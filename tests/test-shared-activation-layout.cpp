#include "shared-activation-buffer.h"

struct result { std::vector<float> values; size_t bytes; shared_activation_layout::reservation reservation; };

static result execute(device & dev,int rows,bool compact,const shared_activation_layout::reservation * reserved=nullptr) {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}));
    require(ctx!=nullptr,"test context allocation failed");
    auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
    auto * x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,32,rows);
    auto * y=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,32,rows);
    auto * z=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,32,rows);
    auto * view=ggml_reshape_2d(ctx.get(),ggml_view_1d(ctx.get(),x,32*rows,0),32,rows);
    auto * first=ggml_mul(ctx.get(),ggml_silu(ctx.get(),ggml_mul(ctx.get(),view,y)),y);
    ggml_build_forward_expand(graph,first);
    // A late use keeps x alive across the first sink; both sinks must survive.
    auto * second=ggml_add(ctx.get(),ggml_silu(ctx.get(),ggml_mul(ctx.get(),x,z)),z);
    ggml_build_forward_expand(graph,second);
    shared_activation_layout layout(dev,ctx.get(),graph);
    for(const auto & a:layout.slots) {
        if(a.tensor==x || a.tensor==y || a.tensor==z) { require(a.first==0,"input allocated after execution starts"); }
        if(a.tensor==first || a.tensor==second) { require(a.last==ggml_graph_n_nodes(graph)+1,"sink not retained"); }
        for(const auto & b:layout.slots) {
            if(&a==&b || a.last<b.first || b.last<a.first) { continue; }
            require(a.offset+a.bytes<=b.offset || b.offset+b.bytes<=a.offset,"overlapping live tensors");
        }
    }
    size_t bytes=reserved?reserved->bytes:shared_activation_buffer::required_size(dev,ctx.get(),compact?graph:nullptr);
    shared_activation_buffer buffer(dev,bytes); buffer.bind(dev,ctx.get(),compact?graph:nullptr,reserved);
    require(view->data==x->data,"view root not bound correctly");
    result r; r.bytes=bytes; r.reservation=layout.freeze();
    for(int round=0;round<3;++round) {
        for(int i=0;i<32*rows;++i) {
            static_cast<float *>(x->data)[i]=float((i+round)%17-8)*.1f;
            static_cast<float *>(y->data)[i]=float((i+round)%11-5)*.2f;
            static_cast<float *>(z->data)[i]=float((i+round)%7-3)*.3f;
        }
        dev.compute(graph);
        for(auto * out:{first,second}) {
            auto * data=static_cast<float *>(out->data); r.values.insert(r.values.end(),data,data+32*rows);
        }
    }
    return r;
}

int main() {
    try {
        ggml_backend_load_all(); device cpu("CPU");
        auto maximum=execute(cpu,1024,true);
        for(int rows:{1,17,768,1024}) {
            auto full=execute(cpu,rows,false),compact=execute(cpu,rows,true);
            auto reserved=execute(cpu,rows,true,&maximum.reservation);
            require(full.values==compact.values,"lifetime layout changed output");
            require(full.values==reserved.values,"maximum-shape reservation changed output");
            require(compact.bytes<full.bytes,"lifetime layout did not reduce capacity");
            printf("rows=%d full=%zu lifetime=%zu outputs identical\n",rows,full.bytes,compact.bytes);
        }
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
