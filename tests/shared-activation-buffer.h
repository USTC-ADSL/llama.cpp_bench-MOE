#pragma once

#include "expert-probe-common.h"
#include "ggml-shared-dma.h"
#include <unordered_map>
#include <unordered_set>

// Closed lifetimes: a node's output never aliases an input of that same node.
// Inputs coexist before execution; graph sinks coexist until host aggregation.
struct shared_activation_layout {
    struct slot { ggml_tensor * tensor; size_t offset,bytes; int first,last; };
    struct reservation {
        struct entry { size_t offset,bytes; int first,last; ggml_type type; ggml_op op; };
        std::vector<entry> entries;
        size_t bytes=0;
    };
    std::vector<slot> slots;
    size_t bytes=0;
    reservation freeze() const {
        reservation r; r.bytes=bytes;
        for(const auto & s:slots) { r.entries.push_back({s.offset,s.bytes,s.first,s.last,s.tensor->type,s.tensor->op}); }
        return r;
    }
    static ggml_tensor * root(ggml_tensor * t) {
        while(t && t->view_src) { t=t->view_src; }
        return t;
    }
    shared_activation_layout(device & dev,ggml_context * ctx,ggml_cgraph * graph) {
        const size_t alignment=std::max<size_t>(128,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev.dev)));
        auto align=[&](size_t n) { return (n+alignment-1)/alignment*alignment; };
        std::unordered_map<ggml_tensor *,size_t> index;
        for(auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            if(t->view_src) { continue; }
            index[t]=slots.size();
            size_t size=std::max(ggml_nbytes(t),ggml_backend_buft_get_alloc_size(ggml_backend_dev_buffer_type(dev.dev),t));
            slots.push_back({t,0,align(size),0,0});
        }
        const int n=ggml_graph_n_nodes(graph);
        std::unordered_set<ggml_tensor *> consumed;
        for(int i=0;i<n;++i) {
            auto * t=ggml_graph_node(graph,i);
            if(!t->view_src && index.count(t)) { slots[index.at(t)].first=slots[index.at(t)].last=i+1; }
            for(auto * src:t->src) { if(src) { consumed.insert(src); } }
        }
        for(int i=0;i<n;++i) {
            auto * t=ggml_graph_node(graph,i);
            for(auto * src:t->src) { auto it=index.find(root(src)); if(it!=index.end()) {
                slots[it->second].last=std::max(slots[it->second].last,i+1);
            } }
            if(!consumed.count(t)) { auto it=index.find(root(t)); if(it!=index.end()) { slots[it->second].last=n+1; } }
        }
        std::vector<size_t> order(slots.size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) {
            return slots[a].first!=slots[b].first?slots[a].first<slots[b].first:slots[a].bytes>slots[b].bytes;
        });
        struct region { size_t offset,bytes; };
        std::vector<region> free;
        std::vector<size_t> active;
        for(size_t id:order) {
            auto & s=slots[id];
            for(auto it=active.begin();it!=active.end();) {
                const auto & old=slots[*it];
                if(old.last<s.first) { free.push_back({old.offset,old.bytes}); it=active.erase(it); }
                else { ++it; }
            }
            std::sort(free.begin(),free.end(),[](const region & a,const region & b) { return a.offset<b.offset; });
            for(size_t j=1;j<free.size();) {
                if(free[j-1].offset+free[j-1].bytes==free[j].offset) { free[j-1].bytes+=free[j].bytes; free.erase(free.begin()+j); }
                else { ++j; }
            }
            auto best=free.end();
            for(auto it=free.begin();it!=free.end();++it) {
                if(it->bytes>=s.bytes && (best==free.end() || it->bytes<best->bytes)) { best=it; }
            }
            if(best==free.end()) { s.offset=bytes; bytes+=s.bytes; }
            else { s.offset=best->offset; best->offset+=s.bytes; best->bytes-=s.bytes; if(!best->bytes) { free.erase(best); } }
            active.push_back(id);
        }
    }
};

struct shared_activation_buffer {
    // Imported views are destroyed before the backing and its host mapping.
    ggml_backend_buffer_ptr backing;
    ggml_backend_buffer_ptr imported;
    ggml_backend_shared_dma_buffer_desc desc{};
    using export_fn = bool (*)(ggml_backend_buffer_t, ggml_backend_shared_dma_buffer_desc *);
    using import_fn = ggml_backend_buffer_t (*)(ggml_backend_t, const ggml_backend_shared_dma_buffer_desc *);

    static import_fn gpu_import(device & gpu) {
        return reinterpret_cast<import_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(gpu.dev), "ggml_backend_opencl_import_shared_dma_compute_buffer"));
    }
    shared_activation_buffer(device & dev, size_t bytes) {
        auto htp=ggml_backend_dev_by_name("HTP0"); require(htp!=nullptr,"shared activation requires HTP0");
        auto export_dma=reinterpret_cast<export_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(htp),"ggml_backend_hexagon_buffer_export_shared_dma"));
        require(export_dma!=nullptr,"HTP shared export API missing");
        backing.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(htp),bytes));
        require(backing && export_dma(backing.get(),&desc),"shared backing allocation/export failed");
        ggml_backend_buffer_set_usage(backing.get(),GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (dev.name=="GPUOpenCL") {
            auto import=gpu_import(dev); require(import!=nullptr,"OpenCL DMA import API missing");
            imported.reset(import(dev.backend.get(),&desc));
            require(imported!=nullptr,"OpenCL IO-coherent DMA import unavailable");
        } else if (dev.name=="CPU") {
            imported.reset(ggml_backend_dev_buffer_from_host_ptr(dev.dev,desc.host_ptr,desc.size,desc.size));
            require(imported!=nullptr,"CPU shared activation view failed");
        } else { require(dev.name=="HTP0","unsupported shared activation backend"); }
        ggml_backend_buffer_set_usage(get(),GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    ggml_backend_buffer_t get() const { return imported?imported.get():backing.get(); }
    static size_t align(size_t bytes,size_t alignment) { return (bytes+alignment-1)/alignment*alignment; }
    static size_t required_size(device & dev,ggml_context * ctx,ggml_cgraph * graph=nullptr) {
        if(graph) { return shared_activation_layout(dev,ctx,graph).bytes; }
        size_t alignment=std::max<size_t>(128,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev.dev)));
        size_t bytes=0;
        for (auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            if (!t->view_src) { bytes=align(bytes,alignment)+ggml_nbytes(t); }
        }
        return align(bytes,alignment);
    }
    void bind(device & dev,ggml_context * ctx,ggml_cgraph * graph=nullptr,const shared_activation_layout::reservation * reserved=nullptr) {
        if(graph) {
            shared_activation_layout layout(dev,ctx,graph);
            require((reserved?reserved->bytes:layout.bytes)<=desc.size,"shared lifetime arena capacity exceeded");
            require(!reserved || reserved->entries.size()==layout.slots.size(),"lifetime graph template shape mismatch");
            for(size_t i=0;i<layout.slots.size();++i) {
                const auto & s=layout.slots[i]; size_t offset=s.offset;
                if(reserved) {
                    const auto & r=reserved->entries[i];
                    require(s.bytes<=r.bytes && s.first==r.first && s.last==r.last && s.tensor->type==r.type && s.tensor->op==r.op,
                            "lifetime graph exceeds reservation contract");
                    offset=r.offset;
                }
                require(ggml_backend_tensor_alloc(get(),s.tensor,static_cast<uint8_t *>(desc.host_ptr)+offset)==GGML_STATUS_SUCCESS,
                        "shared lifetime tensor allocation failed");
            }
            for(auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
                if(t->view_src) { require(ggml_backend_view_init(t)==GGML_STATUS_SUCCESS,"shared lifetime view init failed"); }
            }
            return;
        }
        size_t alignment=std::max<size_t>(128,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev.dev)));
        require(required_size(dev,ctx)<=desc.size,"shared activation arena capacity exceeded");
        size_t offset=0;
        for (auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            if (t->view_src) { require(ggml_backend_view_init(t)==GGML_STATUS_SUCCESS,"shared view init failed"); }
            else {
                offset=align(offset,alignment);
                require(ggml_backend_tensor_alloc(get(),t,static_cast<uint8_t *>(desc.host_ptr)+offset)==GGML_STATUS_SUCCESS,
                        "shared tensor allocation failed");
                offset+=ggml_nbytes(t);
            }
        }
    }
    static std::unique_ptr<shared_activation_buffer> allocate(device & dev,ggml_context * ctx) {
        auto buffer=std::make_unique<shared_activation_buffer>(dev,required_size(dev,ctx));
        buffer->bind(dev,ctx);
        return buffer;
    }
};
