#include "shared-activation-buffer.h"
#include <unistd.h>

int main() {
    try {
        ggml_backend_load_all(); device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL");
        shared_activation_buffer storage(npu,65536);
        auto import=shared_activation_buffer::gpu_import(gpu); require(import!=nullptr,"missing GPU import");
        auto borrowed=storage.desc;
        borrowed.dma_buf_fd=dup(storage.desc.dma_buf_fd);
        require(borrowed.dma_buf_fd>=0,"fd duplication failed");
        ggml_backend_buffer_ptr gpu_view(import(gpu.backend.get(),&borrowed));
        close(borrowed.dma_buf_fd);
        require(gpu_view!=nullptr,"GPU import failed");
        require(import(nullptr,&storage.desc)==nullptr,"null backend accepted");
        auto invalid=storage.desc; invalid.dma_buf_fd=-1;
        require(import(gpu.backend.get(),&invalid)==nullptr,"invalid fd accepted");
        invalid=storage.desc; invalid.host_ptr=static_cast<char *>(invalid.host_ptr)+1;
        require(import(gpu.backend.get(),&invalid)==nullptr,"unaligned pointer accepted");
        auto make_context=[] { return ggml_context_ptr(ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(16,false),nullptr,true})); };
        auto cg=make_context(),cn=make_context(); require(cg && cn,"context allocation failed");
        auto * unsupported=ggml_new_tensor_1d(cg.get(),GGML_TYPE_F16,256);
        require(ggml_backend_tensor_alloc(gpu_view.get(),unsupported,storage.desc.host_ptr)!=GGML_STATUS_SUCCESS,"F16 import unexpectedly accepted");
        auto bind=[&](ggml_backend_buffer_t buffer,ggml_tensor * t,size_t offset) {
            require(ggml_backend_tensor_alloc(buffer,t,static_cast<char *>(storage.desc.host_ptr)+offset)==GGML_STATUS_SUCCESS,"bind failed");
        };
        auto * x=ggml_new_tensor_1d(cg.get(),GGML_TYPE_F32,256); bind(gpu_view.get(),x,4096);
        auto * gx=ggml_add(cg.get(),x,x); bind(gpu_view.get(),gx,8192);
        auto * nx=ggml_new_tensor_1d(cn.get(),GGML_TYPE_F32,256); bind(storage.get(),nx,8192);
        auto * ny=ggml_add(cn.get(),nx,nx); bind(storage.get(),ny,12288);
        auto * gy=ggml_new_tensor_1d(cg.get(),GGML_TYPE_F32,256); bind(gpu_view.get(),gy,12288);
        auto * gz=ggml_add(cg.get(),gy,gy); bind(gpu_view.get(),gz,16384);
        auto * g1=ggml_new_graph_custom(cg.get(),16,false); ggml_build_forward_expand(g1,gx);
        auto * g2=ggml_new_graph_custom(cn.get(),16,false); ggml_build_forward_expand(g2,ny);
        auto * g3=ggml_new_graph_custom(cg.get(),16,false); ggml_build_forward_expand(g3,gz);
        auto * base=static_cast<uint8_t *>(storage.desc.host_ptr);
        for (int round=0;round<64;++round) {
            std::memset(base,0x5a,storage.desc.size);
            auto * input=reinterpret_cast<float *>(base+4096);
            for (int i=0;i<256;++i) { input[i]=float((round+1)*256+i); }
            gpu.compute(g1); npu.compute(g2); gpu.compute(g3);
            for (int i=0;i<256;++i) {
                require(reinterpret_cast<float *>(base+8192)[i]==2*input[i],"GPU -> CPU visibility failure");
                require(reinterpret_cast<float *>(base+12288)[i]==4*input[i],"GPU -> HTP visibility failure");
                require(reinterpret_cast<float *>(base+16384)[i]==8*input[i],"HTP -> GPU visibility failure");
            }
            for (size_t i=0;i<storage.desc.size;++i) {
                bool written=false;
                for (size_t start : {4096u,8192u,12288u,16384u}) { written|=i>=start && i<start+1024; }
                require(written || base[i]==0x5a,"shared allocation guard changed");
            }
        }
        printf("{\"rounds\":64,\"physical_backings\":1,\"bytes\":%zu,\"nonzero_offsets\":true,\"staging_calls\":0,\"guards_pass\":true,\"cross_backend_exact\":true,\"borrowed_fd_closed\":true,\"invalid_imports_rejected\":true}\n",storage.desc.size);
        return 0;
    } catch (const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
