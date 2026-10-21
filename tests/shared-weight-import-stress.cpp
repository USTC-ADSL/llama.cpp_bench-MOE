#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"
#include <cerrno>
#include <fcntl.h>
#include <sys/resource.h>

static size_t open_fds() {
    rlimit limit{}; require(getrlimit(RLIMIT_NOFILE,&limit)==0,"getrlimit failed");
    size_t n=0;
    for(rlim_t fd=0;fd<limit.rlim_cur;++fd) { n+=fcntl(int(fd),F_GETFD)!=-1; }
    return n;
}

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: shared-weight-import-stress MODEL OUTPUT");
        ggml_backend_load_all();
        ggml_context * raw=nullptr; gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gf && meta && f,"model open failed"); source src{gf.get(),meta.get(),f.get()};
        device htp("HTP0"),gpu("GPUOpenCL");
        auto * tensor=src.tensor("blk.0.ffn_gate_exps.weight"); auto bytes=src.read("blk.0.ffn_gate_exps.weight");
        const auto size=ggml_row_size(GGML_TYPE_Q4_0,tensor->ne[0])*tensor->ne[1];
        using handle=moe_shared_weights::handle;
        handle parent(ggml_shared_weight_create_with_layout(htp.backend.get(),tensor->ne[0],tensor->ne[1],bytes.data(),size,
                      GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2),ggml_shared_weight_free);
        require(bool(parent),"parent creation failed");
        rlimit limit{}; require(getrlimit(RLIMIT_NOFILE,&limit)==0,"getrlimit failed");
        // A low soft limit makes resource scaling observable without large memory use.
        rlimit test_limit=limit; test_limit.rlim_cur=std::min<rlim_t>(limit.rlim_cur,512);
        require(setrlimit(RLIMIT_NOFILE,&test_limit)==0,"setrlimit failed");
        ggml_shared_weight_desc desc{}; require(ggml_shared_weight_describe(parent.get(),&desc),"descriptor failed");
        json report={{"fd_soft_limit",test_limit.rlim_cur},{"backing_bytes",size},{"fd_before",open_fds()}};
        std::vector<handle> slices;
        bool passed=true;
        for(int i=0;i<640;++i) {
            int first=(i%int(tensor->ne[1]/32))*32;
            handle slice(ggml_shared_weight_slice_rows(parent.get(),first,32),ggml_shared_weight_free);
            require(bool(slice),"slice creation failed"); errno=0;
            auto * bound=ggml_shared_weight_bind(slice.get(),gpu.backend.get()); int error=errno;
            if(!bound) { report["failed_at"]=i; report["errno"]=error; passed=false; break; }
            require(bound->data==static_cast<uint8_t *>(desc.storage.host_ptr)+size_t(first)*ggml_row_size(GGML_TYPE_Q4_0,tensor->ne[0]),"alias mismatch");
            slices.push_back(std::move(slice));
        }
        report["bindings"]=slices.size(); report["fd_with_bindings"]=open_fds();
        parent.reset(); report["fd_after_parent_free"]=open_fds();
        slices.clear(); report["fd_after_release"]=open_fds(); report["passed"]=passed;
        std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
        fprintf(stderr,"IMPORT_STRESS %s\n",report.dump().c_str()); return passed?0:2;
    } catch(const std::exception & e) { fprintf(stderr,"import stress: %s\n",e.what()); return 1; }
}
