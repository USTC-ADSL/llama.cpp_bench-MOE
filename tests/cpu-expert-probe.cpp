// Complete real Expert FFN: CPU layout vs operator, separately accounting first-use packing.
#define EXPERT_ORACLE_NO_MAIN
#include "expert-oracle-probe.cpp"
#include "online-cpu-experts.h"

static std::vector<float> reference_ffn(const matrix &g,const matrix &u,const matrix &d,
                                      const std::vector<float> &x,int nt) {
    auto gate=g.reference(x,nt),up=u.reference(x,nt);
    for(size_t i=0;i<up.size();++i) { up[i]*=gate[i]/(1.0f+std::exp(-gate[i])); }
    return d.reference(up,nt);
}

static int check_packed_bytes(online_cpu_experts &executor,const bank &source) {
    executor.prepare(source,{0,1});
    for(int interleave:{4,8}) {
        bool equal=true;
        for(int i=0;i<3;++i) {
            auto *s=source.w[i],*t=executor.target.w[i];
            const int blocks=int(s->ne[0]/32),rows=int(s->ne[1]);
            std::vector<uint8_t> expected(ggml_nbytes(t));
            const auto *src=static_cast<const uint8_t *>(s->data);
            for(int slot=0;slot<executor.capacity;++slot) { for(int r=0;r<rows;r+=4) { for(int b=0;b<blocks;++b) {
                auto *out=expected.data()+((size_t(slot)*rows+r)/4*blocks+b)*72;
                for(int lane=0;lane<4;++lane) {
                    const auto *in=src+std::min(slot,1)*s->nb[2]+((r+lane)*blocks+b)*18;
                    std::memcpy(out+lane*2,in,2);
                    for(int q=0;q<16;++q) { out[8+(q/interleave)*4*interleave+lane*interleave+q%interleave]=in[2+q]^0x88; }
                }
            } } }
            equal &= std::memcmp(t->data,expected.data(),expected.size())==0;
        }
        if(equal) { return interleave; }
    }
    throw std::runtime_error("CPU repack differs from scalar layout reference");
}

int main(int argc,char **argv) {
    try {
        require(argc==4,"usage: cpu-expert-probe MODEL OUTPUT REPS");
        const int reps=std::stoi(argv[3]); require(reps>=4 && reps<=100,"probe reps");
        ggml_log_set([](ggml_log_level l,const char *s,void*) { if(l!=GGML_LOG_LEVEL_DEBUG) { fputs(s,stderr); } },nullptr);
        ggml_backend_load_all(); device cpu("CPU");
        ggml_context *raw=nullptr; gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gf && meta,"model metadata");
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose); require(bool(file),"model open");
        source src{gf.get(),meta.get(),file.get()}; std::vector<int> subset={7,19};
        auto repack_buft=cpu.buft; cpu.buft=ggml_backend_dev_buffer_type(cpu.dev);
        bank canonical(cpu,src,0,&subset); cpu.buft=repack_buft;
        online_cpu_experts repack_id(cpu,canonical,"repack-id"),repack_mm(cpu,canonical,"repack-mm"),canonical_mm(cpu,canonical,"canonical-mm");
        std::array<std::unique_ptr<matrix>,6> matrices;
        for(int e=0;e<2;++e) { for(int i=0;i<3;++i) {
            matrices[e*3+i]=std::make_unique<matrix>(gf.get(),meta.get(),file.get(),std::array<const char*,3>{"gate","up","down"}[i],subset[e]);
        } }
        json report={{"protocol","synthetic activations, real layer0 experts7/19; resident compute and repack+full-FFN separately; not model throughput"},
            {"cpu_buffer",ggml_backend_buft_name(cpu.buft)},{"threads",4},{"cases",json::array()}};
        report["direct_repack"]=std::getenv("GGML_CPU_REPACK_DIRECT")!=nullptr;
        report["scalar_layout_exact_interleave"]=check_packed_bytes(repack_id,canonical);
        const int width=int(canonical.w[0]->ne[0]);
        for(int nt:{1,4,8,16,32,64}) {
            const int rows=nt*2; std::vector<float> x(size_t(rows)*width),reference(x.size());
            std::vector<int32_t> ids(rows); std::vector<int> pairs(rows); std::iota(pairs.begin(),pairs.end(),0);
            for(int row=0;row<rows;++row) {
                ids[row]=row%2;
                for(int c=0;c<width;++c) { x[size_t(row)*width+c]=0.2f*std::sin(float(row*13+c)*0.037f); }
            }
            for(int e=0;e<2;++e) {
                std::vector<float> part(size_t(nt)*width);
                for(int t=0;t<nt;++t) { std::copy_n(x.data()+size_t(2*t+e)*width,width,part.data()+size_t(t)*width); }
                auto out=reference_ffn(*matrices[e*3],*matrices[e*3+1],*matrices[e*3+2],part,nt);
                for(int t=0;t<nt;++t) { std::copy_n(out.data()+size_t(t)*width,width,reference.data()+size_t(2*t+e)*width); }
            }
            oracle_job shape(canonical,nullptr,rows,1,rows,false,false,true,{},true);
            auto arena=std::make_shared<shared_activation_buffer>(cpu,shared_activation_buffer::required_size(cpu,shape.ctx.get(),shape.graph));
            json modes=json::object();
            for(const std::string mode:{"canonical-id","repack-id","repack-mm","canonical-mm"}) {
                std::vector<float> actual(x.size()); std::vector<double> total,conversion,compute;
                for(int r=-1;r<reps;++r) {
                    double tu=0,cu=0,pu=0;
                    if(mode=="canonical-id") {
                        auto begin=timer::now(); oracle_job job(canonical,nullptr,rows,1,rows,false,false,true,arena,false,true);
                        std::copy(x.begin(),x.end(),job.input_data()); std::copy(ids.begin(),ids.end(),job.index_data());
                        cu=job.run()[1]; std::copy_n(job.output_data(),actual.size(),actual.data()); tu=elapsed(begin);
                    } else {
                        auto &executor=mode=="repack-id"?repack_id:mode=="repack-mm"?repack_mm:canonical_mm;
                        auto result=executor.run(canonical,x.data(),1,ids,pairs,rows,arena,[&](int p,const float *out) {
                            std::copy_n(out,width,actual.data()+size_t(p)*width);
                        });
                        tu=result.at("total_us"); cu=result.at("compute_us"); pu=result.at("conversion_us");
                    }
                    if(r==0) { modes[mode]["reference_error"]=validate_values(actual,reference,rows,mode.c_str()); }
                    if(r>=0) { total.push_back(tu); conversion.push_back(pu); compute.push_back(cu); }
                }
                modes[mode]["online_total"]=stats(total,rows); modes[mode]["conversion"]=stats(conversion,rows);
                modes[mode]["compute_in_online"]=stats(compute,rows);
                // Resident ceiling: separate target per Expert only in this microbenchmark.
                std::vector<std::unique_ptr<bank>> banks;
                bool grouped=mode=="repack-mm" || mode=="canonical-mm";
                if(grouped) {
                    for(int e=0;e<2;++e) {
                        banks.push_back(std::make_unique<bank>(cpu,canonical,1,mode=="repack-mm"));
                        for(int i=0;i<3;++i) {
                            auto *s=canonical.w[i],*t=banks.back()->w[i]; void *data=static_cast<uint8_t *>(s->data)+e*s->nb[2];
                            if(mode=="repack-mm") { ggml_backend_tensor_set(t,data,0,s->nb[2]); }
                            else { t->buffer=s->buffer; t->data=data; }
                        }
                    }
                } else if(mode=="repack-id") { repack_id.prepare(canonical,{0,1}); }
                std::vector<std::unique_ptr<oracle_job>> jobs;
                const int n_jobs=grouped?2:1;
                for(int e=0;e<n_jobs;++e) {
                    bank &w=mode=="canonical-id"?canonical:mode=="repack-id"?repack_id.target:*banks[e];
                    // Independent activation buffers preserve input while timing only compute.
                    jobs.push_back(std::make_unique<oracle_job>(w,nullptr,n_jobs==2?nt:rows,1,rows,false,n_jobs==2,true));
                    auto &j=*jobs.back();
                    for(int t=0;t<(n_jobs==2?nt:rows);++t) {
                        int p=n_jobs==2?2*t+e:t;
                        std::copy_n(x.data()+size_t(p)*width,width,j.input_data()+size_t(t)*width); j.index_data()[t]=n_jobs==2?0:ids[p];
                    }
                }
                std::vector<double> resident;
                for(int r=-1;r<reps;++r) { double us=0; for(auto &job:jobs) { us+=job->run()[1]; } if(r>=0) { resident.push_back(us); } }
                modes[mode]["resident_compute"]=stats(resident,rows);
            }
            report["cases"].push_back({{"tokens_per_expert",nt},{"experts",2},{"modes",modes}});
            fprintf(stderr,"CPU_EXPERT tokens=%d complete\n",nt);
        }
        report["workspace"]={{"repack_id_bytes",repack_id.bytes()},{"repack_mm_bytes",repack_mm.bytes()}};
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(bool(out),"probe write");
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"cpu-expert-probe: %s\n",e.what());return 1; }
}
