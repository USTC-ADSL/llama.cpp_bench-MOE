// Diagnostic only: reuse production graphs without changing their kernels or dispatch.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"
#include <random>
#include <sched.h>

struct overlap_case {
    std::string name;
    int shared; // full FFN shared owner, or isolated backend
    bool full;
    int dispatch; // 0 inline/async production path, 1 pool, 2 isolated async
    bool routed;
};

int main(int argc, char ** argv) {
    try {
        require(argc==6,"usage: moe-decode-overlap-probe MODEL CAPTURE OUTPUT REPEATS SEED");
        const int repeats=std::stoi(argv[4]);
        require(repeats>0,"positive repeats required");
        std::mt19937 rng(std::stoul(argv[5]));
        require(std::getenv("MOE_SHARED_WEIGHTS") && std::string(std::getenv("MOE_SHARED_WEIGHTS"))=="cpu4x8","CPU4x8 required");
        require(std::getenv("MOE_SHARED_SCRATCH") && std::string(std::getenv("MOE_SHARED_SCRATCH"))=="common","common scratch required");
        ggml_backend_load_all(); llama_backend_init();
        std::ifstream in(argv[2]); json capture; in>>capture;
        require(capture.at("tokens")==1,"only N=1 supported");
        ggml_context * raw=nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gguf && meta && f,"model open failed");
        source src{gguf.get(),meta.get(),f.get()};
        device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
        const std::vector<overlap_case> cases={
            {"all_cpu",0,true,0,false},
            {"cpu_npu_async",1,true,0,false}, {"cpu_npu_pool",1,true,1,false},
            {"cpu_gpu_async",2,true,0,false}, {"cpu_gpu_pool",2,true,1,false},
            {"routed_cpu_inline",0,false,0,true}, {"routed_cpu_async",0,false,2,true},
            {"routed_cpu_pool",0,false,1,true},
            {"shared_cpu_inline",0,false,0,false}, {"shared_npu_inline",1,false,0,false},
            {"shared_gpu_inline",2,false,0,false}};
        json report={{"scope","N=1 captured FFN; same resident weights/arenas/cached graphs per layer; randomized blocks; not end-to-end decode"},
            {"repeats",repeats},{"seed",std::stoul(argv[5])},{"capture",argv[2]},
            {"layers",json::array()},{"complete",false}};
        for(int il:{0,6,12,18}) {
            require(capture.at("layers").at(il).at("tokens")==1,"capture layer shape");
            std::ifstream bin(std::string(argv[2])+".layer"+std::to_string(il)+".bin",std::ios::binary);
            auto x=binary_read<float>(bin,2048), probs=binary_read<float>(bin,60);
            auto ids=binary_read<int32_t>(bin,4); auto reference=binary_read<float>(bin,2048);
            require(std::set<int32_t>(ids.begin(),ids.end()).size()==4,"distinct routed experts required");
            placement plan; plan.name="resident_overlap"; plan.owner.assign(60,0); plan.shared=0;
            resident_placement_model::availability resident;
            for(auto & mask:resident.experts) { mask={true,false,false}; }
            resident.shared={true,true,true};
            full_engine engine;
            engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,1,plan,true,true,&resident));
            engine.prepare_arenas(devs);
            moe_token_workers workers(3);
            auto & ff=*engine.layers.front(); auto & execution=*ff.single_layout;
            std::vector<float> actual(2048);
            std::array<int,61> routed_counts{},shared_counts{};
            for(auto e:ids) { ++routed_counts[e]; } shared_counts[60]=1;
            // Warm every production graph before retaining pointers to its jobs.
            for(int b=0;b<3;++b) {
                ff.plan.shared=b;
                ff.run_into(x.data(),x.size(),probs.data(),probs.size(),ids,false,actual.data());
            }
            auto & cpu_shared=execution.jobs[0][{0,shared_counts}];
            require(!cpu_shared,"unexpected isolated shared graph");
            cpu_shared=std::make_unique<moe_shared_job>(cpu,execution.weights,0,shared_counts,
                execution.arenas[0],0,execution.shared_scratch); ++ff.allocations;
            auto job_for=[&](const overlap_case & c)->moe_shared_job & {
                return *execution.jobs[c.shared].at({0,c.routed?routed_counts:shared_counts});
            };
            auto isolated_output=[&](const overlap_case & c) {
                auto & j=job_for(c); std::vector<float> out;
                if(c.routed) {
                    for(auto e:ids) { auto * p=static_cast<float *>(j.outputs[e]->data); out.insert(out.end(),p,p+2048); }
                } else { auto * p=static_cast<float *>(j.outputs[60]->data); out.assign(p,p+2048); }
                return out;
            };
            auto run=[&](const overlap_case & c) {
                if(c.full) {
                    ff.plan.shared=c.shared; execution.branch_workers=c.dispatch==1?&workers:nullptr;
                    ff.run_into(x.data(),x.size(),probs.data(),probs.size(),ids,false,actual.data());
                    return ff.last;
                }
                auto & j=job_for(c);
                std::array<double,2> interval{}; double gather=0,compute=0;
                int core_begin=-1,core_end=-1; auto origin=timer::now();
                auto execute=[&] {
                    interval[0]=elapsed(origin); core_begin=sched_getcpu(); auto t=timer::now();
                    for(auto * input:j.inputs) { if(input) { std::copy_n(x.data(),2048,static_cast<float *>(input->data)); } }
                    gather=elapsed(t); t=timer::now(); j.dev.compute(j.graph); compute=elapsed(t);
                    core_end=sched_getcpu(); interval[1]=elapsed(origin);
                };
                if(c.dispatch==2) { auto future=std::async(std::launch::async,execute); future.get(); }
                else if(c.dispatch==1) {
                    workers.run(3,[&](int begin,int end) { for(int i=begin;i<end;++i) { if(i==2) { execute(); } } });
                } else { execute(); }
                const double wall=elapsed(origin);
                return json{{"expert_wall_us",wall},{"branch_gather_us",gather},{"branch_compute_us",compute},
                    {"interval_us",interval},{"cpu_begin",core_begin},{"cpu_end",core_end}};
            };
            json layer={{"layer",il},{"ids",ids},{"hidden",execution.weights.hidden},
                {"shared_hidden",execution.weights.shared_hidden},{"payload_bytes",execution.weights.payload_bytes},
                {"physical_bytes",execution.weights.physical_bytes},{"checks",json::object()},{"samples",json::array()}};
            std::map<std::string,std::vector<float>> expected;
            for(const auto & c:cases) {
                for(int w=0;w<5;++w) { run(c); }
                if(c.full) {
                    auto check=chain_difference(actual,reference,1);
                    require(check.at("passes_4percent_per_token").get<bool>(),"full FFN numerical check failed");
                    layer["checks"][c.name]=check; expected[c.name]=actual;
                } else { expected[c.name]=isolated_output(c); }
            }
            // Isolated jobs must reproduce the corresponding outputs in the full graphs.
            for(int b=0;b<3;++b) {
                const overlap_case c{"check",b,true,0,false}; run(c);
                std::array<int,61> counts=routed_counts; if(b==0) { counts[60]=1; }
                auto & r=*execution.jobs[0].at({0,counts});
                std::vector<float> routed;
                for(auto e:ids) { auto * p=static_cast<float *>(r.outputs[e]->data); routed.insert(routed.end(),p,p+2048); }
                require(routed==expected.at("routed_cpu_inline"),"isolated routed output mismatch");
                auto & s=*execution.jobs[b].at({0,b==0?counts:shared_counts});
                auto * p=static_cast<float *>(s.outputs[60]->data);
                const std::string name=b==0?"shared_cpu_inline":b==1?"shared_npu_inline":"shared_gpu_inline";
                require(std::vector<float>(p,p+2048)==expected.at(name),"isolated shared output mismatch");
            }
            const size_t warmed=ff.allocations;
            std::vector<int> order(cases.size()); std::iota(order.begin(),order.end(),0);
            for(int block=0;block<repeats;++block) {
                std::shuffle(order.begin(),order.end(),rng);
                for(int idx:order) {
                    const auto & c=cases[idx]; rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
                    auto start=timer::now(); json sample=run(c); double total=elapsed(start);
                    getrusage(RUSAGE_SELF,&after);
                    require(ff.allocations==warmed,"graph allocation after warmup");
                    const auto output=c.full?actual:isolated_output(c);
                    require(output==expected.at(c.name),"non-deterministic output after switching candidates");
                    sample["name"]=c.name; sample["block"]=block; sample["total_us"]=total;
                    sample["major_faults"]=after.ru_majflt-before.ru_majflt;
                    sample["minor_faults"]=after.ru_minflt-before.ru_minflt;
                    layer["samples"].push_back(sample);
                }
            }
            execution.branch_workers=nullptr;
            layer["graph_allocations"]=warmed; layer["all_outputs_bitwise_stable"]=true;
            report["layers"].push_back(layer);
            std::ofstream out(argv[3]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
            fprintf(stderr,"OVERLAP layer=%d samples=%zu checks=passed\n",il,layer["samples"].size());
        }
        report["complete"]=true;
        std::ofstream out(argv[3]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
