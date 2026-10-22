// Resident Qwen1.5-MoE cost model. Coefficients are fitted by moe-placement-model.py.
// No backend allocation or weight conversion occurs during choose().
#pragma once

static std::vector<placement> model_candidates(const std::vector<int> & counts) {
    const char * names[]={"cpu","npu","gpu"};
    std::vector<placement> out;
    for(int b=0;b<3;++b) { for(int s=0;s<3;++s) {
        out.push_back({std::string(names[b])+"_shared_"+names[s],std::vector<int>(60,b),s});
    } }
    std::vector<int> order; for(int e=0;e<60;++e) { if(counts[e]) { order.push_back(e); } }
    std::sort(order.begin(),order.end(),[&](int a,int b) { return counts[a]==counts[b]?a<b:counts[a]<counts[b]; });
    for(int b:{0,2}) { for(int n:{1,4,8,16,24,32}) { if(n<int(order.size())) { for(int s:{1,2}) {
        placement p{"npu_tail"+std::to_string(n)+"_"+names[b]+"_shared_"+names[s],std::vector<int>(60,1),s};
        for(int i=0;i<n;++i) { p.owner[order[i]]=b; } out.push_back(p);
    } } } }
    return out;
}

struct resident_placement_model {
    std::vector<double> coefficients[3],host_coefficients;
    // Optional independent full-model context correction, interpolated by shape.
    // This is not an expert-ID table or a request timing lookup.
    std::vector<std::pair<int,std::array<double,3>>> context_scales;
    struct availability {
        std::array<std::array<bool,3>,60> experts{};
        std::array<bool,3> shared{};
        // Missing nonresident candidate cost means ineligible. Costs are serial us.
        std::map<std::string,double> preparation_us;
    };
    explicit resident_placement_model(const json & j) {
        require(j.at("version")==1,"unsupported placement model version");
        for(int b=0;b<3;++b) {
            coefficients[b]=j.at("backends").at(b).get<std::vector<double>>();
            require(coefficients[b].size()==26,"backend coefficient shape mismatch");
            for(double w:coefficients[b]) { require(std::isfinite(w),"nonfinite model coefficient"); }
        }
        host_coefficients=j.at("host").get<std::vector<double>>();
        require(host_coefficients.size()==8,"host coefficient shape mismatch");
        for(double w:host_coefficients) { require(std::isfinite(w),"nonfinite model coefficient"); }
        if(j.contains("context_scales")) {
            int previous=0;
            for(const auto & knot:j.at("context_scales")) {
                int nt=knot.at("tokens");
                auto factors=knot.at("backends").get<std::array<double,3>>();
                require(nt>previous,"context lengths must be positive and strictly increasing");
                for(double f:factors) { require(std::isfinite(f) && f>0,"invalid context multiplier"); }
                context_scales.push_back({nt,factors}); previous=nt;
            }
        }
    }
    std::array<double,3> context_scale(int nt) const {
        if(context_scales.empty()) { return {1.,1.,1.}; }
        if(nt<=context_scales.front().first) { return context_scales.front().second; }
        for(size_t i=1;i<context_scales.size();++i) {
            const auto & hi=context_scales[i]; const auto & lo=context_scales[i-1];
            if(nt<=hi.first) {
                double fraction=double(nt-lo.first)/(hi.first-lo.first); auto out=lo.second;
                for(int b=0;b<3;++b) { out[b]+=(hi.second[b]-lo.second[b])*fraction; }
                return out;
            }
        }
        return context_scales.back().second;
    }
    template<class Features> static double dot(const Features & x,const std::vector<double> & w) {
        require(x.size()==w.size(),"model coefficient shape mismatch");
        double sum=0; for(size_t i=0;i<x.size();++i) { sum+=x[i]*w[i]; } return sum;
    }
    struct prediction { std::array<double,3> backend{}; double host=0,total=0; };
    prediction predict(int nt,const std::vector<int> & counts,const placement & p,
                       const std::array<int,3> * resident_width=nullptr) const {
        require(counts.size()==60 && p.owner.size()==60 && p.shared>=0 && p.shared<3,"invalid model input");
        double pairs[3]{},active_experts[3]{},pad8[3]{},pad32[3]{},max_count[3]{},squares[3]{},bank[3]{};
        bool active[3]{};
        double total_pairs=0;
        for(int e=0;e<60;++e) {
            int b=p.owner[e],n=counts[e]; require(b>=0 && b<3 && n>=0,"invalid expert load");
            ++bank[b]; pairs[b]+=n; total_pairs+=n;
            if(n) { ++active_experts[b]; pad8[b]+=(n+7)/8*8; pad32[b]+=(n+31)/32*32;
                max_count[b]=std::max(max_count[b],double(n)); squares[b]+=double(n)*n; }
        }
        require(nt>0 && total_pairs==4.0*nt,"invalid top-k load accounting");
        // A reusable union bank can be wider than the currently assigned group.
        // OpenCL index padding/reorder uses this physical width, not ownership.
        if(resident_width) { for(int b=0;b<3;++b) { bank[b]=(*resident_width)[b]; } }
        bool native=*std::max_element(pairs,pairs+3)==total_pairs;
        for(int b=0;b<3;++b) { active[b]=pairs[b]>0 || p.shared==b; }
        prediction out;
        const auto scale=context_scale(nt);
        for(int b=0;b<3;++b) { if(active[b]) {
            double shared=p.shared==b,has=bool(pairs[b]),rows=native && has?nt:pairs[b];
            std::array<double,26> x={1,pairs[b]/1024,active_experts[b]/60,pad8[b]/1024,pad32[b]/1024,
                max_count[b]/128,squares[b]/std::max(1.,pairs[b])/128,bank[b]/60,rows/2048,double(native)*has,
                double(rows==1),bank[b]*rows/(60*2048),shared,shared*nt/512,shared*((nt+31)/32)/16,
                shared*has,shared*pairs[b]/1024};
            for(int other=0;other<3;++other) {
                x[17+3*other]=other!=b?active[other]:0;
                x[18+3*other]=other!=b?pairs[other]/1024:0;
                x[19+3*other]=other!=b && p.shared==other?double(nt)/512:0;
            }
            out.backend[b]=std::max(1.,dot(x,coefficients[b])*scale[b]);
        } }
        std::array<double,8> x={1,double(nt)/512,double(native),double(native)*nt/512,
            double(active[0]+active[1]+active[2])/3,(active_experts[0]+active_experts[1]+active_experts[2])/60,
            (bank[0]*pairs[0]+bank[1]*pairs[1]+bank[2]*pairs[2])/(60*2048),p.shared==0?double(nt)/512:0};
        out.host=std::max(0.,dot(x,host_coefficients));
        out.total=*std::max_element(out.backend.begin(),out.backend.end())+out.host;
        return out;
    }
    placement choose(int nt,const std::vector<int> & counts,const availability & available,prediction & predicted,double & preparation) const {
        double best=INFINITY; placement chosen;
        for(const auto & p:model_candidates(counts)) {
            bool ready=available.shared[p.shared];
            for(int e=0;e<60;++e) { ready=ready && available.experts[e][p.owner[e]]; }
            auto cost=available.preparation_us.find(p.name);
            if(!ready && cost==available.preparation_us.end()) { continue; }
            double prepare=cost==available.preparation_us.end()?0:cost->second;
            require(std::isfinite(prepare) && prepare>=0,"invalid preparation cost");
            auto estimate=predict(nt,counts,p);
            if(estimate.total+prepare<best) { best=estimate.total+prepare; chosen=p; predicted=estimate; preparation=prepare; }
        }
        require(std::isfinite(best),"no feasible resident/preparable placement");
        return chosen;
    }
};
