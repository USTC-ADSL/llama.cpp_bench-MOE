#pragma once
#include "moe-context-model.h"

// Resident, pre-load planning only. No weight migration is hidden in this cost.
struct fullmodel_execution_model {
    resident_placement_model base;
    json node;
    double decision_budget;
    static int shared(const std::string & f) { return f=="gpu"?1:2; }
    static void coefficient(const json & v,bool zero=false) {
        require(v.is_number(),"nonnumeric execution coefficient");
        double x=v.get<double>(); require(std::isfinite(x) && (zero?x>=0:x>0),"invalid execution coefficient");
    }
    static json lerp(const json & a,const json & b,double w) {
        if(a.is_object()) { json out=json::object(); for(auto it=a.begin();it!=a.end();++it) { out[it.key()]=lerp(it.value(),b.at(it.key()),w); } return out; }
        if(a.is_array()) { require(a.size()==b.size(),"coefficient shape mismatch"); json out=json::array(); for(size_t i=0;i<a.size();++i) { out.push_back(lerp(a[i],b[i],w)); } return out; }
        return (1-w)*a.get<double>()+w*b.get<double>();
    }
    fullmodel_execution_model(const json & m,int nt):base(m.at("base")),decision_budget(m.at("decision_budget_us")) {
        require(m.at("version")==2 && m.at("model_type")=="branch_critical_path","invalid execution model schema");
        require(m.at("calibrated_shared_backend")==json({{"cpu",2},{"gpu",1},{"branch",2}}),"invalid shared calibration policy");
        coefficient(m.at("decision_budget_us"),true);
        for(int b=0;b<3;++b) { require(base.coefficients[b].size()==26,"invalid base coefficient count"); for(double x:base.coefficients[b]) { require(std::isfinite(x),"invalid base coefficient"); } }
        require(base.host_coefficients.size()==8,"invalid host coefficient count");
        for(double x:base.host_coefficients) { require(std::isfinite(x),"invalid host coefficient"); }
        const auto & nodes=m.at("nodes"); require(nodes.size()>=2,"invalid execution nodes");
        int previous=1;
        for(const auto & n:nodes) {
            require(n.at("tokens").is_number_integer(),"invalid node length"); int t=n.at("tokens"); require(t>previous,"unsorted execution nodes"); previous=t;
            for(const char * k:{"native_us","native_base_compute_us"}) { coefficient(n.at(k)); }
            coefficient(n.at("native_uncertainty_us"),true);
            for(const char * f:{"cpu","gpu","branch"}) { for(const char * p:{"barrier","overlap"}) {
                const auto & c=n.at("families").at(f).at(p);
                for(const char * k:{"compute_scales","gather_us_per_mib"}) {
                    require(c.at(k).size()==3,"invalid backend coefficient count");
                    for(const auto & x:c.at(k)) { coefficient(x,std::string(k)=="gather_us_per_mib"); }
                }
                for(const char * k:{"host_us_per_token","outside_us","uncertainty_us"}) { coefficient(c.at(k),true); }
            } }
        }
        require(nt>=nodes.front().at("tokens").get<int>() && nt<=nodes.back().at("tokens").get<int>(),"length outside calibrated execution range");
        for(size_t i=1;i<nodes.size();++i) {
            const auto & a=nodes[i-1]; const auto & b=nodes[i]; int lo=a.at("tokens"),hi=b.at("tokens");
            if(nt>=lo && nt<=hi) { node=lerp(a,b,double(nt-lo)/(hi-lo)); break; }
        }
    }
    static std::array<double,3> gather_mib(int nt,const std::vector<int> & counts,const placement & p) {
        std::array<int,3> pairs{},bank{}; for(int e=0;e<60;++e) { pairs[p.owner[e]]+=counts[e]; ++bank[p.owner[e]]; }
        bool native=*std::max_element(pairs.begin(),pairs.end())==4*nt;
        std::array<double,3> out{};
        for(int b=0;b<3;++b) {
            int rows=native && pairs[b]?nt:pairs[b],stride=b==2?bank[b]:native?4:1;
            out[b]=double(rows)*(2048*4+stride*4);
            if(p.shared==b && !(native && rows)) { out[b]+=double(nt)*2048*4; }
            out[b]/=1048576;
        }
        return out;
    }
    static double critical(const std::array<double,3> & g,const std::array<double,3> & c,bool overlap) {
        if(overlap) { return std::max({g[0]+c[0],g[1]+c[1],g[2]+c[2]}); }
        return g[0]+g[1]+g[2]+*std::max_element(c.begin(),c.end());
    }
    json predict(int nt,const std::vector<int> & counts,const placement & p,const std::string & f,bool overlap) const {
        auto plain=base.predict(nt,counts,p); std::array<double,3> g{}; double path=*std::max_element(plain.backend.begin(),plain.backend.end());
        if(nt>1) {
            require(fullmodel_context_model::family(p)==f && p.shared==shared(f),"uncalibrated family");
            const auto & c=node.at("families").at(f).at(overlap?"overlap":"barrier"); g=gather_mib(nt,counts,p);
            for(int b=0;b<3;++b) { plain.backend[b]*=c.at("compute_scales").at(b).get<double>(); g[b]*=c.at("gather_us_per_mib").at(b).get<double>(); }
            path=critical(g,plain.backend,overlap); plain.host=nt*c.at("host_us_per_token").get<double>(); plain.total=path+plain.host;
        }
        return {{"backend_us",plain.backend},{"gather_us",g},{"critical_us",path},{"host_us",plain.host},{"total_us",plain.total}};
    }
    json decide(double custom,double native_compute,double uncertainty,double observed) const {
        double native=std::max(1.,node.at("native_us").get<double>()+native_compute-node.at("native_base_compute_us").get<double>());
        double budget=std::max(decision_budget,observed),upper=custom+uncertainty+budget,lower=native-node.at("native_uncertainty_us").get<double>();
        return {{"execution_mode",upper<lower?"custom":"npu"},{"custom_predicted_us",custom},{"native_predicted_us",native},
            {"custom_upper_us",upper},{"native_lower_us",lower},{"custom_uncertainty_us",uncertainty},
            {"native_uncertainty_us",node.at("native_uncertainty_us")},{"decision_budget_us",budget},
            {"reserved_decision_budget_us",decision_budget},{"observed_planning_us",observed}};
    }
    json plan(const json & capture,const json & layers) const {
        int tokens=capture.at("tokens"); require(capture.at("token_ids").size()==size_t(tokens) && layers.size()==24,"invalid fullmodel route shape");
        for(int il=0;il<24;++il) { require(layers[il].at("layer")==il && layers[il].at("tokens")== (il==23?1:tokens),"invalid fullmodel layer shape"); }
        double native_compute=0; for(const auto & l:layers) { native_compute+=base.predict(l.at("tokens"),l.at("counts").get<std::vector<int>>(),{"npu_shared_npu",std::vector<int>(60,1),1}).backend[1]; }
        json options=json::array(),best_layers,best; double best_score=INFINITY;
        for(const char * f:{"cpu","gpu","branch"}) { for(bool overlap:{false,true}) {
            const auto & c=node.at("families").at(f).at(overlap?"overlap":"barrier"); json selected=json::array(); double cost=c.at("outside_us");
            for(const auto & l:layers) {
                int nt=l.at("tokens"); auto counts=l.at("counts").get<std::vector<int>>(); double fastest=INFINITY; placement chosen; json prediction;
                for(const auto & p:model_candidates(counts)) {
                    if(nt>1 && (fullmodel_context_model::family(p)!=f || p.shared!=shared(f))) { continue; }
                    auto pred=predict(nt,counts,p,f,overlap); double total=pred.at("total_us");
                    if(total<fastest) { fastest=total; chosen=p; prediction=std::move(pred); }
                }
                require(std::isfinite(fastest),"no eligible execution candidate"); auto row=l;
                row["selected"]={{"name",chosen.name},{"owner",chosen.owner},{"shared",chosen.shared}}; row["prediction"]=prediction; selected.push_back(std::move(row)); cost+=fastest;
            }
            json option={{"family",f},{"policy",overlap?"overlap":"barrier"},{"predicted_us",cost},{"uncertainty_us",c.at("uncertainty_us")},{"outside_us",c.at("outside_us")}};
            options.push_back(option); double score=cost+c.at("uncertainty_us").get<double>();
            if(score<best_score) { best_score=score; best=option; best_layers=std::move(selected); }
        } }
        auto decision=decide(best.at("predicted_us"),native_compute,best.at("uncertainty_us"),0);
        return {{"tokens",tokens},{"token_ids",capture.at("token_ids")},{"layers",best_layers},{"model_type","branch_critical_path"},
            {"execution_mode",decision.at("execution_mode")},{"path_decision",decision},{"native_base_compute_us",native_compute},
            {"execution_policy",{{"family",best.at("family")},{"overlap_gather",best.at("policy")=="overlap"},{"activation_lifetimes",true}}},
            {"custom_candidates",options},{"scope","offline pre-load; one calibrated family/policy per request, layer-specific owners; no migration"}};
    }
};
