#pragma once
#include <cstdint>
#include "moe-context-model.h"

// Resident, pre-load planning only. No weight migration is hidden in this cost.
struct fullmodel_execution_model {
    resident_placement_model base;
    json node;
    json guard;
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
        if(m.contains("calibration_guard")) {
            guard=m.at("calibration_guard"); require(guard.at("version")==1 && !guard.at("native").empty() && !guard.at("profiles").empty(),"invalid calibration guard");
            std::vector<int> lengths; std::vector<std::string> ids;
            for(const auto & n:guard.at("native")) {
                require(n.at("tokens").is_number_integer(),"invalid guard length"); int t=n.at("tokens");
                require(t>=nodes.front().at("tokens").get<int>() && t<=nodes.back().at("tokens").get<int>() && std::find(lengths.begin(),lengths.end(),t)==lengths.end(),"invalid guard length"); lengths.push_back(t);
                coefficient(n.at("native_us")); coefficient(n.at("native_base_compute_us")); coefficient(n.at("uncertainty_us"),true);
            }
            for(const auto & p:guard.at("profiles")) {
                std::string id=p.at("id"),f=p.at("family"),policy=p.at("policy");
                require(!id.empty() && std::find(ids.begin(),ids.end(),id)==ids.end() && p.at("tokens").is_number_integer(),"invalid guard profile"); ids.push_back(id);
                require(std::find(lengths.begin(),lengths.end(),p.at("tokens").get<int>())!=lengths.end() && (f=="cpu" || f=="gpu" || f=="branch") && (policy=="barrier" || policy=="overlap"),"invalid guard profile");
                require(p.at("tails").size()==23,"invalid guard bank sizes");
                for(const auto & t:p.at("tails")) { require(t.is_number_integer() && (f=="branch"?t==0:t.get<int>()>=1 && t.get<int>()<60),"invalid guard bank sizes"); }
                require(p.at("bias_us").is_number() && std::isfinite(p.at("bias_us").get<double>()),"invalid guard bias"); coefficient(p.at("uncertainty_us"),true);
            }
            if(guard.contains("intervals")) {
                const auto & extension=guard.at("intervals");
                require(extension.at("version")==1 && !extension.at("regions").empty(),"invalid interval guard");
                std::map<std::string,const json *> by_id;
                for(const auto & p:guard.at("profiles")) { by_id[p.at("id").get<std::string>()]=&p; }
                int last=nodes.front().at("tokens"); std::vector<std::string> region_ids;
                for(const auto & region:extension.at("regions")) {
                    require(region.at("lower_tokens").is_number_integer() && region.at("upper_tokens").is_number_integer(),"invalid interval bounds");
                    int lo=region.at("lower_tokens"),hi=region.at("upper_tokens");
                    require(lo<hi && lo>=last && std::find(lengths.begin(),lengths.end(),lo)!=lengths.end() && std::find(lengths.begin(),lengths.end(),hi)!=lengths.end(),"invalid interval bounds");
                    for(int t:lengths) { require(t<=lo || t>=hi,"interval skips exact anchor"); } last=hi;
                    std::string id=region.at("id");
                    require(!id.empty() && std::find(region_ids.begin(),region_ids.end(),id)==region_ids.end() && !region.at("profiles").empty(),"invalid interval identity"); region_ids.push_back(id);
                    require(region.at("native_bias_us").is_number() && std::isfinite(region.at("native_bias_us").get<double>()),"invalid interval native bias");
                    coefficient(region.at("native_uncertainty_us"),true);
                    for(const auto & p:region.at("profiles")) {
                        std::string pid=p.at("id"),aid=p.at("lower_profile"),bid=p.at("upper_profile");
                        require(!pid.empty() && std::find(ids.begin(),ids.end(),pid)==ids.end(),"invalid interval profile id"); ids.push_back(pid);
                        require(by_id.count(aid) && by_id.count(bid),"unknown interval endpoint profile");
                        const auto & a=*by_id.at(aid); const auto & b=*by_id.at(bid);
                        require(a.at("tokens")==lo && b.at("tokens")==hi && a.at("family")==b.at("family") && a.at("policy")==b.at("policy"),"incompatible interval profiles");
                        require(p.at("bias_us").is_number() && std::isfinite(p.at("bias_us").get<double>()),"invalid interval bias"); coefficient(p.at("uncertainty_us"),true);
                    }
                }
            }
        }
        require(!guard.is_null() || (nt>=nodes.front().at("tokens").get<int>() && nt<=nodes.back().at("tokens").get<int>()),"length outside calibrated execution range");
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
    void finalize(json & report,double observed) const {
        require(std::isfinite(observed) && observed>=0,"invalid observed planning time");
        auto & d=report["path_decision"]; double budget=std::max(decision_budget,observed);
        d["decision_budget_us"]=budget; d["reserved_decision_budget_us"]=decision_budget; d["observed_planning_us"]=observed;
        d["custom_upper_us"]=d.at("custom_predicted_us").is_null()?json(nullptr):json(d.at("custom_predicted_us").get<double>()+d.at("custom_uncertainty_us").get<double>()+budget);
        d["execution_mode"]=!d["custom_upper_us"].is_null() && d["custom_upper_us"].get<double>()<d.at("native_lower_us").get<double>()?"custom":"npu";
        report["execution_mode"]=d["execution_mode"];
    }
    json resolve_interval(int nt) const {
        if(!guard.contains("intervals")) { return nullptr; }
        for(const auto & region:guard.at("intervals").at("regions")) {
            int lo=region.at("lower_tokens"),hi=region.at("upper_tokens");
            if(nt<=lo || nt>=hi) { continue; }
            double w=double(nt-lo)/(hi-lo),curvature=4*w*(1-w);
            auto blend=[&](double a,double b) { return (1-w)*a+w*b; };
            const json * a=nullptr,*b=nullptr;
            for(const auto & n:guard.at("native")) { if(n.at("tokens")==lo) { a=&n; } if(n.at("tokens")==hi) { b=&n; } }
            json native={{"tokens",nt},{"native_us",blend(a->at("native_us"),b->at("native_us"))+curvature*region.at("native_bias_us").get<double>()},
                {"native_base_compute_us",blend(a->at("native_base_compute_us"),b->at("native_base_compute_us"))},
                {"uncertainty_us",std::max({a->at("uncertainty_us").get<double>(),b->at("uncertainty_us").get<double>(),region.at("native_uncertainty_us").get<double>()})}};
            std::map<std::string,const json *> by_id; for(const auto & p:guard.at("profiles")) { by_id[p.at("id").get<std::string>()]=&p; }
            json profiles=json::array();
            for(const auto & p:region.at("profiles")) {
                const auto & x=*by_id.at(p.at("lower_profile").get<std::string>()); const auto & y=*by_id.at(p.at("upper_profile").get<std::string>());
                std::vector<int> tails;
                for(int il=0;il<23;++il) {
                    int64_t numerator=int64_t(hi-nt)*x.at("tails").at(il).get<int>()+int64_t(nt-lo)*y.at("tails").at(il).get<int>();
                    tails.push_back(int((2*numerator+(hi-lo))/(2*int64_t(hi-lo))));
                }
                profiles.push_back({{"id",p.at("id")},{"tokens",nt},{"family",x.at("family")},{"policy",x.at("policy")},{"tails",tails},
                    {"bias_us",blend(x.at("bias_us"),y.at("bias_us"))+curvature*p.at("bias_us").get<double>()},
                    {"uncertainty_us",std::max({x.at("uncertainty_us").get<double>(),y.at("uncertainty_us").get<double>(),p.at("uncertainty_us").get<double>()})}});
            }
            return {{"native",native},{"profiles",profiles},{"interpolation",{{"id",region.at("id")},{"lower_tokens",lo},{"upper_tokens",hi},{"weight",w},{"curvature_weight",curvature}}}};
        }
        return nullptr;
    }
    json guarded_plan(const json & capture,const json & layers) const {
        int nt=capture.at("tokens"); const json * anchor=nullptr;
        for(const auto & n:guard.at("native")) { if(n.at("tokens")==nt) { anchor=&n; break; } }
        json interval; const json * available=&guard.at("profiles");
        if(!anchor) {
            interval=resolve_interval(nt);
            if(!interval.is_null()) { anchor=&interval.at("native"); available=&interval.at("profiles"); }
        }
        json report={{"tokens",nt},{"token_ids",capture.at("token_ids")},{"model_type","branch_critical_path"},
            {"execution_policy",{{"family","native"},{"overlap_gather",false},{"activation_lifetimes",true}}},
            {"layers",layers},{"custom_candidates",json::array()},
            {"scope","offline exact-length calibrated request templates; empirical error envelope; no migration"},
            {"path_decision",{{"custom_predicted_us",nullptr},{"custom_uncertainty_us",nullptr},{"native_predicted_us",nullptr},
                {"native_uncertainty_us",nullptr},{"native_lower_us",nullptr},{"reason","unsupported_length_or_schedule"}}}};
        for(auto & l:report["layers"]) { l["selected"]={{"name","npu_shared_npu"},{"owner",std::vector<int>(60,1)},{"shared",1}}; }
        if(!interval.is_null()) {
            report["interpolation"]=interval.at("interpolation");
            report["scope"]="offline interval-interpolated request templates; empirical error envelope; no migration";
        }
        std::vector<const json *> profiles; for(const auto & p:*available) {
            if(p.at("tokens")!=nt) { continue; } bool feasible=true;
            for(int il=0;il<23;++il) { int active=0; for(const auto & c:layers[il].at("counts")) { active+=c.get<int>()>0; }
                int tail=p.at("tails").at(il); if(tail && tail>=active) { feasible=false; break; } }
            if(feasible) { profiles.push_back(&p); }
        }
        if(!anchor || profiles.empty()) { finalize(report,0); return report; }
        std::array<std::vector<int>,24> counts,order;
        double native_base=0;
        for(int il=0;il<24;++il) {
            counts[il]=layers[il].at("counts").get<std::vector<int>>();
            for(int e=0;e<60;++e) { if(counts[il][e]) { order[il].push_back(e); } }
            std::sort(order[il].begin(),order[il].end(),[&](int a,int b) { return counts[il][a]!=counts[il][b]?counts[il][a]<counts[il][b]:a<b; });
            native_base+=base.predict(il==23?1:nt,counts[il],{"npu_shared_npu",std::vector<int>(60,1),1}).backend[1];
        }
        placement last; double last_cost=INFINITY;
        for(const auto & p:model_candidates(counts[23])) { double cost=base.predict(1,counts[23],p).total; if(cost<last_cost) { last=p; last_cost=cost; } }
        auto placement_for=[&](const json & profile,int il) {
            if(il==23) { return last; }
            std::string f=profile.at("family"); int tail=profile.at("tails").at(il);
            placement p={"guard_"+f+"_tail"+std::to_string(tail),std::vector<int>(60,1),shared(f)};
            for(int i=0;i<tail;++i) { p.owner[order[il][i]]=f=="cpu"?0:2; } return p;
        };
        const json * winner=nullptr; json best; double best_score=INFINITY;
        for(const json * profile:profiles) {
            const auto & p=*profile; std::string f=p.at("family"),policy=p.at("policy");
            double nominal=node.at("families").at(f).at(policy).at("outside_us");
            for(int il=0;il<24;++il) { nominal+=predict(il==23?1:nt,counts[il],placement_for(p,il),f,policy=="overlap").at("total_us").get<double>(); }
            double corrected=std::max(1.,nominal+p.at("bias_us").get<double>()),score=corrected+p.at("uncertainty_us").get<double>();
            json option={{"id",p.at("id")},{"family",f},{"policy",policy},{"nominal_us",nominal},{"predicted_us",corrected},{"bias_us",p.at("bias_us")},{"uncertainty_us",p.at("uncertainty_us")}};
            report["custom_candidates"].push_back(option);
            if(score<best_score) { best_score=score; winner=profile; best=option; }
        }
        for(int il=0;il<24;++il) {
            auto p=placement_for(*winner,il); auto & l=report["layers"][il];
            l["selected"]={{"name",p.name},{"owner",p.owner},{"shared",p.shared}};
            l["prediction"]=predict(il==23?1:nt,counts[il],p,best.at("family"),best.at("policy")=="overlap");
        }
        double native=std::max(1.,anchor->at("native_us").get<double>()+native_base-anchor->at("native_base_compute_us").get<double>());
        report["selected_profile"]=best.at("id"); report["native_base_compute_us"]=native_base;
        report["execution_policy"]={{"family",best.at("family")},{"overlap_gather",best.at("policy")=="overlap"},{"activation_lifetimes",true}};
        report["path_decision"]={{"custom_predicted_us",best.at("predicted_us")},{"custom_uncertainty_us",best.at("uncertainty_us")},
            {"native_predicted_us",native},{"native_uncertainty_us",anchor->at("uncertainty_us")},{"native_lower_us",native-anchor->at("uncertainty_us").get<double>()},{"reason",interval.is_null()?"calibrated_template_error_guard":"calibrated_interval_error_guard"}};
        finalize(report,0); return report;
    }
    json plan(const json & capture,const json & layers) const {
        require(capture.at("tokens").is_number_integer(),"invalid fullmodel length");
        int tokens=capture.at("tokens"); require(tokens>1 && capture.at("token_ids").size()==size_t(tokens) && layers.size()==24,"invalid fullmodel route shape");
        for(int il=0;il<24;++il) {
            require(layers[il].at("layer")==il && layers[il].at("tokens")== (il==23?1:tokens),"invalid fullmodel layer shape");
            const auto & counts=layers[il].at("counts"); require(counts.size()==60,"invalid route counts"); int total=0;
            for(const auto & c:counts) { require(c.is_number_integer() && c.get<int>()>=0,"invalid route count"); total+=c.get<int>(); }
            require(total==4*(il==23?1:tokens),"invalid route total");
        }
        if(!guard.is_null()) { return guarded_plan(capture,layers); }
        struct candidate {
            placement p;
            std::string family;
            resident_placement_model::prediction plain;
            std::array<double,3> gather;
        };
        std::array<std::vector<candidate>,24> cache;
        std::array<std::vector<int>,24> counts;
        std::array<int,24> lengths{};
        double native_compute=0;
        for(int il=0;il<24;++il) {
            lengths[il]=layers[il].at("tokens"); counts[il]=layers[il].at("counts").get<std::vector<int>>();
            native_compute+=base.predict(lengths[il],counts[il],{"npu_shared_npu",std::vector<int>(60,1),1}).backend[1];
            for(const auto & p:model_candidates(counts[il])) {
                auto f=fullmodel_context_model::family(p);
                if(lengths[il]>1 && (f.empty() || p.shared!=shared(f))) { continue; }
                cache[il].push_back({p,f,base.predict(lengths[il],counts[il],p),gather_mib(lengths[il],counts[il],p)});
            }
        }
        // Evaluate numeric costs only. Route IDs and JSON reports are materialized
        // once for the winning request policy, not six times inside the search.
        json options=json::array(),best; double best_score=INFINITY;
        std::array<size_t,24> best_indices{};
        for(const char * f:{"cpu","gpu","branch"}) { for(bool overlap:{false,true}) {
            const auto & c=node.at("families").at(f).at(overlap?"overlap":"barrier"); double cost=c.at("outside_us");
            auto scales=c.at("compute_scales").get<std::array<double,3>>(),rates=c.at("gather_us_per_mib").get<std::array<double,3>>();
            double host=c.at("host_us_per_token"); std::array<size_t,24> indices{};
            for(int il=0;il<24;++il) {
                int nt=lengths[il]; double fastest=INFINITY;
                for(size_t i=0;i<cache[il].size();++i) {
                    const auto & x=cache[il][i]; if(nt>1 && x.family!=f) { continue; }
                    double total=x.plain.total;
                    if(nt>1) {
                        auto compute=x.plain.backend,gather=x.gather;
                        for(int b=0;b<3;++b) { compute[b]*=scales[b]; gather[b]*=rates[b]; }
                        total=critical(gather,compute,overlap)+nt*host;
                    }
                    if(total<fastest) { fastest=total; indices[il]=i; }
                }
                require(std::isfinite(fastest),"no eligible execution candidate"); cost+=fastest;
            }
            json option={{"family",f},{"policy",overlap?"overlap":"barrier"},{"predicted_us",cost},{"uncertainty_us",c.at("uncertainty_us")},{"outside_us",c.at("outside_us")}};
            options.push_back(option); double score=cost+c.at("uncertainty_us").get<double>();
            if(score<best_score) { best_score=score; best=option; best_indices=indices; }
        } }
        auto decision=decide(best.at("predicted_us"),native_compute,best.at("uncertainty_us"),0);
        json report={{"tokens",tokens},{"token_ids",capture.at("token_ids")},{"model_type","branch_critical_path"},
            {"execution_mode",decision.at("execution_mode")},{"path_decision",decision},{"native_base_compute_us",native_compute},
            {"execution_policy",{{"family",best.at("family")},{"overlap_gather",best.at("policy")=="overlap"},{"activation_lifetimes",true}}},
            {"scope","offline pre-load; one calibrated family/policy per request, layer-specific owners; no migration"}};
        report["custom_candidates"]=std::move(options);
        report["layers"]=layers;
        for(int il=0;il<24;++il) {
            auto & row=report["layers"][il]; const auto & p=cache[il][best_indices[il]].p;
            row["selected"]={{"name",p.name},{"owner",p.owner},{"shared",p.shared}};
            row["prediction"]=predict(lengths[il],counts[il],p,best.at("family"),best.at("policy")=="overlap");
        }
        return report;
    }
};
