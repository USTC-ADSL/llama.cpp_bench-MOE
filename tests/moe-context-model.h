#pragma once
#include "moe-placement-model.h"

// Full-model correction is a small table over the original cost model. Path
// selection is before preparation, never a migration to unresident weights.
struct fullmodel_context_model {
    resident_placement_model base;
    json node;
    double decision_budget;
    static std::string family(const placement & p) {
        bool cpu=std::find(p.owner.begin(),p.owner.end(),0)!=p.owner.end();
        bool npu=std::find(p.owner.begin(),p.owner.end(),1)!=p.owner.end();
        bool gpu=std::find(p.owner.begin(),p.owner.end(),2)!=p.owner.end();
        if(cpu && npu) { return "cpu"; }
        if(gpu && npu) { return "gpu"; }
        if(npu && !cpu && !gpu && p.shared==2) { return "branch"; }
        return "";
    }
    fullmodel_context_model(const json & model,int tokens) : base(model.at("base")),decision_budget(model.at("decision_budget_us")) {
        require(model.at("version")==1,"invalid context model version");
        const auto & nodes=model.at("nodes");
        require(nodes.size()>=2 && tokens>=nodes.front().at("tokens").get<int>() &&
                tokens<=nodes.back().at("tokens").get<int>(),"length outside calibrated context range");
        for(size_t i=1;i<nodes.size();++i) {
            const auto & a=nodes[i-1]; const auto & b=nodes[i];
            int lo=a.at("tokens"),hi=b.at("tokens"); require(hi>lo,"unsorted context nodes");
            if(tokens<lo || tokens>hi) { continue; }
            double w=double(tokens-lo)/(hi-lo);
            auto lerp=[&](const json & x,const json & y) { return (1-w)*x.get<double>()+w*y.get<double>(); };
            node["tokens"]=tokens;
            for(const char * key:{"native_us","native_uncertainty_us","native_base_compute_us"}) { node[key]=lerp(a.at(key),b.at(key)); }
            for(const char * f:{"cpu","gpu","branch"}) {
                auto & dst=node["families"][f]; const auto & x=a.at("families").at(f); const auto & y=b.at("families").at(f);
                for(const char * key:{"outside_us","uncertainty_us"}) { dst[key]=lerp(x.at(key),y.at(key)); }
                dst["scales"]=json::array();
                for(int k=0;k<4;++k) { double scale=lerp(x.at("scales").at(k),y.at("scales").at(k));
                    require(std::isfinite(scale) && scale>0,"invalid context scale"); dst["scales"].push_back(scale); }
            }
            break;
        }
        require(!node.empty(),"missing context node");
    }
    struct prediction {
        resident_placement_model::prediction ffn;
        double outside=0,uncertainty=0,score=0;
    };
    prediction predict(int nt,const std::vector<int> & counts,const placement & p) const {
        prediction out; out.ffn=base.predict(nt,counts,p);
        if(nt>1) {
            auto f=family(p); require(!f.empty(),"candidate family not calibrated");
            const auto & c=node.at("families").at(f);
            for(int b=0;b<3;++b) { out.ffn.backend[b]*=c.at("scales").at(b).get<double>(); }
            out.ffn.host*=c.at("scales").at(3).get<double>();
            out.ffn.total=*std::max_element(out.ffn.backend.begin(),out.ffn.backend.end())+out.ffn.host;
            out.outside=c.at("outside_us").get<double>()/23;
            out.uncertainty=c.at("uncertainty_us").get<double>()/23;
        }
        out.score=out.ffn.total+out.outside;
        return out;
    }
    placement choose(int nt,const std::vector<int> & counts,prediction & predicted) const {
        double best=INFINITY; placement chosen;
        for(const auto & p:model_candidates(counts)) {
            if(nt>1 && family(p).empty()) { continue; }
            auto est=predict(nt,counts,p);
            if(est.score<best) { best=est.score; chosen=p; predicted=est; }
        }
        require(std::isfinite(best),"no calibrated offline candidate");
        return chosen;
    }
    json decide(double custom,double native_compute,double uncertainty,double observed_decision_us=0) const {
        double native=std::max(1.,node.at("native_us").get<double>()+native_compute-node.at("native_base_compute_us").get<double>());
        double budget=std::max(decision_budget,observed_decision_us);
        double upper=custom+uncertainty+budget,lower=native-node.at("native_uncertainty_us").get<double>();
        return {{"execution_mode",upper<lower?"custom":"npu"},{"custom_predicted_us",custom},{"native_predicted_us",native},
                {"custom_upper_us",upper},{"native_lower_us",lower},{"custom_uncertainty_us",uncertainty},
                {"native_uncertainty_us",node.at("native_uncertainty_us")},{"decision_budget_us",budget},
                {"reserved_decision_budget_us",decision_budget},{"observed_planning_us",observed_decision_us}};
    }
};
