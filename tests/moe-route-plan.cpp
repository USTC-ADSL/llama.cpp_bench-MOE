// Host-only planning/calibration utility; shares the exact device search code.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "json.hpp"
using json=nlohmann::ordered_json;
static void require(bool ok,const char * message) { if(!ok) { throw std::runtime_error(message); } }
struct placement { std::string name; std::vector<int> owner; int shared; };
#include "moe-placement-model.h"
#include "moe-route-scheduler.h"

static json plan_json(const placement & p) { return {{"name",p.name},{"owner",p.owner},{"shared",p.shared}}; }

#ifndef MOE_ROUTE_PLAN_NO_MAIN
int main(int argc,char ** argv) {
    try {
        require(argc==5,"usage: moe-route-plan MODEL ROUTES OUTPUT profile|provision");
        json parameters,routes; std::ifstream(argv[1])>>parameters; std::ifstream(argv[2])>>routes;
        resident_placement_model model(parameters);
        const auto algorithm=parameters.value("algorithm",std::string("generic"));
        require(algorithm=="generic" || algorithm=="shared_additive","unsupported route search algorithm");
        moe_route_scheduler search(model,parameters.value("search_budget",2048),algorithm=="shared_additive");
        resident_placement_model::availability all;
        for(auto & row:all.experts) { row.fill(true); } all.shared.fill(true);
        std::string mode=argv[4]; require(mode=="profile" || mode=="provision","invalid plan utility mode");
        json out={{"tokens",routes.at("tokens")},{"token_ids",routes.at("token_ids")},{"layers",json::array()}};
        for(const auto & l:routes.at("layers")) {
            std::vector<int> counts(60); int nt=l.at("tokens");
            if(l.contains("counts")) { counts=l.at("counts").get<std::vector<int>>(); }
            else { for(int id:l.at("ids").get<std::vector<int>>()) { require(id>=0 && id<60,"invalid route ID"); ++counts[id]; } }
            auto picked=search.choose(nt,counts,all);
            json item={{"layer",l.at("layer")},{"tokens",nt},{"counts",counts},{"selected",plan_json(picked.selected)}};
            if(l.contains("ids")) { item["ids"]=l.at("ids"); }
            if(mode=="profile") {
                // Independent calibration placements, not just the optimizer's
                // favored points. Cover shared choices and three-way fractions.
                auto seeds=moe_route_scheduler::seeds(counts);
                std::vector<placement> candidates;
                for(size_t i=0;i<9;++i) { candidates.push_back(seeds[i]); }
                int threeway=0;
                for(const auto & p:seeds) { if(p.name=="threeway_seed") {
                    int index=threeway++;
                    bool tail_selected=nt>=512?index>=18:(index<9 || (index>=18 && index<36));
                    if(tail_selected && index%9<6) { candidates.push_back(p); }
                } }
                candidates.push_back(picked.selected);
                std::set<std::vector<int>> seen;
                item["candidates"]=json::array();
                for(auto p:candidates) {
                    auto key=p.owner; key.push_back(p.shared); if(!seen.insert(key).second) { continue; }
                    p.name="calibration_"+std::to_string(item["candidates"].size()); item["candidates"].push_back(plan_json(p));
                }
            } else {
                resident_placement_model::availability available;
                // Prepare the compact winner and six promising alternative
                // expert layouts, plus competitive shared choices. This union
                // avoids three complete weight copies. It is NOT universal residency.
                for(int e=0;e<60;++e) { available.experts[e][picked.selected.owner[e]]=true; }
                // Do not keep large shared copies that cannot compete even after
                // reoptimizing all routed owners. The 20% preparation margin is
                // deliberately wider than held-length model MAPE (~9..11%).
                // This is an explicit residency policy, not an online free copy.
                std::array<double,3> shared_scores{};
                for(int s=0;s<3;++s) {
                    auto constrained=all; constrained.shared.fill(false); constrained.shared[s]=true;
                    shared_scores[s]=search.choose(nt,counts,constrained).predicted.total;
                    available.shared[s]=shared_scores[s]<=picked.predicted.total*1.20;
                }
                available.shared[picked.selected.shared]=true;
                item["shared_preparation_scores_us"]=shared_scores;
                item["shared_preparation_margin"]=1.20;
                std::vector<std::pair<double,std::pair<int,int>>> moves;
                for(int e=0;e<60;++e) { if(counts[e]) { for(int b=0;b<3;++b) { if(b!=picked.selected.owner[e]) {
                    auto p=picked.selected; p.owner[e]=b;
                    moves.push_back({model.predict(nt,counts,p).total,{e,b}});
                } } } }
                std::sort(moves.begin(),moves.end());
                for(size_t i=0;i<std::min<size_t>(6,moves.size());++i) { available.experts[moves[i].second.first][moves[i].second.second]=true; }
                item["resident"]={{"experts",available.experts},{"shared",available.shared}};
                std::array<int,3> widths{};
                for(auto row:available.experts) { for(int b=0;b<3;++b) { widths[b]+=row[b]; } }
                auto final=search.choose(nt,counts,available,&widths);
                item["selected"]=plan_json(final.selected);
                item["predicted_backend_us"]=final.predicted.backend;
                item["predicted_total_us"]=final.predicted.total;
                item["resident_width"]=widths;
            }
            out["layers"].push_back(item);
        }
        out["scope"]=mode=="profile"?"development calibration placements":"route-conditioned resident discovery only; actual timed callbacks reschedule using actual IDs and explicit availability";
        std::ofstream output(argv[3]); output<<out.dump(2)<<'\n'; require(output.good(),"plan write failed");
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
#endif
