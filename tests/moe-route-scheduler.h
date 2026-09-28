// Whole-expert search on the current route. No timing lookup, weight IO or
// backend calls here. Backend order: CPU, HTP, OpenCL.
#pragma once
#include <set>

struct moe_route_scheduler {
    const resident_placement_model & model;
    int budget=2048;
    struct result {
        placement selected;
        resident_placement_model::prediction predicted;
        int evaluations=0;
        double seed_us=0;
    };
    explicit moe_route_scheduler(const resident_placement_model & m,int limit=2048) : model(m),budget(limit) {
        require(limit>=128 && limit<=16384,"invalid route search budget");
    }
    static bool feasible(const placement & p,const resident_placement_model::availability & a) {
        if(!a.shared[p.shared]) { return false; }
        for(int e=0;e<60;++e) { if(!a.experts[e][p.owner[e]]) { return false; } }
        return true;
    }
    // Diverse initial points cover two/three-way groups. These are only seeds:
    // subsequent moves may put any eligible expert on any eligible backend.
    static std::vector<placement> seeds(const std::vector<int> & counts) {
        auto out=model_candidates(counts);
        std::vector<int> order(60); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return counts[a]<counts[b]; });
        for(int tail:{1,2,4,8,16}) { for(double fraction:{.15,.30,.45}) { for(int shared=0;shared<3;++shared) {
            placement p{"threeway_seed",std::vector<int>(60,1),shared};
            int used=0,remaining=0;
            for(int e:order) { if(counts[e] && used<tail) { p.owner[e]=0; ++used; } }
            for(int e:order) { if(p.owner[e]==1) { remaining+=counts[e]; } }
            double target=remaining*fraction; int assigned=0;
            for(auto it=order.rbegin();it!=order.rend();++it) {
                int e=*it;
                if(p.owner[e]==1 && counts[e] && std::abs(assigned+counts[e]-target)<std::abs(assigned-target)) {
                    p.owner[e]=2; assigned+=counts[e];
                }
            }
            out.push_back(std::move(p));
        } } }
        return out;
    }
    result choose(int nt,const std::vector<int> & counts,const resident_placement_model::availability & available,
                  const std::array<int,3> * widths=nullptr) const {
        require(counts.size()==60 && nt>0 && std::accumulate(counts.begin(),counts.end(),0)==4*nt,
                "invalid current route counts");
        for(int n:counts) { require(n>=0 && n<=nt,"invalid per-expert top-k count"); }
        for(const auto & row:available.experts) { require(row[0] || row[1] || row[2],"expert has no resident layout"); }
        require(available.shared[0] || available.shared[1] || available.shared[2],"shared has no resident layout");
        if(widths) { for(int b=0;b<3;++b) {
            int count=0; for(const auto & row:available.experts) { count+=row[b]; }
            require((*widths)[b]==count,"resident bank width mismatch");
        } }
        result best; best.predicted.total=INFINITY;
        std::vector<std::pair<double,placement>> ranked;
        std::set<std::vector<int>> seen;
        for(auto p:seeds(counts)) {
            // Project seeds onto explicitly available layouts, preferring NPU.
            for(int e=0;e<60;++e) { if(!available.experts[e][p.owner[e]] || !counts[e]) {
                for(int b:{1,0,2}) { if(available.experts[e][b]) { p.owner[e]=b; break; } }
            } }
            if(!available.shared[p.shared]) { continue; }
            auto key=p.owner; key.push_back(p.shared);
            if(!seen.insert(key).second) { continue; }
            auto pred=model.predict(nt,counts,p,widths); ++best.evaluations;
            ranked.push_back({pred.total,p});
            if(pred.total<best.predicted.total) { best.selected=p; best.predicted=pred; }
        }
        require(!ranked.empty(),"no feasible initial route placement");
        std::stable_sort(ranked.begin(),ranked.end(),[](const auto & a,const auto & b) { return a.first<b.first; });
        best.seed_us=best.predicted.total;
        // Two starts, each with a bounded share of the budget. Strict improvement
        // and deterministic traversal make equal-load ties stable across runs.
        for(size_t restart=0;restart<std::min<size_t>(2,ranked.size());++restart) {
            placement current=ranked[restart].second;
            double current_score=ranked[restart].first;
            int stop=restart==0?budget/2:budget;
            for(int iteration=0;iteration<32 && best.evaluations<stop;++iteration) {
                placement next=current; double next_score=current_score;
                auto consider=[&](const placement & p) {
                    if(best.evaluations>=stop || !feasible(p,available)) { return; }
                    auto pred=model.predict(nt,counts,p,widths); ++best.evaluations;
                    if(pred.total+1e-6<next_score) { next=p; next_score=pred.total; }
                    if(pred.total+1e-6<best.predicted.total) { best.selected=p; best.predicted=pred; }
                };
                for(int s=0;s<3;++s) { if(s!=current.shared) { auto p=current; p.shared=s; consider(p); } }
                for(int e=0;e<60;++e) { if(counts[e]) { for(int b=0;b<3;++b) { if(b!=current.owner[e]) {
                    auto p=current; p.owner[e]=b; consider(p);
                } } } }
                // Swaps escape critical-path plateaus where either single move
                // makes the receiving backend slower. Only needed at a move minimum.
                if(next_score==current_score) {
                    // Shared and routed work compete on the same branch; moving
                    // either alone can be worse even when the joint move wins.
                    for(int e=0;e<60 && best.evaluations<stop;++e) { if(counts[e]) {
                        for(int b=0;b<3;++b) { if(b!=current.owner[e]) {
                            for(int s=0;s<3;++s) { if(s!=current.shared) {
                                auto p=current; p.owner[e]=b; p.shared=s; consider(p);
                            } }
                        } }
                    } }
                    for(int e=0;e<60 && best.evaluations<stop;++e) { if(counts[e]) {
                        for(int f=e+1;f<60 && best.evaluations<stop;++f) {
                            if(counts[f] && current.owner[e]!=current.owner[f] && counts[e]!=counts[f]) {
                                auto p=current; std::swap(p.owner[e],p.owner[f]); consider(p);
                            }
                        }
                    } }
                }
                if(next_score==current_score) { break; }
                current=std::move(next); current_score=next_score;
            }
        }
        best.selected.name="current_route_search";
        require(feasible(best.selected,available),"search returned nonresident placement");
        return best;
    }
};
