// Whole-expert search on the current route. No timing lookup, weight IO or
// backend calls here. Backend order: CPU, HTP, OpenCL.
#pragma once
#include <set>

struct moe_route_scheduler {
    const resident_placement_model & model;
    const resident_placement_model * single_token_model=nullptr;
    int budget=2048;
    bool additive=false;
    struct result {
        placement selected;
        resident_placement_model::prediction predicted;
        int evaluations=0;
        double seed_us=0;
    };
    explicit moe_route_scheduler(const resident_placement_model & m,int limit=2048,bool fast=false,
                                 const resident_placement_model * nt1=nullptr)
        : model(m),single_token_model(nt1),budget(limit),additive(fast) {
        require(limit>=128 && limit<=16384,"invalid route search budget");
        if(additive) {
            validate_additive(model);
            if(single_token_model) { validate_additive(*single_token_model); }
        }
    }
    static void validate_additive(const resident_placement_model & m) {
        for(const auto & row:m.coefficients) { for(size_t i=0;i<row.size();++i) {
            require(row[i]>=0 && (i==0 || i==1 || i==2 || i==4 || i==12 || i==13 || i==14 || row[i]==0),
                    "additive scheduler requires nonnegative additive coefficients");
        } }
        for(double w:m.host_coefficients) { require(w==0,"additive scheduler requires zero host coefficients"); }
    }
    const resident_placement_model & model_for(int nt) const {
        return nt==1 && single_token_model?*single_token_model:model;
    }
    result choose_additive(int nt,const std::vector<int> & counts,
                           const resident_placement_model::availability & available,
                           const std::array<int,3> * widths) const {
        const auto & model=model_for(nt);
        const auto scale=model.context_scale(nt);
        std::array<std::array<double,3>,60> costs{};
        std::vector<int> order;
        for(int e=0;e<60;++e) { if(counts[e]) {
            order.push_back(e);
            for(int b=0;b<3;++b) {
                const auto & w=model.coefficients[b];
                costs[e][b]=(w[1]*counts[e]/1024.0+w[2]/60.0+w[4]*((counts[e]+31)/32)/32.0)*scale[b];
            }
        } }
        auto fastest=[&](int e) {
            double value=INFINITY;
            for(int b=0;b<3;++b) { if(available.experts[e][b]) { value=std::min(value,costs[e][b]); } }
            return value;
        };
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return fastest(a)>fastest(b); });
        using loads_type=std::array<double,3>;
        auto score=[](const loads_type & loads) {
            return std::make_pair(*std::max_element(loads.begin(),loads.end()),
                                  std::accumulate(loads.begin(),loads.end(),0.0));
        };
        auto better=[](const auto & a,const auto & b) {
            return a.first+1e-6<b.first || (std::abs(a.first-b.first)<=1e-6 && a.second+1e-6<b.second);
        };
        result best; best.predicted.total=INFINITY;
        auto best_score=std::make_pair(double(INFINITY),double(INFINITY));
        int shared_options=std::accumulate(available.shared.begin(),available.shared.end(),0);
        for(int s=0;s<3;++s) { if(available.shared[s]) {
            placement current{"current_route_additive",std::vector<int>(60,1),s};
            for(int e=0;e<60;++e) { for(int b:{1,0,2}) {
                if(available.experts[e][b]) { current.owner[e]=b; break; }
            } }
            loads_type loads{}; std::array<int,3> active{};
            // nt is fixed during this search. Shared length/tile terms are a
            // constant for each shared-owner candidate, so moves stay O(1).
            const auto & w=model.coefficients[s];
            loads[s]=(w[0]+w[12]+w[13]*nt/512.0+w[14]*((nt+31)/32)/16.0)*scale[s]; active[s]=1;
            for(int e:order) {
                int chosen=-1; loads_type chosen_loads{};
                auto chosen_score=std::make_pair(double(INFINITY),double(INFINITY));
                for(int b=0;b<3;++b) { if(available.experts[e][b]) {
                    auto trial=loads; trial[b]+=costs[e][b]+(active[b]?0:model.coefficients[b][0]*scale[b]);
                    auto candidate=score(trial);
                    if(better(candidate,chosen_score)) { chosen=b; chosen_score=candidate; chosen_loads=trial; }
                } }
                require(chosen>=0,"no additive expert placement");
                current.owner[e]=chosen; ++active[chosen]; loads=chosen_loads;
            }
            auto current_score=score(loads);
            best.seed_us=best.seed_us?std::min(best.seed_us,current_score.first):current_score.first;
            int evaluations=0,limit=budget/shared_options;
            // Each move/swap updates three scalar loads; no feature-vector or
            // placement allocation in the search loop. Strict descent is bounded.
            for(int iteration=0;iteration<32 && evaluations<limit;++iteration) {
                auto next_score=current_score; loads_type next_loads=loads;
                int move_e=-1,move_b=-1,swap_f=-1;
                for(int e:order) { for(int b=0;b<3 && evaluations<limit;++b) {
                    int a=current.owner[e];
                    if(a==b || !available.experts[e][b]) { continue; }
                    auto trial=loads;
                    trial[a]-=costs[e][a]+(active[a]==1?model.coefficients[a][0]*scale[a]:0);
                    trial[b]+=costs[e][b]+(active[b]?0:model.coefficients[b][0]*scale[b]);
                    ++evaluations; auto candidate=score(trial);
                    if(better(candidate,next_score)) { next_score=candidate; next_loads=trial; move_e=e; move_b=b; }
                } }
                if(move_e<0) { for(size_t i=0;i<order.size() && evaluations<limit;++i) {
                    int e=order[i],a=current.owner[e];
                    for(size_t j=i+1;j<order.size() && evaluations<limit;++j) {
                        int f=order[j],b=current.owner[f];
                        if(a==b || !available.experts[e][b] || !available.experts[f][a]) { continue; }
                        auto trial=loads; trial[a]+=costs[f][a]-costs[e][a]; trial[b]+=costs[e][b]-costs[f][b];
                        ++evaluations; auto candidate=score(trial);
                        if(better(candidate,next_score)) { next_score=candidate; next_loads=trial; move_e=e; move_b=b; swap_f=f; }
                    }
                } }
                if(move_e<0) { break; }
                int old=current.owner[move_e]; current.owner[move_e]=move_b;
                if(swap_f>=0) { current.owner[swap_f]=old; }
                else { --active[old]; ++active[move_b]; }
                loads=next_loads; current_score=next_score;
            }
            best.evaluations+=evaluations;
            if(better(current_score,best_score)) { best_score=current_score; best.selected=current; }
        } }
        require(feasible(best.selected,available),"additive search returned nonresident placement");
        best.predicted=model.predict(nt,counts,best.selected,widths);
        require(std::abs(best.predicted.total-std::max(1.,best_score.first))<=1e-6*std::max(1.,best.predicted.total),
                "additive search score differs from full prediction");
        return best;
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
        if(additive) { return choose_additive(nt,counts,available,widths); }
        const auto & model=model_for(nt);
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
