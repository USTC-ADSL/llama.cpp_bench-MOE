#define MOE_ROUTE_PLAN_NO_MAIN
#include "moe-route-plan.cpp"

int main() {
    // Six active experts allow exhaustive verification against 3^6 * 3
    // assignments, including a three-way optimum absent from old families.
    json j={{"version",1},{"backends",std::vector<std::vector<double>>(3,std::vector<double>(26))},
            {"host",std::vector<double>(8)}};
    for(int b=0;b<3;++b) { j["backends"][b][1]=(b==0?3.:b==1?1.:2.)*1024; j["backends"][b][13]=(b==0?2.:b==1?1.:3.)*512; }
    resident_placement_model model(j); moe_route_scheduler search(model);
    resident_placement_model::availability all;
    for(auto & row:all.experts) { row.fill(true); } all.shared.fill(true);
    std::vector<int> counts(60); counts[0]=8; counts[1]=8; counts[2]=6; counts[3]=4; counts[4]=4; counts[5]=2;
    auto r=search.choose(8,counts,all);
    double optimal=INFINITY;
    for(int code=0;code<729;++code) { for(int s=0;s<3;++s) {
        placement p{"exhaustive",std::vector<int>(60,1),s}; int v=code;
        for(int e=0;e<6;++e) { p.owner[e]=v%3; v/=3; }
        optimal=std::min(optimal,model.predict(8,counts,p).total);
    } }
    require(r.predicted.total==optimal,"small exhaustive reference optimum missed");
    require(r.evaluations<=search.budget && r.predicted.total<=r.seed_us,"search bound or improvement violated");
    require(r.predicted.total==*std::max_element(r.predicted.backend.begin(),r.predicted.backend.end()),"parallel costs added");
    auto changed=counts; std::swap(changed[0],changed[2]);
    auto other=search.choose(8,changed,all);
    bool differs=false; for(int e=0;e<6;++e) { differs|=other.selected.owner[e]!=r.selected.owner[e]; }
    require(differs,"decision ignores current route");
    auto npu=all; for(auto & row:npu.experts) { row={false,true,false}; } npu.shared={false,true,false};
    auto restricted=search.choose(8,counts,npu);
    require(restricted.selected.shared==1 && std::count(restricted.selected.owner.begin(),restricted.selected.owner.end(),1)==60,"nonresident assignment");
    npu.experts[59]={true,false,false}; // inactive experts also need legal metadata
    require(search.choose(8,counts,npu).selected.owner[59]==0,"inactive expert projection is not resident");
    for(int b=0;b<3;++b) {
        auto only=all; only.shared.fill(false); only.shared[b]=true;
        require(search.choose(8,counts,only).selected.shared==b,"shared backend unavailable in search");
    }
    bool rejected=false;
    try { auto broken=all; broken.experts[4].fill(false); search.choose(8,counts,broken); }
    catch(const std::runtime_error &) { rejected=true; }
    require(rejected,"missing layout accepted");
    auto repeated=search.choose(8,counts,all);
    require(repeated.selected.owner==r.selected.owner && repeated.selected.shared==r.selected.shared,"nondeterministic search");
    // Physical bank width matters even when ownership is unchanged.
    j["backends"][2][7]=600.; resident_placement_model wide_model(j);
    placement p{"width",std::vector<int>(60,1),2}; std::array<int,3> width={0,60,20};
    require(wide_model.predict(8,counts,p,&width).backend[2]-wide_model.predict(8,counts,p).backend[2]==200.,"bank width omitted");
    std::cout<<"route scheduler contracts passed; exhaustive optimum="<<optimal<<" evaluations="<<r.evaluations<<'\n';
}
