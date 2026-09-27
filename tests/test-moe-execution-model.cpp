#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "json.hpp"
using json=nlohmann::ordered_json;
struct placement { std::string name; std::vector<int> owner; int shared; };
static void require(bool yes,const char * message) { if(!yes) { throw std::runtime_error(message); } }
#ifndef MOE_EXECUTION_MODEL_HEADER
#define MOE_EXECUTION_MODEL_HEADER "moe-execution-model.h"
#endif
#include MOE_EXECUTION_MODEL_HEADER
#include "moe-state-correction.h"
int main(int argc,char ** argv) {
    try {
        require(argc==3 || argc==4,"usage: execution-model-test MODEL ROUTES [REPETITIONS]");
        require(fullmodel_execution_model::critical({3,5,0},{10,4,0},false)==18,"barrier critical path");
        require(fullmodel_execution_model::critical({3,5,0},{10,4,0},true)==13,"overlap critical path");
        std::ifstream m(argv[1]),r(argv[2]); json model,routes; m>>model; r>>routes;
        int reps=argc==4?std::stoi(argv[3]):1; require(reps>0,"invalid repetitions");
        json plan,samples=json::array();
        for(int i=0;i<reps;++i) {
            auto start=std::chrono::steady_clock::now();
            fullmodel_execution_model execution(model,routes.at("tokens"));
            plan=execution.plan(routes,routes.at("layers"));
            if(model.contains("runtime_state_correction")) {
                apply_moe_state_correction(plan,model.at("runtime_state_correction"),routes.value("decision_state",json()),routes.value("decision_now_us",0.));
            }
            execution.finalize(plan,0);
            samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        }
        plan["decision_benchmark_us"]=samples;
        std::cout<<plan.dump()<<'\n';
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
