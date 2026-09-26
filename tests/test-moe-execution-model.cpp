#include <algorithm>
#include <array>
#include <cmath>
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
#include "moe-execution-model.h"
int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: execution-model-test MODEL ROUTES");
        require(fullmodel_execution_model::critical({3,5,0},{10,4,0},false)==18,"barrier critical path");
        require(fullmodel_execution_model::critical({3,5,0},{10,4,0},true)==13,"overlap critical path");
        std::ifstream m(argv[1]),r(argv[2]); json model,routes; m>>model; r>>routes;
        fullmodel_execution_model execution(model,routes.at("tokens"));
        std::cout<<execution.plan(routes,routes.at("layers")).dump()<<'\n';
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
