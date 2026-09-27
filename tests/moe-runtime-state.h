#pragma once
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <sched.h>

// Optional read-only telemetry. Accelerator temperature is never labelled as
// its clock. Missing/permission-denied metrics remain null. No root writes.
struct moe_runtime_state {
    bool enabled=false;
    double setup_us=0;
    std::vector<std::pair<std::string,std::string>> sensors;
    static double now_us() {
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static json number(const std::string & path,double divisor=1) {
        std::ifstream in(path); double x;
        if(!(in>>x) || !std::isfinite(x)) { return nullptr; }
        return x/divisor;
    }
    static json residency(int policy) {
        std::ifstream in("/sys/devices/system/cpu/cpufreq/policy"+std::to_string(policy)+"/stats/time_in_state");
        json values=json::array(); uint64_t khz,ticks;
        while(in>>khz>>ticks) { values.push_back({khz,ticks}); }
        return values.empty()?json(nullptr):values;
    }
    explicit moe_runtime_state(bool use):enabled(use) {
        if(!enabled) { return; } double start=now_us();
        const std::map<std::string,std::string> wanted={{"cpu-0-0-0","cpu0_c"},{"cpu-1-0-0","cpu6_c"},
            {"gpuss-0","gpu_c"},{"nsphvx-0","hvx_c"},{"nsphmx-0","hmx_c"},{"ddr","ddr_c"},
            {"cpu_therm","soc_c"},{"quiet_therm","skin_c"},{"battery","battery_c"}};
        if(auto * dir=opendir("/sys/class/thermal")) {
            while(auto * entry=readdir(dir)) {
                std::string name=entry->d_name; if(name.find("thermal_zone")!=0) { continue; }
                std::string path="/sys/class/thermal/"+name; std::ifstream in(path+"/type"); std::string type; in>>type;
                auto it=wanted.find(type); if(it!=wanted.end()) { sensors.push_back({it->second,path+"/temp"}); }
            }
            closedir(dir);
        }
        std::sort(sensors.begin(),sensors.end()); setup_us=now_us()-start;
    }
    json sample() const {
        if(!enabled) { return nullptr; } double start=now_us();
        json values=json::object();
        for(int p:{0,6}) {
            std::string prefix="cpu"+std::to_string(p),path="/sys/devices/system/cpu/cpufreq/policy"+std::to_string(p)+"/";
            values[prefix+"_mhz"]=number(path+"scaling_cur_freq",1000);
            values[prefix+"_max_mhz"]=number(path+"scaling_max_freq",1000);
            values[prefix+"_hw_max_mhz"]=number(path+"cpuinfo_max_freq",1000);
        }
        for(const char * name:{"cpu0_c","cpu6_c","gpu_c","hvx_c","hmx_c","ddr_c","soc_c","skin_c","battery_c"}) { values[name]=nullptr; }
        for(const auto & s:sensors) { values[s.first]=number(s.second,1000); }
        values["load1"]=number("/proc/loadavg"); values["cpu_id"]=sched_getcpu();
        json state={{"monotonic_us",start},{"features",values},
            {"cpu_residency_khz_ticks",{residency(0),residency(6)}},
            {"gpu_clock_mhz",number("/sys/class/kgsl/kgsl-3d0/gpuclk",1000000)},
            {"npu_clock_mhz",nullptr}};
        state["observation_us"]=now_us()-start; return state;
    }
};
