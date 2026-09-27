#pragma once

// Correct a single calibrated custom/native pair. The caller finalizes the
// decision after charging sampling and planning costs. Does not move weights.
static void apply_moe_state_correction(json & report,const json & c,const json & state,double now_us) {
    require(c.at("version")==1 && c.at("observation_point")=="preload" && c.at("lengths")==json({128,768}),"invalid state correction schema");
    require(c.at("profiles").size()==2 && c.at("profiles")[0].is_string() && c.at("profiles")[1].is_string() && c.at("profiles")[0]!=c.at("profiles")[1],"invalid state profiles");
    for(const char * kind:{"native","custom"}) {
        const auto & m=c.at("models").at(kind); auto k=m.at("kind").get<std::string>();
        require((k=="bias" || k=="state") && m.at("coefficients").size()==5,"invalid state coefficients");
        for(size_t i=0;i<5;++i) { const auto & v=m.at("coefficients")[i]; require(v.is_number() && std::isfinite(v.get<double>()) && (k!="bias" || i<2 || v.get<double>()==0),"invalid state coefficient"); }
        const auto & err=m.at("relative_error"); require(err.is_number() && std::isfinite(err.get<double>()) && err.get<double>()>=0 && err.get<double>()<=2,"invalid state residual envelope");
    }
    require(c.at("bounds").size()==3,"invalid state bounds");
    for(const auto & b:c.at("bounds")) { require(b.size()==2 && b[0].is_number() && b[1].is_number() && std::isfinite(b[0].get<double>()) && std::isfinite(b[1].get<double>()) && b[0].get<double>()<=b[1].get<double>(),"invalid state bounds"); }
    require(c.at("max_age_us").is_number() && std::isfinite(c.at("max_age_us").get<double>()) && c.at("max_age_us").get<double>()>0,"invalid state age");
    json status={{"applied",false},{"observation_point","preload"}};
    auto reject=[&](const char * reason) { status["reason"]=reason; report["state_correction"]=status; };
    int nt=report.at("tokens"); auto profile=report.value("selected_profile",std::string());
    if((nt!=128 && nt!=768) || (profile!=c.at("profiles")[0] && profile!=c.at("profiles")[1]) || report.at("custom_candidates").size()!=1) { reject("unsupported_profile"); return; }
    if(!state.is_object() || !state.contains("features") || !state.contains("monotonic_us") || !state.at("monotonic_us").is_number()) { reject("missing_or_stale_state"); return; }
    const auto & f=state.at("features");
    for(const char * k:{"skin_c","cpu0_mhz","cpu0_hw_max_mhz","cpu6_mhz","cpu6_hw_max_mhz"}) {
        if(!f.contains(k) || !f.at(k).is_number() || !std::isfinite(f.at(k).get<double>())) { reject("missing_or_stale_state"); return; }
    }
    double age=now_us-state.at("monotonic_us").get<double>();
    if(!std::isfinite(age) || age<0 || age>c.at("max_age_us").get<double>() || f.at("cpu0_hw_max_mhz").get<double>()<=0 || f.at("cpu6_hw_max_mhz").get<double>()<=0) { reject("missing_or_stale_state"); return; }
    std::array<double,5> x={1.,double(nt==768),(f.at("skin_c").get<double>()-35)/10,
        1-f.at("cpu0_mhz").get<double>()/f.at("cpu0_hw_max_mhz").get<double>(),1-f.at("cpu6_mhz").get<double>()/f.at("cpu6_hw_max_mhz").get<double>()};
    for(int i=2;i<5;++i) { const auto & b=c.at("bounds")[i-2]; if(x[i]<b[0].get<double>() || x[i]>b[1].get<double>()) { reject("state_outside_calibration"); return; } }
    json factors=json::object(),kinds=json::object(); auto & d=report["path_decision"];
    for(const std::string kind:{"native","custom"}) {
        const auto & m=c.at("models").at(kind); double log_factor=0;
        for(int i=0;i<5;++i) { log_factor+=x[i]*m.at("coefficients")[i].get<double>(); }
        double factor=std::exp(std::max(-std::log(2.),std::min(std::log(2.),log_factor))),old=d.at(kind+"_predicted_us");
        double predicted=old*factor; factors[kind]=factor; kinds[kind]=m.at("kind");
        d[kind+"_before_state_us"]=old; d[kind+"_predicted_us"]=predicted; d[kind+"_uncertainty_us"]=std::max(1000.,predicted*m.at("relative_error").get<double>());
    }
    d["native_lower_us"]=d["native_predicted_us"].get<double>()-d["native_uncertainty_us"].get<double>(); d["reason"]="preload_residual_model";
    auto & option=report["custom_candidates"][0]; option["before_state_us"]=d["custom_before_state_us"]; option["predicted_us"]=d["custom_predicted_us"]; option["uncertainty_us"]=d["custom_uncertainty_us"];
    status["applied"]=true; status["reason"]="calibrated_state"; status["features"]=x; status["age_us"]=age; status["factors"]=factors; status["model_kinds"]=kinds;
    report["state_correction"]=status;
}
