"""Small pre-load residual model; no post-request features or weight migration."""
import math


def features(nt, state):
    f = state["features"]
    values = [f[k] for k in ("skin_c", "cpu0_mhz", "cpu0_hw_max_mhz", "cpu6_mhz", "cpu6_hw_max_mhz")]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values) or values[2] <= 0 or values[4] <= 0:
        raise ValueError("missing state features")
    return [1., float(nt == 768), (values[0]-35)/10, 1-values[1]/values[2], 1-values[3]/values[4]]


def validate(c):
    if c["version"] != 1 or c["observation_point"] != "preload" or c["lengths"] != [128,768]:
        raise ValueError("invalid state correction schema")
    if len(c["profiles"]) != 2 or len(set(c["profiles"])) != 2:
        raise ValueError("invalid state profiles")
    for kind in ("native", "custom"):
        m=c["models"][kind]
        if m["kind"] not in ("state", "bias") or len(m["coefficients"]) != 5 or not all(type(v) in (int,float) and math.isfinite(v) for v in m["coefficients"]):
            raise ValueError("invalid state coefficients")
        if not math.isfinite(m["relative_error"]) or not 0 <= m["relative_error"] <= 2:
            raise ValueError("invalid state residual envelope")
        if m["kind"] == "bias" and any(m["coefficients"][2:]):
            raise ValueError("bias model contains state coefficients")
    if len(c["bounds"])!=3 or any(len(b)!=2 or not all(math.isfinite(v) for v in b) or b[0]>b[1] for b in c["bounds"]):
        raise ValueError("invalid state coverage bounds")
    if not math.isfinite(c["max_age_us"]) or c["max_age_us"] <= 0:
        raise ValueError("invalid state age")


def apply(report, correction, state, now_us):
    validate(correction)
    status={"applied":False,"observation_point":"preload"}
    report["state_correction"]=status
    if report["tokens"] not in correction["lengths"] or report.get("selected_profile") not in correction["profiles"] or len(report["custom_candidates"]) != 1:
        status["reason"]="unsupported_profile"; return report
    try:
        x=features(report["tokens"],state)
        age=now_us-state["monotonic_us"]
        if not math.isfinite(age) or not 0 <= age <= correction["max_age_us"]:
            raise ValueError("stale state")
    except (ValueError,KeyError,TypeError):
        status["reason"]="missing_or_stale_state"; return report
    if any(v<lo or v>hi for v,(lo,hi) in zip(x[2:],correction["bounds"])):
        status["reason"]="state_outside_calibration"; return report
    factors={kind:math.exp(max(-math.log(2),min(math.log(2),sum(a*b for a,b in zip(x,m["coefficients"]))))) for kind,m in correction["models"].items()}
    d=report["path_decision"]
    for kind in ("native","custom"):
        old=d[f"{kind}_predicted_us"]; predicted=old*factors[kind]
        d[f"{kind}_before_state_us"]=old; d[f"{kind}_predicted_us"]=predicted
        d[f"{kind}_uncertainty_us"]=max(1000.,predicted*correction["models"][kind]["relative_error"])
    d["native_lower_us"]=d["native_predicted_us"]-d["native_uncertainty_us"]
    d["reason"]="preload_residual_model"
    report["custom_candidates"][0].update(before_state_us=d["custom_before_state_us"],predicted_us=d["custom_predicted_us"],uncertainty_us=d["custom_uncertainty_us"])
    status.update(applied=True,reason="calibrated_state",features=x,age_us=age,factors=factors,
                  model_kinds={k:v["kind"] for k,v in correction["models"].items()})
    return report
