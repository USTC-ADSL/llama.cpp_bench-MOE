#!/usr/bin/env python3
"""Descriptive timing attribution only; never train/select using these traces."""
import importlib.util
import json
import math
import statistics as st

spec=importlib.util.spec_from_file_location("study",__file__.replace("analyze-moe-device-timing.py","run-moe-device-timing-study.py"))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def stats(values):
    return {"median":st.median(values),"mean":st.mean(values),"min":min(values),"max":max(values),"stdev":st.stdev(values) if len(values)>1 else 0}

def window(trace,start,end):
    # Diagnostic zero-order hold from completed sysfs samples. A sample is not
    # used before its read completed; this is not GPU hardware residency time.
    assert start<end
    n=0;duration=0;hz=0;busy=0;frequencies=set();age=0
    for row,nxt in zip(trace,trace[1:]):
        a=max(start,row["end_us"]);b=min(end,nxt["end_us"])
        if b<=a:continue
        assert row["gpu_hz"] is not None and row["busy_percent"] is not None
        duration+=b-a;hz+=(b-a)*row["gpu_hz"];busy+=(b-a)*row["busy_percent"];frequencies.add(row["gpu_hz"])
        age=max(age,b-row["end_us"]);n+=1
    assert abs(duration-(end-start))<.01,"trace did not cover requested window"
    return {"frequency_mean_mhz":hz/duration/1e6,"busy_mean_percent":busy/duration,"frequency_levels_mhz":sorted(f/1e6 for f in frequencies),"segments":n,"max_hold_age_us":age}

def main():
    reports=[]
    for nt,mode,config,repeat in s.cases():
        d=s.verify(nt,mode,config,repeat);name=s.tag(nt,mode,config,repeat)
        rows=d["samples"];wall=[r["total_us"]/1000 for r in rows]
        report={"tag":name,"tokens":nt,"mode":mode,"config":config,"repeat":repeat,"wall_ms":stats(wall),"rounds":[]}
        if config!="control":
            report["htp"]={k:stats([r["htp_timing"][k] for r in rows]) for k in rows[0]["htp_timing"]}
            report["dsp_counter_mcycles_s"]=stats([r["htp_timing"]["dsp_batch_cycles"]/r["htp_timing"]["dsp_batch_us"] for r in rows])
            if mode=="npu":
                gap=[r["total_us"]-r["htp_timing"]["dsp_batch_us"] for r in rows]
                assert min(gap)>=0
                report["native_wall_minus_dsp_batch_ms"]=stats([x/1000 for x in gap])
        trace=None
        if config=="sampled":
            all_trace=[json.loads(line) for line in s.raw(s.EV/f"{name}.gpu-trace.jsonl").splitlines()]
            trace=all_trace[1:-1];footer=all_trace[-1];header=all_trace[0]
            duration=footer["end_monotonic_us"]-header["start_monotonic_us"]
            report["collector"]={**footer,"duration_us":duration,"one_core_cpu_percent":100*footer["cpu_us"]/duration,
                "mean_read_us":footer["read_total_us"]/footer["samples"],"trace_max_gap_us":max(b["end_us"]-a["end_us"] for a,b in zip(trace,trace[1:]))}
        for row in rows:
            rec={"round":row["round"],"wall_us":row["total_us"]}
            if config!="control":rec["htp"]=row["htp_timing"]
            if mode=="custom":
                rec["backend_run_sum_us"]=[sum(layer["backend_phases_us"][b][1] for layer in row["layers"]) for b in range(3)]
                rec["expert_critical_wall_sum_us"]=sum(layer["expert_wall_us"] for layer in row["layers"])
                rec["gather_sum_us"]=sum(layer["gather_us"] for layer in row["layers"])
                rec["combine_sum_us"]=sum(layer["combine_us"] for layer in row["layers"])
                rec["last_finishing_backend_counts"]=[0,0,0]
                for layer in row["layers"][:23]:
                    latest=max(range(3),key=lambda b:layer["intervals_us"][b][1])
                    rec["last_finishing_backend_counts"][latest]+=1
            if trace:
                state=row["runtime_state"]
                rec["gpu_request"]=window(trace,state["request_begin_monotonic_us"],state["request_end_monotonic_us"])
                ready=[x for x in trace if x["end_us"]<=state["request_begin_monotonic_us"]][-1]
                rec["ready_gpu_hz"]=ready["gpu_hz"];rec["ready_gpu_age_us"]=state["request_begin_monotonic_us"]-ready["end_us"]
                if mode=="custom":
                    rec["gpu_branches"]=[]
                    for layer in row["layers"]:
                        start,end=layer["intervals_us"][2]
                        if end<=start:continue
                        origin=layer["branch_origin_monotonic_us"]
                        rec["gpu_branches"].append({"layer":layer["layer"],"window_us":end-start,**window(trace,origin+start,origin+end)})
            report["rounds"].append(rec)
        if trace:
            report["gpu_request_mhz"]=stats([r["gpu_request"]["frequency_mean_mhz"] for r in report["rounds"]])
            report["gpu_request_busy_percent"]=stats([r["gpu_request"]["busy_mean_percent"] for r in report["rounds"]])
        if mode=="custom":
            report["backend_run_sum_us"]={name:stats([r["backend_run_sum_us"][b] for r in report["rounds"]]) for b,name in enumerate(("CPU","NPU","GPU"))}
            report["last_finishing_backend_counts"]={name:sum(r["last_finishing_backend_counts"][b] for r in report["rounds"]) for b,name in enumerate(("CPU","NPU","GPU"))}
        reports.append(report)
    effects=[]
    for nt,mode in ((128,"custom"),(768,"npu")):
        for repeat in ("a","b"):
            r={x["config"]:x for x in reports if (x["tokens"],x["mode"],x["repeat"])==(nt,mode,repeat)}
            a,b,c=[r[k]["wall_ms"]["median"] for k in ("control","timing","sampled")]
            effects.append({"tokens":nt,"mode":mode,"repeat":repeat,"control_ms":a,"timing_ms":b,"sampled_ms":c,
                "timing_vs_control_latency_percent":100*(b/a-1),"sampling_vs_timing_latency_percent":100*(c/b-1),"combined_vs_control_latency_percent":100*(c/a-1)})
    result={"protocol_sha256":s.sha(s.TASK/"protocol.json"),"successful_processes":len(reports),"all_original_path_logits_bitwise_equal":True,
        "scope":"descriptive fixed-placement diagnostic, not new performance model; nested host/DSP intervals not additive; GPU frequency is sampled/held sysfs state, not hardware residency; sequential controls do not isolate all external drift",
        "observer_comparisons":effects,"processes":reports}
    s.write(s.TASK/"analysis.json",result)
    for r in reports:
        print(r["tag"],"wall",round(r["wall_ms"]["median"],2),"DSP",round(r.get("htp",{}).get("dsp_batch_us",{}).get("median",0)/1000,2),"counter",round(r.get("dsp_counter_mcycles_s",{}).get("median",0),2),"GPU",round(r.get("gpu_request_mhz",{}).get("mean",0),2),"busy",round(r.get("gpu_request_busy_percent",{}).get("mean",0),2))
    print("Observer comparisons",json.dumps(effects,indent=2))

if __name__=="__main__":main()
