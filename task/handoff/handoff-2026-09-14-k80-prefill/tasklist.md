# Redmi K80 Pro CPU / HTP Prefill Retest

Target: fd8657d6, Redmi K80 Pro, SM8750. Model:
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes.
User asks to recheck earlier CPU prefill results that exceeded HTP. No APK.

- [x] Confirm target online, original model present, no active llama benchmark.
- [x] Locate prior comparable results, exact configurations and relevant execution paths.
- [x] Verify and deploy matching benchmark binaries and HTP v79 skeleton.
- [x] Reproduce CPU/HTP prefill with matched workloads and repeated samples.
- [x] Audit actual HTP kernels/fallbacks separately from timed runs and explain observed differences.
- [x] Validate all 54 samples and compressed profile evidence; clean task-owned device artifacts.
- [x] Record results and commit a scoped checkpoint.

Device test directory: /data/local/tmp/llama-moe-prefill-retest (initially absent).
Initial battery temperature 23.1 C, battery 94%, USB powered. No clock or security
settings changed. Earlier OnePlus results are in a separate task directory.

E1 report is the relevant same-device comparison: 512-token request, t/tb4,
b512/ub128, CPU TTFT median 11873.722 ms vs HTP 2224.106 ms. E0_followup CPU perf
records belong to another phone and must not be compared directly with fd HTP.
Current device auto-detects v79. All 11 deployed hashes match host. Model SHA256
matches the prior known value. HTP expert numerical checks passed at T=1 and 128;
T128 full FFN worst-token NRMSE 0.000582735. This is a kernel sanity check, not
the complete-model performance result.
