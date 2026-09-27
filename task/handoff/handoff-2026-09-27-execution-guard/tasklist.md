# Prediction-error guard: implementation and device validation

- [x] Implement optional complete-schedule coverage guard, signed bias and empirical error allowance; preserve unguarded v2/v1.
- [x] Android contract/parity tests and incremental build (18 cases passed).
- [ ] Collect technical A fitting and independent technical B residual calibration; freeze metadata/model.
- [ ] Fresh ecology C confirmation at 128/384/768, native/auto A/B and forced custom when rejected.
- [ ] Audit results/logits/provenance; archive and clean owned device artifacts; Git checkpoints and final handoff.

## Protocol fixed before new measurements

Target fd8657d6 only. Resident weights and full 24 Transformer layers, last FFN at one token. No kernel/logit changes or additional complete weight replica. Parent frozen v2 SHA256: `17bb46360a79573f3e3009c99cb11c1de8d667e0d34eeda095735f2607079b21`.

Guard profiles are complete request schedules: exact length, family, barrier/overlap and 23 expert-bank sizes; expert IDs sorted by current offline route count then ID. Final-token placement uses unchanged base predictor. Initial portfolio: CPU routed tail + NPU remainder + GPU shared; 128 uses original technical CPU tail24/32 layer vector, 384 uniform tail16, 768 uniform tail1/4/8/16. Both policies collected as 12 timed ABBA samples, two warmups. Other lengths fall back to native with explicit unsupported status. No categorical template interpolation.

Tail experts are selected from active experts only, matching the original base candidate definition. A template with insufficient active experts in any regular layer is unsupported and excluded; zero-token Experts are not counted as offloaded work. All choices still use actual route counts as numeric model inputs.

A = original technical prompt; B = independently written technical prompt; C = new ecology prompt. Before device timing, freeze all prompt hashes and exact templates. Existing museum/programming/astronomy confirmation timings are excluded from fitting. For each profile: bias = median(A measured - frozen nominal A); allowance = max(1000us, maximum absolute A residual around corrected prediction, maximum absolute B residual around corrected prediction). Native anchor = A native median; native route adjustment = unchanged base NPU compute delta; native allowance uses the same empirical A/B maximum residual rule. Native controls bracket each length's profiles. These are empirical envelopes, not statistical coverage guarantees.

Choose minimum corrected custom prediction + allowance; enable only if its upper estimate plus max(reserved, observed device decision time) < native lower estimate. Preserve coefficients and all unfavorable observations. Unsupported length returns native, not a fabricated extrapolation. Malformed routes/metadata fail validation. Model selection is offline before weight loading; this does not promise free online migration.

Freeze guarded model before C runs. C128/384/768 each: native→auto [→forced custom if rejected], reverse order second process cohort, six repetitions per process. Include actual device planning time in auto throughput, including fallback. Retain forced-custom results to expose missed gains. Record prediction residuals, policy, allocations, repeated logits and cross-backend differences without claiming quality equivalence.
