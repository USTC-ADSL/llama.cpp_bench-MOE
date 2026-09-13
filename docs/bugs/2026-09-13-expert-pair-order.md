# Rotating Scheduling Trials Kept a Fixed Pair Order

## Symptom And Trigger

During construction of tests/expert-scheduling-probe.cpp, review found that a
comment promising alternating stay/move order was not true for individual routes.
Smoke01 passed numeric tests, but it is excluded from formal scheduling results.

## Root Cause

There are twelve directed route/state combinations. The traversal rotates the
route index as i=(j+round+warmup)%12, but the initial code also chose stay/move
order using (j+round+warmup)%2. Since twelve is even, this equals i%2. Each route
therefore always ran in the same order, and retained-raw versus restore/reread
routes systematically had opposite ordering. Cache and frequency effects could
then be confounded with source-state differences.

## Change

Pair ordering now uses (round+warmup+i)%2, where i is the actual route index.
Every individual route alternates order each round while traversal still rotates.
The change is one expression in the test probe; no production backend changes.

## Verification And Limits

The updated standalone probe is rebuilt with the Android NDK and deployed with
SHA256 verification. Formal run01..03 use twenty measured pairs per route after
two warmup pairs, thus ten observations for each ordering per run. The runs also
use different token orders. All raw recoveries are byte-exact, and initial/final
source/target outputs are checked against an independent dequantized reference.
Initial zero-target computations prevent stale valid weights from hiding a
preparation failure.

Exact build/test commands and measured results are recorded in
task/handoff/handoff-2026-09-13-expert-scheduling/handoff-01-results.md. Phone DVFS
and CPU caches remain uncontrolled; balanced order reduces systematic bias but
does not establish identical hardware-cache state for counterfactual executions.
