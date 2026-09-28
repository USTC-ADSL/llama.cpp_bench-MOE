# Excess shared layouts cause paging during long prefill

In the first current-route scheduler prototype,1536-token prefill on fd8657d6 took2127.14ms (median of6 timed repetitions), slower than native HTP, despite balanced predicted expert branches. Each invocation incurred38410–109000 major faults. The initial resident manifest kept all three backend copies of every shared expert, even where the model predicted those alternatives could not compete.

Root cause: unnecessary shared copies increased logical resident weights to8.484GiB and activation arenas to576.47MiB. Together with complete-model buffers, this caused active working-set paging. Removing timing outliers or changing the cost fit would conceal the resource problem.

The provisioner now evaluates an independently optimized assignment for each shared backend and retains shared layouts within20% of the best score. It still retains the chosen layout, six alternative routed-expert layouts, and explicit backend availability. The current callback neither converts nor implicitly acquires missing weights. This policy is in tests/moe-route-plan.cpp; there is no1536-specific branch.

On the same1536 development workload, logical weights decreased to7.614GiB and arenas to420.47MiB. All6 timed invocations had zero major faults and median latency1890.91ms. Native A/B medians were1951.13/1914.58ms; the remaining gain is small.128 and768 final-residency reruns also had zero major faults. Full-model repeat logits are checked bitwise by the arena run contract. The separate route-resident-probe validates12 ownership/shared combinations against fresh compact banks and revisits15 cached graphs without reallocation.

Evidence and exact commands: task/handoff/handoff-2026-09-28-route-scheduler/evidence/dev-1536-route-v1.{json,run.json,log}, dev-1536-route-v2.*, dev-128-route-v2.*, dev-768-route-v2.*, resident-contract.{json,log}. Model preparation remains outside timed decode. This reduces memory pressure for the measured cases; it does not guarantee universal residency, absence of paging under arbitrary device state, or cold-request speedup. Independent confirmation remains pending at this checkpoint.
