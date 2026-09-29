# Upstream integration and Hexagon impact review

Start: feat/dynamic-moe-placement at 7f224605a. Preserve user-owned untracked TODO.md and all historical experiments. User requests upstream integration with emphasis on Hexagon and impact on ongoing heterogeneous MoE work. Default device fd8657d6 is online.

- [x] Inspect local state and read current execution/scheduler handoffs.
- [x] Fetch upstream and audit Hexagon changes, local intersections and compatibility (420 commits, 24 Hexagon commits).
- [x] Merge upstream, resolve ten conflicted files while preserving local functionality, and inspect resulting diff. Validated integration committed as 0bc3225c3.
- [x] Incrementally build affected Android/HTP targets and run focused device regressions (60 indexed cases, 12 resident, 12 canonical/mixed, GPU conversion, six full-model processes at 128/768).
- [x] Record impact, validation limits and reproducible commands; archive 80 verified members, clean 35 owned device files and commit integration. Final documentation is saved in a separate reporting checkpoint.
