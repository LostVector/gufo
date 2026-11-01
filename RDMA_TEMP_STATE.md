# RDMA merge checkpoint — 2026-11-01

This branch saves unfinished work. It is not qualified for deployment or push to rdma.

## Source and build

- Baseline rdma: `0b6978aa`.
- Merged upstream/main: `12effe369662ac572cb100cd20157f56a74e2da0`.
- Builds and GPU tests use pinned Nix tooling and ROCm 10.1 inside containers on fuzzy and misty.
- TP2 adaptations include rank-local Q8 expansion, shared-expert stream ordering, prompt rendering, cache persistence propagation, logprobs protocol v22, TP2 MTP controller behavior, and Linux TP control isolation from native Windows builds.

## Known blocker

A true continued conversation with Q4 MTP, RAM caching disabled, disk caching disabled, and one session returned HTTP 500 on the second request. Rank 0 reported:

`Qwen3.8-Flash-Next reused prefix does not match checkpoint`

This remains unresolved. Do not interpret the earlier cache tests as covering this case. Diagnose prefix/checkpoint reuse and add a regression test before qualification.

Failure logs are preserved on the lab hosts under `~/gufo-rdma-upstream-20261031/q4-mtp-ramoff-failure-rank0.log` and `q4-mtp-ramoff-failure-rank1.log`.

## Evidence so far

- Seven focused Linux CPU tests passed, including TP control/executor and serving contracts; protocol v22 tests passed.
- Seven GPU operator tests passed on misty using ROCm 10.1.
- 171 functional harness unit tests, serving benchmark harness tests, pinned C++ formatting, documentation links, and dependency checks passed.
- Actual Q4 TP2 MTP API checks passed for rendering, input token counts, logprobs, mixed isolated/concurrent requests, and disk cache persistence exclusion after restart.
- Numerical split/batched model qualification, Q8 model correctness, and matched baseline/candidate performance measurements remain pending. Performance is unqualified. Native Windows builds have not been tested.

## Follow-up resources

Local reports and detailed runtime paths are in `/Users/user/Documents/gufo/artifacts/rdma-upstream-2026-10-31/`, especially `validation-state.md`, `STATE.md`, and `validation-plan.md`. These ignored artifacts are not included in this commit.

Lab root: `~/gufo-rdma-upstream-20261031`. Candidate containers at the last checkpoint were `gufo-rdma-validation-20261101` on fuzzy and `gufo-rdma-validation-20261031` on misty. Inspect current processes before testing. The RAM-cache-off pair was left available for reproduction. Old Q4 user services were stopped to prevent automatic restarts. The user permits stopping Gufo and does not require restoring it.

Resume with the correctness fix, rebuild and retest, then complete numerical and matched performance checks. Preserve all failed results. The earlier agents stopped because of a usage limit. This checkpoint was explicitly requested as local only; no push is authorized by the checkpoint request.
