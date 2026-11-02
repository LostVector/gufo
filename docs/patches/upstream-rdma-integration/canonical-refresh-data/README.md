# Canonical refresh evidence

The [integration record](../README.md) distinguishes the measured739eb86a
extension migration from the later4e8a2dc0/f25f4744 build/runtime checks.
`manifest.json` records file hashes, compressed raw records and report attribution.
The22 raw files contain240 paired measurement records plus12 latest-smoke
records and13 latest cache/cancellation case records (the latter are not13HTTPcalls).

Run from any directory with Python3:

```sh
python3 docs/patches/upstream-rdma-integration/canonical-refresh-data/check.py --statistics
```

This checks hashes, all240 paired identities, exactC1proposal/acceptance/round
work,32C2work changes, per-pair rates, all52flags over20requestcomparisons,
and recomputes the frozen bootstrap/assignment/Holm calculations. It does not
rerun inference or independently repeat the GPU numerical/fault/disk tests.
Their copied validation reports preserve scope and revision. Historical absolute
paths in original reports are provenance; the verifier reads only adjacent files.

`paired_statistics.py` is the frozen original. No paired acquisition was dropped
or replaced. Marginal confidence intervals and conditional sensitivity tests
retain their stated assumptions; nonsignificance is not equivalence. C1rate
uses server-accounted decode time; C2wave includes admission/prefill/terminal
waits.52flags include aliased metrics, not52independent incidents.

The original acquisition used these commands from the investigation root, with
its frozen scripts/settings and shared model lock:

```sh
source cable-prep/env.sh
source native-rdma/env.sh
export LD_LIBRARY_PATH="$PWD/tp-cache-retention/fork-integration-v1/runtime/lib:$LD_LIBRARY_PATH"
unset LD_PRELOAD ROCP_TOOL_ATTACH STRIX_FINISH_TRACE GUFO_TP_TRACE_PREFIX
python3 tp-cache-retention/extensions-release-comparison-v1/run.py
python3 tp-cache-retention/extensions-release-comparison-v1/analyze.py
python3 tp-cache-retention/extensions-release-comparison-v1/summarize.py
python3 tp-cache-retention/extensions-release-comparison-v1/statistical-summary.py
```

Those single-use acquisition scripts refuse existing protocols. A new hardware
run requires a fresh output directory, matching artifacts/settings and the
original model files; the portable verifier above requires none of the hardware.
The original diskV1 fixture failed before requests on removed progress-log
labels. DiskV2 preserves actual restart/output/work assertions and checks all
five full-file identity-cache digests on both hosts instead. Both failed
self-import attempts and the log-only fixture failure remain in the investigation
record; no completed model acquisition was replaced.
