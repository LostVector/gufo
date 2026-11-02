# First-Forward startup wait

This diagnostic uses the same frozen fbacf9cc/85586a63 production binaries and
original five-request history as the earlier bounded B/C/B trace. Public HIP
entry/return probes replace generic recorder IDs; the shape fields and opaque
managed-request identity fields are unchanged. The private HIP DSO hash and
actual executable mapping are verified on each host. Both binaries import the
probed versioned public functions. No SDK shim, model edit, table warming,
driver change or additional idle period is introduced.

All15 output/prompt/cache-work identities, both rank identities/settings, trace
coverage/recursion/maps, six checked transport windows and final kernel/idle
checks pass. Every recorded HIP span is exported in hip-api-spans.csv. The
first-request Forward partition uses only same-thread enclosures and an interval
union, not overlapping API sums. Peer API counts use the peer's complete trace;
there is no cross-host timestamp alignment.

| First12-row Forward, ms | Baseline before | PLE candidate | Baseline after |
| --- | ---: | ---: | ---: |
| Forward host span |78.333039|77.661739|111.360521|
| Probed HIP API interval union |70.135019|70.124709|104.242275|
| Outside those HIP APIs |8.198020|7.537030|7.118246|
| Sum of48 hipEventSynchronize calls |66.253560|66.179345|99.363110|
| Longest hipEventSynchronize |2.106787|3.128576|34.685104|
| Sum of2 hipStreamSynchronize calls |3.220104|3.222344|3.230513|
| Sum of102 hipMemcpyAsync calls |0.661355|0.723020|1.648652|

Warmup TTFT is148.749 /148.273 /180.656ms. The high startup mode reproduces on
the later unchanged baseline. Its longest event wait is the second event in
the first Forward:34.685ms, beginning6.163ms into that frame. The subsequent
seven-row Forward stays60.763–61.779ms in allthree arms. No host/device allocation
or graph launch occurs inside these first-Forward enclosures.

## Attribution and limits

Most of the additional first-Forward wall time is inside one HIP event wait.
It is not predominantly a longer stream-synchronize call, allocation API call,
or extra time outside the probed APIs. The source RouteHints path waits for
expert counts with hipEventSynchronize;48 repeated event waits are consistent
with that per-layer path. The API probe does not capture the callsite/event
handle, so this source association is not an independently recorded stack.

The wait may include GPU execution, peer/network progress, GPU runtime work,
host sleeping/polling or delayed scheduling. This trace does **not** establish
which of those delayed event completion/return, or that the wait is avoidable.
Time outside the API union likewise includes untraced CPU work and asynchronous
GPU execution; it is not a measured GPU-idle interval. Uprobes perturb timing.

There are **zero hipGraphLaunch calls on either rank in allthree complete
traces**. The imported public symbol was probed, and Executor::Run explicitly
keeps TP>1 eager because host-driven communication lacks a graph-safe capture
protocol. Enabling an existing graph flag therefore is not an available TP2
fix. Removing the per-layer host expert-count dependency or making collectives
capture-safe would be separate architectural work with numerical and multi-node
qualification, not a change justified by this startup observation alone.

No new strict5%/3ms phase flags arise in this diagnostic; five adaptive-work
fields differ. All earlier startup, TTFT, completion-tail and maximum-gap flags
remain. This triplet is not pooled into the original ten-pair study or used to
claim a new throughput gain/equivalence result. The remaining question is why
one event wait is long; the trace narrows its location, not its root cause.

## Reproduce

Run the unchanged acquisition harness in a fresh study directory. After it has
completed, run validate.py, analyze.py, then hip-analysis.py. The complete
protocol, scripts, dynamic-library mapping proofs, all raw app traces and all
raw request records are retained. This report is a host API partition; a GPU
timeline or thread-scheduling trace would be needed for a further split.

[Host partition figure](first-forward-host-partition.png) ([PDF](first-forward-host-partition.pdf))

The acquisition.zip archive retains the full lab layout, scripts, protocols,
raw app traces and result JSONL. The HIP CSV is inside that archive. Paths in
the reproduction notes refer to the original lab; this export does not rerun
inference. Run `python3 ../check-new-latency.py` to audit both new raw bundles.
