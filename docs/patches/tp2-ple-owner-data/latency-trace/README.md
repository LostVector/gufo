# Request-identified PLE latency trace

Fixed original-history baseline/candidate/baseline with measured production
sourcesfbacf9cc/85586a63, unchanged81C6/TX32 native driver, UD-Q4_K_XL/shared-Q8_0
MTP, thinkingoff,65K capacity,two sessions,chunk2048,1GiB cache. Original first
five requests: warmup8, first/repeated2K max128, shortC2 max128. No extra idle,
full-table warming, SDK/GPU/NVMe/admin sampler or production binary change.

All15 canonical prompt/output/cache-work identities match the original timing
study prefix. Both ranks' six transport windows, binary/settings/PID audits,
application map/ring/recursion/balance checks, unchanged boot/topology/kernel
checks and final approved idle state pass. No independent kernel uprobe missed
counter is available. The [raw archive](acquisition.zip) includes frozen protocol,
harness, six application traces,15 client records and analysis scripts. Commands
from project root: `python3 -u tp-cache-retention/ple-owner-latency-trace-v1/run.py`,
then `validate.py`, `analyze.py` in that directory; plot uses the lab`.venv/bin/python`.
Paths/hosts/models/privileged helpers require the existing lab setup.

## Direct observations

| Arm | C2 actor0 client TTFT ms | Last prefill return to publication ms | Other actor prefill inside that gap ms |
| --- | ---: | ---: | ---: |
| Baseline before |329.740|61.1479|61.1002|
| Candidate |327.018|61.2691|61.2421|
| Baseline after |263.229|0.0266|0|

[Timeline](publication-timeline.png) shows the actual scheduling difference:
actor0 has finished its last prefill, but the scheduler executes actor1's final
roughly61ms StepPrefill before attempting actor0 publication in the first two
arms. The later baseline publishes actor0 first. Opaque request pointers join
StepPrefill/PrepareFirstSnapshot/PublishSelection; they contain no prompt data.
The roughly64ms candidate-vs-later-baseline TTFT increase recurs as three strict
entries (client/server/raw reporting the same request). All remain recorded.
Candidate new PLE callback host submissions during C2 are at most32us, and first
C1 has42 calls totaling0.354ms. This callback **queues** GPU staging/wait/validation
and records an event; it does not await full transfer/GPU completion. These are
not end-to-end collective timings or proof that all TP communication is cheap.

Warmup TTFT148.24/147.39/182.02ms has its extra roughly35ms inside the later
baseline's first Forward host span:77.66/77.41/112.22ms. Its second Forward is
61.52/61.14/61.15ms; post-prefill publication is29–37us. This localizes the mode
inside Forward, without separating GPU execution, peer work or host waiting.
There is no GPU/kernel attribution from an application-only trace.

## Interpretation and limits

The C2 publication mode exists with unchanged baseline code and with the PLE
candidate. Scheduler capturing-queue checks and prefill/decoder selection
(`text_generation_scheduler.cpp`, source85586a63) permit different interleavings
around asynchronous snapshot completion; actual capture spans also overlap peer
prefill. This supports a shared scheduling explanation for this observed mode,
not a64ms new-collective host-dispatch cost. Snapshot readiness at every branch
was not separately probed, so that branch-level mechanism remains an inference.
Do not reopen the closed initial-batch/coalescing-timer experiments on this basis.

One B/C/B does not waive earlier flags, prove equivalence or quantify frequency.
Three adaptive-work field comparisons remain. No current strict completion-tail
regression recurred; historical tail/gap flags remain. C2 request windows overlap,
span totals can nest, and peer clocks are not aligned: the report omits rank1
window totals and preserves full peer traces instead. Last prefill return is a
host scheduler boundary, and publication entry is an attempt rather than GPU
ready or delivery. Wider concurrency remains before default adoption.
