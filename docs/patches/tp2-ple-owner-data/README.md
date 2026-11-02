# Cooperative PLE owner reads: frozen two-node evidence

The prototype lets rank1 read eligible disk-backed PLE rows and sends exact row
IDs/F32 bits to rank0 through the existing native communicator. Wider ordinary
prefill retains local reads. Each mapped packet has a GPU completion event;
unsafe reuse fails closed. TP2 remains eager. No full n-gram table is warmed.

Measured source: baseline `fbacf9cc`, candidate `85586a63`. Both use the repository
release preset (`RelWithDebInfo`) with BUILD_TESTING=OFF, private ROCm7.14.1/Clang23,
UD-Q4_K_XL target/shared-Q8_0 MTP, thinking off, context capacity65536,2sessions,
prefill chunk2048,1GiB RAM cache, approved81C6/TX32 native driver and crossed40+20Gb/s
links. All77 common model/family compiler commands match after path normalization;
only2 additional PLE implementation sources are candidate-only. These are
measurements of those sources, not a later integration revision. The review
branch additionally merges maintained `tp2` f75bed09 (RAM-off cache correctness
and Windows macro fixes); the tables do not qualify that later combined binary.
The combined source2669e07c has separate production runtime/failure qualification
below; the performance table remains attributed to the measured prototype.

## Measured results

Ten preselected balanced randomized pairs,20 fresh server starts,240 request
records; no exclusions, added pairs, pooling or interim significance testing.

| Metric | Baseline tok/s | Candidate tok/s | Paired change | Marginal95% CI | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| First C1 server decode | 66.72984 | 67.19626 | +0.69897% | +0.49928..+0.86163% | Primary, Holm p=.015873 |
| Short C2 complete client wave | 70.54220 | 71.91057 | +1.93979% | +0.24552..+4.61339% | Primary, Holm p=.031746 |
| Repeated C1 server decode | 67.05267 | 67.02877 | -0.03564% | -0.33168..+0.26247% | Exploratory, inconclusive |
| Coding C2 complete client wave | 79.87246 | 81.81882 | +2.43684% | +1.52257..+3.44278% | Exploratory, no adjusted conclusion |

[Paired figure](before-after.png), [statistics](statistical-summary.json),
[per-request data](per-request.csv), [all timing flags](timing-flags.csv) and
[flag review](timing-flag-review.json) retain every observation, including the
low short-C2 baseline in pair10. C1 speculative work is exact;26 C2 request
comparisons change adaptive work. C2 throughput is a complete client-wave rate,
not a sum of per-request decode rates. Intervals are marginal; assignment/sign
calculations retain their documented assumptions. No equivalence is claimed.

All27 strict timing entries over15 request comparisons remain:8startup,10shortC2,
9codingC2. None concern the main pp2048 C1 case. Startup TTFT has low/high modes
in both arms (two low/eight high each); C2 actor0 also changes modes in some pairs.
Mode causes remain unresolved. Several completion-tail increases accompany
changed adaptive work and earlier whole-request completion. Neither aggregate
gains nor that observation waive phase regressions. The first-pass gain does
not establish a warmed-repeat gain or an SSD/network/CPU-only attribution.

## Correctness and remaining gates

Fresh production smoke12 and cache/two-role cancellation checks pass. Production
baseline and candidate match104 full-logit/RNG/token/position/work records per
rank and32 complete materialized-state hashes per rank at short and actual32K
depths. Logs verify3840owner rows on rank1 and3840received rows on rank0.
These are state/behavior comparisons, not independent official-teacher parity.

The real-peer component probe passes140exact-bit/reuse exchanges/rank plus row
mismatch and owner-read-failure rejection. That probe used a testing-enabled
build, explicitly separate from production timing. Mixed wire21/22 peers reject
placement metadata in both rank orders before model/collective execution.
The same approved module/boot/topology and idle state and checked error/kernel
conditions pass. These do not qualify every asynchronous-reuse pattern.

The combined production binary at source `2669e07c` now passes a fresh
[twelve-request replay](ple-owner-review-http-smoke-v1-validation.json) and
[cache/two-role cancellation lifecycle](ple-owner-review-http-lifecycle-v1-validation.json).
Its [binary/build record](ple-owner-review-release-binary.json) pins the exact
BUILD_TESTING=OFF release artifact; these checks do not remeasure the table above.

[Live owner failure checks](ple-owner-live-faults-v3-validation.json) inject one
aligned O_DIRECT PLE read EIO or terminate the task-owned rank1 process after
streaming begins. Both give explicit stream errors, automatically exit the
failed pair, refuse new connections, and recover with eight exact actual-MTP
outputs across independent fresh restarts. Failure-to-stream-error was186.8ms
for EIO and195.1ms for process loss in these observations, not transport latency.

[Event-record failure checks](ple-owner-event-faults-v1-validation.json) inject
one correctly typed `hipEventRecord` error at the new PLE callback on each rank.
Four already-live client streams fail explicitly; both failed pairs exit and
eight fresh recovery outputs match. The shim omits the real API only at the
selected failed call; its [five CPU controls](event-fault-shim-selftest.json)
include an eight-thread one-failure race. The [exact call-site record](review-ple-event-target.json)
is valid only for the pinned binary. This is process-local API-return injection,
not a physical GPU/cable fault, precise collective-inflight loss, or transparent
in-process reconnect test.

The [qualification harness archive](combined-qualification-harness.zip) preserves
frozen study protocols and their hash-verified harness inputs plus the typed
shim sources. It needs the same lab setup as the timing acquisition. Failed
fixture attempts remain in the lab: v1 expected a surviving HTTP500 server but
current serving intentionally exits a failed TP pair; v2 then hit an undefined
report-variable typo after successful failure handling. Fresh v3 corrects the
oracle/typo and completes every stage. These were fixture failures, not passing
runs or production fixes.

Wider concurrency and bounded request-identified attribution of the retained
latency flags remain before default integration. No production default or upstream-ready RDMA integration
is claimed. Earlier testing-enabled candidate measurements and rejected partial
timingv1 remain preserved; corrected production timingv2 is fresh. See the
[build-mode erratum](cooperative-ple-build-mode-erratum.json) and
[production build rules](production-build-rules.md).

## Verify and reproduce

```sh
python3 docs/patches/tp2-ple-owner-data/check.py --statistics
```

This checks compressed/raw/file hashes, all240 identities and C1 work,26 C2 work
changes, paired rates, every strict5%AND3ms flag and the frozen bootstrap/Holm
calculations. It does not run a model or certify hardware. Raw JSONL records
include client/phase fields; binaries/protocol/design/build manifests pin inputs.
The [frozen acquisition scripts](acquisition-scripts.zip) preserve every script
and external input hash listed in the original protocol. Their lab-specific
paths, SSH hosts, model locations and privileged inventory helper require the
documented workspace setup; they are not a portable installer. Adapting those
paths creates a new acquisition with its own frozen manifest.

Rebuild the source using the documented release prerequisites and native RDMA
setup. Keep tests in a separate build directory. The model target is `gufo`;
the native component correctness target is `qwen38_flash_next_ple_owner_probe`
(testing required). Run the probe with matching rank/host/bootstrap port/device
and `good`, `rows` or `read`; it requires two real native peers. Exact local
acquisition/analysis scripts remain in the lab's
`tp-cache-retention/cooperative-ple-release-comparison-v2` and
`fork-integration-v1/*cooperative-ple*`. The frozen design/sequence and raw records
are exported here; no generic network simulator is substituted.
