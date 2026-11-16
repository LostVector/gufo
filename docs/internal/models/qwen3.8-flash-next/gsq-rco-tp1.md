# GSQ-RCO TP1 dense verification

November 15, 2026: share packed weight decoding across verification tokens for dense
Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS in Flash Next. This makes the measured TP1
GSQ-RCO IQ3_S workload 8.81% faster, without changing weights or output precision.

## Measured result

Eight fresh-process pairs alternate AB/BA order, with four of each order. Each
process warms 32 tokens then generates 128 tokens on three fixed text/code prompts.
All 24 paired responses and recorded prompt/output/draft/acceptance/cache/lookup
work counters agree. No request was excluded. The primary endpoint is 384 tokens
divided by summed engine decode time; each process pair is one sample.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Aggregate C1 decode, median tokens/s | 48.33 | 52.57 |
| Story decode, median tokens/s | 50.42 | 55.71 |
| Explanation decode, median tokens/s | 49.49 | 54.17 |
| Code decode, median tokens/s | 45.46 | 48.47 |
| 2149-token prefill, pilot median tokens/s | 697.43 | 697.98 |

The geometric mean of the eight paired speed ratios is +8.809%, with paired
percentile-bootstrap 95% interval[+8.531,+9.089]% (50000 resamples,seed0).
An exact two-sided within-pair sign-flip test gives p=.0078125. All eight paired
changes are positive, from +8.10% to+9.49%. This is an exploratory same-machine,
same-prompt study, not a broad model/hardware/long-context guarantee. Prefill is
unchanged within noise; the new path is decode/verification only.

Setup: EVO-X2/gfx1151, private ROCm 7.14.1, TP1/native MTP, 262144 configured context,
4 slots/1 active request, thinking on/xhigh,budget 27853,total output default 32768,
temperature 1,top-p .95,top-k 20,min-p .05,seed 42,no penalties, 1 GiB prompt cache.
Timing requests cap total output at 128 tokens, including reasoning. Actual short
prompt counts are 94/100/90; 2149 includes thinking/effort framing. These are not
262K-depth results and must not be compared directly with the older TP2/non-thinking
measurements. Driver, OS, system ROCm, power/boot settings and disk-backed n-grams
are unchanged. The service stays stopped at the user's request.

## Mechanism and numerical contract

The generic dense vector adapter assigned different tokens to different waves,
repeating packed loads/dequantization. The new templated kernel keeps two output
rows and 2–8 token accumulators in one wave, allowing the packed values to be shared.
Each token keeps the original format-specific vector dot, K traversal and 32-lane
reduction. Scalar decoding, Q8 dense/routed paths, prefill, activation quantization,
model state and speculative policy stay unchanged. No flag or weight expansion.

Two separately captured GPU timelines have matched request work. Summed packed
dense kernel time falls from 853.79 to 660.92ms in the 128-token diagnostic request;
GPU busy time falls from 2710.62 to 2513.46ms. Profiles include short prefill and
tracing overhead, so untraced paired inference above is the performance result.

Quality evidence:

- 312 GPU cases / 1,559,000 checked values pass in 3.37 s, excluding compilation/loading.
  Independent CPU-dequant/FP64 checks use exactly quantizable activations; separate
  bitwise scalar/batch tests use nonintegral activations and unequal block scales.
  They cover all widths 1–8 and 17/64, odd output rows, guards, Q6 K6144 and IQ4_NL K640.
  These are distinct oracles; no tolerance widened.
- Real Q6 head weights at output sizes 257/10240/248320 and widths 1/2/3/4/7/8 retain
  all 6,470,425 output floats bitwise against baseline executable goldens, plus guards.
  These component timings show up to 60% savings for wider Q6 vocabulary batches;
  scalar and two-token vocabulary batches change little.
- The full 17-request pilot has identical outputs and all recorded work counters,
  including repeated sampled generations and a repeated greedy non-thinking control.
- Cached/cold continuation and actual cache reuse pass on both builds. A verbatim
  copied-token workload proposes and accepts 110 tokens on each. The inherited
  TP2-oriented fixture produced no copies under the TP1/eval profile; its work
  assertion failure is retained locally and is not counted as a pass. The new
  workload keeps the equality and >8proposed/accepted requirements.

The parent selected-extension BUILD_TESTING registry has an unrelated DeepSeek
failure. Tests were linked explicitly against all current CMake-built archives;
commands are retained. This does not claim normal GPU CTest or whole-repo CI passes.

## Retained limits and rejected paths

Q6-only sharing first showed 4.5–7.5% in a pilot but its six cold-process pairs
were inconclusive: estimate +3.35%,95% interval[-0.89,+7.15]%,exact p=.21875. Those
results remain alongside the independent eight-pair combined study. A four-row
Q6 tile was smaller/mixed and was replaced by token sharing.

The earlier HC vector-load prototype stays out. 64 replays of actual captured HC
matrices/inputs are bitwise identical, but that did not isolate its earlier
whole-engine repeat failure. This study neither fixes nor conclusively attributes it.

A Q6-only diagnostic captured 50–90 ms GPU-idle intervals outside HIP API activity.
A subsequent read-syscall/GPU trace did not reproduce those intervals: its 1561
n-gram reads topped out at 4.148 ms, and its largest 30.08 ms GPU gap had no overlapping
pread. It does not prove an SSD/controller fault. A syscall duration is host/kernel
read-wait time, not NVMe command completion time. Full traces remain local; no
NVMe admin sampler, kernel change or read-path workaround was introduced.

This qualifies TP1,C1 with these prompts/settings. TP2 and concurrent-request
performance are unqualified. Measurements preceded fork publication.

## Reproduction and evidence

[Portable measurements](gsq-rco-tp1-evidence/) include all paired requests, quality,
component timings, profile summaries, cache gates and the inconclusive Q6-only
study. Local raw requests/responses, traces, exact build commands, retained
failures and controllers are under
`~/strix-dispatch-investigation/tp1-quant-optimization-2026-11-15`.

Use that directory's run.py for isolated model checks, build.py for CMake incremental
builds, run-operator.py for numerical binaries and all-confirm.py for the eight-pair
study. Result paths are exclusive. Controllers own the shared model lock, verify
both hosts idle, use private loopback ports and leave the serving unit stopped.

Shared formatting reports seven pre-existing untouched failures; changed non-vendored
C++ passes. Docs and diff whitespace checks pass.
