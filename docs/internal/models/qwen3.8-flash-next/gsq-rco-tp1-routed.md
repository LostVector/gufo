# GSQ-RCO TP1 routed gate/up fusion

November15,2026: enable existing fused/grouped gate/up kernels for IQ2_S,
IQ3_XXS,IQ3_S and IQ4_XS at1..8 tokens. All48 target layers in the official
GSQ-RCO IQ3_S model use these formats. Each dot retains its original arithmetic;
expert masks,finite sanitation and output slots are unchanged. Wider IQ batches
keep their old path. No new flag,weight conversion or n-gram warming.

## Measured C1 decode

| Metric | Prior dense-sharing build | With IQ gate/up fusion |
| --- | ---: | ---: |
| Aggregate median tokens/s | 52.56 | 54.30 |
| Story median tokens/s | 55.68 | 57.18 |
| Explanation median tokens/s | 53.99 | 56.03 |
| Code median tokens/s | 48.49 | 50.18 |
| 2149-token prefill pilot median tokens/s | 698.51 | 697.90 |

Eight fixed fresh-process blocks compare baseline,fusion,and fusion plus down
expert ordering. Orders cover all six permutations plus a forward/reverse pair;
24 starts complete,with no discarded observations. Each process warms32 tokens
then generates128 total tokens on each of three fixed prompts. The primary
endpoint is384 tokens divided by summed engine decode milliseconds; one process
is one observation. All72 response hashes and all recorded prompt/output/draft/
acceptance/cache/lookup work counts match their controls.

Fusion's paired geometric gain is **3.298%**;50000-resample marginal95% bootstrap
interval[3.090,3.521]%,seed0. Exact two-sided paired sign-flip p=.0078125;
Holm-adjusted p=.0234375 across the three planned comparisons. All eight fusion
pairs improve,from2.844% to3.896%. Separate17-request pilots have exact outputs/
work and unchanged prefill within noise. This is a same-day,fixed-prompt C1
result,not a TP2,long-depth,concurrent or broad model guarantee.

Setup:EVO-X2/gfx1151,privateROCm7.14.1,TP1/native shared-Q8 MTP,262144 context
capacity,4 configured slots/1active,thinking on/xhigh,budget27853,total output
default32768,temperature1,top-p.95,top-k20,min-p.05,seed42,no penalties,1GiB
prompt cache. Timing requests cap output at128 including reasoning. Actual short
prompts94/100/90 tokens. Model revision`ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
BaselineSHA256`20be6ff4f2c1aaff7fca0319802fde3f38c5ea253cdec690bab57e9ee933a2a2`.
The previous dense-sharing result is[separate evidence](gsq-rco-tp1.md),not an
additional matched cumulative speed measurement.

## Independent down-ordering experiment

Adding IQ4_NL/Q2_0 to the existing K640 down-view expert ordering gives+0.120%
over fusion alone,95%marginal interval[−0.004,+0.224]%,p=.09375. Six pairs improve,
two regress. Its direction remains inconclusive; no minimum-gain threshold was
used. It stays out of the final source. All observations remain in the evidence.
Its156 numerical cases pass bitwise against unsorted scalar calls before removal.

## Correctness and limitations

The final existing GSQ GPU test adds400 exact fusion checks to its312 independent
packed-arithmetic/dense/embedding cases. Fusion compares to the old paired GPU
projections followed by the original SwiGLU; it is not400 new CPU oracles. It
covers fourIQ formats,widths1..8,M65/640,shared/disjoint/mixed routing,duplicates,
negative IDs,fully inactive tokens,NaN/Inf scales,prequantized/fresh inputs,guards,
32 experts,and rejected width9 with unchanged output. Underlying packed dots
retain the existing independent CPU-dequant/FP64 checks. No tolerance changed.
The final712-case source passes in5.274s when compiled against the production
archives. This is a directly run GPU probe,not a claim that full CTest passed.
[Operator log](gsq-rco-tp1-routed/operator.log) and [checks](gsq-rco-tp1-routed/checks.json)
are retained. Changed-file formatting,docs,generated infrastructure and diff
checks pass. The repository-wide formatter still flags seven untouched files.

All17 pilot outputs/work counts match. Actual prefix restore,cached/cold copying
and prompt lookup pass with110/110 accepted copied tokens. One paired C2/C4
screen preserves all seven responses/work counts;timings are mixed. One C4
request's wall time rises129ms despite faster engine decode. No concurrent speed
or tail-latency claim. C6/C8 and TP2 are unqualified. No model-accuracy benchmark
was run; exact engine replay is the numerical evidence here.

## Trace confirms the execution path

Separate supported rocprofv3 HIP/kernel traces cover one matched explanation
request:100 prompt tokens plus128 generated tokens. Outputs/work agree.

| Kernel launches in request window | Baseline | Fusion |
| --- | ---: | ---: |
| All kernels | 88010 | 84074 |
| Q8 input quantization | 17003 | 15035 |
| Standalone SwiGLU | 4031 | 2063 |
| Routed vector projections | 6204 | 2268 |
| Grouped gate/up projections | 30 | 1998 |
| Expert grouping | 30 | 1998 |

The net3936 fewer launches(4.47%) validate the intended change. Fusion reuses
already-quantized shared-expert input and combines routed projection/activation.
Kernel-active union is2509.48→2429.83ms in traced request windows2762.04→2672.89ms.
Kernel-free wall time252.56→243.06ms includes CPU/client/I/O/waits;it is not pure
dispatch overhead or a measured removable-time budget. HIP API sums can overlap
threads and include waits. Traced timings are diagnostic; the speed claim above
uses only unprofiled runs. No SSD stall fix or distributed transport claim.

## Reproduction and evidence

The permanent operator source is
`tests/models/qwen38_flash_next/gsq_quant_ops_test.cpp`,under its existing GPU
CTest target. Build with the same pinned compiler/dependencies as the baseline.
Raw per-request[CSV](gsq-rco-tp1-routed/confirmation-requests.csv),
[fixed plan](gsq-rco-tp1-routed/confirmation-plan.json),
[paired results](gsq-rco-tp1-routed/confirmation-results.json),
[concurrent screen](gsq-rco-tp1-routed/concurrent-screen.json),and
[trace counts/durations](gsq-rco-tp1-routed/trace-summary.json) are retained.

Full commands,harnesses,binary hashes,raw responses and profiler CSVs remain at
`/home/rkuo/strix-dispatch-investigation/tp1-routed-optimization-2026-11-15/README.md`.
Use unique output names; harnesses refuse overwrites and check model ownership.
Public Gufo remains stopped on both hosts. No system/driver/boot settings changed.
