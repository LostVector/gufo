# GSQ-RCO IQ3_S support

As of November 15, 2026, the original ISTA-DASLab IQ3_S release runs on the
EVO-X2/Corsair Gufo TP2 pair with native MTP. It saves memory but is slower than
UD-Q4_K_XL on the current kernels. This qualifies the tested text paths, not
model intelligence, vision, all IQ artifacts or full 64K prompt depth.

## Artifact and reproduction

Repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Both shards were verified on both hosts:

| Shard | Bytes | SHA256 |
| --- | ---: | --- |
| 1 | 54817524224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| 2 | 28800138432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The label covers BF16/F16/F32, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0,
Q4_K, Q5_K, Q6_K and Q8_0. Q2_0 is type42 with64 values/18 bytes and signed levels
-1,0,1,2; it differs from Q2_K. The embedding is IQ4_XS and target head Q6_K.
Persistent weights remain packed; only current Q2_0 shared-memory tiles expand.
Small F16 convolution weights widen exactly to F32. Mixed QKV/gate projections
keep their own formats; TP requires whole quantization blocks.

Use the public model page for launch options. The tested setup has64K capacity,
2 sessions/1 active request, thinking off,1GiB prompt cache and the existing
shared-Q8_0 MTP artifact. Disk-backed n-gram operation remains enabled. No kernel,
driver, system ROCm, boot or power changes were made. The approved service is
restored after each isolated experiment; the IQ3 runtime is not adopted silently.

## Correctness and fixes

- CPU codec tests cover all65536 half scales, signed packing, unaligned input,
  TP320/640 geometry and guards. Reader, provider probe and mixed TP tests pass.
- Packed GPU tests pass242 cases/1505765 values. Independent CPU dequantization
  and FP64 dots use exactly quantizable activations; additional grouping checks
  compare batch/tile widths bitwise with nonintegral inputs and variable scales.
  These distinct oracles must not be described as all independent FP64 checks.
- Real-weight spot checks cover14 cases/7392 values across all seven new packed
  formats, with a predeclared FP32 gamma_(2K+8)*sum(abs(products)) bound. Q6 dense
  rows repacked as experts check arithmetic, not scheduling. No tolerance widened.
- Full-model repeated greedy text/code, prompt-cache reuse and cached versus cold
  continuation pass. The dedicated copied-token regression proposes and accepts
  109 lookup tokens, exercising wide TP execution.
- Final native-MTP versus AR greedy output matches for story, code and long
  continuation. Seed513/temperature0.8 replay is exact; the generated pure
  integer-comparison function passes four independent execution cases.
- Full/rank-local wide recurrent allocation checks pass at world sizes1 and2.

Tests found fractional-scale integer truncation in IQ2_S/IQ3_XXS vectors and
unwritten inactive tiled slots. Both are fixed. Layer-aligned real-model tracing
then isolated batch-dependent arithmetic: library BF16 K-reduction rotation and
FP32 reassociation/contraction in four quantized MMA helpers. The existing native
BF16 GEMM covers the new shapes, and lexical FP controls preserve MMA arithmetic.
Short prefill tails explicitly retain matrix arithmetic. A captured real Q4_K
gate changed from2598/3072 unequal values (maximum2.86102e-6) to bitwise equality
across widths1/170/1122. BF16 chunk tests fail before and pass after the fix.

Copied-token execution also exposed missing packed DenseBatch dispatch and TP2
wide-state copies/scratch strides sized with full-model heads. They use rank-local
heads throughout; packed batches preserve scalar decode arithmetic in chunks.
These state geometry defects are general TP2 fixes, not quant-specific behavior.

## Matched performance

Same final production binary, ROCm7.14.1/gfx1151, both hosts, TP2 native MTP,
identical requests/settings, three repeats, fixed128 output tokens. Decode prompts
contain54/60/50 tokens; long prefill contains2113 tokens. Values are medians.
Weights, resulting tokens and draft acceptance differ between quantizations; this
is fixed output-count comparison, not identical mathematical work or a significance
claim from three samples.

| Measurement | IQ3_S GSQ-RCO | UD-Q4_K_XL |
| --- | ---: | ---: |
| Story decode, tokens/s | 40.37 | 54.32 |
| Explanation decode, tokens/s | 50.63 | 52.57 |
| Code decode, tokens/s | 49.00 | 69.16 |
| 2113-token prefill, tokens/s | 899.77 | 1770.73 |
| 2113-token TTFT, ms | 2352.0 | 1202.4 |
| GPU allocated at load, rank0/rank1 MiB | 40391/39401 | 53103/52113 |

The saving is12712MiB (12.414GiB) per node, about24%. GPU allocation is not total
system RAM use. Current generic mixed-format paths and the required deterministic
BF16 path need profiling before a speed optimization claim. Keep Q4 for speed
unless the memory saving matters more.

## Evidence and limits

[Portable evidence](gsq-rco-evidence/) retains raw operator/before-after logs,
layout results, per-request timing CSV, benchmark summaries and quality records.
The opt-in API regression is `tests/functional/flash_next_mixed_quant.py`; it checks
actual cache and copied-token work, rather than treating a successful load as proof.

Full local commands, models, build manifests, failure logs and requests/responses
are retained under `~/strix-dispatch-investigation/iq3-s-gsq-rco-2026-11-15`.
The final build is `gpu-build-v17`. The selected-extension BUILD_TESTING registry
has a pre-existing DeepSeek dependency failure; GPU tests were linked explicitly
against all candidate CMake-built archives, with exact commands retained. This is
not a successful normal GPU CTest build. Shared formatting reports seven untouched
pre-existing failures; changed non-vendored C++ files pass. Earlier failed runs and
an invalid callback-alignment comparison remain labeled and excluded from results.
