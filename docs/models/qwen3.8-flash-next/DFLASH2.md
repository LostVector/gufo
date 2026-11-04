# Experimental native DFlash2

Flash Next can attach an external DFlash2 drafter through `--speculative
dflash2`. The implementation supports one GPU and the fork's two-rank native
RDMA path. It reuses the shared Qwen DFlash implementation while capturing
Flash Next's hyper-connection features for the draft.

This is a draft implementation. Correct target output and cache replay have
been tested. Independent author-operator comparisons now qualify the covered
draft execution cases using identical supplied features and converted weights;
see below. This does not establish equivalence to the original FP8 target,
unquantized BF16 draft checkpoint or published acceptance benchmarks.

## Checkpoint and launch

The tested draft is the author's
[`incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF)
checkpoint, revision `4f5a39be8f0a9e9a573ac9824251830a8f4096d9`, file
`Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` (652,101,024 bytes).
SHA256: `541f4d6ee230747a34d4a2c6890e6e08cc8b19c461226dc1829701a544e46e8a`.

```sh
gufo serve llm --model /path/to/target-first-shard.gguf \
  --speculative dflash2 \
  --dflash-model /path/to/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf \
  --draft-policy adaptive --draft-tokens 7 \
  --context 65536 --sessions 2
```

For two nodes, retain the ordinary matching [TP2](TP2.md) options on both ranks
and use the same target, draft, policy and binary on both machines. The control
handshake distinguishes native MTP from DFlash and fixed from adaptive policy.
Rank 0 plans each width; both ranks verify agreement on the complete token and
sampled-probability inputs before target execution. Native `--mtp-model` and
external `--dflash-model` are mutually exclusive.

Borrowed embedding/head formats must be supported by the shared draft kernels.
F16 donor tables are rejected at load time with the table role in the error;
they are not silently interpreted as F32 or packed quantization. The shared
factory also validates row geometry, element count and available storage.

## Implementation and validation

- Captures the mean of raw hyper-connection streams after the selected layers,
  before the next layer's normalization. GGUF taps `[6,16,26,36,46]` are mapped
  to completed zero-based layers `[5,15,25,35,45]`.
- Uses native target verification/rollback for greedy and sampled drafts,
  including residual sampling and deferred EOS. Draft ownership is tied to its
  matching target without a model/draft reference cycle.
- Persists draft KV and adaptive policy state with target snapshots, validates
  compatibility and includes the additional allocations in memory accounting.
- Tests include greedy AR parity, sampled scalar-target verification, seeded
  replay, component snapshots, lifetime checks, malformed proposals, tools,
  structured output, cancellation, long context and disk restart.
- The TP request path also forwards `cache_persist` to scheduler metadata;
  previously the TP path silently used the default `true`.

GPU tests requiring external weights are excluded from the default build. Build
the affected `qwen38_flash_next_dflash_{prototype,sampled,owned,component}_test`
targets and invoke each executable with the target first-shard path followed by
the draft path. Invoking them without model paths is a skip, not a pass. The
`qwen38_flash_next.dflash_features` operator test requires no model weights.

## Initial measurements

Two Strix Halo machines, native RDMA, Release/O3, UD-Q4_K_XL target, Q8 draft,
65,536 context capacity, two session slots and one request at a time. The fixed
workloads generated 256 tokens with EOS ignored; prompt caching/persistence was
disabled. Each mode used one warmup and three retained repeats within one server
launch. Natural-stop behavior was tested separately.

| Workload | AR decode tok/s | Native MTP decode tok/s | DFlash2 decode tok/s |
|---|---:|---:|---:|
| Code continuation, 1,917 prompt tokens, greedy | 38.11 | 122.91 | 101.06 |
| Prose continuation, 22 prompt tokens, sampled | 38.52 | 57.33 | 43.18 |

Code-prefill medians were 1,903.69 / 1,826.67 / 1,355.92 tok/s respectively.
These initial measurements predate Flash Next DFlash prompt lookup. Native MTP
accepted 148 copied tokens per retained code request; DFlash then had no lookup
composition. They are configured-mode measurements, not isolated neural-drafter
comparisons or confidence estimates. See the newer single-node lookup screen
below; its settings and hardware count differ from this historical TP2 table.

The sampled prose request accepted 134 of 432 proposals (31.0%), while the code
request accepted 215 of 267 (80.5%). These ratios are workload-specific and
include rejected proposal tails. The author reports output tokens per
verification step under different target quantization, widths and workloads;
these ratios cannot establish agreement with that reference.

The fixed-length prose run continued past `<|im_end|>` into another assistant
turn and thinking text. It is useful as a retained fixed-work timing probe,
but is not an ordinary completed chat or a representative draft-quality test.
Also, `draft_rounds` counts speculative cycles and excludes AR-only forwards;
it is not generally an exact count of every target call.

### Bounded chat acceptance follow-up

A follow-up used one Strix Halo GPU, 8K context, the same Q4 target/Q8 draft,
fixed three-token drafts, temperature 1.0/top-p 0.95/top-k 20, `xhigh` reasoning,
neutral repetition/frequency/presence penalties and normal EOS stopping. Prompt
caching/persistence was disabled. Native RDMA devices were absent, so this is
not a new TP2 performance comparison. These are two hand-written prompts with
two seeds, not a reproduction of the author's benchmark datasets.

| Task / seed | Output tokens | Accepted / proposed | Ratio | Output / speculative cycle | Finish |
|---|---:|---:|---:|---:|---|
| Arithmetic / 43 | 120 | 76/132 | 57.6% | 2.73 | stop |
| Python / 43 | 476 | 314/486 | 64.6% | 2.94 | stop |
| Arithmetic / 44 | 96 | 61/105 | 58.1% | 2.74 | stop |
| Python / 44 | 512 | 326/556 | 58.6% | 2.75 | length; reasoning censored |

The measured target's two donor tables are Q8_0, so rejecting F16 changes none
of this path's numerical operations. The higher ratios do not demonstrate a
numerical fix or isolate one cause: width, framing, stopping, sampling and TP
configuration differ from the earlier probe. They do show that 31% was not a
general property of the checkpoint. Proposed-token ratios also include suffixes
never tested after an earlier rejection, so they depend strongly on block width.

For these four rows, output minus accepted proposals equals the reported
speculative cycles; that is a useful cross-check, not a reason to reinterpret
the counter for modes that also perform uncounted AR-only forwards. Author
feature/logit/selector parity remains unverified. Compare independent execution
with identical captured features, dequantized weights and donor tables before
attributing remaining differences to quantization or the model itself.

## Prompt lookup

Flash Next DFlash now composes the native prompt-lookup policy with its neural
drafts automatically. A sufficiently strong match can bypass the neural model;
otherwise a retained neural prefix may establish a copied suffix. No neural q
row follows a replaced token. Sampled copies use exact point-mass q rows and
ordinary target p/q verification/residual correction, with constraints and EOS
preserved. The first request step copies nothing, and sampled requests retain
the native longer-match and successful-copy resume gates.

Copies can fill the reserved target verification budget beyond the current
neural width, within request/context/model limits. Target features for every
committed copied token are still injected into draft KV. Existing metadata
already persists the lookup cursor/request boundary; no snapshot format or
public flag changed. The DFlash controller observes the retained neural prefix,
excluding copy successes/failures. Existing `lookup_n`/`lookup_n_accepted` metrics
now report copied proposals separately from total draft/accepted counts.

### Matched single-node screen

Same Release toolchain, Q4XL target/Q8 draft, context capacity 8K, one session,
chunk 512, cache/persistence off. Frozen code (1,917 tokens) and prose (22 tokens) prompts; 256 generated tokens,
EOS ignored for timing, one warmup plus three repeats within each launch.
Code is greedy; prose uses temperature 0.8, top-k 40, top-p 0.95 and seed 513. Omitted
penalties retain server defaults. These differ from the neutral chat probes.

| Workload | DFlash before / after controls | DFlash + lookup | Native MTP |
|---|---:|---:|---:|
| Code decode tok/s | 86.21 / 86.43 | 94.68 | 89.91 |
| Prose decode tok/s | 40.87 / 40.90 | 40.83 | 43.32 |

DFlash accepted 136 of 147 proposed lookup tokens per retained code request; native
MTP accepted 143 of 155. All modes' greedy code text matches exactly. DFlash's no-copy
prose text and draft counts also match both controls exactly. The observed code
gain is 9.5–9.8% over DFlash controls and 5.3% over native MTP on this fixture.
These are descriptive results, not a process-pair confidence study or a general
MTP/TP2 advantage. Repeated requests in one launch are not independent starts.

CPU q tests cover pure/mixed/extended rows, unchanged neural conditioning,
point-mass rejection identity and malformed input. Real target tests cover
active lookup beyond a one-token neural plan, object/serialized replay with
complete unmasked target-logit bits, new-request/EOS gates and 52 certain sampled
copy acceptances. Six sampled modes compare real verification with a scalar
target teacher. Existing owned/adaptive/component tests pass, including the
5,009-token boundary and failed-restore recovery.

All 11 HTTP long-context/cache/cancellation/disk-restart case reports pass on
the final source, with no quality/coverage changes. Its timing comparison also
passes the unchanged 5% and 10 ms limits. An earlier candidate run was
**inconclusive** because startup/restart were 25.8/25.7 s versus 27.9/27.9 s;
those flags remain recorded. A focused startup-only control measured unchanged
baseline 25.7 s, candidate 27.5 s, unchanged baseline 31.2 s. This demonstrates
baseline variation, not startup equivalence. Final review also restored the
original context-full stop sentinel, with an explicit GPU boundary regression;
the final Release binary was rechecked against the same frozen request hashes
and in the HTTP suites. Earlier B/C/B and native-MTP controls are retained.
No fresh two-node lookup test was possible with native RDMA devices absent.
No driver/runtime/system setting was changed.

[Machine-readable lookup results](dflash-lookup-v1.json) retain per-request
throughputs, copied-token counts, source/binary hashes and timing flags.

## Independent operator parity

The trace probe and shared oracle compare actual captured Flash Next features
and the same Q8 GGUF/donor weights with unmodified
[pinned DFlash2 operators](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py).
The reference is PyTorch 2.15.0+cpu / Transformers 5.12.1, eight CPU threads.
Weight decoding is independent of Gufo's GPU upload path. Production kernels
and sampling checks are unchanged.

| Captured state | Prefix tokens | Draft positions | Matching candidate sets | Max full-forward q total variation |
|---|---:|---:|---:|---:|
| Short prefix | 24 | 3 | 3/3 | 2.04e-6 |
| Sliding window and KV ring wrap | 2,064 | 7 | 7/7 | 1.20e-7 |
| After forced rejection and one-row committed injection | 2,065 | 7 | 7/7 | 5.98e-8 |
| After persistent draft restore | 2,065 | 7 | 7/7 | 5.98e-8 |
| Presence 1.5, including earlier proposed tokens | 24 | 7 | 7/7 | 1.15e-6 |

All 31 conditional candidate sets match. Worst stage relative RMSE is 2.25e-5;
worst full-logit maximum error is 1.82e-5. The original strict limits remain
1e-4 stage relative RMSE, 1e-3 logit maximum error and 1e-4 q total variation,
with exact candidate membership. The presence row independently extends the
author's scores with Gufo's requested penalty policy; it is not a claim that
the original author model implements that API policy. State restoration also
preserves Gufo's proposal tokens and probabilities bit-for-bit.

Separate diagnostics round the same converted weights to BF16, then also run
BF16 activations. On the short trace, weight-only rounding changes q by up to
0.00216 total variation; full BF16 execution changes it by up to 0.0232 and
changes one of three candidate sets. These diagnostics are excluded from the
strict FP32 pass decision. They compare precision contracts over already
quantized weights, not the original unquantized checkpoint, and do not establish
whether BF16 would improve live acceptance.

### Independent target-feature diagnostic

A one-token run also compared the five GPU-captured raw HC stream means with
Gufo's separate scalar target implementation, using the same Q4 target weights.
Against scalar FP32 formulas, relative error was 0.12–0.43%, cosine similarity
was at least 0.9999909 and norm ratios were 0.999986–1.000584. Against its decode
storage emulation, relative error was 1.06–2.86%. These are diagnostics without a
predeclared pass threshold, not bitwise or original FP8-target equivalence.
They argue against a gross tap, normalization or stream-layout error for this
first token; they do not qualify later-token attention/cache behavior.

### Draft quantization check

The official BF16 GGUF from the same pinned repository was downloaded and its
LFS SHA256 verified:
`fa63c62374607095cfb07441abfe8c18982ae99d130472e2ea0da35c61d98b20`.
Full header comparison, including tokenizer/mask metadata, differs only in
`general.file_type`. Holding supplied features, Q8 donor tables and FP32
execution fixed isolates the change in draft checkpoint weights.

On one presence-policy row, a token moves from unary rank 16 to 17, removing
its 0.280938 proposal probability. The conditional Q8-versus-BF16 distribution
distance is 0.280938. Retaining the Q8 candidate membership while keeping BF16
draft scores/hidden/codebooks reduces that distance to 0.028525. This is a
counterfactual diagnostic of the hard top-16 gate, not a proposed new selector.

The native target distribution at that same forced prefix gives both proposal
distributions zero expected acceptance after top-k/top-p filtering. Thus this
large distribution change does not demonstrate an acceptance loss there. The
conditional metric is `sum(min(p, q))`; it does not integrate over alternative
generated histories or establish a whole-request acceptance rate.

Four matched-setting single-node chat requests observed 777/1279 accepted
proposals with Q8 (60.8%) and 818/1281 with BF16 draft weights (63.9%). Output
tokens per reported speculative cycle were 2.82 and 2.92 respectively. Both use
Gufo's FP32 arithmetic and the same Q4 target. Sampled trajectories differ and
some reasoning was capped; these four requests do not establish statistical
significance, a speedup, or parity with the original FP8-target evaluation.

The same-Q8 weight reload control passes the strict distribution tolerance.
A negative-control oracle that deliberately omits earlier draft-token presence
penalties fails with a selector-probability mismatch. Independent harness review
caught and verified fixes for that prefix accounting and for separating BF16
diagnostics from strict FP32 results. No strict numerical tolerance was relaxed.

These tests support draft execution parity for the covered inputs and state
paths. They do not validate the GGUF converter, the original target's hidden
features, full 64K behavior, TP2 numerical parity or published acceptance scores.
The CPU oracle and single-node probes do not require changing the RDMA driver.

### Reproduction

Build `qwen38_flash_next_dflash_trace` explicitly in the HIP test build. It is
excluded from normal builds and is not an automatic model CI test:

```sh
qwen38_flash_next_dflash_trace TARGET_FIRST_SHARD DRAFT_GGUF NEW_TRACE_DIR
qwen38_flash_next_dflash_trace TARGET_FIRST_SHARD DRAFT_GGUF NEW_FEATURE_DIR 1 3 features
qwen38_flash_next_dflash_trace TARGET_FIRST_SHARD DRAFT_GGUF NEW_RING_DIR 2064 7 rejection
qwen38_flash_next_dflash_trace TARGET_FIRST_SHARD DRAFT_GGUF NEW_POLICY_DIR 24 7 presence
```

The `features` scenario exports GPU means and independent scalar FP32/decode
storage means for one token; compare its `target_features*.f32` arrays as five
contiguous 2560-element vectors. The scalar observer runs after each completed
transformer layer and averages its raw wide residual across HC streams.

The rejection scenario also exports `after-rejection`, `after-restore` and
`with-presence-penalty` subdirectories. Run the oracle on each desired trace:

```sh
python tools/qwen27b/dflash_reference.py \
  --upstream PINNED_AUTHOR_MODEL_PY --config PINNED_DRAFT_CONFIG_JSON \
  --target DONOR_TABLE_SHARD --draft DRAFT_GGUF \
  --trace TRACE_DIR --output REPORT_JSON \
  --device cpu --external-env --bf16-diagnostic
```

Use `--alternative-draft BF16_DRAFT_GGUF` to measure draft-weight changes with
the supplied features/donor tables fixed. The report's `passed` field describes
the baseline strict execution check; alternative weights are diagnostics, not
an additional pass criterion. An identical-file reload is explicitly checked
as a control. The `presence` trace scenario also exports native target AR logits
for conditional overlap calculations; it uses synthetic generated-token history
to exercise policy, not a published benchmark conversation.

`DONOR_TABLE_SHARD` must contain the embedding and output tensors: shard 2 in
the tested split target. The oracle verifies the pinned author source hash.
`--external-env` explicitly records an environment outside Nix; the default
CUDA/Nix behavior is retained. Use fresh trace/output paths to preserve prior
results. Machine-readable retained results are in [dflash-parity-v1.json](dflash-parity-v1.json).

## Known limitations and review priorities

1. Extend the same-input oracle to more representative histories, deeper
   positions and TP2. Original FP8-target features and unquantized-checkpoint
   equivalence remain separate questions from the execution parity above.
2. Capture sessions currently disable paired TP prefill and use blocking feature
   downloads/reuploads. Their contributions have not been separately profiled.
3. The adaptive cost calibration is inherited from 27B, not measured Flash Next
   costs. Fixed/adaptive width comparisons remain to be done.
4. DFlash execution is serial. Two session slots do not imply batched DFlash or
   a C2 throughput improvement.
5. Tests exercised sequences up to about 6.6K tokens. Successful 64K capacity
   allocation is not qualification of a full 64K prompt or continuation.
