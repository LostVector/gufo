# Experimental native DFlash2

Flash Next can attach an external DFlash2 drafter through `--speculative
dflash2`. The implementation supports one GPU and the fork's two-rank native
RDMA path. It reuses the shared Qwen DFlash implementation while capturing
Flash Next's hyper-connection features for the draft.

This is a draft implementation. Correct target output and cache replay have
been tested, but draft logits/features have not been independently matched
against the author's implementation. Rejection sampling can preserve target
output even when draft quality is wrong. The current results therefore do not
establish reference-equivalent draft fidelity or acceptance.

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
Native MTP accepted 148 prompt-lookup tokens per retained code request; DFlash
does not yet compose its drafts with that lookup path. These are configured-mode
measurements, not isolated neural-drafter comparisons or confidence estimates.

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

## Known limitations and review priorities

1. Independently compare captured features, draft logits and selector
   probabilities with the pinned author implementation. Target verification
   tests do not replace this check.
2. Capture sessions currently disable paired TP prefill and use blocking feature
   downloads/reuploads. Their contributions have not been separately profiled.
3. The adaptive cost calibration is inherited from 27B, not measured Flash Next
   costs. Fixed/adaptive width comparisons remain to be done.
4. DFlash execution is serial. Two session slots do not imply batched DFlash or
   a C2 throughput improvement.
5. Tests exercised sequences up to about 6.6K tokens. Successful 64K capacity
   allocation is not qualification of a full 64K prompt or continuation.
