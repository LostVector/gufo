# Canonical batched target-FFN HC fusion qualification

Candidate `6ec3f75f338d9f2154ef60e1c14e2dd2c70ff53b` is based on canonical
`8ab834a5902bf19341750daecf27516a57840dd1`. The production diff is confined to
the immediate target-FFN combine/mix pair in `batch.cpp`. It reuses the existing
fused route, including its quantized-activation producer, only when its geometry,
format and residual-state guards allow it. This is not a pure launch-only change.
There is no RDMA dependency, new switch, or public extension-API change.

The maintained TP2 branch has not adopted this candidate. The related fork draft
is [LostVector/gufo#15](https://github.com/LostVector/gufo/pull/15); its TP2 study
did not establish a statistically significant serving improvement. Do not
substitute that study for canonical single-node qualification.

## Completed correctness evidence

- `operator/validation.json`, `operator/tests.log` and `operator/operator-tests.xml`:
  fresh canonical GPU-test build, complete existing HC fixture plus 48 additive
  batch-reference conditions, real HIP, assertions enabled, zero skips.
- `../upstream-ffn-hc-state-v1/analysis.json`: 104 records, 64 matched replay
  records, 32 materialized state pairs, context depths 0 and 32768, sampled and
  greedy decoding. Full vocabulary scores, tokens, RNG, positions, accepted work
  and restored state are checked by the diagnostic harness. Borrowed, serialized
  and file-backed restore are covered.
- Diagnostic greedy execution uses a fixed alternating 1/0 shared draft plan to
  compare identical histories. Sampled planning is unchanged. This is not normal
  greedy-controller replay, an independent model teacher, or throughput evidence.
- State-test source and every production `src/`/`include/` input are frozen in
  its protocol/build manifests. The earlier preparation-anchor failure remains
  in `preparation-note.txt`; no numerical assertion was weakened to pass it.
- Canonical extension API is 0.1.15; no private headers/libraries from the fork's
  API 0.1.16 are used.

## Production-build and serving procedure

From `/home/rkuo/strix-dispatch-investigation`:

```bash
bash tp-cache-retention/upstream-ffn-hc-v1/build-release.sh
bash tp-cache-retention/upstream-ffn-hc-v1/build-baseline-release.sh
python3 tp-cache-retention/upstream-ffn-hc-v1/verify-release.py
python3 -u tp-cache-retention/upstream-ffn-hc-serving-v1/run.py
python3 tp-cache-retention/upstream-ffn-hc-serving-v1/analyze.py
```

These are retained exact commands, not instructions to overwrite an existing run:
the serving controller refuses to replace its protocol. Inspect live handles
before a retry and preserve failures in a fresh named experiment if needed.

Both release builds use the same private ROCm 7.14.1/compiler/dependency tree,
the official release preset, `BUILD_TESTING=OFF`, gfx1151 and four build workers.
Neither config enables RDMA. The source proof compares normalized compile
commands and records binary and source identities. The operator executable and
diagnostic state executable are never used for production timing.

The bounded serving screen fixes one fresh baseline/candidate/baseline sequence,
two session slots, context capacity 65536, chunk 2048, RAM checkpoint cache 1 GiB,
UD-Q4_K_XL target weights, shared-Q8_0 MTP, thinking disabled and greedy sampling.
It runs warmup, two actual pp2048/tg128 requests and two distinct coding prompts
solo/concurrent/solo. All requests disable prefix caching. It records normal MTP
work, exact completions and cache work, raw Gufo phase metrics and client content
arrival timestamps. Context capacity is not actual 64K prompt depth.

The three server starts are sequential under the model-matrix lock. No profiler,
NVMe admin sampler, system setting change, peer inference, or full-table warming
is introduced. The controller checks binary/PID/source identities and guards
8 GiB of available host memory. Owned subprocesses are stopped on completion or
failure; Halogen is not restarted.

Per-request candidate phase increases exceeding both 5% and 10 ms are retained
against each control, along with all adaptive draft/acceptance differences. A
single B/C/B screen cannot establish statistical significance, equivalence, or
qualification of C4/C6/C8. It does not replace the existing inconclusive TP2
study or justify repeating that study until a favorable result appears.

## Original HTTP accounting and first-gap investigation

The original 27-request B/C/B runtime completed with all server/worker exits0.
The frozen analyzer failed its erroneous assumption that max_tokens128 requires
128 generated tokens. All six pp2048 replies naturally stop96 identically on both
controls and candidate. Original protocol/harness/failure remain intact;
`../upstream-ffn-hc-serving-v1/analyze-observed-eos.py` is explicitly separate,
posthoc analysis preserving every other assertion. No actual pp2048/tg128 claim.
All coding requests do generate128, with exact solo/concurrent/solo output.

Six mandatory release phase entries flag C2 actor0 first gap108.8ms versus
71.5/70.3ms and maximum interval108.7ms versus80.5/81.3ms; eight adaptive-work
differences remain. They are not waived. Source-stage v1 captured the high gap
on unchanged upstream109.43ms; that gap ended15.57ms before its first target
ForwardBatch started. This does not attribute the original candidate flag.

Source-stage v2 additionally records serial DecodeStep and prefill Feed spans.
All three starts were low-gap68.5/70.9/68.3ms, with exact outputs. One baseline
gap is predominantly64.1ms serial decode; candidate/laterbaseline include a
batched cycle. Engine requests.front() positions alone cannot identify actors,
since request order may change. Use timestamps and recorded session identities.

`../upstream-ffn-hc-first-gap-repro-v1/` is a baseline-only event capture with a
predeclared maximum6fresh starts, stopping on first C2actor0 firstgap>95ms. It
reuses the exact v2 diagnostic binary and original nine-request history. This
adaptive stop is for bad-state capture only: never use it as a performance or
significance comparison, and retain every earlier start and original flag.
