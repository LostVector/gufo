# Flash Next DFlash2

This single-node ROCm implementation adds the `dflash2` speculative backend
alongside native MTP. It shares the current Qwen DFlash runtime, including its
prefix-plus-request-table draft head. The target always verifies proposals over
the full vocabulary; sampled verification uses the exported proposal masses,
rejection correction and the request's RNG state. Prompt lookup can replace a
suffix of neural proposals with point-mass proposals.

## Run

Use a Flash Next target and its matching DFlash2 GGUF. The target supplies the
embedding and output tables; supported donor formats are checked before GPU
execution. The draft is not interchangeable with a different model's sidecar.

```sh
gufo serve llm --model /models/flash-next/target-00001-of-00004.gguf \
  --speculative dflash2 --dflash-model /models/flash-next/dflash2-Q8_0.gguf \
  --draft-policy adaptive --draft-tokens 7 --context 8192 --sessions 1
```

`--draft-policy fixed` retains a fixed neural proposal limit. Adaptive policy
selects the limit from private acceptance history. Both modes preserve target
verification. Explicitly selecting native MTP continues to use `--mtp-model`.
No additional switch is needed for the shared smaller draft head or prompt
lookup. A request with unsuitable vocabulary falls back to the full draft head.

## Continuations and scope

Each session owns its feature capture, draft KV state, length controller and
lookup state. Component-cache captures include the external draft state and
reject incomplete or incompatible restoration. The vocabulary table is derived
from committed tokens and rebuilt after restore; it is not extra persisted state.
Snapshot compatibility version 21 distinguishes the new payload layout.

This port supports serial draft execution and session interleaving. It does not
add TP/RDMA, batched DFlash execution, or a new transport to upstream. Target
prefill passes currently end at external-draft checkpoint boundaries so draft
and target state are captured at the same frontier. This is a correctness
constraint, not a claim of zero-cost in-pass draft capture. Host feature staging
and per-checkpoint draft serialization remain optimization opportunities.

DFlash2 is not automatically faster than native MTP. Existing downstream
single-stream prose comparisons favored MTP. A shared-head projection reduction
alone does not establish an end-to-end gain; compare matched prompts, model
quantization, context depth, sampling, stopping and concurrency.

The completed independent review and matched native AR/MTP controls are in
[DFLASH2-REVIEW.md](DFLASH2-REVIEW.md) and
[DFLASH2-NATIVE-CONTROLS.md](DFLASH2-NATIVE-CONTROLS.md). Lifecycle timing
flags remain inconclusive; no per-request regression flag was observed.

## Focused qualification

Configure a HIP test build for `gfx1151`, then build these targets:

```sh
cmake --build build --parallel 4 --target \
  qwen38_flash_next_dflash_donor_test \
  qwen38_flash_next_dflash_lookup_test \
  qwen38_flash_next_dflash_features_test \
  qwen38_flash_next_dflash_owned_test \
  qwen38_flash_next_dflash_component_test \
  qwen38_flash_next_dflash_sampled_test \
  qwen38_flash_next_dflash_lookup_model_test qwen_dflash_gpu_test
ctest --test-dir build -R '^qwen38_flash_next.dflash_(donor|lookup|features)$' \
  --output-on-failure
```

The model tests require positional target-first-shard and draft-GGUF arguments.
Run `dflash_owned` and `dflash_component` again with a final `adaptive` argument.
The owned-session test asserts that eligible requests actually use the smaller
head, checks greedy AR parity, sampled seeded replay and snapshot restoration.
The sampled test checks p/q rejection against a scalar target reference in both
compact and full target-distribution modes. The component and lookup tests
cover long prompt boundaries, private-piece restoration, EOS and context limits.
The legacy `qwen_dflash_gpu_test` takes a Qwen27B target and its matching draft.
Missing model arguments are skips, not qualification passes.

For HTTP continuation coverage, run `tests/functional/run.py` with the affected
`long-context`, `cache` and `state-edges` suites. Retain raw request timings and
compare against a matched baseline separately; capturing a candidate reference
with `--record-baseline` does not qualify performance regressions.
