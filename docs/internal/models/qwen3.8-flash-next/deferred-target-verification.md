# Deferred target verification over TP2

Each rank keeps its projected vocabulary half and exchanges the candidates that
native sampled MTP verification consumes. Full rows remain available on demand.
This reduces transfer without changing target probabilities or random draws.

## Contract

For each verification row, select each partition's top 64 while excluding all
penalized token IDs. Exchange original candidate score bits and every owned
penalty score, then select the stable global top 64 and append the original
penalty scores. The existing CPU FP64 sampler and rejection/correction RNG stay
unchanged. A global top 64 that ignores penalties is insufficient.

Parked logits have separate storage from expanded full rows. Track their session,
row count and materialization mask. Gather an original full row for rollback,
constraints, logprobs or other full-logit consumers; never expand in place over
another parked row. Clear metadata on new dispatch/view changes. A layer hook
cannot yield with live partial verification rows; clear a nested decode's metadata
before restoring the parent prefill view.

Host-window peers negotiate wire version 27 and collective header version 6.
The header adds a semantic tag within the existing 64-byte payload offset.
Full-row tags identify row/count; compact tags hash row/penalty layout and IDs.
Reject equal-sized but differently tagged exchanges before accepting results.
Unsupported, batch, wide, greedy and external routes retain their full gather.
Check queued completion status after draining. Abort communication before draining
when a queued operation fails, so the peer and stream wait can terminate.

## Reproduce focused tests

Configure the repository's `cpu-test` preset, then build and run the two small
state fixtures. They extract the actual method bodies and compile them with
simulated HIP calls; they check state/order, not GPU execution.

```bash
cmake --preset cpu-test
cmake --build --preset cpu-test --target \
  qwen38_flash_next_deferred_verification_state_test \
  qwen38_flash_next_deferred_verification_hook_test
ctest --test-dir build/cpu-test -R 'qwen38_flash_next.deferred_verification_' --output-on-failure
```

The state fixture covers 24 cases: both ranks, lazy/repeated rows, owner/view/range
rejection, callback failure/exception, abort-before-drain and reset. The hook
fixture covers nine cases: half/F32 scratch, deferred residual/prefetch combinations
and live-partial rejection. The GPU `verification_sampling` test checks split
candidate bits, independent FP64 distributions and verification RNG, including
unaligned controls; see the test source for its full shape matrix.

With HIP and RDMA enabled, explicitly build
`qwen38_flash_next_tp_target_verification_probe`. Run both ranks concurrently;
choose the head's management IP and local RDMA device names for your hosts.
This synthetic-logit probe uses bootstrap port 29769, so reserve it for the test.

```bash
./build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_tp_target_verification_probe \
  0 HEAD_IP HEAD_RDMA_DEVICE matching /tmp/target-rank0.txt
# On the peer, concurrently:
./build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_tp_target_verification_probe \
  1 HEAD_IP PEER_RDMA_DEVICE matching /tmp/target-rank1.txt
```

Repeat with `row-mismatch` and `signature-mismatch` in place of `matching` on
both ranks. Each mismatch must report rejection. Matching checks compact results,
reverse-order full-row materialization and immutable parked data for 2/3/7/8 rows.
It does not load model weights, emulate a network or measure pure wire latency.

## Measured scope and limits — November 14, 2026

A fixed 20-start, ten-pair balanced-order study on two Max+ 395 hosts measured
**+0.4653% sampled decode**, bootstrap 95% interval **[0.1964, 0.6395]%**, exact
balanced-order two-sided **p=0.03968**; nine pairs positive. All 100 output/work
records match; all 40 selected transport-error windows are clear. Parent is the
compact-proposal implementation, not the original serving build. Earlier gains
are separate studies, not a measured stacked total.

UD-Q4_K_XL target/shared-Q8 MTP, 64K capacity, two slots/one active, 1 GiB cache,
thinking off, seed 513/temp .8/top-k 40/top-p .95. Actual short/2106-token prompts
use two prefill chunks with maximum 36/2096 tokens. Legacy chunk/draft CLI flags
are ignored; runtime reports draft limit 7. Completion lengths are 121/115/128.

Four prefill flags remain unresolved: explanation +49.510/+32.525/+49.946 ms,
2K scene +66.832 ms. This is a decode result, not all-phase latency qualification.
Older read-path traces explain similar delays but do not attribute these runs.
No driver, power, kernel, cache-warming or serving-default change was made.

Final-runtime mixed-wire rejection and sampled text/tool continuation checks pass.
Disk restoration does not guarantee exact replay in that fixture, even though
observed comparisons match. General dynamic-grammar transition coverage and a
complete cancellation/error matrix are not established by these tests. Keep
these limits visible before deployment or broader performance claims.

Local raw evidence is retained under
`target-compact-verification-2026-11-14/confirmation/` in the investigation archive:
`analysis.json`, `case-pairs.csv`, `randomization.json`, `prefill-all-starts.csv`,
per-start requests/counters/logs, source/runtime pins and restoration proof.

The portable [paired measurements](deferred-target-verification-pairs.csv) and
[statistics script](deferred-target-verification-statistics.py) reproduce the
primary result without a model or workstation-specific paths.
