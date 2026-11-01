# Let completed TP2 requests finish between decode calls

After the scheduler publishes its terminal result, the HTTP request still calls
`TpRequest::Finish`: it sends the completed request's instruction count/digest to
rank 1 and waits for its verdict. `TpMirroredRunner::EndRequest` previously acquired
the shared recursive call mutex without announcing a pending turn. A sibling
could repeatedly reacquire that mutex for another decode, delaying the terminal
HTTP response after the last content token had already reached the client.

Reserve a pending turn with the existing snapshot gate before acquiring the
mutex, and release that reservation once acquired. Already running GPU work
finishes normally; subsequent decode/advance calls yield at their existing gate.
The instruction stream, verdict checks and model math remain unchanged. There is
no additional collective or option. This does not preempt an active call or add
ordering to prefill calls.

## Evidence and limits

On the parent release build (`ac8f6ed7`), an untraced coding request delivered its
last content token slightly earlier in the batched-head arm but returned its final
HTTP result later: the post-content tail was 40.0 versus 824.9 ms. Long tails also
occurred in the ordinary-head control, up to 749.9 ms. These observations motivated
this separate fix; they are not measurements of this fix's gain.

CPU traces excluded scheduler completion and commit as the slow phase. Direct
`Finish`/`EndRequest` spans place 98–99% of observed 32–54 ms finish time in the call
mutex; worker response then takes less than 1 ms. A separate 399.7 ms tail overlapped
eleven uninterrupted peer decode cycles. This is application finalization, not
per-token compute or the USB4 collective's transfer latency.

The included sparse captures replay the same twelve-request history on actual
TP2 models, three fresh processes per arm. Both use the combined target-head
release implementation. Only the candidate's pending handoff differs. Both arms
have the same bounded in-memory markers around `Finish` and `EndRequest`, with no
markers in the greedy decode path. All 72 request output/prompt/cache-work
identities match. C2 adaptive work may vary and is retained in the raw records.

| First-finishing request | Control worst mutex wait | Candidate worst mutex wait |
| --- | ---: | ---: |
| Short C2 | 71.47 ms | 51.89 ms |
| Coding C2 | 117.56 ms | 38.01 ms |

These are exploratory ranges over three histories per arm, not confidence bounds
or an inferential throughput claim. The remaining one-call wait is expected.
Untraced confirmation and current-patch cache/cancellation/failure checks remain
pending; the patch stays a draft. Earlier head-optimization confidence results are
separate and are not pooled with these profiles.

The first complete control history was retained after the collector wrongly
expected a worker-side trace. These two hooks run only on the HTTP head, so no
worker trace is emitted. The remaining five histories resumed with corrected
collection; the completed control was not repeated or excluded. Both-node PID,
boot, topology, driver-counter, kernel and final-idle checks passed.

Environment: two 128 GB Strix Halo nodes, gfx1151, private ROCm 7.14.1, Clang 23,
Linux 7.3.0-strix-test, unchanged approved driver `81C6FF2FBAC473C7AB21844`, four rails
on crossed 40+20 Gb/s links, mapped host windows. Model: Qwen3.8-Flash-Next
UD-Q4_K_XL with shared-Q8_0 MTP, greedy, context capacity 65,536, two sessions,
scheduler chunk 2,048 and RAM cache 1 GiB; requests bypass cache reads. Actual
prefixes in the focused concurrent cases are short, not 64K prompts. Worker
verdict responses use the TCP control channel; the intervals below are not RDMA
payload latency measurements.

## Reproduction

With the documented source-build dependencies:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --parallel 4 --target tp_executor_test tp_snapshot_gate_test
ctest --test-dir build/cpu-test -R '^(tp_executor_test|tp_snapshot_gate_test)$' --output-on-failure
python3 docs/patches/tp-request-end-handoff/verify.py
```

The executor test retains equal-budget batching checks and adds unequal 2/32 and
8/32 budgets in both arrival orders for AR, greedy MTP and sampled MTP. A two-token
request may finish from preview before batching; longer cases must still batch.
Every result matches its independent control, both ranks' model calls agree and
both requests settle. The gate test covers reservations, recursive turns, unwind
and concurrent users. Both focused CPU suites pass.

The [manifest](data/manifest.json) pins compressed and uncompressed hashes for the
raw client histories and master CPU spans. The verifier reproduces
[the per-request timing table](summary.csv) and [the verification summary](verification.json).
It uses each host's local monotonic clock and never subtracts timestamps across
machines. PID and virtual-address values are trace correlations only.
