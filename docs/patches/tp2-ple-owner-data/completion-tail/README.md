# EndRequest: pre-send versus control-send time

One fresh, bounded B/C/B replay uses the frozen fbacf9cc baseline and85586a63 PLE
candidate, original first-five-request history, UD-Q4_K_XL/shared-Q8_0 MTP,
65,536 context capacity, two sessions, chunk2048 and1GiB cache. The approved
81C6/TX32 driver and boot/link settings stay unchanged throughout this successful
acquisition. Only generic probe7 changes from Capture to the instruction sink's
Send method. Recorder-v3 code and its request-identity/shape ABI reads are intact.

All15 original output/prompt/cache-work identities and six transport, trace,
kernel and final-idle audit windows pass. The analyzer pairs every one of15
EndRequest spans with its one same-thread nested Send; none are dropped. The
analysis is diagnostic, not another sample for the ten-pair confidence study.

| The long EndRequest, ms | Baseline before | PLE candidate | Baseline after |
| --- | ---: | ---: | ---: |
| Total |51.177742|51.236171|50.992828|
| Before Send entry |51.149723|51.199692|50.959649|
| Inside Send |0.021069|0.027050|0.025719|
| After Send return |0.006950|0.009429|0.007460|
| Overlapping Session::DecodeStep |51.129334|51.174553|50.919630|

This uses the original **128-token short-C2** request history. Actor1's
last-content-to-HTTP-finish intervals are51.740 /52.162 /51.647ms. All three
overlap the long EndRequest above. Actor0 is still generating. Its subsequent
completion tails are0.472 /0.524 /1.005ms. Every other EndRequest is below45us;
all15 nested control Send calls are below27.051us.

## What this establishes

The roughly51ms is **before** the end instruction's Send call in both candidate
and baseline. It is not51ms spent inside that send. The same-host trace shows
almost all of that interval overlapping the sibling's model decode. Source
EndRequest and TpMirroredRunner::DecodeStep both take the same recursive call
mutex; the latter holds it around Session::DecodeStep. These facts strongly
support waiting for the active sibling call as the cause. Lock acquisition
itself is not separately measured, and the pre-send interval also contains
request bookkeeping. It must not be relabeled pure CPU dispatch overhead.

Send duration is the host API interval, not completed network transfer, peer
receipt, peer verdict or GPU-ready latency. This does not estimate remote wire
latency or show that all other control operations are equally cheap. The
existing pending-finalizer gate prevents repeated starvation; it does not
preempt an already active model call. This result is consistent with that
documented boundary, rather than a failure of the earlier starvation fix.

No new strict5%/3ms phase flags arise in this triplet; two adaptive-work fields
differ. Every older flag remains. These traces do not individually attribute
all nine completion-tail flags in the original ten-pair study, prove timing
equivalence, or justify waiving the remaining warmup/max-gap observations.

## Next experiment

If optimizing this tail, investigate reserving end-instruction order before the
scheduler starts its next sibling call. Avoid arbitrary sleeps/yields, removing
the required peer verdict, or sending End before a request's pending cache/model
operations have completed. Measure both complete-wave throughput and request
completion latency; this observation predicts a possible short-request tail
improvement, not faster decode or prefill. It is a separate scheduling change,
not evidence to modify the RDMA driver or broaden the PLE patch.

## Reproduction and preserved failure

The completed source/protocol/binary/recorder hashes, both raw app traces, every
EndRequest decomposition, raw request JSONL and audit outputs are retained.
After acquisition, run `validate.py`, `analyze.py`, then `end-analysis.py` from
this directory with the investigation root as the working directory. Runtime
reproduction requires a fresh study directory because run.py refuses to reuse
an acquired protocol. No inference binary or system installation was changed.

The preceding v1 attempt is preserved. A preparation-time namespace collision
selected an old launcher; it failed before requests, briefly loaded an earlier
approved transient driver and cleanly restored the approved baseline on both
hosts. It contributes no timing data. V2 executes its producer separately,
verifies the three harness files against the correct source, rejects the old
driver-admin path, and deploys the complete current launcher. See
preparation-notes.md for the exact distinction.

The acquisition.zip archive retains the full lab layout, scripts, protocols,
raw app traces and result JSONL. The HIP CSV is inside that archive. Paths in
the reproduction notes refer to the original lab; this export does not rerun
inference. Run `python3 ../check-new-latency.py` to audit both new raw bundles.
