# PLE owner-read sharing: review scope

The implementation is ready for code review. Maintained `tp2` remains separate;
review readiness is not blanket timing qualification or automatic adoption.

## Qualified code and provenance

Production review source74bf4b5e includes maintainede250a07c and canonical
gufo-org13f64566. Repository41397279327 and Windows41397282071 pass on that
exact code head. Maintainede250a07c separately passes repository41397237414 and
Windows41397237381. Windows coverage includes packaging, CPU/application and
extension contracts; the fork does not run the Windows distributed GPU job.

Both current production builds pass fresh two-node twelve-output replay and
cached continuation/both-role cancellation checks with actual MTP, exact output
and prompt/cache work, matching settings, and clean transport/kernel/idle audits.
Earlier full-logit/RNG/state-hash, mixed-wire, owner-EIO, peer-exit and event-error
evidence retains its exact documented source revisions. The evidence-only
follow-up changes this documentation subtree, not app code, build options or
workflows. See review-code-qualification.json for the attribution boundary.

The original twenty-start timing study remains baselinefbacf9cc versus candidate
85586a63. First C1 server decode improves0.699% and short C2 complete-wave rate
improves1.940%, with the predeclared paired confidence/ordering/Holm checks.
Repeated C1 and wider timing results stay inconclusive. Diagnostic triplets are
not pooled into those estimates, and later combined binaries inherit no new
numerical performance result by renaming their revision.

## What the targeted traces establish

- The C2 first-publication mode can include another request's prefill between
  this request's final prefill and first publication, in both baseline and PLE.
- All104 captured batch-policy decisions replay from exact inputs. The first
  observed width divergence flips when only learned timing estimates are
  exchanged. This identifies decision sensitivity, not a faster replacement
  policy or a complete replay of the earlier244ms short-reply mode.
- The roughly51ms completion tail occurs before the control Send, overlapping
  a sibling decode. Nested sends take21–27us. Shared call-mutex contention is
  strongly supported by source; acquisition is not independently timed.
- The startup high mode reproduces on unchanged baseline inside one34.685ms
  HIP event wait in the first Forward. The probe does not distinguish GPU/peer
  completion from host sleeping/polling/scheduling. No graph launch occurs;
  source intentionally keeps TP2 eager.

These are distinct observations. They are not a universal explanation for all
flags, a GPU/CPU utilization split, or evidence that the entire wait is avoidable.

## Remaining limitations

All27 original strict timing entries over15 comparisons remain, including two
underlying maximum-gap comparisons. Follow-up flags/work differences are also
retained. Whole-request gains do not waive individual phase increases. Wider
latency remains unqualified. The root cause of the long event wait and the two
original maximum-gap comparisons are unresolved; a candidate-specific cause is
not established by these traces.

An earlier-finalization scheduling change would be a separate patch. It must
retain peer verdicts, cache/model-operation ordering, failure settlement and
cancellation. Shorter HTTP tails alone do not prove higher token throughput.
Likewise, TP graph capture requires a safe communication/control design; it
cannot be enabled by assuming a graph flag already covers this path.

## Review and reproduce

Review the wire22 negotiation, exact row IDs/F32 payload, reusable packet event
lifetime and cancellation/error handling. The test-only component probe and
production performance builds remain separate. Run:

```bash
python3 docs/patches/tp2-ple-owner-data/check.py --statistics
```

This verifies the original study and the exported diagnostic raw data, including
the new completion-send partitions, every HIP CSV span and first-Forward union.
It does not rerun inference or certify historical hardware. Every raw bundle,
rejected attempt, recorded source/hash and scope limitation remains available.
