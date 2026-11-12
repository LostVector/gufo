# Exact batch-policy replay

This diagnostic preserves the original 20-request history on each fresh server
pair, with baseline/candidate/baseline production binaries fbacf9cc/2669e07c.
Only the ragged C4 wave is probed. All 60 prompt/output/cache-work identities,
six audited transport/kernel windows and final cleanup checks pass. These
diagnostic times are not additional samples for the original ten-pair study.

## Direct observations

All **104 actual MtpBatchController::Choose decisions** replay exactly through
the real source header and captured controller/session inputs. The recorder
captures raw float bits, sample counts, cohort history, per-session acceptance
state and budgets; the actual function return supplies the comparison oracle.
Both-host synthetic ABI, bad-pointer, process-filter and nested-counter tests
passed before acquisition. No engine, driver, boot setting or model was changed.

The baseline-before and candidate have the same selected-width sequence and
per-actor completion-work sequence in this wave (row ordering differs). Their
second batch chooses five drafts. The later baseline chooses four. At that
first selected-width divergence, context, actors, remaining budgets and all
per-actor acceptance-state fields are identical.

| Captured C3 timing cell, ms | Baseline before | Baseline after |
| --- | ---: | ---: |
| zero drafts | 38.776814 | 38.804264 |
| one draft | 49.905533 | 49.913540 |
| three drafts | 59.345337 | 59.045834 |
| four drafts | 60.333874 | 60.724503 |
| five drafts | 62.268185 | 66.011230 |

Two drafts have no samples in either cell. These are learned **controller
estimates**, not per-layer measurements or newly measured GPU execution times.

In bidirectional offline input swaps, replacing only these eight timing floats
changes five to four, or four to five. Swapping only sample counts, scheduling
history or acceptance state does not. The same result holds between candidate
and the later baseline. This establishes the cause of this first width choice:
learned timing estimates carried over from the preceding requests.

## Limits and next action

This does not reproduce the earlier 12-batch-step fast mode of the 33-token
request: all three current arms take 18 steps. Its publication spans are
1209.940 / 1210.352 / 1198.502 ms. The earlier roughly 244 ms batch-span gap
therefore remains evidence from a different trace; the current first-choice
counterfactual is not a complete causal replay of that gap.

All 20 new strict phase flags and 20 adaptive-work differences are retained in
phase-flags.csv and adaptive-work.csv. No statistical gain, equivalence,
candidate-specific slowdown, timing waiver or beneficial replacement policy
follows from one traced triplet. Actual proposed work can differ from selected
neural caps because confidence and lookup paths remain active.

The next policy experiment, if pursued, should target how stable learned costs
are across cohorts. It must compare complete-wave throughput and individual
latencies: making the short reply finish earlier is not by itself a net win.
Do not change acceptance math, seed behavior or synchronization to force a
preferred width. The already-fixed C2-to-C1 backoff issue is separate.

## Evidence and commands

After acquisition has finished and released the model lock:

```bash
python3 tp-cache-retention/ple-owner-ragged-policy-trace-v2/analyze.py
python3 tp-cache-retention/ple-owner-ragged-policy-trace-v2/rounds.py
python3 tp-cache-retention/ple-owner-ragged-policy-trace-v2/choices.py
python3 tp-cache-retention/ple-owner-ragged-policy-trace-v2/divergence.py
```

Run from the investigation root. The recorded v6 fixture shared object and all
manifest-listed artifacts must match their hashes. The source acquisition
harness, protocol, raw app.csv on both ranks, CHOICE records, every per-request
counter and all offline swaps are retained in this directory and
`../app-span-recorder-v6/`. The failed v1 attempt is preserved separately: it
stopped at metadata-deployment preflight before any model was launched. V2
deploys and verifies every manifest-listed file automatically.

The adjacent acquisition.zip retains the original protocol, source harness, raw
traces, recorder artifacts/controls and all60 raw result records. Paths and
commands above describe the original lab layout; this export does not run GPU
work or silently pool diagnostic timings into the original confidence study.
