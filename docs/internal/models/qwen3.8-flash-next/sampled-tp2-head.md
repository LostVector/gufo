# Sampled TP2 MTP head sharing

Sharing the native-MTP vocabulary projection across two ranks improves sampled
decode by 1.3409% on the fixed November 14, 2026 Strix Halo experiment.

## Change

Each rank projects disjoint vocabulary rows, then gathers every original logit
before the existing top-64 selection, vocabulary remapping and sampling.
The common request-subset route reuses `draft_head_gather`; the full-vocabulary
fallback reuses `target_head_gather`. GPU arithmetic, normalization, random-number
use, target verification, draft control and disk-backed n-gram policy stay intact.

The new mutually negotiated capability defaults off in other communicators.
Eligible Q8_0 two-rank sampled paths enable it without a CLI option. Greedy and
unsupported paths retain their existing behavior. Both peers need matching
binaries: verbs wire version 25 rejects version 23 parent and version 24 prototype.
No driver update, dedicated buffer or whole-TP graph change is part of this patch.

## Measured result

Twenty fresh starts form ten independent process pairs, balanced five parent-first
and five candidate-first. Each start runs one warmup, a greedy control and three
sampled requests. No cases or starts were excluded or replaced.

| Case | Parent tokens/s | Candidate tokens/s | Paired change |
| --- | ---: | ---: | ---: |
| Greedy scene control | 54.161 | 54.150 | -0.019% |
| Sampled scene | 50.592 | 51.223 | +1.246% |
| Sampled explanation | 56.343 | 57.374 | +1.829% |
| Sampled 2K scene | 45.987 | 46.424 | +0.950% |

The primary averages sampled-case log ratios within each process pair, then the
ten pairs. The gain is **1.3409%**, order-stratified paired-bootstrap 95% interval
**[1.1589%, 1.5035%]**; exact balanced-order two-sided **p = 0.0079365** over 252
assignments. All ten pairs improve. Absolute rates above are arithmetic means;
paired changes use log ratios. Complete-client rate improves 1.1528%, descriptive.
Pilot and diagnostic timings are excluded from confirmation.

All 100 output/work records match. Forty selected request counter windows have
no selected transport-error increments. Three prefill flags remain: explanation
+71.918 ms in pair 1, explanation +11.956 ms in pair 3, and 2K scene +89.711 ms in
pair 7. They exceed both 5% and 10 ms. Their cause is unresolved; this patch does
not establish improved prefill or uniform tail safety. No decode/client flags
exceed that rule.

Conditions: UD-Q4_K_XL target/shared-Q8_0 MTP, 64K capacity, two session slots,
one active request, chunk 512, maximum draft 7, 1 GiB RAM cache, thinking off,
prompt/persistent caching off. Sampling uses seed 513, temperature 0.8, top-k 40,
top-p 0.95. Actual prompts are 46/2106 tokens; sampled completions are 121/115/128
with natural EOS retained. Both Strix Halos use the B6 driver, TX window 32,
four native paths and polling off. This is conditional evidence for these tasks,
settings and machines, not a general cross-model estimate.

## Correctness and lifecycle evidence

The extracted real-Q8 head and real verbs component test matches all full-logit
and candidate bits on eight full and eight subset inputs, including masked extras.
Capability-off, greedy and callback-error controls pass. Twenty records per rank
include the controls. A separate sampled inference trace records 67 successful
198,720-byte subset exchanges from actual head threads, matching 67 proposals.
That trace proves activation and is excluded from speed measurements.

Matching new peers bootstrap and complete a real reduction. Both mixed-version
orientations reject metadata explicitly. Unmodified sampled text/tool fixtures
pass interruption, RAM continuation and disk restoration after restart. All four
observed continuations are exact; disk fixtures do not require exact output when
a checkpoint gap needs re-prefill. These checks are narrower than general
concurrency, long-context or arbitrary peer-failure qualification.

## Provenance and saved-data reproduction

The parent is maintained fork commit `2d4ebb483fec889d7a91d48ed8f3b5662a062c5d`,
binary SHA-256 `ff85589337889357b6af9900cc9099a7eeba7ad011824911842630309d8663e9`.
The tested candidate binary is
`d08d56858517926e99989a358c422b75cefb0f07b570c2f3d431efcb51eeb008`.
All 201 normalized C/C++/HIP compile commands match the parent: RelWithDebInfo,
-O2, Clang 23, ROCm 7.14.1, gfx1151. The review source differs from the tested
source only by clang-format and a callback comment; it is not a newly timed binary.
The older d697 serving binary was restored and is not the measured parent.

[Saved analysis](evidence/tp2-sampled-head-2026-11-14/analysis.json),
[raw paired rates](evidence/tp2-sampled-head-2026-11-14/case-pairs.csv),
[frozen design](evidence/tp2-sampled-head-2026-11-14/DESIGN.json),
[statistical self-check](evidence/tp2-sampled-head-2026-11-14/statistical-selfcheck.json),
[lifecycle results](evidence/tp2-sampled-head-2026-11-14/lifecycle.json) and
[source relationship](evidence/tp2-sampled-head-2026-11-14/source-relationship.json)
are retained together. Full scripts, requests, logs and source/binary manifests
remain in the lab directory `sampled-mtp-head-2026-11-14/v2/`.
To reanalyze there, run `python3 confirmation/analyze.py`; its frozen input data
and restoration proof are required. Do not rerun the completed one-shot controller.

The patch is prepared locally for review, with no serving adoption or publication.
