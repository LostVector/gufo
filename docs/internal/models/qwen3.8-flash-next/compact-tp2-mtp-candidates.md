# Compact sampled TP2 MTP candidates

Exchanging exact partition top-64 candidates improves sampled decode by 0.2734%
over the complete-logit head-sharing build on fixed November 14, 2026 tests.

## Algorithm and compatibility

Both ranks keep the existing scalar half-vocabulary projections. Each selects its
best 64 logical rows and sends 64 IDs plus 64 F32 score bits: 512 payload bytes,
compared with 198656 bytes for the common subset's complete half-logit buffer.
The global top 64 must lie in the union of the partition top 64s. A stable GPU
merge reads rank 0 then rank 1, preserving original logical-row tie order.
Nonfinite keys become negative infinity; zero keys are canonicalized while
original finite signed-zero score bits survive as payload. Subset remapping runs
after global selection. Local normalization, one-winner exchange and ID
deduplication would change the algorithm and are not used.

The bounded proposal sampler consumes exactly these returned IDs and logits.
Sampling, RNG, target verification, controller and n-gram policy are unchanged;
target verification remains complete. The capability defaults off in other
communicators. Matching verbs wire 26 peers are required; the measured parent
uses wire 25. No new CLI switch or driver update. The GPU merge is kernel-only;
the distributed host callback remains eager.

## Measured application result

Forty starts form 20 independent pairs, balanced 10 parent-first and 10
candidate-first. The primary averages three sampled-case log ratios per pair,
then averages pairs: **+0.2734%**, stratified-bootstrap 95% interval
**[+0.1710%, +0.3868%]**, exact balanced-order two-sided **p = 0.0000649505**.
Nineteen pairs improve. All 200 output/work records match. No exclusions,
replacements, optional extensions, interim speed analysis or pilot pooling.

| Case | Parent tokens/s | Compact tokens/s | Paired change |
| --- | ---: | ---: | ---: |
| Greedy control | 54.1712 | 54.1265 | -0.0824% |
| Sampled scene | 51.3174 | 51.3950 | +0.1515% |
| Sampled explanation | 57.3787 | 57.6169 | +0.4154% |
| Sampled 2K scene | 46.3703 | 46.4874 | +0.2535% |

Rates are arithmetic means; paired changes use log ratios. Complete-client rate
improves 0.3261%, descriptive. This is incremental to the d08 head-sharing build;
the earlier 1.3409% result is separate, with no measured combined total.

Six prefill flags remain unresolved: explanation +51.062, +14.792, +57.739 and
+42.355 ms; 2K scene +74.011 and +70.223 ms. All exceed 5% and 10 ms. Earlier d08
flags are separate. One baseline start has an EVO CRC/lost frame and Corsair WR
retransmission; both host windows stay in the primary. No general prefill, tail,
concurrency, peer-failure or reliability improvement is established.

Settings: UD-Q4_K_XL target/shared-Q8 MTP, 64K capacity, two slots/one active,
chunk 512, draft 7, 1 GiB RAM cache, thinking and prompt/persistent caching off.
Sample seed 513, temperature 0.8, top-k 40, top-p 0.95; greedy temperature 0.
Actual prompts 46/2106 tokens, completions 121/115/128 with natural EOS retained.
Two Strix Halos, B6 driver, TX32, four paths, polling off, 64 KiB stripe.
Evidence is conditional on these fixed tasks, configuration and machines.

## Correctness and reproduction

CPU selection identity passes 637 cases. The GPU selector matches the full GPU
selector and independent CPU oracle on 72 shape/pattern cases. Actual extracted
heads with real Q8 weights and production verbs pass 22 records/rank, including
960 proposal/RNG comparisons. These are distinct scopes, not full-model lifecycle
proof. The same-image isolated subset head falls from 741–743 to 675–676 us:
about 9% of that stage, not 9% of inference. GPU events include communication and
scheduling waits. The inconclusive application pilot was excluded from confidence.
The extended repository MTP operator suite passes both-rank bit/guard/oracle tests.
[Operator report](evidence/tp2-compact-candidates-2026-11-14/operator-test.json) and
[source relationship](evidence/tp2-compact-candidates-2026-11-14/source-relationship.json)
retain execution and review provenance.

For GPU coverage, build `qwen38_flash_next_mtp_ops_test` in the `gpu-test` preset,
then run `ctest --preset gpu-full -R '^qwen38_flash_next.mtp_ops$' --output-on-failure`.
For saved-data statistics, run
`python3 docs/internal/models/qwen3.8-flash-next/evidence/tp2-compact-candidates-2026-11-14/recompute.py`.
It needs only Python and verifies the primary, interval and exact test against
[saved analysis](evidence/tp2-compact-candidates-2026-11-14/analysis.json), using
[paired CSV](evidence/tp2-compact-candidates-2026-11-14/case-pairs.csv) and the
[frozen design](evidence/tp2-compact-candidates-2026-11-14/DESIGN.json).

Tested parent binary SHA-256 starts `d08d5685`; candidate `9c8c5a0c`.
All 201 normalized compile commands match: Clang 23, ROCm 7.14.1, gfx1151,
RelWithDebInfo/-O2. The review derivative adds tests and clarifies comments;
runtime code is the frozen tested implementation. Full scripts, raw data,
requests, manifests and restoration proof remain in the lab directory
`sampled-mtp-candidate-merge-2026-11-14/`. The older d697 service was restored.
No serving adoption or publication; focused lifecycle/protocol qualification and
prefill flags remain review/adoption limits. Do not repeat efficacy unchanged.
