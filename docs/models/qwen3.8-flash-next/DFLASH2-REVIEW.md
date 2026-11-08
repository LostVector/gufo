# Independent adversarial review of fork PR 17

Reviewed candidate `e6a769b04918623018696025127acb4ecbdf01d2` against `0fb50ce4`, November 7, 2026. Review covered the actual source diff and new tests. No GitHub publication or GPU workloads were performed. The live MTP trial and shared model lock were left alone.

## Finding

**P2 — Use the extension facade for the model-local draft backend include**
`src/models/qwen38_flash_next/dflash_sampling.hpp:9`

The new model-local helper directly includes `src/core/speculative/draft_backend.hpp`. The repository architecture requires production model code to consume the public extension facade. `include/gufo/ext/draft_backend.hpp` already forwards the necessary declaration. Reproduction: `python3 tools/ci/check-extension-isolation.py --fail-on-family-graph` reports one strict production model-to-core boundary violation, although this invocation only fails on family graph errors and therefore exits successfully. Replace the include with `<gufo/ext/draft_backend.hpp>`. The parent implementation agent acknowledged the finding and is applying that isolated fix; this review pins the original candidate and does not claim to have validated the subsequent commit.

No additional concrete correctness defect was established in the reviewed diff. This is not evidence that unexecuted GPU/model or timing qualification passes.

## Review observations

- Sampled verification uses the exported F32 q rows without renormalizing them; every row is validated before target mutation. The accept/reject path compares target p against the selected q and defers the actual residual draw into the next anchor. Compact target candidates fall back to full rows when constraints become active.
- Prompt lookup retains only an unchanged neural prefix, replaces its trailing rows with point masses, and discards neural suffix rows conditioned on the replaced predecessor history. The CPU test exercises mixed, extended and copy-only proposal shapes and rejection identity.
- The owned smaller head rebuilds its derived vocabulary table after restore/reset and catches up only from committed token history. Target verification remains over the full target vocabulary. Existing model tests exercise actual subset use, seeded replay and restore. Their scalar p/q teacher exercises the full draft head, rather than independently testing the owned subset selector probabilities; that remains a coverage limitation.
- External draft prefill explicitly ends at checkpoint boundaries, consumes the complete committed feature pass before capture, and checks target/draft frontiers during serialization and restore. Component private staging is invalidated by execution and writes. The fixed private component capacity includes the external header and payload-size prefix.
- Draft models retain the target through a borrowed-table owner; the target does not retain the draft model. Session member destruction tears down native graphs before freeing the feature capture buffers they reference. The legacy Qwen adapter preserves the existing entrypoints while routing to shared family code.
- Resource claims include draft packed weights, per-session executor state/scratch, host/device target-feature staging, target rollback capacity, and external component staging. I found no established omission in those accounting paths.
- The shared-family moves retain the prior DFlash selector and block execution behavior apart from donor validation and explicit persistent snapshot validation/size helpers. AR/MTP fast-path conditions added for feature capture are conditional on an attached capture callback.

## Independent checks

Passed on the review host:

1. `python3 tools/ci/check-extension-api.py --base 0fb50ce4` — extension API 0.2.15, 122 surface paths.
2. `python3 tools/ci/check-extension-isolation.py --fail-on-family-graph` — acyclic family graph, 8 libraries, 12 declared edges; also surfaced the include finding above.
3. `c++ -std=c++20 -I. -Iinclude tests/models/qwen38_flash_next/dflash_lookup_test.cpp -o /tmp/dflash-independent-lookup` followed by `/tmp/dflash-independent-lookup` — passed.

No complete independent production build, GPU operator/model tests, long-context model run, matched AR/MTP timing controls, or performance qualification were run by this reviewer. The checked-in qualification JSON explicitly leaves matched existing-mode timing controls outstanding and identifies its HTTP record-baseline run as correctness/reference capture, not regression qualification. Retain those limitations before any upstream submission.

## Verified resolution

Independently verified local HEAD `4e78ff2f7824432aefbc540e292d41f55f5e8b5f`. Its complete diff from the reviewed candidate changes only the draft-backend include in `dflash_sampling.hpp` to `<gufo/ext/draft_backend.hpp>`. The P2 finding above is resolved. Reran the isolation check: zero strict boundary violations and the family graph passes. The extension API gate remains at 0.2.15 with 122 surface paths, and the independently rebuilt CPU lookup test passes. No GPU work or additional performance qualification was performed for this verification.
