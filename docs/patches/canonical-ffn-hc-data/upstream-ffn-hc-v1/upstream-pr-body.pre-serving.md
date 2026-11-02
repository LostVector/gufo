The batched target FFN path dispatches the immediate combine/mix pair separately, even when the existing fused HC implementation supports the row count and layout. Route just that immediate pair through the existing fused implementation, retaining geometry/format and residual-state guards and the original fallback. Extend the existing HC fixture with 48 batch-reference conditions; preserve its original cases.

This is a common-code port based on canonical `8ab834a5`, with no RDMA API, transport dependency, extension-API change, new switch or new arithmetic kernel. It also reuses the existing fused producer's Q8 activation output, so the mechanism is more than launch reduction alone.

Draft scope: component dispatch reduction and exactness review. The related TP2 [fork draft #15](https://github.com/LostVector/gufo/pull/15) independently measured 384 fewer GPU dispatches in eligible 3/8-row target passes (48 fused calls); a 9-row fallback was unchanged. Those are TP2 measurements. Its 12-pair/24-start/624-request serving study was statistically inconclusive at C2/C4/C8, with every timing flag and adaptive-work difference retained. No canonical serving speedup, equivalence, or default adoption is claimed.

Canonical validation completed:

- Fresh HIP GPU-test build on gfx1151; complete existing HC operator fixture plus 48 additive conditions, assertions enabled, no skips. Eager/graph, aliasing and arithmetic checks remain.
- Diagnostic fixed-history model comparison: 104 records, 64 matched replay records and 32 complete materialized state pairs, at depths 0/32768, sampled and greedy. Full scores/RNG/tokens/work and borrowed/serialized/file-backed restore agree. Greedy plans are fixed alternating 1/0 in the test-only engine; sampled policy is unchanged. This does not prove production greedy-controller replay or independent full-model teacher parity.

Production release and bounded natural-controller C1/C2 serving qualification are being collected separately; results and all phase/work flags will be added before this draft is marked ready. No model/driver/system setting is changed by this patch.
