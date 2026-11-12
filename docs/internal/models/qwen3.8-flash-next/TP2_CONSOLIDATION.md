# Maintained TP2 consolidation, November 11, 2026

The maintained `tp2` branch combines the canonical refresh with the tested
capture-scheduling fixes and cancellation assertions. This is a source adoption;
it does not upgrade the running service or change the driver or host setup.

## Included source

- Previous maintained head: `4e30eeeb416622834e9a26a3ff815c0a3e68d6dc`.
- Canonical main: `0d238d4f5fee49720d6be45c2c27795100922d5c`, included by ancestry.
- Neuhaus RDMA: reviewed selective intake through `f070a179`, not full branch parity.
- Integration: `6b0f404f`, API 0.3.4 and control wire 22.
- Scheduler fixes: `6387a35a`; cancellation assertions: `9c79494b`.

Ready partial prompts resume before the most recent peer repeats, preserving
older resident waiters. Native TP prefill observes the pending-snapshot gate
before acquiring the call mutex. Cancellation tests check that an in-flight
capture finishes using its state before the request settles and cache reuse
continues. The gate is not an atomic priority lock and does not preempt active
GPU work or guarantee bounded shutdown for arbitrary peer failures.

Prompt rendering and requested logprobs are mirrored through the TP wrapper.
The model-owned scheduling profile determines pass size and is checked on both
ranks. Both ranks need a matching wire-22 build. The explicit legacy adapter
retains coordinated TP snapshots; canonical one-host runners use component cache.

## Validation

The final focused CPU checks cover TP executor/control behavior, gate nesting,
legacy and component runner ownership, and scheduler ordering. Formatting,
extension API and documentation checks are recorded in the
[machine-readable adoption data](upstream-refresh-data/consolidation.json).
Negative controls reject original tail insertion, unconditional front insertion,
missing TP prefill gating and waiting inside the call mutex.

Earlier selected production AR/MTP/DFlash histories preserve exact outputs/work
and exercise disk restoration of 24,640 tokens followed by seven prefill tokens.
AR retains seven inconclusive timing metrics; MTP retains nine; DFlash's single
pair passes 128. See [production controls](upstream-refresh-data/production-controls.json).
The scope is those histories, not full numerical or release qualification.

The combined scheduling candidate passed four observer-free production runs in
baseline/candidate/candidate/baseline order: 116 generation records and 20
discovery probes, with identical outputs/work. Of 413 timing metrics, 403 pass
the unchanged regression gate and ten remain inconclusive. The raw primary
observations and unresolved timing ranges are retained in the adoption data.

| Independent request | Baseline first token | Candidate first token |
|---|---:|---:|
| ALPHA | 4.014–4.362 s | 2.523–2.534 s |
| BETA | 3.993–4.341 s | 3.832–3.844 s |

These are two observations per role, not a significance test or a standalone
decode-throughput measurement. Runs used UD-Q4_K_XL Flash Next, shared Q8 MTP,
64K capacity, two slots, 1 GiB RAM cache and thinking off. Processes/application
caches were fresh; OS caches were not dropped and the full n-gram table was not
warmed. The B6 driver used host windows, TX32 and active polling disabled, with
the two links' speed/lane fields preserved in the raw run guards.
Both independent prompts started cold with zero reused tokens: ALPHA prefills
2,927 tokens and BETA 2,929. Cache capture is enabled during prefill; these
figures do not describe warm prefix-cache hits.

The measured production binary was built from `6387a35a`:
`b05c12e57753bba941859d56d586aef313374ad5ec649163eb4af53bc2d61030`.
The `9c79494b` addition changes only a test; final adoption documentation changes
no production sources. Reusing that timing evidence relies on this source
identity, not on selecting a mutable build artifact by its timestamp. The
executable was built with only Flash Next, `BUILD_TESTING=OFF`, using the existing
private ROCm 7.14.1 toolchain for gfx1151.

## Limits and separate work

The inherited strict literal-history test still appends a warning after `DONE`
under TP2 and fails its assertion. The deployed pre-refresh binary reproduces
the same continuation for the exact uncached request. This remains unwaived;
neither broader model quality nor all workload timings are qualified.

The smaller-vocabulary DFlash projection experiment `52130313` is excluded.
It has compile/syntax checks but no GPU correctness, acceptance or timing result.
The post-capture notification prototype is also excluded. Existing DFlash support
and the previously merged native MTP/verification optimizations remain included.
Nix/container recipes have not been newly built/run for this consolidated head.

The portable scheduler fix is submitted as
[gufo-org/gufo #1181](https://github.com/gufo-org/gufo/pull/1181); the native TP
prefill guard and cancellation checks are submitted as
[neuhaus/gufo #17](https://github.com/neuhaus/gufo/pull/17). Those PRs use their
own upstream bases and do not share this branch's model-performance attribution.
