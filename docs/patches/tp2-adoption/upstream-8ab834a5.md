# Canonical Gufo cache-admission intake: November 2, 2026

The qualified production source for this intake is
`5e7663534afd17f7d4d18162285d3d81f1acf42d`, tree
`ed90aa82ad3f88097a1bca32dd047f9cf2718524`. It normally merges canonical
`8ab834a5902bf19341750daecf27516a57840dd1` onto maintained `d038b2e8`, with no
conflicts or manual resolutions. All existing native TP work is retained.

This is an upstream correctness sync, **not a new throughput or latency-equivalence
qualification**. The independent Q8-reuse [draft14](https://github.com/LostVector/gufo/pull/14)
and fused HC [draft15](https://github.com/LostVector/gufo/pull/15) are excluded.
The [previous intake](upstream-773a1641.md) preserves its actual 32K continuation,
headroom, scheduling boundaries and retained timing flags; none are waived here.

## Included changes

- [b911c150](https://github.com/gufo-org/gufo/commit/b911c150e9cd6e4d6daf842826d0df5242c5387f):
  capacity errors report per-session bytes, context size and fitting session count.
- Canonical `cefc651e`: unchanged-retry functional checks recognize explicit byte
  and record retention refusals at the exact requested token boundary. Other
  refusal reasons/actions and mismatched boundaries remain failures.
- [8ab834a5](https://github.com/gufo-org/gufo/commit/8ab834a5902bf19341750daecf27516a57840dd1):
  RAM-cache segments measure device-heap free memory before and after host-page
  registration. Uncharged registration is allowed even when the device heap is
  already tight. Registration is retained only if
  `free_after >= min(free_before, dispatch_reserve)`; failed queries refuse growth.
  After a sufficiently large charge identifies a charging host, later segments
  also use a preflight reserve check. Host-memory reserve checks remain active.

The RAM backing here stores component/conversation checkpoints; the n-gram table
remains disk-backed. The reserve value, placement policy, all model/family/kernel sources, public
include sources and native TP scheduling/transport are unchanged. No driver,
ROCm installation, boot option, service setting or table warming changed. The
new RAM-cache behavior is not presented as a measured speedup on these hosts.

## Build and focused checks

Official CMake release, RelWithDebInfo, `BUILD_TESTING=OFF`, native RDMA enabled,
private ROCm 7.14.1/Clang 23/gfx1151. All 336 normalized compiler commands match
qualified `9c715720`; only three host C++ implementations and the internal
committed-backing header change. Both peers ran the identical executable:

```text
2e413c9f6bccbb37f94f95f7f11548cd4591091a29bac64343fb3d835f8c263b
```

Local checks passed:

- Shared formatting, extension API 0.1.16, docs and generated-infrastructure checks.
- Five focused CPU tests: runner capacity validation plus Flash-Next config,
  runner options, partition and GID contracts.
- Twelve cache-workload parser tests, including exact refusal reason/action and
  token-boundary negatives.
- Two real-HIP cache-backing fixtures, normal and restricted-pin modes, with no
  skips. They simulate charging/non-charging heap accounting, tight reserves and
  failed queries while exercising registration, allocation, release and cache
  lifecycle paths. This is not a live full-model low-memory stress measurement.
- Fresh native TP2 smoke: 12 exact prompt/output/cache-work matches against the
  archived qualified `9c715720` control, with actual MTP and source/binary/settings
  checks.
- Four greedy/sampled keep/discard cached continuations and nine cancellation
  records, covering both victim roles, live width-two survivors, solo controls and
  post-cancel replay. All completed substantive calls use actual MTP. Thirteen
  case records are not thirteen HTTP calls.
- Boot, native module, four rails, kernel window, PID exit and final idle checks;
  no transport-error counter changes.

Hosted checks both pass on exact source `5e766353`:
[Repository/Linux](https://github.com/LostVector/gufo/actions/runs/41485872231),
[Windows validation](https://github.com/LostVector/gufo/actions/runs/41485876452).
The fork skips Windows GPU model execution; these are packaging/CPU checks,
not Windows TP numerical qualification.

No new full-model quality sweep, actual 32K/in-pass qualification, physical-fault
campaign or matched performance study was run for this intake. Previous numeric
and performance artifacts keep their original source attribution. Ordinary cache
continuation and the changed allocation/refusal logic are the focused new gates.

## Reproduction and retained evidence

[Acquisition bundle](upstream-8ab834a5-data/acquisition.zip),
[archive scope and hash](upstream-8ab834a5-data/archive.json),
[offline audit](upstream-8ab834a5-data/check.py).

```sh
python3 docs/patches/tp2-adoption/upstream-8ab834a5-data/check.py
```

The bundle retains exact build/runtime commands, the upstream diff and source
proof, test XML/logs, all request/server records, identity/inventory/counter
snapshots and failure notes. The initial merge invocation lacked a committer
identity and stopped before applying changes; the corrected merge used the
existing per-command identity without global configuration changes. Models,
executables, control tokens and multi-GB state files are excluded. The offline
audit checks retained records; it cannot rerun GPU, live PID, kernel or host tests.
