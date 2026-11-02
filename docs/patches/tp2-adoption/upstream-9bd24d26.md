# Canonical Gufo serving compatibility intake: November 2, 2026

Qualified source **75de9f82234da17257321db81bd13e914ea2dbb6** normally merges
canonical **9bd24d26b066ce08a463c2def8da9acb95c75338** onto maintained1ad4bd3b.
The merge is automatic; native TP, inference, cache and public extension-header
paths remain byte-unchanged. This is a compatibility sync, not a new throughput
or latency-equivalence qualification. The [preceding intake](upstream-8ab834a5.md)
retains its cache-accounting checks and older timing flags.

## Included behavior

Canonical9b3b1f48 tolerates unknown embedding/audio/image/video request fields.
Canonical9bd24d26 adds model aliases across modalities and root health responses,
and removes public HTTP API-key support, including the `--api-key` option.
Clients can still send bearer credentials to another endpoint or an authenticating
proxy. The **separate TP control HMAC authentication is unchanged**.

The validated native-driver/model settings remain UD-Q4_K_XL/shared-Q8_0 MTP,
context65536/sessions2/prefill chunk2048/cache1GiB, TX32, active pollingoff,
registered host windows and crossed40+20Gb/s links. The GPU wait/add prototypes,
global-Q8 and FFN-HC drafts are not part of this source. No driver, kernel, boot,
ROCm installation or persistent service configuration changes accompany the sync.

## Qualification

Official fresh RelWithDebInfo production HIP build, BUILD_TESTINGOFF/RDMAON,
ROCm7.14.1/Clang23/gfx1151. All336normalized compiler commands and unit names match
qualified5e766353;13serving implementations change. Changed header closure is
rebuilt fresh. Exact tested binary SHA256:

```text
033595203293ca031c63be72061b9e8fd2a86cccda5798223e45234bb99273b9
```

- 15distinct affected HTTP/CLI/modality/TP CPU checks pass with no skips. The
 original CTest run had14passes and a missing-Matplotlib interpreter failure;
 that same benchmark-client test passes after selecting the existing project
 environment. Original failure retained; no assertion change or replacement case.
- Shared matching-format21, extension API0.1.16, isolation, documentation and
 generated-infrastructure checks pass. An initial formatter23run rejected
 unchanged files; matched21passes without unrelated formatting edits.
- Actual two-node native TP2/MTP startup and12standard requests preserve exact
 prompt/output/cache-work identities against qualified5e766353.
- Four greedy/sampled keep/discard cached continuations (317cached tokens) and
 nine cancellation cases pass, including both victim roles, live width-two
 survivors, solo comparisons and post-cancel replay. Thirteen case records are
 not thirteen HTTP calls.
- Actual binary/options/boot/module/four-rail identities, PID exit, kernel window
 and restored idle/error-counter checks pass. No selected transport error deltas.

Exact-source hosted [repository/Linux package](https://github.com/LostVector/gufo/actions/runs/41524577980)
and [Windows validation](https://github.com/LostVector/gufo/actions/runs/41524799587)
both pass. Windows GPU model execution is skipped on the fork; these results do
not qualify Windows TP numerical execution.

No new full-model sweep, actual32K state qualification, physical-fault campaign,
or matched performance study was run for this serving-only intake. Existing
numerical/performance records keep their original source and limitations. Nothing
here waives earlier per-phase timing flags or adopts inconclusive experiments.

## Evidence and reproduction

[Compact qualification records](upstream-9bd24d26-data/manifest.json) retain source/
compile-input bounds, binary identity, local checks, hosted CI and native runtime
summaries. The original acquired lifecycle summary has a stale8ab/5e human scope
label; its wrapper explicitly records that real binary/source/harness guards bind
the new acquisition to75de/9bd. No raw assertions/timestamps were rewritten and
no reacquisition was used for a label correction.

```sh
python3 docs/patches/tp2-adoption/upstream-9bd24d26-data/check.py
```

This auditor checks retained summary hashes and consistency, not live GPU/hosts
or a complete raw-trace replay. Full build/request/identity/counter/kernel records
remain at the acquisition lab. Rebuild exact75de with the standard release preset
and GUFO_ENABLE_TP2_RDMAON; use the [TP2 setup](../../models/qwen3.8-flash-next/TP2.md)
for matched peer settings. No new upstream PR, ready or upstream merge action was
performed; existing upstream patches remain subject to manual user review.
