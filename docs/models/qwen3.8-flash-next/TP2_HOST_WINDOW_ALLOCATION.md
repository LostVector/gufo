# Allocate TP host windows at unused virtual addresses

The native TP communicator mapped its send and receive host windows at fixed
addresses starting at0x700000000000. If another mapping occupied either range,
MAP_FIXED_NOREPLACE returnedEEXIST and model startup failed. An unchanged6c4f
baseline hit this during a confidence study before model loading; the peer then
failed bootstrap. The exact conflicting mapping was not captured. The study was
stopped, retained as incomplete and contributes no confidence result.

Let Linux choose unused anonymous mappings instead. Each rank already publishes
its receive address and rkey in Wire; RDMA writes use that peer address. The GPU
already obtains its local alias through hipHostGetDevicePointer. Equal virtual
addresses across ranks are unnecessary. Registered bytes, window count/size,
coherence, staging, collectives and cleanup ownership are unchanged. Host peer
metadata now requires a nonzero page-aligned address, a nonzero rkey and room
for the complete buffer without uint64 overflow. Device-window metadata is
unchanged.

Wire version16 rejects older peers before model loading because older host peers
require the former fixed address. Upgrade both nodes together. No kernel, driver,
boot, global ASLR or ROCm change is required. This is startup robustness, not a
claimed inference or transport speedup.

## Validation

One production source/host object changes: verbs.cpp. All headers, GPU kernels,
other objects, model arithmetic and snapshot format19 are unchanged from6c4f.
The isolated measured binary is
e8157ddba559f3f0c4f7b057e7ec6a022b00057ea6fbad48667751cc7a4f27bf.
The review source is byte-identical to that build's production source.

- Real provider new/new bootstrap and10KiB GPU all-reduce produce exactly3.0 on
  both ranks. Both wire14/16 rank orders reject promptly.
- A process-local test reserves one PROT_NONE page at the legacy send base,
  using MAP_FIXED_NOREPLACE without overwriting existing mappings. The old
  factory failsEEXIST on both nodes. The new factory maps two distinct unused
  144MiB windows per rank and completes the real GPU/RDMA sum with the old page
  still occupied. Global ASLR is unchanged.
- With that reservation held on both hosts, all12 canonical C1/C2 full-model
  outputs/prompt/cache work match and C1MTP work matches the actual6c4f parent.
  Adaptive C2work differences are retained. This exercises large prefills and
  repeated queued decode, not just a10KiB message. The initial offline audit
  mistakenly used an older policy's work reference; the original audit is
  preserved and a separate corrected audit uses the SHA-pinned immediate parent.
  The acquisition was not repeated or excluded.
- A separate normal startup with no injected reservation passes13 HTTP cache
  and cancellation cases: greedy/seeded-sampled keep/discard restores,
  both C2cancellation roles, live survivor width2 and exact control/replays.
- All checked transport counters/kernel windows are clean; processes exit and
  both hosts return to the approved native four-rail idle baseline. Test hosts
  use7.3.0-strix-test, ROCm7.14.1, crossed40+20Gb/s links and the same pinned
  Qwen3.8-Flash-Next UD-Q4_K_XL/shared-Q8MTP inputs as the parent study.

The CPU-only regression compiles the actual MapHostBuffer/UnmapHostBuffer helpers
verbatim with real mmap and a fake HIP registrar. It checks the legacy collision
control, distinct usable mappings, null/success cleanup, registration-failure
unmapping and mmap failure. It needs a Linux C++20 compiler, no GPU, RDMA, root
or system configuration changes:

```bash
python3 tests/models/qwen38_flash_next/rdma_host_window_allocation.py
```

This simulated registrar is not transport or GPU evidence; those checks above
used the real provider and hardware. GPU DMA-BUF placement is unavailable on
these hosts and is not newly runtime-qualified. No physical cable/GPU-fault,
transparent reconnect, overnight, performance-equivalence or default-adoption
claim is made. The older batched-head experiment is separate and not included.

[Bootstrap checks](tp2-host-window-data/bootstrap-validation.json),
[raw bootstrap outputs and mapping addresses](tp2-host-window-data/bootstrap-outputs.json),
[model checks](tp2-host-window-data/model-validation.json),
[exact model-work control](tp2-host-window-data/model-work-reference.json),
[cache/cancellation checks](tp2-host-window-data/cache-cancel-validation.json),
[build provenance](tp2-host-window-data/build-provenance.json) and
[interrupted study status](tp2-host-window-data/incomplete-confidence-study.json)
are retained for review. Full raw logs, exact commands and frozen harnesses are
in the investigation's rdma-address-allocation-v1, its model/lifecycle fixtures
and the incomplete batched-target-head-confidence-v1. The old confidence
samples are not reused after changing the allocator.
