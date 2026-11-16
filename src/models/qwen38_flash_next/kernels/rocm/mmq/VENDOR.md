# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q8_0 dense projections and
Q4_K, Q5_K, Q5_1 and Q8_0 routed expert projections, including the paired
Q4_K gate/up path. `qfn_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the Flash-Next operator tests
under `tests/models/qwen38_flash_next/` plus the AR/MTP session replay.
The routed GEMM test compares supported formats against a floating-point
reference and checks the paired adapter against two separate projections.
Build and validate through Nix as described in `AGENTS.md`.

Q2_0 block geometry and the HIP vector dot-product arm are additionally adapted
from `ggml-org/llama.cpp` at
`1719dbb0ee7bdaf56de083079d3613bb670160a5` (MIT). The imported arm retains
its packed-code mapping, DP4A grouping and scale order; the CUDA-only arm is
omitted and the DP4A call uses the local HIP name. This is a scoped import,
not a replacement of the original vendoring pin.

Local packed dense verification shares weight unpacking across2–8 token
accumulators. It retains the vendored format-specific vector dot, K traversal
and wave reduction; Q8 and routed dispatch are unchanged. See the GSQ-RCO TP1
quality/performance evidence under docs/internal/models/qwen3.8-flash-next/.
