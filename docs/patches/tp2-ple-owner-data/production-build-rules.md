# Keep production and test build directories separate

The repository's `release` preset intentionally uses `RelWithDebInfo`, with
`BUILD_TESTING=OFF` and the same optimization as its shipped Nix package. Do not
substitute a build-type name for checking the actual preset and compiler inputs.

A component probe needs testing enabled, but enabling it in a production build
directory changes shared-library compilation: `src/cache/CMakeLists.txt` adds
`GUFO_CACHE_TESTING` publicly and `-UNDEBUG` to the cache target. Test extension
include directories also reach some consumers. Reconfiguring that directory
back and forth can therefore change the binary subsequently called a release.

Use separate directories:

```sh
cmake --preset release -B build/production
cmake --build build/production --target gufo --parallel 4
cmake --preset gpu-test -B build/operator-tests
cmake --build build/operator-tests --target qwen38_flash_next_mtp_ops_test --parallel 4
```

Keep the documented local dependency/compiler arguments when reproducing this
lab's private toolchain. Do not take a component/test binary as production
performance evidence. A useful build-doctor check is:

- Compare `CMakeCache.txt` with the source's actual `release` preset.
- Require `BUILD_TESTING=OFF` for headline production timings.
- Check model/family compile commands for leaked `GUFO_CACHE_TESTING` or test
  include paths; match normalized compiler inputs between arms.
- Freeze source commit/tree, source and build-input hashes, executable hash,
  toolchain/dependencies, model/settings and runtime identity before acquisition.

The corrected PLE experiment checks all77 common Flash-Next/Qwen-family command
lines; only the two additional PLE implementation sources are candidate-only.
`verify-cooperative-ple-production-inputs.py` records and enforces those inputs.
This verifies compiler inputs, not machine-code or numerical equivalence.
Correctness, cache/cancellation/state replay and timings remain separate gates.

The rejected partial v1 acquisition remains preserved with its immutable
protocol and a build-mode erratum. Its correctness checks are attributable to
the actual testing-enabled binaries; its timings support no production claim.
The corrected production acquisition is a fresh v2 study, not a relabeling or
selective extension of v1.
