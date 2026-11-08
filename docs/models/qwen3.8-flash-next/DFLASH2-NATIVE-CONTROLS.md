# Flash Next DFlash2 native-mode controls

Baseline `0fb50ce4` versus candidate `4e78ff2f` on EVO-X2 gfx1151,
private ROCm 7.14.1, release preset (RelWithDebInfo, BUILD_TESTING=OFF),
same UD-Q4_K_XL target and shared Q8_0 MTP sidecar. The extension selection is
qwen + qwen38_flash_next in both binaries.

## CLI controls

ABBA launches, one retained sample per shape per launch, with the engine’s
normal untimed warmup. Thus two observations per variant/shape. CLI capacity
is 6,145 tokens; pp2048 measures the suffix at depth 0 or 4,096, and tg128
uses a 16-token seed at shallow depth or a 4,096-token seed.

| Mode | Test | Baseline tok/s, both observations | Candidate tok/s, both observations |
| --- | --- | --- | --- |
| AR | pp2048 | 1827.02, 1829.53 | 1834.06, 1831.13 |
| AR | tg128 | 30.15, 30.10 | 30.14, 30.12 |
| AR | pp2048 @ d4096 | 1679.51, 1670.99 | 1677.36, 1677.87 |
| AR | tg128 @ d4096 | 29.53, 26.21 | 29.52, 29.49 |
| MTP | pp2048 | 1775.87, 1770.78 | 1775.92, 1771.22 |
| MTP | tg128 | 44.61, 44.70 | 44.79, 44.77 |
| MTP | pp2048 @ d4096 | 1615.66, 1618.85 | 1620.96, 1620.08 |
| MTP | tg128 @ d4096 | 52.83, 46.69 | 52.81, 52.69 |

The final baseline launch slowed deep decode in both modes. It remains in the
table and raw data. Candidate results closely match the earlier faster baseline;
these two-sample observations do not establish statistical superiority.

## HTTP controls and retained flags

Both modes pass all eleven selected functional report groups, including
long-context reuse, cancellation and RAM/restarted-disk cache histories.
All compared outputs and expected cache/prefill work match. Neither mode has
a per-request timing flag under the unchanged 5% AND 10 ms gate.

| Mode | Comparison measurements | Per-request flags | Remaining lifecycle flags |
| --- | ---: | ---: | --- |
| AR | 637 | 0 | startup_ms: 25.13 → 28.66 s, restart_ms: 25.83 → 27.44 s |
| MTP | 637 | 0 | startup_ms: 26.64 → 29.46 s |

Both aggregate comparisons remain **inconclusive** because of those lifecycle
flags. They have not been converted into clean timing passes.

Instead of repeating the passing request suites, four startup-only probes used
the canonical harness server lifecycle, the exact AR server options, fresh
ports and owned empty disk directories. In chronological order:

| Binary | Ready time (s) | Loader time (s) |
| --- | ---: | ---: |
| candidate | 29.68 | 29.48 |
| baseline | 32.80 | 32.61 |
| candidate | 35.01 | 34.83 |
| baseline | 36.32 | 36.13 |

The increase occurs across both binaries and follows test order. It supports
time-dependent loading variability; it does not identify its cause or qualify
the original persisted restart history. Existing CLI model-load logs were
24.3–25.4 s for candidate and 24.5–25.4 s for baseline.

## Reproduction and review scope

Use identical release-preset toolchains and the two revisions above. For each
ABBA launch and mode, run:

```sh
gufo bench --model TARGET -p 2048 -n 128 -d 0,4096 -r 1 \
  --temperature 0 --speculative off
# MTP: replace off with mtp and add --mtp-model MTP_SIDECAR.
```

For each mode, run `tests/functional/run.py --record-baseline` on baseline,
then `--baseline BASELINE_REPORT` on candidate, with `--sampling-preset qwen38`,
`--expected-input-modalities text`, `--suite long-context --suite cache`, and:

```sh
gufo serve llm --model TARGET --speculative off --context 8192 \
  --sessions 2 --cache-ram-bytes 2147483648 --think off
# MTP: replace off with mtp and add --mtp-model MTP_SIDECAR.
```

Raw commands, environment, binary hashes, all flags and measurements are in
[the JSON record](dflash-native-controls-v1.json) and [raw CLI CSV](dflash-native-controls-v1.csv).
Full local reports/logs remain under `~/strix-dispatch-investigation/
dflash-upstream-2026-11-07/results/{native-controls-v1,native-followup-v1}/`.
The independent review and verified include-facade fix are in
[DFLASH2-REVIEW.md](DFLASH2-REVIEW.md).

No end-to-end DFlash gain is claimed. Full-head/small-head comparisons on
natural prose and coding remain performance characterization work. The TP2
implementation remains separate; the original user MTP trial is restored.
