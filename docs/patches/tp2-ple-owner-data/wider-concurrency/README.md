# Four-session PLE qualification

Fixed untraced baseline/candidate/baseline,20requests/start. Baselinefbacf9cc,
combined candidate2669e07c; production OFF,UD-Q4_K_XL/shared-Q8_0 MTP,thinkingoff,
65Kcapacity,four sessions,chunk2048,1GiBcache,approved81C6/TX32/crossed40+20Gb/s.
The combined candidate includes later generic cache/platform changes, so this is
not an isolated-PLE timing estimate. No driver/system change, profiler/admin
sampler or full-table warmup. The inherited engine retains10s post-stop cleanup;
no request-time idle is added.

Warmup8, four solo128 controls, three solo17/33/65 controls, C3x128, C4x128,
raggedC4 limits17/33/65/128, then post-wave solo128 replay. All60 cross-arm
prompt/output/cache-work identities match. Each concurrent/replayed reply also
matches its own same-prompt/same-limit solo control. ActualMTP/no prompt reuse
are verified. Reported resident admissions1..3/1..4 and physical widths3/4 are
observed on every request in those waves. This does not expose every transient
batch transition. Six native error-counter windows, exact binary/settings/PID,
frozen harness, unchanged boot/topology/kernel and final approved idle checks pass.

## Observed complete client-wave throughput

| Wave | Baseline before tok/s | Candidate tok/s | Baseline after tok/s |
| --- | ---: | ---: | ---: |
| C3,128each |86.0133|85.7871|85.6172|
| C4,128each |97.9841|98.5061|98.5071|
| RaggedC4 |65.5751|67.0473|66.5479|

These are aggregate complete-wave rates, not single-client decode or a new
statistically supported gain/equivalence result. One fixed B/C/B cannot supply
that conclusion. Neither wave throughput nor exact outputs waive request flags.

## Retained timing flag

Six strict5%AND3ms entries all concern ragged actor1 versus the first baseline
(wall, client/server interval, decode and duplicate raw fields). Its33-token
reply takes1540.62/1758.70/1779.15ms in baseline/candidate/baseline. MTP work is
10rounds/23drafts/21accepted versus13/22/15 in both candidate and later baseline.
All35 adaptive-work field differences across the full study remain recorded.

The [matched visible-progress plot](matched-progress.png) uses33 identical
emitted byte frontiers, without interpreting them as token IDs. Candidate begins
1.86ms later than the first baseline and finishes visible content250.62ms later;
it is14.66ms earlier initially and20.23ms earlier finally than the later baseline.
The wall delta is218.08ms because completion tails also differ. This observes
a shared work/latency mode; it does not prove that3extra draft rounds caused the
whole delay or identify a per-round scheduling mechanism. No phase flag is waived.
Functional qualification passes; timing remains inconclusive/unqualified.

[Full report](validation.json), [flag review](timing-flag-review.json), and
[raw acquisition archive](acquisition.zip) retain every request and observation.
The archive includes exact protocols/harness,60 JSONL records and analysis/plot
scripts. Run `run.py`, then `analyze.py`, `review-flags.py` from the lab study
directory; plotting uses`.venv/bin/python plot-progress.py`. Existing lab model
paths, SSH and inventory helpers are required; this is not an installer.
