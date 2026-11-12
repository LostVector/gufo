"""Verify the full untraced client family and reproduce completion tails/flags."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import statistics

out = Path(__file__).resolve().parent
manifest = json.loads((out / 'manifest.json').read_text())
assert len(manifest['files']) == 12
key = ('case', 'repetition', 'request_index', 'concurrency', 'prompt_sha256',
       'prompt_tokens', 'prefill_tokens', 'cached_prompt_tokens',
       'completion_tokens', 'completion_sha256')
data = {}
reference = None
for entry in manifest['files']:
    blob = (out / entry['file']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == entry['sha256']
    raw = gzip.decompress(blob)
    assert hashlib.sha256(raw).hexdigest() == entry['raw_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 12
    identities = [{k: row[k] for k in key} for row in rows]
    if reference is None:
        reference = identities
    assert identities == reference
    for row in rows:
        assert row['draft_tokens'] > 0 and row['cached_prompt_tokens'] == 0
        assert row['prefill_tokens'] == row['prompt_tokens']
        row['tail_ms'] = row['finished_at'] * 1000 - row['content_events'][-1]['perf_counter_ns'] / 1e6
        assert row['tail_ms'] >= 0
    pair, role = entry['phase'].split('-', 1)
    assert role == entry['role']
    assert (pair, role) not in data
    data[pair, role] = rows
assert {k for k in data} == {(f'pair{i:02d}', role) for i in range(1, 7)
                           for role in ('baseline', 'candidate')}
flags, c1_work_differences = [], []
metrics = ('wall_ms', 'client_ttft_ms', 'tail_ms', 'prefill_ms', 'decode_ms',
           'server_max_inter_token_ms')
for i in range(1, 7):
    pair = f'pair{i:02d}'
    for baseline, candidate in zip(data[pair, 'baseline'], data[pair, 'candidate']):
        if candidate['case'] == 'pp2048-tg128':
            assert all(baseline[k] == candidate[k] for k in
                       ('draft_tokens', 'draft_accepted_tokens', 'physical_execution_width'))
            assert baseline['raw_gufo']['draft_rounds'] == candidate['raw_gufo']['draft_rounds']
        for metric in metrics:
            a, b = baseline[metric], candidate[metric]
            if a is not None and b is not None and b-a > 3 and b > a*1.05:
                flags.append(dict(pair=pair, case=candidate['case'],
                    repetition=candidate['repetition'], actor=candidate['request_index'],
                    metric=metric, baseline=a, candidate=b, delta_ms=b-a))
tails = []
for case, actor in [('short-c2', 1), ('coding-c2', 0)]:
    for role in ('baseline', 'candidate'):
        values = [r['tail_ms'] for i in range(1, 7)
                  for r in data[f'pair{i:02d}', role]
                  if r['case'] == case and r['request_index'] == actor]
        assert len(values) == 6
        tails.append(dict(case=case, actor=actor, role=role, n=6,
            median_ms=statistics.median(values), worst_ms=max(values), values_ms=values))
with (out / 'timing-flags.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(flags[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(flags)
report = dict(all_144_prompt_output_cachework_identities_exact=True,
              c1_proposal_acceptance_rounds_and_width_exact=True,
              timing_flags=len(flags), first_finisher_completion_tails=tails,
              scope='Descriptive six-pair untraced confirmation. Timings flagged '
                    'at both 5% and 3ms; no waiver, significance or throughput claim. '
                    'This verifier checks client records, not the original host environment.')
(out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
