from pathlib import Path
import hashlib
import json

lab = Path(__file__).resolve().parent
protocol = json.loads((lab / 'protocol.json').read_text())
for name, expected in protocol['harness_sha256'].items():
    assert hashlib.sha256((lab / name).read_bytes()).hexdigest() == expected, name
records = {arm: [json.loads(line) for line in (lab / f'{arm}-requests.jsonl').read_text().splitlines()] for arm, _ in protocol['sequence'] if (lab / f'{arm}-requests.jsonl').exists()}
functional = []
flags = []
work = []
for arm, rows in records.items():
    assert len(rows) == 9
    for row in rows:
        assert row['metrics_source'] == 'gufo'
        assert row['completion_tokens'] == (8 if row['case'] == 'warmup' else 96 if row['case'] == 'pp2048-tg128' else 128)
        assert row['cached_prompt_tokens'] == 0 and not row['cache_hit']
        assert row['prefill_tokens'] == row['prompt_tokens']
        assert row['draft_tokens'] > 0 and row['draft_accepted_tokens'] > 0
        if row['case'] == 'pp2048-tg128':
            assert row['prompt_tokens'] == 2048
    for index in (0, 1):
        same = [row for row in rows if row['case'] in ('solo-before', 'C2', 'solo-after') and row['request_index'] == index]
        for key in ('completion_sha256', 'prompt_sha256', 'completion_tokens', 'prefill_tokens', 'cached_prompt_tokens'):
            assert len({row[key] for row in same}) == 1, (arm, index, key)
        functional.append({'arm': arm, 'actor': index, 'solo_before_C2_solo_after_exact': True})
    for row in rows:
        if row['case'] == 'C2':
            assert row['physical_execution_width'] == 2
    log = (lab / f'{arm}-server.stderr').read_text()
    assert 'runtime error:' not in log and 'HIP error' not in log
    eos_lines = [line for line in log.splitlines() if 'event=completed method=POST path=/v1/chat/completions' in line and 'prompt_tokens=2048 ' in line]
    assert len(eos_lines) == 2 and all('generated_tokens=96 finish=stop ' in line for line in eos_lines)
result = {'status': 'baseline-diagnostic-functional-pass', 'records': sum(len(rows) for rows in records.values()), 'solo_concurrent_exact_checks': functional,
          'phase_flags': flags, 'adaptive_work_changes': work,
          'scope': 'Baseline-only event capture with predeclared six-start cap and >95ms stopping rule; no gain estimates. Exact completions/cache work, actual MTP, pp2048/EOS96 and coding128. Logging may change adaptive timings; not release performance or exoneration of original flags. Timing flags retained; no confidence, equivalence, no-regression or wider-concurrency conclusion. Fixed-history full-state/operator qualification is separate.'}
(lab / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
