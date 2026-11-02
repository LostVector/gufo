"""Read-only retained-packet checks; does not claim GPU requalification."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['entries']:
    path = root / entry['path']
    assert path.resolve().is_relative_to(root)
    raw = path.read_bytes()
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'], entry['path']
proof = json.loads((root / 'upstream-ffn-hc-v1/release-source-proof.json').read_text())
assert proof['status'] == 'pass' and proof['compile_units'] == 327
assert proof['normalized_command_differences'] == []
assert proof['changed_production_paths'] == ['src/models/qwen38_flash_next/kernels/rocm/batch.cpp']
state = json.loads((root / 'upstream-ffn-hc-state-v1/analysis.json').read_text())
assert state['status'] == 'pass' and state['records'] == 104 and state['complete_materialized_state_pairs'] == 32
lab = root / 'upstream-ffn-hc-serving-v1'
protocol = json.loads((lab / 'protocol.json').read_text())
for name, expected in protocol['harness_sha256'].items():
    assert hashlib.sha256((lab / name).read_bytes()).hexdigest() == expected
rows = {arm: [json.loads(line) for line in (lab / f'{arm}-requests.jsonl').read_text().splitlines()] for arm, _ in protocol['sequence']}
flags, work = [], []
for arm in ('before-baseline', 'after-baseline'):
    assert len(rows[arm]) == len(rows['candidate']) == 9
    for baseline, candidate in zip(rows[arm], rows['candidate']):
        assert (baseline['case'], baseline['repetition'], baseline['request_index']) == (candidate['case'], candidate['repetition'], candidate['request_index'])
        for key in ('completion_sha256', 'prompt_sha256', 'completion_tokens', 'prompt_tokens', 'prefill_tokens', 'cached_prompt_tokens'):
            assert baseline[key] == candidate[key]
        for key in ('draft_tokens', 'draft_accepted_tokens'):
            if baseline[key] != candidate[key]:
                work.append({'control': arm, 'case': baseline['case'], 'repetition': baseline['repetition'], 'actor': baseline['request_index'], 'metric': key, 'baseline': baseline[key], 'candidate': candidate[key]})
        for key in ('wall_ms', 'prefill_ms', 'decode_ms', 'client_ttft_ms', 'server_ttft_ms', 'server_max_inter_token_ms', 'first_followup_content_gap_ms', 'maximum_content_gap_ms'):
            before, after = baseline[key], candidate[key]
            if before is not None and after is not None and after - before > 10 and after > before * 1.05:
                flags.append({'control': arm, 'case': baseline['case'], 'repetition': baseline['repetition'], 'actor': baseline['request_index'], 'metric': key, 'baseline_ms': before, 'candidate_ms': after, 'increase_ms': after - before})
observed = json.loads((lab / 'observed-eos-analysis.json').read_text())
assert flags == observed['phase_flags'] and len(flags) == 6
assert work == observed['adaptive_work_changes'] and len(work) == 8
assert 'AssertionError' in (lab / 'analysis.log').read_text()
for arm, data in rows.items():
    pp = [row for row in data if row['case'] == 'pp2048-tg128']
    assert len(pp) == 2 and all(row['prompt_tokens'] == 2048 and row['completion_tokens'] == 96 for row in pp)
repro = json.loads((root / 'upstream-ffn-hc-first-gap-repro-v1/solo-analysis.json').read_text())
assert len(repro['arms']) == 6 and all(row['gap_ms'] <= 95 for row in repro['arms'].values())
bench = json.loads((root / 'upstream-ffn-hc-standard-bench-v1/validation.json').read_text())
assert bench['status'] == 'pass'
assert bench['output_token_hashes']['baseline'] == bench['output_token_hashes']['candidate']
print(f"PASS {len(manifest['entries'])} hashes; 327 compile commands, retained104/32 state summary, 27 exact HTTP records, all6 flags/8 work comparisons, EOS96 accounting failure, six bounded baseline captures, fixed128 benchmark hashes. Offline scope only.")
