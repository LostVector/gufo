"""Read-only payload and publication-chronology audit; no live GPU claims."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['entries']:
    f = root / entry['path']
    assert f.resolve().is_relative_to(root)
    raw = f.read_bytes()
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'], entry['path']
v1 = root / 'first-output-publication-trace-v1'
v2 = root / 'first-output-publication-trace-v2'
for lab in (v1, v2):
    protocol = json.loads((lab / 'protocol.json').read_text())
    for name, expected in protocol['harness_sha256'].items():
        assert hashlib.sha256((lab / name).read_bytes()).hexdigest() == expected, (lab, name)
    if lab == v2:
        assert hashlib.sha256((v1 / 'baseline-requests.jsonl').read_bytes()).hexdigest() == protocol['original_baseline_requests_sha256']
for role in ('baseline', 'candidate'):
    build = json.loads((v1 / 'build-protocol.json').read_text())['records'][role]
    assert build['changed_serving_members'] == ['text_generation_scheduler.cpp.o']
    assert hashlib.sha256((v1 / role / 'text_generation_scheduler.cpp').read_bytes()).hexdigest() == build['source_sha256']
assert 'AssertionError: []' in (v1 / 'wrong-archive-failure.log').read_text()
assert 'resident_requests_at_admission' in (v1 / 'candidate-requests.stderr').read_text()
assert len((v1 / 'candidate-requests.jsonl').read_text().splitlines()) == 5
rows = {'baseline': [json.loads(s) for s in (v1 / 'baseline-requests.jsonl').read_text().splitlines()],
        'candidate': [json.loads(s) for s in (v2 / 'candidate-requests.jsonl').read_text().splitlines()]}
assert all(len(value) == 11 for value in rows.values())
flags, work = [], []
for baseline, candidate in zip(rows['baseline'], rows['candidate']):
    assert (baseline['case'], baseline['request_index'], baseline['repetition']) == (candidate['case'], candidate['request_index'], candidate['repetition'])
    for key in ('completion_sha256', 'prompt_sha256', 'completion_tokens', 'prompt_tokens', 'prefill_tokens', 'cached_prompt_tokens'):
        assert baseline[key] == candidate[key], (baseline['case'], key)
    for key in ('draft_tokens', 'draft_accepted_tokens'):
        if baseline[key] != candidate[key]:
            work.append({'control': 'baseline', 'case': baseline['case'], 'repetition': baseline['repetition'], 'actor': baseline['request_index'], 'metric': key, 'baseline': baseline[key], 'candidate': candidate[key]})
    for key in ('wall_ms', 'prefill_ms', 'decode_ms', 'client_ttft_ms', 'server_ttft_ms', 'server_max_inter_token_ms', 'first_followup_content_gap_ms', 'maximum_content_gap_ms'):
        before, after = baseline[key], candidate[key]
        if before is not None and after is not None and after - before > 10 and after > before * 1.05:
            flags.append({'control': 'baseline', 'case': baseline['case'], 'repetition': baseline['repetition'], 'actor': baseline['request_index'], 'metric': key, 'baseline_ms': before, 'candidate_ms': after, 'increase_ms': after - before})
analysis = json.loads((v2 / 'analysis.json').read_text())
assert flags == analysis['phase_flags'] and len(flags) == 5
assert work == analysis['adaptive_work_changes'] and len(work) == 8
check = subprocess.run([sys.executable, str(v2 / 'publication.py'), '--check'], capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
publication = json.loads((v2 / 'publication-analysis.json').read_text())
assert len(publication['records']) == 8 and len(publication['ready_to_first_publication']) == 53
assert max(row['ms'] for row in publication['ready_to_first_publication']) == 0.034929
selected = next(row for row in publication['records'] if row['arm'] == 'candidate' and row['case'] == 'C2' and row['actor'] == 1)
steps = selected['ready_steps_during_gap']
assert [(row['returned_selections'], row['preview_duplicates'], row['novel_publications']) for row in steps] == [(1, 1, 0), (3, 0, 3)]
# The older all-actor reanalysis uses immutable inputs in the previous packet.
older = json.loads((root / 'first-output-existing-trace-analysis-v1/analysis.json').read_text())
previous = root.with_name('canonical-ffn-hc-data')
for name, expected in older['inputs_sha256'].items():
    relative = Path(name).relative_to('tp-cache-retention')
    assert hashlib.sha256((previous / relative).read_bytes()).hexdigest() == expected, name
print(f"PASS {len(manifest['entries'])} payload hashes; frozen harness/source identities, 22 exact outputs/cache work, all5 flags/8 work changes, eight recomputed publication histories, 53 ready/publication intervals and actual preview-only cycle. Original failures retained. Offline scope only.")
