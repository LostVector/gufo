"""Account for the first follow-up interval without double-counting spans."""
from pathlib import Path
import json
import re

lab = Path(__file__).resolve().parent
pattern = re.compile(r'HC_SOLO unit=(\w+) call=(\d+) stage=(\w+) t_ns=(\d+) elapsed_us=([\d.]+) session=([\da-f]+) position=(\d+) tokens=(\d+)')
stages = json.loads((lab / 'stage-analysis.json').read_text())
report = {}


def union_ms(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1e6


for arm, data in stages['arms'].items():
    spans = {}
    for line in (lab / f'{arm}-server.stderr').read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        unit, call, stage, ns, elapsed, session, position, tokens = match.groups()
        key = (unit, int(call))
        if stage == 'entry':
            assert key not in spans
            spans[key] = {'unit': unit, 'call': int(call), 'start_ns': int(ns), 'session': session,
                          'position': int(position), 'tokens': int(tokens)}
        else:
            assert stage == 'leave' and key in spans and 'end_ns' not in spans[key]
            spans[key]['end_ns'] = int(ns)
    assert spans and all('end_ns' in span for span in spans.values())
    first, followup = data['client_first_content_ns'], data['client_followup_content_ns']
    overlaps = []
    covered = []
    prefill = []
    decode = []
    for span in spans.values():
        start, end = max(first, span['start_ns']), min(followup, span['end_ns'])
        if end <= start:
            continue
        covered.append((start, end))
        (prefill if span['unit'] == 'prefill' else decode).append((start, end))
        overlaps.append({**span, 'wall_ms': (span['end_ns'] - span['start_ns']) / 1e6,
                         'first_gap_overlap_ms': (end - start) / 1e6})
    for span in data['cycles']:
        if span['unit'] != 'engine':
            continue
        start, end = max(first, span['start_ns']), min(followup, span['end_ns'])
        if end > start:
            covered.append((start, end))
    rows = [json.loads(line) for line in (lab / f'{arm}-requests.jsonl').read_text().splitlines()]
    request_rows = [row for row in rows if row['case'] == 'C2']
    report[arm] = {
        'gap_ms': data['first_followup_gap_ms'],
        'C2_prompt_tokens': [row['prompt_tokens'] for row in request_rows],
        'solo_spans_overlapping_gap': overlaps,
        'prefill_overlap_union_ms': union_ms(prefill),
        'serial_decode_overlap_union_ms': union_ms(decode),
        'all_engine_prefill_decode_union_ms': union_ms(covered),
        'uncovered_ms': data['first_followup_gap_ms'] - union_ms(covered),
    }
result = {'arms': report, 'scope': 'Diagnostic same-clock wall-span overlap with client first-followup. Union avoids nested/double-counted time. Prefill/DecodeStep durations include necessary GPU/I/O work, not pure CPU overhead. Uncovered wall time is not automatically wasted. Logging may change arrival/cohort/adaptive timing. No exoneration of original uninstrumented candidate flag from one diagnostic reversal.'}
(lab / 'solo-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
