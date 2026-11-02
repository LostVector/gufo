"""Correlate source-stage wall spans with the client's first follow-up gap."""
from pathlib import Path
import collections
import json
import re

lab = Path(__file__).resolve().parent
stage_re = re.compile(r'HC_STAGE unit=(\w+) call=(\d+) stage=(\w+) t_ns=(\d+) elapsed_us=([\d.]+)')
value_re = re.compile(r'HC_VALUE unit=(\w+) call=(\d+) key=(\w+) value=(\d+)')
result = {}
for arm, _ in json.loads((lab / 'protocol.json').read_text())['sequence']:
    if not (lab / f'{arm}-requests.jsonl').exists():
        continue
    groups = collections.defaultdict(lambda: {'stages': [], 'values': collections.defaultdict(list)})
    for line in (lab / f'{arm}-server.stderr').read_text().splitlines():
        match = stage_re.search(line)
        if match:
            unit, call, stage, ns, elapsed = match.groups()
            groups[(unit, int(call))]['stages'].append({'stage': stage, 't_ns': int(ns), 'elapsed_us': float(elapsed)})
        match = value_re.search(line)
        if match:
            unit, call, key, value = match.groups()
            groups[(unit, int(call))]['values'][key].append(int(value))
    assert {unit for unit, _ in groups} == {'engine', 'batch'}
    assert all(len([key for key in groups if key[0] == unit]) == 16 for unit in ('engine', 'batch'))
    rows = [json.loads(line) for line in (lab / f'{arm}-requests.jsonl').read_text().splitlines()]
    actor = next(row for row in rows if row['case'] == 'C2' and row['request_index'] == 0)
    events = actor['content_events_ns']
    assert len(events) > 1
    first, followup = events[:2]
    intervals = []
    cycles = []
    for (unit, call), group in groups.items():
        stages = group['stages']
        assert stages[0]['stage'] == 'entry' and stages[-1]['stage'] == 'leave'
        assert all(b['t_ns'] >= a['t_ns'] for a, b in zip(stages, stages[1:]))
        cycle = {'unit': unit, 'call': call, 'values': dict(group['values']),
                 'start_ns': stages[0]['t_ns'], 'end_ns': stages[-1]['t_ns'],
                 'wall_ms': (stages[-1]['t_ns'] - stages[0]['t_ns']) / 1e6}
        cycles.append(cycle)
        for a, b in zip(stages, stages[1:]):
            overlap = max(0, min(b['t_ns'], followup) - max(a['t_ns'], first))
            intervals.append({'unit': unit, 'call': call, 'from': a['stage'], 'to': b['stage'],
                              'wall_ms': (b['t_ns'] - a['t_ns']) / 1e6,
                              'first_gap_overlap_ms': overlap / 1e6,
                              'values': dict(group['values'])})
    # Engine and batch spans nest; never add their durations to one total.
    result[arm] = {'first_followup_gap_ms': (followup - first) / 1e6,
                   'client_first_content_ns': first, 'client_followup_content_ns': followup,
                   'cycles': sorted(cycles, key=lambda row: row['start_ns']),
                   'first_gap_intervals': [row for row in intervals if row['first_gap_overlap_ms'] > 0],
                   'largest_intervals': sorted(intervals, key=lambda row: row['wall_ms'], reverse=True)[:12],
                   'all_intervals': intervals}
report = {'arms': result, 'scope': 'Diagnostic CPU wall stages from actual normal-controller path. Linux steady_clock and Python perf_counter use the monotonic clock. Engine/batch intervals nest; do not sum them. GPU submission spans are not GPU execution times, and stream-completion host waits include necessary queued GPU work. Source logging can alter adaptive plans; a clean diagnostic replay does not exonerate the original release timing flag.'}
(lab / 'stage-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
for arm, data in result.items():
    print(arm, 'first gap ms', data['first_followup_gap_ms'])
    for row in data['first_gap_intervals']:
        print(row['unit'], row['call'], row['from'], row['to'], round(row['wall_ms'], 3), row['values'])
