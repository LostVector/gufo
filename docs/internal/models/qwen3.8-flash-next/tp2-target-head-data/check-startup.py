"""CPU-only recomputation of the retained initial SessionSync read/wait spans."""
from pathlib import Path
import csv
import json
import math

root = Path(__file__).resolve().parent
summary = json.loads((root / 'startup-trace-summary.json').read_text())
with (root / 'startup-first-sync-spans.csv').open() as stream:
    spans = list(csv.DictReader(stream))
assert summary['measured_binaries']['baseline']['sha256'] == '1caf9512d974e7f4b93b34ef7f23586d31d61e03efcb3c2003edd64bc686c932'
assert summary['measured_binaries']['candidate']['sha256'] == '6c4f1403d57549e712d70b3c18b36585388b8198712bec0a84c04083afd533ae'

def union_ms(rows, lo, hi):
    merged = []
    intervals = sorted((max(lo, int(x['begin_ns'])), min(hi, int(x['end_ns']))) for x in rows)
    for begin, end in intervals:
        if end <= begin:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return sum(end - begin for begin, end in merged) / 1e6

for row in summary['starts']:
    assert row['EVO_warmup_Forward_rows'] == [12, 7, 4, 4]
    assert row['warmup_head_gather_host_call_union_ms'] == 0
    for rank in (0, 1):
        selected = [x for x in spans if x['phase'] == row['phase'] and int(x['rank']) == rank]
        boundaries = [x for x in selected if x['tag'] == 'session_sync']
        assert len(boundaries) == 1
        begin, end = (int(boundaries[0][key]) for key in ('begin_ns', 'end_ns'))
        assert end > begin
        assert all(int(x['begin_ns']) < end and int(x['end_ns']) > begin for x in selected)
        metrics = {'EVO_first_sync_ms' if rank == 0 else 'Corsair_first_sync_ms': (end - begin) / 1e6}
        if rank == 0:
            metrics['EVO_first_sync_HIP_event_wait_ms'] = union_ms([x for x in selected if x['tag'] == 'hipEventSynchronize'], begin, end)
        else:
            reads = [x for x in selected if x['tag'] == 'pread']
            metrics['Corsair_ngram_wait_ms'] = union_ms([x for x in selected if x['tag'] == 'ngram_wait'], begin, end)
            metrics['Corsair_pread_union_ms'] = union_ms(reads, begin, end)
            metrics['Corsair_longest_pread_ms'] = max((float(x['duration_ms']) for x in reads), default=0)
        for key, value in metrics.items():
            assert math.isclose(row[key], value, rel_tol=0, abs_tol=1e-6), (row['phase'], rank, key, row[key], value)
        assert union_ms([x for x in selected if x['tag'] == 'target_head_gather'], begin, end) == 0
print('PASS: eight initial SessionSync read/wait intervals independently recomputed from retained CSV')
print('Whole-warmup row/gather declarations are checked against the summary; only initial SessionSync spans are bundled.')
print('This checks application wall spans, not native NVMe service, GPU execution or original untraced outliers.')
