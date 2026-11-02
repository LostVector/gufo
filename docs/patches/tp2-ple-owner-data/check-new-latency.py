"""Audit exported completion-send and HIP host partitions against raw spans."""
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

p = Path(__file__).resolve().parent
canonical = [json.loads(s) for s in gzip.decompress(
    (p / 'cooperative-ple-release-comparison-v2-pair01-baseline.jsonl.gz').read_bytes()).splitlines()][:5]
identity = ('case', 'repetition', 'request_index', 'concurrency', 'prompt_sha256',
            'prompt_tokens', 'prefill_tokens', 'cached_prompt_tokens',
            'completion_tokens', 'completion_sha256')


def spans(data):
    entries, result = {}, []
    for line in data.decode().splitlines():
        if not line.startswith('APP,'):
            continue
        s = dict(zip(('kind', 'end', 'begin', 'tid', 'id', 'depth', 'cpu', 'pid', 'value'),
                     map(int, line.split(',')[1:])))
        key = tuple(s[k] for k in ('tid', 'id', 'depth', 'begin'))
        if s['kind'] == 1:
            assert key not in entries
            entries[key] = s
        else:
            assert s['kind'] == 2 and key in entries
            s['entry_value'] = entries.pop(key)['value']
            result.append(s)
    assert not entries
    return sorted(result, key=lambda s: s['begin'])


def union(intervals):
    total, end = 0, -1
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total / 1e6


def aggregate(ss, tags):
    result = {}
    for ident, name in tags.items():
        calls = [s for s in ss if s['id'] == ident]
        if calls:
            times = [(s['end'] - s['begin']) / 1e6 for s in calls]
            result[name] = dict(calls=len(calls), sum_host_ms=sum(times),
                                max_host_ms=max(times), nonzero_returns=dict(collections.Counter(
                                    str(s['value'] & 0xffffffff) for s in calls if s['value'] & 0xffffffff)))
    return result


count = 0
for bundle, study, report_name in (
    ('completion-tail', 'ple-owner-end-send-trace-v2', 'end-analysis.json'),
    ('startup-hip', 'ple-owner-startup-hip-trace-v1', 'hip-analysis.json'),
):
    with zipfile.ZipFile(p / bundle / 'acquisition.zip') as archive:
        index = json.loads(archive.read('archive-index.json'))
        for name, digest in index.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        base = 'tp-cache-retention/' + study + '/'
        read = lambda name: json.loads(archive.read(base + name))
        protocol = read('protocol.json')
        for name, digest in protocol['harness_sha256'].items():
            assert index[base + name] == digest, name
        assert read('cleanup.json')['errors'] == []
        report = json.loads((p / bundle / report_name).read_text())
        assert report == read(report_name)
        for name, digest in report['input_sha256'].items():
            assert index[name] == digest
        csv_expected = []
        for (phase, role), saved in zip(protocol['sequence'], report['phases']):
            assert saved['phase'] == phase
            raw = archive.read('results/model-scaling/' + study + '-' + phase + '.jsonl')
            rows = [json.loads(s) for s in raw.splitlines()]
            assert len(rows) == 5 and all(all(a[k] == b[k] for k in identity) for a, b in zip(rows, canonical))
            count += len(rows)
            both = []
            for rank in (0, 1):
                folder = phase + '/trace-r' + str(rank) + '/'
                status = read(folder + 'status.json')
                stats = status['stats']
                assert status['app_collector_exit_code'] == 0 and status['app_coverage_complete']
                assert not any(stats['counters'][:5]) and not any(stats['program_recursion_misses'])
                assert stats['remaining_span_keys'] == stats['remaining_depth_keys'] == 0
                both.append(spans(archive.read(base + folder + 'app.csv')))
                if bundle == 'startup-hip':
                    library = read('hip-library.json')
                    mapped = read(folder + 'hip-library-map.json')
                    assert mapped['library'] == report['library'] == library and mapped['maps']
                    tags = {f['id']: f['tag'] for f in read('probes.json')[role]['functions'] if f['tag'].startswith('hip')}
                    assert aggregate(both[-1], tags) == saved['full_trace_HIP_APIs_per_rank'][rank]
                    for s in both[-1]:
                        if s['id'] in tags:
                            csv_expected.append((phase, rank, tags[s['id']], s['tid'], s['begin'], s['end'],
                                                 (s['end'] - s['begin']) / 1e6, s['value'] & 0xffffffff))
            head = both[0]
            if bundle == 'completion-tail':
                ends = [s for s in head if s['id'] == 4]
                assert len(ends) == len(saved['EndRequests']) == 5
                for end, record in zip(ends, saved['EndRequests']):
                    sends = [s for s in head if s['id'] == 7 and s['tid'] == end['tid']
                             and end['begin'] <= s['begin'] and s['end'] <= end['end']]
                    assert len(sends) == 1 and end == record['end'] and sends[0] == record['send']
                    send = sends[0]
                    for field, value in [('pre_send_ms', send['begin'] - end['begin']),
                                         ('send_ms', send['end'] - send['begin']),
                                         ('post_send_ms', end['end'] - send['end']),
                                         ('total_ms', end['end'] - end['begin'])]:
                        assert record[field] == value / 1e6
                    steps = [s for s in head if s['id'] == 13 and s['begin'] < end['end'] and s['end'] > end['begin']]
                    assert steps == record['overlapping_SessionDecodeStep']
                    assert record['overlapping_step_ms'] == sum(max(0, min(s['end'], end['end']) - max(s['begin'], end['begin'])) for s in steps) / 1e6
                for row, client in zip(rows, saved['clients']):
                    lo, hi = row['content_events'][-1]['perf_counter_ns'], round(row['finished_at'] * 1e9)
                    assert client['final_content_to_HTTP_finish_ms'] == (hi - lo) / 1e6
                    assert client['overlapping_EndRequest_indices'] == [i for i, end in enumerate(ends) if end['begin'] < hi and end['end'] > lo]
            else:
                row = rows[0]
                lo, hi = round(row['started_at'] * 1e9), round(row['finished_at'] * 1e9)
                pre = [s for s in head if s['id'] == 9 and lo <= s['begin'] and s['end'] <= hi]
                frames = [s for s in head if s['id'] == 3 and any(q['tid'] == s['tid'] and q['begin'] <= s['begin'] and s['end'] <= q['end'] for q in pre)]
                assert sum(s['entry_value'] for s in frames) == row['prompt_tokens']
                assert len(frames) == len(saved['first_request_forwards'])
                for frame, record in zip(frames, saved['first_request_forwards']):
                    assert frame == record['forward']
                    inside = [s for s in head if s['id'] in tags and s['tid'] == frame['tid'] and frame['begin'] <= s['begin'] and s['end'] <= frame['end']]
                    assert aggregate(inside, tags) == record['APIs']
                    covered = union([(s['begin'], s['end']) for s in inside])
                    assert covered == record['HIP_API_union_ms']
                    assert (frame['end'] - frame['begin']) / 1e6 == record['host_ms']
                    assert (frame['end'] - frame['begin'] - round(covered * 1e6)) / 1e6 == record['outside_probed_HIP_APIs_ms']
        if bundle == 'startup-hip':
            raw_csv = list(csv.DictReader(io.StringIO(archive.read(base + 'hip-api-spans.csv').decode())))
            actual = [(r['phase'], int(r['rank']), r['api'], int(r['tid']), int(r['begin_ns']), int(r['end_ns']), float(r['host_ms']), int(r['return_code'])) for r in raw_csv]
            assert actual == csv_expected
assert count == 30
print('PASS:30 canonical diagnostic identities,12 trace audits,all15 EndRequest decompositions,all HIP API CSV rows and first-Forward unions match raw records.')
print('Scope: raw-data arithmetic/identity audit; no hardware requalification, GPU timeline, attribution guarantee or timing-flag waiver.')
