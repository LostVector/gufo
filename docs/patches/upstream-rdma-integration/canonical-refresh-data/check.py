"""Verify exported raw identities, paired measures and retained timing flags."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math
from paired_statistics import summarize, holm_primary

p = Path(__file__).resolve().parent
manifest = json.loads((p/'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == expected, name
raw = {}
for name, meta in manifest['raw'].items():
    data = gzip.decompress((p/name).read_bytes())
    assert hashlib.sha256(data).hexdigest() == meta['uncompressed_sha256'], name
    raw[name] = [json.loads(line) for line in data.splitlines()]
    assert len(raw[name]) == meta['records']
summary = json.loads((p/'statistical-summary.json').read_text())
protocol = json.loads((p/'protocol.json').read_text())
assert len(protocol['sequence']) == 20 and summary['pairs'] == 10
identity = ('case','repetition','request_index','concurrency','prompt_sha256',
            'prompt_tokens','prefill_tokens','cached_prompt_tokens',
            'completion_tokens','completion_sha256')
reference, pairs, count = None, {}, 0
for phase, role in protocol['sequence']:
    name = f'extensions-release-comparison-v1-{phase}.jsonl.gz'
    rows = raw[name]
    assert len(rows) == 12
    original_hash = next(h for n,h in summary['raw_inputs_sha256'].items()
                         if Path(n).name+'.gz' == name)
    assert original_hash == manifest['raw'][name]['uncompressed_sha256']
    ids = [{key: row[key] for key in identity} for row in rows]
    if reference is None: reference = ids
    assert ids == reference
    assert all(row['cached_prompt_tokens']==0 and
               row['prefill_tokens']==row['prompt_tokens'] and row['draft_tokens']>0
               for row in rows)
    pairs[phase.split('-')[0],role] = rows
    count += len(rows)
assert count == 240

def c1(rows, repetition):
    xs = [x for x in rows if x['case']=='pp2048-tg128' and x['repetition']==repetition]
    assert len(xs)==1
    return xs[0]

def wave(rows, case):
    xs = [x for x in rows if x['case']==case]
    assert len(xs)==2
    return sum(x['completion_tokens']for x in xs)/(max(x['finished_at']for x in xs)-min(x['started_at']for x in xs))

metrics = {n:dict(baseline=[],candidate=[])for n in
           ['C1-first-server-decode','C2-short-client-wave',
            'C1-repeat-server-decode','C2-coding-client-wave']}
changed = []
for i in range(1,11):
    pair = f'pair{i:02d}'
    a,b = [pairs[pair,role]for role in ['baseline','candidate']]
    for rep in [1,2]:
        x,y = c1(a,rep),c1(b,rep)
        assert all(x[k]==y[k]for k in ['draft_tokens','draft_accepted_tokens','physical_execution_width'])
        assert x['raw_gufo']['draft_rounds']==y['raw_gufo']['draft_rounds']
    for x,y in zip(a,b):
        if x['concurrency']==2:
            fields=[k for k in ['draft_tokens','draft_accepted_tokens','physical_execution_width']if x[k]!=y[k]]
            if x['raw_gufo']['draft_rounds']!=y['raw_gufo']['draft_rounds']: fields.append('draft_rounds')
            if fields: changed.append(dict(pair=pair,case=x['case'],actor=x['request_index'],fields=fields))
    for role,rows in [('baseline',a),('candidate',b)]:
        for n,v in [('C1-first-server-decode',c1(rows,1)['decode_tokens_per_second']),
                    ('C2-short-client-wave',wave(rows,'short-c2')),
                    ('C1-repeat-server-decode',c1(rows,2)['decode_tokens_per_second']),
                    ('C2-coding-client-wave',wave(rows,'coding-c2'))]: metrics[n][role].append(v)
assert changed==summary['c2_work_changes'] and len(changed)==32
for report in summary['metrics']:
    for role in ['baseline','candidate']: assert metrics[report['metric']][role]==report[role+'_values']
flags=list(csv.DictReader((p/'timing-flags.csv').open()))
assert len(flags)==52
assert len({(x['pair'],x['case'],x['repetition'],x['request_index'])for x in flags})==20
assert all(float(x['delta_ms'])>3 and float(x['candidate'])>1.05*float(x['baseline'])for x in flags)
assert all(x['case']in ['short-c2','coding-c2']for x in flags)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--statistics',action='store_true',help='Recompute frozen bootstrap and assignment sensitivity calculations')
if parser.parse_args().statistics:
    results=[]
    for i,report in enumerate(summary['metrics']):
        values=metrics[report['metric']]
        logs=[math.log(b/a)for a,b in zip(values['baseline'],values['candidate'])]
        result=summarize(logs,summary['observed_order'],202611011540+i)
        for key,value in result.items(): assert value==report[key],key
        results.append(result)
    holm_primary(results[:2])
    for result,report in zip(results[:2],summary['metrics'][:2]): assert result['holm_adjusted_p']==report['holm_adjusted_p']
print('PASS: exported hashes,240identities,exactC1work,32C2changes,pairedrates and all52flags/20requestcomparisons.')
print('Scope: checks exported data; does not rerun inference, certify hardware or establish equivalence.')
