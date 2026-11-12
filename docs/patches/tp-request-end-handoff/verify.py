"""Verify the compressed exploratory finish captures, without external packages."""
from pathlib import Path
import collections,csv,gzip,hashlib,json,math
out=Path(__file__).resolve().parent;data=out/'data';manifest=json.loads((data/'manifest.json').read_text());rows=[];reference=None
identity=['case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256']
for history in manifest['histories']:
 loaded={}
 for kind,entry in history['inputs'].items():
  raw=(data/entry['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['gzip_sha256'];decoded=gzip.decompress(raw);assert hashlib.sha256(decoded).hexdigest()==entry['sha256'];loaded[kind]=[json.loads(line)for line in decoded.decode().splitlines()]
 requests=loaded['requests'];assert len(requests)==12;ids=[{k:r[k]for k in identity}for r in requests]
 if reference is None:reference=ids
 assert ids==reference and all(r['draft_tokens']>0 and r['cached_prompt_tokens']==0 for r in requests)
 meta,*events=loaded['trace'];assert meta['meta']and meta['dropped']==0 and meta['records']==len(events);spans=collections.defaultdict(list)
 for e in events:assert e['tag']!='INCOMPLETE';spans[e['id']].append(e)
 for es in spans.values():es.sort(key=lambda e:e['ns'])
 ends={es[0]['b']:es for es in spans.values()if es[0]['tag']=='tp_end'};finishes={es[0]['a']:es for es in spans.values()if es[0]['tag']=='tp_request_finish'};assert set(ends)==set(range(1,13))
 for seq,request in enumerate(requests,1):
  end=ends[seq];finish=finishes[seq];lock=next(e for e in end if e['stage']=='call_lock');sent=next(e for e in finish if e['stage']=='end_sent');response=next(e for e in finish if e['stage']=='response_done');assert end[0]['ns']<=lock['ns']<=end[-1]['ns'];assert sent['ns']<=response['ns']
  row=dict(run=history['run'],role=history['role'],case=request['case'],actor=request['request_index'],sequence=seq,end_mutex_wait_ms=(lock['ns']-end[0]['ns'])/1e6,end_total_ms=(end[-1]['ns']-end[0]['ns'])/1e6,worker_response_ms=(response['ns']-sent['ns'])/1e6,request_finish_ms=(finish[-1]['ns']-finish[0]['ns'])/1e6,last_content_to_HTTP_finish_ms=(request['finished_at']-request['content_events'][-1]['perf_counter_ns']/1e9)*1e3)
  assert all(math.isfinite(v)and v>=0 for k,v in row.items()if k.endswith('_ms'));rows.append(row)
summary={}
for role in ['baseline','candidate']:
 summary[role]={}
 for case,actor in [('short-c2',1),('coding-c2',0)]:
  selected=[r for r in rows if r['role']==role and r['case']==case and r['actor']==actor];assert len(selected)==3
  summary[role][case]=dict(end_mutex_wait_ms=[r['end_mutex_wait_ms']for r in selected],worst_end_mutex_wait_ms=max(r['end_mutex_wait_ms']for r in selected),worker_response_ms=[r['worker_response_ms']for r in selected],last_content_to_HTTP_finish_ms=[r['last_content_to_HTTP_finish_ms']for r in selected])
with(out/'summary.csv').open('w')as file:
 writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
report=dict(all_72_request_identities_exact=True,six_histories_zero_trace_drops=True,summary=summary,scope='Exploratory per-request CPU wait attribution; no p-value or throughput gain claim. Local monotonic clocks only, no cross-host timestamp subtraction. Workers have no Finish/End hook executions; absent worker trace is expected.')
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
