#!/usr/bin/env python3
"""Offline raw-evidence audit; no GPU, cluster, preload, or original lab paths."""
from pathlib import Path
import collections,hashlib,json,math,re,statistics,sys,zipfile
p=Path(__file__).resolve().parent
meta=json.loads((p/'archive.json').read_text());archive=p/meta['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
z=zipfile.ZipFile(archive)
def raw(n):return z.read(n)
def obj(n):return json.loads(raw(n))
def rows(n):return [json.loads(s)for s in raw(n).decode().splitlines()if s.strip()]
for n,h in obj('index.json').items():
 b=raw(n);assert len(b)==h['bytes']and hashlib.sha256(b).hexdigest()==h['sha256'],n
names=('early-end-release-comparison-v2','early-end-wider-concurrency-v1','early-end-live-faults-v1')
identity=('case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256')
def tail(r):return(round(r['finished_at']*1e9)-r['content_events'][-1]['perf_counter_ns'])/1e6
def healthy(rs):assert all(r['draft_tokens']>0 and r['cached_prompt_tokens']==0 and r['prefill_tokens']==r['prompt_tokens']for r in rs)
allrows={};windows=0
for name in names:
 proto=obj(name+'/protocol.json');assert not obj(name+'/cleanup.json')['errors']
 for n,h in proto['harness_sha256'].items():assert hashlib.sha256(raw(name+'/'+n)).hexdigest()==h,(name,n)
 for rank in (0,1):
  inv=obj(f'{name}/final-inventory-r{rank}.json');assert inv['refcnt']==0 and inv['module']=='81C6FF2FBAC473C7AB21844'and len(inv['devices'])==4
  assert all(inv[k]==proto['initial'][rank][k]for k in('boot_id','cmdline','routers'))
  assert not re.search(r'BUG:|Oops:|kernel panic|soft lockup|hard LOCKUP|general protection fault|frame_length_invalid|RING_DESC_CRC|reorder timeout|amdgpu.*reset|segfault',raw(f'{name}/kernel-window-r{rank}.txt').decode(),re.I)
 for phase,role in proto['sequence']:
  rs=rows(f'raw/{name}-{phase}.jsonl');allrows[name,phase]=rs
  for rank in(0,1):
   ident=obj(f'{name}/{name}-{phase}-identity-r{rank}.json');assert ident['binary_sha256']==proto['binaries'][role if name!=names[2]else'candidate']['sha256']and ident['read_observer_absent']and ident['SDK_trace_absent']
   for k,v in{'--context':'65536','--sessions':'4'if name==names[1]else'2','--prefill-chunk':'2048','--cache-ram-bytes':'1073741824','--speculative':'mtp','--tp-world-size':'2','--tp-rank':str(rank),'--think':'off'}.items():assert ident['selected_safe_options'][k]==v
   def counters(tag):return{k:int(v)for k,v in re.findall(r'^((?:data|dmabuf)_\w+): (\d+)$',raw(f'{name}/{phase}-{tag}-r{rank}.txt').decode().split('\nFILE ',1)[0],re.M)}
   a,b=counters('before'),counters('after');assert a and a.keys()==b.keys()and all(b[k]>=a[k]for k in a)
   error={k:b[k]-a[k]for k in a if b[k]!=a[k]and re.search('error|retransmit|retry_exhausted|timeout|lost|drop|fail|bad_frame|bad_header|poison|overrun',k)}
   if phase!='owner-loss':assert not error,error
   windows+=1
name=names[0];proto=obj(name+'/protocol.json');report=obj(name+'/statistical-summary.json');canonical=None;pairs={}
for phase,role in proto['sequence']:
 rs=allrows[name,phase];assert len(rs)==5;healthy(rs);ids=[{k:r[k]for k in identity}for r in rs]
 if canonical is None:canonical=ids
 assert canonical==ids
 pairs[phase.split('-')[0],role]=rs
 # Verify the original raw-file digest against the committed report.
 original='/home/rkuo/strix-dispatch-investigation/results/model-scaling/'+name+'-'+phase+'.jsonl'
 assert hashlib.sha256(raw('raw/'+name+'-'+phase+'.jsonl')).hexdigest()==report['raw_inputs_sha256'][original]
sys.path.insert(0,str(archive)+'/'+name)
from paired_statistics import summarize
values={'baseline':[],'candidate':[]};flags=[];work=[];order=[]
for i in range(1,11):
 pair=f'pair{i:02d}';a,b=[pairs[pair,r]for r in('baseline','candidate')]
 for role,rs in [('baseline',a),('candidate',b)]:
  group={r['request_index']:r for r in rs if r['case']=='short-c2'};values[role].append(tail(group[1]));order.append(dict(pair=pair,role=role,actor1_finishes_first=group[1]['finished_at']<group[0]['finished_at']))
 for x,y in zip(a,b):
  ident=dict(pair=pair,case=x['case'],repetition=x['repetition'],actor=x['request_index'])
  for k in('draft_tokens','draft_accepted_tokens','physical_execution_width'):
   if x[k]!=y[k]:work.append(ident|dict(metric=k,baseline=x[k],candidate=y[k]))
  if x['raw_gufo']['draft_rounds']!=y['raw_gufo']['draft_rounds']:work.append(ident|dict(metric='raw.draft_rounds',baseline=x['raw_gufo']['draft_rounds'],candidate=y['raw_gufo']['draft_rounds']))
  times={k:(v,y[k])for k,v in x.items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(y.get(k),(int,float))}
  times.update({'raw.'+k:(v,y['raw_gufo'][k])for k,v in x['raw_gufo'].items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(y['raw_gufo'].get(k),(int,float))})
  times['final_content_to_HTTP_finish_ms']=tail(x),tail(y)
  for k,(av,bv)in times.items():
   assert math.isfinite(av)and math.isfinite(bv)
   if bv-av>3 and bv>1.05*av:flags.append(ident|dict(metric=k,baseline_ms=av,candidate_ms=bv,delta_ms=bv-av))
assert flags==report['strict_flags']and work==report['adaptive_work_fields']and order==report['completion_order']
assert values==report['primary']['values_ms']
s=summarize([math.log(a/b)for a,b in zip(values['baseline'],values['candidate'])],proto['design']['observed_order'],2026110203)
assert s['duration_reduction_pct']==report['primary']['tail_reduction_pct']and s['balanced_assignment_sensitivity_p']==report['primary']['balanced_assignment_sensitivity_p']
assert [100*v/(100+v)for v in s['marginal_bootstrap95_throughput_increase_pct']]==report['primary']['marginal_bootstrap95_tail_reduction_pct']
assert all(math.exp(statistics.mean(map(math.log,v)))==report['primary']['geometric_ms'][k]for k,v in values.items())
name=names[1];canonical=None
for phase,role in obj(name+'/protocol.json')['sequence']:
 rs=sorted(allrows[name,phase],key=lambda r:r['order']);assert len(rs)==20;healthy(rs);ids=[{k:r[k]for k in identity}for r in rs]
 if canonical is None:canonical=ids
 assert ids==canonical
 controls={(r['request_index'],r['max_tokens']):r for r in rs if r['case']in('solo','solo-ragged')};assert len(controls)==7
 for r in rs:
  if r['case']=='warmup':continue
  other=controls[r['request_index'],r['max_tokens']]
  assert all(r[k]==other[k]for k in('prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256'))
 for case,width in [('C3',3),('C4',4),('ragged-C4',4)]:
  group=sorted([r for r in rs if r['case']==case],key=lambda r:r['request_index']);assert len(group)==width
  assert [r['raw_gufo']['resident_requests_at_admission']for r in group]==list(range(1,width+1))and max(r['physical_execution_width']for r in group)>=width
  if role=='candidate':assert all(0<tail(r)<1 for r in group)
name=names[2];reference=None
for phase,script in obj(name+'/protocol.json')['sequence']:
 rs=allrows[name,phase]
 if phase.startswith('owner-'):
  assert len(rs)==1;r=rs[0];assert r['explicit_stream_error']and not r['normal_completion_reported']and r['failed_pair_exited']and r['content_pieces_observed_at_least']>=8
  assert all(not x['alive']for x in r['pair_exit']['states'])and r['pair_exit']['new_connection_rejected']
  events=obj(name+('/read-stream-events.json'if phase=='owner-readerror'else'/loss-stream-events.json'));assert any(e.get('error')for e in events)
  assert not any(c.get('finish_reason')is not None for e in events for c in e.get('choices',[]))
  if phase=='owner-readerror':
   fault=obj(name+'/injected.jsonl')
   assert fault==r['injected_fault']and fault['pid']==r['owner']['pid']and fault['errno']==5 and fault['result']==-1 and fault['o_direct']
 else:
  assert len(rs)==4;healthy(rs);ids=[{k:r[k]for k in identity}for r in rs]
  if reference is None:reference=ids
  assert ids==reference
assert windows==56
# Independently pair every diagnostic span and recompute all End/Send splits.
tm=json.loads((p/'dispatch-trace.json').read_text());tz=zipfile.ZipFile(p/tm['archive']);assert hashlib.sha256((p/tm['archive']).read_bytes()).hexdigest()==tm['sha256']
for n,h in json.loads(tz.read('index.json')).items():
 b=tz.read(n);assert len(b)==h['bytes']and hashlib.sha256(b).hexdigest()==h['sha256']
tp=json.loads(tz.read('protocol.json'));tr=json.loads(tz.read('end-analysis.json'));tc=None;ends_checked=0
for phase,role in tp['sequence']:
 for n,h in tp['harness_sha256'].items():assert hashlib.sha256(tz.read(n)).hexdigest()==h
 rr=[json.loads(line)for line in tz.read('raw/early-end-dispatch-trace-v2-'+phase+'.jsonl').decode().splitlines()];assert len(rr)==5;healthy(rr)
 ids=[{k:r[k]for k in identity}for r in rr]
 if tc is None:tc=ids
 assert tc==ids
 spans_by_rank=[]
 for rank in(0,1):
  enters={};spans=[]
  for line in tz.read(f'{phase}/trace-r{rank}/app.csv').decode().splitlines():
   if not line.startswith('APP,'):continue
   x=dict(zip(('kind','end','begin','tid','id','depth','cpu','pid','value'),map(int,line.split(',')[1:])));key=tuple(x[k]for k in('tid','id','depth','begin'))
   if x['kind']==1:assert key not in enters;enters[key]=x
   else:
    assert x['kind']==2 and key in enters;x['entry_value']=enters[key]['value'];spans.append(x)
  assert len(enters)==len(spans)==len({tuple(x[k]for k in('tid','id','depth','begin'))for x in spans})
  spans_by_rank.append(sorted(spans,key=lambda x:x['begin']))
 ss=spans_by_rank[0];ends=[x for x in ss if x['id']==4];assert len(ends)==5
 recorded=next(x for x in tr['phases']if x['phase']==phase)
 for end,saved in zip(ends,recorded['EndRequests']):
  sends=[x for x in ss if x['id']==7 and x['tid']==end['tid']and end['begin']<=x['begin']and x['end']<=end['end']];assert len(sends)==1;send=sends[0]
  assert end==saved['end']and send==saved['send']
  for key,value in {'pre_send_ms':(send['begin']-end['begin'])/1e6,'send_ms':(send['end']-send['begin'])/1e6,'post_send_ms':(end['end']-send['end'])/1e6,'total_ms':(end['end']-end['begin'])/1e6}.items():assert value==saved[key]
  ends_checked+=1
assert ends_checked==15
print('PASS: indexed archives;100 matched primary identities/statistics/all65 flags/37 work fields;60 wider controls;12 healthy fault controls/2 explicit failures;56 audited rank-windows;15 diagnostic outputs/all15 End-Send decompositions. Source/compiler/CI provenance recorded separately, not reproduced by this offline audit.')
