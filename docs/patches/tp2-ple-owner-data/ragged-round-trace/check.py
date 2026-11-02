"""Audit final round attribution directly from exported raw traces; no GPU needed."""
from pathlib import Path
import collections,csv,hashlib,io,json,math,zipfile
p=Path(__file__).resolve().parent;study='tp-cache-retention/ple-owner-ragged-round-trace-v2/'
report=json.loads((p/'round-analysis.json').read_text());validation=json.loads((p/'validation.json').read_text());peer_report=json.loads((p/'peer-work-check.json').read_text());recorded=list(csv.DictReader((p/'rounds.csv').open()));checked=0
with zipfile.ZipFile(p/'recorder-v5.zip')as z:
 base='tp-cache-retention/app-span-recorder-v5/';build=json.loads(z.read(base+'build-info.json'))
 for n,h in build['files_sha256'].items():assert hashlib.sha256(z.read(base+n)).hexdigest()==h,n
 for rank,h in build['selftest_validation_sha256'].items():assert hashlib.sha256(z.read(base+f'selftest-r{rank}/validation.json')).hexdigest()==h
 assert build['both_host_selftests_pass']and all(x['offset']==304 and x['drafted_offset']==312 and x['accepted_offset']==320 for x in build['layouts'].values())

def parse(text):
 starts={};spans={};details=[]
 for line in text.splitlines():
  if line.startswith('APP,'):
   x=dict(zip(('kind','end','begin','tid','id','depth','cpu','pid','value'),map(int,line.split(',')[1:])));key=tuple(x[k]for k in('tid','id','depth','begin'))
   if x['kind']==1:assert key not in starts;starts[key]=x
   else:assert x['kind']==2 and key in starts and key not in spans;x['entry_value']=starts[key]['value'];spans[key]=x
  elif line.startswith('DETAIL,'):details.append(dict(zip(('kind','ns','begin','tid','id','depth','cpu','pid','object','cycles','drafted','accepted'),map(int,line.split(',')[1:]))))
 assert starts.keys()==spans.keys();return list(spans.values()),spans,details

def inside(a,b):return a['tid']==b['tid']and a['begin']<=b['begin']and b.get('end',b['begin'])<=a['end']

raw={};reference=None
with zipfile.ZipFile(p/'acquisition.zip')as z:
 proto=json.loads(z.read(study+'protocol.json'))
 for n,h in proto['harness_sha256'].items():assert hashlib.sha256(z.read(study+n)).hexdigest()==h,n
 for phase,role in proto['sequence']:
  rows=sorted(map(json.loads,z.read(f'results/model-scaling/ple-owner-ragged-round-trace-v2-{phase}.jsonl').splitlines()),key=lambda x:x['order']);assert len(rows)==20;raw[phase]=rows
  keys=('case','order','max_tokens','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256');ids=[{k:x[k]for k in keys}for x in rows]
  if reference is None:reference=ids
  assert ids==reference and all(x['draft_tokens']>0 and x['cached_prompt_tokens']==0 and x['prefill_tokens']==x['prompt_tokens']for x in rows)
  ragged=sorted([x for x in rows if x['case']=='ragged-C4'],key=lambda x:x['request_index']);analysis=next(x for x in report['phases']if x['phase']==phase);head_totals=[]
  for rank in (0,1):
   prefix=study+phase+f'/trace-r{rank}/';stats=json.loads(z.read(prefix+'stats.json'));c=stats['counters'];assert not any(c[:6])and c[6]==c[7]==c[30]==c[46]>0 and c[8]==c[24]==c[40]>0
   assert stats['remaining_span_keys']==stats['remaining_depth_keys']==stats['remaining_object_keys']==0 and not any(stats['program_recursion_misses'])
   spans,bykey,details=parse(z.read(prefix+'app.csv').decode());before={};after={};mapping={};prefill=sorted([x for x in spans if x['id']==9],key=lambda x:x['begin']);ptrs=list(dict.fromkeys(x['entry_value']for x in prefill))
   if rank==0:assert len(ptrs)==4 and all(ptrs)
   for x in details:
    if x['kind']==5:
     if rank==0:
      parent=[a for a in prefill if inside(a,x)];assert len(parent)==1;ptr=parent[0]['entry_value'];assert mapping.setdefault(x['object'],ptr)==ptr
     continue
    assert x['id']==14 and x['kind']in(3,4);key=tuple(x[k]for k in('tid','id','depth','begin'));target=before if x['kind']==3 else after;assert key in bykey and key not in target;target[key]=x
   assert before.keys()==after.keys();totals=collections.defaultdict(lambda:[0,0,0,0]);cycles=[]
   for key,a in before.items():
    b=after[key];span=bykey[key];assert a['object']==b['object']and span['value']&255==1;delta=[b[k]-a[k]for k in('cycles','drafted','accepted')];assert delta[0]in(0,1)and 0<=delta[2]<=delta[1]
    for i,value in enumerate(delta):totals[a['object']][i]+=value
    totals[a['object']][3]+=delta[0]==0
    if rank==0:
     ptr=mapping[a['object']];actor=ptrs.index(ptr);matches=[x for x in recorded if x['phase']==phase and int(x['session'])==a['object']and int(x['begin_ns'])==span['begin']and int(x['end_ns'])==span['end']];assert len(matches)==1;row=matches[0];assert int(row['actor'])==actor and int(row['request'])==ptr
     assert [int(row[k])for k in('cycles','drafted','accepted')]==delta
     batch=[x for x in spans if x['id']==12 and inside(x,span)];scalar=[x for x in spans if x['id']==10 and inside(x,span)];assert len(batch)<=1 and len(scalar)<=1 and(batch or scalar)
     assert row['batch_begin']==(str(batch[0]['begin'])if batch else'')and row['scalar_begin']==(str(scalar[0]['begin'])if scalar else'')
     cycles.append(dict(actor=actor,batch_begin=batch[0]['begin']if batch else None));checked+=1
   if rank==0:
    assert len(mapping)==4 and set(mapping.values())==set(ptrs)
    for actor,row in enumerate(ragged):
     ptr=ptrs[actor];obj=next(s for s,q in mapping.items()if q==ptr);actual=totals[obj];assert actual[:3]==[row['raw_gufo']['draft_rounds'],row['draft_tokens'],row['draft_accepted_tokens']]
     pub=sorted([x for x in spans if x['id']==11 and x['entry_value']==ptr],key=lambda x:x['begin']);assert len(pub)==row['completion_tokens']==actual[0]+actual[2]+actual[3];head_totals.append(tuple(actual[:3]))
     expected=analysis['actors'][actor];assert [expected[k]for k in('cycles','drafted','accepted','plain_completion_calls')]==actual
     lo,hi=pub[0]['begin'],pub[-1]['begin'];segments=[]
     batches=[x for x in spans if x['id']==12]
     for x in batches:
      members={a['actor']for a in cycles if a['batch_begin']==x['begin']};assert members;segments.append((x['begin'],x['end'],'batch_including_request'if actor in members else'batch_other_requests'))
     for x in spans:
      if x['id']==10 and not any(inside(a,x)for a in batches):segments.append((x['begin'],x['end'],'scalar_this_request'if x['entry_value']==ptr else'scalar_other_requests'))
      if x['id']==9:segments.append((x['begin'],x['end'],'prefill'))
     clips=sorted((max(a,lo),min(b,hi),tag)for a,b,tag in segments if a<hi and b>lo);assert all(a[1]<=b[0]for a,b in zip(clips,clips[1:]));parts=collections.defaultdict(float)
     for a,b,tag in clips:parts[tag]+=(b-a)/1e6
     parts['unclassified_host_interval']=(hi-lo)/1e6-sum(parts.values());assert abs(expected['publication_span_ms']-(hi-lo)/1e6)<1e-9
     assert parts.keys()==expected['host_partition_ms'].keys()and all(abs(v-expected['host_partition_ms'][k])<1e-8 for k,v in parts.items())
   else:
    actual=sorted(tuple(v[:3])for v in totals.values());assert actual==sorted(head_totals)
    saved=next(x for x in peer_report if x['phase']==phase);assert saved['same_multiset']and actual==list(map(tuple,saved['peer_counter_triples']))
assert checked==len(recorded)
flags=[];work=[]
for control in('before-baseline','after-baseline'):
 for a,b in zip(raw[control],raw['first-candidate']):
  ident=dict(control=control,case=a['case'],actor=a['request_index'],order=a['order'])
  for k in('draft_tokens','draft_accepted_tokens','physical_execution_width'):
   if a[k]!=b[k]:work.append(ident|dict(metric=k,baseline=a[k],candidate=b[k]))
  if a['raw_gufo']['draft_rounds']!=b['raw_gufo']['draft_rounds']:work.append(ident|dict(metric='raw.draft_rounds',baseline=a['raw_gufo']['draft_rounds'],candidate=b['raw_gufo']['draft_rounds']))
  values={k:(v,b[k])for k,v in a.items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(b.get(k),(int,float))};values.update({'raw.'+k:(v,b['raw_gufo'][k])for k,v in a['raw_gufo'].items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(b['raw_gufo'].get(k),(int,float))})
  for k,(av,bv)in values.items():
   assert math.isfinite(av)and math.isfinite(bv)
   if bv-av>3 and bv>1.05*av:flags.append(ident|dict(metric=k,baseline_ms=av,candidate_ms=bv,delta_ms=bv-av))
assert flags==validation['strict_phase_flags']and len(flags)==6 and work==validation['adaptive_work_fields']and len(work)==26
print(f'PASS:60 final-trace identities,{checked} head completions,peer counter multisets,raw host partitions and all6flags/26work fields; recorder hashes/two-host CPU sensitivity controls pinned. No hardware or timing-equivalence certification.')
