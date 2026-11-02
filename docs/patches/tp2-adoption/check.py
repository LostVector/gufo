#!/usr/bin/env python3
"""Offline archive, output, work, transport and owner-role consistency audit."""
from pathlib import Path
import hashlib,json,math,re,zipfile
p=Path(__file__).resolve().parent
meta=json.loads((p/'archive.json').read_text());archive=p/meta['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
z=zipfile.ZipFile(archive)
def obj(name):return json.loads(z.read(name))
def rows(name):return [json.loads(s)for s in z.read(name).decode().splitlines()if s.strip()]
for name,h in obj('index.json').items():
 b=z.read(name);assert len(b)==h['bytes']and hashlib.sha256(b).hexdigest()==h['sha256'],name
studies=['combined-ple-early-end-http-smoke-v1','combined-ple-early-end-http-lifecycle-v1','combined-ple-early-end-wider-v1','combined-ple-early-end-live-faults-v2']
identity=('case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256')
def ids(rs):return [{k:r[k]for k in identity}for r in rs]
def healthy(rs):assert all(r['draft_tokens']>0 and r['cached_prompt_tokens']==0 and r['prefill_tokens']==r['prompt_tokens']for r in rs)
windows=0
for name in studies:
 proto=obj(name+'/protocol.json');assert not obj(name+'/cleanup.json')['errors']
 for n,h in proto['harness_sha256'].items():assert hashlib.sha256(z.read(name+'/'+n)).hexdigest()==h,(name,n)
 for rank in(0,1):
  inv=obj(f'{name}/final-inventory-r{rank}.json');assert inv['refcnt']==0 and inv['module']=='81C6FF2FBAC473C7AB21844'and len(inv['devices'])==4
  assert all(inv[k]==proto['initial'][rank][k]for k in('boot_id','cmdline','routers'))
  assert not re.search(r'BUG:|Oops:|kernel panic|soft lockup|hard LOCKUP|general protection fault|frame_length_invalid|RING_DESC_CRC|reorder timeout|amdgpu.*reset|segfault',z.read(f'{name}/kernel-window-r{rank}.txt').decode(),re.I)
 for phase,role in proto['sequence']:
  role=role if role in proto['binaries']else'candidate'
  for rank in(0,1):
   ident=obj(f'{name}/{name}-{phase}-identity-r{rank}.json');assert ident['binary_sha256']==proto['binaries'][role]['sha256']
   for k,v in{'--context':'65536','--sessions':'4'if name==studies[2]else'2','--prefill-chunk':'2048','--cache-ram-bytes':'1073741824','--speculative':'mtp','--tp-world-size':'2','--tp-rank':str(rank),'--think':'off'}.items():assert ident['selected_safe_options'][k]==v
   def counters(tag):return{k:int(v)for k,v in re.findall(r'^((?:data|dmabuf)_\w+): (\d+)$',z.read(f'{name}/{phase}-{tag}-r{rank}.txt').decode().split('\nFILE ',1)[0],re.M)}
   a,b=counters('before'),counters('after');assert a and a.keys()==b.keys()and all(b[k]>=a[k]for k in a)
   errors={k:b[k]-a[k]for k in a if b[k]!=a[k]and re.search('error|retransmit|retry_exhausted|timeout|lost|drop|fail|bad_frame|bad_header|poison|overrun',k)}
   if phase!='owner-loss':assert not errors,errors
   windows+=1
assert windows==20
name=studies[0];rs=rows('raw/'+name+'-candidate.jsonl');assert len(rs)==12;healthy(rs)
for reference in obj(name+'/reference.json').values():
 path='references/'+Path(reference['path']).name;assert hashlib.sha256(z.read(path)).hexdigest()==reference['sha256'];assert ids(rs)==ids(rows(path))
name=studies[1];lifecycle=rows('raw/'+name+'-candidate.jsonl');assert len(lifecycle)==13
v=obj(name+'/validation.json');assert v['cache_cases']and v['two_C2_victim_roles_and_live_survivors_width2_exact_solo_and_postcancel_replay']and v['completed_substantive_requests_actual_MTP']
name=studies[2];raw={};canonical=None
for phase,_ in obj(name+'/protocol.json')['sequence']:
 rs=sorted(rows('raw/'+name+'-'+phase+'.jsonl'),key=lambda r:r['order']);assert len(rs)==20;healthy(rs);raw[phase]=rs
 if canonical is None:canonical=ids(rs)
 assert ids(rs)==canonical
 controls={(r['request_index'],r['max_tokens']):r for r in rs if r['case']in('solo','solo-ragged')};assert len(controls)==7
 for r in rs:
  if r['case']=='warmup':continue
  other=controls[r['request_index'],r['max_tokens']]
  assert all(r[k]==other[k]for k in('prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256'))
flags=[];work=[]
for control in('before-baseline','after-baseline'):
 for a,b in zip(raw[control],raw['first-candidate']):
  ident=dict(control=control,case=a['case'],actor=a['request_index'],order=a['order'])
  for k in('draft_tokens','draft_accepted_tokens','physical_execution_width'):
   if a[k]!=b[k]:work.append(ident|dict(metric=k,baseline=a[k],candidate=b[k]))
  if a['raw_gufo']['draft_rounds']!=b['raw_gufo']['draft_rounds']:work.append(ident|dict(metric='raw.draft_rounds',baseline=a['raw_gufo']['draft_rounds'],candidate=b['raw_gufo']['draft_rounds']))
  times={k:(v,b[k])for k,v in a.items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(b.get(k),(int,float))}
  times['final_content_to_HTTP_finish_ms']=tuple((round(r['finished_at']*1e9)-r['content_events'][-1]['perf_counter_ns'])/1e6 for r in(a,b))
  times.update({'raw.'+k:(v,b['raw_gufo'][k])for k,v in a['raw_gufo'].items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(b['raw_gufo'].get(k),(int,float))})
  for k,(av,bv)in times.items():
   assert math.isfinite(av)and math.isfinite(bv)
   if bv-av>3 and bv>1.05*av:flags.append(ident|dict(metric=k,baseline_ms=av,candidate_ms=bv,delta_ms=bv-av))
v=obj(name+'/validation.json');assert flags==v['strict_phase_flags']and work==v['adaptive_work_fields']and len(flags)==12 and len(work)==26
name=studies[3];reference=None
for phase,_ in obj(name+'/protocol.json')['sequence']:
 rs=rows('raw/'+name+'-'+phase+'.jsonl')
 if phase.startswith('owner-'):
  assert len(rs)==1;r=rs[0];assert r['explicit_stream_error']and not r['normal_completion_reported']and r['failed_pair_exited']and all(not s['alive']for s in r['pair_exit']['states'])and r['pair_exit']['new_connection_rejected']
  if phase=='owner-readerror':
   fault=obj(name+'/injected.jsonl');assert fault==r['injected_fault']and fault['errno']==5 and fault['o_direct']and fault['pid']==r['owner']['pid']
 else:
  assert len(rs)==4;healthy(rs)
  if reference is None:reference=ids(rs)
  assert reference==ids(rs)
for arm in obj('provenance/combined-ple-activity.json')['healthy_arms']:
 assert arm['rank0_owner_rows']==arm['rank1_received_rows']==0 and arm['rank0_received_rows']==arm['rank1_owner_rows']>0
 matches=[x for x in obj('provenance/owner-lines.json')if x['label']==arm['label']];assert len(matches)==2
 for line in matches:
  rank,owner,remote=map(int,re.fullmatch(r'cooperative_ple rank=(\d+) owner_rows=(\d+) remote_rows=(\d+)',line['line']).groups());assert owner==arm[f'rank{rank}_owner_rows']and remote==arm[f'rank{rank}_received_rows']
if (p/'adoption.json').exists():
 adoption=json.loads((p/'adoption.json').read_text());assert adoption['status']=='merged'and adoption['qualified_tree']==meta['qualified_tree']
 assert all(r['status']=='completed'and r['conclusion']=='success'for r in adoption['CI'].values())
print('PASS: indexed archive;12 smoke references/13 lifecycle case records;60 exact wider controls/all12 phase flags/26 work fields;12 healthy fault controls/2 explicit failures;20 transport windows;6 matching owner-role arms. Lifecycle semantics/source/compiler/historical hardware attribution remain recorded live-verifier evidence, not new inference or a performance claim.')
