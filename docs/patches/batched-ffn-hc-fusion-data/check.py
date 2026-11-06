"""Offline integrity/recomputation; never substitutes for real GPU/live-host gates."""
from pathlib import Path,PurePosixPath
import collections,csv,hashlib,io,itertools,json,math,random,zipfile
out=Path(__file__).resolve().parent;meta=json.loads((out/'archive.json').read_text());blob=(out/'acquisition.zip').read_bytes();assert hashlib.sha256(blob).hexdigest()==meta['zip_sha256']
with zipfile.ZipFile(io.BytesIO(blob))as z:
 manifest=json.loads(z.read('manifest.json'));assert len(manifest)==meta['entries']and set(z.namelist())==set(manifest)|{'manifest.json'}
 for name,entry in manifest.items():
  path=PurePosixPath(name);assert not path.is_absolute()and '..'not in path.parts
  value=z.read(name);assert len(value)==entry['bytes']and hashlib.sha256(value).hexdigest()==entry['sha256'],name
 def read(name):return json.loads(z.read('tp-cache-retention/'+name))
 family='batched-ffn-hc-release-study-v1';stats=read(family+'/statistics.json');validation=read(family+'/validation.json');protocol=read(family+'/protocol.json');assert validation['functional_qualification_pass']and validation['canonical_records']==624 and len(protocol['sequence'])==24
 expected=None;raw={};identities=('case','order','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256')
 for path,digest in stats['input_sha256'].items():
  data=z.read('results/model-scaling/'+Path(path).name);assert hashlib.sha256(data).hexdigest()==digest
  rows=sorted([json.loads(line)for line in data.splitlines()],key=lambda x:x['order']);assert len(rows)==26 and[x['order']for x in rows]==list(range(26));current=[{k:r[k]for k in identities}for r in rows]
  if expected is None:expected=current
  assert current==expected
  assert all(r['draft_tokens']>0 and r['cached_prompt_tokens']==0 and r['prompt_tokens']==r['prefill_tokens']for r in rows)
  assert all(r['prompt_tokens']==2048 for r in rows if r['case']=='pp2048-tg128')
  raw[Path(path).stem.removeprefix(family+'-')]=rows
 assert len(raw)==24 and sum(map(len,raw.values()))==624
 flags=list(csv.DictReader(io.StringIO(z.read('tp-cache-retention/'+family+'/phase-flags.csv').decode())));work=list(csv.DictReader(io.StringIO(z.read('tp-cache-retention/'+family+'/adaptive-work.csv').decode())));assert len(flags)==228 and len({(f['control'],f['order'])for f in flags})==64 and len(work)==291
 assert len(validation['strict_phase_flags'])==228 and len(validation['adaptive_work_fields'])==291
 for row in flags:assert float(row['candidate_ms'])>1.05*float(row['baseline_ms'])and float(row['delta_ms'])>3
 means=[]
 for primary,width in zip(stats['primary_results'],[2,4,8]):
  assert primary['co_primary']==f'C{width}-client-wave';ratios=[]
  for pair in range(1,13):
   rates={}
   for arm in ['baseline','candidate']:
    group=[x for x in raw[f'pair{pair:02d}-{arm}']if x['case']==f'C{width}'];assert len(group)==width
    duration=max(x['finished_at']for x in group)-min(x['started_at']for x in group);rates[arm]=sum(x['completion_tokens']for x in group)/duration
   ratios.append(math.log(rates['candidate']/rates['baseline']))
  assert all(abs(a-b)<1e-12 for a,b in zip(ratios,primary['paired_log_ratios']))
  mean=sum(ratios)/12;assert abs(100*math.expm1(mean)-primary['paired_geometric_throughput_change_pct'])<1e-10
  null=[sum(x*s for x,s in zip(ratios,signs))/12 for signs in itertools.product([-1,1],repeat=12)];pv=sum(abs(x)>=abs(mean)-1e-12 for x in null)/4096;assert pv==primary['signflip_p']
  rng=random.Random(202611021601+width);boot=sorted(sum(rng.choices(ratios,k=12))/12 for _ in range(20000));ci=[100*math.expm1(boot[i])for i in[500,19499]];assert all(abs(a-b)<1e-10 for a,b in zip(ci,primary['marginal_bootstrap95_change_pct']));assert ci[0]<=0<=ci[1];means.append((primary,pv))
 previous=0
 for i,(primary,pv)in enumerate(sorted(means,key=lambda x:x[1])):
  previous=max(previous,(3-i)*pv);assert primary['holm3_signflip_p']==min(1.,previous)and not primary['confirmatory_gain']
 state=read('batched-ffn-hc-state-v1/validation.json');wide=read('batched-ffn-hc-wide-state-v1/validation.json');assert(state['records_per_rank'],state['complete_postreplay_state_pairs_per_rank'])==(104,32);assert(wide['records_per_rank'],wide['complete_materialized_state_pairs_per_rank'])==(1992,498)
 for name in ['batched-ffn-hc-http-smoke-v1','batched-ffn-hc-c8-http-v1']:assert read(name+'/validation.json')
 life=read('batched-ffn-hc-http-lifecycle-v1/validation.json');assert len(life['cache_cases'])==4 and life['two_C2_victim_roles_and_live_survivors_width2_exact_solo_and_postcancel_replay']
 assert read('batched-hc-fusion-contract-v1/result.json')['exit']==0 and read('batched-ffn-hc-review-operator-v1/result.json')['exit']==0
 log=z.read('tp-cache-retention/batched-ffn-hc-review-operator-v1/run.log').decode();assert'HC mixed outputs and short prefill chunks passed'in log and'SKIP'not in log.upper()
 fixture=z.read('review/hc_mix_ops_test.cpp').decode();assert'MixerReference::Batch'in fixture and'CheckFusedProjection(2049)'in fixture and'CheckFusedInjectPartials(n)'in fixture
 assert read('batched-ffn-hc-release-v1/compile-comparison.json')['changed_normalized_commands']==[]
 comparison=read('batched-ffn-hc-cycle-gpu-build-v1/comparison.json');assert comparison['exact_ordered_work_comparisons']==8
 trace_manifest=json.loads(z.read('scoped-traces/full-trace-manifest.json'));observed={}
 def ns(row):return int(row['Start_Timestamp']),int(row['End_Timestamp'])
 def union(spans):
  total=0;last=None
  for a,b in sorted(spans):
   if b<=a:continue
   if last is None:last=[a,b]
   elif a<=last[1]:last[1]=max(last[1],b)
   else:total+=last[1]-last[0];last=[a,b]
  return total+(last[1]-last[0]if last else 0)
 for arm in ['baseline','candidate']:
  lab=f'batched-ffn-hc-gpu-{arm}-v1';gpu=read(lab+'/gpu-attribution.json');cycles=read(lab+'/cycles.json')['phases'][0]['ranks']
  for rank in[0,1]:
   base=f'scoped-traces/{arm}/r{rank}/';kernels=list(csv.DictReader(io.StringIO(z.read(base+'trace_kernel_trace.csv').decode())));copies=list(csv.DictReader(io.StringIO(z.read(base+'trace_memory_copy_trace.csv').decode())))
   m=trace_manifest[base+'trace_kernel_trace.csv'];assert len(kernels)==m['selected_record_count']and m['parent_record_count']>=len(kernels)
   for window,reported in zip(m['windows'],gpu['ranks'][rank]['windows']):
    a,b=window['begin_sdk_ns'],window['end_sdk_ns'];ks=[k for k in kernels if ns(k)[0]<b and ns(k)[1]>a];cs=[c for c in copies if ns(c)[0]<b and ns(c)[1]>a];assert len(ks)==reported['GPU_kernel_dispatches']
    spans=[(max(a,ns(k)[0]),min(b,ns(k)[1]))for k in ks+cs];assert abs(union(spans)/1e6-reported['GPU_kernel_or_copy_union_ms'])<1e-9
    count=sum('HcMixQ8FusedKernel<'in k['Kernel_Name']for k in ks);assert count==(48 if arm=='candidate'and window['rows']<=8 else 0)
    observed[arm,rank,window['cycle']]=(len(ks),window['rows'],cycles[rank][window['cycle']-1]['work'])
 for rank in[0,1]:
  for cycle in[1,2,3,4]:
   a,b=observed['baseline',rank,cycle],observed['candidate',rank,cycle];assert a[1:]==b[1:];assert a[0]-b[0]==(384 if a[1]<=8 else 0)
print(f"PASS {meta['entries']} hashes;624 identities;all3pairedstatistics/228flags/291work;104/1992state;operator/cache/cancel;16scopedGPUwindows/384eligiblelaunchreduction. No significantservinggain or live-GPU requalification inferred.")
