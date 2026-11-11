"""Recompute all192canonical identities, four paired summaries and28flags."""
from pathlib import Path
import csv,gzip,hashlib,json,math
from paired_statistics import summarize
p=Path(__file__).resolve().parent
workload=json.loads((p/'workload.json').read_text())
assert hashlib.sha256((p/'requests.json.gz').read_bytes()).hexdigest()==workload['request_records_sha256']
assert hashlib.sha256((p/'prompts.json.gz').read_bytes()).hexdigest()==workload['prompts_sha256']
records=json.loads(gzip.decompress((p/'requests.json.gz').read_bytes()));assert len(records)==192
design=json.loads((p/'design.json').read_text());analysis=json.loads((p/'analysis.json').read_text());data={}
for phase,role in design['sequence']:
 xs=[x['record']for x in records if x['phase']==phase and x['role']==role];assert len(xs)==12;data[phase]=xs
keys=['case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256']
reference=[{k:x[k]for k in keys}for x in next(iter(data.values()))]
for xs in data.values():assert[{k:x[k]for k in keys}for x in xs]==reference
flags=[];logs={k:[]for k in ['C2_natural_wave','C2_coding_wave','C1_first2K','C1_repeat2K']};orders=[];work=0

def wave(xs,name):
 rs=[x for x in xs if x['case']==name];assert len(rs)==2 and max(x['physical_execution_width']for x in rs)==2
 assert[x['raw_gufo']['resident_requests_at_admission']for x in rs]==[1,2]
 return sum(x['completion_tokens']for x in rs)/(max(x['finished_at']for x in rs)-min(x['started_at']for x in rs))

for pair in design['pairs']:
 a,b=data[pair['baseline']],data[pair['candidate']];orders.append(1 if pair['order'][0]=='candidate'else-1)
 for x,y in zip(a,b):
  changed=sum(x[k]!=y[k]for k in ['draft_tokens','draft_accepted_tokens','physical_execution_width'])+int(x['raw_gufo']['draft_rounds']!=y['raw_gufo']['draft_rounds'])
  if x['concurrency']==1:assert changed==0
  work+=changed
  vals={k:(v,y[k])for k,v in x.items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(y.get(k),(int,float))};vals.update({'raw.'+k:(v,y['raw_gufo'][k])for k,v in x['raw_gufo'].items()if k.endswith('_ms')and isinstance(v,(int,float))and isinstance(y['raw_gufo'].get(k),(int,float))})
  for k,(old,new)in vals.items():
   assert math.isfinite(old)and math.isfinite(new)
   if new-old>3 and new>old*1.05:flags.append(dict(pair=pair['pair'],case=x['case'],repetition=x['repetition'],index=x['request_index'],metric=k,baseline_ms=old,candidate_ms=new,delta_ms=new-old))
 for k,old,new in [('C2_natural_wave',wave(a,'short-c2'),wave(b,'short-c2')),('C2_coding_wave',wave(a,'coding-c2'),wave(b,'coding-c2')),('C1_first2K',a[1]['decode_tokens_per_second'],b[1]['decode_tokens_per_second']),('C1_repeat2K',a[2]['decode_tokens_per_second'],b[2]['decode_tokens_per_second'])]:logs[k].append(math.log(new/old))
assert len(data)==16 and work==analysis['adaptive_work_differences']==72
with(p/'phase-flags.csv').open()as f:expected=list(csv.DictReader(f))
assert len(flags)==len(expected)==analysis['phase_flags']==28
for x,y in zip(flags,expected):
 for k in ['pair','case','repetition','index','metric']:assert str(x[k])==y[k]
 for k in ['baseline_ms','candidate_ms','delta_ms']:assert math.isclose(x[k],float(y[k]),rel_tol=0,abs_tol=1e-9)
for result in analysis['results']:
 metric=result['metric'];actual=summarize(logs[metric],orders)
 for k in ['paired_geometric_throughput_increase_pct','balanced_assignment_sensitivity_p','predeclared_two_sided_signflip_p']:
  assert math.isclose(actual[k],result[k],rel_tol=0,abs_tol=1e-11),(metric,k)
 assert actual['bootstrap95_throughput_increase_pct']==result['bootstrap95_throughput_increase_pct']
 print(metric,f"{actual['paired_geometric_throughput_increase_pct']:+.6f}%",'95%CI',actual['bootstrap95_throughput_increase_pct'],'balanced-p',actual['balanced_assignment_sensitivity_p'],'primary',result['primary'])
print('PASS192canonical records/C1work,72C2work comparisons,allfour paired summaries and28retained timing flags.')
print('Only natural C2 is confirmatory; secondary/unadjusted results and timing tradeoffs retain their original scope.')
