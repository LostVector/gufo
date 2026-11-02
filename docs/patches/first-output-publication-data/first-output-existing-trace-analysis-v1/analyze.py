from pathlib import Path
import collections, hashlib, json, re
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
stage=re.compile(r'HC_STAGE unit=(\w+) call=(\d+) stage=(\w+) t_ns=(\d+) elapsed_us=([\d.]+)')
solo=re.compile(r'HC_SOLO unit=(\w+) call=(\d+) stage=(\w+) t_ns=(\d+) elapsed_us=([\d.]+) session=([\da-f]+) position=(\d+) tokens=(\d+)')
def union(intervals):
 merged=[]
 for a,b in sorted(intervals):
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 return sum(b-a for a,b in merged)/1e6
results=[];inputs={}
for labname in ['upstream-ffn-hc-first-gap-trace-v2','upstream-ffn-hc-first-gap-repro-v1']:
 lab=root/'tp-cache-retention'/labname
 for f in sorted(lab.glob('*-requests.jsonl')):
  arm=f.name.removesuffix('-requests.jsonl');log=lab/f'{arm}-server.stderr';inputs[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest();inputs[str(log.relative_to(root))]=hashlib.sha256(log.read_bytes()).hexdigest()
  groups=collections.defaultdict(dict)
  for line in log.read_text().splitlines():
   m=stage.search(line)
   if m:
    unit,call,key,ns,elapsed=m.groups();g=groups[('stage',unit,int(call))];g.setdefault('marks',[]).append((key,int(ns)));g.update(unit=unit,call=int(call))
   m=solo.search(line)
   if m:
    unit,call,key,ns,elapsed,session,pos,tokens=m.groups();g=groups[('solo',unit,int(call))];g.setdefault('marks',[]).append((key,int(ns)));g.update(unit=unit,call=int(call),session=session,position=int(pos),tokens=int(tokens))
  spans=[]
  for key,g in groups.items():
   marks=g['marks'];assert marks[0][0]=='entry'and marks[-1][0]=='leave';spans.append({**{k:v for k,v in g.items()if k!='marks'},'kind':key[0],'start_ns':marks[0][1],'end_ns':marks[-1][1]})
  records=[json.loads(s)for s in f.read_text().splitlines()]
  for r in records:
   if r['case']!='C2':continue
   ev=r['content_events_ns'];assert len(ev)>1;maxi=max(range(len(ev)-1),key=lambda i:ev[i+1]-ev[i])
   for label,index in [('first',0),('maximum',maxi)]:
    a,b=ev[index:index+2];selected=[];covered=[];byunit=collections.defaultdict(list)
    for sp in spans:
     lo,hi=max(a,sp['start_ns']),min(b,sp['end_ns'])
     if hi<=lo:continue
     x={**sp,'gap_overlap_ms':(hi-lo)/1e6,'start_relative_gap_ms':(sp['start_ns']-a)/1e6,'end_relative_gap_ms':(sp['end_ns']-a)/1e6};selected.append(x)
     if sp['unit']!='batch':covered.append((lo,hi))
     byunit[sp['unit']].append((lo,hi))
    results.append({'lab':labname,'arm':arm,'actor':r['request_index'],'interval':label,'event_index':index,'gap_ms':(b-a)/1e6,'covered_union_ms':union(covered),'uncovered_ms':(b-a)/1e6-union(covered),'by_unit_union_ms':{k:union(v)for k,v in byunit.items()},'spans':selected,'raw_snapshot_ms':r['raw_gufo']['cache_snapshot_ms']})
(out/'analysis.json').write_text(json.dumps({'inputs_sha256':inputs,'records':results,'scope':'Posthoc read-only analysis of all actors first/maximum content gaps from existing diagnostic traces. Spans nest; union avoids doublecount. GPU completion waits include queuedwork; uncovered time is not proof ofidle orI/O. Doesnotattributeoriginaluninstrumentedcandidateflag or estimateperformancegain.'},indent=2)+'\n')
for r in results:
 if r['interval']=='first'or r['gap_ms']>95:print(r['lab'],r['arm'],'actor',r['actor'],r['interval'],r['event_index'],'gap',round(r['gap_ms'],3),'units',{k:round(v,3)for k,v in r['by_unit_union_ms'].items()},'uncovered',round(r['uncovered_ms'],3),'snapshot',round(r['raw_snapshot_ms'],3))
