from pathlib import Path
import collections,hashlib,json,re,sys
lab=Path(__file__).resolve().parent;root=lab.parents[1]
event_re=re.compile(r'SCHED_EVENT event=(\w+) request=(\d+) t_ns=(\d+) a=(\d+) b=(\d+) c=(\d+)')
span_re=re.compile(r'SCHED_SPAN unit=(\w+) request=(\d+) call=(\d+) stage=(\w+) t_ns=(\d+) a=(\d+) b=(\d+)')
def union_ms(xs):
 m=[]
 for a,b in sorted(xs):
  if m and a<=m[-1][1]:m[-1][1]=max(b,m[-1][1])
  else:m.append([a,b])
 return sum(b-a for a,b in m)/1e6
report=[];inputs={};latencies=[]
for arm,source in [('baseline',lab.with_name('first-output-publication-trace-v1')),('candidate',lab)]:
 files=[source/f'{arm}-server.stderr',source/f'{arm}-requests.jsonl']
 for f in files:inputs['tp-cache-retention/'+f.parent.name+'/'+f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 events=[];groups=collections.defaultdict(list);batch_open=None;batch_intervals=[]
 for line in files[0].read_text().splitlines():
  m=event_re.search(line)
  if m:
   event,rid,ns,a,b,c=m.groups();e=dict(event=event,request=int(rid),ns=int(ns),a=int(a),b=int(b),c=int(c));events.append(e)
   if event=='runner_batch_begin':assert batch_open is None;batch_open=e
   if event=='runner_batch_end':
    assert batch_open is not None;batch_intervals.append(dict(unit='runner_batch',start=batch_open['ns'],end=e['ns']));batch_open=None
  m=span_re.search(line)
  if m:
   unit,rid,call,stage,ns,a,b=m.groups();groups[(unit,int(rid),int(call))].append(dict(stage=stage,ns=int(ns),a=int(a),b=int(b)))
 assert batch_open is None
 spans=[]
 for (unit,rid,call),g in groups.items():
  assert len(g)==2 and g[0]['stage']=='entry'and g[1]['stage']=='leave'
  spans.append(dict(unit=unit,request=rid,call=call,start=g[0]['ns'],end=g[1]['ns'],a=g[0]['a'],b=g[0]['b']))
 rows=[json.loads(line)for line in files[1].read_text().splitlines()]
 for row in rows:
  if row['case']not in('C2','C2-after-preview'):continue
  ids={s['request']for s in spans if s['unit']=='prefill'and s['a']==row['prompt_tokens']and row['started_at']*1e9<=s['start']<=row['finished_at']*1e9}
  assert len(ids)==1,(arm,row['case'],row['request_index'],ids)
  rid=ids.pop();own=[e for e in events if e['request']==rid]
  queues=[e for e in own if e['event']=='piece_queued'and e['a']>0];consumers=[e for e in own if e['event']=='consumer_deliver'and e['a']>0]
  assert len(queues)>=2 and len(consumers)>=2
  first,second=queues[:2];gap=second['ns']-first['ns'];selected=[];coverage=[]
  for s in spans+batch_intervals:
   a,b=max(first['ns'],s['start']),min(second['ns'],s['end'])
   if b>a:
    selected.append({**s,'overlap_ms':(b-a)/1e6,'start_relative_first_piece_ms':(s['start']-first['ns'])/1e6})
    if s['unit']in('single_decode','prefill','runner_batch'):coverage.append((a,b))
  # Every row-ready is classified inside its actual containing scheduler step.
  ready_records=[]
  for e in own:
   if e['event']not in('batch_row_ready','single_return'):continue
   kind='batch_decode'if e['event']=='batch_row_ready'else'single_decode'
   containers=[s for s in spans if s['unit']==kind and s['start']<=e['ns']<=s['end']and(s['request']==0 or s['request']==rid)]
   assert len(containers)==1,(arm,rid,e,containers)
   end=containers[0]['end'];novel=[p for p in own if p['event']=='publish_selection'and e['ns']<=p['ns']<=end];decoded=[p for p in own if p['event']=='decoded_selection'and e['ns']<=p['ns']<=end]
   rec={'event':e['event'],'t_ns':e['ns'],'returned_selections':e['a'],'published_tokens_before_step':e['b'],'preview_pending':bool(e['c']),'preview_duplicates':sum(p['b']!=0 for p in decoded),'novel_publications':len(novel),'nonempty_novel_publications':sum(p['b']>0 for p in novel),'ready_to_first_publication_ms':(novel[0]['ns']-e['ns'])/1e6 if novel else None,'after_first_piece_ms':(e['ns']-first['ns'])/1e6}
   ready_records.append(rec)
   if novel:latencies.append({'arm':arm,'case':row['case'],'actor':row['request_index'],'request':rid,'ms':rec['ready_to_first_publication_ms']})
  contained_ready=[r for r in ready_records if first['ns']<=r['t_ns']<=second['ns']]
  relevant_events=[e for e in own if first['ns']<=e['ns']<=second['ns']and e['event']in('decoded_selection','publish_selection','batch_row_ready','single_return','consumer_deliver')]
  report.append({'arm':arm,'case':row['case'],'actor':row['request_index'],'request':rid,'client_first_gap_ms':row['first_followup_content_gap_ms'],'server_piece_gap_ms':gap/1e6,'client_minus_server_gap_ms':row['first_followup_content_gap_ms']-gap/1e6,'queue_to_consumer_first_two_ms':[(c['ns']-q['ns'])/1e6 for q,c in zip(queues[:2],consumers[:2])],'covered_work_union_ms':union_ms(coverage),'uncovered_ms':gap/1e6-union_ms(coverage),'overlapping_work':selected,'ready_steps_during_gap':contained_ready,'all_initial_ready_steps':ready_records,'own_events_during_gap':relevant_events})
result={'inputs_sha256':inputs,'records':report,'ready_to_first_publication':latencies,'scope':'Diagnostic actual-return/preview/publish/queue/consumer chronology under simultaneous and after-preview arrivals. Baselinev1 completed; candidatev2 corrects admission-order metadata assertion only, retains failedv1candidate/serverlogs. CPUspans include necessary GPU/I/O; use union, not nested sum. No pureCPUoverhead attribution or originalcandidateexoneration/performanceconfidence/adoption claim. Source observer may change arrival/controller timing.'}
if '--check' in sys.argv:
 assert json.loads((lab/'publication-analysis.json').read_text()) == result
else:
 (lab/'publication-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in report:
 print(r['arm'],r['case'],'actor',r['actor'],'request',r['request'],'client/servergap',round(r['client_first_gap_ms'],3),round(r['server_piece_gap_ms'],3),'work',round(r['covered_work_union_ms'],3),'uncovered',round(r['uncovered_ms'],3),'queue-consumer',r['queue_to_consumer_first_two_ms'])
 print(' ready',r['ready_steps_during_gap'])
print('max ready→publish ms',max(r['ms']for r in latencies),'events',len(latencies))
