from pathlib import Path
import collections,json,hashlib
lab=Path(__file__).resolve().parent
protocol=json.loads((lab/'tp1-protocol.json').read_text())
for n,h in protocol['harness_sha256'].items():assert hashlib.sha256((lab/n).read_bytes()).hexdigest()==h,n
for n,h in protocol['build']['production_source_sha256'].items():assert hashlib.sha256((Path(protocol['build']['production_source_root'])/n).read_bytes()).hexdigest()==h,n
result=json.loads((lab/'tp1-result.json').read_text());assert result['status']=='pass'and result['exit']==0 and result['boot_unchanged']
rows=[json.loads(s)for s in(lab/'tp1-results.jsonl').read_text().splitlines()];assert len(rows)==104
phases=('borrowed-same','borrowed-cross','serialized-cross','file-cross');compared=0
for depth in(0,32768):
 for sampled in(0,1):
  ref={(x['step'],x['actor']):x for x in rows if x['depth']==depth and x['sampled']==sampled and x['phase']=='full-reference'};assert len(ref)==4
  for phase in phases:
   actual={(x['step'],x['actor']):x for x in rows if x['depth']==depth and x['sampled']==sampled and x['phase']==phase};assert actual.keys()==ref.keys()
   for key,x in actual.items():
    y=ref[key];assert {k:v for k,v in x.items()if k not in('phase','fused_ffn_hc')}=={k:v for k,v in y.items()if k not in('phase','fused_ffn_hc')},(depth,sampled,phase,key)
    compared+=1
states=[s for s in(lab/'tp1.stdout').read_text().splitlines()if s.startswith('STATE ')];assert len(states)==32
assert 'PASS records=104 actual_ffn_hc_enabled_batches=44 full_state_pairs=32'in(lab/'tp1.stdout').read_text()
summary=dict(status='pass',records=104,matched_replay_records=compared,complete_materialized_state_pairs=32,actual_verification_rows=dict(collections.Counter(str(x['rows'])for x in rows)),depths=[0,32768],sampled_and_greedy=True,scope='Fixed-history FFNHcRoute-off/on numerical/state comparison. Greedy batch plan forced to alternating1/0 in test-only engine object; sampled production policy unchanged. All canonical headers/GPU objects; no RDMA. Not a production greedy-controller replay or independent teacher/full throughput claim.',scope_prior_canonical_Q8_failure='Separate olderQ8 experiment; not a failureof thisHCport or pooled evidence.')
(lab/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
