"""Offline evidence-integrity audit; does not rerun GPU or live-host checks."""
from pathlib import Path, PurePosixPath
import collections,csv,hashlib,io,json,zipfile
out=Path(__file__).resolve().parent;meta=json.loads((out/'archive.json').read_text());raw=(out/'acquisition.zip').read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['zip_sha256']
with zipfile.ZipFile(io.BytesIO(raw))as z:
 manifest=json.loads(z.read('manifest.json'));assert len(manifest)==meta['entries'];assert set(z.namelist())==set(manifest)|{'manifest.json'}
 for name,entry in manifest.items():
  p=PurePosixPath(name);assert not p.is_absolute()and '..'not in p.parts
  b=z.read(name);assert len(b)==entry['bytes']and hashlib.sha256(b).hexdigest()==entry['sha256'],name
 def read(name):return json.loads(z.read('tp-cache-retention/'+name))
 family='batched-hc-q8-release-comparison-v1';stats=read(family+'/statistical-summary.json');native=read(family+'/validation.json');assert native['all_prompt_output_cachework_identities_exact']and stats['c1_speculative_work_exact'];assert len(stats['c2_work_changes'])==21
 expected=None;count=0;keys=('case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256')
 for path,digest in stats['raw_inputs_sha256'].items():
  name='results/model-scaling/'+Path(path).name;b=z.read(name);assert hashlib.sha256(b).hexdigest()==digest
  rows=[json.loads(s)for s in b.splitlines()];assert len(rows)==12;identities=[{k:x[k]for k in keys}for x in rows]
  if expected is None:expected=identities
  assert identities==expected;count+=len(rows)
 assert count==240 and len(stats['raw_inputs_sha256'])==20
 primary=[x for x in stats['metrics']if x['primary']];assert {x['metric']for x in primary}=={'C2-coding-client-wave','C2-short-client-wave'}
 assert all(x['marginal_bootstrap95_throughput_increase_pct'][0]<=0<=x['marginal_bootstrap95_throughput_increase_pct'][1]and x['holm_adjusted_p']>.05 for x in primary)
 flags=list(csv.DictReader(io.StringIO(z.read('tp-cache-retention/'+family+'/timing-flags.csv').decode())));assert len(flags)==33 and len({(x['pair'],x['case'],x['request_index'])for x in flags})==9
 state=read('batched-hc-q8-state-v2/validation.json');assert(state['records_per_rank'],state['complete_postreplay_state_pairs_per_rank'])==(104,32)
 wide=read('batched-hc-q8-wide-state-v2/validation.json');assert(wide['records_per_rank'],wide['complete_materialized_state_pairs_per_rank'])==(1992,498)
 active=read('batched-hc-q8-activation-v1/validation.json');assert active['per_rank_counts_equal']and all((x['total_produced'],x['total_ready_matches'])==(15745,24874)for x in active['ranks'])
 life=read('batched-hc-ready-q8-http-lifecycle-v1/validation.json');assert len(life['cache_cases'])==4 and life['two_C2_victim_roles_and_live_survivors_width2_exact_solo_and_postcancel_replay']
 component=read('batched-hc-ready-q8-component-v1/analysis.json');assert component['conditions']==30 and component['timing_records']==7200 and component['all_exact_F32_inject_Q8_and_guards']
 test=z.read('tp-cache-retention/batched-hc-q8-review-test-v1/test.log').decode();assert 'HC mixed outputs and short prefill chunks passed'in test and 'SKIP'not in test.upper()
 source=z.read('review/hc_mix_ops_test.cpp').decode();assert 'CheckBatchedReadyQ8(rows, inject, scale)'in source and 'q::HcInjectParts(kHidden)'in source
print(f"PASS {meta['entries']} hashed entries;240 canonical outputs;104/1992 state records per rank;activation/cache/cancel;33 flags/21 work comparisons retained;both primary estimates inconclusive")
