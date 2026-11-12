"""Verify the retained intake records without executing GPU/live-host probes."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,xml.etree.ElementTree as E,zipfile
p=Path(__file__).resolve().parent;meta=json.loads((p/'archive.json').read_text());blob=(p/'acquisition.zip').read_bytes();assert hashlib.sha256(blob).hexdigest()==meta['zip_sha256']
with zipfile.ZipFile(io.BytesIO(blob))as z:
 manifest=json.loads(z.read('manifest.json'));assert len(manifest)==meta['entries']and set(z.namelist())==set(manifest)|{'manifest.json'}
 for name,d in manifest.items():
  path=PurePosixPath(name);assert not path.is_absolute()and '..'not in path.parts;data=z.read(name);assert len(data)==d['bytes']and hashlib.sha256(data).hexdigest()==d['sha256'],name
 def read(name):return json.loads(z.read('tp-cache-retention/'+name))
 family='upstream-8ab834a5-intake-v1';tests=read(family+'/validation.json');proof=read(family+'/release-source-proof.json');binary=read(family+'/binary.json');ci=read(family+'/CI.json')
 assert tests['focused_tests']=={'cpu-tests.xml':5,'gpu-tests.xml':2};assert proof['common_compile_units']==336 and not proof['normalized_command_changes']and proof['unchanged_model_family_public_include_heapplacement_sources'];assert len(proof['compiled_source_changes'])==3
 assert binary['source_commit']==meta['source']and binary['canonical_head']==meta['canonical']and binary['build_testing']is False
 for d in ci.values():assert d['status']=='completed'and d['conclusion']=='success'and d['headSha']==meta['source']
 for filename,count in [('cpu-tests.xml',5),('gpu-tests.xml',2)]:
  cases=E.fromstring(z.read('tp-cache-retention/'+family+'/'+filename)).findall('.//testcase');assert len(cases)==count and all(c.find('skipped')is None and c.find('failure')is None and c.find('error')is None for c in cases)
 assert b'Ran 12 tests'in z.read('tp-cache-retention/'+family+'/cache-functional-parser.log')
 smoke=read('upstream-8ab834a5-http-smoke-v1/validation.json');life=read('upstream-8ab834a5-http-lifecycle-v1/validation.json');assert smoke['records_per_variant']==12 and smoke['all_prompt_output_cachework_identities_exact']and smoke['actual_MTP_and_live_binary_settings_verified'];assert len(life['cache_cases'])==4 and life['two_C2_victim_roles_and_live_survivors_width2_exact_solo_and_postcancel_replay']and not any(life['checked_transport_error_deltas'].values())
 data=z.read('results/model-scaling/upstream-8ab834a5-http-smoke-v1-candidate.jsonl');rows=[json.loads(x)for x in data.splitlines()];assert len(rows)==12;keys=('case','repetition','request_index','concurrency','prompt_sha256','prompt_tokens','prefill_tokens','cached_prompt_tokens','completion_tokens','completion_sha256');ids=[{k:x[k]for k in keys}for x in rows]
 for ref in read('upstream-8ab834a5-http-smoke-v1/reference.json').values():
  original=z.read('results/model-scaling/'+Path(ref['path']).name);assert hashlib.sha256(original).hexdigest()==ref['sha256'];old=[json.loads(x)for x in original.splitlines()];assert ids==[{k:x[k]for k in keys}for x in old]
 for x in rows:assert x['draft_tokens']>0 and x['cached_prompt_tokens']==0 and x['prefill_tokens']==x['prompt_tokens']
print(f"PASS {meta['entries']} hashes;5CPU/2HIP/12parser;336commands;12smoke identities/cache4/cancel9/exactCI. Not new performance or live-GPU requalification.")
