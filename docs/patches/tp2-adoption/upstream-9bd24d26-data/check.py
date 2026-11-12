from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
read=lambda n:json.loads((p/n).read_text())
head='75de9f82234da17257321db81bd13e914ea2dbb6'
source=read('source-contract.json');assert source['head']==head and source['changed_files']==46
assert source['parents']==['1ad4bd3b047c8aa3b3526bee6303d8f98f36d441','9bd24d26b066ce08a463c2def8da9acb95c75338']
build=read('build-comparison.json');assert build['old_compile_units']==build['new_compile_units']==336 and not build['changed_normalized_commands']
assert len(build['changed_compiled_source'])==13 and all(n.startswith('src/cli/serve/')for n in build['changed_compiled_source'])
cpu=read('cpu-validation.json');assert cpu['distinct_affected_checks']==15 and cpu['original_passes']==14 and cpu['same_test_source_recovered_in_project_environment']and cpu['no_skips']
assert len(read('repository-checks.json'))==6 and all(x['exit']==0 for x in read('repository-checks.json'))
ci=read('ci.json');assert len(ci)==2
assert all(x['headSha']==head and x['status']=='completed'and x['conclusion']=='success'for x in ci.values())
assert any(j['name']=='Run Windows model checks'and j['conclusion']=='skipped'for x in ci.values()for j in x['jobs'])
binary=read('binary.json');assert binary['source_commit']==head and binary['canonical_head']==source['parents'][1]and not binary['build_testing']and binary['full_fresh_release']
smoke=read('native-smoke.json');assert smoke['records_per_variant']==12 and smoke['all_prompt_output_cachework_identities_exact']and smoke['actual_MTP_and_live_binary_settings_verified']and smoke['driver_error_kernel_idle_source_harness_checks_pass']
life=read('native-lifecycle.json');assert life['source']==head and life['case_counts']=={'cache':4,'cancel':9}
assert len(life['result']['cache_cases'])==4 and all(x['cached']==317 and x['exact']for x in life['result']['cache_cases'])
assert life['result']['two_C2_victim_roles_and_live_survivors_width2_exact_solo_and_postcancel_replay']and life['result']['completed_substantive_requests_actual_MTP']
assert all(not x for x in life['result']['checked_transport_error_deltas'].values())
print('PASS compact qualification summary hashes/source/build/CPU/CI/native records; no live-host or complete raw-trace replay claim')
