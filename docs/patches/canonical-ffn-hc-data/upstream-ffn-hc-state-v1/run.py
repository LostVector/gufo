from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time
lab=Path(__file__).resolve().parent;r=lab.parents[1];lock=(r/'overnight/results/model-matrix.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not(lab/'tp1-protocol.json').exists()
assert json.loads((lab.parent/'upstream-ffn-hc-v1/operator/validation.json').read_text())['real_GPU_operator_tests']==1
build=json.loads((lab/'tp1-build-result.json').read_text());probe=lab/'tp1-probe';assert hashlib.sha256(probe.read_bytes()).hexdigest()==build['binary_sha256']
env=os.environ.copy();env.update(json.loads((lab/'tp1-build-commands.json').read_text())['environment'])
for k in('LD_PRELOAD','ROCP_TOOL_ATTACH','STRIX_READ_OUTPUT','STRIX_FINISH_TRACE','GUFO_TP_TRACE_PREFIX'):env.pop(k,None)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();cmdline=Path('/proc/cmdline').read_text().strip()
def available():return int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))
assert available()>100*1024*1024,'Require idle single-node memory headroom before loading full model.'
command=[str(probe),'/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf','/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',str(lab/'tp1-results.jsonl')]
deps=subprocess.check_output(['ldd',str(probe)],env=env,text=True);(lab/'tp1-ldd.txt').write_text(deps);assert 'not found'not in deps
record=dict(command=command,boot_id=boot,cmdline=cmdline,build=build,design=json.loads((lab/'tp1-design.json').read_text()),initial_available_kib=available(),scope='Correctness only; no profiler, NVMe sampler, transport or settings change. Preserve all failures. Guard against less than 8GiB MemAvailable; no performance inference from this observer.')
record['harness_sha256']={n:hashlib.sha256((lab/n).read_bytes()).hexdigest()for n in ['build.py','run.py','analyze.py','tp1-probe.cpp','tp1-diagnostic-batch.cpp','tp1-diagnostic-engine.cpp','tp1-design.json','tp1-build-commands.json','tp1-build-result.json']}
(lab/'tp1-protocol.json').write_text(json.dumps(record,indent=2)+'\n')
status='unknown';p=None;start=time.monotonic();minimum=available()
try:
 with(lab/'tp1.stdout').open('w')as out,(lab/'tp1.stderr').open('w')as err:
  p=subprocess.Popen(command,env=env,cwd=r,stdout=out,stderr=err,start_new_session=True)
  (lab/'tp1.pid').write_text(str(p.pid)+'\n')
  while p.poll()is None:
   minimum=min(minimum,available())
   if minimum<8*1024*1024:status='memory-headroom-stop';raise RuntimeError(status)
   if time.monotonic()-start>1200:status='deadline-stop';raise RuntimeError(status)
   time.sleep(.25)
  status='pass'if p.returncode==0 else'probe-failed'
finally:
 if p is not None and p.poll()is None:
  os.killpg(p.pid,signal.SIGINT)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
 result=dict(status=status,exit=None if p is None else p.returncode,min_available_kib=minimum,boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,elapsed_seconds=time.monotonic()-start)
 (lab/'tp1-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert status=='pass'
rows=[json.loads(x)for x in(lab/'tp1-results.jsonl').read_text().splitlines()];assert len(rows)==104
assert 'PASS records=104 actual_ffn_hc_enabled_batches=44 full_state_pairs=32'in(lab/'tp1.stdout').read_text()
(lab/'tp1-complete').write_text('104records/44enabledcycles/32fullstatepairs; canonicalfixedhistorygreedy1/0/sampledpolicyunchanged; no productiongreedycontroller/teacher/speed claim\n')
