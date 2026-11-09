from pathlib import Path
import hashlib,json,re,zipfile
p=Path(__file__).resolve().parent;m=json.loads((p/'archive.json').read_text());b=(p/'acquisition.zip').read_bytes();assert hashlib.sha256(b).hexdigest()==m['sha256']
with zipfile.ZipFile(p/'acquisition.zip') as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(m['entries'])
 for name,digest in m['entries'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 a=json.loads(z.read('analysis.json'));assert a['requests']==168 and a['all_output_prompt_cache_identities_match'];assert len(a['flags'])==34
 for role in ('baseline','pr14','pr15'):
  assert re.search(r'Building CXX object.*batch.cpp.o',z.read(f'build-{role}.log').decode())
  log=z.read(f'trace-v3-build-{role}.log').decode()
  for name in ('text_generation_scheduler.cpp.o','text_model_runner.cpp.o','batch.cpp.o'):assert name in log,(role,name)
 t=json.loads(z.read('trace-v3-analysis.json'));assert len(t['records'])==80
 for meta in t['events'].values():assert 'overflow=0' in meta['header']
 assert len(json.loads(z.read('observer-summary.json')))==4
print('PASS archive integrity, retained production flags, build evidence and observer coverage; no live/GPU qualification implied')
