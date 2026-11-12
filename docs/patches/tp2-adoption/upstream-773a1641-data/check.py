from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent; manifest=json.loads((root/'archive.json').read_text());p=root/'acquisition.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['sha256']
with zipfile.ZipFile(p)as z:
 assert set(z.namelist())==set(manifest['entries'])
 for name,e in manifest['entries'].items():
  data=z.read(name);assert len(data)==e['bytes']and hashlib.sha256(data).hexdigest()==e['sha256'],name
 promotion=json.loads(z.read('controllers/upstream-773a1641-promoted.json'));assert promotion['functional_qualified']and not promotion['timing_qualified']
 wider=json.loads(z.read('upstream-773a1641-wider-v1/validation.json'));assert wider['canonical_records']==60 and len(wider['strict_phase_flags'])==14 and not wider['timing_qualified']
 long=json.loads(z.read('upstream-773a1641-native-long-checkpoint-v1/result.json'));assert long['success']and long['exits']==[0,0]
 cycles=json.loads(z.read('upstream-773a1641-cycle-http-v1/cycles.json'));assert len(cycles['comparisons'])==16 and all(x['exact_ordered_work']for x in cycles['comparisons'])
 invalid=json.loads(z.read('upstream-773a1641-native-checkpoint-v1/result.json'));assert not invalid['success']
print('PASS archive integrity and recorded coverage/retained failures; not a GPU rerun or independent model oracle')
