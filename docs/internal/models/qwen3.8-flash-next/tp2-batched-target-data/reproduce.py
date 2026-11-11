from pathlib import Path
import argparse,concurrent.futures,gzip,hashlib,json,sys,threading,urllib.request
parser=argparse.ArgumentParser(description='Replay the exact12-request TP2/MTP benchmark history on an already running matched server.')
parser.add_argument('output',help='Fresh JSONL destination; existing files are refused')
parser.add_argument('base_url',help='Head HTTP URL, e.g. http://127.0.0.1:28530')
args=parser.parse_args();sys.argv=[sys.argv[0],args.output,args.base_url]
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'tools'))
from gufo import serving_bench as bench
base=sys.argv[2];dest=Path(sys.argv[1]);original=bench._required_integer;local=threading.local()
def integer(metrics,key):
 if key=='prefill_tokens':local.metrics=dict(metrics)
 return original(metrics,key)
bench._required_integer=integer
# Same client-side arrival recorder in both arms; no engine/profiler changes.
original_sse=bench._sse_data
import time
def sse(response):
 for event in original_sse(response):
  if event!='[DONE]':
   payload=json.loads(event);choices=payload.get('choices')
   if isinstance(choices,list)and choices:
    delta=choices[0].get('delta',{});content=delta.get('content');tools=delta.get('tool_calls')
    if content or tools:local.content_events.append({'perf_counter_ns':time.perf_counter_ns(),'content_bytes':len(content.encode())if isinstance(content,str)else 0})
  yield event
bench._sse_data=sse
class Gate:
 def __init__(self,index,event):self.index=index;self.event=event
 def wait(self):
  if self.index==0:self.event.set()
  else:self.event.wait();__import__('time').sleep(.01)
prompts=json.loads(gzip.decompress((Path(__file__).resolve().parent/'prompts.json.gz').read_bytes()))
def count(text):
 req=urllib.request.Request(base+'/v1/messages/count_tokens',data=json.dumps({'model':'flash-next-tp-test','messages':[{'role':'user','content':text}]}).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=60))['input_tokens']
assert count(prompts['32768'])==32768 and count(prompts['2048'])==2048
short='Write a detailed numbered list of thirty practical ways to debug a slow distributed inference server. Explain each item.'
def request(case,text,index=0,width=1,tokens=128,gate=None,rep=1):
 local.metrics=None;local.content_events=[];obs=bench.run_request(base_url=base,model='flash-next-tp-test',prompt=text,max_tokens=tokens,temperature=0,timeout_seconds=600,client_id=f'refresh-{case}-{rep}-{index}',concurrency=width,repetition=rep,request_index=index,start_gate=gate,cache_prompt=False,extra_body={'chat_template_kwargs':{'enable_thinking':False}},endpoint_profile='gufo')
 record={'case':case,**obs.public(),'started_at':obs.started_at,'finished_at':obs.finished_at,'raw_gufo':local.metrics,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'content_events':local.content_events,'first_followup_content_gap_ms':(local.content_events[1]['perf_counter_ns']-local.content_events[0]['perf_counter_ns'])/1e6 if len(local.content_events)>1 else None}
 record['maximum_content_gap_ms']=max(((b['perf_counter_ns']-a['perf_counter_ns'])/1e6 for a,b in zip(local.content_events,local.content_events[1:])),default=None)
 return record
with dest.open('x') as out:
 def save(r):
  out.write(json.dumps(r)+'\n');out.flush();print(r['case'],r['repetition'],r['request_index'],'prefill',r['prefill_tokens_per_second'],'decode',r['decode_tokens_per_second'],flush=True)
 save(request('warmup','Explain tensor parallel inference briefly.',tokens=8,rep=0))
 for rep in [1,2]:save(request('pp2048-tg128',prompts['2048'],rep=rep))
 for rep in [1]:
  event=threading.Event()
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(request,'short-c2',short+f'\nRequest variant {i}.',i,2,128,Gate(i,event),rep) for i in range(2)];rs=[j.result() for j in jobs]
   assert [r['raw_gufo']['resident_requests_at_admission'] for r in rs]==[1,2]
   for r in rs:save(r)

 # End after target C2 wave; later 32K arm omitted from this focused study.

 for topic,text in [('cpu-cache', 'Explain how CPU cache hierarchies affect program performance.'), ('red-black-tree', 'Implement a red-black tree insertion algorithm in Python with comments.'), ('tcp', 'Explain TCP congestion control and how it responds to packet loss.'), ('czech-rail', 'Describe the history of rail transport in the Czech lands.'), ('iso8601', 'Implement a robust C++ parser for ISO 8601 timestamps and explain the edge cases.')]:
  save(request('short16-'+topic,text,tokens=16))
 event=threading.Event()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(request,'coding-c2',text,i,2,256,Gate(i,event),1)for i,text in enumerate(['Implement a Python asynchronous work queue with bounded concurrency, fair scheduling and cancellation. Provide code and explain the invariants.', 'Implement a C++ bounded concurrent queue. Explain safe producer and consumer shutdown and provide code.'])];rs=[j.result()for j in jobs]
  assert [r['raw_gufo']['resident_requests_at_admission']for r in rs]==[1,2]
  for r in rs:save(r)
