"""Bounded natural-controller TP1 screen; no engine hooks or forced plans."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
import urllib.request

lab = Path(__file__).resolve().parent
root = lab.parents[1]
sys.path.insert(0, str(root / 'tp-cache-retention/upstream-ffn-hc-worktree/tools'))
from gufo import serving_bench as bench

base, arm = sys.argv[1:]
local = threading.local()
first_preview_event = threading.Event()
original_integer = bench._required_integer
original_sse = bench._sse_data


def integer(metrics, key):
    if key == 'prefill_tokens':
        local.metrics = dict(metrics)
    return original_integer(metrics, key)


def sse(response):
    for event in original_sse(response):
        if event != '[DONE]':
            choices = json.loads(event).get('choices', [])
            if choices:
                delta = choices[0].get('delta', {})
                if delta.get('content') or delta.get('tool_calls'):
                    local.events.append(time.perf_counter_ns())
                    if local.release_after_first and len(local.events) == 1:
                        first_preview_event.set()
        yield event


bench._required_integer = integer
bench._sse_data = sse
prompts = json.loads((root / 'prefill-scaling/results/prompts.json').read_text())
shorts = [
    'Implement a Python asynchronous work queue with bounded concurrency, fair scheduling and cancellation. Provide code and explain the invariants.',
    'Implement a C++ bounded concurrent queue. Explain safe producer and consumer shutdown and provide code.',
]


def count(text):
    req = urllib.request.Request(base + '/v1/messages/count_tokens', data=json.dumps({
        'model': 'flash-next-hc-tp1', 'messages': [{'role': 'user', 'content': text}]
    }).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)['input_tokens']


assert count(prompts['2048']) == 2048


class Gate:
    def __init__(self, index, barrier):
        self.barrier = barrier

    def wait(self):
        self.barrier.wait(timeout=60)


class AfterPreviewGate:
    def __init__(self, index):
        self.index = index

    def wait(self):
        if self.index != 0:
            assert first_preview_event.wait(timeout=60), 'First client produced no content within60s'


def request(case, text, index=0, width=1, tokens=128, rep=1, gate=None):
    local.metrics, local.events = None, []
    local.release_after_first = case == "C2-after-preview" and index == 0
    obs = bench.run_request(
        base_url=base, model='flash-next-hc-tp1', prompt=text, max_tokens=tokens,
        temperature=0, timeout_seconds=600, client_id=f'hc-{arm}-{case}-{rep}-{index}',
        concurrency=width, repetition=rep, request_index=index, start_gate=gate,
        cache_prompt=False, extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        endpoint_profile='gufo',
    )
    return {
        'arm': arm, 'case': case, **obs.public(), 'raw_gufo': local.metrics,
        'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'started_at': obs.started_at, 'finished_at': obs.finished_at,
        'content_events_ns': local.events,
        'first_followup_content_gap_ms': (local.events[1] - local.events[0]) / 1e6 if len(local.events) > 1 else None,
        'maximum_content_gap_ms': max(((b - a) / 1e6 for a, b in zip(local.events, local.events[1:])), default=None),
    }


with (lab / f'{arm}-requests.jsonl').open('x') as out:
    def save(record):
        out.write(json.dumps(record) + '\n')
        out.flush()
        print(record['case'], record['request_index'], record['completion_tokens'],
              record['prefill_tokens_per_second'], record['decode_tokens_per_second'], flush=True)

    save(request('warmup', 'Explain tensor parallel inference briefly.', tokens=8, rep=0))
    for rep in (1, 2):
        save(request('pp2048-tg128', prompts['2048'], rep=rep))
    for index, text in enumerate(shorts):
        save(request('solo-before', text, index=index))
    event = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(request, 'C2', text, index, 2, 128, 1, Gate(index, event))
                for index, text in enumerate(shorts)]
        records = [job.result() for job in jobs]
    assert [r['raw_gufo']['resident_requests_at_admission'] for r in records] == [1, 2]
    for record in records:
        save(record)
    for index, text in enumerate(shorts):
        save(request('solo-after', text, index=index))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(request, 'C2-after-preview', text, index, 2, 128, 1, AfterPreviewGate(index)) for index, text in enumerate(shorts)]
        records = [job.result() for job in jobs]
    assert [r['raw_gufo']['resident_requests_at_admission'] for r in records] == [1, 2]
    for record in records:
        save(record)
