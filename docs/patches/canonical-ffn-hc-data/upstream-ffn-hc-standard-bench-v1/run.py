"""Stock fixed-token C1 benchmark, separate from the HTTP/EOS screen."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time

lab = Path(__file__).resolve().parent
root = lab.parents[1]
lock = (root / 'overnight/results/model-matrix.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (lab / 'protocol.json').exists()
assert (root / 'tp-cache-retention/upstream-ffn-hc-first-gap-repro-v1/runtime-complete').exists()
proof = json.loads((root / 'tp-cache-retention/upstream-ffn-hc-v1/release-source-proof.json').read_text())
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(root / 'dependencies/gufo-dev/usr/lib/x86_64-linux-gnu') + ':/home/rkuo/.local/opt/rocm-7.14.1/lib'
for key in ('LD_PRELOAD', 'ROCP_TOOL_ATTACH', 'STRIX_READ_OUTPUT', 'STRIX_FINISH_TRACE', 'GUFO_TP_TRACE_PREFIX'):
    env.pop(key, None)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))


protocol = {'sequence': ['baseline', 'candidate'], 'binaries': proof['binaries'],
            'boot': boot, 'cmdline': Path('/proc/cmdline').read_text().strip(),
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'One fresh production CLI process per arm, two repetitions; canonical fixed-token standard benchmark. pp2048 and tg128 are separate workloads. Generation prefix16, actual128 tokens with stop_at_eos=false in stock driver; capacity4096 at depth0. Synthetic tokens, not HTTP chat or a 2048-token-history continuation. No significance/equivalence/adoption claim, no pooling with HTTP or TP2 results.'}
(lab / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
outputs = {}
for role in protocol['sequence']:
    assert available() > 100 * 1024 * 1024
    entry = proof['binaries'][role]
    binary = Path(entry['path'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == entry['sha256']
    command = [str(binary), 'bench', '--model',
        '/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
        '--mtp-model', '/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',
        '--speculative', 'mtp', '--n-prompt', '2048', '--n-gen', '128', '--n-depth', '0',
        '--repetitions', '2', '--temperature', '0', '--seed', '1', '--verbose']
    (lab / f'{role}-command.json').write_text(json.dumps(command, indent=2) + '\n')
    process = None
    minimum = available()
    start = time.monotonic()
    try:
        with (lab / f'{role}.stdout').open('w') as out, (lab / f'{role}.stderr').open('w') as err:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=out, stderr=err, start_new_session=True)
        (lab / f'{role}.pid').write_text(str(process.pid) + '\n')
        while process.poll() is None:
            minimum = min(minimum, available())
            assert minimum > 8 * 1024 * 1024 and time.monotonic() - start < 900
            time.sleep(.25)
        assert process.returncode == 0
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        result = {'exit': None if process is None else process.returncode, 'minimum_available_kib': minimum,
                  'elapsed_seconds': time.monotonic() - start,
                  'boot_unchanged': Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot}
        (lab / f'{role}-result.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['exit'] == 0 and result['boot_unchanged']
    raw = (lab / f'{role}.stderr').read_text()
    hashes = re.findall(r'Qwen3\.8-Flash-Next tg depth=0 .*output_sha256=([0-9a-f]{64})', raw)
    assert len(hashes) == 2 and len(set(hashes)) == 1
    outputs[role] = hashes
    print(role, json.dumps(result), hashes, flush=True)
assert outputs['baseline'] == outputs['candidate']
(lab / 'validation.json').write_text(json.dumps({'status': 'pass', 'output_token_hashes': outputs,
    'scope': protocol['scope'], 'HTTP_original_failure_and_flags_retained': True}, indent=2) + '\n')
