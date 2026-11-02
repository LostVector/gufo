"""Matched source-instrumented first-gap attribution, no headline timing claims."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request

lab = Path(__file__).resolve().parent
root = lab.parents[1]
lock = (root / 'overnight/results/model-matrix.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (lab / 'protocol.json').exists()
assert (lab / 'build-protocol.json').exists()
diagnostic_build = json.loads((lab / 'build-protocol.json').read_text())
build_lab = root / 'tp-cache-retention/upstream-ffn-hc-v1'
for name in ('release-build-complete', 'baseline-release-build-complete'):
    assert (build_lab / name).exists(), name
assert json.loads((root / 'tp-cache-retention/upstream-ffn-hc-state-v1/analysis.json').read_text())['status'] == 'pass'
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(root / 'dependencies/gufo-dev/usr/lib/x86_64-linux-gnu') + ':/home/rkuo/.local/opt/rocm-7.14.1/lib'
for key in ('LD_PRELOAD', 'ROCP_TOOL_ATTACH', 'STRIX_READ_OUTPUT', 'STRIX_FINISH_TRACE', 'GUFO_TP_TRACE_PREFIX', 'OPENAI_API_KEY'):
    env.pop(key, None)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
port = 28630
base = f'http://127.0.0.1:{port}'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', port))


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


binaries = {}
for role, stem, revision in (
    ('baseline', 'upstream-ffn-hc-baseline', '8ab834a5902bf19341750daecf27516a57840dd1'),
    ('candidate', 'upstream-ffn-hc', '6ec3f75f338d9f2154ef60e1c14e2dd2c70ff53b'),
):
    source = root / 'tp-cache-retention' / (stem + '-worktree')
    binary = lab / role / 'gufo'
    assert sha(binary) == diagnostic_build['records'][role]['binary_sha256']
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    cache = (root / 'tp-cache-retention' / (stem + '-release-build') / 'CMakeCache.txt').read_text()
    assert 'BUILD_TESTING:BOOL=OFF' in cache and 'CMAKE_BUILD_TYPE:STRING=RelWithDebInfo' in cache
    assert 'GUFO_ENABLE_TP2_RDMA:' not in cache
    deps = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
    assert 'not found' not in deps
    (lab / f'{role}-ldd.txt').write_text(deps)
    binaries[role] = {'path': str(binary), 'sha256': sha(binary), 'source': revision, 'build_testing': False, 'RDMA': False, 'diagnostic_CPU_stage_observer': True}

sequence = [('before-baseline', 'baseline'), ('candidate', 'candidate'), ('after-baseline', 'baseline')]
protocol = {
    'sequence': sequence, 'binaries': binaries, 'boot': boot,
    'cmdline': Path('/proc/cmdline').read_text().strip(),
    'harness_sha256': {name: sha(lab / name) for name in ('run.py', 'requests.py', 'analyze.py')},
    'scope': 'One source-instrumented B/C/B attribution screen, first16 normal-controller C2 engine cycles and batch stages. No forced plans/math changes/newGPUwaits. pp2048 stops atEOS96, coding requests128. Logging may change adaptive plans and timing; no release performance/exoneration/adoption claim. Original uninstrumented flags remain authoritative.',
    'minimum_available_kib': 8 * 1024 * 1024,
}
protocol['scope'] = 'Source-instrumented B/C/B first-gap attribution; first16 actual C2 engine/batch cycles, no forced plans/new waits/math changes. pp2048/EOS96 and coding128. Logging may perturb adaptive timing; not release performance or candidate exoneration. Original uninstrumented flags retained.'
protocol['diagnostic_build'] = diagnostic_build
(lab / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
for arm, role in sequence:
    assert available() > 100 * 1024 * 1024, 'Require idle memory headroom before each fresh model start'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
    binary = Path(binaries[role]['path'])
    assert sha(binary) == binaries[role]['sha256']
    command = [str(binary), 'serve', 'llm', '--model',
        '/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
        '--mtp-model', '/home/rkuo/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',
        '--served-model-name', 'flash-next-hc-tp1', '--context', '65536', '--sessions', '2',
        '--prefill-chunk', '2048', '--cache-ram-bytes', '1073741824', '--speculative', 'mtp',
        '--think', 'off', '--host', '127.0.0.1', '--port', str(port), '--temperature', '0', '--seed', '1']
    (lab / f'{arm}-command.json').write_text(json.dumps(command, indent=2) + '\n')
    server = worker = None
    start = time.monotonic()
    minimum = available()
    status = 'failed'
    try:
        with (lab / f'{arm}-server.stdout').open('w') as out, (lab / f'{arm}-server.stderr').open('w') as err:
            server = subprocess.Popen(command, env=env, cwd=root, stdout=out, stderr=err, start_new_session=True)
        (lab / f'{arm}-server.pid').write_text(str(server.pid) + '\n')
        while True:
            assert server.poll() is None, 'Server exited during startup'
            minimum = min(minimum, available())
            assert minimum > 8 * 1024 * 1024 and time.monotonic() - start < 900
            try:
                with urllib.request.urlopen(base + '/v1/models', timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(.25)
        proc = Path('/proc') / str(server.pid)
        assert (proc / 'exe').resolve() == binary
        maps = (proc / 'maps').read_text()
        assert all(x not in maps for x in ('copy-fault.so', 'pread.so', 'librocprofiler-sdk.so'))
        identity = {'pid': server.pid, 'start_ticks': (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19],
                    'sha256': sha(proc / 'exe'), 'cmdline': (proc / 'cmdline').read_bytes().decode().split('\0'),
                    'boot': boot, 'external_observer_absent': True, 'CPU_stage_observer': True}
        (lab / f'{arm}-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        with (lab / f'{arm}-requests.stdout').open('w') as out, (lab / f'{arm}-requests.stderr').open('w') as err:
            worker = subprocess.Popen(['python3', str(lab / 'requests.py'), base, arm], cwd=root, env=env, stdout=out, stderr=err, start_new_session=True)
        while worker.poll() is None:
            minimum = min(minimum, available())
            assert server.poll() is None
            assert minimum > 8 * 1024 * 1024 and time.monotonic() - start < 1200
            time.sleep(.25)
        assert worker.returncode == 0, f'Request worker exited {worker.returncode}'
        assert len((lab / f'{arm}-requests.jsonl').read_text().splitlines()) == 9
        status = 'pass'
    finally:
        exits = {}
        for label, process in [('worker', worker), ('server', server)]:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                exits[label] = process.returncode
        result = {'status': status, 'exits': exits, 'minimum_available_kib': minimum,
                  'elapsed_seconds': time.monotonic() - start, 'boot_unchanged': Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot}
        (lab / f'{arm}-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(arm, json.dumps(result), flush=True)
    assert status == 'pass' and result['boot_unchanged']
(lab / 'runtime-complete').write_text('All three arms completed. Analysis/flags must be inspected separately.\n')
