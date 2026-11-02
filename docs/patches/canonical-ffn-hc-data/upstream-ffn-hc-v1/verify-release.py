from pathlib import Path
import hashlib
import json
import subprocess

lab = Path(__file__).resolve().parent
root = lab.parents[1]
sources = {
    'baseline': root / 'tp-cache-retention/upstream-ffn-hc-baseline-worktree',
    'candidate': root / 'tp-cache-retention/upstream-ffn-hc-worktree',
}
commands = {}
binaries = {}
for role, source in sources.items():
    build = source.with_name(source.name.replace('-worktree', '-release-build'))
    binary = build / 'gufo'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    normalized = {}
    for entry in json.loads((build / 'compile_commands.json').read_text()):
        file = entry['file'].replace(str(source), 'SOURCE').replace(str(build), 'BUILD')
        command = entry['command'].replace(str(source), 'SOURCE').replace(str(build), 'BUILD')
        # Revision metadata belongs to version translation units, not compiler tuning.
        import re
        command = re.sub(r'GUFO_REVISION=[^ ]+', 'GUFO_REVISION=REVISION', command)
        assert file not in normalized, file
        normalized[file] = command
    commands[role] = normalized
    binaries[role] = {
        'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_tree': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
        'full_fresh_release': True, 'BUILD_TESTING': False, 'RDMA': False,
    }
assert commands['baseline'].keys() == commands['candidate'].keys()
different = [name for name in commands['baseline'] if commands['baseline'][name] != commands['candidate'][name]]
assert not different, different
changed = subprocess.check_output(['git', '-C', str(sources['candidate']), 'diff', '--name-only', binaries['baseline']['source'], '--', 'src', 'include', 'cmake'], text=True).splitlines()
assert changed == ['src/models/qwen38_flash_next/kernels/rocm/batch.cpp'], changed
result = {'status': 'pass', 'compile_units': len(commands['baseline']), 'normalized_command_differences': different,
          'changed_production_paths': changed, 'binaries': binaries,
          'scope': 'Fresh stock canonical baseline and common-code candidate production release builds; normalized source/build prefixes and revision strings only. No RDMA or diagnostic objects.'}
(lab / 'release-source-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
