#!/usr/bin/env python3
"""Offline integrity/coverage/statistics audit; does not run GPU quality tests."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / 'archive.json').read_text())
    archive = root / 'acquisition.zip'
    sha = lambda data: hashlib.sha256(data).hexdigest()
    assert sha(archive.read_bytes()) == inventory['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(inventory['entries'])
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert sha(z.read(name)) == inventory['entries'][name], name
        source = root.parents[2] / 'src/models/qwen38_flash_next/kernels/rocm/executor.cpp'
        assert source.read_bytes() == z.read('candidate/executor.cpp')
        quality = json.loads(z.read('candidate/validation.json'))
        assert quality['pass'] and quality['full_verification_frontier_hashes_match_across_ranks']
        for row in quality['ranks']:
            assert row['records'] == 294 and row['full_state_comparisons'] == 28
            assert row['actual_target_rows'] == list(range(1, 9))
            assert row['actual_depths'] == [0, 16384, 32768]
        assert json.loads(z.read('functional/w7-sdk/report.json'))['status'] == 'passed'
        assert 'functional/isolated-complete' in names
        assert z.read('failed-c2-attempt0/baseline-b1/rows.jsonl') == b''
        with tempfile.TemporaryDirectory(prefix='native-mtp-head-audit-') as tmp:
            directory = Path(tmp)
            for group in ('confirmation', 'c2'):
                expected = json.loads(z.read(group + '/analysis.json'))
                for name in names:
                    if name.startswith(group + '/'):
                        target = directory / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read(name))
                subprocess.run([sys.executable, str(directory / group / 'analyze.py')],
                               check=True, stdout=subprocess.DEVNULL)
                actual = json.loads((directory / group / 'analysis.json').read_text())
                assert actual == expected, group + ' statistics differ'
                assert actual['requests'] == (120 if group == 'confirmation' else 52)
    print(f'PASS: {len(names)} hashed files, source identity, retained quality/failure '
          'coverage and recomputed C1/C2 comparisons/statistics')


if __name__ == '__main__':
    main()
