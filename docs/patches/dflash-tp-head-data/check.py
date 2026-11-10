#!/usr/bin/env python3
"""Audit retained evidence offline. Does not run GPUs or independently prove quality."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / 'archive.json').read_text())
    archive = root / 'acquisition.zip'
    assert sha(archive.read_bytes()) == inventory['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(inventory['entries'])
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            data = z.read(name)
            assert sha(data) == inventory['entries'][name], name
            assert b'-----BEGIN OPENSSH PRIVATE KEY-----' not in data, name
        source = root.parents[2] / 'src/models/qwen38_flash_next/kernels/rocm/executor.cpp'
        assert source.read_bytes() == z.read('production-source.cpp')
        state = json.loads(z.read('state/validation.json'))
        assert state['records_per_rank'] == 294
        assert state['full_state_pairs_per_rank'] == 28
        assert state['same_all_verification_and_frontier_hashes_across_ranks']
        assert state['actual_depths'] == [0, 16384, 32768]
        assert state['candidate_target_rows'] == [4, 8]
        for rank in ('0', '1'):
            assert len(state['state_pair_logs'][rank]) == 28
        behavior = json.loads(z.read('functional/validation.json'))
        assert behavior['candidate_behavior_checks_pass']
        assert behavior['original_unmodified_w3_aggregate_report'] == 'failed, retained'
        assert 'functional/DISK-SPACING-FAILURE.md' in names
        assert 'functional/isolated-complete' in names
        expected = json.loads(z.read('confirmation/analysis.json'))
        assert expected['requests'] == 200
        assert expected['all_output_work_identities_match']
        assert len(expected['phase_flags']) == 14
        assert len(expected['transport_errors']) == 16
        # Run the archived analysis, using only hashed, safe confirmation entries.
        # It rechecks output hashes/work identity and recomputes all statistics.
        with tempfile.TemporaryDirectory(prefix='dflash-tp-head-audit-') as tmp:
            directory = Path(tmp)
            for name in names:
                if name.startswith('confirmation/'):
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
            subprocess.run([sys.executable, str(directory / 'confirmation/analyze.py')],
                           check=True, stdout=subprocess.DEVNULL)
            actual = json.loads((directory / 'confirmation/analysis.json').read_text())
            assert actual == expected, 'offline statistics differ'
        assert all(r['resolved_positive_decode_gain'] for r in expected['results'])
    print(f'PASS: {len(names)} hashed files, production source identity, retained '
          'quality/failure records, 200-request comparisons and recomputed statistics')


if __name__ == '__main__':
    main()
