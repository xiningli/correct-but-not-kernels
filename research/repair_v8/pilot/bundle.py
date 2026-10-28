"""Build a hash-verified investigator package with portable relative paths."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from research.repair_v8.prepare import sha
from .artifacts import write_json
from .prepare import load, PROJECT


def bundle(prepared, output):
    manifest, _ = load(prepared)
    files = {row['path']: PROJECT / row['path'] for row in manifest['code']}
    for p in sorted(prepared.rglob('*')):
        if '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and (p.is_file() or p.is_symlink()):
            files['prepared/' + str(p.relative_to(prepared))] = p
    rows = []
    for name, p in sorted(files.items()):
        if p.is_symlink():
            if not p.resolve().is_relative_to(prepared):
                raise ValueError('Nonportable source link')
            rows.append({'path': name, 'symlink': str(p.readlink())})
        else:
            rows.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    info = {'scope': manifest['scope'], 'files': rows, 'contains_gold_control_modules': True,
            'mount_entire_package_into_candidate': False,
            'candidate_access': 'Only an isolated before checkout with the selected edited module and current input tensors.',
            'startup': 'bash scripts/repair_v8_pilot_autodl.sh prepare; bash scripts/repair_v8_pilot_autodl.sh run'}
    encoded = (json.dumps(info, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output, 'w:gz') as tar:
        for name, p in sorted(files.items()):
            tar.add(p, arcname=name, recursive=False)
        entry = tarfile.TarInfo('PORTABLE_PACKAGE.json'); entry.size = len(encoded)
        tar.addfile(entry, io.BytesIO(encoded))
    with tarfile.open(output, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files) + 1 or len({m.name for m in members}) != len(members):
            raise ValueError('Archive member mismatch')
        for row in rows:
            member = tar.getmember(row['path'])
            if 'symlink' in row:
                if not member.issym() or member.linkname != row['symlink']:
                    raise ValueError('Archive link mismatch')
            elif hashlib.sha256(tar.extractfile(member).read()).hexdigest() != row['sha256']:
                raise ValueError('Archive file mismatch')
    receipt = {'path': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size,
               'files_and_symlinks': len(files), 'readback_verified': True,
               'preparation_sha256': sha(prepared / 'manifest.json'), 'GPU_runs': 0, 'model_runs': 0,
               'contains_gold_control_modules': True, 'scope': manifest['scope']}
    write_json(output.with_suffix('.receipt.json'), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(bundle(a.prepared.resolve(), a.out.resolve()), indent=2))


if __name__ == '__main__':
    main()
