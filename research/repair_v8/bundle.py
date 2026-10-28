"""Build a verified investigator-only AutoDL source package, without a venv."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .cases import TASKS
from .prepare import sha, write_json
from .qualify import PROJECT, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    relative_root = root.relative_to(PROJECT)
    plan = json.loads((root / 'qualification-plan.json').read_text())
    verify_files(PROJECT, plan['files'])
    paths = list((PROJECT / 'research/repair_v8').glob('*.py'))
    paths += [PROJECT / p for p in ['scripts/repair_v8_autodl.sh', 'scripts/repair_v8_cpu_preflight.py',
              'experiments/repair-v8-qualification.in', 'experiments/repair-v8-qualification.lock',
              'tests/test_repair_v8_qualification.py']]
    paths += [root / p for p in ['batch-plan.json', 'source-summary.json']]
    for task in TASKS.values():
        for revision in ['before', 'fixed']:
            receipt = root / 'source-receipts' / (task['slug'] + '-' + revision + '.json')
            data = json.loads(receipt.read_text())
            source = root / 'sources' / task['slug'] / revision
            verify_files(source, data['files'])
            paths.append(receipt)
            paths.extend(source / row['path'] for row in data['files'])
    rows = []
    for path in sorted(set(paths)):
        entry = {'path': str(path.relative_to(PROJECT))}
        if path.is_symlink():
            entry['symlink'] = str(path.readlink())
        else:
            entry.update(bytes=path.stat().st_size, sha256=sha(path))
        rows.append(entry)
    manifest = {'schema': 'repair-v8-investigator-batch1-package-v1', 'files': rows,
                'run_root': str(relative_root), 'scope': 'Four development tasks; no GPU validation yet',
                'contains_gold_fixes': True, 'mount_in_model_container': False,
                'included': 'Pinned original source, licenses, qualification code, hashed dependency lock',
                'not_included': 'venv, dependency wheels, models, caches, GPU results, old experiments',
                'startup': 'bash scripts/repair_v8_autodl.sh prepare; bash scripts/repair_v8_autodl.sh run',
                'portability': 'prepare rebuilds execution paths and plan locally from these exact inputs; no GPU outcomes used'}
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    output = root / 'autodl-batch1-investigator.tgz'
    if output.exists():
        raise FileExistsError('Preserve the previous bundle before rebuilding')
    with tarfile.open(output, 'w:gz') as tar:
        for row in rows:
            tar.add(PROJECT / row['path'], arcname=row['path'], recursive=False)
        info = tarfile.TarInfo('PORTABLE_PACKAGE.json'); info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
    with tarfile.open(output, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(rows) + 1
        assert len({m.name for m in members}) == len(members)
        for row in rows:
            member = tar.getmember(row['path'])
            if 'symlink' in row:
                assert member.issym() and member.linkname == row['symlink']
            else:
                assert member.isfile() and member.size == row['bytes']
                assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == row['sha256']
        assert tar.extractfile('PORTABLE_PACKAGE.json').read() == encoded
    receipt = {'path': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size,
               'files_and_symlinks': len(rows), 'readback_verified': True,
               'package_manifest_sha256': hashlib.sha256(encoded).hexdigest(),
               'contains_gold_fixes': True, 'mount_in_model_container': False}
    write_json(root / 'bundle-receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
