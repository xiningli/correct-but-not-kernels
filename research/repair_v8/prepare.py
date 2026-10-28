"""Fetch complete pinned source snapshots without executing upstream code."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

FIRST_BATCH = [
    'linkedin/Liger-Kernel#1344', 'linkedin/Liger-Kernel#1376',
    'flagos-ai/FlagGems#553', 'fla-org/flash-linear-attention#1103',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def safe_extract(archive, destination):
    """Files/directories and internal relative symlinks; no archive actions."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        roots = {PurePosixPath(m.name).parts[0] for m in members if m.name}
        if len(roots) != 1:
            raise ValueError('Expected one codeload root')
        total = 0
        links = []
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError('Unsafe source archive path')
            if len(name.parts) == 1:
                continue
            target = destination.joinpath(*name.parts[1:])
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Source path escapes root')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if total > 2 * 1024**3:
                    raise ValueError('Expanded source exceeds cap')
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as stream:
                    while block := source.read(1024 * 1024):
                        stream.write(block)
            elif member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or '\\' in member.linkname:
                    raise ValueError('Unsafe archive link')
                links.append((target, member.linkname))
            else:
                raise ValueError('Unsupported source archive member: ' + member.name)
        for target, link in links:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link)
        for target, _ in links:
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive symlink escapes root')


def fetch_one(item, revision, root, screening):
    repo = item['repo']
    commit = item['before_commit' if revision == 'before' else 'fixed_commit']
    slug = repo.replace('/', '--') + '-' + str(item['number'])
    dest = root / 'sources' / slug / revision
    archive = root / 'downloads' / (repo.replace('/', '--') + '-' + commit + '.tgz')
    url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 256 * 1024**2:
                    raise ValueError('Source download exceeds cap')
                stream.write(block)
        temporary.replace(archive)
    receipt = root / 'source-receipts' / (slug + '-' + revision + '.json')
    if dest.exists() and not receipt.exists():
        raise ValueError('Incomplete snapshot exists; inspect before removing/retrying: ' + str(dest))
    if not dest.exists():
        safe_extract(archive, dest)
    manifest = json.loads((screening / 'archives' / slug / 'manifest.json').read_text())
    verified = []
    for record in manifest['files']:
        if record['path'].startswith(revision + '/'):
            relative = record['path'].split('/', 1)[1]
            p = dest / relative
            if not p.is_file() or sha(p) != record['sha256']:
                raise ValueError('Full checkout disagrees with screened artifact: ' + str(p))
            verified.append(relative)
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    # Imports and test discovery can create caches. Recheck every archived file,
    # without adding those generated outputs to the immutable source inventory.
    paths = [dest / row['path'] for row in previous['files']] if previous else sorted(dest.rglob('*'))
    files = []
    for p in paths:
        if p.is_symlink():
            files.append({'path': str(p.relative_to(dest)), 'symlink': str(p.readlink())})
        elif p.is_file():
            files.append({'path': str(p.relative_to(dest)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    result = {'task': item['id'], 'revision': revision, 'commit': commit, 'url': url,
              'archive': str(archive), 'archive_sha256': sha(archive),
              'directory': str(dest), 'screened_files_verified': verified, 'files': files,
              'status': 'complete_source_snapshot_not_GPU_validated'}
    if previous is not None and previous != result:
        raise ValueError('Prepared source snapshot changed: ' + str(dest))
    write_json(receipt, result)
    print(json.dumps({'task': item['id'], 'revision': revision, 'files': len(files)}), flush=True)
    return {k: v for k, v in result.items() if k != 'files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--screening', type=Path, default=Path('runs/repair-v8-pr-screen-20261018'))
    parser.add_argument('--queue', type=Path, default=Path('evidence/repair-v8-pr-screen-20261018/development-queue.json'))
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())['development']
    selected = [next(x for x in queue if x['id'] == task) for task in FIRST_BATCH]
    write_json(args.root / 'batch-plan.json', {
        'status': 'CPU_preparation_not_GPU_qualification', 'tasks': FIRST_BATCH,
        'selection_reason': 'First operational batch from the fixed development queue: three repositories, bounded APIs, archived tests; no model/GPU outcomes used.',
        'remaining_development_tasks': [x['id'] for x in queue if x['id'] not in FIRST_BATCH],
        'no_formal_evaluation': True, 'no_training': True})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_one, item, revision, args.root, args.screening)
                   for item in selected for revision in ['before', 'fixed']]
        rows = [future.result() for future in futures]
    write_json(args.root / 'source-summary.json', rows)


if __name__ == '__main__':
    main()
