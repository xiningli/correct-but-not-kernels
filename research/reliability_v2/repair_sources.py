"""K4/T2: archive real bug/fix provenance without executing upstream code.

A fetched PR is a candidate, NEVER an eligible test task. Eligibility requires
review plus original-fail/fixed-pass receipts in the same GPU environment.
All raw files stay in a new immutable destination; hashes are suitable for
later inclusion in the sealed task manifest. Parent means the first parent of
the merged commit, not the moving target branch in the PR metadata.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'cuda-gemma4-research',
                                              'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=40) as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Upstream file exceeds source-archive limit')
    return data


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:
        raise ValueError('Unsafe upstream path')
    return p


def archive_pr(repo, number, paths, destination):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Expected owner/repository')
    if type(number) is not int or number < 1 or not paths or len(paths) != len(set(paths)):
        raise ValueError('Expected a PR and unique source paths')
    paths = [str(safe_relative(p)) for p in paths]
    api = 'https://api.github.com/repos/' + repo
    pr_bytes = fetch(f'{api}/pulls/{number}')
    pr = json.loads(pr_bytes)
    if not pr.get('merged') or not pr.get('merge_commit_sha'):
        raise ValueError('Only verified merged fixes are eligible for provenance capture')
    commit_bytes = fetch(f'{api}/commits/{pr["merge_commit_sha"]}')
    commit = json.loads(commit_bytes)
    if not commit.get('parents'):
        raise ValueError('Cannot establish pre-fix revision')
    parent, fixed = commit['parents'][0]['sha'], commit['sha']
    files_bytes = fetch(f'{api}/pulls/{number}/files?per_page=100')
    changed = {x['filename']: x for x in json.loads(files_bytes)}
    if any(p not in changed or changed[p]['status'] != 'modified' for p in paths):
        raise ValueError('Requested kernel must be modified in this PR')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = {}

    def save(name, data, url):
        p = out / safe_relative(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        artifacts[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'url': url}

    save('pull.json', pr_bytes, pr['html_url'])
    save('commit.json', commit_bytes, commit['html_url'])
    save('changed-files.json', files_bytes, f'{api}/pulls/{number}/files?per_page=100')
    for revision, label in [(parent, 'before'), (fixed, 'after')]:
        for path in paths:
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
            save(label + '/' + path, fetch(url), url)
    license_url = f'{api}/license?ref={fixed}'
    license_bytes = fetch(license_url)
    lic = json.loads(license_bytes)
    if lic.get('encoding') != 'base64' or not lic.get('content'):
        raise ValueError('No archived license text; task remains incomplete')
    save('license-metadata.json', license_bytes, license_url)
    save('UPSTREAM_LICENSE', base64.b64decode(lic['content']), lic.get('html_url', license_url))
    manifest = {'schema': 1, 'status': 'candidate_not_GPU_validated',
                'origin': 'real_upstream_fix', 'repo': repo, 'pull_request': number,
                'url': pr['html_url'], 'title': pr['title'], 'author': pr['user']['login'],
                'merged_at': pr['merged_at'], 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'parent': parent, 'fixed': fixed, 'source_paths': paths,
                'license_spdx': lic.get('license', {}).get('spdx_id'), 'artifacts': artifacts,
                'eligibility_missing': ['manual_fault_and_license_review', 'adaptation_diff',
                    'independent_visible_and_hidden_regressions', 'same_environment_before_fail_after_pass',
                    'duplicate_defect_and_train_test_overlap_review'],
                'contamination': 'Public upstream history; absence from pretraining is not established.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_archive(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, record in manifest['artifacts'].items():
        p = directory / safe_relative(name)
        if p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Archive file escapes root')
        data = p.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Upstream artifact drift: ' + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--pr', required=True, type=int)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    manifest = archive_pr(args.repo, args.pr, args.source, args.out)
    print(json.dumps({k: manifest[k] for k in ['status', 'url', 'parent', 'fixed', 'license_spdx']}))


if __name__ == '__main__':
    main()
