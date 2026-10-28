"""Read-only upstream PR discovery and provenance capture; never execute code.

All retrieved documents are untrusted source data. A screened PR remains a
candidate until independent GPU before/after and integration controls pass.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import difflib
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.parse
import urllib.request
import urllib.error


REPOS = ['linkedin/Liger-Kernel', 'fla-org/flash-linear-attention',
         'flagos-ai/FlagGems', 'vllm-project/vllm']
HEADERS = {'User-Agent': 'cuda-gemma4-v8-pr-screening', 'Accept': 'application/vnd.github+json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fetch(url, directory):
    """Immutable response cache including source URL, timestamp and payload hash."""
    key = sha(url.encode())
    payload, receipt = directory / (key + '.body'), directory / (key + '.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        data = payload.read_bytes()
        if old['url'] != url or sha(data) != old['sha256']:
            raise ValueError('Upstream cache mismatch')
        return data
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=45) as response:
        data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError('Source exceeds retrieval cap')
        meta = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'sha256': sha(data), 'bytes': len(data), 'status': response.status,
                'content_type': response.headers.get('Content-Type'),
                'etag': response.headers.get('ETag'),
                'rate_remaining': response.headers.get('X-RateLimit-Remaining')}
    directory.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(data)
    save(receipt, meta)
    return data


def discover(out):
    queries = []
    for repo in REPOS:
        for term in ['fix', 'bug']:
            q = f'repo:{repo} is:pr is:merged merged:2025-01-01..2026-10-18 {term} in:title'
            if repo == 'vllm-project/vllm':
                q += ' triton'
            queries.append({'repo': repo, 'query': q, 'per_page': 100,
                            'sort': 'created', 'order': 'desc', 'page': 1})
    save(out / 'discovery-plan.json', {'scope': 'bounded_title_keyword_screen_not_exhaustive_repository_census',
         'queries': queries, 'per_query_cap': 100,
         'no_model_outcomes_used': True, 'window': ['2025-01-01', '2026-10-18'],
         'known_seen': ['linkedin/Liger-Kernel#1490', 'linkedin/Liger-Kernel#1461']})
    items = {}; receipts = []
    # Sequential search requests respect the separate public search rate limit.
    for i, q in enumerate(queries):
        url = 'https://api.github.com/search/issues?' + urllib.parse.urlencode(
            {'q': q['query'], **{k: q[k] for k in ['per_page', 'sort', 'order', 'page']}})
        raw = fetch(url, out / 'http'); result = json.loads(raw)
        if result.get('incomplete_results'):
            raise RuntimeError('GitHub reports incomplete search execution; retain response and retry explicitly')
        receipts.append({**q, 'url': url, 'response_sha256': sha(raw),
                         'total_count': result['total_count'], 'retrieved': len(result['items']),
                         'truncated_by_cap': result['total_count'] > 100})
        for item in result['items']:
            key = q['repo'] + '#' + str(item['number'])
            if key not in items:
                items[key] = {'id': key, 'repo': q['repo'], 'number': item['number'],
                    'url': item['html_url'], 'title': item['title'], 'body': item.get('body'),
                    'created_at': item['created_at'], 'closed_at': item['closed_at'],
                    'labels': [x['name'] for x in item['labels']], 'queries': []}
            items[key]['queries'].append(i)
        save(out / 'search-receipts.json', receipts)
        save(out / 'discovered.json', list(items.values()))
        print(json.dumps({'query': i, 'repo': q['repo'], 'term': q['query'],
                          'total': result['total_count'], 'retrieved': len(result['items'])}), flush=True)
    return list(items.values())


def inspect_one(item, out):
    repo, number = item['repo'], item['number']
    raw = fetch(f'https://api.github.com/repos/{repo}/pulls/{number}', out / 'http')
    pr = json.loads(raw)
    diff = fetch(f'https://github.com/{repo}/pull/{number}.diff', out / 'http')
    dest = out / 'inspected' / (repo.replace('/', '--') + '-' + str(number))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'pull.json').write_bytes(raw); (dest / 'pull.diff').write_bytes(diff)
    summary = {'id': repo + '#' + str(number), 'repo': repo, 'number': number,
        'url': pr['html_url'], 'title': pr['title'], 'merged': pr['merged'],
        'merged_at': pr['merged_at'], 'merge_commit_sha': pr['merge_commit_sha'],
        'base_sha_at_query': pr['base']['sha'], 'head_sha': pr['head']['sha'],
        'changed_files': pr['changed_files'], 'additions': pr['additions'], 'deletions': pr['deletions'],
        'pull_sha256': sha(raw), 'diff_sha256': sha(diff),
        'paths': re.findall(r'^diff --git a/(.+) b/.+$', diff.decode(), re.M),
        'directory': str(dest), 'status': 'inspected_metadata_and_diff_only'}
    save(dest / 'summary.json', summary)
    return summary


def inspect(out, selection):
    items = json.loads(selection.read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(item, pool.submit(inspect_one, item, out)) for item in items]
        rows = []
        for item, future in futures:
            try:
                row = future.result()
                print(json.dumps({k: row[k] for k in ['id', 'merged', 'changed_files', 'title']}), flush=True)
            except Exception as error:
                row = {'id': item['repo'] + '#' + str(item['number']),
                       'status': 'retrieval_failed', 'error': type(error).__name__ + ': ' + str(error)}
                print(json.dumps(row), flush=True)
            rows.append(row)
    save(out / ('inspection-' + selection.stem + '.json'), rows)


def safe_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Unsafe upstream path')
    return path


def archive_one(item, out):
    """Pin the actual merge's first parent; a PR's base.sha is NOT that parent.

    Includes full changed text files, test clues and dependency declarations.
    These declarations are not an installed environment lock or GPU validation.
    Gold and tests here are investigator-only; never expose this archive to a model.
    """
    repo, number = item['repo'], item['number']
    slug = repo.replace('/', '--') + '-' + str(number)
    inspected = out / 'inspected' / slug
    pr = json.loads((inspected / 'pull.json').read_text())
    if not pr.get('merged') or not pr.get('merge_commit_sha'):
        raise ValueError('Not a merged PR')
    fixed = pr['merge_commit_sha']
    commit_raw = fetch(f'https://api.github.com/repos/{repo}/commits/{fixed}', out / 'http')
    commit = json.loads(commit_raw)
    if commit['sha'] != fixed or not commit.get('parents'):
        raise ValueError('Missing or inconsistent ancestry')
    parent = commit['parents'][0]['sha']
    dest = out / 'archives' / slug
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['pull.json', 'pull.diff', 'summary.json']:
        (dest / name).write_bytes((inspected / name).read_bytes())
    (dest / 'merge-commit.json').write_bytes(commit_raw)
    paths = json.loads((inspected / 'summary.json').read_text())['paths']
    required = set(item['owner_modules'])
    paths = sorted(set(paths) | required | set(item.get('extra_paths', [])) |
                   {'LICENSE', 'pyproject.toml'})
    assets, unavailable = [], []
    for revision_name, revision in [('before', parent), ('fixed', fixed)]:
        for path in paths:
            safe_path(path)
            if not path.endswith(('.py', '.toml', '.yaml', '.yml', '.txt', '.md')) and path != 'LICENSE':
                unavailable.append({'revision': revision_name, 'path': path, 'reason': 'non_text_file_not_fetched'})
                continue
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/{urllib.parse.quote(path)}'
            try:
                raw = fetch(url, out / 'http')
            except urllib.error.HTTPError as error:
                if error.code != 404 or path in required or path == 'LICENSE':
                    raise
                unavailable.append({'revision': revision_name, 'path': path, 'reason': 'HTTP_404', 'url': url})
                continue
            target = dest / revision_name / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            assets.append({'path': str(target.relative_to(dest)), 'sha256': sha(raw),
                           'bytes': len(raw), 'url': url})
    for name in ['pull.json', 'pull.diff', 'summary.json', 'merge-commit.json']:
        raw = (dest / name).read_bytes()
        assets.append({'path': name, 'sha256': sha(raw), 'bytes': len(raw)})
    manifest = {'schema': 'cuda-repair-v8-pr-source-archive-v1', 'id': repo + '#' + str(number),
        'repo': repo, 'number': number, 'url': pr['html_url'], 'merged_at': pr['merged_at'],
        'before_commit': parent, 'fixed_commit': fixed, 'merge_parents': [p['sha'] for p in commit['parents']],
        'status': 'source_archived_not_GPU_validated', 'owner_modules_provisional': item['owner_modules'],
        'scope_warning': 'Owning modules are provisional; freeze from before-source issue/stack and call graph, never gold lines.',
        'dependency_warning': 'Upstream declarations only; full repository checkout and reproducible environment lock still required.',
        'visibility': 'investigator_only_includes_gold_and_regression_tests',
        'files': assets, 'unavailable': unavailable}
    save(dest / 'manifest.json', manifest)
    verify_archive(dest)
    return {'id': manifest['id'], 'directory': str(dest), 'files': len(assets),
            'before_commit': parent, 'fixed_commit': fixed, 'manifest_sha256': sha((dest / 'manifest.json').read_bytes())}


def verify_archive(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for row in manifest['files']:
        path = directory / safe_path(row['path'])
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Archive path escapes root')
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
            raise ValueError('Archive hash mismatch: ' + row['path'])
    return len(manifest['files'])


def archive(out, selection):
    items = json.loads(selection.read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(x, pool.submit(archive_one, x, out)) for x in items]
        for item, future in futures:
            try:
                row = future.result()
            except Exception as error:
                row = {'id': item['repo'] + '#' + str(item['number']),
                       'status': 'archive_failed', 'error': type(error).__name__ + ': ' + str(error)}
            rows.append(row)
            save(out / 'archive-receipts.json', rows)
            print(json.dumps(row), flush=True)


def measure_scopes(out, tokenizer_path):
    """Count complete module responses with the pinned local model tokenizer.

    Parsing is syntax-only. No upstream code is imported or executed.
    """
    from tokenizers import Tokenizer
    if sha(tokenizer_path.read_bytes()) != 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f':
        raise ValueError('Tokenizer does not match the pinned Gemma-4-31B snapshot')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = []
    for path in sorted((out / 'archives').glob('*/manifest.json')):
        directory = path.parent
        verify_archive(directory)
        manifest = json.loads(path.read_text())
        row = {'id': manifest['id'], 'modules': manifest['owner_modules_provisional']}
        diffs = []
        for revision in ['before', 'fixed']:
            files = []
            for name in row['modules']:
                source = (directory / revision / name).read_text()
                ast.parse(source, filename=name)
                files.append({'path': name, 'source': source})
                if revision == 'fixed':
                    before = (directory / 'before' / name).read_text()
                    if before == source:
                        raise ValueError('Provisional edited module did not change: ' + name)
                    diffs.extend(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                                                     fromfile='before/' + name, tofile='fixed/' + name))
            response = json.dumps({'files': files}, ensure_ascii=False, separators=(',', ':'))
            row[revision + '_full_response_tokens'] = len(tokenizer.encode(response, add_special_tokens=False).ids)
        diff_path = out / 'merge-parent-diffs' / (directory.name + '.diff')
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(''.join(diffs))
        row.update({'before_within_6144': row['before_full_response_tokens'] <= 6144,
                    'official_fix_within_8192': row['fixed_full_response_tokens'] <= 8192,
                    'syntax_parse': 'pass_before_and_fixed_no_execution',
                    'actual_merge_parent_diff': str(diff_path),
                    'actual_merge_parent_diff_sha256': sha(diff_path.read_bytes())})
        rows.append(row)
    save(out / 'scope-token-counts.json', {
        'tokenizer_model': 'google/gemma-4-31B-it',
        'tokenizer_revision': '842da3794eaa0b77d5f08bae87a17459d91ff475',
        'tokenizer_sha256': sha(tokenizer_path.read_bytes()),
        'wire_format': 'compact JSON: files[{path,source}], ensure_ascii=False, add_special_tokens=False',
        'note': 'Screening only. Full prompt, API card, tests and generation framing still require pre-freeze audit. Whole original modules retained.',
        'tasks': rows})
    print(json.dumps({'complete_module_pairs_parsed_and_counted': len(rows)}))


def finalize(out, review_path, evidence):
    """Join explicit review decisions with source evidence, never infer eligibility.

    Unreviewed discovery results stay deferred. Assignment uses root IDs and
    repository strata, never a model result or a search for a favorable seed.
    """
    review = json.loads(review_path.read_text())
    manual = {x['id']: x for x in review['reviews']}
    if len(manual) != len(review['reviews']):
        raise ValueError('Duplicate manual decision')
    discovered = json.loads((out / 'discovered.json').read_text())
    scopes = {x['id']: x for x in json.loads((out / 'scope-token-counts.json').read_text())['tasks']}
    archives = {}
    for p in sorted((out / 'archives').glob('*/manifest.json')):
        count = verify_archive(p.parent)
        m = json.loads(p.read_text())
        archives[m['id']] = (p, m, count)
    ledger, eligible = [], []
    seen = {'linkedin/Liger-Kernel#1490', 'linkedin/Liger-Kernel#1461'}
    for item in discovered:
        row = {k: item[k] for k in ['id', 'repo', 'number', 'url', 'title', 'queries']}
        row.update({'review_level': 'search_metadata_only', 'decision': 'deferred_not_deep_reviewed',
                    'reason': 'Not selected for the bounded deep-review batch; no ineligibility conclusion.'})
        if item['id'] in seen:
            row.update({'decision': 'excluded_prior_development', 'reason': 'Already used in previous model/harness development.'})
        if item['id'] in manual:
            row.update(manual[item['id']])
            row['review_level'] = 'PR_body_and_diff_review'
            d = out / 'inspected' / (item['repo'].replace('/', '--') + '-' + str(item['number']))
            s = json.loads((d / 'summary.json').read_text())
            row.update({k: s[k] for k in ['merged', 'merged_at', 'changed_files', 'paths', 'pull_sha256', 'diff_sha256']})
            if not s['merged'] or not ('2025-01-01' <= s['merged_at'][:10] <= '2026-10-18'):
                raise ValueError('Deep review outside merged window')
        if item['id'] in archives:
            p, m, count = archives[item['id']]
            row.update({'archive_manifest': str(p), 'archive_manifest_sha256': sha(p.read_bytes()),
                        'before_commit': m['before_commit'], 'fixed_commit': m['fixed_commit'],
                        'owner_modules_provisional': m['owner_modules_provisional'],
                        'archived_files_verified': count, 'source_scope': scopes[item['id']],
                        'archived_test_files': [x['path'] for x in m['files'] if x['path'].startswith(('before/test/', 'before/tests/', 'fixed/test/', 'fixed/tests/'))]})
        row['GPU_qualification'] = 'not_run'
        if row['decision'] == 'cpu_candidate':
            if item['id'] not in archives or not scopes[item['id']]['before_within_6144'] or not scopes[item['id']]['official_fix_within_8192']:
                raise ValueError('CPU candidate lacks archive or fair whole-module budget: ' + item['id'])
            if item['repo'] not in REPOS or not 1 <= len(scopes[item['id']]['modules']) <= 3:
                raise ValueError('CPU candidate outside repository or module-count scope')
            row['assignment_hash'] = sha((review['split_seed'] + '\n' + row['root_cluster']).encode())
            eligible.append(row)
        ledger.append(row)
    if set(manual) - {x['id'] for x in discovered}:
        raise ValueError('Reviewed item missing discovery provenance')
    if len({x['root_cluster'] for x in eligible}) != len(eligible):
        raise ValueError('Root cluster counted twice')
    grouped = defaultdict(list)
    for row in sorted(eligible, key=lambda x: x['assignment_hash']):
        grouped[row['repo']].append(row)
    chosen = []
    while len(chosen) < min(review['development_slots'], len(eligible)):
        for repo in REPOS:
            if grouped[repo] and len(chosen) < review['development_slots']:
                chosen.append(grouped[repo].pop(0))
    chosen_ids = {x['id'] for x in chosen}
    reserve = [x for repo in REPOS for x in grouped[repo]]
    for row in eligible:
        row['split'] = 'development_candidate' if row['id'] in chosen_ids else 'development_reserve'
        row['formal_test_eligible'] = False
    receipts = json.loads((out / 'search-receipts.json').read_text())
    summary = {
        'status': 'bounded_CPU_PR_screen_complete_GPU_qualification_not_started',
        'review_notes': str(review_path), 'review_notes_sha256': sha(review_path.read_bytes()),
        'discovered_unique_PRs': len(discovered), 'deep_reviewed_PRs': len(manual),
        'deferred_metadata_only': sum(x['decision'] == 'deferred_not_deep_reviewed' for x in ledger),
        'decision_counts': dict(Counter(x['decision'] for x in ledger)),
        'archives': len(archives), 'archived_files_verified': sum(x[2] for x in archives.values()),
        'CPU_candidates': len(eligible), 'development_candidates': len(chosen),
        'development_reserves': len(reserve), 'GPU_qualified': 0, 'formal_tasks_sealed': 0,
        'development_by_repository': dict(Counter(x['repo'] for x in chosen)),
        'search_queries': len(receipts), 'capped_queries': sum(x['truncated_by_cap'] for x in receipts),
        'split_seed': review['split_seed'], 'assignment': review['assignment'],
        'replacement': review['replacement'], 'shared_pending_gates': review['shared_pending_gates'],
        'model_runs': 0, 'GPU_runs': 0, 'upstream_code_executed': False,
        'qualification_labor': 'Not measured as person-hours. Retrieval receipts preserve timestamps; these are not human labor or GPU time.',
        'limitations': ['Title-keyword, latest-created, capped discovery is not representative random sampling or an exhaustive PR census.',
                       'All CPU candidates here are development or development reserve, never the formal test set.',
                       'Merged source and upstream claimed tests do not prove independently reproducible correctness.',
                       'Scope holds from PR-head estimates are preliminary, not definitive merge-parent exclusions.',
                       'Public-source pretraining contamination cannot be ruled out; recent merges do not prove unfamiliarity.',
                       'Full repository checkout, exact environment lock, independent tests, integration and GPU controls remain pending.']}
    evidence.mkdir(parents=True, exist_ok=True)
    save(evidence / 'screening-summary.json', summary)
    save(evidence / 'candidate-ledger.json', sorted(ledger, key=lambda x: x['id']))
    save(evidence / 'development-queue.json', {'status': 'provisional_CPU_candidates_not_execution_ready',
         'development': chosen, 'reserves': reserve, 'shared_pending_gates': review['shared_pending_gates'],
         'duplicate_groups': review['duplicate_groups']})
    for name in ['discovery-plan-original.json', 'discovery-plan.json', 'discovery-amendment.json',
                 'search-receipts.json', 'scope-token-counts.json', 'archive-receipts.json',
                 'scope-measurements-head.json', 'scope-measurements-supplement.json']:
        if (out / name).exists():
            (evidence / name).write_bytes((out / name).read_bytes())
    save(evidence / 'archive-verification.json', [
        {'id': key, 'manifest_sha256': sha(p.read_bytes()), 'files_verified': count}
        for key, (p, m, count) in sorted(archives.items())])
    print(json.dumps({k: summary[k] for k in ['discovered_unique_PRs', 'deep_reviewed_PRs', 'CPU_candidates',
                                           'development_candidates', 'development_reserves', 'archived_files_verified']}))


def bundle_sources(out, evidence):
    """Portable investigator evidence with normalized tar metadata and SHA256.

    Never mount this gold-containing bundle inside a model/repair container.
    """
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence / 'investigator-source-bundle.tgz'
    inventory = []
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    with target.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Source bundle contains symlink')
                    name = str(path.relative_to(out))
                    raw = path.read_bytes()
                    inventory.append({'path': name, 'bytes': len(raw), 'sha256': sha(raw)})
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    info.pax_headers = {}
                    with path.open('rb') as source:
                        tar.addfile(info, source)
    expected = {x['path']: x for x in inventory}
    verified = set()
    # Read every bundled byte back without extracting or executing anything.
    with tarfile.open(target, 'r:gz') as tar:
        for member in tar:
            safe_path(member.name)
            if not member.isfile() or member.name in verified or member.name not in expected:
                raise ValueError('Unexpected bundle member')
            raw = tar.extractfile(member).read()
            row = expected[member.name]
            if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
                raise ValueError('Bundle readback mismatch')
            verified.add(member.name)
    if verified != set(expected):
        raise ValueError('Bundle missing a file')
    save(evidence / 'source-bundle-inventory.json', inventory)
    save(evidence / 'source-bundle-receipt.json', {
        'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target.read_bytes()),
        'files_readback_verified': len(verified), 'inventory_sha256': sha((evidence / 'source-bundle-inventory.json').read_bytes()),
        'visibility': 'investigator_only_gold_tests_PR_bodies_and_future_history_do_not_expose_to_models',
        'not_a_complete_repository_checkout': True})
    print(json.dumps({'bundle_bytes': target.stat().st_size, 'files_readback_verified': len(verified)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['discover', 'inspect', 'archive', 'verify', 'scope', 'finalize', 'bundle'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'discover':
        discover(args.out)
    elif args.phase == 'scope':
        if not args.tokenizer:
            parser.error('scope requires --tokenizer')
        measure_scopes(args.out, args.tokenizer)
    elif args.phase == 'finalize':
        if not args.review or not args.evidence:
            parser.error('finalize requires --review and --evidence')
        finalize(args.out, args.review, args.evidence)
    elif args.phase == 'bundle':
        if not args.evidence:
            parser.error('bundle requires --evidence')
        if args.evidence.resolve().is_relative_to(args.out.resolve()):
            parser.error('bundle destination must be outside source directory')
        bundle_sources(args.out, args.evidence)
    elif args.phase == 'verify':
        rows = [{'directory': str(p), 'files_verified': verify_archive(p)}
                for p in sorted((args.out / 'archives').iterdir()) if (p / 'manifest.json').exists()]
        save(args.out / 'archive-verification.json', rows)
        print(json.dumps({'archives_verified': len(rows), 'files_verified': sum(x['files_verified'] for x in rows)}))
    else:
        if not args.selection:
            parser.error(args.phase + ' requires --selection')
        (inspect if args.phase == 'inspect' else archive)(args.out, args.selection)


if __name__ == '__main__':
    main()
