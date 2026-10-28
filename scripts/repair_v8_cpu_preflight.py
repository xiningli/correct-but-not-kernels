"""Import and collect the matched original-repository suites without a GPU."""
import argparse
import json
import sys
from pathlib import Path

from research.repair_v8.cases import TASKS, load_owner
from research.repair_v8.prepare import write_json
from research.repair_v8.qualify import environment, execute, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'cpu-preflight'
    rows = []
    for key, task in TASKS.items():
        for revision in ['before', 'fixed']:
            checkout = root / 'execution' / task['slug'] / revision
            path = output / (key + '-' + revision + '.json')
            env = environment(checkout, task, path)
            env['CUDA_VISIBLE_DEVICES'] = ''
            # Arguments carry paths; no interpolated executable source text.
            statement = ('import sys; from pathlib import Path; '
                         'from research.repair_v8.cases import load_owner; '
                         'print(load_owner(sys.argv[1], Path(sys.argv[2])).__file__)')
            imported = execute([sys.executable, '-c', statement, key, str(checkout)],
                               checkout, env, path.with_suffix('.import.log'), 120)
            collected = execute([sys.executable, '-m', 'pytest', '-p', 'research.repair_v8.pytest_receipt',
                                 '--collect-only', '-q', *task['upstream_tests']],
                                checkout, env, path.with_suffix('.collection.log'), 120)
            receipt = json.loads(path.read_text()) if path.exists() else {}
            manifest = json.loads((root / 'execution-manifests' / (task['slug'] + '-' + revision + '.json')).read_text())
            verify_files(checkout, manifest['files'])
            rows.append({'task': key, 'revision': revision, 'import': imported, 'collection': collected,
                         'collected': receipt.get('collected', 0), 'receipt': receipt})
    matched = all(len({r['collected'] for r in rows if r['task'] == key}) == 1 for key in TASKS)
    passed = matched and all(r['import']['returncode'] == 0 and r['collection']['returncode'] == 0
                             and r['collected'] > 0 and r['receipt']['rows'] == [] for r in rows)
    result = {'status': 'pass' if passed else 'fail', 'matching_suite_counts': matched,
              'scope': 'CPU import and test discovery only; no test bodies or GPU kernels executed',
              'GPU_runs': 0, 'rows': rows}
    write_json(output / 'summary.json', result)
    print(json.dumps({'status': result['status'], 'imports': len(rows), 'matched_suites': matched}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
