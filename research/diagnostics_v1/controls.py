"""D1/D2: current-boot before/fix controls and visible-only diagnostic receipts."""
import argparse
import json
from pathlib import Path

from research.diagnostics_v1.common import digest, save, source_seal, utc
from research.diagnostics_v1.evidence import build_visible_diagnostic, render
from research.diagnostics_v1.prepare import load_prepared
from research.diagnostics_v1.tasks import fixed_control, visible_suite, withheld_suite
from research.reliability_v2.capability_controls import boot_identity
from research.reliability_v2.strict_evaluator import evaluate


def controls(prepared, isolation, out):
    import torch
    torch.set_num_threads(4)
    plan, tasks, policies = load_prepared(prepared)
    isolated = json.loads(Path(isolation).read_text())
    if isolated['status'] != 'PASS' or len(isolated['checks']) != 6 or any(c['status'] != 'PASS' for c in isolated['checks']):
        raise ValueError('All six current isolation controls must pass first')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    initial = source_seal()
    checks, diagnostics = [], {}
    for task_id, task in tasks.items():
        fixed, _ = fixed_control(task_id)
        for label, source in [('before', task['source']), ('fixed', fixed)]:
            for suite, factory in [('visible', visible_suite), ('withheld', withheld_suite)]:
                inputs, expected = factory(task_id)
                directory = out / task_id.split('/')[-1] / label / suite
                verdict = evaluate(source, inputs, expected, directory, seed=plan['seeds']['model_init'])
                ok = (verdict['component'] != 'harness_error' and
                      (verdict['correct'] is True if label == 'fixed' else
                       verdict['numerically_correct'] is False and len(verdict.get('checks', [])) == len(inputs)))
                checks.append({'task': task_id, 'revision': label, 'suite': suite, 'passed': ok,
                               'component': verdict['component'], 'verdict_sha256': digest(directory / 'verdict.json')})
                save(out / 'checks.json', checks)
                print(json.dumps(checks[-1]), flush=True)
                if not ok:
                    raise RuntimeError('Repair diagnostic control failed; no model evaluation')
                if label == 'before' and suite == 'visible':
                    diagnostics[task_id] = build_visible_diagnostic(task_id, source, inputs, expected, verdict)
        rich = [render(diagnostics[task_id]['facts'], c)['exposed_facts_sha256']
                for c in ['facts_flat', 'facts_grouped']]
        if rich[0] != rich[1]:
            raise ValueError('Presentation conditions contain different evidence')
    if initial != source_seal():
        raise ValueError('Source changed during controls')
    save(out / 'visible-diagnostics.json', diagnostics)
    receipt = {'status': 'PASS', 'controls': len(checks), 'utc': utc(), 'boot': boot_identity(),
               'source_sha256': initial, 'prepared_sha256': digest(Path(prepared) / 'prepared.json'),
               'isolation_sha256': digest(isolation), 'checks_sha256': digest(out / 'checks.json'),
               'visible_diagnostics_sha256': digest(out / 'visible-diagnostics.json'),
               'scope': 'two_reduced_development_defects_not_model_results'}
    save(out / 'receipt.json', receipt)
    return receipt


def validate_controls(directory, prepared):
    directory = Path(directory)
    r = json.loads((directory / 'receipt.json').read_text())
    if (r['status'] != 'PASS' or r['controls'] != 8 or r['boot'] != boot_identity()
            or r['source_sha256'] != source_seal()
            or r['prepared_sha256'] != digest(Path(prepared) / 'prepared.json')
            or r['checks_sha256'] != digest(directory / 'checks.json')
            or r['visible_diagnostics_sha256'] != digest(directory / 'visible-diagnostics.json')):
        raise ValueError('Current-source/current-boot diagnostic controls required')
    return json.loads((directory / 'visible-diagnostics.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--isolation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(controls(a.prepared, a.isolation, a.out)))


if __name__ == '__main__':
    main()
