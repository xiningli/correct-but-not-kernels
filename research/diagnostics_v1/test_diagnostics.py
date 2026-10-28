"""D1/D2: semantic evidence parity, leakage, context, and real-task boundaries."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from research.reliability_v2.repair_cases import SOURCES

NEEDS_UPSTREAM = pytest.mark.skipif(not SOURCES.is_dir(), reason='Upstream Liger-Kernel snapshots are fetched, not committed: python -m research.reliability_v2.repair_sources (see its --help)')

from research.diagnostics_v1.common import PLAN, object_hash
from research.diagnostics_v1.evidence import (build_visible_diagnostic, flatten, group, render,
                                             validate_diagnostic, validate_facts)
from research.diagnostics_v1.protocol import candidate_source, entries, messages, validate_entries
from research.diagnostics_v1.report import summarize
from research.diagnostics_v1.sample import check_context, check_source_budget
from research.diagnostics_v1.tasks import CASES, public_task


def facts():
    return {'result.component': 'incorrect', 'result.passed': 1, 'result.total': 2,
            'tests.0.passed': False, 'tests.0.component': 'numerical',
            'tests.0.inputs.0.shape': [2, 3], 'tests.0.inputs.0.stride': [7, 2],
            'tests.0.inputs.0.dtype': 'torch.float32', 'tests.0.inputs.0.storage_offset': 1,
            'tests.0.mismatch_count': 1, 'tests.0.first_index': [0, 1],
            'tests.0.actual': 0.5, 'tests.0.expected': 0.75, 'tests.0.max_abs_error': 0.25}


def diagnostic(task):
    f = facts()
    return {'schema': 1, 'scope': 'development_visible_only', 'task': task['id'],
            'source_sha256': task['source_sha256'], 'facts': f, 'facts_sha256': object_hash(f)}


def test_rich_presentations_preserve_exact_facts_and_verdict_ablation_removes_detail():
    f = facts()
    a, b = [render(f, c) for c in ['facts_flat', 'facts_grouped']]
    parse = lambda x: json.loads(x['text'].split('```json\n')[1].split('\n```')[0])
    assert parse(a) == flatten(parse(b)) == f
    assert a['exposed_facts_sha256'] == b['exposed_facts_sha256']
    assert a['text'] != b['text']
    bare = render(f, 'verdict_only')
    assert set(parse(bare)) == {'result.component', 'result.passed', 'result.total'}
    assert bare['exposed_facts_sha256'] != a['exposed_facts_sha256']


@pytest.mark.parametrize('key', ['hidden.passed', 'gold.patch', 'tests.0.root_cause', 'tests.0.log'])
def test_hidden_or_inferred_fields_are_rejected(key):
    with pytest.raises(ValueError):
        validate_facts({**facts(), key: 'private_or_inferred_marker'})


def test_invalid_numeric_or_component_feedback_is_rejected():
    for k, v in [('tests.0.actual', float('nan')), ('result.passed', True),
                 ('result.component', 'harness_error'), ('result.passed', 3)]:
        with pytest.raises(ValueError):
            validate_facts({**facts(), k: v})


@NEEDS_UPSTREAM
def test_all_conditions_share_task_and_seed_without_private_fix_locators():
    tasks = {k: public_task(k) for k in CASES}
    diagnostics = {k: diagnostic(t) for k, t in tasks.items()}
    plan = yaml.safe_load(PLAN.read_text())
    rows = entries(tasks, diagnostics, plan)
    assert len(rows) == 12 and len({(r['task'], r['replicate'], r['condition']) for r in rows}) == 12
    assert rows == entries(tasks, diagnostics, plan)
    for task in tasks.values():
        assert '/pull/' not in task['source']
        assert 'Redistribution' in task['source']  # Preserve the upstream BSD license.
        for rep in range(2):
            group_rows = [r for r in rows if r['task'] == task['id'] and r['replicate'] == rep]
            assert len({r['sampling_seed'] for r in group_rows}) == 1
            assert len({r['all_facts_sha256'] for r in group_rows}) == 1
            assert all(task['source'] in r['messages'][1]['content'] for r in group_rows)
    changed = copy.deepcopy(rows); changed[0]['messages'][1]['content'] += 'a leaked hint'
    with pytest.raises(ValueError):
        validate_entries(changed, rows)


@NEEDS_UPSTREAM
def test_diagnostics_cannot_move_between_sources_or_suites():
    task = public_task(next(iter(CASES)))
    d = diagnostic(task)
    for change in [{'scope': 'withheld'}, {'source_sha256': '0' * 64}, {'gold_patch': 'private'}]:
        with pytest.raises(ValueError):
            validate_diagnostic({**d, **change}, task)


def test_candidate_failures_are_not_repaired_or_rerolled():
    valid = '```python\nclass ModelNew: pass\n```'
    assert candidate_source(valid, 'stop') == 'class ModelNew: pass\n'
    assert candidate_source(valid, 'length') is None
    assert candidate_source('```python\nclass ModelNew(: pass\n```', 'stop') is None
    assert candidate_source('```python\ndef kernel(): pass\n```', 'stop') is None


def test_prompt_overflow_is_explicit_and_no_prompt_is_truncated():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(messages[0]['content'])))
    prompt = [[{'role': 'user', 'content': 'abcdefghij'}]]
    assert check_context(Tokenizer(), prompt, 5, 15) == [10]
    with pytest.raises(ValueError, match='Context overflow'):
        check_context(Tokenizer(), prompt, 6, 15)
    assert prompt[0][0]['content'] == 'abcdefghij'


def test_output_budget_allows_original_program_without_forced_compression():
    class Tokenizer:
        def encode(self, source, **kwargs):
            return list(source)
    tasks = {'case': {'source': 'class ModelNew: pass'}}
    n = len('```python\nclass ModelNew: pass\n```')
    assert check_source_budget(Tokenizer(), tasks, n + 1) == {'case': n}
    with pytest.raises(ValueError, match='Original program exceeds output cap'):
        check_source_budget(Tokenizer(), tasks, n)


def test_summary_keeps_unknowns_and_does_not_count_replicates_as_defects():
    row = dict(policy='base', task='development/case-001', condition='facts_flat', replicate=0,
               resolved=True, component='correct', prompt_tokens=10, new_tokens=20, evaluation_seconds=1)
    rows = [row, {**row, 'replicate': 1, 'resolved': None, 'component': 'harness_error'}]
    summary = summarize(rows)
    assert summary['attempts'] == 2 and summary['independent_defects'] == 1
    assert summary['cells'][0]['unknown'] == 1 and summary['cells'][0]['unresolved'] == 0
    with pytest.raises(ValueError):
        summarize([row, row])


def test_numeric_diagnostics_preserve_stride_and_measure_only_supplied_visible_outputs(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import research.diagnostics_v1.evidence as module
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    wanted = torch.tensor([1., 2., 3.])
    actual = torch.tensor([1., 4., 3.])
    monkeypatch.setattr(module, 'read_output', lambda path, expected: actual)
    verdict = {'component': 'incorrect', 'checks': [{'correct': False, 'component': 'numerical'}],
               'sandbox': {'workdir': str(tmp_path)}, 'hidden': 'SECRET_NOT_EXPOSED'}
    result = build_visible_diagnostic('development/case-001', 'source', [[x]], [wanted], verdict)
    f = result['facts']
    assert f['tests.0.inputs.0.stride'] == [6, 2]
    assert f['tests.0.mismatch_count'] == 1 and f['tests.0.first_index'] == [1]
    assert f['tests.0.actual'] == 4. and f['tests.0.expected'] == 2.
    assert 'SECRET_NOT_EXPOSED' not in json.dumps(result)
    with pytest.raises(RuntimeError):
        build_visible_diagnostic('development/case-001', 'source', [[x]], [wanted], {'component': 'harness_error'})


def test_handoff_requires_dead_predecessor_deferred_shutdown_and_verified_archive(tmp_path, monkeypatch):
    import research.diagnostics_v1.supervise as module
    from research.diagnostics_v1.common import digest, save
    monkeypatch.setattr(module, 'alive', lambda pid: True)
    assert module.handoff_evidence(tmp_path, 123) is None
    monkeypatch.setattr(module, 'alive', lambda pid: False)
    with pytest.raises(RuntimeError, match='without a handoff'):
        module.handoff_evidence(tmp_path, 123)
    save(tmp_path / 'shutdown-deferred.json', {'utc': 'test'})
    with pytest.raises(RuntimeError, match='without verified backup'):
        module.handoff_evidence(tmp_path, 123)
    archive = tmp_path / 'evidence.tgz'; archive.write_bytes(b'controlled archive fixture')
    save(tmp_path / 'archive.json', {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)})
    save(tmp_path / 'backup-local-ack.json', {'sha256': '0' * 64})
    save(tmp_path / 'outcome.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='identity mismatch'):
        module.handoff_evidence(tmp_path, 123)
    save(tmp_path / 'backup-local-ack.json', {'sha256': digest(archive)})
    assert module.handoff_evidence(tmp_path, 123)['outcome']['status'] == 'complete'
    archive.write_bytes(b'corrupted archive')
    with pytest.raises(ValueError, match='identity mismatch'):
        module.handoff_evidence(tmp_path, 123)
