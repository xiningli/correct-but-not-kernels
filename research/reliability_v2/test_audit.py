"""K1/K3: guard against executing candidates or overinterpreting source/log hints."""
from pathlib import Path
import pytest

from research.reliability_v2.audit import archive_file, log_observation, source_observation, summarize


REFERENCE = 'class Model:\n    def forward(self, x):\n        return x + 1\n'


def fenced(body):
    return '```python\n' + body + '\n```'


def test_candidate_is_never_executed(tmp_path):
    marker = tmp_path / 'must-not-exist'
    src = f'open({str(marker)!r}, "w").write("bad")\n' + REFERENCE.replace('Model:', 'ModelNew:')
    result = source_observation(fenced(src), REFERENCE)
    assert result['exact_forward_ast_match']
    assert not marker.exists()


def test_comment_markers_are_not_imports_and_ast_match_is_conservative():
    src = '# triton __global__ load_inline\nclass ModelNew:\n    def forward(self, y):\n        return y + 1\n'
    result = source_observation(fenced(src), REFERENCE)
    assert result['source_markers'] == 'no_custom_markers'
    assert not result['exact_forward_ast_match']  # Equivalent expression is NOT established by this test.


def test_unused_triton_import_is_only_a_source_marker():
    src = 'import triton as unused\n' + REFERENCE.replace('Model:', 'ModelNew:')
    result = source_observation(fenced(src), REFERENCE)
    assert result['source_markers'] == 'triton_markers'
    assert result['exact_forward_ast_match']
    assert 'runtime_backend' not in result


def test_missing_log_is_not_reclassified_as_numeric_error():
    assert log_observation(None)['log_hint'] == 'missing_log'
    result = log_observation('triton.compiler.errors.CompilationError: failed\nAttributeError: missing API\n')
    assert set(result['log_signals']) == {'api_import_or_name', 'compiler_diagnostic'}


def test_output_write_failure_is_not_a_compiler_diagnostic():
    log = ('File "<private-harness>", line 155, in execute_worker\n'
           'File "/venv/site-packages/safetensors/torch.py", line 323, in save_file\n'
           'safetensors._safetensors_rust.SafetensorError: Error while serializing: '
           'I/O error: No space left on device (os error 28)\n')
    observed = log_observation(log)
    assert observed['log_hint'] == 'storage_io'
    assert observed['log_signals'] == ['storage_io']
    assert observed['output_serialization_enospc_hint']
    assert not log_observation('OSError: No space left on device')['output_serialization_enospc_hint']


def test_archive_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        archive_file(tmp_path, '../secret')
    (tmp_path / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        archive_file(tmp_path, 'outside/secret')


def test_all_original_failures_and_regressions_are_retained():
    def row(label, sample, component):
        return {'label': label, 'arm': label.split('-')[0], 'task': 'task', 'sample': sample,
                'original_component': component, 'correct': component == 'correct',
                'source_markers': 'no_custom_markers', 'exact_forward_ast_match': False, 'log_hint': 'missing_log'}
    result = summarize([row('base', 0, 'correct'), row('base', 1, 'crash'),
                        row('C-1', 0, 'format'), row('C-1', 1, 'correct')])
    assert result['policies']['C-1']['n'] == 2
    assert result['policies']['C-1']['components'] == {'format': 1, 'correct': 1}
    assert {'base': 'correct', 'policy': 'format', 'n': 1} in result['paired_outcome_transitions']['C-1']
