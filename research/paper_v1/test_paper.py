"""K1/K3: classifiers behind the paper's decorative-kernel decomposition."""
import pytest

from research.paper_v1 import claims
from research.paper_v1.shortcut_audit import classify, launched, referenced_by_model
from research.paper_v1.training_shortcut_audit import static_class

HEAD = 'import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef fused_kernel(x_ptr):\n    pass\n\n'

DECORATIVE = HEAD + '''class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
'''

CALLED = HEAD + '''def run(x):
    fused_kernel[(1,)](x)
    return x

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return run(x)
'''

ALIASED = HEAD + '''def run(x):
    fused_kernel[(1,)](x)
    return x

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

class Final(torch.nn.Module):
    def forward(self, x):
        return run(x)

ModelNew = Final
'''

REDEFINED = HEAD + '''class ModelNew(torch.nn.Module):
    def forward(self, x):
        return fused_kernel[(1,)](x)

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
'''


def test_static_classes():
    assert static_class('import torch\nclass ModelNew(torch.nn.Module):\n    pass\n') == 'no_declared_kernel'
    assert static_class(DECORATIVE) == 'definition_only'
    assert static_class(CALLED) == 'reachable'
    assert static_class(None) == 'not_extractable'


def test_reachability_follows_python_rebinding():
    assert referenced_by_model(ALIASED, ['fused_kernel']) is True
    # The last class definition is the one Python binds to ModelNew.
    assert referenced_by_model(REDEFINED, ['fused_kernel']) is False


def test_launch_attribution_uses_name_boundaries():
    # Triton 3.x reports the bare function name (observed in the v4 replay traces).
    assert launched(['fused_kernel'], ['fused_kernel'])
    assert launched(['fused_kernel'], ['void fused_kernel(float*, int)'])
    assert not launched(['fused_kernel'], ['prefused_kernel'])
    # A generically named kernel must not match inside an ATen kernel name.
    assert not launched(['kernel'], ['void at::native::vectorized_elementwise_kernel<4>(int)'])


def test_runtime_classes():
    def row(names, kernels):
        return {'result': {'declared_kernel_names': names,
                           'runtime': {'observations': [{'cuda_kernels': kernels}]}}}
    assert classify(row([], ['cudnn_conv'])) == 'no_declared_kernel'
    assert classify(row(['fused_kernel'], ['cudnn_conv'])) == 'declared_not_launched'
    assert classify(row(['fused_kernel'], ['cudnn_conv', 'fused_kernel'])) == 'launched'


@pytest.mark.skipif(not (claims.ROOT / claims.R3).exists(),
                    reason='runs/ receipts come from the data bundle: python -m research.paper_v1.claims --data release-data/data')
def test_every_paper_number_matches_its_receipt():
    text = ' '.join(claims.PAPER.read_text().split())
    missing = [want for want, _ in claims.claims() if ' '.join(want.split()) not in text]
    missing += [want for want in claims.MANUAL if want not in text]
    assert missing == []


def test_writeup_within_paper_track_word_limit():
    # The Paper Track caps writeups at 3,000 words; count the whole source, references included.
    assert len(claims.PAPER.read_text().split()) <= 3000
