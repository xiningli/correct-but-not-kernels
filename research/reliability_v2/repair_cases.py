"""K4/T2: two real upstream fixes adapted as DEVELOPMENT-only regressions.

Kernel function text is copied byte-for-byte from verified before/after files.
Dependencies/wrappers are reduced and every change is recorded. These tasks are
not full-repository repairs and must never enter the later sealed test set.
Upstream BSD notices are retained in generated code and the source archive.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from research.reliability_v2.repair_sources import verify_archive

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'third_party/cuda-repair-candidates'


def extract_function(source, name):
    tree = ast.parse(source)
    matches = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(matches) != 1:
        raise ValueError('Expected one upstream function: ' + name)
    node = matches[0]
    start = min([node.lineno] + [x.lineno for x in node.decorator_list])
    return '\n'.join(source.splitlines()[start - 1:node.end_lineno]) + '\n'


def build(case, revision):
    if case not in {'softcap_tail', 'independent_strides'} or revision not in {'before', 'after'}:
        raise ValueError('Undeclared development repair')
    number = 1490 if case == 'softcap_tail' else 1461
    directory = SOURCES / f'linkedin--Liger-Kernel-{number}'
    manifest = verify_archive(directory)
    path, = manifest['source_paths']
    original = (directory / revision / path).read_text()
    notice = '\n'.join('# ' + line for line in (directory / 'UPSTREAM_LICENSE').read_text().splitlines())
    prefix = notice + '\n# Reduced development reproduction of ' + manifest['url'] + '\n'
    prefix += 'import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n'
    preserved = []
    if case == 'softcap_tail':
        kernel = extract_function(original, 'liger_cross_entropy_kernel')
        preserved.append(kernel)
        prefix += 'from triton.language.extra.cuda.libdevice import tanh\nLOG2_E = tl.constexpr(1.4426950408889634)\n'
        wrapper = '''class ModelNew(nn.Module):
    def forward(self, x, labels):
        rows, cols = x.shape
        loss = torch.empty((rows,), dtype=torch.float32, device=x.device)
        liger_cross_entropy_kernel[(rows,)](
            x, x.stride(0), labels, labels.stride(0), None, loss, None, 1,
            None, 0, None, 0, cols, rows, 0.0, 0.0, -100,
            0.0, 0.0, "none", 1.0, False, False, False,
            min(triton.next_power_of_2(cols), 32768), False, True, False)
        return loss
'''
        source = prefix + kernel + '\n' + wrapper
        adaptations = ['Extract unchanged cross-entropy Triton function; omit unrelated package code.',
            'Use CUDA tanh import directly instead of multi-backend package dispatch.',
            'Supply a forward-only ModelNew wrapper with softcap=1, no smoothing or weights, reduction=none.',
            'Disable in-place gradient output so inputs remain immutable; this does not alter the faulty softmax pass.']
    else:
        kernel = extract_function(original, '_selective_log_softmax_kernel')
        helper = extract_function(original, 'fused_selective_log_softmax')
        preserved.append(kernel)
        zero = '''@triton.jit
def initialize_logp(Y, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + idx, 0.0, idx < N)
'''
        old = 'log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)'
        new = ('log_p = torch.empty((B, L), dtype=torch.float32, device=logits.device)\n'
               '    initialize_logp[(triton.cdiv(B * L, 256),)](log_p, B * L, 256)')
        if helper.count(old) != 1:
            raise ValueError('Upstream helper changed; do not silently adapt')
        helper = helper.replace(old, new)
        wrapper = '''class ModelNew(nn.Module):
    def forward(self, logits, ids, mask):
        return fused_selective_log_softmax(logits, ids, 0.7, mask)
'''
        source = prefix + zero + kernel + helper + wrapper
        adaptations = ['Extract unchanged selective-log-softmax Triton function and its public helper.',
            'Replace helper torch.zeros with empty plus a handwritten zero-initialization kernel, equally before and after.',
            'Use a ModelNew wrapper with temperature=0.7; keep upstream slicing, stride arguments and kernel math.',
            'Allocate extra backing storage in regressions to avoid out-of-bounds reads by the faulty version.']
    return source, {'case': case, 'split': 'development', 'scope': 'reduced_real_bug_reproduction',
                    'origin': 'real_upstream_fix', 'upstream': manifest['url'],
                    'parent': manifest['parent'], 'fixed': manifest['fixed'], 'revision': revision,
                    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'preserved_kernel_sha256': [hashlib.sha256(x.encode()).hexdigest() for x in preserved],
                    'adaptations': adaptations, 'license': manifest['license_spdx']}


def regressions(case, *, hidden=False):
    """Independent FP64 formulas; public/private cases use distinct seeds/domains.

    All regressions here are developer diagnostics. Later test issues require
    independently sealed inputs and may not reuse these two development bugs.
    """
    import torch
    generator = torch.Generator().manual_seed(2026101304 if hidden else 2026101303)
    cases, expected = [], []
    for i in range(4):
        if case == 'softcap_tail':
            n = ([5, 33, 257, 1001] if hidden else [3, 17, 513, 32])[i]
            dtype = [torch.float32, torch.float16, torch.bfloat16, torch.float32][i]
            x = torch.randn(3, n, generator=generator, dtype=dtype)
            if i == 0 and not hidden:
                x.zero_()
            labels = torch.randint(n, (3,), generator=generator)
            z = x.double().tanh()
            wanted = (torch.logsumexp(z, -1) - z.gather(1, labels[:, None]).squeeze(1)).float()
            cases.append([x, labels]); expected.append(wanted)
        elif case == 'independent_strides':
            b, length, vocab = (3, 5 + i + int(hidden), [17, 31, 67, 129][i])
            logits = torch.randn(b, length + 1, vocab, generator=generator)
            ids_storage = torch.randint(vocab, (b + 1, length * 4 + 8), generator=generator)
            mask_storage = torch.randint(2, (b + 1, length * 6 + 16), generator=generator).bool()
            ids = ids_storage[:b, :length * 2:2]
            mask = mask_storage[:b, :length * 3:3]
            if i == 3:  # Dense non-regression control.
                ids, mask = ids.contiguous(), mask.contiguous()
            values = logits[:, :-1].double() / 0.7
            wanted = (values.gather(-1, ids[:, :, None]).squeeze(-1) - torch.logsumexp(values, -1))
            wanted = wanted.masked_fill(~mask, 0).float()
            cases.append([logits, ids, mask]); expected.append(wanted)
        else:
            raise ValueError('Unknown regression')
    return cases, expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in ['softcap_tail', 'independent_strides']:
        for revision in ['before', 'after']:
            source, meta = build(case, revision)
            name = case + '-' + revision
            (args.out / (name + '.py')).write_text(source)
            if args.gpu:
                from research.reliability_v2.strict_evaluator import evaluate
                meta['regressions'] = {}
                for hidden in [False, True]:
                    inputs, expected = regressions(case, hidden=hidden)
                    label = 'hidden' if hidden else 'visible'
                    result = evaluate(source, inputs, expected, args.out / (name + '-' + label), seed=2026101301)
                    meta['regressions'][label] = result
                    print(json.dumps({'case': case, 'revision': revision, 'suite': label,
                                      'component': result['component'], 'correct': result['correct']}), flush=True)
                    if result['component'] == 'harness_error':
                        raise RuntimeError('Harness error; retain diagnostics, do not certify repair')
            rows.append(meta)
            (args.out / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')
    if args.gpu:
        for case in ['softcap_tail', 'independent_strides']:
            before, after = [r for r in rows if r['case'] == case]
            if any(before['regressions'][s]['numerically_correct'] for s in ['visible', 'hidden']):
                raise RuntimeError('Original defect did not reproduce')
            if not all(after['regressions'][s]['correct'] for s in ['visible', 'hidden']):
                raise RuntimeError('Reference fix did not pass strict checks')
        (args.out / 'receipt.json').write_text(json.dumps({'status': 'PASS', 'defects': 2,
            'scope': 'development_reproductions_not_policy_results', 'test_eligible': False}) + '\n')


if __name__ == '__main__':
    main()
