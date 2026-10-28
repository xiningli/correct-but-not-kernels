"""K1/K3: finite numeric equality, bounded output decoding, persistence probes.

Only the trusted parent compares expected values. Candidate workers never
receive expected values or the reference implementation. Native code/driver
bugs and obfuscated in-process measurement tampering remain a limitation, not
a claimed formal correctness guarantee.
"""
import json
import math
import struct
from pathlib import Path

RTOL, ATOL = 1e-3, 1e-4

def compare(actual, expected):
    import torch
    if type(actual) is not torch.Tensor or type(expected) is not torch.Tensor:
        return {'correct': False, 'component': 'output_type'}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {'correct': False, 'component': 'shape_dtype'}
    if actual.is_floating_point() and not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {'correct': False, 'component': 'nonfinite'}
    ok = bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)) if actual.is_floating_point() else bool(torch.equal(actual, expected))
    return {'correct': ok, 'component': 'correct' if ok else 'numerical', 'rtol': RTOL, 'atol': ATOL}

def read_output(path: Path, expected):
    """Validate shape/dtype/size BEFORE materializing untrusted tensor bytes."""
    from safetensors.torch import load_file
    import torch
    types = {torch.float32:'F32', torch.float64:'F64', torch.float16:'F16',
             torch.bfloat16:'BF16', torch.int64:'I64', torch.int32:'I32',
             torch.int16:'I16', torch.int8:'I8', torch.uint8:'U8', torch.bool:'BOOL'}
    if path.is_symlink() or not path.is_file():
        raise ValueError('Output must be a regular file')
    maximum = expected.numel()*expected.element_size() + 65544
    if path.stat().st_size > maximum:
        raise ValueError('Oversized candidate output')
    with path.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 65536:
            raise ValueError('Oversized safetensors header')
        header = json.loads(f.read(header_size))
    if set(header) != {'output'} or header['output']['shape'] != list(expected.shape) or header['output']['dtype'] != types[expected.dtype]:
        raise ValueError('Candidate output metadata differs from expected contract')
    return load_file(path, device='cpu')['output']

def input_flags(inputs, snapshots, output):
    """K3: conservative no-mutation/no-alias contract; called inside worker."""
    import torch
    flags = []
    for value, before in zip(inputs, snapshots):
        if isinstance(value, torch.Tensor):
            if not torch.equal(value, before):
                flags.append('input_mutation')
            if isinstance(output, torch.Tensor) and output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                flags.append('input_alias')
    return flags
