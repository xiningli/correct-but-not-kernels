"""K3: exact-value persistence checks that do not mistake unchanged NaNs for mutation."""


def same_values(left, right):
    import torch
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    if torch.equal(left, right):
        return True
    if left.is_floating_point() or left.is_complex():
        return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
    return False


def input_flags_v2(inputs, snapshots, output):
    import torch
    flags = []
    for value, before in zip(inputs, snapshots):
        if isinstance(value, torch.Tensor):
            if not same_values(value, before):
                flags.append('input_mutation')
            if isinstance(output, torch.Tensor) and output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                flags.append('input_alias')
    return flags
