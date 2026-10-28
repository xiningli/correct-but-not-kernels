"""K1/K3: trusted tensor-storage budgeting, independent of model outcomes."""
from pathlib import Path
import shutil

HEADROOM = 8 << 30


def storage_budget(input_sizes, output_sizes, *, headroom=HEADROOM):
    if len(input_sizes) != len(output_sizes) or not input_sizes:
        raise ValueError('Need paired nonempty input/output sizes')
    if any(type(x) is not int or x < 0 for x in [*input_sizes, *output_sizes, headroom]):
        raise ValueError('Sizes must be nonnegative integers')
    remaining, written = sum(input_sizes), 0
    peak = remaining
    for inp, out in zip(input_sizes, output_sizes):
        remaining -= inp  # private input is removed after CPU load, before output write
        written += out
        peak = max(peak, remaining + written)
    # Conservatively allow one extra full output plus filesystem/compiler slack.
    required = peak + max(output_sizes) + headroom
    return {'staged_input_bytes': sum(input_sizes), 'streaming_peak_bytes': peak,
            'scratch_bytes': max(output_sizes), 'headroom_bytes': headroom,
            'required_before_staging_bytes': required,
            'required_after_staging_bytes': max(0, required - sum(input_sizes))}


def require_free(path: Path, required: int):
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError(f'Insufficient evaluator storage: free={free}, required={required}')
    return {'free_bytes': free, 'required_bytes': required}
