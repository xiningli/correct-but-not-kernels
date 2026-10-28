"""K2/K3: cold-L2 CUDA-event timing with all-stream completion inside the interval.

Warmup includes JIT, which is never timed. Synchronize before recording the end
event so off-stream work cannot vanish. This deliberately includes host dispatch
and synchronization gaps; reference and candidate use the same protocol. Raw
samples, median, IQR and CV are retained; no outlier deletion or winsorization.
"""
from dataclasses import asdict, dataclass
import statistics

@dataclass(frozen=True)
class TimingConfig:
    """K2: declared measurement effort/cache state; C2 freezes the defaults."""
    warmup: int = 10
    samples: int = 31
    flush_bytes: int = 256 << 20

def summarize(samples):
    if len(samples) < 2 or any(x <= 0 for x in samples):
        raise ValueError('Need positive timing samples')
    q = statistics.quantiles(samples, n=4, method='inclusive')
    return {'samples_ms': samples, 'median_ms': statistics.median(samples),
            'iqr_ms': q[2]-q[0], 'cv': statistics.pstdev(samples)/statistics.fmean(samples)}

class Timer:
    def __init__(self, config=TimingConfig()):
        import torch
        self.config = config
        self.flush = torch.empty(config.flush_bytes, dtype=torch.uint8, device='cuda')
        self.sync = torch.cuda.synchronize
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def warm(self, fn):
        for _ in range(self.config.warmup):
            fn()
        self.sync()

    def once(self, fn):
        self.flush.zero_()
        self.sync()
        self.start.record()
        output = fn()
        self.sync()
        self.end.record()
        self.end.synchronize()
        value = self.start.elapsed_time(self.end)
        del output
        return value

    def measure(self, fn):
        return {**summarize([self.once(fn) for _ in range(self.config.samples)]),
                'protocol': asdict(self.config)}
