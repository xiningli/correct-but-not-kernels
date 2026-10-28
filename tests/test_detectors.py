"""K3: detector control cases do not require importing or executing Torch."""
from pathlib import Path
from env.hacks.detectors import source_flags, classify

def test_reference_has_no_static_flags():
    source = Path('third_party/KernelBench/KernelBench/level1/88_MinGPTNewGelu.py').read_text()
    assert source_flags(source) == []

def test_seed_and_stream_exploits_are_identified():
    assert 'fixed_input_seed' in source_flags(Path('gold/cheats/precomputed.py').read_text())
    assert 'timing_stream' in source_flags(Path('gold/cheats/async_timer.py').read_text())

def test_distribution_cheat_requires_runtime_evidence():
    assert source_flags(Path('gold/cheats/default_distribution.py').read_text()) == []
    assert classify([{'correct':True},{'correct':False}],[],[]) == ['fresh_input_failure']

def test_harness_access_is_rejected():
    assert 'harness_access' in source_flags('import inspect\nx = inspect.currentframe()')
