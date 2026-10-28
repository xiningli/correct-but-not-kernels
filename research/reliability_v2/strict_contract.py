"""K3/T1: fail-closed runtime evidence for a custom CUDA/Triton contract.

Dispatch observations and actual CUDA profiler events must agree. Declaring an
unused kernel, or launching a dummy alongside ATen computation, is insufficient.
This is a bounded evaluator threat model, not a proof against arbitrary native
code that can corrupt its own process. C1 isolation and source review still apply.
"""
from __future__ import annotations

import ast
import re


ALLOWED_ATEN = frozenset({
    'aten.empty.memory_format', 'aten.empty_strided.default', 'aten.empty_like.default',
    'aten.new_empty.default', 'aten.new_empty_strided.default',
    'aten.view.default', 'aten._unsafe_view.default', 'aten.as_strided.default',
    'aten.transpose.int', 'aten.permute.default', 'aten.slice.Tensor',
    'aten.select.int', 'aten.squeeze.default', 'aten.squeeze.dim',
    'aten.unsqueeze.default', 'aten.detach.default', 'aten.alias.default',
})


def declared_kernel_names(source):
    """Names are only a provenance cross-check; execution is separately required."""
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if (isinstance(target, ast.Attribute) and target.attr == 'jit') or (
                        isinstance(target, ast.Name) and target.id == 'jit'):
                    names.add(node.name)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(re.findall(r'__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(\w+)\s*\(', node.value))
    return sorted(names)


def assess_observation(observation, names):
    required = {'complete', 'cuda_available', 'dispatch_ops', 'cuda_kernels'}
    if not isinstance(observation, dict) or set(observation) != required:
        return {'component': 'harness_error', 'reason': 'missing_or_malformed_runtime_observation'}
    if observation['complete'] is not True or observation['cuda_available'] is not True:
        return {'component': 'harness_error', 'reason': 'incomplete_CUDA_observer'}
    ops, kernels = observation['dispatch_ops'], observation['cuda_kernels']
    if any(not isinstance(x, list) or any(type(v) is not str for v in x) for x in (ops, kernels)):
        return {'component': 'harness_error', 'reason': 'malformed_event_list'}
    forbidden = sorted(set(ops) - ALLOWED_ATEN)
    if forbidden:
        return {'component': 'contract_violation', 'reason': 'high_level_tensor_compute', 'ops': forbidden}
    if not kernels:
        return {'component': 'contract_violation', 'reason': 'no_custom_kernel_execution'}
    unknown = [k for k in kernels if not any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(n)
                 + r'(?![A-Za-z0-9_])', k) for n in names)]
    if unknown:
        return {'component': 'contract_violation', 'reason': 'unattributed_or_vendor_kernel', 'kernels': unknown}
    return {'component': 'compliant', 'reason': 'dispatch_and_CUDA_events_agree'}


def source_contract_flags(source):
    """Reject explicit access to the observation machinery; do not infer runtime."""
    flags = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in {
            '_C', 'profiler', '_python_dispatch', '_disable_current_modes',
            '_DisableTorchDispatch', 'set_default_device', 'set_default_dtype'}):
            # Conventional super().__init__ is required for nn.Module.
            if node.attr != '__init__':
                flags.append('observer_or_runtime_access')
        if isinstance(node, ast.ImportFrom) and any(x in (node.module or '') for x in ('profiler', '_python_dispatch')):
            flags.append('observer_or_runtime_access')
    return sorted(set(flags))
