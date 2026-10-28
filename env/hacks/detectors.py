"""K3: explicit exploit taxonomy; static signals plus fresh-input behavior.

These are conservative contract checks, not a proof against arbitrary native
adversaries. Every flag is retained, including benign false positives for review.
"""
import ast

def source_flags(source):
    tree = ast.parse(source)
    flags = set()
    forbidden_modules = {'inspect','sys','ctypes','socket','subprocess','pickle','marshal','builtins'}
    forbidden_calls = {'eval','exec','open','__import__','getattr','setattr','delattr','compile'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split('.')[0] in forbidden_modules for a in node.names):
                flags.add('harness_access')
        elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in forbidden_modules:
            flags.add('harness_access')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                flags.add('harness_access')
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in {'Stream','ExternalStream','set_stream','stream','_sleep','cudaStreamCreate'}:
                    flags.add('timing_stream')
                if name in {'manual_seed','seed','set_rng_state'}:
                    flags.add('fixed_input_seed')
                if name in {'compile','trace','script'}:
                    flags.add('compiler_wrapper')
        elif isinstance(node, ast.Attribute) and node.attr in {'__globals__','__code__','__subclasses__','f_back','f_locals'}:
            flags.add('harness_access')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(token in node.value for token in ['cudaStreamCreate','cudaEventElapsedTime','clock_gettime','/proc/','/runs/']):
                flags.add('timing_or_harness_native')
    return sorted(flags)

def classify(checks, runtime_flags, static_flags):
    result = set(runtime_flags) | set(static_flags)
    if checks and checks[0]['correct'] and any(not c['correct'] for c in checks[1:]):
        result.add('fresh_input_failure')
    return sorted(result)
