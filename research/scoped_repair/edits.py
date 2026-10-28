"""Apply named function replacements without importing or executing candidates.

Scope comes only from the supplied pre-fix module: all top-level functions and
direct class methods are editable. Imports, constants, class declarations and
the external forward signature remain frozen. Internal signatures may change,
so a stride repair can update both a kernel and its caller in one atomic edit.

This is an editing contract, NOT an execution security boundary or correctness
check. GPU candidates still require the existing isolated strict evaluator.
The prototype accepts a tool-neutral mapping; model response serialization and
an experiment runner have not been implemented or frozen.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import textwrap

MAX_SOURCE_BYTES = 1_000_000
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


class EditError(ValueError):
    """Closed edit outcome; the caller must count it against the attempt budget."""


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _parse(source):
    if not isinstance(source, str) or len(source.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise EditError('invalid_source_size_or_type')
    if '\r' in source:
        raise EditError('LF_source_required')
    try:
        tree = ast.parse(source)
        # Compilation checks syntax/context only; no import, decorator or code runs.
        compile(tree, '<candidate>', 'exec', dont_inherit=True)
        return tree
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        raise EditError('invalid_python') from error


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    end: int
    indent: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _regions(source):
    tree = _parse(source)
    lines = source.splitlines(keepends=True)
    found = []
    for item in tree.body:
        if isinstance(item, FUNCTIONS):
            found.append((item.name, item))
        elif isinstance(item, ast.ClassDef):
            found.extend((item.name + '.' + node.name, node)
                         for node in item.body if isinstance(node, FUNCTIONS))
    result = []
    for name, node in found:
        start = min([node.lineno, *[x.lineno for x in node.decorator_list]]) - 1
        line = lines[start]
        indent = line[:len(line) - len(line.lstrip(' \t'))]
        result.append(Region(name, start, node.end_lineno, indent, node))
    result.sort(key=lambda x: x.start)
    if len({x.name for x in result}) != len(result):
        raise EditError('ambiguous_function_name')
    if any(a.end > b.start for a, b in zip(result, result[1:])):
        raise EditError('overlapping_functions')
    return result


def _skeleton(source, regions):
    lines = source.splitlines(keepends=True)
    result = []
    offset = 0
    for item in regions:
        result.extend([('literal', ''.join(lines[offset:item.start])), ('function', item.name)])
        offset = item.end
    result.append(('literal', ''.join(lines[offset:])))
    return result


def _signature(node):
    return (type(node).__name__, ast.dump(node.args, include_attributes=False),
            ast.dump(node.returns, include_attributes=False) if node.returns else None,
            tuple(ast.dump(x, include_attributes=False) for x in node.decorator_list))


@dataclass(frozen=True)
class Scope:
    """Trusted host-side boundary built before model evaluation, not model input."""
    original: str
    protected: tuple[str, ...]

    def describe(self):
        return {'origin': 'all_functions_in_supplied_before_source_no_gold_localization',
                'original_sha256': sha256(self.original),
                'editable': [x.name for x in _regions(self.original)],
                'protected_external_signatures': list(self.protected),
                'outside_functions': 'byte_identical',
                'internal_signatures': 'editable_with_their_call_sites',
                'security_or_correctness_certification': False}


def build_scope(source, *, protected=('ModelNew.forward',)):
    names = {x.name for x in _regions(source)}
    if not names or not set(protected) <= names:
        raise EditError('missing_editable_or_protected_function')
    return Scope(source, tuple(protected))


def validate_candidate(scope, candidate):
    before = _regions(scope.original)
    after = _regions(candidate)
    if _skeleton(scope.original, before) != _skeleton(candidate, after):
        raise EditError('change_outside_editable_functions')
    old = {x.name: x for x in before}
    new = {x.name: x for x in after}
    for name in old:
        if type(old[name].node) is not type(new[name].node):
            raise EditError('function_kind_changed')
    for name in scope.protected:
        if _signature(old[name].node) != _signature(new[name].node):
            raise EditError('external_signature_changed')


def function_text(source, name):
    matches = [x for x in _regions(source) if x.name == name]
    if len(matches) != 1:
        raise EditError('unknown_function')
    item, = matches
    return textwrap.dedent(''.join(source.splitlines(keepends=True)[item.start:item.end]))


def accept_full_program(current, scope, candidate, *, expected_source_sha256):
    """Comparison arm: same boundary, but the model returns the whole module."""
    if sha256(current) != expected_source_sha256:
        raise EditError('stale_source')
    validate_candidate(scope, current)
    validate_candidate(scope, candidate)
    return candidate


def apply_edits(current, scope, replacements, *, expected_source_sha256):
    """Atomically replace complete named definitions, including decorators.

    Replacement definitions start at column zero. Indentation and a terminating
    newline are supplied mechanically; no syntax/semantic repair is performed.
    Rejected edits never modify the caller's current source. In a future runner,
    rejection still consumes one generation/attempt and exposes its closed code.
    """
    if sha256(current) != expected_source_sha256:
        raise EditError('stale_source')
    validate_candidate(scope, current)
    if not isinstance(replacements, dict) or not replacements:
        raise EditError('nonempty_function_mapping_required')
    regions = {x.name: x for x in _regions(current)}
    if not all(isinstance(k, str) and k in regions for k in replacements):
        raise EditError('unknown_function')
    if not all(isinstance(v, str) for v in replacements.values()):
        raise EditError('invalid_replacement_type')
    if sum(len(v.encode('utf-8')) for v in replacements.values()) > MAX_SOURCE_BYTES:
        raise EditError('invalid_source_size_or_type')
    blocks = {}
    for name, replacement in replacements.items():
        tree = _parse(replacement)
        if len(tree.body) != 1 or not isinstance(tree.body[0], FUNCTIONS):
            raise EditError('replacement_must_be_one_function')
        node = tree.body[0]
        if node.name != regions[name].node.name:
            raise EditError('replacement_name_changed')
        start = min([node.lineno, *[x.lineno for x in node.decorator_list]])
        if start != 1 or node.end_lineno != len(replacement.splitlines()):
            raise EditError('text_outside_replacement_function')
        blocks[name] = ''.join(regions[name].indent + line if line.strip() else line
                               for line in replacement.splitlines(keepends=True))
        if not blocks[name].endswith('\n'):
            blocks[name] += '\n'
    lines = current.splitlines(keepends=True)
    for name in sorted(blocks, key=lambda k: regions[k].start, reverse=True):
        item = regions[name]
        block = blocks[name]
        if item.end == len(lines) and not current.endswith('\n'):
            block = block.removesuffix('\n')
        lines[item.start:item.end] = block.splitlines(keepends=True)
    candidate = ''.join(lines)
    validate_candidate(scope, candidate)
    return candidate
