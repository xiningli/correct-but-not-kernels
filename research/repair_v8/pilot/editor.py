"""Two wire formats with the same source boundary. Never executes model text."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import textwrap

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
MAX_BYTES = 262144


class EditError(ValueError):
    pass


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse(source):
    if not isinstance(source, str) or len(source.encode()) > MAX_BYTES or '\r' in source:
        raise EditError('invalid_source_size_or_type')
    try:
        tree = ast.parse(source)
        compile(tree, '<candidate>', 'exec', dont_inherit=True)
        return tree
    except (SyntaxError, ValueError, TypeError, RecursionError) as e:
        raise EditError('invalid_python') from e


def function_nodes(tree):
    result = {}
    for item in tree.body:
        pairs = [(item.name, item)] if isinstance(item, FUNCTIONS) else (
            [(item.name + '.' + n.name, n) for n in item.body if isinstance(n, FUNCTIONS)]
            if isinstance(item, ast.ClassDef) else [])
        for name, node in pairs:
            if name in result:
                raise EditError('ambiguous_function_name')
            result[name] = node
    return result


def constant_name(node):
    target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
        node.target if isinstance(node, ast.AnnAssign) else None)
    return target.id if isinstance(target, ast.Name) else None


def signature(node):
    return (type(node).__name__, ast.dump(node.args), ast.dump(node.returns) if node.returns else None,
            tuple(ast.dump(n) for n in node.decorator_list))


def header(source):
    lines = []
    for line in source.splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            break
        lines.append(line)
    return '\n'.join(lines).rstrip()


@dataclass(frozen=True)
class Scope:
    original: str
    protected: tuple[str, ...]


def build_scope(source, protected):
    nodes = function_nodes(parse(source))
    if not nodes or not set(protected) <= set(nodes):
        raise EditError('missing_public_interface')
    return Scope(source, tuple(protected))


def skeleton(tree, original_functions, constants):
    def node_key(node, prefix=''):
        if isinstance(node, FUNCTIONS):
            name = prefix + node.name
            if name in original_functions:
                return ('function', name, type(node).__name__)
            if not prefix and node.name.startswith('_repair_'):
                return None
            raise EditError('new_helper_requires_repair_prefix')
        if not prefix and isinstance(node, (ast.Import, ast.ImportFrom)):
            return None
        if not prefix and constant_name(node) in constants:
            return ('constant', constant_name(node))
        if isinstance(node, ast.ClassDef):
            return ('class', node.name, tuple(ast.dump(n) for n in node.bases),
                    tuple(ast.dump(n) for n in node.keywords), tuple(ast.dump(n) for n in node.decorator_list),
                    tuple(node_key(n, node.name + '.') for n in node.body))
        return ('frozen', ast.dump(node))
    return tuple(key for node in tree.body if (key := node_key(node)) is not None)


def validate(scope, candidate):
    original, proposed = parse(scope.original), parse(candidate)
    old, new = function_nodes(original), function_nodes(proposed)
    constants = {constant_name(n) for n in original.body} - {None}
    if header(scope.original) != header(candidate):
        raise EditError('license_header_changed')
    if skeleton(original, old, constants) != skeleton(proposed, old, constants):
        raise EditError('change_outside_shared_surface')
    if not set(old) <= set(new):
        raise EditError('original_function_removed')
    for name in scope.protected:
        if signature(old[name]) != signature(new[name]):
            raise EditError('public_signature_changed')
    # A new import or constant cannot silently rebind a function/class/constant.
    bindings = []
    for node in proposed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings.extend(a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)
                            for a in node.names)
        elif isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            bindings.append(node.name)
        elif constant_name(node):
            bindings.append(constant_name(node))
    if len(bindings) != len(set(bindings)) or '*' in bindings:
        raise EditError('ambiguous_top_level_binding')


def region(node):
    return min([node.lineno, *[d.lineno for d in getattr(node, 'decorator_list', [])]]) - 1, node.end_lineno


def function_text(source, name):
    node = function_nodes(parse(source))[name]
    start, end = region(node)
    return textwrap.dedent(''.join(source.splitlines(keepends=True)[start:end]))


def atomic(current, operations):
    if not isinstance(operations, list) or not operations or len(operations) > 64:
        raise EditError('invalid_operations')
    tree, lines = parse(current), current.splitlines(keepends=True)
    functions = function_nodes(tree)
    edits, append, seen = [], [], set()
    for op in operations:
        if not isinstance(op, dict) or set(op) != {'op', 'name', 'source'}:
            raise EditError('invalid_operation_fields')
        kind, name, source = op['op'], op['name'], op['source']
        if not isinstance(name, str) or (kind, name) in seen:
            raise EditError('duplicate_operation')
        seen.add((kind, name))
        fragment = parse(source)
        if kind == 'replace_function':
            if name not in functions or len(fragment.body) != 1 or not isinstance(fragment.body[0], FUNCTIONS):
                raise EditError('replacement_must_be_named_function')
            node = functions[name]
            if fragment.body[0].name != node.name:
                raise EditError('function_name_changed')
            start, end = region(node)
            indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
            edits.append((start, end, textwrap.indent(source.rstrip() + '\n', indent)))
        elif kind == 'upsert_helper':
            if not name.startswith('_repair_') or '.' in name or len(fragment.body) != 1 or not isinstance(fragment.body[0], FUNCTIONS) or fragment.body[0].name != name:
                raise EditError('invalid_helper')
            if name in functions:
                start, end = region(functions[name]); edits.append((start, end, source.rstrip() + '\n'))
            else:
                append.append(source.rstrip() + '\n')
        elif kind == 'replace_import_block':
            if name != 'imports' or not fragment.body or not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in fragment.body):
                raise EditError('imports_only_required')
            imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            if not imports:
                raise EditError('missing_import_block')
            for i, node in enumerate(imports):
                edits.append((*region(node), source.rstrip() + '\n' if i == 0 else ''))
        elif kind == 'replace_constant':
            nodes = [n for n in tree.body if constant_name(n) == name]
            if len(nodes) != 1 or len(fragment.body) != 1 or constant_name(fragment.body[0]) != name:
                raise EditError('existing_constant_required')
            edits.append((*region(nodes[0]), source.rstrip() + '\n'))
        else:
            raise EditError('unknown_operation')
    ordered = sorted(edits)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise EditError('overlapping_edits')
    for start, end, block in reversed(ordered):
        lines[start:end] = [block]
    return ''.join(lines).rstrip() + '\n' + ''.join('\n' + x for x in append)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EditError('duplicate_JSON_key')
        result[key] = value
    return result


def apply_response(current, scopes, text, interface):
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES:
        raise EditError('response_too_large')
    try:
        response = json.loads(text, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError) as e:
        raise EditError('JSON_required') from e
    if not isinstance(response, dict) or set(response) != {'files'} or not isinstance(response['files'], list):
        raise EditError('files_array_required')
    result, seen = dict(current), set()
    if not response['files']:
        raise EditError('empty_response')
    for entry in response['files']:
        field = 'source' if interface == 'full_module' else 'operations'
        if not isinstance(entry, dict) or set(entry) != {'path', 'expected_sha256', field}:
            raise EditError('invalid_file_fields')
        path = entry['path']
        if not isinstance(path, str) or path not in scopes or path in seen:
            raise EditError('unlisted_or_duplicate_path')
        seen.add(path)
        if entry['expected_sha256'] != digest(current[path]):
            raise EditError('stale_source')
        proposed = entry[field] if field == 'source' else atomic(current[path], entry[field])
        validate(scopes[path], proposed)
        result[path] = proposed
    if interface == 'full_module' and seen != set(scopes):
        raise EditError('all_complete_modules_required')
    if interface not in {'full_module', 'atomic_edits'}:
        raise EditError('invalid_interface')
    return result
