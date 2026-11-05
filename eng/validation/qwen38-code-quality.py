#!/usr/bin/env python3
"""Supplemental, read-only prime-function checks; NEVER regrade the frozen benchmark.

Generated modules are never exec/eval/compiled. A small AST interpreter evaluates
only two extracted pure functions, with closed value/call types and bounded work.
The interpreter runs in an isolated Python subprocess with OS resource limits.
Truncated replies stay UNVERIFIED even if their complete extracted functions pass.

Usage: qwen38-code-quality.py 'docs/validation/RUN/raw/*.jsonl' --output
       docs/validation/RUN/code-quality.json

Subset: scalar arithmetic/comparisons, if/for/while, assignment, flat lists,
integer/list slices, append, list comprehensions, and closed range/len/int/float/
bool/list/enumerate/abs/min/max/all/any plus math.isqrt/sqrt/ceil/floor. Imports,
other attributes, closures, arbitrary globals/callables and Python code execution
are unsupported. Annotations are stripped; decorators/defaults are rejected.
Top-level module code and generated unittests are never evaluated. Complete
function checks on a truncated module are reported separately, not as a PASS.
"""
from __future__ import annotations
import argparse
import ast
import glob
import hashlib
import io
import json
import math
import operator
from pathlib import Path
import re
import subprocess
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = {"is_prime", "primes_below"}
MAX_SOURCE = 131072
MAX_ITEMS = 50000
MAX_STEPS = 1500000
MAX_BITS = 256
MAX_ALLOCATED_ITEMS = 500000
BUILTINS = {"range", "len", "int", "float", "bool", "list", "enumerate", "abs", "min", "max", "all", "any"}
MATH = {"isqrt": math.isqrt, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}
RESERVED = BUILTINS | FUNCTIONS | set(MATH) | {"math"}
FORBIDDEN = {"open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
             "input", "object", "type", "super", "help", "breakpoint"}


class Unsupported(Exception): pass
class BudgetExceeded(Unsupported): pass
class Returned(Exception):
    def __init__(self, value): self.value = value
class Broken(Exception): pass
class Continued(Exception): pass


def sha(value):
    return hashlib.sha256(value).hexdigest()


def extract_functions(text):
    """Tokenize top-level def boundaries; never repair a truncated function."""
    if len(text.encode()) > MAX_SOURCE:
        raise Unsupported("source_size_limit")
    source, issues = text, []
    opening = re.match(r"\s*```(?:python|py)?[ \t]*\r?\n", text, re.I)
    if opening:
        rest = text[opening.end():]
        closing = re.search(r"^```[ \t]*$", rest, re.M)
        if closing:
            source = rest[:closing.start()]
            if "```" in rest[closing.end():]: issues.append("multiple_code_fences")
        else:
            source = rest
            issues.append("unclosed_code_fence")
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            tokens.append(token)
            if len(tokens) > 25000: raise Unsupported("token_count_limit")
    except (tokenize.TokenError, IndentationError, SyntaxError):
        issues.append("module_tokenization_incomplete")
    if any(t.type == tokenize.OP and t.string == "@" and t.start[1] == 0 for t in tokens):
        raise Unsupported("top_level_decorator_not_supported")
    lines = source.splitlines(keepends=True)
    functions, identities = {}, {}
    for i, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "def" or token.start[1] != 0:
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].string not in FUNCTIONS:
            continue
        name = tokens[i + 1].string
        if name in functions: raise Unsupported("duplicate_function:" + name)
        depth, entered, end = 0, False, None
        for item in tokens[i + 2:]:
            if item.type == tokenize.INDENT:
                depth += 1
                entered = True
            elif item.type == tokenize.DEDENT:
                depth -= 1
                if entered and depth == 0:
                    end = item.start[0]
                    break
            elif not entered and item.type == tokenize.NEWLINE:
                # A one-line suite has no INDENT/DEDENT.
                header_line = "".join(lines[token.start[0] - 1:item.end[0]])
                try:
                    parsed = ast.parse(header_line)
                    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.FunctionDef):
                        end = item.end[0] + 1
                        break
                except SyntaxError:
                    pass
        if end is None: raise Unsupported("function_not_closed:" + name)
        code = "".join(lines[token.start[0] - 1:end - 1])
        try:
            parsed = ast.parse(code)
        except (SyntaxError, RecursionError, MemoryError) as error:
            raise Unsupported("function_parse_error:" + name) from error
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
            raise Unsupported("ambiguous_function_boundary:" + name)
        functions[name] = parsed.body[0]
        identities[name] = {"source_line": token.start[0], "source_sha256": sha(code.encode())}
    if set(functions) != FUNCTIONS:
        raise Unsupported("missing_required_functions")
    try:
        module = ast.parse(source)
        for item in module.body:
            if isinstance(item, ast.FunctionDef) and item.name in FUNCTIONS: continue
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str): continue
            issues.append("module_statement_not_evaluated:" + type(item).__name__)
    except (SyntaxError, RecursionError, MemoryError):
        issues.append("module_parse_incomplete")
    return functions, identities, sorted(set(issues))


ALLOWED_NODES = {
    ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Expr,
    ast.Name, ast.Constant, ast.List, ast.Tuple, ast.ListComp, ast.comprehension,
    ast.Call, ast.Attribute, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.Load, ast.Store,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
}


def validate_functions(functions):
    for name, function in functions.items():
        args = function.args
        if (function.decorator_list or args.posonlyargs or len(args.args) != 1 or args.vararg or args.kwarg
                or args.kwonlyargs or args.defaults or args.kw_defaults):
            raise Unsupported("function_signature_not_supported:" + name)
        # Annotations are never evaluated, but dangerous annotation names are rejected too.
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and ("__" in node.id or node.id in FORBIDDEN):
                raise Unsupported("forbidden_name:" + node.id)
        function.returns = None
        args.args[0].annotation = None
        if args.args[0].arg in RESERVED or "__" in args.args[0].arg:
            raise Unsupported("reserved_parameter")
        for node in ast.walk(function):
            if isinstance(node, ast.AnnAssign): node.annotation = None
        for node in ast.walk(function):
            if type(node) not in ALLOWED_NODES or (isinstance(node, ast.FunctionDef) and node is not function):
                raise Unsupported("unsupported_ast:" + type(node).__name__)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in RESERVED:
                raise Unsupported("reserved_name_assignment:" + node.id)
            if isinstance(node, ast.Call):
                if node.keywords: raise Unsupported("keyword_calls_not_supported")
                target = node.func
                if isinstance(target, ast.Name):
                    if target.id not in RESERVED - {"math"}: raise Unsupported("call_not_allowed:" + target.id)
                elif isinstance(target, ast.Attribute):
                    if not (target.attr == "append" or (isinstance(target.value, ast.Name)
                            and target.value.id == "math" and target.attr in MATH)):
                        raise Unsupported("attribute_call_not_allowed:" + target.attr)
                else: raise Unsupported("indirect_call_not_allowed")
            if isinstance(node, ast.Attribute) and node.attr not in set(MATH) | {"append"}:
                raise Unsupported("attribute_not_allowed:" + node.attr)
    return {name: sha(ast.dump(function, include_attributes=False).encode()) for name, function in functions.items()}


class Interpreter:
    """No Python code objects or user-callable values ever enter this interpreter."""
    def __init__(self, functions):
        self.functions, self.steps, self.depth, self.allocated = functions, 0, 0, 0

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS: raise BudgetExceeded("step_budget_exhausted")

    def scalar(self, value):
        if type(value) not in (int, float, bool, type(None)):
            raise Unsupported("non_scalar_container_element")
        if type(value) is int and value.bit_length() > MAX_BITS: raise BudgetExceeded("integer_size_limit")
        if type(value) is float and (not math.isfinite(value) or abs(value) > 1e30):
            raise BudgetExceeded("float_size_limit")
        return value

    def allocate(self, items):
        self.allocated += items
        if self.allocated > MAX_ALLOCATED_ITEMS: raise BudgetExceeded("total_allocation_budget")

    def sequence(self, value):
        # Only our closed enumerate builtin can create this iterator. Keep it
        # lazy: snapshotting a list here would change mutation/termination.
        if type(value) is enumerate: return value
        if type(value) not in (list, tuple, range): raise Unsupported("unsupported_sequence_type")
        try: size = len(value)
        except OverflowError as error: raise BudgetExceeded("container_size_limit") from error
        if size > MAX_ITEMS: raise BudgetExceeded("container_size_limit")
        return value

    def materialize(self, value):
        result = []
        for item in self.sequence(value):
            self.tick()
            if len(result) >= MAX_ITEMS: raise BudgetExceeded("container_size_limit")
            self.allocate(1 + (len(item) if type(item) is tuple else 0))
            result.append(item)
        return result

    def call(self, name, values):
        self.tick()
        if name in self.functions:
            if len(values) != 1: raise Unsupported("function_arity")
            self.depth += 1
            if self.depth > 16: raise BudgetExceeded("call_depth_limit")
            try:
                self.block(self.functions[name].body, {self.functions[name].args.args[0].arg: values[0]})
            except Returned as result:
                return result.value
            finally:
                self.depth -= 1
            return None
        if name == "range":
            if not 1 <= len(values) <= 3 or any(type(x) is not int for x in values): raise Unsupported("range_arguments")
            return self.sequence(range(*values))
        if name in ("enumerate", "list", "len", "all", "any"):
            if len(values) != 1: raise Unsupported("builtin_arity")
            value = self.sequence(values[0])
            if name == "enumerate":
                if type(value) is enumerate: raise Unsupported("nested_enumerate_not_supported")
                return enumerate(value)
            if name == "list": return self.materialize(value)
            if name == "len": return len(value)
            return all(value) if name == "all" else any(value)
        if name in ("int", "float", "bool", "abs"):
            if len(values) != 1: raise Unsupported("builtin_arity")
            self.scalar(values[0])
            return self.scalar({"int": int, "float": float, "bool": bool, "abs": abs}[name](values[0]))
        if name in ("min", "max"):
            items = self.sequence(values[0]) if len(values) == 1 else values
            for value in items: self.scalar(value)
            return min(items) if name == "min" else max(items)
        if name in MATH:
            for value in values: self.scalar(value)
            return self.scalar(MATH[name](*values))
        raise Unsupported("unknown_call")

    def binary(self, kind, left, right):
        if kind in (ast.Add, ast.Mult) and (type(left) is list or type(right) is list):
            if kind is ast.Add:
                if type(left) is not list or type(right) is not list: raise TypeError("list addition")
                size = len(left) + len(right)
            else:
                seq, count = (left, right) if type(left) is list else (right, left)
                if type(count) not in (int, bool): raise TypeError("list repetition")
                size = len(seq) * max(0, count)
            if size > MAX_ITEMS: raise BudgetExceeded("container_size_limit")
            self.allocate(size)
            return operator.add(left, right) if kind is ast.Add else operator.mul(left, right)
        self.scalar(left); self.scalar(right)
        if kind is ast.Pow:
            if abs(right) > 64 or (type(left) is int and type(right) is int
                                  and right > 0 and max(1, left.bit_length()) * right > MAX_BITS):
                raise BudgetExceeded("power_size_limit")
        operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}[kind]
        return self.scalar(operation(left, right))

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant): return self.scalar(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env: raise Unsupported("unbound_name:" + node.id)
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.scalar(self.expr(x, env)) for x in node.elts]
            self.sequence(values)
            self.allocate(len(values))
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.BinOp): return self.binary(type(node.op), self.expr(node.left, env), self.expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not): return not value
            self.scalar(value)
            return self.scalar(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BoolOp):
            value = self.expr(node.values[0], env)
            for item in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value): break
                value = self.expr(item, env)
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, other in zip(node.ops, node.comparators):
                right = self.expr(other, env)
                operation = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                             ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                             ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}[type(op)]
                if not operation(left, right): return False
                left = right
            return True
        if isinstance(node, ast.IfExp): return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Subscript):
            value = self.sequence(self.expr(node.value, env))
            index = self.index(node.slice, env)
            if isinstance(index, slice) and type(value) in (list, tuple):
                self.allocate(len(range(*index.indices(len(value)))))
            return value[index]
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return self.call(node.func.id, [self.expr(x, env) for x in node.args])
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                return self.call(node.func.attr, [self.expr(x, env) for x in node.args])
            value = self.expr(node.func.value, env)
            values = [self.expr(x, env) for x in node.args]
            if node.func.attr != "append" or type(value) is not list or len(values) != 1: raise Unsupported("unsafe_list_operation")
            if len(value) >= MAX_ITEMS: raise BudgetExceeded("container_size_limit")
            self.allocate(1)
            value.append(self.scalar(values[0]))
            return None
        if isinstance(node, ast.ListComp):
            result = []
            def visit(level, scope):
                if level == len(node.generators):
                    if len(result) >= MAX_ITEMS: raise BudgetExceeded("container_size_limit")
                    self.allocate(1)
                    result.append(self.scalar(self.expr(node.elt, scope))); return
                generator = node.generators[level]
                if generator.is_async: raise Unsupported("async_comprehension")
                for value in self.sequence(self.expr(generator.iter, scope)):
                    self.tick(); self.assign(generator.target, value, scope)
                    if all(self.expr(test, scope) for test in generator.ifs): visit(level + 1, scope)
            visit(0, dict(env))
            return result
        raise Unsupported("unsupported_expression:" + type(node).__name__)

    def index(self, node, env):
        if isinstance(node, ast.Slice):
            values = [self.expr(x, env) if x is not None else None for x in (node.lower, node.upper, node.step)]
            if any(x is not None and type(x) not in (int, bool) for x in values): raise TypeError("slice index")
            return slice(*values)
        value = self.expr(node, env)
        if type(value) not in (int, bool): raise TypeError("index must be integer")
        return value

    def assign(self, target, value, env):
        if isinstance(target, ast.Name): env[target.id] = value; return
        if isinstance(target, (ast.Tuple, ast.List)):
            self.sequence(value)
            if type(value) is enumerate: raise Unsupported("iterator_unpack_not_supported")
            if len(target.elts) != len(value): raise ValueError("unpack length")
            for item, element in zip(target.elts, value): self.assign(item, element, env)
            return
        if isinstance(target, ast.Subscript):
            container = self.expr(target.value, env)
            if type(container) is not list: raise Unsupported("assignment_target_not_list")
            index = self.index(target.slice, env)
            if isinstance(index, slice):
                self.sequence(value)
                if type(value) is enumerate: value = self.materialize(value)
                for item in value: self.scalar(item)
                removed = len(range(*index.indices(len(container))))
                if len(container) - removed + len(value) > MAX_ITEMS: raise BudgetExceeded("container_size_limit")
                self.allocate(len(value))
            else: self.scalar(value)
            container[index] = value
            return
        raise Unsupported("unsupported_assignment")

    def block(self, statements, env):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return): raise Returned(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets: self.assign(target, value, env)
            elif isinstance(node, ast.AnnAssign):
                if node.value is None: raise Unsupported("annotation_without_assignment")
                self.assign(node.target, self.expr(node.value, env), env)
            elif isinstance(node, ast.AugAssign):
                # Container in-place operations and side-effecting subscript
                # targets have distinct alias/evaluation semantics; reject them.
                if not isinstance(node.target, ast.Name): raise Unsupported("augmented_target_not_name")
                left = self.scalar(self.expr(node.target, env))
                right = self.scalar(self.expr(node.value, env))
                self.assign(node.target, self.binary(type(node.op), left, right), env)
            elif isinstance(node, ast.If): self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, (ast.For, ast.While)):
                iterator = iter(self.sequence(self.expr(node.iter, env))) if isinstance(node, ast.For) else None
                broke = False
                while True:
                    self.tick()
                    if iterator is not None:
                        try: value = next(iterator)
                        except StopIteration: break
                        self.assign(node.target, value, env)
                    elif not self.expr(node.test, env): break
                    try: self.block(node.body, env)
                    except Continued: continue
                    except Broken: broke = True; break
                if not broke: self.block(node.orelse, env)
            elif isinstance(node, ast.Break): raise Broken()
            elif isinstance(node, ast.Continue): raise Continued()
            elif isinstance(node, ast.Pass): pass
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str): continue
                self.expr(node.value, env)
            else: raise Unsupported("unsupported_statement:" + type(node).__name__)


def oracle_prime(n):
    if n < 2: return False
    return all(n % d for d in range(2, math.isqrt(n) + 1))


def check_functions(functions):
    prime_inputs = sorted(set(range(-17, 258)) | {341, 561, 1105, 1729, 7919, 7920, 9409, 65521, 65535, 65536, 999983, 994009, 1000000, 1000003})
    list_inputs = sorted(set(range(-10, 130)) | {169, 170, 255, 256, 257, 289, 290, 511, 512, 513, 997, 998, 1023, 1024, 1025, 4095, 4096, 4097})
    results = {}
    for name, inputs in (("is_prime", prime_inputs), ("primes_below", list_inputs)):
        passed = 0
        for n in inputs:
            expected = oracle_prime(n) if name == "is_prime" else [i for i in range(2, n) if oracle_prime(i)]
            try:
                actual = Interpreter(functions).call(name, [n])
                valid_type = type(actual) is bool if name == "is_prime" else type(actual) is list and all(type(x) is int for x in actual)
                if not valid_type or actual != expected:
                    observed = actual
                    if type(actual) is range: observed = {"range": [actual.start, actual.stop, actual.step]}
                    if type(actual) is enumerate: observed = {"unsupported_return_type": "enumerate"}
                    if type(actual) in (list, tuple) and len(actual) > 40:
                        observed = {"first_40": actual[:40], "length": len(actual)}
                    results[name] = {"status": "FAIL", "passed_cases": passed, "counterexample": {"input": n, "expected": expected, "actual": observed, "actual_type": type(actual).__name__}}
                    break
            except Unsupported as error:
                results[name] = {"status": "UNVERIFIED", "passed_cases": passed, "input": n, "reason": str(error)}; break
            except (ArithmeticError, ValueError, TypeError, IndexError) as error:
                results[name] = {"status": "FAIL", "passed_cases": passed, "counterexample": {"input": n, "expected": expected, "exception": type(error).__name__}}; break
            passed += 1
        else: results[name] = {"status": "PASS", "passed_cases": passed}
    return results


def worker(payload):
    limits = {}
    try:
        import resource
        for name, value in (("RLIMIT_CPU", (4, 4)), ("RLIMIT_AS", (512 << 20, 512 << 20)),
                            ("RLIMIT_FSIZE", (0, 0)), ("RLIMIT_CORE", (0, 0)), ("RLIMIT_NOFILE", (64, 64))):
            try: resource.setrlimit(getattr(resource, name), value); limits[name] = list(value)
            except (OSError, ValueError): limits[name] = "unavailable"
    except ImportError:
        limits["resource_module"] = "unavailable"
    result = {"status": "UNVERIFIED", "resource_limits": limits}
    try:
        functions, identities, issues = extract_functions(payload["text"])
        digests = validate_functions(functions)
        for name in FUNCTIONS: identities[name]["annotation_stripped_ast_sha256"] = digests[name]
        checks = check_functions(functions)
        result.update(functions=identities, function_checks=checks, source_issues=issues)
        if payload.get("finish_reason") != "stop": issues.append("response_not_finished_with_stop")
        if payload.get("request_error"): issues.append("request_error")
        if issues or any(x["status"] == "UNVERIFIED" for x in checks.values()): return result
        result["status"] = "FAIL" if any(x["status"] == "FAIL" for x in checks.values()) else "PASS"
    except (Unsupported, SyntaxError, RecursionError, MemoryError) as error:
        result["reason"] = str(error) or type(error).__name__
    return result


def diagnose_text(text, content_sha256, finish_reason, *, request_error=None, timeout=8):
    actual_sha = sha(text.encode())
    if not content_sha256 or actual_sha != content_sha256:
        return {"status": "UNVERIFIED", "reason": "stored_content_hash_mismatch", "actual_content_sha256": actual_sha}
    if len(text.encode()) > MAX_SOURCE: return {"status": "UNVERIFIED", "reason": "source_size_limit"}
    try:
        completed = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "_worker"],
            input=json.dumps({"text": text, "finish_reason": finish_reason, "request_error": request_error}),
            text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "UNVERIFIED", "reason": "worker_timeout"}
    if completed.returncode != 0:
        return {"status": "UNVERIFIED", "reason": "worker_resource_limit_or_error", "worker_exit_code": completed.returncode}
    try: return json.loads(completed.stdout)
    except ValueError: return {"status": "UNVERIFIED", "reason": "invalid_worker_result"}


def analyze(paths, timeout=8):
    records, sources, cache = [], [], {}
    for path in paths:
        data = path.read_bytes(); sources.append({"path": str(path), "sha256": sha(data)})
        header = {}
        for line_number, line in enumerate(data.decode().splitlines(), 1):
            row = json.loads(line)
            if row.get("kind") == "header": header = row
            if row.get("kind") != "case" or row.get("scenario") != "long_generation": continue
            prompts = row.get("metrics", {}).get("per_prompt", [])
            for index, request in enumerate(row.get("requests", [])):
                kind = prompts[index].get("kind") if index < len(prompts) else None
                if kind != "code": continue
                text = request.get("content") or ""
                key = (sha(text.encode()), request.get("content_sha256"), request.get("finish_reason"), json.dumps(request.get("error"), sort_keys=True))
                if key not in cache:
                    cache[key] = diagnose_text(text, request.get("content_sha256"), request.get("finish_reason"), request_error=request.get("error"), timeout=timeout)
                records.append({"raw_path": str(path), "raw_line": line_number,
                    "engine": row.get("engine", header.get("engine")), "mode": row.get("mode", header.get("mode")),
                    "label": row.get("label", header.get("label")), "run_id": row.get("run_id"),
                    "scenario": "long_generation", "repeat": row.get("repeat"), "request_index": index,
                    "request_sha256": request.get("request_sha256"), "content_sha256": request.get("content_sha256"),
                    "finish_reason": request.get("finish_reason"), "original_verdict": row.get("verdict"),
                    "original_pass": row.get("pass"), "diagnostic": cache[key]})
    return {"scope": "Supplemental finite functional checks of extracted prime functions only. No generated module/import/main/unittest is executed. Frozen benchmark scores, quality gates, performance eligibility and rankings are unchanged.",
            "tool_sha256": sha(Path(__file__).read_bytes()), "sources": sources, "records": records,
            "diagnostic_counts": {status: sum(r["diagnostic"]["status"] == status for r in records) for status in ("PASS", "FAIL", "UNVERIFIED")},
            "unique_content_sha256_count": len({r["content_sha256"] for r in records}),
            "limits": {"source_bytes": MAX_SOURCE, "container_items": MAX_ITEMS, "allocated_items_per_function_call": MAX_ALLOCATED_ITEMS, "integer_bits": MAX_BITS, "ast_steps_per_function_call": MAX_STEPS, "worker_wall_seconds": timeout},
            "limitations": ["A finite integer test suite is not a proof for arbitrary inputs or a complexity benchmark.",
                "The restricted interpreter supports a documented Python subset; unsupported syntax is UNVERIFIED, not a functional failure.",
                "Whole-response truncation, incomplete syntax, ignored top-level statements, hash mismatch and resource limits remain UNVERIFIED even when extracted function checks pass.",
                "Function-level PASS concerns only the extracted complete AST under closed safe builtins/math; generated unittest, annotation behavior and module-level effects are not validated.",
                "No diagnostic can promote an originally failed benchmark scenario or authorize a performance ranking."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=8)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0: parser.error("positive finite timeout required")
    destination = args.output.resolve()
    if not any(destination.is_relative_to(ROOT / name) for name in ("artifacts", "docs/validation")):
        parser.error("diagnostics must stay in ignored artifacts/ or docs/validation/")
    if destination.exists(): parser.error("refusing to overwrite existing evidence")
    paths = sorted({Path(p) for pattern in args.raw for p in glob.glob(pattern)})
    if not paths: parser.error("no raw inputs matched")
    result = analyze(paths, args.timeout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "counts": result["diagnostic_counts"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        print(json.dumps(worker(json.loads(sys.stdin.read(MAX_SOURCE * 8)))))
    else: main()
