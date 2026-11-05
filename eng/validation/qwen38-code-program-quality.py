#!/usr/bin/env python3
"""Ten complete Python-program functional regressions, not a coding leaderboard.

PLAN is the default. --execute requests 10 tasks x 3 repeats, with max_tokens4096.
Each finished, complete module is really compiled and executed in a fresh limited
Python subprocess, consuming JSON stdin and producing exactly one JSON value on
stdout. No function extraction/repair, generated tests, or existing score changes.

The closed AST/import/builtin policy and POSIX resource limits are defense in
depth for this deliberately small program subset, not a general hostile-code
sandbox. Run under an operator-owned evaluation account/container. No server,
SSH, GPU initialization, package installation or network call occurs in PLAN.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request

VERSION = "qwen38-code-program-quality/1"
REPEATS = 3
MAX_TOKENS = 4096
MAX_SOURCE = 65536
MAX_HTTP = 2 << 20
OUTPUT_LIMIT = 65536
MEMORY_BYTES = 512 << 20
CPU_SECONDS = 2
PROGRAM_TIMEOUT = 4.0


def task(name, statement, cases):
    return {"id": name, "statement": statement,
            "cases": [{"id": i, "input": x, "expected": y} for i, (x, y) in enumerate(cases)]}


TASKS = [
    task("primes", "Input {\"n\": integer}. Output the increasing list of primes strictly below n; n<2 gives [].", [
        ({"n": -4}, []), ({"n": 0}, []), ({"n": 2}, []), ({"n": 3}, [2]),
        ({"n": 10}, [2, 3, 5, 7]), ({"n": 25}, [2, 3, 5, 7, 11, 13, 17, 19, 23]),
        ({"n": 30}, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])]),
    task("sort_integers", "Input {\"values\": list of integers}. Output all values sorted ascending, retaining duplicates and exact large integers.", [
        ({"values": []}, []), ({"values": [5]}, [5]), ({"values": [3, -1, 3, 0]}, [-1, 0, 3, 3]),
        ({"values": [5, 4, 3, 2, 1]}, [1, 2, 3, 4, 5]),
        ({"values": [9007199254740993, 9007199254740992, -9007199254740993]}, [-9007199254740993, 9007199254740992, 9007199254740993]),
        ({"values": [0, 0, -2, -2]}, [-2, -2, 0, 0])]),
    task("balanced_brackets", "Input {\"text\": string containing only ()[]{} characters}. Output a JSON boolean indicating properly nested/matched brackets. Empty is true.", [
        ({"text": ""}, True), ({"text": "()[]{}"}, True), ({"text": "([{}])"}, True),
        ({"text": "([)]"}, False), ({"text": "]"}, False), ({"text": "(("}, False),
        ({"text": "{{[()]}}[]"}, True)]),
    task("run_length", "Input {\"text\": Unicode string}. Output consecutive run-length encoding as [[character,count],...]. Empty gives []. Counts are integers.", [
        ({"text": ""}, []), ({"text": "a"}, [["a", 1]]), ({"text": "aaabbbaa"}, [["a", 3], ["b", 3], ["a", 2]]),
        ({"text": "a b"}, [["a", 1], [" ", 1], ["b", 1]]),
        ({"text": "你你🙂🙂你"}, [["你", 2], ["🙂", 2], ["你", 1]]),
        ({"text": "\n\n\t"}, [["\n", 2], ["\t", 1]])]),
    task("merge_intervals", "Input {\"intervals\": list of closed integer [start,end] intervals, start<=end}. Output sorted merged intervals. Merge if next.start<=current.end, including touching endpoints. Empty gives [].", [
        ({"intervals": []}, []), ({"intervals": [[2, 2]]}, [[2, 2]]),
        ({"intervals": [[1, 3], [3, 5], [8, 9]]}, [[1, 5], [8, 9]]),
        ({"intervals": [[5, 7], [-3, 0], [1, 2], [-1, 1]]}, [[-3, 2], [5, 7]]),
        ({"intervals": [[1, 10], [2, 3], [1, 10]]}, [[1, 10]]),
        ({"intervals": [[0, 0], [1, 1]]}, [[0, 0], [1, 1]])]),
    task("top_k_frequent", "Input {\"values\": list of integers, \"k\": integer}. Output at most k distinct values ordered by descending frequency, ties by ascending numeric value. k<=0 gives [].", [
        ({"values": [], "k": 3}, []), ({"values": [1, 1], "k": 0}, []),
        ({"values": [3, 1, 3, 2, 2, 1], "k": 2}, [1, 2]),
        ({"values": [-1, -1, 2, 2, 2, 0], "k": 9}, [2, -1, 0]),
        ({"values": [8, 7, 8, 6], "k": 1}, [8]), ({"values": [4, 4], "k": -1}, [])]),
    task("transpose", "Input {\"matrix\": rectangular list of integer rows}. Output its transpose. [] or any zero-column rectangular matrix gives [].", [
        ({"matrix": []}, []), ({"matrix": [[], []]}, []), ({"matrix": [[7]]}, [[7]]),
        ({"matrix": [[1, 2, 3], [4, 5, 6]]}, [[1, 4], [2, 5], [3, 6]]),
        ({"matrix": [[-1], [0], [2]]}, [[-1, 0, 2]]),
        ({"matrix": [[9007199254740993, -2]]}, [[9007199254740993], [-2]])]),
    task("grid_shortest_path", "Input {\"grid\": rectangular rows of 0=open,1=blocked, \"start\":[row,col], \"end\":[row,col]}. Output minimum 4-neighbor steps between endpoints. Empty grid, invalid coordinates, blocked endpoints or no path give -1; same open cell gives 0.", [
        ({"grid": [], "start": [0, 0], "end": [0, 0]}, -1),
        ({"grid": [[0]], "start": [0, 0], "end": [0, 0]}, 0),
        ({"grid": [[1]], "start": [0, 0], "end": [0, 0]}, -1),
        ({"grid": [[0, 0], [1, 0]], "start": [0, 0], "end": [1, 1]}, 2),
        ({"grid": [[0, 1], [1, 0]], "start": [0, 0], "end": [1, 1]}, -1),
        ({"grid": [[0, 1, 0], [0, 1, 0], [0, 0, 0]], "start": [0, 0], "end": [0, 2]}, 6),
        ({"grid": [[0]], "start": [-1, 0], "end": [0, 0]}, -1)]),
    task("decimal_add", "Input {\"a\": nonempty unsigned decimal digit string, \"b\": same}. Output their exact sum as a canonical decimal STRING without leading zeros except \"0\". Inputs can contain leading zeros and up to 100 digits.", [
        ({"a": "0", "b": "0"}, "0"), ({"a": "00012", "b": "0008"}, "20"),
        ({"a": "999", "b": "1"}, "1000"), ({"a": "123", "b": "456"}, "579"),
        ({"a": "9007199254740993", "b": "2"}, "9007199254740995"),
        ({"a": "9" * 80, "b": "1"}, "1" + "0" * 80)]),
    task("unique_permutations", "Input {\"values\": list of at most 6 integers}. Output all distinct permutations as lists in lexicographic ascending order. Empty has one permutation: [[]].", [
        ({"values": []}, [[]]), ({"values": [5]}, [[5]]), ({"values": [2, 2]}, [[2, 2]]),
        ({"values": [1, 1, 2]}, [[1, 1, 2], [1, 2, 1], [2, 1, 1]]),
        ({"values": [-1, 0]}, [[-1, 0], [0, -1]]),
        ({"values": [2, 1, 0]}, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])]),
]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    return a == b


def parse_json(value):
    def pairs(items):
        result = {}
        for key, val in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = val
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


SAFE_BUILTINS = {"abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "float", "int",
                 "len", "list", "max", "min", "ord", "pow", "print", "range", "reversed", "round",
                 "set", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError",
                 "RuntimeError", "IndexError", "KeyError", "ZeroDivisionError"}
SAFE_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "copy", "count", "index",
                "sort", "reverse", "get", "items", "keys", "values", "setdefault", "update", "add",
                "discard", "union", "intersection", "difference", "split", "join", "strip", "lstrip",
                "rstrip", "replace", "startswith", "endswith", "isdigit", "lower", "upper"}
ALLOWED_NODES = {"Module", "Import", "alias", "FunctionDef", "arguments", "arg", "Return", "Assign",
                 "AnnAssign", "AugAssign", "Expr", "If", "For", "While", "Break", "Continue", "Pass",
                 "Raise", "Try", "ExceptHandler", "Assert", "Name", "Load", "Store", "Constant", "List",
                 "Tuple", "Set", "Dict", "Subscript", "Slice", "Index", "Starred", "Call", "keyword", "Attribute",
                 "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp", "ListComp", "SetComp", "DictComp",
                 "GeneratorExp", "comprehension", "Lambda", "JoinedStr", "FormattedValue", "Add", "Sub",
                 "Mult", "Div", "FloorDiv", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
                 "MatMult", "UAdd", "USub", "Not", "Invert", "And", "Or", "Eq", "NotEq", "Lt", "LtE",
                 "Gt", "GtE", "Is", "IsNot", "In", "NotIn"}


class Rejected(ValueError):
    pass


def attr_path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = attr_path(node.value)
        return head + "." + node.attr if head else ""
    return ""


def extract_program(text):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_SOURCE:
        raise Rejected("empty or oversized complete program")
    source = text.strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines[0].lower() not in ("```python", "```py", "```") or lines[-1] != "```" or any("```" in x for x in lines[1:-1]):
            raise Rejected("require exactly one complete code fence and no surrounding prose")
        source = "\n".join(lines[1:-1])
    if "```" in source:
        raise Rejected("unexpected code fence")
    return source + "\n"


def validate_program(source):
    tree = ast.parse(source, mode="exec")
    nodes = list(ast.walk(tree))
    if len(nodes) > 12000:
        raise Rejected("AST size limit")
    functions = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    protected = SAFE_BUILTINS | functions | {"json", "sys"}
    if functions & (SAFE_BUILTINS | {"json", "sys"}):
        raise Rejected("function may not shadow protected imports/builtins")
    for node in nodes:
        if type(node).__name__ not in ALLOWED_NODES:
            raise Rejected("unsupported AST: " + type(node).__name__)
        for name in (getattr(node, "id", ""), getattr(node, "arg", ""), getattr(node, "name", ""), getattr(node, "attr", "")):
            if isinstance(name, str) and "__" in name:
                raise Rejected("dunder identifier")
        if isinstance(node, ast.Import) and any(x.name not in ("json", "sys") or x.asname for x in node.names):
            raise Rejected("only unaliased import json/sys allowed")
        if ((isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in protected)
                or (isinstance(node, ast.arg) and node.arg in protected)
                or (isinstance(node, ast.ExceptHandler) and node.name in protected)):
            raise Rejected("cannot rebind or alias a protected callable/import")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise Rejected("decorators forbidden")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise Rejected("async comprehension forbidden")
        if isinstance(node, ast.Attribute):
            path = attr_path(node)
            if path not in ("json.loads", "json.dumps", "json.load", "json.dump", "sys.stdin", "sys.stdout", "sys.stdin.read", "sys.stdout.write") and node.attr not in SAFE_METHODS:
                raise Rejected("attribute forbidden: " + (path or node.attr))
        if isinstance(node, ast.Call):
            if any(x.arg is None for x in node.keywords):
                raise Rejected("expanded keyword arguments forbidden")
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS | functions:
                    raise Rejected("call forbidden: " + node.func.id)
                if node.func.id == "print" and any(x.arg not in ("sep", "end", "flush") for x in node.keywords):
                    raise Rejected("print cannot redirect IO")
            elif isinstance(node.func, ast.Attribute):
                path = attr_path(node.func)
                if path.startswith("json."):
                    allowed = {"ensure_ascii", "sort_keys", "separators", "indent"} if path in ("json.dumps", "json.dump") else set()
                    if any(x.arg not in allowed for x in node.keywords):
                        raise Rejected("JSON hooks/decoder options forbidden")
                elif path == "sys.stdin.read":
                    if node.args or node.keywords:
                        raise Rejected("stdin read must consume the single JSON input")
                elif path != "sys.stdout.write" and node.func.attr not in SAFE_METHODS:
                    raise Rejected("method call forbidden")
            else:
                raise Rejected("dynamic call forbidden")
    return tree


def resource_limits():
    import resource
    values = {"RLIMIT_CPU": CPU_SECONDS, "RLIMIT_AS": MEMORY_BYTES, "RLIMIT_FSIZE": OUTPUT_LIMIT,
              "RLIMIT_CORE": 0, "RLIMIT_NOFILE": 32}
    for name, value in values.items():
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError) as error:
            raise RuntimeError("mandatory " + name + " limit unavailable: " + str(error)) from error
    return values


def worker(source_path, metadata_path):
    metadata = {"status": "NOT_EXECUTED", "limits": {}}
    def save():
        Path(metadata_path).write_text(json.dumps(metadata) + "\n")
    try:
        source = Path(source_path).read_text()
        tree = validate_program(source)
        metadata["limits"] = resource_limits()
    except BaseException as error:
        metadata.update(status="SETUP_ERROR_NO_PROGRAM_EXECUTION", error=type(error).__name__ + ": " + str(error)[:500])
        save()
        return 125
    metadata["status"] = "LIMITS_INSTALLED_NOT_COMPLETE"
    save()
    json_proxy = types.SimpleNamespace(load=json.load, loads=json.loads, dump=json.dump, dumps=json.dumps)
    sys_proxy = types.SimpleNamespace(stdin=sys.stdin, stdout=sys.stdout)
    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or fromlist or name not in ("json", "sys"):
            raise ImportError("import outside closed policy")
        return json_proxy if name == "json" else sys_proxy
    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    allowed["__import__"] = limited_import
    namespace = {"__builtins__": allowed}
    try:
        exec(compile(tree, "<complete-generated-program>", "exec"), namespace, namespace)
        sys.stdout.flush()
        metadata["status"] = "COMPLETE"
        save()
        return 0
    except BaseException as error:
        metadata.update(status="PROGRAM_ERROR", error=type(error).__name__ + ": " + str(error)[:500])
        save()
        return 1


def run_case(source_path, case, directory, *, timeout=PROGRAM_TIMEOUT):
    directory.mkdir()
    input_bytes = encoded(case["input"])
    (directory / "stdin.json").write_bytes(input_bytes)
    stdout, stderr, metadata = (directory / name for name in ("stdout.txt", "stderr.txt", "worker.json"))
    result = {"case_id": case["id"], "input": case["input"], "input_sha256": sha(input_bytes),
              "expected": case["expected"], "passed": False, "status": "ERROR", "elapsed_s": None}
    started = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        child = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker", str(source_path), str(metadata)],
                                 stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=directory, start_new_session=True,
                                 env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"})
        try:
            child.communicate(input_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            result.update(status="TIMEOUT", error="program wall-time limit")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    result.update(elapsed_s=time.monotonic() - started, exit_code=child.returncode)
    result["worker"] = json.loads(metadata.read_text()) if metadata.exists() else None
    for path, key in ((stdout, "stdout"), (stderr, "stderr")):
        data = path.read_bytes()
        result[key] = data.decode("utf-8", errors="replace")
        result[key + "_sha256"] = sha(data)
        result[key + "_bytes"] = len(data)
    if result["status"] == "TIMEOUT":
        return result
    if result["stdout_bytes"] >= OUTPUT_LIMIT or result["stderr_bytes"] >= OUTPUT_LIMIT:
        result.update(status="OUTPUT_LIMIT", error="program output limit reached")
        return result
    if child.returncode != 0 or not result["worker"] or result["worker"].get("status") != "COMPLETE":
        result.update(status="PROGRAM_OR_LIMIT_FAILURE", error="worker did not finish with all resource limits installed")
        return result
    try:
        value = parse_json(result["stdout"])
        passed = strict_equal(value, case["expected"])
        result.update(actual=value, passed=passed, status="PASS" if passed else "WRONG_OUTPUT")
    except (ValueError, TypeError) as error:
        result.update(status="INVALID_STDOUT_JSON", error=str(error))
    return result


def evaluate_program(text, finish_reason, cases, directory):
    checks = {"finish_reason_stop": finish_reason == "stop", "complete_module": False, "closed_AST_policy": False}
    result = {"passed": False, "checks": checks, "cases": [], "content_sha256": sha(text.encode()) if isinstance(text, str) else None}
    if not checks["finish_reason_stop"]:
        result["error"] = "incomplete/length-limited response; no repair or program execution"
        return result
    try:
        source = extract_program(text)
        tree = ast.parse(source)
        checks["complete_module"] = bool(tree.body)
        if not checks["complete_module"]:
            raise Rejected("empty complete module")
        validate_program(source)
        checks["closed_AST_policy"] = True
    except (ValueError, SyntaxError, RecursionError) as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        return result
    directory.mkdir()
    program_path = directory / "program.py"
    program_path.write_text(source)
    result.update(program_path=str(program_path), program_sha256=sha(source.encode()), evaluation_scope="actual complete module execution")
    result["cases"] = [run_case(program_path, case, directory / ("case-%02d" % case["id"])) for case in cases]
    result["setup_error"] = any(c.get("worker", {}).get("status") == "SETUP_ERROR_NO_PROGRAM_EXECUTION" for c in result["cases"] if c.get("worker"))
    checks["all_functional_cases"] = bool(cases) and all(x["passed"] is True for x in result["cases"])
    result["passed"] = all(checks.values())
    return result


def sandbox_preflight(directory):
    """Fail before any model request when mandatory process limits cannot run."""
    return evaluate_program("import json, sys\njson.loads(sys.stdin.read())\nprint(0)\n", "stop",
                            [{"id": 0, "input": {}, "expected": 0}], directory)


def request_body(model, task, repeat, nonce_seed):
    prompt = ("Write one COMPLETE Python 3 program, not functions alone. It reads exactly one JSON value from stdin and writes exactly one JSON value to stdout. "
              "Use only import json and/or import sys; standard pure builtins/data structures are available. No other imports, files, network, classes, decorators, dunder identifiers, "
              "__name__ guard, introspection, dynamic calls or JSON hooks. Do not rebind/shadow imports, builtins or function names. Call your main function directly if used. Do not print explanations/debug text. "
              "Return only the complete program, optionally in one closed Python fence. Do not include tests.\nTask: " + task["statement"] +
              "\nRegression identity: " + task["id"] + ":" + str(repeat) + ":" + str(nonce_seed))
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "temperature": 0, "top_p": 1, "seed": 1234, "max_tokens": MAX_TOKENS}


def query(url, body, timeout):
    wire = encoded(body)
    record = {"request_body": body, "request_sha256": sha(wire), "wire_body_sha256": sha(wire), "error": None}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(MAX_HTTP + 1)
            record["http_status"] = response.status
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_HTTP + 1)
        record.update(http_status=error.code, error="HTTP " + str(error.code))
    except (OSError, TimeoutError, http.client.HTTPException) as error:
        record.update(error=type(error).__name__ + ": " + str(error), elapsed_s=time.monotonic() - started)
        return record
    record.update(raw_response=raw.decode("utf-8", errors="replace"), raw_response_sha256=sha(raw), raw_response_bytes=len(raw), elapsed_s=time.monotonic() - started)
    if len(raw) > MAX_HTTP:
        record.update(error="HTTP response size limit; saved prefix only", raw_response_complete=False)
        return record
    record["raw_response_complete"] = True
    if not record["error"]:
        try:
            value = parse_json(raw)
            choices = value["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("require exactly one choice")
            choice = choices[0]
            message = choice["message"]
            if not isinstance(message.get("content"), str) or message.get("tool_calls"):
                raise ValueError("require plain complete program content, no tool calls")
            normalized = {"content": message["content"], "reasoning": message.get("reasoning_content", message.get("reasoning", "")), "tool_calls": message.get("tool_calls", [])}
            record.update(content=message["content"], finish_reason=choice.get("finish_reason"), usage=value.get("usage"),
                          response=value, response_sha256=sha(encoded(normalized)))
        except (ValueError, KeyError, TypeError) as error:
            record["error"] = "invalid completion response: " + str(error)
    return record


def compare_baseline(baseline, records, suite_sha):
    result = {"comparable": False, "no_regression_passed": False, "true_to_false": [], "false_to_true": [], "pairs": []}
    if (baseline.get("schema") != VERSION or baseline.get("suite_sha256") != suite_sha
            or baseline.get("complete") is not True or baseline.get("tool_sha256") != sha(Path(__file__).read_bytes())
            or baseline.get("repeats") != REPEATS or baseline.get("max_tokens") != MAX_TOKENS
            or not isinstance(baseline.get("model"), str) or type(baseline.get("nonce_seed")) is not int):
        result["error"] = "baseline must be a complete execution of the identical suite and tool bytes"
        return result
    def keyed(rows):
        if any(type(x.get("repeat")) is not int or not isinstance(x.get("task"), str) for x in rows):
            raise ValueError("program identities must have typed task/repeat fields")
        pairs = [(x.get("task"), x.get("repeat")) for x in rows]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate baseline/candidate program identity")
        return dict(zip(pairs, rows))
    try:
        old, new = keyed(baseline.get("records", [])), keyed(records)
    except ValueError as error:
        result["error"] = str(error)
        return result
    expected = {(t["id"], r) for t in TASKS for r in range(REPEATS)}
    if old.keys() != new.keys() or set(new) != expected:
        result["error"] = "baseline/candidate do not contain the complete 10x3 matrix"
        return result
    compatible = True
    for key in sorted(new):
        a, b = old[key], new[key]
        current = next(t for t in TASKS if t["id"] == key[0])
        expected_cases = current["cases"]
        try:
            aa = validate_comparison_record(a, current, baseline["model"], baseline["nonce_seed"])
            bb = validate_comparison_record(b, current, baseline["model"], baseline["nonce_seed"])
        except (ValueError, TypeError, KeyError) as error:
            result["error"] = str(key) + ": " + str(error)
            return result
        same = a["request"]["request_sha256"] == b["request"]["request_sha256"]
        row = {"task": key[0], "repeat": key[1], "same_request": same, "baseline_program_passed": a.get("passed") is True,
               "candidate_program_passed": b.get("passed") is True,
               "baseline_case_execution": "COMPLETE_CASE_MATRIX" if aa else "NOT_EXECUTED",
               "candidate_case_execution": "COMPLETE_CASE_MATRIX" if bb else "NOT_EXECUTED", "cases": []}
        compatible &= same
        for case in expected_cases:
            x, y = aa.get(case["id"]), bb.get(case["id"])
            transition = {"case_id": case["id"], "same_suite_fixture": True, "both_cases_executed": x is not None and y is not None,
                          "same_observed_input_and_expected": x is not None and y is not None,
                          "baseline_passed": bool(x and x.get("passed") is True), "candidate_passed": bool(y and y.get("passed") is True),
                          "baseline_status": x.get("status") if x else "NOT_EXECUTED", "candidate_status": y.get("status") if y else "NOT_EXECUTED"}
            row["cases"].append(transition)
            if same and transition["baseline_passed"] != transition["candidate_passed"]:
                result["true_to_false" if transition["baseline_passed"] else "false_to_true"].append({"task": key[0], "repeat": key[1], **transition})
        result["pairs"].append(row)
    result.update(comparable=compatible, no_regression_passed=compatible and not result["true_to_false"] and not any(x["baseline_program_passed"] and not x["candidate_program_passed"] for x in result["pairs"]))
    return result


def validate_comparison_record(row, task, model, nonce_seed):
    request = row["request"]
    expected_body = request_body(model, task, row["repeat"], nonce_seed)
    if encoded(request.get("request_body")) != encoded(expected_body) or any(request.get(k) != sha(encoded(expected_body)) for k in ("request_sha256", "wire_body_sha256")):
        raise ValueError("actual complete request body/hash does not match fixed task/repeat/model/nonce")
    if type(row.get("passed")) is not bool or row.get("verdict") not in ("PASS", "FAIL", "ERROR") or (row["verdict"] == "PASS") != row["passed"]:
        raise ValueError("program verdict/passed mismatch")
    evaluation = row.get("evaluation")
    if evaluation is None:
        if not request.get("error") or row["verdict"] != "ERROR":
            raise ValueError("unexecuted request must retain its actual error")
        return {}
    checks = evaluation.get("checks", {})
    if (not {"finish_reason_stop", "complete_module", "closed_AST_policy"} <= checks.keys()
            or any(type(v) is not bool for v in checks.values())
            or checks["finish_reason_stop"] != (request.get("finish_reason") == "stop")
            or evaluation.get("content_sha256") != sha(request["content"].encode())):
        raise ValueError("completion/check/content identity mismatch")
    cases = evaluation.get("cases", [])
    execution_ready = all(checks[k] for k in ("finish_reason_stop", "complete_module", "closed_AST_policy"))
    if bool(cases) != execution_ready:
        raise ValueError("execution-ready programs require the complete case matrix; pre-execution failures must retain no executed cases")
    ids = [x.get("case_id") for x in cases]
    expected_ids = {x["id"] for x in task["cases"]}
    if cases and (any(type(x) is not int for x in ids) or len(ids) != len(set(ids)) or set(ids) != expected_ids):
        raise ValueError("executed functional cases must have exact typed unique fixture identities")
    keyed = {x["case_id"]: x for x in cases}
    for expected in task["cases"]:
        if not cases:
            break
        case = keyed[expected["id"]]
        if (case.get("input_sha256") != sha(encoded(expected["input"])) or not strict_equal(case.get("input"), expected["input"])
                or not strict_equal(case.get("expected"), expected["expected"]) or type(case.get("passed")) is not bool):
            raise ValueError("functional input/expected/passed metadata mismatch")
        if case.get("status") in ("PASS", "WRONG_OUTPUT", "INVALID_STDOUT_JSON"):
            if case.get("exit_code") != 0 or case.get("worker", {}).get("status") != "COMPLETE":
                raise ValueError("scored stdout must come from a completed actual worker")
            raw = case["stdout"].encode()
            if case.get("stdout_sha256") != sha(raw) or case.get("stdout_bytes") != len(raw):
                raise ValueError("actual scored stdout hash/bytes mismatch")
            try:
                value = parse_json(case["stdout"])
            except (ValueError, TypeError):
                if case.get("status") != "INVALID_STDOUT_JSON":
                    raise ValueError("stdout is not the recorded parsed JSON")
                actual_pass = False
            else:
                if case.get("status") == "INVALID_STDOUT_JSON" or not strict_equal(value, case.get("actual")):
                    raise ValueError("stored JSON status/value contradicts actual parsed stdout")
                actual_pass = strict_equal(value, expected["expected"])
            if case["passed"] != actual_pass or (case.get("status") == "PASS") != actual_pass:
                raise ValueError("functional truth differs from actual stdout")
        elif case["passed"]:
            raise ValueError("failed/unavailable worker cannot claim a passed case")
        elif case.get("exit_code") == 0 and case.get("worker", {}).get("status") == "COMPLETE" and case.get("stdout_bytes", OUTPUT_LIMIT) < OUTPUT_LIMIT:
            raise ValueError("completed in-budget stdout cannot be hidden as an unscored worker failure")
    all_cases = bool(cases) and all(x["passed"] is True for x in cases)
    if (cases and checks.get("all_functional_cases") is not all_cases) or (not cases and checks.get("all_functional_cases") is True):
        raise ValueError("complete-case check contradicts executed cases")
    evaluated_pass = all_cases and all(checks.values())
    if type(evaluation.get("passed")) is not bool or evaluation["passed"] != evaluated_pass or row["passed"] != evaluated_pass:
        raise ValueError("program pass must match all completion/AST/actual functional checks")
    return keyed


def fresh_output(value):
    p = Path(value).absolute()
    if str(p) != os.path.normpath(str(p)) or p.exists() or p.is_symlink():
        raise ValueError("output must be normalized and fresh")
    if "artifacts" not in p.parts and not any(p.parts[i:i + 2] == ("docs", "validation") for i in range(len(p.parts) - 1)):
        raise ValueError("output must be within ignored artifacts/ or docs/validation/")
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink():
            raise ValueError("output ancestors may not be symlinks")
    p.mkdir(parents=True)
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:5088")
    parser.add_argument("--model")
    parser.add_argument("--label", default="")
    parser.add_argument("--nonce-seed", type=int, default=20261102)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.execute and not args.model):
        parser.error("positive finite timeout and explicit execution model are required")
    out = fresh_output(args.out)
    suite_sha = sha(encoded(TASKS))
    source_sha = sha(Path(__file__).read_bytes())
    state = {"schema": VERSION, "label": args.label, "complete": False, "quality_passed": False,
             "status": "PLAN_ONLY_NO_REQUESTS_NO_PROGRAM_EXECUTION", "suite_sha256": suite_sha,
             "tool_sha256": source_sha, "python": sys.version, "python_executable": sys.executable,
             "repeats": REPEATS, "max_tokens": MAX_TOKENS, "planned_programs": len(TASKS) * REPEATS,
             "nonce_seed": args.nonce_seed,
             "planned_functional_cases": sum(len(t["cases"]) for t in TASKS) * REPEATS, "tasks": TASKS, "records": [],
             "limitations": ["Ten algorithm-program regressions, not a public/broad coding benchmark or comprehensive model quality proof.",
                             "Repeated prompts and finite edge cases are not independent model-quality samples; no confidence interval or performance ranking.",
                             "Complete code really executes in a restricted subprocess; unsupported AST, truncation, IO, resource limits and errors are never PASS.",
                             "The AST/import/builtin policy plus POSIX limits is not a general hostile-code sandbox; operator/container isolation remains required.",
                             "No server/runtime identity is inferred; the operator must bind model, server flags, exact binary and exclusive execution externally."]}
    def save():
        temp = out / "campaign.json.tmp"
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temp.replace(out / "campaign.json")
    save()
    if not args.execute:
        print(json.dumps({k: v for k, v in state.items() if k not in ("tasks", "records")}, ensure_ascii=False))
        return 0
    baseline = None
    if args.baseline:
        raw = args.baseline.read_bytes()
        state["baseline"] = {"path": str(args.baseline), "sha256": sha(raw)}
        baseline = parse_json(raw)
    if args.manifest:
        raw = args.manifest.read_bytes()
        state["server_manifest"] = {"path": str(args.manifest), "sha256": sha(raw), "value": parse_json(raw)}
    state.update(status="RUNNING_NOT_PASS", model=args.model, url=args.url)
    save()
    try:
        state["sandbox_preflight"] = sandbox_preflight(out / "sandbox-preflight")
        if state["sandbox_preflight"]["passed"] is not True:
            state.update(status="SETUP_ERROR_NO_MODEL_REQUESTS", acceptance_passed=False)
            print(json.dumps({"status": state["status"], "quality_passed": False, "requests": 0}))
            return 2
        for current in TASKS:
            for repeat in range(REPEATS):
                body = request_body(args.model, current, repeat, args.nonce_seed)
                response = query(args.url, body, args.timeout)
                row = {"task": current["id"], "repeat": repeat, "request": response, "passed": False, "verdict": "ERROR"}
                state["records"].append(row)
                save()
                try:
                    if not response.get("error"):
                        row["evaluation"] = evaluate_program(response.get("content"), response.get("finish_reason"), current["cases"], out / (current["id"] + "-%d" % repeat))
                        row.update(passed=row["evaluation"]["passed"], verdict="ERROR" if row["evaluation"].get("setup_error") else "PASS" if row["evaluation"]["passed"] else "FAIL")
                except BaseException as error:
                    row["evaluation_error"] = type(error).__name__ + ": " + str(error)
                    raise
                finally:
                    with (out / "requests.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    save()
        state["complete"] = len(state["records"]) == len(TASKS) * REPEATS
        state["quality_passed"] = state["complete"] and all(x["passed"] is True for x in state["records"])
        state["verdict_counts"] = {key: sum(x["verdict"] == key for x in state["records"]) for key in ("PASS", "FAIL", "ERROR")}
        if baseline is not None:
            state["baseline_comparison"] = compare_baseline(baseline, state["records"], suite_sha)
        state["acceptance_passed"] = state["quality_passed"] and (baseline is None or state["baseline_comparison"]["no_regression_passed"])
        state["status"] = "EXECUTED_COMPLETE" if state["complete"] else "INCOMPLETE"
    except BaseException as error:
        state.update(complete=False, quality_passed=False, acceptance_passed=False, status="INCOMPLETE_ERROR", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        changed = sha(Path(__file__).read_bytes()) != source_sha
        if args.baseline:
            changed |= sha(args.baseline.read_bytes()) != state["baseline"]["sha256"]
        if args.manifest:
            changed |= sha(args.manifest.read_bytes()) != state["server_manifest"]["sha256"]
        state["identities_unchanged"] = not changed
        if changed:
            state.update(quality_passed=False, acceptance_passed=False, complete=False, status="IDENTITY_CHANGED")
        save()
    print(json.dumps({k: state.get(k) for k in ("status", "complete", "verdict_counts", "quality_passed", "acceptance_passed")}, ensure_ascii=False))
    return 0 if state.get("acceptance_passed") is True else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
