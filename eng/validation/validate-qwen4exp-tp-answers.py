#!/usr/bin/env python3
"""Validate the five fixed Qwen4Exp TP CLI semantic smoke cases.

Supply the CLI output JSONL and its request JSONL to verify complete answers
and per-request completion budgets. Optionally compare exact UTF-8 outputs
against a layer-split reference. Model Python is parsed and interpreted as
bounded, restricted AST data, never executed. Keep generated reports ignored.

Example from the repository root::

    python3 eng/validation/validate-qwen4exp-tp-answers.py OUTPUT.jsonl \\
        --requests InferenceWeb.Tests/Fixtures/Qwen4Exp/tp-quality-requests.jsonl \\
        --report docs/validation/qwen38-q2-tp/answers.json
"""
import argparse
import ast
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

CASE_IDS = ("arithmetic", "extraction", "python", "squares", "thinking")
EXPECTED_SQUARES = [value * value for value in range(1, 21)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError(f"{path}:{line_number}: expected an object with a string id")
        if value["id"] in records:
            raise ValueError(f"{path}:{line_number}: duplicate id {value['id']}")
        records[value["id"]] = value
    if set(records) != set(CASE_IDS):
        raise ValueError(f"{path}: expected exactly {CASE_IDS}; found {tuple(records)}")
    return records


def final_text(output, require_thinking=False):
    if output.count("</think>") > 1:
        raise ValueError("duplicate thinking terminators")
    if require_thinking and "</think>" not in output:
        raise ValueError("thinking answer did not close </think>")
    final = output.split("</think>", 1)[-1].strip()
    if not final or "<think>" in final or "</think>" in final:
        raise ValueError("empty or unclosed final answer")
    if re.search(r"<\|[^>]*\|>", final):
        raise ValueError("unexpected model control token in final answer")
    return final


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class RestrictedSquares:
    """Small, bounded interpreter with only local variables and range calls."""
    NODES = {
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
        ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
        ast.ListComp, ast.comprehension, ast.Call, ast.BinOp, ast.Add,
        ast.Sub, ast.Mult, ast.Pow, ast.UnaryOp, ast.USub, ast.UAdd,
        ast.Not, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Compare,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.BoolOp, ast.And, ast.Or, ast.Pass, ast.Expr,
    }

    def __init__(self, source):
        if len(source) > 10000:
            raise ValueError("Python source exceeds bounded fixture limit")
        fence = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n?```", source.strip(), re.DOTALL)
        if fence:
            source = fence[1]
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Python must contain only one function definition")
        function = tree.body[0]
        args = function.args
        parameters = args.posonlyargs + args.args
        if (function.name != "squares" or len(parameters) != 1 or parameters[0].arg != "n"
                or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
                or args.kw_defaults or function.decorator_list or getattr(function, "type_params", [])):
            raise ValueError("expected undecorated squares(n) with exactly one required parameter")
        # These inert annotations have no runtime semantics in this interpreter.
        annotation = parameters[0].annotation
        if annotation is not None and not (isinstance(annotation, ast.Name) and annotation.id == "int"):
            raise ValueError("only an optional int parameter annotation is accepted")
        returns = function.returns
        if returns is not None:
            valid = isinstance(returns, ast.Name) and returns.id == "list"
            valid |= (isinstance(returns, ast.Subscript) and isinstance(returns.value, ast.Name)
                and returns.value.id == "list" and isinstance(returns.slice, ast.Name) and returns.slice.id == "int")
            if not valid:
                raise ValueError("only an optional list[int] return annotation is accepted")
        parameters[0].annotation = None
        function.returns = None
        nodes = list(ast.walk(tree))
        if len(nodes) > 200:
            raise ValueError("Python AST exceeds bounded fixture limit")
        for node in nodes:
            if type(node) not in self.NODES:
                raise ValueError(f"disallowed Python AST node: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("dunder names are disallowed")
            if isinstance(node, ast.Name) and node.id == "range" and isinstance(node.ctx, ast.Store):
                raise ValueError("the permitted range function cannot be shadowed")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                    or node.func.id != "range" or node.keywords or not 1 <= len(node.args) <= 3):
                raise ValueError("range is the only permitted function call")
            if isinstance(node, (ast.For, ast.comprehension)) and not isinstance(node.target, ast.Name):
                raise ValueError("iteration targets must be local names")
            if isinstance(node, ast.comprehension) and node.is_async:
                raise ValueError("async comprehensions are disallowed")
            if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
                raise ValueError("assignments must target one local name")
            if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
                raise ValueError("augmented assignments must target a local name")
            if isinstance(node, ast.Expr) and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError("only docstrings may be standalone expressions")
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, bool, str)) or (isinstance(node.value, int) and abs(node.value) > 10**9):
                    raise ValueError("unsupported or oversized constant")
        self.body = function.body

    def tick(self):
        self.steps += 1
        if self.steps > 5000:
            raise ValueError("restricted interpreter exceeded its step budget")

    def bounded(self, value):
        if isinstance(value, (list, tuple, range, str)) and len(value) > 10000:
            raise ValueError("oversized interpreted collection")
        if isinstance(value, int) and abs(value) > 10**9:
            raise ValueError("oversized interpreted integer")
        return value

    def binary(self, operator, left, right):
        if isinstance(operator, ast.Add):
            value = left + right
        elif isinstance(operator, ast.Sub):
            value = left - right
        elif isinstance(operator, ast.Mult):
            if type(left) is not int or type(right) is not int:
                raise ValueError("multiplication only accepts bounded integers")
            value = left * right
        elif isinstance(operator, ast.Pow):
            if type(left) is not int or type(right) is not int or not 0 <= right <= 8:
                raise ValueError("power only accepts bounded integer exponents 0..8")
            value = left ** right
        else:
            raise ValueError("unsupported arithmetic operator")
        return self.bounded(value)

    def expression(self, node, environment):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in environment:
                raise ValueError(f"unknown local name {node.id}")
            return environment[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.expression(item, environment) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left, environment), self.expression(node.right, environment))
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, environment)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) is not int:
                raise ValueError("unary arithmetic only accepts integers")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call):
            values = [self.expression(item, environment) for item in node.args]
            if any(type(value) is not int or abs(value) > 10000 for value in values):
                raise ValueError("range arguments must be bounded integers")
            return self.bounded(range(*values))
        if isinstance(node, ast.ListComp):
            result = []
            def collect(index, local):
                if index == len(node.generators):
                    result.append(self.expression(node.elt, local))
                    self.bounded(result)
                    return
                generator = node.generators[index]
                for value in self.expression(generator.iter, local):
                    self.tick()
                    child = dict(local, **{generator.target.id: value})
                    if all(self.expression(condition, child) for condition in generator.ifs):
                        collect(index + 1, child)
            collect(0, dict(environment))
            return result
        if isinstance(node, ast.Compare):
            left = self.expression(node.left, environment)
            for operator, other in zip(node.ops, node.comparators):
                right = self.expression(other, environment)
                passed = (left == right if isinstance(operator, ast.Eq) else left != right if isinstance(operator, ast.NotEq)
                    else left < right if isinstance(operator, ast.Lt) else left <= right if isinstance(operator, ast.LtE)
                    else left > right if isinstance(operator, ast.Gt) else left >= right)
                if not passed:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.expression(item, environment)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        raise ValueError(f"unsupported expression {type(node).__name__}")

    def statements(self, body, environment):
        for node in body:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expression(node.value, environment))
            if isinstance(node, ast.Assign):
                environment[node.targets[0].id] = self.expression(node.value, environment)
            elif isinstance(node, ast.AugAssign):
                environment[node.target.id] = self.binary(node.op, environment[node.target.id], self.expression(node.value, environment))
            elif isinstance(node, ast.For):
                for value in self.expression(node.iter, environment):
                    self.tick()
                    environment[node.target.id] = value
                    self.statements(node.body, environment)
                self.statements(node.orelse, environment)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expression(node.test, environment) else node.orelse, environment)
            elif isinstance(node, (ast.Pass, ast.Expr)):
                continue
            else:
                raise ValueError(f"unsupported statement {type(node).__name__}")

    def run(self, n):
        self.steps = 0
        try:
            self.statements(self.body, {"n": n})
        except Returned as returned:
            return returned.value
        raise ValueError("squares(n) did not return a result")


def semantic(case_id, output):
    text = final_text(output, require_thinking=case_id == "thinking")
    if case_id == "arithmetic":
        return dict(passed=text == "703", expected="703", actual=text)
    if case_id == "extraction":
        return dict(passed=text == "maya.chen@example.org", expected="maya.chen@example.org", actual=text)
    if case_id == "squares":
        if not re.fullmatch(r"\d+(?:\s*,\s*\d+){19}", text):
            raise ValueError("expected exactly twenty comma-separated integers")
        actual = [int(value.strip()) for value in text.split(",")]
        return dict(passed=actual == EXPECTED_SQUARES, expected=EXPECTED_SQUARES, actual=actual)
    if case_id == "thinking":
        matches = re.findall(r"(?<![\d.])(?:\$\s*(\d+(?:\.\d{1,2})?)|(\d+\.\d{2}))(?!\d|\.\d)", text)
        amounts = [currency or decimal for currency, decimal in matches]
        passed = bool(amounts) and amounts[-1] == "66.00" and Decimal(amounts[-1]) == Decimal("66.00")
        return dict(passed=passed, expected_final_amount="66.00", amounts_after_think=amounts, actual=text)
    interpreter = RestrictedSquares(text)
    checks = []
    for n in (0, 1, 5, 10):
        actual = interpreter.run(n)
        expected = [value * value for value in range(1, n + 1)]
        passed = isinstance(actual, list) and all(type(value) is int for value in actual) and actual == expected
        checks.append(dict(n=n, passed=passed, expected=expected, actual=actual))
    return dict(passed=all(item["passed"] for item in checks), interpreter="restricted AST; no Python eval/exec", checks=checks)


def evaluate(records, requests):
    results = []
    for case_id in CASE_IDS:
        record, request = records[case_id], requests[case_id]
        budget, count = request.get("max_tokens"), record.get("tokens_generated")
        budget_passed = (type(budget) is int and budget > 0 and type(count) is int and 0 < count < budget)
        errors = [f"error field present: {record['error']}"] if "error" in record else []
        if record.get("errors"):
            errors.append(str(record["errors"]))
        reason = record.get("finish_reason", record.get("finishReason"))
        if reason is not None and reason not in ("stop", "eos", "end_turn", "stop_sequence"):
            errors.append(f"noncompletion finish reason: {reason}")
        if record.get("truncated") or record.get("status") in ("error", "failed", "aborted", "cancelled"):
            errors.append("explicit failure/truncation status")
        try:
            output = record.get("output")
            if not isinstance(output, str):
                raise ValueError("output is not a string")
            outcome = semantic(case_id, output)
        except Exception as error:
            outcome = dict(passed=False, error=f"{type(error).__name__}: {error}")
        results.append(dict(id=case_id, passed=budget_passed and not errors and outcome["passed"],
            completion_tokens=count, request_budget=budget, completed_below_budget=budget_passed,
            errors=errors, semantic=outcome))
    return dict(passed=all(item["passed"] for item in results), cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=Path, required=True, help="Matching CLI request JSONL with per-case max_tokens budgets")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = dict(passed=False, output=str(args.output), requests=str(args.requests),
        limitations=["Five deterministic semantic smoke cases; no broad model-quality score.",
            "CLI JSONL omits finish reason; below-budget completion plus strict semantics rejects budget truncation.",
            "Reference comparison, when provided, compares exact UTF-8 output bytes without normalization."])
    try:
        requests, output = load_jsonl(args.requests), load_jsonl(args.output)
        report.update(output_sha256=digest(args.output), requests_sha256=digest(args.requests), candidate=evaluate(output, requests))
        report["passed"] = report["candidate"]["passed"]
        if args.reference:
            reference = load_jsonl(args.reference)
            comparison = [dict(id=case_id,
                output_bytes_equal=output[case_id].get("output", "").encode("utf-8") == reference[case_id].get("output", "").encode("utf-8"),
                tokens_equal=output[case_id].get("tokens_generated") == reference[case_id].get("tokens_generated")) for case_id in CASE_IDS]
            report.update(reference=str(args.reference), reference_sha256=digest(args.reference),
                reference_quality=evaluate(reference, requests), reference_comparison=comparison)
            report["passed"] &= report["reference_quality"]["passed"] and all(item["output_bytes_equal"] for item in comparison)
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
