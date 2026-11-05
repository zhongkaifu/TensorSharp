#!/usr/bin/env python3
"""CPU-only tests: complete modules execute; no server, requests or CUDA."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

PATH = Path(__file__).with_name("qwen38-code-program-quality.py")
SPEC = importlib.util.spec_from_file_location("program_quality", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

# This check is itself a separate CPU process: never lower the test runner's
# limits. Unsupported platforms are explicitly skipped, never counted as
# validated full-program execution. Production fails before model requests.
LIMIT_CHECK = subprocess.run([sys.executable, "-I", "-S", "-c",
    "import resource;resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))"], capture_output=True)
LIMITS_SUPPORTED = LIMIT_CHECK.returncode == 0

PROGRAMS = {
    "primes": """n=d['n']
p=[]
for x in range(2,n):
    if all(x%y for y in range(2,int(x**0.5)+1)):
        p.append(x)
r=p
""",
    "sort_integers": "r=sorted(d['values'])\n",
    "balanced_brackets": """stack=[]
closing={')':'(',']':'[','}':'{'}
r=True
for c in d['text']:
    if c in '([{':
        stack.append(c)
    elif not stack or stack.pop()!=closing[c]:
        r=False
        break
r=r and not stack
""",
    "run_length": """r=[]
for c in d['text']:
    if r and r[-1][0]==c:
        r[-1][1]+=1
    else:
        r.append([c,1])
""",
    "merge_intervals": """r=[]
for a,b in sorted(d['intervals']):
    if r and a<=r[-1][1]:
        r[-1][1]=max(r[-1][1],b)
    else:
        r.append([a,b])
""",
    "top_k_frequent": """counts={}
for v in d['values']:
    counts[v]=counts.get(v,0)+1
r=sorted(counts,key=lambda x:(-counts[x],x))[:max(0,d['k'])]
""",
    "transpose": "r=[list(x) for x in zip(*d['matrix'])] if d['matrix'] else []\n",
    "grid_shortest_path": """def solve(g,s,e):
    if not g or not g[0]:
        return -1
    h,w=len(g),len(g[0])
    for x,y in (s,e):
        if x<0 or y<0 or x>=h or y>=w or g[x][y]:
            return -1
    todo=[(s[0],s[1],0)]
    seen={tuple(s)}
    for x,y,n in todo:
        if [x,y]==e:
            return n
        for a,b in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0<=a<h and 0<=b<w and not g[a][b] and (a,b) not in seen:
                seen.add((a,b))
                todo.append((a,b,n+1))
    return -1
r=solve(d['grid'],d['start'],d['end'])
""",
    "decimal_add": "r=str(int(d['a'])+int(d['b']))\n",
    "unique_permutations": """def perm(values):
    if not values:
        return [[]]
    result=[]
    for i,v in enumerate(values):
        if i and v==values[i-1]:
            continue
        for p in perm(values[:i]+values[i+1:]):
            result.append([v]+p)
    return result
r=perm(sorted(d['values']))
""",
}


def complete(body):
    return "import json, sys\nd=json.loads(sys.stdin.read())\n" + body + "print(json.dumps(r,ensure_ascii=False))\n"


def records():
    return [{"task": task["id"], "repeat": repeat, "passed": True, "verdict": "PASS",
             "request": {"request_body": tool.request_body("same-model", task, repeat, 20261102),
                         "request_sha256": tool.sha(tool.encoded(tool.request_body("same-model", task, repeat, 20261102))),
                         "wire_body_sha256": tool.sha(tool.encoded(tool.request_body("same-model", task, repeat, 20261102))),
                         "error": None, "content": "synthetic comparison trace", "finish_reason": "stop"},
             "evaluation": {"passed": True, "content_sha256": tool.sha(b"synthetic comparison trace"),
                            "checks": {"finish_reason_stop": True, "complete_module": True, "closed_AST_policy": True, "all_functional_cases": True},
                            "cases": [{"case_id": c["id"], "input": c["input"], "input_sha256": tool.sha(tool.encoded(c["input"])),
                                       "expected": c["expected"], "actual": c["expected"], "passed": True, "status": "PASS",
                                       "exit_code": 0, "worker": {"status": "COMPLETE"},
                                       "stdout": tool.encoded(c["expected"]).decode(), "stdout_bytes": len(tool.encoded(c["expected"])),
                                       "stdout_sha256": tool.sha(tool.encoded(c["expected"]))} for c in task["cases"]]}}
            for task in tool.TASKS for repeat in range(tool.REPEATS)]


def baseline(rows):
    return {"schema": tool.VERSION, "suite_sha256": tool.sha(tool.encoded(tool.TASKS)), "tool_sha256": tool.sha(PATH.read_bytes()),
            "model": "same-model", "nonce_seed": 20261102, "repeats": 3, "max_tokens": 4096, "complete": True, "records": rows}


def mark_wrong(row):
    row.update(passed=False, verdict="FAIL")
    row["evaluation"].update(passed=False)
    row["evaluation"]["checks"]["all_functional_cases"] = False
    row["evaluation"]["cases"][0].update(passed=False, status="WRONG_OUTPUT", actual=None, stdout="null", stdout_bytes=4, stdout_sha256=tool.sha(b"null"))


class ProgramQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, source, *, cases=None, finish="stop", name="evaluation"):
        return tool.evaluate_program(source, finish, cases or tool.TASKS[0]["cases"], self.root / name)

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; full restricted-program execution NOT_RUN")
    def test_ten_original_task_contracts_and_complete_reference_modules(self):
        self.assertEqual(set(PROGRAMS), {t["id"] for t in tool.TASKS})
        self.assertEqual(len(tool.TASKS), 10)
        self.assertEqual(sum(len(t["cases"]) for t in tool.TASKS), 63)
        for task in tool.TASKS:
            with self.subTest(task=task["id"]):
                result = self.evaluate(complete(PROGRAMS[task["id"]]), cases=task["cases"], name=task["id"])
                self.assertTrue(result["passed"], result)
                self.assertEqual(len(result["cases"]), len(task["cases"]))
                for case in result["cases"]:
                    self.assertEqual(case["worker"]["status"], "COMPLETE")
                    self.assertEqual(set(case["worker"]["limits"]), {"RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_NOFILE"})

    def test_fixture_ground_truth_boundaries_not_inferred_from_candidate(self):
        tasks = {t["id"]: t for t in tool.TASKS}
        self.assertEqual(tasks["primes"]["cases"][5]["expected"], [2, 3, 5, 7, 11, 13, 17, 19, 23])
        self.assertEqual(tasks["merge_intervals"]["cases"][3]["expected"], [[-3, 2], [5, 7]])
        self.assertEqual(tasks["grid_shortest_path"]["cases"][5]["expected"], 6)
        self.assertEqual(tasks["decimal_add"]["cases"][-1]["expected"], "1" + "0" * 80)
        self.assertEqual(tasks["unique_permutations"]["cases"][0]["expected"], [[]])
        self.assertIs(tasks["balanced_brackets"]["cases"][0]["expected"], True)
        for body in PROGRAMS.values():
            tool.validate_program(tool.extract_program(complete(body)))

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; restricted-program execution NOT_RUN")
    def test_function_only_reply_is_not_a_program_pass(self):
        result = self.evaluate("def is_prime(n):\n    return True\n")
        self.assertFalse(result["passed"])
        self.assertTrue(all(c["status"] == "INVALID_STDOUT_JSON" for c in result["cases"]))

    def test_truncation_and_unclosed_or_extra_fences_never_repaired(self):
        with mock.patch.object(tool, "run_case") as run:
            self.assertFalse(self.evaluate(complete(PROGRAMS["primes"]), finish="length")["passed"])
            self.assertFalse(self.evaluate("```python\nprint([])\n")["passed"])
            self.assertFalse(self.evaluate("Explanation\n```python\nprint([])\n```\n")["passed"])
            self.assertFalse(self.evaluate("def main(:\n")["passed"])
            run.assert_not_called()

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; restricted-program execution NOT_RUN")
    def test_one_closed_fence_complete_module_is_executed(self):
        result = self.evaluate("```python\n" + complete(PROGRAMS["primes"]) + "```\n")
        self.assertTrue(result["passed"], result)
        bare_except = "import json, sys\nd=json.loads(sys.stdin.read())\ntry:\n    r=d['n']\nexcept:\n    r=-1\nprint(json.dumps(r))\n"
        cases = [{"id": 0, "input": {"n": 7}, "expected": 7}, {"id": 1, "input": {}, "expected": -1}]
        self.assertTrue(self.evaluate(bare_except, cases=cases, name="bare-except")["passed"])

    def test_dangerous_ast_imports_calls_attributes_and_io_are_rejected(self):
        sources = ["import os\n", "from json import loads\n", "import json as j\n", "print((1).__class__)\n",
                   "open('/tmp/x','w')\n", "eval('1')\n", "exec('print(1)')\n", "getattr(1,'real')\n",
                   "import sys\nprint(sys.modules)\n", "import sys\nsys.stderr.write('x')\n",
                   "import sys\nprint(1,file=sys.stdout)\n", "class X:\n    pass\n",
                   "import json\njson.loads('1',object_hook=print)\n", "print('{0.__class__}'.format(1))\n",
                   "f=print\nf(1)\n", "def f():\n    pass\nf=json.loads\nf('{}',object_hook=print)\n",
                   "def f(print):\n    print('x')\n", "def json():\n    pass\n", "import sys\nsys.stdin.read(1)\n",
                   "if __name__=='main':\n    pass\n"]
        for source in sources:
            with self.subTest(source=source), self.assertRaises((tool.Rejected, SyntaxError)):
                tool.validate_program(source)

    def test_optional_ast_names_allow_bare_except_and_reject_expanded_kwargs(self):
        for handler in ("except:", "except ValueError:"):
            tool.validate_program("try:\n    raise ValueError('x')\n" + handler + "\n    pass\nprint([])\n")
        with mock.patch.object(tool, "run_case") as run:
            result = self.evaluate("print(**{})\n")
            self.assertFalse(result["passed"])
            self.assertIn("expanded keyword", result["error"])
            run.assert_not_called()

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; restricted-program execution NOT_RUN")
    def test_wrong_program_and_bad_stdout_are_scored_not_skipped(self):
        wrong = self.evaluate("print([])\n")
        self.assertFalse(wrong["passed"])
        self.assertEqual(sum(x["passed"] for x in wrong["cases"]), 3)
        malformed = self.evaluate("print('debug'); print([])\n", name="malformed")
        self.assertFalse(malformed["passed"])
        self.assertTrue(all(x["status"] == "INVALID_STDOUT_JSON" for x in malformed["cases"]))

    def test_exact_json_type_and_integer_precision_no_error_tolerance(self):
        self.assertFalse(tool.strict_equal(True, 1))
        self.assertFalse(tool.strict_equal(1.0, 1))
        self.assertFalse(tool.strict_equal(9007199254740992, 9007199254740993))
        self.assertTrue(tool.strict_equal(tool.parse_json("9007199254740993"), 9007199254740993))
        for raw in ('NaN', 'Infinity', '{"x":1,"x":2}', '1 2'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                tool.parse_json(raw)

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; restricted-program execution NOT_RUN")
    def test_worker_exception_timeout_and_output_cap_fail_closed(self):
        case = [{"id": 0, "input": {}, "expected": 0}]
        error = self.evaluate("raise ValueError('intentional')\n", cases=case)
        self.assertFalse(error["passed"])
        self.assertNotEqual(error["cases"][0]["exit_code"], 0)
        self.assertEqual(error["cases"][0]["worker"]["status"], "PROGRAM_ERROR")
        program = self.root / "loop.py"
        program.write_text("while True:\n    pass\n")
        timed = tool.run_case(program, case[0], self.root / "timed", timeout=.1)
        self.assertEqual(timed["status"], "TIMEOUT")
        self.assertFalse(timed["passed"])
        output = self.evaluate("print('x'*200000)\n", cases=case, name="oversized")
        self.assertFalse(output["passed"])
        self.assertLessEqual(output["cases"][0]["stdout_bytes"], tool.OUTPUT_LIMIT)

    @unittest.skipUnless(LIMITS_SUPPORTED, "mandatory RLIMIT_AS unavailable; restricted-program execution NOT_RUN")
    def test_program_memory_budget_is_installed_and_allocation_cannot_pass(self):
        result = self.evaluate("r=[0]*200000000\nprint(0)\n", cases=[{"id": 0, "input": {}, "expected": 0}])
        self.assertFalse(result["passed"])
        self.assertNotEqual(result["cases"][0]["exit_code"], 0)

    def test_request_matrix_is_fixed_complete_and_cross_label_identical(self):
        rows = [tool.request_body("model", t, r, 7) for t in tool.TASKS for r in range(3)]
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({tool.sha(tool.encoded(r)) for r in rows}), 30)
        for row in rows:
            self.assertEqual((row["max_tokens"], row["temperature"], row["top_p"], row["seed"], row["stream"]), (4096, 0, 1, 1234, False))
        self.assertEqual(rows[0], tool.request_body("model", tool.TASKS[0], 0, 7))

    def test_baseline_retains_per_case_regression_without_netting_recovery(self):
        old = records()
        mark_wrong(old[1])
        new = records()
        mark_wrong(new[0])
        reference = baseline(old)
        result = tool.compare_baseline(reference, new, reference["suite_sha256"])
        self.assertTrue(result["comparable"])
        self.assertEqual(len(result["true_to_false"]), 1)
        self.assertEqual(len(result["false_to_true"]), 1)
        self.assertFalse(result["no_regression_passed"])

    def test_incomplete_or_changed_baseline_is_not_comparable(self):
        rows = records()
        reference = baseline(copy.deepcopy(rows))
        rows[0]["request"]["request_sha256"] = "changed"
        result = tool.compare_baseline(reference, rows, reference["suite_sha256"])
        self.assertFalse(result["comparable"])
        self.assertFalse(result["no_regression_passed"])
        reference["complete"] = False
        self.assertFalse(tool.compare_baseline(reference, rows, reference["suite_sha256"])["comparable"])

    def test_corrupt_case_matrices_and_declared_truth_never_become_comparable(self):
        mutations = [lambda x: x.append(copy.deepcopy(x[0])), lambda x: x.pop(),
                     lambda x: x[0].update(case_id=False), lambda x: x[0].update(case_id=99),
                     lambda x: x[0].update(passed=False), lambda x: x[0].update(actual=[999]),
                     lambda x: x[0].update(input={"n": 999}), lambda x: x[0].update(expected=[999]),
                     lambda x: x[0].update(status="INVALID_STDOUT_JSON"), lambda x: x[0].update(status="PROGRAM_OR_LIMIT_FAILURE")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                rows = records()
                old = copy.deepcopy(rows)
                mutate(old[0]["evaluation"]["cases"])
                reference = baseline(old)
                result = tool.compare_baseline(reference, rows, reference["suite_sha256"])
                self.assertFalse(result["comparable"], result)
                self.assertFalse(result["no_regression_passed"])

    def test_declared_hash_does_not_hide_wrong_body_or_task_nonce(self):
        rows = records()
        reference = baseline(copy.deepcopy(rows))
        self.assertTrue(tool.compare_baseline(reference, rows, reference["suite_sha256"])["no_regression_passed"])
        rows[0]["request"]["request_body"]["messages"][0]["content"] += " changed"
        self.assertFalse(tool.compare_baseline(reference, rows, reference["suite_sha256"])["comparable"])
        rows = records()
        rows[0]["request"]["request_body"]["max_tokens"] = 4096.0
        self.assertFalse(tool.compare_baseline(reference, rows, reference["suite_sha256"])["comparable"])
        rows = records()
        reference["nonce_seed"] = 999
        self.assertFalse(tool.compare_baseline(reference, rows, reference["suite_sha256"])["comparable"])

    def test_entire_executed_case_array_cannot_be_erased_as_not_executed(self):
        rows = records()
        old = copy.deepcopy(rows)
        old[0].update(passed=False, verdict="FAIL")
        old[0]["evaluation"].update(passed=False, cases=[])
        old[0]["evaluation"]["checks"]["all_functional_cases"] = False
        reference = baseline(old)
        result = tool.compare_baseline(reference, rows, reference["suite_sha256"])
        self.assertFalse(result["comparable"], result)
        self.assertFalse(result["no_regression_passed"])
        self.assertIn("execution-ready", result["error"])
        with mock.patch.object(tool, "run_case") as run:
            empty = self.evaluate("# no executable program\n")
            self.assertFalse(empty["passed"])
            self.assertEqual(empty["cases"], [])
            run.assert_not_called()

    def test_truncated_candidate_keeps_not_executed_coverage_and_prior_pass_regressions(self):
        rows = records()
        reference = baseline(copy.deepcopy(rows))
        rows[0].update(passed=False, verdict="FAIL")
        rows[0]["request"]["finish_reason"] = "length"
        rows[0]["evaluation"].update(passed=False, cases=[], checks={"finish_reason_stop": False, "complete_module": False, "closed_AST_policy": False})
        result = tool.compare_baseline(reference, rows, reference["suite_sha256"])
        self.assertTrue(result["comparable"])
        self.assertFalse(result["no_regression_passed"])
        pair = next(x for x in result["pairs"] if (x["task"], x["repeat"]) == (rows[0]["task"], 0))
        self.assertEqual(pair["candidate_case_execution"], "NOT_EXECUTED")
        self.assertTrue(all(not x["both_cases_executed"] for x in pair["cases"]))
        self.assertEqual(len(result["true_to_false"]), len(tool.TASKS[0]["cases"]))

    def test_plan_never_requests_or_executes_and_foreign_output_is_preserved(self):
        out = self.root / "artifacts" / "plan"
        with mock.patch.object(tool, "query") as query, mock.patch.object(tool, "evaluate_program") as evaluate, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(tool.main(["--out", str(out)]), 0)
            query.assert_not_called()
            evaluate.assert_not_called()
        data = json.loads((out / "campaign.json").read_text())
        self.assertFalse(data["complete"])
        self.assertFalse(data["quality_passed"])
        self.assertEqual(data["planned_programs"], 30)
        original = (out / "campaign.json").read_bytes()
        with self.assertRaises(ValueError):
            tool.main(["--out", str(out)])
        self.assertEqual((out / "campaign.json").read_bytes(), original)

    def test_error_requests_remain_thirty_errors_not_fake_pass(self):
        out = self.root / "artifacts" / "errors"
        def error(url, body, timeout):
            return {"request_body": body, "request_sha256": tool.sha(tool.encoded(body)), "error": "HTTP 500", "raw_response": "actual error body"}
        with mock.patch.object(tool, "sandbox_preflight", return_value={"passed": True, "scope": "mock preflight only"}), mock.patch.object(tool, "query", side_effect=error), mock.patch.object(tool, "evaluate_program") as evaluate, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(tool.main(["--out", str(out), "--execute", "--model", "model"]), 1)
            evaluate.assert_not_called()
        result = json.loads((out / "campaign.json").read_text())
        self.assertTrue(result["complete"])
        self.assertEqual(result["verdict_counts"], {"PASS": 0, "FAIL": 0, "ERROR": 30})
        self.assertFalse(result["quality_passed"])
        self.assertFalse(result["acceptance_passed"])
        self.assertTrue(all(r["request"]["raw_response"] == "actual error body" for r in result["records"]))

    def test_missing_sandbox_limits_prevents_any_model_request(self):
        out = self.root / "artifacts" / "unavailable"
        with mock.patch.object(tool, "sandbox_preflight", return_value={"passed": False, "error": "mandatory resource limit unavailable"}), mock.patch.object(tool, "query") as query, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(tool.main(["--out", str(out), "--execute", "--model", "model"]), 2)
            query.assert_not_called()
        state = json.loads((out / "campaign.json").read_text())
        self.assertEqual(state["status"], "SETUP_ERROR_NO_MODEL_REQUESTS")
        self.assertFalse(state["complete"])
        self.assertFalse(state["quality_passed"])

    def test_received_response_survives_unexpected_evaluator_exception(self):
        out = self.root / "artifacts" / "unexpected-evaluator-error"
        received = {"request_sha256": "original-request-hash", "raw_response": "the complete received response",
                    "raw_response_sha256": "original-response-hash", "content": "print([])", "finish_reason": "stop", "error": None}
        with mock.patch.object(tool, "sandbox_preflight", return_value={"passed": True}), mock.patch.object(tool, "query", return_value=received), mock.patch.object(tool, "evaluate_program", side_effect=TypeError("unexpected evaluator bug")), mock.patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(TypeError, "unexpected evaluator bug"):
                tool.main(["--out", str(out), "--execute", "--model", "model"])
        state = json.loads((out / "campaign.json").read_text())
        journal = [json.loads(x) for x in (out / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(state["status"], "INCOMPLETE_ERROR")
        self.assertFalse(state["quality_passed"])
        self.assertFalse(state["complete"])
        self.assertEqual(len(state["records"]), 1)
        self.assertEqual(state["records"], journal)
        self.assertEqual(journal[0]["request"], received)
        self.assertIn("TypeError", journal[0]["evaluation_error"])


if __name__ == "__main__":
    unittest.main()
