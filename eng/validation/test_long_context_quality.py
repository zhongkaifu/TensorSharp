#!/usr/bin/env python3
"""CPU-only tests of long-context-quality.py: grading, statistics, the report's pairing and gates on synthetic runs, the
estimate's fit. No model, GPU or network."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATH = Path(__file__).with_name("long-context-quality.py")
SPEC = importlib.util.spec_from_file_location("long_context_quality", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

LIMIT_CHECK = subprocess.run([sys.executable, "-I", "-S", "-c",
    "import resource;resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))"], capture_output=True)
LIMITS_SUPPORTED = LIMIT_CHECK.returncode == 0

RETRIEVAL_GRADE = {"type": "retrieval", "codes": {"alpha": "AMBER-OTTER-1234", "beta": "OCHRE-LYNX-2222", "gamma": "UMBER-IBEX-3333",
                                                  "delta": "INDIGO-HERON-4444", "epsilon": "SAFFRON-TAPIR-5555"},
                   "order": ["beta", "gamma", "alpha", "delta", "epsilon"], "before_gamma": "beta", "after_gamma": "alpha"}


TOPK = "AAAAAAAAAAA="   # one row's top-1 (id 0, log-probability 0) as the probe encodes it
TOPK_SHA = tool.sha256_bytes(TOPK.encode())


def retrieval_answer(**changes):
    answer = dict(RETRIEVAL_GRADE["codes"], order=RETRIEVAL_GRADE["order"], before_gamma="beta", after_gamma="alpha")
    answer.update(changes)
    return json.dumps(answer)


class Grading(unittest.TestCase):
    def test_retrieval_reads_fenced_and_thinking_answers(self):
        checks = tool.grade_retrieval("<think>hmm</think>\n```json\n" + retrieval_answer() + "\n```", RETRIEVAL_GRADE)
        self.assertEqual(8, len(checks))
        self.assertTrue(all(checks.values()))

    def test_retrieval_counts_each_wrong_field(self):
        checks = tool.grade_retrieval(retrieval_answer(alpha="AMBER-OTTER-1235", order=["alpha"]), RETRIEVAL_GRADE)
        self.assertFalse(checks["needle_alpha"])
        self.assertFalse(checks["position_order"])
        self.assertTrue(checks["needle_beta"])
        self.assertFalse(any(tool.grade_retrieval("no json here", RETRIEVAL_GRADE).values()))

    def test_gsm8k_takes_the_last_answer_line(self):
        self.assertTrue(tool.grade_gsm8k("3 + 5 = 8\nAnswer: 8,000", {"answer": "8000"}))
        self.assertTrue(tool.grade_gsm8k("Answer: 7\nwait\n**Answer:** $8000", {"answer": "8000"}))
        self.assertFalse(tool.grade_gsm8k("The answer is 8000.", {"answer": "8000"}))
        self.assertFalse(tool.grade_gsm8k("<think>Answer: 8000", {"answer": "8000"}))

    def test_humaneval_program_prepends_the_prompt_only_for_a_body(self):
        grade = {"prompt": "def f(x):\n    \"\"\"double\"\"\"\n", "test": "def check(c):\n    assert c(2) == 4\n", "entry_point": "f"}
        full = tool.humaneval_program("```python\ndef f(x):\n    return 2 * x\n```", grade)
        self.assertEqual(1, full.count("def f("))
        body = tool.humaneval_program("    return 2 * x\n", grade)
        self.assertTrue(body.startswith(grade["prompt"]))
        self.assertTrue(full.rstrip().endswith("check(f)"))

    @unittest.skipUnless(LIMITS_SUPPORTED, "RLIMIT_AS cannot be set here; HumanEval execution is refused on such hosts")
    def test_humaneval_runs_pass_fail_and_timeout(self):
        grade = {"prompt": "def f(x):\n", "test": "def check(c):\n    assert c(2) == 4\n", "entry_point": "f"}
        self.assertEqual("PASS", tool.run_humaneval(tool.humaneval_program("def f(x):\n    return 2 * x\n", grade)))
        self.assertEqual("FAIL", tool.run_humaneval(tool.humaneval_program("def f(x):\n    return x\n", grade)))
        self.assertEqual("FAIL", tool.run_humaneval(tool.humaneval_program("def f(x):\n    import os\n    os.system('true')\n    return 4\n", grade)))
        self.assertEqual("TIMEOUT", tool.run_humaneval(tool.humaneval_program("def f(x):\n    while True: pass\n", grade)))


class Statistics(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(1.0, tool.mcnemar_p(0, 0))
        self.assertAlmostEqual(2 / 32, tool.mcnemar_p(5, 0))
        self.assertEqual(1.0, tool.mcnemar_p(1, 1))

    def test_prefill_fit_recovers_a_quadratic(self):
        points = [(L, 0.15 * L + 0.0034 * L * L) for L in (4, 8, 32, 64)]
        a, b = tool.fit_prefill(points)
        self.assertAlmostEqual(0.15, a, places=6)
        self.assertAlmostEqual(0.0034, b, places=8)

    def test_block_se_needs_two_blocks(self):
        self.assertTrue(math.isnan(tool.block_se([[0.1] * 100])))
        self.assertAlmostEqual(0.0, tool.block_se([[0.1] * 256]))

    def test_outputs_stay_out_of_the_repository(self):
        with self.assertRaises(SystemExit):
            tool.ignored_output(tool.ROOT / "eng" / "x")
        self.assertEqual(tool.ROOT / "artifacts" / "q", tool.ignored_output(tool.ROOT / "artifacts" / "q"))


class ReportOnSyntheticRuns(unittest.TestCase):
    def write(self, d, name, rows):
        p = Path(d) / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    def runs(self, d, cand_nll_shift=0.0, cand_order=None, greedy_cut=None):
        cases = [{"type": "header", "schema": tool.SCHEMA},
                 {"type": "case", "id": "ppl/prose/8k/w0", "kind": "ppl", "corpus": "prose", "tokens": 8192, "start_token": 0, "score_tokens": 512},
                 {"type": "case", "id": "greedy/code/8k/p0", "kind": "greedy", "corpus": "code", "tokens": 8192, "end_char": 9, "new_tokens": 4},
                 {"type": "case", "id": "retrieval/8k/r0", "kind": "chat", "group": "retrieval", "fit_tokens": 7900, "max_new_tokens": 192,
                  "messages": [], "grade": RETRIEVAL_GRADE},
                 {"type": "case", "id": "gsm8k/test-1", "kind": "chat", "group": "gsm8k", "fit_tokens": 4096, "max_new_tokens": 64,
                  "messages": [], "grade": {"type": "gsm8k", "answer": "12"}}]
        cases_path = self.write(d, "cases.jsonl", cases)
        nll = [0.5 + 0.01 * (i % 7) for i in range(512)]
        gen = [1, 2, 3, 4]

        def records(arm, shift, order, cut):
            g = gen if cut is None else gen[:cut] + [9] * (4 - cut)
            return [
                {"type": "run", "arms": [{"name": arm, "prompt_attention": arm}], "cases_file": str(cases_path), "backend": "cuda"},
                {"type": "record", "case": "ppl/prose/8k/w0", "kind": "ppl", "arm": arm, "status": "ok", "input_sha256": "x",
                 "nll": [x + shift for x in nll], "top1": [7] * 512, "kl": [1e-4] * 512 if shift else None, "topk_k": 1, "topk": TOPK,
                 "kl_reference": "f16 (this run)" if shift else None, "kl_reference_topk_sha256": TOPK_SHA if shift else None,
                 "attn8_launches": 3 if arm == "8bit" else 0},
                {"type": "record", "case": "greedy/code/8k/p0", "kind": "greedy", "arm": arm, "status": "ok", "input_sha256": "y",
                 "generated": g, "gaps": [5.0, 0.4, 3.0, 2.0]},
                {"type": "record", "case": "retrieval/8k/r0", "kind": "chat", "group": "retrieval", "arm": arm, "status": "ok",
                 "prompt_sha256": "z", "prompt_tokens": 7890, "generated": [1], "text": retrieval_answer(**({"order": order} if order else {}))},
                {"type": "record", "case": "gsm8k/test-1", "kind": "chat", "group": "gsm8k", "arm": arm, "status": "ok",
                 "prompt_sha256": "w", "prompt_tokens": 4090, "generated": [1], "text": "Answer: 12"},
            ]
        run = self.write(d, "run.jsonl", records("f16", 0.0, None, None) + records("8bit", cand_nll_shift, cand_order, greedy_cut)[1:])
        return run, cases_path

    def report(self, run, cases, gates):
        ref, cand = tool.load_arm(f"{run}:f16"), tool.load_arm(f"{run}:8bit")
        rep = tool.Report(ref, cand, tool.load_cases(cases)[1], gates, execute_humaneval=False)
        text = rep.run()
        return text, {name: state for name, state, _ in rep.verdicts}, rep.summary

    def test_equal_arms_pass_every_gate(self):
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d)
            text, verdicts, summary = self.report(run, cases, "t2-attn8")
            self.assertTrue(verdicts and all(v == "PASS" for v in verdicts.values()), text)
            self.assertAlmostEqual(0.0, summary["ppl"][0]["delta_ppl_pct"])

    def test_an_arm_with_q8_activations_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d)
            rows = [json.loads(line) for line in run.read_text().splitlines()]
            rows[0]["arms"] = [{"name": "f16", "prompt_attention": "f16", "q8act": True},
                               {"name": "8bit", "prompt_attention": "8bit", "q8act": False}]
            self.write(d, "run.jsonl", rows)
            text, _, _ = self.report(run, cases, "t2-attn8")
            self.assertIn("prompt attention f16, q8 activations (q8act),", text)
            self.assertIn("prompt attention 8bit, records", text)

    def test_a_ppl_shift_a_lost_check_and_a_confident_divergence_fail(self):
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d, cand_nll_shift=0.01, cand_order=["alpha"], greedy_cut=2)
            text, verdicts, summary = self.report(run, cases, "t2-attn8")
            self.assertEqual("FAIL", verdicts["ppl (every corpus and L run)"], text)
            self.assertAlmostEqual((math.exp(0.01) - 1) * 100, summary["ppl"][0]["delta_ppl_pct"], places=6)
            self.assertAlmostEqual(1e-4, summary["ppl"][0]["kl_mean"])
            self.assertEqual("FAIL", verdicts["retrieval (no check the reference passes is lost)"])
            self.assertEqual("FAIL", verdicts["greedy (divergences at near-ties only)"])   # diverged at position 2, gap 3.0
            self.assertEqual("PASS", verdicts["gsm8k"])

    def test_kinds_grades_and_gates_only_those_cases(self):
        # t2-2 on NVFP4 reads its ppl gate from a run whose arms keep q8 activations and every other gate from the served arms:
        # a report over some kinds neither judges nor lists the others, and the exit status is theirs alone.
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d, cand_nll_shift=0.01)
            args = ["report", "--ref", f"{run}:f16", "--cand", f"{run}:8bit", "--cases", str(cases), "--gates", "t2-attn8"]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(1, tool.main(args))
                self.assertEqual(0, tool.main(args + ["--kinds", "greedy,retrieval,gsm8k"]))
                self.assertEqual(1, tool.main(args + ["--kinds", "ppl"]))
            served = out.getvalue().split("# Long-context quality")[2]
            self.assertIn("graded kinds:** greedy, retrieval, gsm8k only", served)
            self.assertNotIn("## ppl", served)
            self.assertNotIn("| ppl", served)
            self.assertIn("| gsm8k | PASS |", served)
            with self.assertRaises(SystemExit):
                tool.main(args + ["--kinds", "humaneval"])   # the case file has none
            with self.assertRaises(SystemExit):
                tool.main(args + ["--kinds", "perplexity"])

    def test_kinds_scope_the_evidence_and_name_each_refused_case(self):
        # Under --kinds the header's record counts and the arms' evidence are the graded cases' alone: a band refusal of a ppl
        # case is not evidence in a report over the served kinds (it read 608 refusals from ungraded 96K ppl cases into a
        # continue-only report), and a report that grades it names the case.
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d)
            rows = [json.loads(line) for line in run.read_text().splitlines()]
            for r in rows:
                if r.get("type") == "record" and r["arm"] == "8bit" and r["kind"] == "ppl":
                    r["attn8_band_refusals"] = 2
            self.write(d, "run.jsonl", rows)
            args = ["report", "--ref", f"{run}:f16", "--cand", f"{run}:8bit", "--cases", str(cases), "--gates", "t2-attn8"]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(0, tool.main(args + ["--kinds", "greedy,retrieval,gsm8k"]))
                self.assertEqual(0, tool.main(args + ["--kinds", "ppl"]))
            served, ppl = out.getvalue().split("# Long-context quality")[1:]
            self.assertIn("8-bit prompt-attention launches 0, band refusals 0,", served)
            self.assertNotIn("ppl/prose/8k/w0", served)
            self.assertEqual(2, served.count("records ok 3;"))
            self.assertIn("band refusals 2 (those chunks ran the 16-bit kernel)", ppl)
            self.assertIn("`ppl/prose/8k/w0`: 2 band refusal(s) beside 3 8-bit launch(es)", ppl)
            self.assertEqual(2, ppl.count("records ok 1;"))

    def test_a_floor_arm_is_tabled_without_gates(self):
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d, cand_nll_shift=0.01, greedy_cut=2)
            ref, cand = tool.load_arm(f"{run}:f16"), tool.load_arm(f"{run}:8bit")
            rep = tool.Report(ref, cand, tool.load_cases(cases)[1], "t2-attn8", False, floor=tool.load_arm(f"{run}:8bit"))
            text = rep.run()
            self.assertIn("### noise floor, ppl", text)
            self.assertAlmostEqual(rep.summary["ppl"][0]["delta_ppl_pct"], rep.summary["floor_ppl"][0]["delta_ppl_pct"])
            self.assertEqual(1, sum(1 for name, _, _ in rep.verdicts if name.startswith("ppl")))

    def test_a_divergence_at_a_near_tie_passes(self):
        with tempfile.TemporaryDirectory() as d:
            run, cases = self.runs(d, greedy_cut=1)
            _, verdicts, summary = self.report(run, cases, "t2-attn8")
            self.assertEqual("PASS", verdicts["greedy (divergences at near-ties only)"])
            self.assertEqual(1, summary["greedy"]["rows"][0]["prefix"])

    # Gates over part of their cases (review of A2): a gate never reads PASS while one of its cases did not pair.
    def case_file(self, d, extra):
        return self.write(d, "cases.jsonl", [{"type": "header", "schema": tool.SCHEMA}] + extra)

    def ppl_case(self, cid, tokens):
        return {"type": "case", "id": cid, "kind": "ppl", "corpus": "prose", "tokens": tokens, "start_token": 0, "score_tokens": 512}

    def ppl_record(self, cid, arm, **fields):
        rec = {"type": "record", "case": cid, "kind": "ppl", "arm": arm, "status": "ok", "input_sha256": cid,
               "nll": [0.5] * 512, "top1": [7] * 512}
        rec.update(fields)
        return rec

    def gated(self, d, cases, rows, gates="t2-attn8"):
        run = self.write(d, "run.jsonl", [{"type": "run"}] + rows)
        ref, cand = tool.load_arm(f"{run}:f16"), tool.load_arm(f"{run}:8bit")
        rep = tool.Report(ref, cand, tool.load_cases(cases)[1], gates, execute_humaneval=False)
        text = rep.run()
        return text, {name: state for name, state, _ in rep.verdicts}, rep

    def test_a_case_the_candidate_failed_fails_its_gate(self):
        with tempfile.TemporaryDirectory() as d:
            cases = self.case_file(d, [self.ppl_case("ppl/prose/8k/w0", 8192), self.ppl_case("ppl/prose/64k/w0", 65536)])
            rows = [self.ppl_record("ppl/prose/8k/w0", "f16"), self.ppl_record("ppl/prose/8k/w0", "8bit"),
                    self.ppl_record("ppl/prose/64k/w0", "f16"),
                    {"type": "record", "case": "ppl/prose/64k/w0", "kind": "ppl", "arm": "8bit", "status": "error",
                     "error": "CUDA_ERROR_OUT_OF_MEMORY", "out_of_memory": True}]
            text, verdicts, rep = self.gated(d, cases, rows)
            self.assertEqual("FAIL", verdicts["ppl (every corpus and L run)"], text)
            self.assertEqual(1, tool.exit_status(rep.verdicts))

    def test_cases_neither_arm_ran_leave_a_passing_gate_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            cases = self.case_file(d, [self.ppl_case("ppl/prose/8k/w0", 8192), self.ppl_case("ppl/prose/96k/w0", 98304)])
            skipped = {"status": "skipped", "reason": "not run: 98304 tokens is above --max-tokens 66000"}
            rows = [self.ppl_record("ppl/prose/8k/w0", "f16"), self.ppl_record("ppl/prose/8k/w0", "8bit"),
                    self.ppl_record("ppl/prose/96k/w0", "f16", **skipped), self.ppl_record("ppl/prose/96k/w0", "8bit", **skipped)]
            text, verdicts, rep = self.gated(d, cases, rows)
            self.assertEqual("INCOMPLETE", verdicts["ppl (every corpus and L run)"], text)
            self.assertEqual(3, tool.exit_status(rep.verdicts))
            self.assertIn("ppl/prose/96k/w0", dict((n, x) for n, _, x in rep.verdicts)["ppl (every corpus and L run)"])
            self.assertEqual(0, tool.exit_status([("x", "PASS", "")]))

    def test_a_layers_case_no_arm_checked_is_a_not_run_gate(self):
        with tempfile.TemporaryDirectory() as d:
            cases = self.case_file(d, [{"type": "case", "id": "layers/prose/32k", "kind": "layers", "corpus": "prose", "tokens": 32768}])
            rows = [{"type": "record", "case": "layers/prose/32k", "kind": "layers", "arm": "f16", "status": "skipped", "reason": "not applicable"},
                    {"type": "record", "case": "layers/prose/32k", "kind": "layers", "arm": "8bit", "status": "error", "error": "out of memory"}]
            text, verdicts, rep = self.gated(d, cases, rows)
            self.assertEqual("NOT RUN", verdicts["layers"], text)
            self.assertIn("not checked: `layers/prose/32k`", text)
            self.assertEqual(3, tool.exit_status(rep.verdicts))

    def greedy_runs(self, d, ref_gaps, ref_generated=(1, 2), stop="eos"):
        cases = self.case_file(d, [{"type": "case", "id": "greedy/code/8k/p0", "kind": "greedy", "corpus": "code", "tokens": 8192,
                                    "end_char": 9, "new_tokens": 4}])
        rows = [{"type": "record", "case": "greedy/code/8k/p0", "kind": "greedy", "arm": "f16", "status": "ok", "input_sha256": "y",
                 "generated": list(ref_generated), "gaps": ref_gaps, "stop": stop},
                {"type": "record", "case": "greedy/code/8k/p0", "kind": "greedy", "arm": "8bit", "status": "ok", "input_sha256": "y",
                 "generated": [1, 2, 3, 4], "gaps": [5.0, 4.0, 0.1, 2.0], "stop": "length"}]
        return self.gated(d, cases, rows)

    def test_a_divergence_where_the_reference_ended_reads_the_end_steps_gap(self):
        name = "greedy (divergences at near-ties only)"
        with tempfile.TemporaryDirectory() as d:
            # The reference chose end of sequence at step 2 with a gap of 3.0; the candidate went on: a confident divergence.
            text, verdicts, _ = self.greedy_runs(d, [5.0, 4.0, 3.0])
            self.assertEqual("FAIL", verdicts[name], text)
            self.assertIn("at 2 (an arm ended): 3.000 / 0.100", text)
        with tempfile.TemporaryDirectory() as d:
            _, verdicts, _ = self.greedy_runs(d, [5.0, 4.0, 0.2])
            self.assertEqual("PASS", verdicts[name])
        with tempfile.TemporaryDirectory() as d:
            # A record without the end step's gap (written before the probe recorded it) cannot show the divergence was a tie.
            text, verdicts, _ = self.greedy_runs(d, [5.0, 4.0])
            self.assertEqual("INCOMPLETE", verdicts[name], text)

    def test_a_kl_is_read_only_against_the_distribution_it_was_computed_from(self):
        with tempfile.TemporaryDirectory() as d:
            cases = self.case_file(d, [self.ppl_case("ppl/prose/8k/w0", 8192)])
            ref = self.ppl_record("ppl/prose/8k/w0", "f16", topk_k=1, topk=TOPK)
            # A candidate whose KL was computed against another run's f16 arm (an arm name alone matches).
            other = self.ppl_record("ppl/prose/8k/w0", "8bit", kl=[0.5] * 512, kl_reference="/elsewhere/run.jsonl:f16",
                                    kl_reference_topk_sha256=tool.sha256_bytes(b"another distribution"))
            _, _, rep = self.gated(d, cases, [ref, other])
            self.assertIsNone(rep.summary["ppl"][0]["kl_mean"])
        with tempfile.TemporaryDirectory() as d:
            cases = self.case_file(d, [self.ppl_case("ppl/prose/8k/w0", 8192)])
            same = self.ppl_record("ppl/prose/8k/w0", "8bit", kl=[0.5] * 512, kl_reference="/elsewhere/run.jsonl:f16",
                                   kl_reference_topk_sha256=TOPK_SHA)
            _, _, rep = self.gated(d, cases, [ref, same])
            self.assertAlmostEqual(0.5, rep.summary["ppl"][0]["kl_mean"])

    def test_estimate_extrapolates_from_measured_points(self):
        with tempfile.TemporaryDirectory() as d:
            cases = self.write(d, "full.jsonl", [{"type": "header"},
                {"type": "case", "id": "ppl/x/64k/w0", "kind": "ppl", "corpus": "prose", "tokens": 65536, "score_tokens": 2048}])
            run = self.write(d, "run.jsonl", [{"type": "run", "load_s": 60},
                {"type": "record", "kind": "ppl", "case": "a", "arm": "x", "status": "ok", "input_tokens": 8192, "score_tokens": 512,
                 "prefill_s": 0.15 * 7.5 + 0.0034 * 7.5 ** 2, "score_s": 0.5},
                {"type": "record", "kind": "continue", "case": "b", "arm": "x", "status": "ok", "input_tokens": 32768, "score_tokens": 64,
                 "prefill_s": 0.15 * 31.9375 + 0.0034 * 31.9375 ** 2, "score_s": 64 * 0.03}])

            class A:
                pass
            args = A()
            args.run, args.cases, args.arm, args.processes = str(run), str(cases), None, 1
            self.assertEqual(0, tool.estimate(args))
            records = [r for r in tool.read_jsonl(run) if r.get("type") == "record"]
            t = tool.arm_timings(records)
            self.assertAlmostEqual(0.15, t["prefill"][0], places=6)
            self.assertAlmostEqual(0.0034, t["prefill"][1], places=8)
            expected = 0.15 * 62 + 0.0034 * 62 ** 2 + (t["score"][0] + t["score"][1] * 64) * 2048
            self.assertAlmostEqual(expected, tool.predict(tool.load_cases(cases)[1]["ppl/x/64k/w0"], t), places=6)


if __name__ == "__main__":
    unittest.main()
