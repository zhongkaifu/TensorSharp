"""Reject incomplete model evidence, wrong token conditioning and weak answers."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import sys
import unittest
import zipfile
from unittest import mock

path = Path(__file__).resolve().parents[1] / "qwen38-strata-compare.py"
spec = importlib.util.spec_from_file_location("qwen38_strata_compare", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StrataComparisonTests(unittest.TestCase):
    def test_process_interrupt_reaps_real_child_and_saves_failure(self):
        launch = runner.subprocess.Popen
        children = []
        def interrupted(*args, **kwargs):
            child = launch(*args, **kwargs)
            children.append(child)
            wait = child.wait
            calls = 0
            def interrupt_once(timeout=None):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt('injected interrupt')
                return wait(timeout=timeout)
            child.wait = interrupt_once
            return child
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.subprocess, 'Popen', side_effect=interrupted), \
                mock.patch.object(runner.cache, 'gpu_sample', return_value={'exit_code': 0, 'csv': 'fixture'}):
            output = Path(directory)/'run'
            with self.assertRaises(KeyboardInterrupt):
                runner.run_process([sys.executable, '-c', 'import time; time.sleep(30)'], directory,
                                   dict(os.environ), output, 10)
            self.assertIsNotNone(children[0].poll())
            result = json.loads((output/'run.json').read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['complete'])
            self.assertIn('KeyboardInterrupt', result['failure'])

    def test_process_launch_failure_has_durable_evidence(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.subprocess, 'Popen', side_effect=OSError('injected launch failure')):
            output = Path(directory)/'run'
            with self.assertRaises(OSError):
                runner.run_process(['missing-executable'], directory, {}, output, 10)
            result = json.loads((output/'run.json').read_text())
            self.assertFalse(result['passed'])
            self.assertIsNone(result['exit_code'])
            self.assertIn('launch failure', result['failure'])

    def test_sampler_failure_is_unknown_telemetry_and_keeps_process_outcome(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.cache, 'gpu_sample', side_effect=RuntimeError('injected sample failure')):
            result = runner.run_process([sys.executable, '-c', 'import time; time.sleep(.05)'], directory,
                                       dict(os.environ), Path(directory)/'run', 10)
            self.assertTrue(result['passed'])  # Process completion, not physical-budget qualification.
            self.assertFalse(result['gpu_telemetry_complete'])
            self.assertIn('sample failure', result['sampling_errors'][0])

    def test_timeout_keeps_incomplete_record_and_reaps_owned_process(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.cache, 'gpu_sample', return_value={'exit_code': 0, 'csv': 'fixture'}):
            output = Path(directory)/'run'
            result = runner.run_process([sys.executable, '-c', 'import time; time.sleep(30)'], directory,
                                       dict(os.environ), output, .05)
            self.assertFalse(result['passed'])
            self.assertFalse(result['complete'])
            self.assertTrue(result['timed_out'])
            self.assertIsNotNone(result['exit_code'])
            self.assertFalse(result['cleanup_errors'])
            self.assertEqual(result, json.loads((output/'run.json').read_text()))

    def archive_dependency(self, directory):
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        (source / "code.cpp").write_bytes(b"original source\n")
        archive = root / "source.zip"
        revision = "3" * 40
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("llama.cpp-" + revision + "/code.cpp", b"original source\n")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"source": str(source), "archive": str(archive), "revision": revision,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}), encoding="utf-8")
        return source, archive, manifest, revision

    def test_archive_dependency_verifies_actual_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, manifest, revision = self.archive_dependency(directory)
            identity = runner.dependency_identity(source, revision, manifest)
            self.assertEqual("archive-payloads", identity["verification"])
            self.assertEqual(1, identity["verified_files"])
            self.assertEqual(revision, identity["revision"])
            self.assertEqual("", identity["status"])

    def test_archive_dependency_rejects_changed_missing_and_extra_source(self):
        for change in ("changed", "missing", "extra"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                source, archive, manifest, revision = self.archive_dependency(directory)
                if change == "changed":
                    (source / "code.cpp").write_bytes(b"modified source\n")
                elif change == "missing":
                    (source / "code.cpp").unlink()
                else:
                    (source / "injected.cpp").write_bytes(b"extra source\n")
                with self.assertRaises(ValueError):
                    runner.dependency_identity(source, revision, manifest)

    def test_archive_dependency_rejects_changed_archive_or_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, manifest, revision = self.archive_dependency(directory)
            with self.assertRaises(ValueError):
                runner.dependency_identity(source, "4" * 40, manifest)
            with archive.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                runner.dependency_identity(source, revision, manifest)

    def test_archive_dependency_rejects_wrong_source_and_zip_root(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, manifest, revision = self.archive_dependency(directory)
            with self.assertRaises(ValueError):
                runner.dependency_identity(source.parent, revision, manifest)
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("llama.cpp-wrong/code.cpp", b"original source\n")
            data = json.loads(manifest.read_text())
            data["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                runner.dependency_identity(source, revision, manifest)

    def test_git_dependency_requires_expected_clean_revision(self):
        with mock.patch.object(runner.cache, "git", side_effect=["3" * 40, ""]):
            self.assertEqual("git", runner.dependency_identity("source", "3" * 40)["verification"])
        for revision, status in (("4" * 40, ""), ("3" * 40, " M code.cpp")):
            with self.subTest(revision=revision, status=status), \
                    mock.patch.object(runner.cache, "git", side_effect=[revision, status]), self.assertRaises(ValueError):
                runner.dependency_identity("source", "3" * 40)

    def test_explicit_worker_count_and_zero_vram_reserve_reach_engine_controls(self):
        with mock.patch.dict(runner.os.environ, {"TS_CPU_MOE_THREADS": "16", "TS_GGML_CPU_THREADS": "32"}, clear=True):
            environment = runner.ts_environment(8192, 8)
        self.assertEqual("8", environment["TS_CPU_MOE_THREADS"])
        self.assertEqual("8", environment["TS_GGML_CPU_THREADS"])
        self.assertEqual(["--resident-budget-gib", "8", "--pool-workers", "8", "--vram-reserve-mib", "0"],
                         runner.strata_resource_arguments(8, 8, 0))
        self.assertEqual(["--resident-budget-gib", "8"], runner.strata_resource_arguments(8))

    def test_strata_baseline_disables_approximation_without_mutating_parent(self):
        parent = {"PATH": "tools", "CUDA_VISIBLE_DEVICES": "0", "CUDA_MODULE_LOADING": "EAGER",
                  "STRATA_ROUTE_TAIL_SKIP": "7", "STRATA_ROUTE_RESIDENT": "0.2",
                  "STRATA_NO_IQ256": "0", "strata_kv_rot": "1", "STRATA_RESIDENT_PIN": "0"}
        with mock.patch.dict(runner.os.environ, parent, clear=True):
            actual = runner.strata_environment()
            self.assertEqual("7", runner.os.environ["STRATA_ROUTE_TAIL_SKIP"])
        self.assertEqual({"PATH": "tools", "CUDA_VISIBLE_DEVICES": "0", "CUDA_MODULE_LOADING": "EAGER",
                          "STRATA_ROUTE_TAIL_SKIP": "0"}, actual)
        self.assertEqual("7", parent["STRATA_ROUTE_TAIL_SKIP"])
        self.assertEqual({"STRATA_ROUTE_TAIL_SKIP": "0"}, runner.strata_environment({}))

    def test_invalid_resident_expert_budget_is_rejected_before_creating_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            for value in ("0", "-1", "nan", "inf"):
                arguments = ["compare", "--output", str(output), "--model", "model.gguf",
                             "--pack", "pack", "--strata", "strata", "--resident-budget-gib", value]
                with self.subTest(budget=value), mock.patch.object(runner.sys, "argv", arguments), \
                        mock.patch.object(runner.sys, "stderr"), self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(2, error.exception.code)
                self.assertFalse(output.exists())

    def test_instruction_format_is_not_repaired_before_scoring(self):
        for case, text in (("math", "42."), ("math", "42!"),
                           ("extraction", "cedar | maple"), ("extraction", "cedar maple"),
                           ("code", "```python\ndef square(x):\n    return x*x\n```")):
            with self.subTest(case=case, text=text):
                self.assertFalse(runner.semantic_check(case, text, True)["passed"])

    def test_tool_arguments_are_complete_json_without_duplicate_keys(self):
        answer = '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}'
        self.assertTrue(runner.semantic_check("tool_json", answer, True)["passed"])
        for text, eos in ((answer, False), (answer.replace('celsius', 'fahrenheit'), True),
                          (answer.replace('"name":', '"name":"wrong","name":'), True),
                          ('```json\n' + answer + '\n```', True),
                          (answer[:-1] + ',"extra":1}', True)):
            with self.subTest(text=text, eos=eos):
                self.assertFalse(runner.semantic_check("tool_json", text, eos)["passed"])

    LOG = "prompt  : 11 22 33\noutput  : 42 99\ndecode  2 tokens in 123.0 ms -> 16.26 tok/s\nprefill  2 tokens in 51.0 ms -> 39.22 tok/s (time to first token 72.0 ms)\n"

    def test_completed_ids_and_different_timing_denominators(self):
        data = runner.parse_strata_log(self.LOG, [11, 22, 33], 64)
        self.assertEqual([42, 99], data["generated_tokens"])
        self.assertEqual(2, data["decode_tokens"])
        self.assertEqual(2, data["prefill_tokens"])
        self.assertEqual(72.0, data["time_to_first_token_ms"])

    def test_missing_duplicate_wrong_conditioning_and_denominators_fail(self):
        for log, prompt in ((self.LOG.replace("output  : 42 99\n", ""), [11, 22, 33]),
                            (self.LOG + "output  : 42 99\n", [11, 22, 33]),
                            (self.LOG, [11, 22, 34]),
                            (self.LOG.replace("decode  2", "decode  1"), [11, 22, 33]),
                            (self.LOG.replace("prefill  2", "prefill  3"), [11, 22, 33])):
            with self.subTest(log=log, prompt=prompt), self.assertRaises(ValueError):
                runner.parse_strata_log(log, prompt, 64)

    def test_eos_stopped_dump_has_known_final_prediction_position(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / "logits.bin"
            # Three prompt positions, two generated IDs: four consumed positions.
            dump.write_bytes(struct.pack("<ii8f", 2, 3 - 1 + 64, 1, 2, 3, 4, 5, 6, 7, 8))
            result = runner.parse_strata_final_logits(dump, 2, 3, 2, 64)
            self.assertEqual([7.0, 8.0], result["final_logits"])
            self.assertEqual(3, result["prediction_position"])
            for raw in (b"\0", struct.pack("<ii2f", 2, 66, 7, 8), struct.pack("<ii8f", 2, 66, 1, 2, 3, 4, 5, 6, float("nan"), 8)):
                dump.write_bytes(raw)
                with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                    runner.parse_strata_final_logits(dump, 2, 3, 2, 64)

    def test_semantics_are_independent_and_require_eos(self):
        for case, answer in (("math", "42"), ("extraction", "cedar, maple"), ("code", "def square(x):\n    return x * x")):
            with self.subTest(case=case):
                self.assertTrue(runner.semantic_check(case, answer, True)["passed"])
                self.assertFalse(runner.semantic_check(case, answer, False)["passed"])
        for case, answer in (("math", "17+25=43"), ("extraction", "cedar, maple, pine"),
                             ("code", "def square(x):\n    return x + x"),
                             ("code", "import os\ndef square(x):\n    return x*x"),
                             ("code", "@dangerous\ndef square(x):\n    return x*x")):
            with self.subTest(case=case, answer=answer):
                self.assertFalse(runner.semantic_check(case, answer, True)["passed"])

    def test_missing_cases_arms_repetitions_and_timeout_cannot_pass(self):
        passed = {"passed": True, "complete": True, "semantic_check": {"passed": True}}
        suite = {"math": {"runs": [{"arms": {"tensor": passed, "strata": passed}}]}}
        self.assertTrue(runner.completed_suite(suite, ["math"], ["tensor", "strata"], 1))
        for actual, cases, arms, repetitions in (({}, ["math"], ["tensor", "strata"], 1),
                                               (suite, ["math", "code"], ["tensor", "strata"], 1),
                                               (suite, ["math"], ["tensor", "strata", "cache"], 1),
                                               (suite, ["math"], ["tensor", "strata"], 2)):
            self.assertFalse(runner.completed_suite(actual, cases, arms, repetitions))
        timed_out = json.loads(json.dumps(suite))
        timed_out["math"]["runs"][0]["arms"]["strata"] = {"passed": False, "failure": "timeout"}
        self.assertFalse(runner.completed_suite(timed_out, ["math"], ["tensor", "strata"], 1))

    def test_supplemental_order_can_run_larger_cache_before_smaller_cache(self):
        self.assertEqual(["tensor-cache9472", "strata", "tensor-cache0", "tensor-cache8192"],
                         runner.engine_order([8192, 9472], 0, 9472))
        self.assertEqual(["strata", "tensor-cache0", "tensor-cache8192", "tensor-cache9472"],
                         runner.engine_order([8192, 9472], 1, 9472))
        self.assertEqual(["tensor-cache0", "tensor-cache8192", "tensor-cache9472", "strata"],
                         runner.engine_order([8192, 9472], 0))
        with self.assertRaises(ValueError):
            runner.engine_order([8192, 9472], 0, 4096)

    def test_strata_can_run_first_with_reused_conditioning_for_full_counterbalance(self):
        self.assertEqual(["strata", "tensor-cache0", "tensor-cache8192", "tensor-cache9472"],
                         runner.engine_order([8192, 9472], 0, strata_first=True))
        self.assertEqual(["tensor-cache0", "tensor-cache8192", "tensor-cache9472", "strata"],
                         runner.engine_order([8192, 9472], 1, strata_first=True))

    def test_long_numeric_answer_requires_all_twenty_correct_ordered_squares(self):
        answer = ", ".join(str(value * value) for value in range(1, 21))
        self.assertTrue(runner.semantic_check("squares", answer, True)["passed"])
        for wrong, complete in ((answer, False), (answer.rsplit(",", 1)[0], True),
                                (answer.replace("400", "401"), True),
                                ("The answer is " + answer, True),
                                (answer + ", 441", True)):
            with self.subTest(answer=wrong, complete=complete):
                self.assertFalse(runner.semantic_check("squares", wrong, complete)["passed"])

    def test_download_provenance_must_cover_exact_paths_sizes_and_publisher_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.gguf"
            checkpoint.write_bytes(b"checkpoint")
            report = Path(directory) / "download.json"
            shard = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                     "sha256": "a" * 64, "actual_sha256": "a" * 64, "verified": True}
            good = {"status": "verified", "shards": [shard]}
            report.write_text(json.dumps(good))
            self.assertEqual("verified", runner.verified_download(report, [checkpoint])["status"])
            for key, value in (("bytes", 1), ("path", str(Path(directory) / "other.gguf")),
                               ("verified", False), ("actual_sha256", "b" * 64), ("sha256", "bogus")):
                wrong = json.loads(json.dumps(good))
                wrong["shards"][0][key] = value
                report.write_text(json.dumps(wrong))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.verified_download(report, [checkpoint])
            report.write_text(json.dumps({"status": "downloading", "shards": [shard]}))
            with self.assertRaises(ValueError):
                runner.verified_download(report, [checkpoint])

    def test_single_row_verifier_requires_observed_zero_drafts_and_complete_windows(self):
        log = "speculation   4 rounds of 2, drafts accepted 0 of 0 (0.000), 1.00 tokens per round\nwindow sizes   T1:4 T2:0 (min draft probability 1.00)\n"
        data = runner.parse_strata_scalar_execution(log, 2, 4)
        self.assertEqual(1, data["observed_target_window"])
        for wrong in (log.replace("T1:4 T2:0", "T1:3 T2:1"), log.replace("0 of 0", "0 of 4"),
                      log.replace("T1:4 T2:0", "T1:4"), log.replace("rounds of 2", "rounds of 4"),
                      log.replace("4 rounds", "3 rounds"), ""):
            with self.subTest(log=wrong), self.assertRaises(ValueError):
                runner.parse_strata_scalar_execution(wrong, 2, 4)


if __name__ == "__main__":
    unittest.main()
