"""Regressions for failures that successful response rows used to conceal."""
import importlib.util
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("parallel_case", Path(__file__).parents[1] / "run-parallel-case.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
eval_spec = importlib.util.spec_from_file_location("parallel_eval", Path(__file__).parents[1] / "parallel-server-eval.py")
evaluator = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluator)
logit_spec = importlib.util.spec_from_file_location("parallel_logits", Path(__file__).parents[1] / "compare-parallel-logits.py")
logits = importlib.util.module_from_spec(logit_spec)
logit_spec.loader.exec_module(logits)


class ShutdownEvidenceTests(unittest.TestCase):
    def test_completed_responses_do_not_hide_native_disposal_abort(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=-signal.SIGABRT)
        failures = runner.shutdown_failures(shutdown,
            "All requests completed\nggml-backend.cpp:426: GGML_ASSERT(backend) failed\n")
        self.assertEqual(2, len(failures))
        self.assertIn("GGML_ASSERT", failures[1])

    def test_native_assertion_is_preserved_even_if_a_parent_masks_the_exit(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=0)
        self.assertTrue(runner.shutdown_failures(shutdown, "GGML_ASSERT(backend) failed"))

    def test_server_must_finish_disposal_but_worker_can_be_terminated(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=-signal.SIGTERM)
        self.assertTrue(runner.shutdown_failures(shutdown, ""))
        self.assertEqual([], runner.shutdown_failures(shutdown, "", allow_sigterm=True))

    def test_timeout_and_unrequested_crash_are_failures(self):
        for shutdown in (
            dict(started=True, requested=True, forced=True, exit_code=-9),
            dict(started=True, requested=False, forced=False, exit_code=-signal.SIGTERM),
        ):
            with self.subTest(shutdown=shutdown):
                self.assertTrue(runner.shutdown_failures(shutdown, "", allow_sigterm=True))

    def test_successful_shutdown_and_unstarted_process_are_distinct_valid_states(self):
        for shutdown in (dict(started=False),
                         dict(started=True, requested=True, forced=False, exit_code=0)):
            self.assertEqual([], runner.shutdown_failures(shutdown, "clean shutdown"))


class CompletedAnswerTests(unittest.TestCase):
    def test_correct_but_truncated_answer_is_not_a_quality_pass(self):
        payload = dict(done=True, done_reason="length", eval_count=1, message={"content": "42"})
        self.assertTrue(evaluator.simple_check("arithmetic", "42", "42"))
        self.assertFalse(evaluator.completed_answer(payload))
        payload["done_reason"] = "stop"
        self.assertTrue(evaluator.completed_answer(payload))

    def test_empty_or_unreported_completion_cannot_pass(self):
        for payload in ({}, dict(done=True, done_reason="stop", eval_count=2, message={"content": ""}),
                        dict(done=True, done_reason="stop", message={"content": "42"})):
            with self.subTest(payload=payload):
                self.assertFalse(evaluator.completed_answer(payload))


class BaselineParityTests(unittest.TestCase):
    def setUp(self):
        self.row = {"case": "arithmetic", "repeat": 0, "content": "42", "simple_check_passed": True,
                    "request": {"messages": [{"role": "user", "content": "17+25?"}], "stream": True},
                    "response": {"done_reason": "stop"}}
        self.baseline = {"status": "completed", "rows": [copy.deepcopy(self.row)]}

    def test_identical_completed_requests_pass(self):
        result = evaluator.compare_baseline([self.row], self.baseline)
        self.assertTrue(result["passed"])
        self.assertEqual(result["identical_content"], 1)

    def test_content_or_finish_drift_fails_even_when_both_answers_are_correct(self):
        for change in ("content", "finish"):
            row = copy.deepcopy(self.row)
            if change == "content":
                row["content"] = "42."
            else:
                row["response"]["done_reason"] = "length"
            self.assertFalse(evaluator.compare_baseline([row], self.baseline)["passed"])

    def test_partial_unfinished_duplicate_or_failed_control_never_passes(self):
        for mutation in ("missing", "extra", "unfinished", "duplicate", "failed"):
            baseline = copy.deepcopy(self.baseline)
            if mutation == "missing":
                baseline["rows"] = []
            elif mutation == "extra":
                baseline["rows"].append({**self.row, "repeat": 1})
            elif mutation == "unfinished":
                baseline["status"] = "running"
            elif mutation == "duplicate":
                baseline["rows"].append(self.row)
            else:
                baseline["rows"][0]["simple_check_passed"] = False
            with self.subTest(mutation=mutation):
                self.assertFalse(evaluator.compare_baseline([self.row], baseline)["passed"])

    def test_changed_prompt_sampling_or_stream_options_do_not_establish_parity(self):
        row = copy.deepcopy(self.row)
        row["request"]["temperature"] = .5
        result = evaluator.compare_baseline([row], self.baseline)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["request_differences"]), 1)


class StreamTimingTests(unittest.TestCase):
    def stream(self, events):
        return io.BytesIO(b"\n".join(json.dumps(event).encode() for event in events))

    def test_queue_and_empty_role_chunks_do_not_count_as_first_token(self):
        events = [{"queue_position": 1, "message": {"role": "assistant", "content": ""}},
                  {"message": {"thinking": "Reasoning"}}, {"message": {"content": "4"}},
                  {"message": {"content": "2"}},
                  {"done": True, "done_reason": "stop", "message": {"content": ""},
                   "eval_count": 2, "eval_duration": 100}]
        evidence = {}
        with patch.object(evaluator.time, "perf_counter", side_effect=[12., 13., 14.]):
            result = evaluator.read_stream(self.stream(events), 10., evidence)
        self.assertEqual(evidence["first_token_seconds"], 2.)
        self.assertEqual(evidence["first_content_seconds"], 3.)
        self.assertEqual(result["message"]["content"], "42")
        self.assertEqual(result["eval_duration"], 100)
        self.assertTrue(evaluator.completed_answer(result))

    def test_truncated_stream_and_error_after_content_cannot_pass(self):
        for tail in ([], [{"error": "CUDA failed"}]):
            with self.assertRaises(ValueError):
                evaluator.read_stream(self.stream([{"message": {"content": "42"}}, *tail]), 0., {})

    def test_duplicate_terminal_event_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "after the terminal"):
            evaluator.read_stream(self.stream([{"done": True}, {"done": True}]), 0., {})


class EvaluatorExecutionTests(unittest.TestCase):
    def execute(self, output, content="42", baseline=None, terminal=True):
        events = [{"message": {"content": content}}]
        if terminal:
            events.append({"done": True, "done_reason": "stop", "eval_count": 2,
                           "eval_duration": 1000000, "prompt_eval_count": 3, "prompt_eval_duration": 2000000})
        stream = io.BytesIO(b"\n".join(json.dumps(event).encode() for event in events))
        argv = ["parallel-server-eval.py", "--output", str(output), "--stream", "--cases", "arithmetic", "--repeats", "1"]
        if baseline:
            argv += ["--baseline", str(baseline)]
        with patch.object(evaluator.sys, "argv", argv), patch.object(evaluator.urllib.request, "urlopen", return_value=stream), contextlib.redirect_stdout(io.StringIO()):
            code = evaluator.main()
        return code, json.loads(output.read_text())

    def test_baseline_drift_propagates_to_exit_status_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory)/"base.json", Path(directory)/"candidate.json"
            code, report = self.execute(baseline)
            self.assertEqual(code, 0)
            self.assertIsNotNone(report["summary"]["median_first_token_seconds"])
            code, report = self.execute(candidate, "42.", baseline)
            self.assertEqual(code, 1)
            self.assertTrue(report["rows"][0]["simple_check_passed"])
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["baseline_comparison"]["passed"])

    def test_failed_stream_is_retained_instead_of_disappearing_from_report(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.execute(Path(directory)/"truncated.json", terminal=False)
            self.assertEqual(code, 1)
            row = report["rows"][0]
            self.assertIn("without a terminal", row["error"])
            self.assertEqual(row["stream_events"], [{"message": {"content": "42"}}])


class LoadTimingTests(unittest.TestCase):
    def test_failed_evidence_write_does_not_orphan_the_model_child(self):
        process = Mock(pid=123)
        with patch.object(runner.subprocess, "Popen", return_value=process), \
                patch.object(Path, "write_text", side_effect=OSError("disk full")), \
                patch.object(runner, "stop") as stop:
            with self.assertRaisesRegex(OSError, "disk full"):
                runner.launch_model(["model"], Path("."), {}, None, Path("."), {})
            stop.assert_called_once_with(process)

    def test_actual_child_identity_is_saved_before_readiness_or_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-c", "import time; time.sleep(30)"]
            report = {"status": "running", "command": command, "started_unix": 1.}
            before = time.time()
            with (root / "process.log").open("w") as log:
                process = runner.launch_model(command, root, None, log, root, report)
                try:
                    saved = json.loads((root / "run.json").read_text())
                    self.assertIsNone(process.poll())
                    self.assertEqual(saved["process_pid"], process.pid)
                    self.assertEqual(saved["status"], "running")
                    self.assertEqual(saved["started_unix"], 1.)
                    self.assertLessEqual(before, saved["process_started_unix"])
                    self.assertLessEqual(saved["process_started_unix"], time.time())
                finally:
                    runner.stop(process)

    def test_only_model_loader_logs_establish_load_duration(self):
        log = "\n".join(["HTTP /health 200 1.5 ms", "Loaded model a architecture=qwen elapsedMs=2000.0",
                         "Loaded model b (architecture=deepseek41, backend=cuda) in 3000.0 ms",
                         "[agent-turn-bench] loaded qwen in 4.0s; dtype=f16"])
        samples = runner.load_timings(log)
        self.assertEqual([row["seconds"] for row in samples], [2., 3., 4.])
        self.assertEqual(samples[-1]["source"], "agent_turn_load_and_kernel_warmup")
        self.assertEqual(runner.load_timings("HTTP /health 200"), [])

    def test_companion_identity_covers_drafter_and_projector_flag_forms(self):
        paths = runner.companion_paths(["--draft-model", "/models/draft.gguf", "--mmproj=/models/vision.gguf"])
        self.assertEqual(paths, {"draft_model": Path("/models/draft.gguf").resolve(), "mmproj": Path("/models/vision.gguf").resolve()})
        self.assertEqual(runner.companion_paths(["--mmproj", "none"]), {})
        self.assertEqual(runner.companion_paths(["--draft-model", "draft.gguf"], Path("/models")),
                         {"draft_model": Path("/models/draft.gguf").resolve()})
        with self.assertRaisesRegex(ValueError, "Missing companion"):
            runner.companion_paths(["--draft-model", "--tp", "2"])


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "bin"
        self.binary.mkdir()
        for name in ("libGgmlOps.so", "TensorSharp.Cli.dll", "ThirdParty.dll", "fr/Resources.dll"):
            path = self.binary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original executable")
        self.report = {"status": "completed", "exit_code": 0,
                       "runtime_file_identity_launch": runner.runtime_file_identity(self.binary)}

    def audit(self):
        runner.check_runtime_file_identity(self.report, self.binary)
        return self.report["runtime_file_identity_check"]

    def test_unchanged_runtime_and_read_only_atime_changes_pass(self):
        path = self.binary / "ThirdParty.dll"
        before = path.stat()
        path.read_bytes()
        result = self.audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.report["status"], "completed")
        files = self.report["runtime_file_identity_launch"]["files"]
        self.assertEqual(set(files), {"libGgmlOps.so", "TensorSharp.Cli.dll", "ThirdParty.dll", "fr/Resources.dll"})
        self.assertNotIn("st_atime_ns", files["ThirdParty.dll"])
        self.assertEqual(files["ThirdParty.dll"]["st_ino"], before.st_ino)

    def test_same_content_rewrite_invalidates_success_even_with_identical_hash(self):
        path = self.binary / "libGgmlOps.so"
        path.write_bytes(path.read_bytes())
        # Make the mutation observable even on filesystems with coarse timestamps.
        old = self.report["runtime_file_identity_launch"]["files"][path.name]
        os.utime(path, ns=(path.stat().st_atime_ns, old["st_mtime_ns"] + 1_000_000_000))
        result = self.audit()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["original_run_status"], "completed")
        self.assertEqual(self.report["status"], "failed")
        self.assertEqual(self.report["exit_code"], 0)
        self.assertEqual(old["sha256"], self.report["runtime_file_identity_after_exit"]["files"][path.name]["sha256"])
        self.assertIn("st_mtime_ns", result["changes"][0]["changed_fields"])

    def test_changed_dependency_hash_and_same_bytes_replacement_are_detected(self):
        dependency = self.binary / "ThirdParty.dll"
        dependency.write_bytes(b"different executable")
        satellite = self.binary / "fr/Resources.dll"
        previous = satellite.stat()
        replacement = self.binary / "replacement.tmp"
        replacement.write_bytes(satellite.read_bytes())
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(satellite)
        result = self.audit()
        changes = {row["path"]: row["changed_fields"] for row in result["changes"]}
        self.assertIn("sha256", changes["ThirdParty.dll"])
        self.assertIn("st_ino", changes["fr/Resources.dll"])
        self.assertEqual(result["status"], "failed")

    def test_removed_or_added_dll_cannot_escape_the_inventory(self):
        (self.binary / "ThirdParty.dll").unlink()
        (self.binary / "NewDependency.dll").write_bytes(b"new")
        result = self.audit()
        self.assertEqual({row["change"] for row in result["changes"]}, {"added", "removed"})
        self.assertEqual(result["status"], "failed")

    def test_original_exception_and_exit_status_survive_integrity_failure(self):
        self.report.update(status="failed", error="TimeoutError('original timeout')", exit_code=37)
        (self.binary / "libGgmlOps.so").unlink()
        result = self.audit()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["original_run_status"], "failed")
        self.assertEqual(self.report["error"], "TimeoutError('original timeout')")
        self.assertEqual(self.report["exit_code"], 37)
        self.assertTrue(self.report["runtime_file_identity_after_exit"]["errors"])

    def test_replacement_during_hashing_is_rejected(self):
        target = self.binary / "libGgmlOps.so"
        original_open = Path.open
        replaced = False
        def opening(path, *args, **kwargs):
            nonlocal replaced
            if path == target and not replaced:
                replaced = True
                temporary = target.with_suffix(".replacement")
                temporary.write_bytes(b"original executable")
                temporary.replace(target)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", opening):
            snapshot = runner.runtime_file_identity(self.binary)
        self.assertIn("Runtime changed while hashing: libGgmlOps.so", snapshot["errors"])
        self.assertIn("capture_mutation", snapshot["files"][target.name])

    def test_launch_refuses_preflight_hash_drift_without_starting_a_child(self):
        report = {"native_sha256": "wrong preflight hash", "managed_sha256": {}}
        with patch.object(runner.subprocess, "Popen") as start:
            with self.assertRaisesRegex(RuntimeError, "differs from preflight hash"):
                runner.launch_model(["unused"], self.root, None, None, self.root, report, self.binary)
            start.assert_not_called()
        self.assertTrue(report["runtime_file_identity_launch"]["errors"])

    def test_real_child_mutation_is_bound_to_launch_and_invalidates_clean_exit(self):
        native = self.binary / "libGgmlOps.so"
        report = {"status": "running", "native_sha256": runner.sha(native), "managed_sha256": {}}
        command = [sys.executable, "-c", "from pathlib import Path; Path('bin/ThirdParty.dll').write_bytes(b'changed by child')"]
        with (self.root / "process.log").open("w") as log:
            process = runner.launch_model(command, self.root, None, log, self.root, report, self.binary)
            self.assertIn("runtime_file_identity_launch", json.loads((self.root / "run.json").read_text()))
            report["exit_code"] = process.wait(timeout=10)
        report["status"] = "completed"
        runner.check_runtime_file_identity(report, self.binary)
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runtime_file_identity_check"]["changes"][0]["path"], "ThirdParty.dll")


class SourceIdentityTests(unittest.TestCase):
    def test_tar_snapshot_identifies_sources_without_claiming_git_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("TensorSharp.GGML.Native/ops.cpp", "TensorSharp.Models/Model.cs",
                         "TensorSharp.Models/bin/generated.cs", "TensorSharp.Models/obj/generated.cs",
                         "TensorSharp.GGML.Native/build/debug.cu", "Directory.Build.props"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            with patch.dict(runner.os.environ, {"TENSORSHARP_SOURCE_REVISION": "declared-base"}), \
                    patch.object(runner.subprocess, "check_output", side_effect=AssertionError("Not a Git checkout")):
                first = runner.source_identity(root)
                self.assertEqual(first["source_kind"], "tar_snapshot")
                self.assertIsNone(first["revision"])
                self.assertEqual(first["declared_base_revision"], "declared-base")
                files = {name for group in first["groups"].values() for name in group["files"]}
                self.assertEqual(files, {"TensorSharp.GGML.Native/ops.cpp", "TensorSharp.Models/Model.cs", "Directory.Build.props"})
                (root / "TensorSharp.Models/bin/generated.cs").write_text("new generated output")
                self.assertEqual(first["groups"], runner.source_identity(root)["groups"])
                (root / "TensorSharp.Models/Model.cs").write_text("changed source")
                self.assertNotEqual(first["groups"]["managed_sources"], runner.source_identity(root)["groups"]["managed_sources"])


class CgroupMemoryTests(unittest.TestCase):
    def test_v1_container_memory_budget_and_cache_are_captured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            memory.mkdir()
            (memory / "memory.usage_in_bytes").write_text("900\n")
            (memory / "memory.limit_in_bytes").write_text("1000\n")
            (memory / "memory.stat").write_text("rss 100\ncache 200\ntotal_rss 300\ntotal_cache 600\n")
            sample = runner.read_cgroup_memory(root)
            self.assertEqual(sample["version"], 1)
            self.assertEqual(sample["limit_bytes"], 1000)
            self.assertEqual(sample["current_bytes"], 900)
            self.assertEqual(sample["anon_bytes"], 300)
            self.assertEqual(sample["file_bytes"], 600)

    def test_v2_unlimited_and_unavailable_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(runner.read_cgroup_memory(root))
            (root / "memory.current").write_text("100\n")
            (root / "memory.max").write_text("max\n")
            (root / "memory.stat").write_text("anon 30\nfile 70\n")
            sample = runner.read_cgroup_memory(root)
            self.assertEqual(sample["version"], 2)
            self.assertIsNone(sample["limit_bytes"])
            self.assertEqual(sample["file_bytes"], 70)


class LogitGateTests(unittest.TestCase):
    def test_ungated_measurement_is_never_reported_as_parity_pass(self):
        self.assertEqual(logits.compare([1., 2.], [3., 0.])["status"], "measured_not_gated")

    def test_argmax_and_error_gates_fail_independently(self):
        for options in ({"require_same_argmax": True}, {"max_absolute_error": .1},
                        {"max_rmse": .1}, {"min_cosine": .999}, {"max_relative_l2": .001}):
            self.assertEqual(logits.compare([1., 2.], [3., 0.], **options)["status"], "failed")

    def test_identical_finite_vectors_pass_declared_gates(self):
        self.assertEqual(logits.compare([1., 2.], [1., 2.], max_rmse=0., min_cosine=.999,
                                       require_same_argmax=True)["status"], "passed")

    def test_undefined_cosine_empty_and_nonfinite_vectors_fail(self):
        self.assertEqual(logits.compare([0., 0.], [0., 0.], min_cosine=.99)["status"], "failed")
        self.assertEqual(logits.compare([0., 0.], [0., 0.], max_relative_l2=0.)["status"], "passed")
        self.assertEqual(logits.compare([0., 0.], [1., 0.], max_relative_l2=.001)["status"], "failed")
        for left, right in (([], []), ([1.], [1., 2.]), ([float("nan")], [1.])):
            with self.assertRaises(ValueError):
                logits.compare(left, right)


if __name__ == "__main__":
    unittest.main()
