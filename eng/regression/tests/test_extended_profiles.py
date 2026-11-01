"""Synthetic TRX/client receipts only; no .NET process, model or GPU is launched."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from eng.regression import extended_profiles as profiles
from eng.regression.scenarios import ScenarioFailure, ScenarioUntested

REPO = Path(__file__).resolve().parents[3]
CUDA_PROOF = "[cuda-graph] captured test|shape (3 owned blocks)\n[cuda-graph] replay test|shape (total 1)\n[cuda-profile] CPU fallback ops:\n[cuda-profile] host<->device syncs:\n  sync 4 calls 1.0 ms 0.1 MB\n"


def result():
    return {"assertions": [], "evidence": {}, "processes": [], "inputs": {}}


def healthy(stdout="", stderr=""):
    return {"stage": "fixture", "exit_code": 0, "stdout": stdout, "stderr": stderr, "timed_out": False,
            "cancelled": False, "output_truncated": False, "teardown": {"attempted": True, "verified": True}}


def native_capture(mode="stream", backend="ggml_cpu", model="fixture.gguf"):
    receipts = [{"schema_version": 1, "turn": i, "text": text, "model": model, "architecture": "qwen35",
        "backend": backend, "mode": mode, "finish": "stop", "native_finish": "eos",
        "prompt_tokens": 20 * i, "generated_tokens": 2, "kv_cache_reused_tokens": 0} for i, text in enumerate(("4", "12"), 1)]
    return healthy("Backend: " + backend + "\n4\n12\n", "\n".join(profiles.RECEIPT_PREFIX + json.dumps(row) for row in receipts))


def write_trx(path, outcomes=("Passed", "Passed"), names=profiles.NATIVE_TESTS, output=CUDA_PROOF):
    root = ET.Element("TestRun")
    definitions, results = ET.SubElement(root, "TestDefinitions"), ET.SubElement(root, "Results")
    for i, (name, outcome) in enumerate(zip(names, outcomes)):
        unit = ET.SubElement(definitions, "UnitTest", id=str(i))
        cls, method = name.rsplit(".", 1)
        ET.SubElement(unit, "TestMethod", className=cls, name=method)
        row = ET.SubElement(results, "UnitTestResult", testId=str(i), testName=method, outcome=outcome, duration="00:00:00.001")
        if i == 0:
            ET.SubElement(ET.SubElement(row, "Output"), "StdOut").text = output
    summary = ET.SubElement(root, "ResultSummary")
    ET.SubElement(summary, "Counters", total=str(len(outcomes)), executed=str(sum(o != "NotExecuted" for o in outcomes)),
                  passed=str(outcomes.count("Passed")), failed=str(outcomes.count("Failed")), notExecuted=str(outcomes.count("NotExecuted")))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class ExtendedProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = {}
        for role, name in (("test_assembly", "InferenceWeb.Tests.dll"), ("native_client", "TensorSharp.NativeChat.dll"),
                           ("http_client", "TensorSharp.LocalChat.dll")):
            path = self.root / name
            path.write_bytes(b"synthetic prebuilt identity, never executed")
            for suffix in (".deps.json", ".runtimeconfig.json"):
                path.with_suffix(suffix).write_text("{}", encoding="utf-8")
            self.fixture[role] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.profile = {"id": "cuda-sentinels", "scenario": "native_unit", "preset": "qwen3.5-0.8b", "backend": "cuda",
            "models": {}, "fixtures": {"test_assembly": self.fixture["test_assembly"]}, "parameters": {"source_commit": "a" * 40},
            "expected": {"required_tests": list(profiles.NATIVE_TESTS), "max_cpu_fallback_calls": 0},
            "resources": {"ram_required_gib": 2, "vram_required_gib": 1}}
        self.context = {"scenario_profile": self.profile, "repo": REPO, "artifact_dir": self.root / "artifacts",
                        "backend": "cuda", "env": {"CUDA_VISIBLE_DEVICES": "0"}, "timeout_seconds": 15}
        self.context["artifact_dir"].mkdir()
        self.oracle = profiles.load_oracle(REPO)

    def client(self):
        profile = copy.deepcopy(self.profile)
        profile.update(scenario="client_live", backend="ggml_cpu", fixtures={role: self.fixture[role] for role in ("native_client", "http_client")},
                       expected={"answers": ["4", "12"], "require_normal_finish": True})
        model = self.root / "fixture.gguf"
        model.write_bytes(b"never loaded")
        return dict(self.context, scenario_profile=profile, backend="ggml_cpu", model_path=str(model),
                    context_tokens=2048, server_command=["dotnet", "owned-server.dll"])

    def validate_capture(self, item):
        return profiles.validate_native_client(item, mode="stream", model="fixture.gguf", backend="ggml_cpu", oracle=self.oracle)

    def test_profiles_accept_only_fixed_contracts_and_tolerate_common_provenance(self):
        profiles.validate_profile(self.profile)
        profiles.validate_profile(dict(self.profile, profile_source={"path": "known.json", "sha256": "f" * 64}))
        profiles.validate_profile(self.client()["scenario_profile"])
        for key, value in (("parameters", {"source_commit": "a" * 40, "command": "arbitrary"}),
                           ("expected", {"required_tests": ["Other.Test"], "max_cpu_fallback_calls": 0}),
                           ("expected", {"required_tests": [{}, []], "max_cpu_fallback_calls": 0}),
                           ("expected", {"required_tests": list(profiles.NATIVE_TESTS), "max_cpu_fallback_calls": True}),
                           ("models", {"extra": "weights"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                profiles.validate_profile(dict(self.profile, **{key: value}))

    def test_unsafe_fixture_identity_or_source_declaration_rejected(self):
        for change in ({"parameters": {"source_commit": "latest"}}, {"fixtures": {"test_assembly": {"path": "other.dll", "sha256": "f" * 64}}},
                       {"fixtures": {"test_assembly": {"path": "InferenceWeb.Tests.dll", "sha256": "bad"}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                profiles.validate_profile(dict(self.profile, **change))

    @patch.object(profiles.shutil, "which", return_value="dotnet")
    def test_backend_matrix_missing_deployment_and_context_are_explicit(self, _):
        self.assertTrue(profiles.availability({}, self.context)[0])
        self.assertFalse(profiles.availability({}, dict(self.context, backend="ggml_cuda"))[0])
        client = self.client()
        self.assertTrue(profiles.availability({}, client)[0])
        self.assertFalse(profiles.availability({}, dict(client, context_tokens=1024))[0])
        Path(self.fixture["native_client"]["path"]).with_suffix(".deps.json").unlink()
        self.assertFalse(profiles.availability({}, client)[0])

    def test_trx_exact_names_and_counters(self):
        path = self.root / "test.trx"
        write_trx(path)
        rows, counts, output = profiles.read_native_trx(path)
        self.assertEqual([row["status"] for row in rows], ["pass", "pass"])
        self.assertEqual(counts["executed"], 2)
        self.assertIn("[cuda-graph]", output)
        for names in ((profiles.NATIVE_TESTS[0], "Other.Test"), (profiles.NATIVE_TESTS[0],) * 2, (profiles.NATIVE_TESTS[0],)):
            write_trx(path, outcomes=("Passed",) * len(names), names=names)
            with self.subTest(names=names), self.assertRaises(ScenarioFailure):
                profiles.read_native_trx(path)

    def test_trx_bad_xml_entities_missing_and_contradictory_counters(self):
        path = self.root / "bad.trx"
        for data in ("", "broken", '<!DOCTYPE root [<!ENTITY x "x">]><TestRun/>', "<TestRun/>"):
            path.write_text(data)
            with self.subTest(data=data), self.assertRaises(ScenarioFailure):
                profiles.read_native_trx(path)
        write_trx(path)
        path.write_text(path.read_text().replace('passed="2"', 'passed="0"'))
        with self.assertRaises(ScenarioFailure): profiles.read_native_trx(path)

    def test_positive_device_and_complete_zero_fallback_evidence_required(self):
        self.assertEqual(profiles.cuda_device_proof(CUDA_PROOF)["fallback_calls"], [])
        for text in ("", CUDA_PROOF.replace("captured test|shape", "captured unrelated"),
                     CUDA_PROOF.split("[cuda-profile]")[0], CUDA_PROOF.replace("replay test|shape", "replay unrelated")):
            with self.subTest(text=text), self.assertRaises(ScenarioUntested): profiles.cuda_device_proof(text)
        for text in (CUDA_PROOF.replace("[cuda-profile] host", " add 1 calls 0.0 ms\n[cuda-profile] host"),
                     CUDA_PROOF + "WARNING: element-by-element CPU fallback", CUDA_PROOF.replace("[cuda-profile] host", " unexpected\n[cuda-profile] host")):
            with self.subTest(text=text), self.assertRaises(ScenarioFailure): profiles.cuda_device_proof(text)

    def test_assembly_identity_change_fails_before_any_execution(self):
        Path(self.fixture["test_assembly"]["path"]).write_bytes(b"changed")
        with patch.object(profiles, "run_process") as process, self.assertRaises(ScenarioFailure):
            profiles._deployment(self.context, result())
        process.assert_not_called()

    @patch.object(profiles.shutil, "which", return_value="dotnet")
    def test_native_fixed_filter_no_model_no_build_and_individual_outcomes(self, _):
        def execute(argv, **kwargs):
            self.assertEqual(argv[:2], ["dotnet", "vstest"])
            self.assertFalse(any(arg in argv for arg in ("build", "restore", "--model")))
            self.assertIn("Requires!=Models", next(arg for arg in argv if arg.startswith("--TestCaseFilter:")))
            self.assertEqual(kwargs["env"]["TS_CUDA_PROFILE"], "1")
            root = Path(next(arg.split(":", 1)[1] for arg in argv if arg.startswith("--ResultsDirectory:")))
            write_trx(root / "sentinels.trx")
            return healthy()
        output = result()
        with patch.object(profiles, "run_process", side_effect=execute):
            profiles.run({}, self.context, output)
        self.assertEqual(len(output["evidence"]["native_tests"]["tests"]), 2)
        self.assertTrue(all(row["passed"] for row in output["assertions"]))
        self.assertIn("runtime_candidates", output["evidence"]["deployment"])
        self.assertEqual(output["evidence"]["deployment"]["declared_build_source_commit"], "a" * 40)

    def test_native_skipped_tests_are_untested_and_failures_retained(self):
        for outcomes, error in ((("NotExecuted", "NotExecuted"), ScenarioUntested), (("Passed", "Failed"), ScenarioFailure)):
            artifact = self.root / outcomes[0]
            artifact.mkdir()
            ctx = dict(self.context, artifact_dir=artifact)
            def execute(argv, **kwargs):
                write_trx(artifact / "native-sentinel-results/sentinels.trx", outcomes=outcomes)
                return healthy()
            output = result()
            with patch.object(profiles, "run_process", side_effect=execute), self.assertRaises(error):
                profiles._native_unit(ctx, output, {"test_assembly": ["dotnet", "InferenceWeb.Tests.dll"]})
            self.assertEqual(len(output["evidence"]["native_tests"]["tests"]), 2)

    def test_stale_results_directory_is_never_reused(self):
        (self.context["artifact_dir"] / "native-sentinel-results").mkdir()
        with patch.object(profiles, "run_process") as process, self.assertRaises(FileExistsError):
            profiles._native_unit(self.context, result(), {"test_assembly": ["dotnet", "InferenceWeb.Tests.dll"]})
        process.assert_not_called()

    def test_receipts_assert_full_text_not_numeric_lines_or_log_previews(self):
        item = native_capture()
        self.assertEqual(len(self.validate_capture(item)), 2)
        item["stdout"] += "Preview: 4 then 12\n"
        item["stderr"] = item["stderr"].replace('"text": "4"', '"text": "5, not 4"')
        with self.assertRaises(ScenarioFailure): self.validate_capture(item)
        with self.assertRaises(ScenarioUntested): self.validate_capture(healthy("Backend: ggml_cpu\n4\n12\n"))

    def test_receipt_identity_finish_usage_and_turn_order_faults(self):
        for old, new in (('"backend": "ggml_cpu"', '"backend": "cpu"'), ('"model": "fixture.gguf"', '"model": "other.gguf"'),
                         ('"finish": "stop"', '"finish": "length"'), ('"native_finish": "eos"', '"native_finish": "aborted"'),
                         ('"generated_tokens": 2', '"generated_tokens": 65'), ('"prompt_tokens": 20', '"prompt_tokens": 0'),
                         ('"kv_cache_reused_tokens": 0', '"kv_cache_reused_tokens": 999'), ('"turn": 1', '"turn": 2'),
                         ('"mode": "stream"', '"mode": "response"'), ('"architecture": "qwen35"', '"architecture": ""')):
            item = native_capture(); item["stderr"] = item["stderr"].replace(old, new)
            with self.subTest(old=old), self.assertRaises(ScenarioFailure): self.validate_capture(item)
        item = native_capture(); item["stderr"] += "\n" + item["stderr"]
        with self.assertRaises(ScenarioFailure): self.validate_capture(item)

    def test_actual_device_fallback_fails_even_with_good_receipts(self):
        item = native_capture(); item["stdout"] += "falling back to CPU\n"
        with self.assertRaises(ScenarioFailure): self.validate_capture(item)
        item = native_capture(); item["stdout"] += "graph unavailable; falling back to per-op kernels\n"
        self.validate_capture(item)

    def test_process_timeouts_truncation_and_unverified_teardown_fail(self):
        for change in ({"exit_code": 1}, {"timed_out": True}, {"cancelled": True}, {"output_truncated": True}, {"teardown": {}},
                       {"teardown": {"attempted": True, "verified": False}}):
            item = healthy(); item.update(change); output = result()
            with self.subTest(change=change), self.assertRaises(ScenarioFailure): profiles._healthy(item, output, "fixture")
            self.assertEqual(len(output["processes"]), 1)

    @patch.object(profiles.shutil, "which", return_value="dotnet")
    def test_native_and_http_parity_use_serial_fixed_commands_and_owned_server(self, _):
        ctx = self.client(); seen = []
        def execute(argv, **kwargs):
            seen.append(argv)
            self.assertIn("--regression-json", argv)
            self.assertEqual(argv[argv.index("--model") + 1], ctx["model_path"])
            return native_capture(mode=argv[argv.index("--mode") + 1])
        def server(case, context, output):
            self.assertEqual(len(seen), 2)
            self.assertEqual(case, {"id": "client_http_live"})
            self.assertEqual(context["http_client_command"][-1], self.fixture["http_client"]["path"])
            output["evidence"]["http_client"] = {"invocations": [{"passed": True, "answers": ["4", "12"]} for _ in range(4)]}
        output = result()
        with patch.object(profiles, "run_process", side_effect=execute), patch.object(profiles, "_server", side_effect=server):
            profiles.run({}, ctx, output)
        self.assertEqual(output["evidence"]["native_client"]["counts"]["native_turns"], 4)
        self.assertTrue(all(row["passed"] for row in output["assertions"]))

    def test_missing_http_parity_cannot_pass_after_native_answers(self):
        ctx = self.client()
        def execute(argv, **kwargs): return native_capture(mode=argv[argv.index("--mode") + 1])
        with patch.object(profiles, "run_process", side_effect=execute), patch.object(profiles, "_server"), self.assertRaises(ScenarioFailure):
            profiles._client_live(ctx, result(), {role: ["dotnet", fixture["path"]] for role, fixture in ctx["scenario_profile"]["fixtures"].items()})


if __name__ == "__main__":
    unittest.main()
