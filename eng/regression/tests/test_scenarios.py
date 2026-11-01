"""Model-free negative regressions for runtime adapters and subprocess isolation."""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch
from types import SimpleNamespace

from eng.regression.processes import ManagedProcess, clean_environment, redact, run_process
from eng.regression.scenarios import (
    ScenarioFailure, ScenarioUntested, _Http, availability, backend_evidence, extract_assistant,
    run_scenario, validate_chat, validate_embeddings, validate_logit_metrics,
    validate_sequence_metrics,
    _history_hashes,
    _json_loads, cancellation_evidence, offload_evidence, _attest_offload, _attest_offload_architecture, _hosted_response_model,
)


def completion(text="56"):
    return {"object": "chat.completion", "model": "synthetic.gguf", "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}


def logit_metrics(comparison=False):
    row = {"finite": True}
    if comparison:
        row.update(relative_l2=0.00001, cosine=0.9999999, kl=0.000001, max_abs=0.0001, top1_equal=True)
    return {"run_id": "fresh", "complete": True, "all_finite": True, "qualified": True if comparison else None,
            "cases_done": 1, "cases_total": 1,
            "cases": [{"name": "prefill", "steps": 16, "metrics": [dict(row, step=i) for i in range(16)]}]}


class ScenarioContracts(unittest.TestCase):
    def test_hosted_stem_alias_uses_exact_filename_response_identity(self):
        loaded = "Qwen3.5-0.8B-Q4_K_M.gguf"
        for alias in (loaded, "Qwen3.5-0.8B-Q4_K_M"):
            result = {"assertions": [], "evidence": {}}
            expected = _hosted_response_model(loaded, [{"id": alias}], result)
            self.assertEqual(expected, loaded)
            response = dict(completion(), model=loaded)
            self.assertEqual(validate_chat(response, expected_model=expected), "56")
            for wrong in ("another-model.gguf", "Qwen3.5-0.8B-Q4_K_M-other.gguf"):
                with self.subTest(wrong=wrong), self.assertRaisesRegex(ScenarioFailure, "identity"):
                    validate_chat(dict(response, model=wrong), expected_model=expected)

    def test_unrelated_advertised_model_cannot_alias_verified_loaded_file(self):
        for listing in ([{"id": "other-model"}], [{"id": "synthetic.gguf.backup"}], [{"id": "dir/synthetic.gguf"}], [], [{"id": "synthetic"}, {"id": "other"}]):
            with self.subTest(listing=listing), self.assertRaises(ScenarioFailure):
                _hosted_response_model("synthetic.gguf", listing, {"assertions": [], "evidence": {}})

    def test_cpu_moe_host_requires_active_all_layer_banner(self):
        active = "MoE CPU offload active: routed experts of all layers stay in system RAM and run on the host (auto threads)"
        self.assertTrue(offload_evidence(active)["verified"])
        for text in ("TS_CPU_MOE=1 requested", "MoE CPU offload: all layers", active.replace("all layers", "first 2 layer(s)"),
                     active + "\n[moe-offload] WARNING: --cpu-moe is not implemented for this backend"):
            with self.subTest(text=text):
                self.assertFalse(offload_evidence(text)["verified"])

    def test_cpu_moe_probe_requires_typed_runtime_policy_not_host_banner(self):
        policy = {"cpu_moe_enabled": True, "all_layers": True, "cpu_moe_layers": 0,
                  "explicitly_set": True, "cpu_threads": 0, "configuration_source": "environment"}
        self.assertTrue(offload_evidence("", {"offload_policy": policy}, probe=True)["verified"])
        self.assertFalse(offload_evidence("MoE CPU offload active: routed experts of all layers stay in system RAM and run on the host",
                                          {}, probe=True)["verified"])
        for key, value in (("cpu_moe_enabled", False), ("all_layers", False), ("cpu_moe_layers", 2),
                           ("explicitly_set", False), ("cpu_threads", 4), ("configuration_source", "default"),
                           ("cpu_moe_enabled", 1), ("cpu_moe_layers", False)):
            with self.subTest(key=key, value=value):
                bad = dict(policy, **{key: value})
                self.assertFalse(offload_evidence("", {"offload_policy": bad}, probe=True)["verified"])
        self.assertFalse(offload_evidence("[moe-offload] WARNING: --cpu-moe is not implemented for this model",
                                          {"offload_policy": policy}, probe=True)["verified"])

    def test_cpu_moe_probe_binds_profile_to_actual_moe_architecture(self):
        context = {"model_family": "qwen35", "offload_policy": {"name": "cpu_moe", "requested": True, "supported": True}}
        metrics = {"offload_policy": {"cpu_moe_enabled": True, "all_layers": True, "cpu_moe_layers": 0,
                  "explicitly_set": True, "cpu_threads": 0, "configuration_source": "environment"}}
        for architecture in ("qwen35moe", "qwen3next"):
            _attest_offload(context, {"assertions": [], "evidence": {}}, "probe", "",
                            dict(metrics, model_architecture=architecture), probe=True)
        for architecture in (None, "qwen35", "llama", "qwen4exp"):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(ScenarioFailure, "actual model architecture"):
                _attest_offload(context, {"assertions": [], "evidence": {}}, "probe", "",
                                dict(metrics, model_architecture=architecture), probe=True)

    def test_cpu_moe_cli_config_banner_cannot_qualify_dense_or_missing_architecture(self):
        context = {"model_family": "qwen35", "offload_policy": {"name": "cpu_moe", "requested": True, "supported": True}}
        active = "MoE CPU offload active: routed experts of all layers stay in system RAM and run on the host (auto threads)\n"
        for architecture in ("qwen35moe", "qwen3next"):
            _attest_offload(context, {"assertions": [], "evidence": {}}, "cli", active + "Model: " + architecture + ", Layers=32",
                            cli=True)
        for banner in ("", "Model: qwen35, Layers=32", "Model: qwen4exp, Layers=32", "model_path=qwen35moe.gguf",
                       "Model: qwen35, Layers=32\nModel: qwen35moe, Layers=32"):
            with self.subTest(banner=banner), self.assertRaisesRegex(ScenarioFailure, "actual model architecture"):
                _attest_offload(context, {"assertions": [], "evidence": {}}, "cli", active + banner, cli=True)

    def test_cpu_moe_server_identity_is_bound_to_requested_family(self):
        context = {"model_family": "qwen4exp", "offload_policy": {"name": "cpu_moe", "requested": True, "supported": True}}
        _attest_offload_architecture(context, {"assertions": [], "evidence": {}}, "server-model", ["qwen4exp"], "/api/models architecture")
        for architecture in (None, "qwen35", "qwen35moe", "llama"):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(ScenarioFailure, "actual model architecture"):
                _attest_offload_architecture(context, {"assertions": [], "evidence": {}}, "server-model", [architecture], "/api/models architecture")

    def test_backend_requires_positive_runtime_banner(self):
        self.assertFalse(backend_evidence("requested backend=ggml_cuda", "ggml_cuda")["verified"])
        self.assertTrue(backend_evidence("Backend: GgmlCuda\n", "ggml_cuda")["verified"])

    def test_backend_fallback_is_never_coverage(self):
        self.assertFalse(backend_evidence("Backend: Cpu\n", "ggml_cuda")["verified"])
        self.assertFalse(backend_evidence("Backend: GgmlCuda\nbackend unavailable, falling back to CPU", "ggml_cuda")["verified"])

    def test_per_op_kernel_fallback_preserves_actual_backend_coverage(self):
        message = "[qwen35] fused whole-model prefill NOT engaged (backend Cuda has no whole-model prefill graph); falling back to the per-op prefill loop."
        self.assertTrue(backend_evidence("Backend: Cuda\n" + message, "cuda")["verified"])
        for switch in ("falling back to CPU", "falling back to the ggml_cpu backend", "fallback backend: cpu"):
            with self.subTest(switch=switch):
                self.assertFalse(backend_evidence("Backend: Cuda\n" + switch, "cuda")["verified"])

    def test_prompt_or_thinking_cannot_satisfy_answer(self):
        text = "info: prompt mentions 56\n=== Generated Output ===\n--- Thinking ---\n56\n--- End Thinking ---\nwrong\n08:41:33 info: TensorSharp.Cli[1701] done"
        self.assertEqual(extract_assistant(text), "wrong")
        self.assertEqual(extract_assistant("prompt 56\nno output marker"), "")

    def test_cli_interleaved_completion_log_cannot_replace_or_hide_answer(self):
        prefix = '=== Generated Output ===\n08:41:33 info: TensorSharp.Cli[1501] cli.inference.complete chars=2 preview="56"\n'
        suffix = '\n08:41:33 info: TensorSharp.Cli[1701] tensorsharp-cli completed exitCode=0\n'
        self.assertEqual(extract_assistant(prefix + '56' + suffix), '56')
        self.assertEqual(extract_assistant(prefix + '55' + suffix), '55')
        self.assertEqual(extract_assistant(prefix + suffix), '')

    def test_visible_error_and_info_lines_are_not_discarded_as_logs(self):
        for visible in ("Error: inference failed\n56", "Exception: invalid state\n56", "info: misleading answer\n56"):
            with self.subTest(visible=visible):
                self.assertEqual(extract_assistant("=== Generated Output ===\n" + visible), visible)

    def test_semantic_and_usage_assertions(self):
        self.assertEqual(validate_chat(completion()), "56")
        for value in ("", "The answer is 55", "Error: 56", "56 is wrong, 55 is right"):
            with self.subTest(value=value), self.assertRaises(ScenarioFailure):
                validate_chat(completion(value))
        value = completion()
        value["usage"]["total_tokens"] = 99
        with self.assertRaises(ScenarioFailure):
            validate_chat(value)

    def test_length_capped_answer_does_not_pass(self):
        value = completion()
        value["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ScenarioFailure):
            validate_chat(value)

    def test_completion_model_and_budget_are_validated(self):
        with self.assertRaisesRegex(ScenarioFailure, "identity"):
            validate_chat(completion(), expected_model="another-model")
        with self.assertRaisesRegex(ScenarioFailure, "budget"):
            validate_chat(completion(), max_tokens=0)

    def test_duplicate_json_properties_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _json_loads('{"answer": 1, "answer": 56}')

    def test_cancel_telemetry_requires_matching_scope_link_and_early_stop(self):
        start = {"scope": {"RequestId": "owned"}, "props": {"ChatcmplId": "chatcmpl-owned"}}
        aborted = {"scope": {"RequestId": "owned"}, "event": {"id": 1503, "name": "ChatAborted"},
                   "message": "chat.cancelled tokens=2", "props": {"Tokens": 2}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensorsharp-server-test.jsonl"
            def proof(first, second):
                path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
                return cancellation_evidence(directory, "owned", "chatcmpl-owned", 512)
            self.assertTrue(proof(start, aborted)["verified"])
            other = copy.deepcopy(aborted); other["scope"]["RequestId"] = "unrelated"
            self.assertFalse(proof(start, other)["verified"])
            other = copy.deepcopy(aborted); other["props"]["Tokens"] = 512
            self.assertFalse(proof(start, other)["verified"])
            other_start = copy.deepcopy(start); other_start["props"]["ChatcmplId"] = "wrong-stream"
            self.assertFalse(proof(other_start, aborted)["verified"])

    def test_embedding_shapes_nonfinite_and_insensitivity(self):
        value = {"object": "list", "data": [{"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0, 0.0]}, {"index": 2, "embedding": [0.8, 0.6]},
                    {"index": 3, "embedding": [0.0, 1.0]}]}
        self.assertEqual(validate_embeddings(value, 2)["duplicate_cosine"], 1)
        for vector in ([float("nan"), 0], [0, 0], [1, 0], [0, 1, 0]):
            bad = copy.deepcopy(value)
            bad["data"][3]["embedding"] = vector
            with self.subTest(vector=vector), self.assertRaises(ScenarioFailure):
                validate_embeddings(bad, 2)

    def test_embedding_semantic_ordering_not_just_difference(self):
        value = {"object": "list", "data": [{"index": i, "embedding": vector} for i, vector in enumerate(
            ([1, 0], [1, 0], [0, 1], [0.8, 0.6]))]}
        with self.assertRaisesRegex(ScenarioFailure, "relevance"):
            validate_embeddings(value, 2)

    def test_stale_incomplete_and_nonfinite_logit_metrics(self):
        value = logit_metrics()
        self.assertIsNone(validate_logit_metrics(value, "fresh", comparison=False))
        for key, actual in (("run_id", "stale"), ("complete", False), ("all_finite", False), ("cases_done", 0)):
            bad = copy.deepcopy(value)
            bad[key] = actual
            with self.subTest(key=key), self.assertRaises(ScenarioFailure):
                validate_logit_metrics(bad, "fresh", comparison=False)

    def test_logit_first_divergence_not_summary_only(self):
        value = logit_metrics(True)
        value["cases"][0]["metrics"][7]["relative_l2"] = 0.2
        self.assertEqual(validate_logit_metrics(value, "fresh", comparison=True)["step"], 7)
        value["cases"][0]["metrics"][3]["cosine"] = None
        self.assertEqual(validate_logit_metrics(value, "fresh", comparison=True)["step"], 3)

    def test_missing_reference_rows_rejected(self):
        value = logit_metrics(True)
        value["cases"][0]["metrics"].pop()
        with self.assertRaises(ScenarioFailure):
            validate_logit_metrics(value, "fresh", comparison=True)

    def sequence_metrics(self, order):
        schedule = [{"name": f"{name}-{i}", "reference_case": name} for i, name in enumerate(order)]
        value = logit_metrics(True)
        template = value["cases"][0]
        template["metrics"] = [dict(template["metrics"][0], step=i) for i in range(32)]
        value["cases"] = [dict(copy.deepcopy(template), name=case["name"], reference_case=case["reference_case"],
                               reference_scope="external", reset_kv_cache=True, session_case_index=i, steps=32,
                               prompt_tokens=32 if case["reference_case"] == "short" else 512)
                          for i, case in enumerate(schedule)]
        value["cases_done"] = value["cases_total"] = len(schedule)
        return value, schedule

    def test_length_order_divergence_fails_when_same_length_passes(self):
        stable, stable_schedule = self.sequence_metrics(["short", "short"])
        self.assertIsNone(validate_sequence_metrics(stable, stable_schedule, "fresh"))
        changed, changed_schedule = self.sequence_metrics(["short", "long", "short"])
        changed["cases"][2]["metrics"][0].update(top1_equal=False, relative_l2=0.12, cosine=0.91)
        first = validate_sequence_metrics(changed, changed_schedule, "fresh")
        self.assertEqual((first["case"], first["step"]), ("short-2", 0))

    def test_length_order_requires_explicit_reset_and_fresh_reference(self):
        for key, actual in (("reference_scope", "process"), ("reset_kv_cache", False), ("prompt_tokens", 31)):
            value, schedule = self.sequence_metrics(["long", "short"])
            value["cases"][1][key] = actual
            with self.subTest(key=key), self.assertRaises(ScenarioFailure):
                validate_sequence_metrics(value, schedule, "fresh")

    def test_length_order_history_hash_cannot_silently_change(self):
        value, schedule = self.sequence_metrics(["short", "long", "short"])
        references = {name: {"prompt_tokens": [1] * count, "generated_tokens": [2] * 32}
                      for name, count in (("short", 32), ("long", 512))}
        for actual, expected in zip(value["cases"], schedule):
            actual.update(_history_hashes(references[expected["reference_case"]]))
        self.assertIsNone(validate_sequence_metrics(value, schedule, "fresh", references))
        value["cases"][2]["forced_tokens_sha256"] = "0" * 64
        with self.assertRaisesRegex(ScenarioFailure, "history hash"):
            validate_sequence_metrics(value, schedule, "fresh", references)

    def test_environment_permissions_and_secrets_are_not_inherited(self):
        with patch.dict(os.environ, {"TS_CODE_EXEC": "1", "TS_KNOWLEDGE_DIRECTORY": "private", "TENSORSHARP_BACKEND": "cuda", "HF_TOKEN": "secret", "MAX_CONTEXT": "999999"}):
            env = clean_environment({"TS_CODE_EXEC": "1"}, "artifacts")
            self.assertEqual(env["TS_CODE_EXEC"], "0")
            self.assertEqual(env["MAX_CONTEXT"], "2048")
            for key in ("HF_TOKEN", "TS_KNOWLEDGE_DIRECTORY", "TENSORSHARP_BACKEND"):
                self.assertNotIn(key, env)
            self.assertTrue(env["TENSORSHARP_PREFIX_CACHE_DIR"].endswith("prefix-cache"))

    def test_redaction(self):
        value = redact("Bearer abcd.123 https://user:pass@example.test/x?token=abcd password=private literal_secret", ["literal_secret"])
        for secret in ("abcd.123", "user:pass", "token=abcd", "password=private", "literal_secret"):
            self.assertNotIn(secret, value)


class RealSubprocessRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "synthetic.gguf"
        self.model.write_bytes(b"GGUFfixture")
        self.context = {"repo": self.root, "artifact_dir": self.root / "result", "backend": "cpu",
                        "model_path": self.model, "timeout_seconds": 3}

    def tearDown(self):
        self.temp.cleanup()

    def cpu_moe_profile(self):
        self.context.update(backend="ggml_cuda", model_family="qwen35", model_layout="moe", env={"TS_CPU_MOE": "1"},
                            offload_policy={"name": "cpu_moe", "requested": True, "supported": True})

    def test_cpu_moe_ignored_by_old_cli_cannot_pass(self):
        self.cpu_moe_profile()
        self.context["cli_command"] = [sys.executable, "-c", "print('Backend: GgmlCuda\\n=== Generated Output ===\\n56')"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("runtime CPU MoE", value["reason"])

    def test_cpu_moe_active_cli_preserves_backend_and_output_assertions(self):
        self.cpu_moe_profile()
        output = "Backend: GgmlCuda\nModel: qwen35moe, Layers=32\nMoE CPU offload active: routed experts of all layers stay in system RAM and run on the host (auto threads)\ncli.inference decode complete: tokens=2 ms=1 tokensPerSec=2\ncli.inference finishReason=eos tokens=2\n=== Generated Output ===\n56"
        self.context["cli_command"] = [sys.executable, "-c", "print(" + repr(output) + ")"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(len(value["evidence"]["offload"]), 2)
        self.context["cli_command"] = [sys.executable, "-c", "print(" + repr(output.replace("Backend: GgmlCuda", "Backend: Cpu")) + ")"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("actual backend", value["reason"])

    def test_cpu_moe_valid_model_file_and_banner_do_not_make_dense_cli_pass(self):
        self.cpu_moe_profile()
        output = "Backend: GgmlCuda\nModel: qwen35, Layers=32\nMoE CPU offload active: routed experts of all layers stay in system RAM and run on the host (auto threads)\ncli.inference decode complete: tokens=2\ncli.inference finishReason=eos tokens=2\n=== Generated Output ===\n56"
        self.context["cli_command"] = [sys.executable, "-c", "print(" + repr(output) + ")"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("actual model architecture", value["reason"])

    def test_cpu_moe_ignored_by_old_probe_cannot_pass_either_logit_adapter(self):
        self.cpu_moe_profile()
        script = self.root / "old_probe.py"
        script.write_text("import json,sys\nfrom pathlib import Path\np=Path(sys.argv[2]);p.mkdir(parents=True,exist_ok=True)\n(p/'metrics.json').write_text(json.dumps({}))\nprint('Backend: GgmlCuda')\n")
        self.context["logit_command"] = [sys.executable, str(script)]
        for kind in ("forced_logits", "forced_logits_sequence"):
            with self.subTest(kind=kind), patch("eng.regression.scenarios.shutil.disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                value = run_scenario({"id": kind}, self.context)
            self.assertEqual(value["status"], "fail", value)
            self.assertIn("runtime CPU MoE", value["reason"])

    def fake_cli(self, answer="56", backend="Cpu", finish="eos", exit_code=0):
        output = f"Backend: {backend}\ninfo: cli.inference decode complete: tokens=1 ms=1 tokensPerSec=1000\ninfo: cli.inference finishReason={finish} tokens=1\n=== Generated Output ===\n{answer}\n"
        return [sys.executable, "-c", "import sys; print(" + repr(output) + "); sys.exit(" + str(exit_code) + ")"]

    def test_real_subprocess_validates_and_repeats(self):
        self.context["cli_command"] = self.fake_cli()
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(len(value["processes"]), 2)
        self.assertTrue((self.root / "result/scenario.json").is_file())

    def test_exit_zero_wrong_answer_is_failure(self):
        self.context["cli_command"] = self.fake_cli("55")
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertIn("semantic", value["reason"])

    def test_visible_error_followed_by_correct_integer_cannot_pass(self):
        self.context["cli_command"] = self.fake_cli("Error: inference failed\n56")
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("no inference error text", value["reason"])

    def test_cli_tool_prose_without_formatted_section_fails(self):
        self.context["cli_command"] = self.fake_cli('Function: get_regression_value Arguments: {"key":"fixture"}')
        value = run_scenario({"id": "cli_tools"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertIn("formatted", value["reason"])

    def test_backend_mismatch_is_failure(self):
        self.context["cli_command"] = self.fake_cli(backend="GgmlCuda")
        self.assertEqual(run_scenario({"id": "cli_text"}, self.context)["status"], "fail")

    def test_oom_exit_captures_diagnostic(self):
        self.context["cli_command"] = [sys.executable, "-c", "import sys; print('CUDA out of memory',file=sys.stderr); sys.exit(9)"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertEqual(value["processes"][0]["exit_code"], 9)
        self.assertIn("out of memory", Path(value["processes"][0]["stderr_path"]).read_text())

    def test_timeout_kills_owned_process_and_records_stage(self):
        process = ManagedProcess([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
                                 cwd=self.root, artifact_dir=self.root, stage="timeout")
        self.addCleanup(process.stop)
        deadline = time.monotonic() + 5
        while "started" not in process.stdout.read()[0] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn("started", process.stdout.read()[0], "synthetic child did not start")
        value = process.wait(0.2)
        self.assertTrue(value["timed_out"])
        self.assertIsNotNone(value["exit_code"])
        self.assertIn("started", value["stdout"])

    def test_reject_untrusted_effective_config_before_start(self):
        config = self.root / "config.json"
        config.write_text('{"code-exec": true}')
        self.context.update(cli_command=self.fake_cli(), effective_config_path=config)
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertEqual(value["processes"], [])

    def test_reject_download_object_before_start(self):
        config = self.root / "config.json"
        config.write_text('{"model":{"path":"x","url":"https://example.test/model"}}')
        self.context.update(cli_command=self.fake_cli(), effective_config_path=config)
        self.assertEqual(run_scenario({"id": "cli_text"}, self.context)["status"], "fail")

    def test_download_preflight_does_not_require_existing_weights(self):
        self.context.update(cli_command=self.fake_cli(), model_path=self.root / "missing.gguf")
        self.assertFalse(availability({"id": "cli_text"}, self.context)[0])
        self.context["preparing_model"] = True
        self.assertTrue(availability({"id": "cli_text"}, self.context)[0])

    def test_missing_media_and_unknown_adapter_remain_untested(self):
        self.context["cli_command"] = self.fake_cli()
        self.assertEqual(run_scenario({"id": "cli_image"}, self.context)["status"], "untested")
        self.assertEqual(run_scenario({"id": "distributed"}, self.context)["status"], "untested")

    def test_logit_artifact_disk_guard_stops_before_launch(self):
        self.context["logit_command"] = [sys.executable, "-c", "raise RuntimeError('must not launch')"]
        with patch("eng.regression.scenarios.shutil.disk_usage", return_value=SimpleNamespace(free=1024)):
            result = run_scenario({"id": "forced_logits"}, self.context)
        self.assertEqual(result["status"], "untested")
        self.assertEqual(result["processes"], [])
        self.assertIn("disk", result["reason"])

    def test_nonfinite_benchmark_failure_still_writes_valid_json(self):
        output = "Backend: Cpu\ninference benchmark starting: prefillTokens=128 decodeTokens=0 runs=1\nbenchmark summary: bestPrefillMs=1 bestPrefillTps=NaN bestDecodeMs=0 bestDecodeTps=NaN\n"
        self.context["cli_command"] = [sys.executable, "-c", "print(" + repr(output) + ")"]
        value = run_scenario({"id": "cli_prefill"}, self.context)
        self.assertEqual(value["status"], "fail")
        json.loads((self.root / "result/scenario.json").read_text())

    def run_fake_server(self, scenario, mode="normal"):
        script = self.root / "synthetic_server.py"
        script.write_text(r'''import json, sys, os, time, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
mode = sys.argv[1]
port = int(sys.argv[sys.argv.index('--port') + 1])
model = Path(sys.argv[sys.argv.index('--model') + 1]).name
active = False
sessions = set()
session_cache = set()
session_counter = 0
def telemetry(request_id, event, message, props):
    folder = Path(os.environ['TENSORSHARP_LOG_DIR']); folder.mkdir(parents=True, exist_ok=True)
    record = {'scope':{'RequestId':request_id},'event':event,'message':message,'props':props}
    with (folder/'tensorsharp-server-fixture.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record)+'\n')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/api/models': value = {'loaded':model,'loadedBackend':sys.argv[sys.argv.index('--backend')+1],
                                               'architecture':'qwen35' if mode=='cpu_moe_dense' else 'qwen35moe'}
        elif self.path == '/api/queue/status': value = {'processing':int(active),'pending_requests':0}
        else: value = {'data':[{'id':'unrelated-model' if mode=='advertised_other_model' else Path(model).stem,'embedding_dimensions':2}]}
        self.reply(value)
    def do_POST(self):
        global active, session_counter
        payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))) or b'{}')
        if self.path == '/api/sessions':
            session_counter += 1; sid = 'fixture-session-'+str(session_counter); sessions.add(sid)
            self.reply({'sessionId':sid}); return
        if self.path == '/api/chat':
            sid = payload['sessionId']; prompt = payload['messages'][-1]['content']
            if payload.get('newChat'):
                if mode == 'reset_global_clear': session_cache.clear()
                else: session_cache.discard(sid)
            reused = 256 if sid in session_cache and mode != 'reset_no_private' else 8
            session_cache.add(sid)
            markers = re.findall(r'REGRESSION_[A-Z]+_\d+', '\n'.join(m['content'] for m in payload['messages']))
            if 'My session marker is' in prompt: answer = re.search(r'REGRESSION_[A-Z]+_\d+',prompt)[0]
            elif 'earlier messages' in prompt: answer = markers[-1] if markers else ('REGRESSION_COBALT_7319' if mode=='reset_leaks' else 'NONE')
            else: answer = '56'
            if mode != 'reset_no_prefix_telemetry':
                public = 256 if mode == 'reset_public_only' else 8
                telemetry(self.headers.get('X-TensorSharp-Request-Id'), {'id':1,'name':'KvCacheReusePlan'},
                          'prompt.shared_prefix tokens='+str(public), {'SharedPrefixTokens':public})
            self.sse_headers()
            self.emit({'token':answer})
            self.emit({'done':True,'aborted':False,'truncated':False,'promptTokens':400,'tokenCount':1,'kvReusedTokens':reused,'sessionId':sid})
            return
        if self.path == '/v1/embeddings':
            if payload.get('model') != model: self.error(404,'model','model_not_found'); return
            if payload.get('input') == [] and mode != 'embedding_accepts_empty': self.error(400,'input'); return
            if payload.get('dimensions') == 0: self.error(400,'dimensions'); return
            self.reply({'object':'list','data':[{'index':i,'embedding':v} for i,v in enumerate([[1,0],[1,0],[.8,.6],[0,1]])]}); return
        if mode.startswith('embedding') and mode != 'embedding_allows_generation':
            self.error(400,None,message='This server hosts an embedding model. Use /v1/embeddings or /api/embed.'); return
        if payload.get('stream'):
            if mode == 'normal_stream':
                self.sse_headers()
                self.emit({'id':'chatcmpl-fixture','model':model,'object':'chat.completion.chunk','choices':[{'delta':{'content':'56'},'finish_reason':None}]})
                self.emit({'id':'chatcmpl-fixture','model':model,'object':'chat.completion.chunk','choices':[{'delta':{},'finish_reason':'stop'}],
                           'usage':{'prompt_tokens':12,'completion_tokens':1,'total_tokens':13}})
                self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush(); return
            active = True
            rid = self.headers.get('X-TensorSharp-Request-Id'); stream_id = 'chatcmpl-fixture-cancel'
            telemetry(rid, {'id':1501,'name':'ChatStarted'}, 'request starts', {'ChatcmplId':stream_id})
            self.sse_headers()
            self.emit({'id':stream_id,'model':model,'object':'chat.completion.chunk','choices':[{'delta':{'content':'1'},'finish_reason':None}]})
            time.sleep(.3)
            if mode == 'cancel_aborts':
                telemetry(rid, {'id':1503,'name':'ChatAborted'}, 'chat.cancelled tokens=2', {'Tokens':2,'PromptTokens':12})
            else:
                # Deliberately ignore disconnect and report normal completion of
                # the entire budget: queue recovery must not pass this scenario.
                telemetry(rid, {'id':1502,'name':'ChatCompleted'}, 'chat.complete tokens=512', {'Tokens':512,'FinishReason':'max_tokens'})
            active = False
            return
        self.reply({'object':'chat.completion','model':'another-model.gguf' if mode=='response_other_model' else model,'choices':[{'message':{'role':'assistant','content':'56'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':12,'completion_tokens':1,'total_tokens':13,'prompt_tokens_details':{'cached_tokens':8}}})
    def do_DELETE(self):
        sid = self.path.rsplit('/',1)[-1]
        if sid not in sessions: self.error(404,'session'); return
        sessions.remove(sid); self.reply({'ok':True,'sessionId':sid})
    def error(self, status, param, code=None, message='Invalid fixture input'):
        self.reply({'error':{'type':'invalid_request_error','param':param,'code':code,'message':message}},status)
    def sse_headers(self):
        self.send_response(200); self.send_header('Content-Type','text/event-stream')
        if self.headers.get('X-TensorSharp-Request-Id'): self.send_header('X-TensorSharp-Request-Id',self.headers['X-TensorSharp-Request-Id'])
        self.end_headers()
    def emit(self, value):
        try: self.wfile.write(('data: '+json.dumps(value)+'\n\n').encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
    def reply(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
if mode in {'cpu_moe_active','cpu_moe_dense'}: print('MoE CPU offload active: routed experts of all layers stay in system RAM and run on the host (auto threads)', flush=True)
print('Now listening on: http://127.0.0.1:' + str(port), flush=True)
server.serve_forever()
''', encoding="utf-8")
        self.context["server_command"] = [sys.executable, str(script), mode]
        self.context["timeout_seconds"] = 10
        return run_scenario({"id": scenario}, self.context)

    def test_owned_server_bootstrap_and_real_loopback_replay(self):
        value = self.run_fake_server("server_chat")
        self.assertEqual(value["status"], "pass", value)
        self.assertTrue(value["evidence"]["server_shutdown"]["verified"])
        self.assertEqual(value["evidence"]["model_alias"], {"loaded_filename": "synthetic.gguf", "advertised_id": "synthetic",
                         "request_model": "synthetic.gguf", "expected_response_model": "synthetic.gguf"})
        self.assertTrue(all(r["body"]["model"] == "synthetic.gguf" for r in value["inputs"]["requests"] if r["route"] == "/v1/chat/completions"))

    def test_extensionless_listing_and_filename_sse_are_exactly_bound(self):
        value = self.run_fake_server("server_stream", "normal_stream")
        self.assertEqual(value["status"], "pass", value)

    def test_advertised_or_response_other_model_cannot_pass(self):
        for mode, reason in (("advertised_other_model", "advertised model alias"), ("response_other_model", "identity")):
            with self.subTest(mode=mode):
                value = self.run_fake_server("server_chat", mode)
                self.assertEqual(value["status"], "fail", value)
                self.assertIn(reason, value["reason"])

    def test_agent_dispatch_uses_bounded_host_options_and_owned_loopback(self):
        with patch("eng.regression.agent_regression.run_agents") as run_agents:
            value = self.run_fake_server("server_agents")
        self.assertEqual(value["status"], "pass", value)
        run_agents.assert_called_once()
        self.assertTrue(run_agents.call_args.args[0].base.startswith("http://127.0.0.1:"))
        command = value["processes"][0]["command"]
        self.assertNotIn("--no-multi-agent", command)
        self.assertIn("--no-skills", command)
        self.assertNotIn("--code-exec", command)
        for flag, expected in (("--max-tokens", "256"), ("--agents-max-count", "2"), ("--agents-max-concurrent", "2"),
                               ("--agents-max-depth", "1"), ("--agents-max-rounds", "3"), ("--agents-max-generations", "6"),
                               ("--agents-timeout", "10"), ("--agents-max-result-chars", "2000")):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_cpu_moe_ignored_by_old_server_cannot_pass(self):
        self.cpu_moe_profile()
        value = self.run_fake_server("server_chat")
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("runtime CPU MoE", value["reason"])

    def test_cpu_moe_active_server_is_attested_before_and_after_requests(self):
        self.cpu_moe_profile()
        value = self.run_fake_server("server_chat", "cpu_moe_active")
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual([proof["stage"] for proof in value["evidence"]["offload"]], ["server-startup", "server-final"])

    def test_cpu_moe_valid_model_file_and_banner_do_not_make_dense_server_pass(self):
        self.cpu_moe_profile()
        value = self.run_fake_server("server_chat", "cpu_moe_dense")
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("actual model architecture", value["reason"])

    def test_embedding_negatives_are_executed(self):
        value = self.run_fake_server("server_embeddings", "embedding")
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(len(value["evidence"]["http_errors"]), 4)

    def test_embedding_accepting_malformed_input_fails(self):
        value = self.run_fake_server("server_embeddings", "embedding_accepts_empty")
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("rejects invalid request", value["reason"])

    def test_embedding_serving_generation_fails(self):
        value = self.run_fake_server("server_embeddings", "embedding_allows_generation")
        self.assertEqual(value["status"], "fail", value)

    def test_two_sessions_reset_and_cache_evidence(self):
        value = self.run_fake_server("server_reset_cache", "reset")
        self.assertEqual(value["status"], "pass", value)
        self.assertIn("session B cannot recover A marker", [a["name"] for a in value["assertions"]])
        self.assertEqual(len(value["evidence"]["session_cache_controls"]), 2)
        requests = [r["body"] for r in value["inputs"]["requests"] if r["route"] == "/api/chat"]
        self.assertEqual([r["newChat"] for r in requests], [True, False, True, False, False, False, True, False, True])
        self.assertEqual([len(r["messages"]) for r in requests], [1, 3, 1, 5, 3, 5, 1, 7, 1])

    def test_new_chat_globally_clearing_other_session_cache_cannot_pass(self):
        value = self.run_fake_server("server_reset_cache", "reset_global_clear")
        self.assertEqual(value["status"], "untested", value)
        self.assertIn("A after B starts", value["reason"])
        self.assertTrue(next(a["passed"] for a in value["assertions"] if a["name"] == "session A recalls its marker after B starts"))

    def test_absent_private_cache_positive_control_is_untested(self):
        value = self.run_fake_server("server_reset_cache", "reset_no_private")
        self.assertEqual(value["status"], "untested", value)
        self.assertIn("private-cache retention was not demonstrated", value["reason"])

    def test_warmed_public_prefix_alone_does_not_prove_private_retention(self):
        value = self.run_fake_server("server_reset_cache", "reset_public_only")
        self.assertEqual(value["status"], "untested", value)

    def test_missing_request_scoped_public_boundary_is_untested(self):
        value = self.run_fake_server("server_reset_cache", "reset_no_prefix_telemetry")
        self.assertEqual(value["status"], "untested", value)

    def test_cross_session_marker_leak_fails(self):
        value = self.run_fake_server("server_reset_cache", "reset_leaks")
        self.assertEqual(value["status"], "fail", value)
        self.assertIn("session B", value["reason"])

    def test_ignored_disconnect_normal_completion_is_untested(self):
        value = self.run_fake_server("server_cancel", "cancel_ignores")
        self.assertEqual(value["status"], "untested", value)
        self.assertIn("normal completion", value["reason"])
        self.assertFalse(value["evidence"]["cancellation"]["verified"])

    def test_request_scoped_early_abort_and_recovery_pass(self):
        value = self.run_fake_server("server_cancel", "cancel_aborts")
        self.assertEqual(value["status"], "pass", value)
        self.assertTrue(value["evidence"]["cancellation"]["verified"])


class HttpStreamRegressions(unittest.TestCase):
    def setUp(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if getattr(owner, "redirect", None):
                    self.send_response(302)
                    self.send_header("Location", owner.redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(owner.body.encode())
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.result = {"inputs": {}, "evidence": {}}
        self.http = _Http(self.server.server_port, time.monotonic() + 5, self.result)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def stream_body(self, content="56", done=True, finish="stop"):
        value = {"id": "chatcmpl-fixture", "model": "synthetic.gguf", "object": "chat.completion.chunk", "choices": [{"delta": {"content": content}, "finish_reason": None}]}
        final = {"id": "chatcmpl-fixture", "model": "synthetic.gguf", "object": "chat.completion.chunk", "choices": [{"delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}
        return "data: " + json.dumps(value) + "\n\ndata: " + json.dumps(final) + "\n\n" + ("data: [DONE]\n\n" if done else "")

    def test_valid_stream_reconstructs_answer(self):
        self.body = self.stream_body()
        self.assertEqual(self.http.stream("/v1/chat/completions", {})["content"], "56")

    def test_loopback_redirect_is_refused(self):
        self.redirect = "http://example.invalid/never-request-this"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.http.json("/v1/chat/completions", {})
        self.assertEqual(caught.exception.code, 302)
        self.assertEqual(self.result["evidence"]["http_errors"][0]["status"], 302)

    def test_stream_missing_terminal_does_not_pass(self):
        self.body = self.stream_body(done=False)
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_stream_duplicate_terminal_does_not_pass(self):
        self.body = self.stream_body() + "data: [DONE]\n\n"
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_stream_cannot_switch_completion_identity(self):
        self.body = self.stream_body().replace('"id": "chatcmpl-fixture"', '"id": "another-id"', 1)
        with self.assertRaisesRegex(ScenarioFailure, "ID"):
            self.http.stream("/v1/chat/completions", {})

    def test_stream_model_must_match_request(self):
        self.body = self.stream_body()
        with self.assertRaisesRegex(ScenarioFailure, "model"):
            self.http.stream("/v1/chat/completions", {"model": "another-model"})

    def test_completed_stream_cannot_count_as_cancellation_coverage(self):
        self.body = self.stream_body()
        with patch.object(self.http, "json", return_value={"processing": 0, "pending_requests": 0}):
            with self.assertRaises(ScenarioUntested):
                self.http.stream("/v1/chat/completions", {}, cancel=True)

    def test_stream_error_and_budget_exhaustion(self):
        self.body = self.stream_body(content="Error: bad model")
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})
        self.body = self.stream_body(finish="length")
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_web_ui_replace_is_string(self):
        events = [{"token": "obsolete"}, {"replace": "56"},
                  {"done": True, "aborted": False, "truncated": False, "promptTokens": 12, "tokenCount": 1}]
        self.body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        self.assertEqual(self.http.stream("/api/chat", {}, web_ui=True)["content"], "56")


if __name__ == "__main__":
    unittest.main()
