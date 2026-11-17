"""Synthetic-only scaling profile contracts; no binaries, models or network run."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from eng.regression import scaling_profiles as scaling
from eng.regression.scenarios import ScenarioFailure, ScenarioUntested


def rows():
    plain = dict(Scenario="spec", Label="plain greedy", Finish="eos", Note="", Prompt=128,
                 Reused=0, Fresh=128, OutTokens=3, Tokens=[7, 8, 9], PrefillTps=12.0,
                 DecodeTps=5.0, TtftMs=10.0, Drafted=0, Accepted=0, VerifySteps=0, PlainSteps=3, Rollbacks=0)
    return [plain, dict(plain, Label="draft head (auto)", Drafted=3, Accepted=2, VerifySteps=1, PlainSteps=1)]


def result():
    return {"assertions": [], "inputs": {}, "evidence": {}, "processes": []}


class LoraHttpFixture:
    """Protocol-shaped responses; the production chat validator stays active."""
    def __init__(self, evidence, texts=("blue", "orange", "blue", "blue"), *, removed=True, retain=False):
        self.evidence, self.texts = evidence, iter(texts)
        self.removed, self.retain, self.loaded = removed, retain, False
        self.calls = []
        self.response_mutator = lambda response: response

    def json(self, route, body=None, method=None):
        self.calls.append((route, body, method))
        if (body or {}).get("scale") == 1e60 or ((body or {}).get("lora") and not self.loaded):
            param = "scale" if "scale" in body else "lora"
            self.evidence["evidence"].setdefault("http_errors", []).append({
                "body": json.dumps({"error": {"type": "invalid_request_error", "param": param}})})
            raise urllib.error.HTTPError("http://fixture", 400, "invalid", {}, None)
        if method == "DELETE":
            self.loaded = self.retain
            return {"removed": self.removed}
        if route == "/v1/loras" and body:
            self.loaded = True
            return {"id": "fixture-lora", "modules": 1}
        if route == "/v1/loras":
            return {"data": [{"id": "fixture-lora", "applicable": True}] if self.loaded else []}
        if route != "/v1/chat/completions":
            raise AssertionError("unexpected fixture route: " + route)
        return self.response_mutator({
            "object": "chat.completion", "model": "model.gguf",
            "choices": [{"message": {"role": "assistant", "content": next(self.texts)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
        })


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[3]
        self.model = self.root / "synthetic.gguf"
        self.model.write_bytes(b"never loaded")
        self.benchmark = self.root / "AgentTurnBench.exe"
        self.server = self.root / "TensorSharp.Server.Host.exe"
        self.probe = self.root / "DeepSeek41DistributedProbe.exe"
        for path in (self.benchmark, self.server, self.probe): path.write_bytes(b"never executed")
        self.tokens = self.root / "tokens.json"
        self.tokens.write_text(json.dumps([4, 5, 6, 7, 8, 9]))
        self.adapter = self.root / "adapter.safetensors"
        self.adapter.write_bytes(b"synthetic adapter marker, never parsed")
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("Return the fixture color.")
        self.context = dict(repo=self.repo, artifact_dir=self.root / "output", backend="ggml_cuda", context_tokens=2048,
                            model_path=str(self.model), timeout_seconds=30,
                            benchmark_command=[str(self.benchmark)], server_command=[str(self.server)],
                            hardware={"gpus": [{"index": 0}, {"index": 1}]},
                            profile_models={"primary": {"model_path": str(self.model), "preset": {"id": "synthetic"},
                                "options": {}, "identity": [{"path": str(self.model), "sha256": scaling._hash(self.model)}]}})

    def fixture(self, path):
        return {"path": str(path), "sha256": scaling._hash(path)}

    def profile(self, kind):
        value = dict(id="fixture-profile", scenario=kind, preset="synthetic", backend="ggml_cuda", models={}, fixtures={},
                     resources={"ram_required_gib": 8, "vram_required_gib": 8}, parameters={}, expected={})
        if kind == "speculation":
            value.update(parameters={"head_kind": "pertoken", "draft_tokens": 4},
                         expected={"require_token_parity": True, "require_acceptance": True})
        elif kind == "multi_gpu":
            value.update(parameters={"mode": "tp", "devices": [0, 1], "chunk": 2, "decode_count": 2},
                         fixtures={"probe": self.fixture(self.probe), "tokens": self.fixture(self.tokens)},
                         expected=dict(max_absolute_error=0, max_rmse=0, max_relative_l2=0, min_cosine=1, require_same_argmax=True))
        elif kind == "lora":
            value.update(parameters={"adapter_name": "fixture-lora", "scale": 1, "max_tokens": 16},
                         fixtures={"adapter": self.fixture(self.adapter), "prompt": self.fixture(self.prompt)},
                         expected={"base_text": "blue", "adapted_text": "orange"})
        return value

    def receipt(self, command, **changes):
        value = dict(command=command, stdout="Backend: GgmlCuda\n[agent-turn-bench] loaded qwen4exp in1s; context=2048 drafter=pertoken\n",
                     stderr="", exit_code=0, timed_out=False, cancelled=False, output_truncated=False,
                     teardown={"attempted": True, "verified": True})
        value.update(changes)
        return value

    def invoke_learned(self, data=None, changes=None):
        profile = self.profile("speculation")
        self.context["scenario_profile"] = profile
        evidence = result()
        calls = []
        def execute(command, **kwargs):
            calls.append((command, kwargs))
            Path(command[command.index("--out") + 1]).write_text(json.dumps(rows() if data is None else data))
            return self.receipt(command, **(changes or {}))
        with patch.object(scaling, "run_process", side_effect=execute):
            try:
                scaling.run({"id": "speculation"}, self.context, evidence)
                error = None
            except (ScenarioFailure, ScenarioUntested, ValueError) as caught:
                error = caught
        return evidence, calls, error

    def test_typed_profiles_reject_unknowns_weak_gates_and_duplicate_devices(self):
        for kind in ("multi_gpu", "speculation", "lora"):
            scaling.validate_profile(self.profile(kind))
            value = self.profile(kind)
            value["parameters"]["shell"] = "arbitrary command"
            with self.assertRaises(ValueError): scaling.validate_profile(value)
        for key, bad in (("max_rmse", 2), ("max_relative_l2", .002), ("min_cosine", .99),
                         ("max_absolute_error", float("nan")), ("require_same_argmax", False)):
            value = self.profile("multi_gpu")
            value["expected"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): scaling.validate_profile(value)
        for devices in ([0], [0, 0], [0, True], [-1, 0]):
            value = self.profile("multi_gpu")
            value["parameters"]["devices"] = devices
            with self.assertRaises(ValueError): scaling.validate_profile(value)

    def test_distributed_machine_pins_gate_public_plan_and_execution_entrypoints(self):
        profile = self.profile("multi_gpu")
        profile["scenario"] = "distributed"
        profile["fixtures"]["known_hosts"] = self.fixture(self.tokens)
        profile["expected"]["weights_id"] = "synthetic-pinned-checkpoint"
        profile["resources"]["per_gpu_required_gib"] = {"0": 8}
        source_names = ("TensorSharp.Core/Fixture.cs", "TensorSharp.Models/ModelBase.cs",
                        "TensorSharp.Distributed/TcpCommunicator.cs",
                        "eng/validation/DeepSeek41DistributedProbe/Program.cs",
                        "eng/validation/DeepSeek41DistributedProbe/DeepSeek41DistributedProbe.csproj")
        nodes = []
        for index in range(2):
            nodes.append(dict(host=f"node{index}.invalid", user="fixture", port=22, authorized=True,
                              expected_machine_id_sha256=hashlib.sha256(f"machine-{index}\n".encode()).hexdigest(),
                              root=f"/node{index}/artifacts", model_path=f"/node{index}/model.gguf",
                              probe_path=f"/node{index}/bin/DeepSeek41DistributedProbe.dll", dotnet="/opt/dotnet/dotnet",
                              model_files=[{"path": f"/node{index}/model.gguf", "sha256": "a" * 64}],
                              runtime_files=[{"path": f"/node{index}/bin/DeepSeek41DistributedProbe.dll", "sha256": "b" * 64},
                                             {"path": "/opt/dotnet/dotnet", "sha256": "c" * 64}],
                              source_files=[{"path": f"/node{index}/repo/" + name, "sha256": "d" * 64}
                                            for name in source_names]))
        profile["parameters"] = dict(mode="tp", local_degree=1, devices=[0], nodes=nodes,
                                     peer_endpoints=["node0.invalid:19000", "node1.invalid:19001"],
                                     chunk=2, decode_count=2, timeout_seconds=30)
        scaling.validate_profile(profile)
        for change in (lambda p: p["parameters"]["nodes"][0].pop("expected_machine_id_sha256"),
                       lambda p: p["parameters"]["nodes"][1].update(
                           expected_machine_id_sha256=p["parameters"]["nodes"][0]["expected_machine_id_sha256"]),
                       lambda p: p["parameters"]["nodes"][0].update(expected_machine_id_sha256="invalid")):
            invalid = copy.deepcopy(profile)
            change(invalid)
            with self.subTest(nodes=invalid["parameters"]["nodes"]):
                with self.assertRaises(ValueError): scaling.validate_profile(invalid)
                self.context["scenario_profile"] = invalid
                with patch.object(scaling, "run_process") as execute, patch.object(scaling, "_probe") as probe:
                    self.assertFalse(scaling.availability({"id": "distributed"}, self.context)[0])
                    with self.assertRaises(ScenarioUntested):
                        scaling.run({"id": "distributed"}, self.context, result())
                    execute.assert_not_called()
                    probe.assert_not_called()

    def test_availability_never_launches_with_missing_profile_hardware_or_model(self):
        with patch.object(scaling, "run_process") as execute:
            self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])
            self.context["scenario_profile"] = self.profile("multi_gpu")
            self.context["hardware"] = {"gpus": [{"index": 0}]}
            self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])
            self.context["hardware"]["gpus"].append({"index": 1})
            self.context["profile_models"]["primary"]["identity"] = []
            self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])
            self.context["preparing_model"] = True
            self.assertTrue(scaling.availability({"id": "multi_gpu"}, self.context)[0])
            execute.assert_not_called()

    def test_preparation_checks_fixture_bounds_without_model_loading(self):
        self.context.update(scenario_profile=self.profile("multi_gpu"), preparing_model=True)
        self.tokens.write_text("[1,2]")
        self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])
        self.tokens.write_text("[1,2,3,4,NaN]")
        self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])
        self.tokens.write_text("[1,2,3,4]")
        self.context["context_tokens"] = 256
        self.assertFalse(scaling.availability({"id": "multi_gpu"}, self.context)[0])

    def test_profile_provenance_is_allowed_but_command_flags_are_not(self):
        profile = self.profile("speculation")
        profile["profile_source"] = {"path": "manifest.json", "sha256": "f" * 64}
        scaling.validate_profile(profile)
        self.context["scenario_profile"] = profile
        self.context["benchmark_command"].append("--draft-model")
        self.assertFalse(scaling.availability({"id": "speculation"}, self.context)[0])

    def test_learned_positive_requires_exact_rows_and_observed_head(self):
        self.context["env"] = {"TS_SPEC_DRAFT_MODEL": "must-not-load", "TS_SPEC_DRAFT": "999"}
        evidence, calls, error = self.invoke_learned()
        self.assertIsNone(error)
        command, kwargs = calls[0]
        self.assertEqual(command[command.index("--spec-candidates") + 1], "auto")
        self.assertEqual(command[command.index("--spec-new") + 1], "64")
        self.assertEqual(command[command.index("--spec-file") + 1], "128")
        self.assertNotIn("--draft-model", command)
        self.assertNotIn("TS_SPEC_DRAFT_MODEL", kwargs["env"])
        self.assertEqual(kwargs["env"]["TS_SPEC_DRAFT"], "4")
        self.assertEqual(evidence["evidence"]["learned_parity"]["status"], "passed")

    def test_learned_inactive_is_untested_and_ngram_cannot_substitute(self):
        data = rows()
        data[1].update(Drafted=0, Accepted=0, VerifySteps=0)
        _, _, error = self.invoke_learned(data)
        self.assertIsInstance(error, ScenarioUntested)
        data = rows()
        data[1]["Label"] = "ngram"
        _, _, error = self.invoke_learned(data)
        self.assertIsInstance(error, ScenarioFailure)
        _, _, error = self.invoke_learned(changes={"stdout": "Backend: GgmlCuda\ndrafter=none\n"})
        self.assertIsInstance(error, ScenarioFailure)

    def test_learned_divergence_is_recorded_even_if_benchmark_exits_nonzero(self):
        data = rows()
        data[1]["Tokens"] = [7, 88, 9]
        evidence, _, error = self.invoke_learned(data, {"exit_code": 1})
        self.assertIsInstance(error, ScenarioFailure)
        self.assertEqual(evidence["evidence"]["learned_parity"]["first_divergence"], 1)

    def test_learned_rejects_nonfinite_inconsistent_counters_and_dirty_baseline(self):
        for key, bad in (("TtftMs", float("nan")), ("PrefillTps", 0), ("Accepted", 100), ("Rollbacks", 2), ("Drafted", True)):
            data = rows()
            data[1][key] = bad
            with self.subTest(key=key):
                self.assertIsNotNone(self.invoke_learned(data)[2])
        data = rows()
        data[0].update(Drafted=1)
        self.assertIsInstance(self.invoke_learned(data)[2], ScenarioFailure)

    def test_timeout_oom_cleanup_and_backend_fallback_fail(self):
        for changes in ({"timed_out": True}, {"teardown": {"attempted": True, "verified": None}},
                        {"stderr": "CUDA out of memory"}, {"output_truncated": True},
                        {"stdout": "Backend: GgmlCpu\n[agent-turn-bench] loaded x context=2048 drafter=pertoken"}):
            with self.subTest(changes=changes):
                self.assertIsInstance(self.invoke_learned(changes=changes)[2], ScenarioFailure)

    def test_fixture_mutation_blocks_all_processes(self):
        self.context["scenario_profile"] = self.profile("multi_gpu")
        self.tokens.write_text("[10,11,12,13,14,15]")
        with patch.object(scaling, "run_process") as execute, self.assertRaises(ScenarioFailure):
            scaling.run({"id": "multi_gpu"}, self.context, result())
        execute.assert_not_called()

    def test_multi_gpu_exact_operation_oracle_and_device_placement(self):
        profile = self.profile("multi_gpu")
        self.context["scenario_profile"] = profile
        evidence = result()
        calls = []
        def execute(command, **kwargs):
            calls.append((command, kwargs))
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            tokens = json.loads(self.tokens.read_text())
            expected = [("reset", []), ("prefill", tokens[:4]), ("decode", [8]), ("decode", [9]),
                        ("reset", []), ("chunk", [4,5]), ("chunk", [6,7]), ("refill", [8]),
                        ("refill", [9]), ("reset", []), ("reset-prefill", tokens[:4])]
            steps = []
            for i, (operation, values) in enumerate(expected):
                entry = {"operation": operation, "tokens": values, "logits": None}
                if operation != "reset":
                    name = f"step-{i}.f32"
                    (folder / name).write_bytes(struct.pack("<16f", *range(1, 17)))
                    entry.update(logits=name, count=16, seconds=.01)
                steps.append(entry)
            degree = 2 if "--tp" in command else 1
            report = dict(schema_version=1, status="completed", disposed=True, steps=steps, vocab_size=16,
                          tokens=tokens, tokens_sha256=scaling._hash(self.tokens), weights_id=command[command.index("--weights-id")+1],
                          topology=dict(mode="tensor_parallel", local_degree=degree, global_degree=degree, rank_offset=0, node_count=1))
            (folder / "probe.json").write_text(json.dumps(report))
            text = "Backend: GgmlCuda\nKV cache: fixture (dtype: f16)\n"
            text += "TP weight sharding complete (2 GPUs)\n" if degree == 2 else "ggml_cuda_init: found 1 CUDA devices\n"
            return self.receipt(command, stdout=text)
        with patch.object(scaling, "run_process", side_effect=execute):
            scaling.run({"id": "multi_gpu"}, self.context, evidence)
        self.assertEqual([call[1]["env"]["CUDA_VISIBLE_DEVICES"] for call in calls], ["0", "0,1"])
        self.assertEqual(evidence["evidence"]["numerical_comparison"]["status"], "passed")
        self.assertEqual(len(evidence["evidence"]["numerical_comparison"]["vectors"]), 8)

    def test_probe_cannot_omit_decode_reset_or_refill(self):
        profile = self.profile("multi_gpu")
        report = {"steps": [{"operation": "prefill", "tokens": [4,5,6,7]}]}
        with self.assertRaises(ScenarioFailure): scaling._probe_report(report, profile, result(), "probe", "id")

    def test_numerical_comparator_rejects_nonfinite_and_argmax_divergence(self):
        profile = self.profile("multi_gpu")
        first, second = self.root / "reference", self.root / "candidate"
        first.mkdir(); second.mkdir()
        base = {"schema_version": 1, "status": "completed", "disposed": True, "weights_id": "same",
                "tokens": [1], "tokens_sha256": "f" * 64, "vocab_size": 3,
                "steps": [{"operation": "decode", "tokens": [1], "count": 3, "logits": "row.f32"}]}
        for folder in (first, second):
            (folder / "probe.json").write_text(json.dumps(base))
        (first / "row.f32").write_bytes(struct.pack("<3f", 1, 2, 3))
        for values in ((1, 2, float("nan")), (1, 4, 3)):
            (second / "row.f32").write_bytes(struct.pack("<3f", *values))
            evidence = result()
            with self.subTest(values=values), self.assertRaises(ScenarioFailure):
                scaling._compare(first / "probe.json", second / "probe.json", profile, self.context, evidence)
            self.assertEqual(evidence["evidence"]["numerical_comparison"]["status"], "failed")

    def test_lora_ignored_adapter_cannot_pass_just_because_loading_succeeded(self):
        profile = self.profile("lora")
        folder = self.root / "ignored-lora"
        folder.mkdir()
        evidence = result()
        http = LoraHttpFixture(evidence, texts=("blue", "blue"))
        with self.assertRaises(ScenarioFailure):
            scaling._lora_requests(http, "model.gguf", profile, evidence, self.adapter, folder)
        self.assertEqual(evidence["assertions"][-1]["name"], "exact adapted oracle differs from baseline")

    def test_lora_changes_expected_answer_and_unload_restores_base(self):
        profile = self.profile("lora")
        evidence = result()
        folder = self.root / "loras"
        folder.mkdir()
        http = LoraHttpFixture(evidence)
        scaling._lora_requests(http, "model.gguf", profile, evidence, self.adapter, folder)
        oracle = evidence["evidence"]["lora_oracle"]
        self.assertEqual((oracle["before"], oracle["adapted"], oracle["after"]), ("blue", "orange", "blue"))
        self.assertTrue(any(method == "DELETE" for _, _, method in http.calls))
        self.assertEqual(self.adapter.read_bytes(), (folder / "verified-adapter.safetensors").read_bytes())

    def test_lora_wrong_baseline_adaptation_or_unload_cannot_satisfy_profile_oracles(self):
        cases = (
            ({"texts": ("56",)}, "exact baseline oracle"),
            ({"texts": ("blue", "56")}, "exact adapted oracle differs from baseline"),
            ({"texts": ("blue", "orange", "orange")}, "per-request baseline remains unmodified"),
            ({"texts": ("blue", "orange", "blue", "orange")}, "unload restores exact baseline"),
            ({"removed": False}, "adapter unload acknowledged"),
            ({"retain": True}, "no adapter remains after unload"),
        )
        for index, (changes, assertion) in enumerate(cases):
            with self.subTest(assertion=assertion):
                evidence = result()
                folder = self.root / f"lora-negative-{index}"
                folder.mkdir()
                with self.assertRaisesRegex(ScenarioFailure, assertion):
                    scaling._lora_requests(LoraHttpFixture(evidence, **changes), "model.gguf",
                                           self.profile("lora"), evidence, self.adapter, folder)
                self.assertEqual(assertion, evidence["assertions"][-1]["name"])
                self.assertFalse(evidence["assertions"][-1]["passed"])
                self.assertNotIn("lora_oracle", evidence["evidence"])

    def test_lora_profile_oracle_does_not_disable_chat_envelope_identity_or_usage_checks(self):
        def bad_finish(response):
            response["choices"][0]["finish_reason"] = "length"
            return response
        for index, (mutator, reason) in enumerate((
            (lambda response: dict(response, object="invalid"), "envelope"),
            (lambda response: dict(response, model="another-model.gguf"), "identity"),
            (lambda response: dict(response, usage={"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 8}), "usage"),
            (lambda response: dict(response, usage={"prompt_tokens": 8, "completion_tokens": 17, "total_tokens": 25}), "budget"),
            (bad_finish, "finish normally"),
        )):
            with self.subTest(reason=reason):
                evidence = result()
                folder = self.root / f"lora-protocol-{index}"
                folder.mkdir()
                http = LoraHttpFixture(evidence)
                http.response_mutator = mutator
                with self.assertRaisesRegex(ScenarioFailure, reason):
                    scaling._lora_requests(http, "model.gguf", self.profile("lora"), evidence, self.adapter, folder)
                self.assertFalse(any(body for route, body, _ in http.calls if route == "/v1/loras"),
                                 "invalid baseline envelope must block adapter loading")

    def test_lora_requires_effect_oracle_and_safe_parameters(self):
        for key, bad in (("adapter_name", "../escape"), ("scale", 0), ("scale", float("inf")), ("max_tokens", 1000)):
            profile = self.profile("lora")
            profile["parameters"][key] = bad
            with self.assertRaises(ValueError): scaling.validate_profile(profile)
        profile = self.profile("lora")
        profile["expected"]["adapted_text"] = profile["expected"]["base_text"]
        with self.assertRaises(ValueError): scaling.validate_profile(profile)

    def test_lora_family_and_backend_are_gated_before_model_preparation(self):
        self.context.update(scenario_profile=self.profile("lora"), preparing_model=True, model_family="mistral3")
        self.assertTrue(scaling.availability({"id": "lora"}, self.context)[0])
        self.context["model_family"] = "qwen4exp"
        self.assertFalse(scaling.availability({"id": "lora"}, self.context)[0])
        self.context.update(model_family="nemotron_h", backend="mlx")
        self.context["scenario_profile"]["backend"] = "mlx"
        self.assertFalse(scaling.availability({"id": "lora"}, self.context)[0])

    def test_distributed_safe_prelaunch_unavailability_remains_untested(self):
        from eng.regression import scaling_remote
        profile = self.profile("multi_gpu")
        profile.update(scenario="distributed")
        profile["expected"]["weights_id"] = "pinned-weights"
        self.context["scenario_profile"] = profile
        with patch.object(scaling, "availability", return_value=(True, "")), \
             patch.object(scaling, "_probe", return_value=self.root / "reference.json") as control, \
             patch.object(scaling_remote, "run_remote_nodes", side_effect=scaling_remote.RemoteUnavailable("all nodes lack memory")), \
             patch.object(scaling, "_compare") as compare, self.assertRaises(ScenarioUntested):
            scaling.run({"id": "distributed"}, self.context, result())
        control.assert_called_once()
        compare.assert_not_called()

    def test_distributed_wrapper_compares_the_collected_driver_with_fresh_control(self):
        from eng.regression import scaling_remote
        profile = self.profile("multi_gpu")
        profile.update(scenario="distributed")
        profile["expected"]["weights_id"] = "pinned-weights"
        self.context["scenario_profile"] = profile
        driver = self.root / "driver.json"
        driver.write_text('{"schema_version":1}')
        reference = self.root / "reference.json"
        receipt = {"node_index": 0, "local_probe_path": str(driver), "teardown": {"verified": True}}
        with patch.object(scaling, "availability", return_value=(True, "")), \
             patch.object(scaling, "_probe", return_value=reference), \
             patch.object(scaling_remote, "run_remote_nodes", return_value=[receipt]), \
             patch.object(scaling, "_probe_report") as validate, patch.object(scaling, "_compare") as compare:
            evidence = result()
            scaling.run({"id": "distributed"}, self.context, evidence)
        validate.assert_called_once()
        self.assertEqual(compare.call_args.args[:2], (reference, driver))
        self.assertEqual(evidence["evidence"]["remote_nodes"], [receipt])


if __name__ == "__main__":
    unittest.main()
