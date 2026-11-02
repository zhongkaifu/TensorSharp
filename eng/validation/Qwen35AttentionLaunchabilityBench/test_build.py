"""CPU-only source, admission, failure and ownership tests. No nvcc/CUDA/GPU."""
from __future__ import annotations
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


build = module("launchability_build_cpu", HERE / "build.py")
run = module("launchability_run_cpu", HERE / "run.py")


def resource_text(regs=240, stack=0, stores=0, loads=0):
    return ("ptxas info : Compiling entry function 'ts_bench_attn_prefill' for 'sm_120a'\n"
            "ptxas info : Function properties for ts_bench_attn_prefill\n"
            f" {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
            f"ptxas info : Used {regs} registers, 0 bytes smem\n")


def observations(variant=3, mode="bench", long_cases=True):
    shared = {0: 68608, 2: 68608, 3: 51200, 4: 50176}
    facts = {str(i): dict(registers_per_thread=254 if i == 0 else 240,
                         dynamic_shared_bytes=shared[i], static_shared_bytes=0) for i in (0, variant)}
    manifest = dict(candidate_variant=variant, resources=facts)
    events = [dict(event="device", name="CPU_MOCK_NOT_GPU", sms=170,
                   registers_per_sm=65536, shared_per_sm=102400)]
    for i in (0, variant):
        events.append(dict(event="device_resources", variant=i, regs=facts[str(i)]["registers_per_thread"],
                           local_bytes=24 if i == 0 else 0, dynamic_shared=shared[i],
                           static_shared=0, potential_active_ctas=1))
    result = dict(complete=True, device_resource_rejected_variants=[])
    if mode == "resources":
        result["validation"] = "resources_only_not_quality"
    else:
        ss = run.shapes(long_cases)
        events += [dict(event="strict_shape_pass", shape=run.name(s), variants=2,
                        full_output_and_partial_bits=True) for s in ss]
        result.update(strict_shapes=len(ss), strict_candidate_comparisons=3 * len(ss),
                      active_variants=[0, variant], long_cases=long_cases, timed=mode == "bench")
        if mode == "bench":
            events += [dict(event="abba_gpu_ms", shape=run.name(s), repetition=r,
                            total_keys=s[0] + s[1], variant=variant, kernels_per_half=3,
                            complete_subtiles_and_combine=True, poison_and_verify_each_half=True,
                            a=1.0, b=1.2, c=1.3, d=1.1)
                       for s in run.timed_shapes(long_cases) for r in range(5)]
    return result, events, manifest


class SourceTests(unittest.TestCase):
    def test_pinned_kernels_and_only_runner_policy_delta(self):
        source = build.SOURCE.read_text()
        sealed = build.frozen_module()
        for variant in (0, 2, 3, 4):
            self.assertEqual(build.generated(source, variant), sealed.generate(source, variant))
        original = (build.FROZEN / "bench.cu").read_text()
        private = build.runner_source()
        self.assertEqual(private.replace(build.NEW_POLICY, build.OLD_POLICY, 1), original)
        self.assertEqual(private.count(build.NEW_POLICY), 1)
        self.assertIn("Equal(oracle,f.Read(output)", private)
        self.assertIn("static Result Oracle(", private)
        self.assertIn("complete_subtiles_and_combine", private)

    def test_five_arms_control_never_receives_candidate_O1(self):
        self.assertEqual(len(build.ARMS), 5)
        for arm, (variant, o1) in build.ARMS.items():
            control = build.compile_command("nvcc", "sm_120a", Path("/unused"), 0, arm)
            candidate = build.compile_command("nvcc", "sm_120a", Path("/unused"), variant, arm)
            self.assertIn("-O3", control)
            self.assertNotIn("-Xptxas=-v,-warn-spills,-O1", control)
            self.assertEqual("-Xptxas=-v,-warn-spills,-O1" in candidate, o1)
            self.assertIn("--extra-device-vectorization", control)
            self.assertFalse(any("maxrregcount" in x or "fast_math" in x for x in candidate))

    def test_sealed_pin_and_changed_attention_source_rejected(self):
        with patch.dict(build.PINS, {"checks.h": "0" * 64}):
            with self.assertRaises(ValueError):
                build.frozen_module()
        changed = build.SOURCE.read_text().replace("TS_Q35_AI_KT 32", "TS_Q35_AI_KT 16")
        with self.assertRaises(ValueError):
            build.generated(changed, 0)

    def test_actual_high_register_and_spill_facts_are_not_old_gate(self):
        for variant, regs in ((2, 237), (3, 240), (4, 246)):
            r = build.parse_resources(resource_text(regs), variant)
            self.assertEqual(r["registers_per_thread"], regs)
            self.assertFalse(r["sealed_policy_would_admit"])
            self.assertEqual(r["status"], "OFFLINE_RESOURCE_FACTS_ONLY_DRIVER_NOT_RUN")
        r = build.parse_resources(resource_text(128, 184, 492, 508), 3)
        self.assertEqual(r["stack_bytes"], 184)
        self.assertEqual(r["spill_load_bytes"], 508)
        for text in ("", resource_text() * 2, resource_text().replace("Used 240 registers", "unknown")):
            with self.assertRaises(ValueError):
                build.parse_resources(text, 3)

    def test_original_shape_and_timing_sets(self):
        self.assertEqual(len(run.shapes(False)), 79)
        self.assertEqual(len(run.shapes(True)), 81)
        self.assertEqual(len({run.name(s) for s in run.shapes(True)}), 81)
        self.assertEqual(len(run.timed_shapes(False)), 4)
        self.assertEqual(len(run.timed_shapes(True)), 6)
        self.assertEqual({s[0] + s[1] for s in run.timed_shapes(True)[-2:]}, {131072, 250000})


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        root = build.ROOT / "artifacts/qwen38-regression/launchability64-cpu-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def invoke(self, out, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return build.main(["--out", str(out), "--arm", "v3-original", *extra])

    def fake_command(self, fail=None):
        def call(argv, log, timeout):
            if argv[-1] == "--version":
                if fail == "version":
                    raise subprocess.CalledProcessError(1, argv)
                log.write_text("CPU_MOCK_VERSION_NOT_CUDA_EVIDENCE\n")
                return log.read_text()
            output = Path(argv[argv.index("-o") + 1])
            if fail == "runner" and output.name == build.RUNNER:
                log.write_text("CPU MOCK injected runner failure\n")
                raise subprocess.CalledProcessError(1, argv)
            if fail == "candidate" and output.name == "variant-3.cubin":
                log.write_text("CPU MOCK injected candidate failure\n")
                raise subprocess.CalledProcessError(1, argv)
            output.write_bytes(b"CPU_MOCK_NOT_EXECUTABLE_NOT_GPU_EVIDENCE")
            text = resource_text(254, 24, 28, 40) if output.name == "variant-0.cubin" else resource_text()
            log.write_text(text)
            return text
        return call

    def mock_build(self, directory):
        with patch.object(build, "command", side_effect=self.fake_command()):
            self.assertEqual(self.invoke(directory), 0)
        return directory

    def test_generate_only_has_no_native_command_marker_or_quality(self):
        out = self.directory / "generate"
        with patch.object(build, "command", side_effect=AssertionError("unexpected compiler")):
            self.assertEqual(self.invoke(out, "--generate-only"), 0)
        m = json.loads((out / "manifest.json").read_text())
        self.assertFalse(m["complete"])
        self.assertEqual(m["gpu_quality"], "NOT_RUN")
        self.assertFalse((out / build.PERMIT).exists())
        self.assertFalse((out / "qualified-variants.txt").exists())

    def test_existing_output_and_hidden_nvcc_flags_rejected_before_mutation(self):
        out = self.directory / "existing"
        out.mkdir()
        (out / "keep").write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(out, "--generate-only")
        self.assertEqual((out / "keep").read_text(), "keep")
        for variable in build.ENV_NAMES[:2]:
            target = self.directory / variable
            with patch.dict(os.environ, {variable: "--use_fast_math"}), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(target, "--generate-only")
            self.assertFalse(target.exists())

    def test_compile_failure_any_phase_has_no_permit(self):
        for failure in ("version", "candidate", "runner"):
            out = self.directory / failure
            with patch.object(build, "command", side_effect=self.fake_command(failure)):
                self.assertEqual(self.invoke(out), 1)
            self.assertFalse((out / build.PERMIT).exists())
            self.assertFalse((out / "qualified-variants.txt").exists())
            self.assertFalse(json.loads((out / "manifest.json").read_text())["complete"])

    def test_new_schema_compiled_facts_and_source_hash_gate(self):
        directory = self.mock_build(self.directory / "mock")
        m = run.verify_build(directory)
        self.assertFalse(m["resources"]["3"]["sealed_policy_would_admit"])
        (directory / "bench.cu").write_text("changed")
        with self.assertRaises(ValueError):
            run.verify_build(directory)

    def test_sealed_schema_or_marker_cannot_authorize_new_campaign(self):
        directory = self.mock_build(self.directory / "oldschema")
        permit = json.loads((directory / build.PERMIT).read_text())
        permit["schema"] = "q35-attention-compiler-ablation/1"
        (directory / build.PERMIT).write_text(json.dumps(permit))
        with self.assertRaises(ValueError):
            run.verify_build(directory)

    def test_missing_offline_permission_never_invokes_gpu(self):
        directory = self.directory / "source-only"
        self.assertEqual(self.invoke(directory, "--generate-only"), 0)
        with patch.object(run.build, "command", side_effect=AssertionError("unexpected GPU")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--build", str(directory), "--name", "refused"]), 1)
        state = json.loads((directory / "refused/controller.json").read_text())
        self.assertFalse(state["gpu_invoked"])
        self.assertFalse(state["quality_pass"])

    def test_resource_only_never_quality_and_one_CTA_is_explicitly_eligible(self):
        for mode in ("resources", "strict", "bench"):
            result, events, manifest = observations(mode=mode)
            gate = run.validate_observation(result, events, manifest, mode, True)
            self.assertEqual(gate["quality"], "NOT_RUN" if mode == "resources" else "PASS")
            self.assertTrue(all(e["potential_active_ctas"] == 1 for e in gate["resources"]))

    def test_missing_duplicate_wrong_counts_or_bad_half_rejected(self):
        for fault in ("missing", "duplicate", "driver_zero", "driver_duplicate", "nan", "negative", "incomplete_work", "wrong_variant", "wrong_counts", "unexpected"):
            result, events, manifest = observations()
            if fault == "missing":
                events.pop()
            elif fault == "duplicate":
                events.append(copy.deepcopy(next(e for e in events if e["event"] == "strict_shape_pass")))
            elif fault == "driver_zero":
                events[2]["potential_active_ctas"] = 0
            elif fault == "driver_duplicate":
                events.append(copy.deepcopy(events[2]))
            elif fault in ("nan", "negative"):
                next(e for e in events if e["event"] == "abba_gpu_ms")["b"] = float("nan") if fault == "nan" else -1
            elif fault == "incomplete_work":
                next(e for e in events if e["event"] == "abba_gpu_ms")["complete_subtiles_and_combine"] = False
            elif fault == "wrong_variant":
                result["active_variants"] = [0, 4]
            elif fault == "wrong_counts":
                result["strict_candidate_comparisons"] -= 1
            else:
                events.append(dict(event="resource_rejected_no_gpu_run", variant=3))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                run.validate_observation(result, events, manifest, "bench", True)

    def test_timeout_kills_only_owned_unreaped_session(self):
        class Process:
            pid = 123456789
            returncode = None
            waits = 0
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired(["CPU_MOCK"], 1)
                self.returncode = -9
                return -9
        process = Process()
        with patch.object(build.subprocess, "Popen", return_value=process) as popen, patch.object(build.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                build.command(["CPU_MOCK_NOT_EXECUTED"], self.directory / "timeout.log", 1)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(process.waits, 2)


if __name__ == "__main__":
    unittest.main()
