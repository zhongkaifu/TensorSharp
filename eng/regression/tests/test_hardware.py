import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "hardware.py"
SPEC = importlib.util.spec_from_file_location("regression_hardware", MODULE)
hardware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardware)
GIB = hardware.GIB


def host(**changes):
    result = {"memory": {"total_bytes": 32 * GIB, "available_bytes": 30 * GIB},
              "disk": {"free_bytes": 100 * GIB},
              "gpus": [{"index": 0, "memory_free_bytes": 16 * GIB}],
              "backends": {name: {"available": True} for name in hardware.BACKENDS}}
    result.update(changes)
    return result


class HardwareTests(unittest.TestCase):
    def test_csv_parses_mib_and_quoted_name(self):
        gpu, = hardware.parse_nvidia_csv('0, "NVIDIA, model", 16384, 14000, 581.42, 8.6\n')
        self.assertEqual(16384 * 1024 ** 2, gpu["memory_total_bytes"])
        self.assertEqual(8.6, gpu["compute_capability"])

    def test_older_csv_missing_capability_is_unknown(self):
        gpu, = hardware.parse_nvidia_csv("0, GPU, 1024, 512, 581.1\n")
        self.assertIsNone(gpu["compute_capability"])

    def test_bad_csv_never_invents_capacity(self):
        self.assertEqual([], hardware.parse_nvidia_csv("N/A\n0, GPU, N/A, 4, 580\n"))

    def test_missing_tools_are_not_available(self):
        with patch.object(hardware.shutil, "which", return_value=None):
            self.assertFalse(hardware._tool("nvcc", ["--version"])["available"])

    def test_probe_timeout_is_recorded(self):
        with patch.object(hardware.subprocess, "run", side_effect=hardware.subprocess.TimeoutExpired(["x"], 1)):
            result = hardware._probe(["x"])
        self.assertIsNone(result["exit_code"])
        self.assertEqual("TimeoutExpired", result["stderr"])

    def test_disk_checks_nearest_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            result = hardware._disk(Path(directory) / "not-yet" / "models")
            self.assertEqual(str(Path(directory).resolve()), result["path"])
            self.assertGreater(result["free_bytes"], 0)

    def test_ggml_wrapper_alone_does_not_claim_cuda(self):
        runtime = {"files": {"ggmlops.dll": ["GgmlOps.dll"]}, "build_flags": {}, "build_metadata": []}
        result = hardware._backend_inventory("Windows", "AMD64", {"nvcc": {"available": True}},
                                             [{"name": "GPU"}], runtime)
        self.assertTrue(result["ggml_cpu"]["available"])
        self.assertFalse(result["ggml_cuda"]["available"])
        self.assertFalse(result["cuda"]["available"])
        self.assertFalse(result["ggml_vulkan"]["available"])

    def test_cuda_compile_flag_and_device_form_candidate(self):
        runtime = {"files": {"ggmlops.dll": ["GgmlOps.dll"]},
                   "build_flags": {"TENSORSHARP_GGML_NATIVE_ENABLE_CUDA": True}, "build_metadata": ["CMakeCache.txt"]}
        result = hardware._backend_inventory("Windows", "AMD64", {"nvcc": {"available": False}},
                                             [{"name": "GPU"}], runtime)
        self.assertTrue(result["ggml_cuda"]["available"])
        self.assertFalse(result["ggml_cuda"]["execution_verified"])

    def test_direct_cuda_requires_driver_and_compute_floor(self):
        runtime = {"files": {"tensorsharp.backends.cuda.dll": ["cuda.dll"]}, "build_flags": {}, "build_metadata": []}
        for driver, compute, expected in (("579.1", 8.6, False), ("581.1", 7.0, False),
                                           ("581.1", None, False), ("581.1", 8.6, True)):
            with self.subTest(driver=driver, compute=compute):
                result = hardware._backend_inventory("Windows", "AMD64", {"nvcc": {"available": True}},
                    [{"name": "GPU", "driver_version": driver, "compute_capability": compute}], runtime)
                self.assertEqual(expected, result["cuda"]["available"])

    def test_runtime_inventory_only_explicit_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "GgmlOps.dll").write_bytes(b"fake")
            (root / "CMakeCache.txt").write_text("TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:BOOL=ON\n")
            result = hardware._runtime_evidence([root])
            self.assertIn("ggmlops.dll", result["files"])
            self.assertTrue(result["build_flags"]["TENSORSHARP_GGML_NATIVE_ENABLE_CUDA"])

    def test_runtime_inventory_includes_conventional_cuda_kernel_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernels = root / "cuda_kernels"
            kernels.mkdir()
            expected = {"tensorsharp.ptx": b"synthetic ptx",
                        "tensorsharp.fatbin": b"synthetic fatbin",
                        "tensorsharp.fatbin.sha256": b"synthetic identity sidecar"}
            for name, contents in expected.items():
                (kernels / name).write_bytes(contents)
            # Neither model files, arbitrary sidecars nor deeper trees are assets.
            for name in ("private.gguf", "notes.json", "unrelated.sha256", "unexpected.dll"):
                (kernels / name).write_bytes(b"not a kernel")
            for folder in (kernels / "nested", root / "models"):
                folder.mkdir()
                (folder / "excluded.ptx").write_bytes(b"not a conventional asset")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("inventory must not read payloads")):
                result = hardware._runtime_evidence([root, kernels])
            # An explicitly supplied kernels dir may contain its own DLL; the
            # parent-dir inventory must restrict conventional children to kernels.
            parent_only = hardware._runtime_evidence([root])["files"]
            self.assertEqual(set(expected), set(parent_only))
            for name in expected:
                # Windows CI may canonicalize TEMP through its 8.3 alias. Assert
                # the exact file identity and single entry, not path spelling.
                self.assertEqual(1, len(result["files"][name]))
                self.assertTrue((kernels / name).samefile(result["files"][name][0]))
            self.assertNotIn("excluded.ptx", result["files"])

    def test_conventional_cuda_kernel_inventory_reaches_report_hashes(self):
        report_spec = importlib.util.spec_from_file_location("regression_reports", MODULE.with_name("reports.py"))
        reports = importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(reports)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernels = root / "cuda_kernels"
            kernels.mkdir()
            for name in ("test.ptx", "test.fatbin", "test.fatbin.sha256"):
                (kernels / name).write_bytes(name.encode())
            assets = reports.runtime_identity({"runtime": hardware._runtime_evidence([root])})
            self.assertEqual({"test.ptx", "test.fatbin", "test.fatbin.sha256"},
                             {Path(row["path"]).name for row in assets})
            self.assertTrue(all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in assets))

    def test_unknown_size_or_memory_not_eligible(self):
        self.assertFalse(hardware.plan_resources(None, "cpu", host())["eligible"])
        self.assertFalse(hardware.plan_resources(0, "cpu", host())["eligible"])
        self.assertFalse(hardware.plan_resources(2 * GIB, "cpu", host(memory={}))["eligible"])

    def test_missing_backend_needs_explicit_override(self):
        machine = host(backends={})
        self.assertFalse(hardware.plan_resources(GIB, "ggml_cpu", machine)["eligible"])
        result = hardware.plan_resources(GIB, "ggml_cpu", machine, {"assume_backend_available": True})
        self.assertTrue(result["eligible"])
        self.assertIn("overridden", result["warnings"][0])

    def test_model_fits_cpu_after_headroom(self):
        result = hardware.plan_resources(4 * GIB, "cpu", host())
        self.assertTrue(result["eligible"])
        self.assertEqual(7 * GIB, result["estimates"]["ram_required_bytes"])
        self.assertAlmostEqual(6.4, result["estimates"]["ram_reserve_bytes"] / GIB)

    def test_ram_oom_budget_rejected(self):
        result = hardware.plan_resources(22 * GIB, "cpu", host())
        self.assertFalse(result["eligible"])
        self.assertIn("RAM", result["reason"])

    def test_vram_not_summed_across_devices(self):
        machine = host(gpus=[{"index": 0, "memory_free_bytes": 8 * GIB},
                             {"index": 1, "memory_free_bytes": 8 * GIB}])
        result = hardware.plan_resources(5 * GIB, "ggml_cuda", machine)
        self.assertFalse(result["eligible"])
        self.assertIn("single-GPU", result["reason"])

    def test_partial_offload_requires_explicit_budget_fraction(self):
        machine = host(gpus=[{"index": 0, "memory_free_bytes": 8 * GIB}])
        result = hardware.plan_resources(5 * GIB, "ggml_cuda", machine, {"gpu_weight_fraction": .4})
        self.assertTrue(result["eligible"])
        self.assertIn("explicit partial offload", result["offload"])
        self.assertTrue(result["warnings"])

    def test_unknown_gpu_memory_not_eligible_without_override(self):
        self.assertFalse(hardware.plan_resources(GIB, "ggml_cuda", host(gpus=[]))["eligible"])
        self.assertTrue(hardware.plan_resources(GIB, "ggml_cuda", host(gpus=[]), {"vram_budget_gib": 8})["eligible"])

    def test_budget_overrides_cannot_invent_observed_capacity(self):
        machine = host(memory={"total_bytes": 32 * GIB, "available_bytes": 3 * GIB})
        result = hardware.plan_resources(2 * GIB, "cpu", machine, {"ram_budget_gib": 100})
        self.assertFalse(result["eligible"])
        self.assertEqual(3 * GIB, result["estimates"]["ram_budget_bytes"])

    def test_selected_direct_cuda_gpu_must_meet_requirements(self):
        machine = host(gpus=[{"index": 0, "memory_free_bytes": 16 * GIB,
                             "driver_version": "581.1", "compute_capability": 7.0},
                            {"index": 1, "memory_free_bytes": 16 * GIB,
                             "driver_version": "581.1", "compute_capability": 8.6}])
        self.assertFalse(hardware.plan_resources(GIB, "cuda", machine)["eligible"])
        self.assertTrue(hardware.plan_resources(GIB, "cuda", machine, {"gpu_index": 1})["eligible"])

    def test_fractional_device_index_is_not_truncated(self):
        self.assertFalse(hardware.plan_resources(GIB, "ggml_cuda", host(), {"gpu_index": .5})["eligible"])

    def test_disk_budget_and_headroom_are_enforced(self):
        result = hardware.plan_resources(GIB, "cpu", host(disk={"free_bytes": 9 * GIB}), {"download_bytes": 5 * GIB})
        self.assertFalse(result["eligible"])
        self.assertIn("disk", result["reason"])

    def test_nonfinite_negative_and_fraction_overrides_rejected(self):
        for options in ({"ram_budget_gib": float("inf")}, {"download_bytes": -1},
                        {"gpu_weight_fraction": 1.1}, {"vram_required_gib": float("nan")}):
            self.assertFalse(hardware.plan_resources(GIB, "ggml_cuda", host(), options)["eligible"])

    def test_cpu_offload_flag_is_rejected(self):
        self.assertFalse(hardware.plan_resources(GIB, "cpu", host(), {"gpu_weight_fraction": .5})["eligible"])

    def test_unified_memory_does_not_add_vram_and_ram_capacity(self):
        result = hardware.plan_resources(4 * GIB, "mlx", host(gpus=[]), {"ram_budget_gib": 8})
        self.assertTrue(result["eligible"])
        self.assertEqual(8 * GIB, result["estimates"]["vram_budget_bytes"])


if __name__ == "__main__":
    unittest.main()
