#!/usr/bin/env python3
"""CPU-only source/mapping/resource/admission checks. No CUDA process is run."""
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
spec = importlib.util.spec_from_file_location("int8mma_explicit_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class SourceAndMappingTests(unittest.TestCase):
    def test_oracle_prefix_is_imported_verbatim_without_rewriting_arithmetic(self):
        source = build.SOURCE.read_text()
        prefix = build.oracle_prefix(source)
        self.assertTrue(source.startswith(prefix))
        generated = build.generate(source)
        self.assertEqual(generated[generated.index(prefix):generated.index(prefix) + len(prefix)], prefix)
        self.assertNotIn("quantize_nvfp4_f32(", generated)
        self.assertIn("TS_Q35_GEMV(8, TS_Q35_GEMV_ROWS(8))", generated)

    def test_changed_oracle_partition_or_accumulation_requires_manual_review(self):
        source = build.SOURCE.read_text()
        for old, new in (("b += 32", "b += 16"), ("ds[j] * xs * (float)sumi", "xs * (ds[j] * (float)sumi)")):
            with self.assertRaises(ValueError):
                build.generate(source.replace(old, new))

    def test_all_mma_fragments_cover_each_element_once_and_output_scales_differ_from_B_loader(self):
        aa, bb, dd = [], [], []
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            aa.extend((g + (8 if reg & 1 else 0), 4 * t + byte + (16 if reg >= 2 else 0))
                      for reg in range(4) for byte in range(4))
            bb.extend((4 * t + byte + 16 * reg, g) for reg in range(2) for byte in range(4))
            dd.extend((g + (8 if c >= 2 else 0), 2 * t + (c & 1)) for c in range(4))
        self.assertEqual(len(aa), len(set(aa)));self.assertEqual(set(aa), {(m, k) for m in range(16) for k in range(32)})
        self.assertEqual(len(bb), len(set(bb)));self.assertEqual(set(bb), {(k, n) for k in range(32) for n in range(8)})
        self.assertEqual(len(dd), len(set(dd)));self.assertEqual(set(dd), {(m, n) for m in range(16) for n in range(8)})
        self.assertTrue(any((lane >> 2) != 2 * (lane & 3) for lane in range(32)))

    def test_nvfp4_word_selection_for_every_nibble_value_and_logical_position(self):
        table = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
        for position in range(16):
            for code in range(16):
                logical = [0] * 16;logical[position] = code
                words = [sum((logical[4*w+i] | (logical[4*w+i+8] << 4)) << (8*i) for i in range(4)) for w in range(2)]
                actual = []
                for t in range(4):
                    word = words[t & 1]
                    actual.extend(table[(word >> (8*i + (0 if t < 2 else 4))) & 15] for i in range(4))
                self.assertEqual(actual, [table[q] for q in logical])

    def test_integer_dot_exact_bound_and_float_conversion(self):
        rng = random.Random(20261101)
        values = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
        for _ in range(500):
            w = [rng.choice(values) for _ in range(16)];x = [rng.randrange(-128,128) for _ in range(16)]
            old = sum(sum(w[i] * x[i] for i in range(4*j, 4*j+4)) for j in range(4))
            mma = sum(a*b for a,b in zip(w+[0]*16, x+[0]*16))
            self.assertEqual(old, mma)
            self.assertLessEqual(abs(mma), 24576)
            self.assertEqual(struct.unpack("f", struct.pack("f", float(mma)))[0], mma)
        self.assertEqual(16 * 12 * 128, 24576)

    def test_old_lane_scale_order_and_empty_warps_are_preserved(self):
        for blocks in (1, 2, 31, 32, 33, 80, 96, 272):
            all_groups = []
            for old_lane in range(32):
                expected = [(b, s, 2*b+(s >> 1)) for b in range(old_lane, blocks, 32) for s in range(4)]
                actual = [];b = old_lane
                while b < blocks:
                    actual.extend((b, s, (64*b+16*s) >> 5) for s in range(4));b += 32
                self.assertEqual(expected, actual)
                all_groups.extend((b, s) for b, s, sb in actual)
                if old_lane >= blocks:self.assertEqual(actual, [])
            self.assertEqual(len(all_groups), blocks*4)
            self.assertEqual(len(set(all_groups)), blocks*4)

    def test_shared_partial_coverage_and_reducer_bank_layout(self):
        for warp in range(32):
            writes = []
            for lane in range(32):
                g,t = lane >> 2,lane & 3
                writes.extend((warp, row*8+2*t+offset) for row in (g,g+8) for offset in (0,1))
            self.assertEqual(len(set(writes)),128)
        for output in range(128):
            self.assertEqual(len({(lane*129+output)%32 for lane in range(32)}),32)
            self.assertEqual(len({(lane*128+output)%32 for lane in range(32)}),1)
        kernel = (HERE / "int8mma.cuh").read_text()
        self.assertIn("partial[32][129]", kernel)
        self.assertNotIn("reinterpret_cast<float2", kernel)
        self.assertIn("ts_q35_warp_sum(partial[lane][output])", kernel)
        self.assertIn("alpha * v", kernel)

    def test_required_shape_alpha_and_subnormal_coverage_is_planned_not_claimed_passed(self):
        plan = build.plan()
        self.assertEqual({(s["k"], s["out_dim"]) for s in plan if s["timed"]},
            {(k,m) for k in (5120,6144,17408) for m in (5120,14336,16480,34816)})
        self.assertTrue(all(s["k"] > 0 and s["k"] % 64 == 0 and s["out_dim"] > 0 for s in plan))
        self.assertTrue({1,7,8,15,16,17,31,33}.issubset({s["out_dim"] for s in plan}))
        self.assertTrue({0,0x80000000,1,0x00800000,0x7f7fffff,0xff7fffff}.issubset({s["alpha_bits"] for s in plan}))
        source = (HERE / "bench.cu").read_text()
        self.assertIn("0x007fffff",source);self.assertIn("0x80000001",source)

    def test_no_forced_register_constraint_or_quality_bypass(self):
        kernel = (HERE / "int8mma.cuh").read_text()
        self.assertNotIn("__launch_bounds__(",kernel)
        self.assertFalse(any("maxrregcount" in flag for flag in build.FLAGS))
        source = (HERE / "bench.cu").read_text()
        self.assertLess(source.index("for(const Shape& s:StrictPlan) {"),source.index('if(mode=="--bench")'))
        self.assertIn('"oracle_A-A"',source);self.assertIn('"candidate_A-A"',source)


class ResourceAndAdmissionTests(unittest.TestCase):
    def setUp(self):
        parent = build.ROOT / "artifacts/qwen38-regression/int8mma-cpu-tests"
        parent.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def record(regs=64, stack=0, stores=0, loads=0, shared=16512):
        return (f"ptxas info : Compiling entry function '{build.CANDIDATE}' for 'sm_120a'\n"
                f"ptxas info : Function properties for {build.CANDIDATE}\n"
                f" {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
                f"ptxas info : Used {regs} registers, {shared} bytes smem\n")

    def test_actual_ptxas_over_budget_or_any_local_spill_is_rejected(self):
        good = build.parse_resources(self.record())
        self.assertTrue(good["offline_resource_qualified"])
        self.assertEqual(good["allocated_registers_per_cta"],65536)
        for change in ({"regs":65},{"stack":4},{"stores":4},{"loads":4},{"shared":16384},{"shared":16528}):
            self.assertFalse(build.parse_resources(self.record(**change))["offline_resource_qualified"],change)
        for log in ("",self.record().replace("Used 64 registers", "unknown registers"),self.record()+self.record()):
            with self.assertRaises(ValueError):build.parse_resources(log)

    def test_generation_only_never_creates_a_gpu_quality_permit(self):
        out = self.root / "generated"
        with patch("subprocess.run",side_effect=AssertionError("generation invoked a compiler/GPU process")):
            self.assertEqual(build.main(["--out",str(out),"--generate-only"]),0)
        state = json.loads((out/"manifest.json").read_text())
        self.assertTrue(state["generation_complete"])
        self.assertFalse(state["complete"])
        self.assertEqual(state["cuda_compilation"],"NOT_RUN")
        self.assertEqual(state["gpu_quality"],"NOT_RUN")
        self.assertFalse((out/"gpu-quality-permitted.json").exists())

    def test_N1_to_N7_are_NOT_SUPPORTED_without_invoking_cuda_or_compiler(self):
        for acts in range(1,8):
            with patch.object(run,"verify_build",side_effect=AssertionError("unsupportedN checked GPU build")), \
                 patch("subprocess.run",side_effect=AssertionError("unsupportedN invoked a process")):
                self.assertEqual(run.main(["--build",str(self.root),"--name",f"n{acts}","--acts",str(acts)]),2)
            state=json.loads((self.root/f"n{acts}/run.json").read_text())
            self.assertEqual(state["status"],"NOT_SUPPORTED")
            self.assertFalse(state["gpu_invoked"])
            self.assertFalse(state["complete"])

    def test_missing_or_stale_offline_gate_cannot_start_gpu(self):
        (self.root/"manifest.json").write_text(json.dumps({"complete":False}))
        (self.root/"gpu-quality-permitted.json").write_text("{}")
        with self.assertRaises(ValueError):run.verify_build(self.root)
        with patch("subprocess.run",side_effect=AssertionError("rejected build invoked GPU")):
            self.assertEqual(run.main(["--build",str(self.root),"--name","blocked"]),1)
        self.assertFalse(json.loads((self.root/"blocked/run.json").read_text())["gpu_invoked"])

    def test_changed_artifact_or_missing_identity_cannot_start_gpu(self):
        source = self.root / "oracle.cu"
        source.write_text("unchanged source")
        for name in build.ARTIFACTS:
            (self.root/name).write_text("compiled fixture identity, never executed")
        manifest = dict(complete=True, activation_rows=8, resources={"offline_resource_qualified":True},
            source=str(source), source_sha256=build.digest(source),
            tools={name:build.digest(HERE/name) for name in build.TOOLS},
            artifacts={name:build.digest(self.root/name) for name in build.ARTIFACTS})
        def issue_permit():
            (self.root/"manifest.json").write_text(json.dumps(manifest))
            (self.root/"gpu-quality-permitted.json").write_text(json.dumps(dict(
                manifest_sha256=build.digest(self.root/"manifest.json"),artifacts=manifest["artifacts"])))
        issue_permit()
        self.assertEqual(run.verify_build(self.root),manifest)
        (self.root/"kernels.cubin").write_text("changed artifact")
        with self.assertRaisesRegex(ValueError,"artifact changed"):
            run.verify_build(self.root)
        with patch("subprocess.run",side_effect=AssertionError("changed artifact invoked GPU")):
            self.assertEqual(run.main(["--build",str(self.root),"--name","changed-blocked"]),1)
        self.assertFalse(json.loads((self.root/"changed-blocked/run.json").read_text())["gpu_invoked"])
        del manifest["artifacts"]["kernels.cubin"]
        issue_permit()
        with self.assertRaisesRegex(ValueError,"does not identify"):
            run.verify_build(self.root)

    def test_missing_oracle_source_records_failure_and_removes_stale_permit(self):
        (self.root/"gpu-quality-permitted.json").write_text("stale")
        with patch("subprocess.run",side_effect=AssertionError("missing source invoked a process")):
            self.assertEqual(build.main(["--out",str(self.root),"--source",str(self.root/"missing.cu")]),1)
        state=json.loads((self.root/"manifest.json").read_text())
        self.assertFalse(state["complete"])
        self.assertEqual(state["cuda_compilation"],"NOT_RUN")
        self.assertFalse((self.root/"gpu-quality-permitted.json").exists())

    def test_injected_nvcc_flags_cannot_relax_precision_or_resource_gate(self):
        for name in ("NVCC_PREPEND_FLAGS","NVCC_APPEND_FLAGS"):
            with patch.dict("os.environ",{name:"--use_fast_math --maxrregcount=64"}), \
                 patch("subprocess.run",side_effect=AssertionError("injected flags invoked a process")):
                self.assertEqual(build.main(["--out",str(self.root),"--generate-only"]),1)
            state=json.loads((self.root/"manifest.json").read_text())
            self.assertFalse(state["complete"])
            self.assertIn("must be unset",state["error"])
            self.assertFalse((self.root/"gpu-quality-permitted.json").exists())


if __name__ == "__main__":
    unittest.main()
