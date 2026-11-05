"""CPU source/coverage/resource/admission gates only, never nvcc or CUDA execution."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
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


build = module("dp4a_rows_tested_build", HERE / "build.py")
run = module("dp4a_rows_tested_run", HERE / "run.py")


def resource_record(symbol, regs=172, stack=0, stores=0, loads=0, shared=0):
    return (f"ptxas info : Compiling entry function '{symbol}' for 'sm_120a'\n"
            f"ptxas info : Function properties for {symbol}\n"
            f" {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
            f"ptxas info : Used {regs} registers, {shared} bytes smem\n")


def resource_log(**change):
    return "".join(resource_record(s, **change) for s in build.SYMBOLS)


class SourceTests(unittest.TestCase):
    def test_original_prefix_and_template_are_verbatim_only_four_private_wrappers_added(self):
        source = build.SOURCE.read_text()
        prefix = build.SEALED.oracle_prefix(source)
        generated = build.generate(source)
        self.assertTrue(generated.startswith("#include <cuda_runtime.h>\n" + prefix))
        suffix = generated[len("#include <cuda_runtime.h>\n" + prefix):]
        for r in build.ROWS:
            self.assertEqual(suffix.count(f'void ts_bench_dp4a_rows_8_r{r}('), 1)
            self.assertEqual(suffix.count(f'ts_q35_gemv_nvfp4<8, {r}>('), 1)
        self.assertNotIn("__launch_bounds__", suffix)
        self.assertNotIn("__shared__", generated)
        self.assertNotIn("int8mma", generated)
        self.assertFalse(any("maxrregcount" in f or "fast_math" in f for f in build.FLAGS))
        self.assertEqual(build.FLAGS, build.SEALED.FLAGS)

    def test_changed_arithmetic_layout_or_default_R_requires_manual_review(self):
        source = build.SOURCE.read_text()
        for old, new in (("b += 32", "b += 16"),
                ("ds[j] * xs * (float)sumi", "xs * (ds[j] * (float)sumi)"),
                ("(N) <= 3 ? 2 : 4", "(N) <= 3 ? 2 : 3")):
            with self.assertRaises(ValueError):
                build.generate(source.replace(old, new))

    def test_exact_frozen_plan_96_cases_12_timings_and_fixture_initialization(self):
        plan = build.SEALED.plan()
        self.assertEqual(len(plan), 96)
        self.assertEqual(sum(p["timed"] for p in plan), 12)
        self.assertEqual({(p["k"], p["out_dim"]) for p in plan if p["timed"]},
            {(k,m) for k in (5120,6144,17408) for m in (5120,14336,16480,34816)})
        frozen = (build.FROZEN / "bench.cu").read_text()
        current = (HERE / "bench.cu").read_text()
        for a,b in (("struct Buffer {", "struct Module {"),
                    ("struct Fixture {", "    void Issue("),
                    ("    std::vector<uint32_t> Read()", "static void Strict(")):
            old_start, old_end = frozen.index(a), frozen.index(b)
            new_end_marker = "struct Variant {" if b == "struct Module {" else "static std::array<int,8> Argmax(" if b == "static void Strict(" else b
            self.assertEqual(frozen[old_start:old_end], current[current.index(a):current.index(new_end_marker)])
        self.assertTrue({1,7,8,15,16,17,31,33}.issubset({p["out_dim"] for p in plan}))
        self.assertTrue({0,0x80000000,1,0x00800000,0x7f7fffff,0xff7fffff}.issubset({p["alpha_bits"] for p in plan}))

    def test_grid_and_row_coverage_all_tail_sizes_primary_threads_are_128(self):
        for m in (1,2,3,4,7,8,15,16,17,31,33,5119,5120,5121,34817):
            for r in build.ROWS:
                threads = 32 if r == 4 else 128
                grid, block = build.launch_geometry(m,r,threads)
                covered = []
                for cta in range(grid):
                    for warp in range(block//32):
                        first = (cta*(block//32)+warp)*r
                        if first < m:
                            covered.extend(first+j for j in range(r) if first+j<m)
                self.assertEqual(covered,list(range(m)))
            grid,threads = build.launch_geometry(m,4,128)
            self.assertEqual(grid,(m+15)//16)
        source = (HERE / "bench.cu").read_text()
        self.assertIn('Variant oracle{4,128,"ts_q35_gemv_nvfp4_8"}',source)
        for r in (1,2,3):self.assertIn(f'{{{r},128,"ts_bench_dp4a_rows_8_r{r}"}}',source)
        self.assertIn('{4,32,"ts_bench_dp4a_rows_8_r4"}',source)
        self.assertIn('v.function,v.threads,0',source)
        self.assertIn('cuModuleGetFunction(&query,module,"ts_q35_gemv_rows_per_warp")',source)
        self.assertIn('if(rows[8]!=4)throw',source)
        self.assertIn('table.CheckGuard()',source)

    def test_old_lane_partition_scale_and_warp_sum_tree_do_not_change_with_R(self):
        for blocks in (1,2,31,32,33,80,96,272):
            oracle = [[(b,s) for b in range(lane,blocks,32) for s in range(4)] for lane in range(32)]
            for r in build.ROWS:
                for j in range(r):
                    self.assertEqual([[(b,s) for b in range(lane,blocks,32) for s in range(4)] for lane in range(32)],oracle)
        # Shuffle XOR partner graph is shared by the exact same source template.
        source = build.generate(build.SOURCE.read_text())
        self.assertIn("for (int off = 16; off > 0; off >>= 1)",source)
        self.assertIn("v += __shfl_xor_sync(0xffffffff, v, off, 32)",source)
        self.assertIn("acc[n][j] += ds[j] * xs * (float)sumi;",source)

    def test_whole_output_guards_argmax_and_AA_gate_before_ABBA_with_every_half_verified(self):
        source = (HERE / "bench.cu").read_text()
        self.assertLess(source.index('for(const Shape& s:StrictPlan) {'), source.index('if(mode=="--bench")'))
        for text in ('"oracle_A-A"','"candidate_A-A"','Argmax(a,s)!=Argmax(b,s)',
                     'if(value>best)', 'EqualOutput(oracle,f.Read(),f.shape,"timed_half_fullbits")',
                     'codes.CheckGuard();scales.CheckGuard();xq.CheckGuard();xd.CheckGuard();'):
            self.assertIn(text,source)
        self.assertEqual(source.count('=VerifiedTime(f,'),4)
        self.assertNotIn('candidate?1024',source)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        parent=build.ROOT / "artifacts/qwen38-regression/dp4a-rows-cpu-tests"
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def invoke(self,out,*extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return build.main(["--out",str(out),*extra])

    def test_resources_accept_natural_172_regs_but_reject_spill_stack_shared_and_missing_records(self):
        resources=build.parse_resources(resource_log())
        self.assertTrue(all(v["offline_resource_qualified"] for v in resources.values()))
        self.assertEqual(resources[build.ORACLE]["block_threads"],128)
        self.assertEqual(resources[build.SYMBOLS[4]]["block_threads"],32)
        for changes in ({"regs":0},{"regs":256},{"stack":4},{"stores":4},{"loads":4},{"shared":4}):
            self.assertFalse(all(v["offline_resource_qualified"] for v in build.parse_resources(resource_log(**changes)).values()))
        for log in ("",resource_log()+resource_record(build.ORACLE),resource_log().replace("Used 172 registers","unknown registers")):
            with self.assertRaises(ValueError):build.parse_resources(log)

    def test_generation_does_not_invoke_compiler_or_cuda_and_never_authorizes_gpu(self):
        out=self.root/"generated"
        with patch.object(build.subprocess,"run",side_effect=AssertionError("unexpected process")):
            self.assertEqual(self.invoke(out,"--generate-only"),0)
        manifest=json.loads((out/"manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertTrue(manifest["generation_complete"])
        self.assertEqual(manifest["cuda_compilation"],"NOT_RUN")
        self.assertEqual(manifest["gpu_quality"],"NOT_RUN")
        self.assertEqual(manifest["strict_cases"],96)
        self.assertEqual(manifest["planned_strict_candidate_comparisons"],384)
        self.assertFalse((out/build.PERMIT).exists())
        self.assertEqual(json.loads((out/"plan.json").read_text()),build.SEALED.plan())

    def test_existing_output_cannot_be_overwritten_even_if_it_is_failed(self):
        out=self.root/"existing";out.mkdir();(out/"manifest.json").write_text("keep failed evidence")
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            self.invoke(out,"--generate-only")
        self.assertEqual((out/"manifest.json").read_text(),"keep failed evidence")

    def test_injected_flags_rejected_before_compilation_no_permit(self):
        for n in ("NVCC_PREPEND_FLAGS","NVCC_APPEND_FLAGS"):
            out=self.root/n
            with patch.dict(os.environ,{n:"--use_fast_math --maxrregcount=64"}),patch.object(build.subprocess,"run",side_effect=AssertionError("unexpected compiler")):
                self.assertEqual(self.invoke(out,"--generate-only"),1)
            self.assertFalse((out/build.PERMIT).exists())

    @staticmethod
    def compiler(fail="",bad_resources=False):
        def mock(command,**kwargs):
            if command[-1]=="--version":
                if fail=="version":raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0,"UNIT TEST FAKE NVCC, NOT CUDA EVIDENCE","")
            output=Path(command[command.index("-o")+1])
            if (fail=="runner" and output.name==build.RUNNER) or (fail=="cubin" and "-cubin" in command):
                return subprocess.CompletedProcess(command,1,"","injected failure")
            if fail!="missing":output.write_bytes(b"CPU_MOCK_NOT_CUDA")
            return subprocess.CompletedProcess(command,0,"",resource_log(stack=4 if bad_resources else 0) if "-cubin" in command else "")
        return mock

    def test_permit_only_after_actual_compile_resources_and_runner_success_mock_logic_only(self):
        out=self.root/"fakebuild"
        fake=self.compiler();calls=[]
        def observe(command,**kwargs):
            self.assertFalse((out/build.PERMIT).exists())
            calls.append(command)
            return fake(command,**kwargs)
        with patch.object(build.subprocess,"run",side_effect=observe),patch.object(build.shutil,"which",return_value=None):
            self.assertEqual(self.invoke(out),0)
        self.assertTrue((out/build.PERMIT).exists())
        manifest=json.loads((out/"manifest.json").read_text())
        self.assertEqual(run.verify_build(out),manifest)
        self.assertEqual(calls[-1][-1],str(out/build.RUNNER))
        (out/"kernels.cubin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"artifact changed"):run.verify_build(out)

    def test_all_build_failures_and_resource_rejections_do_not_authorize(self):
        for name in ("version","cubin","runner","missing","bad_resources"):
            out=self.root/name
            with patch.object(build.subprocess,"run",side_effect=self.compiler(fail=name,bad_resources=name=="bad_resources")),patch.object(build.shutil,"which",return_value=None):
                self.assertEqual(self.invoke(out),1)
            self.assertFalse((out/build.PERMIT).exists())
            self.assertFalse(json.loads((out/"manifest.json").read_text())["complete"])

    def test_N1_to_N7_never_check_build_or_invoke_cuda(self):
        for acts in range(1,8):
            with patch.object(run,"verify_build",side_effect=AssertionError("unsupported N checked build")),patch.object(run.subprocess,"run",side_effect=AssertionError("unsupported N launched GPU")),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--build",str(self.root),"--name",f"n{acts}","--acts",str(acts)]),2)
            state=json.loads((self.root/f"n{acts}/run.json").read_text())
            self.assertEqual(state["status"],"NOT_SUPPORTED")
            self.assertFalse(state["gpu_invoked"])

    def test_incomplete_or_foreign_int8_permit_cannot_launch_gpu(self):
        (self.root/"manifest.json").write_text(json.dumps(dict(schema="qwen35-int8mma-build/1",complete=True)))
        (self.root/build.PERMIT).write_text("{}")
        with patch.object(run.subprocess,"run",side_effect=AssertionError("invalid gate launched GPU")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--build",str(self.root),"--name","blocked"]),1)
        self.assertFalse(json.loads((self.root/"blocked/run.json").read_text())["gpu_invoked"])

    def test_result_does_not_pass_partial_comparison_plan_or_missing_timing(self):
        observed=dict(complete=True,status="STRICT_OUTPUT_BITS_AND_ARGMAX_PASS",activation_rows=8,
            strict_cases_per_candidate=96,strict_candidate_comparisons=384,candidates=4,
            timed_shapes_per_candidate=12,rounds=5,iterations_per_sample=3)
        self.assertTrue(run.validate_result(observed,True,5,3))
        for key,value in (("strict_cases_per_candidate",95),("strict_candidate_comparisons",96),
                          ("candidates",3),("timed_shapes_per_candidate",0),("complete",False)):
            bad=dict(observed,**{key:value})
            self.assertFalse(run.validate_result(bad,True,5,3))


if __name__=="__main__":unittest.main()
