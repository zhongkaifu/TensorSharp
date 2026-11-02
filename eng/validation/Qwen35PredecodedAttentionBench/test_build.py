"""CPU source/layout/admission and actual host-checker tests; not CUDA math proof."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
build=load("predecoded_attention_tested_build",HERE/"build.py")
run=load("predecoded_attention_tested_run",HERE/"run.py")


def resource_log(stack=24,stores=28,loads=40):
    return "".join(f"ptxas info : Compiling entry function '{s}' for 'sm_120a'\n"
        f"ptxas info : Function properties for {s}\n {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
        "ptxas info : Used 254 registers\n" for s in build.SYMBOLS)


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=build.SOURCE.read_text();cls.generated=build.generate(cls.source)
        cls.original=build.BASE.generate(cls.source,0)
        cls.oracle=build.BASE.block(cls.original,'extern "C" __global__ void __launch_bounds__(256) ts_bench_attn_prefill(')
        cls.candidate=build.BASE.block(cls.generated,'extern "C" __global__ void __launch_bounds__(256) ts_bench_attn_f16_prefill(')

    def test_baseline0_is_the_entire_verbatim_prefix(self):
        self.assertTrue(self.generated.startswith(self.original+"\n"))

    def test_plain_decode_changes_only_helper_name_and_destination_stride(self):
        old=build.BASE.block(self.source,"__device__ __forceinline__ void ts_q35_tile_to_h(")
        new=build.BASE.block(self.generated,"__device__ __forceinline__ void ts_bench_tile_to_plain(")
        restored=new.replace("ts_bench_tile_to_plain(","ts_q35_tile_to_h(").replace("dst + key * TS_Q35_HD + b * 32","dst + key * TS_Q35_ATTN_ROW + b * 32")
        self.assertEqual(old,restored)
        self.assertIn("ts_q35_tile_to_h(raw, dst, threadIdx.x);",self.generated)

    def test_every_QK_softmax_PV_and_output_operation_stays_in_original_order(self):
        start="        float s[4][4];"
        math=self.candidate[self.candidate.index(start):]
        math=math.replace("sm.kv[st][0]","sm.k").replace("sm.kv[st][1]","sm.v")
        self.assertEqual(math,self.oracle[self.oracle.index(start):])
        a="    const float log2Scale";b="    for (int first = kLo"
        self.assertEqual(self.candidate[self.candidate.index(a):self.candidate.index(b)],self.oracle[self.oracle.index(a):self.oracle.index(b)])

    def test_double_bank_pipeline_retains_barriers_and_split_specific_zero_fill(self):
        self.assertIn("const bool ok = first + row < kHi;",self.candidate)
        self.assertIn("(ok ? (long long)(first + row) * TS_Q35_HD + col : 0)",self.candidate)
        self.assertIn("load(st ^ 1, first + TS_Q35_AI_KT)",self.candidate)
        self.assertEqual(self.candidate.count("__syncthreads();"),self.oracle.count("__syncthreads();"))
        self.assertNotIn("ts_q35_tile_to_h(",self.candidate)
        self.assertIn("static_assert(sizeof(TsBenchHalfBanks) == 67584",self.generated)
        self.assertEqual(self.candidate.count("__shared__"),1)
        self.assertLess(self.candidate.index("__syncthreads();"),self.candidate.index("load(st ^ 1"))
        for iteration in range(10):
            current=iteration&1;prefetch=current^1
            self.assertNotEqual(current,prefetch)
            if iteration:self.assertEqual(prefetch,(iteration-1)&1)

    def test_optional_variant_only_removes_the_redundant_post_decode_barrier(self):
        optional=build.BASE.block(self.generated,'extern "C" __global__ void __launch_bounds__(256) '+build.NO_SECOND_BARRIER+'(')
        wanted=self.candidate.replace(build.CANDIDATE+"(",build.NO_SECOND_BARRIER+"(").replace("        __syncthreads();   // this tile's K and V are in F16\n","")
        self.assertEqual(optional,wanted)
        self.assertEqual(optional.count("__syncthreads();"),1)
        self.assertIn("ts_q35_cp_wait<0>();\n        __syncthreads();",optional)
        self.assertLess(optional.index("__syncthreads();"),optional.index("load(st ^ 1"))
        self.assertIn("if (!warpLive)\n            continue;",optional)
        self.assertNotIn("tile_to_h",optional)

    def test_predecode_uses_last_sequence_end_before_any_page_lookup(self):
        for symbol in ("ts_bench_decode_plain","ts_bench_decode_reference"):
            code=build.BASE.block(self.generated,'extern "C" __global__ void '+symbol+'(')
            self.assertLess(code.index("if (key >= validKeys) return;"),code.index("table[(long long)seq"))
            self.assertIn("keyEnd[(long long)seq * rowsPerSeq + rowsPerSeq - 1]",code)
            self.assertIn("(((long long)seq * 2 + isV) * nKv + head)",code)

    def test_runner_pipeline_and_exact_timed_gates_include_predecode_every_call(self):
        text=(HERE/"bench.cu").read_text()
        issue=text[text.index("    void Issue(Variant& v)"):text.index("    Result Read(")]
        self.assertIn("Decode(v,decoded,false);",issue)
        self.assertLess(issue.index("Decode(v,decoded,false)"),issue.rindex("cuLaunchKernel(v.attention"))
        self.assertIn("if(sp>1)",issue)
        self.assertIn("f.Reset();float ms=Time(f,v,3);Equal(oracle,f.Read(output)",text)
        self.assertEqual(text.count("f.VerifyDecode(*variants[1],output)"),2)
        self.assertIn("CompareHalves(first,second,264,0x3c3c,0xc3c3",text)
        self.assertIn("for(unsigned char color:{(unsigned char)0xa5,(unsigned char)0x7e})",text)
        self.assertIn("validEnds[seq]=seq==0?std::max(0,keys-7):keys",text)
        self.assertIn("VerifyDecodePass(v,output,validEnds,alternate.base,alternateTable.base)",text)
        self.assertIn("limitedPages[(size_t)seq*blocks+b]=nullptr",text)
        self.assertIn("void* ep=alternateEnds?alternateEnds:ends.base",text)
        self.assertIn("if(!std::isfinite(ms)||ms<=0)",text)
        self.assertIn("reg",text)
        self.assertNotIn("regs<=128",text)
        self.assertNotIn("active>=2",text)

    def test_sealed_random_inputs_and_shape_plan_are_unchanged(self):
        old=(build.FROZEN/"bench.cu").read_text();new=(HERE/"bench.cu").read_text()
        start="        std::mt19937 rng";end="        Rt(cudaMemcpy(cache.base"
        old_body=old[old.index(start):old.index(end)]
        new_body=new[new.index(start):new.index(end)]
        # Only add explicit validity checks to the identical generated fixture.
        new_body=new_body[:new_body.index("        // Predecode's last-key bound")]
        self.assertEqual(old_body,new_body)
        start="    std::vector<Shape> shapes;";end="    int strict="
        self.assertEqual(old[old.index(start):old.index(end)],new[new.index(start):new.index(end)])


class GeometryTests(unittest.TestCase):
    def test_every_valid_half_is_written_once_padding_and_unused_rows_never_written(self):
        for keys in (1,31,32,33,255,256,257):
            seqs,heads=2,3;ends=(max(0,keys-7),keys)
            counts=bytearray(seqs*2*heads*keys*256)
            for seq in range(seqs):
                for plane in range(2):
                    for h in range(heads):
                        for first in range(0,keys,32):
                            self.assertLessEqual(first%256,224)
                            for tid in range(256):
                                key=first+(tid&31);b=tid>>5
                                if key>=ends[seq]:continue
                                row=(((seq*2+plane)*heads+h)*keys+key)
                                for d in range(b*32,b*32+32):counts[row*256+d]+=1
                                for stride in (256,264):
                                    for c in range(4):self.assertEqual((256+row*stride*2+b*64+c*16)%16,0)
            for seq in range(seqs):
                for plane in range(2):
                    for h in range(heads):
                        for key in range(keys):
                            row=(((seq*2+plane)*heads+h)*keys+key)
                            self.assertEqual(set(counts[row*256:(row+1)*256]),{1 if key<ends[seq] else 0})

    def test_cp_async_vectors_cover_data_once_skip_padding_and_banks_never_alias(self):
        destinations=set()
        for stage in (0,1):
            for i in range(2048):
                plane=i//1024;c=i%1024;row=c//32;col=(c%32)*8
                destination=((stage*2+plane)*32+row)*264+col
                self.assertEqual(destination*2%16,0)
                for d in range(8):
                    self.assertNotIn(destination+d,destinations);destinations.add(destination+d)
        self.assertEqual(len(destinations),2*2*32*256)
        self.assertLess(max(destinations)*2,67584)
        for key_hi in (1,31,32,33,255,257):
            first=(key_hi-1)//32*32
            for row in range(32):
                key=first+row
                address_key=key if key<key_hi else 0
                self.assertGreaterEqual(address_key,0);self.assertLess(address_key,key_hi)

    def test_workspace_bytes_count_current_layer_and_use_wide_arithmetic(self):
        self.assertEqual(build.workspace_bytes(131072),512*1024**2)
        self.assertEqual(build.workspace_bytes(250000),1024000000)
        self.assertEqual(build.workspace_bytes(256000),1048576000)
        self.assertEqual(build.workspace_bytes(262144,sequences=2),2**31)
        self.assertEqual(build.workspace_bytes(131072,stride=264),528*1024**2)
        for bad in (0,-1,True):
            with self.assertRaises(ValueError):build.workspace_bytes(bad)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        parent=build.ROOT/"artifacts/qwen38-regression/predecoded-attention-cpu-tests";parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def invoke(self,out,*extra):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):return build.main(["--out",str(out),*extra])
    @staticmethod
    def compiler(fail=""):
        def fake(command,**kwargs):
            if command[-1]=="--version":
                if fail=="version":raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0,"CPU_MOCK_VERSION_NOT_CUDA","")
            target=Path(command[command.index("-o")+1])
            if(fail=="cubin" and "-cubin"in command)or(fail=="runner"and target.name==build.RUNNER):return subprocess.CompletedProcess(command,1,"","injected failure")
            if fail!="missing":target.write_bytes(b"CPU_MOCK_OUTPUT_NOT_NATIVE_EVIDENCE")
            return subprocess.CompletedProcess(command,0,"",resource_log()if"-cubin"in command else"")
        return fake
    def test_resources_are_complete_facts_not_old_occupancy_or_spill_filter(self):
        data=build.parse_resources(resource_log())
        self.assertEqual(data[build.CANDIDATE]["registers_per_thread"],254)
        self.assertEqual(data[build.CANDIDATE]["stack_bytes"],24)
        self.assertEqual(data[build.CANDIDATE]["dynamic_shared_bytes"],67584)
        for log in ("",resource_log()+resource_log(),resource_log().replace("Used 254 registers","missing")):
            with self.assertRaises(ValueError):build.parse_resources(log)
    def test_generate_only_does_not_invoke_a_compiler_or_publish_marker(self):
        out=self.root/"generated"
        with patch.object(build.subprocess,"run",side_effect=AssertionError("unexpected compiler")):
            self.assertEqual(self.invoke(out,"--generate-only"),0)
        m=json.loads((out/"manifest.json").read_text());self.assertFalse(m["complete"])
        self.assertEqual(m["cuda_compilation"],"NOT_RUN");self.assertFalse((out/build.PERMIT).exists())
    def test_old_evidence_and_hidden_flags_are_not_overwritten_or_compiled(self):
        out=self.root/"old";out.mkdir();(out/"manifest.json").write_text("retain old failure")
        with self.assertRaises(SystemExit):self.invoke(out,"--generate-only")
        self.assertEqual((out/"manifest.json").read_text(),"retain old failure")
        for flag in("NVCC_PREPEND_FLAGS","NVCC_APPEND_FLAGS"):
            destination=self.root/flag
            with patch.dict(os.environ,{flag:"--use_fast_math"}),self.assertRaises(SystemExit):self.invoke(destination,"--generate-only")
            self.assertFalse(destination.exists())
    def test_build_marker_only_after_complete_native_and_runner_mock_logic(self):
        out=self.root/"mocked"
        fake=self.compiler()
        def observe(command,**kwargs):
            self.assertFalse((out/build.PERMIT).exists());return fake(command,**kwargs)
        with patch.object(build.subprocess,"run",side_effect=observe):self.assertEqual(self.invoke(out),0)
        self.assertTrue((out/build.PERMIT).exists());run.verify_build(out)
        (out/"kernels.cubin").write_bytes(b"changed")
        with self.assertRaises(ValueError):run.verify_build(out)
    def test_failed_builds_never_authorize(self):
        for fail in("version","cubin","runner","missing"):
            out=self.root/fail
            with patch.object(build.subprocess,"run",side_effect=self.compiler(fail)):self.assertEqual(self.invoke(out),1)
            self.assertFalse((out/build.PERMIT).exists())
    def test_foreign_marker_and_incomplete_shapes_never_pass(self):
        (self.root/"manifest.json").write_text(json.dumps(dict(schema="q35-attention-compiler-ablation/1",complete=True)))
        (self.root/build.PERMIT).write_text("{}")
        with self.assertRaises(ValueError):run.verify_build(self.root)
        result=dict(complete=True,strict_shapes=81,strict_candidate_comparisons=243,decoded_fixture_checks=87,decoded_boundary_checks=73,no_second_barrier_requested=False,
                    active_variants=[0,1],device_resource_rejected_variants=[],long_cases=True,timed=True)
        self.assertTrue(run.validated_result(result,"--bench",True))
        for key,value in(("decoded_fixture_checks",81),("decoded_boundary_checks",72),("strict_shapes",79),("strict_candidate_comparisons",81),("active_variants",[0])):
            self.assertFalse(run.validated_result(dict(result,**{key:value}),"--bench",True))
        resource=dict(complete=True,validation="resources_only_not_quality",active_variants=[0,1],device_resource_rejected_variants=[],no_second_barrier_requested=False)
        self.assertTrue(run.validated_result(resource,"--resources",False))
        for change in(dict(active_variants=[0]),dict(device_resource_rejected_variants=[1]),dict(validation="strict")):
            self.assertFalse(run.validated_result(dict(resource,**change),"--resources",False))
        optional=dict(result,active_variants=[0,1,2],strict_candidate_comparisons=486,no_second_barrier_requested=True)
        self.assertTrue(run.validated_result(optional,"--bench",True,True))
        self.assertFalse(run.validated_result(optional,"--bench",True,False))
        self.assertFalse(run.validated_result(result,"--bench",True,True))
        self.assertFalse(run.validated_result(dict(optional,strict_candidate_comparisons=243),"--bench",True,True))


class ActualHostCheckerTests(unittest.TestCase):
    def test_actual_cpp_checker_accepts_all_half_payloads_and_rejects_common_missing_writes(self):
        compiler=shutil.which("c++")or shutil.which("clang++")
        if not compiler:self.skipTest("host C++ compiler unavailable")
        parent=build.ROOT/"artifacts/qwen38-regression/predecoded-attention-cpu-tests";parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent)as temp:
            directory=Path(temp);source=directory/"check.cpp";binary=directory/"check"
            source.write_text(r'''#include "half_checks.h"
#include <cassert>
int main(){
  const int keys=128;const size_t rows=2*keys;std::vector<int> end{keys};
  std::vector<uint16_t> reference(rows*264,0x3c3c),a(rows*256,0xa5a5),b(rows*256,0x7e7e),ref2(rows*264,0xc3c3);
  for(size_t row=0;row<rows;row++)for(int d=0;d<256;d++){
    uint16_t value=uint16_t(row*256+d);reference[row*264+d]=ref2[row*264+d]=a[row*256+d]=b[row*256+d]=value;
  }
  // Every uint16 payload, including all NaNs and every poison value, is legal.
  assert(!FindDecodedHalfMismatch(reference,a,1,1,keys,end,256,0x3c3c,0xa5a5));
  assert(!FindDecodedHalfMismatch(reference,b,1,1,keys,end,256,0x3c3c,0x7e7e));
  assert(!FindDecodedHalfMismatch(reference,ref2,1,1,keys,end,264,0x3c3c,0xc3c3));
  for(size_t at:{size_t(0),size_t(65535)}){a[at]^=1;assert(FindDecodedHalfMismatch(reference,a,1,1,keys,end,256,0x3c3c,0xa5a5));a[at]^=1;}
  // Both kernels omit the same valid cell: different poisons cannot pass.
  auto old=reference[0];reference[0]=0x3c3c;a[0]=0xa5a5;
  assert(FindDecodedHalfMismatch(reference,a,1,1,keys,end,256,0x3c3c,0xa5a5));reference[0]=old;a[0]=old;
  reference[256]=0;assert(FindDecodedHalfMismatch(reference,a,1,1,keys,end,256,0x3c3c,0xa5a5));reference[256]=0x3c3c;
  std::vector<uint16_t> tailRef(2*7*264,0x3c3c),tail(2*7*256,0xa5a5);std::vector<int> none{0};
  assert(!FindDecodedHalfMismatch(tailRef,tail,1,1,7,none,256,0x3c3c,0xa5a5));tail.back()=0;
  assert(FindDecodedHalfMismatch(tailRef,tail,1,1,7,none,256,0x3c3c,0xa5a5));
  bool caught=false;try{FindDecodedHalfMismatch({},a,1,1,keys,end,256,0,0);}catch(const std::invalid_argument&){caught=true;}assert(caught);
  caught=false;try{FindDecodedHalfMismatch(reference,a,1,1,keys,{keys+1},256,0,0);}catch(const std::invalid_argument&){caught=true;}assert(caught);
}''')
            r=subprocess.run([compiler,"-std=c++17","-O2","-I",str(HERE),str(source),"-o",str(binary)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            r=subprocess.run([str(binary)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=="__main__":unittest.main()
