#!/usr/bin/env python3
"""CPU-only generation/resource gates. CUDA compilation and execution are separate."""
import math
import unittest
import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from unittest import mock
import build


class SourceTests(unittest.TestCase):
    def test_oracle_body_is_current_production_body_except_private_name(self):
        source = build.SOURCE.read_text()
        expected = build.block(source, 'extern "C" __global__ void __launch_bounds__(256) ts_q35_attn_prefill_f32(')
        expected = expected.replace("ts_q35_attn_prefill_f32(", "ts_bench_attn_prefill(", 1)
        actual = build.block(build.generate(source, 0), 'extern "C" __global__ void __launch_bounds__(256) ts_bench_attn_prefill(')
        self.assertEqual(expected, actual)

    def test_single_bank_preserves_entire_arithmetic_suffix(self):
        source = build.SOURCE.read_text()
        a, b = (build.generate(source, v) for v in (0, 1))
        marker = "        if (!warpLive)\n            continue;"
        self.assertEqual(a[a.index(marker):], b[b.index(marker):])
        self.assertEqual(1, b.count("Every raw reader is finished"))

    def test_all_variants_have_exactly_one_dynamic_shared_object(self):
        source = build.SOURCE.read_text()
        for v, size in enumerate((68608, 51200, 68608, 51200, 50176)):
            generated = build.generate(source, v)
            self.assertEqual(1, generated.count("__shared__"))
            self.assertIn(f"sizeof(TsQ35AttnPrefill)=={size}", generated)

    def test_row_mapping_keeps_every_original_row_and_mma_half(self):
        for g in (1, 2, 3, 6, 8, 16, 64):
            for rows in (1, 7, 11, 21, 22, 23, 63, 64, 65, 127, 128, 129, 257):
                count = rows*g
                actual=[]
                for tile in range((count+127)//128):
                    for sub in range(2):
                        for warp in range(8):
                            for lane_group in range(8):
                                row=tile*128+sub*64+warp*8+lane_group
                                if row>=count: continue
                                self.assertEqual((row%16)//8,warp&1)
                                actual.append(row)
                self.assertEqual(list(range(count)),actual)

    def test_key_partitions_keep_original_128_row_bound_even_in_ragged_subtiles(self):
        for prefix in (0, 31, 255, 8191, 131072, 250000):
            for g in (1, 3, 6, 64):
                for rows in (1, 11, 22, 43, 129):
                    count=rows*g
                    ends=[prefix+t+1 for t in range(rows)]
                    for tile in range((count+127)//128):
                        end=ends[(min(tile*128+128,count)-1)//g]
                        for splits in range(1,9):
                            share=((end+splits-1)//splits+31)//32*32
                            intervals=[(min(end,s*share),min(end,(s+1)*share)) for s in range(splits)]
                            self.assertEqual(0,intervals[0][0]);self.assertEqual(end,intervals[-1][1])
                            self.assertEqual([hi for lo,hi in intervals[:-1]],[lo for lo,hi in intervals[1:]])
                            for sub in range(2):
                                row=tile*128+sub*64
                                if row>=count:continue
                                reference_end=ends[(min((row//128+1)*128,count)-1)//g]
                                reference_width=math.ceil(reference_end/(splits*32))*32
                                self.assertEqual(reference_width,share)
                                self.assertGreaterEqual(end,ends[row//g])

    def test_changed_production_arithmetic_constants_require_review(self):
        source=build.SOURCE.read_text()
        with self.assertRaises(ValueError):build.generate(source.replace("#define TS_Q35_AH_CHAIN 1", "#define TS_Q35_AH_CHAIN 2"),0)

    def test_buddy_vote_matches_original_warp_for_every_query_update_pattern(self):
        # Original warp has16queries, each four-lane group shares one c value.
        for mask in range(1<<16):
            lower=all(not(mask&(1<<i)) for i in range(8))
            upper=all(not(mask&(1<<i)) for i in range(8,16))
            original=all(not(mask&(1<<i)) for i in range(16))
            self.assertEqual(original,lower and upper)

    def test_vote_storage_occupies_only_never_read_k_padding(self):
        locations=[]
        for warp in range(8):
            byte=(warp//4)*264*2+256*2+(warp%4)*4
            self.assertEqual(0,byte%4)
            for b in range(byte,byte+4):
                self.assertGreaterEqual(b%(264*2),256*2)
                locations.append(b)
        self.assertEqual(32,len(set(locations)))
        # Every ldmatrix source lane supplies one16-byte row. Check both source
        # mappings against padding, including the largestKstep/dimension tile.
        for lane in range(32):
            for np in range(2):
                for kk in range(16):
                    key=(lane&7)+((lane>>4)<<3)+np*16
                    col=((lane>>3)&1)*8+kk*16
                    self.assertLess(key,32);self.assertLessEqual(col+8,256)
            for kk in range(2):
                for dn in range(16):
                    key=(lane&7)+(((lane>>3)&1)<<3)+kk*16
                    col=(lane>>4)*8+dn*16
                    self.assertLess(key,32);self.assertLessEqual(col+8,256)


class ResourceTests(unittest.TestCase):
    def record(self, regs=128, stack=0, stores=0, loads=0, smem=0):
        return f"ptxas info : Compiling entry function 'ts_bench_attn_prefill' for 'sm_120a'\nptxas info : Function properties for ts_bench_attn_prefill\n {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\nptxas info : Used {regs} registers, {smem} bytes smem\n"

    def test_only_qualified_64_single_bank_may_enter_gpu_gate(self):
        self.assertTrue(build.parse_resources(self.record(),3)["gpu_gate_permitted"])
        self.assertFalse(build.parse_resources(self.record(),2)["gpu_gate_permitted"])
        for kwargs in ({"regs":129},{"stack":16},{"stores":4},{"loads":4},{"smem":16}):
            self.assertFalse(build.parse_resources(self.record(**kwargs),3)["gpu_gate_permitted"],kwargs)

    def test_ordinary_variants_do_not_claim_two_cta_resources(self):
        result=build.parse_resources(self.record(regs=254),0)
        self.assertTrue(result["gpu_gate_permitted"])
        self.assertFalse(result["two_cta_resource_gate"])

    def test_compact_variant_still_requires_register_and_spill_gate(self):
        result=build.parse_resources(self.record(),4)
        self.assertEqual(50176,result["dynamic_shared_bytes"])
        self.assertTrue(result["gpu_gate_permitted"])
        self.assertFalse(build.parse_resources(self.record(regs=129),4)["gpu_gate_permitted"])
        self.assertFalse(build.parse_resources(self.record(stores=4),4)["gpu_gate_permitted"])

    def test_incomplete_resource_output_is_not_treated_as_a_pass(self):
        for text in ("", "ptxas info : Compiling entry function 'other'\n", "ptxas info : Compiling entry function 'ts_bench_attn_prefill'\nUsed128 registers"):
            with self.assertRaises(ValueError): build.parse_resources(text,3)


class BuildPublicationTests(unittest.TestCase):
    def run_fake_build(self, directory, runner_ok=True, cube_ok=True, compiler_missing=False):
        output=pathlib.Path(directory);(output/"qualified-variants.txt").write_text("0\n1\n")
        (output/"attention-bench").write_bytes(b"old runner must never authorize a failed build")
        logs=ResourceTests().record(regs=254)
        def run(command,**kwargs):
            if command[0]=="git":return subprocess.CompletedProcess(command,0,b"",b"")
            self.assertFalse((output/"qualified-variants.txt").exists())
            if compiler_missing:raise FileNotFoundError("missing test compiler")
            if "-cubin" in command:
                if cube_ok:pathlib.Path(command[command.index("-o")+1]).write_bytes(b"fake cubin for publication test only")
                return subprocess.CompletedProcess(command,0 if cube_ok else 1,"",logs if cube_ok else "fixture compiler failure")
            if runner_ok:pathlib.Path(command[command.index("-o")+1]).write_bytes(b"fake new runner for publication test only")
            return subprocess.CompletedProcess(command,0 if runner_ok else 1,"","" if runner_ok else "fixture runner failure")
        argv=["build.py","--out",str(output),"--nvcc","publication-test-only","--variants","0,1"]
        with mock.patch.object(sys,"argv",argv),mock.patch.object(build.subprocess,"run",side_effect=run),contextlib.redirect_stdout(io.StringIO()):
            result=build.main()
        return result,json.loads((output/"manifest.json").read_text())

    def test_failed_runner_cannot_publish_authorization_for_a_stale_binary(self):
        with tempfile.TemporaryDirectory() as d:
            result,manifest=self.run_fake_build(d,runner_ok=False)
            self.assertEqual(1,result);self.assertFalse(manifest["complete"])
            self.assertFalse((pathlib.Path(d)/"qualified-variants.txt").exists())

    def test_failed_or_missing_compiler_cannot_publish_authorization(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as d:
                result,manifest=self.run_fake_build(d,cube_ok=False,compiler_missing=missing)
                self.assertEqual(1,result);self.assertFalse(manifest["complete"])
                self.assertFalse((pathlib.Path(d)/"qualified-variants.txt").exists())
                if missing:self.assertEqual("compiler_unavailable_not_run",manifest["variants"][0]["status"])

    def test_authorization_is_published_only_after_every_binary_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            result,manifest=self.run_fake_build(d)
            self.assertEqual(0,result);self.assertTrue(manifest["complete"])
            self.assertEqual("0\n1\n",(pathlib.Path(d)/"qualified-variants.txt").read_text())


class ActualHostBufferCheckTests(unittest.TestCase):
    def test_gpu_runner_write_coverage_checker_rejects_shared_missing_writes(self):
        compiler=shutil.which("clang++") or shutil.which("c++")
        if compiler is None:self.skipTest("host C++ compiler unavailable; CUDA is not used")
        source=r'''
#include "checks.h"
int main() {
    const uint32_t c=0x4f123456;const size_t n=6,g=2;
    std::vector<uint32_t> empty(n+2*g,c),good=empty;
    const uint32_t bits[]={0,0x80000000,0x7fc00001,0x7f800000,0xff800000,1};
    for(size_t i=0;i<n;i++)good[g+i]=bits[i];
    auto rejects=[&](const std::vector<uint32_t>& v,bool written){try{CheckGuardedWords(v,n,g,c,written,"test");return false;}catch(const std::exception&){return true;}};
    CheckGuardedWords(good,n,g,c,true,"valid");CheckGuardedWords(empty,n,g,c,false,"unused partial");
    if(!rejects(empty,true))return 1;auto missing=good;missing[g+3]=c;if(!rejects(missing,true))return 2;
    auto front=good;front[0]=0;if(!rejects(front,true))return 3;auto back=good;back.back()=0;if(!rejects(back,true))return 4;
    auto unexpected=empty;unexpected[g]=0;if(!rejects(unexpected,false))return 5;
    auto wrong=good;wrong.pop_back();if(!rejects(wrong,true))return 6;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/"host.cpp").write_text(source)
            command=[compiler,"-std=c++17","-Wall","-Wextra","-Werror","-I",str(build.HERE),str(p/"host.cpp"),"-o",str(p/"host-check")]
            compiled=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(0,compiled.returncode,compiled.stdout+compiled.stderr)
            executed=subprocess.run([str(p/"host-check")],capture_output=True,text=True)
            self.assertEqual(0,executed.returncode,executed.stdout+executed.stderr)


class CompactSwizzleTests(unittest.TestCase):
    @staticmethod
    def physical(key,col):return key*256+(col^((key&7)*8))

    def store(self,epoch=0):
        memory=[None]*(32*256)
        for tid in range(256):
            key,b=tid&31,tid>>5
            for c in range(2):
                for vector in range(2):
                    col=b*32+c*16+vector*8
                    at=self.physical(key,col)
                    self.assertEqual(0,at%8)
                    for i in range(8):
                        self.assertIsNone(memory[at+i])
                        memory[at+i]=(key*256+col+i)^(epoch<<13)
        self.assertNotIn(None,memory)
        return memory

    def test_uint4_stores_are_bijective_and_keep_every_half_bit_payload(self):
        # Eight epochs exhaust all65536 possible16-bit half payloads, including
        # signedzeros, subnormals, infinities and NaN payloads. Conversion is not
        # reimplemented; this proves the store/load permutation is bit-neutral.
        seen=set()
        for epoch in range(8):
            memory=self.store(epoch)
            for key in range(32):
                for col in range(256):
                    value=memory[self.physical(key,col)]
                    self.assertEqual((key*256+col)^(epoch<<13),value)
                    seen.add(value)
        self.assertEqual(set(range(65536)),seen)

    def test_every_qk_and_transposed_pv_lane_vector_recovers_original_values(self):
        memory=self.store()
        for lane in range(32):
            for np in range(2):
                for kk in range(16):
                    key=(lane&7)+((lane>>4)<<3)+np*16
                    col=((lane>>3)&1)*8+kk*16
                    at=self.physical(key,col)
                    self.assertEqual(list(range(key*256+col,key*256+col+8)),memory[at:at+8])
            for kk in range(2):
                for dn in range(16):
                    key=(lane&7)+(((lane>>3)&1)<<3)+kk*16
                    col=(lane>>4)*8+dn*16
                    at=self.physical(key,col)
                    self.assertEqual(list(range(key*256+col,key*256+col+8)),memory[at:at+8])

    def banks(self,addresses):
        banks=[]
        for key,col in addresses:
            start=self.physical(key,col)*2//4
            banks.extend((start+word)%32 for word in range(4))
        self.assertEqual(set(range(32)),set(banks))
        self.assertEqual(32,len(banks))

    def test_each_8_row_ldmatrix_address_group_covers_all_32_word_banks(self):
        for first in range(0,32,8):
            for np in range(2):
                for kk in range(16):
                    self.banks([((lane&7)+((lane>>4)<<3)+np*16,((lane>>3)&1)*8+kk*16) for lane in range(first,first+8)])
            for kk in range(2):
                for dn in range(16):
                    self.banks([((lane&7)+(((lane>>3)&1)<<3)+kk*16,(lane>>4)*8+dn*16) for lane in range(first,first+8)])

    def test_each_8_thread_uint4_store_group_covers_all_32_word_banks(self):
        for first in range(0,256,8):
            for c in range(2):
                for vector in range(2):
                    self.banks([(tid&31,(tid>>5)*32+c*16+vector*8) for tid in range(first,first+8)])

    def test_dequant_conversion_calls_and_scale_load_are_the_original_operations(self):
        import re
        original=build.block(build.SOURCE.read_text(),"__device__ __forceinline__ void ts_q35_tile_to_h(")
        compact=build.block((build.HERE/"attention64.cuh").read_text(),"__device__ __forceinline__ void ts_bench_tile_to_h_swizzled(")
        conversion=lambda x:re.findall(r"ts_q35_i8x4_to_h2\([^;]+;",x)
        self.assertEqual(conversion(original),conversion(compact))
        scale="const half2 d = __half2half2(*(const half*)(row + TS_Q35_Q8_SCALES + 2 * b));"
        self.assertIn(scale,original);self.assertIn(scale,compact)


if __name__=="__main__":unittest.main()
