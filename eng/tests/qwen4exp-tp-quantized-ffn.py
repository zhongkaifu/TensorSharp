#!/usr/bin/env python3
"""Tensor-sliced Qwen FFN quantized-kernel probe with independent FP64 arithmetic.

Uses the real intermediate width640 by default. Slices bytes
of the once-quantized source, never requantizes shards. Reports both native
unsharded-versus-TP error and each path's error against dequantized FP64 weights.
The FP64 oracle does not model CUDA activation quantization; its errors are
reported separately, never used to excuse a sharding discrepancy.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np

spec = importlib.util.spec_from_file_location("op", Path(__file__).with_name("qwen4exp-mtp-operator.py"))
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
P,I,L,F = C.c_void_p,C.c_int,C.c_int64,C.c_float
_library = None
QUANT_TYPES = {8:"Q8_0",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",
    16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS"}
DEFAULT_WIDTHS = (1,2,3,4,5,6,7,8,17,31,128)


def main():
    global _library
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--backend", choices=("CPU","CUDA"), default="CPU")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ff", type=int, default=640,
        help="Intermediate width; use a multiple of256 for K/IQ down weights")
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8,17,31,128", help="Comma-separated batch widths, including every supported verify width")
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--require-bitwise", action="store_true",
        help="Fail on any F32 bit difference between the unsharded and TP outputs")
    ap.add_argument("--gate-type", type=int, choices=tuple(QUANT_TYPES), default=23,
        help="Routed gate/up ggml type; 17 IQ2_XS and18 IQ3_XXS match Qwen3.8-Flash-Next UD-Q2_K_XL")
    ap.add_argument("--down-type", type=int, choices=tuple(QUANT_TYPES), default=8,
        help="Routed down ggml type; 20 IQ4_NL matches Qwen3.8-Flash-Next UD-Q2_K_XL")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--used", type=int, default=4)
    ap.add_argument("--shared-q8", action="store_true", help="Use Q8_0 for all shared FFN matrices")
    ap.add_argument("--shared-gate-type", type=int, choices=tuple(QUANT_TYPES), default=None,
        help="Shared gate/up ggml type; 13 Q5_K and14 Q6_K match Qwen3.8-Flash-Next UD-Q2_K_XL")
    ap.add_argument("--shared-down-type", type=int, choices=tuple(QUANT_TYPES), default=8)
    ap.add_argument("--ggml-dir", type=Path, default=Path(__file__).resolve().parents[2]/"ExternalProjects/ggml")
    args = ap.parse_args()
    degree = args.degree
    H,HC,LOW,FF,EXPERTS,USED = args.hidden,4,16,args.ff,args.experts,args.used
    if EXPERTS < 1 or USED < 1 or USED > EXPERTS:
        ap.error("Use 1 <= --used <= --experts")
    if degree < 2 or H <= 0 or FF <= 0 or H%degree or FF%degree:
        ap.error("Positive --hidden and --ff must be divisible by --degree >=2")
    widths = [int(value) for value in args.tokens.split(',')]
    if not widths or min(widths) < 1:
        ap.error("Use positive --tokens widths")
    devices = [int(value) for value in args.devices.split(',')]
    if args.backend == "CUDA" and len(devices) != degree:
        ap.error("--devices must contain exactly --degree device IDs")
    shared_gate_type = 8 if args.shared_q8 else args.shared_gate_type if args.shared_gate_type is not None else 23
    shared_down_type = 8 if args.shared_q8 else args.shared_down_type
    lib = C.CDLL(str(args.library.resolve()))
    _library = lib
    err=lib.TSGgml_GetLastError; err.restype=C.c_char_p
    if args.backend == "CPU":
        init=lib.TSGgml_TensorParallelInitLoopback;init.argtypes=[I,I];init.restype=I
        assert init(2,degree),err()
    else:
        init=lib.TSGgml_TensorParallelInit;init.argtypes=[I,P,I,I];init.restype=I
        assert init(3,(I*degree)(*devices),degree,0),err()
        if not args.float:
            capability=lib.TSGgml_Qwen4ExpTpWeightSupported
            capability.argtypes=[I,I,I,I,I,I];capability.restype=I
            assert capability(args.gate_type,H,FF,EXPERTS,degree,1),err()
            assert capability(args.down_type,FF,H,EXPERTS,degree,0),err()
            assert capability(shared_gate_type,H,FF,1,degree,1),err()
            assert capability(shared_down_type,FF,H,1,degree,0),err()
            assert not capability(29,H,FF,EXPERTS,degree,1), "Unqualified IQ1_M must refuse before data loading"
            assert not capability(8,H,641,EXPERTS,degree,1), "Nondivisible row tail must refuse before data loading"
    quant=lib.ggml_quantize_chunk;quant.argtypes=[I,P,P,L,L,L,P];quant.restype=C.c_size_t
    quant_init=lib.ggml_quantize_init;quant_init.argtypes=[I];quant_init.restype=None
    requires_importance=lib.ggml_quantize_requires_imatrix;requires_importance.argtypes=[I];requires_importance.restype=C.c_bool
    block_size=lib.ggml_blck_size;block_size.argtypes=[I];block_size.restype=L
    row_size=lib.ggml_row_size;row_size.argtypes=[I,L];row_size.restype=C.c_size_t
    dequant=lib.TSGgml_DequantizeToF32;dequant.argtypes=[I,P,L,P];dequant.restype=I
    signature=[P,P,P,P,I,I,P,P]+[I]*16+[F]*3+[I]*4+[F,I,P,P,P,I,P,P,P,I,I]
    normal=lib.TSGgml_Qwen4ExpTokenSpan;normal.argtypes=signature+[P,I,P,P,I];normal.restype=I
    tp=lib.TSGgml_Qwen4ExpTokenSpanTp;tp.argtypes=normal.argtypes+[C.POINTER(P)];tp.restype=I
    execute=lib.TSGgml_TensorParallelExecutePlans;execute.argtypes=[P,I];execute.restype=I
    release=lib.TSGgml_Qwen4ExpReleaseAllSeqState;release.argtypes=[];release.restype=None
    rng=np.random.default_rng(93017)
    desc=op.Ffn();pinned=[];weights={};raw={};types={}
    matrices={"hc_down":(LOW,HC*H),"hc_up":(HC*H,LOW),"hc_inject":(HC,HC*H),
        "router":(EXPERTS,H),"gate_exps":(EXPERTS,FF,H),"up_exps":(EXPERTS,FF,H),
        "down_exps":(EXPERTS,H,FF),"sh_gate":(FF,H),"sh_up":(FF,H),"sh_down":(H,FF)}
    for field,shape in matrices.items():
        value=np.asarray(rng.normal(0,1/np.sqrt(shape[-1]),shape),np.float32)
        kind=0 if args.float else (args.gate_type if field in ("gate_exps","up_exps")
            else shared_gate_type if field in ("sh_gate","sh_up")
            else args.down_type if field == "down_exps"
            else shared_down_type if field == "sh_down" else 0)
        if shape[-1]%int(block_size(kind)):
            ap.error(f"{field} input width {shape[-1]} is not divisible by the {QUANT_TYPES.get(kind,'F32')} block size")
        size=int(row_size(kind,shape[-1])); packed=np.empty((*shape[:-1],size),np.uint8)
        quant_init(kind)
        importance=np.ones(shape[-1],np.float32) if requires_importance(kind) else None
        assert quant(kind,value.ctypes.data,packed.ctypes.data,0,value.size//shape[-1],shape[-1],
            importance.ctypes.data if importance is not None else None)==packed.nbytes
        decoded=np.empty_like(value);assert dequant(kind,packed.ctypes.data,value.size,decoded.ctypes.data)==0
        weights[field]=decoded.astype(np.float64);raw[field]=packed;types[field]=kind;pinned.append(packed)
        setattr(desc,field,packed.ctypes.data);setattr(desc,field+'_bytes',packed.nbytes);setattr(desc,field+'_type',kind)
    for field,shape in (("hc_norm",(HC,H)),("sh_gate_inp",(H,))):
        value=np.ones(shape,np.float32) if field=="hc_norm" else np.asarray(rng.normal(0,1/np.sqrt(H),shape),np.float32)
        weights[field]=value.astype(np.float64);pinned.append(value);setattr(desc,field,value.ctypes.data)
    full=(op.Ffn*1)(desc); shards=[];shard_weights=[]
    for rank in range(degree):
        sd=op.Ffn.from_buffer_copy(desc);sw={}
        for field,axis in (("gate_exps",1),("up_exps",1),("down_exps",1),("sh_gate",0),("sh_up",0),("sh_down",0)):
            rows=raw[field].shape[axis];first=rank*rows//degree;end=(rank+1)*rows//degree
            # The production loader retains complete MMQ output tiles around
            # each logical gate/up slice; native execution crops the overlap.
            if not args.float and field in ("gate_exps","up_exps","sh_gate","sh_up"):
                first=first//128*128;end=(end+127)//128*128
            selection=[slice(None)]*raw[field].ndim;selection[axis]=slice(first,end)
            value=np.ascontiguousarray(raw[field][tuple(selection)]);pinned.append(value)
            setattr(sd,field,value.ctypes.data);setattr(sd,field+'_bytes',value.nbytes)
            sw[field]=np.split(weights[field],degree,axis=axis)[rank]
        shards.append((op.Ffn*1)(sd));shard_weights.append(sw)
    # The current public span ABI executes an attention/recurrent half before
    # its FFN. A valid all-zero output projection makes that half an exact
    # identity, keeping the FFN oracle independent of attention arithmetic.
    capacity=max(256,(max(widths)+255)//256*256)
    attention=op.Attn()
    for field in ("hc_norm","hc_down","hc_up","hc_inject"):
        setattr(attention,field,getattr(desc,field))
        if field!="hc_norm":
            setattr(attention,field+'_bytes',getattr(desc,field+'_bytes'))
            setattr(attention,field+'_type',getattr(desc,field+'_type'))
    for field,shape in (("wq",(64,H)),("wk",(16,H)),("wv",(16,H)),("wo",(H,32)),
        ("q_norm",(8,)),("k_norm",(8,)),("k_cache",(2,capacity,8)),("v_cache",(2,capacity,8))):
        value=np.ones(shape,np.float32) if field.endswith("norm") else np.zeros(shape,np.float32)
        pinned.append(value);setattr(attention,field,value.ctypes.data)
        if field in ("wq","wk","wv","wo"):
            setattr(attention,field+'_bytes',value.nbytes);setattr(attention,field+'_type',0)
        elif field=="k_cache":attention.kv_bytes=value.nbytes
    attn=(op.Attn*1)(attention)
    mask=np.zeros((max(widths),capacity),np.float16);pinned.append(mask)
    kinds=(C.c_uint8*1)(0);dummy=C.create_string_buffer(1024)
    sigmoid=lambda x:1/(1+np.exp(-x))
    def oracle(res, partition=False):
        T=len(res);xn=res/np.sqrt(np.mean(res*res,axis=2,keepdims=True)+1e-6)*weights['hc_norm'];xn=xn.reshape(T,HC*H)
        lo=xn@weights['hc_down'].T/HC;lo=lo*sigmoid(lo)
        mixed=(xn*sigmoid(lo@weights['hc_up'].T)).reshape(T,HC,H).mean(axis=1)
        injection=2*sigmoid(xn@weights['hc_inject'].T/HC)
        router=mixed@weights['router'].T; probs=np.exp(router-router.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
        ids=np.argsort(-probs,axis=1)[:,:USED];scores=np.take_along_axis(probs,ids,axis=1);scores/=scores.sum(axis=1,keepdims=True)
        def partial(w):
            routed=np.zeros((T,w['down_exps'].shape[1]))
            for t in range(T):
                for k,e in enumerate(ids[t]):
                    gate=w['gate_exps'][e]@mixed[t];up=w['up_exps'][e]@mixed[t]
                    routed[t]+=scores[t,k]*(w['down_exps'][e]@(gate*sigmoid(gate)*up))
            gate=mixed@w['sh_gate'].T;up=mixed@w['sh_up'].T
            return routed+(gate*sigmoid(gate)*up)@w['sh_down'].T*sigmoid(mixed@weights['sh_gate_inp'])[:,None]
        # The rank's down rows consume the gathered, full-width activation.
        ffn=np.concatenate([partial(dict(weights,down_exps=w['down_exps'],sh_down=w['sh_down']))
            for w in shard_weights],axis=1) if partition else partial(weights)
        return res+injection[:,:,None]*ffn[:,None,:]
    def run(parallel,res):
        T=len(res);out=res.copy();plans=(P*degree)()
        values=[C.addressof(full),C.addressof(dummy),C.addressof(attn),C.addressof(kinds),0,1,out.ctypes.data,mask.ctypes.data,
            H,HC,LOW,T,4,4,1,2,3,8,4,2,capacity,T,0,4,10000.,1.,1.,EXPERTS,USED,FF//degree if parallel else FF,FF//degree if parallel else FF,
            1e-6,3,None,None,None,-1,None,None,None,0,0,None,1,None,None,0]
        assert len(values)==len(normal.argtypes),"Span ABI fixture signature mismatch"
        if parallel:
            for rank in range(degree):
                values[0]=C.addressof(shards[rank]);values[len(signature)-1]=rank;plan=P()
                assert tp(*values,C.byref(plan)),err();plans[rank]=plan
            assert execute(plans,degree),err()
            if os.environ.get("TS_Q4E_NODE_DUMP"):
                dump = lib.TSGgml_Qwen4ExpTestDumpTpPlans
                dump.argtypes = [P,I,I,I,I,I,I,I]; dump.restype = None
                dump(plans,degree,0,1,T,0,1,T)
        else: assert normal(*values),err()
        return out
    def git_output(*command):
        result=subprocess.run(["git","-C",str(args.ggml_dir),*command],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    report=dict(hidden=H,ff=FF,experts=EXPERTS,used=USED,degree=degree,types=types,
        type_names={field:QUANT_TYPES.get(kind,"F32") for field,kind in types.items()},backend=args.backend,
        devices=devices if args.backend=="CUDA" else [],loopback_correctness_only=args.backend=="CPU",
        native_sha256=op.sha(args.library),ggml_revision=git_output("rev-parse","HEAD"),
        ggml_dirty=git_output("status","--porcelain"),unexecuted_default_widths=[T for T in DEFAULT_WIDTHS if T not in widths],
        limitations=["Synthetic direct-ABI FFN fixture; excludes model loading, attention, tokenizer, and output quality",
            "Per-path timings are diagnostics; exclude model throughput and warm-up qualification",
            "Independent FP64 oracle omits backend activation quantization; only plain-versus-TP error is an acceptance gate"],
        gate=dict(require_bitwise=args.require_bitwise,atol=2e-5,rtol=2e-5),checks=[])
    def stats(x,y):
        diff=x.astype(np.float64)-y.astype(np.float64)
        return dict(max_abs=float(np.max(np.abs(diff))),relative_l2=float(np.linalg.norm(diff)/max(1e-30,np.linalg.norm(y))))
    passed=True
    for T in widths:
        residual=np.asarray(rng.normal(0,.4,(T,HC,H)),np.float32)
        start=time.perf_counter();plain=run(False,residual);plain_ms=1000*(time.perf_counter()-start)
        start=time.perf_counter();parallel=run(True,residual);tp_ms=1000*(time.perf_counter()-start)
        bitwise_equal = bool(np.array_equal(plain.view(np.uint32), parallel.view(np.uint32)))
        # Some native BLAS/quantization routines leave FP status flags set.
        # NumPy can report those old flags on a finite matmul; the explicit
        # finiteness checks below remain the authoritative oracle gate.
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            reference=oracle(residual.astype(np.float64));partitioned=oracle(residual.astype(np.float64),True)
        check_passed = bool(all(np.isfinite(x).all() for x in (plain,parallel,reference,partitioned))
            and np.allclose(plain,parallel,atol=2e-5,rtol=2e-5)
            and (not args.require_bitwise or bitwise_equal)
            and np.allclose(reference,partitioned,atol=1e-12,rtol=1e-12))
        passed &= check_passed
        report['checks'].append(dict(tokens=T,passed=check_passed,plain_vs_tp=dict(stats(parallel,plain),bitwise_equal=bitwise_equal),plain_vs_f64=stats(plain,reference),tp_vs_f64=stats(parallel,reference),f64_partition=stats(partitioned,reference),diagnostic_ms=dict(plain=plain_ms,tp=tp_ms)))
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
        print(json.dumps(report['checks'][-1]),flush=True)
    release();report['passed']=passed;args.report.write_text(json.dumps(report,indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    try:
        result = main()
    finally:
        if _library is not None:
            _library.TSGgml_Shutdown.argtypes = []
            _library.TSGgml_Shutdown.restype = None
            _library.TSGgml_Shutdown()
    raise SystemExit(result)
