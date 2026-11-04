// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>TensorSharp-owned numerical adapters for the original NInfer Qwen3.8 target.
/// All public activations use F32 storage with explicit BF16 rounding boundaries.
/// GDN state and AB projection accumulators retain F32. This is a separate format policy,
/// never an implicit fallback for GGUF model execution.</summary>
public sealed unsafe class CudaNinferKernels : IDisposable
{
    public const string EnvironmentVariable = "TS_Q35_NINFER_KERNELS";
    private readonly CudaModule _module;
    private readonly IntPtr _round, _cast, _embed, _norm, _quant, _swiglu, _a16;
    private readonly IntPtr _conv, _convState, _prep, _scan, _replay, _out;
    private IntPtr _blas, _inputBf16;
    private long _inputCapacity;
    private bool _disposed;
    public string ImageSha256 => _module.ImageSha256;

    public CudaNinferKernels(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        _module = CudaModule.LoadFromFile(path, identify: true);
        try
        {
            _round=Function("ts_ni_round"); _cast=Function("ts_ni_f32_bf16");
            _embed=Function("ts_ni_embed_w8"); _norm=Function("ts_ni_add_norm");
            _quant=Function("ts_ni_quant_nvfp4"); _swiglu=Function("ts_ni_swiglu");
            _a16=Function("ts_ni_project_a16"); _conv=Function("ts_ni_gdn_conv");
            _convState=Function("ts_ni_gdn_conv_state"); _prep=Function("ts_ni_gdn_prep");
            _scan=Function("ts_ni_gdn_scan"); _replay=Function("ts_ni_gdn_replay");
            _out=Function("ts_ni_gdn_out");
            CheckBlas(CublasApi.cublasCreate(out _blas));
        }
        catch { _module.Dispose(); throw; }
    }
    public static CudaNinferKernels Create(string path = null)
    {
        path ??= Environment.GetEnvironmentVariable(EnvironmentVariable);
        if(string.IsNullOrWhiteSpace(path))
        {
            foreach(string name in new[]{"tensorsharp_q35_ninfer.fatbin","tensorsharp_q35_ninfer.ptx"})
            {
                foreach(string directory in new[]{AppContext.BaseDirectory,Path.Combine(AppContext.BaseDirectory,"cuda_kernels")})
                {
                    string candidate=Path.Combine(directory,name);
                    if(File.Exists(candidate)) { path=candidate; break; }
                }
                if(path!=null)break;
            }
        }
        if(string.IsNullOrWhiteSpace(path))
            throw new FileNotFoundException("NInfer CUDA adapter module is required. Set "+EnvironmentVariable+" to its fatbin or PTX path.");
        return new(path);
    }
    private IntPtr Function(string name)
    {
        IntPtr f=_module.GetFunction(name);
        CudaDriverApi.cuFuncLoad(f).ThrowOnError();
        return f;
    }
    private static void CheckBlas(int status)
    {
        if(status!=0)throw new InvalidOperationException("NInfer cuBLAS failed with status "+status+".");
    }
    private static uint Grid(long n,int block=256) => checked((uint)((n+block-1)/block));
    private void Launch(IntPtr f,uint x,uint y,uint z,int block,IntPtr stream,void** args)
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        CudaDriverApi.cuLaunchKernel(f,x,y,z,(uint)block,1,1,0,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();
    }
    private static void Shape(int rows,int width)
    {
        if(rows<1||width<1)throw new ArgumentOutOfRangeException(nameof(rows),"Positive dimensions required.");
        _=checked((long)rows*width*sizeof(float));
    }
    public void Round(IntPtr values,long count,IntPtr stream)
    {
        if(count<1)throw new ArgumentOutOfRangeException(nameof(count));
        void** args=stackalloc void*[]{&values,&count}; Launch(_round,Grid(count),1,1,256,stream,args);
    }
    internal void CastToBf16(IntPtr input,IntPtr output,long count,IntPtr stream)
    {
        if(count<1||input==IntPtr.Zero||output==IntPtr.Zero)throw new ArgumentOutOfRangeException(nameof(count));
        _=checked(count*sizeof(float));
        void** args=stackalloc void*[]{&input,&output,&count}; Launch(_cast,Grid(count),1,1,256,stream,args);
    }
    public void EmbedW8(IntPtr weights,IntPtr tokenIds,IntPtr output,int rows,int width,int vocabulary,IntPtr stream)
    {
        Shape(rows,width);
        if(width%128!=0||vocabulary<1)throw new ArgumentOutOfRangeException(nameof(width));
        IntPtr scales=weights+(nint)checked((long)vocabulary*width);
        void** args=stackalloc void*[]{&weights,&scales,&tokenIds,&output,&width,&vocabulary};
        Launch(_embed,(uint)rows,1,1,256,stream,args);
    }
    /// <summary>Mutates x to BF16(x+delta), then writes BF16(RMSNorm(x,effectiveWeight)).
    /// The norm weight already contains the F32 unit offset where the format requires it.</summary>
    public void AddRmsNorm(IntPtr x,IntPtr delta,IntPtr effectiveWeight,IntPtr output,int rows,int width,float eps,IntPtr stream)
    {
        Shape(rows,width);
        if(x==output)throw new ArgumentException("Residual and norm output must not alias.");
        if(!(eps>0)||!float.IsFinite(eps))throw new ArgumentOutOfRangeException(nameof(eps));
        void** args=stackalloc void*[]{&x,&delta,&effectiveWeight,&output,&width,&eps};
        Launch(_norm,(uint)rows,1,1,256,stream,args);
    }
    public void QuantizeNvfp4(IntPtr input,IntPtr codes,IntPtr scales,IntPtr rowScale,int columns,int rows,float inputDivisor,IntPtr stream)
    {
        Shape(rows,columns);
        if(columns%256!=0)throw new ArgumentOutOfRangeException(nameof(columns));
        if(!(inputDivisor>0)||!float.IsFinite(inputDivisor))throw new ArgumentOutOfRangeException(nameof(inputDivisor));
        void** args=stackalloc void*[]{&input,&codes,&scales,&rowScale,&columns,&rows,&inputDivisor};
        Launch(_quant,Grid(checked((long)rows*columns)),1,1,256,stream,args);
    }
    /// <summary>Consumes unrounded F32 projection accumulators in gate/up groups of eight;
    /// rounds only the final SwiGLU result, matching the fused source operation.</summary>
    public void SwiGlu(IntPtr packed,IntPtr output,int intermediate,int rows,IntPtr stream)
    {
        Shape(rows,intermediate);
        if(intermediate%8!=0)throw new ArgumentOutOfRangeException(nameof(intermediate));
        void** args=stackalloc void*[]{&packed,&output,&intermediate};
        Launch(_swiglu,Grid(intermediate),(uint)rows,1,256,stream,args);
    }
    /// <summary>A16 projection over losslessly converted TS code/scale planes. Output remains
    /// F32 so the caller can apply a fused SwiGLU, residual add or explicit BF16 boundary.</summary>
    public void ProjectA16(IntPtr codes,IntPtr scales,IntPtr input,IntPtr output,int outputRows,int columns,int rows,
        float inverseWeightDivisor,IntPtr stream,int inputStride=0,int outputStride=0)
    {
        Shape(rows,columns); Shape(rows,outputRows);
        if(columns%16!=0||!(inverseWeightDivisor>0)||!float.IsFinite(inverseWeightDivisor))throw new ArgumentOutOfRangeException(nameof(columns));
        inputStride=inputStride==0?columns:inputStride; outputStride=outputStride==0?outputRows:outputStride;
        if(inputStride<columns||outputStride<outputRows)throw new ArgumentOutOfRangeException(nameof(inputStride));
        void** args=stackalloc void*[]{&codes,&scales,&input,&output,&outputRows,&columns,&rows,&inputStride,&outputStride,&inverseWeightDivisor};
        Launch(_a16,Grid(outputRows,4),(uint)rows,1,128,stream,args);
    }
    /// <summary>Native BF16 tensor-core projection with F32 accumulation/output. The nine
    /// original BF16 exception matrices and the independent AB projection use this path.
    /// No activation quantization or silent weight conversion is performed.</summary>
    public void ProjectBf16(IntPtr weights,IntPtr input,IntPtr output,int outputRows,int columns,int rows,IntPtr stream,int outputStride=0)
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        Shape(rows,columns); Shape(rows,outputRows);
        outputStride=outputStride==0?outputRows:outputStride;
        if(outputStride<outputRows)throw new ArgumentOutOfRangeException(nameof(outputStride));
        long elements=checked((long)rows*columns);
        ReserveBf16Input(elements,stream);
        IntPtr converted=_inputBf16;
        void** args=stackalloc void*[]{&input,&converted,&elements};
        Launch(_cast,Grid(elements),1,1,256,stream,args);
        CheckBlas(CublasApi.cublasSetStream(_blas,stream));
        float alpha=1,beta=0;
        // Row-major W[N,K] is column-major [K,N]. X[T,K] is column-major [K,T].
        CheckBlas(CublasApi.cublasGemmEx(_blas,1,0,outputRows,rows,columns,ref alpha,
            weights,CublasApi.CUDA_R_16BF,columns,converted,CublasApi.CUDA_R_16BF,columns,
            ref beta,output,CublasApi.CUDA_R_32F,outputStride,CublasApi.CUBLAS_COMPUTE_32F,-1));
    }
    internal void ReserveBf16Input(long elements,IntPtr stream)
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        if(elements<=0)throw new ArgumentOutOfRangeException(nameof(elements));
        long bytes=checked(elements*2);
        if(bytes>_inputCapacity)
        {
            if(_inputBf16!=IntPtr.Zero) { CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError(); CudaDriverApi.cuMemFree(_inputBf16).ThrowOnError(); _inputBf16=IntPtr.Zero; _inputCapacity=0; }
            CudaDriverApi.cuMemAlloc(out _inputBf16,(UIntPtr)bytes).ThrowOnError(); _inputCapacity=bytes;
        }
    }
    public void GdnConv(IntPtr packed,int packedWidth,int rowsPerSequence,int sequences,int convWidth,int convolutionWidth,
        IntPtr weights,IntPtr states,IntPtr qkv,IntPtr raw,bool advanceState,IntPtr stream,IntPtr count=default)
    {
        Shape(rowsPerSequence,sequences); ValidateGdn(16,48,convWidth);
        if(convolutionWidth<2||convolutionWidth>8||packedWidth<convWidth)throw new ArgumentOutOfRangeException(nameof(convolutionWidth));
        void** args=stackalloc void*[]{&packed,&packedWidth,&rowsPerSequence,&convWidth,&convolutionWidth,&weights,&states,&qkv,&raw,&count};
        Launch(_conv,Grid(convWidth),(uint)rowsPerSequence,(uint)sequences,256,stream,args);
        if(advanceState)GdnConvState(packed,packedWidth,0,rowsPerSequence,count,sequences,convWidth,convolutionWidth,states,0,1,stream);
    }
    public void GdnConvState(IntPtr input,int inputWidth,long layerStride,int rowsPerSequence,IntPtr count,int sequences,
        int convWidth,int convolutionWidth,IntPtr states,int pointerStride,int layers,IntPtr stream)
    {
        Shape(rowsPerSequence,sequences); Shape(layers,convWidth);
        void** args=stackalloc void*[]{&input,&inputWidth,&layerStride,&rowsPerSequence,&count,&convWidth,&convolutionWidth,&states,&pointerStride};
        Launch(_convState,Grid(convWidth),(uint)sequences,(uint)layers,256,stream,args);
    }
    public void GdnPrep(IntPtr qkv,IntPtr alphaBeta,int alphaBetaWidth,int rows,int keyHeads,int valueHeads,int convWidth,
        IntPtr dtBias,IntPtr rawALog,IntPtr decay,IntPtr beta,IntPtr stream,IntPtr count=default,int rowsPerSequence=1)
    {
        ValidateGdn(keyHeads,valueHeads,convWidth); Shape(rows,convWidth);
        if(alphaBetaWidth<2*valueHeads||rowsPerSequence<1)throw new ArgumentOutOfRangeException(nameof(alphaBetaWidth));
        void** args=stackalloc void*[]{&qkv,&alphaBeta,&alphaBetaWidth,&rows,&keyHeads,&valueHeads,&convWidth,&dtBias,&rawALog,&decay,&beta,&count,&rowsPerSequence};
        Launch(_prep,Grid(checked((long)rows*(2*keyHeads+valueHeads)*32)),1,1,256,stream,args);
    }
    public void GdnScan(IntPtr qkv,int convWidth,int keyHeads,int valueHeads,int rowsPerSequence,int sequences,
        IntPtr decay,IntPtr beta,IntPtr states,IntPtr output,bool verify,IntPtr stream,IntPtr count=default)
    {
        ValidateGdn(keyHeads,valueHeads,convWidth); Shape(rowsPerSequence,sequences); int verifyValue=verify?1:0;
        void** args=stackalloc void*[]{&qkv,&convWidth,&keyHeads,&valueHeads,&rowsPerSequence,&decay,&beta,&states,&output,&count,&verifyValue};
        Launch(_scan,(uint)valueHeads,(uint)sequences,8,128,stream,args);
    }
    public void GdnReplay(IntPtr qkv,long qkvStride,int convWidth,int keyHeads,int valueHeads,int rowsPerSequence,IntPtr kept,
        IntPtr decay,IntPtr beta,long decayStride,IntPtr states,int pointerStride,int sequences,int layers,IntPtr stream)
    {
        ValidateGdn(keyHeads,valueHeads,convWidth); Shape(sequences,layers);
        void** args=stackalloc void*[]{&qkv,&qkvStride,&convWidth,&keyHeads,&valueHeads,&rowsPerSequence,&kept,&decay,&beta,&decayStride,&states,&pointerStride};
        Launch(_replay,(uint)valueHeads,(uint)sequences,checked((uint)layers*8),128,stream,args);
    }
    public void GdnOut(IntPtr recurrent,IntPtr packed,int packedWidth,int convWidth,int rows,int valueHeads,IntPtr plainNorm,
        float eps,IntPtr output,IntPtr stream)
    {
        Shape(rows,valueHeads); if(valueHeads!=48||packedWidth<convWidth+valueHeads*128)throw new ArgumentOutOfRangeException(nameof(valueHeads));
        void** args=stackalloc void*[]{&recurrent,&packed,&packedWidth,&convWidth,&rows,&valueHeads,&plainNorm,&eps,&output};
        Launch(_out,Grid(checked((long)rows*valueHeads*32)),1,1,256,stream,args);
    }
    private static void ValidateGdn(int keyHeads,int valueHeads,int convWidth)
    {
        if(keyHeads!=16||valueHeads!=48||convWidth!=(2*keyHeads+valueHeads)*128)
            throw new NotSupportedException("Only the bound Qwen3.8 target GDN geometry is supported.");
    }
    public void Dispose()
    {
        if(_disposed)return;
        if(_blas!=IntPtr.Zero) { CheckBlas(CublasApi.cublasDestroy(_blas)); _blas=IntPtr.Zero; }
        if(_inputBf16!=IntPtr.Zero) { CudaDriverApi.cuMemFree(_inputBf16).ThrowOnError(); _inputBf16=IntPtr.Zero; _inputCapacity=0; }
        _module.Dispose(); _disposed=true;
    }
}
