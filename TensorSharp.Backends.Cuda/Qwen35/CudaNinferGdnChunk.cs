// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Explicit standalone candidate using unchanged original Ignis e600
/// BF16/TF32 GDN chunk kernels. No model or environment setting selects this
/// class automatically. Only complete 64-token chunks of a single sequence
/// are supported. This changes the prefill numerical recipe from sequential
/// F32-normalized recurrence; independent model qualification is required.</summary>
public sealed unsafe class CudaNinferGdnChunk : IDisposable
{
    public const string Profile="ignis-e600-original-chunk64-bf16-tf32-candidate-v1";
    public const int KeyHeads=16,ValueHeads=48,Dimension=128,ChunkSize=64,PackedWidth=10240;
    private readonly CudaModule _module;
    private readonly IntPtr _prepare,_state,_output,_outputMulti,_pack,_unpack;
    private readonly IntPtr _context;
    private bool _disposed;
    public string ImageSha256=>_module.ImageSha256;
    public string[] KernelSymbols {get;}
    [StructLayout(LayoutKind.Sequential)]
    private struct HeadMap
    {
        public int KeyHeads,ValueHeads;
        public uint Multiplier,Shift,Divisor;
    }
    private static readonly HeadMap Heads=new(){KeyHeads=16,ValueHeads=48,Multiplier=1431655766,Shift=2,Divisor=3};
    public readonly record struct Layout(long Query,long Key,long Value,long Output,long CumulativeDecay,
        long W,long U,long VNew,long ChunkStates,long Bytes);
    public CudaNinferGdnChunk(string modulePath)
    {
        ArgumentException.ThrowIfNullOrEmpty(modulePath);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if(_context==IntPtr.Zero||IntPtr.Size!=8||Marshal.SizeOf<HeadMap>()!=20)
            throw new PlatformNotSupportedException("Chunk candidate requires the current 64-bit CUDA context and a 20-byte head-map ABI.");
        _module=CudaModule.LoadFromFile(modulePath,identify:true);
        try
        {
            var symbols=_module.EnumerateFunctions();
            if(symbols.Length!=6)throw new InvalidOperationException("Chunk candidate requires exactly four original kernels plus its two bridge kernels.");
            _prepare=Find(symbols,"20prepare_wy_wu_kernelILi64ELi32ELi8EEE");
            _state=Find(symbols,"20state_passing_kernelILi16EEE");
            _output=Find(symbols,"13output_kernelILb0EEE");
            _outputMulti=Find(symbols,"13output_kernelILb1EEE");
            _pack=symbols.Single(x=>x.Name=="ts_ni_chunk_pack").Function;
            _unpack=symbols.Single(x=>x.Name=="ts_ni_chunk_unpack").Function;
            KernelSymbols=symbols.Select(x=>x.Name).OrderBy(x=>x,StringComparer.Ordinal).ToArray();
            foreach(var symbol in symbols)CudaDriverApi.cuFuncLoad(symbol.Function).ThrowOnError();
            CheckParameters(_prepare,new ulong[]{0,8,16,24,32,40,48,56},new ulong[]{8,8,8,8,8,8,8,20});
            CheckParameters(_state,new ulong[]{0,8,16,24,32,40,48,56,64,84},new ulong[]{8,8,8,8,8,8,8,8,20,4});
            foreach(IntPtr f in new[]{_output,_outputMulti})
                CheckParameters(f,new ulong[]{0,8,16,24,32,40,48,68,72},new ulong[]{8,8,8,8,8,8,20,4,4});
            CheckParameters(_pack,new ulong[]{0,8,16,24,32,36},new ulong[]{8,8,8,8,4,4});
            CheckParameters(_unpack,new ulong[]{0,8,16},new ulong[]{8,8,8});
            CudaDriverApi.cuFuncSetAttribute(_prepare,8,29440).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_state,8,45312).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_output,8,24576).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_outputMulti,8,24576).ThrowOnError();
        }
        catch{_module.Dispose();throw;}
    }
    private static IntPtr Find(CudaFunctionSymbol[] symbols,string template)
    {
        const string prefix="_ZN6ninfer3ops6detail15gated_delta_net7chunked";
        var found=symbols.Where(x=>x.Name.StartsWith(prefix,StringComparison.Ordinal)&&x.Name.Contains(template,StringComparison.Ordinal)).ToArray();
        if(found.Length!=1||found[0].Function==IntPtr.Zero)
            throw new InvalidOperationException("Original chunk kernel specialization missing or ambiguous: "+template);
        return found[0].Function;
    }
    private static void CheckParameters(IntPtr function,ulong[] offset,ulong[] size)
    {
        for(int i=0;i<offset.Length;i++)
        {
            CudaDriverApi.cuFuncGetParamInfo(function,(UIntPtr)(uint)i,out var actualOffset,out var actualSize).ThrowOnError();
            if(actualOffset.ToUInt64()!=offset[i]||actualSize.ToUInt64()!=size[i])
                throw new InvalidOperationException("Original chunk kernel ABI differs at argument "+i);
        }
        if(CudaDriverApi.cuFuncGetParamInfo(function,(UIntPtr)(uint)offset.Length,out _,out _)!=1)
            throw new InvalidOperationException("Original chunk kernel argument count differs.");
    }
    public static Layout WorkspaceLayout(int rows)
    {
        if(rows<64||rows%64!=0)throw new ArgumentOutOfRangeException(nameof(rows),"Only complete 64-token chunks are admitted.");
        long cursor=0;
        long Take(long bytes){long start=cursor;cursor=checked((cursor+bytes+255)&~255L);return start;}
        long q=Take(checked((long)rows*KeyHeads*Dimension*2));
        long k=Take(checked((long)rows*KeyHeads*Dimension*2));
        long v=Take(checked((long)rows*ValueHeads*Dimension*2));
        long o=Take(checked((long)rows*ValueHeads*Dimension*2));
        long g=Take(checked((long)rows*ValueHeads*4));
        long w=Take(checked((long)rows*ValueHeads*Dimension*2));
        long u=Take(checked((long)rows*ValueHeads*Dimension*2));
        long vn=Take(checked((long)rows*ValueHeads*Dimension*2));
        long hc=Take(checked((long)(rows/ChunkSize)*ValueHeads*Dimension*Dimension*2));
        return new(q,k,v,o,g,w,u,vn,hc,cursor);
    }
    public static long WorkspaceBytes(int rows)=>WorkspaceLayout(rows).Bytes;
    private void Current()
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        CudaDriverApi.cuCtxGetCurrent(out var current).ThrowOnError();
        if(current!=_context)throw new InvalidOperationException("Make the chunk module's owning context current.");
    }
    /// <summary>Runs original stages on BF16-rounded normalized q/k and BF16 v.
    /// State input/output may alias. Caller owns all buffers and stream lifetime.</summary>
    public void RunNormalized(IntPtr normalizedQkv,int rows,IntPtr decay,IntPtr beta,IntPtr state,
        IntPtr output,IntPtr workspace,long workspaceBytes,IntPtr stream,int inputStride=PackedWidth)
        =>RunNormalizedCore(normalizedQkv,rows,decay,beta,state,output,workspace,workspaceBytes,stream,inputStride,false);
    /// <summary>Diagnostic-only schedule isolation. Uses the unchanged original
    /// single-job output specialization even above the normal multi-job cutoff.
    /// This is not selected by the engine or included as an adopted policy.</summary>
    public void RunNormalizedDiagnostic(IntPtr normalizedQkv,int rows,IntPtr decay,IntPtr beta,IntPtr state,
        IntPtr output,IntPtr workspace,long workspaceBytes,IntPtr stream,bool forceSingleJobOutput,int inputStride=PackedWidth)
        =>RunNormalizedCore(normalizedQkv,rows,decay,beta,state,output,workspace,workspaceBytes,stream,inputStride,forceSingleJobOutput);
    private void RunNormalizedCore(IntPtr normalizedQkv,int rows,IntPtr decay,IntPtr beta,IntPtr state,
        IntPtr output,IntPtr workspace,long workspaceBytes,IntPtr stream,int inputStride,bool forceSingleJobOutput)
    {
        Current();var plan=WorkspaceLayout(rows);
        if(inputStride<PackedWidth||workspace==IntPtr.Zero||workspaceBytes<plan.Bytes
            ||normalizedQkv==IntPtr.Zero||decay==IntPtr.Zero||beta==IntPtr.Zero||state==IntPtr.Zero||output==IntPtr.Zero)
            throw new ArgumentException("Chunk candidate buffer or stride is invalid.");
        IntPtr q=workspace+(nint)plan.Query,k=workspace+(nint)plan.Key,v=workspace+(nint)plan.Value,o=workspace+(nint)plan.Output;
        IntPtr gc=workspace+(nint)plan.CumulativeDecay,w=workspace+(nint)plan.W,u=workspace+(nint)plan.U,vn=workspace+(nint)plan.VNew,hc=workspace+(nint)plan.ChunkStates;
        void** packArgs=stackalloc void*[]{&normalizedQkv,&q,&k,&v,&rows,&inputStride};
        Launch(_pack,checked((uint)(((long)rows*PackedWidth+255)/256)),1,256,0,stream,packArgs);
        HeadMap heads=Heads;int chunks=rows/ChunkSize;
        void** prepareArgs=stackalloc void*[]{&k,&v,&decay,&beta,&w,&u,&gc,&heads};
        Launch(_prepare,(uint)chunks,ValueHeads,256,29440,stream,prepareArgs);
        void** stateArgs=stackalloc void*[]{&w,&u,&k,&gc,&state,&vn,&hc,&state,&heads,&chunks};
        Launch(_state,ValueHeads*8,1,256,45312,stream,stateArgs);
        const int targetCtas=170*4;int jobs=(chunks*ValueHeads+targetCtas-1)/targetCtas;
        if(forceSingleJobOutput)jobs=1;
        int gridChunks=(chunks+jobs-1)/jobs;float scale=0.08838834764831845f;
        void** outputArgs=stackalloc void*[]{&q,&k,&vn,&gc,&hc,&o,&heads,&scale,&chunks};
        Launch(jobs==1?_output:_outputMulti,(uint)gridChunks,ValueHeads,128,24576,stream,outputArgs);
        long elements=(long)rows*ValueHeads*Dimension;
        void** unpackArgs=stackalloc void*[]{&o,&output,&elements};
        Launch(_unpack,checked((uint)((elements+255)/256)),1,256,0,stream,unpackArgs);
    }
    private static void Launch(IntPtr f,uint x,uint y,uint block,uint shared,IntPtr stream,void** args)
        =>CudaDriverApi.cuLaunchKernel(f,x,y,1,block,1,1,shared,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();
    public void Dispose(){if(_disposed)return;Current();_module.Dispose();_disposed=true;}
}
