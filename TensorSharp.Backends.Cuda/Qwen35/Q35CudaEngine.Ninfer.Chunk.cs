// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Globalization;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferGdnChunkEnvironment="TS_Q35_NINFER_GDN_CHUNK_PTX";
    private CudaNinferGdnChunk _ninferGdnChunk;
    private IntPtr _ninferGdnChunkWorkspace;
    private long _ninferGdnChunkWorkspaceBytes;
    private string _ninferGdnChunkNamespace=string.Empty;
    private long _ninferGdnChunkLaunches;
    public bool NinferGdnChunkEnabled=>_ninferGdnChunk!=null;
    public long NinferGdnChunkWorkspaceBytes=>_ninferGdnChunkWorkspaceBytes;
    public long NinferGdnChunkLaunches=>_ninferGdnChunkLaunches;
    private string NinferGdnChunkNamespaceSuffix=>_ninferGdnChunkNamespace??string.Empty;

    internal readonly record struct NinferGdnChunkPlan(int ActualRows,int FullRows,int TailRows,int PaddingRows,bool MaterializeBf16Qk);
    internal static NinferGdnChunkPlan PlanNinferGdnChunk(int rowsPerSequence,int actualRows)
    {
        if(rowsPerSequence<1||actualRows<0||actualRows>rowsPerSequence)
            throw new ArgumentOutOfRangeException(nameof(actualRows),"Real rows must fit the supplied sequence span.");
        int full=actualRows/64*64;
        return new(actualRows,full,actualRows-full,rowsPerSequence-actualRows,actualRows>=64);
    }
    internal static string ResolveNinferGdnChunkPath(string configured)
    {
        if(configured==null)return null;
        if(string.IsNullOrWhiteSpace(configured))
            throw new ArgumentException(NinferGdnChunkEnvironment+" must name an original-source candidate CUDA module when set.");
        if(!Path.IsPathFullyQualified(configured))
            throw new ArgumentException("The NInfer GDN chunk module path must be fully qualified.",nameof(configured));
        string path=Path.GetFullPath(configured);
        if(!string.Equals(path,configured,StringComparison.Ordinal))
            throw new ArgumentException("The NInfer GDN chunk module path must be lexically normalized.",nameof(configured));
        if(!File.Exists(path))throw new FileNotFoundException("Configured NInfer GDN chunk module does not exist.",path);
        return path;
    }
    internal static string NinferGdnChunkStateSuffix(string profile,string moduleSha,int maximumRows)
    {
        if(string.IsNullOrEmpty(profile)||profile.Length>128||maximumRows<64)
            throw new ArgumentException("A bounded chunk profile and prefill capacity are required.");
        foreach(char c in profile)
            if(!char.IsAsciiLetterOrDigit(c)&&c!='-'&&c!='_'&&c!='.')
                throw new ArgumentException("Chunk profile contains a namespace delimiter or unsupported character.");
        if(moduleSha?.Length!=64)throw new ArgumentException("The full loaded module SHA256 is required.");
        foreach(char c in moduleSha)if(!Uri.IsHexDigit(c))throw new ArgumentException("The module identity is not SHA256 hexadecimal.");
        return "|gdn-prefill="+profile+":"+moduleSha.ToLowerInvariant()+":ub="+maximumRows.ToString(CultureInfo.InvariantCulture);
    }
    /// <summary>Called explicitly by the NInfer constructor after its CUDA context
    /// is current. No setting means OFF; there is no module discovery fallback.</summary>
    private void InitializeNinferGdnChunk()
    {
        string configured=Environment.GetEnvironmentVariable(NinferGdnChunkEnvironment);
        if(configured==null)return;
        if(_m.Ninfer==null)throw new NotSupportedException("The original NInfer GDN chunk policy requires the NInfer model route.");
        if(_ninferGdnChunk!=null)throw new InvalidOperationException("NInfer GDN chunk module was already initialized.");
        string path=ResolveNinferGdnChunkPath(configured);
        if(_m.NUbatch<64)throw new NotSupportedException("The configured NInfer chunk policy requires a prefill capacity of at least 64 rows.");
        // This buffer is owned by the engine's ordinary allocation list. Reserve
        // it before requests; no allocation occurs on the recurrence hot path.
        int fullCapacity=_m.NUbatch/64*64;
        var module=new CudaNinferGdnChunk(path);
        try
        {
            long bytes=CudaNinferGdnChunk.WorkspaceBytes(fullCapacity);
            IntPtr workspace=Alloc(bytes);
            string identity=NinferGdnChunkStateSuffix(CudaNinferGdnChunk.Profile,module.ImageSha256,_m.NUbatch);
            _ninferGdnChunkWorkspace=workspace;
            _ninferGdnChunkWorkspaceBytes=bytes;
            _ninferGdnChunkNamespace=identity;
            _ninferGdnChunk=module;
            Console.Error.WriteLine("[q35-cuda] explicit original-source NInfer GDN chunk candidate "+JsonSerializer.Serialize(new
            {
                profile=CudaNinferGdnChunk.Profile,modulePath=path,moduleImageSha256=module.ImageSha256,
                workspaceBytes=bytes,prefillCapacity=_m.NUbatch,fullChunkCapacity=fullCapacity,
                qkPolicy="spans-at-least64-materialize-normalized-qk-bf16-including-tail",
                defaultEnabled=false,qualityQualification="required"
            }));
        }
        catch{module.Dispose();throw;}
    }
    /// <summary>The disabled path is exactly the original batched helper dispatch.
    /// The optional policy applies the original chunk recipe to each sequence's
    /// real token span, then advances its tail through the existing F32 state.</summary>
    private void RunNinferGdnScan(Layer layer,in RowGroup group,int sequenceOffset,
        IntPtr qkv,IntPtr decay,IntPtr beta,IntPtr output)
    {
        if(_ninferGdnChunk==null)
        {
            IntPtr count=group.Counts!=null?_countDev+(nint)(4L*sequenceOffset):IntPtr.Zero;
            _ninfer.GdnScan(qkv,ConvDim,_m.GdnKHeads,_m.GdnVHeads,group.RowsPerSeq,group.Slots.Length,
                decay,beta,TabState(layer)+(nint)(8L*sequenceOffset),output,false,Stream,count);
            return;
        }
        if(group.Verify)throw new NotSupportedException("NInfer chunk candidate does not implement speculative verification or rollback.");
        if(group.Counts!=null&&group.Counts.Length!=group.Slots.Length)
            throw new ArgumentException("NInfer chunk real-row counts do not match sequence slots.");
        for(int sequence=0;sequence<group.Slots.Length;sequence++)
        {
            int actual=group.Counts==null?group.RowsPerSeq:group.Counts[sequence];
            var plan=PlanNinferGdnChunk(group.RowsPerSeq,actual);
            long first=(long)sequence*group.RowsPerSeq;
            IntPtr q=qkv+(nint)(first*ConvDim*sizeof(float));
            IntPtr gd=decay+(nint)(first*_m.GdnVHeads*sizeof(float));
            IntPtr b=beta+(nint)(first*_m.GdnVHeads*sizeof(float));
            IntPtr y=output+(nint)(first*ValueDim*sizeof(float));
            IntPtr table=TabState(layer)+(nint)(8L*(sequenceOffset+sequence));
            if(plan.MaterializeBf16Qk)
            {
                // Original NInfer also materializes the tail's normalized Q/K
                // when there is at least one full chunk. V is already BF16;
                // rerounding it is idempotent. Do not normalize a second time.
                _ninfer.Round(q,(long)plan.ActualRows*ConvDim,Stream);
                if(CudaNinferGdnChunk.WorkspaceBytes(plan.FullRows)>_ninferGdnChunkWorkspaceBytes)
                    throw new InvalidOperationException("NInfer GDN full span exceeds the preallocated chunk capacity.");
                IntPtr state=group.Slots[sequence].State[layer.Ordinal];
                _ninferGdnChunk.RunNormalized(q,plan.FullRows,gd,b,state,y,
                    _ninferGdnChunkWorkspace,_ninferGdnChunkWorkspaceBytes,Stream);
                _ninferGdnChunkLaunches++;
            }
            if(plan.TailRows>0)
            {
                long tail=plan.FullRows;
                _ninfer.GdnScan(q+(nint)(tail*ConvDim*sizeof(float)),ConvDim,_m.GdnKHeads,_m.GdnVHeads,
                    plan.TailRows,1,gd+(nint)(tail*_m.GdnVHeads*sizeof(float)),b+(nint)(tail*_m.GdnVHeads*sizeof(float)),
                    table,y+(nint)(tail*ValueDim*sizeof(float)),false,Stream);
            }
            if(plan.PaddingRows>0)
                CudaDriverApi.cuMemsetD32Async(y+(nint)((long)plan.ActualRows*ValueDim*sizeof(float)),0,
                    (UIntPtr)checked((long)plan.PaddingRows*ValueDim),Stream).ThrowOnError();
        }
    }
    /// <summary>Module lifetime only; the ordinary engine owner frees the
    /// preallocated workspace exactly once. Safe to call repeatedly.</summary>
    private void DisposeNinferGdnChunk()
    {
        _ninferGdnChunk?.Dispose();_ninferGdnChunk=null;
        _ninferGdnChunkWorkspace=IntPtr.Zero;_ninferGdnChunkWorkspaceBytes=0;
        _ninferGdnChunkNamespace=string.Empty;
    }
}
