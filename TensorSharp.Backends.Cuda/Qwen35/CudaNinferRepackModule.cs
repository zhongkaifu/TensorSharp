// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Explicit original NInfer NVFP4 byte-layout converter. It neither
/// validates payload values on the device nor changes projection math. The
/// caller must first validate all source scales and the original divisor,
/// retain that divisor in the descriptor and provide the declared capacities.</summary>
public sealed unsafe class CudaNinferRepackModule : IDisposable
{
    public const long MaximumOriginalBytes=128L<<20;
    public const int MaximumColumns=65536;
    public const string Symbol="ts_ni_repack_nvfp4";
    private readonly CudaModule _module;
    private readonly IntPtr _function,_context;
    private bool _disposed;
    public string ImageSha256=>_module.ImageSha256;
    internal readonly record struct Geometry(long CodeBytes,long ScaleBytes,long DestinationBytes,long OriginalBytes,long Groups);
    public CudaNinferRepackModule(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if(_context==IntPtr.Zero||IntPtr.Size!=8)throw new PlatformNotSupportedException("The repacker requires a current 64-bit CUDA context.");
        _module=CudaModule.LoadFromFile(path,identify:true);
        try
        {
            var functions=_module.EnumerateFunctions();
            if(functions.Length!=1||functions[0].Name!=Symbol||functions[0].Function==IntPtr.Zero)
                throw new InvalidOperationException("The NVFP4 repack module must contain exactly its expected encoded-byte kernel.");
            _function=functions[0].Function;CudaDriverApi.cuFuncLoad(_function).ThrowOnError();
            ulong[] offsets={0,8,16,20,24},sizes={8,8,4,4,4};
            for(int i=0;i<offsets.Length;i++)
            {
                CudaDriverApi.cuFuncGetParamInfo(_function,(UIntPtr)(uint)i,out var offset,out var size).ThrowOnError();
                if(offset.ToUInt64()!=offsets[i]||size.ToUInt64()!=sizes[i])throw new InvalidOperationException("NVFP4 repack kernel ABI differs.");
            }
            if(CudaDriverApi.cuFuncGetParamInfo(_function,(UIntPtr)5,out _,out _)!=1)
                throw new InvalidOperationException("NVFP4 repack kernel argument count differs.");
        }
        catch{_module.Dispose();throw;}
    }
    internal static Geometry ValidateGeometry(int rows,int columns,CudaNinferOriginalRowMapping rowMapping)
    {
        if(rows<=0||rows%128!=0||columns<=0||columns%64!=0||columns>MaximumColumns)
            throw new ArgumentOutOfRangeException(nameof(rows),"Original NVFP4 repacking requires complete m128/k64 tiles within the bounded column width.");
        switch(rowMapping)
        {
            case CudaNinferOriginalRowMapping.Identity:break;
            case CudaNinferOriginalRowMapping.GateUpRows8:
                if(rows%256!=0)throw new ArgumentException("Gate/up halves must contain complete original m128 tiles.");break;
            case CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue:
                if(rows!=14336)throw new ArgumentException("Only the declared Q6144/K1024/gate6144/V1024 attention row order is supported.");break;
            default:throw new ArgumentOutOfRangeException(nameof(rowMapping));
        }
        long values=checked((long)rows*columns),codes=values/2,scales=values/16;
        long destination=checked(codes+scales),original=checked(destination+4);
        if(original>MaximumOriginalBytes)throw new NotSupportedException("Original NVFP4 projection exceeds the bounded 128 MiB staging budget.");
        return new(codes,scales,destination,original,values/16);
    }
    internal static void ValidateBuffers(IntPtr raw,IntPtr destination,Geometry geometry)
    {
        long source=raw.ToInt64(),target=destination.ToInt64();
        if(source<=0||target<=0||source%8!=0||target%8!=0
            ||source>long.MaxValue-geometry.OriginalBytes||target>long.MaxValue-geometry.DestinationBytes)
            throw new ArgumentException("Repacking requires nonzero aligned, nonoverflowing original and destination buffers.");
        if(source<target+geometry.DestinationBytes&&target<source+geometry.OriginalBytes)
            throw new ArgumentException("Original and destination repack ranges must not alias.");
    }
    /// <summary>Writes exactly code|scale destination bytes; the four-byte
    /// calibration trailer remains unread. Both buffers stay caller-owned.
    /// Caller must synchronize before reusing staging or unloading the module.</summary>
    public void Repack(IntPtr raw,IntPtr destination,int rows,int columns,CudaNinferOriginalRowMapping rowMapping,IntPtr stream)
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        var geometry=ValidateGeometry(rows,columns,rowMapping);ValidateBuffers(raw,destination,geometry);
        CudaDriverApi.cuCtxGetCurrent(out var current).ThrowOnError();
        if(current!=_context)throw new InvalidOperationException("Make the repack module's owning CUDA context current.");
        int mapping=(int)rowMapping;
        void** args=stackalloc void*[]{&raw,&destination,&rows,&columns,&mapping};
        uint blocks=checked((uint)((geometry.Groups+255)/256));
        CudaDriverApi.cuLaunchKernel(_function,blocks,1,1,256,1,1,0,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();
    }
    public void Dispose()
    {
        if(_disposed)return;
        CudaDriverApi.cuCtxGetCurrent(out var current).ThrowOnError();
        if(current!=_context)throw new InvalidOperationException("Make the repack module's owning CUDA context current before disposal.");
        _module.Dispose();_disposed=true;
    }
}
