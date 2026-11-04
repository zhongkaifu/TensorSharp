// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Same bounded memory/stream/public graph ABI pattern as NinferNormCacheProbe,
// without its norm-specific launchers. No existing probe is edited.
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    private const int Guard=256;
    private static byte[] Guarded(byte[] body){var a=new byte[body.Length+2*Guard];Array.Fill(a,(byte)0xa5);body.CopyTo(a,Guard);return a;}
    private sealed unsafe class Memory:IDisposable
    {
        internal sealed record Buffer(string Name,IntPtr Allocation,IntPtr Pointer,int Bytes,string? ImmutableSha);
        private readonly List<Buffer> _buffers=new();
        internal Buffer Put(string name,byte[] body,bool immutable=true){byte[] all=Guarded(body);CudaDriverApi.cuMemAlloc(out var p,(UIntPtr)all.Length).ThrowOnError();var b=new Buffer(name,p,p+Guard,body.Length,immutable?Sha(all):null);_buffers.Add(b);Write(p,all);return b;}
        internal Buffer Empty(string name,int bytes)=>Put(name,new byte[bytes],false);
        internal static void Write(IntPtr p,byte[] data){fixed(byte* v=data)CudaDriverApi.cuMemcpyHtoD(p,(IntPtr)v,(UIntPtr)data.Length).ThrowOnError();}
        internal static byte[] Read(IntPtr p,int bytes){var a=new byte[bytes];fixed(byte* v=a)CudaDriverApi.cuMemcpyDtoH((IntPtr)v,p,(UIntPtr)bytes).ThrowOnError();return a;}
        internal static byte[] Body(Buffer b)=>Read(b.Pointer,b.Bytes);
        internal static void Poison(Buffer b,byte poison){var a=new byte[b.Bytes];Array.Fill(a,poison);Write(b.Allocation,Guarded(a));}
        internal bool Inputs()=>_buffers.Where(x=>x.ImmutableSha!=null).All(b=>Sha(Read(b.Allocation,b.Bytes+2*Guard))==b.ImmutableSha);
        internal bool Guards()=>_buffers.All(b=>{byte[] a=Read(b.Allocation,b.Bytes+2*Guard);return a.AsSpan(0,Guard).IndexOfAnyExcept((byte)0xa5)<0&&a.AsSpan(a.Length-Guard).IndexOfAnyExcept((byte)0xa5)<0;});
        public void Dispose(){foreach(var b in _buffers.AsEnumerable().Reverse())CudaDriverApi.cuMemFree(b.Allocation).ThrowOnError();}
    }
    private sealed class Stream:IDisposable
    {internal IntPtr Pointer{get;}internal Stream(){CudaDriverApi.cuStreamCreate(out var p,1).ThrowOnError();Need(p!=IntPtr.Zero,"Nondefault stream required.");Pointer=p;}internal void Sync()=>CudaDriverApi.cuStreamSynchronize(Pointer).ThrowOnError();public void Dispose(){Sync();CudaDriverApi.cuStreamDestroy(Pointer).ThrowOnError();}}
    private static class GraphApi
    {
        [StructLayout(LayoutKind.Sequential)]internal struct KernelParams{internal IntPtr Function;internal uint GridX,GridY,GridZ,BlockX,BlockY,BlockZ,Shared;internal IntPtr Parameters,Extra;}
        [DllImport("libcuda.so.1")]internal static extern int cuGraphGetNodes(IntPtr graph,IntPtr nodes,ref UIntPtr count);
        [DllImport("libcuda.so.1")]internal static extern int cuGraphNodeGetType(IntPtr node,out int type);
        [DllImport("libcuda.so.1")]internal static extern int cuGraphKernelNodeGetParams(IntPtr node,out KernelParams parameters);
    }
}
