// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private const int Guard=256;
    private static byte[] Guarded(byte[] body){var a=new byte[body.Length+2*Guard];Array.Fill(a,(byte)0xa5);body.CopyTo(a,Guard);return a;}
    private sealed unsafe class Memory:IDisposable
    {
        internal sealed record Buffer(string Name,IntPtr Allocation,IntPtr Pointer,int Bytes,string? ImmutableSha);
        private readonly List<Buffer> _buffers=new();
        internal Buffer Put(string name,byte[] body,bool immutable=true)
        {
            byte[] all=Guarded(body);CudaDriverApi.cuMemAlloc(out var p,(UIntPtr)all.Length).ThrowOnError();var b=new Buffer(name,p,p+Guard,body.Length,immutable?Sha(all):null);_buffers.Add(b);Write(p,all);return b;
        }
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
    {
        internal IntPtr Pointer{get;}
        internal Stream(){CudaDriverApi.cuStreamCreate(out var p,1).ThrowOnError();Need(p!=IntPtr.Zero,"Nondefault stream required.");Pointer=p;}
        internal void Sync()=>CudaDriverApi.cuStreamSynchronize(Pointer).ThrowOnError();
        public void Dispose(){Sync();CudaDriverApi.cuStreamDestroy(Pointer).ThrowOnError();}
    }
    private sealed unsafe class Norm:IDisposable
    {
        private readonly CudaModule _module;internal IntPtr Function{get;}internal string Symbol{get;}internal string ImageSha256=>S.Normalize(_module.ImageSha256);
        internal long Submissions{get;private set;}
        internal Norm(string path,string expected,bool candidate)
        {
            Symbol=candidate?"ts_ni_add_norm_t8_cached":"ts_ni_add_norm";_module=CudaModule.LoadFromFile(path,identify:true);
            try
            {
                Need(S.Matches(_module.ImageSha256,expected),"Actual loaded image differs from explicit expected SHA.");Function=_module.GetFunction(Symbol);CudaDriverApi.cuFuncLoad(Function).ThrowOnError();
                ulong[] offsets={0,8,16,24,32,36},sizes={8,8,8,8,4,4};for(uint i=0;i<6;i++){CudaDriverApi.cuFuncGetParamInfo(Function,(UIntPtr)i,out var o,out var s).ThrowOnError();Need(o.ToUInt64()==offsets[i]&&s.ToUInt64()==sizes[i],"Six-argument norm ABI differs.");}
                Need(CudaDriverApi.cuFuncGetParamInfo(Function,(UIntPtr)6,out _,out _)==1,"Unexpected norm argument.");
            }catch{_module.Dispose();throw;}
        }
        internal void Run(IntPtr x,IntPtr d,IntPtr w,IntPtr y,IntPtr stream)
        {
            Need(x!=IntPtr.Zero&&w!=IntPtr.Zero&&y!=IntPtr.Zero&&x!=y&&stream!=IntPtr.Zero,"Exact supported disjoint x/y and nondefault stream required.");int width=Width;float eps=Epsilon;
            void** args=stackalloc void*[]{&x,&d,&w,&y,&width,&eps};CudaDriverApi.cuLaunchKernel(Function,Rows,1,1,256,1,1,0,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();Submissions++;
        }
        public void Dispose()=>_module.Dispose();
    }
    private sealed class Arm
    {
        internal readonly Memory.Buffer X,Y;internal readonly IntPtr Delta;
        internal Arm(Memory memory,string name,string mode,Memory.Buffer delta)
        {X=memory.Empty(name+"-residual",Count*4);Y=memory.Empty(name+"-norm",Count*4);Delta=mode switch{"null"=>IntPtr.Zero,"delta-x"=>X.Pointer,"delta-output"=>Y.Pointer,_=>delta.Pointer};}
        internal void Reset(Fixture f,string mode,byte poison)
        {Memory.Poison(X,poison);Memory.Poison(Y,poison);Memory.Write(X.Pointer,Bytes(f.X));if(mode=="delta-output")Memory.Write(Y.Pointer,Bytes(f.Delta));}
    }
    private static class GraphApi
    {
        [StructLayout(LayoutKind.Sequential)]internal struct KernelParams{internal IntPtr Function;internal uint GridX,GridY,GridZ,BlockX,BlockY,BlockZ,Shared;internal IntPtr Parameters,Extra;}
        [DllImport("libcuda.so.1")]internal static extern int cuGraphGetNodes(IntPtr graph,IntPtr nodes,ref UIntPtr count);
        [DllImport("libcuda.so.1")]internal static extern int cuGraphNodeGetType(IntPtr node,out int type);
        [DllImport("libcuda.so.1")]internal static extern int cuGraphKernelNodeGetParams(IntPtr node,out KernelParams parameters);
    }
    private sealed unsafe class Captured:IDisposable
    {
        private IntPtr _graph,_exec;internal object[] Nodes{get;}
        internal Captured(Norm norm,Arm arm,Memory.Buffer weight,Memory.Buffer seedX,Memory.Buffer seedY,IntPtr stream)
        {
            CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try
            {
                CudaDriverApi.cuMemcpyDtoDAsync(arm.X.Pointer,seedX.Pointer,(UIntPtr)(Count*4),stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync(arm.Y.Pointer,seedY.Pointer,(UIntPtr)(Count*4),stream).ThrowOnError();
                norm.Run(arm.X.Pointer,arm.Delta,weight.Pointer,arm.Y.Pointer,stream);
            }
            catch{_=CudaDriverApi.cuStreamEndCapture(stream,out var bad);if(bad!=IntPtr.Zero)_=CudaDriverApi.cuGraphDestroy(bad);throw;}
            CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();Need(count.ToUInt64()==3,"Each full chain must capture exactly two resets plus one norm.");
                IntPtr* nodes=stackalloc IntPtr[3];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();var rows=new List<object>();int copies=0,kernels=0;
                for(int i=0;i<3;i++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[i],out int type).ThrowOnError();
                    if(type==1){copies++;rows.Add(new{type="device-copy-reset",bytes=Count*4});continue;}
                    Need(type==0,"Only kernel/copy nodes are admitted.");kernels++;GraphApi.cuGraphKernelNodeGetParams(nodes[i],out var p).ThrowOnError();
                    Need(p.Function==norm.Function&&p.GridX==Rows&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1&&p.Shared==0&&p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero,"Captured norm geometry/function differs.");
                    IntPtr Arg(int a)=>Marshal.ReadIntPtr(p.Parameters,a*IntPtr.Size);
                    Need(Marshal.ReadIntPtr(Arg(0))==arm.X.Pointer&&Marshal.ReadIntPtr(Arg(1))==arm.Delta&&Marshal.ReadIntPtr(Arg(2))==weight.Pointer&&Marshal.ReadIntPtr(Arg(3))==arm.Y.Pointer&&Marshal.ReadInt32(Arg(4))==Width&&Marshal.ReadInt32(Arg(5))==BitConverter.SingleToInt32Bits(Epsilon),"Captured norm six arguments differ.");
                    rows.Add(new{type="norm",symbol=norm.Symbol,grid=new[]{p.GridX,p.GridY,p.GridZ},block=new[]{p.BlockX,p.BlockY,p.BlockZ},sharedBytes=p.Shared});
                }
                Need(copies==2&&kernels==1,"Full symmetric reset+norm chain required.");Nodes=rows.ToArray();CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }catch{Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
}
