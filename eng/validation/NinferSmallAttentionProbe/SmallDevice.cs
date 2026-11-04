// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    // The actual R12 Verify launch: split128-thread kernel plus its combine.
    // This deliberately does not use the frozen helper's nonsplit256 launcher.
    private sealed unsafe class LegacySplit : IDisposable
    {
        private readonly CudaModule _module;
        private readonly IntPtr _split, _combine;
        internal string ImageSha256 => _module.ImageSha256;
        internal long KernelSubmissions { get; private set; }
        internal LegacySplit(string path)
        {
            _module = CudaModule.LoadFromFile(path, identify: true);
            try
            {
                _split = Bind("ts_ninfer_attn_bf16_split_f32", new ulong[] { 0,8,16,24,32,40,48,52,56,60,64,72 }, new ulong[] { 8,8,8,4,8,8,4,4,4,4,4,8 });
                _combine = Bind("ts_ninfer_attn_combine_f32", new ulong[] { 0,8,16,24 }, new ulong[] { 8,8,4,8 });
            }
            catch { _module.Dispose(); throw; }
        }
        private IntPtr Bind(string name, ulong[] offsets, ulong[] sizes)
        {
            IntPtr f = _module.GetFunction(name); CudaDriverApi.cuFuncLoad(f).ThrowOnError();
            for (uint i=0;i<offsets.Length;i++)
            {
                CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)i,out var o,out var s).ThrowOnError();
                Need(o.ToUInt64()==offsets[i] && s.ToUInt64()==sizes[i], "Actual legacy split/combine ABI differs.");
            }
            Need(CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)(uint)offsets.Length,out _,out _)==1,"Extra legacy argument.");
            return f;
        }
        internal void Run(IntPtr q,IntPtr gate,IntPtr ends,IntPtr blocks,long kOff,long vOff,int keys,IntPtr partial,IntPtr output,IntPtr stream)
        {
            int maxBlocks=MaxBlocks,rows=8,heads=24,kv=4; float scale=.0625f;
            var plan=Q35Kernels.GetSplitAttentionPlan(keys); int chunk=plan.Chunk,splits=plan.Splits;
            void** a=stackalloc void*[] { &q,&ends,&blocks,&maxBlocks,&kOff,&vOff,&rows,&heads,&kv,&scale,&chunk,&partial };
            CudaDriverApi.cuLaunchKernel(_split,(uint)splits,4,1,128,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();
            void** b=stackalloc void*[] { &partial,&gate,&splits,&output };
            CudaDriverApi.cuLaunchKernel(_combine,192,1,1,256,1,1,0,stream,(IntPtr)b,IntPtr.Zero).ThrowOnError(); KernelSubmissions+=2;
        }
        public void Dispose()=>_module.Dispose();
    }

    private sealed class StreamScope : IDisposable
    {
        internal IntPtr Pointer { get; }
        internal StreamScope(){CudaDriverApi.cuStreamCreate(out var p,0).ThrowOnError();Pointer=p;}
        internal void Sync()=>CudaDriverApi.cuStreamSynchronize(Pointer).ThrowOnError();
        public void Dispose(){Sync();CudaDriverApi.cuStreamDestroy(Pointer).ThrowOnError();}
    }

    // Probe-only graph inspection. CUDA's public v1 kernel-node parameter ABI
    // is fixed here; these entry points are never called by CPU self-tests.
    private static class GraphApi
    {
        [StructLayout(LayoutKind.Sequential)] internal struct KernelParams
        { internal IntPtr Function; internal uint GridX,GridY,GridZ,BlockX,BlockY,BlockZ,Shared; internal IntPtr Parameters,Extra; }
        [DllImport("libcuda.so.1")] internal static extern int cuGraphGetNodes(IntPtr graph,IntPtr nodes,ref UIntPtr count);
        [DllImport("libcuda.so.1")] internal static extern int cuGraphNodeGetType(IntPtr node,out int type);
        [DllImport("libcuda.so.1")] internal static extern int cuGraphKernelNodeGetParams(IntPtr node,out KernelParams parameters);
        [DllImport("libcuda.so.1")] internal static extern int cuFuncGetName(out IntPtr name,IntPtr function);
    }
    private sealed unsafe class Captured : IDisposable
    {
        private IntPtr _graph,_exec;
        internal object[] Nodes { get; }
        internal Captured(IntPtr stream,Action launch,IntPtr position,IntPtr unusedPosition,bool candidate=true)
        {
            CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try { launch(); }
            catch { _=CudaDriverApi.cuStreamEndCapture(stream,out var invalid);if(invalid!=IntPtr.Zero) _=CudaDriverApi.cuGraphDestroy(invalid);throw; }
            CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();
                int expected=candidate?4:2;Need(count.ToUInt64()==(ulong)expected,"Graph must contain the entire candidate4/legacy2-kernel chain.");
                IntPtr* nodes=stackalloc IntPtr[expected];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();
                var results=new List<object>();var roles=new HashSet<string>();
                for(int i=0;i<expected;i++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[i],out int type).ThrowOnError();Need(type==0,"Hidden non-kernel graph node refused.");
                    GraphApi.cuGraphKernelNodeGetParams(nodes[i],out var p).ThrowOnError();GraphApi.cuFuncGetName(out var n,p.Function).ThrowOnError();
                    Need(p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero,"Expected public kernelParams graph ABI, not extra-buffer launch encoding.");
                    string name=Marshal.PtrToStringAnsi(n)??"";
                    string role=candidate?(name=="ts_ni_prefill_pack"?"pack":name=="ts_ni_prefill_gate"?"gate":name.Contains("ts_ni_prefill_bf16_kernel",StringComparison.Ordinal)?"core":name.Contains("gqa_attention_prefill_reduce_kernel",StringComparison.Ordinal)?"reduce":"unknown"):
                        name=="ts_ninfer_attn_bf16_split_f32"?"split":name=="ts_ninfer_attn_combine_f32"?"combine":"unknown";
                    Need(role!="unknown"&&roles.Add(role),"Unexpected/duplicate graph kernel.");
                    bool dynamic=false;
                    if(role=="core")
                    {
                        Need(p.GridX==1&&p.GridY==24&&p.GridZ==4&&p.BlockX==128&&p.Shared==98304,"Captured core launch differs from fixed original T8/S4.");
                        IntPtr argument=Marshal.ReadIntPtr(p.Parameters,4*IntPtr.Size),captured=Marshal.ReadIntPtr(argument);
                        dynamic=captured==position&&captured!=unusedPosition;Need(dynamic,"Graph captured a CPU-derived/internal base-position pointer.");
                    }
                    if(role=="pack")
                    {
                        Need(p.GridX==192&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.Shared==0,"Captured pack launch differs.");
                        int baseValue=Marshal.ReadInt32(Marshal.ReadIntPtr(p.Parameters,8*IntPtr.Size)),pages=Marshal.ReadInt32(Marshal.ReadIntPtr(p.Parameters,10*IntPtr.Size));
                        Need(baseValue==0&&pages==640,"Pack must capture unusedbase0/fixed640-page metadata.");
                    }
                    if(role=="gate")Need(p.GridX==192&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.Shared==0,"Captured gate launch differs.");
                    if(role=="reduce")Need(p.GridX==8&&p.GridY==24&&p.GridZ==1&&p.BlockX==256&&p.Shared==0,"Captured reducer launch differs.");
                    if(role=="split")Need(p.GridY==4&&p.GridZ==1&&p.BlockX==128&&p.Shared==0,"Actual legacy128-thread split launch required.");
                    if(role=="combine")Need(p.GridX==192&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.Shared==0,"Actual legacy combine launch required.");
                    results.Add(new {role,name,grid=new[]{p.GridX,p.GridY,p.GridZ},block=new[]{p.BlockX,p.BlockY,p.BlockZ},sharedBytes=p.Shared,dynamicExternalPosition=dynamic});
                }
                Need(roles.SetEquals(candidate?new[]{"pack","core","reduce","gate"}:new[]{"split","combine"}),"Complete captured chain is required.");Nodes=results.ToArray();
                CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }
            catch {Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
}
