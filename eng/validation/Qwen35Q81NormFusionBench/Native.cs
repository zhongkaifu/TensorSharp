using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.Cuda;

internal sealed unsafe class Native : IDisposable
{
    private const string Lib = "libcuda.so.1";
    [DllImport(Lib)] private static extern int cuMemAlloc_v2(out IntPtr p, UIntPtr bytes);
    [DllImport(Lib)] private static extern int cuMemFree_v2(IntPtr p);
    [DllImport(Lib)] private static extern int cuMemcpyHtoD_v2(IntPtr dst, IntPtr src, UIntPtr bytes);
    [DllImport(Lib)] private static extern int cuMemcpyDtoH_v2(IntPtr dst, IntPtr src, UIntPtr bytes);
    [DllImport(Lib)] private static extern int cuModuleLoadData(out IntPtr module, IntPtr image);
    [DllImport(Lib)] private static extern int cuModuleUnload(IntPtr module);
    [DllImport(Lib)] private static extern int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);
    [DllImport(Lib)] private static extern int cuLaunchKernel(IntPtr function, uint gx,uint gy,uint gz,uint bx,uint by,uint bz,uint smem,IntPtr stream,IntPtr args,IntPtr extra);
    [DllImport(Lib)] private static extern int cuFuncGetAttribute(out int value,int attr,IntPtr function);
    [DllImport(Lib)] private static extern int cuOccupancyMaxActiveBlocksPerMultiprocessor(out int value,IntPtr function,int block,UIntPtr dynamicShared);
    [DllImport(Lib)] private static extern int cuEventCreate(out IntPtr e,uint flags);
    [DllImport(Lib)] private static extern int cuEventRecord(IntPtr e,IntPtr stream);
    [DllImport(Lib)] private static extern int cuEventSynchronize(IntPtr e);
    [DllImport(Lib)] private static extern int cuEventElapsedTime(out float ms,IntPtr start,IntPtr end);
    [DllImport(Lib)] private static extern int cuEventDestroy_v2(IntPtr e);
    internal static MethodInfo Q35LocatorMethod()=>typeof(Q35CudaEngine).Assembly.GetType("TensorSharp.Cuda.CudaKernels",true)!.GetMethod("LocatePtxPath",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic,null,new[]{typeof(string)},null)??throw new InvalidOperationException("Q35 string-filename locator missing.");
    internal static bool IsOriginalImage(string? actual)=>string.Equals(actual,"bbdaca53a0fb56ec78644356edfa981f6552e57c011f6b8cbfe015ac7b608515",StringComparison.OrdinalIgnoreCase);
    internal delegate void Norm(IntPtr x,IntPtr delta,IntPtr weights,IntPtr y,int rows,int width,float eps,IntPtr stream);
    internal delegate void Quant(IntPtr input,IntPtr codes,IntPtr scales,int width,int rows,IntPtr stream);
    internal delegate void Gemv(IntPtr codes,IntPtr scales,int outDim,int k,IntPtr xq,IntPtr xd,float alpha,IntPtr y,int rows,IntPtr stream);
    internal readonly Norm OriginalNorm;
    internal readonly Quant OriginalQuant;
    internal readonly Gemv OriginalGemv;
    internal readonly Q35CudaEngine Engine;
    internal readonly IntPtr Stream, Function;
    internal readonly object ReferenceIdentity;
    internal readonly string PrivateImageSha;
    private readonly object referenceKernels;
    private IntPtr module;
    internal static void Check(int code) { if(code!=0) throw new InvalidOperationException("CUDA driver error "+code); }
    internal static FieldInfo Field(Type t,string name,Type? expected=null,bool publicField=false)
    {
        var f=t.GetField(name,BindingFlags.Instance|BindingFlags.DeclaredOnly|(publicField?BindingFlags.Public:BindingFlags.NonPublic));
        Bench.Require(f!=null&&!f.IsStatic&&(expected==null||f.FieldType==expected),"Unsupported diagnostic field "+t+"."+name);return f!;
    }
    internal Native(Q35CudaEngine engine,string cubin)
    {
        Engine=engine;var engineKernels=Field(typeof(Q35CudaEngine),"_k").GetValue(engine)!;var type=engineKernels.GetType();
        var cuda=typeof(Q35CudaEngine).Assembly;var mt=cuda.GetType("TensorSharp.Cuda.CudaModule",true)!;
        var locator=Q35LocatorMethod();
        string ptx=(string)locator.Invoke(null,new object[]{Bench.Q35KernelFile})!;
        object originalModule=mt.GetMethod("LoadFromFile",new[]{typeof(string),typeof(bool)})!.Invoke(null,new object[]{ptx,true})!;
        object? created=null;
        try
        {
        object kernels=created=type.GetMethod("FromModule",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic)!.Invoke(null,new[]{originalModule})!;referenceKernels=kernels;
        Bench.Require(IsOriginalImage((string?)mt.GetProperty("ImageSha256")!.GetValue(originalModule)),"Actual separately loaded original reference image must be BBD, not PTX fallback.");
        T Bind<T>(string name) where T:Delegate => type.GetMethod(name,BindingFlags.Public|BindingFlags.Instance|BindingFlags.DeclaredOnly)!.CreateDelegate<T>(kernels);
        OriginalNorm=Bind<Norm>("AddRmsNorm");OriginalQuant=Bind<Quant>("QuantizeQ81");OriginalGemv=Bind<Gemv>("GemvNvfp4");
        var get=mt.GetMethod("GetFunction",BindingFlags.Public|BindingFlags.Instance|BindingFlags.DeclaredOnly)!;
        ReferenceIdentity=new{module_handle=((IntPtr)Field(mt,"module",typeof(IntPtr)).GetValue(originalModule)!).ToInt64(),image_sha256=mt.GetProperty("ImageSha256")!.GetValue(originalModule),functions=new[]{"ts_q35_add_rmsnorm_f32","ts_q35_quantize_q81_f32"}.ToDictionary(n=>n,n=>((IntPtr)get.Invoke(originalModule,new object[]{n})!).ToInt64()),actual_reference_calls="Separately loaded identified original BBD module; original Norm/Quant/Gemv wrappers, actual model weight planes"};
        var stream=typeof(Q35CudaEngine).GetProperty("Stream",BindingFlags.Instance|BindingFlags.NonPublic|BindingFlags.DeclaredOnly);
        Bench.Require(stream?.PropertyType==typeof(IntPtr),"Stream contract changed.");Stream=(IntPtr)stream!.GetValue(engine)!;
        byte[] image=File.ReadAllBytes(cubin);PrivateImageSha=Bench.Hash(image);fixed(byte* p=image)Check(cuModuleLoadData(out module,(IntPtr)p));
        Check(cuModuleGetFunction(out var f,module,"ts_bench_add_rmsnorm_q81_f32"));Function=f;
        }
        catch{if(module!=IntPtr.Zero){cuModuleUnload(module);module=IntPtr.Zero;}if(created is IDisposable owner)owner.Dispose();else((IDisposable)originalModule).Dispose();throw;}
    }
    internal object Resources()
    {
        int Attr(int id){Check(cuFuncGetAttribute(out int v,id,Function));return v;}
        Check(cuOccupancyMaxActiveBlocksPerMultiprocessor(out int active,Function,512,UIntPtr.Zero));
        int maxThreads=Attr(0);Bench.Require(maxThreads>=512&&active>=1,"Private kernel not launchable at actual512 threads.");
        return new{block_threads=512,static_shared_bytes=Attr(1),local_bytes=Attr(3),registers=Attr(4),binary_arch=Attr(6),max_threads=maxThreads,active_ctas=active,occupancy_is_not_measured_stall_or_speedup=true};
    }
    internal void Fused(IntPtr x,IntPtr delta,IntPtr weights,IntPtr y,int rows,int width,float eps,IntPtr q,IntPtr d)
    {
        void** args=stackalloc void*[]{&x,&delta,&weights,&y,&width,&eps,&q,&d};
        Check(cuLaunchKernel(Function,(uint)rows,1,1,512,1,1,0,Stream,(IntPtr)args,IntPtr.Zero));
    }
    internal static byte[] Download(IntPtr pointer,int bytes)
    {
        byte[] data=new byte[bytes];fixed(byte* p=data)Check(cuMemcpyDtoH_v2((IntPtr)p,pointer,new UIntPtr((uint)bytes)));return data;
    }
    internal float Time(Action issue,int iterations)
    {
        IntPtr a=IntPtr.Zero,b=IntPtr.Zero;
        try{Check(cuEventCreate(out a,0));Check(cuEventCreate(out b,0));Check(cuEventRecord(a,Stream));for(int i=0;i<iterations;i++)issue();Check(cuEventRecord(b,Stream));Check(cuEventSynchronize(b));Check(cuEventElapsedTime(out float ms,a,b));Bench.Require(float.IsFinite(ms)&&ms>0,"Invalid CUDA event time.");return ms/iterations;}
        finally{if(b!=IntPtr.Zero)Check(cuEventDestroy_v2(b));if(a!=IntPtr.Zero)Check(cuEventDestroy_v2(a));}
    }
    public void Dispose(){try{Engine.Synchronize();}finally{try{if(module!=IntPtr.Zero){var owned=module;module=IntPtr.Zero;Check(cuModuleUnload(owned));}}finally{((IDisposable)referenceKernels).Dispose();}}}
    internal sealed class Buffer : IDisposable
    {
        internal const int Guard=256;
        internal readonly int Bytes;internal readonly byte Poison;internal IntPtr Base;internal IntPtr Data=>Base+Guard;
        private readonly Action sync;
        internal Buffer(int bytes,byte poison,Action sync){Bench.Require(bytes>0,"Empty allocation.");Bytes=bytes;Poison=poison;this.sync=sync;Check(cuMemAlloc_v2(out Base,new UIntPtr(checked((uint)(bytes+2*Guard)))));try{Reset();}catch{Check(cuMemFree_v2(Base));Base=IntPtr.Zero;throw;}}
        internal void Reset(){var fill=Enumerable.Repeat(Poison,checked(Bytes+2*Guard)).ToArray();fixed(byte* p=fill)Check(cuMemcpyHtoD_v2(Base,(IntPtr)p,new UIntPtr((uint)fill.Length)));}
        internal void Load(byte[] data){Bench.Require(data.Length==Bytes,"Payload mismatch.");fixed(byte* p=data)Check(cuMemcpyHtoD_v2(Data,(IntPtr)p,new UIntPtr((uint)data.Length)));}
        internal byte[] Read()=>Download(Base,Bytes+2*Guard);
        internal static bool Guarded(byte[] all,int bytes,byte poison,int offset,int count)
        {if(all.Length!=bytes+2*Guard||offset<0||count<0||(long)offset+count>bytes)return false;return all.AsSpan(0,Guard+offset).IndexOfAnyExcept(poison)<0&&all.AsSpan(Guard+offset+count).IndexOfAnyExcept(poison)<0;}
        internal bool Guards(byte[] all,int offset,int count)=>Guarded(all,Bytes,Poison,offset,count);
        public void Dispose(){if(Base!=IntPtr.Zero){try{sync();}finally{var p=Base;Base=IntPtr.Zero;Check(cuMemFree_v2(p));}}}
    }
}
