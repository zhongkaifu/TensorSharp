// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private static readonly string[] ArmNames={"actual-old8-old-addnorm","same-source8-old-addnorm","fused8-plain-norm"};
    private static readonly ulong[] GemmOffsets={0,8,16,20,24,32,40,48,52,56,64,72},GemmSizes={8,8,4,4,8,8,8,4,4,8,8,8};
    private static string[] PtxTypes(string text,string symbol)
    {var m=Regex.Match(text,@"\.visible\s+\.entry\s+"+Regex.Escape(symbol)+@"\s*\((.*?)\)",RegexOptions.Singleline);Need(m.Success,"Actual PTX entry missing: "+symbol);return Regex.Matches(m.Groups[1].Value,@"\.param\s+\.(\w+)\b").Select(x=>x.Groups[1].Value).ToArray();}
    private static void PtxAbi(string path)
    {
        string text=File.ReadAllText(path);
        foreach(var spec in new[]{("ts_q35_gemm_w4a4_8_residual_bf16",new[]{"u64","u64","u32","u32","u64","u64","u64","u32","f32","u64","u64","u64","u64"}),("ts_q35_ninfer_plain_norm_f32",new[]{"u64","u64","u64","u32","f32"})})
        {Need(PtxTypes(text,spec.Item1).SequenceEqual(spec.Item2),"Actual scalar PTX types differ: "+spec.Item1);}
    }
    private sealed unsafe class Modules:IDisposable
    {
        private readonly CudaModule _helper,_old,_new;
        internal readonly IntPtr Quant,OldNorm,Old,Same,Fused,Plain;
        internal long HostSubmissions;
        internal Modules(string helper,string helperSha,string old,string oldSha,string candidate,string candidateSha)
        {
            _helper=CudaModule.LoadFromFile(helper,identify:true);
            try{_old=CudaModule.LoadFromFile(old,identify:true);try{_new=CudaModule.LoadFromFile(candidate,identify:true);try{
                Need(S.Matches(_helper.ImageSha256,helperSha)&&S.Matches(_old.ImageSha256,oldSha)&&S.Matches(_new.ImageSha256,candidateSha),"Actual loaded image differs from byte pin.");
                Quant=Bind(_helper,"ts_ni_quant_nvfp4",new ulong[]{0,8,16,24,32,36,40},new ulong[]{8,8,8,8,4,4,4});
                OldNorm=Bind(_helper,"ts_ni_add_norm",new ulong[]{0,8,16,24,32,36},new ulong[]{8,8,8,8,4,4});
                Old=Bind(_old,"ts_q35_gemm_w4a4_8",GemmOffsets,GemmSizes);Same=Bind(_new,"ts_q35_gemm_w4a4_8",GemmOffsets,GemmSizes);
                Fused=Bind(_new,"ts_q35_gemm_w4a4_8_residual_bf16",GemmOffsets.Concat(new ulong[]{80}).ToArray(),GemmSizes.Concat(new ulong[]{8}).ToArray());
                Plain=Bind(_new,"ts_q35_ninfer_plain_norm_f32",new ulong[]{0,8,16,24,28},new ulong[]{8,8,8,4,4});
                foreach(IntPtr fn in new[]{Old,Same,Fused})CudaDriverApi.cuFuncSetAttribute(fn,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,43520).ThrowOnError();
            }catch{_new.Dispose();throw;}}catch{_old.Dispose();throw;}}catch{_helper.Dispose();throw;}
        }
        private static IntPtr Bind(CudaModule m,string symbol,ulong[] offsets,ulong[] sizes)
        {IntPtr fn=m.GetFunction(symbol);CudaDriverApi.cuFuncLoad(fn).ThrowOnError();for(uint i=0;i<offsets.Length;i++){CudaDriverApi.cuFuncGetParamInfo(fn,(UIntPtr)i,out var o,out var s).ThrowOnError();Need(o.ToUInt64()==offsets[i]&&s.ToUInt64()==sizes[i],symbol+" actual ABI differs.");}Need(CudaDriverApi.cuFuncGetParamInfo(fn,(UIntPtr)(uint)offsets.Length,out _,out _)==1,"Unexpected device argument.");return fn;}
        private void Launch(IntPtr f,uint gx,uint gy,uint gz,uint shared,IntPtr stream,void** a)
        {CudaDriverApi.cuLaunchKernel(f,gx,gy,gz,256,1,1,shared,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();HostSubmissions++;}
        internal void Quantize(Fixture f,Memory.Buffer input,Arm arm,IntPtr stream)
        {IntPtr x=input.Pointer,c=arm.Codes.Pointer,s=arm.Scales.Pointer,r=arm.RowScale.Pointer;int k=Width,t=Rows;float d=f.InputD;void** a=stackalloc void*[]{&x,&c,&s,&r,&k,&t,&d};Launch(Quant,(uint)((Count+255)/256),1,1,0,stream,a);}
        internal void Project(Fixture f,Memory.Buffer wc,Memory.Buffer ws,Arm arm,int kind,IntPtr stream)
        {Need(kind is >=0 and <=2,"Unknown arm.");IntPtr w=wc.Pointer,s=ws.Pointer,x=arm.Codes.Pointer,xs=arm.Scales.Pointer,r=arm.RowScale.Pointer,y=arm.Block.Pointer,p=arm.Partial.Pointer,c=arm.Counters.Pointer,residual=arm.X.Pointer;int n=N,k=Width,t=Rows;float alpha=f.Alpha;void** a=stackalloc void*[]{&w,&s,&n,&k,&x,&xs,&r,&t,&alpha,&y,&p,&c,&residual};Launch(kind==0?Old:kind==1?Same:Fused,(uint)(N/128),1,8,43520,stream,a);}
        internal void Norm(Arm arm,Memory.Buffer weight,int kind,IntPtr stream)
        {if(kind==2)PlainRun(arm.X.Pointer,weight.Pointer,arm.Y.Pointer,stream);else AddRun(arm.X.Pointer,arm.Block.Pointer,weight.Pointer,arm.Y.Pointer,stream);}
        internal void PlainRun(IntPtr x,IntPtr weight,IntPtr y,IntPtr stream)
        {int k=NormWidth;float e=Epsilon;void** a=stackalloc void*[]{&x,&weight,&y,&k,&e};Launch(Plain,Rows,1,1,0,stream,a);}
        internal void AddRun(IntPtr x,IntPtr delta,IntPtr weight,IntPtr y,IntPtr stream)
        {int k=NormWidth;float e=Epsilon;void** a=stackalloc void*[]{&x,&delta,&weight,&y,&k,&e};Launch(OldNorm,Rows,1,1,0,stream,a);}
        public void Dispose(){_new.Dispose();_old.Dispose();_helper.Dispose();}
    }
    private sealed class Arm
    {
        internal readonly Memory.Buffer Codes,Scales,RowScale,Block,Partial,Counters,X,Y;
        internal Arm(Memory m,string name)
        {Codes=m.Empty(name+"-codes",Count/2);Scales=m.Empty(name+"-scales",Count/16);RowScale=m.Empty(name+"-rowscale",Rows*4);Block=m.Empty(name+"-block",OutputCount*4);Partial=m.Empty(name+"-partial",8*OutputCount*4);Counters=m.Empty(name+"-counters",N/128*4);X=m.Empty(name+"-x",OutputCount*4);Y=m.Empty(name+"-norm",OutputCount*4);}
        internal void Reset(float[] seed,byte poison)
        {foreach(var b in new[]{Codes,Scales,RowScale,Block,Partial,X,Y})Memory.Poison(b,poison);Memory.Write(X.Pointer,Bytes(seed));Memory.Write(Counters.Allocation,Guarded(new byte[N/128*4]));}
        internal bool QuantBits(Fixture f)=>Memory.Body(Codes).AsSpan().SequenceEqual(f.Codes)&&Memory.Body(Scales).AsSpan().SequenceEqual(f.Scales)&&Memory.Body(RowScale).AsSpan().SequenceEqual(Bytes(Enumerable.Repeat(1f,Rows).ToArray()));
        internal bool CountersZero()=>Memory.Body(Counters).AsSpan().IndexOfAnyExcept((byte)0)<0;
    }
    private static class CopyGraphApi
    {
        [StructLayout(LayoutKind.Sequential)]internal struct Copy3D
        {internal ulong SX,SY,SZ,SLod;internal uint ST;internal IntPtr SH,SD,SA,SR;internal ulong SP,SHt,DX,DY,DZ,DLod;internal uint DT;internal IntPtr DH,DD,DA,DR;internal ulong DP,DHt,W,H,D;}
        [DllImport("libcuda.so.1")]internal static extern int cuGraphMemcpyNodeGetParams(IntPtr node,out Copy3D parameters);
        [DllImport("libcuda.so.1")]internal static extern int cuGraphGetEdges(IntPtr graph,IntPtr from,IntPtr to,ref UIntPtr count);
    }
    private sealed record Address(Memory.Buffer Wc,Memory.Buffer Ws,Memory.Buffer SeedX,Arm Old,Arm New);
    private sealed unsafe class ChainGraph:IDisposable
    {
        private IntPtr _graph,_exec;internal readonly object[] Nodes;
        internal ChainGraph(Modules modules,Fixture fixture,Memory.Buffer input,Memory.Buffer weight,Address[] addresses,bool candidate,IntPtr stream)
        {
            CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try{foreach(var a in addresses){var arm=candidate?a.New:a.Old;CudaDriverApi.cuMemcpyDtoDAsync(arm.X.Pointer,a.SeedX.Pointer,(UIntPtr)(OutputCount*4),stream).ThrowOnError();modules.Quantize(fixture,input,arm,stream);modules.Project(fixture,a.Wc,a.Ws,arm,candidate?2:0,stream);modules.Norm(arm,weight,candidate?2:0,stream);}}
            catch{_=CudaDriverApi.cuStreamEndCapture(stream,out var bad);if(bad!=IntPtr.Zero)_=CudaDriverApi.cuGraphDestroy(bad);throw;}
            CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                int size=4*addresses.Length;UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();Need(count.ToUInt64()==(ulong)size,"Complete reset/quant/GEMM/norm chain required.");IntPtr* nodes=stackalloc IntPtr[size];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();var named=new Dictionary<string,IntPtr>();var result=new List<object>();
                for(int z=0;z<size;z++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[z],out int type).ThrowOnError();string role="unknown";int index=-1;uint[]? geometry=null;
                    if(type==1)
                    {CopyGraphApi.cuGraphMemcpyNodeGetParams(nodes[z],out var p).ThrowOnError();index=Array.FindIndex(addresses,a=>p.SD==a.SeedX.Pointer);Need(index>=0,"Unknown reset source.");var arm=candidate?addresses[index].New:addresses[index].Old;Need(p.ST==2&&p.DT==2&&p.DD==arm.X.Pointer&&p.W==(ulong)(OutputCount*4)&&p.H==1&&p.D==1&&(p.SX|p.SY|p.SZ|p.SLod|p.DX|p.DY|p.DZ|p.DLod)==0,"Captured D2D reset differs.");role="reset";}
                    else
                    {
                        Need(type==0,"Hidden graph work refused.");GraphApi.cuGraphKernelNodeGetParams(nodes[z],out var p).ThrowOnError();Need(p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1,"Captured block/ABI differs.");
                        IntPtr A(int j)=>Marshal.ReadIntPtr(p.Parameters,j*IntPtr.Size);IntPtr P(int j)=>Marshal.ReadIntPtr(A(j));int I(int j)=>Marshal.ReadInt32(A(j));
                        if(p.Function==modules.Quant){index=Array.FindIndex(addresses,a=>P(1)==(candidate?a.New:a.Old).Codes.Pointer);role="quant";}
                        else if(p.Function==(candidate?modules.Fused:modules.Old)){index=Array.FindIndex(addresses,a=>P(9)==(candidate?a.New:a.Old).Block.Pointer);role="gemm";}
                        else if(p.Function==(candidate?modules.Plain:modules.OldNorm)){index=Array.FindIndex(addresses,a=>P(0)==(candidate?a.New:a.Old).X.Pointer);role="norm";}
                        Need(index>=0,"Captured unknown function/address.");var a=addresses[index];var arm=candidate?a.New:a.Old;
                        if(role=="quant")Need(p.GridX==(uint)((Count+255)/256)&&p.GridY==1&&p.GridZ==1&&p.Shared==0&&P(0)==input.Pointer&&P(1)==arm.Codes.Pointer&&P(2)==arm.Scales.Pointer&&P(3)==arm.RowScale.Pointer&&I(4)==Width&&I(5)==Rows&&I(6)==BitConverter.SingleToInt32Bits(fixture.InputD),"Quant node differs.");
                        if(role=="gemm")Need(p.GridX==(uint)(N/128)&&p.GridY==1&&p.GridZ==8&&p.Shared==43520&&P(0)==a.Wc.Pointer&&P(1)==a.Ws.Pointer&&I(2)==N&&I(3)==Width&&P(4)==arm.Codes.Pointer&&P(5)==arm.Scales.Pointer&&P(6)==arm.RowScale.Pointer&&I(7)==Rows&&I(8)==BitConverter.SingleToInt32Bits(fixture.Alpha)&&P(9)==arm.Block.Pointer&&P(10)==arm.Partial.Pointer&&P(11)==arm.Counters.Pointer&&(!candidate||P(12)==arm.X.Pointer),"GEMM node differs.");
                        if(role=="norm")Need(p.GridX==Rows&&p.GridY==1&&p.GridZ==1&&p.Shared==0&&(candidate?P(0)==arm.X.Pointer&&P(1)==weight.Pointer&&P(2)==arm.Y.Pointer&&I(3)==NormWidth&&I(4)==BitConverter.SingleToInt32Bits(Epsilon):P(0)==arm.X.Pointer&&P(1)==arm.Block.Pointer&&P(2)==weight.Pointer&&P(3)==arm.Y.Pointer&&I(4)==NormWidth&&I(5)==BitConverter.SingleToInt32Bits(Epsilon)),"Norm node differs.");geometry=new[]{p.GridX,p.GridY,p.GridZ,p.Shared};
                    }
                    Need(role!="unknown"&&named.TryAdd(index+"/"+role,nodes[z]),"Duplicate/unknown captured role.");result.Add(new{address=index,role,geometry});
                }
                var order=Enumerable.Range(0,addresses.Length).SelectMany(i=>new[]{i+"/reset",i+"/quant",i+"/gemm",i+"/norm"}).ToArray();Need(order.All(named.ContainsKey),"Captured role coverage incomplete.");UIntPtr edges=UIntPtr.Zero;CopyGraphApi.cuGraphGetEdges(_graph,IntPtr.Zero,IntPtr.Zero,ref edges).ThrowOnError();Need(edges.ToUInt64()==(ulong)(size-1),"Captured chain dependencies differ.");IntPtr* from=stackalloc IntPtr[size-1];IntPtr* to=stackalloc IntPtr[size-1];CopyGraphApi.cuGraphGetEdges(_graph,(IntPtr)from,(IntPtr)to,ref edges).ThrowOnError();for(int i=1;i<size;i++){bool found=false;for(int j=0;j<size-1;j++)found|=from[j]==named[order[i-1]]&&to[j]==named[order[i]];Need(found,"A reset/producer/consumer dependency is missing.");}
                Nodes=result.ToArray();CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }catch{Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
}
