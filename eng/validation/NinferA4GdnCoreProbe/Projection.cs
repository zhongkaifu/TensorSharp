// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private sealed unsafe class Modules:IDisposable
    {
        private readonly CudaModule _helper,_old,_candidate;
        internal readonly IntPtr Quant,Round,SwiGlu,Old,SameSource,Candidate;
        internal Modules(string helper,string helperSha,string old,string oldSha,string candidate,string candidateSha)
        {
            _helper=CudaModule.LoadFromFile(helper,identify:true);
            try
            {
                _old=CudaModule.LoadFromFile(old,identify:true);
                try
                {
                    _candidate=CudaModule.LoadFromFile(candidate,identify:true);
                    try
                    {
                        Need(S.Matches(_helper.ImageSha256,helperSha)&&S.Matches(_old.ImageSha256,oldSha)&&S.Matches(_candidate.ImageSha256,candidateSha),"Actual loaded module differs from explicit byte pins.");
                        Quant=Bind(_helper,"ts_ni_quant_nvfp4",new ulong[]{0,8,16,24,32,36,40},new ulong[]{8,8,8,8,4,4,4});Round=Bind(_helper,"ts_ni_round",new ulong[]{0,8},new ulong[]{8,8});
                        SwiGlu=Bind(_helper,"ts_ni_swiglu",new ulong[]{0,8,16},new ulong[]{8,8,4});
                        ulong[] o={0,8,16,20,24,32,40,48,52,56,64,72},s={8,8,4,4,8,8,8,4,4,8,8,8};Old=Bind(_old,LegacyPlan.Symbol,o,s);SameSource=Bind(_candidate,ProjectionPlan(1,LegacySchedule).Symbol,o,s);Candidate=Bind(_candidate,ProjectionPlan(2,LegacySchedule).Symbol,o,s);
                        CudaDriverApi.cuFuncSetAttribute(Old,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,checked((int)LegacyPlan.Shared)).ThrowOnError();CudaDriverApi.cuFuncSetAttribute(SameSource,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,76800).ThrowOnError();CudaDriverApi.cuFuncSetAttribute(Candidate,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,43520).ThrowOnError();
                    }catch{_candidate.Dispose();throw;}
                }catch{_old.Dispose();throw;}
            }catch{_helper.Dispose();throw;}
        }
        private static IntPtr Bind(CudaModule m,string name,ulong[] offsets,ulong[] sizes)
        {var f=m.GetFunction(name);CudaDriverApi.cuFuncLoad(f).ThrowOnError();for(uint i=0;i<offsets.Length;i++){CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)i,out var o,out var s).ThrowOnError();Need(o.ToUInt64()==offsets[i]&&s.ToUInt64()==sizes[i],name+" ABI differs.");}Need(CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)(uint)offsets.Length,out _,out _)==1,"Extra device argument.");return f;}
        internal void Quantize(Fixture f,Memory.Buffer input,GemmArm arm,IntPtr stream)
        {IntPtr x=input.Pointer,c=arm.Codes.Pointer,s=arm.Scales.Pointer,r=arm.RowScale.Pointer;int k=Width,t=Rows;float d=f.InputD;void**a=stackalloc void*[]{&x,&c,&s,&r,&k,&t,&d};CudaDriverApi.cuLaunchKernel(Quant,(uint)((Count+255)/256),1,1,256,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
        internal void Project(Fixture f,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream)
        {IntPtr w=wc.Pointer,s=ws.Pointer,x=arm.Codes.Pointer,xs=arm.Scales.Pointer,r=arm.RowScale.Pointer,y=arm.Output.Pointer,p=arm.Partial.Pointer,c=arm.Counters.Pointer;int n=N,k=Width,t=Rows;float alpha=f.Alpha;void**a=stackalloc void*[]{&w,&s,&n,&k,&x,&xs,&r,&t,&alpha,&y,&p,&c};CudaDriverApi.cuLaunchKernel(kind==0?Old:kind==1?SameSource:Candidate,(uint)(N/128),1,(uint)Geometry.Splits,256,1,1,ProjectionPlan(kind,LegacySchedule).Shared,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
        internal void Consume(GemmArm arm,IntPtr stream)
        {
            if(Geometry.Consumer=="none")return;
            IntPtr y=arm.Output.Pointer;
            if(Geometry.Consumer=="swiglu"){IntPtr act=arm.Final.Pointer;int ff=N/2;void**a=stackalloc void*[]{&y,&act,&ff};CudaDriverApi.cuLaunchKernel(SwiGlu,(uint)((ff+255)/256),Rows,1,256,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
            else{long count=OutputCount;void**a=stackalloc void*[]{&y,&count};CudaDriverApi.cuLaunchKernel(Round,(uint)((OutputCount+255)/256),1,1,256,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
        }
        internal void Chain(Fixture f,Memory.Buffer x,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream){Quantize(f,x,arm,stream);Project(f,wc,ws,arm,kind,stream);Consume(arm,stream);}
        public void Dispose(){_candidate.Dispose();_old.Dispose();_helper.Dispose();}
    }
    private sealed class GemmArm
    {
        internal readonly Memory.Buffer Codes,Scales,RowScale,Output,Partial,Counters;
        private readonly Memory.Buffer? _activation;
        internal Memory.Buffer Final=>_activation??Output;
        internal GemmArm(Memory m,string name){Codes=m.Empty(name+"-codes",Count/2);Scales=m.Empty(name+"-scales",Count/16);RowScale=m.Empty(name+"-rows",Rows*4);Output=m.Empty(name+"-output",OutputCount*4);Partial=m.Empty(name+"-partial",Geometry.Splits==1?16:Geometry.Splits*OutputCount*4);Counters=m.Empty(name+"-counters",N/128*4);if(Geometry.Consumer=="swiglu")_activation=m.Empty(name+"-activation",FinalCount*4);}
        internal void Reset(byte poison){foreach(var b in new[]{Codes,Scales,RowScale,Output,Partial})Memory.Poison(b,poison);if(_activation!=null)Memory.Poison(_activation,poison);Memory.Write(Counters.Allocation,Guarded(new byte[N/128*4]));}
        internal bool QuantBits(Fixture f)=>Memory.Body(Codes).AsSpan().SequenceEqual(f.Codes)&&Memory.Body(Scales).AsSpan().SequenceEqual(f.Scales)&&Memory.Body(RowScale).AsSpan().SequenceEqual(Bytes(Enumerable.Repeat(1f,Rows).ToArray()));
        internal bool CountersZero()=>Memory.Body(Counters).AsSpan().IndexOfAnyExcept((byte)0)<0;
    }
    private sealed unsafe class ChainGraph:IDisposable
    {
        private IntPtr _graph,_exec;internal object[] Nodes{get;}
        internal ChainGraph(Modules modules,Fixture fixture,Memory.Buffer x,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream)
        {
            CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try{modules.Chain(fixture,x,wc,ws,arm,kind,stream);}catch{_=CudaDriverApi.cuStreamEndCapture(stream,out var bad);if(bad!=IntPtr.Zero)_=CudaDriverApi.cuGraphDestroy(bad);throw;}
            CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                Need(kind is 0 or 2,$"Timing only compares {LegacyPlan.Arm} with candidate8.");UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();Need(count.ToUInt64()==(ulong)NodesPerChain,"Only the real role's complete consumer chain is admitted.");IntPtr* nodes=stackalloc IntPtr[NodesPerChain];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();var seen=new HashSet<string>();var result=new List<object>();
                for(int i=0;i<NodesPerChain;i++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[i],out int type).ThrowOnError();Need(type==0,"Hidden graph work refused.");GraphApi.cuGraphKernelNodeGetParams(nodes[i],out var p).ThrowOnError();string role=p.Function==modules.Quant?"quant":p.Function==modules.Round?"round":p.Function==modules.SwiGlu?"swiglu":p.Function==(kind==0?modules.Old:modules.Candidate)?"gemm":"unknown";
                    Need(role!="unknown"&&seen.Add(role)&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1&&p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero,"Unexpected captured kernel/ABI.");
                    IntPtr A(int j)=>Marshal.ReadIntPtr(p.Parameters,j*IntPtr.Size);IntPtr P(int j)=>Marshal.ReadIntPtr(A(j));int I(int j)=>Marshal.ReadInt32(A(j));
                    if(role=="quant")Need(p.GridX==(uint)((Count+255)/256)&&p.GridY==1&&p.GridZ==1&&p.Shared==0&&P(0)==x.Pointer&&P(1)==arm.Codes.Pointer&&P(2)==arm.Scales.Pointer&&P(3)==arm.RowScale.Pointer&&I(4)==Width&&I(5)==Rows&&I(6)==BitConverter.SingleToInt32Bits(fixture.InputD),"Quant capture differs.");
                    if(role=="gemm")Need(p.GridX==(uint)(N/128)&&p.GridY==1&&p.GridZ==(uint)Geometry.Splits&&p.Shared==ProjectionPlan(kind,LegacySchedule).Shared&&P(0)==wc.Pointer&&P(1)==ws.Pointer&&I(2)==N&&I(3)==Width&&P(4)==arm.Codes.Pointer&&P(5)==arm.Scales.Pointer&&P(6)==arm.RowScale.Pointer&&I(7)==Rows&&I(8)==BitConverter.SingleToInt32Bits(fixture.Alpha)&&P(9)==arm.Output.Pointer&&P(10)==arm.Partial.Pointer&&P(11)==arm.Counters.Pointer,"GEMM capture differs.");
                    if(role=="round")Need(Geometry.Consumer=="round"&&p.GridX==(uint)((OutputCount+255)/256)&&p.GridY==1&&p.GridZ==1&&p.Shared==0&&P(0)==arm.Output.Pointer&&Marshal.ReadInt64(A(1))==OutputCount,"Round capture differs.");
                    if(role=="swiglu")Need(Geometry.Consumer=="swiglu"&&p.GridX==(uint)((N/2+255)/256)&&p.GridY==Rows&&p.GridZ==1&&p.Shared==0&&P(0)==arm.Output.Pointer&&P(1)==arm.Final.Pointer&&I(2)==N/2,"SwiGLU capture differs.");
                    result.Add(new{role,grid=new[]{p.GridX,p.GridY,p.GridZ},block=new[]{p.BlockX,p.BlockY,p.BlockZ},dynamicShared=p.Shared});
                }
                Need(seen.SetEquals(Geometry.Consumer=="none"?new[]{"quant","gemm"}:new[]{"quant","gemm",Geometry.Consumer}),"Complete actual consumer roles mandatory.");Nodes=result.ToArray();CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }catch{Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
}
