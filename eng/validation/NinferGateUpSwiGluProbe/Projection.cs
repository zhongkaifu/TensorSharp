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
        private readonly CudaModule _helper,_old,_candidate;internal readonly IntPtr Quant,SwiGlu,Old,SameSource,Candidate;internal object Resources{get;}
        internal Modules(string helper,string helperSha,string old,string oldSha,string candidate,string candidateSha)
        {
            FixedGeometry(Rows,N,Width,Geometry.Splits);_helper=CudaModule.LoadFromFile(helper,identify:true);
            try
            {
                _old=CudaModule.LoadFromFile(old,identify:true);try
                {
                    _candidate=CudaModule.LoadFromFile(candidate,identify:true);try
                    {
                        Need(S.Matches(_helper.ImageSha256,helperSha)&&S.Matches(_old.ImageSha256,oldSha)&&S.Matches(_candidate.ImageSha256,candidateSha),"Loaded module image differs from explicit pins.");
                        Quant=Bind(_helper,"ts_ni_quant_nvfp4",new ulong[]{0,8,16,24,32,36,40},new ulong[]{8,8,8,8,4,4,4});SwiGlu=Bind(_helper,"ts_ni_swiglu",new ulong[]{0,8,16},new ulong[]{8,8,4});
                        ulong[] offsets={0,8,16,20,24,32,40,48,52,56,64,72},sizes={8,8,4,4,8,8,8,4,4,8,8,8};Old=Bind(_old,"ts_q35_gemm_w4a4_8",offsets,sizes);SameSource=Bind(_candidate,"ts_q35_gemm_w4a4_8",offsets,sizes);Candidate=Bind(_candidate,FusedSymbol,offsets,sizes);
                        foreach(IntPtr f in new[]{Old,SameSource,Candidate})CudaDriverApi.cuFuncSetAttribute(f,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,(int)DynamicShared).ThrowOnError();
                        Resources=new[]{Old,SameSource,Candidate}.Select((f,i)=>{CudaDriverApi.cuFuncGetAttribute(out int regs,4,f).ThrowOnError();CudaDriverApi.cuFuncGetAttribute(out int local,3,f).ThrowOnError();CudaDriverApi.cuFuncGetAttribute(out int shared,1,f).ThrowOnError();Need(local==0,"Predeclared zero local-memory/spill scope required.");return new{arm=ArmNames[i],registers=regs,localBytes=local,staticSharedBytes=shared,dynamicSharedBytes=DynamicShared};}).ToArray();
                    }catch{_candidate.Dispose();throw;}
                }catch{_old.Dispose();throw;}
            }catch{_helper.Dispose();throw;}
        }
        private static IntPtr Bind(CudaModule module,string name,ulong[] offsets,ulong[] sizes)
        {
            IntPtr f=module.GetFunction(name);CudaDriverApi.cuFuncLoad(f).ThrowOnError();for(uint i=0;i<offsets.Length;i++){CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)i,out var o,out var s).ThrowOnError();Need(o.ToUInt64()==offsets[i]&&s.ToUInt64()==sizes[i],name+" ABI differs.");}Need(CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)(uint)offsets.Length,out _,out _)==1,"Unexpected extra device argument.");return f;
        }
        internal void Quantize(Fixture f,Memory.Buffer input,GemmArm arm,IntPtr stream)
        {IntPtr x=input.Pointer,c=arm.Codes.Pointer,s=arm.Scales.Pointer,r=arm.RowScale.Pointer;int k=Width,t=Rows;float d=f.InputD;void** a=stackalloc void*[]{&x,&c,&s,&r,&k,&t,&d};CudaDriverApi.cuLaunchKernel(Quant,(uint)((Count+255)/256),1,1,256,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
        internal void Project(Fixture f,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream)
        {
            FixedGeometry(Rows,N,Width,Geometry.Splits);Need(kind is >=0 and <=2&&arm.IsFused==(kind==2),"Wrong projection/output arm.");IntPtr w=wc.Pointer,s=ws.Pointer,x=arm.Codes.Pointer,xs=arm.Scales.Pointer,r=arm.RowScale.Pointer,y=(kind==2?arm.Final:arm.Output).Pointer,p=arm.Partial.Pointer,c=arm.Counters.Pointer;int n=N,k=Width,t=Rows;float alpha=f.Alpha;void** a=stackalloc void*[]{&w,&s,&n,&k,&x,&xs,&r,&t,&alpha,&y,&p,&c};CudaDriverApi.cuLaunchKernel(kind==0?Old:kind==1?SameSource:Candidate,(uint)(N/128),1,1,256,1,1,DynamicShared,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();
        }
        internal void Consume(GemmArm arm,IntPtr stream)
        {Need(!arm.IsFused,"Fused arm must not launch a second SwiGLU.");IntPtr raw=arm.Output.Pointer,final=arm.Final.Pointer;int ff=N/2;void** a=stackalloc void*[]{&raw,&final,&ff};CudaDriverApi.cuLaunchKernel(SwiGlu,(uint)((ff+255)/256),Rows,1,256,1,1,0,stream,(IntPtr)a,IntPtr.Zero).ThrowOnError();}
        internal void Chain(Fixture f,Memory.Buffer x,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream){Quantize(f,x,arm,stream);Project(f,wc,ws,arm,kind,stream);if(kind!=2)Consume(arm,stream);}
        public void Dispose(){_candidate.Dispose();_old.Dispose();_helper.Dispose();}
    }
    private sealed class GemmArm
    {
        internal readonly Memory.Buffer Codes,Scales,RowScale,Output,Final,Partial,Counters;internal readonly bool IsFused;
        internal GemmArm(Memory memory,string name,bool fused)
        {IsFused=fused;Codes=memory.Empty(name+"-codes",Count/2);Scales=memory.Empty(name+"-scales",Count/16);RowScale=memory.Empty(name+"-row-scale",Rows*4);Output=memory.Empty(name+"-raw-f32",fused?16:OutputCount*4);Final=memory.Empty(name+"-final",FinalCount*4);Partial=memory.Empty(name+"-unused-partial",16);Counters=memory.Empty(name+"-unused-counters",N/128*4);}
        internal void Reset(byte poison){foreach(var b in new[]{Codes,Scales,RowScale,Output,Final,Partial,Counters})Memory.Poison(b,poison);}
        internal bool QuantBits(Fixture f)=>Memory.Body(Codes).AsSpan().SequenceEqual(f.Codes)&&Memory.Body(Scales).AsSpan().SequenceEqual(f.Scales)&&Memory.Body(RowScale).AsSpan().SequenceEqual(Bytes(Enumerable.Repeat(1f,Rows).ToArray()));
    }
    private static class DependencyApi
    { [DllImport("libcuda.so.1")]internal static extern int cuGraphGetEdges(IntPtr graph,IntPtr from,IntPtr to,ref UIntPtr count); }
    private sealed unsafe class ChainGraph:IDisposable
    {
        private IntPtr _graph,_exec;internal readonly object[] Nodes;internal readonly object Dependencies;
        internal ChainGraph(Modules modules,Fixture f,Memory.Buffer x,Memory.Buffer wc,Memory.Buffer ws,GemmArm arm,int kind,IntPtr stream)
        {
            Need(kind is 0 or 2,"Timing compares only actualEA0 with fused2.");int countExpected=kind==0?3:2;CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try{modules.Chain(f,x,wc,ws,arm,kind,stream);}catch{_=CudaDriverApi.cuStreamEndCapture(stream,out var bad);if(bad!=IntPtr.Zero)_=CudaDriverApi.cuGraphDestroy(bad);throw;}CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();Need(count.ToUInt64()==(ulong)countExpected,"Only the exact complete chain is admitted.");IntPtr* nodes=stackalloc IntPtr[countExpected];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();var roles=new Dictionary<string,IntPtr>();var records=new List<object>();
                for(int i=0;i<countExpected;i++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[i],out int type).ThrowOnError();Need(type==0,"Hidden graph work refused.");GraphApi.cuGraphKernelNodeGetParams(nodes[i],out var p).ThrowOnError();string role=p.Function==modules.Quant?"quant":p.Function==modules.SwiGlu?"swiglu":p.Function==(kind==0?modules.Old:modules.Candidate)?(kind==0?"gemm":"fused-gemm-swiglu"):"unknown";
                    Need(role!="unknown"&&roles.TryAdd(role,nodes[i])&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1&&p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero,"Unexpected captured function/block/ABI.");IntPtr A(int j)=>Marshal.ReadIntPtr(p.Parameters,j*IntPtr.Size);IntPtr P(int j)=>Marshal.ReadIntPtr(A(j));int I(int j)=>Marshal.ReadInt32(A(j));
                    if(role=="quant")Need(p.GridX==(uint)((Count+255)/256)&&p.GridY==1&&p.GridZ==1&&p.Shared==0&&P(0)==x.Pointer&&P(1)==arm.Codes.Pointer&&P(2)==arm.Scales.Pointer&&P(3)==arm.RowScale.Pointer&&I(4)==Width&&I(5)==Rows&&I(6)==BitConverter.SingleToInt32Bits(f.InputD),"Actual quant arguments differ.");
                    if(role is "gemm" or "fused-gemm-swiglu")Need(p.GridX==272&&p.GridY==1&&p.GridZ==1&&p.Shared==DynamicShared&&P(0)==wc.Pointer&&P(1)==ws.Pointer&&I(2)==N&&I(3)==Width&&P(4)==arm.Codes.Pointer&&P(5)==arm.Scales.Pointer&&P(6)==arm.RowScale.Pointer&&I(7)==Rows&&I(8)==BitConverter.SingleToInt32Bits(f.Alpha)&&P(9)==(kind==0?arm.Output:arm.Final).Pointer&&P(10)==arm.Partial.Pointer&&P(11)==arm.Counters.Pointer,"Actual GEMM/fused arguments differ.");
                    if(role=="swiglu")Need(kind==0&&p.GridX==68&&p.GridY==8&&p.GridZ==1&&p.Shared==0&&P(0)==arm.Output.Pointer&&P(1)==arm.Final.Pointer&&I(2)==17408,"Original helper SwiGLU arguments differ.");
                    records.Add(new{role,grid=new[]{p.GridX,p.GridY,p.GridZ},block=new[]{p.BlockX,p.BlockY,p.BlockZ},dynamicSharedBytes=p.Shared,argumentCount=role=="quant"?7:role=="swiglu"?3:12});
                }
                string[] order=kind==0?new[]{"quant","gemm","swiglu"}:new[]{"quant","fused-gemm-swiglu"};Need(roles.Keys.Order().SequenceEqual(order.Order()),"Missing exact chain role.");UIntPtr edgeCount=UIntPtr.Zero;DependencyApi.cuGraphGetEdges(_graph,IntPtr.Zero,IntPtr.Zero,ref edgeCount).ThrowOnError();Need(edgeCount.ToUInt64()==(ulong)(countExpected-1),"Full producer/consumer dependency count differs.");IntPtr* from=stackalloc IntPtr[countExpected-1];IntPtr* to=stackalloc IntPtr[countExpected-1];DependencyApi.cuGraphGetEdges(_graph,(IntPtr)from,(IntPtr)to,ref edgeCount).ThrowOnError();for(int i=1;i<order.Length;i++){bool found=false;for(int j=0;j<countExpected-1;j++)found|=from[j]==roles[order[i-1]]&&to[j]==roles[order[i]];Need(found,"Quant/GEMM/consumer dependency missing.");}
                Nodes=records.ToArray();Dependencies=new{verified=true,order,edges=countExpected-1,hiddenResets=0};CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }catch{Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
}
