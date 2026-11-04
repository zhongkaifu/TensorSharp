// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Runtime;

internal static partial class Program
{
    private static class CopyGraph
    {
        [StructLayout(LayoutKind.Sequential)]internal struct Params
        {internal ulong SX,SY,SZ,SLod;internal uint ST;internal IntPtr SH,SD,SA,SR;internal ulong SP,SHt,DX,DY,DZ,DLod;internal uint DT;internal IntPtr DH,DD,DA,DR;internal ulong DP,DHt,W,H,D;}
        [DllImport("libcuda.so.1")]internal static extern int cuGraphMemcpyNodeGetParams(IntPtr node,out Params parameters);
    }
    private sealed record Address(Case Case,Memory.Buffer Weight,Memory.Buffer Input,Memory.Buffer Norm,Memory.Buffer Seed,Arm Old,Arm New,Captures OldExpected,Captures NewExpected);
    private sealed unsafe class Chain:IDisposable
    {
        private IntPtr _graph,_exec;
        internal readonly object[] Nodes;
        internal Chain(Modules modules,Address[] addresses,bool candidate,IntPtr stream)
        {
            var s=modules.Geometry;CudaDriverApi.cuStreamBeginCapture(stream,CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
            try{foreach(var a in addresses){var arm=candidate?a.New:a.Old;if(s.Name=="output")CudaDriverApi.cuMemcpyDtoDAsync(arm.X.Pointer,a.Seed.Pointer,(UIntPtr)(8*s.N*4),stream).ThrowOnError();modules.Project(arm,a.Input,a.Weight,candidate,stream);modules.Consume(arm,a.Norm,stream);}}
            catch{_=CudaDriverApi.cuStreamEndCapture(stream,out var bad);if(bad!=IntPtr.Zero)_=CudaDriverApi.cuGraphDestroy(bad);throw;}
            CudaDriverApi.cuStreamEndCapture(stream,out _graph).ThrowOnError();
            try
            {
                UIntPtr count=UIntPtr.Zero;GraphApi.cuGraphGetNodes(_graph,IntPtr.Zero,ref count).ThrowOnError();int n=checked((int)count.ToUInt64());Need(n>=addresses.Length*(s.Name=="output"?4:3),"Full cast/GEMM/consumer/reset graph absent.");IntPtr* nodes=stackalloc IntPtr[n];GraphApi.cuGraphGetNodes(_graph,(IntPtr)nodes,ref count).ThrowOnError();var result=new List<object>();int casts=0,consumers=0,cores=0,resets=0;
                for(int i=0;i<n;i++)
                {
                    GraphApi.cuGraphNodeGetType(nodes[i],out int type).ThrowOnError();
                    if(type==1)
                    {CopyGraph.cuGraphMemcpyNodeGetParams(nodes[i],out var cp).ThrowOnError();int a=Array.FindIndex(addresses,x=>x.Seed.Pointer==cp.SD);if(a>=0){var arm=candidate?addresses[a].New:addresses[a].Old;Need(s.Name=="output"&&cp.ST==2&&cp.DT==2&&cp.DD==arm.X.Pointer&&cp.W==(ulong)(8*s.N*4)&&cp.H==1&&cp.D==1&&(cp.SX|cp.SY|cp.SZ|cp.SLod|cp.DX|cp.DY|cp.DZ|cp.DLod)==0,"Captured reset_x differs.");resets++;}else Need(!candidate,"Unknown candidate copy.");result.Add(new{type=a>=0?"exact-D2D-reset":"actual-cuBLAS-library-copy",nodeIndex=i,address=a});continue;}
                    Need(type==0,"Unsupported hidden graph work; do not qualify incomplete chain.");GraphApi.cuGraphKernelNodeGetParams(nodes[i],out var p).ThrowOnError();string role="actual-cuBLAS-library-kernel";int address=-1;if(p.Function==modules.Cast||p.Function==modules.Candidate||p.Function==modules.Round||p.Function==modules.Norm)Need(p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero,"Known node requires original parameter-array ABI.");
                    IntPtr A(int j)=>Marshal.ReadIntPtr(p.Parameters,j*IntPtr.Size);IntPtr P(int j)=>Marshal.ReadIntPtr(A(j));long L(int j)=>Marshal.ReadInt64(A(j));int I(int j)=>Marshal.ReadInt32(A(j));
                    if(p.Function==modules.Cast)
                    {role="cast";casts++;address=Array.FindIndex(addresses,x=>x.Input.Pointer==P(0));Need(address>=0&&p.Parameters!=IntPtr.Zero&&p.Extra==IntPtr.Zero&&p.GridX==(uint)((8*s.K+255)/256)&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1&&p.Shared==0&&L(2)==8L*s.K,"Captured original cast differs.");if(candidate)Need(P(1)==addresses[address].New.Cast.Pointer,"Candidate cast destination differs.");}
                    else if(candidate&&p.Function==modules.Candidate)
                    {role="owned-f32-original-core";cores++;address=Array.FindIndex(addresses,x=>x.New.Projection.Pointer==P(2));Need(address>=0&&P(0)==addresses[address].New.Cast.Pointer&&P(1)==addresses[address].Weight.Pointer&&p.Extra==IntPtr.Zero&&p.GridX==modules.Blocks&&p.GridY==1&&p.GridZ==1&&p.BlockX==modules.Threads&&p.BlockY==1&&p.BlockZ==1&&p.Shared==modules.Shared&&(!modules.Mma||I(3)==8),"Candidate full core ABI/geometry differs.");}
                    else if(p.Function==(s.Name=="input"?modules.Round:modules.Norm))
                    {
                        role="original-consumer";consumers++;address=Array.FindIndex(addresses,x=>(candidate?x.New:x.Old).Projection.Pointer==P(s.Name=="input"?0:1));Need(address>=0&&p.Extra==IntPtr.Zero&&p.GridY==1&&p.GridZ==1&&p.BlockX==256&&p.BlockY==1&&p.BlockZ==1&&p.Shared==0,"Captured original consumer differs.");var a=addresses[address];var arm=candidate?a.New:a.Old;
                        if(s.Name=="input")Need(p.GridX==(uint)((8*s.N+255)/256)&&L(1)==8L*s.N,"Original Round count differs.");else Need(p.GridX==8&&P(0)==arm.X.Pointer&&P(2)==a.Norm.Pointer&&P(3)==arm.Y.Pointer&&I(4)==5120&&I(5)==BitConverter.SingleToInt32Bits(Epsilon),"Original AddNorm operands differ.");
                    }
                    else{Need(!candidate,"Unknown kernel in candidate chain.");cores++;}
                    result.Add(new{nodeIndex=i,role,address,grid=new[]{p.GridX,p.GridY,p.GridZ},block=new[]{p.BlockX,p.BlockY,p.BlockZ},p.Shared});
                }
                Need(casts==addresses.Length&&consumers==addresses.Length&&cores>=addresses.Length&&(!candidate||cores==addresses.Length)&&resets==(s.Name=="output"?addresses.Length:0),"Every physical address needs complete producer/consumer nodes.");Nodes=result.ToArray();CudaDriverApi.cuGraphInstantiateWithFlags(out _exec,_graph,0).ThrowOnError();
            }catch{Dispose();throw;}
        }
        internal void Run(IntPtr stream)=>CudaDriverApi.cuGraphLaunch(_exec,stream).ThrowOnError();
        public void Dispose(){if(_exec!=IntPtr.Zero){CudaDriverApi.cuGraphExecDestroy(_exec).ThrowOnError();_exec=IntPtr.Zero;}if(_graph!=IntPtr.Zero){CudaDriverApi.cuGraphDestroy(_graph).ThrowOnError();_graph=IntPtr.Zero;}}
    }
    private sealed class Events:IDisposable
    {
        internal readonly IntPtr Start,End;internal Events(){CudaDriverApi.cuEventCreate(out Start,0).ThrowOnError();try{CudaDriverApi.cuEventCreate(out End,0).ThrowOnError();}catch{CudaDriverApi.cuEventDestroy(Start);throw;}}
        internal float Run(Chain chain,IntPtr stream){CudaDriverApi.cuEventRecord(Start,stream).ThrowOnError();for(int i=0;i<64;i++)chain.Run(stream);CudaDriverApi.cuEventRecord(End,stream).ThrowOnError();CudaDriverApi.cuEventSynchronize(End).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out float ms,Start,End).ThrowOnError();Need(float.IsFinite(ms)&&ms>0,"Invalid actual CUDA event duration.");return ms/64f;}
        public void Dispose(){CudaDriverApi.cuEventDestroy(End).ThrowOnError();CudaDriverApi.cuEventDestroy(Start).ThrowOnError();}
    }
    private static double Median(IEnumerable<double> v){var a=v.Order().ToArray();return a.Length%2==1?a[a.Length/2]:(a[a.Length/2-1]+a[a.Length/2])/2;}
    private static Captures Direct(Modules m,Address a,bool candidate,IntPtr stream)
    {var arm=candidate?a.New:a.Old;Memory.Write(arm.X.Pointer,File.ReadAllBytes(a.Case.Seed.Path));m.Project(arm,a.Input,a.Weight,candidate,stream);CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();byte[] pre=Memory.Body(arm.Projection);m.Consume(arm,a.Norm,stream);CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();return new(pre,arm.Output(m.Geometry),m.Geometry.Name=="output"?Memory.Body(arm.X):null);}
    private static object Timings(Shape s,Prepared prepared,OracleSet oracle,System.Text.Json.JsonElement native,string helper,string helperSha,string candidate,string candidateSha,string output,Dictionary<string,string> files)
    {
        var records=new List<object>();var modes=new List<object>();bool allWins=true;
        foreach(int count in new[]{1,s.RealCount})
        {
            using var reader=new NinferArtifactReader(prepared.Artifact);using var stream=new Stream();using var m=new Modules(helper,helperSha,candidate,candidateSha,native,s);using var memory=new Memory();var addresses=new List<Address>();
            foreach(var c in prepared.Cases.Where(x=>x.Shape==s.Name&&x.Kind=="real").Take(count))
            {
                var o=oracle.Cases.Single(x=>x.Id==c.Id);byte[] w=Weight(c,s,reader);Need(Sha(w)==o.WeightSha256,"Timing actual weights differ.");var weight=memory.Put(c.Id+"-W",w);w=Array.Empty<byte>();var input=memory.Put(c.Id+"-input",File.ReadAllBytes(c.Input.Path));var norm=memory.Put(c.Id+"-norm",File.ReadAllBytes(c.Norm.Path));var seed=memory.Put(c.Id+"-seed",File.ReadAllBytes(c.Seed.Path));var old=new Arm(memory,c.Id+"-old",s);var next=new Arm(memory,c.Id+"-new",s);var a=new Address(c,weight,input,norm,seed,old,next,null!,null!);var oldData=Direct(m,a,false,stream.Pointer);var newData=Direct(m,a,true,stream.Pointer);Need(Dot(oldData.Projection,MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(o.Ideal.Path)).ToArray(),MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(o.L1.Path)).ToArray(),s.K,false).Passed&&Dot(newData.Projection,MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(o.Ideal.Path)).ToArray(),MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(o.L1.Path)).ToArray(),s.K,false).Passed&&Consumer(c,s,oldData.Projection,oldData.Consumer,oldData.Residual)&&Consumer(c,s,newData.Projection,newData.Consumer,newData.Residual),"Fresh timing address full numeric gate failed.");Need(!m.Mma||(oldData.Projection.AsSpan().SequenceEqual(newData.Projection)&&oldData.Consumer.AsSpan().SequenceEqual(newData.Consumer)&&(oldData.Residual==null||oldData.Residual.AsSpan().SequenceEqual(newData.Residual!))),"Fresh MMA timing address does not preserve actual old projection/consumer/residual bits.");addresses.Add(a with{OldExpected=oldData,NewExpected=newData});
            }
            var array=addresses.ToArray();Need(array.Length==count,"Timing real-address matrix incomplete.");using var oldGraph=new Chain(m,array,false,stream.Pointer);using var newGraph=new Chain(m,array,true,stream.Pointer);using var events=new Events();var rounds=new List<(double Old,double New)>();var oldTimes=new List<double>();var newTimes=new List<double>();
            for(int round=0;round<5;round++)
            {
                var a=new List<double>();var b=new List<double>();foreach(bool next in new[]{false,true,true,false})
                {
                    Chain g=next?newGraph:oldGraph;g.Run(stream.Pointer);stream.Sync();float time=events.Run(g,stream.Pointer);stream.Sync();var raw=new List<object>();
                    foreach(var address in array)
                    {
                        var arm=next?address.New:address.Old;var expected=next?address.NewExpected:address.OldExpected;byte[] post=arm.Output(s);byte[]? residual=s.Name=="output"?Memory.Body(arm.X):null;Need(post.AsSpan().SequenceEqual(expected.Consumer)&&(residual==null||residual.AsSpan().SequenceEqual(expected.Residual!)),"Captured full chain changed own admitted output.");
                        if(s.Name=="output")Need(Memory.Body(arm.Projection).AsSpan().SequenceEqual(expected.Projection),"Captured raw F32 projection changed.");raw.Add(new{address.Case.Id,consumer=Capture(post,output,files),residual=residual==null?null:Capture(residual,output,files)});
                    }
                    Need(memory.Guards()&&memory.Inputs(),"Timing guard/input mutation.");records.Add(new{physicalAddresses=count,round,arm=next?"candidate":"actual-cuBLAS",passMs=time,replays=64,outsideWarmup=1,fullOutputAndInputChecks=true,raw});(next?b:a).Add(time);(next?newTimes:oldTimes).Add(time);
                }
                rounds.Add((a.Average(),b.Average()));
            }
            double oldMedian=Median(oldTimes),newMedian=Median(newTimes),ratio=oldMedian/newMedian;int wins=rounds.Count(r=>r.Old/r.New>1.05);bool win=ratio>1.05&&wins>=4;allWins&=win;
            modes.Add(new{physicalAddresses=count,weightWorkingSetBytes=(long)count*s.N*s.K*2,mode=count==1?"one-real-address":"all-distinct-real-layer-weights",oldGraphNodes=oldGraph.Nodes,candidateGraphNodes=newGraph.Nodes,oldMedianMsPerFullPass=oldMedian,newMedianMsPerFullPass=newMedian,speedup=ratio,roundWinsAbove1_05=wins,predeclaredWin=win,ordinaryLaunchOnly=true,scope="Complete cast→GEMM→original consumer; output reset_x included both arms. Address-stream component, not E2E or hardware DRAM-cold proof."});
        }
        return new{passed=true,predeclaredWin=allWins,records,modes,expectedTimedGraphReplays=2*20*64,outsideWarmupReplays=40,rounds=5};
    }
}
