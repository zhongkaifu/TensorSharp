// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Runtime;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    [StructLayout(LayoutKind.Sequential)]private struct StockPolicy{internal IntPtr Data;internal int N;internal int Padding;}
    private static T Private<T>(object o,string name)=>(T)(o.GetType().GetField(name,BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(o)??throw new InvalidDataException("Actual legacy helper field missing: "+name));
    private sealed unsafe class Modules:IDisposable
    {
        internal readonly CudaNinferKernels Old;
        private readonly CudaModule _candidate;
        internal readonly IntPtr Candidate,Stock,Cast,Round,Norm;
        internal readonly Shape Geometry;
        internal int MathMode{get;}
        internal readonly bool Mma;
        internal readonly int Blocks,Threads,Shared;
        internal Modules(string helper,string helperSha,string candidate,string candidateSha,JsonElement manifest,Shape shape)
        {
            Geometry=shape;Mma=manifest.GetProperty("arithmeticPolicy").GetString()==MmaPolicy;Blocks=shape.N/(Mma?32:16);Threads=Mma?256:128;Shared=Mma?98304:0;Old=new CudaNinferKernels(helper);
            try{Need(S.Matches(Old.ImageSha256,helperSha),"Actual helper image differs.");_candidate=CudaModule.LoadFromFile(candidate,identify:true);try
            {
                Need(S.Matches(_candidate.ImageSha256,candidateSha)&&_candidate.EnumerateFunctions().Length==4,"Actual isolated four-entry image differs.");
                var entries=manifest.GetProperty("specializations").EnumerateArray().Where(x=>x.GetProperty("outputRows").GetInt32()==shape.N&&x.GetProperty("columns").GetInt32()==shape.K).ToArray();Need(entries.Length==2,"Both actual output policies required.");
                IntPtr Bind(string policy,int size)
                {
                    var e=entries.Single(x=>x.GetProperty("outputPolicy").GetString()==policy);Need(e.GetProperty("rows").GetInt32()==8&&e.GetProperty("grid").EnumerateArray().Select(x=>x.GetInt32()).SequenceEqual(new[]{Blocks,1,1})&&e.GetProperty("block").EnumerateArray().Select(x=>x.GetInt32()).SequenceEqual(new[]{Threads,1,1})&&e.GetProperty("dynamicSmemBytes").GetInt32()==Shared,"Actual launch geometry differs.");
                    IntPtr f=_candidate.GetFunction(e.GetProperty("kernel").GetString()!);CudaDriverApi.cuFuncLoad(f).ThrowOnError();ValidateParameterMetadata(e.GetProperty("parameters"),Mma,size);var expectedParameters=ParameterLayout(Mma,size);int count=expectedParameters.Length;
                    for(uint i=0;i<count;i++)
                    {
                        var expected=expectedParameters[i];CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)i,out var off,out var n).ThrowOnError();
                        Need(off.ToUInt64()==(ulong)expected.Offset&&n.ToUInt64()==(ulong)expected.Size,"Actual driver/PTX parameter ABI differs.");
                    }
                    Need(CudaDriverApi.cuFuncGetParamInfo(f,(UIntPtr)count,out _,out _)==1,"Unexpected extra CUDA argument.");
                    if(Mma){CudaDriverApi.cuFuncSetAttribute(f,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,Shared).ThrowOnError();CudaDriverApi.cuFuncGetAttribute(out int limit,CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,f).ThrowOnError();Need(limit>=Shared,"Original MMA dynamic shared-memory opt-in unavailable.");}
                    return f;
                }
                Candidate=Bind("f32",8);Stock=Bind("original-bf16",16);Cast=Private<IntPtr>(Old,"_cast");Round=Private<IntPtr>(Old,"_round");Norm=Private<IntPtr>(Old,"_norm");
                CublasApi.cublasGetMathMode(Private<IntPtr>(Old,"_blas"),out int mode).ThrowOnError();MathMode=mode;
            }catch{_candidate.Dispose();throw;}}catch{Old.Dispose();throw;}
        }
        internal void Project(Arm arm,Memory.Buffer input,Memory.Buffer weight,bool candidate,IntPtr stream)
        {
            if(!candidate){Old.ProjectBf16(weight.Pointer,input.Pointer,arm.Projection.Pointer,Geometry.N,Geometry.K,8,stream);return;}
            Old.CastToBf16(input.Pointer,arm.Cast.Pointer,(long)8*Geometry.K,stream);IntPtr x=arm.Cast.Pointer,w=weight.Pointer,y=arm.Projection.Pointer;int tokens=8;void** args=stackalloc void*[]{&x,&w,&y,&tokens};CudaDriverApi.cuLaunchKernel(Candidate,(uint)Blocks,1,1,(uint)Threads,1,1,(uint)Shared,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();
        }
        internal void OriginalReference(Arm candidate,Memory.Buffer weight,Memory.Buffer output,IntPtr stream)
        {IntPtr x=candidate.Cast.Pointer,w=weight.Pointer;var policy=new StockPolicy{Data=output.Pointer,N=Geometry.N,Padding=0};int tokens=8;void** args=stackalloc void*[]{&x,&w,&policy,&tokens};CudaDriverApi.cuLaunchKernel(Stock,(uint)Blocks,1,1,(uint)Threads,1,1,(uint)Shared,stream,(IntPtr)args,IntPtr.Zero).ThrowOnError();}
        internal void Consume(Arm arm,Memory.Buffer norm,IntPtr stream)
        {if(Geometry.Name=="input")Old.Round(arm.Projection.Pointer,(long)8*Geometry.N,stream);else Old.AddRmsNorm(arm.X.Pointer,arm.Projection.Pointer,norm.Pointer,arm.Y.Pointer,8,5120,Epsilon,stream);}
        internal byte[] ActualOldCast()=>Memory.Read(Private<IntPtr>(Old,"_inputBf16"),8*Geometry.K*2);
        public void Dispose(){_candidate.Dispose();Old.Dispose();}
    }
    private sealed class Arm
    {
        internal readonly Memory.Buffer Projection,Cast,X,Y;
        internal Arm(Memory m,string id,Shape s){Projection=m.Empty(id+"-projection",8*s.N*4);Cast=m.Empty(id+"-bf16-input",8*s.K*2);X=m.Empty(id+"-residual",8*s.N*4);Y=m.Empty(id+"-norm",8*s.N*4);}
        internal void Reset(Case c,byte poison){foreach(var b in new[]{Projection,Cast,X,Y})Memory.Poison(b,poison);Memory.Write(X.Pointer,File.ReadAllBytes(c.Seed.Path));}
        internal byte[] Output(Shape s)=>Memory.Body(s.Name=="input"?Projection:Y);
    }
    private sealed record Captures(byte[] Projection,byte[] Consumer,byte[]? Residual);
    private static byte[] CastBits(byte[] f32)
    {var a=MemoryMarshal.Cast<byte,float>(f32);var b=new ushort[a.Length];for(int i=0;i<a.Length;i++)b[i]=(ushort)(BitConverter.SingleToUInt32Bits(Bf(a[i]))>>16);return Bytes(b);}
    private sealed record ArmCheck(Numeric Dot,bool ConsumerCpuContract,bool OwnArmAA,bool Guards,bool Immutable,object Raw){public bool Passed=>Dot.Passed&&ConsumerCpuContract&&OwnArmAA&&Guards&&Immutable;}
    private static ArmCheck CheckArm(Case c,Shape s,OracleCase oracle,Arm arm,byte[] before,Memory memory,string output,Dictionary<string,string> files,Dictionary<string,string> aa,string key)
    {
        double[] ideal=MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(oracle.Ideal.Path)).ToArray(),l1=MemoryMarshal.Cast<byte,double>(File.ReadAllBytes(oracle.L1.Path)).ToArray();var dot=Dot(before,ideal,l1,s.K,c.Exact);byte[] after=arm.Output(s);byte[]? residual=s.Name=="output"?Memory.Body(arm.X):null;
        var data=new{projection=Capture(before,output,files),consumer=Capture(after,output,files),residual=residual==null?null:Capture(residual,output,files)};
        bool consumer=Consumer(c,s,before,after,residual),same=Repeat(aa,key+"/projection",before)&&Repeat(aa,key+"/consumer",after)&&(residual==null||Repeat(aa,key+"/residual",residual)),guards=memory.Guards(),immutable=memory.Inputs();
        return new(dot,consumer,same,guards,immutable,data);
    }
    private static void Components(Shape s,Prepared prepared,OracleSet oracle,JsonElement native,string helper,string helperSha,string candidate,string candidateSha,string output,Dictionary<string,string> files,List<object> results,List<object> libraries)
    {
        var selected=prepared.Cases.Where(x=>x.Shape==s.Name).ToArray();Need(selected.Length==s.RealCount+3&&selected.Count(x=>x.Kind=="real")==s.RealCount,"Complete real/synthetic shape matrix required.");var aa=new Dictionary<string,string>();
        using var reader=new NinferArtifactReader(prepared.Artifact);Need(reader.DirectorySha256==prepared.DirectorySha256,"Artifact directory differs.");
        for(int life=0;life<2;life++)using(var stream=new Stream())using(var modules=new Modules(helper,helperSha,candidate,candidateSha,native,s))
        {
            foreach(var c in selected)
            {
                var o=oracle.Cases.Single(x=>x.Id==c.Id);Check(o.Ideal);Check(o.L1);using var memory=new Memory();byte[] w=Weight(c,s,reader);Need(Sha(w)==o.WeightSha256,"Actual uploaded matrix differs from CPU oracle: "+c.Id);
                var weights=memory.Put(c.Id+"-weights",w);w=Array.Empty<byte>();var input=memory.Put("input",File.ReadAllBytes(c.Input.Path));var norm=memory.Put("norm",File.ReadAllBytes(c.Norm.Path));var old=new Arm(memory,"old",s);var next=new Arm(memory,"candidate",s);var stock=memory.Empty("stock-bf16",8*s.N*2);byte[] expectedCast=CastBits(File.ReadAllBytes(c.Input.Path));
                for(int pass=0;pass<3;pass++)
                {
                    old.Reset(c,Poisons[pass]);next.Reset(c,Poisons[pass]);Memory.Poison(stock,Poisons[pass]);modules.Project(old,input,weights,false,stream.Pointer);modules.Project(next,input,weights,true,stream.Pointer);modules.OriginalReference(next,weights,stock,stream.Pointer);stream.Sync();byte[] oldRaw=Memory.Body(old.Projection),newRaw=Memory.Body(next.Projection),refRaw=Memory.Body(stock);
                    bool casts=expectedCast.AsSpan().SequenceEqual(modules.ActualOldCast())&&expectedCast.AsSpan().SequenceEqual(Memory.Body(next.Cast)),stockSame=CastBits(newRaw).AsSpan().SequenceEqual(refRaw),stockAA=Repeat(aa,c.Id+"/stock",refRaw);var stockCapture=Capture(refRaw,output,files);bool stockFinite=true;foreach(ushort v in MemoryMarshal.Cast<byte,ushort>(refRaw))stockFinite&=(v&0x7f80)!=0x7f80;
                    modules.Consume(old,norm,stream.Pointer);modules.Consume(next,norm,stream.Pointer);stream.Sync();var a=CheckArm(c,s,o,old,oldRaw,memory,output,files,aa,c.Id+"/old");var b=CheckArm(c,s,o,next,newRaw,memory,output,files,aa,c.Id+"/candidate");
                    int rawDiff=DifferentWords(oldRaw,newRaw),consumerDiff=DifferentWords(old.Output(s),next.Output(s));int? residualDiff=s.Name=="output"?DifferentWords(Memory.Body(old.X),Memory.Body(next.X)):null;bool equal=rawDiff==0&&consumerDiff==0&&(residualDiff==null||residualDiff==0);
                    results.Add(new{c.Id,life,pass,castsExactCpu=casts,candidateBf16ExactlyStock=stockSame,stockAA,stockFinite,stockReference=stockCapture,legacy=a,candidate=b,rawF32DifferentWords=rawDiff,consumerDifferentWords=consumerDiff,residualDifferentWords=residualDiff,legacyBitsExact=equal,legacyBitsRequired=modules.Mma,fullValuesPerProjection=8*s.N,actualLegacyCuBlas=true,ordinaryCandidateLaunch=true});Save(Path.Combine(output,"component-progress.json"),new{complete=false,records=results});Need(casts&&stockSame&&stockAA&&stockFinite&&a.Passed&&b.Passed&&(!modules.Mma||equal),"Full original/candidate projection/consumer/bound/AA/guard/exact-old gate failed: "+c.Id);
                }
            }
            libraries.Add(new{life,modules.MathMode,mappedNativeLibraries=MappedLibraries()});stream.Sync();
        }
    }
    // Deliberately bounded admission screen: nine actual matrices, one pass each.
    // This is not the two-lifetime full component or a performance measurement.
    private static void Screen(Prepared prepared,OracleSet oracle,JsonElement native,string helper,string helperSha,string candidate,string candidateSha,string output,Dictionary<string,string> files,List<object> results,List<object> libraries)
    {
        var aa=new Dictionary<string,string>();using var reader=new NinferArtifactReader(prepared.Artifact);Need(reader.DirectorySha256==prepared.DirectorySha256,"Artifact directory differs.");
        foreach(var s in Shapes)using(var stream=new Stream())using(var modules=new Modules(helper,helperSha,candidate,candidateSha,native,s))
        {
            Need(modules.Mma,"Screen requires the separately pinned original MMA candidate.");var selected=prepared.Cases.Where(c=>c.Shape==s.Name&&c.Kind=="real").ToArray();Need(selected.Length==s.RealCount,"All real matrices required.");
            foreach(var c in selected)
            {
                var o=oracle.Cases.Single(x=>x.Id==c.Id);using var memory=new Memory();byte[] w=Weight(c,s,reader);Need(Sha(w)==o.WeightSha256,"Actual uploaded matrix differs from fixed oracle.");
                var weight=memory.Put(c.Id+"-weights",w);w=Array.Empty<byte>();var input=memory.Put("input",File.ReadAllBytes(c.Input.Path));var norm=memory.Put("norm",File.ReadAllBytes(c.Norm.Path));var old=new Arm(memory,"old",s);var next=new Arm(memory,"candidate",s);var stock=memory.Empty("stock-bf16",8*s.N*2);
                old.Reset(c,Poisons[0]);next.Reset(c,Poisons[0]);Memory.Poison(stock,Poisons[0]);modules.Project(old,input,weight,false,stream.Pointer);modules.Project(next,input,weight,true,stream.Pointer);modules.OriginalReference(next,weight,stock,stream.Pointer);stream.Sync();
                byte[] oldRaw=Memory.Body(old.Projection),newRaw=Memory.Body(next.Projection),refRaw=Memory.Body(stock),cast=CastBits(File.ReadAllBytes(c.Input.Path));
                bool casts=cast.AsSpan().SequenceEqual(modules.ActualOldCast())&&cast.AsSpan().SequenceEqual(Memory.Body(next.Cast)),stockSame=CastBits(newRaw).AsSpan().SequenceEqual(refRaw);var stockCapture=Capture(refRaw,output,files);bool stockFinite=true;foreach(ushort v in MemoryMarshal.Cast<byte,ushort>(refRaw))stockFinite&=(v&0x7f80)!=0x7f80;
                modules.Consume(old,norm,stream.Pointer);modules.Consume(next,norm,stream.Pointer);stream.Sync();
                var a=CheckArm(c,s,o,old,oldRaw,memory,output,files,aa,c.Id+"/old");var b=CheckArm(c,s,o,next,newRaw,memory,output,files,aa,c.Id+"/candidate");
                byte[] oldConsumer=old.Output(s),newConsumer=next.Output(s);byte[]? oldX=s.Name=="output"?Memory.Body(old.X):null,newX=s.Name=="output"?Memory.Body(next.X):null;
                int rawDiff=DifferentWords(oldRaw,newRaw),consumerDiff=DifferentWords(oldConsumer,newConsumer);int? residualDiff=oldX==null?null:DifferentWords(oldX,newX!);bool equal=rawDiff==0&&consumerDiff==0&&(residualDiff==null||residualDiff==0);
                results.Add(new{c.Id,shape=s.Name,pass=0,aaScope="BASELINE_ONLY_NOT_REPEATED_SCREEN",castsExactCpu=casts,candidateBf16ExactlyStock=stockSame,stockFinite,stockReference=stockCapture,legacy=a,candidate=b,rawF32DifferentWords=rawDiff,consumerDifferentWords=consumerDiff,residualDifferentWords=residualDiff,legacyBitsExact=equal,fullValuesPerProjection=8*s.N,actualLegacyCuBlas=true,ordinaryCandidateLaunch=true});
                // Raw captures are already retained in the record before any gate fails.
                Save(Path.Combine(output,"screen-progress.json"),new{complete=false,records=results});
                Need(casts&&stockSame&&stockFinite&&a.Passed&&b.Passed&&equal,"MMA exact-old screen failed: "+c.Id+"; mismatching full raw arrays retained; no timing permitted.");
            }
            libraries.Add(new{shape=s.Name,modules.MathMode,mappedNativeLibraries=MappedLibraries()});stream.Sync();
        }
        Need(results.Count==9,"All nine real matrices must complete; missing cases never pass.");
    }
    private static int DifferentWords(byte[] a,byte[] b)
    {Need(a.Length==b.Length&&a.Length%4==0,"Full F32 parity spans differ.");int count=0;var x=MemoryMarshal.Cast<byte,uint>(a);var y=MemoryMarshal.Cast<byte,uint>(b);for(int i=0;i<x.Length;i++)if(x[i]!=y[i])count++;return count;}
    private static (int Offset,int Size)[] ParameterLayout(bool mma,int policyBytes)
    {Need(policyBytes is 8 or 16,"Only actual F32-pointer or original BF16 output policy supported.");return mma?new[]{(0,8),(8,8),(16,policyBytes),(16+policyBytes,4)}:new[]{(0,8),(8,8),(16,policyBytes)};}
    private static void ValidateParameterMetadata(JsonElement parameters,bool mma,int policyBytes)
    {var expected=ParameterLayout(mma,policyBytes);var actual=parameters.EnumerateArray().ToArray();Need(actual.Length==expected.Length,"Unexpected actual manifest parameter count.");for(int i=0;i<expected.Length;i++)Need(actual[i].GetProperty("offset").GetInt32()==expected[i].Offset&&actual[i].GetProperty("size").GetInt32()==expected[i].Size,"Unexpected actual manifest parameter layout.");}
    private static object PrimitiveContracts()
    {
        Need(IntPtr.Size==8&&BitConverter.IsLittleEndian&&Marshal.SizeOf<StockPolicy>()==16&&Marshal.SizeOf<GraphApi.KernelParams>()==56&&Marshal.SizeOf<CopyGraph.Params>()==200,"Fixed Linux64 device/graph ABI differs.");
        var small=new Shape("cpu",32,64,0);var dummy=new Pin("",0,"");var hashes=new Dictionary<string,string>();
        foreach(string kind in new[]{"dense","cancellation","rounding-lowbits"}){var c=new Case(kind,"cpu",kind,null,0,32*64*2,false,true,dummy,dummy,dummy);hashes[kind]=Sha(Weight(c,small,null));}
        Need(Dot(Bytes(new[]{1.00390625f}),new[]{1.00390625d},new[]{1.00390625d},5120,true).Passed&&!Dot(Bytes(new[]{1f}),new[]{1.00390625d},new[]{1.00390625d},5120,true).Passed,"Unrounded FP32 gate failed to detect premature BF16.");
        Need(DifferentWords(Bytes(new[]{0f,-0f}),Bytes(new[]{-0f,0f}))==2&&DifferentWords(Bytes(new[]{1f}),Bytes(new[]{BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(1f)+1)}))==1,"Full raw parity must detect signed zero and one-bit F32 changes.");
        int negatives=0;void Refused(Action action){try{action();}catch(InvalidDataException){negatives++;return;}throw new InvalidDataException("Changed screen/ABI metadata admitted.");}
        foreach(int policy in new[]{8,16})
        {using var doc=JsonDocument.Parse(JsonSerializer.Serialize(ParameterLayout(true,policy).Select(x=>new{offset=x.Offset,size=x.Size})));ValidateParameterMetadata(doc.RootElement,true,policy);Refused(()=>ValidateParameterMetadata(doc.RootElement,false,policy));}
        using(var bad=JsonDocument.Parse("[{\"offset\":0,\"size\":8},{\"offset\":8,\"size\":8},{\"offset\":16,\"size\":8},{\"offset\":32,\"size\":4}]"))Refused(()=>ValidateParameterMetadata(bad.RootElement,true,8));
        ValidateScreenOptions(Options(new[]{"--execute","--screen-only"}));
        foreach(var args in new[]{new[]{"--screen-only"},new[]{"--execute","--screen-only","--timing"},new[]{"--execute","--screen-only","--shape","input"}})Refused(()=>ValidateScreenOptions(Options(args)));
        return new{screenAndFourSlotNegativeCases=negatives,rawParitySignedZeroAndOneBit="PASS",driverAbi="PASS",syntheticWeight32x64Sha256=hashes,fullRawF32Witness="PASS",mmaFourSlotAbi="8/8/8/4 or 8/8/16/4; actual driver check before launch",GPU="NOT_RUN"};
    }
    private static Pin[] MappedLibraries()=>File.Exists("/proc/self/maps")?File.ReadLines("/proc/self/maps").Where(x=>x.Contains("libcublas",StringComparison.Ordinal)).Select(x=>x.Split(' ',StringSplitOptions.RemoveEmptyEntries)[^1]).Where(File.Exists).Distinct().Select(FilePin).ToArray():Array.Empty<Pin>();
}
