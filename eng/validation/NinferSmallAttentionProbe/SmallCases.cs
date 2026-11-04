// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;

internal static partial class Program
{
    private const int Rows=8,Elements=Rows*6144,MaxBlocks=160,MaxPages=640;
    private const long KOffset=64,VOffset=KOffset+CudaNinferPrefillAttentionModule.KvPlaneBlockBytes+128;
    private const int BlockBodyBytes=(int)(VOffset+CudaNinferPrefillAttentionModule.KvPlaneBlockBytes+64);
    private static R.Case[] Cases()=>new[]{
        new R.Case("dense-t8-k74",8,66,"dense",4),new R.Case("dense-t8-k128",8,120,"dense",4),
        new R.Case("dense-t8-k202",8,194,"dense",4),new R.Case("dense-t8-k512",8,504,"dense",4),
        new R.Case("dense-t8-k1024",8,1016,"dense",4),new R.Case("dense-t8-k4096",8,4088,"dense",4),
        new R.Case("sentinel-t8-k65",8,57,"sentinel",4),new R.Case("sentinel-t8-k257",8,249,"sentinel",4)};
    private static int[] Permutation(int count,uint seed)
    {
        var p=Enumerable.Range(0,count).ToArray();for(int i=count-1;i>0;i--){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;int j=(int)(seed%(uint)(i+1));(p[i],p[j])=(p[j],p[i]);}return p;
    }
    private sealed record Cache(Memory.Buffer Table,byte[] TableBytes,Memory.Buffer OriginalK,Memory.Buffer OriginalV,
        Memory.Buffer OriginalTable,int[] PhysicalPermutation,int[] OriginalPermutation);
    private static Cache MakeCache(Memory memory,R.Fixture f,uint seed=0x65e19bd3U)
    {
        int count=(f.Case.Keys+255)/256+1,pages=(f.Case.Keys+63)/64+1;
        int[] permutation=Permutation(count,seed),inverse=new int[count],originalPermutation=Permutation(pages,seed^0x14a5b43aU);
        for(int logical=0;logical<count;logical++)inverse[permutation[logical]]=logical;
        var blocks=new Memory.Buffer[count];
        for(int physical=0;physical<count;physical++)
        {
            int logical=inverse[physical];var data=new byte[BlockBodyBytes];Array.Fill(data,(byte)0xa5);
            foreach(long offset in new[]{KOffset,VOffset})
            {
                var words=MemoryMarshal.Cast<byte,ushort>(data.AsSpan((int)offset,(int)CudaNinferPrefillAttentionModule.KvPlaneBlockBytes));words.Fill(0x7fc1);
                ushort[] source=offset==KOffset?f.LogicalK:f.LogicalV;
                for(int t=0;t<256&&logical*256+t<f.Case.Keys;t++)for(int h=0;h<4;h++)
                    source.AsSpan(((logical*256+t)*4+h)*256,256).CopyTo(words.Slice((h*256+t)*256,256));
            }
            blocks[physical]=memory.Put($"ts256-{seed:x8}-physical-{physical}",data);
        }
        var tablePointers=Enumerable.Range(0,MaxBlocks).Select(i=>blocks[permutation[Math.Min(i,count-1)]].Pointer.ToInt64()).ToArray();
        byte[] tableBytes=Bytes(tablePointers);var table=memory.Put($"ts256-{seed:x8}-table",tableBytes);
        ushort[] Original(ushort[] source)
        {
            var data=new ushort[pages*4*64*256];Array.Fill(data,(ushort)0x7fc1);
            for(int k=0;k<f.Case.Keys;k++)for(int h=0;h<4;h++)source.AsSpan((k*4+h)*256,256).CopyTo(data.AsSpan(((originalPermutation[k/64]*4+h)*64+k%64)*256,256));return data;
        }
        var originalK=memory.Put($"original64-{seed:x8}-K",Bytes(Original(f.LogicalK)));
        var originalV=memory.Put($"original64-{seed:x8}-V",Bytes(Original(f.LogicalV)));
        var originalTable=memory.Put($"original64-{seed:x8}-table",Bytes(Enumerable.Range(0,MaxPages).Select(i=>originalPermutation[Math.Min(i,pages-1)]).ToArray()));
        return new(table,tableBytes,originalK,originalV,originalTable,permutation,originalPermutation);
    }
    private sealed record Numeric(double MaxAbsolute,double MaxScaledError,int Nonfinite,int Violations,int ExactAnalyticViolations);
    private static Numeric Check(byte[] body,R.Fixture f,bool core)
    {
        Need(body.Length==Elements*(core?2:4),"Full output span differs.");double max=0,scaled=0;int nonfinite=0,violations=0,exact=0;
        for(int i=0;i<Elements;i++)
        {
            double actual=core?R.Float(MemoryMarshal.Cast<byte,ushort>(body)[i]):MemoryMarshal.Cast<byte,float>(body)[i];
            double expected=core?f.IdealCore[i]:f.IdealGated[i],error=Math.Abs(actual-expected),bound=R.Absolute+R.Relative*Math.Abs(expected);
            if(!double.IsFinite(actual)){nonfinite++;continue;}if(!(error<=bound))violations++;max=Math.Max(max,error);scaled=Math.Max(scaled,error/bound);
            int d=i%256,h=i/256%24,t=i/6144;
            if(f.Case.Pattern=="constant"||f.Case.Pattern=="sentinel"&&d%17!=0)
            {
                float analytic=R.Constant(h/6,d)*(core?1:.5f);
                if(core?MemoryMarshal.Cast<byte,ushort>(body)[i]!=R.Bf(analytic):BitConverter.SingleToInt32Bits((float)actual)!=BitConverter.SingleToInt32Bits(analytic))exact++;
            }
            else if(f.Case.Pattern=="sentinel"&&f.Case.Past+t<f.Case.Sentinel&&actual!=0)exact++;
        }
        return new(max,scaled,nonfinite,violations,exact);
    }
    private static bool Pass(Numeric n)=>n.Nonfinite==0&&n.Violations==0&&n.ExactAnalyticViolations==0;
    private static Dictionary<string,string> Digests(R.Fixture f)=>new(){
        ["q.bf16"]=Sha(Bytes(f.Query)),["q.f32"]=Sha(Bytes(f.QueryF32)),["gate.f32"]=Sha(Bytes(f.Gate)),
        ["logical-k.bf16"]=Sha(Bytes(f.LogicalK)),["logical-v.bf16"]=Sha(Bytes(f.LogicalV)),
        ["ideal-core.f64"]=Sha(Bytes(f.IdealCore)),["ideal-gated.f32"]=Sha(Bytes(f.IdealGated))};
    private static void SamePartials(CudaNinferPrefillAttentionModule.Layout p,Memory.Buffer workspace,Memory.Buffer acc,Memory.Buffer m,Memory.Buffer l,
        string label,string output,Dictionary<string,string> files)
    {
        foreach(var pair in new[]{(acc,p.PartialAcc),(m,p.PartialM),(l,p.PartialL)})
        {
            byte[] actual=Memory.Read(workspace.Pointer+(nint)pair.Item2,pair.Item1.BodyBytes),expected=Memory.Body(pair.Item1);
            if(!actual.AsSpan().SequenceEqual(expected))
            {Cas(actual,output,files);Cas(expected,output,files);throw new InvalidDataException(label+": full original partial bits differ: "+pair.Item1.Name);}
        }
    }
    private static void RunCase(CudaNinferPrefillAttentionModule module,LegacySplit legacy,R.Fixture f,int life,string output,
        List<Dictionary<string,object?>> records,Dictionary<string,string> repeats,Dictionary<string,string> files,Action checkpoint)
    {
        using var memory=new Memory();using var stream=new StreamScope();var c=f.Case;var p=CudaNinferPrefillAttentionModule.PlanVerify8(c.Keys,MaxBlocks);
        var q=memory.Put("Q-BF16",Bytes(f.Query));var q32=memory.Put("Q-F32",Bytes(f.QueryF32));var gate=memory.Put("gate",Bytes(f.Gate));
        var positions=memory.Put("external-base-position",Bytes(new[]{c.Past}));var ends=memory.Put("legacy-key-ends",Bytes(Enumerable.Range(1,8).Select(i=>c.Past+i).ToArray()));
        var cache=MakeCache(memory,f);var workspace=memory.Empty("workspace",p.Bytes);var original=memory.Empty("original-core",Elements*2);
        var gated=memory.Empty("candidate-gated",Elements*4);var old=memory.Empty("legacy-gated",Elements*4);
        var acc=memory.Empty("original-partial-acc",Elements*4L*4);var m=memory.Empty("original-partial-m",Rows*24*4L*4);var l=memory.Empty("original-partial-l",Rows*24*4L*4);
        var oldPlan=Q35Kernels.GetSplitAttentionPlan(c.Keys);var oldPartial=memory.Empty("legacy-partial",Elements/256L*oldPlan.Splits*258*4);
        int keyBlocks=(c.Keys+63)/64,perSplit=(keyBlocks+3)/4,active=Math.Min(4,(keyBlocks+perSplit-1)/perSplit);
        var passes=new List<object>();var record=new Dictionary<string,object?>{["id"]=$"{c.Name}/life{life}",["case"]=c.Name,["lifetime"]=life,["rows"]=Rows,["keys"]=c.Keys,["past"]=c.Past,
            ["splits"]=4,["nonemptyKeySplits"]=active,["neutralKeySplits"]=4-active,["launchedCoreCtas"]=96,["legacySplits"]=oldPlan.Splits,["legacyChunk"]=oldPlan.Chunk,
            ["inputSha256"]=Digests(f),["physicalPermutation"]=cache.PhysicalPermutation,["originalPermutation"]=cache.OriginalPermutation,["passed"]=false,["passes"]=passes};records.Add(record);checkpoint();
        for(int pass=0;pass<3;pass++)
        {
            foreach(var b in memory.Buffers.Where(x=>x.ImmutableSha==null))Memory.Poison(b,Poisons[pass]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            module.RunF32Verify8(q32.Pointer,gate.Pointer,cache.Table.Pointer,MaxBlocks,KOffset,VOffset,positions.Pointer,c.Keys,.0625f,gated.Pointer,workspace.Pointer,p.Bytes,stream.Pointer);
            module.RunOriginalVerify8Bf16(q.Pointer,cache.OriginalK.Pointer,cache.OriginalV.Pointer,cache.OriginalTable.Pointer,positions.Pointer,original.Pointer,.0625f,acc.Pointer,m.Pointer,l.Pointer,stream.Pointer);
            legacy.Run(q32.Pointer,gate.Pointer,ends.Pointer,cache.Table.Pointer,KOffset,VOffset,c.Keys,oldPartial.Pointer,old.Pointer,stream.Pointer);stream.Sync();
            byte[] core=Memory.Read(workspace.Pointer+(nint)p.Output,Elements*2),refCore=Memory.Body(original),candidate=Memory.Body(gated),control=Memory.Body(old);
            bool exact=core.AsSpan().SequenceEqual(refCore),packed=Memory.Read(workspace.Pointer+(nint)p.Query,Elements*2).AsSpan().SequenceEqual(Bytes(f.Query));
            // Preserve complete outputs even if the following partial-bit gate
            // fails before constructing the pass record. Equal bodies dedup.
            foreach(var bytes in new[]{core,refCore,candidate,control})Cas(bytes,output,files);
            SamePartials(p,workspace,acc,m,l,c.Name,output,files);
            var coreNumeric=Check(core,f,true);var candidateNumeric=Check(candidate,f,false);var controlNumeric=Check(control,f,false);
            bool repeated=true;var captured=new Dictionary<string,string>();
            foreach(var item in new[]{("core",core),("candidate",candidate),("legacy",control)})
            {
                string key=c.Name+"/"+item.Item1,hash=Sha(item.Item2);if(repeats.TryGetValue(key,out var expected)&&hash!=expected)repeated=false;repeats.TryAdd(key,hash);
                captured.Add(item.Item1,Cas(item.Item2,output,files));
            }
            bool immutable=memory.CheckInputs(),guards=memory.CheckGuards(),ok=exact&&packed&&repeated&&immutable&&guards&&Pass(coreNumeric)&&Pass(candidateNumeric)&&Pass(controlNumeric);
            passes.Add(new{pass,poisonByte=Poisons[pass],originalCoreBitsExact=exact,originalPartialBitsExact=true,packedQueryBitsExact=packed,aaAndCrossLifetimeBits=repeated,
                immutableInputs=immutable,allAllocationGuards=guards,coreFp64=coreNumeric,candidateFp64=candidateNumeric,legacyFp64=controlNumeric,captured,passed=ok});checkpoint();Need(ok,"Small attention component gate failed; complete outputs retained.");
        }
        Need(module.PagedCoreLaunches==3&&module.OriginalCoreLaunches==3&&legacy.KernelSubmissions==6,"Actual T8 core/legacy submission coverage differs.");
        record["kernelSubmissions"]=24;record["passed"]=true;checkpoint();
    }
}
