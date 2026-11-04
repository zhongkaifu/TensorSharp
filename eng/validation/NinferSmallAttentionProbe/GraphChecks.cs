// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;

internal static partial class Program
{
    private static R.Fixture RotateHeads(R.Fixture f,int phase)
    {
        // Rotate query heads only within their six-head KV group. The ideal
        // attention and gate tensors undergo the identical permutation; K/V
        // are unchanged. This is an exact algebraic transform, not a new oracle.
        T[] Rotate<T>(T[] source)
        {
            var result=new T[source.Length];
            for(int t=0;t<Rows;t++)for(int h=0;h<24;h++)
            {int src=h/6*6+(h%6+phase)%6;Array.Copy(source,(t*24+src)*256,result,(t*24+h)*256,256);}return result;
        }
        return new(f.Case,Rotate(f.Query),Rotate(f.QueryF32),Rotate(f.Gate),f.LogicalK,f.LogicalV,Rotate(f.IdealCore),Rotate(f.IdealGated));
    }
    private static R.Fixture[] GraphFixtures()=>new[]{57,249,255,1017}.Select((past,i)=>RotateHeads(R.Make(new R.Case($"graph-t8-p{past}",8,past,"dense",4)),i)).ToArray();
    private static void WriteExpected(Memory.Buffer buffer,byte[] bytes,Dictionary<string,string> expected)
    {Need(buffer.BodyBytes==bytes.Length,"Dynamic upload length differs.");Memory.Write(buffer.Allocation,Guarded(bytes));expected[buffer.Name]=Sha(Guarded(bytes));}
    private static bool DynamicInputs(Memory memory,Dictionary<string,string> expected)=>memory.Buffers.Where(x=>expected.ContainsKey(x.Name)).All(x=>Sha(Memory.All(x))==expected[x.Name]);
    private sealed record Snapshot(byte[] Query,byte[] Core,byte[] Gated,byte[] Acc,byte[] M,byte[] L);
    private static Snapshot ReadSnapshot(Memory.Buffer workspace,Memory.Buffer output,CudaNinferPrefillAttentionModule.Layout p)=>new(
        Memory.Read(workspace.Pointer+(nint)p.Query,Elements*2),Memory.Read(workspace.Pointer+(nint)p.Output,Elements*2),Memory.Body(output),
        Memory.Read(workspace.Pointer+(nint)p.PartialAcc,Elements*4*4),Memory.Read(workspace.Pointer+(nint)p.PartialM,Rows*24*4*4),Memory.Read(workspace.Pointer+(nint)p.PartialL,Rows*24*4*4));
    private static bool SameSnapshot(Snapshot a,Snapshot b)=>a.Query.AsSpan().SequenceEqual(b.Query)&&a.Core.AsSpan().SequenceEqual(b.Core)&&a.Gated.AsSpan().SequenceEqual(b.Gated)
        &&a.Acc.AsSpan().SequenceEqual(b.Acc)&&a.M.AsSpan().SequenceEqual(b.M)&&a.L.AsSpan().SequenceEqual(b.L);
    private static void RunGraphChecks(CudaNinferPrefillAttentionModule module,R.Fixture[] fixtures,int life,string output,List<Dictionary<string,object?>> graphRecords,
        Dictionary<string,string> files,Dictionary<string,string> repeats,Action checkpoint)
    {
        using var memory=new Memory();using var stream=new StreamScope();var maximum=fixtures[^1];var p=CudaNinferPrefillAttentionModule.PlanVerify8(maximum.Case.Keys,MaxBlocks);
        var cacheA=MakeCache(memory,maximum,0x65e19bd3U);var cacheB=MakeCache(memory,maximum,0xb4072ea9U);
        Need(!cacheA.TableBytes.AsSpan().SequenceEqual(cacheB.TableBytes),"Graph pointer-table updates require distinct live physical cache sets.");
        var q=memory.Empty("dynamic-Q-BF16",Elements*2);var q32=memory.Empty("dynamic-Q-F32",Elements*4);var gate=memory.Empty("dynamic-gate",Elements*4);
        var table=memory.Empty("dynamic-table",MaxBlocks*8);var positions=memory.Empty("dynamic-external-base",4);
        var workspace=memory.Empty("graph-workspace",p.Bytes);var directWorkspace=memory.Empty("direct-workspace",p.Bytes);
        var gated=memory.Empty("graph-gated",Elements*4);var directGated=memory.Empty("direct-gated",Elements*4);var original=memory.Empty("original-core",Elements*2);
        var acc=memory.Empty("original-acc",Elements*4*4);var m=memory.Empty("original-m",Rows*24*4*4);var l=memory.Empty("original-l",Rows*24*4*4);
        var dynamic=new Dictionary<string,string>();
        void Upload(R.Fixture f,int variant)
        {
            WriteExpected(q,Bytes(f.Query),dynamic);WriteExpected(q32,Bytes(f.QueryF32),dynamic);WriteExpected(gate,Bytes(f.Gate),dynamic);
            WriteExpected(positions,Bytes(new[]{f.Case.Past}),dynamic);WriteExpected(table,variant%2==0?cacheA.TableBytes:cacheB.TableBytes,dynamic);
            // The helper's uploads are synchronous default-stream operations.
            // Explicitly finish that stream before this nondefault graph stream.
            CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
        }
        Upload(fixtures[0],0);
        using var graph=new Captured(stream.Pointer,()=>module.RunF32Verify8(q32.Pointer,gate.Pointer,table.Pointer,MaxBlocks,KOffset,VOffset,positions.Pointer,
            fixtures[0].Case.Keys,.0625f,gated.Pointer,workspace.Pointer,p.Bytes,stream.Pointer),positions.Pointer,workspace.Pointer+(nint)p.Positions);
        var passes=new List<object>();var record=new Dictionary<string,object?>{["id"]=$"graph/life{life}",["lifetime"]=life,["capturedNodes"]=graph.Nodes,
            ["captureHostAdmissionKeys"]=fixtures[0].Case.Keys,["fixedWorkspaceBytes"]=p.Bytes,["fixedPages"]=p.Pages,["sameDeviceAddressesAcrossReplays"]=true,["passed"]=false,["passes"]=passes};graphRecords.Add(record);checkpoint();
        foreach(var fixture in fixtures)
        {
            Need(maximum.LogicalK.AsSpan(0,fixture.LogicalK.Length).SequenceEqual(fixture.LogicalK)&&maximum.LogicalV.AsSpan(0,fixture.LogicalV.Length).SequenceEqual(fixture.LogicalV),"Graph reference cache prefix differs.");
        }
        for(int pass=0;pass<3;pass++)for(int variant=0;variant<fixtures.Length;variant++)
        {
            var f=fixtures[variant];foreach(var b in new[]{workspace,directWorkspace,gated,directGated,original,acc,m,l})Memory.Poison(b,Poisons[pass]);Upload(f,variant);
            // No Pack is launched outside the graph before replay. Its query,
            // metadata, table and output regions have all just been poisoned.
            graph.Run(stream.Pointer);stream.Sync();var replay=ReadSnapshot(workspace,gated,p);
            module.RunF32Verify8(q32.Pointer,gate.Pointer,table.Pointer,MaxBlocks,KOffset,VOffset,positions.Pointer,f.Case.Keys,.0625f,directGated.Pointer,directWorkspace.Pointer,p.Bytes,stream.Pointer);
            module.RunOriginalVerify8Bf16(q.Pointer,cacheA.OriginalK.Pointer,cacheA.OriginalV.Pointer,cacheA.OriginalTable.Pointer,positions.Pointer,original.Pointer,.0625f,acc.Pointer,m.Pointer,l.Pointer,stream.Pointer);stream.Sync();
            var direct=ReadSnapshot(directWorkspace,directGated,p);bool sameDirect=SameSnapshot(replay,direct),sameOriginal=replay.Core.AsSpan().SequenceEqual(Memory.Body(original));
            bool samePartials=replay.Acc.AsSpan().SequenceEqual(Memory.Body(acc))&&replay.M.AsSpan().SequenceEqual(Memory.Body(m))&&replay.L.AsSpan().SequenceEqual(Memory.Body(l));
            bool packed=replay.Query.AsSpan().SequenceEqual(Bytes(f.Query)),inputs=memory.CheckInputs()&&DynamicInputs(memory,dynamic),guards=memory.CheckGuards();
            var numeric=Check(replay.Gated,f,false);var coreNumeric=Check(replay.Core,f,true);bool repeated=true;var captured=new Dictionary<string,string>();
            foreach(var pair in new[]{("core",replay.Core),("gated",replay.Gated)})
            {
                string key=f.Case.Name+"/"+pair.Item1,hash=Sha(pair.Item2);if(repeats.TryGetValue(key,out var previous)&&hash!=previous)repeated=false;repeats.TryAdd(key,hash);captured.Add(pair.Item1,Cas(pair.Item2,output,files));
            }
            bool ok=sameDirect&&sameOriginal&&samePartials&&packed&&inputs&&guards&&repeated&&Pass(numeric)&&Pass(coreNumeric);
            passes.Add(new{pass,variant,past=f.Case.Past,keys=f.Case.Keys,poisonByte=Poisons[pass],querySha256=Sha(Bytes(f.Query)),pointerTableSha256=dynamic[table.Name],
                originalCoreBitsExact=sameOriginal,originalPartialBitsExact=samePartials,graphVsUncapturedAllBitsExact=sameDirect,packedQueryBitsExact=packed,
                immutableAndUpdatedInputs=inputs,allAllocationGuards=guards,aaAndCrossLifetimeBits=repeated,coreFp64=coreNumeric,gatedFp64=numeric,captured,passed=ok});checkpoint();
            if(!ok){foreach(var bytes in new[]{replay.Query,replay.Core,replay.Gated,replay.Acc,replay.M,replay.L,direct.Core,direct.Gated,direct.Acc,direct.M,direct.L,Memory.Body(original),Memory.Body(acc),Memory.Body(m),Memory.Body(l)})Cas(bytes,output,files);}
            Need(ok,"Dynamic captured Verify8 replay differs from same-input uncaptured/original reference; diagnostics retained.");
        }
        Need(module.PagedCoreLaunches==13&&module.OriginalCoreLaunches==12,"Capture/direct/original host submission count differs; graph replays do not increment host counters.");
        record["actualGraphReplays"]=12;record["actualGpuKernelExecutions"]=120;record["capturedKernelNodes"]=4;record["passed"]=true;checkpoint();
    }
}
