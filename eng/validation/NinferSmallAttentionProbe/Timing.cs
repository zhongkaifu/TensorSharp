// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;
using S = TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private sealed record Qualified(Dictionary<string,string> Candidate,Dictionary<string,string> Legacy,Dictionary<string,string> Core,Dictionary<string,string> Files);
    private static Qualified ReadQualification(byte[] bytes,string folder)
    {
        using var d=JsonDocument.Parse(bytes);var root=d.RootElement;
        Need(root.GetProperty("schema").GetString()=="ninfer-original-bf16-small-attention-verify8/1"&&root.GetProperty("complete").GetBoolean()&&root.GetProperty("passed").GetBoolean()
            &&root.GetProperty("actualGpuKernelExecutions").GetInt32()==624&&root.GetProperty("allRawEvidenceRehashed").GetBoolean(),"Only the complete624-kernel component+dynamic-graph qualification admits timing.");
        Need(S.Matches(root.GetProperty("moduleSha256").GetString()!,CandidateSha)&&S.Matches(root.GetProperty("legacyModuleSha256").GetString()!,LegacySha)
            &&S.Matches(root.GetProperty("sourceReferenceSha256").GetString()!,ReferenceSha)&&S.Matches(root.GetProperty("sourceDeviceHelpersSha256").GetString()!,DeviceHelpersSha)
            &&root.GetProperty("policy").GetString()==CudaNinferPrefillAttentionModule.Verify8Policy,"Qualified code/image policy differs.");
        var files=root.GetProperty("files").EnumerateObject().ToDictionary(x=>x.Name,x=>S.Normalize(x.Value.GetString()!));Need(files.Count is >0 and <=32,"Qualified CAS count differs.");
        foreach(var x in files){Need(x.Key=="cas/"+x.Value+".bin","Noncanonical CAS path refused.");Pinned(Path.Combine(folder,x.Key),x.Value);}
        var candidate=new Dictionary<string,string>();var legacy=new Dictionary<string,string>();var core=new Dictionary<string,string>();var allRefs=new HashSet<string>();var ids=new HashSet<string>();
        void Numeric(JsonElement n)=>Need(n.GetProperty("nonfinite").GetInt32()==0&&n.GetProperty("violations").GetInt32()==0&&n.GetProperty("exactAnalyticViolations").GetInt32()==0,"Qualified numeric gate failed.");
        void Bind(Dictionary<string,string> arm,string name,string file)
        {Need(files.ContainsKey(file),"Missing qualified full output.");allRefs.Add(file);if(arm.TryGetValue(name,out var old))Need(file==old,"Qualified bytes changed across poisons/lifetimes.");else arm.Add(name,file);}
        var records=root.GetProperty("records").EnumerateArray().ToArray();Need(records.Length==16,"Qualified component lifecycle count differs.");
        foreach(var r in records)
        {
            string name=r.GetProperty("case").GetString()!;var c=Cases().Single(x=>x.Name==name);int life=r.GetProperty("lifetime").GetInt32();
            Need(life is 0 or 1&&r.GetProperty("id").GetString()==$"{name}/life{life}"&&ids.Add($"{name}/life{life}")&&r.GetProperty("passed").GetBoolean()
                &&r.GetProperty("keys").GetInt32()==c.Keys&&r.GetProperty("past").GetInt32()==c.Past&&r.GetProperty("rows").GetInt32()==8&&r.GetProperty("splits").GetInt32()==4&&r.GetProperty("kernelSubmissions").GetInt32()==24,"Qualified case identity differs.");
            var digest=r.GetProperty("inputSha256").EnumerateObject().ToDictionary(x=>x.Name,x=>x.Value.GetString()!);var expected=Digests(R.Make(c));
            Need(digest.Count==7&&expected.All(x=>digest.TryGetValue(x.Key,out var s)&&S.Matches(s,x.Value)),"Qualified input/oracle reconstruction differs.");
            var passes=r.GetProperty("passes").EnumerateArray().ToArray();Need(passes.Length==3,"All qualified poison passes mandatory.");
            for(int p=0;p<3;p++)
            {
                var x=passes[p];Need(x.GetProperty("pass").GetInt32()==p&&x.GetProperty("poisonByte").GetInt32()==Poisons[p],"Qualified poison identity differs.");
                foreach(string key in new[]{"passed","originalCoreBitsExact","originalPartialBitsExact","packedQueryBitsExact","aaAndCrossLifetimeBits","immutableInputs","allAllocationGuards"})Need(x.GetProperty(key).GetBoolean(),"Qualified gate failed: "+key);
                Numeric(x.GetProperty("coreFp64"));Numeric(x.GetProperty("candidateFp64"));Numeric(x.GetProperty("legacyFp64"));
                var captured=x.GetProperty("captured");Bind(candidate,name,captured.GetProperty("candidate").GetString()!);Bind(legacy,name,captured.GetProperty("legacy").GetString()!);Bind(core,name,captured.GetProperty("core").GetString()!);
            }
        }
        foreach(var c in Cases())foreach(int life in new[]{0,1})Need(ids.Contains($"{c.Name}/life{life}"),"Qualified component record missing.");
        var graphs=root.GetProperty("graphs").EnumerateArray().ToArray();var graphFixtures=GraphFixtures();Need(graphs.Length==2,"Both graph module lifetimes mandatory.");
        for(int life=0;life<2;life++)
        {
            var g=graphs.Single(x=>x.GetProperty("lifetime").GetInt32()==life);Need(g.GetProperty("passed").GetBoolean()&&g.GetProperty("actualGraphReplays").GetInt32()==12&&g.GetProperty("actualGpuKernelExecutions").GetInt32()==120
                &&g.GetProperty("captureHostAdmissionKeys").GetInt32()==65&&g.GetProperty("fixedWorkspaceBytes").GetInt64()==992256&&g.GetProperty("fixedPages").GetInt32()==640&&g.GetProperty("sameDeviceAddressesAcrossReplays").GetBoolean(),"Qualified graph coverage/identity differs.");
            var nodes=g.GetProperty("capturedNodes").EnumerateArray().ToArray();Need(nodes.Length==4&&nodes.Select(x=>x.GetProperty("role").GetString()).ToHashSet().SetEquals(new[]{"pack","core","reduce","gate"})&&nodes.Single(x=>x.GetProperty("role").GetString()=="core").GetProperty("dynamicExternalPosition").GetBoolean(),"Qualified complete-chain/external-position graph proof missing.");
            var passes=g.GetProperty("passes").EnumerateArray().ToArray();Need(passes.Length==12,"All dynamic graph passes mandatory.");
            for(int p=0;p<3;p++)for(int variant=0;variant<4;variant++)
            {
                var x=passes.Single(x=>x.GetProperty("pass").GetInt32()==p&&x.GetProperty("variant").GetInt32()==variant);
                Need(x.GetProperty("past").GetInt32()==graphFixtures[variant].Case.Past&&x.GetProperty("keys").GetInt32()==graphFixtures[variant].Case.Keys&&x.GetProperty("poisonByte").GetInt32()==Poisons[p]
                    &&S.Matches(x.GetProperty("querySha256").GetString()!,Sha(Bytes(graphFixtures[variant].Query))),"Graph update/input identity differs.");
                foreach(string key in new[]{"passed","originalCoreBitsExact","originalPartialBitsExact","graphVsUncapturedAllBitsExact","packedQueryBitsExact","immutableAndUpdatedInputs","allAllocationGuards","aaAndCrossLifetimeBits"})Need(x.GetProperty(key).GetBoolean(),"Qualified graph gate failed: "+key);
                Numeric(x.GetProperty("coreFp64"));Numeric(x.GetProperty("gatedFp64"));foreach(var file in x.GetProperty("captured").EnumerateObject()){string path=file.Value.GetString()!;Need(files.ContainsKey(path),"Qualified graph output missing.");allRefs.Add(path);}
            }
        }
        Need(allRefs.SetEquals(files.Keys),"Qualified raw CAS objects omitted or unreferenced.");return new(candidate,legacy,core,files);
    }
    private static double Median(IEnumerable<double> values){var a=values.OrderBy(x=>x).ToArray();Need(a.Length==10,"Ten paired samples perarm required.");return(a[4]+a[5])/2;}
    private static bool RunTiming(CudaNinferPrefillAttentionModule module,LegacySplit legacy,R.Fixture f,byte[] qualifiedCandidate,byte[] qualifiedLegacy,byte[] qualifiedCore,
        string output,Dictionary<string,string> files,List<Dictionary<string,object?>> timings,Action checkpoint)
    {
        Need(Pass(Check(qualifiedCandidate,f,false))&&Pass(Check(qualifiedLegacy,f,false))&&Pass(Check(qualifiedCore,f,true)),"Qualified complete outputs violate fixed independent bounds.");
        using var memory=new Memory();using var stream=new StreamScope();var c=f.Case;var p=CudaNinferPrefillAttentionModule.PlanVerify8(c.Keys,MaxBlocks);
        var q=memory.Put("Q-F32",Bytes(f.QueryF32));var gate=memory.Put("gate",Bytes(f.Gate));var positions=memory.Put("external-base",Bytes(new[]{c.Past}));
        var ends=memory.Put("key-ends",Bytes(Enumerable.Range(1,8).Select(i=>c.Past+i).ToArray()));var cache=MakeCache(memory,f);
        var workspace=memory.Empty("candidate-workspace",p.Bytes);var result=memory.Empty("candidate-gated",Elements*4);var control=memory.Empty("legacy-gated",Elements*4);
        var oldPlan=Q35Kernels.GetSplitAttentionPlan(c.Keys);var partial=memory.Empty("legacy-partial",Rows*24L*oldPlan.Splits*258*4);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
        using var candidateGraph=new Captured(stream.Pointer,()=>module.RunF32Verify8(q.Pointer,gate.Pointer,cache.Table.Pointer,MaxBlocks,KOffset,VOffset,positions.Pointer,c.Keys,.0625f,result.Pointer,workspace.Pointer,p.Bytes,stream.Pointer),positions.Pointer,workspace.Pointer+(nint)p.Positions);
        using var controlGraph=new Captured(stream.Pointer,()=>legacy.Run(q.Pointer,gate.Pointer,ends.Pointer,cache.Table.Pointer,KOffset,VOffset,c.Keys,partial.Pointer,control.Pointer,stream.Pointer),IntPtr.Zero,IntPtr.Zero,false);
        var intervals=new List<object>();var candidateSamples=new List<double>();var legacySamples=new List<double>();var roundWins=new List<bool>();
        var record=new Dictionary<string,object?>{["case"]=c.Name,["keys"]=c.Keys,["past"]=c.Past,["rows"]=8,["candidateCapturedNodes"]=candidateGraph.Nodes,["legacyCapturedNodes"]=controlGraph.Nodes,
            ["candidateQualifiedSha256"]=Sha(qualifiedCandidate),["legacyQualifiedSha256"]=Sha(qualifiedLegacy),["intervals"]=intervals,["passed"]=false};timings.Add(record);checkpoint();
        for(int round=0;round<5;round++)
        {
            var samples=new double[4];bool[] order={false,true,true,false};
            for(int position=0;position<4;position++)
            {
                bool candidate=order[position];var graph=candidate?candidateGraph:controlGraph;var armOutput=candidate?result:control;
                foreach(var b in candidate?new[]{workspace,result}:new[]{partial,control})Memory.Poison(b,Poisons[(round+position)%3]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                graph.Run(stream.Pointer);stream.Sync(); // one full-chain warmup outside events, both arms symmetric.
                CudaDriverApi.cuEventCreate(out var begin,0).ThrowOnError();IntPtr end=IntPtr.Zero;float milliseconds;
                try
                {
                    CudaDriverApi.cuEventCreate(out end,0).ThrowOnError();CudaDriverApi.cuEventRecord(begin,stream.Pointer).ThrowOnError();
                    for(int replay=0;replay<6;replay++)graph.Run(stream.Pointer);
                    CudaDriverApi.cuEventRecord(end,stream.Pointer).ThrowOnError();CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out milliseconds,begin,end).ThrowOnError();
                }
                finally{if(end!=IntPtr.Zero)CudaDriverApi.cuEventDestroy(end).ThrowOnError();CudaDriverApi.cuEventDestroy(begin).ThrowOnError();}
                Need(float.IsFinite(milliseconds)&&milliseconds>0,"Invalid full-chain CUDA event interval.");
                byte[] actual=Memory.Body(armOutput);bool exact=actual.AsSpan().SequenceEqual(candidate?qualifiedCandidate:qualifiedLegacy),bridge=true;
                if(candidate)bridge=Memory.Read(workspace.Pointer+(nint)p.Query,Elements*2).AsSpan().SequenceEqual(Bytes(f.Query))&&Memory.Read(workspace.Pointer+(nint)p.Output,Elements*2).AsSpan().SequenceEqual(qualifiedCore);
                bool guards=memory.CheckGuards(),immutable=memory.CheckInputs();var numeric=Check(actual,f,false);bool ok=exact&&bridge&&guards&&immutable&&Pass(numeric);
                double perReplay=milliseconds/6; samples[position]=perReplay;(candidate?candidateSamples:legacySamples).Add(perReplay);
                intervals.Add(new{round,position,arm=candidate?"candidate4-node":"legacy2-node",timedGraphReplays=6,outsideWarmups=1,intervalMilliseconds=milliseconds,millisecondsPerFullChain=perReplay,
                    outputFile=Cas(actual,output,files),fullQualifiedOutputBitsEqual=exact,candidatePackAndCoreBitsExact=bridge,allAllocationGuards=guards,immutableInputs=immutable,fp64=numeric,passed=ok});checkpoint();Need(ok,"Timed captured full chain changed qualified bits/inputs/guards.");
            }
            roundWins.Add((samples[1]+samples[2])/2<(samples[0]+samples[3])/2);
        }
        Need(module.PagedCoreLaunches==1&&legacy.KernelSubmissions==2&&intervals.Count==20,"Only one capture per full chain is permitted; all timings must be graph replays.");
        double legacyMedian=Median(legacySamples),candidateMedian=Median(candidateSamples),speedup=legacyMedian/candidateMedian;int wins=roundWins.Count(x=>x);bool win=speedup>1.05&&wins>=4;
        record["legacyMedianMilliseconds"]=legacyMedian;record["candidateMedianMilliseconds"]=candidateMedian;record["medianSpeedup"]=speedup;record["pairedRoundWins"]=wins;record["roundWinFlags"]=roundWins;
        record["predeclaredFullChainWin"]=win;record["actualTimedGraphReplays"]=120;record["actualOutsideWarmupReplays"]=20;record["actualGpuKernelExecutions"]=420;record["passed"]=true;checkpoint();return win;
    }
    private static int TimingMain(string[] args)
    {
        string? output=null;bool owned=false;var timings=new List<Dictionary<string,object?>>();var files=new Dictionary<string,string>();
        var state=new Dictionary<string,object?>{["schema"]="ninfer-original-bf16-small-attention-verify8-timing/1",["complete"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["timings"]=timings,["files"]=files};
        try
        {
            Need(args.Length==15&&args[0]=="--timing"&&args[1]=="--module"&&args[3]=="--expect-sha256"&&args[5]=="--legacy-module"&&args[7]=="--expect-legacy-sha256"&&args[9]=="--component-certificate"&&args[11]=="--expect-certificate-sha256"&&args[13]=="--out","Exact explicit timing arguments required.");
            Need(S.Matches(args[4],CandidateSha)&&S.Matches(args[8],LegacySha),"Timing requires unchanged qualified images.");
            string image=Pinned(args[2],args[4]),legacyImage=Pinned(args[6],args[8]),certificate=Pinned(args[10],args[12]),folder=Path.GetDirectoryName(certificate)!;byte[] certificateBytes=File.ReadAllBytes(certificate);
            Need(S.Matches(Sha(certificateBytes),args[12]),"Certificate changed while opening.");var qualified=ReadQualification(certificateBytes,folder);
            output=args[14];Fresh(output);Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;NewFile(Path.Combine(output,"component-certificate.json"),certificateBytes);
            state["moduleSha256"]=CandidateSha;state["legacyModuleSha256"]=LegacySha;state["componentCertificateSha256"]=S.Normalize(args[12]);state["toolAssemblySha256"]=FileSha(typeof(Program).Assembly.Location);state["backendAssemblySha256"]=FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"]="Paired CUDA-event captured full chains: originalPack+Core+Reduce+Gate versus actualR12 legacySplit128+Combine. FixedT8/B1, samebuffers/stream, symmetric one-replay warmups outside events. No physicalcoldcache or model/E2E claim.";
            state["protocol"]=new{keys=new[]{74,202,512,1024},rounds=5,order=new[]{"legacy","candidate","candidate","legacy"},timedReplaysPerInterval=6,outsideWarmupsPerInterval=1,minimumMedianSpeedupExclusive=1.05,minimumPairedRoundWins=4,allFourCasesMustWin=true};
            void Checkpoint()=>Save(Path.Combine(output!,"result.json"),state);Checkpoint();bool allWin=true;CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
            byte[] Read(string key){var bytes=File.ReadAllBytes(Path.Combine(folder,key));Need(S.Matches(Sha(bytes),qualified.Files[key]),"Qualified output changed while opening.");return bytes;}
            using(var context=CudaContext.Create(0))
            {
                state["device"]=DeviceIdentity();state["GPU"]="ATTEMPTED";
                foreach(int keys in new[]{74,202,512,1024})
                {
                    var c=Cases().Single(x=>x.Keys==keys&&x.Pattern=="dense");var f=R.Make(c);using var module=new CudaNinferPrefillAttentionModule(image);using var legacy=new LegacySplit(legacyImage);
                    Need(S.Matches(module.ImageSha256,CandidateSha)&&S.Matches(legacy.ImageSha256,LegacySha),"Actually loaded timed images differ.");
                    allWin&=RunTiming(module,legacy,f,Read(qualified.Candidate[c.Name]),Read(qualified.Legacy[c.Name]),Read(qualified.Core[c.Name]),output,files,timings,Checkpoint);
                }
            }
            Need(timings.Count==4&&timings.All(x=>(bool)x["passed"]!&&((List<object>)x["intervals"]!).Count==20),"All four complete timing series mandatory.");
            Need(files.Count<=8&&files.All(x=>S.Matches(FileSha(Path.Combine(output,x.Key)),x.Value))&&qualified.Files.All(x=>S.Matches(FileSha(Path.Combine(folder,x.Key)),x.Value)),"Full qualified/timed CAS evidence changed.");
            Need(S.Matches(FileSha(image),CandidateSha)&&S.Matches(FileSha(legacyImage),LegacySha)&&S.Matches(FileSha(certificate),args[12]),"Pinned inputs changed during timing.");
            state["GPU"]="EXECUTED";state["complete"]=true;state["passed"]=true;state["kernelPerformance"]="MEASURED";state["allFourCasesMeetPredeclaredFullChainWin"]=allWin;state["actualTimedGraphReplays"]=480;state["actualOutsideWarmupReplays"]=80;state["actualGpuKernelExecutions"]=1680;Checkpoint();Complete(output,state);
            Console.WriteLine(JsonSerializer.Serialize(new{complete=true,passed=true,cases=4,allFourCasesMeetPredeclaredFullChainWin=allWin,actualGpuKernelExecutions=1680,modelQuality="NOT_RUN",endToEndPerformance="NOT_RUN"}));return 0;
        }
        catch(Exception e){state["passed"]=false;state["error"]=e.ToString();if(owned&&output!=null)Save(Path.Combine(output,"result.json"),state);Console.Error.WriteLine(e);return 1;}
    }
}
