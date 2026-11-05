// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private const string ActualEaSha="ea0e322a36c48211750e0f569bd12c44f9d62caa6d3249c37f1547679bdccbea";
    private const string FusedSymbol="ts_q35_gemm_w4a4_8_swiglu_bf16";
    private const uint DynamicShared=43520;
    private const long RawBudget=128L<<20;
    private static readonly string[] ArmNames={"actual-ea8-plus-helper-swiglu","same-source8-plus-helper-swiglu","fused8-swiglu-bf16"};
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.CamelCase};
    static Program(){Geometry=Shapes.Single(s=>s.Name=="gate-up");}
    private static void Need(bool value,string message){if(!value)throw new InvalidDataException(message);}
    private static string Sha(ReadOnlySpan<byte> bytes)=>Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileSha(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static void NewFile(string path,byte[] bytes){using var s=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);s.Write(bytes);s.Flush(true);}
    private static void Save(string path,object value)=>File.WriteAllBytes(path,JsonSerializer.SerializeToUtf8Bytes(value,Json));
    private static string Cas(byte[] bytes,string output,Dictionary<string,string> files)
    {
        string h=Sha(bytes),rel="cas/"+h+".bin",p=Path.Combine(output,rel);
        if(File.Exists(p))Need(File.ReadAllBytes(p).AsSpan().SequenceEqual(bytes),"CAS byte conflict.");else{Need(Directory.EnumerateFiles(Path.Combine(output,"cas")).Sum(x=>new FileInfo(x).Length)+bytes.Length<=RawBudget,"Fixed128MiB raw budget exceeded.");NewFile(p,bytes);}files[rel]=h;return rel;
    }
    private static void Pinned(string p,string hash)=>Need(Path.IsPathFullyQualified(p)&&Path.GetFullPath(p)==p&&File.Exists(p)&&S.Matches(FileSha(p),hash),"Existing normalized module and explicit SHA required.");
    private static bool Repeat(Dictionary<string,string> aa,string key,byte[] bytes){string h=Sha(bytes);if(aa.TryGetValue(key,out var previous))return h==previous;aa[key]=h;return true;}
    private static void FixedGeometry(int rows,int n,int k,int splits)=>Need(rows==8&&n==34816&&k==5120&&splits==1,"Only T8,N34816,K5120,split1 GateUp is admitted.");
    private static void Component(string name,int life,Modules modules,Stream stream,string output,List<Dictionary<string,object?>> records,Dictionary<string,string> files,Dictionary<string,string> aa,Action checkpoint)
    {
        var f=Make(name);using var memory=new Memory();var x=memory.Put("canonical-input",Bytes(f.Input));var wc=memory.Put("weight-codes",f.WeightCodes);var ws=memory.Put("weight-scales",f.WeightScales);var arms=Enumerable.Range(0,3).Select(i=>new GemmArm(memory,ArmNames[i],i==2)).ToArray();var passes=new List<object>();
        var record=new Dictionary<string,object?>{["id"]=$"{name}/life{life}",["fixture"]=name,["lifetime"]=life,["shape"]=new[]{Rows,N,Width},["splits"]=1,["inputDivisor"]=f.InputD,["weightDivisor"]=f.WeightD,["alphaBits"]=BitConverter.SingleToInt32Bits(f.Alpha),["inputSha256"]=new{x=Sha(Bytes(f.Input)),weightCodes=Sha(f.WeightCodes),weightScales=Sha(f.WeightScales),expectedCodes=Sha(f.Codes),expectedScales=Sha(f.Scales)},["passes"]=passes,["passed"]=false};records.Add(record);checkpoint();
        for(int pass=0;pass<3;pass++)
        {
            foreach(var arm in arms)arm.Reset(Poisons[pass]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            for(int i=0;i<3;i++){modules.Quantize(f,x,arms[i],stream.Pointer);modules.Project(f,wc,ws,arms[i],i,stream.Pointer);}stream.Sync();
            var before=arms.Take(2).Select(a=>Memory.Body(a.Output)).ToArray();var numeric=before.Select(b=>Check(b,f,false)).ToArray();bool rawBits=before[0].AsSpan().SequenceEqual(before[1]);
            for(int i=0;i<2;i++)modules.Consume(arms[i],stream.Pointer);stream.Sync();var after=arms.Select(a=>Memory.Body(a.Final)).ToArray();var formats=after.Select(ConsumerFormat).ToArray();bool finalBits=after.Skip(1).All(b=>b.AsSpan().SequenceEqual(after[0]));
            bool quant=arms.All(a=>a.QuantBits(f)),guards=memory.Guards(),inputs=memory.Inputs(),repeated=true;var captures=new List<object>();
            for(int i=0;i<3;i++)
            {
                repeated&=Repeat(aa,name+"/"+i+"/final",after[i]);if(i<2)repeated&=Repeat(aa,name+"/"+i+"/raw",before[i]);
                captures.Add(new{arm=ArmNames[i],beforeConsumerStatus=i<2?"CAPTURED_FULL_F32":"NOT_CAPTURED_FUSED_EPILOGUE",beforeConsumerFile=i<2?Cas(before[i],output,files):null,beforeConsumerFp64=i<2?numeric[i]:null,finalFile=Cas(after[i],output,files),finalFormat=formats[i],quantCodesFile=Cas(Memory.Body(arms[i].Codes),output,files),quantScalesFile=Cas(Memory.Body(arms[i].Scales),output,files),quantRowScaleFile=Cas(Memory.Body(arms[i].RowScale),output,files)});
            }
            bool ok=rawBits&&numeric.All(n=>n.Passed)&&finalBits&&formats.All(f=>f.Passed)&&quant&&guards&&inputs&&repeated;
            passes.Add(new{pass,poison=Poisons[pass],controlProjectionFullF32BitsExact=rawBits,full139264FinalF32BitsExact=finalBits,quantizedPayloadCpuBits=quant,unusedScratchGuardsOnly=true,countersReturnedZero=(bool?)null,allocationGuards=guards,immutableInputs=inputs,aaAndCrossLifetime=repeated,candidateBeforeConsumerStatus="NOT_CAPTURED",captures,passed=ok});checkpoint();Need(ok,"GateUp fusion component failed; full available outputs retained.");
        }
        record["actualKernelExecutions"]=24;record["passed"]=true;checkpoint();
    }
    private static void GateUpSelfTest()
    {
        FixedGeometry(Rows,N,Width,Geometry.Splits);Need(FinalCount==139264&&OutputCount==278528&&DynamicShared==43520,"Fixed raw/final dimensions changed.");
        var original=Console.Out;using var discarded=new StringWriter();try{Console.SetOut(discarded);SelfTest();}finally{Console.SetOut(original);}
        Action[] invalid={()=>FixedGeometry(7,N,Width,1),()=>FixedGeometry(8,N,Width,2),()=>FixedGeometry(8,N-128,Width,1),()=>FixedGeometry(8,N,Width+256,1)};
        foreach(var call in invalid){bool refused=false;try{call();}catch(InvalidDataException){refused=true;}Need(refused,"Unsupported fused geometry admitted.");}
        Need(System.Runtime.InteropServices.Marshal.SizeOf<GraphApi.KernelParams>()==56,"CUDA graph v1 ABI changed.");
        Console.WriteLine(JsonSerializer.Serialize(new{CPU="PASS",GPU="NOT_RUN",fixtures=3,shape=new[]{Rows,N,Width},splits=1,rawControlValues=OutputCount,finalValues=FinalCount,moduleLifetimes=2,poisons=3,componentKernelExecutions=144,candidateRaw="NOT_CAPTURED",numericBeforeConsumerScope="Full controls0/1 only; unchanged reference and tolerance",unsupportedGeometryContracts=invalid.Length,modelQuality="NOT_RUN",endToEndPerformance="NOT_RUN"}));
    }
    private static int Main(string[] args)
    {
        if(args.SequenceEqual(new[]{"--self-test"})){GateUpSelfTest();return 0;}if(args.Length==0){Console.WriteLine("{\"mode\":\"PLAN\",\"GPU\":\"NOT_RUN\"}");return 0;}
        string? output=null;bool owned=false;var records=new List<Dictionary<string,object?>>();var files=new Dictionary<string,string>();var aa=new Dictionary<string,string>();var state=new Dictionary<string,object?>{["schema"]="ninfer-gateup-swiglu-fusion-component-and-timing/1",["complete"]=false,["passed"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["records"]=records,["files"]=files};
        try
        {
            bool timing=args.Length==15&&args[^1]=="--timing";Need((args.Length==14||timing)&&args[0]=="--helper-module"&&args[2]=="--expect-helper-sha256"&&args[4]=="--legacy-module"&&args[6]=="--expect-legacy-sha256"&&args[8]=="--candidate-module"&&args[10]=="--expect-candidate-sha256"&&args[12]=="--out","Usage: --helper-module FILE --expect-helper-sha256 SHA --legacy-module ACTUAL_EA --expect-legacy-sha256 SHA --candidate-module FILE --expect-candidate-sha256 SHA --out FRESH_IGNORED_DIR [--timing]");
            foreach(int i in new[]{1,5,9})Pinned(args[i],S.Normalize(args[i+2]));Need(S.Normalize(args[7])==ActualEaSha,"Primary control must be actual previously qualified EA image.");output=args[13];Need(Path.IsPathFullyQualified(output)&&Path.GetFullPath(output)==output&&!Directory.Exists(output)&&!File.Exists(output)&&(output.Split(Path.DirectorySeparatorChar).Contains("artifacts")||output.Contains("/docs/validation/",StringComparison.Ordinal)),"Fresh ignored output required.");Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;
            state["modules"]=new{helper=new{path=args[1],sha256=S.Normalize(args[3])},actualEa=new{path=args[5],sha256=S.Normalize(args[7])},candidate=new{path=args[9],sha256=S.Normalize(args[11])}};state["toolAssemblySha256"]=FileSha(typeof(Program).Assembly.Location);state["backendAssemblySha256"]=FileSha(typeof(CudaBackend).Assembly.Location);state["geometry"]=Geometry;
            state["scope"]="Fixed T8/N34816/K5120/split1 actualEA8+originalhelper, same-source8+originalhelper, fused8 finalBF16widened. Full controls rawF32 exact/F64; fused raw is NOT_CAPTURED. All final values must equal actual original consumer bits. No model/quality/E2E, cold-memory or all-shape claim.";
            state["boundPolicy"]="Unchanged Reference.cs: gamma_(3K+32,u=2^-24)*abs(alpha)*sumAbsDecodedProducts+2u*abs(FP64ideal), beforeconsumer only. Full control output on three fixtures; dense timing only fixed active-row witnesses. No independent expf/SwiGLU FP64 claim and no invented fused raw values.";
            void Checkpoint()=>Save(Path.Combine(output,"result.json"),state);Checkpoint();CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
            using(var context=CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int major,75,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int minor,76,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int sms,16,device).ThrowOnError();Need(major==12&&minor==0&&sms==170,"Fixed qualified device required.");state["device"]=new{ordinal=0,driver,major,minor,sms};using var stream=new Stream();
                for(int life=0;life<2;life++){using var modules=new Modules(args[1],args[3],args[5],args[7],args[9],args[11]);state["GPU"]="EXECUTED";state["actualGemmResources"]=modules.Resources;foreach(string name in new[]{"exact-binary","broad-finite","ties-zero-and-last-column"})Component(name,life,modules,stream,output,records,files,aa,Checkpoint);if(life==1&&timing)RunTiming(modules,stream,output,files,state,Checkpoint);stream.Sync();}
                Need(records.Count==6&&records.Select(r=>(string)r["id"]!).Distinct().Count()==6&&records.All(r=>r["passed"] is true),"All six fixture/lifetime records required.");stream.Sync();
            }
            foreach(var f in files)Need(f.Key=="cas/"+f.Value+".bin"&&S.Matches(FileSha(Path.Combine(output,f.Key)),f.Value),"Raw evidence rehash failed.");state["componentPassed"]=true;state["actualComponentKernelExecutions"]=144;state["allRawEvidenceRehashed"]=true;state["complete"]=true;state["passed"]=true;byte[] b=JsonSerializer.SerializeToUtf8Bytes(state,Json);NewFile(Path.Combine(output,"completed-result.json"),b);NewFile(Path.Combine(output,"completed-result.json.sha256"),System.Text.Encoding.ASCII.GetBytes(Sha(b)+"  completed-result.json\n"));Save(Path.Combine(output,"result.json"),state);Console.WriteLine(JsonSerializer.Serialize(new{passed=true,output,componentKernelExecutions=144,timingRequested=timing}));return 0;
        }catch(Exception e){state["complete"]=false;state["passed"]=false;state["error"]=e.ToString();if(owned&&output!=null)Save(Path.Combine(output,"result.json"),state);Console.Error.WriteLine(e);return 1;}
    }
}
