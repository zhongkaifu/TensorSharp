// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private static void Need(bool value,string message){if(!value)throw new InvalidDataException(message);}
    private static string Sha(ReadOnlySpan<byte> data)=>Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    private static string FileSha(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    private static void NewFile(string path,byte[] bytes){using var s=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);s.Write(bytes);s.Flush(true);}
    private static void Save(string path,object value)=>File.WriteAllBytes(path,JsonSerializer.SerializeToUtf8Bytes(value,Json));
    private static string Cas(byte[] bytes,string output,Dictionary<string,string> files)
    {string h=Sha(bytes),rel="cas/"+h+".bin",p=Path.Combine(output,rel);if(File.Exists(p))Need(File.ReadAllBytes(p).AsSpan().SequenceEqual(bytes),"CAS byte conflict.");else NewFile(p,bytes);files[rel]=h;return rel;}
    private static void Pinned(string p,string sha)=>Need(Path.IsPathFullyQualified(p)&&Path.GetFullPath(p)==p&&File.Exists(p)&&S.Matches(FileSha(p),sha),"Existing normalized module path and exact SHA required.");
    private static void Component(string name,int life,Modules modules,Stream stream,string output,List<Dictionary<string,object?>> records,Dictionary<string,string> files,Dictionary<string,string> aa,Action checkpoint)
    {
        var f=Make(name);using var memory=new Memory();var x=memory.Put("input-f32-bf16",Bytes(f.Input));var wc=memory.Put("weight-codes-ts",f.WeightCodes);var ws=memory.Put("weight-scales-ts",f.WeightScales);var arms=Enumerable.Range(0,3).Select(i=>new GemmArm(memory,"arm"+i)).ToArray();var passes=new List<object>();
        var record=new Dictionary<string,object?>{["id"]=$"{Geometry.Name}/{name}/life{life}",["fixture"]=name,["lifetime"]=life,["shape"]=new[]{Rows,N,Width},["role"]=Geometry.Name,["splits"]=Geometry.Splits,["consumer"]=Geometry.Consumer,["partialPolicy"]=Geometry.Splits==1?"unused-guards-only":"full-split-partials-bits",["inputDivisor"]=f.InputD,["weightDivisor"]=f.WeightD,["alphaBits"]=BitConverter.SingleToInt32Bits(f.Alpha),["sameImmutableOperandPointers"]=true,
            ["inputSha256"]=new{x=Sha(Bytes(f.Input)),weightCodes=Sha(f.WeightCodes),weightScales=Sha(f.WeightScales),codes=Sha(f.Codes),scales=Sha(f.Scales)},["passes"]=passes,["passed"]=false};records.Add(record);checkpoint();
        for(int pass=0;pass<3;pass++)
        {
            foreach(var arm in arms)arm.Reset(Poisons[pass]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            for(int i=0;i<3;i++){modules.Quantize(f,x,arms[i],stream.Pointer);modules.Project(f,wc,ws,arms[i],i,stream.Pointer);}stream.Sync();
            var before=arms.Select(a=>Memory.Body(a.Output)).ToArray();var partial=Geometry.Splits==1?Array.Empty<byte[]>():arms.Select(a=>Memory.Body(a.Partial)).ToArray();var numeric=before.Select(a=>Check(a,f,false)).ToArray();
            bool exact=before[0].AsSpan().SequenceEqual(before[1])&&before[0].AsSpan().SequenceEqual(before[2]),part=Geometry.Splits==1||(partial[0].AsSpan().SequenceEqual(partial[1])&&partial[0].AsSpan().SequenceEqual(partial[2])),quant=arms.All(a=>a.QuantBits(f)),counters=Geometry.Splits==1||arms.All(a=>a.CountersZero());
            foreach(var arm in arms)modules.Consume(arm,stream.Pointer);stream.Sync();var after=arms.Select(a=>Memory.Body(a.Final)).ToArray();var finalFormat=after.Select(ConsumerFormat).ToArray();bool boundary=true;
            if(Geometry.Consumer=="round")for(int i=0;i<3;i++){var want=System.Runtime.InteropServices.MemoryMarshal.Cast<byte,float>(before[i]).ToArray().Select(Bf).ToArray();boundary&=after[i].AsSpan().SequenceEqual(Bytes(want));}
            bool finalExact=after[0].AsSpan().SequenceEqual(after[1])&&after[0].AsSpan().SequenceEqual(after[2]),inputs=memory.Inputs(),guards=memory.Guards(),repeated=true;var captured=new List<object>();
            for(int i=0;i<3;i++)
            {
                var outputs=new List<(string,byte[])>{("f32",before[i]),("consumer",after[i])};if(Geometry.Splits>1)outputs.Add(("partial",partial[i]));
                foreach(var pair in outputs)
                {string key=name+"/"+i+"/"+pair.Item1,h=Sha(pair.Item2);if(aa.TryGetValue(key,out var previous)&&h!=previous)repeated=false;else aa.TryAdd(key,h);}
                captured.Add(new{arm=i==0?"actual-old32":i==1?"same-source32":"candidate8",f32=Cas(before[i],output,files),consumer=Cas(after[i],output,files),partial=Geometry.Splits>1?Cas(partial[i],output,files):null});
            }
            bool ok=exact&&part&&quant&&counters&&boundary&&finalExact&&inputs&&guards&&repeated&&numeric.All(a=>a.Passed)&&finalFormat.All(a=>a.Passed);
            passes.Add(new{pass,poison=Poisons[pass],fullF32BitsEqual=exact,fullPartialF32BitsEqual=Geometry.Splits>1?(bool?)part:null,unusedScratchGuardsOnly=Geometry.Splits==1,sameQuantAndCpuBits=quant,countersReturnedZero=Geometry.Splits>1?(bool?)counters:null,cpuRoundBoundaryChecked=Geometry.Consumer=="round",roundBoundaryCpuBits=boundary,fullConsumerBitsEqual=finalExact,immutableInputs=inputs,allocationGuards=guards,aaAndCrossLifetime=repeated,fp64Before=numeric,consumerFormat=finalFormat,captured,passed=ok});checkpoint();Need(ok,"A4 component failed; full three-arm outputs retained.");
        }
        record["actualKernelLaunches"]=9*NodesPerChain;record["passed"]=true;checkpoint();
    }
    private static int Main(string[] args)
    {
        int shapeIndex=Array.IndexOf(args,"--shape");if(shapeIndex>=0){Need(shapeIndex+1<args.Length&&args.Count(x=>x=="--shape")==1,"Exactly one named --shape required.");string name=args[shapeIndex+1];Geometry=Shapes.SingleOrDefault(x=>x.Name==name)??throw new ArgumentException("Unknown shape; use gdn-input,gate-up,down,output,gqa-input.");args=args.Take(shapeIndex).Concat(args.Skip(shapeIndex+2)).ToArray();}
        if(args.SequenceEqual(new[]{"--self-test"})){SelfTest();return 0;}string? output=null;bool owned=false;var records=new List<Dictionary<string,object?>>();var files=new Dictionary<string,string>();var aa=new Dictionary<string,string>();
        var state=new Dictionary<string,object?>{["schema"]="ninfer-a4-t8-component/2",["complete"]=false,["passed"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["records"]=records,["files"]=files};
        try
        {
            bool timing=args.Length==15&&args[^1]=="--timing";Need((args.Length==14||timing)&&args[0]=="--helper-module"&&args[2]=="--expect-helper-sha256"&&args[4]=="--legacy-module"&&args[6]=="--expect-legacy-sha256"&&args[8]=="--candidate-module"&&args[10]=="--expect-candidate-sha256"&&args[12]=="--out","Usage: --shape ENUM --helper-module FILE --expect-helper-sha256 SHA --legacy-module FILE --expect-legacy-sha256 SHA --candidate-module FILE --expect-candidate-sha256 SHA --out FRESH_ABSOLUTE_ARTIFACT_DIR [--timing]");
            foreach(int i in new[]{1,5,9})Pinned(args[i],S.Normalize(args[i+2]));output=args[13];Need(Path.IsPathFullyQualified(output)&&Path.GetFullPath(output)==output&&!Directory.Exists(output)&&!File.Exists(output)&&(output.Split(Path.DirectorySeparatorChar).Contains("artifacts")||output.Contains("/docs/validation/",StringComparison.Ordinal)),"Fresh ignored output path required.");
            Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;
            state["modules"]=new{helper=new{path=args[1],sha256=S.Normalize(args[3])},legacy=new{path=args[5],sha256=S.Normalize(args[7])},candidate=new{path=args[9],sha256=S.Normalize(args[11])}};state["toolAssemblySha256"]=FileSha(typeof(Program).Assembly.Location);state["backendAssemblySha256"]=FileSha(typeof(CudaBackend).Assembly.Location);
            state["geometry"]=Geometry;state["scope"]="Fixed five actual T8 projection shapes only; actual old32 versus same-source32 versus new8. Same TS quantization/planes/alpha/rowScale. Strict beforeconsumer F32 bits and decoded FP64 mandatory. Actual unchanged consumer: round forGDN/GQA, SwiGLU forGateUp, none forOutput/Down. No model/quality/E2E proof.";
            state["boundPolicy"]="Beforeconsumer only: gamma_(3K+32,u=2^-24)*abs(alpha)*sumAbsDecodedProducts+2u*abs(FP64ideal). Exact binary fixture independently proves accumulation on fixed lattice<2^24. After unchanged consumers require full GPUbits+finite and BF16format where applicable; no independent SwiGLU FP64 claim.";
            void Checkpoint()=>Save(Path.Combine(output,"result.json"),state);Checkpoint();CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
            using(var context=CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int major,75,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int minor,76,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int sms,16,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int l2,38,device).ThrowOnError();
                Need(major==12&&minor==0&&sms==170,"This fixed scheduling trial is qualified only on the measured170SM CC12.0 device.");state["device"]=new{major,minor,sms,l2Bytes=l2};using var stream=new Stream();state["GPU"]="EXECUTED";
                for(int life=0;life<2;life++)
                {
                    using var modules=new Modules(args[1],args[3],args[5],args[7],args[9],args[11]);foreach(string name in new[]{"exact-binary","broad-finite","ties-zero-and-last-column"})Component(name,life,modules,stream,output,records,files,aa,Checkpoint);
                    if(life==1&&timing)RunTiming(modules,stream,output,files,state,Checkpoint);
                }
                Need(records.Count==6&&records.Select(x=>(string)x["id"]!).Distinct().Count()==6&&records.All(x=>x["passed"] is true),"All six unique lifecycle records required.");stream.Sync();
            }
            foreach(var f in files)Need(f.Key=="cas/"+f.Value+".bin"&&S.Matches(FileSha(Path.Combine(output,f.Key)),f.Value),"Full evidence rehash failed.");state["componentPassed"]=true;state["actualComponentKernelLaunches"]=54*NodesPerChain;state["allRawEvidenceRehashed"]=true;state["complete"]=true;state["passed"]=true;
            byte[] completed=JsonSerializer.SerializeToUtf8Bytes(state,Json);NewFile(Path.Combine(output,"completed-result.json"),completed);NewFile(Path.Combine(output,"completed-result.json.sha256"),System.Text.Encoding.ASCII.GetBytes(Sha(completed)+"  completed-result.json\n"));Save(Path.Combine(output,"result.json"),state);Console.WriteLine(JsonSerializer.Serialize(new{passed=true,output,role=Geometry.Name,componentKernelLaunches=54*NodesPerChain,timingRequested=timing}));return 0;
        }catch(Exception error){state["complete"]=false;state["passed"]=false;state["error"]=error.ToString();if(owned&&output!=null)Save(Path.Combine(output,"result.json"),state);Console.Error.WriteLine(error);return 1;}
    }
}
