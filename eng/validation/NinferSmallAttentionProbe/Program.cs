// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;
using S = TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private const string CandidateSha="4ba17472c196242b88adfe27c4446c136abae8c57225593e1eac16acfe6d4602";
    private const string LegacySha="b6754c46a8b0888eb6760a5027ae91a6f971e7b54e72adeb8ba98945ea37abc0";
    private const string ReferenceSha="533b1b1ac29f623b9b255bb8c94321743b2b90f3d0b4f904bbd48ec51e3ff5d7";
    private const string DeviceHelpersSha="759c122e8a6c718f07b87f832415a4329772a0f8224ef8dd3bea2432f657035a";
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.CamelCase};
    private static void Save(string path,object data){using var f=File.Create(path);JsonSerializer.Serialize(f,data,Json);f.WriteByte(10);}
    private static void Complete(string path,object data)
    {
        byte[] bytes=JsonSerializer.SerializeToUtf8Bytes(data,Json).Concat(new byte[]{10}).ToArray();
        NewFile(Path.Combine(path,"completed-result.json"),bytes);NewFile(Path.Combine(path,"completed-result.json.sha256"),Encoding.ASCII.GetBytes(Sha(bytes)+"  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path)&&Path.GetFullPath(path)==path,"Canonical absolute output required.");
        var parts=path.Split(Path.DirectorySeparatorChar,StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts")||Enumerable.Range(0,Math.Max(0,parts.Length-1)).Any(i=>parts[i]=="docs"&&parts[i+1]=="validation"),"Ignored output required.");
        bool Link(string p)=>new FileInfo(p).LinkTarget!=null||new DirectoryInfo(p).LinkTarget!=null;
        Need(!Path.Exists(path)&&!Link(path),"Fresh output required.");for(string? p=Path.GetDirectoryName(path);p!=null;p=Path.GetDirectoryName(p))Need(!Link(p)&&!File.Exists(p),"Aliased output parent refused.");
    }
    private static string Pinned(string path,string sha)
    {Need(Path.IsPathFullyQualified(path)&&Path.GetFullPath(path)==path&&File.Exists(path),"Canonical existing pinned file required.");Need(S.Matches(FileSha(path),sha),"Pinned file bytes differ.");return path;}
    private static string Cas(byte[] bytes,string output,Dictionary<string,string> files)
    {
        string sha=Sha(bytes),key="cas/"+sha+".bin",path=Path.Combine(output,key);
        if(files.ContainsKey(key))Need(File.ReadAllBytes(path).AsSpan().SequenceEqual(bytes),"CAS byte collision/change.");
        else{Need(!File.Exists(path),"Unregistered CAS object.");NewFile(path,bytes);files.Add(key,sha);}return key;
    }
    private static object DeviceIdentity()
    {
        CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError();
        CudaDriverApi.cuDeviceGetAttribute(out int sms,16,device).ThrowOnError();Need(sms==170,"This original170SM/T8/S4 qualification is fixed to the measured GPU.");
        byte[] name=new byte[256];CudaDriverApi.cuDeviceGetName(name,name.Length,device).ThrowOnError();return new{ordinal=device,name=Encoding.UTF8.GetString(name).TrimEnd('\0'),driverVersion=driver,sms};
    }
    private static void SelfTest()
    {
        Need(BitConverter.IsLittleEndian&&IntPtr.Size==8,"Little-endian64-bit contract required.");
        Need(Marshal.SizeOf<GraphApi.KernelParams>()==56&&Marshal.OffsetOf<GraphApi.KernelParams>(nameof(GraphApi.KernelParams.Parameters)).ToInt32()==40,"Public CUDA v1 kernel-node ABI differs.");
        Need(S.Matches(CandidateSha,CandidateSha.ToUpperInvariant())&&!S.Matches(CandidateSha,LegacySha),"SHA byte identity failed.");
        foreach(string bad in new[]{CandidateSha[..63],CandidateSha+"0","g"+CandidateSha[1..]})
        {bool rejected=false;try{_=S.Matches(CandidateSha,bad);}catch(ArgumentException){rejected=true;}Need(rejected,"Malformed digest admitted.");}
        var cases=Cases();Need(cases.Length==8&&cases.Select(c=>c.Name).Distinct().Count()==8,"Eight unique T8 cases required.");
        long elements=0;foreach(var c in cases)
        {
            var f=R.Make(c);elements+=f.Query.Length;Need(c.Rows==8&&f.Query.Length==Elements&&f.LogicalK.Length==c.Keys*1024,"Frozen reference T8 span differs.");
            Need(Pass(Check(Bytes(f.IdealGated),f,false))&&Pass(Check(Bytes(f.IdealCore.Select(x=>R.Bf((float)x)).ToArray()),f,true)),"Fixed independent FP64 oracle gate changed.");
            Need(Digests(f).Count==7,"Seven reconstructable source/oracle digests required.");
            var p=CudaNinferPrefillAttentionModule.PlanVerify8(c.Keys,MaxBlocks);Need(p.Bytes==992256&&p.Pages==640&&p.Splits==4&&p.Bytes==CudaNinferPrefillAttentionModule.WorkspaceVerify8(),"Fixed graph-stable layout differs.");
            var old=Q35Kernels.GetSplitAttentionPlan(c.Keys);Need(old.Chunk==128&&old.Splits==(c.Keys+127)/128,"Actual R12 split policy differs.");
        }
        Need(elements==393216,"Fixed T8 full-output count differs.");var fixtures=GraphFixtures();Need(fixtures.Length==4&&fixtures.Select(x=>Sha(Bytes(x.Query))).Distinct().Count()==4,"Graph inputs must change, not only position labels.");
        foreach(var f in fixtures)
        {
            Need(fixtures[^1].LogicalK.AsSpan(0,f.LogicalK.Length).SequenceEqual(f.LogicalK)&&fixtures[^1].LogicalV.AsSpan(0,f.LogicalV.Length).SequenceEqual(f.LogicalV),"Graph fixture cache prefix differs.");
            Need(Pass(Check(Bytes(f.IdealGated),f,false)),"Within-KV-group head transform broke the independent oracle.");
        }
        foreach(int phase in Enumerable.Range(0,4))foreach(int h in Enumerable.Range(0,24))Need(h/6==(h/6*6+(h%6+phase)%6)/6,"Head permutation changed KV mapping.");
        var badBytes=Bytes(fixtures[0].IdealGated);BitConverter.GetBytes(float.NaN).CopyTo(badBytes,0);Need(!Pass(Check(badBytes,fixtures[0],false)),"Nonfinite result admitted.");
        Console.WriteLine(JsonSerializer.Serialize(new{CPU="PASS",cases=8,lifecycleRecords=16,componentPasses=48,componentKernelExecutions=384,graphLifetimes=2,graphReplays=24,graphKernelExecutions=240,
            totalGpuKernelExecutionsPlanned=624,fullOutputValuesPerPass=elements,sourceReferenceSha256=ReferenceSha,sourceDeviceHelpersSha256=DeviceHelpersSha,GPU="NOT_RUN",modelQuality="NOT_RUN",performance="NOT_RUN"}));
    }
    internal static int Main(string[] args)
    {
        string? output=null;bool owned=false;var records=new List<Dictionary<string,object?>>();var graphs=new List<Dictionary<string,object?>>();var files=new Dictionary<string,string>();
        var state=new Dictionary<string,object?>{["schema"]="ninfer-original-bf16-small-attention-verify8/1",["complete"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["performance"]="NOT_RUN",["records"]=records,["graphs"]=graphs,["files"]=files};
        try
        {
            Need(BitConverter.IsLittleEndian,"Little-endian fixture bytes required.");if(args.SequenceEqual(new[]{"--self-test"})){SelfTest();return 0;}
            if(args.Length>0&&args[0]=="--timing")return TimingMain(args);
            if(args.Length==0){Console.WriteLine("PLAN: --execute --module NEW_IMAGE --expect-sha256 SHA --legacy-module OLD_IMAGE --expect-legacy-sha256 SHA --out FRESH_IGNORED_DIR; --timing adds --component-certificate JSON --expect-certificate-sha256 SHA; --self-test CPU-only.");return 0;}
            Need(args.Length==11&&args[0]=="--execute"&&args[1]=="--module"&&args[3]=="--expect-sha256"&&args[5]=="--legacy-module"&&args[7]=="--expect-legacy-sha256"&&args[9]=="--out","Exact explicit execution arguments required.");
            Need(S.Matches(args[4],CandidateSha)&&S.Matches(args[8],LegacySha),"Only unchanged qualified4ba/original andb675/legacy images admitted.");
            string image=Pinned(args[2],args[4]),legacyImage=Pinned(args[6],args[8]);output=args[10];Fresh(output);Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;
            state["modulePath"]=image;state["moduleSha256"]=CandidateSha;state["legacyModulePath"]=legacyImage;state["legacyModuleSha256"]=LegacySha;
            state["policy"]=CudaNinferPrefillAttentionModule.Verify8Policy;state["sourceReferenceSha256"]=ReferenceSha;state["sourceDeviceHelpersSha256"]=DeviceHelpersSha;
            state["toolAssemblySha256"]=FileSha(typeof(Program).Assembly.Location);state["backendAssemblySha256"]=FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"]="Synthetic T8/B1 native BF16 cache: exact unchanged original64-page core and partials versus ownedTS256 accessor; fixed FP64 gated checks; actualR12 split128+combine diagnostic. Dynamic-device-base graph proof over four positions/Q/table updates. Candidate-vs-legacy arithmetic is validly different and is not required bit-equal. No model quality/E2E proof.";
            state["fixedFp64Envelope"]=new{absolute=R.Absolute,relative=R.Relative,scope="Bounded represented-BF16 synthetic fixtures; not a universal error theorem."};
            state["expected"]=new{componentCases=8,componentRecords=16,componentPasses=48,componentKernelExecutions=384,graphRecords=2,graphReplays=24,graphKernelExecutions=240,totalGpuKernelExecutions=624};
            void Checkpoint()=>Save(Path.Combine(output!,"result.json"),state);Checkpoint();var repeats=new Dictionary<string,string>();var graphRepeats=new Dictionary<string,string>();
            CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
            using(var context=CudaContext.Create(0))
            {
                state["device"]=DeviceIdentity();state["GPU"]="ATTEMPTED";
                foreach(var c in Cases())
                {
                    var f=R.Make(c);for(int life=0;life<2;life++)
                    {using var module=new CudaNinferPrefillAttentionModule(image);using var legacy=new LegacySplit(legacyImage);Need(S.Matches(module.ImageSha256,CandidateSha)&&S.Matches(legacy.ImageSha256,LegacySha),"Actually loaded image differs.");RunCase(module,legacy,f,life,output,records,repeats,files,Checkpoint);}
                }
                var fixtures=GraphFixtures();for(int life=0;life<2;life++)
                {using var module=new CudaNinferPrefillAttentionModule(image);Need(S.Matches(module.ImageSha256,CandidateSha),"Actually loaded graph image differs.");RunGraphChecks(module,fixtures,life,output,graphs,files,graphRepeats,Checkpoint);}
            }
            Need(records.Count==16&&records.Select(x=>(string)x["id"]!).Distinct().Count()==16&&records.All(x=>(bool)x["passed"]!&&((List<object>)x["passes"]!).Count==3),"All component cases/lifetimes/poisons mandatory.");
            Need(graphs.Count==2&&graphs.All(x=>(bool)x["passed"]!&&((List<object>)x["passes"]!).Count==12)&&repeats.Count==24&&graphRepeats.Count==8,"Complete graph/AA/cross-life coverage mandatory.");
            Need(files.Count<=32&&files.Sum(x=>new FileInfo(Path.Combine(output,x.Key)).Length)<32L*1024*1024&&files.All(x=>S.Matches(FileSha(Path.Combine(output,x.Key)),x.Value)),"Bounded complete CAS evidence changed.");
            Need(S.Matches(FileSha(image),CandidateSha)&&S.Matches(FileSha(legacyImage),LegacySha),"Pinned images changed during execution.");
            state["GPU"]="EXECUTED";state["complete"]=true;state["passed"]=true;state["actualGpuKernelExecutions"]=624;state["allRawEvidenceRehashed"]=true;Checkpoint();Complete(output,state);
            Console.WriteLine(JsonSerializer.Serialize(new{complete=true,passed=true,componentRecords=16,graphRecords=2,actualGpuKernelExecutions=624,modelQuality="NOT_RUN",performance="NOT_RUN"}));return 0;
        }
        catch(Exception e){state["passed"]=false;state["error"]=e.ToString();if(owned&&output!=null)Save(Path.Combine(output,"result.json"),state);Console.Error.WriteLine(e);return 1;}
    }
}
