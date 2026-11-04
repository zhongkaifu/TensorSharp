// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Runtime;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private const long RawCap=512L<<20;
    private const string MmaPolicy="ignis-e600-bf16-t8-original-mma-f32-store-v1";
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.CamelCase};
    private sealed record Pin(string Path,long Bytes,string Sha256);
    private static void Need(bool b,string m){if(!b)throw new InvalidDataException(m);}
    private static string Sha(ReadOnlySpan<byte>b)=>Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();
    private static string FileSha(string p){using var f=File.OpenRead(p);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static Pin FilePin(string p)=>new(Path.GetFullPath(p),new FileInfo(p).Length,FileSha(p));
    private static void Check(Pin p)=>Need(FilePin(p.Path)==p,"Pinned bytes changed: "+p.Path);
    private static void Save(string p,object v,bool once=false)
    {byte[] b=System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(v,Json)+"\n");using(var f=new FileStream(p,once?FileMode.CreateNew:FileMode.Create,FileAccess.Write,FileShare.None)){f.Write(b);f.Flush(true);}if(once)File.WriteAllText(p+".sha256",Sha(b)+"  "+Path.GetFileName(p)+"\n");}
    private static void Fresh(string p)
    {Need(Path.IsPathFullyQualified(p)&&Path.GetFullPath(p)==p&&!Path.Exists(p),"Canonical fresh output required.");var a=p.Split(Path.DirectorySeparatorChar,StringSplitOptions.RemoveEmptyEntries);Need(a.Contains("artifacts")||Enumerable.Range(0,a.Length-1).Any(i=>a[i]=="docs"&&a[i+1]=="validation"),"Ignored evidence output required.");for(string? q=Path.GetDirectoryName(p);q!=null;q=Path.GetDirectoryName(q))Need(new DirectoryInfo(q).LinkTarget==null&&!File.Exists(q),"Aliased output parent.");}
    private static Dictionary<string,string> Options(string[] a)
    {var d=new Dictionary<string,string>();for(int i=0;i<a.Length;i++){string k=a[i];Need(k is "--self-test" or "--prepare" or "--execute" or "--timing" or "--screen-only" or "--artifact" or "--expect-directory-sha" or "--oracle" or "--expect-oracle-sha" or "--helper" or "--expect-helper-sha" or "--candidate" or "--expect-candidate-sha" or "--native-manifest" or "--expect-native-manifest-sha" or "--shape" or "--out","Unknown option.");Need(!d.ContainsKey(k),"Duplicate option.");if(k is "--self-test" or "--prepare" or "--execute" or "--timing" or "--screen-only")d[k]="1";else{Need(++i<a.Length,"Missing option value.");d[k]=a[i];}}Need(new[]{"--self-test","--prepare","--execute"}.Count(d.ContainsKey)<=1,"Exclusive modes.");return d;}
    private static void ValidateScreenOptions(Dictionary<string,string> options)
    {if(options.ContainsKey("--screen-only"))Need(options.ContainsKey("--execute")&&!options.ContainsKey("--timing")&&!options.ContainsKey("--shape"),"Screen requires --execute and forbids --shape/--timing.");}
    private static string Required(Dictionary<string,string>d,string k)=>d.TryGetValue(k,out var x)?x:throw new ArgumentException("Missing "+k);
    private sealed record Raw(string File,int Bytes,string Sha256);
    private static long StoredBytes;
    private static long CaptureLimit=RawCap;
    private static Raw Capture(byte[] bytes,string output,Dictionary<string,string> files)
    {string h=Sha(bytes),relative="cas/"+h+".bin",path=Path.Combine(output,relative);if(files.TryAdd(h,relative)){Need(StoredBytes+bytes.Length<=CaptureLimit,"Raw evidence cap exhausted; incomplete, never PASS.");using var f=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);f.Write(bytes);StoredBytes+=bytes.Length;}else Need(File.ReadAllBytes(path).AsSpan().SequenceEqual(bytes),"CAS bytes changed/collision.");return new(relative,bytes.Length,h);}
    private static bool Repeat(Dictionary<string,string> aa,string key,byte[] bytes){string h=Sha(bytes);if(aa.TryGetValue(key,out var old))return old==h;aa.Add(key,h);return true;}
    private static void Execute(Dictionary<string,string> options)
    {
        string oraclePath=Required(options,"--oracle"),nativePath=Required(options,"--native-manifest"),helper=Required(options,"--helper"),candidate=Required(options,"--candidate"),output=Required(options,"--out");
        var oraclePin=FilePin(oraclePath);Need(S.Matches(oraclePin.Sha256,Required(options,"--expect-oracle-sha")),"Oracle manifest identity differs.");var oracle=JsonSerializer.Deserialize<OracleSet>(File.ReadAllBytes(oraclePath),Json)!;Need(oracle.Schema=="ninfer-bf16-small-t-oracle/1","CPU oracle schema differs.");Check(oracle.Prepared);var prepared=JsonSerializer.Deserialize<Prepared>(File.ReadAllBytes(oracle.Prepared.Path),Json)!;Need(prepared.Schema=="ninfer-bf16-small-t-prepared/1"&&prepared.GeneratorPolicy=="bf16-smallt-fixtures-v1"&&prepared.Cases.Length==15&&oracle.Cases.Length==15&&oracle.Cases.Select(x=>x.Id).ToHashSet().SetEquals(prepared.Cases.Select(x=>x.Id)),"Complete fixed CPU preparation required.");
        var nativePin=FilePin(nativePath);Need(S.Matches(nativePin.Sha256,Required(options,"--expect-native-manifest-sha")),"Actual native metadata differs.");using var doc=JsonDocument.Parse(File.ReadAllBytes(nativePath));var native=doc.RootElement.Clone();bool screen=options.ContainsKey("--screen-only"),mma=native.GetProperty("arithmeticPolicy").GetString()==MmaPolicy;Need(native.GetProperty("complete").GetBoolean()&&(mma?native.GetProperty("schema").GetString()=="tensorsharp-ninfer-bf16-mma-aot/1":native.GetProperty("schema").GetString()=="tensorsharp-ninfer-bf16-small-t-aot/1"&&native.GetProperty("arithmeticPolicy").GetString()=="ignis-e600-bf16-t8-original-simt-f32-store-v1"),"Actual native production schedule/adapter metadata differs.");Need(!screen||mma,"Screen requires the separately pinned MMA module.");ValidateScreenOptions(options);CaptureLimit=screen?32L<<20:RawCap;
        var helperPin=FilePin(helper);var candidatePin=FilePin(candidate);Need(S.Matches(helperPin.Sha256,Required(options,"--expect-helper-sha"))&&S.Matches(candidatePin.Sha256,Required(options,"--expect-candidate-sha"))&&S.Matches(candidatePin.Sha256,native.GetProperty("fatbin").GetProperty("sha256").GetString()!),"Actual selected native file bytes differ.");
        string ptx=Path.Combine(Path.GetDirectoryName(Path.GetFullPath(nativePath))!,native.GetProperty("ptx").GetProperty("path").GetString()!);var ptxPin=FilePin(ptx);Need(S.Matches(ptxPin.Sha256,native.GetProperty("ptx").GetProperty("sha256").GetString()!),"Actual compiled PTX differs.");
        Shape? s=screen?null:Shapes.Single(x=>x.Name==Required(options,"--shape"));var pins=new List<Pin>{oraclePin,oracle.Prepared,nativePin,helperPin,candidatePin,ptxPin,FilePin(typeof(Program).Assembly.Location),FilePin(typeof(CudaNinferKernels).Assembly.Location),FilePin(typeof(NinferArtifactReader).Assembly.Location)};
        foreach(var c in prepared.Cases){pins.AddRange(new[]{c.Input,c.Seed,c.Norm});Check(c.Input);Check(c.Seed);Check(c.Norm);}foreach(var o in oracle.Cases){pins.Add(o.Ideal);pins.Add(o.L1);Check(o.Ideal);Check(o.L1);}
        using(var r=new NinferArtifactReader(prepared.Artifact))Need(r.DirectorySha256==prepared.DirectorySha256&&r.FileBytes==prepared.ArtifactStat.Bytes&&r.FileLastWriteUtcTicks==prepared.ArtifactStat.MtimeTicks,"Original artifact identity/stat changed.");Fresh(output);Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));var files=new Dictionary<string,string>();var results=new List<object>();var libraries=new List<object>();
        var report=new Dictionary<string,object?>{["schema"]="ninfer-bf16-small-t-component/1",["complete"]=false,["passed"]=false,["GPU"]="ATTEMPTED",["shape"]=s,["screenOnly"]=screen,["screenExpectedRealMatrices"]=screen?9:null,["screenExpectedProjectionValues"]=screen?811008:null,["arithmeticPolicy"]=native.GetProperty("arithmeticPolicy").GetString(),["numeric"]=results,["cuBlasRuntime"]=libraries,["inputPins"]=pins,["casFiles"]=files,["rawCap"]=CaptureLimit,["quality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["ordinaryCandidateLaunchOnly"]=true};
        void Persist()=>Save(Path.Combine(output,"result.json"),report);
        try
        {
            foreach(var o in oracle.Cases.Where(x=>prepared.Cases.Any(c=>c.Id==x.Id&&(screen?c.Kind=="real":c.Shape==s!.Name)))){Capture(File.ReadAllBytes(o.Ideal.Path),output,files);Capture(File.ReadAllBytes(o.L1.Path),output,files);}Persist();
            using(var context=CudaContext.Create(0))
            {if(screen)Screen(prepared,oracle,native,helper,helperPin.Sha256,candidate,candidatePin.Sha256,output,files,results,libraries);else{Components(s!,prepared,oracle,native,helper,helperPin.Sha256,candidate,candidatePin.Sha256,output,files,results,libraries);Need(results.Count==6*(s!.RealCount+3),"Numeric campaign incomplete.");}report["numericPassed"]=true;Persist();if(options.ContainsKey("--timing"))report["timing"]=Timings(s!,prepared,oracle,native,helper,helperPin.Sha256,candidate,candidatePin.Sha256,output,files);}
            foreach(var p in pins)Check(p);foreach(var pair in files)Need(FileSha(Path.Combine(output,pair.Value))==pair.Key,"Retained raw CAS changed.");using(var r=new NinferArtifactReader(prepared.Artifact))Need(r.DirectorySha256==prepared.DirectorySha256&&r.FileBytes==prepared.ArtifactStat.Bytes&&r.FileLastWriteUtcTicks==prepared.ArtifactStat.MtimeTicks,"Artifact changed during component.");
            report["rawBytesStored"]=StoredBytes;report["complete"]=true;report["passed"]=true;report["GPU"]="EXECUTED";report["numericCases"]=results.Count;report["limitations"]=new[]{screen?"Nine-real single-pass exact-old MMA admission screen only; no repeated poisoning/two-lifetime qualification, model quality or timing is claimed.":mma?"MMA full component retains two lifetimes/three poisons/all real+synthetic, with mandatory full raw/consumer/canonical exact-old bits; no end-to-end or task quality claim.":"Original SIMT full component scope.","Original SIMT may differ from cuBLAS; this MMA screen additionally requires full raw F32 and full consumers/canonical residual to equal actual old cuBLAS bits. Each full projection has its own fixed CPU FP64 bound; downstream consumer CPU contract checked independently.","Stock BF16 output is only a BF16(candidateF32) bit oracle; it is not a substitute for full unrounded F32 validation.","All real6/3 BF16 weights streamed unchanged, TS-order digest bound to independent NumPy preparation; original weights not retained as raw evidence.","All tool-owned guards and input immutability are checked. Opaque cuBLAS internal workspace/allocation guards are outside tool control.","Actual original Ignis runtime may dispatch Output to another implementation; this isolated SmallT output candidate is experimental, not a claim about its actual fastest path.","Paired full-chain component timing includes cast/consumer/output reset, ordinary CUDA launch only; no PDL or end-to-end speed/quality qualification."};Persist();Save(Path.Combine(output,"completed-result.json"),report,true);
        }
        catch(Exception e){report["error"]=e.ToString();report["rawBytesStored"]=StoredBytes;report["attemptedRecords"]=results.Count;Persist();if(screen){foreach(var pair in files)Need(FileSha(Path.Combine(output,pair.Value))==pair.Key,"Retained failed-screen CAS changed.");Save(Path.Combine(output,"failed-screen.json"),report,true);}throw;}
    }
    public static int Main(string[] args)
    {
        try
        {
            var d=Options(args);ValidateScreenOptions(d);if(d.ContainsKey("--self-test")){Console.WriteLine(JsonSerializer.Serialize(new{CPU="PASS",GPU="NOT_RUN",protocol=SelfTest(),contracts=PrimitiveContracts()},Json));return 0;}
            if(d.ContainsKey("--prepare")){Prepare(Path.GetFullPath(Required(d,"--artifact")),S.Normalize(Required(d,"--expect-directory-sha")),Required(d,"--out"));return 0;}
            if(d.ContainsKey("--execute")){Execute(d);return 0;}
            Console.WriteLine("PLAN: --prepare --artifact ORIGINAL --expect-directory-sha SHA --out FRESH_IGNORED; then prepare-oracle.py before CUDA. --self-test is CPU only. MMA: --execute --screen-only runs all9 real matrices once, requires exact old cuBLAS raw/consumer bits, forbids timing. Fixed T8/input14336x5120/output5120x6144. No GPU without --execute.");return 0;
        }
        catch(Exception e){Console.Error.WriteLine(e);return 1;}
    }
}
