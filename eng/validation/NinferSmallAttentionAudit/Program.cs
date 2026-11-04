// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// CPU-only retained-evidence audit. Never initializes CUDA or runs a kernel.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using R=TensorSharp.Validation.NinferPrefillAttentionReference;

namespace TensorSharp.Validation.SmallAttentionAudit;

internal static class Audit
{
    private sealed class FrozenContext(string directory):AssemblyLoadContext(isCollectible:true)
    {
        protected override Assembly? Load(AssemblyName name)
        {string p=Path.Combine(directory,name.Name+".dll");return File.Exists(p)?LoadFromAssemblyPath(p):null;}
    }
    private static void Need(bool b,string message){if(!b)throw new InvalidDataException(message);}
    private static string Sha(byte[] bytes)=>Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileSha(string p)=>Sha(File.ReadAllBytes(p));
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    private static void Create(string path,object value){using var f=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);JsonSerializer.Serialize(f,value,Json);f.WriteByte(10);f.Flush(true);}
    private static void True(JsonElement e,params string[] fields){foreach(string f in fields)Need(e.GetProperty(f).GetBoolean(),"Failed retained gate: "+f);}
    private static double Median(IEnumerable<double> samples){var a=samples.OrderBy(x=>x).ToArray();Need(a.Length==10,"Exactly ten samples perarm required.");return(a[4]+a[5])/2;}
    private static Dictionary<string,string> VerifyFiles(JsonElement root,string folder)
    {
        var files=root.GetProperty("files").EnumerateObject().ToDictionary(x=>x.Name,x=>x.Value.GetString()!);
        foreach(var p in files){Need(p.Key=="cas/"+p.Value+".bin"&&p.Value.Length==64,"Noncanonical CAS object.");Need(FileSha(Path.Combine(folder,p.Key))==p.Value,"Changed retained CAS bytes.");}
        Need(Directory.GetFiles(Path.Combine(folder,"cas")).Select(Path.GetFileName).ToHashSet().SetEquals(files.Keys.Select(Path.GetFileName)),"Missing/unregistered physical CAS file.");return files;
    }
    private static object Numeric(byte[] bytes,R.Fixture f,bool core)
    {
        const int count=8*24*256;Need(bytes.Length==count*(core?2:4),"Complete raw output span differs.");int nonfinite=0,violations=0,analytic=0;double maxAbsolute=0,maxScaled=0;
        for(int i=0;i<count;i++)
        {
            double actual=core?R.Float(MemoryMarshal.Cast<byte,ushort>(bytes)[i]):MemoryMarshal.Cast<byte,float>(bytes)[i];double expected=core?f.IdealCore[i]:f.IdealGated[i];
            if(!double.IsFinite(actual)){nonfinite++;continue;}double error=Math.Abs(actual-expected),bound=R.Absolute+R.Relative*Math.Abs(expected);maxAbsolute=Math.Max(maxAbsolute,error);maxScaled=Math.Max(maxScaled,error/bound);if(error>bound)violations++;
            int dim=i%256,head=i/256%24,token=i/6144;
            if(f.Case.Pattern=="sentinel"&&dim%17!=0)
            {float exact=R.Constant(head/6,dim)*(core?1:.5f);if(core?MemoryMarshal.Cast<byte,ushort>(bytes)[i]!=R.Bf(exact):BitConverter.SingleToInt32Bits((float)actual)!=BitConverter.SingleToInt32Bits(exact))analytic++;}
            else if(f.Case.Pattern=="sentinel"&&f.Case.Past+token<f.Case.Sentinel&&actual!=0)analytic++;
        }
        Need(nonfinite==0&&violations==0&&analytic==0,"Retained full output fails independent FP64/analytic gate.");return new{values=count,maxAbsolute,maxScaled,nonfinite,violations,analytic};
    }
    private static R.Fixture RotatedGraph(int past,int phase)
    {
        var f=R.Make(new R.Case("graph-t8-p"+past,8,past,"dense",4));
        T[] Rotate<T>(T[] a){var b=new T[a.Length];for(int t=0;t<8;t++)for(int h=0;h<24;h++)Array.Copy(a,(t*24+h/6*6+(h%6+phase)%6)*256,b,(t*24+h)*256,256);return b;}
        return new(f.Case,Rotate(f.Query),Rotate(f.QueryF32),Rotate(f.Gate),f.LogicalK,f.LogicalV,Rotate(f.IdealCore),Rotate(f.IdealGated));
    }
    private static void Nodes(JsonElement nodes,bool candidate,int keys)
    {
        var a=nodes.EnumerateArray().ToArray();Need(a.Length==(candidate?4:2),"Full captured chain missing.");
        var roles=a.Select(x=>x.GetProperty("role").GetString()).ToHashSet();Need(roles.SetEquals(candidate?new[]{"pack","core","reduce","gate"}:new[]{"split","combine"}),"Wrong captured chain roles.");
        foreach(var x in a)
        {
            string role=x.GetProperty("role").GetString()!;var g=x.GetProperty("grid").EnumerateArray().Select(x=>x.GetInt32()).ToArray();var b=x.GetProperty("block").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
            int[] expectedGrid=role switch{"core"=>new[]{1,24,4},"reduce"=>new[]{8,24,1},"split"=>new[]{(keys+127)/128,4,1},_=>new[]{192,1,1}};
            Need(g.SequenceEqual(expectedGrid)&&b.SequenceEqual(new[]{role is "core" or "split"?128:256,1,1})&&x.GetProperty("sharedBytes").GetInt32()==(role=="core"?98304:0),"Retained fullchain geometry differs.");
            if(role=="core")True(x,"dynamicExternalPosition");
        }
    }
    public static int Main(string[] args)
    {
        try
        {
            Need(args.Length is 5 or 7&&args[0]=="--packet"&&args[2]=="--expect-component-sha256"&&args[4].EndsWith(".json",StringComparison.Ordinal)
                &&(args.Length==5||args[5]=="--frozen-probe"),"Usage: --packet ABS_PACKET --expect-component-sha256 SHA ABS_FRESH_OUTPUT.json [--frozen-probe ABS_R23_DLL]");
            string packet=Path.GetFullPath(args[1]),output=Path.GetFullPath(args[4]);Need(!File.Exists(output),"Fresh ignored audit output required.");
            Need(output.Contains(Path.DirectorySeparatorChar+"artifacts"+Path.DirectorySeparatorChar,StringComparison.Ordinal)||output.Contains(Path.DirectorySeparatorChar+"docs"+Path.DirectorySeparatorChar+"validation"+Path.DirectorySeparatorChar,StringComparison.Ordinal),"Generated audit output must be ignored.");
            string componentFolder=Path.Combine(packet,"component"),timingFolder=Path.Combine(packet,"timing"),componentPath=Path.Combine(componentFolder,"completed-result.json"),timingPath=Path.Combine(timingFolder,"completed-result.json");
            byte[] componentBytes=File.ReadAllBytes(componentPath),timingBytes=File.ReadAllBytes(timingPath);Need(Sha(componentBytes)==args[3].ToLowerInvariant(),"Component certificate pin mismatch.");
            using var cDoc=JsonDocument.Parse(componentBytes);using var tDoc=JsonDocument.Parse(timingBytes);var c=cDoc.RootElement;var t=tDoc.RootElement;
            foreach(var pair in new[]{(componentPath,componentBytes),(timingPath,timingBytes)})Need(File.ReadAllText(pair.Item1+".sha256").Split((char[]?)null,StringSplitOptions.RemoveEmptyEntries)[0]==Sha(pair.Item2),"Write-once sidecar mismatch.");
            Need(File.ReadAllBytes(Path.Combine(timingFolder,"component-certificate.json")).AsSpan().SequenceEqual(componentBytes),"Timing qualification bytes differ.");
            True(c,"complete","passed","allRawEvidenceRehashed");True(t,"complete","passed","allFourCasesMeetPredeclaredFullChainWin");
            Need(c.GetProperty("actualGpuKernelExecutions").GetInt32()==624&&t.GetProperty("actualGpuKernelExecutions").GetInt32()==1680,"Complete actual kernel counts differ.");
            var cFiles=VerifyFiles(c,componentFolder);var tFiles=VerifyFiles(t,timingFolder);
            var fixtureMismatches=new List<object>();
            foreach(var record in c.GetProperty("records").EnumerateArray().Where(x=>x.GetProperty("lifetime").GetInt32()==0))
            {
                string name=record.GetProperty("case").GetString()!;int keys=record.GetProperty("keys").GetInt32();var f=R.Make(new R.Case(name,8,keys-8,name.StartsWith("sentinel",StringComparison.Ordinal)?"sentinel":"dense",4));
                byte[] Bytes<T>(T[] a)where T:unmanaged=>MemoryMarshal.AsBytes(a.AsSpan()).ToArray();
                var actual=new Dictionary<string,string>{{"q.bf16",Sha(Bytes(f.Query))},{"q.f32",Sha(Bytes(f.QueryF32))},{"gate.f32",Sha(Bytes(f.Gate))},{"logical-k.bf16",Sha(Bytes(f.LogicalK))},{"logical-v.bf16",Sha(Bytes(f.LogicalV))},{"ideal-core.f64",Sha(Bytes(f.IdealCore))},{"ideal-gated.f32",Sha(Bytes(f.IdealGated))}};
                foreach(var field in record.GetProperty("inputSha256").EnumerateObject())if(actual[field.Name]!=field.Value.GetString())fixtureMismatches.Add(new{@case=name,field=field.Name,expected=field.Value.GetString(),local=actual[field.Name]});
            }
            // Re-execute the frozen producer's complete strict qualification
            // admission. Reflection calls this pure CPU method only; Main,
            // constructors, CUDA registration and device entry points are unused.
            string producerPath=Path.GetFullPath(args.Length==7?args[6]:Path.Combine(AppContext.BaseDirectory,"NinferSmallAttentionProbe.dll"));
            string producerSha256=FileSha(producerPath);bool frozenProducerImageMatched=producerSha256==c.GetProperty("toolAssemblySha256").GetString();
            if(args.Length==7)Need(frozenProducerImageMatched,"Explicit frozen producer DLL differs from actualR23certificate.");
            var loadContext=new FrozenContext(Path.GetDirectoryName(producerPath)!);var producer=loadContext.LoadFromAssemblyPath(producerPath);
            var method=producer.GetType("Program",true)!.GetMethod("ReadQualification",BindingFlags.Static|BindingFlags.NonPublic)!;
            string? strictAdmissionError=null;
            try{_=method.Invoke(null,new object[]{componentBytes,componentFolder});}catch(TargetInvocationException e){strictAdmissionError=(e.InnerException??e).ToString();}
            var rawNumerics=new List<object>();var first=new Dictionary<string,string>();var componentRows=c.GetProperty("records").EnumerateArray().ToArray();
            foreach(var record in componentRows)
            {
                string name=record.GetProperty("case").GetString()!;int keys=record.GetProperty("keys").GetInt32();var f=R.Make(new R.Case(name,8,keys-8,name.StartsWith("sentinel",StringComparison.Ordinal)?"sentinel":"dense",4));
                foreach(var pass in record.GetProperty("passes").EnumerateArray())foreach(var arm in pass.GetProperty("captured").EnumerateObject())
                {
                    string file=arm.Value.GetString()!,identity=name+"/"+arm.Name;if(first.TryGetValue(identity,out string? old))Need(old==file,"Actual raw outputs differ acrosslifetimes/poisons.");else{first.Add(identity,file);rawNumerics.Add(new{kind="component",@case=name,arm=arm.Name,file,fp64=Numeric(File.ReadAllBytes(Path.Combine(componentFolder,file)),f,arm.Name=="core")});}
                }
            }
            int graphReplays=0;foreach(var graph in c.GetProperty("graphs").EnumerateArray())
            {
                Nodes(graph.GetProperty("capturedNodes"),true,65);var tables=new Dictionary<int,string>();
                foreach(var pass in graph.GetProperty("passes").EnumerateArray())
                {
                    graphReplays++;int variant=pass.GetProperty("variant").GetInt32(),past=pass.GetProperty("past").GetInt32();var f=RotatedGraph(past,variant);
                    string table=pass.GetProperty("pointerTableSha256").GetString()!;if(tables.TryGetValue(variant%2,out var old))Need(old==table,"Dynamic table variant changed withinlife.");else tables.Add(variant%2,table);
                    foreach(var arm in pass.GetProperty("captured").EnumerateObject())
                    {
                        string file=arm.Value.GetString()!,identity=f.Case.Name+"/"+arm.Name;if(first.TryGetValue(identity,out string? previous))Need(previous==file,"Graph outputs changed acrosspoisons/lifetimes.");else{first.Add(identity,file);rawNumerics.Add(new{kind="graph",@case=f.Case.Name,arm=arm.Name,file,fp64=Numeric(File.ReadAllBytes(Path.Combine(componentFolder,file)),f,arm.Name=="core")});}
                    }
                }
                Need(tables.Count==2&&tables.Values.Distinct().Count()==2,"Graph physical pointer table did not change.");
            }
            Need(graphReplays==24&&first.Count==32,"Complete actual raw role/crosslife coverage differs.");
            var summaries=new List<object>();int timingIntervals=0,timedReplays=0,warmups=0;foreach(var record in t.GetProperty("timings").EnumerateArray())
            {
                int keys=record.GetProperty("keys").GetInt32();Need(new[]{74,202,512,1024}.Contains(keys),"Unapproved timing geometry.");Nodes(record.GetProperty("candidateCapturedNodes"),true,keys);Nodes(record.GetProperty("legacyCapturedNodes"),false,keys);
                var intervals=record.GetProperty("intervals").EnumerateArray().ToArray();Need(intervals.Length==20,"Complete timing series missing.");var candidate=new List<double>();var legacy=new List<double>();int wins=0;
                for(int round=0;round<5;round++)
                {
                    var pair=new double[4];for(int pos=0;pos<4;pos++)
                    {
                        var x=intervals[round*4+pos];True(x,"passed","fullQualifiedOutputBitsEqual","candidatePackAndCoreBitsExact","allAllocationGuards","immutableInputs");Need(x.GetProperty("round").GetInt32()==round&&x.GetProperty("position").GetInt32()==pos,"ABBA order altered.");
                        bool isCandidate=pos is 1 or 2;Need(x.GetProperty("arm").GetString()==(isCandidate?"candidate4-node":"legacy2-node"),"Fullchain arm order differs.");
                        // The frozen producer divides its CUDA-event float by
                        // six before widening the result to double for JSON.
                        double elapsed=x.GetProperty("intervalMilliseconds").GetDouble(),per=x.GetProperty("millisecondsPerFullChain").GetDouble();Need(double.IsFinite(elapsed)&&elapsed>0&&(double)((float)elapsed/6f)==per&&x.GetProperty("timedGraphReplays").GetInt32()==6&&x.GetProperty("outsideWarmups").GetInt32()==1,"Event float-division/replay counts differ.");
                        string file=x.GetProperty("outputFile").GetString()!,sha=record.GetProperty(isCandidate?"candidateQualifiedSha256":"legacyQualifiedSha256").GetString()!;
                        Need(tFiles[file]==sha&&cFiles.TryGetValue("cas/"+sha+".bin",out var original)&&original==sha&&File.ReadAllBytes(Path.Combine(timingFolder,file)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(componentFolder,"cas/"+sha+".bin"))),"Timed complete bytes differ fromqualifiedownarm.");
                        pair[pos]=per;(isCandidate?candidate:legacy).Add(per);timingIntervals++;timedReplays+=6;warmups++;
                    }
                    if((pair[1]+pair[2])/2<(pair[0]+pair[3])/2)wins++;
                }
                double a=Median(legacy),b=Median(candidate),speedup=a/b;Need(a==record.GetProperty("legacyMedianMilliseconds").GetDouble()&&b==record.GetProperty("candidateMedianMilliseconds").GetDouble()&&speedup==record.GetProperty("medianSpeedup").GetDouble()&&wins==record.GetProperty("pairedRoundWins").GetInt32()&&speedup>1.05&&wins>=4,"Retained median/win computation differs.");
                summaries.Add(new{keys,legacyMilliseconds=a,candidateMilliseconds=b,speedup,pairedRoundWins=wins,fullChainWin=true});
            }
            Need(summaries.Count==4&&timingIntervals==80&&timedReplays==480&&warmups==80,"Complete timing campaign count differs.");
            bool passed=strictAdmissionError==null&&fixtureMismatches.Count==0&&frozenProducerImageMatched;
            var result=new{schema="ninfer-small-attention-offline-actual-audit/1",passed,scope="Retained actual GPU evidence rehashed and checked on CPU; no newGPU launches or model/E2E evidence. A strict source-reconstruction or producer-image mismatch remains a failure even when all raw numerical/timing gates pass.",componentCertificateSha256=Sha(componentBytes),timingCertificateSha256=Sha(timingBytes),producerSha256,frozenProducerImageMatched,strictFrozenQualificationAdmission=strictAdmissionError==null?"PASS":"FAIL",strictAdmissionError,fixtureMismatches,platform=RuntimeInformation.OSDescription,componentCasObjects=cFiles.Count,timingCasObjects=tFiles.Count,actualComponentKernelExecutionsAdmitted=624,actualTimingKernelExecutionsAdmitted=1680,graphReplaysAdmitted=graphReplays,timingIntervals,timedReplays,warmups,fullRawFp64Checks=rawNumerics,summaries,modelQuality="NOT_RUN",endToEndPerformance="NOT_RUN"};
            Create(output,result);Console.WriteLine(JsonSerializer.Serialize(new{passed,strictQualification=passed?"PASS":"FAIL",fixtureMismatches,componentCas=cFiles.Count,timingCas=tFiles.Count,fullRawChecks=rawNumerics.Count,summaries}));return passed?0:1;
        }
        catch(Exception e){Console.Error.WriteLine(e);return 1;}
    }
}
