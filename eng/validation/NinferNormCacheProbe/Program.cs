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
    private static string Sha(ReadOnlySpan<byte> bytes)=>Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileSha(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    private static void NewFile(string path,byte[] bytes){using var f=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);f.Write(bytes);f.Flush(true);}
    private static void Save(string path,object value)=>File.WriteAllBytes(path,JsonSerializer.SerializeToUtf8Bytes(value,Json));
    private static string Cas(byte[] bytes,string output,Dictionary<string,string> files)
    {
        string hash=Sha(bytes),relative="cas/"+hash+".bin",path=Path.Combine(output,relative);
        if(File.Exists(path))Need(File.ReadAllBytes(path).AsSpan().SequenceEqual(bytes),"CAS bytes differ.");else NewFile(path,bytes);
        files[relative]=hash;return relative;
    }
    private static string Pinned(string path,string expected)
    {Need(Path.IsPathFullyQualified(path)&&Path.GetFullPath(path)==path&&File.Exists(path),"Absolute normalized existing module path required.");Need(S.Matches(FileSha(path),expected),"Supplied module bytes differ from explicit SHA.");return path;}
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path)&&Path.GetFullPath(path)==path&&!File.Exists(path)&&!Directory.Exists(path),"Fresh normalized absolute output folder required.");
        string[] parts=path.Split(Path.DirectorySeparatorChar);Need(parts.Contains("artifacts")||Enumerable.Range(0,parts.Length-1).Any(i=>parts[i]=="docs"&&parts[i+1]=="validation"),"Output must be ignored validation/artifact evidence.");
    }
    private static void Complete(string output,Dictionary<string,object?> state,Dictionary<string,string> files)
    {
        foreach(var x in files)Need(x.Key=="cas/"+x.Value+".bin"&&S.Matches(FileSha(Path.Combine(output,x.Key)),x.Value),"Raw evidence rehash failed.");
        state["allRawEvidenceRehashed"]=true;state["complete"]=true;state["passed"]=true;
        byte[] bytes=JsonSerializer.SerializeToUtf8Bytes(state,Json);NewFile(Path.Combine(output,"completed-result.json"),bytes);NewFile(Path.Combine(output,"completed-result.json.sha256"),System.Text.Encoding.ASCII.GetBytes(Sha(bytes)+"  completed-result.json\n"));
        Save(Path.Combine(output,"result.json"),state);
    }
    private static void RunCase(Fixture f,string mode,int life,Norm old,Norm candidate,Stream stream,string output,
        List<Dictionary<string,object?>> records,Dictionary<string,string> files,Dictionary<string,string> aa,Action checkpoint)
    {
        using var memory=new Memory();var delta=memory.Put("delta",Bytes(f.Delta));var weight=memory.Put("effective-f32-weight",Bytes(f.Weight));
        var left=new Arm(memory,"legacy",mode,delta);var right=new Arm(memory,"candidate",mode,delta);var oracle=Reference(f,mode);byte[] residual=Bytes(oracle.Residual);var passes=new List<object>();
        var record=new Dictionary<string,object?>{["id"]=$"{f.Name}/{mode}/life{life}",["fixture"]=f.Name,["aliasMode"]=mode,["lifetime"]=life,["rows"]=Rows,["width"]=Width,["epsilon"]=Epsilon,
            ["inputSha256"]=new{x=Sha(Bytes(f.X)),delta=Sha(Bytes(f.Delta)),effectiveWeight=Sha(Bytes(f.Weight)),residual=Sha(residual)},["nondefaultStream"]=true,["passes"]=passes,["passed"]=false};records.Add(record);checkpoint();
        for(int pass=0;pass<3;pass++)
        {
            left.Reset(f,mode,Poisons[pass]);right.Reset(f,mode,Poisons[pass]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            old.Run(left.X.Pointer,left.Delta,weight.Pointer,left.Y.Pointer,stream.Pointer);candidate.Run(right.X.Pointer,right.Delta,weight.Pointer,right.Y.Pointer,stream.Pointer);stream.Sync();
            byte[] lx=Memory.Body(left.X),ly=Memory.Body(left.Y),rx=Memory.Body(right.X),ry=Memory.Body(right.Y);
            bool exactX=lx.AsSpan().SequenceEqual(rx),exactY=ly.AsSpan().SequenceEqual(ry),referenceX=lx.AsSpan().SequenceEqual(residual)&&rx.AsSpan().SequenceEqual(residual);
            bool guards=memory.Guards(),inputs=memory.Inputs(),repeated=true;var captures=new Dictionary<string,string>();
            foreach(var pair in new[]{("residual",rx),("norm",ry)})
            {
                string key=f.Name+"/"+mode+"/"+pair.Item1,hash=Sha(pair.Item2);if(aa.TryGetValue(key,out var previous)&&previous!=hash)repeated=false;else aa.TryAdd(key,hash);
                captures[pair.Item1]=Cas(pair.Item2,output,files);
            }
            var numericOld=Check(ly,oracle);var numericCandidate=Check(ry,oracle);bool ok=exactX&&exactY&&referenceX&&guards&&inputs&&repeated&&numericOld.Passed&&numericCandidate.Passed;
            passes.Add(new{pass,poison=Poisons[pass],outputSeedIsDelta=mode=="delta-output",legacyResidualFile=Cas(lx,output,files),legacyNormFile=Cas(ly,output,files),candidateFiles=captures,
                fullResidualBitsEqual=exactX,fullNormBitsEqual=exactY,independentResidualBitsEqual=referenceX,allocationGuards=guards,immutableInputs=inputs,aaAndCrossLifetime=repeated,legacyFp64=numericOld,candidateFp64=numericCandidate,passed=ok});checkpoint();
            Need(ok,"Norm component differs; full candidate/control outputs retained.");
        }
        record["actualNormLaunches"]=6;record["passed"]=true;checkpoint();
    }
    private static int Main(string[] args)
    {
        if(args.SequenceEqual(new[]{"--self-test"})){SelfTest();return 0;}
        string? output=null;bool owned=false;var records=new List<Dictionary<string,object?>>();var files=new Dictionary<string,string>();var aa=new Dictionary<string,string>();
        var state=new Dictionary<string,object?>{["schema"]="ninfer-norm-cache-component-and-graph-timing/1",["complete"]=false,["passed"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["records"]=records,["files"]=files};
        try
        {
            bool timing=args.Length==11&&args[^1]=="--timing";
            Need((args.Length==10||timing)&&args[0]=="--legacy-module"&&args[2]=="--expect-legacy-sha256"&&args[4]=="--candidate-module"&&args[6]=="--expect-candidate-sha256"&&args[8]=="--out","Usage: --legacy-module PATH --expect-legacy-sha256 SHA --candidate-module PATH --expect-candidate-sha256 SHA --out FRESH_ABSOLUTE_DIR [--timing]");
            string oldPath=Pinned(args[1],S.Normalize(args[3])),candidatePath=Pinned(args[5],S.Normalize(args[7]));output=args[9];Fresh(output);Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;
            state["legacyModulePath"]=oldPath;state["legacyModuleSha256"]=S.Normalize(args[3]);state["candidateModulePath"]=candidatePath;state["candidateModuleSha256"]=S.Normalize(args[7]);
            state["toolAssemblySha256"]=FileSha(typeof(Program).Assembly.Location);state["backendAssemblySha256"]=FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"]="Synthetic exact T8/K5120 legacy arithmetic comparison only. Four supported delta aliases, two module lifetimes, A/B/A output poisons, nondefault stream. No model/quality/E2E claim.";
            state["fp64Envelope"]="BF16 residual from independent F32 addition/RNE; FP64 RMS reference. rho=8*gamma(K+64,F32u=2^-24)+2^-18; absolute bound=abs(ideal)*(rho+(1+rho)/256)+2^-133. Fixed beforeGPU, finite normal scoped fixtures; exact legacy GPU bits remain mandatory.";
            void Checkpoint()=>Save(Path.Combine(output,"result.json"),state);Checkpoint();CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
            using(var context=CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int major,75,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int minor,76,device).ThrowOnError();CudaDriverApi.cuDeviceGetAttribute(out int sms,16,device).ThrowOnError();
                state["device"]=new{ordinal=0,driver,major,minor,sms};using var stream=new Stream();var fixtures=Fixtures();
                for(int life=0;life<2;life++)
                {
                    using var old=new Norm(oldPath,args[3],false);using var candidate=new Norm(candidatePath,args[7],true);Need(old.ImageSha256!=candidate.ImageSha256,"Candidate and control must be distinct module images.");
                    state["GPU"]="EXECUTED";
                    foreach(var f in fixtures)foreach(string mode in Modes)RunCase(f,mode,life,old,candidate,stream,output,records,files,aa,Checkpoint);
                    Need(old.Submissions==84&&candidate.Submissions==84,"Each module lifetime requires84 norm submissions per arm.");
                    if(life==1&&timing)RunTiming(fixtures.Single(x=>x.Name=="nonuniform"),old,candidate,stream,output,files,state,Checkpoint);
                }
                Need(records.Count==56&&records.Select(x=>(string)x["id"]!).Distinct().Count()==56&&records.All(x=>x["passed"] is true),"All56 unique lifecycle records are mandatory.");
                stream.Sync();
            }
            state["actualComponentNormLaunches"]=336;state["actualComponentFullF32OutputValuesCompared"]=336L*Count;state["componentPassed"]=true;
            Complete(output,state,files);Console.WriteLine(JsonSerializer.Serialize(new{passed=true,output,componentRecords=56,normLaunches=336,timingRequested=timing}));return 0;
        }
        catch(Exception error)
        {
            state["error"]=error.ToString();state["complete"]=false;state["passed"]=false;if(owned&&output!=null)Save(Path.Combine(output,"result.json"),state);Console.Error.WriteLine(error);return 1;
        }
    }
}
