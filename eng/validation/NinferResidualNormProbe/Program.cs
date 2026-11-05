// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using S=TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private const long RawBudget=512L<<20;
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.CamelCase};
    private static void Need(bool value,string message){if(!value)throw new InvalidDataException(message);}
    private static string Sha(ReadOnlySpan<byte> data)=>Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    private static string FileSha(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static void Save(string path,object state,bool once=false)
    {byte[] b=System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(state,Json)+"\n");using(var f=new FileStream(path,once?FileMode.CreateNew:FileMode.Create,FileAccess.Write,FileShare.None)){f.Write(b);f.Flush(true);}if(once){using var f=new FileStream(path+".sha256",FileMode.CreateNew,FileAccess.Write,FileShare.None);f.Write(System.Text.Encoding.ASCII.GetBytes(Sha(b)+"  "+Path.GetFileName(path)+"\n"));f.Flush(true);}}
    private sealed record Pin(string Path,long Bytes,string Sha256);
    private static Pin PinFile(string path){Need(Path.IsPathFullyQualified(path)&&Path.GetFullPath(path)==path&&File.Exists(path),"Canonical existing input required.");return new(path,new FileInfo(path).Length,FileSha(path));}
    private static void VerifyPins(IEnumerable<Pin> pins){foreach(var p in pins)Need(PinFile(p.Path)==p,"Pinned source/runtime/native changed.");}
    private static long StoredBytes;
    private static string Cas(byte[] bytes,string output,Dictionary<string,string> files)
    {string h=Sha(bytes),p=Path.Combine(output,"cas",h+".bin");if(files.TryAdd(h,"cas/"+h+".bin")){Need(StoredBytes+bytes.Length<=RawBudget,"Raw evidence limit reached; incomplete, never PASS.");using var f=new FileStream(p,FileMode.CreateNew,FileAccess.Write,FileShare.None);f.Write(bytes);StoredBytes+=bytes.Length;}else Need(File.ReadAllBytes(p).AsSpan().SequenceEqual(bytes),"CAS retained bytes differ.");return files[h];}
    private static bool Repeat(Dictionary<string,string> aa,string key,byte[] bytes){string h=Sha(bytes);if(aa.TryGetValue(key,out string? prior))return prior==h;aa.Add(key,h);return true;}
    private static void Fresh(string p)
    {Need(Path.IsPathFullyQualified(p)&&Path.GetFullPath(p)==p&&!Path.Exists(p)&&new FileInfo(p).LinkTarget==null,"Fresh canonical output required.");Need(p.Split(Path.DirectorySeparatorChar).Contains("artifacts")||p.Contains("/docs/validation/",StringComparison.Ordinal),"Ignored output required.");for(string? a=Path.GetDirectoryName(p);a!=null;a=Path.GetDirectoryName(a))Need(new DirectoryInfo(a).LinkTarget==null&&!File.Exists(a),"Aliased output parent refused.");}
    private static void Component(Modules m,Stream stream,Fixture f,int life,string output,Dictionary<string,string> files,Dictionary<string,string> aa,List<object> records,Action checkpoint)
    {
        using var memory=new Memory();var input=memory.Put("input",Bytes(f.Input));var wc=memory.Put("weight-codes",f.WeightCodes);var ws=memory.Put("weight-scales",f.WeightScales);float[] seed=Seed(f),weights=NormWeights();var w=memory.Put("norm-weights",Bytes(weights));var arms=ArmNames.Select(n=>new Arm(memory,n)).ToArray();var passes=new List<object>();
        var record=new Dictionary<string,object?>{["life"]=life,["fixture"]=f.Name,["shape"]=Geometry,["projectionFp64Scope"]="All output values, including independently known zero rows.",["inputSha256"]=Sha(Bytes(f.Input)),["weightCodesSha256"]=Sha(f.WeightCodes),["weightScalesSha256"]=Sha(f.WeightScales),["seedSha256"]=Sha(Bytes(seed)),["effectiveWeightSha256"]=Sha(Bytes(weights)),["inputD"]=f.InputD,["weightD"]=f.WeightD,["alpha"]=f.Alpha,["passes"]=passes,["passed"]=false};records.Add(record);checkpoint();
        try{for(int pass=0;pass<3;pass++)
        {
            foreach(var a in arms)a.Reset(seed,Poisons[pass]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            for(int i=0;i<3;i++){m.Quantize(f,input,arms[i],stream.Pointer);m.Project(f,wc,ws,arms[i],i,stream.Pointer);}stream.Sync();
            byte[][] block=arms.Select(a=>Memory.Body(a.Block)).ToArray(),partial=arms.Select(a=>Memory.Body(a.Partial)).ToArray(),beforeX=arms.Select(a=>Memory.Body(a.X)).ToArray();
            var oracle=NormReference(seed,block[0],weights);bool projection=block.Skip(1).All(x=>x.AsSpan().SequenceEqual(block[0])),parts=partial.Skip(1).All(x=>x.AsSpan().SequenceEqual(partial[0])),fused=beforeX[2].AsSpan().SequenceEqual(Bytes(oracle.Residual))&&beforeX.Take(2).All(x=>x.AsSpan().SequenceEqual(Bytes(seed)));
            for(int i=0;i<3;i++)m.Norm(arms[i],w,i,stream.Pointer);stream.Sync();byte[][] residual=arms.Select(a=>Memory.Body(a.X)).ToArray(),norm=arms.Select(a=>Memory.Body(a.Y)).ToArray();
            bool residualBits=residual.All(x=>x.AsSpan().SequenceEqual(Bytes(oracle.Residual))),normBits=norm.Skip(1).All(x=>x.AsSpan().SequenceEqual(norm[0])),readonlyX=residual[2].AsSpan().SequenceEqual(beforeX[2]),blockImmutable=arms.Select((a,i)=>Memory.Body(a.Block).AsSpan().SequenceEqual(block[i])).All(v=>v);
            var numeric=block.Select(x=>Check(x,f,false)).ToArray();var normNumeric=norm.Select(x=>CheckNorm(x,oracle)).ToArray();bool quant=arms.All(a=>a.QuantBits(f)),counter=arms.All(a=>a.CountersZero()),guards=memory.Guards(),inputs=memory.Inputs();var captures=new List<object>();
            bool repeated=true;for(int i=0;i<3;i++)
            {captures.Add(new{arm=ArmNames[i],block=Cas(block[i],output,files),partial=Cas(partial[i],output,files),beforeNormResidual=Cas(beforeX[i],output,files),residual=Cas(residual[i],output,files),norm=Cas(norm[i],output,files)});foreach(var item in new[]{("block",block[i]),("partial",partial[i]),("canonical-x",residual[i]),("norm-y",norm[i])})repeated&=Repeat(aa,f.Name+"/"+i+"/"+item.Item1,item.Item2);}
            bool ok=projection&&parts&&fused&&residualBits&&normBits&&readonlyX&&blockImmutable&&quant&&counter&&guards&&inputs&&repeated&&numeric.All(x=>x.Passed)&&normNumeric.All(x=>x.Passed);
            passes.Add(new{pass,poison=Poisons[pass],fullBlockBits=projection,fullPartialBits=parts,fusedBoundaryCpuBits=fused,canonicalResidualCpuBits=residualBits,fullNormBits=normBits,plainNormDoesNotWriteX=readonlyX,blockUnchangedByNorm=blockImmutable,cpuQuantBits=quant,countersReturnedZero=counter,allocationGuards=guards,immutableInputs=inputs,aaAndCrossLifetime=repeated,projectionFp64=numeric,normFp64=normNumeric,captures,passed=ok});checkpoint();Need(ok,"Residual norm component failed; all complete outputs retained.");
        }}finally{stream.Sync();}
        record["passed"]=true;checkpoint();
    }
    private static void ZeroSemantics(Modules m,Stream stream,int life,string output,Dictionary<string,string> files,Dictionary<string,string> aa,List<object> records,Action checkpoint)
    {
        using var memory=new Memory();float z=BitConverter.UInt32BitsToSingle(0x80000000u);float[] negative=Enumerable.Repeat(z,OutputCount).ToArray(),positive=new float[OutputCount],weight=NormWeights();var d=memory.Put("negative-zero-delta",Bytes(negative));var w=memory.Put("effective-weight",Bytes(weight));var xs=Enumerable.Range(0,3).Select(i=>memory.Empty("zero-x"+i,OutputCount*4)).ToArray();var ys=Enumerable.Range(0,3).Select(i=>memory.Empty("zero-y"+i,OutputCount*4)).ToArray();var passes=new List<object>();
        var record=new Dictionary<string,object?>{["life"]=life,["fixture"]="negative-zero-initial-null-vs-canonical",["passes"]=passes,["passed"]=false,["scope"]="Native primitive distinction only; no production first-layer routing claim."};records.Add(record);checkpoint();
        try{for(int pass=0;pass<3;pass++)
        {
            for(int i=0;i<3;i++){Memory.Write(xs[i].Allocation,Guarded(Bytes(negative)));Memory.Poison(ys[i],Poisons[pass]);}CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            m.AddRun(xs[0].Pointer,d.Pointer,w.Pointer,ys[0].Pointer,stream.Pointer);m.PlainRun(xs[1].Pointer,w.Pointer,ys[1].Pointer,stream.Pointer);m.AddRun(xs[2].Pointer,IntPtr.Zero,w.Pointer,ys[2].Pointer,stream.Pointer);stream.Sync();
            byte[][] x=xs.Select(Memory.Body).ToArray(),y=ys.Select(Memory.Body).ToArray();bool ok=x.Take(2).All(v=>v.AsSpan().SequenceEqual(Bytes(negative)))&&y[0].AsSpan().SequenceEqual(y[1])&&x[2].AsSpan().SequenceEqual(Bytes(positive))&&y[2].AsSpan().SequenceEqual(Bytes(positive))&&CheckNorm(y[1],CanonicalNormReference(negative,weight)).Passed&&memory.Guards()&&memory.Inputs();
            bool repeated=true;var captured=new List<object>();for(int i=0;i<3;i++){captured.Add(new{arm=i==0?"old-add-negative-zero":i==1?"plain-canonical-negative-zero":"initial-old-null-plus-zero",residual=Cas(x[i],output,files),norm=Cas(y[i],output,files)});repeated&=Repeat(aa,"zero/"+i+"/x",x[i])&&Repeat(aa,"zero/"+i+"/y",y[i]);}ok&=repeated;passes.Add(new{pass,captured,aaAndCrossLifetime=repeated,passed=ok});checkpoint();Need(ok,"Initial-null/canonical signed-zero semantic boundary failed.");
        }}finally{stream.Sync();}
        record["passed"]=true;checkpoint();
    }
    private static int Main(string[] args)
    {
        string? output=null;bool owned=false;var records=new List<object>();var semantics=new List<object>();var files=new Dictionary<string,string>();var aa=new Dictionary<string,string>();var state=new Dictionary<string,object?>{["schema"]="ninfer-residual-norm-component/1",["complete"]=false,["passed"]=false,["GPU"]="NOT_RUN",["modelQuality"]="NOT_RUN",["endToEndPerformance"]="NOT_RUN",["records"]=records,["semanticCases"]=semantics,["files"]=files};
        void Checkpoint(){if(owned)Save(Path.Combine(output!,"result.json"),state);}
        try
        {
            var opt=new Dictionary<string,string>();for(int i=0;i<args.Length;i++){string k=args[i];Need(k is "--shape" or "--execute" or "--self-test" or "--timing" or "--helper-module" or "--expect-helper-sha256" or "--legacy-module" or "--expect-legacy-sha256" or "--candidate-module" or "--expect-candidate-sha256" or "--candidate-ptx" or "--expect-candidate-ptx-sha256" or "--out","Unknown option.");Need(!opt.ContainsKey(k),"Duplicate option.");if(k is "--execute" or "--self-test" or "--timing")opt.Add(k,"1");else{Need(++i<args.Length,"Missing option value.");opt.Add(k,args[i]);}}
            string Required(string k)=>opt.TryGetValue(k,out string? v)?v:throw new ArgumentException("Missing "+k);string role=opt.GetValueOrDefault("--shape","output");Geometry=Shapes.SingleOrDefault(s=>s.Name==role&&s.Name is "output" or "down")??throw new ArgumentException("Only output/down are admitted.");Need(!(opt.ContainsKey("--execute")&&opt.ContainsKey("--self-test")),"Exclusive modes required.");
            if(opt.ContainsKey("--self-test")){Need(opt.Keys.All(k=>k is "--self-test" or "--shape"),"Self-test admits no runtime options.");Need(Marshal.SizeOf<GraphApi.KernelParams>()==56&&Marshal.SizeOf<CopyGraphApi.Copy3D>()==200&&Marshal.OffsetOf<CopyGraphApi.Copy3D>("SD").ToInt32()==48&&Marshal.OffsetOf<CopyGraphApi.Copy3D>("DD").ToInt32()==136,"Public x64 graph ABI differs.");ResidualSelfTest();return 0;}
            if(!opt.ContainsKey("--execute")){Need(!opt.ContainsKey("--timing"),"Timing requires explicit execution.");Console.WriteLine("PLAN: --execute --shape output|down --helper-module FILE --expect-helper-sha256 SHA --legacy-module FILE --expect-legacy-sha256 SHA --candidate-module FILE --expect-candidate-sha256 SHA --candidate-ptx FILE --expect-candidate-ptx-sha256 SHA --out FRESH_ABSOLUTE_IGNORED [--timing]. Two module lifetimes, five mandatory fixtures, three poisons, three arms, full residual/norm bits; optional reset/quant/GEMM/norm timing1 and6 independent matrices. --self-test performs no CUDA work.");return 0;}
            var native=new[]{("--helper-module","--expect-helper-sha256"),("--legacy-module","--expect-legacy-sha256"),("--candidate-module","--expect-candidate-sha256"),("--candidate-ptx","--expect-candidate-ptx-sha256")}.Select(pair=>{var p=PinFile(Required(pair.Item1));Need(S.Matches(p.Sha256,Required(pair.Item2)),"Input byte pin differs.");return p;}).ToArray();PtxAbi(native[3].Path);output=Required("--out");Fresh(output);Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"cas"));owned=true;
            var assemblies=new[]{typeof(Program).Assembly,typeof(CudaBackend).Assembly}.Select(a=>PinFile(a.Location)).ToArray();state["nativeFiles"]=native;state["assemblies"]=assemblies;state["geometry"]=Geometry;state["rawByteBudget"]=RawBudget;state["scope"]="Only two fixed T8/B1 projection/norm components, no engine wiring/initial routing/quality/E2E claims. Same-source8 is a numerical control; timing uses actual old8 versus fused8 only.";state["normBound"]="Independent exact represented BF16 residual; positive FP64 square sum; rho=8gamma(K+64,u=2^-24)+2^-18; output BF16 RNE <=2^-8 relative plus2^-133. Projection bound is the unchanged linked decoded FP64 oracle. Dense timing FP64 projection coverage is only explicit active witnesses.";Checkpoint();
            CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();state["GPU"]="ATTEMPTED";
            using(var context=CudaContext.Create(0))using(var stream=new Stream())
            {
                for(int life=0;life<2;life++)using(var modules=new Modules(native[0].Path,native[0].Sha256,native[1].Path,native[1].Sha256,native[2].Path,native[2].Sha256))
                {
                    foreach(string name in FixtureNames)Component(modules,stream,ResidualFixture(name),life,output,files,aa,records,Checkpoint);ZeroSemantics(modules,stream,life,output,files,aa,semantics,Checkpoint);Need(modules.HostSubmissions==144,"Incomplete fixed direct-kernel campaign.");
                    if(life==1&&opt.ContainsKey("--timing")){Need(records.Count==10&&semantics.Count==2,"Timing requires the complete numeric campaign first.");RunTiming(modules,stream,output,files,state,Checkpoint);}
                    stream.Sync();
                }
            }
            Need(records.Count==10&&semantics.Count==2,"Incomplete mandatory campaign.");VerifyPins(native.Concat(assemblies));foreach(var f in files)Need(FileSha(Path.Combine(output,f.Value))==f.Key,"Raw CAS changed before finalization.");state["GPU"]="EXECUTED";state["componentKernelExecutionsSubmitted"]=288;state["actualShape"]=Geometry.Name;state["storedRawBytes"]=StoredBytes;state["passed"]=true;state["complete"]=true;Checkpoint();Save(Path.Combine(output,"completed-result.json"),state,true);Console.WriteLine(JsonSerializer.Serialize(new{complete=true,passed=true,shape=Geometry.Name,componentKernelExecutionsSubmitted=288,rawBytes=StoredBytes,performanceScope="component-only; inspect predeclaredWin separately",modelQuality="NOT_RUN"}));return 0;
        }
        catch(Exception e){state["error"]=e.ToString();state["passed"]=false;Checkpoint();Console.Error.WriteLine(e);return 1;}
    }
}
