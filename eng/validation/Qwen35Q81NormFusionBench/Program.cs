using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;

return Bench.Run(args);

internal static class Bench
{
    private const string ModelSha="d5659ecb29adaf75f313fdae03eb7ff17baf3f986f0c3b1348e8afda1670c9a8";
    internal const string Q35KernelFile="tensorsharp_q35_kernels.ptx";
    internal static MethodInfo Q35LocatorMethod()=>typeof(Q35CudaEngine).Assembly.GetType("TensorSharp.Cuda.CudaKernels",true)!.GetMethod("LocatePtxPath",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic,null,new[]{typeof(string)},null)??throw new InvalidOperationException("Q35 string-filename locator missing.");
    private const string Bbd="bbdaca53a0fb56ec78644356edfa981f6552e57c011f6b8cbfe015ac7b608515";
    internal static void Require(bool ok,string why){if(!ok)throw new InvalidOperationException(why);}
    internal static string Hash(byte[] data)=>Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    internal static string FileHash(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    private static void Write(string path,object value)=>File.WriteAllText(path,JsonSerializer.Serialize(value,new JsonSerializerOptions{WriteIndented=true})+"\n");
    private static byte[] Bytes(float[] values)=>MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static byte[] Active(byte[] all,int first,int rows,int stride)=>all.AsSpan(Native.Buffer.Guard+first*stride,rows*stride).ToArray();
    private sealed record Plane(string Role,int Layer,IntPtr Codes,IntPtr Scales,int Out,int K);
    private sealed record Snapshot(Dictionary<string,byte[]> Values,Dictionary<string,byte[]> Raw,bool Guards);
    private static object Compare(Snapshot reference,Snapshot actual)
    {
        var differences=new Dictionary<string,object>();bool pass=reference.Guards&&actual.Guards;
        foreach(var item in reference.Values){var b=actual.Values[item.Key];Require(b.Length==item.Value.Length,"Comparison size differs.");int count=0,first=-1;for(int i=0;i<b.Length;i++)if(b[i]!=item.Value[i]){count++;if(first<0)first=i;}pass&=count==0;differences[item.Key]=new{differing_bytes=count,first_byte=first,reference_sha256=Hash(item.Value),actual_sha256=Hash(b)};}
        return new{strict_pass=pass,reference_guards=reference.Guards,actual_guards=actual.Guards,differences,no_tolerance=true};
    }
    private static bool Passed(object value)=>JsonSerializer.SerializeToElement(value).GetProperty("strict_pass").GetBoolean();
    private static void SaveRaw(string output,string id,Snapshot value)
    {string dir=Path.Combine(output,id);Directory.CreateDirectory(dir);foreach(var item in value.Raw)File.WriteAllBytes(Path.Combine(dir,item.Key+".bin"),item.Value);}
    private static float[] Values(int count,int k,int pattern)
    {
        var result=new float[count];for(int i=0;i<count;i++){uint u=unchecked((uint)(i+1)*2654435761u);float v=((int)((u>>9)&2047)-1024)/1024f*(1<<((i/k)%3));
            if(pattern==2&&i%11==0)v=BitConverter.UInt32BitsToSingle((i%4)switch{0=>0u,1=>0x80000000u,2=>1u,_=>0x80000001u});
            if(pattern==3&&i%97==0)v=BitConverter.UInt32BitsToSingle((i%3)switch{0=>0x7fc00001u,1=>0x7f800000u,_=>0xff800000u});result[i]=v;}return result;
    }
    private sealed class Fixture : IDisposable
    {
        internal readonly Plane W;internal readonly int N,Pattern,First=3,Total;internal readonly float Eps,Alpha;
        private readonly Native api;private readonly byte[] input,delta,weights;
        private readonly Native.Buffer d=null!,w=null!;
        internal readonly bool HasDelta;
        internal Fixture(Native api,Plane plane,int n,int pattern,float eps,float[] actualNorm)
        {
            this.api=api;W=plane;N=n;Pattern=pattern;Total=n+6;Eps=eps;HasDelta=pattern is 1 or 2;Alpha=pattern switch{1=>.5f,2=>-1.25f,3=>BitConverter.UInt32BitsToSingle(0x3f800001),_=>1f};
            input=Bytes(Values(Total*W.K,W.K,pattern));delta=Bytes(Values(Total*W.K,W.K,0).Select(x=>x*.125f).ToArray());
            var norm=W.K==actualNorm.Length?(float[])actualNorm.Clone():Enumerable.Range(0,W.K).Select(i=>.75f+(i%31)/64f).ToArray();
            if(pattern==1)for(int i=0;i<norm.Length;i++)norm[i]*=1.25f;
            weights=Bytes(norm);try{d=new Native.Buffer(delta.Length,0x25,api.Engine.Synchronize);w=new Native.Buffer(weights.Length,0x36,api.Engine.Synchronize);d.Load(delta);w.Load(weights);}catch{Dispose();throw;}
        }
        internal object Identity()=>new{role=W.Role,layer=W.Layer,n=N,pattern=Pattern,k=W.K,out_dim=W.Out,first=First,total=Total,has_delta=HasDelta,alpha_bits=BitConverter.SingleToUInt32Bits(Alpha),input_sha256=Hash(input),delta_sha256=Hash(delta),norm_sha256=Hash(weights),norm_weight_scope=W.K==api.Engine.HiddenSize?"actual loaded first-GDN AttnNorm (pattern1 scales it1.25 for stress)":"synthetic nonunit norm of this width; not a model norm routing claim"};
        internal Work Create(byte poison)=>new(this,poison);
        internal sealed class Work : IDisposable
        {
            private readonly Fixture f;private readonly Native.Buffer x=null!,y=null!,q=null!,s=null!,outbuf=null!;
            internal Work(Fixture fixture,byte poison)
            {
                f=fixture;try{x=new(f.input.Length,poison,f.api.Engine.Synchronize);y=new(f.input.Length,poison,f.api.Engine.Synchronize);
                q=new(f.Total*f.W.K,poison,f.api.Engine.Synchronize);s=new(f.Total*(f.W.K/32)*4,poison,f.api.Engine.Synchronize);outbuf=new(f.Total*f.W.Out*4,poison,f.api.Engine.Synchronize);x.Load(f.input);
                }catch{Dispose();throw;}
            }
            internal void Issue(bool fused)
            {
                IntPtr xp=x.Data+f.First*f.W.K*4,dp=f.HasDelta?f.d.Data+f.First*f.W.K*4:IntPtr.Zero,yp=y.Data+f.First*f.W.K*4,qp=q.Data+f.First*f.W.K,sp=s.Data+f.First*(f.W.K/32)*4;
                if(fused)f.api.Fused(xp,dp,f.w.Data,yp,f.N,f.W.K,f.Eps,qp,sp);
                else{f.api.OriginalNorm(xp,dp,f.w.Data,yp,f.N,f.W.K,f.Eps,f.api.Stream);f.api.OriginalQuant(yp,qp,sp,f.W.K,f.N,f.api.Stream);}
            }
            internal Snapshot Read()
            {
                f.api.OriginalGemv(f.W.Codes,f.W.Scales,f.W.Out,f.W.K,q.Data+f.First*f.W.K,s.Data+f.First*(f.W.K/32)*4,f.Alpha,outbuf.Data+f.First*f.W.Out*4,f.N,f.api.Stream);f.api.Engine.Synchronize();
                var xr=x.Read();var yr=y.Read();var qr=q.Read();var sr=s.Read();var output=outbuf.Read();var dr=f.d.Read();var wr=f.w.Read();
                bool guards=x.Guards(xr,0,x.Bytes)&&y.Guards(yr,f.First*f.W.K*4,f.N*f.W.K*4)&&q.Guards(qr,f.First*f.W.K,f.N*f.W.K)&&s.Guards(sr,f.First*(f.W.K/32)*4,f.N*(f.W.K/32)*4)&&outbuf.Guards(output,f.First*f.W.Out*4,f.N*f.W.Out*4)
                    &&f.d.Guards(dr,0,f.d.Bytes)&&f.w.Guards(wr,0,f.w.Bytes)&&dr.AsSpan(Native.Buffer.Guard,f.delta.Length).SequenceEqual(f.delta)&&wr.AsSpan(Native.Buffer.Guard,f.weights.Length).SequenceEqual(f.weights)
                    &&xr.AsSpan(Native.Buffer.Guard,f.First*f.W.K*4).SequenceEqual(f.input.AsSpan(0,f.First*f.W.K*4))&&xr.AsSpan(Native.Buffer.Guard+(f.First+f.N)*f.W.K*4,(f.Total-f.First-f.N)*f.W.K*4).SequenceEqual(f.input.AsSpan((f.First+f.N)*f.W.K*4));
                return new(new(){["x-residual-full-payload"]=xr.AsSpan(Native.Buffer.Guard,x.Bytes).ToArray(),["norm-f32"]=Active(yr,f.First,f.N,f.W.K*4),["quant-i8"]=Active(qr,f.First,f.N,f.W.K),["scales-f32"]=Active(sr,f.First,f.N,f.W.K/32*4),["projection-f32"]=Active(output,f.First,f.N,f.W.Out*4)},new(){["x"]=xr,["norm"]=yr,["quant"]=qr,["scales"]=sr,["projection"]=output,["delta"]=dr,["norm-weights"]=wr},guards);
            }
            public void Dispose(){try{outbuf?.Dispose();}finally{try{s?.Dispose();}finally{try{q?.Dispose();}finally{try{y?.Dispose();}finally{x?.Dispose();}}}}}
        }
        public void Dispose(){try{w?.Dispose();}finally{d?.Dispose();}}
    }
    private static string Fresh(string path)
    {Require(Path.GetFullPath(path)==path&&!File.Exists(path)&&!Directory.Exists(path),"Use fresh absolute normalized output.");for(var p=new DirectoryInfo(Path.GetDirectoryName(path)!);p!=null;p=p.Parent)Require(p.LinkTarget==null,"Output ancestor alias.");Directory.CreateDirectory(path);return path;}
    public static int Run(string[] args)
    {
        string? output=null,model=null,bundle=null,sourceRoot=null;bool execute=false,self=false;
        for(int i=0;i<args.Length;i++)switch(args[i]){case"--out":output=args[++i];break;case"--model":model=args[++i];break;case"--bundle":bundle=args[++i];break;case"--source-root":sourceRoot=args[++i];break;case"--execute":execute=true;break;case"--self-test":self=true;break;default:throw new ArgumentException(args[i]);}
        if(self){var a=Enumerable.Repeat((byte)0xa5,1024).ToArray();Require(Native.Buffer.Guarded(a,512,0xa5,0,512),"CPU guard positive");a[255]=0;Require(!Native.Buffer.Guarded(a,512,0xa5,0,512),"CPU guard negative");var program=Q35LocatorMethod();var native=Native.Q35LocatorMethod();var flags=BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic;
            bool oldAmbiguous=false;try{program.DeclaringType!.GetMethod("LocatePtxPath",flags);}catch(AmbiguousMatchException){oldAmbiguous=true;}Require(oldAmbiguous,"CPU fixture must expose actual old overload ambiguity.");
            foreach(var method in new[]{program,native})Require(method.IsStatic&&method.ReturnType==typeof(string)&&method.GetParameters().Length==1&&method.GetParameters()[0].ParameterType==typeof(string),"Exact string overload not selected.");
            Require(Q35KernelFile=="tensorsharp_q35_kernels.ptx","Must select Q35 filename, not generic kernels.");
            string fixture=Path.Combine(Path.GetTempPath(),"ts-q35-locator-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(fixture);
            try{string q35=Path.Combine(fixture,Q35KernelFile);File.WriteAllText(q35,"CPU locator fixture; not CUDA code.");File.WriteAllText(Path.Combine(fixture,"tensorsharp_kernels.ptx"),"different generic CPU fixture");foreach(var method in new[]{program,native})Require((string?)method.Invoke(null,new object[]{q35})==q35,"Actual string overload did not locate Q35 fixture.");}
            finally{Directory.Delete(fixture,true);}
            Require(Native.IsOriginalImage(Bbd.ToUpperInvariant()),"Uppercase original BBD hash must match exactly ignoring case.");
            Require(!Native.IsOriginalImage("0"+Bbd[1..]),"Wrong reference image hash must be rejected.");
            Require(!Native.IsOriginalImage(null),"Missing reference image hash must be rejected.");
            Console.WriteLine("11 CPU-only assertions PASS (2 guards, old ambiguity, 2 exact overloads, Q35 name, 2 real locator calls, uppercase/wrong/null image hashes); GPU/model NOT_RUN.");return 0;}
        Require(output!=null,"--out ABS_FRESH required");string dir=Fresh(output!);var records=new List<object>();var timed=new List<object>();var state=new Dictionary<string,object?>{["schema"]="q35-norm-q81-fusion-component/1",["complete"]=false,["quality_passed"]=false,["gpu_not_run"]=true,["records"]=records,["ABBA"]=timed,["scope"]="Standalone component, no serving/default/graph/mixed changes. Complete original norm+quant pipeline is timed; GEMV is a strict downstream gate outside timer.",["planned_strict_fixtures"]=192,["planned_ABBA_cells"]=30};
        void Persist()=>Write(Path.Combine(dir,"run.json"),state);Persist();
        if(!execute){state["status"]="PLAN_ONLY_GPU_NOT_RUN";Persist();return 0;}
        string? modelHash=null,cubinHash=null,ptxPath=null,ptxHash=null,nativeSourceHash=null,buildHash=null;string? cubin=null;Dictionary<string,string>? sourceBefore=null,assemblyBefore=null;
        Dictionary<string,string> Sources(string root)=>new[]{"build.py","Program.cs","Native.cs","Qwen35Q81NormFusionBench.csproj"}.ToDictionary(n=>n,n=>FileHash(Path.Combine(root,"eng/validation/Qwen35Q81NormFusionBench",n)));
        Dictionary<string,string> Assemblies()=>AppDomain.CurrentDomain.GetAssemblies().Where(a=>!a.IsDynamic&&!string.IsNullOrEmpty(a.Location)).ToDictionary(a=>a.Location,a=>FileHash(a.Location));
        try
        {
            Require(OperatingSystem.IsLinux()&&BitConverter.IsLittleEndian&&model!=null&&bundle!=null,"Linux GPU run requires pinned --model and offline --bundle.");
            foreach(string key in new[]{"TS_Q35_SHARED_SCRATCH","TS_Q35_GDN_PREPARED_PREFILL","TS_Q35_GREEDY_VERIFY","TS_Q35_CUDA_GRAPHS","TS_Q35_VERIFY_GRAPHS","TS_Q35_TOPK_PARTITIONED","TS_Q35_VERIFY_DECODE_Q81"})Require(Environment.GetEnvironmentVariable(key)is null or"0","Keep "+key+" OFF.");
            model=Path.GetFullPath(model!);bundle=Path.GetFullPath(bundle!);modelHash=FileHash(model);Require(modelHash==ModelSha,"Wrong target model.");
            using var build=JsonDocument.Parse(File.ReadAllText(Path.Combine(bundle,"build.json")));var b=build.RootElement;Require(b.GetProperty("compiled").GetBoolean()&&b.GetProperty("complete").GetBoolean(),"No actual offline compiled bundle.");cubin=Path.Combine(bundle,"kernels.cubin");cubinHash=FileHash(cubin);Require(cubinHash==b.GetProperty("cubin_sha256").GetString()&&FileHash(Path.Combine(bundle,"kernels.cu"))==b.GetProperty("generated_sha256").GetString(),"Bundle bytes changed.");
            if(sourceRoot==null){for(var at=new DirectoryInfo(Environment.CurrentDirectory);at!=null;at=at.Parent)if(File.Exists(Path.Combine(at.FullName,"eng/validation/Qwen35Q81NormFusionBench/Program.cs"))){sourceRoot=at.FullName;break;}}
            Require(sourceRoot!=null,"Pass --source-root for actual sources.");sourceRoot=Path.GetFullPath(sourceRoot!);sourceBefore=Sources(sourceRoot);foreach(var item in sourceBefore)Require(item.Value==b.GetProperty("tools").GetProperty(item.Key).GetString(),"Tool source changed: "+item.Key);
            nativeSourceHash=FileHash(Path.Combine(sourceRoot,"TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"));Require(nativeSourceHash==b.GetProperty("source_sha256").GetString(),"Production source differs from reviewed private build input.");state["source_before"]=sourceBefore;state["source_to_assembly_match_requires_external_build_proof"]=true;
            buildHash=FileHash(Path.Combine(bundle,"build.json"));state["build_manifest_sha256"]=buildHash;state["model"]=new{path=model,sha256=modelHash,bytes=new FileInfo(model).Length};state["cubin_sha256"]=cubinHash;state["host"]=Environment.MachineName;state["flags"]=Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(e=>((string)e.Key).StartsWith("TS_",StringComparison.Ordinal)).ToDictionary(e=>(string)e.Key,e=>e.Value?.ToString());
            assemblyBefore=Assemblies();state["assemblies_before"]=assemblyBefore;
            using var exec=new Qwen35CudaExecutor(model,8192,4096,draftGgufPath:null!);var engine=exec.Engine;Require(!engine.HasDrafter,"No drafter allowed.");
            var locate=Q35LocatorMethod();string ptx=(string)locate.Invoke(null,new object[]{Q35KernelFile})!;Require(FileHash(Path.ChangeExtension(ptx,".fatbin"))==Bbd,"Reference provider is not original BBD.");
            ptxPath=ptx;ptxHash=FileHash(ptx);state["reference_provider"]=new{ptx,ptx_sha256=ptxHash,fatbin_sha256=Bbd};
            using var api=new Native(engine,cubin);Require(api.PrivateImageSha==cubinHash,"Loaded private image bytes differ.");state["actual_private_loaded_image_sha256"]=api.PrivateImageSha;state["actual_reference_module"]=api.ReferenceIdentity;state["driver_resources"]=api.Resources();state["gpu_not_run"]=false;state["status"]="RUNNING_NOT_PASS";Persist();
            var desc=(Q35CudaEngine.ModelDesc)Native.Field(typeof(Q35CudaEngine),"_m").GetValue(engine)!;var layers=(Array)Native.Field(typeof(Q35CudaEngine),"_layers").GetValue(engine)!;
            int gdn=Enumerable.Range(0,layers.Length).First(i=>(bool)Native.Field(layers.GetValue(i)!.GetType(),"Recurrent",typeof(bool),true).GetValue(layers.GetValue(i))!);int attn=Enumerable.Range(0,layers.Length).First(i=>!(bool)Native.Field(layers.GetValue(i)!.GetType(),"Recurrent",typeof(bool),true).GetValue(layers.GetValue(i))!);
            var normp=(IntPtr)Native.Field(layers.GetValue(gdn)!.GetType(),"AttnNorm",typeof(IntPtr),true).GetValue(layers.GetValue(gdn))!;byte[] normBytes=Native.Download(normp,engine.HiddenSize*4);float[] actualNorm=new float[engine.HiddenSize];Buffer.BlockCopy(normBytes,0,actualNorm,0,normBytes.Length);state["actual_norm_weights_sha256"]=Hash(normBytes);
            var planes=new List<Plane>();foreach(var(l,role,label)in new[]{(gdn,"InProj","gdn-in"),(attn,"InProj","attn-in"),(gdn,"OutProj","gdn-out"),(attn,"OutProj","attn-out"),(gdn,"GateUp","gate-up"),(gdn,"Down","down")}){object layer=layers.GetValue(l)!,w=Native.Field(layer.GetType(),role,null,true).GetValue(layer)!;T Get<T>(string name)=>(T)Native.Field(w.GetType(),name,typeof(T),true).GetValue(w)!;planes.Add(new(label,l,Get<IntPtr>("Codes"),Get<IntPtr>("Scales"),Get<int>("Rows"),Get<int>("K")));}
            state["actual_six_planes"]=planes.Select(p=>new{p.Role,p.Layer,p.Out,p.K}).ToArray();bool pass=true;var references=new Dictionary<string,Snapshot>();
            foreach(var plane in planes)for(int n=1;n<=8;n++)for(int pattern=0;pattern<4;pattern++)
            {
                using var fixture=new Fixture(api,plane,n,pattern,desc.Eps,actualNorm);string id=plane.Role+"-n"+n+"-p"+pattern;Snapshot Get(bool fused,byte poison){using var work=fixture.Create(poison);work.Issue(fused);return work.Read();}
                var a=Get(false,0x35);var aa=Get(false,0xa6);var c=Get(true,0x57);var cc=Get(true,0xc8);var comparisons=new{reference_AA=Compare(a,aa),candidate_AA=Compare(c,cc),reference_candidate=Compare(a,c)};bool ok=Passed(comparisons.reference_AA)&&Passed(comparisons.candidate_AA)&&Passed(comparisons.reference_candidate);pass&=ok;
                records.Add(new{id,fixture=fixture.Identity(),strict_pass=ok,comparisons});if(!ok){SaveRaw(dir,id+"-referenceA",a);SaveRaw(dir,id+"-referenceB",aa);SaveRaw(dir,id+"-candidateA",c);SaveRaw(dir,id+"-candidateB",cc);}if(n==8&&pattern==0)references.Add(plane.Role,a);Persist();
            }
            Require(records.Count==192,"Incomplete strict inventory.");state["strict_all_pass"]=pass;
            if(pass)foreach(var plane in planes)
            {
                using var fixture=new Fixture(api,plane,8,0,desc.Eps,actualNorm);Require(!fixture.HasDelta,"Timed input must remain identical across iterations.");var reference=references[plane.Role];
                for(int round=0;round<5;round++)
                {
                    var halves=new List<object>();foreach(var(fused,name,poison)in new[]{(false,"A1",(byte)0x39),(true,"B1",(byte)0x4a),(true,"B2",(byte)0x5b),(false,"A2",(byte)0x6c)})
                    {using var work=fixture.Create(poison);float ms=api.Time(()=>work.Issue(fused),32);var result=work.Read();var comparison=Compare(reference,result);bool ok=Passed(comparison);pass&=ok;if(!ok)SaveRaw(dir,plane.Role+"-r"+round+"-"+name,result);halves.Add(new{name,fused,milliseconds_per_complete_norm_quant=ms,iterations=32,full_bits_and_guards_verified=ok,comparison});}
                    timed.Add(new{role=plane.Role,n=8,round,scope="norm+quant complete per iteration; downstream GEMV validated after each half, outside timer",halves});Persist();
                }
            }
            Require(!pass||timed.Count==30,"Incomplete ABBA inventory.");state["complete"]=true;state["quality_passed"]=pass;state["status"]=pass?"STRICT_PASS_COMPONENT_ONLY":"STRICT_FAIL_NO_INTEGRATION";
        }
        catch(Exception error){state["error"]=error.ToString();state["quality_passed"]=false;state["status"]="ERROR_NOT_PASS";}
        finally{try{if(modelHash!=null)Require(FileHash(model!)==modelHash,"Target changed.");if(cubinHash!=null)Require(FileHash(cubin!)==cubinHash,"Private cubin changed.");if(buildHash!=null)Require(FileHash(Path.Combine(bundle!,"build.json"))==buildHash,"Build manifest changed.");if(sourceBefore!=null){var after=Sources(sourceRoot!);state["source_after"]=after;Require(sourceBefore.All(p=>after[p.Key]==p.Value),"Sources changed during GPU run.");Require(FileHash(Path.Combine(sourceRoot!,"TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"))==nativeSourceHash,"Production native source changed.");}if(assemblyBefore!=null){var after=Assemblies();state["assemblies_after"]=after;Require(assemblyBefore.All(p=>after.TryGetValue(p.Key,out var hash)&&hash==p.Value),"Loaded assemblies changed.");}if(ptxPath!=null)Require(FileHash(Path.ChangeExtension(ptxPath,".fatbin"))==Bbd&&FileHash(ptxPath)==ptxHash,"Reference provider changed.");state["identity_postchecks_passed"]=modelHash!=null&&cubinHash!=null&&sourceBefore!=null&&assemblyBefore!=null&&ptxPath!=null;}catch(Exception e){state["postcheck_error"]=e.ToString();state["quality_passed"]=false;state["complete"]=false;}Persist();}
        return state["quality_passed"]is true?0:1;
    }
}
