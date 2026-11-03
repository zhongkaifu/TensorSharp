using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using W8HeadProbe;

internal static unsafe class Program
{
    private sealed record ReferenceCase(int Rows,byte[] Input,byte[] Bf16,byte[] Output,byte[] Reference,byte[] SumAbs);
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    private static void Save(string path,object value)=>File.WriteAllText(path,JsonSerializer.Serialize(value,Json)+"\n");
    private static string Required(Dictionary<string,string> args,string key)=>args.TryGetValue(key,out string? value)&&value.Length>0?value:throw new ArgumentException("Missing "+key);
    private static Dictionary<string,string> Args(string[] args)
    {
        var result=new Dictionary<string,string>(StringComparer.Ordinal);
        for(int i=0;i<args.Length;i++)
        {
            string key=args[i];Checks.Need(key is "--execute" or "--self-test" or "--self-test-output" or "--check-references" or "--help" or "--data" or "--inventory" or "--original-run" or "--native-manifest" or "--native-manifest-sha" or "--backend-sha" or "--out","unknown argument "+key);
            Checks.Need(!result.ContainsKey(key),"duplicate argument "+key);
            if(key is "--execute" or "--self-test" or "--self-test-output" or "--check-references" or "--help")result.Add(key,"1");else{Checks.Need(i+1<args.Length,"missing argument value");result.Add(key,args[++i]);}
        }
        return result;
    }
    private static void ValidateFreshOutput(string output,string allowedRoot)
    {
        string root=Path.GetFullPath(allowedRoot).TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar;
        Checks.Need(output==Path.GetFullPath(output)&&output.StartsWith(root,StringComparison.Ordinal)&&!Path.Exists(output),"fresh normalized output under allowed root required");
        for(string? p=output;p is not null;p=Path.GetDirectoryName(p))Checks.Need(new DirectoryInfo(p).LinkTarget is null,"output alias ancestor");
    }
    private static void SaveOwnedFailure(string? output,bool owned,object state)
    {
        if(owned&&output is not null&&Directory.Exists(output))Save(Path.Combine(output,"result.json"),state);
    }
    private static void OutputSelfTest()
    {
        string root=Path.Combine(AppContext.BaseDirectory,"output-contract-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(root);int checks=0;
        try
        {
            string foreign=Path.Combine(root,"foreign");Directory.CreateDirectory(foreign);string sentinel=Path.Combine(foreign,"result.json");File.WriteAllText(sentinel,"foreign-sentinel");
            void Refuse(string path)
            {
                bool rejected=false;try{ValidateFreshOutput(path,root);}catch(InvalidDataException){rejected=true;}
                Checks.Need(rejected,"output negative accepted");SaveOwnedFailure(path,false,new{complete=false});Checks.Need(File.ReadAllText(sentinel)=="foreign-sentinel","foreign output overwritten");checks++;
            }
            Refuse(foreign);
            string alias=Path.Combine(root,"final-alias");Directory.CreateSymbolicLink(alias,foreign);try{Refuse(alias);Refuse(Path.Combine(alias,"not-created"));Checks.Need(!Directory.Exists(Path.Combine(foreign,"not-created")),"alias parent write");}finally{Directory.Delete(alias);}
            Refuse(Path.Combine(root,"..","outside"));Refuse(Path.GetDirectoryName(root)!);
            string own=Path.Combine(root,"own");ValidateFreshOutput(own,root);Directory.CreateDirectory(own);SaveOwnedFailure(own,true,new{complete=false});Checks.Need(File.Exists(Path.Combine(own,"result.json")),"owned failure not preserved");checks++;
            Console.WriteLine(JsonSerializer.Serialize(new{output_contract_pass=true,actual_path_contracts=checks,gpu_not_run=true,complete=false}));
        }
        finally{Directory.Delete(root,true);}
    }
    private static void FileIdentity(string path,JsonElement row)
    {
        Checks.Need(File.Exists(path)&&new FileInfo(path).LinkTarget is null,"regular reference file required");
        Checks.Need(new FileInfo(path).Length==row.GetProperty("bytes").GetInt64()&&Checks.Sha(path)==row.GetProperty("sha256").GetString(),"frozen full array identity differs: "+path);
    }
    private static ReferenceCase LoadReference(string root,JsonElement inventory,int rows)
    {
        string shape=$"r248320-k5120-t{rows}-p3";
        byte[] Read(string name)
        {
            string relative=shape+"/"+name,path=Path.Combine(root,shape,name);FileIdentity(path,inventory.GetProperty(relative));return File.ReadAllBytes(path);
        }
        var r=new ReferenceCase(rows,Read("input.f32"),Read("input.bf16"),Read("official.bf16"),Read("reference.f64"),Read("sumAbs.f64"));
        Checks.Need(r.Input.Length==rows*Checks.K*4&&r.Bf16.Length==rows*Checks.K*2&&r.Output.Length==rows*Checks.Vocab*2,"original real shape lengths");
        for(int i=0;i<rows*Checks.K;i++)
        {
            uint bits=BinaryPrimitives.ReadUInt32LittleEndian(r.Input.AsSpan(4*i));Checks.Need(bits==Checks.InputBits(i),"input not original fixed pattern3");
            Checks.Need(BinaryPrimitives.ReadUInt16LittleEndian(r.Bf16.AsSpan(2*i))==Checks.Bf16(bits),"original BF16 cast differs from independent RNE");
        }
        Checks.Output(r.Output,r.Output,r.Reference,r.SumAbs);return r;
    }
    private static void VerifyPrefixReferences(Dictionary<int,ReferenceCase> references)
    {
        var largest=references[32];
        foreach(int n in Checks.DirectRows.Where(n=>n<=32))
        {
            var r=references[n];Checks.Need(r.Input.AsSpan().SequenceEqual(largest.Input.AsSpan(0,r.Input.Length)),"original input-prefix consistency");
            Checks.Need(r.Output.AsSpan().SequenceEqual(largest.Output.AsSpan(0,r.Output.Length)),"original same-Kwarps BF16-prefix consistency");
            Checks.Need(r.Reference.AsSpan().SequenceEqual(largest.Reference.AsSpan(0,r.Reference.Length))&&r.SumAbs.AsSpan().SequenceEqual(largest.SumAbs.AsSpan(0,r.SumAbs.Length)),"original FP64 prefix consistency");
        }
    }
    private static nint GenerateWeights(out string hash)
    {
        nint memory=(nint)NativeMemory.AlignedAlloc((nuint)Checks.WeightBytes,256);if(memory==0)throw new OutOfMemoryException();
        try
        {
            var bytes=new Span<byte>((void*)memory,checked((int)Checks.WeightBytes));
            for(int r=0;r<Checks.Vocab;r++)
            {
                int at=r*Checks.K;
                for(int c=0;c<Checks.K;c++)bytes[at+c]=Checks.Code(r,c);
                for(int g=0;g<Checks.K/32;g++)BinaryPrimitives.WriteUInt16LittleEndian(bytes.Slice(checked((int)Checks.ScaleOffset)+(r*160+g)*2,2),Checks.Scale(r,g));
            }
            hash=Checks.Sha(bytes);return memory;
        }
        catch{NativeMemory.AlignedFree((void*)memory);throw;}
    }
    private static string ReadbackHash(DeviceBuffer buffer)
    {
        using var hash=System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);byte[] chunk=new byte[8<<20];
        for(long off=0;off<buffer.Length;off+=chunk.Length)
        {
            int n=(int)Math.Min(chunk.Length,buffer.Length-off);fixed(byte*p=chunk)Native.Ok(Native.DtoH((nint)p,buffer.Pointer+(nint)off,(nuint)n));hash.AppendData(chunk,0,n);
        }
        buffer.Guard();return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
    public static int Main(string[] argv)
    {
        string? output=null;bool ownedOutput=false;nint weightsHost=0;var cases=new List<object>();var state=new Dictionary<string,object?>{{"schema","q35-original-W8-CSharp-driver-probe/1"},{"complete",false},{"gpu_not_run",true},{"model_quality","NOT_RUN"},{"performance_qualification","NOT_RUN"}};
        try
        {
            var args=Args(argv);
            Checks.Need(new[]{"--execute","--self-test","--self-test-output","--check-references"}.Count(args.ContainsKey)<=1,"select one execution mode");
            if(args.ContainsKey("--self-test")){Checks.SelfTest();return 0;}
            if(args.ContainsKey("--self-test-output")){OutputSelfTest();return 0;}
            if(args.ContainsKey("--check-references"))
            {
                string inv=Required(args,"--inventory"),run=Required(args,"--original-run");Checks.Need(Checks.Sha(inv)==Checks.OriginalInventorySha&&Checks.Sha(run)==Checks.OriginalRunSha,"frozen original metadata differs");
                using var d=JsonDocument.Parse(File.ReadAllBytes(inv));using var r=JsonDocument.Parse(File.ReadAllBytes(run));Checks.Need(r.RootElement.GetProperty("complete").GetBoolean()&&r.RootElement.GetProperty("observed").GetProperty("strict_unique").GetInt32()==32,"original run incomplete");
                var loaded=Checks.DirectRows.ToDictionary(n=>n,n=>LoadReference(Required(args,"--data"),d.RootElement,n));VerifyPrefixReferences(loaded);
                Console.WriteLine(JsonSerializer.Serialize(new{complete=false,gpu_not_run=true,reference_check_pass=true,original_real_cases=8,actual_files_read_and_SHA_checked=40,all_existing_same_Kwarps_prefixes_exact=true}));return 0;
            }
            if(!args.ContainsKey("--execute"))
            {
                Console.WriteLine(JsonSerializer.Serialize(new{complete=false,gpu_not_run=true,rows=Enumerable.Range(1,33),direct_original_rows=Checks.DirectRows,prefix_reference_rows=Enumerable.Range(1,33).Select(n=>new{n,oracle=Checks.OracleRows(n)}),required=new[]{"--data","--inventory","--original-run","--native-manifest","--native-manifest-sha","--backend-sha","--out"},Root_only_actual_execution=true}));return 0;
            }
            Checks.Need(OperatingSystem.IsLinux()&&BitConverter.IsLittleEndian&&IntPtr.Size==8,"execution requires Linux64 little endian; CPU plan/self-test is portable");
            output=Required(args,"--out");ValidateFreshOutput(output,"/dev/shm");
            Directory.CreateDirectory(output);ownedOutput=true;Save(Path.Combine(output,"result.json"),state);
            string data=Required(args,"--data"),inventoryPath=Required(args,"--inventory"),originalRun=Required(args,"--original-run"),manifestPath=Required(args,"--native-manifest"),expectedManifest=Required(args,"--native-manifest-sha"),backend=typeof(CudaW8G32HeadModule).Assembly.Location;
            Checks.Need(Checks.Sha(inventoryPath)==Checks.OriginalInventorySha&&Checks.Sha(originalRun)==Checks.OriginalRunSha,"frozen original reference metadata SHA differs");
            Checks.Need(Checks.Sha(manifestPath)==expectedManifest,"new actual native manifest SHA differs");Checks.Need(string.Equals(Checks.Sha(backend),Required(args,"--backend-sha"),StringComparison.OrdinalIgnoreCase),"actual loaded backend assembly differs");
            using var inventory=JsonDocument.Parse(File.ReadAllBytes(inventoryPath));using var original=JsonDocument.Parse(File.ReadAllBytes(originalRun));using var nativeManifest=JsonDocument.Parse(File.ReadAllBytes(manifestPath));var native=nativeManifest.RootElement;
            Checks.Need(original.RootElement.GetProperty("complete").GetBoolean()&&original.RootElement.GetProperty("observed").GetProperty("strict_unique").GetInt32()==32,"original strict run not complete");
            Checks.Need(native.GetProperty("schema").GetString()=="qwen35-original-W8-head-module-build/1"&&native.GetProperty("complete").GetBoolean()&&native.GetProperty("closure_sha256").GetString()=="66b48ffe40fc820ac4cfab5c9cfcc32ee93042514c1944dafd7a16a44f2e7c9c","original-source native build required");
            string dir=Path.GetDirectoryName(manifestPath)!;var artifacts=native.GetProperty("artifacts");
            foreach(string name in new[]{"tensorsharp_w8g32_head.ptx","tensorsharp_w8g32_head.fatbin","tensorsharp_w8g32_head.fatbin.sha256"})Checks.Need(Checks.Sha(Path.Combine(dir,name))==artifacts.GetProperty(name).GetString(),"actual native artifact changed");
            string fatbinHash=artifacts.GetProperty("tensorsharp_w8g32_head.fatbin").GetString()!;
            var identities=new Dictionary<string,string>(StringComparer.Ordinal){{inventoryPath,Checks.OriginalInventorySha},{originalRun,Checks.OriginalRunSha},{manifestPath,expectedManifest},{backend,Checks.Sha(backend)},{Assembly.GetExecutingAssembly().Location,Checks.Sha(Assembly.GetExecutingAssembly().Location)}};
            foreach(string name in new[]{"tensorsharp_w8g32_head.ptx","tensorsharp_w8g32_head.fatbin","tensorsharp_w8g32_head.fatbin.sha256"})identities.Add(Path.Combine(dir,name),artifacts.GetProperty(name).GetString()!);
            foreach(int n in Checks.DirectRows)foreach(string name in new[]{"input.f32","input.bf16","official.bf16","reference.f64","sumAbs.f64"}){string relative=$"r248320-k5120-t{n}-p3/"+name;identities.Add(Path.Combine(data,relative),inventory.RootElement.GetProperty(relative).GetProperty("sha256").GetString()!);}
            state["immutable_input_files_before"]=identities;
            state["inputs"]=new{data,inventory_sha256=Checks.Sha(inventoryPath),original_run_sha256=Checks.Sha(originalRun),native_manifest_sha256=expectedManifest,original_cpp_bench_sha256=Checks.OriginalBenchSha,probe_assembly_sha256=Checks.Sha(Assembly.GetExecutingAssembly().Location),backend_assembly_sha256=Checks.Sha(backend),native_resources=native.GetProperty("resources").Clone()};
            var references=Checks.DirectRows.ToDictionary(n=>n,n=>LoadReference(data,inventory.RootElement,n));VerifyPrefixReferences(references);
            weightsHost=GenerateWeights(out string weightHash);state["synthetic_weights"]=new{bytes=Checks.WeightBytes,sha256=weightHash,formula="frozen_CPP_pattern3_signed_code_and_original_half_scale_generator",actual_Ignis_artifact_loaded=false};
            using var context=CudaContext.Create(0);using var stream=CudaStream.Create();using var module=CudaW8G32HeadModule.LoadFromFile(Path.Combine(dir,"tensorsharp_w8g32_head.ptx"));
            Checks.Need(string.Equals(module.ImageSha256,fatbinHash,StringComparison.OrdinalIgnoreCase),"expected actual SASS fatbin not loaded; no fallback certification");Checks.Need(module.Kernels.Count==34&&module.Kernels.Select(k=>k.Rows).Order().SequenceEqual(Enumerable.Range(0,34)),"all actual driver resources/paraminfo not present");state["actual_module"]=new{module.ImageSha256,module.LaunchPolicy,module.Kernels};
            state["parameter_info_gate"]=new{verified_by_actual_component_constructor=true,head_function_count=33,cast_function_count=1,head_offsets=new[]{0,8,16,24,40,41},head_sizes=new[]{8,8,8,16,1,1},cast_offsets=new[]{0,8,16},cast_sizes=new[]{8,8,8},extra_parameter_result_required=1,raw_parameter_observations_independently_exported=false};state["driver_context_and_module_loaded"]=true;state["actual_device"]=Native.DeviceIdentity();
            using var weights=new DeviceBuffer(Checks.WeightBytes);weights.Upload(weightsHost);NativeMemory.AlignedFree((void*)weightsHost);weightsHost=0;
            try
            {
                foreach(int n in Enumerable.Range(1,33))
                {
                    var reference=references[Checks.OracleRows(n)];int x=n*Checks.K*4,b=n*Checks.K*2,y=n*Checks.Vocab*2;var expected=reference.Output.AsSpan(0,y);var fp64=reference.Reference.AsSpan(0,y*4);var sumAbs=reference.SumAbs.AsSpan(0,y*4);
                    using var input=new DeviceBuffer(x);using var workspace=new DeviceBuffer(b);using var result=new DeviceBuffer(y);input.Upload(reference.Input.AsSpan(0,x));var intervals=new List<float>();byte[]? first=null;
                    try
                    {
                    for(int pass=0;pass<4;pass++)
                    {
                        byte poison=(pass&1)==0?(byte)0xa5:(byte)0x5a;workspace.Paint(poison);result.Paint(poison);using var timer=new EventInterval();
                        state["gpu_not_run"]=false;
                        if(pass<2){timer.Start(stream.Handle);module.LaunchF32(weights.Pointer,weights.Length,input.Pointer,input.Length,workspace.Pointer,workspace.Length,result.Pointer,result.Length,n,stream.Handle);intervals.Add(timer.Finish(stream.Handle));}
                        else{workspace.Upload(reference.Bf16.AsSpan(0,b));module.LaunchBf16(weights.Pointer,weights.Length,workspace.Pointer,workspace.Length,result.Pointer,result.Length,n,stream.Handle);}
                        stream.Synchronize();byte[] actual=result.Read();
                        // Persist each successful read before validation; an incorrect output remains reviewable.
                        File.WriteAllBytes(Path.Combine(output,$"T{n}-pass{pass}.bf16"),actual);
                        state["last_observation"]=new{rows=n,pass,output_sha256=Checks.Sha(actual)};Save(Path.Combine(output,"result.json"),state);
                        Checks.Output(actual,expected,fp64,sumAbs);Checks.Need(workspace.Read().AsSpan().SequenceEqual(reference.Bf16.AsSpan(0,b)),"GPU input cast differs from frozen independent CPU/GPU input");Checks.Need(input.Read().AsSpan().SequenceEqual(reference.Input.AsSpan(0,x)),"F32 input mutated");if(first is null)first=actual;else Checks.Need(first.AsSpan().SequenceEqual(actual),"F32/BF16 paths or AA bits differ");input.Guard();workspace.Guard();result.Guard();weights.Guard();
                    }
                    }
                    finally{context.MakeCurrent();stream.Synchronize();}
                    result.NegativeGuardControl();File.WriteAllBytes(Path.Combine(output,$"T{n}.bf16"),first!);
                    for(int pass=0;pass<4;pass++)File.Delete(Path.Combine(output,$"T{n}-pass{pass}.bf16"));
                    cases.Add(new{rows=n,original_oracle_rows=reference.Rows,oracle_kind=reference.Rows==n?"direct_frozen_CPP":"source_proven_independent_token_bucket_prefix",fullbits=true,unchanged_FP64_bound=true,F32_AA=true,BF16_AA=true,dual_poison=true,guards=true,actual_driver_allocation_ranges=true,negative_guard_rejected=true,output_sha256=Checks.Sha(first!),complete_F32_cast_matmul_BF16_store_gpu_ms_DIAGNOSTIC_ONLY=intervals});state["cases"]=cases;Save(Path.Combine(output,"result.json"),state);
                }
                stream.Synchronize();Checks.Need(ReadbackHash(weights)==weightHash,"entire GPU weight code/scale bytes changed");
                using var lifeInput=new DeviceBuffer(Checks.K*4);using var lifeWorkspace=new DeviceBuffer(Checks.K*2);using var lifeOutput=new DeviceBuffer(Checks.Vocab*2);lifeInput.Upload(references[1].Input);bool wrongContext=false;
                Native.Ok(Native.SetContext(0));try{module.LaunchF32(weights.Pointer,weights.Length,lifeInput.Pointer,lifeInput.Length,lifeWorkspace.Pointer,lifeWorkspace.Length,lifeOutput.Pointer,lifeOutput.Length,1,stream.Handle);}catch(InvalidOperationException){wrongContext=true;}finally{context.MakeCurrent();}
                Checks.Need(wrongContext,"missing current-context rejection");stream.Synchronize();module.Dispose();bool disposed=false;try{module.LaunchF32(weights.Pointer,weights.Length,lifeInput.Pointer,lifeInput.Length,lifeWorkspace.Pointer,lifeWorkspace.Length,lifeOutput.Pointer,lifeOutput.Length,1,stream.Handle);}catch(ObjectDisposedException){disposed=true;}Checks.Need(disposed,"disposed module allowed launch");
                lifeOutput.Dispose();bool bufferDisposed=false;try{_ =lifeOutput.Pointer;}catch(ObjectDisposedException){bufferDisposed=true;}Checks.Need(bufferDisposed,"disposed allocation still exposes pointer");state["lifecycle_negative_controls"]=new{wrong_context_rejected=wrongContext,disposed_module_rejected=disposed,disposed_buffer_rejected=bufferDisposed,all_work_synchronized_before_disposal=true};
            }
            finally{context.MakeCurrent();Native.Ok(Native.SyncContext());}
            Checks.Need(cases.Count==33,"not all33 numeric cases");
            foreach(var identity in identities)Checks.Need(Checks.Sha(identity.Key)==identity.Value,"immutable input changed after execution: "+identity.Key);
            state["immutable_input_files_after_equal"]=true;state["complete"]=true;state["numeric_pass"]=true;state["coverage"]=new{new_CSharp_rows=33,direct_original_CPP_rows=8,source_proven_prefix_rows=25,original_CPP_rows_NOT_independently_reexecuted=true};Save(Path.Combine(output,"result.json"),state);Console.WriteLine(JsonSerializer.Serialize(new{complete=true,numeric_pass=true,cases=33,model_quality="NOT_RUN",performance_qualification="NOT_RUN"}));return 0;
        }
        catch(Exception e){state["complete"]=false;state["error"]=e.ToString();SaveOwnedFailure(output,ownedOutput,state);Console.Error.WriteLine(e);return 1;}
        finally{if(weightsHost!=0)NativeMemory.AlignedFree((void*)weightsHost);}
    }
}
