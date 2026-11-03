using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class Program
{
    private static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    private sealed record Case(string Id,string Text);
    private readonly record struct Reads(long Bytes,long Calls);
    private static Reads Stats(NinferArtifactReader reader)=>new(reader.TotalBytesRead,reader.ReadOperations);
    private static object Delta(Reads before,Reads after)=>new{bytes=after.Bytes-before.Bytes,calls=after.Calls-before.Calls};
    private static void Save(string file,object data)
    {
        using var output=new FileStream(file,FileMode.Create,FileAccess.Write,FileShare.None);JsonSerializer.Serialize(output,data,Json);output.WriteByte(10);
    }
    private static Dictionary<string,string> Args(string[] argv)
    {
        var result=new Dictionary<string,string>(StringComparer.Ordinal);
        for(int i=0;i<argv.Length;i++)
        {
            string key=argv[i];Checks.Need(key is "--execute" or "--self-test" or "--self-test-output" or "--help" or "--artifact" or "--directory-sha" or "--known-full-sha" or "--out" or "--read-calibrations","Unknown argument: "+key);
            Checks.Need(!result.ContainsKey(key),"Duplicate argument: "+key);
            if(key is "--execute" or "--self-test" or "--self-test-output" or "--help" or "--read-calibrations")result.Add(key,"1");
            else{Checks.Need(++i<argv.Length&&!string.IsNullOrWhiteSpace(argv[i]),"Missing value: "+key);result.Add(key,argv[i]);}
        }
        Checks.Need(new[]{"--execute","--self-test","--self-test-output"}.Count(result.ContainsKey)<=1,"Select one execution mode.");
        foreach(string key in new[]{"--directory-sha","--known-full-sha"})if(result.TryGetValue(key,out string? value)){Checks.Need(value.Length==64&&value.All(Uri.IsHexDigit),"Invalid SHA: "+key);result[key]=value.ToLowerInvariant();}
        return result;
    }
    private static string Required(Dictionary<string,string> options,string key)=>options.TryGetValue(key,out string? value)?value:throw new ArgumentException("Missing "+key);
    private static void FreshOutput(string output)
    {
        Checks.Need(output==Path.GetFullPath(output),"Canonical absolute output required.");
        string[] parts=output.Split(Path.DirectorySeparatorChar,StringSplitOptions.RemoveEmptyEntries);
        Checks.Need(parts.Contains("artifacts",StringComparer.Ordinal)||Enumerable.Range(0,Math.Max(0,parts.Length-1)).Any(i=>parts[i]=="docs"&&parts[i+1]=="validation"),"Output must be in ignored artifacts/ or docs/validation/.");
        Checks.Need(!Path.Exists(output)&&!IsLink(output),"Fresh output directory required.");
        for(string? p=Path.GetDirectoryName(output);p is not null;p=Path.GetDirectoryName(p))Checks.Need(!IsLink(p)&&!File.Exists(p),"Output parent alias/non-directory refused.");
    }
    private static bool IsLink(string path)=>new FileInfo(path).LinkTarget is not null||new DirectoryInfo(path).LinkTarget is not null;
    private static void OwnFailure(string? output,bool owned,object state){if(owned&&output is not null)Save(Path.Combine(output,"result.json"),state);}
    private static void OutputSelfTest()
    {
        string root=Path.Combine(AppContext.BaseDirectory,"artifacts","import-output-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(root);int count=0;
        try
        {
            string foreign=Path.Combine(root,"foreign");Directory.CreateDirectory(foreign);string sentinel=Path.Combine(foreign,"result.json");File.WriteAllText(sentinel,"foreign-sentinel");
            void Refuse(string path){bool refused=false;try{FreshOutput(path);}catch(InvalidDataException){refused=true;}Checks.Need(refused,"Unsafe output accepted.");OwnFailure(path,false,new{complete=false});Checks.Need(File.ReadAllText(sentinel)=="foreign-sentinel","Foreign output overwritten.");count++;}
            Refuse(foreign);string alias=Path.Combine(root,"alias");Directory.CreateSymbolicLink(alias,foreign);try{Refuse(alias);Refuse(Path.Combine(alias,"child"));}finally{Directory.Delete(alias);}
            Refuse(Path.Combine(root,"..","raw"));Refuse(Path.GetPathRoot(root)!);
            string own=Path.Combine(root,"owned");FreshOutput(own);Directory.CreateDirectory(own);OwnFailure(own,true,new{complete=false});Checks.Need(File.Exists(Path.Combine(own,"result.json")),"Owned failure lost.");count++;
            Console.WriteLine(JsonSerializer.Serialize(new{output_contract_pass=true,CPU_path_contracts=count,GPU="NOT_RUN",model="NOT_RUN"}));
        }
        finally{Directory.Delete(root,true);}
    }
    private static object Identity(Assembly assembly)=>new{path=assembly.Location,sha256=Checks.HashFile(assembly.Location)};
    private static object ArtifactStat(string path)
    {
        var f=new FileInfo(path);Checks.Need(f.Exists&&f.LinkTarget is null,"Regular original artifact required.");
        return new{f.FullName,f.Length,f.LastWriteTimeUtc,f.CreationTimeUtc,f.LinkTarget};
    }
    private static List<Case> Cases(out byte[] bytes)
    {
        using var input=Assembly.GetExecutingAssembly().GetManifestResourceStream("NinferQwen38ImportProbe.tokenizer-cases.json")??throw new InvalidDataException("Missing built-in cases.");
        using var memory=new MemoryStream();input.CopyTo(memory);bytes=memory.ToArray();using var doc=JsonDocument.Parse(bytes);
        var rows=doc.RootElement.EnumerateArray().Select(x=>new Case(x.GetProperty("id").GetString()!,x.GetProperty("text").GetString()!)).ToList();
        Checks.Need(rows.Count==22&&rows.Select(x=>x.Id).Distinct(StringComparer.Ordinal).Count()==rows.Count,"Fixed unique tokenizer cases differ.");return rows;
    }
    private static object ExportTokenizer(NinferQwen38Frontend frontend,string output)
    {
        var tokenizer=frontend.Tokenizer;Checks.Need(tokenizer.VocabSize==248077&&tokenizer.HeadVocabSize==248320&&tokenizer.SpecialTokenIds.Count==21&&tokenizer.BosTokenId==-1,"Actual tokenizer counts/BOS differ.");
        var cases=Cases(out byte[] originalCases);File.WriteAllBytes(Path.Combine(output,"tokenizer-cases.input.json"),originalCases);var records=new List<object>();bool roundtrip=true;
        foreach(var item in cases)
        {
            var ids=tokenizer.Encode(item.Text,false);var withSpecial=tokenizer.Encode(item.Text,true);var bytes=new List<byte>();foreach(int id in ids)tokenizer.AppendTokenBytes(id,bytes);
            string decoded=tokenizer.Decode(ids);bool same=decoded==item.Text.Normalize(NormalizationForm.FormC),noInjection=ids.SequenceEqual(withSpecial);roundtrip&=same&&noInjection;
            records.Add(new{id=item.Id,text=item.Text,utf8Base64=Convert.ToBase64String(Encoding.UTF8.GetBytes(item.Text)),ids,addSpecialFalse=true,addSpecialTrueIds=withSpecial,noBosEosInjection=noInjection,decoded,decodedTokenBytesBase64=Convert.ToBase64String(bytes.ToArray()),decodedEqualsNFC=same,HF_ID_oracle="NOT_RUN"});
        }
        Save(Path.Combine(output,"tokenizer-cases.output.json"),records);
        using var framed=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);string tokenPath=Path.Combine(output,"token-bytes.jsonl");var piece=new List<byte>();long byteCount=0;int[] oneByteIds=Enumerable.Repeat(-1,256).ToArray();
        using(var writer=new StreamWriter(new FileStream(tokenPath,FileMode.CreateNew,FileAccess.Write,FileShare.None),new UTF8Encoding(false)))
        {
            for(int id=0;id<tokenizer.VocabSize;id++)
            {
                piece.Clear();tokenizer.AppendTokenBytes(id,piece);byte[] raw=piece.ToArray();byteCount+=raw.Length;Checks.TokenFrame(framed,id,raw);
                if(id>=248044)Checks.Need(tokenizer.Encode(Encoding.UTF8.GetString(raw),false).SequenceEqual(new[]{id}),"Actual added-token atomic encoding differs.");
                if(raw.Length==1){Checks.Need(oneByteIds[raw[0]]==-1,"Duplicate single-byte token.");oneByteIds[raw[0]]=id;}
                writer.WriteLine(JsonSerializer.Serialize(new{id,byteLength=raw.Length,bytesBase64=Convert.ToBase64String(raw),sha256=Checks.Hash(raw)}));
            }
        }
        Checks.Need(oneByteIds.All(id=>id>=0),"Complete256-byte tokenizer inverse not found.");string singleBytePath=Path.Combine(output,"single-byte-token-ids.json");Save(singleBytePath,Enumerable.Range(0,256).Select(b=>new{byteValue=b,id=oneByteIds[b]}).ToArray());
        var result=new{validVocabulary=tokenizer.VocabSize,headRows=tokenizer.HeadVocabSize,baseVocabulary=248044,atomicAdded=tokenizer.VocabSize-248044,HF_special_true=tokenizer.SpecialTokenIds.Count,bosAutomaticallyAdded=false,tokenizer.BosTokenId,eos=tokenizer.EosTokenIds,normalizer="NFC_SCHEMA_VALIDATED",caseCount=records.Count,internalNfcRoundtripAndNoBosInjection=roundtrip,caseInputSha256=Checks.Hash(originalCases),caseOutputSha256=Checks.HashFile(Path.Combine(output,"tokenizer-cases.output.json")),singleByteTokens=new{rows=256,file="single-byte-token-ids.json",sha256=Checks.HashFile(singleBytePath)},allValidTokenBytes=new{file="token-bytes.jsonl",fileSha256=Checks.HashFile(tokenPath),rows=tokenizer.VocabSize,totalTokenBytes=byteCount,framedSha256=Checks.Finish(framed),framing="uint32LE(id) || uint32LE(byteLength) || bytes, IDs 0..248076 inclusive"},oracle=new{encode="NOT_RUN",decode="NOT_RUN",required="Original HF tokenizer.json; add_special_tokens=false; decode(skip_special_tokens=false, clean_up_tokenization_spaces=false). Compare exact IDs and per-ID token bytes, not just decoded text."}};
        Save(Path.Combine(output,"tokenizer-export.json"),result);Checks.Need(roundtrip,"An internal tokenizer NFC roundtrip/BOS injection check failed; original outputs are preserved.");return result;
    }
    private static object ReadCalibrations(NinferArtifactReader reader,NinferQwen38TargetPlan plan)
    {
        var list=new List<object>();Span<byte> raw=stackalloc byte[4];var tensors=plan.Tensors.Where(x=>x.Descriptor.Format==NinferTensorFormat.NvFp4).ToArray();Checks.Need(tensors.Length==247,"247 target NVFP4 calibrations required.");var names=new HashSet<string>(StringComparer.Ordinal);
        foreach(var tensor in tensors)
        {
            var d=tensor.InputScaleDivisor??throw new InvalidDataException("Missing separate activation divisor.");Checks.Need(names.Add(d.Name)&&d.ByteLength==4&&d.WeightDivisor is null,"Input divisor must be distinct scalar, not weight divisor.");
            reader.ReadObjectRange(d,0,raw);uint bits=BinaryPrimitives.ReadUInt32LittleEndian(raw);float value=Checks.PositiveF32(bits);
            list.Add(new{name=d.Name,d.AbsoluteOffset,rawF32Bits=bits.ToString("x8"),value,weightDivisorRawBits=tensor.WeightDivisorBits!.Value.ToString("x8"),combined=false});
        }
        return new{count=list.Count,contents="EXPLICITLY_READ_POSITIVE_FINITE_ORIGINAL_RAW_BITS",originalImmutablePlanStatus="NotRead declaration remains immutable",rows=list,notAppliedToActivations=true};
    }
    private static object Planes(NinferArtifactReader reader,NinferQwen38TargetPlan plan)
    {
        var binding=plan.Layers[0].OutputProjection;Checks.Need(binding.Descriptor.Name=="text/layers/0/gdn/output"&&binding.Descriptor.ByteLength==17694724,"Default bounded real tensor differs.");
        using var converter=NinferNvfp4PlaneReader.FromTarget(reader,binding);Checks.Need(converter.RowMapping==NinferNvfp4RowMapping.Identity&&converter.Rows==5120&&converter.Columns==6144&&converter.TileCount==40,"Default Identity geometry differs.");
        using var sourceCodes=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);using var inverseCodes=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);using var sourceScales=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);using var inverseScales=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] originalCodes=new byte[converter.CodeBytesPerTile],originalScales=new byte[converter.ScaleBytesPerTile];var rows=new List<object>();
        for(int index=0;index<converter.TileCount;index++)
        {
            var tile=converter.ReadTile(index);var inverse=Checks.Inverse(tile.Codes.Span,tile.Scales.Span,tile.Columns);
            reader.ReadObjectRange(binding.Descriptor,checked((ulong)index*(ulong)originalCodes.Length),originalCodes);
            reader.ReadObjectRange(binding.Descriptor,checked(binding.Descriptor.Geometry.ScalePlaneOffset!.Value+(ulong)index*(ulong)originalScales.Length),originalScales);
            Checks.Need(inverse.Codes.AsSpan().SequenceEqual(originalCodes)&&inverse.Scales.AsSpan().SequenceEqual(originalScales),"Original code/scale bytes differ from independent inverse.");
            Checks.Need(tile.FirstRow==index*128&&tile.OriginalSourceRowTiles.SequenceEqual(new[]{index})&&tile.WeightDivisorBits==binding.WeightDivisorBits&&ReferenceEquals(tile.InputScaleDivisor,binding.InputScaleDivisor),"Tile mapping/divisor ownership differs.");
            sourceCodes.AppendData(originalCodes);inverseCodes.AppendData(inverse.Codes);sourceScales.AppendData(originalScales);inverseScales.AppendData(inverse.Scales);
            rows.Add(new{index,tile.FirstRow,rows=128,sourceTile=index,codesSha256=Checks.Hash(originalCodes),scalesSha256=Checks.Hash(originalScales),allBytesExact=true});
        }
        Span<byte> divisor=stackalloc byte[4];reader.ReadObjectRange(binding.Descriptor,binding.Descriptor.ByteLength-4,divisor);uint weightBits=BinaryPrimitives.ReadUInt32LittleEndian(divisor);Checks.Need(weightBits==converter.WeightDivisorBits,"Original weight divisor bits changed.");
        string c=Checks.Finish(sourceCodes),ci=Checks.Finish(inverseCodes),s=Checks.Finish(sourceScales),si=Checks.Finish(inverseScales);Checks.Need(c==ci&&s==si,"Whole plane SHA mismatch.");
        return new{name=binding.Descriptor.Name,formatOnly=true,sourceBytes=binding.Descriptor.ByteLength,rows=converter.Rows,columns=converter.Columns,tiles=converter.TileCount,mapping="Identity",originalCodesSha256=c,inverseCodesSha256=ci,originalScalesSha256=s,inverseScalesSha256=si,weightDivisorRawBits=weightBits.ToString("x8"),weightDivisorUncombined=true,inputScaleDivisor=binding.InputScaleDivisor!.Name,inputScaleDivisorNotReadByPlanes=true,allOriginalBytesExact=true,perTile=rows,independentInverse="logical K adjacent nibble reconstruction + original five-axis scale iteration; no converter mapping helper",numericCompute="NOT_RUN",modelQuality="NOT_RUN",nativeCompatibility="Requires divisor-aware dispatch; E4M3FN sign/NaN unsupported by old UE decoder are refused, never clipped."};
    }
    public static int Main(string[] argv)
    {
        string? output=null;bool owned=false;var state=new Dictionary<string,object?>{{"schema","ninfer-qwen38-import-format-probe/1"},{"complete",false},{"GPU","NOT_RUN"},{"inference","NOT_IMPLEMENTED"},{"modelQuality","NOT_RUN"},{"performance","NOT_RUN"},{"HF_tokenizer_oracle","NOT_RUN"}};
        try
        {
            var args=Args(argv);if(args.ContainsKey("--help")){Console.WriteLine("PLAN default. --execute --artifact FILE --directory-sha SHA --out FRESH_IGNORED_DIRECTORY [--known-full-sha SUPPLIED_UNVERIFIED_SHA] [--read-calibrations]; --self-test; --self-test-output");return 0;}
            if(args.ContainsKey("--self-test")){Cases(out _);Checks.SelfTest();return 0;}if(args.ContainsKey("--self-test-output")){OutputSelfTest();return 0;}
            if(!args.ContainsKey("--execute")){Console.WriteLine(JsonSerializer.Serialize(new{planOnly=true,complete=false,required=new[]{"--execute","--artifact","--directory-sha","--out"},frontend="PUBLIC_FromReader_ACTUAL_ARTIFACT_ONLY",planes="GDN0out_Identity_40tiles_17694724bytes",optional="--read-calibrations reads247 separate4B scalars",GPU="NOT_RUN",HF_ID_oracle="NOT_RUN",model="NOT_RUN"},Json));return 0;}
            string artifact=Path.GetFullPath(Required(args,"--artifact")),directory=Required(args,"--directory-sha");output=Path.GetFullPath(Required(args,"--out"));FreshOutput(output);
            Checks.Need(!artifact.StartsWith(output+Path.DirectorySeparatorChar,StringComparison.Ordinal),"Output cannot own artifact ancestor.");
            object before=ArtifactStat(artifact);Directory.CreateDirectory(output);owned=true;Save(Path.Combine(output,"result.json"),state);
            state["artifact"]=artifact;state["artifactBefore"]=before;state["knownFullSha256"]=args.GetValueOrDefault("--known-full-sha");state["fullArtifactSha256Recomputed"]=false;state["knownFullSha256Status"]=args.ContainsKey("--known-full-sha")?"SUPPLIED_NOT_VERIFIED":"NOT_SUPPLIED";
            state["execution"]=new{tool=Identity(Assembly.GetExecutingAssembly()),runtime=Identity(typeof(NinferArtifactReader).Assembly),models=Identity(typeof(NinferQwen38TargetPlan).Assembly)};
            using var reader=new NinferArtifactReader(artifact);Checks.Need(reader.DirectorySha256==directory,"Actual directory SHA differs.");state["directorySha256"]=reader.DirectorySha256;state["initialization"]=new{reader.InitializationBytesRead,reader.InitializationReadOperations};
            Reads b=Stats(reader);var plan=NinferQwen38TargetPlan.Create(reader);Checks.Need(Stats(reader)==b&&plan.Tensors.Count==906&&plan.Layers.Count==64,"Actual target plan counts/zero-extra-IO differ.");state["targetPlan"]=new{tensors=906,layers=64,largeBf16Exceptions=9,weightDivisors=247,extraRead=Delta(b,Stats(reader)),inference=false};
            b=Stats(reader);var frontend=NinferQwen38Frontend.FromReader(reader);state["frontendAPI"]="PUBLIC NinferQwen38Frontend.FromReader(actualReader); no reflection/tiny-profile entry";state["frontendReads"]=Delta(b,Stats(reader));Checks.Need(reader.TotalBytesRead-b.Bytes==12837177&&reader.ReadOperations-b.Calls==6,"Actual six-resource read budget differs.");
            state["resources"]=frontend.Resources.Values.OrderBy(x=>x.Name,StringComparer.Ordinal).Select(x=>new{x.Name,x.ByteLength,x.Sha256}).ToArray();state["generationMetadata"]=new{original=frontend.Generation,appliedToSampling=false,templateRendered=false,visionProcessing=false};
            state["tokenizer"]=ExportTokenizer(frontend,output);Save(Path.Combine(output,"result.json"),state);
            b=Stats(reader);state["calibrations"]=args.ContainsKey("--read-calibrations")?ReadCalibrations(reader,plan):new{contents="NOT_READ_NOT_VALIDATED",descriptors=247,combinedWithWeightDivisors=false};state["calibrationReads"]=Delta(b,Stats(reader));
            if(args.ContainsKey("--read-calibrations"))Checks.Need(reader.TotalBytesRead-b.Bytes==988&&reader.ReadOperations-b.Calls==247,"Separate calibration read budget differs.");
            b=Stats(reader);state["planes"]=Planes(reader,plan);state["planeReads"]=Delta(b,Stats(reader));Checks.Need(reader.TotalBytesRead-b.Bytes==35389444&&reader.ReadOperations-b.Calls==161,"Bounded converter plus independent source audit read budget differs.");
            state["totalReads"]=Stats(reader);object after=ArtifactStat(artifact);state["artifactAfter"]=after;Checks.Need(JsonSerializer.Serialize(before)==JsonSerializer.Serialize(after),"Artifact stat/path changed.");state["fileIdentityScope"]="Length/mtime/creation/link only. External full-SHA/inode guard required; this tool does not rehash19GB.";
            state["complete"]=true;Save(Path.Combine(output,"result.json"),state);Console.WriteLine(JsonSerializer.Serialize(new{complete=true,output,formatOnly=true,GPU="NOT_RUN",modelQuality="NOT_RUN",HF_tokenizer_oracle="NOT_RUN"}));return 0;
        }
        catch(Exception e){state["complete"]=false;state["error"]=e.ToString();OwnFailure(output,owned,state);Console.Error.WriteLine(e);return 1;}
    }
}
