// Standalone CPU investigation; never integrated into production dispatch.
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

string Required(string option) { int i=Array.IndexOf(args,option);return i>=0&&i+1<args.Length?args[i+1]:throw new ArgumentException($"Missing {option}"); }
int Number(string option,int fallback) { int i=Array.IndexOf(args,option);return i>=0?int.Parse(args[i+1]):fallback; }
string tokenizerPath=Required("--tokenizer"), schemaPath=Required("--schema"), output=Required("--out");
int rounds=Number("--rounds",5), iterations=Number("--iterations",30);
if(rounds<1||iterations<1)throw new ArgumentException("Positive rounds/iterations required");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
var result=new Dictionary<string,object?> { ["schema"]="qwen38-grammar-argmax-bench/1",["complete"]=false,
    ["scope"]="CPU standalone mask/sampler only; not production integration, VM EPYC, model quality or E2E",
    ["GPU"]="NOT_RUN",["network"]="NOT_RUN",["VM"]="NOT_ACCESSED" };
void Save()=>File.WriteAllText(output,JsonSerializer.Serialize(result,new JsonSerializerOptions {WriteIndented=true}));
Save();
try
{
    long stamp=Stopwatch.GetTimestamp();
    BpeTokenizer tokenizer=LoadTokenizer(tokenizerPath);
    result["tokenizer_load_ms"]=Stopwatch.GetElapsedTime(stamp).TotalMilliseconds;
    stamp=Stopwatch.GetTimestamp();
    string schemaText=File.ReadAllText(schemaPath);
    var cache=GrammarLibrary.ForJsonSchema(schemaText,tokenizer);
    result["first_compile_and_vocabulary_ms"]=Stopwatch.GetElapsedTime(stamp).TotalMilliseconds;
    stamp=Stopwatch.GetTimestamp();
    for(int i=0;i<100;i++)GC.KeepAlive(GrammarLibrary.ForJsonSchema(schemaText,tokenizer));
    result["warm_same_schema_compile_cache_lookup_mean_us"]=Stopwatch.GetElapsedTime(stamp).TotalMicroseconds/100;
    result["framework"]=RuntimeInformation.FrameworkDescription;result["os"]=RuntimeInformation.OSDescription;
    result["architecture"]=RuntimeInformation.ProcessArchitecture.ToString();
    result["SIMD"]=new {Vector.IsHardwareAccelerated,float_lanes=Vector<float>.Count,Vector128=Vector128.IsHardwareAccelerated,
        Vector256=Vector256.IsHardwareAccelerated,Vector512=Vector512.IsHardwareAccelerated,Avx2=Avx2.IsSupported,AdvSimd=AdvSimd.IsSupported};
    result["runtime_dll_sha256"]=Hash(typeof(TokenSampler).Assembly.Location);
    result["probe_dll_sha256"]=Hash(Assembly.GetExecutingAssembly().Location);
    result["tokenizer_sha256"]=Hash(tokenizerPath);result["schema_sha256"]=Hash(schemaPath);
    result["tokenizer_vocab_size"]=tokenizer.VocabSize;result["tokenizer_eos_ids"]=tokenizer.EosTokenIds;
    result["configuration"]="temp0/repeat1/presence0/frequency0; active nondead grammar; no forced-thinking, first-token allowlist or penalty path";
    result["tiered_compilation"]=Environment.GetEnvironmentVariable("DOTNET_TieredCompilation");
    result["rounds"]=rounds;result["iterations"]=iterations;
    var prefixes=new (string Name,string Prefix)[] {
        ("object_start",""), ("customer_string","{\"customer\":\""),
        ("product_string","{\"customer\":\"Maria Lopez\",\"items\":[{\"product\":\""),
        ("quantity_integer","{\"customer\":\"Maria Lopez\",\"items\":[{\"product\":\"book\",\"quantity\":"),
        ("due_date_digits","{\"customer\":\"Maria\",\"items\":[{\"product\":\"book\",\"quantity\":3,\"unit_price\":4.50}],\"due_date\":\"2026-"),
        ("complete_allow_eos","{\"customer\":\"Maria\",\"items\":[{\"product\":\"book\",\"quantity\":3,\"unit_price\":4.50}],\"due_date\":\"2026-11-05\"}") };
    var fixtures=new List<Fixture>();var cold=new List<object>();
    foreach(var (name,prefix) in prefixes)
    {
        var grammar=GrammarLibrary.NewConstraint(cache,tokenizer);grammar.AcceptBytes(Encoding.UTF8.GetBytes(prefix));
        if(grammar.IsDead)throw new InvalidOperationException("Invalid real grammar prefix "+name);
        stamp=Stopwatch.GetTimestamp();ulong[] mask=grammar.CurrentMask();
        cold.Add(new {name,first_mask_ms=Stopwatch.GetElapsedTime(stamp).TotalMilliseconds,
            allowed_tokens=mask.Sum(x=>(long)BitOperations.PopCount(x)),mask_words=mask.Length,complete=grammar.IsComplete});
        fixtures.Add(new Fixture(name,tokenizer,grammar,tokenizer.VocabSize));
    }
    result["cold_mask_states"]=cold;
    // Small/ragged/padded and unusual EOS sets exercise the actual Runtime
    // sampler, not a copied oracle. Raw prefixes are legal forced continuations.
    foreach(var (name,prefix) in prefixes)
    {
        var grammar=GrammarLibrary.NewConstraint(cache,tokenizer);grammar.AcceptBytes(Encoding.UTF8.GetBytes(prefix));
        fixtures.Add(new Fixture(name+"-padded256",tokenizer,grammar,(tokenizer.VocabSize+255)/256*256));
    }
    foreach(int v in new[]{1,2,7,31,63,64,65,127,129})
    foreach(bool complete in new[]{false,true})
    foreach(int length in (v>3?new[]{v-3,v,v+3}:new[]{v,v+3}))
    {
        var tok=new TinyTokenizer(v,new[]{0,v-1,v+2,-1,0,1,2,3,4,5});
        var grammar=new GrammarConstraint(Grammar.JsonObject(),tok);
        grammar.AcceptBytes(Encoding.UTF8.GetBytes(complete?"{\"a\":\"x\"}":"{\"a\":\""));
        fixtures.Add(new Fixture($"tiny{v}-complete{complete}-length{length}",tok,grammar,length));
    }
    int gates=0;
    foreach(Fixture f in fixtures)for(int pattern=0;pattern<9;pattern++)
    {
        float[] source=Scores(f.Length,pattern),a=(float[])source.Clone(),b=(float[])source.Clone(),c=(float[])source.Clone();
        int expected=f.Baseline(a),fused=f.Fused(b),vector=f.Vector(c);
        if(expected!=fused||expected!=vector||!SameBits(a,b)||!SameBits(a,c))
            throw new InvalidOperationException($"Exact token/full modified-logits parity failed {f.Name} pattern{pattern}: {expected}/{fused}/{vector}");
        gates++;
    }
    result["strict_gate"]=new {status="PASS",cases=gates,full_post_mask_float_bits=true,token_ids=true,
        coverage="finite/all-inf/ties/±0/leading-later-all NaNs incl signaling bits/±Inf/all raw float bits; ragged vocab and unmasked logits tail; duplicate/negative/out-of-range/>8 EOS"};
    var timings=new List<object>();
    var scalarArgmax=(Func<float[],int>)typeof(TokenSampler).GetMethod("Argmax",BindingFlags.NonPublic|BindingFlags.Static)!.CreateDelegate(typeof(Func<float[],int>));
    foreach(Fixture f in fixtures.Take(prefixes.Length))
    {
        float[] source=Scores(f.Length,0),work=new float[f.Length];
        Func<float[],int>[] actions={f.Baseline,f.Fused,f.Vector};
        for(int i=0;i<30;i++)foreach(var action in actions){source.CopyTo(work,0);action(work);}
        var samples=new List<object>();
        foreach(int candidate in new[]{1,2})for(int round=0;round<rounds;round++)
        {
            double a=Measure(actions[0],source,work,iterations),b=Measure(actions[candidate],source,work,iterations);
            double b2=Measure(actions[candidate],source,work,iterations),a2=Measure(actions[0],source,work,iterations);
            samples.Add(new {candidate=candidate==1?"scalar_fused":"original_mask_plus_safe_SIMD_argmax",round,
                baseline_A1_us=a,candidate_B1_us=b,candidate_B2_us=b2,baseline_A2_us=a2});
        }
        var phases=new List<object>();float[] masked=(float[])source.Clone();f.Grammar.ApplyMask(masked,f.Grammar.IsComplete);
        for(int round=0;round<rounds;round++)
            phases.Add(new {round,mask_only_us=Measure(x=>{f.Grammar.ApplyMask(x,f.Grammar.IsComplete);return 0;},source,work,iterations),
                actual_scalar_argmax_only_us=Measure(scalarArgmax,masked,work,iterations),safe_SIMD_argmax_only_us=Measure(Prototypes.VectorArgmax,masked,work,iterations)});
        timings.Add(new {name=f.Name,logits=f.Length,mask_words=f.Mask.Length,
            allowed_tokens=f.Mask.Sum(x=>(long)BitOperations.PopCount(x)),allow_eos=f.Grammar.IsComplete,samples,phases});
    }
    result["timings"]=timings;result["complete"]=true;
    result["limitations"]=new[]{"Local CPU architecture only; no inference/EPYC VM/GPU/E2E result.",
        "Real HF tokenizer.json vocabulary, not asserted identical to the model GGUF tokenizer/padded logits; EOS defaults to im_end and is recorded.",
        "Standalone prototypes apply only the eligible temp0/no-penalty/no-forced-override path. They are NOT replacements for full Sample semantics.",
        "Every timing iteration resets identical input outside the timed window, which warms cache. Timed operations include the current grammar mask lookup in all variants.",
        "Real masks are warm, immutable and share the actual Runtime grammar cache; grammar build/cold masks are reported separately.",
        "Fused prototype precomputes immutable per-tokenizer EOS word masks; production would need a lifetime/identity review.",
        "ABBA samples are descriptive within one local process, no confidence interval or portable speed claim. No production source changed."};
    Save();Console.WriteLine($"PASS {gates} full-bit/token gates; local CPU-only report {output}");
    return 0;
}
catch(Exception error){result["error"]=error.ToString();Save();Console.Error.WriteLine(error);return 1;}

static string Hash(string p){using var s=File.OpenRead(p);return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant();}
static bool SameBits(float[] a,float[] b)=>MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan()));
static float[] Scores(int n,int pattern)
{
    var x=new float[n];uint h=0x12345678;
    for(int i=0;i<n;i++){h^=h<<13;h^=h>>17;h^=h<<5;x[i]=pattern==8?BitConverter.Int32BitsToSingle((int)h):((int)(h%100003)-50001)/1024f;}
    if(pattern==1)Array.Fill(x,float.NegativeInfinity);
    if(pattern==2){Array.Fill(x,0f);x[0]=BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));}
    if(pattern==3){Array.Fill(x,-1f);x[0]=x[n/2]=x[^1]=3f;}
    if(pattern==4)x[0]=BitConverter.Int32BitsToSingle(0x7fc01234);
    if(pattern==5){x[Math.Min(1,n-1)]=BitConverter.Int32BitsToSingle(0x7f800001);x[^1]=BitConverter.Int32BitsToSingle(unchecked((int)0xffc12345));}
    if(pattern==6)for(int i=0;i<n;i++)x[i]=BitConverter.Int32BitsToSingle(unchecked((int)(0x7f800001u+(uint)(i&0x3fffff))));
    if(pattern==7){x[0]=float.NegativeInfinity;x[n/2]=x[^1]=float.PositiveInfinity;}
    return x;
}
static double Measure(Func<float[],int> action,float[] source,float[] work,int count)
{
    long ticks=0;int checksum=0;
    for(int i=0;i<count;i++){source.CopyTo(work,0);long t=Stopwatch.GetTimestamp();checksum^=action(work);ticks+=Stopwatch.GetTimestamp()-t;}
    GC.KeepAlive(checksum);return ticks*1_000_000.0/Stopwatch.Frequency/count;
}
static BpeTokenizer LoadTokenizer(string path)
{
    using var doc=JsonDocument.Parse(File.ReadAllBytes(path));var root=doc.RootElement;var model=root.GetProperty("model");
    var vocabulary=new Dictionary<int,string>();foreach(var p in model.GetProperty("vocab").EnumerateObject())vocabulary[p.Value.GetInt32()]=p.Name;
    var special=new HashSet<int>();foreach(var p in root.GetProperty("added_tokens").EnumerateArray())
    {int id=p.GetProperty("id").GetInt32();vocabulary[id]=p.GetProperty("content").GetString()!;if(p.GetProperty("special").GetBoolean())special.Add(id);}
    int n=vocabulary.Keys.Max()+1;if(vocabulary.Count!=n)throw new InvalidOperationException("Sparse HF vocabulary: no fabricated missing token bytes permitted");
    string[] vocab=Enumerable.Range(0,n).Select(i=>vocabulary[i]).ToArray();int[] types=Enumerable.Range(0,n).Select(i=>special.Contains(i)?3:1).ToArray();
    string[] merges=model.GetProperty("merges").EnumerateArray().Select(p=>p.ValueKind==JsonValueKind.Array?string.Join(" ",p.EnumerateArray().Select(v=>v.GetString())):p.GetString()!).ToArray();
    int eos=Array.IndexOf(vocab,"<|im_end|>");if(eos<0)throw new InvalidOperationException("im_end EOS absent; specify a reviewed tokenizer loader");
    return new BpeTokenizer(vocab,types,merges,-1,[eos],false,false,"qwen35");
}

sealed class Fixture
{
    public readonly string Name;public readonly int Length;public readonly GrammarConstraint Grammar;public readonly ulong[] Mask;
    readonly ITokenizer tokenizer;readonly TokenSampler sampler;readonly ulong[] eosMasks;
    public Fixture(string name,ITokenizer tok,GrammarConstraint grammar,int length)
    {
        Name=name;tokenizer=tok;Grammar=grammar;Length=length;Mask=grammar.CurrentMask();
        var config=SamplingConfig.Greedy;config.Grammar=grammar;sampler=new TokenSampler(config);eosMasks=new ulong[Mask.Length];
        foreach(int id in tok.EosTokenIds)if((uint)id<(uint)Math.Min(tok.VocabSize,length)&&id/64<Mask.Length)eosMasks[id>>6]|=1UL<<(id&63);
        if(grammar.IsDead)throw new InvalidOperationException("Dead grammar excluded from fused eligible path");
    }
    public int Baseline(float[] x)=>sampler.Sample(x);
    public int Vector(float[] x){Grammar.ApplyMask(x,Grammar.IsComplete);return Prototypes.VectorArgmax(x);}
    public int Fused(float[] x)=>Prototypes.Fused(x,Grammar.CurrentMask(),tokenizer.VocabSize,Grammar.IsComplete?eosMasks:null);
}

static class Prototypes
{
    static readonly Vector<int> LaneIds=new(Enumerable.Range(0,Vector<float>.Count).ToArray());
    public static int Fused(float[] values,ulong[] mask,int vocab,ulong[]? eos)
    {
        int best=0,index=0,covered=Math.Min(Math.Min(values.Length,vocab),mask.Length*64);float bestValue=0f;
        for(int word=0;index<covered;word++)
        {
            ulong bits=mask[word]|(eos is null?0:eos[word]);int end=Math.Min(index+64,covered);
            for(int b=0;index<end;index++,b++)
            {
                float value=values[index];if((bits&(1UL<<b))==0){value=float.NegativeInfinity;values[index]=value;}
                if(index==0)bestValue=value;else if(value>bestValue){bestValue=value;best=index;}
            }
        }
        for(;index<values.Length;index++){float value=values[index];if(index==0)bestValue=value;else if(value>bestValue){bestValue=value;best=index;}}
        return best;
    }
    public static int VectorArgmax(float[] values)
    {
        // The real mask has already run: leading NaN can return0 without
        // skipping the required in-place side effects.
        if(float.IsNaN(values[0]))return 0;
        int width=Vector<float>.Count,i=0,best=0;float bestValue=values[0];
        var max=new Vector<float>(bestValue);var indices=Vector<int>.Zero;
        for(;i+width<=values.Length;i+=width)
        {
            var x=new Vector<float>(values,i);var greater=Vector.GreaterThan(x,max);
            max=Vector.ConditionalSelect(greater,x,max);
            indices=Vector.ConditionalSelect(greater,LaneIds+new Vector<int>(i),indices);
        }
        for(int lane=0;lane<width;lane++)if(max[lane]>bestValue||(max[lane]==bestValue&&indices[lane]<best))
        {bestValue=max[lane];best=indices[lane];}
        for(;i<values.Length;i++)if(values[i]>bestValue){bestValue=values[i];best=i;}
        return best;
    }
}

sealed class TinyTokenizer:ITokenizer
{
    public string[] Vocab {get;}public int BosTokenId=>-1;public int[] EosTokenIds {get;}public int VocabSize=>Vocab.Length;
    public TinyTokenizer(int size,int[] eos){Vocab=Enumerable.Range(0,size).Select(i=>((char)('a'+i%26)).ToString()).ToArray();EosTokenIds=eos;}
    public List<int> Encode(string text,bool addSpecial=true)=>throw new NotSupportedException();
    public string Decode(List<int> ids)=>string.Concat(ids.Select(i=>Vocab[i]));
    public void AppendTokenBytes(int tokenId,List<byte> buffer)=>buffer.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
    public bool IsEos(int id)=>EosTokenIds.Contains(id);public int LookupToken(string value)=>Array.IndexOf(Vocab,value);
}
