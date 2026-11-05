// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Validation;
using static Support;

internal static class Protocol
{
    internal const string OldMain = "4242d3e8891c6b8fec2fcd216069c3ce40698cd63416f96f6bb084614324d76b";
    internal const string NewMain = "ea0e322a36c48211750e0f569bd12c44f9d62caa6d3249c37f1547679bdccbea";
    internal const int Context = 40960, Ubatch = 2048, HeadRows = 326, TapBytes = 8294400;
    internal const long RawBound = 332103680;
    internal static readonly int[] Bases = { 127, 255, 1023 }, Keeps = { 0, 1, 8 };
    internal sealed record Seed(string Id, int[] Prompt, int[] Window, int Anchor, int NextToken, int CloneToken, int NextAnchor);
    internal sealed record NativePin(string Role, string Path, long Bytes, string Sha256);
    internal static (int[] Prompt, int[] Continuation) Inputs(byte[] bytes)
    {
        using var doc = JsonDocument.Parse(bytes); Unique(doc.RootElement);
        Need(doc.RootElement.GetProperty("schema").GetString() == "ninfer-qwen38-body-inputs/1", "Original body input schema required.");
        var pools = new List<(int[] Prompt, int[] Continuation)>();
        foreach (var row in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int[] p = row.GetProperty("promptTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int[] c = row.GetProperty("continuationTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Need(p.Length is > 0 and <= 16384 && c.Length is >= 8 and <= 128 && p.Concat(c).All(t => t >= 0 && t < Valid), "Bounded valid original tokens required."); pools.Add((p, c));
        }
        Need(pools.Count is > 0 and <= 128, "Bounded input cases required."); var longest = pools.OrderByDescending(x => x.Prompt.Length).First();
        Need(longest.Prompt.Length >= 4096, "Original4096-token pool required."); return longest;
    }
    internal static void Unique(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.Object) { var names = new HashSet<string>(); foreach (var p in e.EnumerateObject()) { Need(names.Add(p.Name), "Duplicate JSON field."); Unique(p.Value); } }
        else if (e.ValueKind == JsonValueKind.Array) foreach (var p in e.EnumerateArray()) Unique(p);
    }
    internal static Seed[] Seeds(int[] p, int[] c) => Bases.Select((b, i) => new Seed("base" + b,
        Enumerable.Range(0, b).Select(j => p[(j + 17*i) % p.Length]).ToArray(),
        Enumerable.Range(0, 8).Select(j => c[(i+j) % c.Length]).ToArray(), p[(111+i*13)%p.Length], p[(211+i*13)%p.Length],
        p[(311+i*13)%p.Length], p[(411+i*13)%p.Length])).ToArray();
    internal static NativePin[] Pins(byte[] bytes, string mode)
    {
        using var doc = JsonDocument.Parse(bytes); Unique(doc.RootElement);
        Need(doc.RootElement.GetProperty("schema").GetString() == "ninfer-residual-body-native-pins/1", "Native pin schema differs.");
        var pins = doc.RootElement.GetProperty("files").EnumerateArray().Select(x => new NativePin(x.GetProperty("role").GetString()!,
            x.GetProperty("path").GetString()!, x.GetProperty("bytes").GetInt64(), ValidationSha256.Normalize(x.GetProperty("sha256").GetString()!))).ToArray();
        string[] roles = { "main", "helper", "attention", "a16", "draft", "repack", "w8" };
        Need(pins.Length == roles.Length && pins.Select(p => p.Role).ToHashSet().SetEquals(roles), "Exactly eight distinct actual loaded native roles required.");
        foreach (var p in pins) Need(Path.IsPathFullyQualified(p.Path) && p.Path == Path.GetFullPath(p.Path) && p.Bytes > 0 && FilePin(p.Path) == new Pin(p.Path,p.Bytes,p.Sha256), "Actual native bytes differ: " + p.Role);
        Need(pins.Single(p => p.Role == "main").Sha256 == (mode == "old" ? OldMain : NewMain), "Wrong actual baseline/candidate main image."); return pins;
    }
    internal static Dictionary<string,string?> EnvironmentSnapshot() => Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(x => ((string)x.Key).StartsWith("TS_Q35_", StringComparison.Ordinal) || (string)x.Key is "CUDA_VISIBLE_DEVICES" or "CUDA_CACHE_DISABLE")
        .OrderBy(x => (string)x.Key, StringComparer.Ordinal).ToDictionary(x => (string)x.Key,x => (string?)x.Value,StringComparer.Ordinal);
    internal static Dictionary<string,string?> NumericalEnvironment(Dictionary<string,string?> env, NativePin[] pins)
    {
        var result = new Dictionary<string,string?>(env, StringComparer.Ordinal); result.Remove("TS_Q35_NINFER_RESIDUAL_NORM");
        foreach (var p in env.Where(p => p.Key.EndsWith("_PTX",StringComparison.Ordinal) && p.Value != null))
        {
            string sha = FilePin(p.Value!).Sha256;
            Need(pins.Any(x => x.Sha256 == sha) || p.Value!.EndsWith(".ptx",StringComparison.OrdinalIgnoreCase) &&
                File.Exists(Path.ChangeExtension(p.Value,".fatbin")) && pins.Any(x => x.Sha256 == FilePin(Path.ChangeExtension(p.Value,".fatbin")).Sha256), "Configured native path not bound by actual pins.");
            result[p.Key] = "native-role:" + pins.First(x => x.Sha256 == sha || p.Value!.EndsWith(".ptx",StringComparison.OrdinalIgnoreCase) && File.Exists(Path.ChangeExtension(p.Value,".fatbin")) && x.Sha256 == FilePin(Path.ChangeExtension(p.Value,".fatbin")).Sha256).Role;
        }
        return result;
    }
    internal static object Loaded(Q35CudaEngine e, Qwen35CudaExecutor executor, NativePin[] pins, string mode)
    {
        Need(e.IsNinferModel && e.VocabSize == Vocab && e.HasNinferDrafter && e.HasDrafter && executor.Draft == null && e.DraftBlockSize == 8
            && e.DraftRingRows == 2048 && e.NinferDraftHeadRows == Vocab && !e.NinferDraftUsesShortlist && e.NinferDecodeGraphsEnabled
            && e.NinferVerify8GemmEnabled && e.OutputW8TargetEnabled
            && !e.NinferFlashInferEnabled && !e.NinferExactVerifyEnabled && !e.NinferGdnChunkEnabled && !e.NinferPrefillAttentionEnabled, "Qualified existing native full-head policies required.");
        string Image(string role, object? actual) { string s = ValidationSha256.Normalize(actual as string ?? ""); Need(s == pins.Single(x => x.Role == role).Sha256, "Loaded image differs: " + role); return s; }
        var k = Member<object>(e,"_k"); var attention = Member<object>(e,"_ninferAttention");
        var loaded = new Dictionary<string,string> {
            ["main"] = Image("main",Member<string>(k,"ImageSha256")), ["helper"] = Image("helper",Member<string>(Member<object>(e,"_ninfer"),"ImageSha256")),
            ["attention"] = Image("attention",Member<string>(attention,"ImageSha256")),
            ["a16"] = Image("a16",e.NinferA16ImageSha256), ["draft"] = Image("draft",e.NinferDraftModuleSha256), ["repack"] = Image("repack",e.NinferLoadRepackImageSha256),
            ["w8"] = Image("w8",Member<string>(e.OutputW8Info,"ModuleImageSha256")) };
        var counters = State.Counters(e); Need(mode == "old" ? counters.Enabled == null : counters.Enabled == (mode == "on"), "New residual mode/old compatibility differs.");
        if (mode == "on") Need(ValidationSha256.Matches(Optional(e,"NinferResidualNormImageSha256") as string ?? "",NewMain), "Enabled residual image differs.");
        return loaded;
    }
    internal static string[] Cases() => Bases.SelectMany(b => Keeps.Select(k => "base"+b+"/keep"+k)).Append("base255/pending1").ToArray();
    internal static HashSet<string> VectorKeys()
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        void Add(string id,int pass,int rows,bool taps=false)
        { for(int r=0;r<rows;r++)Need(result.Add(id+"/"+pass+"/"+r+"/f32"),"Duplicate vector plan.");if(taps)Need(result.Add(id+"/native-taps/"+pass+"/0/bf16"),"Duplicate tap plan."); }
        for(int pass=0;pass<2;pass++)foreach(string id in Cases())
        {
            Add(id+"/verify8",pass,8,true);Add(id+"/decode",pass,1);Add(id+"/proposal",pass,7);
            if(id.EndsWith("pending1",StringComparison.Ordinal))Add(id+"/pending-anchor",pass,1,true);
            if(id=="base255/keep8"){Add(id+"/clone-source",pass,1);Add(id+"/clone-copy",pass,1);}
        }
        return result;
    }
    internal static HashSet<string> StateKeys()
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        for(int pass=0;pass<2;pass++)foreach(string id in Cases())
        {
            foreach(string phase in new[]{"initial","committed","decoded","proposed"})result.Add(id+"/"+pass+"/"+phase);
            if(id.EndsWith("pending1",StringComparison.Ordinal))result.Add(id+"/"+pass+"/pending");
            if(id=="base255/keep8"){result.Add(id+"/"+pass+"/clone-source");result.Add(id+"/"+pass+"/clone-copy");}
        }
        return result;
    }
    internal static object SelfTest()
    {
        var seeds = Seeds(Enumerable.Range(0,4096).Select(i => i%257).ToArray(),Enumerable.Range(17,16).ToArray());
        Need(seeds.Length == 3 && seeds.Select(s => s.Prompt.Length).SequenceEqual(Bases) && seeds.All(s => s.Window.Length == 8), "Fixed protocol shape differs.");
        long rows = 2*(3*3*16+17+2); Need(rows == HeadRows && rows*Vocab*4+TapBytes == RawBound && RawBound < Budget, "Raw bound differs.");
        Need(2*(3*3+1)==20 && 2*3*3+2==20 && TapBytes == (20*8+2)*51200 && VectorKeys().Count==348 && StateKeys().Count==86, "Operation/tap/key count differs.");
        var p = Poison(1,0); for(int i=0;i<Vocab;i++)p[i]=0; int[] a=Enumerable.Repeat(IntGuard,1+Guard).ToArray();a[0]=0;Head(p,1,a);
        bool rejected=false; a[0]=1;try{Head(p,1,a);}catch(InvalidDataException){rejected=true;}Need(rejected,"CPU argmax gate not enforced.");
        Need(ValidationSha256.Matches(OldMain,OldMain.ToUpperInvariant()) && !ValidationSha256.Matches(OldMain,NewMain),"Image identity gate differs.");
        return new { CPU="PASS", fixedBases=Bases, keeps=Keeps, resetPasses=2, verify8Calls=20,verify1Calls=2,commitCalls=22,decode1Calls=24,proposalCalls=20,
            fullHeadRows=HeadRows,nativeTapCaptures=22,rawByteBound=RawBound,rawBudget=Budget,KV="only live committed prefix; explicit slice offsets; future keys and RingPos excluded",GPU="NOT_RUN",performance="NOT_MEASURED",quality="NOT_RUN" };
    }
}
