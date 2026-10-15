using TensorSharp.GGML;

namespace GlmUnifiedSessionProbe;

internal static class CacheTelemetryPolicy
{
    public static readonly string[] RequestResultKeys = ["Completion", "BudgetAfterCompletion", "PhysicalAfterCompletion",
        "RawText", "Content", "Thinking", "ToolCalls", "ThinkingClosed", "OutputTokens", "PublishedTokens", "ComputedTokens",
        "StatusBeforeDispose", "NaturalEos", "StructuralPassed", "SemanticPassed", "SemanticPolicy", "FinalLogitsFinite",
        "ComputedTokenConvention", "Timing", "Passed", "Status"];

    public static void BeginRound(Dictionary<string, object?> report)
    {
        foreach (string key in RequestResultKeys) report.Remove(key);
        report["Passed"] = false; report["Status"] = "running-round";
    }

    public static int ParseRepetitions(string text) => int.TryParse(text,
        System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int n)
        && n is >= 1 and <= 8 ? n : throw new ArgumentException("repetitions must be an integer from 1 to 8.");

    public static bool ParseBoolean(string text) => text switch
    { "true" => true, "false" => false, _ => throw new ArgumentException("Require true or false.") };

    public static bool AllRoundsPass(int requested, IEnumerable<bool> results)
    {
        if (requested is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(requested));
        bool[] values = results.ToArray();
        return values.Length == requested && values.All(p => p);
    }

    public static object Read() => new
    {
        Usage = GgmlBasicOps.GetHostMoeExpertCacheUsage(),
        Entries = GgmlBasicOps.GetHostMoeExpertCacheEntries(),
        Growth = GgmlBasicOps.GetHostMoeExpertCacheGrowthUsage(),
        Pipeline = GgmlBasicOps.GetHostMoeFilePipelineUsage(),
        CounterScope = "Process-wide cumulative native counters until full cache teardown. Capture exact snapshots; do not subtract across a reset.",
        EntryScope = "Pooled cache entries represent quantization/layout groups. Layer is the first-creator label, not a separate table for every layer. AccountedBytes is logical charged cache payload plus allowance, not whole-board VRAM.",
        FileReadCounterCoverage = "Per-entry file_read_calls/file_read_bytes are not publicly exported. Preserve native stderr teardown markers with inherited TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1. Pipeline.SourceBytes covers only the pipeline path; it is not total expert reads.",
        SelectedSeamCounterCoverage = "The native selected graph/seam counters require an opaque native model handle, which the public model/session does not expose. This public-path probe does not bypass that boundary."
    };

    public static void SelfTest()
    {
        int checks = 0;
        void Check(bool condition) { checks++; if (!condition) throw new Exception("Cache telemetry self-test " + checks + " failed."); }
        Check(ParseRepetitions("1") == 1); Check(ParseRepetitions("8") == 8);
        foreach (string text in new[] { "0", "9", "-1", "1.0", " 2", "NaN", "2147483648" })
        { bool rejected = false; try { ParseRepetitions(text); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        Check(ParseBoolean("true")); Check(!ParseBoolean("false"));
        foreach (string text in new[] { "True", "0", "" })
        { bool rejected = false; try { ParseBoolean(text); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        Check(AllRoundsPass(1, [true])); Check(AllRoundsPass(2, [true, true]));
        Check(!AllRoundsPass(2, [true])); Check(!AllRoundsPass(1, [true, true]));
        Check(!AllRoundsPass(2, [false, true])); Check(!AllRoundsPass(2, [true, false]));
        Check(!AllRoundsPass(1, [])); Check(AllRoundsPass(8, Enumerable.Repeat(true, 8)));
        Dictionary<string, object?> prior = new() { ["Completion"] = "prior-eos", ["NaturalEos"] = true,
            ["StructuralPassed"] = true, ["Passed"] = true, ["Status"] = "completed", ["ModelIdentityLinked"] = true };
        BeginRound(prior);
        Check(!prior.ContainsKey("Completion") && !prior.ContainsKey("NaturalEos") && !prior.ContainsKey("StructuralPassed"));
        Check(prior["Passed"] is false && prior["Status"] is "running-round");
        Check(prior["ModelIdentityLinked"] is true);
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { Passed = true, Checks = checks,
            Scope = "Model-free telemetry option policy; no cache/native/model/device invocation." }));
    }
}
