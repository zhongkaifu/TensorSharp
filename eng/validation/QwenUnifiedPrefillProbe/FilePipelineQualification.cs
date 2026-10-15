using TensorSharp.GGML;

namespace QwenUnifiedPrefillProbe;

/// <summary>Pure checks over the public process-cumulative counters. This does
/// not invoke native code or infer transfer/compute overlap from counters.</summary>
internal static class FilePipelineQualification
{
    internal static bool ParseEnabled(string value = "false") => value switch
    {
        "true" => true,
        "false" => false,
        _ => throw new ArgumentException("async-prefill true|false")
    };

    internal static bool Valid(GgmlHostMoeFilePipelineUsage usage)
        => usage.Calls >= 0 && usage.SourceBytes >= 0 && usage.RetainedRingBytes >= 0;

    internal static bool Monotonic(GgmlHostMoeFilePipelineUsage before, GgmlHostMoeFilePipelineUsage after)
        => Valid(before) && Valid(after) && after.Calls >= before.Calls && after.SourceBytes >= before.SourceBytes;

    internal static bool HasActualUse(GgmlHostMoeFilePipelineUsage before, GgmlHostMoeFilePipelineUsage after)
        => Monotonic(before, after) && after.Calls > before.Calls && after.SourceBytes > before.SourceBytes;

    internal static bool Qualifies(bool enabled, GgmlHostMoeFilePipelineUsage? before,
        GgmlHostMoeFilePipelineUsage? after, GgmlHostMoeFilePipelineUsage? final)
        => before is { } start && after is { } end && final is { } clean
            && Monotonic(start, end) && Monotonic(end, clean) && clean.RetainedRingBytes == 0
            && clean.Calls == end.Calls && clean.SourceBytes == end.SourceBytes
            && (enabled ? HasActualUse(start, end) : end.Calls == start.Calls && end.SourceBytes == start.SourceBytes);

    internal static void SelfTest()
    {
        int checks = 0;
        void Check(bool ok, string name) { checks++; if (!ok) throw new InvalidOperationException(name); }
        Check(!ParseEnabled() && !ParseEnabled("false") && ParseEnabled("true"), "Default false and explicit async boolean");
        foreach (string invalid in new[] { "TRUE", "False", "1", "0", "yes", "", " true", "false " })
        {
            bool refused = false;
            try { ParseEnabled(invalid); } catch (ArgumentException) { refused = true; }
            Check(refused, "Invalid async boolean refused: " + invalid);
        }
        var start = new GgmlHostMoeFilePipelineUsage(7, 100, 0);
        var end = new GgmlHostMoeFilePipelineUsage(8, 200, 128);
        var clean = end with { RetainedRingBytes = 0 };
        Check(Qualifies(true, start, end, clean), "Both cumulative deltas prove actual path use");
        Check(!Qualifies(true, start, start, start), "Enabling an unused path cannot qualify");
        Check(!Qualifies(true, start, end with { Calls = start.Calls }, clean), "Bytes alone cannot qualify");
        Check(!Qualifies(true, start, end with { SourceBytes = start.SourceBytes }, clean), "Calls alone cannot qualify");
        Check(!Qualifies(true, start, end, end), "Retained ring must be gone after cleanup");
        Check(!Qualifies(true, start, end, clean with { Calls = 6 }), "Counters cannot regress during cleanup");
        Check(!Qualifies(true, start, end, clean with { Calls = 9 }), "No unobserved calls during quiescent cleanup");
        Check(!Qualifies(true, start, end, clean with { SourceBytes = 201 }), "No unobserved source payload during quiescent cleanup");
        Check(!Qualifies(true, start with { SourceBytes = -1 }, end, clean), "Negative native counter rejected");
        Check(!Qualifies(true, null, end, clean) && !Qualifies(true, start, null, clean)
            && !Qualifies(true, start, end, null), "Unavailable stages never become zero or pass");
        Check(Qualifies(false, start, start, start), "Disabled baseline needs no actual pipeline calls");
        Check(!Qualifies(false, start, end, clean), "Disabled option must not silently run the pipeline");
        Check(!Qualifies(false, start, start, null), "Disabled baseline still requires final telemetry");
        Check(HasActualUse(new(long.MaxValue - 1, long.MaxValue - 1, 0), new(long.MaxValue, long.MaxValue, 0)),
            "Large counters checked without overflowing arithmetic");
        Console.WriteLine($"PASS async option and public file-pipeline counter qualification: {checks} checks; no model/native execution.");
    }
}
