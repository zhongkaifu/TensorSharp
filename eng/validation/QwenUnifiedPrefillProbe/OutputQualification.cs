using System.Text.Json;
using TensorSharp.Runtime.Scheduling;

namespace QwenUnifiedPrefillProbe;

/// <summary>Independent, model-free checks over unmodified production output.
/// JSON whitespace is an explicit alternative policy, never a text rewrite.</summary>
internal static class OutputQualification
{
    internal static bool IsExact42(string? content) => string.Equals(content, "42", StringComparison.Ordinal);

    internal static bool IsJsonInteger42(string? content)
    {
        if (content == null) return false;
        try
        {
            using var json = JsonDocument.Parse(content);
            // GetRawText identifies the JSON token; the original Content is
            // retained separately, including every leading/trailing character.
            // Reject strings, decimal/exponent notation and every other integer.
            return json.RootElement.ValueKind == JsonValueKind.Number
                && json.RootElement.GetRawText() == "42";
        }
        catch (JsonException) { return false; }
    }

    internal static bool MatchesPolicy(string policy, string? content) => policy switch
    {
        "exact" => IsExact42(content),
        "json-integer" => IsJsonInteger42(content),
        _ => throw new ArgumentException("answer-policy exact|json-integer")
    };

    internal static bool HasTerminalEos(IReadOnlyList<int> output, IReadOnlyCollection<int> eosIds)
        => output.Count > 0 && output.All(t => t >= 0) && eosIds.Contains(output[^1])
            && !output.Take(output.Count - 1).Any(eosIds.Contains);

    internal static bool IsNaturalFusedCompletion(InferenceCompletion completion, int promptTokens,
        IReadOnlyList<int> output, IReadOnlyList<int> published, int computedTokens,
        int maxNewTokens, bool sequenceHasError, IReadOnlyCollection<int> eosIds)
        // InferenceEngine.NotifyStop publishes the internal reason "eos" and
        // InferenceRequestHandle copies it verbatim; Server API "stop" is a
        // separate vocabulary. Plain fused AR forwards every output, even EOS.
        => completion.Status == SequenceStatus.FinishedStopped && completion.FinishReason == "eos"
            && !sequenceHasError && HasTerminalEos(output, eosIds)
            && completion.OutputTokenCount == output.Count && output.Count <= maxNewTokens
            && completion.PromptTokenCount == promptTokens
            && (long)computedTokens == (long)promptTokens + output.Count
            && published.SequenceEqual(output.Take(output.Count - 1))
            && completion.PrefixCacheReusedTokens == 0;

    internal static void SelfTest()
    {
        int checks = 0;
        void Check(bool ok, string name) { checks++; if (!ok) throw new InvalidOperationException(name); }
        Check(IsExact42("42"), "Exact integer");
        Check(!IsExact42("\n\n42"), "Exact policy must preserve strict formatting failure");
        foreach (string value in new[] { "42", "\n\n42", " \t\r\n42\r\n\t " })
            Check(IsJsonInteger42(value), "Valid JSON integer with formal JSON whitespace");
        foreach (string? value in new[] { "42 is the answer", "answer42", "42garbage", "42 0", "4 2",
            "４２", "٤٢", "43", "042", "+42", "42.0", "4.2e1", "[42]", "\"42\"", "{\"answer\":42}",
            "\u00a042", "42\v", "", null })
            Check(!IsJsonInteger42(value), "Reject noninteger or non-JSON output: " + value);
        Check(!MatchesPolicy("exact", "\n\n42") && MatchesPolicy("json-integer", "\n\n42"), "Policy remains explicit");
        bool invalidPolicy = false;
        try { MatchesPolicy("auto", "42"); } catch (ArgumentException) { invalidPolicy = true; }
        Check(invalidPolicy, "Unknown answer policy refused");

        int[] eos = [99], two = [42, 99], four = [271, 19, 17, 99];
        InferenceCompletion Completion(int count, string reason = "eos", SequenceStatus status = SequenceStatus.FinishedStopped,
            int prompt = 3, int reused = 0) => new() { Status = status, FinishReason = reason,
                PromptTokenCount = prompt, OutputTokenCount = count, PrefixCacheReusedTokens = reused };
        bool Valid(InferenceCompletion c, int[] output, int[] published, int computed, int max = 8, bool error = false)
            => IsNaturalFusedCompletion(c, 3, output, published, computed, max, error, eos);
        Check(Valid(Completion(2), two, [42], 5), "Two-output target+EOS completion");
        Check(Valid(Completion(4), four, [271, 19, 17], 7), "Four-output target+EOS completion");
        Check(!Valid(Completion(2, "stop"), two, [42], 5), "Server vocabulary cannot replace internal eos reason");
        Check(!Valid(Completion(2, "EOS"), two, [42], 5), "Finish reason is exact");
        Check(!Valid(Completion(2), [42, 43], [42], 5), "Raw terminal EOS required");
        Check(!Valid(Completion(3), [99, 42, 99], [99, 42], 6), "No embedded EOS accepted");
        Check(!Valid(Completion(2), two, [42, 99], 5), "EOS must not be published");
        Check(!Valid(Completion(4), four, [19, 271, 17], 7), "Complete publication order required");
        Check(!Valid(Completion(2), two, [42], 4), "All fused outputs were forwarded; missing EOS advancement rejected");
        Check(!Valid(Completion(2), two, [42], 6), "Extra computed advancement rejected");
        Check(!Valid(Completion(1), two, [42], 5), "Completion count must match raw outputs");
        Check(!Valid(Completion(2, prompt: 4), two, [42], 5), "Completion prompt count must match raw input");
        Check(!Valid(Completion(2), two, [42], 5, max: 1), "Output budget boundary");
        Check(!Valid(Completion(2, reused: 1), two, [42], 5), "Prefix-off run cannot report reused tokens");
        Check(!Valid(Completion(2), two, [42], 5, error: true), "Sequence error cannot qualify");
        Check(!Valid(Completion(2, status: SequenceStatus.FinishedLengthCapped), two, [42], 5), "Length termination is not natural EOS");
        Console.WriteLine($"PASS exact/JSON-integer output policy and fused EOS structure: {checks} checks; no model/native execution.");
    }
}
