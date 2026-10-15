using System.Globalization;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

namespace GlmUnifiedSessionProbe;

internal static class ProbePolicy
{
    public const string MathPrompt = "Compute 17 + 25. Reply with only the integer.";
    public static bool AdaptivePrefill(string text, bool selected)
    {
        if (text is not ("true" or "false")) throw new ArgumentException("adaptive-prefill must be true or false.");
        if (text == "true" && !selected) throw new ArgumentException("adaptive-prefill requires selected=true.");
        return text == "true";
    }
    public static int[] ParseTokens(string text)
    {
        if (text.Length is 0 or > 4_194_304) throw new ArgumentException("Raw token file is empty or too large.");
        if (!Regex.IsMatch(text, @"\A[ \t\r\n]*[0-9]+(?:(?:[ \t\r\n]*,[ \t\r\n]*|[ \t\r\n]+)[0-9]+)*[ \t\r\n]*\z",
                RegexOptions.NonBacktracking | RegexOptions.CultureInvariant))
            throw new ArgumentException("Malformed token separators or empty CSV field.");
        string[] words = text.Split([',', ' ', '\r', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries);
        if (words.Length is 0 or > 1_048_576) throw new ArgumentException("Raw token count is outside the bounded range.");
        return words.Select(w => Regex.IsMatch(w, @"\A(?:0|[1-9][0-9]*)\z", RegexOptions.CultureInvariant)
            && int.TryParse(w, NumberStyles.None, CultureInfo.InvariantCulture, out int n) && n >= 0
            ? n : throw new ArgumentException("Invalid nonnegative int32 token ID: " + w)).ToArray();
    }
    public static bool MathPromptMatches(string prompt) => prompt.Contains(
        "<|user|>" + MathPrompt + "<|assistant|>", StringComparison.Ordinal);
    public static bool NaturalEos(int[] tokens, int[] eos, string? finish)
        => finish == "eos" && tokens.Length > 0 && eos.Contains(tokens[^1])
            && !tokens.Take(tokens.Length - 1).Any(eos.Contains);
    public static bool StrictMath(string content, int toolCalls, bool thinkingClosed)
        => content == "42" && toolCalls == 0 && thinkingClosed;
    public static int? CaptureToken(string? raw, bool captureFinalLogits, int maxNew, string task)
    {
        if (raw == null) return null;
        if (!captureFinalLogits || maxNew != 1 || task != "unknown"
            || !Regex.IsMatch(raw, @"\A(?:0|[1-9][0-9]*)\z", RegexOptions.CultureInvariant)
            || !int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out int token))
            throw new ArgumentException("capture-token-id requires capture-final-logits true, max-new1, unknown task and one nonnegative int32 ID.");
        return token;
    }
    public static void SelfTest()
    {
        FinalLogitCapture.SelfTest();
        int checks = 0;
        void Check(bool condition) { checks++; if (!condition) throw new Exception("Self-test " + checks + " failed."); }
        Check(CaptureToken(null, false, 128, "math42") == null);
        Check(CaptureToken("0", true, 1, "unknown") == 0);
        Check(CaptureToken("2147483647", true, 1, "unknown") == int.MaxValue);
        foreach (string bad in new[] { "", "-1", "01", "١", "2147483648", "1.0" })
        { bool rejected = false; try { CaptureToken(bad, true, 1, "unknown"); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        foreach (var (capture, maximum, task) in new[] { (false, 1, "unknown"), (true, 2, "unknown"), (true, 1, "math42") })
        { bool rejected = false; try { CaptureToken("1", capture, maximum, task); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        Check(!AdaptivePrefill("false", false)); Check(!AdaptivePrefill("false", true)); Check(AdaptivePrefill("true", true));
        foreach (var pair in new[] { ("true", false), ("1", true), ("TRUE", true), ("", true) })
        { bool rejected = false; try { AdaptivePrefill(pair.Item1, pair.Item2); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        Check(ParseTokens("1,2,2147483647\n").SequenceEqual(new[] { 1, 2, int.MaxValue }));
        foreach (string bad in new[] { "", "-1", "1.0", "01", "NaN", "2147483648", "true", "١", "1,,2", "1,", ",1" })
        { bool rejected = false; try { ParseTokens(bad); } catch (ArgumentException) { rejected = true; } Check(rejected); }
        Check(MathPromptMatches("[gMASK]<sop><|user|>" + MathPrompt + "<|assistant|><think>"));
        Check(!MathPromptMatches("Compute 17 + 26. Reply with only the integer."));
        Check(NaturalEos([31, 99], [99], "eos"));
        Check(!NaturalEos([99, 31, 99], [99], "eos"));
        Check(!NaturalEos([31, 99], [99], "max_tokens"));
        Check(!NaturalEos([31], [99], "eos"));
        Check(!NaturalEos([], [99], "eos"));
        Check(StrictMath("42", 0, true));
        foreach (string bad in new[] { " 42", "42\n", "42.", "The answer is 42", "42</think>42", "", "43" })
            Check(!StrictMath(bad, 0, true));
        Check(!StrictMath("42", 1, true)); Check(!StrictMath("42", 0, false));
        var parser = OutputParserFactory.Create("glm5next"); parser.Init(false, null);
        ParsedOutput parsed = parser.Add("42</think>42", true);
        Check(StrictMath(parsed.Content, parsed.ToolCalls?.Count ?? 0, true));
        parser.Init(false, null); parsed = parser.Add("working out17+25</think>43", true);
        Check(!StrictMath(parsed.Content, parsed.ToolCalls?.Count ?? 0, true));
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { Passed = true, Checks = checks,
            Scope = "Model-free token/prompt/EOS/exact-answer policy; no native, model or device execution." }));
    }
}
