// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.RegularExpressions;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;

namespace DirectCudaProbe;

internal sealed record QualitySelection(string RawText, string Content, string Thinking, ToolCall[] ToolCalls);

internal sealed record QualityPrompt(string Text, string Utf8Sha256)
{
    public static string? ParseReasoningEffort(string? value)
    {
        if (value is null) return null; // Preserve the historical family default.
        if (!ReasoningEffort.TryNormalize(value, out string? normalized) || normalized is null)
            throw new ArgumentException("quality-reasoning-effort must be low, medium or high when provided.");
        return normalized;
    }

    public static QualityPrompt Create(string template, string architecture, string text, string? reasoningEffort)
    {
        string rendered = new GgufPromptRenderer().Render(template, new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant." },
            new() { Role = "user", Content = text },
        }, addGenerationPrompt: true, architecture: architecture, tools: null,
            enableThinking: false, reasoningEffort: reasoningEffort);
        return new(rendered, Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(rendered))).ToLowerInvariant());
    }
}

internal static class QualityOutput
{
    // Extraction previously collected every alphabetic word, accidentally
    // accepting JSON arrays and Markdown-fenced answers as plain comma lists.
    public const string SemanticPolicy = "direct-answer-only-v3";
    public static readonly (string Name, string Prompt)[] Cases =
    [
        ("math", "Compute 17 + 25. Reply with only the integer."),
        ("extraction", "Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order."),
        ("squares", "Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list."),
        ("code", "Write a Python function square(x) that returns x*x. Return only the function, without Markdown."),
        ("tool_json", "Available tool: get_weather(city, unit), where unit is \"celsius\" or \"fahrenheit\". User asks: What is the weather in Hangzhou in Celsius? Return only one JSON object with exactly the keys \"name\" and \"arguments\", containing the tool name and its required arguments. Do not answer the weather question."),
    ];
    public static bool Semantic(string name, string answer) => name switch
    {
        "math" => Regex.IsMatch(answer, @"^42[.!]?$"),
        "extraction" => Regex.IsMatch(answer, @"\Acedar\s*,\s*maple\z", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        "squares" => Regex.IsMatch(answer, @"^\d+(?:\s*,\s*\d+){19}$") && answer.Split(',').Select(s => int.TryParse(s.Trim(), out int value) ? value : -1).SequenceEqual(Enumerable.Range(1, 20).Select(x => x * x)),
        "code" => SquareFunction(answer),
        "tool_json" => ToolJson(answer),
        _ => false,
    };

    public static string[] SelectCases(string value)
    {
        string[] cases = value.Split(',', StringSplitOptions.TrimEntries);
        if (cases.Length == 0 || cases.Distinct(StringComparer.Ordinal).Count() != cases.Length ||
            cases.Any(name => !Cases.Any(test => test.Name == name)))
            throw new ArgumentException("quality-cases must be a nonempty, duplicate-free subset of math,extraction,squares,code,tool_json.");
        return cases;
    }

    private static bool ToolJson(string answer)
    {
        try
        {
            using JsonDocument json = JsonDocument.Parse(answer);
            JsonElement root = json.RootElement;
            if (!ExactKeys(root, "name", "arguments") || root.GetProperty("name").ValueKind != JsonValueKind.String ||
                root.GetProperty("name").GetString() != "get_weather") return false;
            JsonElement args = root.GetProperty("arguments");
            return ExactKeys(args, "city", "unit") && args.GetProperty("city").ValueKind == JsonValueKind.String &&
                args.GetProperty("city").GetString() == "Hangzhou" && args.GetProperty("unit").ValueKind == JsonValueKind.String &&
                args.GetProperty("unit").GetString() == "celsius";
        }
        catch (JsonException) { return false; }
        catch (InvalidOperationException) { return false; } // Invalid UTF-16 escape when decoding a JSON string/key.
    }

    private static bool ExactKeys(JsonElement value, params string[] expected)
    {
        if (value.ValueKind != JsonValueKind.Object) return false;
        string[] actual = value.EnumerateObject().Select(p => p.Name).ToArray();
        return actual.Length == expected.Length && actual.Distinct(StringComparer.Ordinal).Count() == actual.Length &&
            expected.All(key => actual.Contains(key, StringComparer.Ordinal));
    }

    // Recognize the requested pure Python function without executing code or
    // accepting arbitrary text containing its keywords. The independent Python
    // checker compares full ASTs; this managed check recognizes the ordinary
    // single-return grammar, including comments and redundant parentheses.
    private static bool SquareFunction(string answer)
    {
        if (answer.Length > 65536) return false;
        string[] lines = answer.Trim().Replace("\r\n", "\n", StringComparison.Ordinal).Replace('\r', '\n')
            .Split('\n').Select(line => line.Split('#', 2)[0].TrimEnd(' ', '\t'))
            .Where(line => line.Length != 0).ToArray();
        const string header = @"def[ \t]+square[ \t]*\([ \t]*x[ \t]*\)[ \t]*:";
        if (lines.Length == 0) return false;
        Match definition = Regex.Match(lines[0], @"\A" + header + @"[ \t]*(.*)\z");
        if (!definition.Success) return false;
        int bodyLine = 0;
        string body = definition.Groups[1].Value;
        if (body.Length == 0)
        {
            if (lines.Length < 2 || lines[1][0] is not (' ' or '\t')) return false;
            body = lines[++bodyLine].TrimStart(' ', '\t');
        }
        Match returned = Regex.Match(body, @"\Areturn(?=[ \t(])[ \t]*(.+)\z");
        if (!returned.Success) return false;
        var expressionText = new StringBuilder();
        int nesting = 0;
        for (int line = bodyLine; line < lines.Length; ++line)
        {
            // Implicit Python continuation is legal only while an expression
            // parenthesis is open; a second statement cannot become whitespace.
            if (line > bodyLine && nesting == 0) return false;
            string fragment = line == bodyLine ? returned.Groups[1].Value : lines[line];
            foreach (char ch in fragment)
            {
                if (ch == '(' && ++nesting > 64) return false;
                if (ch == ')' && --nesting < 0) return false;
            }
            expressionText.Append(fragment).Append(' ');
        }
        if (nesting != 0) return false;
        string expression = expressionText.ToString();
        expression = expression.TrimEnd(' ', '\t');
        if (expression.EndsWith(';')) expression = expression[..^1];
        int position = 0, leaves = 0, multiplies = 0;
        void Space() { while (position < expression.Length && expression[position] is ' ' or '\t') position++; }
        bool Atom(int depth)
        {
            if (depth > 64) return false;
            Space();
            if (position >= expression.Length) return false;
            if (expression[position] == 'x') { position++; leaves++; return true; }
            if (expression[position++] != '(' || !Product(depth + 1)) return false;
            Space();
            return position < expression.Length && expression[position++] == ')';
        }
        bool Product(int depth)
        {
            if (!Atom(depth)) return false;
            Space();
            while (position < expression.Length && expression[position] == '*')
            {
                position++; multiplies++;
                if (!Atom(depth)) return false;
                Space();
            }
            return true;
        }
        return Product(0) && position == expression.Length && leaves == 2 && multiplies == 1;
    }

    public static QualitySelection Parse(string raw, string architecture, string prompt, string mode)
    {
        if (mode == "raw")
        {
            // Preserve the probe's historical default. Only an explicitly empty
            // thinking block is removed, never arbitrary reasoning or prose.
            return new(raw, Regex.Replace(raw, @"^\s*<think>\s*</think>\s*", "").Trim(), "", []);
        }
        if (mode != "production") throw new ArgumentException("quality-parser must be raw or production.");
        IOutputParser parser = OutputParserFactory.Create(architecture);
        parser.Init(enableThinking: false, tools: null);
        parser.SetGenerationPromptSuffix(prompt);
        ParsedOutput parsed = parser.Add(raw, done: true);
        return new(raw, parsed.Content, parsed.Thinking, parsed.ToolCalls?.ToArray() ?? []);
    }

    // Parsing a final fragment does not prove generation completed. The actual
    // checkpoint EOS must have been sampled, and these are answer-only tasks.
    public static bool CompletePlainAnswer(QualitySelection output, bool eos) => eos && output.ToolCalls.Length == 0;

    // The GLM/Qwen angle-bracket protocol also permits whitespace after the
    // prompt opener and an opener emitted in the response itself. This does
    // not claim to interpret other architectures' channel-token protocols.
    public static bool ThinkingClosed(string prompt, string raw)
    {
        bool open = prompt.TrimEnd().EndsWith("<think>", StringComparison.Ordinal);
        foreach (Match tag in Regex.Matches(raw, @"</?think>")) open = tag.Value == "<think>";
        return !open;
    }

    public static bool CompleteCaseAnswer(string name, QualitySelection output, bool eos, string prompt) =>
        CompletePlainAnswer(output, eos) && (name is not ("code" or "tool_json") || ThinkingClosed(prompt, output.RawText));
}
