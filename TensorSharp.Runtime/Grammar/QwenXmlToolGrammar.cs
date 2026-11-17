// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.Runtime.Grammar;

public enum QwenXmlToolChoice { Required, Named }

/// <summary>
/// <c>tool_choice</c> <c>required</c> or a named function for families that write Qwen3-Coder
/// XML calls (LFM3), as an immutable recipe; every generation gets its own constraint.
/// <code>
/// root    := ws BLOCK (ws BLOCK)* ws          (one BLOCK for a named choice or without parallel_tool_calls)
/// BLOCK   := &lt;tool_call&gt; ws CALL ws &lt;/tool_call&gt;
/// CALL    := &lt;function=NAME&gt; MEMBERS ws &lt;/function&gt;
/// MEMBERS := required parameters in schema order, then optional ones in any order
/// PARAM   := ws &lt;parameter=KEY&gt; VALUE &lt;/parameter&gt;
/// VALUE   := raw text without &lt;/parameter&gt; (a schema that may be a string; the
///            framing newlines are the parser's) | a listed enum value | ws JSON ws
/// </code>
/// It arms after <c>&lt;/think&gt;</c> when the request reasons (a thinking budget closes the
/// block too) and at the first token otherwise, where the prompt itself closed the block.
/// <c>auto</c> and <c>none</c> stay unconstrained: the parser reads whatever the model writes,
/// and <c>none</c> leaves the tools out of the prompt.
/// </summary>
public sealed class QwenXmlToolGrammar : IToolCallGrammar
{
    public const string CallOpen = "<tool_call>";
    public const string CallClose = "</tool_call>";
    private const string FunctionOpen = "<function=";
    private const string FunctionClose = "</function>";
    private const string ParameterOpen = "<parameter=";
    private const string ParameterClose = "</parameter>";
    private const string ReasoningEnd = "</think>";
    private static readonly JsonSerializerOptions JsonOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public string Source { get; }
    public QwenXmlToolChoice Choice { get; }

    private QwenXmlToolGrammar(string source, QwenXmlToolChoice choice) => (Source, Choice) = (source, choice);

    public GrammarConstraint NewConstraint(ITokenizer tokenizer, bool thinking)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        // The call tags are USER_DEFINED tokens in the vocabularies that have them, which the
        // grammar trie leaves out unless they are allowed by name; spelled out in plain tokens
        // they match the same text.
        var allowed = new List<int>();
        foreach (string tag in new[] { CallOpen, CallClose, FunctionOpen, FunctionClose, ParameterOpen, ParameterClose })
        {
            int id = tokenizer.LookupToken(tag);
            if (id >= 0) allowed.Add(id);
        }
        var constraint = GrammarLibrary.NewConstraint(GrammarLibrary.ForGbnf(Source, tokenizer, allowed), tokenizer);
        if (thinking) constraint.ActivateAfter(ReasoningEnd);
        return constraint;
    }

    public static QwenXmlToolGrammar Compile(IReadOnlyList<ToolFunction> tools, QwenXmlToolChoice choice,
        string? namedTool = null, bool parallelToolCalls = true)
    {
        if (tools == null || tools.Count == 0) throw Unsupported("tool_choice requires at least one declared function");
        if (tools.Count > 64) throw Unsupported("at most 64 functions are supported by the tool-call grammar");
        var rules = new Rules();
        var names = new HashSet<string>(StringComparer.Ordinal);
        var calls = new List<string>();
        foreach (ToolFunction tool in tools)
        {
            ValidateName(tool.Name, "function");
            if (!names.Add(tool.Name)) throw Unsupported($"duplicate function name '{tool.Name}'");
            if (choice == QwenXmlToolChoice.Named && tool.Name != namedTool) continue;
            using JsonDocument document = JsonDocument.Parse(ParameterSchema(tool));
            JsonElement root = document.RootElement;
            string members = rules.Members(root, root);
            string call = rules.Next("call");
            rules.Add(call, Literal(FunctionOpen + tool.Name + ">") + " " + members + " ws " + Literal(FunctionClose));
            calls.Add(call);
        }
        if (calls.Count == 0) throw Unsupported($"tool_choice names undeclared function '{namedTool}'");
        rules.Add("call", string.Join(" | ", calls));
        rules.Add("block", Literal(CallOpen) + " ws call ws " + Literal(CallClose));
        // A named choice is one call to that function, as OpenAI and vLLM serve it: with thinking
        // on, LFM3 forced to a function it did not want wrote the same call twice.
        bool several = parallelToolCalls && choice != QwenXmlToolChoice.Named;
        rules.Add("root", "ws block" + (several ? " (ws block)*" : "") + " ws");
        return new(rules.Emit(), choice);
    }

    /// <summary>The client's schema when it sent one, else the flattened parameters.</summary>
    private static string ParameterSchema(ToolFunction tool)
    {
        if (tool.ParametersSchemaJson != null) return tool.ParametersSchemaJson;
        var properties = new Dictionary<string, object>();
        foreach (var (name, parameter) in tool.Parameters ?? new())
        {
            var schema = new Dictionary<string, object> { ["type"] = string.IsNullOrEmpty(parameter.Type) ? "string" : parameter.Type };
            if (parameter.Enum is { Count: > 0 } && (string)schema["type"] == "string") schema["enum"] = parameter.Enum;
            properties[name] = schema;
        }
        return JsonSerializer.Serialize(new { type = "object", properties, required = tool.Required ?? new() });
    }

    private static NotSupportedException Unsupported(string detail)
        => new("Unsupported tool schema/policy for the Qwen-XML tool-call grammar: " + detail + ".");

    private static void ValidateName(string name, string kind)
    {
        if (string.IsNullOrEmpty(name) || name.Length > 128 || name.Any(c => char.IsControl(c) || char.IsWhiteSpace(c) || c is '<' or '>'))
            throw Unsupported($"{kind} names must be nonempty and contain no whitespace, angle brackets or control characters");
    }

    private static string Literal(string value) => JsonSerializer.Serialize(value, JsonOptions);

    private sealed class Rules
    {
        private readonly Dictionary<string, string> _rules = new(StringComparer.Ordinal);
        private int _next;
        private string? _raw;

        public Rules()
        {
            foreach (string line in JsonSchemaGrammarCompiler.JsonValueGrammar.Split('\n'))
            {
                int separator = line.IndexOf("::=", StringComparison.Ordinal);
                if (separator < 0) continue;
                string name = line[..separator].Trim();
                if (name != "root") Add(name, line[(separator + 3)..].Trim());
            }
            Add("integer", "\"-\"? (\"0\" | [1-9] [0-9]*)");
        }

        public string Next(string prefix) => prefix + "-" + _next++;
        public void Add(string name, string body) => _rules[name] = body;
        public string Emit() => string.Join("\n", _rules.Select(p => p.Key + " ::= " + p.Value)) + "\n";

        /// <summary>A function's parameters: required ones in schema order, then any number of
        /// optional ones in any order (llama.cpp's members_of).</summary>
        public string Members(JsonElement schema, JsonElement root)
        {
            if (schema.ValueKind != JsonValueKind.Object
                || !schema.TryGetProperty("properties", out JsonElement properties) || properties.ValueKind != JsonValueKind.Object)
                return "\"\"";
            var required = new HashSet<string>(StringComparer.Ordinal);
            if (schema.TryGetProperty("required", out JsonElement list) && list.ValueKind == JsonValueKind.Array)
                foreach (JsonElement name in list.EnumerateArray())
                    if (name.ValueKind == JsonValueKind.String) required.Add(name.GetString()!);
            var mandatory = new List<string>();
            var optional = new List<string>();
            foreach (JsonProperty property in properties.EnumerateObject())
            {
                ValidateName(property.Name, "parameter");
                string parameter = Next("param");
                Add(parameter, Literal(ParameterOpen + property.Name + ">") + " " + Value(Resolve(property.Value, root, 0), root)
                    + " " + Literal(ParameterClose));
                (required.Contains(property.Name) ? mandatory : optional).Add(parameter);
            }
            string body = string.Join(" ", mandatory.Select(p => "ws " + p));
            if (optional.Count > 0)
                body += (body.Length > 0 ? " " : "") + "(ws (" + string.Join(" | ", optional) + "))*";
            if (body.Length == 0) return "\"\"";
            string rule = Next("members");
            Add(rule, body);
            return rule;
        }

        private string Value(JsonElement s, JsonElement root)
        {
            if (s.ValueKind != JsonValueKind.Object) return Raw();
            if (s.TryGetProperty("enum", out JsonElement values) && values.ValueKind == JsonValueKind.Array && values.GetArrayLength() > 0)
            {
                // The parser keeps a string value's text and JSON-parses anything else, so
                // strings are written bare and other literals as JSON.
                var alternatives = values.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.String
                    ? Literal(v.GetString()!) : Literal(v.GetRawText())).Distinct().ToList();
                return "\"\\n\"? (" + string.Join(" | ", alternatives) + ") \"\\n\"?";
            }
            string type = Type(s);
            return type switch
            {
                "integer" => "ws integer ws",
                "number" => "ws number ws",
                "boolean" => "ws boolean ws",
                "null" => "ws null ws",
                "object" or "array" => "ws value ws",
                _ => Raw(),   // string, a union with string, or no type at all
            };
        }

        /// <summary>Raw text up to the parameter's close (the parser reads a declared string
        /// as text, so it may hold newlines, quotes and markup other than that close).</summary>
        private string Raw()
        {
            if (_raw == null)
            {
                _raw = "raw";
                ForbiddenText(_raw, ParameterClose);
            }
            return _raw;
        }

        private static JsonElement Resolve(JsonElement schema, JsonElement root, int depth)
        {
            if (schema.ValueKind != JsonValueKind.Object || !schema.TryGetProperty("$ref", out JsonElement reference))
                return schema;
            if (depth > 16) throw Unsupported("recursive $ref schemas are not supported");
            string path = reference.ValueKind == JsonValueKind.String ? reference.GetString()! : "";
            if (!path.StartsWith("#/", StringComparison.Ordinal)) throw Unsupported($"$ref '{path}' is not a local reference");
            JsonElement target = root;
            foreach (string segment in path[2..].Split('/'))
            {
                string key = segment.Replace("~1", "/", StringComparison.Ordinal).Replace("~0", "~", StringComparison.Ordinal);
                if (target.ValueKind != JsonValueKind.Object || !target.TryGetProperty(key, out target))
                    throw Unsupported($"$ref '{path}' does not resolve");
            }
            return Resolve(target, root, depth + 1);
        }

        private static string Type(JsonElement s)
        {
            if (s.TryGetProperty("type", out JsonElement type))
            {
                if (type.ValueKind == JsonValueKind.String) return type.GetString()!;
                if (type.ValueKind == JsonValueKind.Array)
                {
                    string[] names = type.EnumerateArray().Where(t => t.ValueKind == JsonValueKind.String)
                        .Select(t => t.GetString()!).Where(t => t != "null").Distinct().ToArray();
                    return names.Length == 1 ? names[0] : "string";
                }
            }
            if (s.TryGetProperty("properties", out _)) return "object";
            if (s.TryGetProperty("items", out _)) return "array";
            return "string";
        }

        // A small DFA over the characters of the forbidden word: any text that never contains
        // it (llama.cpp's until(close), which the parser mirrors).
        private void ForbiddenText(string prefix, string forbidden)
        {
            var states = new List<string> { "" };
            for (int i = 1; i < forbidden.Length; ++i) states.Add(forbidden[..i]);
            char[] alphabet = forbidden.Distinct().OrderBy(c => c).ToArray();
            string excluded = new(alphabet.SelectMany(c => c is '\\' or ']' or '[' or '^' or '-' ? new[] { '\\', c } : new[] { c }).ToArray());
            for (int i = 0; i < states.Count; ++i)
            {
                var alternatives = new List<string> { "\"\"", "[^" + excluded + "] " + prefix + "-0" };
                foreach (char c in alphabet)
                {
                    string next = states[i] + c;
                    if (next.EndsWith(forbidden, StringComparison.Ordinal)) continue;
                    int longest = Enumerable.Range(0, states.Count).Where(k => next.EndsWith(states[k], StringComparison.Ordinal))
                        .OrderByDescending(k => states[k].Length).First();
                    alternatives.Add(Literal(c.ToString(CultureInfo.InvariantCulture)) + " " + prefix + "-" + longest);
                }
                Add(prefix + "-" + i, string.Join(" | ", alternatives));
            }
            Add(prefix, prefix + "-0");
        }
    }
}
