// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class OpenAIChatAdapter
{
    /// <summary>Families whose required / named <c>tool_choice</c> is enforced with
    /// <see cref="QwenXmlToolGrammar"/>. LFM3 only for now: other Qwen-XML families (Qwen 3.5,
    /// Qwen 3.8) keep their unconstrained behavior until their own suites have run with it.</summary>
    private static bool UsesQwenXmlToolGrammar(string architecture)
        => ChatProtocolRegistry.For(architecture)?.Id == "lfm3";

    /// <summary>
    /// <c>tool_choice</c> <c>required</c> or a named function on LFM3: the reply must be
    /// well-formed Qwen3-Coder XML calls to the client's functions (only the named one), from
    /// the first token with thinking off and after <c>&lt;/think&gt;</c> with it on. Without
    /// this the model answered a named choice with a call to another function, or with prose
    /// (eng/lfm3/e2e); llama.cpp does not enforce a named choice for this template either.
    /// <c>auto</c> and <c>none</c> stay unconstrained. <see cref="ValidateClientToolChoice"/>
    /// has already checked the choice against the declared functions.
    /// </summary>
    internal static QwenXmlToolGrammar? PrepareQwenXmlToolGrammar(JsonElement body,
        List<ToolFunction>? clientTools, StructuredOutputFormat? responseFormat)
    {
        if (!body.TryGetProperty("tool_choice", out var requested))
            return null;
        QwenXmlToolChoice choice;
        string? name = null;
        if (requested.ValueKind == JsonValueKind.String && requested.GetString() == "required")
            choice = QwenXmlToolChoice.Required;
        else if (requested.ValueKind == JsonValueKind.Object)
        {
            choice = QwenXmlToolChoice.Named;
            name = requested.GetProperty("function").GetProperty("name").GetString();
        }
        else
            return null;
        if (responseFormat != null)
            throw new NotSupportedException("A required tool call cannot be combined with response_format.");

        bool parallel = !body.TryGetProperty("parallel_tool_calls", out var parallelProperty)
            || parallelProperty.ValueKind != JsonValueKind.False;
        var plan = QwenXmlToolGrammar.Compile(clientTools ?? [], choice, name, parallel);
        _ = Grammar.Parse(plan.Source); // Reject unsupported recipes before any SSE headers or inference.
        return plan;
    }
}
