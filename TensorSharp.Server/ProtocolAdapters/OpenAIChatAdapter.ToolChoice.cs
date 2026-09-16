// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class OpenAIChatAdapter
{
    /// <summary>
    /// Reject impossible client tool selections before queue admission on every
    /// model family. Grammar compilation remains architecture-specific.
    /// </summary>
    private static void ValidateClientToolChoice(JsonElement body, List<ToolFunction>? clientTools)
    {
        if (body.TryGetProperty("parallel_tool_calls", out var parallel) &&
            parallel.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new JsonException("parallel_tool_calls must be a boolean.");
        if (!body.TryGetProperty("tool_choice", out var requested)) return;
        bool required = false;
        string? named = null;
        if (requested.ValueKind == JsonValueKind.String)
        {
            switch (requested.GetString())
            {
                case "auto":
                case "none": return;
                case "required": required = true; break;
                default: throw new NotSupportedException("tool_choice must be auto, none, required, or a named function.");
            }
        }
        else if (requested.ValueKind == JsonValueKind.Object &&
                 requested.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String && type.GetString() == "function" &&
                 requested.TryGetProperty("function", out var function) && function.ValueKind == JsonValueKind.Object &&
                 function.TryGetProperty("name", out var name) && name.ValueKind == JsonValueKind.String &&
                 !string.IsNullOrEmpty(name.GetString()))
            named = name.GetString();
        else
            throw new NotSupportedException("tool_choice names a function as {type: function, function: {name: ...}} with a nonempty name.");
        if (!required && named == null) return;

        // Internal skill/shell rounds are serviced invisibly by this server;
        // they cannot satisfy a caller's required/named client-function result.
        // Absent/auto/none selections retain their existing internal-tool paths.
        if (clientTools is not { Count: > 0 })
            throw new NotSupportedException("tool_choice requires at least one client-declared function.");
        if (named != null && !clientTools.Any(tool => string.Equals(tool.Name, named, StringComparison.Ordinal)))
            throw new NotSupportedException($"The named tool_choice function '{named}' is not declared in tools.");
    }
}
