// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Parsers for the <c>tools</c> field across all three protocols. Returns
    /// null when no tools are declared so downstream code can short-circuit
    /// (the parsing path is identical, only the wrapping differs: Ollama nests
    /// the function under <c>"function"</c>; OpenAI additionally requires
    /// <c>"type": "function"</c>).
    /// </summary>
    internal static class ToolFunctionParser
    {
        public static List<ToolFunction> ParseOllama(JsonElement body)
        {
            if (!body.TryGetProperty("tools", out var toolsEl) || toolsEl.ValueKind != JsonValueKind.Array)
                return null;

            var tools = new List<ToolFunction>();
            foreach (var toolEl in toolsEl.EnumerateArray())
            {
                if (!toolEl.TryGetProperty("function", out var fnEl))
                    continue;
                var tf = ParseFunction(fnEl);
                if (tf != null)
                    tools.Add(tf);
            }
            return tools.Count > 0 ? tools : null;
        }

        public static List<ToolFunction> ParseOpenAI(JsonElement body)
        {
            if (!body.TryGetProperty("tools", out var toolsEl) || toolsEl.ValueKind != JsonValueKind.Array)
                return null;

            var tools = new List<ToolFunction>();
            foreach (var toolEl in toolsEl.EnumerateArray())
            {
                string type = toolEl.TryGetProperty("type", out var t) ? t.GetString() : "function";
                if (type != "function") continue;
                if (!toolEl.TryGetProperty("function", out var fnEl)) continue;

                var tf = ParseFunction(fnEl);
                if (tf != null)
                    tools.Add(tf);
            }
            return tools.Count > 0 ? tools : null;
        }

        private static ToolFunction ParseFunction(JsonElement fnEl)
        {
            var tf = new ToolFunction
            {
                Name = fnEl.TryGetProperty("name", out var n) ? n.GetString() : "",
                Description = fnEl.TryGetProperty("description", out var d) ? d.GetString() : ""
            };

            if (fnEl.TryGetProperty("parameters", out var paramsEl))
            {
                if (paramsEl.TryGetProperty("properties", out var propsEl) &&
                    propsEl.ValueKind == JsonValueKind.Object)
                {
                    tf.Parameters = new Dictionary<string, ToolParameter>();
                    foreach (var prop in propsEl.EnumerateObject())
                    {
                        var tp = new ToolParameter
                        {
                            Type = prop.Value.TryGetProperty("type", out var pt) ? pt.GetString() : "string",
                            Description = prop.Value.TryGetProperty("description", out var pd) ? pd.GetString() : null
                        };
                        if (prop.Value.TryGetProperty("enum", out var enumEl) && enumEl.ValueKind == JsonValueKind.Array)
                            tp.Enum = enumEl.EnumerateArray().Select(e => e.GetString()).ToList();
                        tf.Parameters[prop.Name] = tp;
                    }
                }
                if (paramsEl.TryGetProperty("required", out var reqEl) && reqEl.ValueKind == JsonValueKind.Array)
                    tf.Required = reqEl.EnumerateArray().Select(e => e.GetString()).ToList();
            }

            return tf;
        }
    }
}
