// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.Server.ResponseSerializers;

/// <summary>
/// A tool call's <c>arguments</c> as the JSON text OpenAI-compatible responses carry (chat completions, their stream deltas,
/// Responses function calls): raw UTF-8, as OpenAI and llama.cpp write it. System.Text.Json's default encoder wrote every
/// non-ASCII character and <c>'</c>, <c>+</c>, <c>&lt;</c>, <c>&gt;</c>, <c>&amp;</c> as <c>\uXXXX</c> escapes
/// (<c>{"city":"北京"}</c>): valid JSON, but not the text the model wrote, which clients that show or compare the
/// string saw. Only what JSON requires is escaped now (quotes, backslashes, control characters).
/// </summary>
internal static class ToolCallArgumentsJson
{
    private static readonly JsonSerializerOptions Options = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    internal static string Serialize(Dictionary<string, object?>? arguments) => JsonSerializer.Serialize(arguments, Options);
}
