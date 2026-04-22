// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Cached <see cref="JsonSerializerOptions"/> used by the Ollama and OpenAI
    /// adapters. Reused instead of allocating per-write because the Ollama
    /// streaming hot path emits one of these per generated token.
    /// </summary>
    internal static class JsonOptions
    {
        /// <summary>Skip emitting properties that are null. Used for both protocols' "thinking" and "tool_calls" fields.</summary>
        public static readonly JsonSerializerOptions IgnoreNulls = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
    }
}
