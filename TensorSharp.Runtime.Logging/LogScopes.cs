// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Well-known property names used in structured log scopes across TensorSharp
    /// hosts. Centralising these keys keeps log queries (in any sink that supports
    /// scopes) consistent between the Web server, the CLI, and the test harness.
    /// </summary>
    public static class LogScopeKeys
    {
        /// <summary>Server-assigned id for an HTTP request (correlates middleware + handler logs).</summary>
        public const string RequestId = "RequestId";

        /// <summary>Logical operation name (e.g. <c>chat.completions</c>, <c>model.load</c>).</summary>
        public const string Operation = "Operation";

        /// <summary>Conversation session identifier.</summary>
        public const string SessionId = "SessionId";

        /// <summary>Loaded model identifier (file name or basename).</summary>
        public const string Model = "Model";

        /// <summary>Active backend name (<c>ggml_cpu</c>, <c>ggml_metal</c>, ...).</summary>
        public const string Backend = "Backend";

        /// <summary>Identifier of the API client (Ollama, OpenAI, WebUI, CLI, ...).</summary>
        public const string Client = "Client";

        /// <summary>Server-side process or correlation id used when no request context is available.</summary>
        public const string CorrelationId = "CorrelationId";
    }
}
