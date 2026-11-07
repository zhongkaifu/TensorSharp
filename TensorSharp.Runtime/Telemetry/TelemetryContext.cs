// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Threading;

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// Request-scoped facts the HTTP layer knows and the generation code below it does not,
    /// carried in an <see cref="AsyncLocal{T}"/> so no signature has to change. The request
    /// logger sets them once per request; everything that request calls reads them.
    /// </summary>
    public static class TelemetryContext
    {
        private static readonly AsyncLocal<string?> s_protocol = new();
        private static readonly AsyncLocal<string?> s_requestId = new();

        /// <summary><c>openai</c>, <c>ollama</c>, <c>webui</c>, or null outside an HTTP request (the CLI).</summary>
        public static string? Protocol
        {
            get => s_protocol.Value;
            set => s_protocol.Value = value;
        }

        /// <summary>The <c>X-TensorSharp-Request-Id</c> of the current request, if any.</summary>
        public static string? RequestId
        {
            get => s_requestId.Value;
            set => s_requestId.Value = value;
        }

        private static readonly AsyncLocal<int?> s_round = new();
        private static readonly AsyncLocal<string?> s_origin = new();

        /// <summary>
        /// 1-based round of a skills or multi-agent turn, set by the loop around each model
        /// call so the generation's span carries <c>tensorsharp.round</c>; null for a single call.
        /// </summary>
        public static int? Round
        {
            get => s_round.Value;
            set => s_round.Value = value;
        }

        /// <summary>
        /// Forget every request fact on this execution context. Long-lived worker threads (the
        /// inference engine, the diffusion scheduler) call it first: <c>Thread.Start</c> flowed
        /// the context of whichever request created them, and they serve every request.
        /// </summary>
        public static void Clear()
        {
            s_protocol.Value = null;
            s_requestId.Value = null;
            s_round.Value = null;
            s_origin.Value = null;
        }

        /// <summary>
        /// Why TensorSharp itself is making a model call (<c>knowledge-rerank</c>,
        /// <c>knowledge-media-extract</c>), set around that call; null for a client's request.
        /// Becomes <c>tensorsharp.origin</c> and <c>tensorsharp.internal=true</c>.
        /// </summary>
        public static string? Origin
        {
            get => s_origin.Value;
            set => s_origin.Value = value;
        }
    }
}
