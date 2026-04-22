// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Multipart file upload endpoint used by the Web UI (and indirectly by
    /// Ollama / OpenAI clients that prefer references to base64).
    /// </summary>
    internal static class UploadEndpoints
    {
        public static IEndpointRouteBuilder MapUploadEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapPost("/api/upload",
                (HttpRequest req, WebUiAdapter adapter) => adapter.UploadAsync(req));
            return endpoints;
        }
    }
}
