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
    /// Routes for chat-session lifecycle. Each Web UI tab creates a session on
    /// load, attaches its id to every <c>/api/chat</c> request, and disposes it
    /// when the user clicks "New Chat" so the prior conversation's KV cache is
    /// released.
    /// </summary>
    internal static class SessionEndpoints
    {
        public static IEndpointRouteBuilder MapSessionEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapPost("/api/sessions",
                (WebUiAdapter adapter) => adapter.CreateSession());
            endpoints.MapDelete("/api/sessions/{id}",
                (string id, HttpContext ctx, WebUiAdapter adapter) => adapter.DisposeSessionAsync(id, ctx));
            return endpoints;
        }
    }
}
