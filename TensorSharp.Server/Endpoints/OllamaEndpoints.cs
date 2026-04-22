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
    /// Routes for the Ollama-compatible HTTP surface. Maps each HTTP path onto
    /// an instance method on <see cref="OllamaAdapter"/>; behaviour lives in
    /// the adapter so the routing table stays trivially auditable.
    /// </summary>
    internal static class OllamaEndpoints
    {
        public static IEndpointRouteBuilder MapOllamaEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapGet("/api/version", () => Results.Json(new { version = "0.1.0" }));
            endpoints.MapGet("/api/tags", (OllamaAdapter adapter) => adapter.GetTags());
            endpoints.MapPost("/api/show", (HttpContext ctx, OllamaAdapter adapter) => adapter.ShowAsync(ctx));
            endpoints.MapPost("/api/generate", (HttpContext ctx, OllamaAdapter adapter) => adapter.GenerateAsync(ctx));
            endpoints.MapPost("/api/chat/ollama", (HttpContext ctx, OllamaAdapter adapter) => adapter.ChatAsync(ctx));
            return endpoints;
        }
    }
}
