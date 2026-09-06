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
    /// Routes for the OpenAI-compatible chat-completions surface.
    /// </summary>
    public static class OpenAIEndpoints
    {
        public static IEndpointRouteBuilder MapOpenAIEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapPost("/v1/chat/completions",
                (HttpContext ctx, OpenAIChatAdapter adapter) => adapter.ChatCompletionsAsync(ctx));
            endpoints.MapGet("/v1/models",
                (OpenAIChatAdapter adapter) => adapter.ListModels());
            endpoints.MapPost("/v1/responses",
                (HttpContext ctx, OpenAIResponsesAdapter adapter) => adapter.CreateResponseAsync(ctx));
            endpoints.MapGet("/v1/responses/{id}",
                (HttpContext ctx, OpenAIResponsesAdapter adapter, string id) => adapter.GetResponseAsync(ctx, id));
            // Text-to-video generation (Wan models). OpenAI has no stable public video
            // API yet; this follows the images/generations envelope: prompt in, a data
            // array with url and (optionally) b64_json out.
            endpoints.MapPost("/v1/videos/generations",
                (HttpRequest req, WebUiAdapter adapter) => adapter.OpenAIVideoGenerationsAsync(req))
                .DisableRequestTimeout();
            return endpoints;
        }
    }
}
