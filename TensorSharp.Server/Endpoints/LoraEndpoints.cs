// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace TensorSharp.Server.Endpoints;

/// <summary>
/// Administrative routes for loading and unloading text-LoRA adapters at runtime,
/// without reloading the base model. A loaded adapter can then be selected per
/// request with the <c>lora</c> field on <c>/v1/chat/completions</c> or
/// <c>/v1/responses</c>, and is listed under <c>GET /v1/models</c>.
///
/// <para>
/// These are operator-only actions: <c>path</c> is an arbitrary file path on the
/// server host, so they must be protected by the operator's own network boundary
/// exactly as the model-file loading surface is.
/// </para>
/// </summary>
public static class LoraEndpoints
{
    public static IEndpointRouteBuilder MapLoraEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/v1/loras", (ModelService svc) =>
        {
            var data = svc.LoraRegistry.List()
                .Select(a => new Dictionary<string, object>
                {
                    ["id"] = a.Name,
                    ["rank"] = a.Rank,
                    ["alpha"] = a.Alpha,
                    ["modules"] = a.Modules,
                    ["active"] = string.Equals(svc.ActiveLoraName, a.Name, StringComparison.Ordinal),
                })
                .ToList();
            return Results.Json(new { @object = "list", data });
        });

        endpoints.MapPost("/v1/loras", async (HttpContext ctx, ModelService svc) =>
        {
            string name;
            string path;
            string? config = null;
            float? scale = null;
            try
            {
                var doc = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body).ConfigureAwait(false);
                if (!doc.TryGetProperty("name", out var nameEl) || nameEl.ValueKind != JsonValueKind.String ||
                    string.IsNullOrWhiteSpace(nameEl.GetString()))
                {
                    await BadAsync(ctx, "a non-empty \"name\" is required").ConfigureAwait(false);
                    return;
                }
                name = nameEl.GetString()!;

                if (!doc.TryGetProperty("path", out var pathEl) || pathEl.ValueKind != JsonValueKind.String ||
                    string.IsNullOrWhiteSpace(pathEl.GetString()))
                {
                    await BadAsync(ctx, "a non-empty \"path\" to a .safetensors file is required").ConfigureAwait(false);
                    return;
                }
                path = pathEl.GetString()!;

                if (doc.TryGetProperty("config", out var configEl) && configEl.ValueKind == JsonValueKind.String)
                    config = configEl.GetString();
                if (doc.TryGetProperty("scale", out var scaleEl) && scaleEl.ValueKind == JsonValueKind.Number)
                    scale = scaleEl.GetSingle();
            }
            catch (JsonException)
            {
                await BadAsync(ctx, "request body must be a JSON object").ConfigureAwait(false);
                return;
            }

            // Detach before replacing so a reload of an active adapter releases the old
            // device tensors rather than leaving them pinned by the model.
            if (string.Equals(svc.ActiveLoraName, name, StringComparison.Ordinal))
                svc.ClearActiveLora();

            try
            {
                var adapter = svc.LoraRegistry.Load(name, path, config, scale);
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsJsonAsync(new
                {
                    id = adapter.Name,
                    rank = adapter.Rank,
                    alpha = adapter.Alpha,
                    modules = adapter.Deltas.Count,
                    summary = adapter.Summary,
                }).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or ArgumentException)
            {
                await BadAsync(ctx, ex.Message).ConfigureAwait(false);
            }
        });

        endpoints.MapDelete("/v1/loras/{name}", (ModelService svc, string name) =>
        {
            if (string.Equals(svc.ActiveLoraName, name, StringComparison.Ordinal))
                svc.ClearActiveLora();

            bool removed = svc.LoraRegistry.Unload(name);
            return removed
                ? Results.Json(new { removed = true })
                : Results.Json(new { error = $"No LoRA adapter named '{name}' is loaded." }, statusCode: 404);
        });

        return endpoints;
    }

    private static async Task BadAsync(HttpContext ctx, string message)
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = message }).ConfigureAwait(false);
    }
}