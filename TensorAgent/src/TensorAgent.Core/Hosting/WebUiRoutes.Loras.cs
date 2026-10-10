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
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Settings;

namespace TensorAgent.Core.Hosting;

public static partial class WebUiRoutes
{
    /// <summary>
    /// The page's LoRA sheet: the plug-ins of the image models this device is offered, what
    /// is downloaded and turned on, and the actions on them. A choice is saved, not applied:
    /// the next picture applies it (<see cref="AgentAppHost.PrepareImageTurn"/>), so
    /// turning a plug-in on while a picture is being made never touches that picture.
    /// </summary>
    public static void MapLoras(this LoopbackServer server, AgentAppHost host)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(host);

        CatalogLora? Find(string id) =>
            LoraCatalog.Find(id) is { } lora && host.Catalog.Any(m => lora.AppliesTo(m.Id)) ? lora : null;

        LoopbackResponse? NotFound(string id) => LoopbackResponse.Json(new { error = Loc.T("host.loras.notFound", ("id", id)) }, 404);

        server.MapGet("/api/agent/loras", (_, _) => Ok(DescribeLoras(host)));

        // A stream like a model's download: a window on the job, which outlives it.
        server.MapPost("/api/agent/loras/{id}/download", (request, ct) =>
        {
            string id = request.RouteValues["id"];
            if (Find(id) is not { } lora)
                return Task.FromResult(NotFound(id));
            host.StartLoraDownload(lora);
            return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                WatchFrames(host.Downloads, AgentAppHost.LoraDownloadKey(lora), ct), request.Cancellation));
        });

        server.MapPost("/api/agent/loras/{id}/download/cancel", (request, _) =>
        {
            string id = request.RouteValues["id"];
            if (Find(id) is not { } lora)
                return Task.FromResult(NotFound(id));
            return Ok(new { cancelled = host.Downloads.Cancel(AgentAppHost.LoraDownloadKey(lora)), id = lora.Id });
        });

        server.MapDelete("/api/agent/loras/{id}", (request, _) =>
        {
            string id = request.RouteValues["id"];
            if (Find(id) is not { } lora)
                return Task.FromResult(NotFound(id));
            host.DeleteLora(lora);
            return Ok(DescribeLoras(host));
        });

        // The whole choice at once, in the order the plug-ins were turned on: what the page
        // shows is what is saved, and two requests can never interleave into a third choice.
        server.MapPost("/api/agent/loras/choice", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            var requested = new List<ImageLoraChoice>();
            if (body.ValueKind == JsonValueKind.Object
                && body.TryGetProperty("loras", out JsonElement loras)
                && loras.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement item in loras.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.Object
                        || !item.TryGetProperty("id", out JsonElement id) || id.ValueKind != JsonValueKind.String)
                        return LoopbackResponse.Json(new { error = Loc.T("host.loras.choice.badItem") }, 400);
                    // No strength, or null, is the plug-in's own; anything else must be a number.
                    float strength = LoraCatalog.Find(id.GetString()!)?.DefaultStrength ?? 1f;
                    if (item.TryGetProperty("strength", out JsonElement s) && s.ValueKind != JsonValueKind.Null)
                    {
                        if (s.ValueKind != JsonValueKind.Number || !s.TryGetSingle(out strength))
                            return LoopbackResponse.Json(new { error = Loc.T("host.loras.choice.badStrength", ("id", id.GetString())) }, 400);
                    }
                    requested.Add(new ImageLoraChoice(id.GetString()!, strength));
                }
            }
            else
            {
                return LoopbackResponse.Json(new { error = Loc.T("host.loras.choice.badBody") }, 400);
            }

            return host.ChooseLoras(requested, out string? error) is null
                ? LoopbackResponse.Json(new { error }, 400)
                : LoopbackResponse.Json(DescribeLoras(host));
        });
    }

    /// <summary>The LoRA sheet's state: the plug-ins of the loaded image model (with none loaded,
    /// every plug-in of an offered model), and the choice. A Qwen-Image 2.1 Turbo entry is offered
    /// only the plug-ins validated on it, never a speed plug-in.</summary>
    internal static object DescribeLoras(AgentAppHost host)
    {
        IReadOnlyList<ImageLoraChoice> chosen = host.Settings.Load().ImageLoras;
        string? loaded = host.ModelService.LoadedModelPath;
        CatalogModel? loadedModel = host.Catalog.FirstOrDefault(m =>
            string.Equals(Path.GetFileName(Path.GetDirectoryName(loaded)), m.Id, StringComparison.Ordinal)
            && string.Equals(Path.GetFileName(loaded), m.Weights.FileName, StringComparison.Ordinal));
        return new
        {
            loadedModel = loadedModel?.Id,
            minStrength = LoraCatalog.MinStrength,
            maxStrength = LoraCatalog.MaxStrength,
            loras = LoraCatalog.Offered(loadedModel, host.Catalog)
                .Select(l =>
                {
                    ImageLoraChoice? choice = chosen.FirstOrDefault(c => string.Equals(c.Id, l.Id, StringComparison.OrdinalIgnoreCase));
                    return new
                    {
                        id = l.Id,
                        name = l.DisplayName,
                        baseModel = l.BaseModelId,
                        kind = l.Kind.ToString(),
                        purpose = l.Purpose,
                        trigger = l.Trigger,
                        needsPhoto = l.NeedsPhoto,
                        needsModelSteps = l.NeedsModelSteps,
                        license = l.License,
                        steps = l.Steps,
                        defaultStrength = l.DefaultStrength,
                        strengthAdjustable = l.StrengthAdjustable,
                        totalBytes = l.TotalBytes,
                        state = host.Loras.StateOf(l).ToString(),
                        installedBytes = host.Loras.InstalledBytes(l),
                        chosen = choice is not null,
                        strength = choice is not null && l.StrengthAdjustable ? choice.Strength : l.DefaultStrength,
                        download = host.Downloads.StatusOf(AgentAppHost.LoraDownloadKey(l)) is { } status ? Describe(status) : null,
                    };
                })
                .ToArray(),
            chosen = chosen.Select(c => new { id = c.Id, strength = c.Strength }).ToArray(),
        };
    }
}
