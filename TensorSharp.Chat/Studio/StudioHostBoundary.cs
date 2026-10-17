// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Net;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Studio;

namespace TensorSharp.Chat.Studio;

/// <summary>The two transports share request dispatch and the only bridge into pre-existing chat uploads.</summary>
public sealed class StudioHostBoundary
{
    private readonly string _uploads;
    private readonly StudioApi _api;
    public StudioService Service { get; }
    public StudioHostBoundary(StudioService service, string uploadDirectory)
    { Service = service; _uploads = Path.GetFullPath(uploadDirectory); _api = new StudioApi(service); }

    public async Task<JsonElement> DispatchAsync(JsonElement body, CancellationToken ct)
    {
        if (body.ValueKind != JsonValueKind.Object || !body.TryGetProperty("action", out var action) || action.ValueKind != JsonValueKind.String)
            throw new StudioException("invalid_request", "Provide action and payload.");
        JsonElement payload = body.TryGetProperty("payload", out var p) ? p : JsonSerializer.SerializeToElement(new { });
        if (action.GetString() == "assets/import-chat")
        {
            string url = payload.GetProperty("url").GetString() ?? "";
            string path = ResolveChatUpload(url);
            await using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
            var project = await Service.ImportAssetAsync(payload.GetProperty("projectId").GetString()!, file, Path.GetFileName(path), ct).ConfigureAwait(false);
            return JsonSerializer.SerializeToElement(project, StudioJson.Options);
        }
        return await _api.DispatchAsync(action.GetString()!, payload, ct).ConfigureAwait(false);
    }

    public string ResolveChatUpload(string url)
    {
        // No fetches, arbitrary server paths, nested directories, query credentials, or symlink escapes.
        if (!url.StartsWith("/uploads/", StringComparison.Ordinal) || url.IndexOfAny(new[] { '?', '#' }) >= 0)
            throw new StudioException("invalid_media_url", "Only this host's /uploads/ media can be opened from chat. Import remote files explicitly.");
        string name = Uri.UnescapeDataString(url.Substring("/uploads/".Length));
        if (string.IsNullOrWhiteSpace(name) || name is "." or ".." || name.IndexOfAny(new[] { '/', '\\', '\0' }) >= 0)
            throw new StudioException("invalid_media_url", "Invalid chat upload name.");
        string path = Path.Combine(_uploads, name);
        var file = new FileInfo(path);
        if (!file.Exists) throw new StudioException("missing_media", "The chat upload has expired or is missing. Import the original file again.");
        if (file.LinkTarget != null) throw new StudioException("invalid_media_url", "Symbolic links are not accepted as chat media.");
        return path;
    }

    public static int Status(Exception error) => error switch
    {
        StudioException e when e.Code.Contains("revision", StringComparison.Ordinal) || e.Code.Contains("conflict", StringComparison.Ordinal) => 409,
        StudioException e when e.Code.Contains("missing", StringComparison.Ordinal) || e.Code.Contains("not_found", StringComparison.Ordinal) => 404,
        StudioException => 400,
        FileNotFoundException or DirectoryNotFoundException => 404,
        JsonException or ArgumentException or KeyNotFoundException => 400,
        NotSupportedException => 400,
        UnauthorizedAccessException => 403,
        _ => 500
    };

    public static bool AllowsMutation(string marker, string origin, string scheme, string authority)
    {
        if (marker != "1") return false;
        // Native/API clients omit Origin. Browser callers must match the origin of this host.
        return string.IsNullOrEmpty(origin) || Uri.TryCreate(origin, UriKind.Absolute, out var uri)
            && uri.Scheme.Equals(scheme, StringComparison.OrdinalIgnoreCase)
            && uri.Authority.Equals(authority, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Standalone server Studio is local-only unless an operator explicitly enables protected remote deployment.</summary>
    public static bool AllowsServerConnection([System.Diagnostics.CodeAnalysis.AllowNull] IPAddress peer, string host, bool allowRemote = false)
    {
        if (allowRemote) return true; // This opt-in does not authenticate the caller; deployment must do that separately.
        if (peer == null || !IPAddress.IsLoopback(peer.IsIPv4MappedToIPv6 ? peer.MapToIPv4() : peer)) return false;
        if (string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase)) return true;
        if (string.IsNullOrEmpty(host)) return false;
        if (host.StartsWith("[", StringComparison.Ordinal) && host.EndsWith("]", StringComparison.Ordinal)) host = host[1..^1];
        return IPAddress.TryParse(host, out var address)
            && IPAddress.IsLoopback(address.IsIPv4MappedToIPv6 ? address.MapToIPv4() : address);
    }
}
