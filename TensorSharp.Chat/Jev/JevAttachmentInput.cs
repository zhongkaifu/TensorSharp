// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Jev;

/// <summary>A named inline attachment or an opaque filename returned by /api/upload.</summary>
public sealed record JevAttachment(string Name, string Kind, string Extension, byte[]? Bytes, string? File);

internal static class JevAttachmentInput
{
    internal const int MaxAttachments = 8;
    internal const int MaxAttachmentBytes = 32 * 1024 * 1024;
    internal const long MaxTotalBytes = 64L * 1024 * 1024;

    internal static JevAttachment[] Parse(JsonElement root, long totalBytes)
    {
        var result = new List<JevAttachment>();
        foreach (string field in new[] { "files", "documents", "videos", "audios" })
        {
            if (!root.TryGetProperty(field, out var array) || array.ValueKind == JsonValueKind.Null) continue;
            if (array.ValueKind != JsonValueKind.Array) throw Error(field, "must be an array of attachment objects");
            int index = 0;
            foreach (var value in array.EnumerateArray())
            {
                string label = $"{field}[{index++}]";
                if (result.Count >= MaxAttachments) throw Error(label, $"at most {MaxAttachments} attachments per request");
                if (value.ValueKind != JsonValueKind.Object) throw Error(label, "must be an object with 'name' and 'data', or 'file'");
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (var property in value.EnumerateObject())
                {
                    if (!seen.Add(property.Name)) throw Error(label, $"duplicate field '{property.Name}'");
                    if (property.Name is not ("name" or "data" or "file")) throw Error(label, $"unsupported field '{property.Name}'");
                    if (property.Value.ValueKind != JsonValueKind.String) throw Error(label, $"'{property.Name}' must be a string");
                }
                bool inline = value.TryGetProperty("data", out var data);
                bool uploaded = value.TryGetProperty("file", out var file);
                if (inline == uploaded) throw Error(label, "provide exactly one of 'data' or 'file'");
                string? reference = uploaded ? file.GetString() : null;
                if (uploaded && !IsBareName(reference!)) throw Error(label, "'file' must be a bare upload filename; paths and URLs are not accepted");
                string? name = value.TryGetProperty("name", out var n) ? n.GetString() : reference;
                if (string.IsNullOrWhiteSpace(name) || name.Length > 255 || name.Any(char.IsControl) || name.IndexOfAny(['/', '\\']) >= 0)
                    throw Error(label, "'name' must be a filename of 1 to 255 characters without paths or control characters");
                string extension = Path.GetExtension(reference ?? name).ToLowerInvariant();
                string kind = UploadContentPolicy.Classify(extension);
                if (kind == "unknown") throw Error(label, $"unsupported file type '{extension}'; use text/code, PDF, DOCX, XLSX, PPTX, image, audio or video");
                if (field == "documents" && kind is not ("text" or "pdf" or "document") ||
                    field == "videos" && kind != "video" || field == "audios" && kind != "audio")
                    throw Error(label, $"file type '{extension}' does not match '{field}'");
                byte[]? bytes = inline ? Decode(data.GetString()!, label) : null;
                totalBytes += bytes?.Length ?? 0;
                if (totalBytes > MaxTotalBytes) throw Error(label, $"combined decoded attachments exceed {MaxTotalBytes} bytes");
                result.Add(new(name!, kind, extension, bytes, reference));
            }
        }
        if (totalBytes > MaxTotalBytes) throw Error("attachments", $"combined decoded attachments exceed {MaxTotalBytes} bytes");
        return result.ToArray();
    }

    private static byte[] Decode(string text, string label)
    {
        text = text.Trim();
        if (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            int comma = text.IndexOf(',');
            if (comma < 0 || !text[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw Error(label, "data URLs must use base64 encoding");
            text = text[(comma + 1)..];
        }
        // Check encoded length before allocating a potentially enormous decoded array.
        if (text.Length > ((long)MaxAttachmentBytes + 2) / 3 * 4 + 1024)
            throw Error(label, $"attachment exceeds {MaxAttachmentBytes} bytes");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(text); }
        catch (FormatException) { throw Error(label, "'data' must be base64 or a base64 data URL; paths and remote URLs are not read"); }
        if (bytes.Length == 0 || bytes.Length > MaxAttachmentBytes)
            throw Error(label, $"decoded attachment must contain 1 to {MaxAttachmentBytes} bytes");
        return bytes;
    }

    internal static bool IsBareName(string name) => !string.IsNullOrWhiteSpace(name) && name.Length <= 255 &&
        name is not ("." or "..") && name.IndexOfAny(['/', '\\', ':']) < 0 && !name.Any(char.IsControl) && !Path.IsPathRooted(name);

    private static JevValidationException Error(string label, string message) => new($"{label}: {message}");
}
