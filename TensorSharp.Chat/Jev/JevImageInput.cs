// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace TensorSharp.Server.Jev;

/// <summary>
/// Decodes the inline images of a <c>/v1/systemone</c> request and materializes them for the
/// vision tower.
///
/// <para>Pixels must arrive IN the request body, as a <c>data:</c> URL or bare base64. A remote
/// URL is never fetched and a filesystem path is never read: an endpoint that took either would
/// let any client of the inference server read the host's files or reach its network, and the
/// chat endpoints refuse both for the same reason.</para>
///
/// <para>Decoded bytes are written content-addressed into the same upload storage the chat
/// endpoints use, so the operator's quota and TTL apply, resending the same picture stores one
/// file, and the vision-embedding cache (keyed by content) encodes it once.</para>
/// </summary>
internal static class JevImageInput
{
    /// <summary>Largest decoded image accepted. The request-body cap bounds the base64 form
    /// already; this bounds what one entry may expand to.</summary>
    internal const int MaxImageBytes = 16 * 1024 * 1024;

    /// <summary>Decode one <c>images</c> entry. Format is sniffed from the leading bytes, never
    /// taken from the declared media type, because clients routinely mislabel (a phone's
    /// ".jpg" is often HEIC).</summary>
    internal static JevImage Decode(JsonElement value, int index)
    {
        if (value.ValueKind != JsonValueKind.String)
            throw Error(index, "must be a base64 string or a data: URL");
        string text = (value.GetString() ?? string.Empty).Trim();
        if (text.Length == 0) throw Error(index, "must not be empty");

        if (text.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            text.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            throw Error(index, "remote image URLs are not fetched; send the image as base64 or a data: URL");
        if (text.StartsWith("file://", StringComparison.OrdinalIgnoreCase))
            throw Error(index, "file paths are not read; send the image as base64 or a data: URL");

        string payload = text;
        if (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            int comma = text.IndexOf(',');
            if (comma < 0) throw Error(index, "malformed data: URL");
            string header = text[5..comma];
            if (!header.Contains(";base64", StringComparison.OrdinalIgnoreCase))
                throw Error(index, "only base64 data: URLs are supported");
            payload = text[(comma + 1)..];
        }

        byte[] bytes;
        try { bytes = Convert.FromBase64String(payload); }
        catch (FormatException)
        {
            // A path is the likely mistake here and base64's alphabet contains '/', so it
            // cannot be ruled out before decoding (a JPEG data: URL starts "/9j/"). Say which
            // of the two failures happened rather than reporting "invalid base64" for a path.
            throw Error(index, LooksLikePath(text)
                ? "file paths are not read; send the image as base64 or a data: URL"
                : "is not valid base64");
        }
        if (bytes.Length == 0) throw Error(index, "decodes to an empty image");
        if (bytes.Length > MaxImageBytes)
            throw Error(index, $"decodes to {bytes.Length} bytes; the limit is {MaxImageBytes}");

        string format = SniffFormat(bytes)
            ?? throw Error(index, "is not a recognized image; PNG, JPEG, GIF, BMP, WebP, TIFF and HEIC are accepted");
        return new JevImage(bytes, format);
    }

    /// <summary>Write each decoded image into <paramref name="storage"/> and return the paths in
    /// request order, which is the order their soft tokens occupy in the prompt.</summary>
    internal static string[] Materialize(IReadOnlyList<JevImage> images, UploadStoragePolicy storage)
    {
        ArgumentNullException.ThrowIfNull(storage);
        if (images.Count == 0) return Array.Empty<string>();
        var paths = new string[images.Count];
        for (int i = 0; i < images.Count; i++)
            paths[i] = ChatMessageParser.WriteContentAddressed(images[i].Bytes, "." + images[i].Format, storage);
        return paths;
    }

    /// <summary>
    /// The container formats the image processors decode, told apart by their leading bytes.
    /// <c>TensorSharp.Models.Media.ImageFormatSniffer</c> is the authority that actually
    /// dispatches decoding; this check only fails a non-image fast, with a 422 naming the
    /// reason, instead of letting it reach the vision tower as a decode failure.
    /// </summary>
    internal static string SniffFormat(ReadOnlySpan<byte> d)
    {
        if (d.Length >= 8 && d[0] == 0x89 && d[1] == 0x50 && d[2] == 0x4E && d[3] == 0x47 &&
            d[4] == 0x0D && d[5] == 0x0A && d[6] == 0x1A && d[7] == 0x0A) return "png";
        if (d.Length >= 2 && d[0] == 0xFF && d[1] == 0xD8) return "jpeg";
        if (IsHeif(d)) return "heic";
        if (d.Length >= 6 && d[0] == (byte)'G' && d[1] == (byte)'I' && d[2] == (byte)'F' &&
            d[3] == (byte)'8' && (d[4] == (byte)'7' || d[4] == (byte)'9') && d[5] == (byte)'a') return "gif";
        if (d.Length >= 12 && d[0] == (byte)'R' && d[1] == (byte)'I' && d[2] == (byte)'F' && d[3] == (byte)'F' &&
            d[8] == (byte)'W' && d[9] == (byte)'E' && d[10] == (byte)'B' && d[11] == (byte)'P') return "webp";
        if (d.Length >= 4 && ((d[0] == (byte)'I' && d[1] == (byte)'I' && d[2] == 0x2A && d[3] == 0x00) ||
                              (d[0] == (byte)'M' && d[1] == (byte)'M' && d[2] == 0x00 && d[3] == 0x2A))) return "tiff";
        // "BM" is a weak two-byte magic, so every stronger signature gets first refusal.
        if (d.Length >= 14 && d[0] == (byte)'B' && d[1] == (byte)'M') return "bmp";
        return null;
    }

    /// <summary>ISOBMFF container ("ftyp" box) whose major or compatible brand is a HEIF one,
    /// the same test <c>ImageFormatSniffer.IsHeic</c> applies before handing the bytes to the
    /// platform image provider.</summary>
    private static bool IsHeif(ReadOnlySpan<byte> d)
    {
        if (d.Length < 12 || d[4] != (byte)'f' || d[5] != (byte)'t' || d[6] != (byte)'y' || d[7] != (byte)'p')
            return false;
        int boxSize = (d[0] << 24) | (d[1] << 16) | (d[2] << 8) | d[3];
        if (boxSize <= 0 || boxSize > d.Length) boxSize = d.Length;
        if (IsHeifBrand(d, 8)) return true;
        for (int offset = 16; offset + 4 <= boxSize; offset += 4)
            if (IsHeifBrand(d, offset)) return true;
        return false;
    }

    private static bool IsHeifBrand(ReadOnlySpan<byte> d, int offset)
    {
        if (offset < 0 || offset + 4 > d.Length) return false;
        return System.Text.Encoding.ASCII.GetString(d.Slice(offset, 4)) switch
        {
            "heic" or "heix" or "heim" or "heis" or
            "hevc" or "hevx" or "hevm" or "hevs" or
            "mif1" or "msf1" or "heif" => true,
            _ => false,
        };
    }

    private static bool LooksLikePath(string text)
        => text.Length <= 4096 && text.IndexOfAny([' ', '\t', '\n']) < 0 &&
           (text.Contains('/', StringComparison.Ordinal) || text.Contains('\\', StringComparison.Ordinal)) &&
           text.Contains('.', StringComparison.Ordinal);

    private static JevValidationException Error(int index, string message)
        => new($"images[{index}]: {message}");
}
