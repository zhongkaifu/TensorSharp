// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models.Embeddings;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class EmbeddingAdapter
{
    private async Task<EmbeddingBatchResult> EmbedMediaAsync(JsonElement input, CancellationToken cancellationToken)
    {
        if (_multimodal == null)
            throw Invalid("Media embeddings require EmbeddingGemma 2 with --mmproj <projector.gguf>.", "input");
        var items = input.ValueKind == JsonValueKind.Object ? new[] { input } : ReadItems(input);
        if (items.Length == 0 || items.Length > MaxBatchInputs)
            throw Invalid($"A request may contain between 1 and {MaxBatchInputs} inputs.", "input");
        string directory = Path.Combine(Path.GetTempPath(), "tensorsharp-embedding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var sequences = new List<EmbeddingGemma2InputPart>[items.Length];
            for (int i = 0; i < items.Length; i++)
            {
                var item = items[i];
                if (item.ValueKind != JsonValueKind.Object || !item.TryGetProperty("content", out var content) ||
                    content.ValueKind != JsonValueKind.Array || content.GetArrayLength() == 0)
                    throw Invalid("Every media input must be an object with a nonempty content array.", "input");
                var parts = sequences[i] = new();
                foreach (var part in content.EnumerateArray())
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (part.ValueKind != JsonValueKind.Object || !part.TryGetProperty("type", out var type) || type.ValueKind != JsonValueKind.String)
                        throw Invalid("Every content part must specify a type.", "input");
                    switch (type.GetString())
                    {
                        case "text":
                            parts.Add(EmbeddingGemma2InputPart.Text(RequiredString(part, "text")));
                            break;
                        case "image_url":
                            parts.Add(EmbeddingGemma2InputPart.Image(SaveDataUri(RequiredString(RequiredObject(part, "image_url"), "url"), directory, "image")));
                            break;
                        case "input_audio":
                            var audio = RequiredObject(part, "input_audio");
                            string format = audio.TryGetProperty("format", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString()! : "wav";
                            string mime = format.ToLowerInvariant() switch
                            {
                                "wav" => "audio/wav", "mp3" => "audio/mpeg", "ogg" => "audio/ogg", "flac" => "audio/flac",
                                _ => throw Invalid("Audio format must be wav, mp3, ogg, or flac.", "input"),
                            };
                            parts.Add(EmbeddingGemma2InputPart.Audio(SaveDataUri("data:" + mime + ";base64," + RequiredString(audio, "data"), directory, "audio")));
                            break;
                        case "input_video":
                            parts.Add(EmbeddingGemma2InputPart.Video(SaveDataUri(RequiredString(RequiredObject(part, "input_video"), "url"), directory, "video")));
                            break;
                        default:
                            throw Invalid("Content type must be text, image_url, input_audio, or input_video.", "input");
                    }
                }
            }
            var vectors = new float[items.Length][];
            int totalTokens = 0;
            for (int i = 0; i < sequences.Length; i++)
            {
                var result = await _multimodal.EmbedAsync(sequences[i], cancellationToken).ConfigureAwait(false);
                totalTokens = checked(totalTokens + result.PromptTokens);
                if (totalTokens > MaxBatchTokens)
                    throw Invalid($"A request may contain at most {MaxBatchTokens} tokens.", "input");
                vectors[i] = result.Embeddings[0];
            }
            return new(vectors, totalTokens);
        }
        catch (FormatException)
        {
            throw Invalid("Media payload must contain valid base64 data.", "input");
        }
        catch (Exception ex) when (ex is InvalidDataException or TensorSharp.Models.Media.MediaInputException)
        {
            throw Invalid(ex.Message, "input");
        }
        catch (NotSupportedException ex)
        {
            throw Invalid(ex.Message, "input");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static JsonElement[] ReadItems(JsonElement input)
    {
        var items = new JsonElement[input.GetArrayLength()];
        int i = 0;
        foreach (var item in input.EnumerateArray()) items[i++] = item;
        return items;
    }

    private static JsonElement RequiredObject(JsonElement value, string name) =>
        value.TryGetProperty(name, out var child) && child.ValueKind == JsonValueKind.Object
            ? child : throw Invalid($"Content part requires an object '{name}'.", "input");

    private static string RequiredString(JsonElement value, string name) =>
        value.TryGetProperty(name, out var child) && child.ValueKind == JsonValueKind.String
            ? child.GetString()! : throw Invalid($"Content part requires a string '{name}'.", "input");

    internal static string SaveDataUri(string uri, string directory, string modality)
    {
        int comma = uri.IndexOf(',');
        if (comma < 0 || !uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ||
            !uri.AsSpan(0, comma).EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
            throw Invalid("Media URLs must be base64 data URIs.", "input");
        string mime = uri[5..(comma - 7)].ToLowerInvariant();
        string extension = (modality, mime) switch
        {
            ("image", "image/png") => ".png", ("image", "image/jpeg") => ".jpg",
            ("image", "image/webp") => ".webp", ("image", "image/gif") => ".gif",
            ("audio", "audio/wav" or "audio/x-wav") => ".wav", ("audio", "audio/mpeg") => ".mp3",
            ("audio", "audio/ogg") => ".ogg", ("audio", "audio/flac") => ".flac",
            ("video", "video/mp4") => ".mp4", ("video", "video/webm") => ".webm",
            _ => throw Invalid($"Unsupported {modality} media type.", "input"),
        };
        byte[] bytes = Convert.FromBase64String(uri[(comma + 1)..]);
        if (bytes.Length == 0) throw Invalid("Media payload must not be empty.", "input");
        string path = Path.Combine(directory, Guid.NewGuid().ToString("N") + extension);
        File.WriteAllBytes(path, bytes);
        return path;
    }
}
