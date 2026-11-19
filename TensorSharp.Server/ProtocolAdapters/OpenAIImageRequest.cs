// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Chat;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>
/// The OpenAI image routes' requests, in the shapes SGLang serves them for MiniMax-H3-Image (MiniMax's examples/t2i.py and
/// examples/edit.py): <c>/v1/images/generations</c> as JSON, <c>/v1/images/edits</c> as multipart with one <c>image</c>
/// part a reference (repeated, in <c>&lt;Picture n&gt;</c> order; <c>image[]</c> as the OpenAI SDKs name them) and an
/// optional <c>mask</c>. Their fields map onto the image routes' own parameters (WebUiChatService.ParseImageParameters
/// validates them): <c>prompt</c>; <c>size</c> <c>WIDTHxHEIGHT</c> (or <c>auto</c>) to width and height;
/// <c>seed</c>; <c>num_inference_steps</c> to steps; <c>guidance_scale</c> to cfg; <c>negative_prompt</c> to
/// negativePrompt. <c>n</c> (1-10) pictures are made one after another, picture i at seed + i. <c>response_format</c> is
/// <c>url</c> (the default) or <c>b64_json</c>; <c>output_format</c> must be png. Other fields SGLang clients send
/// (model, generator_device, quality, user, ...) are accepted and ignored.
/// </summary>
internal sealed record OpenAIImageRequest(JsonElement Parameters, int Count, bool Base64, long Seed)
{
    /// <summary>The most pictures one request makes.</summary>
    internal const int MaxCount = 10;

    /// <summary>The image routes' parameters for picture <paramref name="index"/> (its seed is <see cref="Seed"/> + index).</summary>
    internal JsonElement ParametersFor(int index)
    {
        if (index == 0) return Parameters;
        var fields = new Dictionary<string, object>();
        foreach (JsonProperty property in Parameters.EnumerateObject()) fields[property.Name] = property.Value;
        fields["seed"] = Seed + index;
        return JsonSerializer.SerializeToElement(fields);
    }

    /// <summary>A <c>/v1/images/generations</c> body.</summary>
    internal static OpenAIImageRequest FromJson(JsonElement body)
    {
        if (body.ValueKind != JsonValueKind.Object)
            throw Invalid("Expected a JSON object.");
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (JsonProperty property in body.EnumerateObject())
        {
            switch (property.Value.ValueKind)
            {
                case JsonValueKind.String: values[property.Name] = property.Value.GetString(); break;
                case JsonValueKind.Number: values[property.Name] = property.Value.GetRawText(); break;
                case JsonValueKind.Null: break;
                default:
                    if (Mapped.Contains(property.Name)) throw Invalid($"{property.Name} must be a string or a number.");
                    break;
            }
        }
        return FromFields(values);
    }

    /// <summary>A <c>/v1/images/edits</c> form's fields (each one value).</summary>
    internal static OpenAIImageRequest FromForm(IFormCollection form)
    {
        ArgumentNullException.ThrowIfNull(form);
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, value) in form)
        {
            if (value.Count != 1)
            {
                if (Mapped.Contains(name)) throw Invalid($"{name} must have one value.");
                continue;
            }
            values[name] = value[0];
        }
        return FromFields(values);
    }

    /// <summary>The reference images of an edit form, in part order (<c>image</c> or <c>image[]</c>), and its mask.</summary>
    internal static (IReadOnlyList<IFormFile> Images, IFormFile Mask) FilesOf(IFormCollection form, long maxFileBytes)
    {
        ArgumentNullException.ThrowIfNull(form);
        IFormFile[] images = form.Files.Where(f => f.Name is "image" or "image[]").ToArray();
        if (images.Length == 0)
            throw Invalid("No image uploaded (field 'image').");
        IReadOnlyList<IFormFile> masks = form.Files.GetFiles("mask");
        if (masks.Count > 1)
            throw Invalid("Only one mask file is allowed; it applies to the first image.");
        foreach (IFormFile file in images.Concat(masks))
            if (file.Length == 0 || file.Length > maxFileBytes)
                throw new WebUiRequestRejectedException(file.Length == 0 ? 400 : 413,
                    new { error = "An image or the mask is empty or exceeds the upload size limit." });
        return (images, masks.Count == 1 ? masks[0] : null);
    }

    private static readonly HashSet<string> Mapped = new(StringComparer.Ordinal)
    {
        "prompt", "size", "seed", "n", "response_format", "output_format", "num_inference_steps", "guidance_scale",
        "negative_prompt",
    };

    private static OpenAIImageRequest FromFields(IReadOnlyDictionary<string, string> values)
    {
        var fields = new Dictionary<string, object>();
        if (!values.TryGetValue("prompt", out string prompt) || string.IsNullOrWhiteSpace(prompt))
            throw Invalid("prompt is required.");
        fields["prompt"] = prompt;

        if (values.TryGetValue("size", out string size) && !string.IsNullOrWhiteSpace(size)
            && !size.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            string[] parts = size.Trim().Split('x', 'X');
            if (parts.Length != 2
                || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int width)
                || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int height)
                || width <= 0 || height <= 0)
                throw Invalid($"size must be WIDTHxHEIGHT (for example 2048x2048) or auto, not '{size}'.");
            fields["width"] = width;
            fields["height"] = height;
        }

        long seed = 0;
        if (values.TryGetValue("seed", out string seedText))
        {
            if (!long.TryParse(seedText, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out seed))
                throw Invalid("seed must be an integer.");
            fields["seed"] = seed;
        }

        int count = 1;
        if (values.TryGetValue("n", out string countText)
            && (!int.TryParse(countText, NumberStyles.None, CultureInfo.InvariantCulture, out count) || count < 1 || count > MaxCount))
            throw Invalid($"n must be an integer from 1 to {MaxCount}.");

        if (values.TryGetValue("num_inference_steps", out string steps))
        {
            if (!int.TryParse(steps, NumberStyles.None, CultureInfo.InvariantCulture, out int s))
                throw Invalid("num_inference_steps must be a nonnegative integer.");
            fields["steps"] = s;
        }
        if (values.TryGetValue("guidance_scale", out string cfg))
        {
            if (!float.TryParse(cfg, NumberStyles.Float, CultureInfo.InvariantCulture, out float c))
                throw Invalid("guidance_scale must be a number.");
            fields["cfg"] = c;
        }
        if (values.TryGetValue("negative_prompt", out string negative) && !string.IsNullOrWhiteSpace(negative))
            fields["negativePrompt"] = negative;

        bool base64 = false;
        if (values.TryGetValue("response_format", out string format))
            base64 = format switch
            {
                "b64_json" => true,
                "url" => false,
                _ => throw Invalid("response_format must be url or b64_json."),
            };
        if (values.TryGetValue("output_format", out string output) && !output.Equals("png", StringComparison.OrdinalIgnoreCase))
            throw Invalid("output_format must be png.");

        JsonElement parameters = JsonSerializer.SerializeToElement(fields);
        // The image routes' validation, before any upload is decoded or the model is asked.
        WebUiChatService.ParseImageParameters(parameters);
        return new OpenAIImageRequest(parameters, count, base64, seed);
    }

    private static WebUiRequestRejectedException Invalid(string message) => new(400, new { error = message });
}
