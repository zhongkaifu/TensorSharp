// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Globalization;
using System.IO;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Cli;

/// <summary>Mask options shared by command-line and config-file invocations.</summary>
internal sealed class ImageMaskCliOptions
{
    internal string Path { get; private set; }
    internal QwenImageMaskMode Mode { get; private set; } = QwenImageMaskMode.Grayscale;
    internal bool Invert { get; private set; }
    internal int Feather { get; private set; }
    internal bool Crop { get; private set; }
    internal int CropPadding { get; private set; } = 64;
    private bool _configured;

    internal void Read(string[] args, ref int i)
    {
        string flag = args[i];
        _configured = true;
        if (flag == "--mask-crop") { Crop = true; return; }
        if (flag == "--mask-invert") { Invert = true; return; }
        if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException($"{flag} requires a value.");
        string value = args[++i];
        switch (flag)
        {
            case "--mask":
                if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("--mask requires a file path.");
                Path = value;
                break;
            case "--mask-mode":
                Mode = value.ToLowerInvariant() switch
                {
                    "alpha" => QwenImageMaskMode.Alpha,
                    "grayscale" => QwenImageMaskMode.Grayscale,
                    _ => throw new ArgumentException("--mask-mode must be alpha (transparent edits) or grayscale (white edits)."),
                };
                break;
            case "--mask-feather": Feather = ParsePixels(flag, value, 1024); break;
            case "--mask-crop-padding": CropPadding = ParsePixels(flag, value, 16384); break;
            default: throw new ArgumentException($"Unknown mask option: {flag}");
        }
    }

    private static int ParsePixels(string flag, string value, int maximum)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int pixels)
            || pixels < 0 || pixels > maximum)
            throw new ArgumentException($"{flag} must be an integer from 0 to {maximum}.");
        return pixels;
    }

    internal void Validate(int imageCount)
    {
        if (!_configured) return;
        if (Path == null) throw new ArgumentException("Mask options require --mask <file>.");
        if (imageCount == 0) throw new ArgumentException("--mask requires --image; the mask applies to the first image.");
        if (!File.Exists(Path)) throw new ArgumentException($"Mask file not found: {Path}");
    }

    internal void ValidateModel(bool takesMasks)
    {
        if (_configured && !takesMasks)
            throw new ArgumentException("--mask requires an image model that takes edit masks (Qwen-Image-2.1, MiniMax-H3-Image).");
    }

    internal void Apply(QwenImageParams parameters)
    {
        if (Path == null) return;
        try { parameters.Mask = ImageIO.Load(Path, preserveAlpha: true); }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        { throw new ArgumentException("Cannot decode mask image: " + ex.Message, ex); }
        parameters.MaskMode = Mode;
        parameters.MaskInvert = Invert;
        parameters.MaskFeather = Feather;
        parameters.MaskCrop = Crop;
        parameters.MaskCropPadding = CropPadding;
    }
}
