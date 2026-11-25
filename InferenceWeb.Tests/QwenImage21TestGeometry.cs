// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen-Image-2.1's geometry and sampling as a host's parameters reach them: each <see cref="QwenImageParams"/> becomes the
/// model's own runtime request (<see cref="QwenImage21DiffusionModel.RequestOf"/>), resolved by the family's canvas
/// (<see cref="QwenImage21Canvas"/>) and recipe, and the decoded image finished as the pipeline finishes it.
/// </summary>
internal static class QwenImage21TestGeometry
{
    internal const long KeptSourceSamplingFloor = QwenImage21Canvas.KeptSourceSamplingFloor;

    private static DiffusionRequest Request(QwenImageParams p, RgbImage reference) =>
        QwenImage21DiffusionModel.RequestOf("", p, reference == null ? Array.Empty<RgbImage>() : new[] { reference });

    internal static (int Width, int Height) ResolveDimensions(QwenImageParams p, RgbImage reference, BackendType? backend = null) =>
        QwenImage21Canvas.ResolveDimensions(Request(p, reference), reference, backend);

    internal static ((int Width, int Height) Sampling, (int Width, int Height) Output) ResolveGeometry(
        QwenImageParams p, RgbImage reference, EditMask mask, BackendType? backend = null)
    {
        DiffusionCanvas canvas = QwenImage21Canvas.Resolve(Request(p, reference), reference, mask, backend);
        return (canvas.Sampling, canvas.Output);
    }

    internal static (int Width, int Height) ResolveReferenceDimensions(RgbImage image, int outputWidth, int outputHeight) =>
        QwenImage21Canvas.ReferenceDimensions(image, outputWidth, outputHeight);

    internal static bool UsesHostCpuAutomaticSize(QwenImageParams p, BackendType backend) =>
        QwenImage21Canvas.UsesHostCpuAutomaticSize(Request(p, null), backend);

    private static readonly IDiffusionFamily Family = new QwenImage21Family(BackendType.GgmlCpu, QwenImageVariant.Base);

    /// <summary>The decoded image at the request's output size, as the runtime's composite phase makes it.</summary>
    internal static RgbImage FinishOutput(RgbImage decoded, EditMask mask, (int Width, int Height) output) =>
        mask != null ? mask.Composite(decoded, Family.ResizeOutput)
        : decoded.Width == output.Width && decoded.Height == output.Height ? decoded
        : Family.ResizeOutput(decoded, output.Width, output.Height);

    /// <summary>The steps, CFG and sigmas a request samples with on a checkpoint whose own recipe is <paramref name="recipe"/>
    /// (Turbo's, or null for the base checkpoint), as the runtime resolves them.</summary>
    internal static (int Steps, float Cfg, float[] Sigmas) ResolveSampling(QwenImageParams p, QwenImage21LoraRecipe recipe, int imageTokens)
    {
        var family = new QwenImage21Family(BackendType.GgmlCpu,
            ReferenceEquals(recipe, QwenImage21Turbo.Recipe) ? QwenImageVariant.Turbo : QwenImageVariant.Base);
        if (recipe != null && !ReferenceEquals(recipe, QwenImage21Turbo.Recipe)) family.LoraRecipe = family.RecipeOf(recipe);
        SamplingRecipe r = family.DefaultRecipe;
        DiffusionRequest request = QwenImage21DiffusionModel.RequestOf("", p, Array.Empty<RgbImage>());
        int steps = r.ResolveSteps(request.Steps);
        float cfg = request.Cfg > 0 ? request.Cfg : r.Cfg;
        return (steps, cfg, r.SigmasFor(steps, imageTokens));
    }
}
