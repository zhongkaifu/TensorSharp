// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// Qwen-Image-2.1-Turbo: the 2.1 transformer distilled to 8 Euler steps at CFG 1, with its own
/// fixed schedule, and how a load decides that a GGUF holds it.
/// </summary>
/// <remarks>
/// <para><b>The schedule.</b> <c>sample_sigmas</c> from Qwen/Qwen-Image-2.1-Turbo's
/// <c>model_index.json</c> (revision d65dbc9), then a terminal 0. Its scheduler has shift 1.0 and
/// no dynamic shifting, so the values are used as they are at every resolution (diffusers loads
/// them in place of the scheduler's own; the AtomicChat GGUF card passes the same list to
/// stable-diffusion.cpp as <c>--sigmas</c>).</para>
/// <para><b>Other step counts are refused.</b> Qwen's card: "Setting num_inference_steps alone does
/// not override it ... other schedules have not been evaluated for this checkpoint", and the GGUF
/// card measured 8 steps only. Resampling the eight values for another count would be a schedule
/// nobody trained or measured, so the recipe defines 8 steps and nothing else, like a LoRA recipe
/// with one schedule.</para>
/// <para><b>Identification.</b> The GGUFs have no metadata and Turbo's tensors have the base
/// checkpoint's names and shapes, so the host declares the variant
/// (<see cref="QwenImageVariantFlag"/>). Without a declaration a file name with the word
/// <c>turbo</c> is assumed to be Turbo, the way Wan's step-distilled checkpoints are recognized
/// (<c>WanVideoModel.ParseDistilledSteps</c>), and the load prints the assumption: merges of the
/// Viggle step-distillation LoRA into the base checkpoint are published under the same file names
/// (Abiray/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-GGUF ships qwen_image_2.1_turbo_Q4_K_M.gguf, as
/// does Abiray/Qwen-Image-2.1-Turbo-GGUF), so the name is a guess the line lets the operator check.</para>
/// </remarks>
internal static class QwenImage21Turbo
{
    internal const int Steps = 8;

    /// <summary>The published schedule without its terminal 0.</summary>
    internal static readonly float[] SampleSigmas =
        { 1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f };

    /// <summary>The sampling recipe of the Turbo checkpoint: 8 steps on <see cref="SampleSigmas"/>
    /// unshifted, CFG 1, fp32 timesteps like the base checkpoint and stable-diffusion.cpp.</summary>
    internal static QwenImage21LoraRecipe Recipe { get; } = new()
    {
        Source = "Qwen/Qwen-Image-2.1-Turbo model_index.json sample_sigmas",
        Owner = "Qwen-Image-2.1-Turbo, which was distilled for one published 8-step schedule (Qwen has evaluated no other),",
        LogLabel = "[qwen21] Turbo schedule (model_index.json sample_sigmas)",
        DefaultSteps = Steps,
        Nodes = new Dictionary<int, float[]> { [Steps] = SampleSigmas },
        Shift = QwenImage21SigmaShift.None,
        Cfg = 1f,
    };

    // The word "turbo" on its own: "Qwen-Image-2.1-Turbo-AD-Q4_K", "qwen_image_2.1_turbo_Q8_0".
    private static readonly Regex TurboWord = new(@"(?<![a-z])turbo(?![a-z])", RegexOptions.CultureInvariant);

    /// <summary>Whether <paramref name="fileName"/> names a Turbo checkpoint by the fallback rule.</summary>
    internal static bool NameSaysTurbo(string fileName) =>
        !string.IsNullOrEmpty(fileName) && TurboWord.IsMatch(Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant());

    /// <summary>
    /// The variant of the transformer in <paramref name="ditPath"/>: the declaration in
    /// <paramref name="declared"/> (the <see cref="QwenImageVariantFlag.EnvironmentVariable"/>
    /// value) when there is one, otherwise the file-name rule. <paramref name="note"/> is the
    /// load's line about it.
    /// </summary>
    /// <exception cref="ArgumentException">The declared value is not a variant.</exception>
    internal static QwenImageVariant Resolve(string declared, string ditPath, out string note)
    {
        if (!string.IsNullOrWhiteSpace(declared))
        {
            var variant = QwenImageVariantFlag.Parse(declared, QwenImageVariantFlag.EnvironmentVariable);
            note = $"{QwenImageVariantFlag.Name(variant)} (declared)";
            return variant;
        }
        if (NameSaysTurbo(Path.GetFileName(ditPath)))
        {
            note = "turbo, ASSUMED from the word \"turbo\" in the file name: the GGUF has no metadata that tells " +
                "Qwen-Image-2.1-Turbo from the base checkpoint, and step-distillation LoRA merges are published " +
                $"under the same names. Declare it with {QwenImageVariantFlag.Flag} turbo or base ({QwenImageVariantFlag.EnvironmentVariable}).";
            return QwenImageVariant.Turbo;
        }
        note = "base";
        return QwenImageVariant.Base;
    }
}
