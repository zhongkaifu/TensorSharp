// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Qwen-Image's binding of the shared <see cref="EditMask"/>: the mask settings of a <see cref="QwenImageParams"/>,
    /// and reinjection over Qwen-Image-2.1's latent tokens (one mask weight a token, 64 contiguous channels).
    /// </summary>
    internal static class QwenImageEditMask
    {
        /// <summary>Latent channels a Qwen-Image-2.1 token holds, which one mask weight covers.</summary>
        internal const int TokenChannels = 64;

        internal static EditMask Create(QwenImageParams p, RgbImage source) => EditMask.Create(Options(p), source);

        /// <summary>The mask settings of <paramref name="p"/>. A mode value outside the enum stays outside the shared
        /// one, so <see cref="EditMask.Create"/> refuses it the same way.</summary>
        internal static EditMaskOptions Options(QwenImageParams p)
        {
            ArgumentNullException.ThrowIfNull(p);
            return new EditMaskOptions
            {
                Mask = p.Mask,
                MaskMode = (EditMaskMode)(int)p.MaskMode,
                MaskInvert = p.MaskInvert,
                MaskFeather = p.MaskFeather,
                MaskCrop = p.MaskCrop,
                MaskCropPadding = p.MaskCropPadding,
            };
        }

        /// <summary>Fixed-seed flow reinjection at the UPDATED state's sigma (<see cref="EditMask.Reinject"/>) over
        /// 64-channel tokens.</summary>
        internal static void Reinject(Span<float> latents, ReadOnlySpan<float> source, ReadOnlySpan<float> noise,
            ReadOnlySpan<float> mask, float sigma)
            => EditMask.Reinject(latents, source, noise, mask, TokenChannels, sigma);
    }
}
