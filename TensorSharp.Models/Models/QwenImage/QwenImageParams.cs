// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Sampling / generation parameters for a Qwen-Image-Edit run. Defaults track the
    /// reference <c>QwenImageEditPlusPipeline</c> (FlowMatch Euler, true-CFG).
    /// </summary>
    public sealed class QwenImageParams
    {
        /// <summary>Number of denoising (FlowMatch Euler) steps.</summary>
        public int Steps { get; set; } = 30;

        /// <summary>True-CFG guidance scale; &lt;= 1 disables the negative pass (single forward/step).</summary>
        public float CfgScale { get; set; } = 4.0f;

        /// <summary>Negative prompt for the true-CFG pass (empty = unconditional).</summary>
        public string NegativePrompt { get; set; } = " ";

        public long Seed { get; set; } = 0;

        /// <summary>
        /// Target output area in pixels (aspect ratio follows the input image). The
        /// reference pipeline targets ~1 megapixel; dims are snapped to a multiple of 16.
        /// </summary>
        public long TargetArea { get; set; } = 1024 * 1024;

        /// <summary>Optional explicit output width/height override (0 = derive from input + TargetArea).</summary>
        public int Width { get; set; } = 0;
        public int Height { get; set; } = 0;
    }
}
