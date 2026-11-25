// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// A text-to-image (and optionally image-edit) model, as the image routes and the CLI
    /// drive it: a prompt and <see cref="QwenImageParams"/> in, an <see cref="RgbImage"/> out.
    /// Qwen-Image-2.1 (<c>QwenImage21DiffusionModel</c>) and MiniMax-H3-Image
    /// (<c>MiniMaxH3ImageDiffusionModel</c>), both on the shared diffusion runtime, implement it; settings a model does not take (a negative
    /// prompt, CFG above 1, a mask) are refused by that model rather than ignored.
    /// </summary>
    public interface IImageGenerationModel
    {
        /// <summary>The architecture id, for logs and refusals ("qwen_image", "minimax-h3-image").</summary>
        string ImageModelFamily { get; }

        /// <summary>Whether <see cref="EditImage"/> is available.</summary>
        bool SupportsImageEdit { get; }

        /// <summary>Whether <see cref="EditImage"/> takes a mask (<see cref="QwenImageParams.Mask"/> and its settings),
        /// which limits the edit to a selection of the first picture; a model that does not refuses one.</summary>
        bool SupportsEditMask => false;

        /// <summary>
        /// The smallest output area the model makes its pictures at as it was trained (diffusers' min_generation_area), or 0
        /// for none: MiniMax-H3-Image's is 2048 x 2048 (4 MP; an edit below it samples there and is resized back). A host
        /// that picks the size for its users (TensorAgent's chat turns) asks for at least this.
        /// </summary>
        long MinimumGenerationArea => 0;

        RgbImage GenerateImage(string prompt, QwenImageParams p = null);

        RgbImage EditImage(string prompt, IReadOnlyList<RgbImage> inputs, QwenImageParams p);
    }
}
