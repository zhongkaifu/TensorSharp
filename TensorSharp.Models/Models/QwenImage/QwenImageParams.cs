// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Interpretation of the explicit local editing mask.</summary>
    public enum QwenImageMaskMode
    {
        /// <summary>RGB luminance: white edits, black preserves. Alpha is ignored.</summary>
        Grayscale,
        /// <summary>Inverse alpha: transparent edits, opaque preserves. RGB is ignored.</summary>
        Alpha
    }

    /// <summary>
    /// Qwen-Image-2.1 sampling parameters.
    /// Zero-valued step/CFG settings select the model's defaults.
    /// </summary>
    public sealed class QwenImageParams
    {
        /// <summary>Number of denoising (FlowMatch Euler) steps. 0 = auto: 40 steps.</summary>
        public int Steps { get; set; } = 0;

        /// <summary>
        /// Classifier-free guidance scale; &lt;= 1 disables the negative pass (single forward/step).
        /// 0 = auto: 1.0 (the checkpoint's recommended unguided sampling).
        /// Explicit guidance above 1 uses standard CFG without per-token renormalization.
        /// </summary>
        public float CfgScale { get; set; } = 0f;

        /// <summary>Negative prompt for the CFG pass (empty = unconditional).</summary>
        public string NegativePrompt { get; set; } = " ";

        /// <summary>
        /// Philox key of the initial noise. Text-to-image noise depends on the seed and size alone
        /// (stable-diffusion.cpp --rng cuda). An edit's noise also depends on its reference images,
        /// so an edit never restarts from the noise that drew its source; TS_QWEN21_EDIT_NOISE=seed
        /// restores seed-only edit noise for matched-noise comparisons.
        /// </summary>
        public long Seed { get; set; } = 0;

        /// <summary>
        /// Target output area in pixels (aspect ratio follows the input image).
        /// 0 = the model's native 2048² area, except on the pure-C# cpu backend, where the
        /// automatic area is 1024² (see QwenImage21Pipeline.HostCpuAutomaticArea).
        /// Dimensions are snapped to multiples of 32.
        /// An explicit positive area takes precedence over the model default; a request that
        /// passes exactly the native 2048² area cannot be told from an automatic one (the
        /// server resolves omitted areas to it), so ask for 2048² on a CPU with width/height.
        /// </summary>
        public long TargetArea { get; set; } = 0;

        /// <summary>Resolve the automatic output area, retaining an explicit area.</summary>
        public long ResolveTargetArea() =>
            TargetArea > 0 ? TargetArea : 2048L * 2048;

        /// <summary>Optional explicit output width/height override (0 = derive from input + TargetArea).</summary>
        public int Width { get; set; } = 0;
        public int Height { get; set; } = 0;

        /// <summary>
        /// Edit only: return the first reference image's exact width and height, so an edit of
        /// an edit keeps the picture's size. Sampling stays within the area the request would
        /// otherwise use (TargetArea, the server's default size or the automatic area) and
        /// within the source's own area, at the source's aspect ratio; the decoded result is
        /// resized to the source when the 32-pixel grid or that cap differs from it. Masked
        /// edits already return the source canvas. Cannot be combined with Width/Height, which
        /// choose the output size themselves, and requires a reference image.
        /// </summary>
        public bool KeepSourceSize { get; set; } = false;

        /// <summary>
        /// Optional mask matching the first reference image's exact dimensions. Masked edits
        /// return that image's original dimensions and preserve unselected pixels exactly.
        /// Width/Height and TargetArea control sampling resolution, before restoring the canvas.
        /// </summary>
        public RgbImage Mask { get; set; }

        /// <summary>Explicit mask interpretation; default is white edits, black preserves.</summary>
        public QwenImageMaskMode MaskMode { get; set; } = QwenImageMaskMode.Grayscale;

        /// <summary>Invert the selection after interpreting its grayscale or alpha channel.</summary>
        public bool MaskInvert { get; set; } = false;

        /// <summary>Inward box feather radius in source pixels (0..1024). Never expands the selection.</summary>
        public int MaskFeather { get; set; } = 0;

        /// <summary>
        /// Sample only the selection bounds plus context, reducing target and first-reference tokens.
        /// Sampling resolution scales with crop area; additional references remain available.
        /// Final output is composited back onto the original canvas. Default false retains full context.
        /// </summary>
        public bool MaskCrop { get; set; } = false;

        /// <summary>Context padding around a cropped selection, in source pixels (0..16384).</summary>
        public int MaskCropPadding { get; set; } = 64;

        /// <summary>
        /// Optional per-step progress callback for live UI feedback during the denoise loop.
        /// Invoked once after every step as <c>(step, totalSteps, preview)</c> where <c>step</c> is
        /// 1-based and <c>preview</c> is a decoded RGB snapshot of the current (partially denoised)
        /// latent on throttled steps, or <c>null</c> on the steps in between (a progress-only tick).
        /// </summary>
        public Action<int, int, RgbImage> OnStep { get; set; }

        /// <summary>
        /// How many decoded image previews to emit across the denoise loop (0 = progress ticks only,
        /// no decode). Previews are spaced evenly and decoded at reduced resolution to keep the
        /// per-preview VAE cost (and VRAM) small relative to the denoise itself.
        /// Masked previews are composited back to the original canvas with exact protected pixels.
        /// </summary>
        public int PreviewCount { get; set; } = 0;
    }
}
