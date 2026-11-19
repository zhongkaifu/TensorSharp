// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>What a reference picture is to the model.</summary>
    public enum DiffusionReferenceRole
    {
        /// <summary>A picture the prompt names (<c>&lt;Picture n&gt;</c> in reference order); the first is the one an
        /// edit changes and a mask belongs to.</summary>
        Picture,
    }

    /// <summary>One ordered reference picture.</summary>
    public sealed record DiffusionReference(RgbImage Image, DiffusionReferenceRole Role = DiffusionReferenceRole.Picture);

    /// <summary>
    /// One image (or clip) request. The runtime is stateless across requests: a multi-turn host resolves which pictures
    /// a turn uses and passes them as <see cref="References"/>, whose order defines <c>&lt;Picture n&gt;</c>. A seed and
    /// the same inputs reproduce the same picture. Zero steps or guidance select the recipe's (or the speed adapter's
    /// contract's) values.
    /// </summary>
    public sealed class DiffusionRequest
    {
        public string Prompt { get; init; } = string.Empty;
        public IReadOnlyList<DiffusionReference> References { get; init; } = Array.Empty<DiffusionReference>();

        /// <summary>The mask and its settings; null (or a null <see cref="EditMaskOptions.Mask"/>) for none.</summary>
        public EditMaskOptions Mask { get; init; }

        /// <summary>Explicit output size (both or neither; 0 derives it).</summary>
        public int Width { get; init; }
        public int Height { get; init; }

        /// <summary>Output area in pixels when no size is given (0: the family's default).</summary>
        public long TargetArea { get; init; }

        /// <summary>Return the first reference's exact pixel size.</summary>
        public bool KeepSourceSize { get; init; }

        public long Seed { get; init; }
        public int Steps { get; init; }
        public float Cfg { get; init; }

        /// <summary>The negative branch's prompt (guided families); whitespace is no negative prompt.</summary>
        public string NegativePrompt { get; init; }

        public AdapterSet Adapters { get; init; } = AdapterSet.Empty;

        /// <summary>Decoded previews to emit across the loop (0: progress only).</summary>
        public int PreviewCount { get; init; }

        /// <summary>Receives each phase entered and each finished step (<see cref="DiffusionProgress"/>; a preview is an
        /// <see cref="RgbImage"/>), synchronously, in order.</summary>
        public Action<DiffusionProgress> Progress { get; init; }
    }

    /// <summary>Where a request samples and the size it returns (they differ when the decode is resized: a
    /// generation floor, a kept source size).</summary>
    public readonly record struct DiffusionCanvas((int Width, int Height) Sampling, (int Width, int Height) Output)
    {
        /// <summary>A canvas that returns what it samples.</summary>
        public static DiffusionCanvas Of(int width, int height) => new((width, height), (width, height));

        /// <summary>
        /// Text to image: an explicit width and height (both, multiples of <see cref="DiffusionGeometry.SizeMultiple"/>)
        /// win; otherwise a square of the request's target area (its side rounded to the multiple) or the family's
        /// default size. The canvas returns what it samples.
        /// </summary>
        public static DiffusionCanvas TextToImage(DiffusionRequest request, DiffusionGeometry geometry, string familyName)
        {
            ArgumentNullException.ThrowIfNull(request);
            ArgumentNullException.ThrowIfNull(geometry);
            int multiple = geometry.SizeMultiple, width = request.Width, height = request.Height;
            if (width <= 0 && height <= 0)
            {
                if (request.TargetArea > 0)
                {
                    int side = (int)Math.Round(Math.Sqrt(request.TargetArea) / multiple) * multiple;
                    width = height = Math.Max(multiple, side);
                }
                else
                {
                    width = geometry.DefaultWidth;
                    height = geometry.DefaultHeight;
                }
            }
            else if (width <= 0 || height <= 0)
                throw new ArgumentException($"{familyName} needs both width and height, or neither.");
            if (width % multiple != 0 || height % multiple != 0)
                throw new ArgumentException($"{familyName} sizes are multiples of {multiple}; got {width}x{height}.");
            return Of(width, height);
        }
    }

    /// <summary>Seconds each phase of a request took. <see cref="Text"/> is the conditioning less the vision tower's
    /// share (<see cref="Vision"/>).</summary>
    public sealed record DiffusionTimings
    {
        public double Text { get; init; }
        public double Vision { get; init; }
        public double VaeEncode { get; init; }
        public double Denoise { get; init; }
        public IReadOnlyList<double> Steps { get; init; } = Array.Empty<double>();
        public double Decode { get; init; }
        public double Composite { get; init; }
        public double Total { get; init; }
    }

    /// <summary>What a request actually ran: what every host prints and records.</summary>
    public sealed record DiffusionEffectiveRecipe
    {
        public required int Steps { get; init; }
        public required float[] Sigmas { get; init; }
        public required float Cfg { get; init; }
        public required long Seed { get; init; }
        public required DiffusionCanvas Canvas { get; init; }
        public required SamplingRecipe Recipe { get; init; }

        /// <summary>The adapters' names, in order.</summary>
        public IReadOnlyList<string> Adapters { get; init; } = Array.Empty<string>();
    }

    /// <summary>A finished request: the image, what ran, how long each phase took, and the residency plan it ran under.</summary>
    public sealed class DiffusionResult
    {
        public required RgbImage Image { get; init; }
        public required DiffusionEffectiveRecipe Recipe { get; init; }
        public required DiffusionTimings Timings { get; init; }

        /// <summary>The plan, or null when the request ran no model (an empty mask).</summary>
        public ResidencyPlan Residency { get; init; }
    }
}
