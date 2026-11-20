// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>How a family guides its denoiser.</summary>
    public enum DiffusionGuidance
    {
        /// <summary>Guidance-distilled: one branch, guidance 1, no negative prompt (MiniMax-H3, Qwen-Image Turbo).</summary>
        None,
        /// <summary>Classifier-free guidance over a negative branch: <c>u + cfg * (c - u)</c> above guidance 1.</summary>
        Cfg,
        /// <summary>Several guidance terms (LTX 2.5); a family that declares it combines them in its denoiser.</summary>
        Multi,
    }

    /// <summary>A family's pixel and latent geometry, and the request bounds it accepts.</summary>
    public sealed record DiffusionGeometry
    {
        /// <summary>Latent channels the VAE produces (24 for MiniMax-H3-Image).</summary>
        public required int LatentChannels { get; init; }

        /// <summary>Pixels a latent cell spans on each side (16).</summary>
        public required int VaeFactor { get; init; }

        /// <summary>Latent cells a denoiser token spans on each side (2).</summary>
        public required int Patch { get; init; }

        /// <summary>Output sides are multiples of this (32).</summary>
        public required int SizeMultiple { get; init; }

        /// <summary>The size a request that names none samples at.</summary>
        public required int DefaultWidth { get; init; }
        public required int DefaultHeight { get; init; }

        /// <summary>The smallest area the family samples at (a generation floor; smaller requests sample larger and are
        /// resized down), or 0 for none.</summary>
        public long MinGenerationArea { get; init; }

        /// <summary>The largest area a request may ask for, or 0 for no bound.</summary>
        public long MaxArea { get; init; }

        /// <summary>The most reference images an edit takes (0: the family does not edit).</summary>
        public int MaxReferences { get; init; }

        /// <summary>Widest and tallest aspect (width / height) a reference may have, or 0 for no bound.</summary>
        public double MinAspect { get; init; }
        public double MaxAspect { get; init; }

        /// <summary>Values a denoiser token holds: channels x patch x patch (96).</summary>
        public int ValuesPerToken => LatentChannels * Patch * Patch;

        /// <summary>The latent a <paramref name="width"/> x <paramref name="height"/> picture encodes to.</summary>
        public LatentShape LatentOf(int width, int height)
        {
            if (width <= 0 || height <= 0 || width % VaeFactor != 0 || height % VaeFactor != 0)
                throw new ArgumentException($"{width}x{height} is not a whole number of {VaeFactor}-pixel latent cells.");
            return new LatentShape(LatentChannels, 1, height / VaeFactor, width / VaeFactor);
        }
    }

    /// <summary>A latent's extents: channels, frames, height and width in latent cells.</summary>
    public readonly record struct LatentShape(int Channels, int Frames, int Height, int Width)
    {
        /// <summary>Values the latent holds.</summary>
        public long Count => (long)Channels * Frames * Height * Width;
    }

    /// <summary>A latent in channel-planar order <c>[C, F, H, W]</c> (what <c>torch.randn</c> fills and the VAEs
    /// produce); a family converts it to and from its denoiser's token rows.</summary>
    public sealed class LatentTensor
    {
        public LatentTensor(LatentShape shape, float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.LongLength != shape.Count)
                throw new ArgumentException($"A {shape} latent holds {shape.Count} values, not {data.LongLength}.", nameof(data));
            Shape = shape;
            Data = data;
        }

        public LatentShape Shape { get; }
        public float[] Data { get; }
    }

    /// <summary>
    /// A VAE encoder's posterior: the per-value mean and log-variance, in the latent's <c>[C, F, H, W]</c> order.
    /// <see cref="Sample"/> is <c>mean + exp(0.5 * clamp(logvar, -30, 20)) * eps</c> (diffusers'
    /// DiagonalGaussianDistribution); <see cref="Mode"/> is the mean.
    /// </summary>
    public sealed class LatentPosterior
    {
        public LatentPosterior(LatentShape shape, float[] mean, float[] logVariance)
        {
            ArgumentNullException.ThrowIfNull(mean);
            if (mean.LongLength != shape.Count || (logVariance != null && logVariance.LongLength != shape.Count))
                throw new ArgumentException($"A {shape} posterior holds {shape.Count} values a moment.");
            Shape = shape;
            Mean = mean;
            LogVariance = logVariance;
        }

        public LatentShape Shape { get; }
        public float[] Mean { get; }

        /// <summary>Null for an encoder that gives the mean alone (a deterministic VAE).</summary>
        public float[] LogVariance { get; }

        public LatentTensor Mode() => new(Shape, (float[])Mean.Clone());

        /// <summary>A posterior sample with <paramref name="eps"/> (standard normal, one value per latent value).</summary>
        public LatentTensor Sample(ReadOnlySpan<float> eps)
        {
            if (eps.Length != Mean.Length) throw new ArgumentException("The posterior needs one noise value per latent value.", nameof(eps));
            var x = new float[Mean.Length];
            if (LogVariance == null)
            {
                Array.Copy(Mean, x, x.Length);
                return new LatentTensor(Shape, x);
            }
            for (int i = 0; i < x.Length; i++)
                x[i] = Mean[i] + MathF.Exp(0.5f * Math.Clamp(LogVariance[i], -30f, 20f)) * eps[i];
            return new LatentTensor(Shape, x);
        }
    }

    /// <summary>A reference picture as the conditioner sees it: the vision copy and its 0-based order
    /// (<c>&lt;Picture Index + 1&gt;</c>).</summary>
    public sealed record PresentedImage(RgbImage Image, int Index);

    /// <summary>A reference's latent as the denoiser conditions on it: its token rows (fixed for the whole loop) and
    /// the latent grid they cover.</summary>
    public sealed class LatentRef
    {
        public LatentRef(int index, LatentShape shape, float[] rows)
        {
            Index = index;
            Shape = shape;
            Rows = rows ?? throw new ArgumentNullException(nameof(rows));
        }

        /// <summary>0-based order among the request's references.</summary>
        public int Index { get; }
        public LatentShape Shape { get; }

        /// <summary>The token rows (a family's augmentation updates them in place before the loop).</summary>
        public float[] Rows { get; }
    }

    /// <summary>
    /// A conditioner's output for one branch: hidden rows <c>[Rows, Width]</c> row-major, each with a modality tag
    /// whose values the family defines (MiniMax-H3: 1 text, 0 image), and how long the vision tower took of it.
    /// </summary>
    public sealed class ConditioningSet
    {
        public ConditioningSet(float[] hidden, int rows, int width, int[] modalities = null)
        {
            ArgumentNullException.ThrowIfNull(hidden);
            if (rows < 0 || width <= 0 || hidden.LongLength != (long)rows * width)
                throw new ArgumentException($"Conditioning of {rows} rows of {width} holds {(long)rows * width} values, not {hidden.LongLength}.");
            if (modalities != null && modalities.Length != rows)
                throw new ArgumentException("Conditioning needs one modality tag a row.", nameof(modalities));
            Hidden = hidden;
            Rows = rows;
            Width = width;
            Modalities = modalities;
        }

        public float[] Hidden { get; }
        public int Rows { get; }
        public int Width { get; }

        /// <summary>One tag a row, or null when every row is text.</summary>
        public int[] Modalities { get; }

        /// <summary>Seconds of the encode the vision tower took (0 without references).</summary>
        public double VisionSeconds { get; init; }

        /// <summary>Whatever else the family's denoiser needs from its conditioner (DeepStack taps), or null.</summary>
        public object FamilyState { get; init; }
    }

    /// <summary>One denoising step: its 0-based index of <see cref="Count"/>, its sigmas and the head interval the
    /// recipe's head policy picked ((0, 0) for a model with one head).</summary>
    public readonly record struct StepState(int Index, int Count, float Sigma, float Next, int HeadStart, int HeadStop);

    /// <summary>A run of rows of the packed sequence modulated alike: a modality and the index of its timestep in
    /// <see cref="DiffusionLayout.Timesteps"/>.</summary>
    public readonly record struct DiffusionSegment(int Start, int Length, int Modality, int Timestep);

    /// <summary>
    /// The packed sequence of one step (a family's <see cref="IDiffusionFamily.Pack"/>): its rows, where the target
    /// rows sit, the segments and their timesteps, and every row's position (<see cref="PositionAxes"/> floats a row).
    /// </summary>
    public sealed class DiffusionLayout
    {
        public required int Rows { get; init; }
        public required int TargetStart { get; init; }
        public required int TargetRows { get; init; }
        public IReadOnlyList<DiffusionSegment> Segments { get; init; } = Array.Empty<DiffusionSegment>();
        public float[] Timesteps { get; init; } = Array.Empty<float>();
        public float[] Positions { get; init; }
        public int PositionAxes { get; init; }

        /// <summary>The family's own layout object, for its denoiser.</summary>
        public object FamilyLayout { get; init; }
    }

    /// <summary>What a denoiser step reads.</summary>
    public sealed class DenoiseStepInputs
    {
        /// <summary>The target's token rows (the sampler's state).</summary>
        public required float[] Latent { get; init; }
        public required ConditioningSet Conditioning { get; init; }
        public IReadOnlyList<LatentRef> References { get; init; } = Array.Empty<LatentRef>();
        public required DiffusionLayout Layout { get; init; }
        public required StepState Step { get; init; }
        public required SamplingRecipe Recipe { get; init; }
    }

    /// <summary>A denoiser step's prediction: the target rows' flow velocity, and any further named streams (MiniMax-H3
    /// video's audio).</summary>
    public sealed class StepOutput
    {
        public StepOutput(float[] velocity, IReadOnlyDictionary<string, float[]> streams = null)
        {
            Velocity = velocity ?? throw new ArgumentNullException(nameof(velocity));
            Streams = streams;
        }

        public float[] Velocity { get; }
        public IReadOnlyDictionary<string, float[]> Streams { get; }
    }

    /// <summary>What a denoiser can do beyond one step on one device.</summary>
    public sealed record DenoiserCaps
    {
        /// <summary>The step can be emitted per tensor-parallel rank (heads and FFN columns sharded).</summary>
        public bool Shardable { get; init; }

        /// <summary>Text and reference rows are modulated at a fixed timestep and do not attend the target, so their
        /// keys and values can be kept across steps.</summary>
        public bool PrefixCacheable { get; init; }

        /// <summary>The further streams <see cref="StepOutput.Streams"/> holds.</summary>
        public IReadOnlyList<string> Streams { get; init; } = Array.Empty<string>();
    }

    /// <summary>What a family's components are made from: the backend and the component files (by role:
    /// <see cref="Denoiser"/>, <see cref="TextEncoder"/>, <see cref="Vae"/>, or a family's own).</summary>
    public sealed class DiffusionLoad
    {
        public const string Denoiser = "denoiser", TextEncoder = "text-encoder", Vae = "vae";

        public required BackendType Backend { get; init; }
        public IReadOnlyDictionary<string, string> Files { get; init; } = new Dictionary<string, string>();

        /// <summary>The tensor-parallel group a shardable denoiser spreads over, or null.</summary>
        public ITensorParallelGroup TensorParallelGroup { get; init; }

        /// <summary>The file of <paramref name="role"/>; a missing one is a load refusal naming the role.</summary>
        public string File(string role) =>
            Files != null && Files.TryGetValue(role, out string path) && !string.IsNullOrWhiteSpace(path)
                ? path
                : throw new InvalidOperationException($"The diffusion load has no {role} file.");
    }

    /// <summary>The request a component sizes its working memory for: the target's and the references' latents, the
    /// prompt, and the references as the conditioner sees them (<see cref="Pictures"/>, empty without references).</summary>
    public readonly record struct DiffusionWorkload(LatentShape Target, IReadOnlyList<LatentShape> References, string Prompt)
    {
        /// <summary>The references' vision copies (<see cref="IReferencePolicy.ForVision"/>), in order.</summary>
        public IReadOnlyList<PresentedImage> Pictures { get; init; } = Array.Empty<PresentedImage>();
    }
}
