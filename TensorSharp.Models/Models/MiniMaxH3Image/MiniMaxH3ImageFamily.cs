// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image as a family of the shared diffusion runtime (docs/design/diffusion-runtime.md section 3.2): its
// geometry, its recipe (ComfyUI's "simple" schedule over the checkpoint's PDD grid, the grid interval's head), the CPU
// torch noise, the packed layout of MiniMaxH3ImageLayout and the 2x2-patch token rows MiniMaxH3ImageModel denoises on.
// The denoiser runs on the runtime (MiniMaxH3ImageDenoiser); the conditioner and the VAE come with stage S4, editing
// with S5.
using System;
using System.Collections.Generic;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    public sealed class MiniMaxH3ImageFamily : IDiffusionFamily
    {
        public const string ArchitectureId = "minimax-h3-image";
        private const int Channels = 24, Downsample = 16, Patch = 2;

        private readonly float[] _grid;

        /// <param name="grid">The checkpoint's decreasing PDD sigma grid (<see cref="ForDenoiser"/> reads it).</param>
        public MiniMaxH3ImageFamily(float[] grid)
        {
            ArgumentNullException.ThrowIfNull(grid);
            _grid = (float[])grid.Clone();
            DefaultRecipe = new SamplingRecipe
            {
                Schedule = DiffusionSchedule.ComfySimpleOverGrid,
                Steps = MiniMaxH3ImageSampling.Steps,
                Grid = _grid,
                Heads = HeadPolicy.GridInterval,
            };
            DefaultRecipe.Validate();
        }

        /// <summary>The family of the denoiser in <paramref name="path"/>: its grid from the safetensors metadata, or the
        /// release grid for a file without it (MiniMaxH3ImageModel's rule).</summary>
        public static MiniMaxH3ImageFamily ForDenoiser(string path)
        {
            using var file = new MiniMaxH3WeightFile(path ?? throw new ArgumentNullException(nameof(path)));
            MiniMaxH3Config config = MiniMaxH3Config.Detect(file.Tensors);
            if (config.HasAudio || config.PddSteps < 2)
                throw new NotSupportedException($"{System.IO.Path.GetFileName(path)} is not a MiniMax-H3-Image denoiser.");
            return new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.GridFrom(file.SafetensorsMetadata, config.PddSteps));
        }

        public string Id => ArchitectureId;
        public string DisplayName => "MiniMax-H3-Image";

        public DiffusionGeometry Geometry { get; } = new()
        {
            LatentChannels = Channels,
            VaeFactor = Downsample,
            Patch = Patch,
            SizeMultiple = MiniMaxH3ImageModel.SizeMultiple,
            DefaultWidth = MiniMaxH3ImageModel.DefaultSize,
            DefaultHeight = MiniMaxH3ImageModel.DefaultSize,
        };

        public DiffusionGuidance Guidance => DiffusionGuidance.None;
        public SamplingRecipe DefaultRecipe { get; }

        /// <summary>The PDD grid the recipe samples over.</summary>
        public IReadOnlyList<float> Grid => _grid;

        public IReferencePolicy References { get; } = new NoReferences();

        public IConditioner CreateConditioner(DiffusionLoad load) =>
            throw new NotSupportedException("MiniMax-H3-Image's conditioner on the diffusion runtime arrives with stage S4.");

        public IDenoiser CreateDenoiser(DiffusionLoad load)
        {
            ArgumentNullException.ThrowIfNull(load);
            return new MiniMaxH3ImageDenoiser(load.File(DiffusionLoad.Denoiser), load.Backend);
        }

        public IVaeCodec CreateVae(DiffusionLoad load) =>
            throw new NotSupportedException("MiniMax-H3-Image's VAE on the diffusion runtime arrives with stage S4.");

        /// <summary>torch.randn on the CPU generator, as ComfyUI draws MiniMax-H3-Image's initial latent.</summary>
        public INoiseSource Noise(long seed) => new TorchCpuRandn(unchecked((ulong)seed));

        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask)
        {
            ArgumentNullException.ThrowIfNull(request);
            if (mask != null || request.References is { Count: > 0 })
                throw new ArgumentException($"{DisplayName} does not edit images on the diffusion runtime yet; drop the reference images and the mask.");
            return DiffusionCanvas.TextToImage(request, Geometry, DisplayName);
        }

        /// <summary>[text | references | target] (<see cref="MiniMaxH3ImageLayout.Build"/>), carried as the denoiser's
        /// <see cref="DiffusionLayout.FamilyLayout"/>.</summary>
        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step)
        {
            ArgumentNullException.ThrowIfNull(text);
            List<(int, int)> shapes = null;
            if (references is { Count: > 0 })
            {
                shapes = new List<(int, int)>(references.Count);
                foreach (LatentRef r in references) shapes.Add((r.Shape.Height, r.Shape.Width));
            }
            MiniMaxH3PackedLayout packed = MiniMaxH3ImageLayout.Build(text.Rows, target.Height, target.Width, step.Sigma, text.Modalities, shapes);
            var segments = new DiffusionSegment[packed.ModulationSpans.Count];
            for (int i = 0; i < segments.Length; i++)
            {
                MiniMaxH3ModulationSpan span = packed.ModulationSpans[i];
                segments[i] = new DiffusionSegment(span.Start, span.End - span.Start, span.Row % MiniMaxH3Modality.Count, span.Row / MiniMaxH3Modality.Count);
            }
            return new DiffusionLayout
            {
                Rows = packed.TokenCount,
                TargetStart = packed.VideoSpan.Start,
                TargetRows = packed.VideoSpan.End - packed.VideoSpan.Start,
                Segments = segments,
                Timesteps = packed.Timesteps,
                Positions = packed.Positions,
                PositionAxes = 3,
                FamilyLayout = packed,
            };
        }

        /// <summary>[C, 1, H, W] to the denoiser's token rows: channels last, then 2x2 patches of 96 values, channel-major
        /// within a token (MiniMaxH3ImageModel.Generate's exact steps).</summary>
        public float[] ToRows(LatentTensor latent)
        {
            ArgumentNullException.ThrowIfNull(latent);
            LatentShape s = Check(latent.Shape);
            int plane = s.Height * s.Width;
            var hwc = new float[latent.Data.LongLength];
            for (int c = 0; c < s.Channels; c++)
                for (int i = 0; i < plane; i++)
                    hwc[(long)i * s.Channels + c] = latent.Data[(long)c * plane + i];
            return MiniMaxH3DiT.PatchifyVideo(hwc, 1, s.Height, s.Width, s.Channels);
        }

        public LatentTensor FromRows(float[] rows, LatentShape shape)
        {
            ArgumentNullException.ThrowIfNull(rows);
            LatentShape s = Check(shape);
            if (rows.LongLength != s.Count) throw new ArgumentException($"A {s} latent holds {s.Count} values, not {rows.LongLength}.", nameof(rows));
            float[] hwc = MiniMaxH3DiT.UnpatchifyVideo(rows, 1, s.Height, s.Width, s.Channels);
            int plane = s.Height * s.Width;
            var chw = new float[hwc.LongLength];
            for (int c = 0; c < s.Channels; c++)
                for (int i = 0; i < plane; i++)
                    chw[(long)c * plane + i] = hwc[(long)i * s.Channels + c];
            return new LatentTensor(s, chw);
        }

        /// <summary>A cell weight per value of the token rows: token (y, x) holds value c * 4 + ky * 2 + kx of cell
        /// (2y + ky, 2x + kx).</summary>
        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape)
        {
            ArgumentNullException.ThrowIfNull(cellWeights);
            LatentShape s = Check(shape);
            if (cellWeights.LongLength != (long)s.Height * s.Width)
                throw new ArgumentException($"A {s.Height}x{s.Width} latent's mask has {(long)s.Height * s.Width} cells, not {cellWeights.LongLength}.");
            int gh = s.Height / Patch, gw = s.Width / Patch, values = s.Channels * Patch * Patch;
            var weights = new float[(long)gh * gw * values];
            for (int y = 0; y < gh; y++)
                for (int x = 0; x < gw; x++)
                {
                    long token = ((long)y * gw + x) * values;
                    for (int c = 0; c < s.Channels; c++)
                        for (int ky = 0; ky < Patch; ky++)
                            for (int kx = 0; kx < Patch; kx++)
                                weights[token + c * Patch * Patch + ky * Patch + kx] = cellWeights[(long)(y * Patch + ky) * s.Width + x * Patch + kx];
                }
            return (weights, 1);
        }

        private static LatentShape Check(LatentShape s)
        {
            if (s.Channels != Channels || s.Frames != 1 || s.Height <= 0 || s.Width <= 0 || s.Height % Patch != 0 || s.Width % Patch != 0)
                throw new ArgumentException($"MiniMax-H3-Image latents are [{Channels}, 1, H, W] with even H and W, not {s}.");
            return s;
        }

        /// <summary>Editing (reference pictures) is stage S5's.</summary>
        private sealed class NoReferences : IReferencePolicy
        {
            private static NotSupportedException Refused() => new("MiniMax-H3-Image does not edit images on the diffusion runtime yet.");
            public PresentedImage ForVision(RgbImage image, int index) => throw Refused();
            public RgbImage ForVae(RgbImage image, int index) => throw Refused();
            public LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise) => throw Refused();
            public void Augment(LatentRef reference, INoiseSource noise) => throw Refused();
        }
    }
}
