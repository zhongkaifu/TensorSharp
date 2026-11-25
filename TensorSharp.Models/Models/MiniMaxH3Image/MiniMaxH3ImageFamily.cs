// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image as a family of the shared diffusion runtime (docs/design/diffusion-runtime.md section 3.2): its
// geometry, its recipe (ComfyUI's "simple" schedule over the checkpoint's PDD grid, the grid interval's head), the CPU
// torch noise, the packed layout of MiniMaxH3ImageLayout and the 2x2-patch token rows the legacy MiniMaxH3ImageModel denoised on.
// Its components are the MiniMax-H3 text encoder with its vision tower (MiniMaxH3ImageConditioner), the DiT
// (MiniMaxH3ImageDenoiser) and the image VAE (MiniMaxH3ImageVaeCodec), all on the runtime. An edit's references follow
// MiniMaxH3ImageReferences; its canvas follows diffusers' MiniMax-H3-Image (ResolveCanvas).
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
            References = new MiniMaxH3ImageReferences(
                () => Vae != null ? (Vae.LatentsMean, Vae.LatentsStd)
                    : (MiniMaxH3ImageVaeCodec.ReleaseLatentsMean, MiniMaxH3ImageVaeCodec.ReleaseLatentsStd),
                ToRows);
        }

        /// <summary>The most reference pictures an edit takes (MiniMax's README: up to 16).</summary>
        public const int MaxReferences = 16;

        /// <summary>diffusers' min_generation_area: an edit smaller than 2048 x 2048 samples on a larger canvas of the same
        /// aspect and is resized back (decision D3; text to image keeps its own size, as before).</summary>
        public const long MinGenerationArea = 2048L * 2048;

        /// <summary>The largest area an edit may ask for: the model's 4-20 MP range (4096 x 5120).</summary>
        public const long MaxEditArea = 4096L * 5120;

        /// <summary>The family of the denoiser in <paramref name="path"/>: its grid from the safetensors metadata, or the
        /// release grid for a file without it (the legacy model's rule).</summary>
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
            SizeMultiple = MiniMaxH3ImageDiffusionModel.SizeMultiple,
            DefaultWidth = MiniMaxH3ImageDiffusionModel.DefaultSize,
            DefaultHeight = MiniMaxH3ImageDiffusionModel.DefaultSize,
            MinGenerationArea = MinGenerationArea,
            MaxArea = MaxEditArea,
            MaxReferences = MaxReferences,
        };

        public DiffusionGuidance Guidance => DiffusionGuidance.None;
        public SamplingRecipe DefaultRecipe { get; }

        /// <summary>The PDD grid the recipe samples over.</summary>
        public IReadOnlyList<float> Grid => _grid;

        public IReferencePolicy References { get; }

        public IConditioner CreateConditioner(DiffusionLoad load)
        {
            ArgumentNullException.ThrowIfNull(load);
            return Conditioner = new MiniMaxH3ImageConditioner(load.File(DiffusionLoad.TextEncoder), load.Backend);
        }

        public IDenoiser CreateDenoiser(DiffusionLoad load)
        {
            ArgumentNullException.ThrowIfNull(load);
            return Denoiser = new MiniMaxH3ImageDenoiser(load.File(DiffusionLoad.Denoiser), load.Backend)
            {
                AdapterBytes = () => Lora?.FactorBytes ?? 0,
            };
        }

        /// <summary>The denoiser this family made last (the model releases its LoRA binding before replacing a set).</summary>
        internal MiniMaxH3ImageDenoiser Denoiser { get; private set; }

        /// <summary>The LoRA set the model's requests run with (null: none), whose factors residency counts with the denoiser's
        /// weights.</summary>
        internal MiniMaxH3ImageLoraSet Lora { get; set; }

        public IVaeCodec CreateVae(DiffusionLoad load)
        {
            ArgumentNullException.ThrowIfNull(load);
            return Vae = new MiniMaxH3ImageVaeCodec(load.File(DiffusionLoad.Vae), load.Backend);
        }

        /// <summary>The conditioner this family made last (its token count is the model's timing line's).</summary>
        internal MiniMaxH3ImageConditioner Conditioner { get; private set; }

        /// <summary>The VAE this family made last (its tile count and decoder time are the model's timing line's).</summary>
        internal MiniMaxH3ImageVaeCodec Vae { get; private set; }

        /// <summary>Whether <paramref name="names"/> are a MiniMax-H3-Image denoiser's tensors: the H3 DiT's video patch and
        /// text projections under one prefix, and no audio stream (an H3 audio-video denoiser is minimax-h3).</summary>
        public static bool IsDenoiser(IEnumerable<string> names)
        {
            ArgumentNullException.ThrowIfNull(names);
            const string VideoPatch = "video_patch_proj.weight", AudioPatch = "audio_patch_proj.weight", Condition = "condition_proj.weight";
            var set = new HashSet<string>(names, StringComparer.Ordinal);
            bool image = false;
            foreach (string name in set)
            {
                if (!name.EndsWith(VideoPatch, StringComparison.Ordinal)) continue;
                string prefix = name[..^VideoPatch.Length];
                if (set.Contains(prefix + AudioPatch)) return false;
                if (set.Contains(prefix + Condition)) image = true;
            }
            return image;
        }

        /// <summary>torch.randn on the CPU generator, as ComfyUI draws MiniMax-H3-Image's initial latent.</summary>
        public INoiseSource Noise(long seed) => new TorchCpuRandn(unchecked((ulong)seed));

        /// <summary>
        /// Text to image: <see cref="DiffusionCanvas.TextToImage"/>, unchanged. An edit (docs/design/diffusion-runtime.md section
        /// 4.4): an explicit width and height win (multiples of 32, at most <see cref="MaxEditArea"/>); keepSourceSize returns
        /// <c>&lt;Picture 1&gt;</c>'s exact size; otherwise <c>&lt;Picture 1&gt;</c>'s aspect at the request's target area, or
        /// without one its VAE copy's size (a 1024-2048 short edge). Below <see cref="MinGenerationArea"/> (or a speed adapter's
        /// own floor, <see cref="SamplingContract.MinGenerationArea"/>) the picture samples on diffusers' larger canvas
        /// (get_image_canvas_size) and is resized back.
        /// <para>A masked edit (section 4.5) returns its source canvas: the same rules give the whole picture's canvas, and the
        /// selection's crop (or the whole picture) samples at that canvas's density but never above its own resolution
        /// (<see cref="EditMask.SamplingDimensions"/>), so neither the floor nor the latent mask raises it past the picture it
        /// is composited into; the 20 MP ceiling still applies to what it samples, at the crop's aspect.</para>
        /// </summary>
        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask)
        {
            ArgumentNullException.ThrowIfNull(request);
            if (request.References is not { Count: > 0 })
            {
                if (mask != null) throw new ArgumentException("A mask requires a first reference image to edit.");
                return DiffusionCanvas.TextToImage(request, Geometry, DisplayName);
            }
            RgbImage first = request.References[0].Image;
            int multiple = Geometry.SizeMultiple, width = request.Width, height = request.Height;
            if ((width > 0) != (height > 0))
                throw new ArgumentException($"{DisplayName} needs both width and height, or neither.");
            if (request.KeepSourceSize && width > 0)
                throw new ArgumentException("keepSourceSize returns the first image's own size; it cannot be combined with a width and height.");
            if (width > 0)
            {
                if (width % multiple != 0 || height % multiple != 0)
                    throw new ArgumentException($"{DisplayName} sizes are multiples of {multiple}; got {width}x{height}.");
            }
            else if (request.KeepSourceSize)
            {
                (width, height) = (first.Width, first.Height);
            }
            else if (request.TargetArea > 0)
            {
                double aspect = (double)first.Width / first.Height;
                width = Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(Math.Sqrt(request.TargetArea * aspect), multiple));
                height = Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(Math.Sqrt(request.TargetArea / aspect), multiple));
            }
            else
            {
                var (vw, vh) = MiniMaxH3ImageReferences.VaeCopySize(first.Width, first.Height);
                width = Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(vw, multiple));
                height = Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(vh, multiple));
            }
            if (mask != null)
            {
                var selection = mask.SamplingDimensions(width, height, multiple);
                if ((long)selection.Width * selection.Height > MaxEditArea)
                {
                    if (request.Width > 0)
                        throw new ArgumentException(
                            $"{DisplayName} edits at most {MaxEditArea / 1e6:F1} MP ({selection.Width}x{selection.Height} asked for the selection); ask for a smaller size.");
                    double f = Math.Sqrt(MaxEditArea / ((double)selection.Width * selection.Height));
                    selection = (Math.Max(multiple, (int)Math.Floor(selection.Width * f / multiple) * multiple),
                                 Math.Max(multiple, (int)Math.Floor(selection.Height * f / multiple) * multiple));
                }
                return new DiffusionCanvas(selection, (mask.Source.Width, mask.Source.Height));
            }
            // A speed adapter's contract may lower the floor (pDMD's 1024 px: an edit samples at the asked size, as the
            // adapter's diffusers pipeline does); otherwise the family's 4 MP floor. A masked selection takes neither (above).
            long floor = request.Adapters?.Contract is { MinGenerationArea: > 0 } contract ? contract.MinGenerationArea : MinGenerationArea;
            var sampling = EditCanvas(width, height, multiple, floor);
            if ((long)sampling.Width * sampling.Height > MaxEditArea)
            {
                if (!request.KeepSourceSize)
                    throw new ArgumentException(
                        $"{DisplayName} edits at most {MaxEditArea / 1e6:F1} MP ({width}x{height} asked); ask for a smaller size.");
                // A large source keeps its size and samples at the largest canvas of its aspect.
                double f = Math.Sqrt(MaxEditArea / ((double)width * height));
                sampling = (Math.Max(multiple, (int)Math.Floor(width * f / multiple) * multiple),
                            Math.Max(multiple, (int)Math.Floor(height * f / multiple) * multiple));
            }
            return new DiffusionCanvas(sampling, (width, height));
        }

        /// <summary>diffusers' get_image_canvas_size: a picture below <paramref name="floor"/> (0: <see cref="MinGenerationArea"/>)
        /// is scaled up by <c>sqrt(floor / area)</c>; either way both sides round to <paramref name="multiple"/>.</summary>
        public static (int Width, int Height) EditCanvas(int width, int height, int multiple = MiniMaxH3ImageDiffusionModel.SizeMultiple,
            long floor = 0)
        {
            double scale = Math.Max(1.0, Math.Sqrt((floor > 0 ? floor : MinGenerationArea) / ((double)width * height)));
            return (Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(width * scale, multiple)),
                    Math.Max(multiple, MiniMaxH3ImageReferences.RoundToMultiple(height * scale, multiple)));
        }

        /// <summary>A decode resized to the returned size with Pillow's Lanczos on 8 bits, as diffusers' decode step resizes
        /// it (VaeImageProcessor.resize).</summary>
        public RgbImage ResizeOutput(RgbImage image, int width, int height) =>
            MiniMaxH3ImageReferences.PillowResize(image, width, height);

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
        /// within a token (the legacy MiniMaxH3ImageModel.Generate's exact steps).</summary>
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
    }
}
