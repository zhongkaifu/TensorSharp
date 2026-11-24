// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-Image-2.1 (and its 8-step Turbo distillation) as a family of the shared diffusion runtime
// (docs/design/diffusion-runtime.md section 3.2): 64-channel /16 latents denoised a cell a token, Qwen3-VL-8B conditioning
// with each reference's vision slots in the text, classifier-free guidance above CFG 1, the scheduler's dynamic-shift schedule
// (or Turbo's or a step-distillation LoRA's own sigmas), Philox noise keyed to an edit's pictures, and a prefix the denoiser
// caches across steps. Its components are QwenImage21ConditionerComponent, QwenImage21Denoiser and QwenImage21VaeCodec.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage21
{
    public sealed class QwenImage21Family : IDiffusionFamily
    {
        public const string ArchitectureId = "qwen_image";
        internal const int Channels = 64, Downsample = 16, HeadDim = 128, TextDim = 4096;

        /// <summary>Steps the base checkpoint takes by default (Qwen's recommended unguided sampling).</summary>
        public const int BaseSteps = 40;

        private readonly BackendType _backend;
        private readonly ConditionalWeakTable<ConditioningSet, QwenImage21Layout> _layouts = new();

        public QwenImage21Family(BackendType backend, QwenImageVariant variant)
        {
            _backend = backend;
            Variant = variant;
            References = new QwenImage21References();
        }

        public string Id => ArchitectureId;

        public string DisplayName => Variant == QwenImageVariant.Turbo ? "Qwen-Image-2.1-Turbo" : "Qwen-Image-2.1";

        public QwenImageVariant Variant { get; }

        public DiffusionGeometry Geometry { get; } = new()
        {
            LatentChannels = Channels,
            VaeFactor = Downsample,
            Patch = 1,
            SizeMultiple = 32,
            DefaultWidth = 2048,
            DefaultHeight = 2048,
            MaxReferences = int.MaxValue,
        };

        public DiffusionGuidance Guidance => DiffusionGuidance.Cfg;

        /// <summary>The recipe a request samples with: a LoRA plug-in's (<see cref="LoraRecipe"/>, never together with the
        /// checkpoint's own: the loader refuses that), else Turbo's, else the base checkpoint's.</summary>
        public SamplingRecipe DefaultRecipe => LoraRecipe ?? CheckpointRecipe;

        /// <summary>The checkpoint's own recipe: Turbo's 8 published sigmas, or the base scheduler's dynamic shift at 40 steps.</summary>
        public SamplingRecipe CheckpointRecipe => Variant == QwenImageVariant.Turbo ? TurboRecipe : BaseRecipe;

        internal static readonly SamplingRecipe BaseRecipe = new()
        {
            Schedule = DiffusionSchedule.DynamicShift,
            Steps = BaseSteps,
            Cfg = 1f,
        };

        /// <summary>Turbo's recipe: <see cref="QwenImage21Turbo.SampleSigmas"/> unshifted at every size, 8 steps only, CFG 1.</summary>
        internal static readonly SamplingRecipe TurboRecipe = new()
        {
            Schedule = DiffusionSchedule.NodeSets,
            Steps = QwenImage21Turbo.Steps,
            NodeSets = new Dictionary<int, float[]> { [QwenImage21Turbo.Steps] = QwenImage21Turbo.SampleSigmas },
            Cfg = 1f,
            Owner = "Qwen-Image-2.1-Turbo, which was distilled for one published 8-step schedule (Qwen has evaluated no other),",
        };

        /// <summary>The loaded LoRA plug-ins' recipe as a sampling recipe (null: none), set by the model with its plug-ins.</summary>
        internal SamplingRecipe LoraRecipe { get; set; }

        /// <summary>A LoRA recipe as the runtime's: its sigma nodes per step count (shifted dynamically or used as they are), or
        /// only its default steps and CFG over the checkpoint's own schedule.</summary>
        internal SamplingRecipe RecipeOf(QwenImage21LoraRecipe recipe)
        {
            if (recipe == null) return null;
            SamplingRecipe checkpoint = CheckpointRecipe;
            if (!recipe.HasSchedule)
            {
                int steps = recipe.DefaultSteps > 0 ? recipe.DefaultSteps : checkpoint.Steps;
                var defaults = checkpoint with { Steps = steps, Cfg = recipe.Cfg ?? checkpoint.Cfg, TimestepBf16 = recipe.TimestepBf16 };
                defaults.Validate();
                return defaults;
            }
            var sets = new Dictionary<int, float[]>();
            foreach (var (count, nodes) in recipe.Nodes) sets[count] = (float[])nodes.Clone();
            var result = new SamplingRecipe
            {
                Schedule = DiffusionSchedule.NodeSets,
                Steps = recipe.DefaultSteps > 0 ? recipe.DefaultSteps : checkpoint.Steps,
                NodeSets = sets,
                ShiftNodes = recipe.Shift == QwenImage21SigmaShift.Dynamic,
                Cfg = recipe.Cfg ?? 1f,
                TimestepBf16 = recipe.TimestepBf16,
                Owner = recipe.Owner ?? "The LoRA sampling recipe from " + recipe.Source,
            };
            result.Validate();
            return result;
        }

        public IReferencePolicy References { get; }

        // ---- components (the model supplies them through these factories)

        internal Func<DiffusionLoad, IConditioner> ConditionerFactory { get; set; }
        internal Func<DiffusionLoad, IDenoiser> DenoiserFactory { get; set; }
        internal Func<DiffusionLoad, IVaeCodec> VaeFactory { get; set; }

        public IConditioner CreateConditioner(DiffusionLoad load) =>
            (ConditionerFactory ?? throw new InvalidOperationException("Qwen-Image-2.1's family has no conditioner factory."))(load);

        public IDenoiser CreateDenoiser(DiffusionLoad load) =>
            (DenoiserFactory ?? throw new InvalidOperationException("Qwen-Image-2.1's family has no denoiser factory."))(load);

        public IVaeCodec CreateVae(DiffusionLoad load) =>
            (VaeFactory ?? throw new InvalidOperationException("Qwen-Image-2.1's family has no VAE factory."))(load);

        // ---- noise, canvas, layout

        public INoiseSource Noise(long seed) => new QwenImage21Noise(seed, 0);

        /// <summary>Whether an edit's noise follows its pictures (<see cref="QwenImage21Noise.EditNoiseVariable"/>), read when the
        /// model takes a request so a misspelled setting fails before any work.</summary>
        internal bool FollowReferences { get; set; } = true;

        public INoiseSource Noise(DiffusionRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var pictures = new List<RgbImage>(request.References?.Count ?? 0);
            if (request.References != null)
                foreach (DiffusionReference r in request.References) pictures.Add(r.Image);
            ulong stream = QwenImage21Noise.StreamFor(pictures, FollowReferences);
            LastStream = stream;
            return new QwenImage21Noise(request.Seed, stream);
        }

        /// <summary>The stream the last request drew from (0: text to image or seed noise), for its log line.</summary>
        internal ulong LastStream { get; private set; }

        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask)
        {
            ArgumentNullException.ThrowIfNull(request);
            RgbImage first = request.References is { Count: > 0 } ? request.References[0].Image : null;
            if (mask != null && first == null) throw new ArgumentException("A mask requires a first reference image to edit.");
            return QwenImage21Canvas.Resolve(request, first, mask, _backend);
        }

        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step)
        {
            ArgumentNullException.ThrowIfNull(text);
            QwenImage21Layout layout = _layouts.GetValue(text, t =>
            {
                var shapes = new List<(int, int)>();
                if (references != null)
                    foreach (LatentRef r in references) shapes.Add((r.Shape.Height, r.Shape.Width));
                shapes.Add((target.Height, target.Width));
                int[] slots = null;
                if (t.Modalities != null)
                    foreach (int tag in t.Modalities)
                        if (tag != 0) { slots = t.Modalities; break; }
                return QwenImage21Layout.Build(t.Rows, slots, shapes, HeadDim);
            });
            if (layout.Target.Length != target.Height * target.Width)
                throw new InvalidOperationException("A conditioning set was packed for another target.");
            return new DiffusionLayout
            {
                Rows = layout.Rows,
                TargetStart = layout.Prefix,
                TargetRows = layout.Target.Length,
                FamilyLayout = layout,
            };
        }

        /// <summary>A latent [64, 1, H, W] as token-major rows [H * W, 64].</summary>
        public float[] ToRows(LatentTensor latent)
        {
            ArgumentNullException.ThrowIfNull(latent);
            LatentShape s = Check(latent.Shape);
            int count = s.Height * s.Width;
            var rows = new float[latent.Data.Length];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < Channels; c++) rows[i * Channels + c] = latent.Data[c * count + i];
            return rows;
        }

        public LatentTensor FromRows(float[] rows, LatentShape shape)
        {
            ArgumentNullException.ThrowIfNull(rows);
            LatentShape s = Check(shape);
            int count = s.Height * s.Width;
            if (rows.Length != (long)count * Channels) throw new ArgumentException("Expected 64-channel latent rows.", nameof(rows));
            var chw = new float[rows.Length];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < Channels; c++) chw[c * count + i] = rows[i * Channels + c];
            return new LatentTensor(s, chw);
        }

        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape) => (cellWeights, Channels);

        private static LatentShape Check(LatentShape s)
        {
            if (s.Channels != Channels || s.Frames != 1 || s.Height <= 0 || s.Width <= 0)
                throw new ArgumentException($"Qwen-Image-2.1 latents are [{Channels}, 1, H, W], not {s}.");
            return s;
        }
    }

    /// <summary>
    /// Qwen-Image-2.1's references: each picture resized once to its conditioning size (<see cref="QwenImage21Canvas.ReferenceDimensions"/>),
    /// the same copy for the vision tower and the VAE (one image slot of the vision tower stands for four /16 VAE tokens), its
    /// latent the VAE's deterministic encoding, no augmentation.
    /// </summary>
    internal sealed class QwenImage21References : IReferencePolicy
    {
        private (RgbImage Source, DiffusionCanvas Canvas, RgbImage Copy) _last;

        private RgbImage Copy(RgbImage image, DiffusionCanvas canvas)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (ReferenceEquals(_last.Source, image) && _last.Canvas == canvas) return _last.Copy;
            var (w, h) = QwenImage21Canvas.ReferenceDimensions(image, canvas.Sampling.Width, canvas.Sampling.Height);
            RgbImage copy = ImageIO.Resize(image, w, h);
            _last = (image, canvas, copy);
            return copy;
        }

        public PresentedImage ForVision(RgbImage image, int index) =>
            throw new InvalidOperationException("Qwen-Image-2.1's references follow the output's size; pass the canvas.");

        public RgbImage ForVae(RgbImage image, int index) =>
            throw new InvalidOperationException("Qwen-Image-2.1's references follow the output's size; pass the canvas.");

        public PresentedImage ForVision(RgbImage image, int index, DiffusionCanvas canvas) => new(Copy(image, canvas), index);

        public RgbImage ForVae(RgbImage image, int index, DiffusionCanvas canvas) => Copy(image, canvas);

        public LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise)
        {
            ArgumentNullException.ThrowIfNull(posterior);
            LatentShape s = posterior.Shape;
            int count = s.Height * s.Width;
            var rows = new float[posterior.Mean.Length];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < s.Channels; c++) rows[i * s.Channels + c] = posterior.Mean[c * count + i];
            return new LatentRef(index, s, rows);
        }

        public void Augment(LatentRef reference, INoiseSource noise) { }
    }
}
