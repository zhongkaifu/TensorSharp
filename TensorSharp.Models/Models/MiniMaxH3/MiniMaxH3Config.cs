// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 architecture configuration, derived entirely from the checkpoint's
// tensor table.
//
// The published H3 GGUFs carry ZERO metadata key/value pairs -- no
// general.architecture, no hyperparameters, no tokenizer. They are pure tensor
// bags. So both "is this an H3 checkpoint?" and "what shape is it?" have to be
// answered from tensor names and shapes, which is what this file does.
//
// GGUF stores shapes in ne order, i.e. Shape[0] is the fastest-moving dimension.
// For a Linear that means Shape = [inFeatures, outFeatures], the transpose of the
// PyTorch weight's own shape -- worth keeping in mind when reading the numbers
// below against the reference config.json.
using System;
using System.Collections.Generic;
using System.Globalization;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Which conditioning inputs a MiniMax-H3 checkpoint accepts. The two
    /// published denoisers are separate checkpoints, not modes.</summary>
    public enum MiniMaxH3Partition
    {
        /// <summary>Text plus zero, one or two keyframes (t2va / i2va / fl2va).</summary>
        FirstLastFrame,
        /// <summary>Text plus reference images, videos and audio (ref2va).</summary>
        Reference,
    }

    /// <summary>How the timestep reaches the AdaLN modulation projection.</summary>
    public enum MiniMaxH3TimeEmbedding
    {
        /// <summary>Sinusoidal features through a two-layer MLP.</summary>
        TimeEmbedder,
        /// <summary>A learned curve table, interpolated at the requested timestep. Much
        /// smaller, and what the released checkpoints ship.</summary>
        CurveTable,
    }

    /// <summary>Hyperparameters of a MiniMax-H3 diffusion transformer.</summary>
    public sealed class MiniMaxH3Config
    {
        public int HiddenSize { get; init; }
        public int NumLayers { get; init; }
        public int TokenRefinerLayers { get; init; }
        public int NumAttentionHeads { get; init; }
        public int AttentionHeadDim { get; init; }
        public int FfnHiddenSize { get; init; }
        public int VideoLatentChannels { get; init; }
        /// <summary>Audio latent channels, or 0 for an image checkpoint (MiniMax-H3-Image),
        /// which has no audio stream at all.</summary>
        public int AudioLatentChannels { get; init; }
        /// <summary>True for the audio-video denoisers; false for MiniMax-H3-Image.</summary>
        public bool HasAudio => AudioLatentChannels > 0;
        /// <summary>Output heads stacked in <c>final_layer.video_out</c>: 1 for the video
        /// denoisers, the distilled step count (6) for a PDD checkpoint such as
        /// MiniMax-H3-Image, whose head for a step is the first head plus that step's offset.</summary>
        public int PddSteps { get; init; } = 1;
        public int TextDim { get; init; }
        public int RopeInvFreqLength { get; init; }
        public MiniMaxH3TimeEmbedding TimeEmbedding { get; init; }
        /// <summary>Width of the vector fed to the AdaLN projection.</summary>
        public int TimeEmbedDim { get; init; }
        /// <summary>Entries in the AdaLN curve table, or 0 for the MLP variant.</summary>
        public int AdaLnCurveGrid { get; init; }
        /// <summary>Tensor-name prefix the weights live under ("" for the bare GGUFs).</summary>
        public string Prefix { get; init; } = string.Empty;

        public float NormEps { get; init; } = 1e-5f;
        public float QkNormEps { get; init; } = 1e-5f;

        /// <summary>Inner width of the attention projections. NOTE this is deliberately
        /// NOT equal to <see cref="HiddenSize"/> in H3: the residual stream is 5376 wide
        /// while attention runs at 56 x 128 = 7168.</summary>
        public int AttentionInnerDim => NumAttentionHeads * AttentionHeadDim;

        /// <summary>Elements of a patchified video token (latent channels x patch volume).</summary>
        public int VideoPatchDim =>
            VideoLatentChannels * MiniMaxH3Geometry.PatchT * MiniMaxH3Geometry.PatchH * MiniMaxH3Geometry.PatchW;

        /// <summary>Rotated head dimensions: 3 axes x inv-freq length x 2. The rest of the
        /// head passes through unrotated.</summary>
        public int RopeRotaryDim => 3 * RopeInvFreqLength * 2;

        /// <summary>Modulation values the AdaLN projection emits per timestep row.</summary>
        public int AdaLnOutFeatures =>
            HiddenSize * MiniMaxH3Modality.VectorsPerRow * MiniMaxH3Modality.Count;

        /// <summary>Modulation values the final layer's projection emits (shift + scale).</summary>
        public int FinalAdaLnOutFeatures => HiddenSize * 2;

        // ---- detection ------------------------------------------------------

        /// <summary>The two tensors whose joint presence identifies an H3 denoiser: it is
        /// the only architecture that patchifies a video AND an audio latent.</summary>
        private const string VideoPatchProj = "video_patch_proj.weight";
        private const string AudioPatchProj = "audio_patch_proj.weight";

        /// <summary>The text projection every H3 denoiser has; with the video patch
        /// projection and no audio one it identifies MiniMax-H3-Image.</summary>
        private const string ConditionProj = "condition_proj.weight";

        /// <summary>True when the tensor table looks like a MiniMax-H3 audio-video diffusion
        /// model. Used to route a metadata-less GGUF, which no architecture string can do.</summary>
        public static bool IsMiniMaxH3(IReadOnlyDictionary<string, GgufTensorInfo> tensors)
            => TryFindPrefix(tensors, out _);

        /// <summary>True when the tensor table is a MiniMax-H3-Image denoiser: the H3 DiT
        /// with no audio stream.</summary>
        public static bool IsMiniMaxH3Image(IReadOnlyDictionary<string, GgufTensorInfo> tensors)
            => TryFindPrefix(tensors, out _, requireAudio: false) && !TryFindPrefix(tensors, out _);

        /// <summary>Locate the prefix the H3 weights sit under, if any. The audio-video
        /// denoisers are the ones with both patch projections; <paramref name="requireAudio"/>
        /// false also accepts an image denoiser (video patch and text projections only).</summary>
        public static bool TryFindPrefix(
            IReadOnlyDictionary<string, GgufTensorInfo> tensors, out string prefix, bool requireAudio = true)
        {
            prefix = null;
            if (tensors is null) return false;
            foreach (var name in tensors.Keys)
            {
                if (!name.EndsWith(VideoPatchProj, StringComparison.Ordinal)) continue;
                string candidate = name[..^VideoPatchProj.Length];
                if (tensors.ContainsKey(candidate + AudioPatchProj)
                    || (!requireAudio && tensors.ContainsKey(candidate + ConditionProj)))
                {
                    prefix = candidate;
                    return true;
                }
            }
            return false;
        }

        /// <summary>Derive the full configuration from a checkpoint's tensor table, an
        /// audio-video denoiser or a MiniMax-H3-Image one.</summary>
        public static MiniMaxH3Config Detect(IReadOnlyDictionary<string, GgufTensorInfo> tensors)
        {
            if (tensors is null) throw new ArgumentNullException(nameof(tensors));
            if (!TryFindPrefix(tensors, out string prefix, requireAudio: false))
                throw new InvalidOperationException(
                    "Not a MiniMax-H3 diffusion checkpoint: video_patch_proj and condition_proj are absent.");

            GgufTensorInfo Get(string suffix)
            {
                if (!tensors.TryGetValue(prefix + suffix, out var t))
                    throw new InvalidOperationException(
                        $"MiniMax-H3 checkpoint is missing '{prefix + suffix}'.");
                return t;
            }
            bool Has(string suffix) => tensors.ContainsKey(prefix + suffix);
            int Dim(GgufTensorInfo t, int axis) => checked((int)t.Shape[axis]);

            var videoPatch = Get(VideoPatchProj);
            int hidden = Dim(videoPatch, 1);
            int patchVolume = MiniMaxH3Geometry.PatchT * MiniMaxH3Geometry.PatchH * MiniMaxH3Geometry.PatchW;
            int videoChannels = Dim(videoPatch, 0) / patchVolume;

            int layers = CountIndexed(tensors, prefix + "blocks.");
            int refinerLayers = CountIndexed(tensors, prefix + "token_refiner.blocks.");

            int headDim = Dim(Get("blocks.0.attn.q_norm.weight"), 0);
            int qkvOut = Dim(Get("blocks.0.attn.qkv_proj.weight"), 1);
            int heads = qkvOut / (3 * headDim);
            // fc1 emits gate and value side by side for the SwiGLU, hence the halving.
            int ffn = Dim(Get("blocks.0.mlp.fc1.weight"), 1) / 2;

            var timeEmbedding = Has("adaln_t_table")
                ? MiniMaxH3TimeEmbedding.CurveTable
                : MiniMaxH3TimeEmbedding.TimeEmbedder;
            int timeEmbedDim, curveGrid;
            if (timeEmbedding == MiniMaxH3TimeEmbedding.CurveTable)
            {
                var table = Get("adaln_t_table");
                timeEmbedDim = Dim(table, 0);
                curveGrid = Dim(table, 1);
            }
            else
            {
                timeEmbedDim = Dim(Get("time_embedder.proj_out.weight"), 1);
                curveGrid = 0;
            }

            // A PDD checkpoint stacks its step heads in video_out: [hidden, patchDim * steps].
            int pddSteps = 1;
            if (Has("final_layer.video_out.weight"))
            {
                int videoOutRows = Dim(Get("final_layer.video_out.weight"), 1);
                int patchDim = videoChannels * patchVolume;
                if (patchDim <= 0 || videoOutRows % patchDim != 0)
                    throw new InvalidOperationException(
                        $"MiniMax-H3 final_layer.video_out has {videoOutRows} rows, not a multiple of the {patchDim}-value patch.");
                pddSteps = videoOutRows / patchDim;
            }

            return new MiniMaxH3Config
            {
                Prefix = prefix,
                HiddenSize = hidden,
                NumLayers = layers,
                TokenRefinerLayers = refinerLayers,
                NumAttentionHeads = heads,
                AttentionHeadDim = headDim,
                FfnHiddenSize = ffn,
                VideoLatentChannels = videoChannels,
                AudioLatentChannels = Has(AudioPatchProj) ? Dim(Get(AudioPatchProj), 0) : 0,
                PddSteps = pddSteps,
                TextDim = Dim(Get("condition_proj.weight"), 0),
                RopeInvFreqLength = Dim(Get("rope.inv_freq"), 0),
                TimeEmbedding = timeEmbedding,
                TimeEmbedDim = timeEmbedDim,
                AdaLnCurveGrid = curveGrid,
            };
        }

        /// <summary>The partition, from the checkpoint's file name — which is the
        /// only place fl2va and ref2va differ.</summary>
        public static MiniMaxH3Partition PartitionFromFileName(string path)
            => !string.IsNullOrEmpty(path)
               && System.IO.Path.GetFileName(path).Contains("ref2va", StringComparison.OrdinalIgnoreCase)
                ? MiniMaxH3Partition.Reference
                : MiniMaxH3Partition.FirstLastFrame;

        /// <summary>Count the distinct indices under a "<c>prefix</c>N." naming scheme.</summary>
        internal static int CountIndexed(
            IReadOnlyDictionary<string, GgufTensorInfo> tensors, string prefix)
        {
            int max = -1;
            foreach (var name in tensors.Keys)
            {
                if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                int start = prefix.Length;
                int end = start;
                while (end < name.Length && char.IsAsciiDigit(name[end])) end++;
                if (end == start || end >= name.Length || name[end] != '.') continue;
                if (int.TryParse(name.AsSpan(start, end - start),
                                 NumberStyles.None, CultureInfo.InvariantCulture, out int index))
                    max = Math.Max(max, index);
            }
            return max + 1;
        }

        /// <summary>One-line summary for startup logs.</summary>
        public override string ToString() =>
            $"MiniMax-H3 DiT: hidden={HiddenSize} layers={NumLayers}(+{TokenRefinerLayers} refiner) " +
            $"heads={NumAttentionHeads}x{AttentionHeadDim} (inner={AttentionInnerDim}) ffn={FfnHiddenSize} " +
            $"video_ch={VideoLatentChannels} audio_ch={AudioLatentChannels} text_dim={TextDim} " +
            (PddSteps > 1 ? $"pdd_heads={PddSteps} " : "") +
            $"rope={RopeInvFreqLength}x3 (rot={RopeRotaryDim}/{AttentionHeadDim}) " +
            $"time={TimeEmbedding}({TimeEmbedDim}" +
            (AdaLnCurveGrid > 0 ? $", grid={AdaLnCurveGrid}" : "") + ")";
    }
}
