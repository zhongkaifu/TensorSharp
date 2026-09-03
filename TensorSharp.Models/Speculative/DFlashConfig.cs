// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Hyper-parameters of a DFlash speculative drafter GGUF
    /// (general.architecture = "dflash"), the block drafter that ships alongside a
    /// target model (Muse-Glimmer, Qwen 3.8, ...). Parsed from the drafter file
    /// only; every tensor the drafter needs beyond its own blocks (token_embd,
    /// output) is borrowed from the target model.
    ///
    /// TWO GENERATIONS share this architecture id, and which one a file is comes
    /// from the keys it carries, not from its name:
    ///
    ///   DFlash  - plain block diffusion. Each block position's token is the
    ///             argmax of the target LM head over that position's draft hidden
    ///             state, chosen INDEPENDENTLY of its neighbours.
    ///   DFlash2 - the same backbone plus two additions
    ///             (<see cref="HasConv"/> / <see cref="HasSelector"/>):
    ///               * a grouped dynamic depthwise K-tap convolution wrapped
    ///                 around every attention and every FFN sublayer
    ///                 (dflash.conv_kernel_size / dflash.conv_group_size), whose
    ///                 taps are produced per token by a projection of that
    ///                 sublayer's own input and masked at the block boundary, and
    ///               * a CANDIDATE SELECTOR (dflash.selector_rank /
    ///                 dflash.selector_top_k): instead of an independent argmax,
    ///                 the top-K candidates of adjacent positions are scored
    ///                 pairwise through two low-rank [vocab, r] codebooks and the
    ///                 block is read off as a walk through that lattice, so a
    ///                 position's token is conditioned on the one before it.
    ///             Both are no-ops when their keys are absent, which is what lets
    ///             one code path serve both generations.
    ///
    /// DFlash runs three passes (llama.cpp src/models/dflash.cpp):
    ///   A. ENCODE  - the target's per-layer INPUT residuals at
    ///                <see cref="TargetLayerIds"/>, concatenated into one
    ///                <see cref="FeatureSize"/>-wide row per committed position,
    ///                projected by fc.weight and RMS-normed by enc.output_norm.
    ///   B. INJECT  - that row feeds attn_k / attn_v of every draft layer; the
    ///                keys get a per-head RMSNorm + NeoX RoPE at the target
    ///                position and land in the drafter's own KV ring. No Q, no
    ///                attention, no FFN.
    ///   C. DRAFT   - [anchor, MASK x (block_size-1)] runs the 5 draft blocks
    ///                non-causally over [ring window | the block's own keys] and
    ///                the TARGET's LM head turns the result into
    ///                <see cref="BlockSize"/> rows of logits. Row 0 is the
    ///                anchor's own prediction and is discarded.
    /// </summary>
    public sealed class DFlashConfig
    {
        /// <summary>general.architecture of a DFlash drafter GGUF.</summary>
        public const string ArchName = "dflash";

        /// <summary>Name prefix under which the drafter's tensors are merged into
        /// the target model's shared weight dictionaries (mirrors Gemma 4's
        /// "mtp." prefix).</summary>
        public const string WeightPrefix = "dflash.";

        /// <summary>ggml GGML_ROPE_TYPE_NEOX. llama.cpp maps LLM_ARCH_DFLASH to
        /// LLAMA_ROPE_TYPE_NEOX, i.e. the split-halves flavour -- NOT the
        /// interleaved-pair (NORM, mode 0) RoPE the Muse-Glimmer target uses.</summary>
        public const int RopeTypeNeoX = 2;

        /// <summary>dflash.block_count (draft decoder blocks).</summary>
        public int NumLayers { get; private set; }

        /// <summary>dflash.embedding_length; must equal the target's hidden size.</summary>
        public int HiddenSize { get; private set; }

        /// <summary>dflash.feed_forward_length.</summary>
        public int IntermediateSize { get; private set; }

        /// <summary>dflash.attention.head_count (draft query heads).</summary>
        public int NumHeads { get; private set; }

        /// <summary>dflash.attention.head_count_kv. Note this is the DRAFTER's own
        /// GQA ratio (8 kv heads for Muse-Glimmer's dflash) and is unrelated to the
        /// target's (2) -- the drafter keeps a KV cache of its own.</summary>
        public int NumKVHeads { get; private set; }

        /// <summary>dflash.attention.key_length == value_length.</summary>
        public int HeadDim { get; private set; }

        /// <summary>dflash.rope.freq_base.</summary>
        public float RopeBase { get; private set; }

        /// <summary>dflash.attention.layer_norm_rms_epsilon. Every RMSNorm in the
        /// drafter (encoder, per-head Q/K, pre-attention, pre-FFN, final) uses it.</summary>
        public float Eps { get; private set; }

        /// <summary>dflash.block_size: the WIDTH of one draft block, including the
        /// anchor row. The number of usable drafts is <see cref="MaxDraftTokens"/>.</summary>
        public int BlockSize { get; private set; }

        /// <summary>dflash.target_layers: the target layers whose INPUT residual
        /// feeds the encoder, in the order they are concatenated. The converter
        /// already shifted these by +1, so id L means "the hidden state entering
        /// 0-based target layer L" == "the output of target layer L-1".</summary>
        public int[] TargetLayerIds { get; private set; } = Array.Empty<int>();

        /// <summary>dflash.attention.sliding_window (the drafter's own SWA span,
        /// applied against its KV ring).</summary>
        public int SlidingWindow { get; private set; }

        /// <summary>dflash.attention.sliding_window_pattern, one bool per draft
        /// layer (all true for the shipped drafter).</summary>
        public bool[] SwaPattern { get; private set; } = Array.Empty<bool>();

        /// <summary>tokenizer.ggml.mask_token_id: the id filling every draft slot
        /// past the anchor.</summary>
        public int MaskTokenId { get; private set; } = -1;

        /// <summary>dflash.conv_kernel_size: taps of the grouped dynamic
        /// convolution (2 for the shipped DFlash2 drafters). 0 = no convolution,
        /// i.e. a first-generation DFlash file.</summary>
        public int ConvKernelSize { get; private set; }

        /// <summary>dflash.conv_group_size: channels sharing one dynamic tap
        /// coefficient (16). The STATIC part of the kernel is per channel
        /// (blk.N.attn_conv_base); only the per-token delta is per group.</summary>
        public int ConvGroupSize { get; private set; }

        /// <summary>dflash.selector_rank: width of the two [vocab, r] transition
        /// codebooks. 0 = no selector (plain DFlash).</summary>
        public int SelectorRank { get; private set; }

        /// <summary>dflash.selector_top_k: candidates kept per block position
        /// before the lattice walk (16).</summary>
        public int SelectorTopK { get; private set; }

        /// <summary>Channel groups of the dynamic convolution
        /// (<see cref="HiddenSize"/> / <see cref="ConvGroupSize"/>).</summary>
        public int ConvNumGroups => ConvGroupSize > 0 ? HiddenSize / ConvGroupSize : 0;

        /// <summary>Columns of one conv_proj output row: both sides (the sublayer's
        /// input and its output) x taps x groups.</summary>
        public int ConvProjOutSize => 2 * ConvKernelSize * ConvNumGroups;

        /// <summary>True when this drafter wraps its sublayers in the DFlash2
        /// grouped dynamic convolution.</summary>
        public bool HasConv => ConvKernelSize > 0 && ConvGroupSize > 0;

        /// <summary>True when this drafter picks its block through the DFlash2
        /// candidate-selector lattice instead of a per-position argmax.</summary>
        public bool HasSelector => SelectorRank > 0 && SelectorTopK > 0;

        /// <summary>dflash.markov_rank: width of the DSpark Markov head (w1 rows /
        /// w2 columns) that conditions every draft position on the one before it,
        /// like the DSV4 DSpark head. 0 = no Markov head (a plain DFlash / DFlash2
        /// drafter). The Nemotron-3.5 DSpark drafter ships rank 512; llama.cpp
        /// derives it from the markov_w1.weight tensor shape rather than a key,
        /// which is why the shipped GGUF carries no dflash.markov_rank.</summary>
        public int MarkovRank { get; private set; }

        /// <summary>dflash.sample_from_anchor: true when EVERY block row (the
        /// anchor's own prediction included) is a draft and gets the Markov bias
        /// chained from the anchor; false when the anchor row is a "bonus" that
        /// passes through unbiased and drafting starts at row 1. llama.cpp treats a
        /// MISSING key as true; the key itself then means "the anchor's own row is
        /// a full draft, Markov bias included", which is how the reference runtimes
        /// drive the Nemotron-3.5 module (the checkpoint's own config field of the
        /// same name is about sampling, not drafting, and is not exported).</summary>
        public bool SampleFromAnchor { get; private set; } = true;

        /// <summary>True when this drafter carries a per-head attention-sink bias
        /// (blk.N.attn_sinks, e.g. the Nemotron-3.5 DSpark module). The sink is a
        /// per-head constant added to the normalized attention weights, giving the
        /// head a fixed attention-mass floor.</summary>
        public bool HasAttentionSinks { get; private set; }

        /// <summary>
        /// dflash.logit_scale: the multiplier the TARGET applies to its LM-head
        /// output. Only the DFlash2 selector needs it, and it needs it badly: the
        /// lattice ADDS the unary logit to a transition score, so an unscaled unary
        /// term is simply the wrong size and swamps the transition it is supposed to
        /// compete with. (Plain DFlash takes an argmax, which is invariant under a
        /// positive scale, which is why llama.cpp's DFlash graph can ignore it and
        /// why this key only appears on a DFlash2 file whose target has one -
        /// Muse-Glimmer's 0.196, against Qwen 3.8's absent = 1.0.)
        /// </summary>
        public float LogitScale { get; private set; } = 1f;

        /// <summary>dflash.final_logit_softcapping: the target's tanh softcap, applied
        /// to the selector's unary term after <see cref="LogitScale"/>. 0 = none.
        /// Same reasoning as <see cref="LogitScale"/>: monotonic, so it cannot change
        /// which candidates the top-k picks, but it very much changes how they weigh
        /// against the transition scores.</summary>
        public float FinalLogitSoftcap { get; private set; }

        /// <summary>True when the selector's unary term needs the target's logit
        /// transform applied before it enters the lattice.</summary>
        public bool HasUnaryLogitTransform => LogitScale != 1f || FinalLogitSoftcap > 0f;

        /// <summary>True for a second-generation drafter (either extension
        /// present). Descriptive only - every code path keys on
        /// <see cref="HasConv"/> / <see cref="HasSelector"/> individually.</summary>
        public bool IsDFlash2 => HasConv || HasSelector;

        /// <summary>Width of one encoder input row = TargetLayerIds.Length * HiddenSize
        /// (33280 for the Muse-Glimmer drafter). This is what the model reports as
        /// ISpeculativeModel.SpecFeatureSize.</summary>
        public int FeatureSize => TargetLayerIds.Length * HiddenSize;

        /// <summary>Tokens a block actually proposes. Plain DFlash / DFlash2:
        /// the block is [anchor, MASK x (BlockSize-1)] and row 0 (the anchor's own
        /// prediction) is discarded. A DSpark (Markov-head) drafter is the
        /// opposite: row 0 IS the anchor's own prediction through the block and
        /// every row is a draft, so the whole block width is usable.</summary>
        public int MaxDraftTokens => MarkovRank > 0 ? BlockSize : BlockSize - 1;

        /// <summary>Rows of the drafter's KV ring: the SWA span plus one whole
        /// block plus the anchor, rounded up to 32 so a wrapped write never aliases
        /// a row the same pass still reads (same sizing rule as the DSpark ring in
        /// Dsv4CudaEngine.Dspark.cs).</summary>
        public int RingRows => Pad(SlidingWindow + BlockSize + 1, 32);

        private static int Pad(int value, int alignment)
            => ((value + alignment - 1) / alignment) * alignment;

        /// <summary>Reads every "dflash.*" key from an opened drafter GGUF.</summary>
        public static DFlashConfig FromGguf(GgufFile gguf)
        {
            ArgumentNullException.ThrowIfNull(gguf);

            string arch = gguf.GetString("general.architecture") ?? string.Empty;
            if (!string.Equals(arch, ArchName, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Expected a '{ArchName}' GGUF for the Muse-Glimmer DFlash drafter but got '{arch}'.");
            }

            int keyLength = (int)gguf.GetUint32($"{ArchName}.attention.key_length");
            int valueLength = (int)gguf.GetUint32($"{ArchName}.attention.value_length", (uint)keyLength);
            if (keyLength != valueLength)
            {
                throw new NotSupportedException(
                    $"DFlash key_length {keyLength} != value_length {valueLength}; only square head dims are supported.");
            }

            var cfg = new DFlashConfig
            {
                NumLayers = (int)gguf.GetUint32($"{ArchName}.block_count"),
                HiddenSize = (int)gguf.GetUint32($"{ArchName}.embedding_length"),
                IntermediateSize = (int)gguf.GetUint32($"{ArchName}.feed_forward_length"),
                NumHeads = (int)gguf.GetUint32($"{ArchName}.attention.head_count"),
                NumKVHeads = (int)gguf.GetUint32($"{ArchName}.attention.head_count_kv"),
                HeadDim = keyLength,
                RopeBase = gguf.GetFloat32($"{ArchName}.rope.freq_base", 10000f),
                Eps = gguf.GetFloat32($"{ArchName}.attention.layer_norm_rms_epsilon", 1e-5f),
                BlockSize = (int)gguf.GetUint32($"{ArchName}.block_size"),
                TargetLayerIds = gguf.GetInt32Array($"{ArchName}.target_layers") ?? Array.Empty<int>(),
                // DSpark files (llama.cpp's export of the Nemotron-3.5 DSpark
                // module) omit these keys entirely: the drafter is SWA-1024 with
                // every layer sliding. SlidingWindow is resolved below once the
                // Markov head (the DSpark discriminator) is known.
                SlidingWindow = (int)gguf.GetUint32($"{ArchName}.attention.sliding_window", 0),
                SwaPattern = gguf.GetBoolArray($"{ArchName}.attention.sliding_window_pattern") ?? Array.Empty<bool>(),
                // A missing key yields uint.MaxValue, which casts to -1 and is
                // rejected by ValidateSelfConsistent below.
                MaskTokenId = (int)gguf.GetUint32("tokenizer.ggml.mask_token_id", uint.MaxValue),
                // DFlash2 extensions. Absent in a first-generation file, and a zero
                // there means the same thing as absent: the feature is off.
                ConvKernelSize = (int)gguf.GetUint32($"{ArchName}.conv_kernel_size", 0),
                ConvGroupSize = (int)gguf.GetUint32($"{ArchName}.conv_group_size", 0),
                SelectorRank = (int)gguf.GetUint32($"{ArchName}.selector_rank", 0),
                SelectorTopK = (int)gguf.GetUint32($"{ArchName}.selector_top_k", 0),
                LogitScale = gguf.GetFloat32($"{ArchName}.logit_scale", 1f),
                FinalLogitSoftcap = gguf.GetFloat32($"{ArchName}.final_logit_softcapping", 0f),
            };

            // DSpark Markov head. The GGUF carries no dflash.markov_rank key
            // (llama.cpp derives the rank from the markov_w1.weight shape) and its
            // TENSOR names are bare ("markov_w1.weight") - the "dflash." prefix is
            // applied when the loader merges them into the weight dictionaries, so
            // the probes below look at the raw file, not the merged names.
            if (gguf.Tensors.TryGetValue("markov_w1.weight", out var markovW1))
                cfg.MarkovRank = (int)markovW1.Shape[0];
            string sampleAnchor = gguf.GetString($"{ArchName}.sample_from_anchor");
            cfg.SampleFromAnchor = sampleAnchor == null
                || string.Equals(sampleAnchor.Trim(), "true", StringComparison.OrdinalIgnoreCase);
            cfg.HasAttentionSinks = gguf.Tensors.ContainsKey("blk.0.attn_sinks");
            if (cfg.MarkovRank > 0 && cfg.SlidingWindow == 0)
            {
                // DSpark convention: the Nemotron-3.5 module trains with a 1024-token
                // sliding window and the export omits the key. The mask is a no-op
                // until the context passes 1024 tokens, so a wrong default is
                // low-stakes, but matching the trained window is the faithful choice.
                cfg.SlidingWindow = 1024;
                Console.WriteLine("  DFlash: dflash.attention.sliding_window missing; defaulting to 1024 (DSpark convention).");
            }
            if (cfg.SwaPattern.Length == 0)
                cfg.SwaPattern = Enumerable.Repeat(true, cfg.NumLayers).ToArray();

            cfg.ApplyDiagnosticOverrides();
            cfg.ValidateSelfConsistent();
            return cfg;
        }

        /// <summary>
        /// TS_DFLASH_SELECTOR=0 / TS_DFLASH_CONV=0 turn off a DFlash2 extension and
        /// run the checkpoint as a first-generation DFlash drafter.
        ///
        /// DIAGNOSTIC ONLY. Neither is a supported way to run the model: the weights
        /// were trained WITH both, so switching one off changes what the drafter
        /// predicts. What they are for is attribution - how much of the acceptance
        /// rate comes from the selector, how much from the convolution, and what each
        /// costs per draft step - which is otherwise unanswerable without a second
        /// checkpoint.
        /// </summary>
        private void ApplyDiagnosticOverrides()
        {
            if (IsDisabled("TS_DFLASH_SELECTOR") && HasSelector)
            {
                Console.WriteLine("  DFlash: TS_DFLASH_SELECTOR=0 - drafting with per-position argmax instead of the candidate lattice (diagnostic).");
                SelectorRank = 0;
                SelectorTopK = 0;
            }
            if (IsDisabled("TS_DFLASH_CONV") && HasConv)
            {
                Console.WriteLine("  DFlash: TS_DFLASH_CONV=0 - drafting without the grouped dynamic convolution (diagnostic).");
                ConvKernelSize = 0;
                ConvGroupSize = 0;
            }
        }

        private static bool IsDisabled(string envVar)
            => string.Equals(Environment.GetEnvironmentVariable(envVar), "0", StringComparison.Ordinal);

        private void ValidateSelfConsistent()
        {
            if (NumLayers <= 0)
                throw new InvalidOperationException($"{ArchName}.block_count is missing or zero.");
            if (HiddenSize <= 0)
                throw new InvalidOperationException($"{ArchName}.embedding_length is missing or zero.");
            if (IntermediateSize <= 0)
                throw new InvalidOperationException($"{ArchName}.feed_forward_length is missing or zero.");
            if (NumHeads <= 0 || NumKVHeads <= 0 || NumHeads % NumKVHeads != 0)
            {
                throw new InvalidOperationException(
                    $"DFlash head counts do not group: head_count={NumHeads}, head_count_kv={NumKVHeads}.");
            }
            if (HeadDim <= 0 || (HeadDim & 1) != 0)
                throw new InvalidOperationException($"DFlash head dim {HeadDim} must be positive and even (NeoX RoPE splits it in half).");
            if (BlockSize < 2)
                throw new InvalidOperationException($"{ArchName}.block_size {BlockSize} must be at least 2 (anchor + one draft).");
            if (TargetLayerIds.Length == 0)
                throw new InvalidOperationException($"{ArchName}.target_layers is missing or empty.");
            if (SlidingWindow <= 0)
                throw new InvalidOperationException($"{ArchName}.attention.sliding_window {SlidingWindow} must be positive.");
            if (MaskTokenId < 0)
                throw new InvalidOperationException("tokenizer.ggml.mask_token_id is missing from the DFlash GGUF; the block draft has no mask id to fill its slots with.");
            // Both DFlash2 extensions are all-or-nothing: half a convolution, or a
            // rank with no top-k, describes a file we cannot execute, and quietly
            // ignoring the half that IS present would draft from a different model
            // than the one that was trained.
            if ((ConvKernelSize > 0) != (ConvGroupSize > 0))
            {
                throw new InvalidOperationException(
                    "DFlash grouped convolution needs conv_kernel_size and conv_group_size together "
                    + $"(got {ConvKernelSize} / {ConvGroupSize}).");
            }
            if (HasConv)
            {
                if (HiddenSize % ConvGroupSize != 0)
                {
                    throw new InvalidOperationException(
                        $"{ArchName}.conv_group_size {ConvGroupSize} must divide embedding_length {HiddenSize}.");
                }
                if (ConvKernelSize > BlockSize)
                {
                    // Tap t is masked out for the first t positions of a block, so a
                    // kernel wider than the block silently reduces to a narrower one -
                    // a shape the trained weights were never meant to run in.
                    throw new NotSupportedException(
                        $"{ArchName}.conv_kernel_size {ConvKernelSize} exceeds block_size {BlockSize}.");
                }
            }
            if ((SelectorRank > 0) != (SelectorTopK > 0))
            {
                throw new InvalidOperationException(
                    "DFlash selector needs selector_rank and selector_top_k together "
                    + $"(got {SelectorRank} / {SelectorTopK}).");
            }
            if (SwaPattern.Length != NumLayers)
            {
                throw new InvalidOperationException(
                    $"{ArchName}.attention.sliding_window_pattern has {SwaPattern.Length} entries for {NumLayers} layers.");
            }
            for (int il = 0; il < NumLayers; il++)
            {
                if (!SwaPattern[il])
                {
                    // The drafter's history lives in a RingRows-row ring sized for
                    // the SWA span; a full-attention draft layer would have to read
                    // arbitrarily far back, which the ring cannot serve.
                    throw new NotSupportedException(
                        $"DFlash draft layer {il} is not a sliding-window layer; the drafter's KV ring only holds {RingRows} positions.");
                }
            }
        }

        public override string ToString()
            => $"{(MarkovRank > 0 ? "dspark" : IsDFlash2 ? "dflash2" : "dflash")}(layers={NumLayers}, hidden={HiddenSize}, " +
               $"ffn={IntermediateSize}, heads={NumHeads}/{NumKVHeads}x{HeadDim}, " +
               $"block={BlockSize}, drafts={MaxDraftTokens}, swa={SlidingWindow}, ring={RingRows}, " +
               $"targets=[{string.Join(",", TargetLayerIds)}], feature={FeatureSize}, mask={MaskTokenId}" +
               (MarkovRank > 0 ? $", markov=r{MarkovRank}{(SampleFromAnchor ? "" : ", bonus-anchor")}" : string.Empty) +
               (HasAttentionSinks ? ", attn-sinks" : string.Empty) +
               (HasConv ? $", conv={ConvKernelSize}x{ConvGroupSize}({ConvNumGroups}g)" : string.Empty) +
               (HasSelector ? $", selector=r{SelectorRank}/k{SelectorTopK}" : string.Empty) +
               (LogitScale != 1f ? $", logit_scale={LogitScale:G6}" : string.Empty) +
               (FinalLogitSoftcap > 0f ? $", softcap={FinalLogitSoftcap:G6}" : string.Empty) + ")";
    }
}
