// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// The speech-output hyperparameters of an LFM3-Omni / LFM3-Audio mmproj (projector "lfm3a_gen"): the depthformer that
    /// turns one LM hidden row into a frame of 12 codes, the MaskGIT sampling defaults, and the LiquidCodec decoder with its
    /// host ISTFT. The GGUF stores the depthformer's sizes and the sampling defaults (<c>clip.gen.audio.*</c>); the codec's
    /// configuration is fixed by LiquidCodecConfig (llama.cpp mtmd-model.cpp hard-codes it and the converter refuses other
    /// checkpoints), and every value read from tensor shapes is checked against it at load.
    /// </summary>
    public sealed class Lfm3SpeechConfig
    {
        public const string ProjectorType = "lfm3a_gen";

        /// <summary>Codebooks a frame (num_codebooks).</summary>
        public const int Codebooks = 12;

        /// <summary>Codes of a codebook including the end-of-audio code (codebook_size).</summary>
        public const int CodebookSize = 2017;

        /// <summary>The end-of-audio code (audio_eos_code): any codebook sampling it ends the speech.</summary>
        public const int EndOfAudioCode = CodebookSize - 1;

        /// <summary>Codebooks revealed by each of a frame's depthformer passes (maskgit_schedule).</summary>
        public static ReadOnlySpan<int> MaskGitSchedule => new[] { 4, 4, 4 };

        /// <summary>FSQ levels of a code (prod = 2016 codes).</summary>
        public static ReadOnlySpan<int> FsqLevels => new[] { 8, 7, 6, 6 };

        /// <summary>Text slots, then audio slots, of an interleaved block (interleaved_n_text / interleaved_n_audio).</summary>
        public const int InterleavedTextSlots = 6, InterleavedAudioSlots = 12;

        // ---- depthformer (a.gen.code.*) ----

        /// <summary>The LM's hidden width (projection_dim): the depthformer's input and the feedback row's width.</summary>
        public int HiddenSize { get; private init; }
        public int DepthWidth { get; private init; }
        public int DepthFeedForward { get; private init; }
        public int DepthLayers { get; private init; }
        public int DepthHeads { get; private init; }
        public int DepthKvHeads { get; private init; }
        public int DepthHeadDim { get; private init; }
        public float DepthEps { get; private init; }
        public float DepthRopeTheta { get; private init; } = 1_000_000f;

        // ---- MaskGIT sampling defaults (clip.gen.audio.sampling.*; llama.cpp's fallbacks are the Omni card's) ----

        public float Temperature { get; private init; } = 0.4f;
        public int TopK { get; private init; }
        public float RepetitionPenalty { get; private init; } = 1.2f;

        /// <summary>Frames the repetition penalty looks back on (penalty_last_n).</summary>
        public int RepetitionWindow { get; private init; } = 64;

        // ---- LiquidCodec decoder (a.gen.wav.*) ----

        public int CodecWidth { get; private init; } = 512;
        public int CodecFeedForward { get; private init; } = 2304;
        public int CodecLayers { get; private init; } = 8;
        public int CodecHeads { get; private init; } = 8;
        public int CodecKvHeads { get; private init; } = 2;
        public int CodecHeadDim { get; private init; } = 64;
        public float CodecEps { get; private init; } = 1e-5f;
        public float CodecRopeTheta { get; private init; } = 1_000_000f;

        /// <summary>The decoder's sliding window, the query's own position included (decoder_window).</summary>
        public int CodecWindow { get; private init; } = 30;

        /// <summary>Taps of the decoder's causal depthwise convs (conv_L_cache); the carried state is Taps - 1 columns.</summary>
        public int CodecConvTaps { get; private init; } = 7;

        /// <summary>Decoder positions (STFT frames) a codec frame becomes (stft_frame_stack).</summary>
        public int FrameStack { get; private init; } = 6;

        /// <summary>Which decoder layers are short convs (the rest are sliding-window attention): 0, 1, 3, 5, 7.</summary>
        public bool[] CodecConvLayer { get; private init; }

        /// <summary>FSQ values a code (len(levels)) and the quantizer's width.</summary>
        public int FsqDim { get; private init; } = 4;
        public int QuantizerDim { get; private init; } = 256;

        // ---- ISTFT (host) ----

        public int SampleRate { get; private init; } = 24000;
        public int FftSize { get; private init; } = 1280;
        public int HopLength { get; private init; } = 320;

        /// <summary>Frequency bins of a spectrum frame (n_fft / 2 + 1); the decoder outputs log-magnitudes then phases.</summary>
        public int Bins => FftSize / 2 + 1;

        /// <summary>PCM samples a codec frame (80 ms at 24 kHz).</summary>
        public int SamplesPerFrame => FrameStack * HopLength;

        /// <summary>Whether <paramref name="mmproj"/> carries LFM3 speech output.</summary>
        public static bool IsPresent(GgufFile mmproj)
            => mmproj != null && string.Equals(mmproj.GetString("clip.gen.audio.projector_type"), ProjectorType, StringComparison.Ordinal)
               && mmproj.Tensors.ContainsKey("a.gen.code.proj_in.weight") && mmproj.Tensors.ContainsKey("a.gen.wav.tfm.out_proj.weight");

        /// <summary>The configuration of <paramref name="mmproj"/>; throws <see cref="InvalidDataException"/> naming what does
        /// not match the LFM3 speech-output networks.</summary>
        public static Lfm3SpeechConfig FromGguf(GgufFile mmproj)
        {
            ArgumentNullException.ThrowIfNull(mmproj);
            if (!IsPresent(mmproj))
                throw new InvalidDataException($"the mmproj has no LFM3 speech output (clip.gen.audio.projector_type '{mmproj.GetString("clip.gen.audio.projector_type")}', expected '{ProjectorType}' with a.gen.code.* and a.gen.wav.* tensors)");

            long[] Shape(string name)
            {
                if (!mmproj.Tensors.TryGetValue(name, out GgufTensorInfo info)) throw new InvalidDataException($"the mmproj has no tensor '{name}'");
                var ne = new long[info.Shape.Length];
                for (int i = 0; i < ne.Length; i++) ne[i] = (long)info.Shape[i];
                return ne;
            }
            void Expect(string name, params long[] ne)
            {
                long[] got = Shape(name);
                bool same = got.Length == ne.Length;
                for (int i = 0; same && i < ne.Length; i++) same = got[i] == ne[i];
                if (!same) throw new InvalidDataException($"mmproj tensor '{name}' is [{string.Join(", ", got)}], expected [{string.Join(", ", ne)}]");
            }
            int Key(string key, int fallback = -1)
            {
                if (!mmproj.Metadata.ContainsKey(key))
                {
                    if (fallback >= 0) return fallback;
                    throw new InvalidDataException($"the mmproj has no '{key}'");
                }
                return Convert.ToInt32(mmproj.Metadata[key]);
            }

            int hidden = Key("clip.gen.audio.projection_dim");
            int width = Key("clip.gen.audio.embedding_length");
            int ffn = Key("clip.gen.audio.feed_forward_length");
            int layers = Key("clip.gen.audio.block_count");
            int heads = Key("clip.gen.audio.attention.head_count");
            int kvHeads = Key("clip.gen.audio.attention.head_count_kv");
            float eps = mmproj.GetFloat32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
            int headDim = (int)Shape("a.gen.code.blk.0.attn_q_norm.weight")[0];
            if (heads * headDim != width || kvHeads < 1 || heads % kvHeads != 0)
                throw new InvalidDataException($"depthformer heads {heads} / {kvHeads} of {headDim} do not tile its width {width}");
            Expect("a.gen.code.proj_in.weight", hidden, (long)Codebooks * width);
            Expect("a.gen.code.proj_in.bias", (long)Codebooks * width);
            Expect("a.gen.code.mask_embd", width);
            Expect("a.gen.code.embd.weight", width, CodebookSize, Codebooks);
            Expect("a.gen.code.head_norm.weight", width, Codebooks);
            Expect("a.gen.code.output_norm.weight", width);
            Expect("a.gen.code.out_embd.weight", hidden, (long)Codebooks * CodebookSize);
            for (int l = 0; l < layers; l++)
            {
                string p = $"a.gen.code.blk.{l}.";
                Expect(p + "attn_q.weight", width, width);
                Expect(p + "attn_k.weight", width, (long)kvHeads * headDim);
                Expect(p + "attn_v.weight", width, (long)kvHeads * headDim);
                Expect(p + "attn_gate.weight", width, heads);
                Expect(p + "attn_out.weight", width, width);
                Expect(p + "ffn_gate.weight", width, ffn);
                Expect(p + "ffn_up.weight", width, ffn);
                Expect(p + "ffn_down.weight", ffn, width);
            }

            // The codec: LiquidCodecConfig's values, checked against the tensors.
            var config = new Lfm3SpeechConfig
            {
                HiddenSize = hidden,
                DepthWidth = width,
                DepthFeedForward = ffn,
                DepthLayers = layers,
                DepthHeads = heads,
                DepthKvHeads = kvHeads,
                DepthHeadDim = headDim,
                DepthEps = eps,
                Temperature = mmproj.Metadata.ContainsKey("clip.gen.audio.sampling.temp") ? mmproj.GetFloat32("clip.gen.audio.sampling.temp") : 0.4f,
                TopK = Key("clip.gen.audio.sampling.top_k", 0),
                RepetitionPenalty = mmproj.Metadata.ContainsKey("clip.gen.audio.sampling.penalty_repeat") ? mmproj.GetFloat32("clip.gen.audio.sampling.penalty_repeat") : 1.2f,
                RepetitionWindow = Key("clip.gen.audio.sampling.penalty_last_n", 64),
                CodecConvLayer = new bool[8],
            };
            int c = config.CodecWidth, levels = config.FsqDim;
            Expect("a.gen.wav.quant.first.codebook.weight", levels, EndOfAudioCode);
            Expect("a.gen.wav.quant.first.out_proj.weight", (long)levels * Codebooks, config.QuantizerDim);
            Expect("a.gen.wav.tfm.in_proj.weight", config.QuantizerDim, c);
            Expect("a.gen.wav.tfm.out_proj.weight", c, 2L * config.Bins);
            Expect("a.gen.wav.tfm.output_norm.weight", c);
            for (int l = 0; l < config.CodecLayers; l++)
            {
                string p = $"a.gen.wav.tfm.blk.{l}.";
                bool conv = mmproj.Tensors.ContainsKey(p + "conv_dw.weight");
                config.CodecConvLayer[l] = conv;
                if (conv)
                {
                    Expect(p + "conv_dw.weight", config.CodecConvTaps, c);
                    Expect(p + "conv_pw1.weight", c, 3L * c);
                    Expect(p + "conv_pw2.weight", c, c);
                }
                else
                {
                    Expect(p + "attn_q.weight", c, (long)config.CodecHeads * config.CodecHeadDim);
                    Expect(p + "attn_k.weight", c, (long)config.CodecKvHeads * config.CodecHeadDim);
                    Expect(p + "attn_v.weight", c, (long)config.CodecKvHeads * config.CodecHeadDim);
                    Expect(p + "attn_q_norm.weight", config.CodecHeadDim);
                    Expect(p + "attn_k_norm.weight", config.CodecHeadDim);
                    Expect(p + "attn_gate.weight", c, config.CodecHeads);
                    Expect(p + "attn_out.weight", (long)config.CodecHeads * config.CodecHeadDim, c);
                }
                Expect(p + "ln1.weight", c);
                Expect(p + "attn_post_norm.weight", c);
                Expect(p + "ln2.weight", c);
                Expect(p + "ffn_post_norm.weight", c);
                Expect(p + "ffn_gate.weight", c, config.CodecFeedForward);
                Expect(p + "ffn_up.weight", c, config.CodecFeedForward);
                Expect(p + "ffn_down.weight", config.CodecFeedForward, c);
            }
            if (Array.IndexOf(config.CodecConvLayer, false) < 0)
                throw new InvalidDataException("the codec decoder has no attention layer (expected sliding-window attention at layers 2, 4, 6)");
            return config;
        }

        /// <summary>A configuration for tests (small synthetic networks): every size given, the rest LFM3-Omni's.</summary>
        internal static Lfm3SpeechConfig ForTest(int hidden, int depthWidth, int depthFfn, int depthLayers, int depthHeads, int depthKvHeads,
            int codecWidth, int codecFfn, int codecHeads, int codecKvHeads, int codecHeadDim, bool[] codecConvLayer, int quantizerDim)
            => new()
            {
                HiddenSize = hidden,
                DepthWidth = depthWidth,
                DepthFeedForward = depthFfn,
                DepthLayers = depthLayers,
                DepthHeads = depthHeads,
                DepthKvHeads = depthKvHeads,
                DepthHeadDim = depthWidth / depthHeads,
                DepthEps = 1e-5f,
                CodecWidth = codecWidth,
                CodecFeedForward = codecFfn,
                CodecLayers = codecConvLayer.Length,
                CodecHeads = codecHeads,
                CodecKvHeads = codecKvHeads,
                CodecHeadDim = codecHeadDim,
                CodecConvLayer = (bool[])codecConvLayer.Clone(),
                QuantizerDim = quantizerDim,
            };
    }
}
