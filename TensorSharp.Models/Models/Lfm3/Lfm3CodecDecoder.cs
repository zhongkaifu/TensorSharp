// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;
using TensorSharp.Models.Graph;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// The LiquidCodec decoder (a.gen.wav.*; llama.cpp mtmd lfm3a-gen.cpp build_gen_wav, HF LiquidCodec decoder): frames of
    /// 12 codes to 24 kHz PCM. A call's frames run as one graph program on the weights' executor:
    /// <list type="number">
    /// <item>FSQ decode: each code's 4 level values (the GGUF's generated table), a frame's 48 values projected to 256, then
    /// to the decoder width 512.</item>
    /// <item>Nearest upsampling: a frame becomes 6 decoder positions (75 Hz).</item>
    /// <item>8 LFM3 blocks with sandwich norms: short convs (0, 1, 3, 5, 7: in-projection to B | C | x, B * x through a causal
    /// 7-tap depthwise conv, C * that, out-projection) and sliding-window attention (2, 4, 6: per-head q/k norms, NeoX rotary
    /// at absolute positions, 8 / 2 heads of 64, window 30 including the query, LFM3's per-head gate).</item>
    /// <item>The output norm and projection to 641 log-magnitudes and 641 phases a position, which the host ISTFT
    /// (<see cref="Lfm3StreamingIstft"/>) turns into 320 samples.</item>
    /// </list>
    /// Streams (<see cref="CreateStream"/>) carry the decoder's state between calls - the position counter, each conv's last
    /// 6 inputs and each attention layer's last 29 keys and values (device tensors the programs update in place) - so frames
    /// decode one at a time as they are sampled, and the result equals decoding the utterance at once.
    /// </summary>
    public sealed class Lfm3CodecDecoder
    {
        private readonly IGraphExecutor _executor;

        internal Lfm3CodecDecoder(Lfm3SpeechConfig config, NetworkWeights weights, IGraphExecutor executor = null)
        {
            Config = config;
            Weights = weights;
            _executor = executor;
        }

        public Lfm3SpeechConfig Config { get; }

        internal NetworkWeights Weights { get; }

        /// <summary>A new decoding stream (cold: position 0, empty state). Dispose it when the speech ends.</summary>
        public Lfm3CodecStream CreateStream() => new(this, _executor);
    }

    /// <summary>
    /// One speech stream of the codec decoder: <see cref="Decode(ReadOnlySpan{int}, List{float})"/> takes whole frames as they
    /// are sampled and returns their PCM (the first frame 1440 samples, every later one 1920; the ISTFT holds back 40 ms of
    /// overlap), <see cref="Flush"/> ends the segment (the held-back samples less the trailing padding) and returns the stream
    /// to its cold start, as llama.cpp resets its codec at every end-of-audio frame.
    /// </summary>
    public sealed class Lfm3CodecStream : IDisposable
    {
        private readonly Lfm3SpeechConfig _c;
        private readonly NetworkGraph _graph;
        private readonly Tensor[] _conv, _keys, _values;
        private readonly Lfm3StreamingIstft _istft;
        private bool _cold = true;
        private bool _disposed;

        internal Lfm3CodecStream(Lfm3CodecDecoder decoder, IGraphExecutor executor)
        {
            _c = decoder.Config;
            NetworkWeights w = decoder.Weights;
            _graph = new NetworkGraph(w, executor, "lfm3 codec decoder")
            {
                // As llama.cpp attends the codec (flash attention off for the projector): F32 products around a softmax. On
                // ggml-metal the PCM then equals llama.cpp's to float rounding (~122 dB SNR; ~83 dB with the flash kernel), at
                // the same speed for 6 rows over 35 keys.
                AttentionHalfKv = false,
                AttentionByProducts = true,
            };
            int layers = _c.CodecLayers, prefix = _c.CodecWindow - 1;
            _conv = new Tensor[layers];
            _keys = new Tensor[layers];
            _values = new Tensor[layers];
            for (int l = 0; l < layers; l++)
            {
                if (_c.CodecConvLayer[l])
                {
                    _conv[l] = Zeroed(new Tensor(w.Allocator, DType.Float32, _c.CodecWidth, _c.CodecConvTaps - 1));
                }
                else
                {
                    _keys[l] = Zeroed(new Tensor(w.Allocator, DType.Float32, prefix, _c.CodecKvHeads, _c.CodecHeadDim));
                    _values[l] = Zeroed(new Tensor(w.Allocator, DType.Float32, prefix, _c.CodecKvHeads, _c.CodecHeadDim));
                }
            }
            _istft = new Lfm3StreamingIstft(_c.FftSize, _c.HopLength);
        }

        private static Tensor Zeroed(Tensor t)
        {
            t.SetElementsAsFloat(new float[t.ElementCount()]);
            return t;
        }

        /// <summary>Decoder positions decoded since the stream's start (6 a frame): the next call's first rotary position.</summary>
        public int Position { get; private set; }

        /// <summary>Frames decoded since the stream's start.</summary>
        public int Frames => Position / _c.FrameStack;

        /// <summary>The graph the stream's calls run on (counters: Builds, Replays, Runs).</summary>
        public NetworkGraph Graph => _graph;

        /// <summary>
        /// Decode whole frames of <paramref name="codes"/> (frame-major, 12 a frame, each in [0, 2016): the end-of-audio code
        /// has no waveform) and append their samples to <paramref name="output"/>; returns how many.
        /// </summary>
        public int Decode(ReadOnlySpan<int> codes, List<float> output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(output);
            const int Cb = Lfm3SpeechConfig.Codebooks;
            if (codes.Length == 0 || codes.Length % Cb != 0) throw new ArgumentException($"whole frames of {Cb} codes", nameof(codes));
            for (int i = 0; i < codes.Length; i++)
                if ((uint)codes[i] >= Lfm3SpeechConfig.EndOfAudioCode)
                    throw new ArgumentOutOfRangeException(nameof(codes), $"code {codes[i]} (frame {i / Cb}, codebook {i % Cb}): codes are in [0, {Lfm3SpeechConfig.EndOfAudioCode})");
            int frames = codes.Length / Cb, positions = frames * _c.FrameStack, bins = _c.Bins;
            int[] input = codes.ToArray();
            bool cold = _cold;
            int start = Position;
            float[] spectra = _graph.Run(b => b.Output(Emit(b, input, frames, cold, start)))[0];
            _cold = false;
            Position += positions;
            int before = output.Count;
            for (int t = 0; t < positions; t++)
                _istft.ProcessFrame(spectra.AsSpan(t * 2 * bins, bins), spectra.AsSpan(t * 2 * bins + bins, bins), output);
            return output.Count - before;
        }

        /// <summary>Decode whole frames of codes; returns their samples.</summary>
        public float[] Decode(ReadOnlySpan<int> codes)
        {
            var output = new List<float>(codes.Length / Lfm3SpeechConfig.Codebooks * _c.SamplesPerFrame);
            Decode(codes, output);
            return output.ToArray();
        }

        /// <summary>The end of a speech segment: appends the samples the ISTFT holds back (less the trailing padding) and returns
        /// the stream to its cold start; returns how many.</summary>
        public int Flush(List<float> output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int n = Position > 0 ? _istft.Flush(output) : 0;
            Reset();
            return n;
        }

        /// <summary>The end of a speech segment; returns the samples the ISTFT held back.</summary>
        public float[] Flush()
        {
            var output = new List<float>(_c.FftSize);
            Flush(output);
            return output.ToArray();
        }

        /// <summary>Back to the cold start without the tail (a cancelled response).</summary>
        public void Reset()
        {
            _cold = true;
            Position = 0;
            _istft.Reset();
        }

        private NetValue Emit(NetworkBuilder b, int[] codes, int frames, bool cold, int start)
        {
            const int Cb = Lfm3SpeechConfig.Codebooks;
            int width = _c.CodecWidth, headDim = _c.CodecHeadDim, positions = frames * _c.FrameStack, prefix = _c.CodecWindow - 1;
            float eps = _c.CodecEps;
            // FSQ decode: one projection over a frame's 12 x 4 level values.
            NetValue q = b.GetRows(b.Weight("a.gen.wav.quant.first.codebook.weight"), b.InputI32(codes));           // [4, 12 frames]
            q = b.Reshape(q, (long)_c.FsqDim * Cb, frames);
            NetValue x = b.Linear(q, "a.gen.wav.quant.first.out_proj.weight", "a.gen.wav.quant.first.out_proj.bias");
            x = b.Linear(x, "a.gen.wav.tfm.in_proj.weight", "a.gen.wav.tfm.in_proj.bias");                           // [512, frames]
            x = b.UpsampleNearest(x, _c.FrameStack);                                                                   // [512, positions]
            NetValue rotary = b.Positions(start, positions);
            // Keys: the carried window (positions start - 29 .. start - 1; negative ones are a cold stream's empty slots), then
            // this call's.
            NetValue mask = b.Mask(NetworkMasks.SlidingWindow(positions, prefix + positions, _c.CodecWindow, start, start - prefix),
                prefix + positions, positions);
            for (int l = 0; l < _c.CodecLayers; l++)
            {
                string p = $"a.gen.wav.tfm.blk.{l}.";
                NetValue h = b.RmsNorm(x, p + "ln1.weight", eps);
                NetValue mixed;
                if (_c.CodecConvLayer[l])
                {
                    NetValue bcx = b.Linear(h, p + "conv_pw1.weight");                                                 // B | C | x
                    NetValue bx = b.Mul(b.Columns(bcx, 0, width), b.Columns(bcx, 2L * width, width));
                    NetValue state = cold ? default : b.State(_conv[l]);
                    NetValue conv = b.CausalConv1d(bx, b.Weight(p + "conv_dw.weight"), state, out NetValue window);
                    b.Write(b.ConvTail(window, _c.CodecConvTaps - 1), b.State(_conv[l]));
                    mixed = b.Linear(b.Mul(b.Columns(bcx, width, width), conv), p + "conv_pw2.weight");
                }
                else
                {
                    NetValue keyPrefix = cold ? b.Zeros(headDim, _c.CodecKvHeads, prefix) : b.State(_keys[l]);
                    NetValue valuePrefix = cold ? b.Zeros(headDim, _c.CodecKvHeads, prefix) : b.State(_values[l]);
                    NetValue attention = b.SelfAttention(h, _c.CodecHeads, _c.CodecKvHeads, headDim, p + "attn_q.weight", p + "attn_k.weight",
                        p + "attn_v.weight", null, p + "attn_q_norm.weight", p + "attn_k_norm.weight", eps,
                        v => b.RopeNeox(v, rotary, headDim, _c.CodecRopeTheta), 1f / MathF.Sqrt(headDim), mask, keyPrefix, valuePrefix,
                        out NetValue keys, out NetValue values);
                    b.Write(b.Tail(keys, 2, prefix), b.State(_keys[l]));
                    b.Write(b.Tail(values, 2, prefix), b.State(_values[l]));
                    mixed = b.Linear(b.HeadGate(attention, b.Linear(h, p + "attn_gate.weight"), headDim), p + "attn_out.weight");
                }
                // llama.cpp's operand order (build_lfm_layer): the normed mixer output first, which ggml-metal fuses with its norm.
                x = b.Add(b.RmsNorm(mixed, p + "attn_post_norm.weight", eps), x);
                NetValue ffn = b.FeedForward(b.RmsNorm(x, p + "ln2.weight", eps), p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight");
                x = b.Add(x, b.RmsNorm(ffn, p + "ffn_post_norm.weight", eps));
            }
            x = b.RmsNorm(x, "a.gen.wav.tfm.output_norm.weight", eps);
            return b.Linear(x, "a.gen.wav.tfm.out_proj.weight", "a.gen.wav.tfm.out_proj.bias");                         // [1282, positions]
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            IGraphExecutor executor = _graph.Executor;
            _graph.Dispose();
            foreach (Tensor[] set in new[] { _conv, _keys, _values })
            {
                foreach (Tensor t in set)
                {
                    if (t == null) continue;
                    // A GGML executor keeps device copies keyed on the state's host bytes: drop them before the bytes go.
                    if (executor.Capabilities.BindsHostPointers) executor.Invalidate(TensorComputePrimitives.GetStoragePointer(t));
                    t.Dispose();
                }
            }
        }
    }
}
