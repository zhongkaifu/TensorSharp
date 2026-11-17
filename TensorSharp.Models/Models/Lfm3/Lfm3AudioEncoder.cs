// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;

using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// LFM3-Omni's audio encoder (mmproj <c>clip.audio.projector_type = lfm3a</c>; llama.cpp tools/mtmd/models/lfm3a.cpp, HF
    /// <c>Lfm3OmniEncoder</c> + <c>Lfm3OmniProjector</c>): normalized log-mel frames (<see cref="Lfm3AudioPreprocessor"/>) to
    /// the LM's input rows, one row per 8 frames (80 ms), run as one graph program per call on the executor of the model's
    /// allocator (GGML metal / cuda / cpu / vulkan, the managed CPU, MLX, direct CUDA).
    /// <list type="bullet">
    /// <item>Downsampler: Conv2d(1 -> 256, k3, s2, p1) + SiLU, then twice depthwise Conv2d(k3, s2, p1) + pointwise Conv2d +
    /// SiLU (8x in time, 128 -> 16 mel), flattened channel-major (c * 16 + f) and projected 4096 -> 768.</item>
    /// <item>17 LFM2 layers (pre-norm RMSNorm, eps from the GGUF): causal short convolutions (kernel 3) at 0, 1, 3, 4, 6, 7, 9,
    /// 10, 12, 13; at 2, 5, 8, 11, 14, 15, 16 causal sliding-window attention (32 keys including the query's own, GQA 8:4,
    /// head 96, per-head q/k RMSNorm, NeoX rotary theta 1e6); SwiGLU feed-forward; final RMSNorm.</item>
    /// <item>Projector: LayerNorm (eps 1e-5) -> Linear 768 -> 2048 -> exact GELU -> Linear 2048 -> 2048.</item>
    /// </list>
    /// <para>
    /// The network is causal but for one future column per stride-2 conv, so it streams exactly: a call over a multiple of 8
    /// frames, its carried state (each conv stage's last input column, each short conv's last 2 inputs, each attention
    /// layer's last 31 keys and values, the position) then the next call, gives the rows of the whole clip.
    /// <see cref="Encode(ReadOnlySpan{float}, Action)"/> uses that to bound a long clip's memory: it runs chunks of
    /// <see cref="MaxChunkFrames"/> frames (default 3000 = 30 s, one program for any clip up to that). Layout inside the
    /// program: feature maps are [channels, mel, time] (time outermost, so a carried column is a contiguous slab); the conv
    /// kernels are transposed once at load to match.
    /// </para>
    /// <para>
    /// Offline encodes are serialized (they share one set of carried tensors); each <see cref="Lfm3AudioEncoderStream"/> has
    /// its own state and programs. The caller holds the model's GPU lock around a call, as for any encoder, and may yield it
    /// between a long clip's chunks (<c>betweenChunks</c>).
    /// </para>
    /// </summary>
    public sealed class Lfm3AudioEncoder : IDisposable
    {
        public const string ProjectorType = "lfm3a";

        /// <summary>The encoder's rotary base: llama.cpp hard-codes it (it is not in the GGUF); HF's converter requires it.</summary>
        public const float RopeTheta = 1e6f;

        /// <summary>The projector LayerNorm's eps (torch's nn.LayerNorm default; hard-coded in llama.cpp).</summary>
        public const float ProjectorEps = 1e-5f;

        private const int MelBins = Lfm3AudioPreprocessor.MelBins;
        private readonly NetworkWeights _weights;
        private readonly NetworkGraph _graph;
        private readonly StateSet _offlineStates;
        private readonly object _offlineSync = new();
        private readonly bool[] _convLayer;
        private readonly IGraphExecutor _executor;
        private int _maxChunkFrames = 3000;
        private bool _disposed;

        private Lfm3AudioEncoder(NetworkWeights weights, IGraphExecutor executor, int width, int layers, int heads, int kvHeads, int ffn,
            int window, int outputWidth, int channels, float eps, bool[] convLayer)
        {
            _weights = weights;
            _executor = executor;
            Width = width;
            LayerCount = layers;
            Heads = heads;
            KvHeads = kvHeads;
            HeadDim = width / heads;
            FeedForwardWidth = ffn;
            Window = window;
            OutputWidth = outputWidth;
            Channels = channels;
            Eps = eps;
            _convLayer = convLayer;
            _graph = new NetworkGraph(weights, executor, "lfm3 audio encoder");
            _offlineStates = new StateSet(this);
        }

        public int Width { get; }
        public int LayerCount { get; }
        public int Heads { get; }
        public int KvHeads { get; }
        public int HeadDim { get; }
        public int FeedForwardWidth { get; }

        /// <summary>Keys an attention layer sees, the query's own included (<c>clip.audio.window_size</c>).</summary>
        public int Window { get; }

        /// <summary>Width of the rows handed to the LM (<c>clip.audio.projection_dim</c>; the LM's hidden size).</summary>
        public int OutputWidth { get; }

        /// <summary>The downsampler's channels (256).</summary>
        public int Channels { get; }

        public float Eps { get; }

        /// <summary>Whether layer <paramref name="layer"/> is a short-conv layer (else attention).</summary>
        public bool IsConvLayer(int layer) => _convLayer[layer];

        /// <summary>The executor the programs run on.</summary>
        public IGraphExecutor Executor => _graph.Executor;

        /// <summary>The offline graph (counters, plan cache).</summary>
        public NetworkGraph Graph => _graph;

        public NetworkWeights Weights => _weights;

        /// <summary>Frames one program of <see cref="Encode(ReadOnlySpan{float}, Action)"/> covers at most (a multiple of 8):
        /// longer clips run as several programs with carried state (the same rows; bounded memory, and a point between
        /// chunks where the caller can yield the GPU).</summary>
        public int MaxChunkFrames
        {
            get => _maxChunkFrames;
            set
            {
                if (value < Lfm3AudioPreprocessor.FramesPerRow || value % Lfm3AudioPreprocessor.FramesPerRow != 0)
                    throw new ArgumentOutOfRangeException(nameof(value), "a positive multiple of 8 frames");
                _maxChunkFrames = value;
            }
        }

        // ================================================================ loading

        /// <summary>Whether <paramref name="mmproj"/> holds an LFM3 audio tower (<c>clip.has_audio_encoder</c> with
        /// projector type lfm3a).</summary>
        public static bool HasAudioEncoder(GgufFile mmproj)
        {
            ArgumentNullException.ThrowIfNull(mmproj);
            return mmproj.GetBool("clip.has_audio_encoder") && mmproj.GetString("clip.audio.projector_type") == ProjectorType;
        }

        /// <summary>Load the audio tower of the mmproj at <paramref name="mmprojPath"/> onto <paramref name="allocator"/>
        /// (the model's), or null when the file has no audio tower.</summary>
        public static Lfm3AudioEncoder Load(string mmprojPath, IAllocator allocator, BackendType backend)
        {
            using var gguf = new GgufFile(mmprojPath);
            return Load(gguf, allocator, backend);
        }

        /// <summary>Load the audio tower (<c>a.*</c> but the speech generator's <c>a.gen.*</c>, and <c>mm.a.*</c>) of
        /// <paramref name="mmproj"/>, or null when it has none. A tower that is not lfm3a's shape is refused with the reason.</summary>
        public static Lfm3AudioEncoder Load(GgufFile mmproj, IAllocator allocator, BackendType backend) => Load(mmproj, allocator, backend, null);

        /// <summary>As <see cref="Load(GgufFile, IAllocator, BackendType)"/>, the programs on <paramref name="executor"/> in place
        /// of the one registered for the allocator (null: that one): a reference executor in a test.</summary>
        internal static Lfm3AudioEncoder Load(GgufFile mmproj, IAllocator allocator, BackendType backend, IGraphExecutor executor)
        {
            ArgumentNullException.ThrowIfNull(mmproj);
            ArgumentNullException.ThrowIfNull(allocator);
            if (!mmproj.GetBool("clip.has_audio_encoder")) return null;
            string type = mmproj.GetString("clip.audio.projector_type");
            if (type != ProjectorType)
                throw new NotSupportedException($"the mmproj's audio tower is '{type}', not LFM3's '{ProjectorType}'");
            int width = Required(mmproj, "clip.audio.embedding_length");
            int layers = Required(mmproj, "clip.audio.block_count");
            int heads = Required(mmproj, "clip.audio.attention.head_count");
            int kvHeads = (int)mmproj.GetUint32("clip.audio.attention.head_count_kv", (uint)heads);
            int ffn = Required(mmproj, "clip.audio.feed_forward_length");
            int window = Required(mmproj, "clip.audio.window_size");
            int outputWidth = Required(mmproj, "clip.audio.projection_dim");
            int mels = Required(mmproj, "clip.audio.num_mel_bins");
            float eps = mmproj.GetFloat32("clip.audio.attention.layer_norm_epsilon", 1e-5f);
            if (mels != MelBins)
                throw new NotSupportedException($"the lfm3a tower takes {mels} mel bins; the LFM3 front end computes {MelBins}");
            if (width % heads != 0 || heads % kvHeads != 0)
                throw new NotSupportedException($"lfm3a tower with {width} wide, {heads} heads and {kvHeads} KV heads");
            if (window < 1) throw new InvalidDataException("clip.audio.window_size must be positive");

            var convLayer = new bool[layers];
            for (int l = 0; l < layers; l++)
            {
                bool conv = mmproj.Tensors.ContainsKey($"a.blk.{l}.conv_pw1.weight");
                bool attn = mmproj.Tensors.ContainsKey($"a.blk.{l}.attn_q.weight");
                if (conv == attn)
                    throw new InvalidDataException($"lfm3a layer {l} must be either a short-conv layer or an attention layer");
                convLayer[l] = conv;
            }

            var weights = new NetworkWeights(allocator, backend);
            try
            {
                weights.LoadGguf(mmproj, n => n.StartsWith("a.", StringComparison.Ordinal) && !n.StartsWith("a.gen.", StringComparison.Ordinal)
                    || n.StartsWith("mm.a.", StringComparison.Ordinal));
                NetworkWeight conv0 = weights.Get("a.conv1d.0.weight");
                if (conv0.Ne.Length != 4 || conv0.Ne[0] != 3 || conv0.Ne[1] != 3 || conv0.Ne[2] != 1)
                    throw new InvalidDataException($"a.conv1d.0.weight is [{string.Join(", ", conv0.Ne)}], not [3, 3, 1, C]");
                int channels = (int)conv0.Ne[3];
                if (weights.Get("a.pre_encode.out.weight").RowWidth != channels * (MelBins / 8))
                    throw new InvalidDataException($"a.pre_encode.out.weight takes {weights.Get("a.pre_encode.out.weight").RowWidth} features, not {channels} x {MelBins / 8}");
                // The kernels in this program's [channels, mel, time] layout: ggml's [KW = time, KH = mel] transposed.
                foreach (string k in new[] { "a.conv1d.0.weight", "a.conv1d.2.weight", "a.conv1d.5.weight" })
                {
                    NetworkWeight w = weights.Get(k);
                    float[] src = weights.ReadF32(k);
                    var dst = new float[src.Length];
                    long planes = src.Length / 9;
                    for (long p = 0; p < planes; p++)
                        for (int kt = 0; kt < 3; kt++)
                            for (int kf = 0; kf < 3; kf++)
                                dst[p * 9 + kt * 3 + kf] = src[p * 9 + kf * 3 + kt];
                    weights.AddF32(k + "#mel-time", dst, w.Ne);
                }
                return new Lfm3AudioEncoder(weights, executor, width, layers, heads, kvHeads, ffn, window, outputWidth, channels, eps, convLayer);
            }
            catch
            {
                weights.Dispose();
                throw;
            }
        }

        private static int Required(GgufFile f, string key)
        {
            uint v = f.GetUint32(key, 0);
            if (v == 0) throw new InvalidDataException($"the mmproj lacks '{key}'");
            return checked((int)v);
        }

        // ================================================================ offline

        /// <summary>
        /// The LM rows of a 16 kHz mono clip: [rows][<see cref="OutputWidth"/>] row-major, rows =
        /// <see cref="Lfm3AudioPreprocessor.RowCountForSamples"/>. <paramref name="betweenChunks"/> runs between the programs
        /// of a clip longer than <see cref="MaxChunkFrames"/> (the model's GPU-lock yield).
        /// </summary>
        public float[] Encode(ReadOnlySpan<float> samples, Action betweenChunks = null)
        {
            float[] features = Lfm3AudioPreprocessor.ComputeFeatures(samples, out int frames);
            return EncodeFeatures(features, frames, betweenChunks);
        }

        /// <summary>The LM rows of a clip's features (frame-major [frames][128], <see cref="Lfm3AudioPreprocessor"/>).</summary>
        public float[] EncodeFeatures(float[] features, int frames, Action betweenChunks = null)
            => EncodeFeatures(features, frames, null, betweenChunks);

        /// <summary>
        /// As <see cref="EncodeFeatures(float[], int, Action)"/>; where <paramref name="intermediates"/> is given (validation),
        /// the clip runs as one program and also returns the values llama.cpp's graph names: "downsample.conv" ([time][mel 16]
        /// [channel 256], this program's layout), "downsample.out", "operator_norm-{layer}", "embedding_norm" (each
        /// [rows][768]).
        /// </summary>
        public float[] EncodeFeatures(float[] features, int frames, IDictionary<string, float[]> intermediates, Action betweenChunks = null)
        {
            ArgumentNullException.ThrowIfNull(features);
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (frames < 0 || features.LongLength != (long)frames * MelBins)
                throw new ArgumentException($"{features.LongLength} feature values for {frames} frames of {MelBins} mels");
            int rows = Lfm3AudioPreprocessor.RowCount(frames);
            var result = new float[(long)rows * OutputWidth];
            if (frames == 0) return result;
            lock (_offlineSync)
            {
                int chunk = intermediates != null ? frames : MaxChunkFrames;
                int position = 0;
                for (int first = 0; first < frames; first += chunk)
                {
                    if (first > 0) betweenChunks?.Invoke();
                    int count = Math.Min(chunk, frames - first);
                    float[] part = count == frames ? features : features.AsSpan((int)((long)first * MelBins), count * MelBins).ToArray();
                    float[] y = Run(_graph, _offlineStates, part, count, fresh: first == 0, position, intermediates);
                    Array.Copy(y, 0, result, (long)position * OutputWidth, y.LongLength);
                    position += Lfm3AudioPreprocessor.RowCount(count);
                }
            }
            return result;
        }

        /// <summary>A tensor [rows, <see cref="OutputWidth"/>] of <paramref name="rows"/> on the encoder's allocator (what
        /// a model's QueueMedia takes).</summary>
        public Tensor ToTensor(float[] rows)
        {
            ArgumentNullException.ThrowIfNull(rows);
            if (rows.LongLength % OutputWidth != 0) throw new ArgumentException($"{rows.LongLength} values are not rows of {OutputWidth}");
            var t = new Tensor(_weights.Allocator, DType.Float32, rows.LongLength / OutputWidth, OutputWidth);
            t.SetElementsAsFloat(rows);
            return t;
        }

        // ================================================================ streaming

        /// <summary>A stream of this encoder: frames in (a multiple of 8 a call until the last), rows out, equal to the
        /// whole clip's. Each stream carries its own state and programs; dispose it when the audio ends.</summary>
        public Lfm3AudioEncoderStream OpenStream() => new(this);

        /// <summary>A live stream from 16 kHz samples to rows (front end and encoder), encoding every
        /// <paramref name="stepFrames"/> frames (a multiple of 8; default 32 = 320 ms, llama.cpp's default step).</summary>
        public Lfm3AudioStream OpenAudioStream(int stepFrames = 32) => new(this, stepFrames);

        internal NetworkGraph NewStreamGraph() => new(_weights, _executor, "lfm3 audio stream") { MaxCachedPlans = 3 };

        // ================================================================ the program

        /// <summary>Each carried value of a stream: per conv stage its last input column, per short conv its last
        /// K - 1 inputs, per attention layer its last Window - 1 keys and values (post-norm, post-rotary).</summary>
        internal sealed class StateSet : IDisposable
        {
            public readonly Tensor Ds0, Ds1, Ds2;
            public readonly Tensor[] Conv, Keys, Values;

            private readonly IGraphExecutor _executor;

            public StateSet(Lfm3AudioEncoder e)
            {
                _executor = e.Executor;
                IAllocator a = e._weights.Allocator;
                int prefix = e.Window - 1, kvWidth = e.HeadDim * e.KvHeads;
                Ds0 = Zeroed(a, MelBins);
                Ds1 = Zeroed(a, (MelBins / 2) * e.Channels);
                Ds2 = Zeroed(a, (MelBins / 4) * e.Channels);
                Conv = new Tensor[e.LayerCount];
                Keys = new Tensor[e.LayerCount];
                Values = new Tensor[e.LayerCount];
                for (int l = 0; l < e.LayerCount; l++)
                {
                    if (e._convLayer[l]) Conv[l] = Zeroed(a, e.Width * (KernelOf(e, l) - 1));
                    else
                    {
                        Keys[l] = Zeroed(a, (long)kvWidth * prefix);
                        Values[l] = Zeroed(a, (long)kvWidth * prefix);
                    }
                }
            }

            private static Tensor Zeroed(IAllocator a, long n)
            {
                var t = new Tensor(a, DType.Float32, n);
                t.SetElementsAsFloat(new float[n]);
                return t;
            }

            /// <summary>Frees the tensors; an executor that binds host memory (GGML) first drops what it keeps for them, so
            /// a later allocation at the same address is not taken for this one.</summary>
            public void Dispose()
            {
                IGraphExecutor executor = _executor;
                void Free(Tensor t)
                {
                    if (t == null) return;
                    if (executor is { Capabilities.BindsHostPointers: true }) executor.Invalidate(TensorComputePrimitives.GetStoragePointer(t));
                    t.Dispose();
                }
                Free(Ds0);
                Free(Ds1);
                Free(Ds2);
                foreach (Tensor t in Conv) Free(t);
                foreach (Tensor t in Keys) Free(t);
                foreach (Tensor t in Values) Free(t);
            }
        }

        private static int KernelOf(Lfm3AudioEncoder e, int layer) => (int)e._weights.Get($"a.blk.{layer}.conv_dw.weight").Ne[0];

        /// <summary>One program over <paramref name="frames"/> frames whose first row sits at <paramref name="position"/>:
        /// a fresh clip reads zeros for every carried value (no state read), a continued one reads <paramref name="states"/>;
        /// both write the states for a next call. Returns a copy of the rows.</summary>
        internal float[] Run(NetworkGraph graph, StateSet states, float[] features, int frames, bool fresh, int position,
            IDictionary<string, float[]> intermediates = null)
        {
            var captured = intermediates != null ? new List<(string Name, int Index)>() : null;
            int rowsOutput = 0;
            float[][] outputs = graph.Run(b =>
            {
                captured?.Clear();
                NetValue rows = Emit(b, states, features, frames, fresh, position, captured);
                rowsOutput = b.Output(rows);
            });
            if (intermediates != null)
                foreach ((string name, int index) in captured) intermediates[name] = (float[])outputs[index].Clone();
            return (float[])outputs[rowsOutput].Clone();
        }

        private NetValue Emit(NetworkBuilder b, StateSet st, float[] features, int frames, bool fresh, int position,
            List<(string Name, int Index)> capture)
        {
            int c = Channels;
            NetValue x = b.Input(features, 1, MelBins, frames);   // [1, mel, time]

            // Downsampler: three stride-2 convs over (mel, time); time gets the carried column in front, a zero column behind.
            x = b.Silu(b.Conv2d(Down(b, x, st.Ds0, fresh), "a.conv1d.0.weight#mel-time", "a.conv1d.0.bias", 2, 2, 1, 0));
            x = Pointwise(b, b.Add(b.DepthwiseConv2d(Down(b, x, st.Ds1, fresh), b.Weight("a.conv1d.2.weight#mel-time"), 2, 2, 1, 0),
                b.Vector("a.conv1d.2.bias", c)), "a.conv1d.3");
            x = Pointwise(b, b.Add(b.DepthwiseConv2d(Down(b, x, st.Ds2, fresh), b.Weight("a.conv1d.5.weight#mel-time"), 2, 2, 1, 0),
                b.Vector("a.conv1d.5.bias", c)), "a.conv1d.6");
            long mels = x.Ne1, n = x.Ne2;   // [c, 16, rows]
            Capture(b, capture, "downsample.conv", x);
            // Features channel-major (c * 16 + f), as HF's flatten(1, 2) of (B, C, F, T).
            x = b.Linear(b.Reshape(b.Permute(x, 1, 0, 2, 3), c * mels, n), "a.pre_encode.out.weight", "a.pre_encode.out.bias");
            Capture(b, capture, "downsample.out", x);

            int prefix = Window - 1;
            NetValue positions = default, mask = default;
            for (int l = 0; l < LayerCount; l++)
            {
                string p = $"a.blk.{l}.";
                NetValue h = b.RmsNorm(x, p + "ln1.weight", Eps);
                Capture(b, capture, $"operator_norm-{l}", h);
                NetValue mixed;
                if (_convLayer[l])
                {
                    // Short conv: in_proj -> B * x -> causal depthwise conv (k taps) -> C * conv -> out_proj (HF chunk order B, C, x).
                    NetValue bcx = b.Linear(h, p + "conv_pw1.weight");
                    NetValue gated = b.Mul(b.Columns(bcx, 0, Width), b.Columns(bcx, 2 * Width, Width));
                    NetValue kernel = b.Weight(p + "conv_dw.weight");
                    long taps = kernel.Ne0 - 1;
                    NetValue carried = fresh ? b.Zeros(taps, Width) : b.State(st.Conv[l], taps, Width);
                    NetValue conv = b.CausalConv1d(gated, kernel, carried, out NetValue window);
                    b.Write(b.ConvTail(window, taps), b.State(st.Conv[l], taps, Width));
                    mixed = b.Linear(b.Mul(b.Columns(bcx, Width, Width), conv), p + "conv_pw2.weight");
                }
                else
                {
                    if (!positions.Valid)
                    {
                        positions = b.Positions(position, (int)n);
                        mask = b.Mask(NetworkMasks.SlidingWindow((int)n, prefix + (int)n, Window, position, position - prefix), prefix + n, n);
                    }
                    NetValue rope = positions;
                    NetValue keyPrefix = fresh ? b.Zeros(HeadDim, KvHeads, prefix) : b.State(st.Keys[l], HeadDim, KvHeads, prefix);
                    NetValue valuePrefix = fresh ? b.Zeros(HeadDim, KvHeads, prefix) : b.State(st.Values[l], HeadDim, KvHeads, prefix);
                    mixed = b.SelfAttention(h, Heads, KvHeads, HeadDim, p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
                        _weights.Contains(p + "attn_out.weight") ? p + "attn_out.weight" : null, p + "attn_q_norm.weight", p + "attn_k_norm.weight",
                        Eps, v => b.RopeNeox(v, rope, HeadDim, RopeTheta), 1f / MathF.Sqrt(HeadDim), mask, keyPrefix, valuePrefix,
                        out NetValue keys, out NetValue values);
                    b.Write(b.Tail(keys, 2, prefix), b.State(st.Keys[l], HeadDim, KvHeads, prefix));
                    b.Write(b.Tail(values, 2, prefix), b.State(st.Values[l], HeadDim, KvHeads, prefix));
                }
                x = b.Add(x, mixed);
                x = b.Add(x, b.FeedForward(b.RmsNorm(x, p + "ln2.weight", Eps), p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"));
            }
            x = b.RmsNorm(x, "a.post_ln.weight", Eps);
            Capture(b, capture, "embedding_norm", x);
            x = b.LayerNorm(x, "mm.a.mlp.0.weight", "mm.a.mlp.0.bias", ProjectorEps);
            return b.GeluMlp(x, "mm.a.mlp.1.weight", "mm.a.mlp.1.bias", "mm.a.mlp.3.weight", "mm.a.mlp.3.bias");
        }

        /// <summary>A stride-2 conv stage's input over [ch, mel, time]: the carried column (zeros for a fresh clip) in front,
        /// this call's last column written back as the next carried one, a zero column behind (the conv's right padding; with
        /// a multiple of 8 frames a call never reads it but at the clip's end).</summary>
        private static NetValue Down(NetworkBuilder b, NetValue x, Tensor state, bool fresh)
        {
            NetValue left = fresh ? b.Zeros(x.Ne0, x.Ne1, 1) : b.State(state, x.Ne0, x.Ne1, 1);
            NetValue joined = b.Concat(left, x, 2);
            b.Write(b.Tail(joined, 2, 1), b.State(state, x.Ne0, x.Ne1, 1));
            return b.Concat(joined, b.Zeros(x.Ne0, x.Ne1, 1), 2);
        }

        /// <summary>A pointwise conv (a [1, 1, C, C] kernel and its bias) over [C, mel, time], then SiLU.</summary>
        private NetValue Pointwise(NetworkBuilder b, NetValue x, string name)
        {
            long c = x.Ne0, mels = x.Ne1, n = x.Ne2;
            NetValue y = b.Linear(b.Reshape(x, c, mels * n), b.Weight(name + ".weight", c, Channels), b.Vector(name + ".bias", Channels));
            return b.Reshape(b.Silu(y), Channels, mels, n);
        }

        private static void Capture(NetworkBuilder b, List<(string, int)> capture, string name, NetValue x)
        {
            if (capture != null) capture.Add((name, b.Output(x)));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _graph.Dispose();
            _offlineStates.Dispose();
            _weights.Dispose();
        }
    }

    /// <summary>
    /// A stream of an <see cref="Lfm3AudioEncoder"/>: features in, rows out. Every call but the last takes a multiple of 8
    /// frames (a row's worth); the rows then equal the whole clip's (the encoder's carried state makes the cut invisible).
    /// The last call (<c>final</c>) may take any count, padded as the clip's end.
    /// </summary>
    public sealed class Lfm3AudioEncoderStream : IDisposable
    {
        private readonly Lfm3AudioEncoder _encoder;
        private readonly NetworkGraph _graph;
        private readonly Lfm3AudioEncoder.StateSet _states;
        private bool _finished, _disposed;

        internal Lfm3AudioEncoderStream(Lfm3AudioEncoder encoder)
        {
            _encoder = encoder;
            _graph = encoder.NewStreamGraph();
            _states = new Lfm3AudioEncoder.StateSet(encoder);
        }

        /// <summary>Rows produced so far (the next row's position in the encoder).</summary>
        public int Rows { get; private set; }

        /// <summary>Frames consumed so far.</summary>
        public long Frames { get; private set; }

        public bool Finished => _finished;

        /// <summary>The stream's own programs (counters).</summary>
        public NetworkGraph Graph => _graph;

        /// <summary>Encode <paramref name="frames"/> frames (frame-major [frames][128]); a non-final call takes a multiple of
        /// 8. Returns their rows [rows][OutputWidth] (rows = frames / 8, or ceil for the final call).</summary>
        public float[] Encode(float[] features, int frames, bool final)
        {
            ArgumentNullException.ThrowIfNull(features);
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_finished) throw new InvalidOperationException("the encoder stream has finished");
            if (frames < 0 || features.LongLength != (long)frames * Lfm3AudioPreprocessor.MelBins)
                throw new ArgumentException($"{features.LongLength} feature values for {frames} frames");
            if (!final && frames % Lfm3AudioPreprocessor.FramesPerRow != 0)
                throw new ArgumentException($"a non-final call takes a multiple of {Lfm3AudioPreprocessor.FramesPerRow} frames, not {frames}");
            if (final) _finished = true;
            if (frames == 0) return Array.Empty<float>();
            float[] rows = _encoder.Run(_graph, _states, features, frames, fresh: Frames == 0, Rows);
            Frames += frames;
            Rows += Lfm3AudioPreprocessor.RowCount(frames);
            return rows;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _graph.Dispose();
            _states.Dispose();
        }
    }

    /// <summary>
    /// Live 16 kHz audio to LM rows: the front end (<see cref="Lfm3AudioFeatureStream"/>) and an encoder stream, encoding
    /// whenever <see cref="StepFrames"/> new frames are complete, as llama.cpp's <c>mtmd_audio_stream</c> does; the rows
    /// equal the whole clip's. <see cref="Finish"/> flushes the rest with the clip's end padding.
    /// </summary>
    public sealed class Lfm3AudioStream : IDisposable
    {
        private readonly Lfm3AudioFeatureStream _features = new();
        private readonly Lfm3AudioEncoderStream _encoder;
        private float[] _pending = Array.Empty<float>();
        private int _pendingFrames;

        internal Lfm3AudioStream(Lfm3AudioEncoder encoder, int stepFrames)
        {
            if (stepFrames < Lfm3AudioPreprocessor.FramesPerRow || stepFrames % Lfm3AudioPreprocessor.FramesPerRow != 0)
                throw new ArgumentOutOfRangeException(nameof(stepFrames), "a positive multiple of 8 frames");
            StepFrames = stepFrames;
            OutputWidth = encoder.OutputWidth;
            _encoder = encoder.OpenStream();
        }

        public int StepFrames { get; }

        public int OutputWidth { get; }

        /// <summary>Rows produced so far.</summary>
        public int Rows => _encoder.Rows;

        public Lfm3AudioEncoderStream Encoder => _encoder;

        /// <summary>Append samples; returns the rows of every whole step now complete ([rows][OutputWidth], possibly none).</summary>
        public float[] Push(ReadOnlySpan<float> samples)
        {
            _features.Push(samples);
            Append(_features.Read(out int frames), frames);
            int steps = _pendingFrames / StepFrames;
            return steps == 0 ? Array.Empty<float>() : Take(steps * StepFrames, final: false);
        }

        /// <summary>End of audio: the remaining rows (the tail padded as the whole clip's).</summary>
        public float[] Finish()
        {
            Append(_features.Finish(out int frames), frames);
            return Take(_pendingFrames, final: true);
        }

        private void Append(float[] features, int frames)
        {
            if (frames == 0) return;
            int mels = Lfm3AudioPreprocessor.MelBins;
            var joined = new float[(long)(_pendingFrames + frames) * mels];
            Array.Copy(_pending, joined, (long)_pendingFrames * mels);
            Array.Copy(features, 0, joined, (long)_pendingFrames * mels, (long)frames * mels);
            _pending = joined;
            _pendingFrames += frames;
        }

        private float[] Take(int frames, bool final)
        {
            int mels = Lfm3AudioPreprocessor.MelBins;
            float[] chunk = frames == _pendingFrames ? _pending : _pending.AsSpan(0, frames * mels).ToArray();
            float[] rows = _encoder.Encode(chunk, frames, final);
            _pending = _pending.AsSpan(frames * mels).ToArray();
            _pendingFrames -= frames;
            return rows;
        }

        public void Dispose() => _encoder.Dispose();
    }
}
