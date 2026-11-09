// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's text encoder: Qwen3-VL-32B truncated to 50 language layers with the
// final norm removed. The DiT consumes the RAW layer-50 hidden state.
//
// Two things about this checkpoint drive the design here:
//
//   * the GGUF carries no metadata at all, so every hyperparameter is derived from
//     tensor shapes, and the tokenizer has to come from separate vocab files; and
//   * there is no chat template. The prompt is the reference-presentation text
//     followed by the raw user prompt, with nothing stripped from the front.
//     Applying a Qwen chat template here shifts every hidden state and breaks the
//     DiT's text/vision token tagging.
//
// For plain text-to-video the vision tower is never used, and interleaved M-RoPE
// collapses to ordinary RoPE because all three position axes are equal for text
// tokens — so the rope tables below are built with a single position per token.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Hyperparameters of the H3 text encoder, derived from tensor shapes.</summary>
    public sealed class MiniMaxH3TextEncoderConfig
    {
        public int NumLayers { get; init; }
        public int Hidden { get; init; }
        public int Heads { get; init; }
        public int KvHeads { get; init; }
        public int HeadDim { get; init; }
        public int Intermediate { get; init; }
        public int VocabSize { get; init; }
        /// <summary>True when the checkpoint keeps a final norm. H3's does not.</summary>
        public bool HasFinalNorm { get; init; }
        public float Eps { get; init; } = 1e-6f;
        public float RopeTheta { get; init; } = 5_000_000f;

        public override string ToString() =>
            $"Qwen3-VL text encoder: layers={NumLayers} hidden={Hidden} " +
            $"heads={Heads}/{KvHeads}x{HeadDim} ffn={Intermediate} vocab={VocabSize} " +
            $"finalNorm={HasFinalNorm} theta={RopeTheta:0}";
    }

    /// <summary>Qwen3-VL text encoder producing DiT conditioning.</summary>
    public sealed class MiniMaxH3TextEncoder : IDisposable
    {
        private readonly GgufFile _gguf;
        private MiniMaxH3VisionEncoder _vision;
        private readonly List<GCHandle> _pins = new();
        // Weights bound resident on the device are cached by host pointer; the cache
        // has to be told when those pointers stop being valid, or the backend still
        // holds their buffers at teardown.
        private readonly List<IntPtr> _bound = new();
        private readonly H3TeLayerW[] _layers;
        /// <summary>Bytes of the encoder file, used only to size layer groups.</summary>
        private readonly long _trunkBytes;
        private GCHandle _layersPin;
        private readonly GgufTensorInfo _embed;
        // Non-null on the backends with no ggml graph to call (BackendType.Cpu).
        private readonly MiniMaxH3DirectTextEncoder _direct;
        // Carried so the vision tower, built lazily below, gets the same routing.
        private readonly BackendType _backendType;
        private readonly IAllocator _allocatorForVision;
        // Mappings the native uploads may read from the file instead (MiniMaxH3DirectRead).
        private readonly IntPtr[] _directRead;
        private bool _disposed;

        public MiniMaxH3TextEncoderConfig Config { get; }
        public BpeTokenizer Tokenizer { get; }

        public MiniMaxH3TextEncoder(string ggufPath, string tokenizerDir = null)
            : this(ggufPath, tokenizerDir, BackendType.GgmlCuda, null) { }

        public MiniMaxH3TextEncoder(string ggufPath, string tokenizerDir,
                                    BackendType backend, IAllocator allocator)
        {
            _gguf = new GgufFile(ggufPath);
            try { _trunkBytes = new FileInfo(ggufPath).Length; } catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            // The text encoder is a SEPARATE file from the denoiser, so nothing on the
            // main load path has looked at it. A bad copy here fails as NaN deep in the
            // vision tower rather than as a bad file, so check it while the answer is
            // still cheap.
            _gguf.ThrowIfTruncated();
            Config = DetectConfig(_gguf);

            _embed = Get("model.embed_tokens.weight");
            _layers = new H3TeLayerW[Config.NumLayers];
            for (int i = 0; i < Config.NumLayers; i++)
            {
                string p = $"model.layers.{i}.";
                _layers[i] = new H3TeLayerW
                {
                    InputNorm = PinF32(p + "input_layernorm.weight"),
                    PostAttnNorm = PinF32(p + "post_attention_layernorm.weight"),
                    QNorm = PinF32(p + "self_attn.q_norm.weight"),
                    KNorm = PinF32(p + "self_attn.k_norm.weight"),
                    Q = Lin(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"),
                    K = Lin(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias"),
                    V = Lin(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"),
                    O = Lin(p + "self_attn.o_proj.weight", null),
                    Gate = Lin(p + "mlp.gate_proj.weight", null),
                    Up = Lin(p + "mlp.up_proj.weight", null),
                    Down = Lin(p + "mlp.down_proj.weight", null),
                };
            }
            _layersPin = GCHandle.Alloc(_layers, GCHandleType.Pinned);

            // Backends with no ggml graph to call run the same trunk through the
            // shared direct primitives instead.
            _backendType = backend;
            _allocatorForVision = allocator;
            if (allocator != null && backend is not (BackendType.GgmlCuda or BackendType.GgmlCpu
                    or BackendType.GgmlMetal or BackendType.GgmlVulkan))
                _direct = new MiniMaxH3DirectTextEncoder(_gguf, Config, allocator);

            Tokenizer = LoadTokenizer(tokenizerDir ?? Path.GetDirectoryName(Path.GetFullPath(ggufPath)));
            _directRead = MiniMaxH3DirectRead.Register(_gguf, backend);
        }

        /// <summary>Derive the configuration from tensor shapes; the GGUF has no metadata.</summary>
        public static MiniMaxH3TextEncoderConfig DetectConfig(GgufFile gguf)
        {
            int layers = MiniMaxH3Config.CountIndexed(gguf.Tensors, "model.layers.");
            if (layers == 0)
                throw new InvalidOperationException(
                    "not a MiniMax-H3 text encoder: no 'model.layers.N.' tensors.");
            var embed = gguf.Tensors["model.embed_tokens.weight"];
            int hidden = (int)embed.Shape[0];
            int vocab = (int)embed.Shape[1];
            int headDim = (int)gguf.Tensors["model.layers.0.self_attn.q_norm.weight"].Shape[0];
            int qOut = (int)gguf.Tensors["model.layers.0.self_attn.q_proj.weight"].Shape[1];
            int kOut = (int)gguf.Tensors["model.layers.0.self_attn.k_proj.weight"].Shape[1];
            return new MiniMaxH3TextEncoderConfig
            {
                NumLayers = layers,
                Hidden = hidden,
                VocabSize = vocab,
                HeadDim = headDim,
                Heads = qOut / headDim,
                KvHeads = kOut / headDim,
                Intermediate = (int)gguf.Tensors["model.layers.0.mlp.gate_proj.weight"].Shape[1],
                HasFinalNorm = gguf.Tensors.ContainsKey("model.norm.weight"),
            };
        }

        // ---- weights --------------------------------------------------------

        private GgufTensorInfo Get(string name) =>
            _gguf.Tensors.TryGetValue(name, out var t)
                ? t
                : throw new InvalidOperationException($"missing tensor '{name}'");

        private H3Lin Lin(string weight, string bias)
        {
            var info = Get(weight);
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            long bytes = ByteSizeOf(info);
            _bound.Add(ptr);
            return new H3Lin
            {
                W = ptr,
                B = bias != null && _gguf.Tensors.ContainsKey(bias) ? PinF32(bias) : IntPtr.Zero,
                Ne0 = (long)info.Shape[0],
                Ne1 = (long)info.Shape[1],
                Bytes = bytes,
                Type = (int)info.Type,
            };
        }

        // GGUF tensor data is contiguous, so the byte size is the distance to the
        // next tensor; deriving it from the quant block layout instead would mean
        // duplicating ggml's type table here.
        private long ByteSizeOf(GgufTensorInfo info)
        {
            long n = info.NumElements;
            return info.Type switch
            {
                GgmlTensorType.F32 => n * 4,
                GgmlTensorType.F16 or GgmlTensorType.BF16 => n * 2,
                GgmlTensorType.Q8_0 => n / 32 * 34,
                GgmlTensorType.Q4_K => n / 256 * 144,
                GgmlTensorType.Q5_K => n / 256 * 176,
                GgmlTensorType.Q6_K => n / 256 * 210,
                GgmlTensorType.Q3_K => n / 256 * 110,
                GgmlTensorType.Q2_K => n / 256 * 84,
                _ => throw new NotSupportedException(
                    $"unsupported text-encoder weight type {info.Type}"),
            };
        }

        private IntPtr PinF32(string name)
        {
            if (!_gguf.Tensors.TryGetValue(name, out var info)) return IntPtr.Zero;
            var h = GCHandle.Alloc(ReadVectorF32(info, name), GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        /// <summary>Read a small unquantized tensor as F32, converting by dtype.
        /// GgufFile.ReadTensorDataToFloat32 is a raw memcpy that assumes the tensor
        /// is already F32; every norm weight in this checkpoint is BF16, so going
        /// through it would silently produce garbage.</summary>
        private unsafe float[] ReadVectorF32(GgufTensorInfo info, string name)
        {
            long n = info.NumElements;
            var data = new float[n];
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{name}'.");
            switch (info.Type)
            {
                case GgmlTensorType.F32:
                {
                    float* src = (float*)ptr;
                    for (long i = 0; i < n; i++) data[i] = src[i];
                    break;
                }
                case GgmlTensorType.F16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++)
                        data[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                    break;
                }
                case GgmlTensorType.BF16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++)
                        data[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
                    break;
                }
                default:
                    throw new NotSupportedException(
                        $"'{name}' has type {info.Type}; expected an unquantized vector.");
            }
            return data;
        }

        // ---- tokenizer ------------------------------------------------------

        /// <summary>Load the Qwen2 byte-level BPE vocabulary. The GGUF has no
        /// tokenizer, so vocab.json + merges.txt must sit next to it (or be pointed
        /// at by TS_VIDEO_TOKENIZER), with tokenizer_config.json beside them for a
        /// photo or a reference: it alone defines the vision markers.</summary>
        public static BpeTokenizer LoadTokenizer(string dir)
        {
            string overrideDir = Environment.GetEnvironmentVariable("TS_VIDEO_TOKENIZER");
            if (!string.IsNullOrWhiteSpace(overrideDir))
                dir = Directory.Exists(overrideDir) ? overrideDir : Path.GetDirectoryName(overrideDir);

            string vocabPath = Path.Combine(dir ?? ".", "vocab.json");
            string mergesPath = Path.Combine(dir ?? ".", "merges.txt");
            if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
                throw new FileNotFoundException(
                    "MiniMax-H3's text-encoder GGUF carries no tokenizer. Place vocab.json, " +
                    "merges.txt and tokenizer_config.json (from MiniMaxAI/MiniMax-H3/processor) next " +
                    $"to the GGUF, or set TS_VIDEO_TOKENIZER. Looked in '{dir}'.");

            using var doc = JsonDocument.Parse(File.ReadAllText(vocabPath));
            var byId = new SortedDictionary<int, string>();
            foreach (var p in doc.RootElement.EnumerateObject())
                byId[p.Value.GetInt32()] = p.Name;

            // Special tokens live in tokenizer_config.json's added_tokens_decoder,
            // above the base vocabulary's id range.
            string cfgPath = Path.Combine(dir, "tokenizer_config.json");
            var special = new HashSet<int>();
            if (File.Exists(cfgPath))
            {
                using var cfg = JsonDocument.Parse(File.ReadAllText(cfgPath));
                if (cfg.RootElement.TryGetProperty("added_tokens_decoder", out var added))
                    foreach (var p in added.EnumerateObject())
                        if (int.TryParse(p.Name, out int id) &&
                            p.Value.TryGetProperty("content", out var c))
                        {
                            byId[id] = c.GetString();
                            special.Add(id);
                        }
            }

            int size = byId.Count == 0 ? 0 : byId.Keys.GetEnumerator().MoveNext() ? 0 : 0;
            int maxId = -1;
            foreach (int id in byId.Keys) if (id > maxId) maxId = id;
            size = maxId + 1;

            var vocab = new string[size];
            var types = new int[size];
            for (int i = 0; i < size; i++) vocab[i] = string.Empty;
            foreach (var kv in byId)
            {
                vocab[kv.Key] = kv.Value;
                types[kv.Key] = special.Contains(kv.Key) ? 3 : 1;   // 3 = control
            }

            var merges = new List<string>();
            foreach (string line in File.ReadLines(mergesPath))
            {
                if (line.Length == 0 || line[0] == '#') continue;
                merges.Add(line);
            }

            // H3 adds no BOS and no EOS: `add_bos_token = add_eos_token = false`.
            int eos = Array.IndexOf(vocab, "<|endoftext|>");
            return new BpeTokenizer(vocab, types, merges.ToArray(),
                bosTokenId: eos, eosTokenIds: eos >= 0 ? new[] { eos } : Array.Empty<int>(),
                addBos: false, addEos: false);
        }

        // ---- rope -----------------------------------------------------------

        /// <summary>Qwen3-VL's interleaved M-RoPE section widths, over the temporal,
        /// height, width and "extra" axes. The pairs each axis owns are interleaved
        /// rather than contiguous, which is what makes it <i>interleaved</i> M-RoPE.</summary>
        internal static readonly int[] RopeSections = { 24, 20, 20, 0 };

        /// <summary>Rotate-half cos/sin tables. Dim j and j+headDim/2 share an angle,
        /// which is what makes the (j, j+64) pairing a rotation. Text tokens have all
        /// three M-RoPE axes equal, so a single position per token suffices.</summary>
        internal static (float[] Cos, float[] Sin) BuildRope(int seq, int headDim, float theta)
        {
            int half = headDim / 2;
            var cos = new float[(long)headDim * seq];
            var sin = new float[(long)headDim * seq];
            for (int pos = 0; pos < seq; pos++)
            {
                long b = (long)pos * headDim;
                for (int i = 0; i < half; i++)
                {
                    double inv = Math.Pow(theta, -(2.0 * i) / headDim);
                    double a = pos * inv;
                    float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                    cos[b + i] = c; cos[b + i + half] = c;
                    sin[b + i] = s; sin[b + i + half] = s;
                }
            }
            return (cos, sin);
        }

        /// <summary>Interleaved M-RoPE tables, for prompts that carry image tokens.
        ///
        /// <para>Each rotation pair belongs to one axis by its index modulo three, so
        /// the temporal, row and column positions are woven through the head dimension
        /// instead of occupying contiguous blocks. Pairs past a section's width fall
        /// through to the fourth axis, whose position is always zero — those pairs are
        /// simply not rotated.</para>
        ///
        /// <para>With all three positions equal (a prompt of pure text) this reduces
        /// exactly to <see cref="BuildRope"/>.</para></summary>
        internal static (float[] Cos, float[] Sin) BuildMRope(
            IReadOnlyList<int> t, IReadOnlyList<int> h, IReadOnlyList<int> w, int headDim, float theta)
        {
            int seq = t.Count;
            int half = headDim / 2;
            var cos = new float[(long)headDim * seq];
            var sin = new float[(long)headDim * seq];
            for (int pos = 0; pos < seq; pos++)
            {
                long b = (long)pos * headDim;
                for (int i = 0; i < half; i++)
                {
                    int axisPos;
                    if (i % 3 == 0 && i < 3 * RopeSections[0]) axisPos = t[pos];
                    else if (i % 3 == 1 && i < 3 * RopeSections[1]) axisPos = h[pos];
                    else if (i % 3 == 2 && i < 3 * RopeSections[2]) axisPos = w[pos];
                    else axisPos = 0;

                    double inv = Math.Pow(theta, -(2.0 * i) / headDim);
                    double a = axisPos * inv;
                    float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                    cos[b + i] = c; cos[b + i + half] = c;
                    sin[b + i] = s; sin[b + i + half] = s;
                }
            }
            return (cos, sin);
        }

        // ---- encode ---------------------------------------------------------

        /// <summary>Gather the input embeddings for a token sequence.</summary>
        public float[] Embed(IReadOnlyList<int> tokens)
        {
            int hidden = Config.Hidden;
            var outp = new float[(long)tokens.Count * hidden];
            if (!_gguf.TryGetTensorDataPointer(_embed, out IntPtr basePtr))
                throw new InvalidOperationException("could not map the embedding table.");
            unsafe
            {
                for (int i = 0; i < tokens.Count; i++)
                {
                    int id = tokens[i];
                    if (id < 0 || id >= Config.VocabSize)
                        throw new ArgumentOutOfRangeException(nameof(tokens), $"token id {id} out of range");
                    long row = (long)id * hidden;
                    long dst = (long)i * hidden;
                    switch (_embed.Type)
                    {
                        case GgmlTensorType.F32:
                        {
                            float* src = (float*)basePtr + row;
                            for (int c = 0; c < hidden; c++) outp[dst + c] = src[c];
                            break;
                        }
                        case GgmlTensorType.F16:
                        {
                            ushort* src = (ushort*)basePtr + row;
                            for (int c = 0; c < hidden; c++)
                                outp[dst + c] = (float)BitConverter.UInt16BitsToHalf(src[c]);
                            break;
                        }
                        case GgmlTensorType.BF16:
                        {
                            ushort* src = (ushort*)basePtr + row;
                            for (int c = 0; c < hidden; c++)
                                outp[dst + c] = BitConverter.UInt32BitsToSingle((uint)src[c] << 16);
                            break;
                        }
                        default:
                            throw new NotSupportedException(
                                $"embedding table type {_embed.Type} is not supported.");
                    }
                }
            }
            return outp;
        }

        /// <summary>Build the interleaved M-RoPE positions for a prompt whose image
        /// spans are listed in <paramref name="images"/>.
        ///
        /// <para>An image occupies its whole span at a single temporal position while
        /// its row and column axes walk the merged patch grid. The span therefore
        /// advances the shared cursor by max(rows, columns) rather than by its token
        /// count, and every later token is renumbered from there — a wide image costs
        /// the text after it more positions than a tall one of equal token count.</para></summary>
        internal static (int[] T, int[] H, int[] W) BuildImagePositions(
            int seq, IReadOnlyList<MiniMaxH3PromptImage> images)
        {
            var t = new int[seq];
            var h = new int[seq];
            var w = new int[seq];
            for (int i = 0; i < seq; i++) t[i] = h[i] = w[i] = i;
            if (images == null) return (t, h, w);

            int offset = 0;
            foreach (var image in images)
            {
                int index = image.TokenIndex;
                int size = image.Vision.TokenCount;
                int gh = image.Vision.GridHeight / MiniMaxH3VisionEncoder.SpatialMerge;
                int gw = image.Vision.GridWidth / MiniMaxH3VisionEncoder.SpatialMerge;
                if (index < 0 || index + size > seq)
                    throw new ArgumentOutOfRangeException(nameof(images),
                        $"image span [{index}, {index + size}) does not fit in {seq} tokens.");
                if (size != gh * gw)
                    throw new ArgumentException(
                        $"image reports {size} tokens but a {gh}x{gw} merged grid.", nameof(images));

                int lenMax = Math.Max(gh, gw);
                int nextPos = index + lenMax + offset;
                for (int token = index + size; token < seq; token++)
                {
                    int pos = nextPos + token - (index + size);
                    t[token] = h[token] = w[token] = pos;
                }
                for (int token = 0; token < size; token++)
                {
                    t[index + token] = index + offset;
                    h[index + token] = index + offset + token / gw;
                    w[index + token] = index + offset + token % gw;
                }
                offset += lenMax - size;
            }
            return (t, h, w);
        }

        /// <summary>Run the trunk. Returns the raw hidden states of the last layer,
        /// [seq][hidden], with no final normalization applied — that is exactly what
        /// the DiT's condition_proj expects.</summary>
        /// <summary>Write the trunk output to TS_H3_DUMP_TE when set, so the managed
        /// and GGML paths can be compared directly instead of through a chaotic
        /// multi-step denoise. Off unless the variable is set.</summary>
        private static float[] Dump(float[] hidden)
        {
            string path = Environment.GetEnvironmentVariable("TS_H3_DUMP_TE");
            if (string.IsNullOrEmpty(path)) return hidden;
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            foreach (float v in hidden) bw.Write(v);
            return hidden;
        }

        public float[] Encode(IReadOnlyList<int> tokens, int layerLimit = 0) =>
            Encode(tokens, null, layerLimit);

        /// <summary>Run the trunk over a prompt that may carry reference images.
        ///
        /// <para>Each image contributes three things: its merged embeddings replace the
        /// placeholder tokens of its span, its three DeepStack taps are added to the
        /// hidden state after the first three layers, and its span takes 3-D M-RoPE
        /// positions instead of the running token index.</para></summary>
        public float[] Encode(IReadOnlyList<int> tokens,
                              IReadOnlyList<MiniMaxH3PromptImage> images,
                              int layerLimit = 0)
        {
            if (tokens is null || tokens.Count == 0)
                throw new ArgumentException("at least one token is required", nameof(tokens));
            int seq = tokens.Count, hidden = Config.Hidden;
            int layers = layerLimit > 0 ? Math.Min(layerLimit, Config.NumLayers) : Config.NumLayers;
            bool hasImages = images != null && images.Count > 0;

            float[] embeddings = Embed(tokens);
            float[] cos, sin, deepstack = null;
            int numDeepstack = 0;
            if (hasImages)
            {
                foreach (var image in images)
                {
                    // The placeholder tokens carry no information; the tower's output
                    // takes their place outright.
                    Array.Copy(image.Vision.Merged, 0, embeddings,
                               (long)image.TokenIndex * hidden, image.Vision.Merged.LongLength);
                }
                numDeepstack = images[0].Vision.DeepStack.Length;
                // Dense and mostly zero: the alternative is a sparse scatter in the
                // graph, and the sequences here are short enough that it is not worth it.
                deepstack = new float[(long)numDeepstack * seq * hidden];
                foreach (var image in images)
                {
                    if (image.Vision.DeepStack.Length != numDeepstack)
                        throw new ArgumentException(
                            "every reference image must come from the same vision tower.", nameof(images));
                    for (int l = 0; l < numDeepstack; l++)
                        Array.Copy(image.Vision.DeepStack[l], 0, deepstack,
                                   ((long)l * seq + image.TokenIndex) * hidden,
                                   image.Vision.DeepStack[l].LongLength);
                }
                var (pt, ph, pw) = BuildImagePositions(seq, images);
                (cos, sin) = BuildMRope(pt, ph, pw, Config.HeadDim, Config.RopeTheta);
            }
            else
            {
                (cos, sin) = BuildRope(seq, Config.HeadDim, Config.RopeTheta);
            }
            if (_direct != null)
            {
                // Pure-managed path: the identical trunk, with no native call.
                return Dump(_direct.Encode(embeddings, seq, cos, sin, deepstack, numDeepstack, layers));
            }

            var outp = new float[(long)seq * hidden];

            var ePin = GCHandle.Alloc(embeddings, GCHandleType.Pinned);
            var cPin = GCHandle.Alloc(cos, GCHandleType.Pinned);
            var sPin = GCHandle.Alloc(sin, GCHandleType.Pinned);
            var oPin = GCHandle.Alloc(outp, GCHandleType.Pinned);
            var dPin = deepstack == null ? default : GCHandle.Alloc(deepstack, GCHandleType.Pinned);
            try
            {
                int groupSize = ResolveLayerGroup(layers, numDeepstack);
                long layerStride = Marshal.SizeOf<H3TeLayerW>();
                IntPtr layerBase = _layersPin.AddrOfPinnedObject();
                // Ping-pong the hidden state between the caller's output buffer and a scratch
                // one, so the last group always lands in outp.
                float[] scratch = groupSize >= layers ? null : new float[(long)seq * hidden];
                var scratchPin = scratch == null
                    ? default : GCHandle.Alloc(scratch, GCHandleType.Pinned);
                try
                {
                    int groups = (layers + groupSize - 1) / groupSize;
                    IntPtr src = ePin.AddrOfPinnedObject();
                    for (int g = 0; g < groups; g++)
                    {
                        int start = g * groupSize;
                        int count = Math.Min(groupSize, layers - start);
                        bool last = g == groups - 1;
                        IntPtr dst = last ? oPin.AddrOfPinnedObject()
                                          : ((g % 2 == 0 && scratch != null)
                                             ? scratchPin.AddrOfPinnedObject()
                                             : oPin.AddrOfPinnedObject());
                        var args = new H3TextEncodeArgs
                        {
                            StructBytes = Marshal.SizeOf<H3TextEncodeArgs>(),
                            NumLayers = count,
                            Embeddings = src,
                            Out = dst,
                            Cos = cPin.AddrOfPinnedObject(),
                            Sin = sPin.AddrOfPinnedObject(),
                            // H3 removed the final norm; the DiT wants the unnormalized state.
                            FinalNorm = IntPtr.Zero,
                            // DeepStack taps land on the first NumDeepstack layers only, which
                            // ResolveLayerGroup keeps inside group 0.
                            Deepstack = (g == 0 && deepstack != null)
                                ? dPin.AddrOfPinnedObject() : IntPtr.Zero,
                            NumDeepstack = g == 0 ? numDeepstack : 0,
                            Layers = layerBase + (nint)(start * layerStride),
                            Hidden = hidden,
                            Heads = Config.Heads,
                            KvHeads = Config.KvHeads,
                            HeadDim = Config.HeadDim,
                            Seq = seq,
                            Causal = 1,
                            Eps = Config.Eps,
                        };
                        if (!GgmlBasicOps.TryMiniMaxH3TextEncode(in args))
                            throw new InvalidOperationException(
                                "MiniMax-H3 text encode failed.");
                        if (!last)
                        {
                            ReleaseLayerResidency(start, count);
                            src = dst;
                        }
                    }
                }
                finally
                {
                    if (scratchPin.IsAllocated) scratchPin.Free();
                }
            }
            finally
            {
                ePin.Free(); cPin.Free(); sPin.Free(); oPin.Free();
                if (dPin.IsAllocated) dPin.Free();
            }
            return Dump(outp);
        }

        /// <summary>True when this checkpoint carries the Qwen3-VL vision tower, which
        /// Ref2VA needs and the text-only FL2VA path never touches.</summary>
        public bool HasVision => _gguf.Tensors.ContainsKey("visual.patch_embed.proj.weight");

        /// <summary>The vision tower, built on first use and sharing this file's mapping.
        /// Throws when the checkpoint has none — check <see cref="HasVision"/> first.</summary>
        public MiniMaxH3VisionEncoder Vision =>
            _vision ??= new MiniMaxH3VisionEncoder(_gguf, _backendType, _allocatorForVision);

        /// <summary>Tokenize a prompt. There is NO chat template: the raw text is
        /// encoded as-is with no BOS/EOS.</summary>
        public List<int> Tokenize(string prompt) =>
            Tokenizer.Encode(prompt ?? string.Empty, addSpecial: false);

        /// <summary>How many trunk layers to make device-resident at once.
        ///
        /// <para>Qwen3-VL-32B is ~17 GB at Q4_K_M, which does not fit a 16 GB card. The driver
        /// does not refuse the allocation - on Windows/WDDM it backs the overflow with shared
        /// host memory, so the whole trunk then runs at PCIe speed. Splitting the trunk into
        /// groups and handing each group's device copy back after it runs keeps peak residency
        /// at one group instead of fifty layers, at the cost of re-uploading on the next call.
        ///
        /// <para>Returns the full layer count - a single call, byte-for-byte today's behaviour -
        /// whenever the trunk already fits, so cards with room pay nothing. The first group is
        /// never smaller than the DeepStack tap count, because those taps are applied by
        /// layer index within a call.</para></summary>
        private int ResolveLayerGroup(int layers, int numDeepstack)
        {
            if (layers <= 1)
                return layers;
            // Escape hatch, and how the grouped path is A/B tested against the
            // single-call one: TS_H3_TE_GROUP=<n> pins the group size, and any
            // n >= layers reproduces the original whole-trunk call exactly.
            string ov = Environment.GetEnvironmentVariable("TS_H3_TE_GROUP");
            if (!string.IsNullOrWhiteSpace(ov) && int.TryParse(ov, out int forced) && forced > 0)
                return Math.Min(layers, Math.Max(forced, Math.Max(1, numDeepstack)));
            // Grouping is OFF unless asked for, and that is a measured decision rather
            // than a default-safe one. On a 16 GB card the 17 GB trunk does overflow into
            // shared host memory, and grouping does remove the overflow - peak device use
            // fell from 16041 MiB to 12981 MiB at 640x384. It was still 3 s SLOWER over
            // three runs (68.6 s against 65.8 s best-of-3).
            //
            // The reason is that the trunk is a ONE-SHOT prefill: a prompt is typically a
            // few dozen tokens, so every weight is read exactly once and the overflowed
            // ~1.3 GB costs a single PCIe crossing. Grouping cannot make that cheaper - it
            // still moves all 17 GB - and it adds an allocate/invalidate cycle per group.
            // Spill only compounds when weights are re-read per step, which is the DENOISER
            // (see MiniMaxH3DiT.ReleaseDeviceResidency, where releasing before the VAE was
            // worth 22 s), not this trunk.
            //
            // Kept because it is the right tool when device memory, not time, is the
            // binding constraint - a smaller card, or another process needing the VRAM.
            return layers;
        }

        /// <summary>Drop the device-resident copies of one layer group's weights. The host
        /// pointers are mmapped GGUF pages and stay valid, so a later call re-uploads them.</summary>
        private void ReleaseLayerResidency(int start, int count)
        {
            for (int i = start; i < start + count && i < _layers.Length; i++)
            {
                var lw = _layers[i];
                Drop(lw.InputNorm); Drop(lw.PostAttnNorm); Drop(lw.QNorm); Drop(lw.KNorm);
                Drop(lw.Q.W); Drop(lw.Q.B); Drop(lw.K.W); Drop(lw.K.B);
                Drop(lw.V.W); Drop(lw.V.B); Drop(lw.O.W); Drop(lw.O.B);
                Drop(lw.Gate.W); Drop(lw.Gate.B); Drop(lw.Up.W); Drop(lw.Up.B);
                Drop(lw.Down.W); Drop(lw.Down.B);
            }
            static void Drop(IntPtr p)
            {
                if (p != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(p);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _direct?.Dispose();
            if (_layersPin.IsAllocated) _layersPin.Free();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
            _bound.Clear();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _vision?.Dispose();
            _vision = null;
            MiniMaxH3DirectRead.Unregister(_directRead);
            _gguf?.Dispose();
        }
    }
}
