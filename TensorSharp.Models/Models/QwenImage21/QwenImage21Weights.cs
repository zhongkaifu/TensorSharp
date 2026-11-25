// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The tensors of a Qwen-Image-2.1 diffusion transformer GGUF as graph leaves: every projection keyed by its bytes in the
// file's mapping (in the file's own format, so a GGML backend's device copy is the quantized weight), the per-head and
// text norms as F32 vectors. The dimensions come from the tensors, so a small synthetic transformer loads the same way.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>A weight leaf keyed by its host bytes: a projection [Ne0 = in, Ne1 = out] in the file's mapping (or a copy
    /// where the file is not mapped), or (<see cref="Vector"/>) an F32 vector [Ne0].</summary>
    internal sealed record QwenImage21Weight(int Format, long Ne0, long Ne1, IntPtr Data, long Bytes, bool Vector = false) : IGraphWeight
    {
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;

        /// <summary>The weight's first <paramref name="count"/> output rows from <paramref name="start"/> (rows are whole in every
        /// GGUF format): a view of the same bytes.</summary>
        internal QwenImage21Weight Rows(long start, long count)
        {
            if (start < 0 || count <= 0 || start + count > Ne1) throw new ArgumentOutOfRangeException(nameof(count));
            long rowBytes = Bytes / Ne1;
            return this with { Data = Data + checked((nint)(start * rowBytes)), Ne1 = count, Bytes = count * rowBytes };
        }
    }

    /// <summary>One transformer block. A checkpoint with a fused <c>img_mlp.gate_up</c> projection has <see cref="Up"/> null and
    /// <see cref="Gate"/> [dim, 2 ff] ([gate | up] rows).</summary>
    internal sealed class QwenImage21Block
    {
        public QwenImage21Weight Q, K, V, Out, Gate, Up, Down, NormQ, NormK;
        public bool FusedGateUp => Up == null;
    }

    /// <summary>
    /// A Qwen-Image-2.1 transformer file: 64-channel latent tokens, a 4096-wide single stream of 32 blocks, Qwen3-VL-8B's
    /// 4096-dimensional conditioning (the released checkpoint; a synthetic file of other widths loads the same way). Refuses a
    /// file that is not a Qwen-Image-2.1 transformer, naming the tensor it lacks.
    /// </summary>
    internal sealed class QwenImage21Weights : IDisposable
    {
        internal const int TimestepChannels = 256;

        /// <summary>The released checkpoint's attention heads (32 x 128), which a tensor-parallel GPU count must divide.</summary>
        internal const int ReleaseHeads = 32;

        /// <summary>The released checkpoint's shape, which LoRA files are keyed and validated against: 32 blocks, a 4096-wide
        /// stream of 128-wide heads, 64 latent channels and Qwen3-VL-8B's 4096-wide conditioning.</summary>
        internal const int ReleaseLayers = 32, ReleaseDim = 4096, ReleaseHeadDim = 128, ReleaseChannels = 64, ReleaseTextDim = 4096;
        internal const float Eps = 1e-6f;

        private readonly GgufFile _gguf;
        private readonly bool _ownsFile;
        private readonly List<IntPtr> _owned = new();
        private readonly List<QwenImage21Weight> _all = new();
        private readonly Dictionary<string, QwenImage21Weight> _vectors = new(StringComparer.Ordinal);
        private bool _disposed;

        public QwenImage21Weights(string path)
            : this(new GgufFile(path ?? throw new ArgumentNullException(nameof(path))), ownsFile: true, Path.GetFileName(path))
        {
        }

        /// <param name="name">The file's name, for refusals.</param>
        public QwenImage21Weights(GgufFile gguf, bool ownsFile, string name)
        {
            _gguf = gguf ?? throw new ArgumentNullException(nameof(gguf));
            _ownsFile = ownsFile;
            try
            {
                Prefix = gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
                if (!gguf.Tensors.ContainsKey(Prefix + "txt_in.text_norm.weight"))
                    throw new ModelLoadRefusedException(
                        $"'{name}' is not a Qwen-Image-2.1 diffusion transformer (no txt_in.text_norm.weight tensor). " +
                        "Earlier Qwen-Image / Qwen-Image-Edit checkpoints such as Qwen-Image-Edit-2511 are no longer supported; " +
                        "use a Qwen-Image-2.1 GGUF (see docs/models/qwenimage21.md).");
                var imageIn = Info("img_in.weight", 2);
                Channels = checked((int)imageIn.Shape[0]);
                Dim = checked((int)imageIn.Shape[1]);
                TextDim = checked((int)Info("txt_in.in_layer.weight", 2).Shape[0]);
                HeadDim = checked((int)Info("transformer_blocks.0.attn.norm_q.weight", 1).NumElements);
                if (HeadDim <= 0 || HeadDim % 2 != 0 || Dim % HeadDim != 0)
                    throw new NotSupportedException($"Qwen-Image-2.1 requires an even head width that divides the hidden width {Dim}, not {HeadDim}.");
                Heads = Dim / HeadDim;
                int layers = 0;
                while (gguf.Tensors.ContainsKey($"{Prefix}transformer_blocks.{layers}.attn.to_q.weight")) layers++;
                Layers = layers;
                Ff = checked((int)Info("transformer_blocks.0.img_mlp.out.weight", 2).Shape[0]);

                ImageIn = Weight("img_in.weight", Channels, Dim);
                TextIn = Weight("txt_in.in_layer.weight", TextDim, Dim);
                TextOut = Weight("txt_in.out_layer.weight", Dim, Dim);
                TimeIn = Weight("time_text_embed.timestep_embedder.linear_1.weight", TimestepChannels, Dim);
                TimeOut = Weight("time_text_embed.timestep_embedder.linear_2.weight", Dim, Dim);
                Modulation = Weight("modulation.1.weight", Dim, 4 * Dim);
                NormOut = Weight("norm_out.linear.weight", Dim, Dim);
                ProjOut = Weight("proj_out.weight", Dim, Channels);
                TextNorm = Vector("txt_in.text_norm.weight", TextDim);
                Blocks = new QwenImage21Block[Layers];
                for (int i = 0; i < Layers; i++)
                {
                    string p = $"transformer_blocks.{i}.";
                    bool fused = gguf.Tensors.ContainsKey(Prefix + p + "img_mlp.gate_up.weight");
                    Blocks[i] = new QwenImage21Block
                    {
                        Q = Weight(p + "attn.to_q.weight", Dim, Dim),
                        K = Weight(p + "attn.to_k.weight", Dim, Dim),
                        V = Weight(p + "attn.to_v.weight", Dim, Dim),
                        Out = Weight(p + "attn.to_out.0.weight", Dim, Dim),
                        Gate = Weight(p + (fused ? "img_mlp.gate_up.weight" : "img_mlp.gate_layer.weight"), Dim, fused ? 2 * Ff : Ff),
                        Up = fused ? null : Weight(p + "img_mlp.proj.weight", Dim, Ff),
                        Down = Weight(p + "img_mlp.out.weight", Ff, Dim),
                        NormQ = Vector(p + "attn.norm_q.weight", HeadDim),
                        NormK = Vector(p + "attn.norm_k.weight", HeadDim),
                    };
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public GgufFile Gguf => _gguf;

        /// <summary>The tensor-name prefix of the file ("" or ComfyUI's "model.diffusion_model.").</summary>
        public string Prefix { get; }

        public int Dim { get; }
        public int Heads { get; }
        public int HeadDim { get; }
        public int Channels { get; }
        public int TextDim { get; }
        public int Layers { get; }
        public int Ff { get; }

        public QwenImage21Weight ImageIn { get; }
        public QwenImage21Weight TextIn { get; }
        public QwenImage21Weight TextOut { get; }
        public QwenImage21Weight TimeIn { get; }
        public QwenImage21Weight TimeOut { get; }
        public QwenImage21Weight Modulation { get; }
        public QwenImage21Weight NormOut { get; }
        public QwenImage21Weight ProjOut { get; }
        public QwenImage21Weight TextNorm { get; }
        public QwenImage21Block[] Blocks { get; }

        /// <summary>Every leaf this file binds (projections and vectors), each once.</summary>
        public IReadOnlyList<QwenImage21Weight> All => _all;

        /// <summary>The projections' bytes: what the transformer holds on a device.</summary>
        public long Bytes
        {
            get
            {
                long total = 0;
                foreach (QwenImage21Weight w in _all) total += w.Bytes;
                return total;
            }
        }

        private GgufTensorInfo Info(string name, int rank)
        {
            if (!_gguf.Tensors.TryGetValue(Prefix + name, out GgufTensorInfo info) || (rank > 0 && info.Shape.Length != rank))
                throw new NotSupportedException($"Qwen-Image-2.1 requires {Prefix + name}.");
            return info;
        }

        private QwenImage21Weight Weight(string name, long input, long output)
        {
            string full = Prefix + name;
            if (!_gguf.Tensors.TryGetValue(full, out GgufTensorInfo info) || info.Shape.Length != 2 ||
                (long)info.Shape[0] != input || (long)info.Shape[1] != output)
                throw new NotSupportedException($"Qwen-Image-2.1 requires {full} with GGUF shape [{input},{output}].");
            long bytes = _gguf.GetTensorByteCount(info);
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
            {
                ptr = QuantizedWeight.AllocateBuffer(bytes);
                _owned.Add(ptr);
                _gguf.ReadTensorDataToNative(info, ptr, bytes);
            }
            var w = new QwenImage21Weight((int)info.Type, input, output, ptr, bytes);
            _all.Add(w);
            return w;
        }

        /// <summary>An F32 vector: the file's bytes when they are F32 and mapped, otherwise an exact F32 copy.</summary>
        private QwenImage21Weight Vector(string name, int count)
        {
            string full = Prefix + name;
            if (_vectors.TryGetValue(full, out QwenImage21Weight known)) return known;
            if (!_gguf.Tensors.TryGetValue(full, out GgufTensorInfo info) || info.NumElements != count)
                throw new NotSupportedException($"Qwen-Image-2.1 requires {full} with {count} elements.");
            IntPtr ptr;
            if (info.Type != GgmlTensorType.F32 || !_gguf.TryGetTensorDataPointer(info, out ptr))
            {
                ptr = QuantizedWeight.AllocateBuffer((long)count * sizeof(float));
                _owned.Add(ptr);
                long bytes = _gguf.GetTensorByteCount(info);
                IntPtr source = QuantizedWeight.AllocateBuffer(bytes);
                try
                {
                    _gguf.ReadTensorDataToNative(info, source, bytes);
                    // Managed dequantization: exact for the norms' formats (F32, F16, BF16), and no native call on the cpu backend.
                    ManagedQuantizedOps.DequantizeToFloat32Native((int)info.Type, source, ptr, count);
                }
                finally { QuantizedWeight.FreeBuffer(source); }
            }
            var w = new QwenImage21Weight(GraphFormat.F32, count, 1, ptr, (long)count * sizeof(float), Vector: true);
            _vectors[full] = w;
            _all.Add(w);
            return w;
        }

        /// <summary>Host memory this binding allocated (vectors converted to F32, projections read from an unmapped file).</summary>
        internal IReadOnlyList<IntPtr> Owned => _owned;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (IntPtr p in _owned) QuantizedWeight.FreeBuffer(p);
            _owned.Clear();
            if (_ownsFile) _gguf.Dispose();
        }
    }
}
