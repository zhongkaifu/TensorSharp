// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Graph;
using TensorSharp.Models.Graph;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// LFM3's depthformer (a.gen.code.*; llama.cpp mtmd lfm3a-gen.cpp build_gen_code, HF Lfm3OmniDepthDecoder): one MaskGIT
    /// pass over a frame's 12 codebook positions, as one graph program on the weights' executor.
    /// <list type="bullet">
    /// <item>Input: the LM's hidden row (after its final norm) projected to 12 x width (<c>proj_in</c>), plus each
    /// codebook's code embedding where its code is known, else the learned mask embedding.</item>
    /// <item>Trunk: bidirectional pre-norm blocks: per-head RMS-normed q/k, NeoX rotary over positions 0..11, grouped-query
    /// attention without a mask, LFM3's per-head sigmoid gate, a SwiGLU feed-forward.</item>
    /// <item>Head: the output norm, each codebook's head norm, then the codebook's own (tied) table: 2017 logits.</item>
    /// </list>
    /// Two costs llama.cpp pays every pass are not paid here (bit-identical): the projection depends only on the hidden row,
    /// so a frame's first pass computes it and keeps it on the device (a state) for the next two; and only the codebooks a
    /// pass samples get logits, through one product whose matrices are picked per row at run time (GraphOp.MulMatId), so the
    /// same program serves every codebook order. Where the table is held dense (the backend has no product kernel for its
    /// format) every codebook's logits are computed, as llama.cpp does, and the requested ones returned (noted once).
    /// The code embeddings and the mask embedding are gathered on the host from the table's bytes (a dozen rows).
    /// Passes of different frames must not interleave: hold <see cref="Sync"/> across a frame's passes.
    /// </summary>
    public sealed class Lfm3Depthformer : IDisposable
    {
        internal const string ProjIn = "a.gen.code.proj_in.weight", ProjInBias = "a.gen.code.proj_in.bias", MaskEmbedding = "a.gen.code.mask_embd",
            Table = "a.gen.code.embd.weight", HeadNorm = "a.gen.code.head_norm.weight", OutputNorm = "a.gen.code.output_norm.weight";

        private readonly Lfm3SpeechConfig _c;
        private readonly NetworkWeights _weights;
        private readonly NetworkGraph _graph;
        private readonly Tensor _projection;
        private readonly Lfm3SpeechHostRows _codeRows;
        private readonly float[] _maskEmbedding;
        private readonly float[] _inputRows;
        private readonly float[] _hidden;
        private readonly int[] _positions;
        private static bool s_denseHeadNoted;
        private bool _projectionKept;
        private bool _disposed;

        internal Lfm3Depthformer(Lfm3SpeechConfig config, NetworkWeights weights, Lfm3SpeechHostRows codeRows, IGraphExecutor executor = null)
        {
            _c = config;
            _weights = weights;
            _codeRows = codeRows;
            SelectedHead = weights.Get(Table).IsQuantized;
            _graph = new NetworkGraph(weights, executor, "lfm3 depthformer")
            {
                // As llama.cpp attends this network (flash attention off for the projector): F32 products around a softmax. On
                // ggml-metal the logits then equal llama.cpp's bit for bit (the flash kernel's differ by ~6e-4 of their range,
                // which flips ~1% of the sampled codes); the speed is the same at these sizes.
                AttentionHalfKv = false,
                AttentionByProducts = true,
            };
            _projection = new Tensor(weights.Allocator, DType.Float32, Lfm3SpeechConfig.Codebooks, config.DepthWidth);
            _projection.SetElementsAsFloat(new float[(long)Lfm3SpeechConfig.Codebooks * config.DepthWidth]);
            _maskEmbedding = weights.ReadF32(MaskEmbedding);
            _inputRows = new float[(long)Lfm3SpeechConfig.Codebooks * config.DepthWidth];
            _hidden = new float[config.HiddenSize];
            _positions = new int[Lfm3SpeechConfig.Codebooks];
            for (int i = 0; i < _positions.Length; i++) _positions[i] = i;
            if (!SelectedHead && !s_denseHeadNoted)
            {
                s_denseHeadNoted = true;
                Console.WriteLine($"  lfm3 depthformer: the codebook table is held as F32 on {weights.Backend} (no device product kernel for its " +
                    $"format): each pass computes all {Lfm3SpeechConfig.Codebooks} codebooks' logits (3x the head's work) instead of the " +
                    $"{Lfm3SpeechConfig.MaskGitSchedule[0]} it samples.");
            }
        }

        /// <summary>Hold across one frame's passes (the projection the first pass keeps is read by the next ones).</summary>
        public object Sync { get; } = new();

        /// <summary>The graph the passes run on (counters: Builds, Replays, Runs).</summary>
        public NetworkGraph Graph => _graph;

        /// <summary>True where a pass computes the requested codebooks' logits only (the table is held in its block format).</summary>
        public bool SelectedHead { get; }

        /// <summary>
        /// One MaskGIT pass. <paramref name="hidden"/>: the LM's hidden row for a frame's first pass (the projection is computed
        /// and kept), empty for its later passes (the kept projection is read). <paramref name="codes"/>: the frame's 12 codes,
        /// -1 where a codebook is still masked. Writes the logits of each codebook in <paramref name="selected"/> (2017 a
        /// codebook, in that order) to <paramref name="logits"/>.
        /// </summary>
        public void Pass(ReadOnlySpan<float> hidden, ReadOnlySpan<int> codes, ReadOnlySpan<int> selected, Span<float> logits)
        {
            lock (Sync) PassLocked(hidden, codes, selected, logits);
        }

        private void PassLocked(ReadOnlySpan<float> hidden, ReadOnlySpan<int> codes, ReadOnlySpan<int> selected, Span<float> logits)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            const int Cb = Lfm3SpeechConfig.Codebooks, V = Lfm3SpeechConfig.CodebookSize;
            int width = _c.DepthWidth;
            if (codes.Length != Cb) throw new ArgumentException($"{Cb} codes (-1 masked)", nameof(codes));
            if (selected.Length < 1 || selected.Length > Cb) throw new ArgumentException($"1 to {Cb} codebooks", nameof(selected));
            if (logits.Length < selected.Length * V) throw new ArgumentException($"{V} logits a selected codebook", nameof(logits));
            bool first = !hidden.IsEmpty;
            if (first && hidden.Length != _c.HiddenSize) throw new ArgumentException($"a hidden row of {_c.HiddenSize}", nameof(hidden));
            if (!first && !_projectionKept) throw new InvalidOperationException("a frame's later pass before its first one (which takes the hidden row)");
            for (int c = 0; c < Cb; c++)
            {
                Span<float> row = _inputRows.AsSpan(c * width, width);
                int code = codes[c];
                if (code < 0) _maskEmbedding.CopyTo(row);
                else if (code < V) _codeRows.Read((long)c * V + code, row);
                else throw new ArgumentOutOfRangeException(nameof(codes), $"code {code} of codebook {c} (codebook size {V})");
            }
            for (int i = 0; i < selected.Length; i++)
                if ((uint)selected[i] >= Cb) throw new ArgumentOutOfRangeException(nameof(selected), $"codebook {selected[i]}");
            if (first) hidden.CopyTo(_hidden);
            int[] ids = SelectedHead ? selected.ToArray() : null;
            float[] output = _graph.Run(b => b.Output(Emit(b, first, ids)))[0];
            _projectionKept = true;
            if (SelectedHead)
            {
                output.AsSpan(0, selected.Length * V).CopyTo(logits);
            }
            else
            {
                for (int i = 0; i < selected.Length; i++) output.AsSpan(selected[i] * V, V).CopyTo(logits.Slice(i * V, V));
            }
        }

        private NetValue Emit(NetworkBuilder b, bool first, int[] ids)
        {
            const int Cb = Lfm3SpeechConfig.Codebooks;
            int width = _c.DepthWidth, headDim = _c.DepthHeadDim;
            float eps = _c.DepthEps;
            NetValue projection;
            if (first)
            {
                projection = b.Reshape(b.Linear(b.Input(_hidden, _c.HiddenSize, 1), ProjIn, ProjInBias), width, Cb);
                b.Write(projection, b.State(_projection));
            }
            else
            {
                projection = b.State(_projection);
            }
            NetValue x = b.Add(projection, b.Input(_inputRows, width, Cb));
            NetValue positions = b.InputI32(_positions);
            for (int l = 0; l < _c.DepthLayers; l++)
            {
                string p = $"a.gen.code.blk.{l}.";
                NetValue h = b.RmsNorm(x, p + "ln1.weight", eps);
                NetValue attention = b.SelfAttention(h, _c.DepthHeads, _c.DepthKvHeads, headDim, p + "attn_q.weight", p + "attn_k.weight",
                    p + "attn_v.weight", null, p + "attn_q_norm.weight", p + "attn_k_norm.weight", eps,
                    v => b.RopeNeox(v, positions, headDim, _c.DepthRopeTheta), 1f / MathF.Sqrt(headDim), default, default, default, out _, out _);
                attention = b.HeadGate(attention, b.Linear(h, p + "attn_gate.weight"), headDim);
                x = b.Add(x, b.Linear(attention, p + "attn_out.weight"));
                x = b.Add(x, b.FeedForward(b.RmsNorm(x, p + "ln2.weight", eps), p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"));
            }
            NetValue y = b.RmsNorm(x, OutputNorm, eps);
            y = b.RmsNorm(y, eps, b.Weight(HeadNorm));                                   // each codebook its own norm weight
            if (ids != null)
            {
                NetValue rows = b.Reshape(b.GetRows(y, b.InputI32(ids)), width, 1, ids.Length);
                return b.MulMatId(b.Experts(Table, Cb), rows, b.InputI32(ids, 1, ids.Length));   // [2017, 1, n]
            }
            return b.MatMul(b.Weight(Table, width, Lfm3SpeechConfig.CodebookSize, Cb), b.Reshape(y, width, 1, Cb));   // [2017, 1, 12]
        }

        public void Dispose()
        {
            lock (Sync)
            {
                if (_disposed) return;
                _disposed = true;
                IGraphExecutor executor = _graph.Executor;
                _graph.Dispose();
                // The projection's device copies go with it (DeviceResidency: whichever executor bound its bytes).
                DeviceResidency.ReleaseTensor(_projection, executor);
            }
        }
    }
}
