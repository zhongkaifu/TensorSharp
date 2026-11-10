// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.GGML;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// The NextN / MTP draft head, once for every family on the runtime: a decoder layer after the
    /// stack (<c>blk.{NumLayers}</c>) that reads the next token's embedding and the trunk's final-normed
    /// hidden row, each RMS-normed (<c>nextn.enorm</c>, <c>nextn.hnorm</c>), concatenated and projected
    /// (<c>nextn.eh_proj</c>), runs the layer's attention (its own K/V rows in the request's slot) and
    /// FFN or experts through the same graph code as the trunk, and ends in <c>nextn.shared_head_norm</c>
    /// and its own head or the trunk's. The normed row it returns chains the next draft step.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>The first draft layer's weights are all present (the family loaded them).</summary>
        protected bool HasDraftLayer
        {
            get
            {
                if (DraftLayerCount <= 0) return false;
                int n = Config.NumLayers;
                foreach (string suffix in new[] { "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight", "attn_output.weight" })
                    if (!HasWeight(LayerName(n, suffix))) return false;
                return _cacheK != null && _cacheK.Length > n && _cacheK[n] != null;
            }
        }

        /// <summary>Run the draft layer over <paramref name="tokens"/> paired with the trunk rows
        /// <paramref name="hRows"/> at <paramref name="startPos"/>; with <paramref name="logitsOut"/> it
        /// returns the last row's logits and its normed hidden row (the next step's input).</summary>
        private void RunDraftLayer(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
        {
            int rows = tokens.Length, hiddenSize = Config.HiddenSize;
            if (hRows == null || hRows.LongLength < (long)rows * hiddenSize)
                throw new ArgumentException($"{rows} draft rows need {(long)rows * hiddenSize} hidden floats.", nameof(hRows));
            SettlePendingVerify();
            EnsureCacheCapacity(checked(startPos + rows));
            if (GraphUnsupportedReason() != null)
            {
                RunDraftLayerPerOp(tokens, hRows, startPos, logitsOut, hOut);
                return;
            }
            EnterStage(StageOfLayer(Config.NumLayers));
            Tensor embedded = DraftEmbedding(tokens);
            Tensor hidden = CreateFloatTensor(hRows.AsSpan(0, rows * hiddenSize).ToArray(), rows, hiddenSize);
            Slot active = SnapshotActive();
            bool wantLogits = logitsOut != null;
            CachedGraph graph = null;
            bool owned = false;
            StepOutputs outputs = default;
            _forwardSw.Start();
            try
            {
                graph = RunTraced(rows, new[] { active }, new[] { 0 }, new[] { rows }, new[] { startPos },
                    replayable: rows <= SmallChunkRows, snapshots: false, attentionFloor: active.DraftStart,
                    step => EmitDraftLayer(step, embedded, hidden, wantLogits), out outputs, out owned);
                if (graph == null)
                    throw new NotSupportedException(SpeculationRefusal ?? $"{Config.Architecture}: the draft layer cannot run as a graph.");
                _cacheDeviceDirty = true;
                if (wantLogits)
                {
                    Array.Copy(graph.Outputs[outputs.Logits], logitsOut, Math.Min(logitsOut.Length, Config.VocabSize));
                    if (hOut != null) Array.Copy(graph.Outputs[outputs.Final], hOut, Math.Min(hOut.Length, hiddenSize));
                }
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded.Dispose();
                hidden.Dispose();
                EnterStage(0);
                _forwardSw.Stop();
            }
        }

        /// <summary>The draft layer off the graphs (MLX, direct CUDA, the managed CPU): the same block
        /// through the per-op layer, its attention floored at the rows it replayed.</summary>
        private void RunDraftLayerPerOp(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
        {
            int rows = tokens.Length, hiddenSize = Config.HiddenSize, n = Config.NumLayers;
            _forwardSw.Start();
            Tensor x = null;
            try
            {
                EnterStage(_mesh.Stages - 1);
                using (Tensor embedded = DraftEmbedding(tokens))
                using (Tensor e = RMSNormOp(embedded, LayerName(n, "nextn.enorm.weight")))
                using (Tensor trunk = CreateFloatTensor(hRows.AsSpan(0, rows * hiddenSize).ToArray(), rows, hiddenSize))
                using (Tensor h = RMSNormOp(trunk, LayerName(n, "nextn.hnorm.weight")))
                using (Tensor joined = Ops.Concat(null, 1, e, h))
                    x = LinearForward(joined, LayerName(n, "nextn.eh_proj.weight"));
                _attentionFloor = _draftStart;
                try
                {
                    x = LayerOnOneDevice(x, n, TokenPositions.Contiguous(startPos, rows).WithRotary(_ropeDelta, null), batch: null);
                }
                finally
                {
                    _attentionFloor = 0;
                }
                if (logitsOut == null) return;
                string headNorm = HasWeight(LayerName(n, "nextn.shared_head_norm.weight"))
                    ? LayerName(n, "nextn.shared_head_norm.weight") : "output_norm.weight";
                using Tensor lastView = x.Narrow(0, rows - 1, 1);
                using Tensor last = Ops.NewContiguous(lastView);
                using Tensor normed = RMSNormOp(last, headNorm);
                if (hOut != null) Array.Copy(normed.GetElementsAsFloat(hiddenSize), hOut, Math.Min(hOut.Length, hiddenSize));
                string head = HasWeight(LayerName(n, "nextn.shared_head_head.weight")) ? LayerName(n, "nextn.shared_head_head.weight")
                    : HasWeight("output.weight") ? "output.weight" : "token_embd.weight";
                using Tensor logits = LinearForward(normed, head);
                Array.Copy(logits.GetElementsAsFloat(Config.VocabSize), logitsOut, Math.Min(logitsOut.Length, Config.VocabSize));
            }
            finally
            {
                x?.Dispose();
                EnterStage(0);
                _forwardSw.Stop();
            }
        }

        /// <summary>Trace the draft layer: projected input, one decoder layer, shared head (last row).</summary>
        private StepOutputs EmitDraftLayer(GraphStep step, Tensor embedded, Tensor hidden, bool wantLogits)
        {
            int rows = step.Rows, hiddenSize = Config.HiddenSize, n = Config.NumLayers;
            var outputs = new StepOutputs { Logits = -1, Final = -1 };
            int e = step.RmsNorm(step.Input(embedded, hiddenSize, rows), LayerName(n, "nextn.enorm.weight"));
            int h = step.RmsNorm(step.Input(hidden, hiddenSize, rows), LayerName(n, "nextn.hnorm.weight"));
            var positions = new int[rows];
            for (int i = 0; i < rows; i++) positions[i] = step.Starts[0] + i + step.Slots[0].RopeDelta;
            step.SetPositions(positions);
            int x = step.MatMul(LayerName(n, "nextn.eh_proj.weight"),
                step.Op(GraphOp.Concat, stackalloc[] { e, h }, stackalloc long[] { 0 }));
            x = step.Op(GraphOp.Add, x, EmitAttention(step, n, step.RmsNorm(x, LayerName(n, "attn_norm.weight"))));
            // A catch-up only needs the layer's K/V rows (written as effects): the graph holds what
            // feeds an output or an effect, so its attention output and FFN are never computed.
            if (!wantLogits) return outputs;
            int ffnIn = step.RmsNorm(x, FfnNormWeight(n));
            x = step.Op(GraphOp.Add, x, _moeLayer[n] ? EmitMoe(step, n, ffnIn) : EmitDenseFfn(step, n, ffnIn));
            string headNorm = HasWeight(LayerName(n, "nextn.shared_head_norm.weight"))
                ? LayerName(n, "nextn.shared_head_norm.weight") : "output_norm.weight";
            int last = rows == 1 ? x : step.View2(x, hiddenSize, 1, hiddenSize * 4L, (rows - 1) * hiddenSize * 4L);
            int normed = step.RmsNorm(last, headNorm);
            string head = HasWeight(LayerName(n, "nextn.shared_head_head.weight")) ? LayerName(n, "nextn.shared_head_head.weight")
                : HasWeight("output.weight") ? "output.weight" : "token_embd.weight";
            outputs.Final = step.Output(normed, hiddenSize);
            outputs.Logits = step.Output(step.MatMul(head, normed), Config.VocabSize);
            return outputs;
        }

        /// <summary>The draft layer's own token embedding when it ships one, else the trunk's.</summary>
        private Tensor DraftEmbedding(int[] tokens)
        {
            string own = LayerName(Config.NumLayers, "nextn.embed_tokens.weight");
            if (_quantWeights.TryGetValue(own, out QuantizedWeight q) && q.HasHostData)
            {
                var result = new Tensor(_allocator, DType.Float32, tokens.Length, Config.HiddenSize);
                PopulateQuantizedRows(result, q, tokens);
                return result;
            }
            return Embedding(tokens);
        }

        // ---- IDraftHead: the per-token head (DFlash blocks are in Speculative.cs)

        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            if (HasDFlash) throw new NotSupportedException("A DFlash drafter proposes whole blocks; use DraftBlock.");
            if (!HasDraftLayer) throw new InvalidOperationException($"{Config.Architecture} has no NextN/MTP draft layer.");
            RunDraftLayer(new[] { token }, hPrev, pos, logitsOut, hOut);
        }

        /// <summary>A draft step folds into the catch-up over the accepted rows (one graph).</summary>
        public bool SupportsFusedCatchUpStep => HasDraftLayer && !HasDFlash;

        public void DraftCatchUpAndStep(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
        {
            if (!SupportsFusedCatchUpStep) throw new InvalidOperationException($"{Config.Architecture} has no NextN/MTP draft layer to fold.");
            RunDraftLayer(tokens, hRows, startPos, logitsOut, hOut);
        }

        /// <summary>After plain steps (no trunk rows to replay) the draft layer resumes on a fresh suffix.</summary>
        public bool DraftHeadResumesAfterGap => HasDraftLayer && !HasDFlash;

        private void DraftLayerCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (hRows == null)
            {
                // Warmup, prefix reuse and plain steps carry no features: the draft layer ignores the
                // rows before this point (the trunk keeps its full context).
                _draftStart = checked(startPos + tokens.Length);
                return;
            }
            RunDraftLayer(tokens, hRows, startPos, null, null);
        }
    }
}
