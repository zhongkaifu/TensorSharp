// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

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
        /// <summary>The draft layer reads (and chains) the trunk's whole residual stream, not its final-normed row
        /// (hyper-connections: the head's mixer is the draft head's own).</summary>
        protected virtual bool DraftChainsStream => false;

        /// <summary>Floats per row of the draft head's input: the stream's width or the hidden size.</summary>
        protected int DraftFeatureWidth => DraftChainsStream ? StreamWidth : Config.HiddenSize;

        /// <summary>Token ids [0, n) the draft head scores: its proposals only (the verify scores the whole vocabulary,
        /// so the answers are the same; a token past them is never drafted).</summary>
        protected virtual int DraftVocabulary => Config.VocabSize;

        /// <summary>Tests score another draft vocabulary than the family's (<see cref="DraftVocabulary"/>).</summary>
        internal int? DraftVocabularyForTests { get; set; }

        private int ScoredDraftVocabulary => Math.Min(DraftVocabularyForTests ?? DraftVocabulary, Config.VocabSize);

        /// <summary>The draft layer's weights that must be present (suffixes of blk.{NumLayers}).</summary>
        protected virtual string[] DraftLayerWeights => new[] { "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight", "attn_output.weight" };

        /// <summary>The first draft layer's weights are all present (the family loaded them).</summary>
        protected bool HasDraftLayer
        {
            get
            {
                if (DraftLayerCount <= 0) return false;
                int n = Config.NumLayers;
                foreach (string suffix in DraftLayerWeights)
                    if (!HasWeight(LayerName(n, suffix))) return false;
                return _cacheK != null && _cacheK.Length > n && _cacheK[n] != null;
            }
        }

        /// <summary>Run the draft layer over <paramref name="tokens"/> paired with the trunk rows
        /// <paramref name="hRows"/> at <paramref name="startPos"/>; with <paramref name="logitsOut"/> it
        /// returns the last row's logits and its normed hidden row (the next step's input).</summary>
        /// <summary>The replay shapes of the draft layer's steps start with this.</summary>
        private const string DraftShapePrefix = "draft-";

        private void RunDraftLayer(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut, float[] topOut = null)
        {
            int rows = tokens.Length, hiddenSize = DraftFeatureWidth;
            if (hRows == null || hRows.LongLength < (long)rows * hiddenSize)
                throw new ArgumentException($"{rows} draft rows need {(long)rows * hiddenSize} hidden floats.", nameof(hRows));
            SettlePendingVerify();
            EnsureCacheCapacity(checked(startPos + rows));
            // The layer's rows from here on are the speculator's now, and the carry is not its last row.
            _draftFollowed = -1;
            // Per-rank graphs do not trace the (unsharded) draft layer: it keeps the per-op path, as under --tp before.
            if (GraphUnsupportedReason() != null || _rankGraphs)
            {
                RunDraftLayerPerOp(tokens, hRows, startPos, logitsOut, hOut);
                return;
            }
            EnterStage(StageOfLayer(Config.NumLayers));
            Tensor embedded = DraftEmbedding(tokens);
            Tensor hidden = CreateFloatTensor(hRows.AsSpan(0, rows * hiddenSize).ToArray(), rows, hiddenSize);
            Slot active = SnapshotActive();
            bool top = topOut != null, wantLogits = logitsOut != null || top;
            CachedGraph graph = null;
            bool owned = false;
            StepOutputs outputs = default;
            _forwardSw.Start();
            try
            {
                // A draft step runs as the kept one of its width advanced to its position (and replays its recording),
                // as decode steps and verify windows do: its rows are the token embeddings and the trunk's rows.
                graph = RunTraced(rows, new[] { active }, new[] { 0 }, new[] { rows }, new[] { startPos },
                    replayable: rows <= SmallChunkRows, snapshots: false, attentionFloor: active.DraftStart,
                    step => EmitDraftLayer(step, embedded, hidden, wantLogits, top), out outputs, out owned,
                    shape: DraftShapePrefix + (top ? "top" : wantLogits ? "logits" : "rows"), advanceInputs: new[] { embedded, hidden });
                if (graph == null)
                    throw new NotSupportedException(SpeculationRefusal ?? $"{Config.Architecture}: the draft layer cannot run as a graph.");
                _cacheDeviceDirty = true;
                if (top)
                {
                    Array.Copy(graph.Outputs[outputs.Logits], topOut, DraftTopK);
                    Array.Copy(graph.Outputs[outputs.Hidden], 0, topOut, DraftTopK, DraftTopK);
                }
                else if (wantLogits)
                {
                    int scored = Math.Min(logitsOut.Length, ScoredDraftVocabulary);
                    Array.Copy(graph.Outputs[outputs.Logits], logitsOut, scored);
                    if (scored < logitsOut.Length) Array.Fill(logitsOut, float.NegativeInfinity, scored, logitsOut.Length - scored);
                }
                if (wantLogits && hOut != null) Array.Copy(graph.Outputs[outputs.Final], hOut, Math.Min(hOut.Length, hiddenSize));
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

        /// <summary>Trace the draft layer: projected input, one decoder layer (the family's block hooks), its head
        /// (last row). The chained row is the head's normed row, or the stream (<see cref="DraftChainsStream"/>).</summary>
        private StepOutputs EmitDraftLayer(GraphStep step, Tensor embedded, Tensor hidden, bool wantLogits, bool top = false)
        {
            int rows = step.Rows, hiddenSize = Config.HiddenSize, n = Config.NumLayers, width = DraftFeatureWidth;
            var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            var positions = new int[rows];
            for (int i = 0; i < rows; i++) positions[i] = step.Starts[0] + i + step.Slots[0].RopeDelta;
            step.SetPositions(positions);
            int x = EmitDraftInput(step, n, step.HiddenInput(embedded, hiddenSize, rows), step.HiddenInput(hidden, width, rows));
            int normed = EmitBlockInput(step, n, x, BlockSite.Mixer, out int mixContext);
            x = EmitBlockOutput(step, n, x, BlockSite.Mixer, mixContext, EmitAttention(step, n, normed));
            // A catch-up only needs the layer's K/V rows (written as effects): the graph holds what
            // feeds an output or an effect, so its attention output and FFN are never computed.
            if (!wantLogits) return outputs;
            int ffnIn = EmitBlockInput(step, n, x, BlockSite.Ffn, out int ffnContext);
            x = EmitBlockOutput(step, n, x, BlockSite.Ffn, ffnContext, _moeLayer[n] ? EmitMoe(step, n, ffnIn) : EmitDenseFfn(step, n, ffnIn));
            int streamWidth = DraftChainsStream ? StreamWidth : hiddenSize;
            int last = rows == 1 ? x : step.View2(x, streamWidth, 1, streamWidth * 4L, (rows - 1) * streamWidth * 4L);
            int head = EmitDraftHead(step, n, last);
            string headWeight = HasWeight(LayerName(n, "nextn.shared_head_head.weight")) ? LayerName(n, "nextn.shared_head_head.weight")
                : HasWeight("output.weight") ? "output.weight" : "token_embd.weight";
            outputs.Final = step.Output(DraftChainsStream ? last : head, width);
            int vocab = ScoredDraftVocabulary;
            int logits = vocab < Config.VocabSize ? step.MatMulRows(headWeight, 0, vocab, head, alone: true) : step.MatMul(headWeight, head);
            if (!top)
            {
                outputs.Logits = step.Output(logits, vocab);
                return outputs;
            }
            // A chain step's ten largest logits, as ids (a tie to the lower id) and values: what its confidence is made of.
            int ids = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { DraftTopK }));
            outputs.Logits = step.Output(ids, DraftTopK);
            outputs.Hidden = step.Output(step.Op(GraphOp.GetRows, step.Reshape(logits, 1, vocab), ids), DraftTopK);
            return outputs;
        }

        /// <summary>The largest logits a draft chain's step reads back (the speculator's confidence is a softmax over ten).</summary>
        private const int DraftTopK = 10;

        /// <summary>The draft layer's input: the next tokens' embedding and the trunk's rows, each RMS-normed
        /// (<c>nextn.enorm</c>, <c>nextn.hnorm</c>), concatenated and projected (<c>nextn.eh_proj</c>).</summary>
        protected virtual int EmitDraftInput(GraphStep step, int layer, int embedded, int hidden)
        {
            int e = step.RmsNorm(embedded, LayerName(layer, "nextn.enorm.weight"));
            int h = step.RmsNorm(hidden, LayerName(layer, "nextn.hnorm.weight"));
            return step.MatMul(LayerName(layer, "nextn.eh_proj.weight"), step.Op(GraphOp.Concat, stackalloc[] { e, h }, stackalloc long[] { 0 }));
        }

        /// <summary>The draft head's normed row [hidden, 1] from the layer's last row.</summary>
        protected virtual int EmitDraftHead(GraphStep step, int layer, int last)
        {
            string headNorm = HasWeight(LayerName(layer, "nextn.shared_head_norm.weight"))
                ? LayerName(layer, "nextn.shared_head_norm.weight") : "output_norm.weight";
            return step.RmsNorm(last, headNorm);
        }

        /// <summary>The draft layer ships its own token embedding (<see cref="DraftEmbedding"/> reads it).</summary>
        private bool DraftOwnEmbedding
            => _quantWeights.TryGetValue(LayerName(Config.NumLayers, "nextn.embed_tokens.weight"), out QuantizedWeight q) && q.HasHostData;

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

        /// <summary>
        /// The draft window in one call (<see cref="IDraftHead.DraftChain"/>) on executors that return integer outputs: each
        /// step reads back its ten largest logits as ids and values (80 bytes) where a step had read back the scored
        /// vocabulary's logits (593 KB of Qwen3.8's) for the speculator to scan on the host; the confidence is then the
        /// speculator's own (<see cref="TensorSharp.Runtime.Speculative.DraftHeadSpeculator.TopConfidence"/> over the same
        /// ten values), and the chain stops after the first step below <paramref name="stopBelow"/> as the per-step loop did.
        /// </summary>
        public bool SupportsDraftChain => HasDraftLayer && !HasDFlash && !_rankGraphs && GraphUnsupportedReason() == null
            && GraphExecutor?.Capabilities.GreedyIdOutputs == true;

        public int DraftChain(int[] tokens, float[] hRows, int startPos, int maxTokens, int[] draftOut, float[] probOut, float stopBelow = 0f)
        {
            if (!SupportsDraftChain) throw new InvalidOperationException($"{Config.Architecture} cannot run a draft chain.");
            int width = DraftFeatureWidth, steps = 0, pos = startPos;
            var top = new float[2 * DraftTopK];
            float[] hIn = hRows, hOut = new float[width], spare = new float[width];
            int[] stepTokens = tokens;
            while (steps < maxTokens)
            {
                RunDraftLayer(stepTokens, hIn, pos, null, hOut, top);
                int id = (int)top[0];
                float prob = TensorSharp.Runtime.Speculative.DraftHeadSpeculator.TopConfidence(top.AsSpan(DraftTopK, DraftTopK));
                draftOut[steps] = id;
                probOut[steps] = prob;
                steps++;
                if (prob < stopBelow) break;
                pos += stepTokens.Length;
                stepTokens = new[] { id };
                float[] next = ReferenceEquals(hIn, hRows) ? spare : hIn;
                hIn = hOut;
                hOut = next;
            }
            return steps;
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

        // ---- the draft layer following the trunk's own forwards

        /// <summary>
        /// The trunk's own steps hand the draft layer their rows (<see cref="EmitDraftFollow"/>): a speculating request's
        /// prompt, read on the plain path, and the step that captures the carry after it leave the layer's K/V for every
        /// row, so the drafts read the prompt. Before, the layer restarted after the prompt and the first window drafted
        /// with no context at all (the direct engines' MTP block followed the forward on its device the same way). Whole
        /// graphs on this process only: a per-rank (tensor-parallel) or multi-node step leaves the gap restart.
        /// </summary>
        public bool DraftSelfCatchUp => HasDraftLayer && !HasDFlash && !_rankGraphs && !IsMultiNodePipeline && GraphUnsupportedReason() == null;

        /// <summary>The drafts condition on the prompt's rows (<see cref="DraftSelfCatchUp"/> brings them).</summary>
        public bool DraftHeadConditionsOnPrompt => DraftSelfCatchUp;

        public void FollowPlainPrompt(string requestId)
        {
            if (requestId == null) _followPlainPrompt = true;
            else (_followPromptRequests ??= new HashSet<string>(StringComparer.Ordinal)).Add(requestId);
        }

        /// <summary>How a step of the active cache takes the draft layer along (<see cref="EmitDraftFollow"/>).</summary>
        private enum DraftFollow { None, FromStart, FromCarry }

        private bool _followPlainPrompt;   // the primary cache's next plain prompt chunk is a speculating request's
        private HashSet<string> _followPromptRequests;   // per-request slots whose next prompt chunk is (FollowPlainPrompt)
        private int _draftFollowed = -1;   // the active cache's Slot.DraftFollowed
        private Tensor _draftCarry;        // the active cache's Slot.DraftCarry

        /// <summary>
        /// Whether a step of the active cache over rows from <paramref name="start"/> can hand the draft layer its rows:
        /// from position 0 (zeros before the first row), or right behind the rows it handed it last, nothing rewound
        /// below them since (the carry then holds the row before <paramref name="start"/>). A position table (multi-axis
        /// rotary) is not the layer's positions.
        /// </summary>
        private DraftFollow PlanDraftFollow(Slot slot, int start, int[] positionTable)
        {
            if (positionTable != null || !DraftSelfCatchUp) return DraftFollow.None;
            if (start == 0) return DraftFollow.FromStart;
            return slot.DraftFollowed == start && start > slot.DraftStart && slot.DraftCarry != null ? DraftFollow.FromCarry : DraftFollow.None;
        }

        /// <summary>The carry a followed step writes (allocated on the draft layer's stage), and the followed rows' draft
        /// embedding when the step's input rows are not it (media rows, the layer's own embedding, a later stage's step):
        /// null reads the step's input rows.</summary>
        private float[] PrepareDraftFollow(Slot slot, int[] tokens, bool media)
        {
            if (slot.DraftCarry == null)
            {
                EnterStage(StageOfLayer(Config.NumLayers));
                try { slot.DraftCarry = new Tensor(_allocator, DType.Float32, DraftFeatureWidth); }
                finally { EnterStage(0); }
            }
            if (_mesh.Stages == 1 && !media && !DraftOwnEmbedding) return null;
            using Tensor embedded = DraftEmbedding(tokens);
            return embedded.GetElementsAsFloat(tokens.Length * Config.HiddenSize);
        }

        /// <summary>A followed step of <paramref name="rows"/> from <paramref name="start"/> ran on <paramref name="slot"/>.</summary>
        private static void NoteDraftFollowed(Slot slot, int start, int rows)
        {
            if (start == 0) slot.DraftStart = 0;
            slot.DraftFollowed = checked(start + rows);
        }

        /// <summary>The active cache's follow fields from its snapshot (a step of the active cache ran on it).</summary>
        private void LoadDraftFollow(Slot snapshot)
        {
            _draftCarry = snapshot.DraftCarry;
            _draftFollowed = snapshot.DraftFollowed;
            _draftStart = snapshot.DraftStart;
        }

        /// <summary>
        /// The draft layer over a step's rows as the trunk finishes them (one sequence): row p joins token p's embedding
        /// with the trunk's feature row p - 1 (the slot's carry for the step's first row, zeros at position 0), as the
        /// speculator's catch-up pairs them, and writes the layer's K/V (its attention output and FFN feed nothing, so
        /// the graph drops them). The step's last feature row becomes the carry.
        /// </summary>
        private void EmitDraftFollow(GraphStep step, int features, int embedded, GraphRequest request)
        {
            if (step.Sequences != 1) throw new InvalidOperationException("The draft layer follows one sequence's rows.");
            int n = Config.NumLayers, rows = step.Rows, width = DraftFeatureWidth;
            const long f = sizeof(float);
            int e = request.DraftRows != null ? step.InputF32(request.DraftRows, Config.HiddenSize, rows)
                : embedded >= 0 ? embedded : throw new InvalidOperationException("A followed step needs its rows' draft embedding.");
            int carry = step.State(step.Slots[0].DraftCarry, GraphFormat.F32, width, 1);
            int before = request.FollowDraft == DraftFollow.FromStart ? step.ConstantInputF32(new float[width], width, 1) : carry;
            // [the row before | the step's rows]: the first rows feed the layer, the last is the next carry (read from the
            // joined rows, so it is written after the join read the old carry).
            int joined = step.Op(GraphOp.Concat, stackalloc[] { before, features }, stackalloc long[] { 1 });
            int previous = step.View2(joined, width, rows, width * f, 0);
            step.Effect(step.Op(GraphOp.Cpy, step.View2(joined, width, 1, width * f, rows * width * f), carry));
            int x = EmitDraftInput(step, n, e, previous);
            EmitAttention(step, n, EmitBlockInput(step, n, x, BlockSite.Mixer, out _));
        }

        private void DraftLayerCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (hRows == null)
            {
                // Rows the trunk's own forwards handed the layer as they ran (a followed prompt, the step that captured
                // the carry after it) are no gap: it holds them already.
                int end = checked(startPos + tokens.Length);
                if (_draftFollowed == end && _draftStart <= startPos) return;
                // Warmup, prefix reuse and plain steps carry no features: the draft layer ignores the
                // rows before this point (the trunk keeps its full context).
                _draftStart = end;
                _draftFollowed = -1;
                return;
            }
            RunDraftLayer(tokens, hRows, startPos, null, null);
        }
    }
}
