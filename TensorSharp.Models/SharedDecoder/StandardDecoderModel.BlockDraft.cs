// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// The DFlash / DFlash2 block drafter as graph programs, once for every family and backend on the runtime. The passes
    /// are ModelBase.DFlash.cs's (its header documents them); this file lowers PASS A+B and PASS C to two programs:
    ///
    ///   inject  feature rows -> fc -> enc.output_norm -> per draft layer K = rope(headnorm(attn_k g)) at the target
    ///           positions and V = attn_v g, written into the bound request's ring (graph state) at position % ringRows.
    ///   block   [anchor, MASK x (b-1)] (the target's embedding) through the draft layers. Each layer writes the block's
    ///           own K/V into the ring at (p + i) % ringRows, then attends the WHOLE ring under an additive mask made from
    ///           positions alone: query p + s sees a slot holding position q in the ring's valid run with q &lt; p and
    ///           (p + s) - q &lt; sliding_window, and every slot of the block (non-causal inside it). The block's slots are
    ///           future positions: ringRows &gt;= sliding_window + block, so none of them holds a key any query of this
    ///           block may see, and the injects of the positions they stand for overwrite them before they are context
    ///           (the fused GGML graph's df_ring_block does the same). DFlash2's grouped dynamic convolution wraps both
    ///           sublayers. The head is the target's LM head without its logit transform: plain DFlash takes each row's
    ///           top-1 of its softmax; DFlash2's selector (whose unary term does take the transform) walks its lattice on
    ///           the device.
    ///
    /// The per-op drafter read the scores back in every layer to fill its window mask and scanned the vocabulary's logits
    /// on the host; here only the b - 1 drafted ids (as F32 values) and their confidences come back. Both programs run
    /// through <see cref="RunTraced"/> as the MTP draft layer's steps do: replayed by signature, and on a family whose
    /// decode steps advance, run as the kept step moved on to the new positions (positions advance, the ring rows, mask
    /// and anchor are refilled, the embedded rows rebound), which a backend that records decode steps records.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Test seam: true keeps the drafter on its fused or per-op path.</summary>
        internal bool GraphBlockDraftDisabledForTests { get; set; }

        /// <summary>Tests: the graph programs draft where the fused GGML drafter would be kept for speed (ggml_metal's
        /// DFlash2 selector tail).</summary>
        internal bool GraphBlockDraftOnMetalForTests { get; set; }

        /// <summary>Tests: DFlash2's convolutions are emitted as their ops alone, not as a region an executor may serve with
        /// its kernel (<see cref="GraphFusion.DFlashConv"/>).</summary>
        internal bool DFlashConvRegionsDisabledForTests { get; set; }

        /// <summary>Blocks and ring injects the graph programs served (tests check the lowering ran).</summary>
        internal int GraphBlockDrafts { get; private set; }

        internal int GraphDFlashInjects { get; private set; }

        /// <summary>Why the graph drafter stopped (an executor refused or failed one of its programs), or null.</summary>
        internal string GraphBlockDraftDeclineReason { get; private set; }

        private const string DFlashInjectShape = DraftShapePrefix + "dflash-inject";
        private const string DFlashBlockShape = DraftShapePrefix + "dflash-block";

        /// <summary>Each ring's tensors as a slot (its K/V per layer): the steps bind it as their sequence's caches, which is
        /// what keeps a request's kept step from advancing over another request's ring.</summary>
        private Dictionary<Tensor[], Slot> _dflashRingSlots;

        private RotarySpec _dflashRotary;

        /// <summary>The anchor of the block being drafted (the selector's first predecessor), read when a kept step advances.</summary>
        private int _graphDraftAnchor;

        /// <summary>The token ids 0 .. vocab - 1 as F32 values [vocab, 1]: a drafted id is gathered from here by its I32
        /// index, an F32 row gather every executor has (the managed CPU copy refuses an I32 to F32 cast).</summary>
        private Tensor _dflashTokenIds;

        /// <summary>
        /// Why the drafter does not run as graph programs here, or null. Whole graphs on one device (not per-rank graphs,
        /// not a multi-node pipeline), and a plain DFlash or DFlash2 drafter: a DSpark Markov head chains every row on the
        /// host, attention sinks join the softmax as extra logits, and DSpine has its own graphs; each keeps its path.
        /// </summary>
        private string GraphBlockDraftRefusal()
        {
            if (GraphBlockDraftDisabledForTests) return "disabled for tests";
            if (GraphBlockDraftDeclineReason != null) return GraphBlockDraftDeclineReason;
            DFlashConfig cfg = DFlashSettings;
            if (cfg == null || _dflashRingK == null || _dflashLayerNames == null) return "no drafter";
            if (GraphUnsupportedReason() is { } reason) return reason;
            if (_rankGraphs || IsMultiNodePipeline || !_mesh.IsSingleDevice) return "the drafter runs as graphs on one device";
            if (cfg.IsDSpine) return "a DSpine drafter";
            if (cfg.MarkovRank > 0) return "a DSpark Markov head";
            if (cfg.HasAttentionSinks) return "attention sinks";
            if (DFlashRingRows < cfg.SlidingWindow + cfg.BlockSize) return "a ring narrower than the window and a block";
            if (DFlashDraftVocabSubsetRequested) return "a draft vocabulary subset (fused graph only)";
            // ggml_metal's fused drafter runs DFlash2's head, top-k and selector walk on TensorSharp's own Metal kernels after
            // its graph (tsm::df_tail_run): 12.3 ms a block of Qwen3.8-27B UD-IQ3_XXS's drafter on an M5 Pro against 15.5 ms
            // for these programs, whose selector walk is ~70 dependent kernels. It keeps that path until the walk is a region
            // the GGML builder serves with those kernels. Several requests' blocks keep it too (StandardDecoderModel.BlockDraftLanes.cs):
            // the program over four requests' blocks took 48.0 ms a step against 42.6 for four fused drafts, and over two 23.1
            // against 21.3 (the drafter's products are compute-bound on the M5 Pro past 8 rows, as the trunk's verify is).
            if (_backend == BackendType.GgmlMetal && cfg.HasSelector && !GraphBlockDraftOnMetalForTests) return "ggml_metal's fused selector tail";
            return null;
        }

        /// <summary>A program of the drafter failed or was refused: the drafter keeps its other path from here on.</summary>
        private void DeclineGraphBlockDraft(Exception e)
        {
            GraphBlockDraftDeclineReason = e.Message;
            Console.Error.WriteLine($"[dflash] the drafter's graph programs failed on {_backend} ({e.Message}); " +
                "it drafts on its fused or per-op path from here on.");
        }

        /// <summary>The model's graph drafter state goes with its other decoder resources.</summary>
        private void ReleaseGraphBlockDraft()
        {
            // A backend that keeps device copies by host address must not serve this table's to a later tensor there.
            InvalidateTensorDeviceCache(_dflashTokenIds);
            _dflashTokenIds?.Dispose();
            _dflashTokenIds = null;
            _dflashRingSlots = null;
        }

        /// <summary>The F32 ids <paramref name="indices"/> (I32 [n], contiguous) stand for: [1, n].</summary>
        private int EmitDFlashTokenIds(GraphStep step, int indices)
        {
            int vocab = Config.VocabSize;
            if (_dflashTokenIds == null)
            {
                var ids = new float[vocab];
                for (int i = 0; i < vocab; i++) ids[i] = i;
                _dflashTokenIds = CreateFloatTensor(ids, vocab, 1);
            }
            return step.Op(GraphOp.GetRows, step.TensorWeight(_dflashTokenIds, 1, vocab), indices);
        }

        private Slot DFlashRingSlot() => DFlashRingSlotOf(_dflashRingK, _dflashRingV);

        /// <summary>The slot of the ring whose layers are <paramref name="ringK"/> and <paramref name="ringV"/>.</summary>
        private Slot DFlashRingSlotOf(Tensor[] ringK, Tensor[] ringV)
        {
            _dflashRingSlots ??= new Dictionary<Tensor[], Slot>(ReferenceEqualityComparer.Instance);
            if (_dflashRingSlots.TryGetValue(ringK, out Slot slot)) return slot;
            var k = new Tensor[ringK.Length][];
            var v = new Tensor[ringV.Length][];
            for (int l = 0; l < k.Length; l++)
            {
                k[l] = new[] { ringK[l] };
                v[l] = new[] { ringV[l] };
            }
            slot = new Slot { K = k, V = v, Capacity = DFlashRingRows, DraftRing = true };
            _dflashRingSlots[ringK] = slot;
            return slot;
        }

        private protected override void OnDFlashRingOwnerChanged(Tensor[] ringK)
        {
            if (_dflashRingSlots != null && _dflashRingSlots.TryGetValue(ringK, out Slot slot)) slot.DraftRingOwners++;
        }

        /// <summary>A pooled ring given back (ModelBase.ReleaseIdleDFlashRings): the steps, recordings and graphs over it go,
        /// as a disposed slot's do.</summary>
        private protected override void OnDFlashRingDisposing(Tensor[] ringK)
        {
            if (_dflashRingSlots == null || !_dflashRingSlots.Remove(ringK, out Slot slot)) return;
            ForgetGraphsOf(slot);
        }

        /// <summary>The drafter's rotary: NeoX over the whole head at its own base (llama.cpp maps LLM_ARCH_DFLASH to
        /// LLAMA_ROPE_TYPE_NEOX), the per-op <c>DFlashRoPE</c>'s parameters.</summary>
        private RotarySpec DFlashRotary => _dflashRotary ??= new RotarySpec(DFlashSettings.HeadDim, neoX: true, DFlashSettings.RopeBase);

        // ---------------------------------------------------------------- PASS A + B: the ring inject

        private protected override unsafe bool TryGraphDFlashInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            if (n <= 0 || GraphBlockDraftRefusal() != null) return false;
            DFlashConfig cfg = DFlashSettings;
            int feat = cfg.FeatureSize;
            long need = (long)(rowOffset + n) * feat;
            if (hRows == null || hRows.LongLength < need)
                throw new ArgumentException($"DFlash encoder needs {need} feature floats but got {hRows?.LongLength ?? 0}.", nameof(hRows));

            // The ring keeps the last ringRows positions: older rows would be overwritten in place.
            int keep = Math.Min(n, DFlashRingRows), skip = n - keep, first = startPos + skip;
            // The rows go straight into the step's input tensor, as DFlashEncode copies them: a 1024-row prompt chunk of
            // Muse-Glimmer's 33280-float feature rows is 130 MB, so no intermediate array. They fill the whole fresh tensor, so
            // its checkout reads nothing back: a read-write checkout of a fresh CUDA tensor had copied its device bytes to the
            // host after the stream drained, every verify step's catch-up (800 KB of Qwen3.8-27B's 8 rows behind the commit).
            long offset = (long)(rowOffset + skip) * feat, bytes = (long)keep * feat * sizeof(float);
            var features = new Tensor(_allocator, DType.Float32, keep, feat);
            fixed (float* src = &hRows[offset])
                Buffer.MemoryCopy(src, GetFloatPtrForOverwrite(features), bytes, bytes);
            InvalidateTensorDeviceCache(features);
            CachedGraph graph = null;
            bool owned = false;
            try
            {
                graph = RunTraced(keep, new[] { DFlashRingSlot() }, new[] { 0 }, new[] { keep }, new[] { first },
                    replayable: keep <= SmallChunkRows, snapshots: false, attentionFloor: 0,
                    step => EmitDFlashInject(step, features, keep), out _, out owned,
                    shape: DFlashInjectShape, advanceInputs: new[] { features });
            }
            catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
            {
                DeclineGraphBlockDraft(e);
                return false;
            }
            finally
            {
                if (owned) graph?.Dispose();
                features.Dispose();
            }
            if (graph == null) return false;   // a trace the lowering does not express: nothing ran
            // See DFlashInjectKv: the frontier is an assignment, and the device copy of the ring is ahead of the host's.
            NoteDFlashRingWrite(first, keep);
            _dflashRingHostStale = true;
            GraphDFlashInjects++;
            return true;
        }

        /// <summary>g = rmsnorm(fc · feat, enc.output_norm); per draft layer the ring's K (head-normed, rotated at the target
        /// positions) and V rows. Writes only (effects): the program has no outputs.</summary>
        private StepOutputs EmitDFlashInject(GraphStep step, Tensor features, int n)
        {
            DFlashConfig cfg = DFlashSettings;
            string prefix = DFlashConfig.WeightPrefix;
            int hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads;
            var positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = step.Starts[0] + i;
            step.SetPositions(positions);
            int ringRows = DFlashRingRowsInput(step, n);
            int x = step.HiddenInput(features, cfg.FeatureSize, n);
            int g = step.RmsNorm(step.MatMul(prefix + "fc.weight", x), step.Weight(prefix + "enc.output_norm.weight"), cfg.Eps);
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];
                int k = EmitDFlashHeads(step, names[DfAttnK], g, kvHeads, n, names[DfAttnKNorm]);
                int v = step.Reshape(step.MatMul(names[DfAttnV], g), hd, kvHeads, n);
                EmitDFlashRingWrite(step, il, k, v, n, ringRows, out _, out _);
            }
            return new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
        }

        /// <summary>The ring rows of sequence <paramref name="sequence"/>'s positions from its start plus
        /// <paramref name="offset"/>, (start + offset + i) % ringRows (I64 [n, 1]), refilled when the step advances.</summary>
        private int DFlashRingRowsInput(GraphStep step, int n, int sequence = 0, int offset = 0)
        {
            int ringRows = DFlashRingRows;
            var rows = new int[n];
            void Fill()
            {
                for (int i = 0; i < n; i++) rows[i] = (step.Starts[sequence] + offset + i) % ringRows;
            }
            Fill();
            int node = step.InputRows(rows);
            step.RefillOnAdvance(node, () =>
            {
                Fill();
                step.RewriteInput(node, rows);
                return true;
            });
            return node;
        }

        /// <summary>weight · x as heads [head_dim, heads, rows], each RMS-normed with <paramref name="norm"/> (the drafter's
        /// epsilon) and rotated at the step's positions, or at <paramref name="positions"/> (I32 [rows]) when given.</summary>
        private int EmitDFlashHeads(GraphStep step, string weight, int x, int heads, int rows, string norm, int positions = -1)
        {
            DFlashConfig cfg = DFlashSettings;
            int y = step.Reshape(step.MatMul(weight, x), cfg.HeadDim, heads, rows);
            int normed = step.RmsNorm(y, step.Weight(norm), cfg.Eps);
            return positions >= 0 ? step.RopeAt(normed, DFlashRotary, positions) : step.Rope(normed, DFlashRotary);
        }

        /// <summary>K and V [head_dim, kv_heads, n] into draft layer <paramref name="layer"/>'s ring at the rows
        /// <paramref name="ringRows"/>; returns the written rings (reading them orders the read after the write). The bound
        /// ring, or <paramref name="ringKTensor"/> / <paramref name="ringVTensor"/> (another request's).</summary>
        private void EmitDFlashRingWrite(GraphStep step, int layer, int k, int v, int n, int ringRows, out int ringK, out int ringV,
            Tensor ringKTensor = null, Tensor ringVTensor = null)
        {
            DFlashConfig cfg = DFlashSettings;
            int hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads, rows = DFlashRingRows;
            int ck = step.State(ringKTensor ?? _dflashRingK[layer], GraphFormat.F32, hd, rows, kvHeads);
            int cv = step.State(ringVTensor ?? _dflashRingV[layer], GraphFormat.F32, hd, rows, kvHeads);
            // [head_dim, kv_heads, n] -> [head_dim, n, kv_heads], the ring's layout; one row is the same bytes either way.
            int kRows = n == 1 ? step.Reshape(k, hd, 1, kvHeads) : step.Cont(step.Permute(k, 0, 2, 1, 3));
            int vRows = n == 1 ? step.Reshape(v, hd, 1, kvHeads) : step.Cont(step.Permute(v, 0, 2, 1, 3));
            ringK = step.Op(GraphOp.SetRows, ck, kRows, ringRows);
            ringV = step.Op(GraphOp.SetRows, cv, vRows, ringRows);
            step.Effect(ringK);
            step.Effect(ringV);
        }

        // ---------------------------------------------------------------- the trunk's own steps keep the ring

        /// <summary>Tests: the prompt reaches the ring through the speculative context (its feature rows to the host and back
        /// through the inject program), as it does wherever the graph drafter is refused.</summary>
        internal bool DFlashSelfCatchUpDisabledForTests { get; set; }

        /// <summary>Sequences' prompt chunks whose rows a trunk step wrote into their rings (tests check the follow ran).</summary>
        internal int GraphDFlashFollows { get; private set; }

        /// <summary>
        /// The trunk's own steps write the drafter's ring (<see cref="DraftSelfCatchUp"/>): a speculating request's prompt,
        /// read on the plain path, takes the drafter's target layers inside the trunk's graph and writes their K/V rows into
        /// the request's ring there (<see cref="EmitDFlashFollow"/>), as the direct engine's forward did. Through the
        /// speculative context instead, the prompt's step returned five layers' rows to the host (25,600 floats a row of
        /// Qwen3.8-27B's DFlash2), the executor shifted them a row, and the inject program copied and uploaded them again
        /// (a 4K prompt's first token 2.06 s against 1.07 s without the drafter on an RTX PRO 4000). Where the drafter keeps
        /// its fused or per-op path (<see cref="GraphBlockDraftRefusal"/>), the prompt still goes through the context.
        /// </summary>
        private bool DFlashSelfCatchUp => HasDFlash && !DFlashSelfCatchUpDisabledForTests && GraphBlockDraftRefusal() == null;

        /// <summary>Which sequences of a trunk step write their rows into a ring, and which rows.</summary>
        private sealed class DFlashFollowPlan
        {
            /// <summary>Whose ring each sequence's rows go to (null: the bound ring).</summary>
            public required string[] Requests { get; init; }
            /// <summary>Each sequence's last rows the ring takes (0: the sequence does not follow).</summary>
            public required int[] Keep { get; init; }
            /// <summary>The sequence's ring also takes a zero row for the position before its rows (see
            /// <see cref="PlanDFlashFollow"/>).</summary>
            public required bool[] ZeroBefore { get; init; }
            /// <summary>Each following sequence's ring, per draft layer.</summary>
            public required Tensor[][] RingK { get; init; }
            public required Tensor[][] RingV { get; init; }
            /// <summary>Each following sequence's ring's carry: its last row's feature row is written there, so the first
            /// draft after the prompt has its carry without the step returning features (<see cref="TryTakeHeldCarry"/>).</summary>
            public required Tensor[] Carry { get; init; }

            /// <summary>What the plan changes in the emitted program.</summary>
            public string Key
            {
                get
                {
                    var key = new System.Text.StringBuilder();
                    for (int s = 0; s < Keep.Length; s++) key.Append(s > 0 ? "." : "").Append(Keep[s]).Append(ZeroBefore[s] ? "z" : "");
                    return key.ToString();
                }
            }
        }

        /// <summary>A sequence that does not follow (<see cref="PlanDFlashFollow"/>'s prompt ends).</summary>
        private const int DFlashNoFollow = int.MinValue;

        /// <summary>Rows of the drafter's encoder product per block: the tapped layers' rows are joined a block at a time, so
        /// the joined rows (a feature row is five hidden rows) never hold a whole prompt chunk.</summary>
        private const int DFlashFollowBlockRows = 256;

        /// <summary>
        /// Which sequences of a step over rows starts[s] .. + counts[s] write them into a ring: those whose prompt end
        /// (<paramref name="promptEnds"/>[s]; -1 unknown) is not <see cref="DFlashNoFollow"/>, each the rows its ring holds
        /// when the prompt ends (the chunk's last ring-length rows, none older than the prompt's last ring-length
        /// positions: a 16K prompt's earlier chunks would be overwritten). Null when none does.
        ///
        /// A ring row is its own position's features, so a chunk needs nothing of the rows before it, with one exception: the
        /// last row a turn forwarded is the first row of the next turn's prompt to have none (its features were the carry
        /// the next draft would have injected). A chunk starting right past that one-row hole writes a zero row there, as
        /// the speculative context's catch-up did (its carry is zeros when a request arms after a reused prefix), so the
        /// ring keeps the turn before as context instead of restarting at the hole.
        /// <para>A sequence marked in <paramref name="aside"/> does not speculate (<see cref="FollowUnspeculatedPrompt"/>): it
        /// takes a ring only where no running request loses its own.</para>
        /// </summary>
        private DFlashFollowPlan PlanDFlashFollow(string[] requests, int[] starts, int[] counts, int[] promptEnds, bool[] aside = null)
        {
            if (!DFlashSelfCatchUp) return null;
            int n = counts.Length, ringRows = DFlashRingRows;
            var keep = new int[n];
            var zero = new bool[n];
            var ringK = new Tensor[n][];
            var ringV = new Tensor[n][];
            var carry = new Tensor[n];
            bool any = false;
            for (int s = 0; s < n; s++)
            {
                if (promptEnds[s] == DFlashNoFollow || counts[s] <= 0) continue;
                int end = starts[s] + counts[s], first = Math.Max(starts[s], end - ringRows);
                if (promptEnds[s] > 0) first = Math.Max(first, promptEnds[s] - ringRows);
                if (first >= end) continue;
                (Tensor[] k, Tensor[] v, int filled, Tensor ringCarry) = DFlashRingOf(requests[s], mayTake: aside == null || !aside[s]);
                // Two sequences on one ring (more speculating requests than rings): the later one's prompt is not kept.
                bool shared = false;
                for (int o = 0; o < s; o++) shared |= ReferenceEquals(ringK[o], k);
                if (k == null || shared) continue;
                keep[s] = end - first;
                zero[s] = first == starts[s] && first > 0 && filled == first - 1;
                ringK[s] = k;
                ringV[s] = v;
                carry[s] = ringCarry;
                any = true;
            }
            return any ? new DFlashFollowPlan { Requests = requests, Keep = keep, ZeroBefore = zero, RingK = ringK, RingV = ringV, Carry = carry } : null;
        }

        /// <summary>A step with <paramref name="plan"/> ran: each following sequence's ring holds its kept rows.</summary>
        private void NoteDFlashFollowed(DFlashFollowPlan plan, int[] starts, int[] counts)
        {
            if (plan == null) return;
            for (int s = 0; s < plan.Keep.Length; s++)
            {
                if (plan.Keep[s] <= 0) continue;
                int zero = plan.ZeroBefore[s] ? 1 : 0;
                NoteDFlashRingWriteOf(plan.Requests[s], starts[s] + counts[s] - plan.Keep[s] - zero, plan.Keep[s] + zero,
                    carried: plan.Carry[s] != null);
                GraphDFlashFollows++;
            }
        }

        /// <summary>
        /// Run a trunk step with <paramref name="plan"/> (<paramref name="run"/> returns false when the step does not lower:
        /// nothing ran). When the ring's writes are what the lowering or the executor refused, the drafter keeps its other
        /// path from here on (the prompt then goes through the context) and the step runs without them. Returns whether the
        /// step ran; <paramref name="plan"/> is null after when the ring did not take its rows.
        /// </summary>
        private bool RunFollowingDFlash(ref DFlashFollowPlan plan, Func<DFlashFollowPlan, bool> run)
        {
            if (plan == null) return run(null);
            try
            {
                if (run(plan)) return true;
            }
            catch (NotSupportedException e)
            {
                // An executor refuses an op when it compiles the step, before anything runs.
                plan = null;
                DeclineGraphBlockDraft(e);
                return run(null);
            }
            plan = null;
            if (!run(null)) return false;
            DeclineGraphBlockDraft(new NotSupportedException("the trunk's step does not lower with the drafter's ring writes"));
            return true;
        }

        /// <summary>
        /// The drafter's ring over a step's rows as the trunk finishes them: per following sequence, its kept rows'
        /// target-layer rows (<paramref name="taps"/>, [hidden, rows] each, in the drafter's feature order) joined into
        /// feature rows a block at a time, g = rmsnorm(fc · feat, enc.output_norm), and per draft layer the K (head-normed,
        /// rotated at the rows' positions) and V rows written into the sequence's ring at position % ringRows: PASS A+B of
        /// the inject program over the same rows, which the speculative context fed it from the host. A ring row is its own
        /// position's features, so the step needs nothing from the steps before it.
        /// </summary>
        private void EmitDFlashFollow(GraphStep step, int[] taps, DFlashFollowPlan plan)
        {
            DFlashConfig cfg = DFlashSettings;
            string prefix = DFlashConfig.WeightPrefix;
            int hidden = Config.HiddenSize, hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads, ringRows = DFlashRingRows;
            const long f = sizeof(float);
            for (int s = 0; s < step.Sequences; s++)
            {
                int keep = plan.Keep[s];
                if (keep <= 0) continue;
                int count = step.Counts[s], firstRow = step.First[s] + count - keep, firstPos = step.Starts[s] + count - keep;
                int g = -1, lastBlock = -1, lastRows = 0;
                for (int b0 = 0; b0 < keep; b0 += DFlashFollowBlockRows)
                {
                    int nb = Math.Min(DFlashFollowBlockRows, keep - b0);
                    int x = -1;
                    foreach (int tap in taps)
                    {
                        int rows = step.View2(tap, hidden, nb, hidden * f, (long)(firstRow + b0) * hidden * f);
                        x = x < 0 ? rows : step.Op(GraphOp.Concat, stackalloc[] { x, rows }, stackalloc long[] { 0 });
                    }
                    int part = step.MatMul(prefix + "fc.weight", x);
                    g = g < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { g, part }, stackalloc long[] { 1 });
                    (lastBlock, lastRows) = (x, nb);
                }
                if (plan.Carry[s] != null)
                {
                    // The chunk's last feature row into the ring's carry, the first draft's: read from the rows the encoder
                    // joined (a one-tap drafter's are the tap's own rows); the step returns no features.
                    int feat = cfg.FeatureSize;
                    int last = taps.Length == 1
                        ? step.View2(taps[0], hidden, 1, hidden * f, (long)(firstRow + keep - 1) * hidden * f)
                        : step.View2(lastBlock, feat, 1, feat * f, (long)(lastRows - 1) * feat * f);
                    step.Effect(step.Op(GraphOp.Cpy, last, step.State(plan.Carry[s], GraphFormat.F32, feat, 1)));
                }
                g = step.RmsNorm(g, step.Weight(prefix + "enc.output_norm.weight"), cfg.Eps);
                int written = keep;
                if (plan.ZeroBefore[s])
                {
                    // The encoder's row of zero features is zeros, and so are its K and V.
                    g = step.Op(GraphOp.Concat, stackalloc[] { step.ConstantInputF32(new float[cfg.HiddenSize], cfg.HiddenSize, 1), g },
                        stackalloc long[] { 1 });
                    firstPos--;
                    written++;
                }
                // The rows' positions and ring rows are the step's own (opaque inputs: a followed step is never kept to advance).
                var positions = new int[written];
                var slots = new int[written];
                for (int i = 0; i < written; i++)
                {
                    positions[i] = firstPos + i;
                    slots[i] = (firstPos + i) % ringRows;
                }
                int positionsNode = step.InputI32(positions, written);
                int slotsNode = step.InputRows(slots);
                for (int il = 0; il < cfg.NumLayers; il++)
                {
                    string[] names = _dflashLayerNames[il];
                    int k = EmitDFlashHeads(step, names[DfAttnK], g, kvHeads, written, names[DfAttnKNorm], positionsNode);
                    int v = step.Reshape(step.MatMul(names[DfAttnV], g), hd, kvHeads, written);
                    EmitDFlashRingWrite(step, il, k, v, written, slotsNode, out _, out _, plan.RingK[s][il], plan.RingV[s][il]);
                }
            }
        }

        // ---------------------------------------------------------------- PASS C: the block

        private protected override int TryGraphDFlashDraftBlock(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            if (b < 2 || GraphBlockDraftRefusal() != null) return -1;
            DFlashConfig cfg = DFlashSettings;
            int gamma = b - 1;
            var ids = new int[b];
            ids[0] = anchorToken;
            for (int i = 1; i < b; i++) ids[i] = cfg.MaskTokenId;
            // llama.cpp's dflash graph feeds build_inp_embd straight in: no trunk embedding scale (DFlashEmbedding).
            Tensor embedded = DFlashEmbedding(ids);
            _graphDraftAnchor = anchorToken;
            CachedGraph graph = null;
            bool owned = false;
            try
            {
                graph = RunTraced(b, new[] { DFlashRingSlot() }, new[] { 0 }, new[] { b }, new[] { position },
                    replayable: b <= SmallChunkRows, snapshots: false, attentionFloor: 0,
                    step => EmitDFlashBlock(step, embedded, b), out StepOutputs outputs, out owned,
                    shape: DFlashBlockShape, advanceInputs: new[] { embedded });
                if (graph == null) return -1;   // a trace the lowering does not express: nothing ran
                // The ids' outputs, then the confidences', each half gamma values in block order.
                int half = outputs.Draft.Length / 2, at = 0;
                for (int o = 0; o < half; o++)
                {
                    float[] idValues = graph.Outputs[outputs.Draft[o]], confValues = graph.Outputs[outputs.Draft[half + o]];
                    for (int i = 0; i < idValues.Length && at < gamma; i++, at++)
                    {
                        draftOut[at] = (int)idValues[i];
                        if (confOut != null && at < confOut.Length) confOut[at] = confValues[i];
                    }
                }
                if (outputs.Trace != null) LastDFlashTraceForTests = ReadDFlashTrace(graph, outputs.Trace, gamma);
            }
            catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
            {
                DeclineGraphBlockDraft(e);
                return -1;
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded.Dispose();
            }
            // The block's K/V rows went into the ring's future slots on the device.
            _dflashRingHostStale = true;
            GraphBlockDrafts++;
            return gamma;
        }

        /// <summary>Tests: a lone block's trace outputs (<see cref="StepOutputs.Trace"/>) as each slot's logits, candidates and
        /// scores.</summary>
        private DFlashSlotTrace[] ReadDFlashTrace(CachedGraph graph, int[] trace, int gamma)
        {
            int vocab = Config.VocabSize;
            float[] logits = graph.Outputs[trace[0]];
            var slots = new DFlashSlotTrace[gamma];
            for (int e = 0; e < gamma; e++)
            {
                slots[e] = new DFlashSlotTrace { Logits = logits.AsSpan(e * vocab, vocab).ToArray() };
                if (trace.Length > 1 + 2 * e + 1)
                {
                    slots[e].Candidates = Array.ConvertAll(graph.Outputs[trace[1 + 2 * e]], v => (int)v);
                    slots[e].Scores = (float[])graph.Outputs[trace[2 + 2 * e]].Clone();
                }
            }
            return slots;
        }

        /// <summary>
        /// The block's program: the draft layers over [anchor, MASK x (b-1)] at positions p .. p + b - 1, then the head over
        /// rows 1 .. b - 1. Its outputs (<see cref="StepOutputs.Draft"/>) are the drafted ids as F32 values, then their
        /// confidences, in block order.
        /// </summary>
        private StepOutputs EmitDFlashBlock(GraphStep step, Tensor embedded, int b)
        {
            DFlashConfig cfg = DFlashSettings;
            int hidden = cfg.HiddenSize, hd = cfg.HeadDim, heads = cfg.NumHeads, kvHeads = cfg.NumKVHeads, gamma = b - 1;
            var positions = new int[b];
            for (int i = 0; i < b; i++) positions[i] = step.Starts[0] + i;
            step.SetPositions(positions);
            int ringRows = DFlashRingRowsInput(step, b);
            int mask = DFlashBlockMaskInput(step, b);
            // The convolution's taps reach back up to taps - 1 rows; the rows before the block are zeros.
            int zeros = cfg.HasConv && cfg.ConvKernelSize > 1
                ? step.ConstantInputF32(new float[(long)hidden * (cfg.ConvKernelSize - 1)], hidden, cfg.ConvKernelSize - 1) : -1;
            float scale = 1f / MathF.Sqrt(hd);

            int x = step.HiddenInput(embedded, hidden, b);
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];
                int h = step.RmsNorm(x, step.Weight(names[DfAttnNorm]), cfg.Eps);
                // DFlash2: one projection of the sublayer's input gives both its input filter and its output filter.
                int attnTaps = -1;
                if (cfg.HasConv)
                {
                    attnTaps = step.MatMul(names[DfAttnConvProj], h);
                    h = EmitDFlashConv(step, h, attnTaps, names[DfAttnConvBase], side: 0, b, zeros);
                }
                int q = EmitDFlashHeads(step, names[DfAttnQ], h, heads, b, names[DfAttnQNorm]);
                int k = EmitDFlashHeads(step, names[DfAttnK], h, kvHeads, b, names[DfAttnKNorm]);
                int v = step.Reshape(step.MatMul(names[DfAttnV], h), hd, kvHeads, b);
                EmitDFlashRingWrite(step, il, k, v, b, ringRows, out int ringK, out int ringV);
                int fa = step.FlashAttention(step.Permute(q, 0, 2, 1, 3), ringK, ringV, mask, scale);
                int attn = step.MatMul(names[DfAttnOutput], step.Reshape(fa, (long)heads * hd, b));
                if (cfg.HasConv) attn = EmitDFlashConv(step, attn, attnTaps, names[DfAttnConvBase], side: 1, b, zeros);
                x = step.Op(GraphOp.Add, attn, x);

                int f = step.RmsNorm(x, step.Weight(names[DfFfnNorm]), cfg.Eps);
                int ffnTaps = -1;
                if (cfg.HasConv)
                {
                    ffnTaps = step.MatMul(names[DfFfnConvProj], f);
                    f = EmitDFlashConv(step, f, ffnTaps, names[DfFfnConvBase], side: 0, b, zeros);
                }
                // The drafter ships gate and up apart (the trunk's gate/up fusion covers blk.N. names only).
                f = step.MatMul(names[DfFfnDown], step.Op(GraphOp.SwigluSplit, step.MatMul(names[DfFfnGate], f), step.MatMul(names[DfFfnUp], f)));
                if (cfg.HasConv) f = EmitDFlashConv(step, f, ffnTaps, names[DfFfnConvBase], side: 1, b, zeros);
                x = step.Op(GraphOp.Add, f, x);
            }
            int cur = step.RmsNorm(x, step.Weight(DFlashConfig.WeightPrefix + "output_norm.weight"), cfg.Eps);
            // Row 0 is the anchor's own prediction, which neither head reads.
            int rows = step.View2(cur, hidden, gamma, hidden * 4L, hidden * 4L);
            var trace = DFlashTraceForTests ? new List<int>() : null;
            int[] draft = cfg.HasSelector ? EmitDFlashSelector(step, rows, gamma, trace: trace) : EmitDFlashTop1(step, rows, gamma, trace);
            return new StepOutputs { Logits = -1, Final = -1, Hidden = -1, Draft = draft, Trace = trace?.ToArray() };
        }

        /// <summary>The additive mask [ringRows, b] (F16) of sequence <paramref name="sequence"/>'s block attending its ring,
        /// whose slots hold the positions <paramref name="held"/> returns (the bound ring's when null), refilled when the step
        /// advances.</summary>
        private int DFlashBlockMaskInput(GraphStep step, int b, int sequence = 0, Func<int[]> held = null)
        {
            held ??= BuildRingSlotPositions;
            var mask = new ushort[(long)DFlashRingRows * b];
            FillDFlashBlockMask(mask, held(), step.Starts[sequence], b);
            int node = step.InputF16(mask, DFlashRingRows, b);
            step.RefillOnAdvance(node, () =>
            {
                FillDFlashBlockMask(mask, held(), step.Starts[sequence], b);
                return true;
            });
            return node;
        }

        /// <summary>
        /// Query p + s sees the ring slot of a position q of the ring's valid run (<paramref name="held"/>: BuildRingSlotPositions'
        /// map) with q &lt; p and
        /// (p + s) - q &lt; sliding_window, as the per-op drafter's window mask does, and every slot of the block itself
        /// (rows (p + i) % ringRows: attention inside the block is non-causal). Built from the ring's bookkeeping alone, so
        /// nothing is read back.
        /// </summary>
        private void FillDFlashBlockMask(ushort[] mask, int[] held, int p, int b)
        {
            const ushort open = 0, closed = 0xFC00;   // F16 0 and -inf
            int ringRows = DFlashRingRows, swa = DFlashSettings.SlidingWindow;
            for (int s = 0; s < b; s++)
            {
                Span<ushort> row = mask.AsSpan(s * ringRows, ringRows);
                int query = p + s;
                for (int slot = 0; slot < ringRows; slot++)
                {
                    int q = held[slot];
                    row[slot] = q >= 0 && q < p && query - q < swa ? open : closed;
                }
                for (int i = 0; i < b; i++) row[(p + i) % ringRows] = open;
            }
        }

        /// <summary>
        /// DFlash2's grouped dynamic convolution on [hidden, b] rows <paramref name="x"/>: tap t of channel c at block row r
        /// is (base[side][t][c] + taps[r][side][t][c / group]) times x[r - t][c], nothing for r &lt; t (the filter never reaches
        /// across the block). <paramref name="taps"/> is the conv projection [2 * taps * groups, b] of the sublayer's input;
        /// side 0 filters the input, side 1 the sublayer's output. Each tap is the product of the shifted rows with the
        /// per-channel base and with the per-group delta (each broadcast), the shift a concatenation behind zeros. The ops are
        /// a region (<see cref="GraphFusion.DFlashConv"/>): an executor with its kernel computes them in one dispatch, with
        /// their roundings (about ten small kernels a convolution, twenty convolutions a Qwen3.8-27B DFlash2 block).
        /// <para>With <paramref name="lanes"/> blocks (several requests' blocks, one after another: [hidden, lanes * b]) each is
        /// convolved on its own, the taps never reaching from one into the next: the same ops over a lane dimension (4-D views,
        /// so the count of kernels does not grow with the lanes), and the region names the block's rows.</para>
        /// </summary>
        private int EmitDFlashConv(GraphStep step, int x, int taps, string baseName, int side, int b, int zeros, int lanes = 1)
        {
            DFlashConfig cfg = DFlashSettings;
            int hidden = cfg.HiddenSize, width = cfg.ConvKernelSize, group = cfg.ConvGroupSize, groups = cfg.ConvNumGroups;
            long stride = cfg.ConvProjOutSize, half = (long)width * groups;
            const long f = sizeof(float);
            int baseKernel = step.Weight(baseName);   // [hidden, taps, 2]
            bool region = !DFlashConvRegionsDisabledForTests;
            if (region && lanes == 1)
                step.BeginFused(GraphFusion.DFlashConv, stackalloc[] { x, taps, baseKernel }, stackalloc long[] { width, group, groups, side, stride });
            else if (region)
                step.BeginFused(GraphFusion.DFlashConv, stackalloc[] { x, taps, baseKernel }, stackalloc long[] { width, group, groups, side, stride, b });
            int result = -1;
            for (int t = 0; t < width && t < b; t++)
            {
                int n = b - t;
                long channelsAt = ((long)side * width + t) * hidden * f;
                int term;
                if (lanes == 1)
                {
                    int shifted = step.View3(x, group, groups, n, group * f, hidden * f, 0);                              // x rows 0 .. n
                    int channels = step.View3(baseKernel, group, groups, 1, group * f, hidden * f, channelsAt);
                    int delta = step.Cont(step.View3(taps, 1, groups, n, f, stride * f, (t * stride + side * half + (long)t * groups) * f));
                    term = step.Op(GraphOp.Add, step.Op(GraphOp.Mul, shifted, channels), step.Op(GraphOp.Mul, shifted, delta));
                    term = step.Reshape(term, hidden, n);                                                               // rows t .. b
                    if (t > 0)
                        term = step.Op(GraphOp.Concat, stackalloc[] { step.View2(zeros, hidden, t, hidden * f, 0), term }, stackalloc long[] { 1 });
                }
                else
                {
                    // Each lane's rows 0 .. n, and the deltas of its rows t .. b.
                    int shifted = step.View4(x, group, groups, n, lanes, group * f, hidden * f, b * hidden * f, 0);
                    int channels = step.View3(baseKernel, group, groups, 1, group * f, hidden * f, channelsAt);
                    int delta = step.Cont(step.View4(taps, 1, groups, n, lanes, f, stride * f, b * stride * f,
                        (t * stride + side * half + (long)t * groups) * f));
                    term = step.Op(GraphOp.Add, step.Op(GraphOp.Mul, shifted, channels), step.Op(GraphOp.Mul, shifted, delta));
                    term = step.Reshape(term, hidden, n, lanes);                                                        // each lane's rows t .. b
                    if (t > 0)
                        term = step.Op(GraphOp.Concat, stackalloc[] { step.View3(zeros, hidden, t, lanes, hidden * f, t * hidden * f, 0), term },
                            stackalloc long[] { 1 });
                }
                result = result < 0 ? term : step.Op(GraphOp.Add, result, term);
            }
            if (lanes > 1) result = step.Reshape(result, hidden, (long)lanes * b);
            if (region) step.EndFused(stackalloc[] { result });
            return result;
        }

        /// <summary>Plain DFlash: each row's top-1 of the target head's softmax and its probability (the head's raw logits:
        /// no logit scale, no softcap). Outputs: the ids as F32 [gamma], the confidences [gamma].</summary>
        private int[] EmitDFlashTop1(GraphStep step, int rows, int gamma, List<int> trace = null)
        {
            int vocab = Config.VocabSize;
            int logits = step.MatMul(DFlashTargetOutputWeightName, rows);
            trace?.Add(step.Output(logits, (long)vocab * gamma));
            int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { logits }, default, stackalloc[] { 1f, 0f });
            int best = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { probs }, stackalloc long[] { 1 }));        // I32 [1, gamma]
            int conf = step.Op(GraphOp.GetRows, step.Reshape(probs, 1, vocab, gamma), best);                                // [1, 1, gamma]
            int ids = EmitDFlashTokenIds(step, step.Reshape(best, gamma));
            return new[] { step.Output(ids, gamma), step.Output(conf, gamma) };
        }

        /// <summary>
        /// DFlash2's candidate selector on the device: each row's top selector_top_k candidates of the target head, their
        /// logits through the target's logit transform (scale, then softcap: monotonic, so it leaves the candidates as they
        /// are and is applied to k values instead of the vocabulary), and the greedy walk through the lattice
        ///     score[e][c] = unary[e][c] + &lt;A[pred_e] * (P h_e), B[cand[e][c]]&gt;
        /// with pred_0 the anchor and pred_e the candidate position e - 1 chose, picked by its index on the device. Outputs:
        /// each position's id (F32 [1]) in order, then each position's confidence (the softmax of its scores at the choice).
        /// <para>Several requests' blocks (<paramref name="lanes"/> &gt; 1) walk their lattices together: <paramref name="rows"/>
        /// are position-major (row e * lanes + j is lane j's position e), so each step's rows of every lane are one block, the
        /// anchors are <paramref name="anchors"/> (I32 [lanes]), the products and gathers run over the lane dimension, and
        /// each output holds the lanes' values in order (F32 [lanes]).</para>
        /// </summary>
        private int[] EmitDFlashSelector(GraphStep step, int rows, int gamma, int lanes = 1, int anchors = -1, List<int> trace = null)
        {
            DFlashConfig cfg = DFlashSettings;
            int vocab = Config.VocabSize, k = cfg.SelectorTopK, rank = cfg.SelectorRank;
            string prefix = DFlashConfig.WeightPrefix;
            const long f = sizeof(float);
            if (lanes > 1) return EmitDFlashSelectorLanes(step, rows, gamma, lanes, anchors);
            int logits = step.MatMul(DFlashTargetOutputWeightName, rows);                                                  // [vocab, gamma]
            trace?.Add(step.Output(logits, (long)vocab * gamma));
            int cand = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { k }));      // I32 [k, gamma]
            int unary = EmitDFlashUnaryTransform(step,
                step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(logits, 1, vocab, gamma), cand), k, gamma));
            int flat = step.Reshape(cand, (long)k * gamma);
            int candIds = EmitDFlashTokenIds(step, flat);                                                                  // [1, k * gamma]
            int projected = step.MatMul(prefix + "selector_hidden.weight", rows);                                         // [rank, gamma]
            int successors = EmitDFlashCodebookRows(step, prefix + "selector_successor.weight", flat);                     // [rank, k * gamma]
            int predecessors = gamma > 1 ? EmitDFlashCodebookRows(step, prefix + "selector_predecessor.weight", flat) : -1;

            var anchor = new[] { _graphDraftAnchor };
            int anchorNode = step.InputI32(anchor, 1);
            step.RefillOnAdvance(anchorNode, () =>
            {
                anchor[0] = _graphDraftAnchor;
                return true;
            });
            int pred = EmitDFlashCodebookRows(step, prefix + "selector_predecessor.weight", anchorNode);                    // [rank, 1]

            var outputs = new int[2 * gamma];
            int chosen = -1;
            for (int e = 0; e < gamma; e++)
            {
                if (e > 0)
                    pred = step.Op(GraphOp.GetRows, step.View2(predecessors, rank, k, rank * f, (long)(e - 1) * k * rank * f), chosen);
                int m = step.Op(GraphOp.Mul, pred, step.View2(projected, rank, 1, rank * f, (long)e * rank * f));
                int transition = step.Op(GraphOp.MulMat, step.View2(successors, rank, k, rank * f, (long)e * k * rank * f), m);   // [k, 1]
                int score = step.Op(GraphOp.Add, step.View2(unary, k, 1, k * f, (long)e * k * f), transition);
                chosen = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { score }, stackalloc long[] { 1 }));          // I32 [1, 1]
                int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { score }, default, stackalloc[] { 1f, 0f });
                outputs[e] = step.Output(step.Op(GraphOp.GetRows, step.View2(candIds, 1, k, f, (long)e * k * f), chosen), 1);
                outputs[gamma + e] = step.Output(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, k), chosen), 1);
                if (trace != null)
                {
                    trace.Add(step.Output(step.Cont(step.View2(candIds, 1, k, f, (long)e * k * f)), k));
                    trace.Add(step.Output(score, k));
                }
            }
            return outputs;
        }

        /// <summary>The selector's unary scores through the target's logit transform (scale, then softcap), where it has one.</summary>
        private int EmitDFlashUnaryTransform(GraphStep step, int unary)
        {
            DFlashConfig cfg = DFlashSettings;
            if (!cfg.HasUnaryLogitTransform) return unary;
            unary = step.Op(GraphOp.Scale, stackalloc[] { unary }, default, stackalloc[] { cfg.LogitScale });
            if (cfg.FinalLogitSoftcap > 0f)
            {
                unary = step.Op(GraphOp.Scale, stackalloc[] { unary }, default, stackalloc[] { 1f / cfg.FinalLogitSoftcap });
                unary = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, unary) }, default, stackalloc[] { cfg.FinalLogitSoftcap });
            }
            return unary;
        }

        /// <summary>
        /// <see cref="EmitDFlashSelector"/>'s walk for several requests' blocks at once: the lattice of every lane's position e
        /// is one contiguous block of the position-major rows, its predecessor rows gathered by each lane's choice at e - 1
        /// (a gather over the lane dimension), its transitions one product a lane (a batched product), its choice each lane's
        /// first-max candidate. The same ops a lane as the walk of one block, with the lanes as a dimension.
        /// </summary>
        private int[] EmitDFlashSelectorLanes(GraphStep step, int rows, int gamma, int lanes, int anchors)
        {
            DFlashConfig cfg = DFlashSettings;
            int vocab = Config.VocabSize, k = cfg.SelectorTopK, rank = cfg.SelectorRank, n = gamma * lanes;
            string prefix = DFlashConfig.WeightPrefix;
            const long f = sizeof(float);
            int logits = step.MatMul(DFlashTargetOutputWeightName, rows);                                                  // [vocab, n]
            int cand = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { k }));      // I32 [k, n]
            int unary = EmitDFlashUnaryTransform(step,
                step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(logits, 1, vocab, n), cand), k, n));
            int flat = step.Reshape(cand, (long)k * n);
            int candIds = EmitDFlashTokenIds(step, flat);                                                                  // [1, k * n]
            int projected = step.MatMul(prefix + "selector_hidden.weight", rows);                                         // [rank, n]
            int successors = EmitDFlashCodebookRows(step, prefix + "selector_successor.weight", flat);                     // [rank, k * n]
            int predecessors = gamma > 1 ? EmitDFlashCodebookRows(step, prefix + "selector_predecessor.weight", flat) : -1;
            int pred = step.Reshape(EmitDFlashCodebookRows(step, prefix + "selector_predecessor.weight", anchors), rank, 1, lanes);

            long block = (long)lanes * k;   // a position's candidates, every lane's
            var outputs = new int[2 * gamma];
            int chosen = -1;
            for (int e = 0; e < gamma; e++)
            {
                if (e > 0)
                    pred = step.Op(GraphOp.GetRows,
                        step.View3(predecessors, rank, k, lanes, rank * f, k * rank * f, (e - 1) * block * rank * f), chosen); // [rank, 1, lanes]
                int m = step.Op(GraphOp.Mul, pred, step.View3(projected, rank, 1, lanes, rank * f, rank * f, (long)e * lanes * rank * f));
                int transition = step.Op(GraphOp.MulMat,
                    step.View3(successors, rank, k, lanes, rank * f, k * rank * f, e * block * rank * f), m);              // [k, 1, lanes]
                int score = step.Reshape(step.Op(GraphOp.Add, step.View3(unary, k, 1, lanes, k * f, k * f, e * block * f), transition), k, lanes);
                chosen = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { score }, stackalloc long[] { 1 }));          // I32 [1, lanes]
                int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { score }, default, stackalloc[] { 1f, 0f });
                outputs[e] = step.Output(step.Op(GraphOp.GetRows, step.View3(candIds, 1, k, lanes, f, k * f, e * block * f), chosen), lanes);
                outputs[gamma + e] = step.Output(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, k, lanes), chosen), lanes);
            }
            return outputs;
        }

        /// <summary>Rows <paramref name="ids"/> of a [vocab, rank] selector codebook as [rank, ids], with a quantized table's
        /// per-tensor scale (as the per-op gather applies it).</summary>
        private int EmitDFlashCodebookRows(GraphStep step, string name, int ids)
        {
            int rows = step.Op(GraphOp.GetRows, step.Weight(name), ids);
            return _quantWeights.TryGetValue(name, out QuantizedWeight q) && q.Scale != 1f
                ? step.Op(GraphOp.Scale, stackalloc[] { rows }, default, stackalloc[] { q.Scale })
                : rows;
        }
    }
}
