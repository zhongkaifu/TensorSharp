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
            // the GGML builder serves with those kernels.
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

        private Slot DFlashRingSlot()
        {
            Tensor[] ringK = _dflashRingK, ringV = _dflashRingV;
            _dflashRingSlots ??= new Dictionary<Tensor[], Slot>(ReferenceEqualityComparer.Instance);
            if (_dflashRingSlots.TryGetValue(ringK, out Slot slot)) return slot;
            var k = new Tensor[ringK.Length][];
            var v = new Tensor[ringV.Length][];
            for (int l = 0; l < k.Length; l++)
            {
                k[l] = new[] { ringK[l] };
                v[l] = new[] { ringV[l] };
            }
            slot = new Slot { K = k, V = v, Capacity = DFlashRingRows };
            _dflashRingSlots[ringK] = slot;
            return slot;
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
            // Muse-Glimmer's 33280-float feature rows is 130 MB, so no intermediate array.
            long offset = (long)(rowOffset + skip) * feat, bytes = (long)keep * feat * sizeof(float);
            var features = new Tensor(_allocator, DType.Float32, keep, feat);
            fixed (float* src = &hRows[offset])
                Buffer.MemoryCopy(src, GetFloatPtr(features), bytes, bytes);
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
            step.RunsAtAnyPosition();
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

        /// <summary>The ring rows of the step's positions, (start + i) % ringRows (I64 [n, 1]), refilled when it advances.</summary>
        private int DFlashRingRowsInput(GraphStep step, int n)
        {
            int ringRows = DFlashRingRows;
            var rows = new int[n];
            void Fill()
            {
                for (int i = 0; i < n; i++) rows[i] = (step.Starts[0] + i) % ringRows;
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
        /// epsilon) and rotated at the step's positions.</summary>
        private int EmitDFlashHeads(GraphStep step, string weight, int x, int heads, int rows, string norm)
        {
            DFlashConfig cfg = DFlashSettings;
            int y = step.Reshape(step.MatMul(weight, x), cfg.HeadDim, heads, rows);
            return step.Rope(step.RmsNorm(y, step.Weight(norm), cfg.Eps), DFlashRotary);
        }

        /// <summary>K and V [head_dim, kv_heads, n] into draft layer <paramref name="layer"/>'s ring at the rows
        /// <paramref name="ringRows"/>; returns the written rings (reading them orders the read after the write).</summary>
        private void EmitDFlashRingWrite(GraphStep step, int layer, int k, int v, int n, int ringRows, out int ringK, out int ringV)
        {
            DFlashConfig cfg = DFlashSettings;
            int hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads, rows = DFlashRingRows;
            int ck = step.State(_dflashRingK[layer], GraphFormat.F32, hd, rows, kvHeads);
            int cv = step.State(_dflashRingV[layer], GraphFormat.F32, hd, rows, kvHeads);
            // [head_dim, kv_heads, n] -> [head_dim, n, kv_heads], the ring's layout; one row is the same bytes either way.
            int kRows = n == 1 ? step.Reshape(k, hd, 1, kvHeads) : step.Cont(step.Permute(k, 0, 2, 1, 3));
            int vRows = n == 1 ? step.Reshape(v, hd, 1, kvHeads) : step.Cont(step.Permute(v, 0, 2, 1, 3));
            ringK = step.Op(GraphOp.SetRows, ck, kRows, ringRows);
            ringV = step.Op(GraphOp.SetRows, cv, vRows, ringRows);
            step.Effect(ringK);
            step.Effect(ringV);
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
            step.RunsAtAnyPosition();
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
            int[] draft = cfg.HasSelector ? EmitDFlashSelector(step, rows, gamma) : EmitDFlashTop1(step, rows, gamma);
            return new StepOutputs { Logits = -1, Final = -1, Hidden = -1, Draft = draft };
        }

        /// <summary>The additive mask [ringRows, b] (F16) of the block's attention over the ring, refilled when the step
        /// advances.</summary>
        private int DFlashBlockMaskInput(GraphStep step, int b)
        {
            var mask = new ushort[(long)DFlashRingRows * b];
            FillDFlashBlockMask(mask, step.Starts[0], b);
            int node = step.InputF16(mask, DFlashRingRows, b);
            step.RefillOnAdvance(node, () =>
            {
                FillDFlashBlockMask(mask, step.Starts[0], b);
                return true;
            });
            return node;
        }

        /// <summary>
        /// Query p + s sees the ring slot of a position q of the ring's valid run (BuildRingSlotPositions) with q &lt; p and
        /// (p + s) - q &lt; sliding_window, as the per-op drafter's window mask does, and every slot of the block itself
        /// (rows (p + i) % ringRows: attention inside the block is non-causal). Built from the ring's bookkeeping alone, so
        /// nothing is read back.
        /// </summary>
        private void FillDFlashBlockMask(ushort[] mask, int p, int b)
        {
            const ushort open = 0, closed = 0xFC00;   // F16 0 and -inf
            int ringRows = DFlashRingRows, swa = DFlashSettings.SlidingWindow;
            int[] held = BuildRingSlotPositions();
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
        /// per-channel base and with the per-group delta (each broadcast), the shift a concatenation behind zeros.
        /// </summary>
        private int EmitDFlashConv(GraphStep step, int x, int taps, string baseName, int side, int b, int zeros)
        {
            DFlashConfig cfg = DFlashSettings;
            int hidden = cfg.HiddenSize, width = cfg.ConvKernelSize, group = cfg.ConvGroupSize, groups = cfg.ConvNumGroups;
            long stride = cfg.ConvProjOutSize, half = (long)width * groups;
            const long f = sizeof(float);
            int baseKernel = step.Weight(baseName);   // [hidden, taps, 2]
            int result = -1;
            for (int t = 0; t < width && t < b; t++)
            {
                int n = b - t;
                int shifted = step.View3(x, group, groups, n, group * f, hidden * f, 0);                                  // x rows 0 .. n
                int channels = step.View3(baseKernel, group, groups, 1, group * f, hidden * f, ((long)side * width + t) * hidden * f);
                int delta = step.Cont(step.View3(taps, 1, groups, n, f, stride * f, (t * stride + side * half + (long)t * groups) * f));
                int term = step.Op(GraphOp.Add, step.Op(GraphOp.Mul, shifted, channels), step.Op(GraphOp.Mul, shifted, delta));
                term = step.Reshape(term, hidden, n);                                                                   // rows t .. b
                if (t > 0)
                    term = step.Op(GraphOp.Concat, stackalloc[] { step.View2(zeros, hidden, t, hidden * f, 0), term }, stackalloc long[] { 1 });
                result = result < 0 ? term : step.Op(GraphOp.Add, result, term);
            }
            return result;
        }

        /// <summary>Plain DFlash: each row's top-1 of the target head's softmax and its probability (the head's raw logits:
        /// no logit scale, no softcap). Outputs: the ids as F32 [gamma], the confidences [gamma].</summary>
        private int[] EmitDFlashTop1(GraphStep step, int rows, int gamma)
        {
            int vocab = Config.VocabSize;
            int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { step.MatMul(DFlashTargetOutputWeightName, rows) }, default, stackalloc[] { 1f, 0f });
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
        /// </summary>
        private int[] EmitDFlashSelector(GraphStep step, int rows, int gamma)
        {
            DFlashConfig cfg = DFlashSettings;
            int vocab = Config.VocabSize, k = cfg.SelectorTopK, rank = cfg.SelectorRank;
            string prefix = DFlashConfig.WeightPrefix;
            const long f = sizeof(float);
            int logits = step.MatMul(DFlashTargetOutputWeightName, rows);                                                  // [vocab, gamma]
            int cand = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { k }));      // I32 [k, gamma]
            int unary = step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(logits, 1, vocab, gamma), cand), k, gamma);
            if (cfg.HasUnaryLogitTransform)
            {
                unary = step.Op(GraphOp.Scale, stackalloc[] { unary }, default, stackalloc[] { cfg.LogitScale });
                if (cfg.FinalLogitSoftcap > 0f)
                {
                    unary = step.Op(GraphOp.Scale, stackalloc[] { unary }, default, stackalloc[] { 1f / cfg.FinalLogitSoftcap });
                    unary = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, unary) }, default, stackalloc[] { cfg.FinalLogitSoftcap });
                }
            }
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
