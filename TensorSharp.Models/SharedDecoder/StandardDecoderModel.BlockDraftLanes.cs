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
    /// The DFlash / DFlash2 block drafter for several requests at once (<see cref="TensorSharp.Runtime.Speculative.IBatchedSpeculativeTarget.TryDraftBatch"/>),
    /// as one graph program over their rings, on every executor the drafter's programs run on:
    ///
    ///   block    each lane (request) is a sequence of the step over its own ring: its carried feature row (the last position
    ///            before its anchor, which a lone request's draft injects in a program of its own) through the encoder and each
    ///            draft layer's K/V projection into its ring, then its block [anchor, MASK x (b-1)]. Every projection, FFN,
    ///            convolution, the head and the selector run once over every lane's rows; each lane's block rows go into its
    ///            ring and its queries attend that ring alone, under its own mask. The DFlash2 convolution keeps to each lane's
    ///            block, and the selector walks every lane's lattice in the same kernels (StandardDecoderModel.BlockDraft.cs).
    ///   commits  a batched step's catch-ups (each lane's kept tokens' feature rows) wait on their rings until the step ends
    ///            (ModelBase.DFlash.Lanes.cs), and are written then in one inject program over the rings.
    ///
    /// Drafted one request at a time, four Qwen3.8-27B NVFP4 + DFlash2 requests on direct CUDA spent 12-14 ms a request per
    /// verify step drafting, as much as one request alone, where the direct engine's batched draft spent 3.4-4.6 (issue #391);
    /// and each request's commit and catch-up cost 1.5 ms a step. Both programs run as the kept step of their lanes advanced to
    /// the lanes' new positions (their rings' rows, masks and anchors refilled), which an executor that records steps records.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        private const string DFlashLanesShape = DraftShapePrefix + "dflash-lanes";
        private const string DFlashInjectLanesShape = DraftShapePrefix + "dflash-inject-lanes";

        /// <summary>Tests: every request drafts on its own, as before batched blocks.</summary>
        internal bool BatchedBlockDraftDisabledForTests { get; set; }

        /// <summary>Programs that drafted several requests' blocks (tests check the step used them).</summary>
        internal int BatchedBlockDrafts { get; private set; }

        /// <summary>Why the programs over several rings stopped (an executor refused or failed one), or null.</summary>
        internal string BatchedBlockDraftDeclineReason { get; private set; }

        /// <summary>Why several requests' blocks do not draft in one program here, or null: wherever the drafter's graph programs
        /// run (<see cref="GraphBlockDraftRefusal"/>; ggml_metal's DFlash2 keeps its fused selector tail, request by request, which
        /// it measured faster).</summary>
        private string BatchedBlockDraftRefusal()
        {
            if (BatchedBlockDraftDisabledForTests) return "disabled for tests";
            return BatchedBlockDraftDeclineReason ?? GraphBlockDraftRefusal();
        }

        private void DeclineBatchedBlockDraft(Exception e)
        {
            BatchedBlockDraftDeclineReason = e.Message;
            Console.Error.WriteLine($"[dflash] the drafter's program over several requests failed on {_backend} ({e.Message}); " +
                "each request drafts on its own from here on.");
        }

        /// <summary>Drafts <see cref="TryDraftBatch"/> fills per request: the drafter's whole block but its anchor row, as a
        /// lone request's draft runs it (DFlashPropose).</summary>
        public int BatchDraftBlockWidth => HasDFlash && BatchedBlockDraftRefusal() == null ? DFlashSettings.MaxDraftTokens : 0;

        /// <summary>The lanes of the block program being run: a kept step refills each lane's mask from its ring's map and
        /// the selector's anchors from here.</summary>
        private DFlashLaneRing[] _dflashLanes;
        private int[] _dflashLaneAnchors;

        /// <summary>
        /// One block for each of <paramref name="requestIds"/> in one program, each over its own ring: lane i's anchor
        /// <paramref name="anchors"/>[i] at <paramref name="positions"/>[i], its carried feature row <paramref name="carries"/>[i]
        /// (the features of the position before the anchor) written into its ring first. Fills each lane's whole block before
        /// any gate. False, with nothing drafted, when the program does not run here, a lane has no ring of its own or the
        /// step finds no device memory for its compute buffer: the caller then drafts each request on its own (which writes
        /// the carried rows again).
        /// </summary>
        public unsafe bool TryDraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
            int[][] tokensOut, float[][] confOut)
        {
            int m = requestIds?.Count ?? 0;
            if (m < 1 || BatchedBlockDraftRefusal() != null) return false;
            DFlashConfig cfg = DFlashSettings;
            int gamma = cfg.MaxDraftTokens, b = gamma + 1, feat = cfg.FeatureSize;
            for (int j = 0; j < m; j++)
                if (tokensOut[j].Length < gamma || confOut[j].Length < gamma)
                    throw new ArgumentException($"Lane {j} cannot take a {gamma}-token block.", nameof(tokensOut));
            DFlashLaneRing[] lanes = PrepareDFlashLaneRings(requestIds, positions);
            if (lanes == null) return false;
            // The lanes whose rings do not hold their carried row yet (a batched verify writes its kept rows into the rings,
            // the carried one among them): their carries go in with the block.
            var carriedLanes = new List<int>(m);
            var key = new System.Text.StringBuilder();
            for (int j = 0; j < m; j++)
            {
                key.Append(lanes[j].HoldsCarry ? '0' : '1');
                if (lanes[j].HoldsCarry) continue;
                if (carries[j] == null || carries[j].Length < feat)
                    throw new ArgumentException($"Lane {j}: the carried row needs {feat} floats.", nameof(carries));
                carriedLanes.Add(j);
            }
            int mc = carriedLanes.Count;

            var slots = new Slot[m];
            var first = new int[m];
            var counts = new int[m];
            var starts = new int[m];
            var ids = new int[m * b];
            for (int j = 0; j < m; j++)
            {
                slots[j] = DFlashRingSlotOf(lanes[j].K, lanes[j].V);
                first[j] = j * b;
                counts[j] = b;
                starts[j] = positions[j];
                ids[j * b] = anchors[j];
                for (int i = 1; i < b; i++) ids[j * b + i] = cfg.MaskTokenId;
            }
            // llama.cpp's dflash graph feeds build_inp_embd straight in: no trunk embedding scale (DFlashEmbedding).
            Tensor embedded = DFlashEmbedding(ids);
            Tensor carried = null;
            if (mc > 0)
            {
                carried = new Tensor(_allocator, DType.Float32, mc, feat);
                float* rows = GetFloatPtrForOverwrite(carried);
                for (int c = 0; c < mc; c++)
                    fixed (float* src = carries[carriedLanes[c]])
                        Buffer.MemoryCopy(src, rows + (long)c * feat, (long)feat * sizeof(float), (long)feat * sizeof(float));
                InvalidateTensorDeviceCache(carried);
            }
            int[] carriedAt = carriedLanes.ToArray();
            _dflashLanes = lanes;
            _dflashLaneAnchors = (int[])anchors.Clone();
            CachedGraph graph = null;
            bool owned = false;
            try
            {
                graph = RunTraced(m * b, slots, first, counts, starts, replayable: b <= SmallChunkRows, snapshots: false, attentionFloor: 0,
                    step => EmitDFlashBlockLanes(step, embedded, carried, carriedAt), out StepOutputs outputs, out owned,
                    shape: DFlashLanesShape + m + (mc == m ? "" : ":c" + key), advanceInputs: mc > 0 ? new[] { embedded, carried } : new[] { embedded });
                if (graph == null) return false;   // a trace the lowering does not express: nothing ran
                // Each output is a run of every lane's values of one position or more, positions in order (row e * m + j is
                // lane j's position e): the ids' outputs, then the confidences'.
                int half = outputs.Draft.Length / 2, at = 0;
                for (int o = 0; o < half; o++)
                {
                    float[] idValues = graph.Outputs[outputs.Draft[o]], confValues = graph.Outputs[outputs.Draft[half + o]];
                    for (int i = 0; i < idValues.Length && at < gamma * m; i++, at++)
                    {
                        int e = at / m, j = at % m;
                        tokensOut[j][e] = (int)idValues[i];
                        confOut[j][e] = confValues[i];
                    }
                }
            }
            catch (InvalidOperationException ex) when (IsGraphAllocationFailure(ex))
            {
                // The lanes' rows want more compute memory than one lane's: each drafts on its own instead.
                FreeGraphMemory();
                return false;
            }
            catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
            {
                DeclineBatchedBlockDraft(e);
                return false;
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded.Dispose();
                carried?.Dispose();
                _dflashLanes = null;
            }
            BatchedBlockDrafts++;
            return true;
        }

        /// <summary>
        /// The block program over the step's lanes (each a sequence on its own ring): the carried rows of the lanes
        /// <paramref name="carriedLanes"/> into their rings (the other lanes' rings hold theirs: a batched verify wrote them),
        /// then the draft layers over [anchor, MASK x (b-1)] of every lane at its positions p .. p + b - 1 (each lane's K/V rows
        /// into its ring, its queries over its ring under its mask), then the head over each lane's rows 1 .. b - 1 in position
        /// order. Its outputs (<see cref="StepOutputs.Draft"/>) are the drafted ids as F32 values, then their confidences.
        /// </summary>
        private StepOutputs EmitDFlashBlockLanes(GraphStep step, Tensor embedded, Tensor carried, int[] carriedLanes)
        {
            DFlashConfig cfg = DFlashSettings;
            string prefix = DFlashConfig.WeightPrefix;
            int m = step.Sequences, b = step.Counts[0], rows = m * b, gamma = b - 1;
            int hidden = cfg.HiddenSize, hd = cfg.HeadDim, heads = cfg.NumHeads, kvHeads = cfg.NumKVHeads;
            const long f = sizeof(float);
            var positions = new int[rows];
            for (int j = 0; j < m; j++)
                for (int i = 0; i < b; i++) positions[j * b + i] = step.Starts[j] + i;
            step.SetPositions(positions);
            // Each carried lane's carried row's position (the one before its anchor) and ring row, each lane's block's ring rows
            // and its mask.
            int mc = carriedLanes.Length;
            var carryOf = new int[m];
            Array.Fill(carryOf, -1);
            for (int c = 0; c < mc; c++) carryOf[carriedLanes[c]] = c;
            var carryPositions = new int[mc];
            void FillCarryPositions()
            {
                for (int c = 0; c < mc; c++) carryPositions[c] = step.Starts[carriedLanes[c]] - 1;
            }
            int carryAt = -1;
            if (mc > 0)
            {
                FillCarryPositions();
                carryAt = step.InputI32(carryPositions, mc);
                step.RefillOnAdvance(carryAt, () =>
                {
                    FillCarryPositions();
                    step.RewriteInput(carryAt, carryPositions);
                    return true;
                });
            }
            var carryRows = new int[m];
            var blockRows = new int[m];
            var masks = new int[m];
            for (int j = 0; j < m; j++)
            {
                int lane = j;
                carryRows[j] = carryOf[j] >= 0 ? DFlashRingRowsInput(step, 1, lane, offset: -1) : -1;
                blockRows[j] = DFlashRingRowsInput(step, b, lane);
                masks[j] = DFlashBlockMaskInput(step, b, lane, () => _dflashLanes[lane].SlotPositions);
            }
            // The convolution's taps reach back up to taps - 1 rows; the rows before each lane's block are zeros.
            int zeros = cfg.HasConv && cfg.ConvKernelSize > 1
                ? step.ConstantInputF32(new float[(long)hidden * (cfg.ConvKernelSize - 1) * m], hidden, (long)(cfg.ConvKernelSize - 1) * m) : -1;
            float scale = 1f / MathF.Sqrt(hd);

            int x = step.HiddenInput(embedded, hidden, rows);
            // PASS A on the carried rows: g = rmsnorm(fc · feat, enc.output_norm), one row a carried lane.
            int g = mc == 0 ? -1 : step.RmsNorm(step.MatMul(prefix + "fc.weight", step.HiddenInput(carried, cfg.FeatureSize, mc)),
                step.Weight(prefix + "enc.output_norm.weight"), cfg.Eps);
            int fa = -1;
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];
                // PASS B on the carried rows: their K (head-normed, rotated at their positions) and V, [head_dim, kv_heads, mc].
                int carryK = mc == 0 ? -1 : EmitDFlashHeads(step, names[DfAttnK], g, kvHeads, mc, names[DfAttnKNorm], carryAt);
                int carryV = mc == 0 ? -1 : step.Reshape(step.MatMul(names[DfAttnV], g), hd, kvHeads, mc);
                int h = step.RmsNorm(x, step.Weight(names[DfAttnNorm]), cfg.Eps);
                int attnTaps = -1;
                if (cfg.HasConv)
                {
                    attnTaps = step.MatMul(names[DfAttnConvProj], h);
                    h = EmitDFlashConv(step, h, attnTaps, names[DfAttnConvBase], side: 0, b, zeros, m);
                }
                int q = EmitDFlashHeads(step, names[DfAttnQ], h, heads, rows, names[DfAttnQNorm]);
                int k = EmitDFlashHeads(step, names[DfAttnK], h, kvHeads, rows, names[DfAttnKNorm]);
                int v = step.Reshape(step.MatMul(names[DfAttnV], h), hd, kvHeads, rows);
                var laneRings = new int[2 * m];
                for (int j = 0; j < m; j++)
                {
                    DFlashLaneRing lane = _dflashLanes[j];
                    long kRow = (long)hd * kvHeads * f;
                    // The lane's carried row (unless its ring holds it), then its block, into its ring (the block's write reads
                    // the carried row's, so it comes after); its queries over the ring.
                    int laneK = step.View3(k, hd, kvHeads, b, hd * f, kRow, j * b * kRow), laneV = step.View3(v, hd, kvHeads, b, hd * f, kRow, j * b * kRow);
                    int ringK, ringV, c = carryOf[j];
                    if (c >= 0)
                    {
                        EmitDFlashRingWrite(step, il, step.View3(carryK, hd, kvHeads, 1, hd * f, kRow, c * kRow),
                            step.View3(carryV, hd, kvHeads, 1, hd * f, kRow, c * kRow), 1, carryRows[j], out ringK, out ringV,
                            lane.K[il], lane.V[il]);
                        EmitDFlashRingRowsAfter(step, ringK, ringV, laneK, laneV, b, blockRows[j], out ringK, out ringV);
                    }
                    else
                    {
                        EmitDFlashRingWrite(step, il, laneK, laneV, b, blockRows[j], out ringK, out ringV, lane.K[il], lane.V[il]);
                    }
                    laneRings[2 * j] = ringK;
                    laneRings[2 * j + 1] = ringV;
                }
                // Each lane's queries over its ring, the lanes' attentions one region an executor may run in one launch
                // (GraphFusion.RingAttentionLanes); the rings' writes come before it.
                long qRow = (long)hd * heads * f;
                var attendSrcs = new int[4 * m];
                for (int j = 0; j < m; j++)
                {
                    attendSrcs[4 * j] = step.Permute(step.View3(q, hd, heads, b, hd * f, qRow, j * b * qRow), 0, 2, 1, 3);
                    attendSrcs[4 * j + 1] = laneRings[2 * j];
                    attendSrcs[4 * j + 2] = laneRings[2 * j + 1];
                    attendSrcs[4 * j + 3] = masks[j];
                }
                bool attendRegion = m > 1;
                if (attendRegion) step.BeginFused(GraphFusion.RingAttentionLanes, attendSrcs, stackalloc long[] { m }, stackalloc[] { scale });
                fa = -1;
                for (int j = 0; j < m; j++)
                {
                    int attended = step.FlashAttention(attendSrcs[4 * j], attendSrcs[4 * j + 1], attendSrcs[4 * j + 2], attendSrcs[4 * j + 3], scale);   // [hd, heads, b]
                    fa = fa < 0 ? attended : step.Op(GraphOp.Concat, stackalloc[] { fa, attended }, stackalloc long[] { 2 });
                }
                if (attendRegion) step.EndFused(stackalloc[] { fa });
                int attn = step.MatMul(names[DfAttnOutput], step.Reshape(fa, (long)heads * hd, rows));
                if (cfg.HasConv) attn = EmitDFlashConv(step, attn, attnTaps, names[DfAttnConvBase], side: 1, b, zeros, m);
                x = step.Op(GraphOp.Add, attn, x);

                int fh = step.RmsNorm(x, step.Weight(names[DfFfnNorm]), cfg.Eps);
                int ffnTaps = -1;
                if (cfg.HasConv)
                {
                    ffnTaps = step.MatMul(names[DfFfnConvProj], fh);
                    fh = EmitDFlashConv(step, fh, ffnTaps, names[DfFfnConvBase], side: 0, b, zeros, m);
                }
                fh = step.MatMul(names[DfFfnDown], step.Op(GraphOp.SwigluSplit, step.MatMul(names[DfFfnGate], fh), step.MatMul(names[DfFfnUp], fh)));
                if (cfg.HasConv) fh = EmitDFlashConv(step, fh, ffnTaps, names[DfFfnConvBase], side: 1, b, zeros, m);
                x = step.Op(GraphOp.Add, fh, x);
            }
            int cur = step.RmsNorm(x, step.Weight(prefix + "output_norm.weight"), cfg.Eps);
            // Rows 1 .. b - 1 of each lane (row 0 is the anchor's own prediction, which neither head reads), position-major.
            var pick = new int[gamma * m];
            for (int e = 0; e < gamma; e++)
                for (int j = 0; j < m; j++) pick[e * m + j] = j * b + 1 + e;
            int headRows = step.Op(GraphOp.GetRows, cur, step.ConstantInputI32(pick, pick.Length));                    // [hidden, gamma * m]
            int[] draft;
            if (cfg.HasSelector)
            {
                int anchorsAt = step.InputI32((int[])_dflashLaneAnchors.Clone(), m);
                step.RefillOnAdvance(anchorsAt, () =>
                {
                    step.RewriteInput(anchorsAt, _dflashLaneAnchors);
                    return true;
                });
                draft = EmitDFlashSelector(step, headRows, gamma, m, anchorsAt);
            }
            else
            {
                draft = EmitDFlashTop1(step, headRows, gamma * m);
            }
            return new StepOutputs { Logits = -1, Final = -1, Hidden = -1, Draft = draft };
        }

        /// <summary>K and V [head_dim, kv_heads, n] into the rings <paramref name="ringK"/> / <paramref name="ringV"/> (the result of
        /// an earlier write into them) at <paramref name="ringRows"/>: the written rings, after both writes.</summary>
        private void EmitDFlashRingRowsAfter(GraphStep step, int ringK, int ringV, int k, int v, int n, int ringRows, out int writtenK, out int writtenV)
        {
            DFlashConfig cfg = DFlashSettings;
            int hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads;
            int kRows = n == 1 ? step.Reshape(k, hd, 1, kvHeads) : step.Cont(step.Permute(k, 0, 2, 1, 3));
            int vRows = n == 1 ? step.Reshape(v, hd, 1, kvHeads) : step.Cont(step.Permute(v, 0, 2, 1, 3));
            writtenK = step.Op(GraphOp.SetRows, ringK, kRows, ringRows);
            writtenV = step.Op(GraphOp.SetRows, ringV, vRows, ringRows);
            step.Effect(writtenK);
            step.Effect(writtenV);
        }

        // ---------------------------------------------------------------- the commits' catch-ups in one program

        /// <summary>The rows of the inject program being run (each lane's padded rows' positions); a kept step refills them.</summary>
        private int[] _dflashInjectLanePositions;

        /// <summary>
        /// Several rings' queued rows in one program (ModelBase.DFlash.Lanes.cs: a batched step's commits): lane j's rows at
        /// their positions into its ring, PASS A+B over every lane's rows at once. Lanes are padded to a block's rows (a
        /// commit's catch-up holds at most a block: the accepted drafts and their anchor) by repeating their last row (the same
        /// features at the same position into the same slot: the bytes the slot takes anyway), so every step's program is the
        /// kept one of its lane count advanced, and records once: padded to the most rows any lane held, each of the eight
        /// widths kept a step and a recording of its own.
        /// </summary>
        private protected override unsafe bool TryInjectDFlashLanes(DFlashLaneRows[] lanes)
        {
            DFlashConfig cfg = DFlashSettings;
            int m = lanes.Length, width = cfg.BlockSize;
            if (m < 2 || BatchedBlockDraftRefusal() != null) return false;
            foreach (DFlashLaneRows lane in lanes)
            {
                if (lane.Count < 1) return false;
                width = Math.Max(width, lane.Count);
            }
            int feat = cfg.FeatureSize;
            var slots = new Slot[m];
            var first = new int[m];
            var counts = new int[m];
            var starts = new int[m];
            var positions = new int[m * width];
            var features = new Tensor(_allocator, DType.Float32, m * width, feat);
            float* dst = GetFloatPtrForOverwrite(features);
            long rowBytes = (long)feat * sizeof(float);
            for (int j = 0; j < m; j++)
            {
                DFlashLaneRows lane = lanes[j];
                slots[j] = DFlashRingSlotOf(lane.K, lane.V);
                first[j] = j * width;
                counts[j] = width;
                starts[j] = lane.Start;
                fixed (float* src = lane.Rows)
                    for (int i = 0; i < width; i++)
                    {
                        int row = Math.Min(i, lane.Count - 1);
                        positions[j * width + i] = lane.Start + row;
                        Buffer.MemoryCopy(src + (long)row * feat, dst + (long)(j * width + i) * feat, rowBytes, rowBytes);
                    }
            }
            InvalidateTensorDeviceCache(features);
            _dflashInjectLanePositions = positions;
            CachedGraph graph = null;
            bool owned = false;
            try
            {
                graph = RunTraced(m * width, slots, first, counts, starts, replayable: width <= SmallChunkRows, snapshots: false, attentionFloor: 0,
                    step => EmitDFlashInjectLanes(step, features), out _, out owned,
                    shape: DFlashInjectLanesShape + m, advanceInputs: new[] { features });
            }
            catch (InvalidOperationException ex) when (IsGraphAllocationFailure(ex))
            {
                FreeGraphMemory();
                return false;
            }
            catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
            {
                DeclineBatchedBlockDraft(e);
                return false;
            }
            finally
            {
                if (owned) graph?.Dispose();
                features.Dispose();
                _dflashInjectLanePositions = null;
            }
            return graph != null;   // null: a trace the lowering does not express, nothing ran
        }

        /// <summary>PASS A+B over the step's lanes: g = rmsnorm(fc · feat, enc.output_norm) for every lane's rows, then per draft
        /// layer the K (head-normed, rotated at the rows' positions) and V rows of each lane written into its ring.</summary>
        private StepOutputs EmitDFlashInjectLanes(GraphStep step, Tensor features)
        {
            DFlashConfig cfg = DFlashSettings;
            string prefix = DFlashConfig.WeightPrefix;
            int m = step.Sequences, width = step.Counts[0], rows = m * width, hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads, ringRows = DFlashRingRows;
            const long f = sizeof(float);
            // The rows' positions and ring rows (a lane's padding repeats its last row's), refilled from the rows being run.
            var positions = (int[])_dflashInjectLanePositions.Clone();
            int positionsAt = step.InputI32(positions, rows);
            step.RefillOnAdvance(positionsAt, () =>
            {
                step.RewriteInput(positionsAt, _dflashInjectLanePositions);
                return true;
            });
            var slotsAt = new int[m];
            for (int j = 0; j < m; j++)
            {
                int lane = j;
                var slots = new int[width];
                void Fill()
                {
                    for (int i = 0; i < width; i++) slots[i] = _dflashInjectLanePositions[lane * width + i] % ringRows;
                }
                Fill();
                int node = step.InputRows(slots);
                step.RefillOnAdvance(node, () =>
                {
                    Fill();
                    step.RewriteInput(node, slots);
                    return true;
                });
                slotsAt[j] = node;
            }
            int g = step.RmsNorm(step.MatMul(prefix + "fc.weight", step.HiddenInput(features, cfg.FeatureSize, rows)),
                step.Weight(prefix + "enc.output_norm.weight"), cfg.Eps);
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];
                int k = EmitDFlashHeads(step, names[DfAttnK], g, kvHeads, rows, names[DfAttnKNorm], positionsAt);
                int v = step.Reshape(step.MatMul(names[DfAttnV], g), hd, kvHeads, rows);
                long kRow = (long)hd * kvHeads * f;
                for (int j = 0; j < m; j++)
                {
                    Slot slot = step.Slots[j];
                    EmitDFlashRingWrite(step, il, step.View3(k, hd, kvHeads, width, hd * f, kRow, j * width * kRow),
                        step.View3(v, hd, kvHeads, width, hd * f, kRow, j * width * kRow), width, slotsAt[j], out _, out _,
                        slot.K[il][0], slot.V[il][0]);
                }
            }
            return new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
        }
    }
}
