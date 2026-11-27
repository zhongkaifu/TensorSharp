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
    /// A batched verify keeps every sequence's DFlash ring in step itself (<see cref="BatchedSpecVerifyKeepsDrafter"/>): the
    /// verify's own program takes the drafter's target layers of every window row and writes each row's K/V into its
    /// sequence's ring at its position (the follow a prompt's steps run, <see cref="EmitDFlashFollow"/>, over the windows),
    /// and the commit bookkeeps each sequence's kept rows as written, the next draft's carry row among them
    /// (DFlashRing.HeldTo). The rows past a sequence's kept ones sit in ring slots the next block's mask closes (their
    /// positions are past its anchor) until the next verify writes them again.
    ///
    /// The verify had returned every row's five target-layer rows to the host (25,600 floats a row of Qwen3.8-27B's DFlash2),
    /// the speculative contexts kept each sequence's kept rows and its next carry, and the catch-up program and the next block
    /// uploaded them again: four requests' steps on direct CUDA spent about 1.2 ms of host time a step on the pageable copies
    /// (four 655 KB reads, one 3.3 MB write) with the device idle for most of it, and 0.8 ms of device time on the catch-up
    /// program (RTX PRO 4500), where the direct engine's verify kept its drafter's inputs on the device.
    ///
    /// Where a verify cannot follow (a sequence without a ring of its own, an executor that refuses the follow's ops), it
    /// returns the rows' features to the model instead, and the commit writes each sequence's kept rows from them: the rings
    /// end the step the same, and the speculative contexts never hold the rows.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Tests: a batched verify returns its rows' drafter features to the caller, as before it kept the rings.</summary>
        internal bool VerifyFollowDisabledForTests { get; set; }

        /// <summary>Tests: a batched verify that would follow returns its features to the model instead (the commit writes the
        /// kept rows from them).</summary>
        internal bool VerifyFollowWritesFromFeaturesForTests { get; set; }

        /// <summary>Batched verifies whose program wrote their windows into the rings (tests check the follow ran).</summary>
        internal int VerifyFollows { get; private set; }

        /// <summary>Batched verifies whose commit wrote the kept rows from the features the verify returned to the model.</summary>
        internal int VerifyFeatureCommits { get; private set; }

        /// <summary>Whether a batched verify keeps every sequence's DFlash ring in step itself: wherever the drafter's graph
        /// programs draft several requests' blocks (<see cref="BatchedBlockDraftRefusal"/>) and the trunk's own steps write the
        /// rings (<see cref="DFlashSelfCatchUp"/>).</summary>
        public bool BatchedSpecVerifyKeepsDrafter
            => HasDFlash && !VerifyFollowDisabledForTests && DFlashSelfCatchUp && BatchedBlockDraftRefusal() == null;

        /// <summary>The last batched verify that kept the rings, for its commit: its sequences, their positions and window, and
        /// either the follow (the rings hold every window row) or the rows' features the commit writes from.</summary>
        private sealed class KeptVerify
        {
            public required string[] Requests { get; init; }
            public required int[] Positions { get; init; }
            public required int Window { get; init; }
            public bool Followed { get; set; }
            public float[] Features { get; set; }
        }

        private KeptVerify _keptVerify;
        private float[] _keptVerifyFeatures;

        /// <summary>The follow of a batched verify over <paramref name="requestIds"/>' windows: every sequence's whole window into
        /// its ring. Null when some sequence has no ring of its own (more speculating requests than rings): the verify then
        /// returns its features to the model.</summary>
        private DFlashFollowPlan PlanDFlashVerifyFollow(IReadOnlyList<string> requestIds, int window)
        {
            if (VerifyFollowWritesFromFeaturesForTests) return null;
            int n = requestIds.Count;
            var requests = new string[n];
            var keep = new int[n];
            var ringK = new Tensor[n][];
            var ringV = new Tensor[n][];
            for (int s = 0; s < n; s++)
            {
                requests[s] = requestIds[s];
                (Tensor[] k, Tensor[] v, _, _) = DFlashRingOf(requests[s], mayTake: true);
                if (k == null) return null;
                for (int o = 0; o < s; o++)
                    if (ReferenceEquals(ringK[o], k)) return null;
                keep[s] = window;
                ringK[s] = k;
                ringV[s] = v;
            }
            return new DFlashFollowPlan
            {
                Requests = requests, Keep = keep, ZeroBefore = new bool[n], RingK = ringK, RingV = ringV, Carry = new Tensor[n], Windows = true,
            };
        }

        /// <summary>The features buffer a verify that does not follow returns its rows' features into (the model's own).</summary>
        private float[] KeptVerifyFeatures(int rows)
        {
            long floats = (long)rows * DFlashSettings.FeatureSize;
            if (_keptVerifyFeatures == null || _keptVerifyFeatures.LongLength < floats)
                _keptVerifyFeatures = new float[floats];
            return _keptVerifyFeatures;
        }

        /// <summary>
        /// The verify windows' rows into their sequences' rings (<see cref="DFlashFollowPlan.Windows"/>): the target-layer rows
        /// <paramref name="taps"/> ([hidden, rows] each) joined into feature rows, g = rmsnorm(fc · feat, enc.output_norm) over
        /// every row of the step at once, and per draft layer the K (head-normed, rotated at each row's position) and V rows,
        /// each sequence's written into its ring at position % ringRows. The rows' positions and ring rows are refilled as the
        /// step advances, so a kept verify step keeps its follow.
        /// </summary>
        private void EmitDFlashVerifyFollow(GraphStep step, int[] taps, DFlashFollowPlan plan)
        {
            DFlashConfig cfg = DFlashSettings;
            string prefix = DFlashConfig.WeightPrefix;
            int hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads, rows = step.Rows, n = step.Sequences;
            const long f = sizeof(float);
            int x = -1;
            foreach (int tap in taps)
                x = x < 0 ? tap : step.Op(GraphOp.Concat, stackalloc[] { x, tap }, stackalloc long[] { 0 });   // [feature, rows]
            int g = step.RmsNorm(step.MatMul(prefix + "fc.weight", x), step.Weight(prefix + "enc.output_norm.weight"), cfg.Eps);
            var positions = new int[rows];
            void Fill()
            {
                for (int s = 0; s < n; s++)
                    for (int i = 0; i < step.Counts[s]; i++) positions[step.First[s] + i] = step.Starts[s] + i;
            }
            Fill();
            int positionsNode = step.InputI32(positions, rows);
            step.RefillOnAdvance(positionsNode, () =>
            {
                Fill();
                step.RewriteInput(positionsNode, positions);
                return true;
            });
            var ringRows = new int[n];
            for (int s = 0; s < n; s++)
                ringRows[s] = plan.Keep[s] > 0 ? DFlashRingRowsInput(step, step.Counts[s], s) : -1;
            long kRow = (long)hd * kvHeads * f;
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];
                int k = EmitDFlashHeads(step, names[DfAttnK], names[DfAttnKBias], g, kvHeads, rows, names[DfAttnKNorm], positionsNode);
                int v = step.Reshape(DFlashProjection(step, names[DfAttnV], names[DfAttnVBias], g), hd, kvHeads, rows);
                for (int s = 0; s < n; s++)
                {
                    if (plan.Keep[s] <= 0) continue;
                    int count = step.Counts[s];
                    EmitDFlashRingWrite(step, il, step.View3(k, hd, kvHeads, count, hd * f, kRow, step.First[s] * kRow),
                        step.View3(v, hd, kvHeads, count, hd * f, kRow, step.First[s] * kRow), count, ringRows[s], out _, out _,
                        plan.RingK[s][il], plan.RingV[s][il]);
                }
            }
        }

        /// <summary>The commit of a verify that kept the rings (<see cref="_keptVerify"/>): each sequence's kept rows bookkept
        /// as written into its ring (the follow wrote them), or written from the features the verify returned to the model;
        /// either way the ring then holds the next draft's carry row (DFlashRing.HeldTo).</summary>
        private void CommitKeptVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            KeptVerify held = _keptVerify;
            _keptVerify = null;
            if (held == null || held.Requests.Length != requestIds.Count) return;
            for (int s = 0; s < requestIds.Count; s++)
                if (!string.Equals(held.Requests[s], requestIds[s], StringComparison.Ordinal) || held.Positions[s] != positions[s])
                    return;
            for (int s = 0; s < requestIds.Count; s++)
            {
                if (held.Followed)
                {
                    NoteDFlashRingWriteOf(requestIds[s], positions[s], kept[s], holdsCarry: true);
                    continue;
                }
                if (held.Features == null) continue;
                WriteKeptRowsFromFeatures(requestIds[s], held.Features, (long)s * held.Window, kept[s], positions[s]);
            }
            if (held.Followed) VerifyFollows++;
            else if (held.Features != null) VerifyFeatureCommits++;
        }
    }
}
