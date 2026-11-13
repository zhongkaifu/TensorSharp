// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Speculative decoding across the nodes of a multi-node layer split (each node a stage, or a stage of
    /// tensor-parallel GPUs). The driver runs the last stage (the default order), so the drafter, the head and the
    /// sampler stay on it: an MTP draft layer is placed with the last decoder layer, and a DFlash drafter attaches there.
    /// What crosses the nodes is the verify window: the driver mirrors it to every node (<see cref="SpecControlVerify"/>),
    /// each runs its layers over the window's rows with its recurrent states keeping the window, the last stage projects
    /// every row, and the nodes that own a DFlash drafter's target layers send those rows to the driver. The accept count
    /// (<see cref="SpecControlCommit"/>) and the cache rewind (<see cref="SpecControlRewind"/>) are mirrored the same way,
    /// so every node keeps the same accepted prefix.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>[flags, captures, capture layer ids..., tokens...]: a verify window (<see cref="RunDistributedVerify"/>).</summary>
        private const int SpecControlVerify = TpControlModelFirst + 40;

        /// <summary>[kept]: the last verify's first <c>kept</c> rows stay (0: none).</summary>
        private const int SpecControlCommit = TpControlModelFirst + 41;

        /// <summary>[length]: attention rows past <c>length</c> are dead.</summary>
        private const int SpecControlRewind = TpControlModelFirst + 42;

        private const int VerifyAllRows = 1, VerifyCaptureFinal = 2, VerifySnapshots = 4;

        private bool IsMultiNodePipeline => LayerSplitGroup?.NodeCount > 1;

        /// <summary>Why a multi-node pipeline cannot speculate, or null: the driver drafts, so it must run the last
        /// stage; and the per-op verify needs recurrent states that keep a window.</summary>
        private string DistributedSpeculationRefusal(string reason)
        {
            if (LayerPlacement is not { DriverRunsLastStage: true })
                return $"speculative decoding across nodes drafts on the driver, which must run the last stage " +
                    $"({LayerSplitPlan.DriverStageVariable}=last, the default)";
            foreach (RecurrentState state in ActiveRecurrentStates())
                if (!state.SupportsVerify)
                    return $"speculative decoding off whole-forward graphs ({reason}) needs recurrent states that keep a verify window";
            return null;
        }

        /// <summary>A drafter is attached on the node that runs the last stage (the driver) only.</summary>
        private bool DraftsOnThisNode => !IsMultiNodePipeline || LayerPlacement?.IsLast == true;

        /// <summary>The driver's verify: mirrored to every node, then run as they run it.</summary>
        private void SpecForwardDistributed(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (SpeculationRefusal is { } refusal) throw new NotSupportedException(refusal);
            int rows = tokens.Length;
            bool dflash = HasDFlash;
            int[] captureLayers = dflash && hAllOut != null ? DFlashSettings.TargetLayerIds : Array.Empty<int>();
            int flags = (allLogitsRows ? VerifyAllRows : 0)
                | (!dflash && hAllOut != null ? VerifyCaptureFinal : 0)
                | (_anyRecurrent && rows > 1 && rows <= SmallChunkRows ? VerifySnapshots : 0);
            var payload = new int[2 + captureLayers.Length + rows];
            payload[0] = flags;
            payload[1] = captureLayers.Length;
            Array.Copy(captureLayers, 0, payload, 2, captureLayers.Length);
            Array.Copy(tokens, 0, payload, 2 + captureLayers.Length, rows);
            BroadcastModelControl(SpecControlVerify, payload);
            try
            {
                RunDistributedVerify(tokens, flags, captureLayers, hAllOut, logitsOut);
            }
            catch (Exception failure)
            {
                AbortDistributedModelControl(failure);
                throw;
            }
        }

        /// <summary>
        /// One verify window through this node's layers, as every node runs it: the recurrent states keep the window
        /// (with <see cref="VerifySnapshots"/>), the last stage projects every row (or the last) and keeps the final
        /// normed rows, and the rows entering each capture layer go to the driver (a DFlash drafter's features).
        /// </summary>
        private void RunDistributedVerify(int[] tokens, int flags, int[] captureLayers, float[] hAllOut, float[] logitsOut)
        {
            SettlePendingVerify();
            int rows = tokens.Length, start = _cacheSeqLen, hiddenSize = Config.HiddenSize;
            bool allRows = (flags & VerifyAllRows) != 0, captureFinal = (flags & VerifyCaptureFinal) != 0;
            bool snapshots = (flags & VerifySnapshots) != 0;
            EnsureCacheCapacity(checked(start + rows));
            TokenPositions positions = BeginRotaryPositions(start, rows);
            var captures = new float[captureLayers.Length][];
            float[] final = null;
            if (snapshots)
                foreach (RecurrentState state in ActiveRecurrentStates()) state.BeginVerify();
            _forwardSw.Start();
            try
            {
                Tensor Embed()
                {
                    Tensor hidden = EmbedStep(tokens);
                    InjectMedia(hidden, start);
                    return hidden;
                }
                void Capture(int layer, Tensor input)
                {
                    for (int c = 0; c < captureLayers.Length; c++)
                        if (captureLayers[c] == layer) captures[c] = input.GetElementsAsFloat(rows * hiddenSize);
                }
                float[] Project(Tensor hidden)
                {
                    for (int c = 0; c < captureLayers.Length; c++)
                        if (captureLayers[c] >= Config.NumLayers) captures[c] = hidden.GetElementsAsFloat(rows * hiddenSize);
                    if (captureFinal)
                    {
                        using Tensor normed = RMSNormOp(hidden, "output_norm.weight");
                        final = normed.GetElementsAsFloat(rows * hiddenSize);
                    }
                    var which = new int[allRows ? rows : 1];
                    for (int i = 0; i < which.Length; i++) which[i] = allRows ? i : rows - 1;
                    return ProjectRows(hidden, which);
                }
                int outputFloats = checked((allRows ? rows : 1) * Config.VocabSize);
                float[] logits = _mesh.TensorParallel == 1
                    ? ForwardLayerPipeline(tokens, Embed, runLayer: (hidden, layer) =>
                    {
                        Capture(layer, hidden);
                        return LayerOnOneDevice(hidden, layer, positions, null);
                    }, Project, outputFloats)
                    : ForwardLayerPipeline(tokens, Embed, runLocalLayers: hidden => RunLayers(hidden, positions, batch: null, Capture),
                        Project, outputFloats);
                if (captureLayers.Length > 0) GatherCaptures(captureLayers, captures, rows);
                if (logitsOut != null && logits != null)
                    Array.Copy(logits, 0, logitsOut, 0, Math.Min(logits.LongLength, logitsOut.LongLength));
            }
            catch
            {
                if (snapshots)
                    foreach (RecurrentState state in ActiveRecurrentStates()) state.CommitVerify(0);
                throw;
            }
            finally
            {
                _forwardSw.Stop();
            }
            _cacheSeqLen += rows;
            _forwardCount++;
            _verifyRows = snapshots ? rows : 0;
            if (hAllOut != null) WriteSpecFeatures(hAllOut, rows, captures, final);
        }

        /// <summary>
        /// The capture rows of layers another node runs, to the driver: each other node sends a header (how many of the
        /// capture layers it ran, and which), then those layers' rows; the driver reads every other node's in turn. Sent
        /// after the node passed its activation on, so the driver reads its pipeline input first on a shared link.
        /// </summary>
        private void GatherCaptures(int[] captureLayers, float[][] captures, int rows)
        {
            int rowFloats = checked(rows * Config.HiddenSize);
            int headerFloats = 1 + captureLayers.Length;
            if (LayerPlacement.NodeId != 0)
            {
                var header = new float[headerFloats];
                int sent = 0;
                for (int c = 0; c < captureLayers.Length; c++)
                    if (captures[c] != null) header[1 + sent++] = c;
                header[0] = sent;
                LayerSplitGroup.Send(header, headerFloats, 0);
                for (int c = 0; c < captureLayers.Length; c++)
                    if (captures[c] != null) LayerSplitGroup.Send(captures[c], rowFloats, 0);
                return;
            }
            for (int node = 1; node < LayerSplitGroup.NodeCount; node++)
            {
                var header = new float[headerFloats];
                LayerSplitGroup.Receive(header, headerFloats, node);
                int count = (int)header[0];
                for (int i = 0; i < count; i++)
                {
                    int c = (int)header[1 + i];
                    captures[c] = new float[rowFloats];
                    LayerSplitGroup.Receive(captures[c], rowFloats, node);
                }
            }
            for (int c = 0; c < captureLayers.Length; c++)
                if (captures[c] == null)
                    throw new InvalidOperationException($"No node captured layer {captureLayers[c]} of the verify.");
        }

        /// <summary>The verify's accept count on every node (the driver broadcasts it first).</summary>
        private void MirrorVerifyCommit(int kept)
        {
            if (IsMultiNodePipeline) BroadcastModelControl(SpecControlCommit, new[] { kept });
        }

        /// <summary>The cache rewind on every node (the driver broadcasts it first).</summary>
        private void MirrorRewind(int length)
        {
            if (IsMultiNodePipeline) BroadcastModelControl(SpecControlRewind, new[] { length });
        }

        protected override bool HandleModelControl(int op, int[] payload)
        {
            switch (op)
            {
                case SpecControlVerify:
                {
                    int flags = payload[0], captures = payload[1];
                    int[] layers = payload.AsSpan(2, captures).ToArray();
                    int[] tokens = payload.AsSpan(2 + captures).ToArray();
                    RunDistributedVerify(tokens, flags, layers, hAllOut: null, logitsOut: null);
                    return true;
                }
                case SpecControlCommit:
                    if (_verifyRows > 1)
                    {
                        int rows = _verifyRows;
                        _verifyRows = 0;
                        CommitRecurrentPrefix(payload[0], rows);
                    }
                    return true;
                case SpecControlRewind:
                    if (payload[0] < 0 || payload[0] > _cacheSeqLen)
                        throw new InvalidOperationException($"Rewind to {payload[0]} of {_cacheSeqLen} rows.");
                    _cacheSeqLen = payload[0];
                    return true;
                default:
                    return base.HandleModelControl(op, payload);
            }
        }
    }
}
