// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next on the direct CUDA engine with a layer split: two groups' batched decode steps in flight, one on each
// of the engine's decode lanes (Q4eCudaEngine.DecodeLanes.cs), so the first GPU runs one group's layers while the last GPU
// runs the other's. Each row's logits go to its slot's other array of the pair TryForwardBatchedFusedDecode uses in turn:
// the array its sequence sampled from this step stays as it is, and the new one is filled when the step finishes.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public sealed partial class Qwen4ExpCudaModel : IPipelinedBatchedDecode
    {
        /// <summary>Two on a layer split with captured batched steps, none on one GPU. With a draft head too when
        /// <see cref="PipelinedDecodeOutrunsSpeculation"/>: a lone request speculates, two or more decode plainly as groups.</summary>
        public int PipelinedDecodeGroups
            => _exec?.Engine?.SupportsDecodeLanes == true && (!HasDraftHead || s_pipelineOverSpeculation) ? 2 : 0;

        /// <summary>
        /// With the MTP head on a layer split, two or more requests decode faster as two plain groups in flight than
        /// through the batched verify, which runs one step at a time: two H100 NVL, Q8_0, greedy, 232 / 373 / 552 / 754
        /// tok/s at 2 / 4 / 8 / 16 streams as groups against 185 / 243 with the batched verify at 2 / 4 (level with the
        /// plain batched step from 6). A request alone keeps its drafts. TS_Q4E_PIPELINE_OVER_SPEC=0 puts the batched
        /// verify first again (and a draft head then turns the groups off, as before).
        /// </summary>
        public bool PipelinedDecodeOutrunsSpeculation => HasDraftHead && s_pipelineOverSpeculation;

        private static readonly bool s_pipelineOverSpeculation =
            Environment.GetEnvironmentVariable("TS_Q4E_PIPELINE_OVER_SPEC") != "0";

        public bool HasPendingPipelinedDecode => _exec?.Engine?.HasPendingDecodeLanes == true;

        public bool TryIssuePipelinedDecode(int group, IReadOnlyList<string> requestIds, int[] tokens, int[] positions,
            float[][] outLogits)
        {
            lock (_sync)
            {
                var engine = _exec?.Engine;
                if (engine == null || _slotByRequest == null || group < 0 || group > 1)
                    return false;
                int n = requestIds.Count;
                if (n < 1 || n > TensorSharp.Cuda.Q4eCudaEngine.MaxBatchedDecodeRows)
                    return false;
                var slots = new int[n];
                for (int i = 0; i < n; i++)
                    if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                        return false;
                int vocab = Config.VocabSize;
                int maxSlot = 0;
                for (int i = 0; i < n; i++)
                    maxSlot = Math.Max(maxSlot, slots[i]);
                if (_slotLogits == null || _slotLogits.Length <= maxSlot)
                {
                    Array.Resize(ref _slotLogits, maxSlot + 1);
                    Array.Resize(ref _slotLogitsTurn, maxSlot + 1);
                }
                var rows = new float[n][];
                for (int i = 0; i < n; i++)
                {
                    int slot = slots[i];
                    var pair = _slotLogits[slot] ??= new float[2][];
                    int turn = _slotLogitsTurn[slot] ^ 1;
                    rows[i] = pair[turn] != null && pair[turn].Length == vocab ? pair[turn] : (pair[turn] = new float[vocab]);
                }
                if (engine.IssueBatchedDecodeLane(group, slots, tokens, positions, rows) == 0)
                    return false;
                for (int i = 0; i < n; i++)
                {
                    _slotLogitsTurn[slots[i]] ^= 1;
                    outLogits[i] = rows[i];
                }
                return true;
            }
        }

        public void FinishPipelinedDecode(int group)
        {
            lock (_sync)
                _exec?.Engine?.FinishBatchedDecodeLane(group);
        }

        public void DrainPipelinedDecode()
        {
            lock (_sync)
                _exec?.Engine?.DrainDecodeLanes();
        }
    }
}
