// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// glm-dsa MoE FFN — the DeepSeek-V3 routing rule:
//
//   probs      = sigmoid(gate_inp @ x)                     [n_expert]
//   selected   = top_k(probs + exp_probs_b, n_expert_used) (bias steers SELECTION only)
//   weights    = probs[selected]
//   weights   /= max(sum(weights), 6.103515625e-5)         (expert_weights_norm)
//   weights   *= expert_weights_scale                      (2.5 for GLM-5.2)
//   out        = sum_k weights[k] * down_k(silu(gate_k(x)) * up_k(x))
//              + down_sh(silu(gate_sh(x)) * up_sh(x))      (always-on shared expert)
//
// Mirrors llm_graph_context::build_moe_ffn (llama.cpp src/llama-graph.cpp),
// including the F16-denormal clamp on the weight sum.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>llama.cpp clamps the routed-weight sum to the smallest normal F16 before dividing.</summary>
        private const float RouteWeightSumFloor = 6.103515625e-5f;

        private Tensor MoEBlock(Tensor input, int layer, string[] wn, int seqLen)
        {
            Tensor routed = RoutedExperts(input, layer, wn, seqLen);
            Tensor shared = SharedExpert(input, wn, seqLen);
            if (shared != null)
            {
                Ops.Add(routed, routed, shared);
                shared.Dispose();
            }
            return routed;
        }

        private Tensor SharedExpert(Tensor input, string[] wn, int seqLen)
        {
            Tensor gate = LinearForward(input, wn[WN_SH_GATE]);
            if (gate == null)
                return null;
            Tensor up = LinearForward(input, wn[WN_SH_UP]);
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, wn[WN_SH_DOWN]);
            gate.Dispose();
            return down;
        }

        /// <summary>
        /// Router logits -> per-token expert selection and weights, filling
        /// <paramref name="selected"/> and <paramref name="weights"/> with
        /// seqLen * n_expert_used entries each (token-major).
        /// </summary>
        private unsafe void RouteTokens(Tensor input, int layer, string[] wn, int seqLen,
            int[] selected, float[] weights)
        {
            Tensor logits = LinearForward(input, wn[WN_GATE_INP]);   // [T, n_expert]
            InvalidateTensorDeviceCache(logits);
            float* logitPtr = GetFloatPtr(logits);

            float[] bias = null;
            if (_weights.TryGetValue(wn[WN_EXP_PROBS_B], out var biasTensor))
                bias = TensorToFloatArray(biasTensor);

            int nExp = _numExperts;
            int nUsed = _numExpertsUsed;
            var probs = new float[nExp];
            var selection = new float[nExp];

            for (int t = 0; t < seqLen; t++)
            {
                float* row = logitPtr + (long)t * nExp;

                if (_expertGatingFunc == 2)
                {
                    for (int e = 0; e < nExp; e++)
                        probs[e] = TensorComputePrimitives.Sigmoid(row[e]);
                }
                else
                {
                    float max = float.NegativeInfinity;
                    for (int e = 0; e < nExp; e++) if (row[e] > max) max = row[e];
                    float sum = 0f;
                    for (int e = 0; e < nExp; e++) { probs[e] = MathF.Exp(row[e] - max); sum += probs[e]; }
                    float inv = 1.0f / sum;
                    for (int e = 0; e < nExp; e++) probs[e] *= inv;
                }

                if (bias != null)
                {
                    for (int e = 0; e < nExp; e++) selection[e] = probs[e] + bias[e];
                }
                else
                {
                    Array.Copy(probs, selection, nExp);
                }

                int outBase = t * nUsed;
                TensorComputePrimitives.SelectTopKInPlace(selection, nExp, nUsed, _selectedExperts);

                float wsum = 0f;
                for (int k = 0; k < nUsed; k++)
                {
                    int e = _selectedExperts[k];
                    selected[outBase + k] = e;
                    float w = probs[e];
                    weights[outBase + k] = w;
                    wsum += w;
                }

                if (_expertWeightsNorm)
                {
                    float denom = 1.0f / MathF.Max(wsum, RouteWeightSumFloor);
                    for (int k = 0; k < nUsed; k++)
                        weights[outBase + k] *= denom;
                }
                if (_expertWeightsScale != 0.0f && _expertWeightsScale != 1.0f)
                {
                    for (int k = 0; k < nUsed; k++)
                        weights[outBase + k] *= _expertWeightsScale;
                }
            }

            logits.Dispose();
        }

        private Tensor RoutedExperts(Tensor input, int layer, string[] wn, int seqLen)
        {
            int hidden = Config.HiddenSize;
            int nUsed = _numExpertsUsed;

            int need = seqLen * nUsed;
            if (_selectedExpertsBatch == null || _selectedExpertsBatch.Length < need)
            {
                _selectedExpertsBatch = new int[need];
                _routingWeightsBatch = new float[need];
            }

            long tRoute = Stopwatch.GetTimestamp();
            RouteTokens(input, layer, wn, seqLen, _selectedExpertsBatch, _routingWeightsBatch);
            _linearTicks += Stopwatch.GetTimestamp() - tRoute;

            var output = new Tensor(_allocator, DType.Float32, seqLen, hidden);

            if (TryRoutedExpertsGgml(input, output, layer, seqLen, hidden))
                return output;

            RoutedExpertsGrouped(input, output, layer, seqLen, hidden);
            return output;
        }

        private int[] _selectedExpertsBatch;
        private float[] _routingWeightsBatch;

        /// <summary>
        /// One ggml graph for the whole routed-expert FFN: <c>mul_mat_id</c> over the
        /// stacked per-layer expert blocks, exactly the shape llama.cpp builds. Also
        /// carries the MoE CPU-offload switch — an offloaded layer runs this same
        /// graph on the ggml CPU backend, reading the experts zero-copy out of the
        /// GGUF mmap instead of VRAM.
        /// </summary>
        private bool TryRoutedExpertsGgml(Tensor input, Tensor output, int layer, int seqLen, int hidden)
        {
            if (!IsGgmlBackend)
                return false;
            var gate = _stackedGate[layer];
            var up = _stackedUp[layer];
            var down = _stackedDown[layer];
            if (gate == null || up == null || down == null)
                return false;

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.MoEFFNPrefillSwiGLU(
                    input, output,
                    seqLen, hidden, _expertFfnLength, _numExperts, _numExpertsUsed,
                    _selectedExpertsBatch, _routingWeightsBatch,
                    gate.Data, gate.GgmlType, gate.PerExpertNe0, gate.PerExpertNe1, gate.TotalRawBytes,
                    up.Data, up.GgmlType, up.PerExpertNe0, up.PerExpertNe1, up.TotalRawBytes,
                    down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, down.TotalRawBytes,
                    null, null, null,
                    useSwiGLUOAI: false,
                    runOnCpu: MoeOffload.IsLayerOnCpu(layer));
                InvalidateTensorDeviceCache(output);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
            catch (InvalidOperationException ex)
            {
                Console.Error.WriteLine($"[glm-dsa] fused MoE declined on layer {layer} ({ex.Message}); using the grouped path.");
                return false;
            }
        }

        /// <summary>
        /// Backend-neutral routed-expert FFN: bucket the batch by expert, run one
        /// gate/up/down triple per touched expert, and scatter the weighted results
        /// back. One matmul per touched expert instead of one per (token, expert)
        /// pair — the difference between 3*256 and 3*8*n_tokens dispatches on a
        /// 256-expert layer.
        /// </summary>
        private unsafe void RoutedExpertsGrouped(Tensor input, Tensor output, int layer, int seqLen, int hidden)
        {
            int nUsed = _numExpertsUsed;
            int nFf = _expertFfnLength;

            float* outPtr = GetFloatPtr(output);
            TensorComputePrimitives.Zero(outPtr, seqLen * hidden);

            // token lists per expert
            var buckets = new List<int>[_numExperts];
            for (int t = 0; t < seqLen; t++)
            {
                for (int k = 0; k < nUsed; k++)
                {
                    int e = _selectedExpertsBatch[t * nUsed + k];
                    (buckets[e] ??= new List<int>()).Add(t * nUsed + k);
                }
            }

            InvalidateTensorDeviceCache(input);
            float* inPtr = GetFloatPtr(input);

            for (int e = 0; e < _numExperts; e++)
            {
                var slots = buckets[e];
                if (slots == null || slots.Count == 0)
                    continue;

                int rows = slots.Count;
                using var gathered = new Tensor(_allocator, DType.Float32, rows, hidden);
                float* gp = GetFloatPtr(gathered);
                for (int r = 0; r < rows; r++)
                {
                    int token = slots[r] / nUsed;
                    Buffer.MemoryCopy(inPtr + (long)token * hidden, gp + (long)r * hidden,
                        (long)hidden * sizeof(float), (long)hidden * sizeof(float));
                }
                // Host-gathered rows: drop the stale device copy before the
                // expert matmuls read it.
                InvalidateTensorDeviceCache(gathered);

                using var gate = ExpertLinear(gathered, layer, e, ExpertKind.Gate, rows, nFf);
                using var up = ExpertLinear(gathered, layer, e, ExpertKind.Up, rows, nFf);
                Ops.SiLUMul(gate, gate, up);
                using var down = ExpertLinear(gate, layer, e, ExpertKind.Down, rows, hidden);

                InvalidateTensorDeviceCache(down);
                float* dp = GetFloatPtr(down);
                for (int r = 0; r < rows; r++)
                {
                    int slot = slots[r];
                    int token = slot / nUsed;
                    float w = _routingWeightsBatch[slot];
                    TensorComputePrimitives.ScaleAdd(outPtr + (long)token * hidden,
                        dp + (long)r * hidden, w, hidden);
                }
            }

            // The accumulation happened on the host, so the device copy of this
            // buffer is stale for whatever consumes `output` next.
            InvalidateTensorDeviceCache(output);
        }

        private enum ExpertKind { Gate, Up, Down }

        private Tensor ExpertLinear(Tensor input, int layer, int expert, ExpertKind kind, int rows, int outDim)
        {
            QuantizedWeight qw = kind switch
            {
                ExpertKind.Gate => _expertGateQW[layer]?[expert],
                ExpertKind.Up => _expertUpQW[layer]?[expert],
                _ => _expertDownQW[layer]?[expert],
            };

            var result = new Tensor(_allocator, DType.Float32, rows, outDim);
            if (qw != null)
            {
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                return result;
            }

            // F32 checkpoints keep the experts as one [n_expert, out, in] block
            // (ModelBase only splits QUANTIZED "_exps." tensors per expert), so slice
            // this expert's plane out of it. Select is a view, not a copy.
            string stackedName = kind switch
            {
                ExpertKind.Gate => $"blk.{layer}.ffn_gate_exps.weight",
                ExpertKind.Up => $"blk.{layer}.ffn_up_exps.weight",
                _ => $"blk.{layer}.ffn_down_exps.weight",
            };
            if (!_weights.TryGetValue(stackedName, out var stacked) || stacked.Sizes.Length != 3)
            {
                result.Dispose();
                throw new InvalidOperationException($"glm-dsa: missing expert weight {stackedName}.");
            }
            using var w = stacked.Select(0, expert);
            using var wT = w.Transpose();
            Ops.Addmm(result, 0, result, 1.0f, input, wT);
            return result;
        }
    }
}
