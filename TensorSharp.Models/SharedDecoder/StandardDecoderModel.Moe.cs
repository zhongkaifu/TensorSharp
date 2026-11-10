// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Routed-expert (MoE) FFN layers for every shared-runtime family: a layer whose checkpoint
    /// carries <c>ffn_gate_inp</c> + stacked <c>ffn_{gate,up,down}_exps</c> (optionally a shared
    /// expert <c>ffn_*_shexp</c>) is served here, with expert offload and parallelism owned by
    /// the runtime:
    /// <list type="bullet">
    /// <item><b>Expert offload</b> (<c>--n-cpu-moe N</c> / <c>--cpu-moe</c>): the selected layers'
    /// routed experts are never made device-resident and run on the host - through ggml's CPU
    /// backend on the GGML backends, through the managed quantized kernels elsewhere - while
    /// the router, the shared expert, attention and norms stay on the device. Same meaning on
    /// every backend.</item>
    /// <item><b>Tensor parallelism</b>: expert parallelism - each rank owns a contiguous range of
    /// whole experts and computes their contribution for every token; the router is replicated
    /// and the shared expert runs on rank 0. The FFN branch's reduction sums the partials.</item>
    /// <item><b>Layer split / hybrid</b>: the layer's experts live with its stage.</item>
    /// </list>
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        private bool[] _moeLayer;
        private int _numExperts;
        private int _expertsUsed;
        private bool _normalizeExpertWeights;
        private bool _anyMoe;

        /// <summary>Whether the selected experts' softmax weights are renormalized to sum to 1
        /// (Mixtral / Qwen-MoE <c>norm_topk_prob</c>). Families may override; the default reads
        /// <c>{arch}.expert_weights_norm</c> and falls back to true.</summary>
        protected virtual bool NormalizeExpertWeights => _normalizeExpertWeights;

        /// <summary>Whether any layer this process owns has routed experts.</summary>
        protected bool HasRoutedExpertLayers => _anyMoe;

        /// <summary>Routing decided for one batch of tokens: K experts and weights per token.</summary>
        private sealed class Routing
        {
            public required int[] Experts { get; init; }   // [tokens * K]
            public required float[] Weights { get; init; } // [tokens * K]
            public required int K { get; init; }
        }

        private void DetectMoeLayers()
        {
            _moeLayer = new bool[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _moeLayer[l] = OwnsLayer(l) && HasWeight($"blk.{l}.ffn_gate_inp.weight");
                _anyMoe |= _moeLayer[l];
            }
            if (!_anyMoe) return;
            string arch = MetadataArchitecture ?? Config.Architecture;
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _expertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _normalizeExpertWeights = _gguf.GetBool($"{arch}.expert_weights_norm", true);
            if (_numExperts <= 0 || _expertsUsed <= 0 || _expertsUsed > _numExperts)
                throw new InvalidOperationException(
                    $"{Config.Architecture}: routed-expert layers need {arch}.expert_count and {arch}.expert_used_count " +
                    $"(got {_numExperts} and {_expertsUsed}).");
            int offloaded = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_moeLayer[l] && MoeCpuOffloadConfig.IsLayerOnCpu(l)) offloaded++;
            Console.WriteLine($"  Routed experts: {_numExperts} per layer, top-{_expertsUsed}" +
                (offloaded > 0 ? $"; {offloaded} layer(s) keep their experts in host memory and run them there" : "") + ".");
        }

        private bool IsExpertMember(string name) => _stackedExpertMemberNames.Contains(name);

        /// <summary>Offloaded experts are never uploaded; under tensor parallelism each expert is
        /// made resident by the rank that owns it (on first use), not copied to rank 0; on a
        /// single GGML device the stacked-expert kernel serves them, so the per-expert views are
        /// not uploaded a second time.</summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
        {
            if (!base.ShouldPreloadCudaQuantWeightToDevice(weightName)) return false;
            if (!IsExpertMember(weightName)) return true;
            return _mesh != null && _mesh.TensorParallel == 1 && !IsGgmlBackend;
        }

        protected override bool ShouldPreloadMlxQuantWeightToDevice(string weightName, QuantizedWeight weight)
            => !MoeCpuOffloadConfig.IsOffloadedExpertWeightName(weightName) && base.ShouldPreloadMlxQuantWeightToDevice(weightName, weight);

        private static string ExpertName(int layer, string kind, int expert) => $"blk.{layer}.ffn_{kind}_exps.{expert}.weight";

        // ---------------------------------------------------------------- routing

        /// <summary>Router logits on <paramref name="scope"/>'s device, softmax and top-K on the host.</summary>
        private Routing Route(RankScope scope, Tensor input, int layer)
        {
            int tokens = (int)input.Sizes[0];
            // The router is replicated (never sharded): run it on the scope's device.
            Tensor logits = ReplicatedLinear(scope, input, $"blk.{layer}.ffn_gate_inp.weight");
            float[] values = logits.GetElementsAsFloat(tokens * _numExperts);
            logits.Dispose();

            int k = _expertsUsed;
            var experts = new int[tokens * k];
            var weights = new float[tokens * k];
            var probs = new float[_numExperts];
            for (int t = 0; t < tokens; t++)
            {
                int row = t * _numExperts;
                float max = float.NegativeInfinity;
                for (int e = 0; e < _numExperts; e++) max = MathF.Max(max, values[row + e]);
                float sum = 0;
                for (int e = 0; e < _numExperts; e++) { probs[e] = MathF.Exp(values[row + e] - max); sum += probs[e]; }
                for (int e = 0; e < _numExperts; e++) probs[e] /= sum;
                float picked = 0;
                for (int j = 0; j < k; j++)
                {
                    int best = -1;
                    for (int e = 0; e < _numExperts; e++)
                    {
                        if (probs[e] < 0) continue;
                        if (best < 0 || probs[e] > probs[best]) best = e;
                    }
                    experts[t * k + j] = best;
                    weights[t * k + j] = probs[best];
                    picked += probs[best];
                    probs[best] = -1f;   // taken
                }
                if (NormalizeExpertWeights && picked > 0)
                    for (int j = 0; j < k; j++) weights[t * k + j] /= picked;
            }
            return new Routing { Experts = experts, Weights = weights, K = k };
        }

        // ---------------------------------------------------------------- the FFN branch

        /// <summary>
        /// MoE FFN branch body for one rank: the rank's routed experts for every token, plus the
        /// shared expert on rank 0. <paramref name="routing"/> is computed once per layer (it is
        /// identical on every rank) and passed in under tensor parallelism; null computes it here.
        /// </summary>
        private Tensor MoeFfn(RankScope scope, Tensor input, int layer, Routing routing)
        {
            routing ??= Route(scope, input, layer);
            int tokens = (int)input.Sizes[0];
            int hidden = Config.HiddenSize;
            bool onHost = MoeCpuOffloadConfig.IsLayerOnCpu(layer);

            Tensor output = null;
            if (scope.Degree == 1 && IsGgmlBackend)
                output = StackedExperts(input, layer, routing, onHost);
            if (output == null)
            {
                int first = scope.Rank * _numExperts / scope.Degree;
                int end = (scope.Rank + 1) * _numExperts / scope.Degree;
                float[] y = GroupedExperts(scope, input, layer, routing, first, end, onHost);
                output = new Tensor(scope.Allocator, DType.Float32, tokens, hidden);
                output.SetElementsAsFloat(y);
                HostWritten(output);
            }

            if (scope.Rank == 0 && HasWeight($"blk.{layer}.ffn_gate_shexp.weight"))
            {
                Tensor shared = SharedExpert(scope, input, layer);
                Ops.Add(output, output, shared);
                shared.Dispose();
            }
            return output;
        }

        /// <summary>ggml's stacked-expert kernel (mul_mat_id over the selected experts), on the
        /// device or - for an offloaded layer - on ggml's CPU backend reading the experts straight
        /// from the GGUF mapping. Null when the layer's experts are not stacked.</summary>
        private Tensor StackedExperts(Tensor input, int layer, Routing routing, bool onHost)
        {
            if (!_stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_gate_exps.weight", out var gate) ||
                !_stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_up_exps.weight", out var up) ||
                !_stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_down_exps.weight", out var down))
                return null;
            int tokens = (int)input.Sizes[0];
            var output = new Tensor(_allocator, DType.Float32, tokens, Config.HiddenSize);
            GgmlBasicOps.MoEFFNPrefill(
                input, output, tokens, Config.HiddenSize, (int)gate.PerExpertNe1,
                gate.NumExperts, routing.K, routing.Experts, routing.Weights,
                gate.Data, gate.GgmlType, gate.PerExpertNe0, gate.PerExpertNe1, gate.TotalRawBytes,
                up.Data, up.GgmlType, up.PerExpertNe0, up.PerExpertNe1, up.TotalRawBytes,
                down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, down.TotalRawBytes,
                gateBias: null, upBias: null, downBias: null,
                activation: GgmlBasicOps.MoEActivation.SwiGLUSplit,
                runOnCpu: onHost);
            return output;
        }

        /// <summary>
        /// Experts [<paramref name="first"/>, <paramref name="end"/>) for every token that routed
        /// to them, grouped by expert so each expert runs one matmul over its rows. On the host
        /// (offloaded layer) the managed quantized kernels read the weights in place; otherwise
        /// each expert runs on the scope's device. Returns the weighted sum, [tokens, hidden].
        /// </summary>
        private unsafe float[] GroupedExperts(RankScope scope, Tensor input, int layer, Routing routing,
            int first, int end, bool onHost)
        {
            int tokens = (int)input.Sizes[0];
            int hidden = Config.HiddenSize;
            float[] x = input.GetElementsAsFloat(tokens * hidden);
            var y = new float[tokens * hidden];

            var rowsByExpert = new List<(int Token, float Weight)>[end - first];
            for (int t = 0; t < tokens; t++)
                for (int j = 0; j < routing.K; j++)
                {
                    int e = routing.Experts[t * routing.K + j];
                    if (e < first || e >= end) continue;
                    (rowsByExpert[e - first] ??= new()).Add((t, routing.Weights[t * routing.K + j]));
                }

            for (int e = first; e < end; e++)
            {
                var rows = rowsByExpert[e - first];
                if (rows == null) continue;
                QuantizedWeight gate = _quantWeights[ExpertName(layer, "gate", e)];
                QuantizedWeight up = _quantWeights[ExpertName(layer, "up", e)];
                QuantizedWeight down = _quantWeights[ExpertName(layer, "down", e)];
                int ff = (int)gate.Ne1;
                var xe = new float[rows.Count * hidden];
                for (int i = 0; i < rows.Count; i++)
                    Array.Copy(x, (long)rows[i].Token * hidden, xe, (long)i * hidden, hidden);

                float[] de = onHost
                    ? HostExpert(xe, rows.Count, hidden, ff, gate, up, down)
                    : DeviceExpert(scope, xe, rows.Count, hidden, gate, up, down);
                for (int i = 0; i < rows.Count; i++)
                {
                    float w = rows[i].Weight;
                    long dst = (long)rows[i].Token * hidden, src = (long)i * hidden;
                    for (int c = 0; c < hidden; c++) y[dst + c] += w * de[src + c];
                }
            }
            return y;
        }

        /// <summary>One expert on the host: the managed quantized kernels, whatever the backend.</summary>
        private static unsafe float[] HostExpert(float[] x, int rows, int hidden, int ff,
            QuantizedWeight gate, QuantizedWeight up, QuantizedWeight down)
        {
            var g = new float[rows * ff];
            var u = new float[rows * ff];
            var d = new float[rows * hidden];
            fixed (float* px = x, pg = g, pu = u, pd = d)
            {
                ManagedQuantizedOps.AddmmQuantizedToFloat32(gate.GgmlType, gate.Data, gate.Ne0, gate.Ne1, px, hidden, rows, pg, ff);
                ManagedQuantizedOps.AddmmQuantizedToFloat32(up.GgmlType, up.Data, up.Ne0, up.Ne1, px, hidden, rows, pu, ff);
                for (int i = 0; i < g.Length; i++)
                    g[i] = g[i] / (1f + MathF.Exp(-g[i])) * u[i];
                ManagedQuantizedOps.AddmmQuantizedToFloat32(down.GgmlType, down.Data, down.Ne0, down.Ne1, pg, ff, rows, pd, hidden);
            }
            return d;
        }

        /// <summary>One expert on the scope's device.</summary>
        private float[] DeviceExpert(RankScope scope, float[] x, int rows, int hidden,
            QuantizedWeight gate, QuantizedWeight up, QuantizedWeight down)
        {
            using var xt = new Tensor(scope.Allocator, DType.Float32, rows, hidden);
            xt.SetElementsAsFloat(x);
            HostWritten(xt);
            using var g = new Tensor(scope.Allocator, DType.Float32, rows, (int)gate.Ne1);
            using var u = new Tensor(scope.Allocator, DType.Float32, rows, (int)up.Ne1);
            AddmmQuantManaged(g, xt, gate);
            AddmmQuantManaged(u, xt, up);
            Ops.SiLUMul(g, g, u);
            using var d = new Tensor(scope.Allocator, DType.Float32, rows, (int)down.Ne1);
            AddmmQuantManaged(d, g, down);
            return d.GetElementsAsFloat(rows * hidden);
        }

        /// <summary>The always-active shared expert: a dense SwiGLU FFN on the scope's device,
        /// scaled per token by <c>sigmoid(x . ffn_gate_inp_shexp)</c> when the layer has that
        /// gate (Qwen 3.5 / Qwen3-Next MoE).</summary>
        private Tensor SharedExpert(RankScope scope, Tensor input, int layer)
        {
            string p = $"blk.{layer}.";
            Tensor gate = ReplicatedLinear(scope, input, p + "ffn_gate_shexp.weight");
            Tensor up = ReplicatedLinear(scope, input, p + "ffn_up_shexp.weight");
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = ReplicatedLinear(scope, gate, p + "ffn_down_shexp.weight");
            gate.Dispose();
            if (_weights.TryGetValue(p + "ffn_gate_inp_shexp.weight", out Tensor gateVector))
                ScaleRowsBySigmoidGate(down, input, gateVector);
            return down;
        }

        /// <summary><paramref name="rows"/>[t] *= sigmoid(<paramref name="input"/>[t] . <paramref name="gateVector"/>).</summary>
        private void ScaleRowsBySigmoidGate(Tensor rows, Tensor input, Tensor gateVector)
        {
            int tokens = (int)input.Sizes[0];
            int width = (int)input.Sizes[1];
            int outWidth = (int)rows.Sizes[1];
            float[] x = input.GetElementsAsFloat(tokens * width);
            float[] w = gateVector.GetElementsAsFloat((int)Math.Min(gateVector.ElementCount(), width));
            float[] y = rows.GetElementsAsFloat(tokens * outWidth);
            for (int t = 0; t < tokens; t++)
            {
                float dot = 0;
                for (int i = 0; i < w.Length; i++) dot += x[(long)t * width + i] * w[i];
                float g = 1f / (1f + MathF.Exp(-dot));
                for (int c = 0; c < outWidth; c++) y[(long)t * outWidth + c] *= g;
            }
            rows.SetElementsAsFloat(y);
            HostWritten(rows);
        }

        /// <summary>An unsharded projection on the scope's device (LinearForward on one device).</summary>
        private Tensor ReplicatedLinear(RankScope scope, Tensor input, string name)
        {
            if (scope.Degree == 1)
                return LinearForward(input, name) ?? throw new InvalidOperationException($"{Config.Architecture}: '{name}' is missing.");
            int rows = (int)input.Sizes[0];
            if (_quantWeights.TryGetValue(name, out QuantizedWeight q))
            {
                var result = new Tensor(scope.Allocator, DType.Float32, rows, (int)q.Ne1);
                AddmmQuantManaged(result, input, q);
                return result;
            }
            Tensor w = Replica(_weights[name], scope.Stage, scope.Rank);
            using Tensor wT = w.Transpose();
            var dense = new Tensor(scope.Allocator, DType.Float32, rows, (int)w.Sizes[0]);
            Ops.Addmm(dense, 0, dense, 1.0f, input, wT);
            return dense;
        }
    }
}
