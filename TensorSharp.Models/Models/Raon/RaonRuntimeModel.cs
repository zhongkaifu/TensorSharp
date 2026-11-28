// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using TensorSharp.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// KRAFTON Raon-1 nano (GGUF <c>raon1nano</c>) on the shared decoder runtime, as whole-forward graphs on one device.
    /// Ported from the HF reference (<c>modeling_raon1_nano.py</c>) and KRAFTON's llama.cpp graph
    /// (<c>src/models/raon1nano.cpp</c>, which defines the GGUF's folded tensor layout):
    /// <list type="bullet">
    /// <item>A 4-stream residual (Raon-mHC): every block reads a sigmoid-weighted sum of the streams and writes back
    /// through a per-token stream mix built in closed form from the layer's <c>hc_*_chart</c> (no Sinkhorn).</item>
    /// <item>18 RDA layers: the gated delta rule with a per-channel decay, a separate write gate folded into the values,
    /// short convolutions over q|k|v, a per-head RMS norm and a sigmoid output gate.</item>
    /// <item>6 gated MLA layers, NoPE: the absorbed form attends one 576-wide key (512 latent + 64 shared, never rotated
    /// channels) per cell with 32 query heads, scale 1/sqrt(192), values the 512 latent channels.</item>
    /// <item>A per-layer embedding added to every stream after each layer's feed-forward.</item>
    /// </list>
    /// The trailing NextN block of the <c>*_MTP</c> files is not run.
    /// </summary>
    public sealed partial class RaonRuntimeModel : StandardDecoderModel
    {
        private const string PleTable = "per_layer_token_embd.weight";
        private const float L2Eps = 1e-12f;   // F.normalize(eps=1e-6): x / max(|x|, 1e-6), here x / sqrt(|x|^2 + 1e-12)

        private RaonSpec _spec;
        private readonly RotarySpec _rotary;
        private readonly Dictionary<(int Layer, bool Ffn), (Tensor K, Tensor R, Tensor C0)> _charts = new();
        private readonly Dictionary<(IntPtr Key, int Kept, int Rows), IGraphPlan> _commitGraphs = new();

        public RaonRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "raon1nano";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            if (tpDegree > 1 || tpGroup != null)
                throw new NotSupportedException($"{arch}: tensor parallelism is not implemented yet: its lowering traces no rank's shard. " +
                    "Use --layer-split N for several GPUs.");

            RaonSpec p = Spec;
            Config.NumLayers = p.Layers;
            Config.NumHeads = p.MlaHeads;
            Config.NumKVHeads = 1;
            Config.KeyLength = p.MlaKey;
            Config.ValueLength = p.KvLora;
            _rotary = new RotarySpec(p.Extra, neoX: true, 10000f, 1f);   // never applied: the extra channels are NoPE
            ParseTokenizer();

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"Model: {arch}, Layers={p.Layers} ({p.Layers - p.RdaLayers} gated MLA, {p.RdaLayers} RDA), Hidden={p.Hidden}, MLA heads={p.MlaHeads} (latent {p.KvLora} + {p.Extra} NoPE, scale 1/sqrt({p.QkHead})), RDA heads={p.RdaHeads}x{p.RdaDim}, hc={p.Hc}, PLE={p.PleDim}, Vocab={Config.VocabSize}{(p.NextN > 0 ? $" (NextN block {p.Layers} not run)" : "")}"));

            InitializeDecoder(p.MlaKey, p.KvLora, string.Create(CultureInfo.InvariantCulture,
                $"{arch}|{p.Layers}|{p.Hidden}|{p.MlaHeads}|{p.KvLora}|{p.Extra}|{p.QkHead}|{p.RdaHeads}|{p.RdaDim}|{p.DecayRank}|{p.Hc}|" +
                $"{p.PleDim}|{Config.Eps:R}|{p.HcEps:R}|{p.GateLowerBound:R}|{p.Magnitude:R}|{LayerBits(p.IsRda)}"));
            try
            {
                ValidateWeights();
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        // ---------------------------------------------------------------- configuration

        /// <summary>The family's constants, read from the GGUF on first use (the base constructor may call hooks first).</summary>
        private RaonSpec Spec => _spec ??= new RaonSpec(_gguf);

        private sealed class RaonSpec
        {
            public readonly int Layers, NextN, Hidden, Hc, PleDim;
            public readonly int MlaHeads, KvLora, QLora, Extra, QkHead, QkNope, VMla;
            public readonly int RdaHeads, RdaDim, DecayRank, ConvKernel, RdaLayers;
            public readonly float HcEps, Magnitude, GateLowerBound;
            public readonly bool[] IsRda;

            public RaonSpec(GgufFile gguf)
            {
                string a = gguf.GetString("general.architecture") ?? "raon1nano";
                int blocks = (int)gguf.GetUint32($"{a}.block_count");
                NextN = (int)gguf.GetUint32($"{a}.nextn_predict_layers", 0);
                Layers = blocks - NextN;
                Hidden = (int)gguf.GetUint32($"{a}.embedding_length");
                Hc = (int)gguf.GetUint32($"{a}.hyper_connection.count", 0);
                HcEps = gguf.GetFloat32($"{a}.hyper_connection.epsilon", 1e-5f);
                Magnitude = gguf.GetFloat32($"{a}.hyper_connection.magnitude", 1f);
                PleDim = (int)gguf.GetUint32($"{a}.embedding_length_per_layer_input", 0);
                KvLora = (int)gguf.GetUint32($"{a}.attention.kv_lora_rank");
                QLora = (int)gguf.GetUint32($"{a}.attention.q_lora_rank");
                Extra = (int)gguf.GetUint32($"{a}.rope.dimension_count");
                QkHead = (int)gguf.GetUint32($"{a}.attention.key_length_mla");
                VMla = (int)gguf.GetUint32($"{a}.attention.value_length_mla");
                QkNope = QkHead - Extra;
                RdaDim = (int)gguf.GetUint32($"{a}.kda.head_dim");
                GateLowerBound = gguf.GetFloat32($"{a}.kda.gate_lower_bound", -5f);
                DecayRank = (int)gguf.GetUint32($"{a}.ssm.time_step_rank");
                ConvKernel = (int)gguf.GetUint32($"{a}.ssm.conv_kernel");

                uint[] heads = gguf.GetUint32Array($"{a}.attention.head_count");
                uint[] kvHeads = gguf.GetUint32Array($"{a}.attention.head_count_kv");
                if (heads == null || kvHeads == null || heads.Length < Layers || kvHeads.Length < Layers)
                    throw new NotSupportedException($"{a}: per-layer attention.head_count and head_count_kv arrays are required.");
                IsRda = new bool[Layers];
                for (int l = 0; l < Layers; l++)
                {
                    IsRda[l] = kvHeads[l] == 0;
                    if (IsRda[l])
                    {
                        RdaLayers++;
                        RdaHeads = RdaHeads == 0 ? (int)heads[l] : RdaHeads;
                        if (heads[l] != RdaHeads) throw new NotSupportedException($"{a}: RDA layers with {heads[l]} and {RdaHeads} heads.");
                    }
                    else
                    {
                        if (kvHeads[l] != 1) throw new NotSupportedException($"{a}: MLA layer {l} declares {kvHeads[l]} KV heads (absorbed MLA has one).");
                        MlaHeads = MlaHeads == 0 ? (int)heads[l] : MlaHeads;
                        if (heads[l] != MlaHeads) throw new NotSupportedException($"{a}: MLA layers with {heads[l]} and {MlaHeads} heads.");
                    }
                }
                if (Hc < 2 || PleDim <= 0 || KvLora <= 0 || Extra <= 0 || QkNope <= 0 || VMla <= 0 || RdaDim <= 0 || DecayRank <= 0
                    || ConvKernel < 2 || RdaHeads == 0 || MlaHeads == 0 || !(GateLowerBound < 0) || Magnitude <= 0)
                    throw new NotSupportedException($"{a}: incomplete Raon metadata (hc {Hc}, PLE {PleDim}, latent {KvLora}, extra {Extra}, " +
                        $"MLA key {QkHead}/value {VMla}, RDA {RdaHeads}x{RdaDim}, decay rank {DecayRank}, conv {ConvKernel}).");
            }

            public int MlaKey => KvLora + Extra;
            public long StreamFloats => (long)Hc * Hidden;
            public int MixWidth => 2 * Hc + (Hc - 1) * (Hc - 1);
            public int RdaInner => RdaHeads * RdaDim;
            public int RdaChannels => 3 * RdaInner;
            public int PleWidth => Layers * PleDim;
        }

        private static string LayerBits(bool[] flags)
        {
            var c = new char[flags.Length];
            for (int i = 0; i < flags.Length; i++) c[i] = flags[i] ? 'r' : 'm';
            return new string(c);
        }

        // ---------------------------------------------------------------- runtime hooks

        protected override bool RequiresGraphs => true;
        protected override bool RecurrentMixerLowers => true;
        protected override bool AllowsUnequalKeyValueDims => true;
        protected override RotarySpec GraphRotary => _rotary;
        protected override int StreamWidth => checked((int)Spec.StreamFloats);
        protected override bool IsRecurrentLayer(int layer) => Spec.IsRda[layer];
        protected override bool FusesQkv(int layer) => false;
        protected override string FingerprintExtras => $"|raon|{LayerBits(Spec.IsRda)}";

        /// <summary>The per-layer token table (1.6 B values) is read on the host, a token's rows a step.</summary>
        protected override bool IsHostOnlyWeight(string name) => name == PleTable;

        /// <summary>The NextN block of an <c>*_MTP</c> file is not run: its tensors are not loaded.</summary>
        protected override bool ShouldLoadWeight(GgufTensorInfo info)
            => Spec.NextN == 0 || !info.Name.StartsWith($"blk.{Spec.Layers}.", StringComparison.Ordinal);

        /// <summary>The per-op path is not implemented: Raon runs whole-forward graphs only.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
            => throw new NotSupportedException($"{Config.Architecture} runs whole-forward graphs only.");

        protected override RecurrentState CreateRecurrentState(RankScope scope, int layer)
        {
            RaonSpec p = Spec;
            return new Qwen35RuntimeModel.DeviceGdnState(scope.Allocator, ExecutorFor(scope.Allocator), HostPointer,
                p.ConvKernel - 1, p.RdaChannels, p.RdaHeads, p.RdaDim, p.RdaDim);
        }

        private static unsafe IntPtr HostPointer(Tensor tensor) => (IntPtr)GetFloatPtr(tensor);

        // ---------------------------------------------------------------- weights

        /// <summary>Each site's chart [K (17 x 16) | r (16) | c0 (16)] as three F32 constants, read while the host copies exist.</summary>
        protected override void PrepareLoadedWeights()
        {
            RaonSpec p = Spec;
            int mix = p.MixWidth, cells = p.Hc * p.Hc;
            for (int l = 0; l < p.Layers; l++)
                foreach (bool ffn in new[] { false, true })
                {
                    string name = $"blk.{l}.hc_{(ffn ? "ffn" : "attn")}_chart.weight";
                    if (!_weights.TryGetValue(name, out Tensor chart)) continue;   // ValidateWeights names it
                    int expected = mix * cells + 2 * cells;
                    if (chart.ElementCount() != expected)
                        throw new InvalidOperationException($"{name} holds {chart.ElementCount()} values, not {expected}.");
                    float[] v = chart.GetElementsAsFloat(expected);
                    // On the layer's whole-layer stage, which reads them.
                    IAllocator stage = Mesh.Allocator(Mesh.StageOf(l), 0);
                    _charts[(l, ffn)] = (Constant(stage, v, 0, mix * cells, cells, mix), Constant(stage, v, mix * cells, cells, cells),
                        Constant(stage, v, mix * cells + cells, cells, cells));
                }
        }

        private Tensor Constant(IAllocator allocator, float[] source, int offset, int count, params long[] sizes)
        {
            var values = new float[count];
            Array.Copy(source, offset, values, 0, count);
            var t = new Tensor(allocator, DType.Float32, sizes);
            t.SetElementsAsFloat(values);
            HostWritten(t);
            return t;
        }

        private void ValidateWeights()
        {
            RaonSpec p = Spec;
            var names = new List<string> { "token_embd.weight", "output_norm.weight", PleTable, "per_layer_model_proj.weight", "per_layer_proj_norm.weight" };
            for (int l = 0; l < p.Layers; l++)
            {
                foreach (string s in new[] { "attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_ffn_fn", "hc_ffn_base",
                             "ffn_up", "ffn_down", "attn_output", "inp_gate", "proj", "post_norm" })
                    names.Add($"blk.{l}.{s}.weight");
                string[] block = p.IsRda[l]
                    ? new[] { "attn_qkv.weight", "ssm_conv1d.weight", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_dt.bias", "ssm_norm.weight" }
                    : new[] { "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                        "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight" };
                foreach (string s in block) names.Add($"blk.{l}.{s}");
                if (!_charts.ContainsKey((l, false)) || !_charts.ContainsKey((l, true)))
                    throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing its hc_attn_chart / hc_ffn_chart.");
            }
            foreach (string name in names)
                if (!HasWeight(name)) throw new InvalidOperationException($"{Config.Architecture} is missing '{name}'.");
            long fab = WeightRows($"blk.{Array.IndexOf(p.IsRda, true)}.ssm_f_a.weight");
            if (fab != p.DecayRank + 2L * p.RdaHeads)
                throw new InvalidOperationException($"{Config.Architecture}: ssm_f_a has {fab} rows, not decay rank {p.DecayRank} + 2 x {p.RdaHeads} gates.");
        }

        private long WeightRows(string name)
            => _quantWeights.TryGetValue(name, out QuantizedWeight q) ? q.Ne1 : _weights[name].Sizes[0];

        // ---------------------------------------------------------------- per-layer embedding rows

        private GraphStep _pleStep;
        private int _pleNode = -1;

        /// <summary>The step's tokens' per-layer table rows [layers x ple, rows], dequantized on the host.</summary>
        private unsafe float[] GatherPleRows(int rows, float[] into = null)
        {
            int[] tokens = StepTokens;
            if (tokens == null || tokens.Length != rows)
                throw new InvalidOperationException($"{Config.Architecture}: the per-layer embedding needs the step's {rows} tokens.");
            int width = Spec.PleWidth;
            float[] result = into ?? new float[(long)rows * width];
            if (_quantWeights.TryGetValue(PleTable, out QuantizedWeight table))
            {
                int type = table.GgmlType;
                long rowBytes = NativeDequant.RowSize(type, table.Ne0);
                long source = (long)table.Data;
                fixed (float* dst = result)
                {
                    long dstA = (long)dst;
                    void Row(int i) => NativeDequant.DequantizeToFloat32Native(type, (IntPtr)(source + tokens[i] * rowBytes),
                        (IntPtr)(dstA + (long)i * width * sizeof(float)), width);
                    if (rows >= 16) Parallel.For(0, rows, Row);
                    else for (int i = 0; i < rows; i++) Row(i);
                }
            }
            else
            {
                Tensor dense = _weights[PleTable];
                for (int i = 0; i < rows; i++)
                {
                    using Tensor row = dense.Narrow(0, tokens[i], 1);
                    Array.Copy(row.GetElementsAsFloat(width), 0, result, (long)i * width, width);
                }
            }
            return result;
        }

        private int PleRows(GraphStep step)
        {
            if (ReferenceEquals(_pleStep, step) && _pleNode >= 0) return _pleNode;
            throw new InvalidOperationException($"{Config.Architecture}: the step's per-layer embeddings were not emitted.");
        }

        public override void Dispose()
        {
            foreach ((Tensor k, Tensor r, Tensor c0) in _charts.Values)
            {
                k.Dispose();
                r.Dispose();
                c0.Dispose();
            }
            _charts.Clear();
            foreach (IGraphPlan g in _commitGraphs.Values) g.Dispose();
            _commitGraphs.Clear();
            _pleStep = null;
            base.Dispose();
        }
    }
}
