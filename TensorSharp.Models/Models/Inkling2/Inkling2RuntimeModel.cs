// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace TensorSharp.Models
{
    /// <summary>
    /// Thinking Machines Inkling 2 (GGUF <c>inkling2</c>) and Inkling 2 Mini's looped stack (<c>inkling2_loop</c>) on the shared
    /// decoder runtime, as llama.cpp's <c>src/models/inkling2.cpp</c> and <c>inkling2-loop.cpp</c> compute it (the recipe is
    /// written out in <c>docs/design/inkling2-on-shared-runtime.md</c>):
    /// <list type="bullet">
    /// <item>no rotary: the attention scores carry a per-head relative-position bias by query-key distance and a per-query sink
    /// logit; sliding-window and full layers; L2-clipped queries, keys and relative queries, normed values, a sigmoid
    /// output gate;</item>
    /// <item>causal short convolutions (4 taps) on K, V, the attention output and the MoE output, carried per sequence in a
    /// <see cref="StandardDecoderModel.ShortConvState"/> on every layer (<see cref="AttentionLayerHasState"/>); the full layers'
    /// K and V convolutions add input-dependent taps per head;</item>
    /// <item>attention residuals: every block of <c>attn_res.block_size</c> layers starts a new residual stream, and each
    /// block input is a softmax mix over depth of the earlier blocks' streams and the current one; the stream between layers
    /// keeps them all (<see cref="StreamWidth"/>);</item>
    /// <item>sigmoid-routed experts (the bias picks, the weights stay unbiased, normalized, times
    /// <c>expert_weights_scale</c>) with a fused gate/up stack and the SITU activation; a final logit softcap and a padded
    /// vocabulary;</item>
    /// <item><c>inkling2_loop</c>: stored layers [loop.start, loop.end) run loop.count passes, each pass from the prelude's
    /// stream and with its own KV cache and conv state, later passes mixing gated messages from the previous pass. The
    /// runtime's layers are the runs in execution order (<see cref="PhysicalLayer"/> names the stored layer).</item>
    /// </list>
    /// Milestone 1 lowers whole-forward graphs only (<see cref="StandardDecoderModel.RequiresGraphs"/>) on one device.
    /// </summary>
    public partial class Inkling2RuntimeModel : StandardDecoderModel
    {
        private readonly int _physicalLayers;
        private readonly bool _loop;
        private readonly int _loopStart, _loopEnd, _loopPasses, _gateRank;
        private readonly bool[] _isSliding;   // per physical layer
        private readonly int _window;
        private readonly int _convTaps;
        private readonly int _blockSize;
        private readonly int _relDim, _relExtent;
        private readonly float _qkClip, _relClip;
        private readonly int _logScaleFloor;
        private readonly float _logScaleAlpha;
        private readonly float _expertScale, _situBeta, _situLinearBeta, _softcap;
        private readonly int _vocabPadded;
        private readonly int _headDim;
        private readonly int _kvDim;              // KV heads x head dim
        private readonly int _convChannels;       // K + V + attention output + MoE output

        // ---- the residual stream's slots, each [hidden] a row: CUR, the blocks B_0 .., the loop's bank BK_0 .. and BO.
        private readonly int _blocks;             // blocks pushed over the stored stack: ceil(layers / block size)
        private readonly int _bankBlocks;         // a pass's blocks after its first core block
        private readonly int _streamSlots;

        /// <summary>Padded relative-bias tables [extent + 1, rel] (the last row zero) per stored layer, on the model's device.</summary>
        private Tensor[] _relTables;
        // The full layers' log-scaled positions by position [1, positions]: tau(p) = 1 + alpha log(max(1, (p + 1) / floor)),
        // read by the step's positions so a decode step advances without a trace.
        private Tensor _tauTable;

        public Inkling2RuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "inkling2";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            if (tpDegree > 1 || layerSplitDegree > 1)
                throw new NotSupportedException($"{arch}: tensor parallelism and layer split are not implemented yet (one device).");

            _physicalLayers = Config.NumLayers;
            int layers = _physicalLayers;
            uint Key(string name) => _gguf.GetUint32($"{arch}.{name}", 0);
            float KeyF(string name) => _gguf.GetFloat32($"{arch}.{name}", 0f);

            _isSliding = new bool[layers];
            bool[] pattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            if (pattern == null || pattern.Length != layers)
                throw new InvalidOperationException($"{arch}: attention.sliding_window_pattern must have one entry per layer.");
            Array.Copy(pattern, _isSliding, layers);
            _window = (int)Key("attention.sliding_window");
            _convTaps = (int)Key("shortconv.l_cache");
            _blockSize = (int)Key("attn_res.block_size");
            _relDim = (int)Key("attention.relative_length");
            _relExtent = (int)Key("attention.relative_extent");
            _qkClip = KeyF("attention.qk_clip");
            _relClip = KeyF("attention.relative_clip");
            _logScaleFloor = (int)Key("attention.log_scale_floor");
            _logScaleAlpha = KeyF("attention.log_scale_alpha");
            _expertScale = KeyF("expert_weights_scale");
            _situBeta = KeyF("activation.situ_beta");
            _situLinearBeta = KeyF("activation.situ_linear_beta");
            _softcap = KeyF("final_logit_softcapping");
            _vocabPadded = (int)Key("vocab_size");
            int vocab = (int)Key("vocab_size_unpadded");
            if (_window <= 0 || _convTaps < 2 || _blockSize <= 0 || _relDim <= 0 || _relExtent <= 0 || _logScaleFloor <= 0
                || !(_qkClip > 0) || !(_relClip > 0) || !(_situBeta > 0) || vocab <= 0 || _vocabPadded < vocab)
                throw new InvalidOperationException($"{arch}: invalid window, convolution, block, relative-position, clip or vocabulary metadata.");

            _loop = arch == "inkling2_loop";
            if (_loop)
            {
                _loopStart = (int)Key("loop.start");
                _loopEnd = (int)Key("loop.end");
                _loopPasses = (int)Key("loop.count");
                _gateRank = (int)Key("loop.gate_rank");
                if (_gguf.GetBool($"{arch}.loop.varying", false))
                    throw new NotSupportedException($"{arch}: a variable loop (loop.varying) is not implemented.");
                if (_loopPasses < 2 || _gateRank <= 0 || _loopStart <= 0 || _loopStart >= _loopEnd || _loopEnd > layers
                    || _loopStart % _blockSize != 0 || _loopEnd % _blockSize != 0)
                    throw new InvalidOperationException($"{arch}: invalid loop core [{_loopStart}, {_loopEnd}) x {_loopPasses} for block size {_blockSize}.");
            }
            else
            {
                _loopStart = _loopEnd = 0;
                _loopPasses = 1;
            }

            _headDim = Config.KeyLength > 0 ? Config.KeyLength : Config.HiddenSize / Config.NumHeads;
            int valueDim = Config.ValueLength > 0 ? Config.ValueLength : _headDim;
            if (valueDim != _headDim)
                throw new NotSupportedException($"{arch}: key width {_headDim} != value width {valueDim}.");
            _kvDim = Config.NumKVHeads * _headDim;
            _convChannels = 2 * _kvDim + 2 * Config.HiddenSize;
            _blocks = (layers + _blockSize - 1) / _blockSize;
            _bankBlocks = _loop ? (_loopEnd - 1) / _blockSize - _loopStart / _blockSize : 0;
            _streamSlots = 1 + _blocks + (_loop ? _bankBlocks + 1 : 0);

            // The runtime's layers: the stored ones, then every later pass of the loop core as layers of its own.
            Config.NumLayers = layers + (_loopPasses - 1) * (_loopEnd - _loopStart);
            ParseTokenizer();
            Config.VocabSize = vocab;

            Console.WriteLine($"Model: {arch}, Layers={layers} stored" + (_loop ? $" ({_loopStart}-{_loopEnd - 1} x {_loopPasses} passes: {Config.NumLayers} runs)" : "") +
                $", {_isSliding.Count(s => !s)} full attention, Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"HeadDim={_headDim}, Window={_window}, Experts={_gguf.GetUint32($"{arch}.expert_count")}x{_gguf.GetUint32($"{arch}.expert_used_count")}, " +
                $"Vocab={vocab} (padded {_vocabPadded})");

            InitializeDecoder(_headDim, _headDim, string.Create(CultureInfo.InvariantCulture,
                $"{arch}|{Config.NumHeads}|{Config.NumKVHeads}|{_headDim}|{Config.Eps:R}|{_window}|{_convTaps}|{_blockSize}|{_relDim}|{_relExtent}|" +
                $"{_qkClip:R}|{_relClip:R}|{_logScaleFloor}|{_logScaleAlpha:R}|{_expertScale:R}|{_situBeta:R}|{_situLinearBeta:R}|{_softcap:R}|" +
                $"{_loopStart}|{_loopEnd}|{_loopPasses}|{LayerBits(_isSliding)}"));
            try
            {
                ValidateLayerWeights();
                // The checkpoint's DSpark drafter (an inkling2_dspark GGUF), a block drafter of the family's own.
                if (!string.IsNullOrEmpty(draftModelPath))
                    FamilyDrafter = new DSpark(this, draftModelPath);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>The padded bias tables, while every host copy of the loaded weights still exists.</summary>
        protected override void PrepareLoadedWeights()
        {
            BuildRelativeTables();
            BuildTauTable();
            PackProjections();
        }

        /// <summary>Row offsets of a stored layer's packed attention projections (<see cref="PackProjections"/>).</summary>
        private sealed class ProjectionRows
        {
            public int Q, Gate, Rel, Sink, K, KConv = -1, V, VConv = -1, Total;
            /// <summary>Rows read with xq ([Q, K)), with xk ([K, V)) and with xv ([V, Total)).</summary>
            public int KRows => V - K;
            public int VRows => Total - V;
        }

        /// <summary>Stored layers whose projections are packed (null: none, or the layer kept its separate weights).</summary>
        private ProjectionRows[] _projections;

        private static readonly bool PackedProjectionsDisabled = Environment.GetEnvironmentVariable("TS_INKLING_FUSED_PROJECTIONS") == "0";

        /// <summary>
        /// Each stored layer's attention projections as one quantized weight, its output rows
        /// [q | gate | rel | sink | k (| k_conv) | v (| v_conv)]: a decode step reads them in one product (or one a distinct
        /// input) instead of six to eight, each of which paid its own activation quantize and launch tail. Only where every
        /// source is quantized with one type, input width and no per-tensor scale; the sources are released.
        /// </summary>
        private void PackProjections()
        {
            if (PackedProjectionsDisabled) return;
            int packed = 0;
            for (int l = 0; l < _physicalLayers; l++)
            {
                string Name(string suffix) => $"blk.{l}.{suffix}";
                var names = new List<string> { Name("attn_q.weight"), Name("attn_gate.weight"), Name("attn_rel.weight"), Name("attn_sink_proj.weight"), Name("attn_k.weight") };
                bool conv = _quantWeights.ContainsKey(Name("attn_k_conv_proj.weight")) && _quantWeights.ContainsKey(Name("attn_v_conv_proj.weight"));
                if (conv) names.Add(Name("attn_k_conv_proj.weight"));
                names.Add(Name("attn_v.weight"));
                if (conv) names.Add(Name("attn_v_conv_proj.weight"));
                var weights = new QuantizedWeight[names.Count];
                bool ok = true;
                for (int i = 0; i < names.Count && ok; i++)
                    ok = _quantWeights.TryGetValue(names[i], out weights[i]) && weights[i].GgmlType == weights[0].GgmlType
                         && weights[i].Ne0 == weights[0].Ne0 && weights[i].Scale == 1.0f;
                if (!ok || !TryCreateFusedQuantizedWeight(out QuantizedWeight fused, weights))
                    continue;
                var rows = new ProjectionRows();
                int at = 0;
                int Take(int i) { int start = at; at += (int)weights[i].Ne1; return start; }
                rows.Q = Take(0);
                rows.Gate = Take(1);
                rows.Rel = Take(2);
                rows.Sink = Take(3);
                rows.K = Take(4);
                int next = 5;
                if (conv) rows.KConv = Take(next++);
                rows.V = Take(next++);
                if (conv) rows.VConv = Take(next++);
                rows.Total = at;
                _quantWeights[Name("attn_proj.weight")] = fused;
                foreach (string n in names)
                {
                    _quantWeights[n].Dispose();
                    _quantWeights.Remove(n);
                }
                _projections ??= new ProjectionRows[_physicalLayers];
                _projections[l] = rows;
                packed++;
            }
            if (packed > 0)
                Console.WriteLine($"  Packed attention projections: {packed} layers");
        }

        private static string LayerBits(bool[] flags) => new(flags.Select(f => f ? '1' : '0').ToArray());

        // ---------------------------------------------------------------- the loop as layers of the stack

        /// <summary>The stored layer a run of the stack reads: runs [0, loop.end) are the prelude and the first pass, then each
        /// later pass's core, then the layers after the core.</summary>
        protected override int PhysicalLayer(int layer)
        {
            if (!_loop || layer < _loopEnd) return layer;
            int core = _loopEnd - _loopStart, later = (_loopPasses - 1) * core;
            return layer < _loopEnd + later ? _loopStart + (layer - _loopEnd) % core : layer - later;
        }

        /// <summary>The loop pass a run belongs to (0 outside the core).</summary>
        private int PassOf(int layer)
        {
            if (!_loop || layer < _loopEnd) return 0;
            int core = _loopEnd - _loopStart;
            return layer < _loopEnd + (_loopPasses - 1) * core ? 1 + (layer - _loopEnd) / core : 0;
        }

        /// <summary>The run that starts loop pass <paramref name="pass"/> (1 ..), where the stream restarts from the prelude.</summary>
        private bool StartsPass(int layer) => _loop && layer >= _loopEnd && PassOf(layer) > 0 && PhysicalLayer(layer) == _loopStart;

        /// <summary>A later pass's run of a core layer: its sublayer inputs mix in the previous pass's messages.</summary>
        private bool ReadsMessages(int layer) => PassOf(layer) > 0;

        private bool IsSliding(int layer) => _isSliding[PhysicalLayer(layer)];

        protected override int SlidingWindow(int layer) => IsSliding(layer) ? _window : 0;

        // ---------------------------------------------------------------- runtime hooks

        protected override bool AttentionLayerHasState(int layer) => true;

        protected override RecurrentState CreateRecurrentState(RankScope scope, int layer)
            => new ShortConvState(scope.Allocator, ExecutorFor(scope.Allocator), _convChannels, _convTaps - 1);

        /// <summary>Rows a prompt step takes a sequence (TS_INKLING_RING_STEP_ROWS, experiment): a sliding ring keeps its window
        /// plus these rows.</summary>
        protected override int RingStepRowsForPlacement
            => int.TryParse(Environment.GetEnvironmentVariable("TS_INKLING_RING_STEP_ROWS"), out int rows) && rows >= 64
                ? rows : base.RingStepRowsForPlacement;

        /// <summary>Several sequences verify 4 rows each, up to 16 in all: the experts product reads a verify's distinct experts
        /// once (Inkling 2 Mini, 128 experts, 8 a row), so 16 rows cost far less than twice 8. Four code conversations: 103.7
        /// tok/s in all at 4 rows each against 97.4 at 2; two: 76.2 at 4 rows each against 67.5 at 8.</summary>
        protected override int ExpertVerifyRowBudget => 16;

        protected override int ExpertVerifySequenceRows => 4;

        /// <summary>Verify windows pad their attention windows to 2,048 rows where the fused attention runs: it finds a row's keys
        /// from the positions and its key splits past the written ones exit at once, so the wider bucket costs nothing, while
        /// a 256-row bucket traced every verify width again every 256 positions (25-95 ms each; a 384-token reply at 24K
        /// traced about 16 of them).</summary>
        protected override int DecodeWindowRows(int executorRows) => FusedAttention ? Math.Max(executorRows, 2048) : executorRows;

        /// <summary>The full layers read separate Q, K and V inputs, and so does every layer of a later pass.</summary>
        protected override bool FusesQkv(int layer) => false;

        protected override bool RequiresGraphs => true;


        protected override string FingerprintExtras =>
            $"|inkling2|swa={_window}:{LayerBits(_isSliding)}|conv={_convTaps}|rel={_relDim}x{_relExtent}|block={_blockSize}" +
            (_loop ? $"|loop={_loopStart}-{_loopEnd}x{_loopPasses}" : "");

        /// <summary>The coefficient tensors (convolution taps, clip scales, bias tables, depth queries, loop controllers, the
        /// router) are read as F32.</summary>
        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            string n = info.Name;
            if (n.EndsWith("_conv.weight", StringComparison.Ordinal) || n.EndsWith("_clip.weight", StringComparison.Ordinal)
                || n.EndsWith("attn_rel_bias.weight", StringComparison.Ordinal) || n.EndsWith("_res_query.weight", StringComparison.Ordinal)
                || n.Contains(".loop_", StringComparison.Ordinal) || n.EndsWith("ffn_gate_inp.weight", StringComparison.Ordinal))
                return false;
            return base.IsQuantizedLinearWeight(info);
        }

        /// <summary>The per-op path is not implemented (milestone 1 runs whole-forward graphs only).</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
            => throw new NotSupportedException($"{Config.Architecture} runs whole-forward graphs only.");

        // ---------------------------------------------------------------- weights

        private void ValidateLayerWeights()
        {
            for (int l = 0; l < _physicalLayers; l++)
            {
                var required = new List<string>
                {
                    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight", "attn_q_clip.weight",
                    "attn_sink_proj.weight", "attn_rel.weight", "attn_rel_bias.weight", "attn_rel_clip.weight", "attn_k_conv.weight",
                    "attn_v_conv.weight", "attn_post_conv.weight", "ffn_post_conv.weight", "ffn_res_query.weight", "ffn_norm.weight",
                    "ffn_gate_inp.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight",
                };
                if (_isSliding[l]) required.Add("attn_norm.weight");
                else required.AddRange(new[] { "attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
                    "attn_k_conv_proj.weight", "attn_v_conv_proj.weight" });
                if (l > 0) required.Add("attn_res_query.weight");
                if (_loop && l >= _loopStart && l < _loopEnd)
                {
                    required.Add("loop_msg_query.weight");
                    foreach (string c in Controllers)
                        foreach (string part in new[] { "down", "gain", "forget" }) required.Add($"loop_{c}_ctrl_{part}.weight");
                }
                if (_projections?[l] != null)
                {
                    // Packed into attn_proj.weight at load (PackProjections).
                    required.RemoveAll(n => n is "attn_q.weight" or "attn_k.weight" or "attn_v.weight" or "attn_gate.weight"
                        or "attn_sink_proj.weight" or "attn_rel.weight" or "attn_k_conv_proj.weight" or "attn_v_conv_proj.weight");
                    required.Add("attn_proj.weight");
                }
                foreach (string suffix in required)
                {
                    string name = $"blk.{l}.{suffix}";
                    if (!HasWeight(name) && !_stackedExpertWeights.ContainsKey(name))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing '{name}'.");
                }
            }
            foreach (string name in new[] { "token_embd_norm.weight", "output_norm.weight", "output_res_query.weight" })
                if (!HasWeight(name)) throw new InvalidOperationException($"{Config.Architecture} is missing '{name}'.");
        }

        private static readonly string[] Controllers = { "q", "k", "v", "ffn" };

        /// <summary>Each stored layer's relative-bias table [rel, extent] (ggml order) padded with a zero column: distances
        /// outside [0, extent) gather the zero row, as llama.cpp's <c>ggml_pad(attn_rel_bias, 0, 1)</c> does.</summary>
        private void BuildRelativeTables()
        {
            _relTables = new Tensor[_physicalLayers];
            for (int l = 0; l < _physicalLayers; l++)
            {
                Tensor bias = _weights[$"blk.{l}.attn_rel_bias.weight"];
                int extent = (int)(bias.ElementCount() / _relDim);
                float[] values = bias.GetElementsAsFloat(extent * _relDim);
                var padded = new float[(extent + 1) * _relDim];
                Array.Copy(values, padded, values.Length);
                var t = new Tensor(Mesh.Allocator(Mesh.StageOf(l), 0), DType.Float32, extent + 1, _relDim);
                t.SetElementsAsFloat(padded);
                HostWritten(t);
                _relTables[l] = t;
            }
        }

        private void BuildTauTable()
        {
            int n = Config.DeclaredContextLength > 0 ? Config.DeclaredContextLength : 1 << 20;
            var tau = new float[n];
            for (int p = 0; p < n; p++) tau[p] = 1f + _logScaleAlpha * MathF.Log(MathF.Max(1f, (p + 1f) / _logScaleFloor));
            var t = new Tensor(Mesh.Allocator(0, 0), DType.Float32, n, 1);
            t.SetElementsAsFloat(tau);
            HostWritten(t);
            _tauTable = t;
        }

        /// <summary>The bias table's distance bound of a run: the window on a sliding layer, relative_extent on a full one.</summary>
        private int ExtentOf(int layer) => (int)(_relTables[PhysicalLayer(layer)].Sizes[0] - 1);

        public override void Dispose()
        {
            FamilyDrafter?.Dispose();
            FamilyDrafter = null;
            if (_relTables != null)
                foreach (Tensor t in _relTables) t?.Dispose();
            _relTables = null;
            _tauTable?.Dispose();
            _tauTable = null;
            base.Dispose();
        }
    }
}
