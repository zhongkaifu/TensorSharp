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
    /// LiquidAI LFM3 (GGUF <c>lfm3</c>: LFM3-1.7B, the text decoder of LFM3-Omni-2B) on the shared decoder runtime, which
    /// provides every placement, continuous batching, checkpoints and speculation. The recipe (llama.cpp
    /// <c>src/models/lfm3.cpp</c>, HF <c>modeling_lfm3.py</c>):
    /// <list type="bullet">
    /// <item>a hybrid stack: layers whose <c>attention.head_count_kv</c> is 0 are gated short convolutions
    /// (<see cref="StandardDecoderModel.EmitShortConvMixer"/>: <c>[b | c | x] = in_proj · h</c>, <c>c * conv(b * x)</c> with a
    /// causal depthwise filter of <c>shortconv.l_cache</c> taps, no bias or activation, <c>out_proj</c>); the rest are GQA
    /// attention, full or sliding-window (<c>attention.sliding_window_pattern</c>, keys (p - window, p]);</item>
    /// <item>attention: per-head Q/K RMSNorm BEFORE the NeoX rotary - base <c>rope.freq_base</c> on full layers,
    /// <c>rope.freq_base_swa</c> on sliding ones - and a head-wise output gate, <c>attn * sigmoid(attn_gate · h)</c> per head,
    /// before <c>attn_output</c>;</item>
    /// <item>the full layers' rotary has frequency factors <c>rope_freqs.weight</c> = k ones then a "zero tail" (1e30): NeoX
    /// pairs (i, i + head/2) rotate at <c>base^(-2i/head)</c> for i &lt; k and stand still past it. TensorSharp runs that as a
    /// plain partial rotary: the q/k projection rows and Q/K norm weights of each head are permuted at load to
    /// [0..k, head/2..head/2+k, k..head/2, head/2+k..head] (whole rows: exact for every quantization), and a NeoX rotary of
    /// 2k dims at base <c>base^(2k/head)</c> turns pair (i, i + k) - the original (i, i + head/2) - by the same angle. Scores
    /// are permutation-invariant, so attention is unchanged; K rows are cached permuted (the fingerprint says so);</item>
    /// <item>sandwich norms: <c>post_attention_norm</c> and <c>post_ffw_norm</c> on each block's output before its residual
    /// add; a dense SwiGLU FFN in every layer; the head tied to <c>token_embd</c>.</item>
    /// </list>
    /// Media towers (LFM3-Omni) inject through <see cref="StandardDecoderModel.QueueMedia"/>: the family is partial so the
    /// multimodal part lives beside it.
    /// </summary>
    public partial class Lfm3RuntimeModel : StandardDecoderModel
    {
        private readonly bool[] _isConv;
        private readonly bool[] _isSliding;
        private readonly int _window;
        private readonly int _convTaps;
        private readonly int _headDim;
        private readonly bool _postNorms;
        private readonly RotarySpec _ropeFull;
        private readonly RotarySpec _ropeSliding;
        /// <summary>Per-head dim order of the full layers' Q/K rows (new dim j holds old dim [j]), or null: no zero tail.</summary>
        private readonly int[] _headPermutation;
        /// <summary>Rotated pairs of the full layers (k of rope_freqs' ones); head/2 without a zero tail.</summary>
        private readonly int _rotatedPairs;

        private float[][][] _convKernelHost;   // [layer][rank] = [channels, taps] (the rank's channels, taps innermost)
        private Tensor[][] _convKernelRank;    // [layer][rank] on the rank's device (per-rank graphs)

        public Lfm3RuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "lfm3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            int layers = Config.NumLayers;

            // head_count_kv is per layer: 0 marks a short-conv layer (llama.cpp: is_recr = n_head_kv(il) == 0).
            _isConv = new bool[layers];
            int[] kvPerLayer = _gguf.GetInt32Array($"{arch}.attention.head_count_kv");
            if (kvPerLayer != null)
            {
                if (kvPerLayer.Length != layers)
                    throw new InvalidOperationException($"{arch}: attention.head_count_kv has {kvPerLayer.Length} entries for {layers} layers.");
                int kv = 0;
                for (int l = 0; l < layers; l++)
                {
                    _isConv[l] = kvPerLayer[l] == 0;
                    if (kvPerLayer[l] != 0 && kv != 0 && kvPerLayer[l] != kv)
                        throw new NotSupportedException($"{arch}: attention layers with {kv} and {kvPerLayer[l]} KV heads; the shared runtime keeps one KV width.");
                    kv = Math.Max(kv, kvPerLayer[l]);
                }
                if (kv == 0) throw new NotSupportedException($"{arch}: no attention layer (every head_count_kv is 0).");
                Config.NumKVHeads = kv;
            }

            _isSliding = new bool[layers];
            bool[] pattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            if (pattern != null)
            {
                if (pattern.Length != layers)
                    throw new InvalidOperationException($"{arch}: attention.sliding_window_pattern has {pattern.Length} entries for {layers} layers.");
                for (int l = 0; l < layers; l++) _isSliding[l] = pattern[l] && !_isConv[l];
            }
            _window = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 0);
            if (_isSliding.Any(s => s) && _window <= 0)
                throw new InvalidOperationException($"{arch}: sliding-window layers without attention.sliding_window.");

            _convTaps = (int)_gguf.GetUint32($"{arch}.shortconv.l_cache", 0);
            if (_isConv.Any(c => c) && _convTaps < 2)
                throw new InvalidOperationException($"{arch}: short-conv layers need shortconv.l_cache of at least 2, got {_convTaps}.");

            _headDim = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            int valueDim = Config.ValueLength > 0 ? Config.ValueLength : _headDim;
            if (valueDim != _headDim)
                throw new NotSupportedException($"{arch}: key width {_headDim} != value width {valueDim}.");
            if (Config.RopeBase <= 0) Config.RopeBase = 10000f;
            float slidingBase = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", Config.RopeBase);
            int ropeDims = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_headDim);
            (_headPermutation, _rotatedPairs) = ZeroTailRotary(ropeDims);
            _ropeFull = _headPermutation == null
                ? new RotarySpec(ropeDims, neoX: true, Config.RopeBase, 1.0f / Config.RopeScale)
                : new RotarySpec(2 * _rotatedPairs, neoX: true, PartialRotaryBase(Config.RopeBase, _rotatedPairs, ropeDims), 1.0f / Config.RopeScale);
            _ropeSliding = new RotarySpec(ropeDims, neoX: true, slidingBase, 1.0f / Config.RopeScale);

            // Sandwich norms come as a pair (llama.cpp refuses one without the other).
            bool postAttn = _gguf.Tensors.ContainsKey("blk.0.post_attention_norm.weight");
            bool postFfn = _gguf.Tensors.ContainsKey("blk.0.post_ffw_norm.weight");
            if (postAttn != postFfn)
                throw new InvalidOperationException($"{arch}: post_attention_norm and post_ffw_norm come together.");
            _postNorms = postAttn;
            ParseTokenizer();

            int conv = _isConv.Count(c => c), sliding = _isSliding.Count(s => s);
            Console.WriteLine($"Model: {arch}, Layers={layers} ({conv} short-conv, {layers - conv - sliding} full attention, {sliding} sliding-window), " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={_headDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  RoPE (NeoX, after QK-norm): full base={Config.RopeBase}" +
                (_headPermutation != null ? $" over {_rotatedPairs} of {ropeDims / 2} pairs (zero tail, permuted heads: {2 * _rotatedPairs} dims at base {_ropeFull.FreqBase})" : $" dims={ropeDims}") +
                $", sliding base={slidingBase} window={_window}; short conv taps={_convTaps}; sandwich norms={_postNorms}");

            InitializeDecoder(_headDim, _headDim, string.Create(CultureInfo.InvariantCulture,
                $"lfm3|{Config.NumHeads}|{Config.NumKVHeads}|{_headDim}|{Config.Eps:R}|{Config.RopeBase:R}|{slidingBase:R}|{ropeDims}|{_rotatedPairs}|" +
                $"{_window}|{_convTaps}|{LayerBits(_isConv)}|{LayerBits(_isSliding)}|{_postNorms}"));
            try
            {
                ValidateLayerWeights();
                BuildConvKernels();
            }
            catch
            {
                Dispose();
                throw;
            }
            AttachDraftModel(draftModelPath);
        }

        private static string LayerBits(bool[] flags) => new(flags.Select(f => f ? '1' : '0').ToArray());

        /// <summary>
        /// The full layers' rotary from <c>rope_freqs.weight</c>: no tensor (or every factor 1) is a plain rotary of every
        /// dim (null permutation); k factors of exactly 1 followed by a "zero tail" (factors of at least 1e20: the pair turns
        /// by less than 3e-15 rad at 262K positions, and llama.cpp's sin is ~1e-25) is a partial rotary of the first k NeoX
        /// pairs, run on heads permuted so they are the leading 2k dims. Any other factors are refused: the shared runtime
        /// has no frequency-factor rotary.
        /// </summary>
        private (int[] Permutation, int Pairs) ZeroTailRotary(int ropeDims)
        {
            int half = ropeDims / 2;
            if (!_gguf.Tensors.TryGetValue("rope_freqs.weight", out GgufTensorInfo info)) return (null, half);
            if (ropeDims != _headDim)
                throw new NotSupportedException($"{Config.Architecture}: frequency factors over a partial rotary ({ropeDims} of {_headDim} dims).");
            var factors = new float[info.NumElements];
            _gguf.ReadTensorDataToFloat32(info, factors, factors.Length);
            if (factors.Length != half)
                throw new NotSupportedException($"{Config.Architecture}: rope_freqs.weight holds {factors.Length} factors for {half} rotary pairs.");
            int k = 0;
            while (k < half && factors[k] == 1.0f) k++;
            for (int i = k; i < half; i++)
                if (!(factors[i] >= 1e20f))
                    throw new NotSupportedException($"{Config.Architecture}: rope_freqs.weight factor {i} is {factors[i]}: TensorSharp runs only " +
                        "LFM3's zero-tail form (ones, then factors of at least 1e20) as a permuted partial rotary.");
            if (k == half) return (null, half);
            if (k == 0) throw new NotSupportedException($"{Config.Architecture}: rope_freqs.weight turns no pair.");
            var permutation = new int[_headDim];
            int at = 0;
            for (int i = 0; i < k; i++) permutation[at++] = i;
            for (int i = 0; i < k; i++) permutation[at++] = half + i;
            for (int i = k; i < half; i++) permutation[at++] = i;
            for (int i = half + k; i < 2 * half; i++) permutation[at++] = i;
            return (permutation, k);
        }

        /// <summary>The base at which a NeoX rotary of 2k dims turns pair i by <paramref name="freqBase"/>^(-2i/dims):
        /// freqBase^(2k/dims) (sqrt(1e6) = 1000 for LFM3's 32 of 64 pairs).</summary>
        internal static float PartialRotaryBase(float freqBase, int pairs, int dims) => (float)Math.Pow(freqBase, 2.0 * pairs / dims);

        /// <summary>The full layers' q/k head layout (tests): new dim j holds the checkpoint's dim [j], or null.</summary>
        internal int[] HeadPermutationForTests => _headPermutation;

        private bool IsAttentionLayer(int layer) => !_isConv[layer];
        private bool IsFullAttention(int layer) => !_isConv[layer] && !_isSliding[layer];

        /// <summary>The rotary of an attention layer.</summary>
        private RotarySpec RopeOf(int layer) => _isSliding[layer] ? _ropeSliding : _ropeFull;

        protected override bool IsRecurrentLayer(int layer) => _isConv[layer];

        protected override int SlidingWindow(int layer) => _isSliding[layer] ? _window : 0;

        protected override string MixerPostNormWeight(int layer) => _postNorms ? LayerName(layer, "post_attention_norm.weight") : null;

        protected override string FfnPostNormWeight(int layer) => _postNorms ? LayerName(layer, "post_ffw_norm.weight") : null;

        protected override string FingerprintExtras =>
            $"|lfm3|swa={_window}:{LayerBits(_isSliding)}|conv={_convTaps}:{LayerBits(_isConv)}|rope={_ropeFull.Dims}@{_ropeFull.FreqBase:R}" +
            $"/{_ropeSliding.Dims}@{_ropeSliding.FreqBase:R}" + (_headPermutation != null ? $"|kperm={_rotatedPairs}" : "") + (_postNorms ? "|post" : "");

        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
            => _isConv[layer] ? Array.Empty<string>() : new[] { LayerName(layer, "attn_q_norm.weight"), LayerName(layer, "attn_k_norm.weight") };

        protected override IEnumerable<string> HeadParallelLayerWeights(int layer)
            => _isConv[layer] ? Array.Empty<string>() : new[] { LayerName(layer, "attn_gate.weight") };

        protected override IEnumerable<string> ShardingProblems(int degree)
        {
            if (Config.HiddenSize % degree != 0)
                yield return $"{Config.HiddenSize} short-conv channels";
        }

        /// <summary>The short convolution's filter and the rotary factors are coefficients read as F32.</summary>
        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
            => !info.Name.EndsWith(".shortconv.conv.weight", StringComparison.Ordinal) && info.Name != "rope_freqs.weight"
               && base.IsQuantizedLinearWeight(info);

        // ---------------------------------------------------------------- LoRA
        // Text LoRA is not opted in (ModelBase refuses adapters for this family at load): every projection does run through
        // LinearForward on the per-op path an adapter selects, but no merged-twin parity test covers LFM3 yet, and the full
        // layers' q/k rows are permuted at load (the zero-tail rotary), so an adapter's deltas for them would need the same
        // permutation of their B factors' rows. See the LFM3 phase-2 report's open issues.

        // ---------------------------------------------------------------- weights

        /// <summary>The full layers' Q/K rows and norms in the permuted head layout (see the class summary), before sharding and
        /// residency. Rows move whole, so every quantized format stays exact; the q/k fused into attn_qkv permute in place there.</summary>
        protected override void PrepareLoadedWeights()
        {
            if (_headPermutation == null) return;
            int heads = Config.NumHeads, kvHeads = Config.NumKVHeads, d = _headDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!IsFullAttention(l) || !OwnsLayer(l)) continue;
                string qkv = LayerName(l, "attn_qkv.weight");
                if (HasWeight(qkv))
                {
                    PermuteHeadRows(qkv, 0, heads);
                    PermuteHeadRows(qkv, (long)heads * d, kvHeads);
                }
                else
                {
                    PermuteHeadRows(LayerName(l, "attn_q.weight"), 0, heads);
                    PermuteHeadRows(LayerName(l, "attn_k.weight"), 0, kvHeads);
                }
                PermuteHeadRows(LayerName(l, "attn_q_norm.weight"), 0, 1);
                PermuteHeadRows(LayerName(l, "attn_k_norm.weight"), 0, 1);
            }
        }

        /// <summary>Permute rows [firstRow, firstRow + heads * head_dim) of a projection (or the elements of a norm vector) per
        /// head: new row h*d + j is old row h*d + permutation[j].</summary>
        private unsafe void PermuteHeadRows(string name, long firstRow, int heads)
        {
            int d = _headDim;
            int[] perm = _headPermutation;
            if (_quantWeights.TryGetValue(name, out QuantizedWeight q))
            {
                long rowBytes = NativeDequant.RowSize(q.GgmlType, q.Ne0);
                if (firstRow + (long)heads * d > q.Ne1)
                    throw new InvalidOperationException($"{Config.Architecture}: '{name}' has {q.Ne1} rows, not {heads} heads of {d} from row {firstRow}.");
                IntPtr dst = QuantizedWeight.AllocateBuffer(q.RawBytes);
                byte* source = (byte*)q.Data.ToPointer(), target = (byte*)dst.ToPointer();
                Buffer.MemoryCopy(source, target, q.RawBytes, q.RawBytes);
                for (int h = 0; h < heads; h++)
                    for (int j = 0; j < d; j++)
                    {
                        long to = firstRow + (long)h * d + j, from = firstRow + (long)h * d + perm[j];
                        Buffer.MemoryCopy(source + from * rowBytes, target + to * rowBytes, rowBytes, rowBytes);
                    }
                var permuted = new QuantizedWeight(dst, q.RawBytes, q.GgmlType, q.Ne0, q.Ne1) { Scale = q.Scale };
                _quantWeights[name] = permuted;
                q.Dispose();
                return;
            }
            if (!_weights.TryGetValue(name, out Tensor w))
                throw new InvalidOperationException($"{Config.Architecture}: '{name}' is missing.");
            // F32: a [rows, cols] matrix or a [d] vector (one "row" per element).
            long cols = w.DimensionCount == 1 ? 1 : w.Sizes[w.DimensionCount - 1];
            long rows = w.ElementCount() / cols;
            if (firstRow + (long)heads * d > rows)
                throw new InvalidOperationException($"{Config.Architecture}: '{name}' has {rows} rows, not {heads} heads of {d} from row {firstRow}.");
            float[] values = w.GetElementsAsFloat(checked((int)w.ElementCount()));
            var result = (float[])values.Clone();
            for (int h = 0; h < heads; h++)
                for (int j = 0; j < d; j++)
                    Array.Copy(values, (firstRow + (long)h * d + perm[j]) * cols, result, (firstRow + (long)h * d + j) * cols, cols);
            w.SetElementsAsFloat(result);
            HostWritten(w);
        }

        /// <summary>Every owned layer carries the weights its blocks read: refused at load, not at the first forward.</summary>
        private void ValidateLayerWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                IEnumerable<string> required = _isConv[l]
                    ? new[] { "shortconv.conv.weight" }
                    : new[] { "attn_gate.weight", "attn_q_norm.weight", "attn_k_norm.weight" };
                foreach (string suffix in required)
                {
                    string name = LayerName(l, suffix);
                    if (!HasWeight(name))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing '{name}'.");
                }
            }
        }

        private static string ConvWeight(int layer) => $"blk.{layer}.shortconv.conv.weight";
        private static string InProj(int layer) => $"blk.{layer}.shortconv.in_proj.weight";
        private static string OutProj(int layer) => $"blk.{layer}.shortconv.out_proj.weight";

        /// <summary>Each rank's slice of every short-conv filter [channels, taps]: host copies for the per-op mixer and, where
        /// per-rank graphs run, tensors on the rank's device.</summary>
        private void BuildConvKernels()
        {
            int layers = Config.NumLayers, ranks = Mesh.TensorParallel, degree = ShardDegree, hidden = Config.HiddenSize;
            int channels = hidden / degree;
            _convKernelHost = new float[layers][][];
            _convKernelRank = new Tensor[layers][];
            for (int l = 0; l < layers; l++)
            {
                if (!_isConv[l] || !OwnsLayer(l)) continue;
                float[] filter = _weights[ConvWeight(l)].GetElementsAsFloat(hidden * _convTaps);
                _convKernelHost[l] = new float[ranks][];
                if (ranks > 1) _convKernelRank[l] = new Tensor[ranks];
                for (int r = 0; r < ranks; r++)
                {
                    int firstChannel = (ShardRankOffset + r) * channels;
                    var slice = new float[channels * _convTaps];
                    Array.Copy(filter, (long)firstChannel * _convTaps, slice, 0, slice.Length);
                    _convKernelHost[l][r] = slice;
                    if (ranks == 1) continue;
                    var t = new Tensor(Mesh.Allocator(Mesh.StageOf(l), r), DType.Float32, channels, _convTaps);
                    t.SetElementsAsFloat(slice);
                    HostWritten(t);
                    _convKernelRank[l][r] = t;
                }
            }
        }

        public override void Dispose()
        {
            DisposeMediaTowers();
            DisposeSpeechOutput();
            if (_convKernelRank != null)
                foreach (Tensor[] layer in _convKernelRank)
                    if (layer != null)
                        foreach (Tensor t in layer) t?.Dispose();
            _convKernelRank = null;
            base.Dispose();
        }

        // ---------------------------------------------------------------- short conv

        protected override RecurrentState CreateRecurrentState(RankScope scope, int layer)
            => new ShortConvState(scope.Allocator, ExecutorFor(scope.Allocator), Config.HiddenSize / ShardDegree, _convTaps - 1);

        protected override Tensor RecurrentMixer(RankScope scope, Tensor normed, int layer, RecurrentState state)
            => ShortConvMixer(scope, normed, (ShortConvState)state, InProj(layer), _convKernelHost[layer][scope.Rank], OutProj(layer));

        /// <summary>Channel-sharded short convolutions (llama.cpp mirrors them): rank r keeps its channels' rows of each of the
        /// in-projection's b, c and x blocks, its slice of the filter, and its channels' columns of the out-projection, whose
        /// partials the runtime reduces like attention's.</summary>
        protected override void ShardRecurrentLayer(int layer)
        {
            int degree = ShardDegree, hidden = Config.HiddenSize, channels = hidden / degree;
            ShardOutputRows(InProj(layer), r =>
            {
                var rows = new int[3 * channels];
                for (int block = 0; block < 3; block++)
                    for (int c = 0; c < channels; c++)
                        rows[block * channels + c] = block * hidden + r * channels + c;
                return rows;
            });
            ShardInputColumnGroups(OutProj(layer), r => new[] { r }, channels);
        }

        // ---------------------------------------------------------------- attention (per-op path)

        /// <summary>Per-head Q/K RMSNorm, then the layer's rotary (full or sliding base; the full layers' partial one on their
        /// permuted heads).</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            bool hostDecode = positions.Count == 1 && positions.IsContiguous && HostDecodeOps;
            HeadNorm(q, Replica(_weights[LayerName(layer, "attn_q_norm.weight")], rank.Stage, rank.Rank), rank.LocalHeads, positions.Count, hostDecode);
            HeadNorm(k, Replica(_weights[LayerName(layer, "attn_k_norm.weight")], rank.Stage, rank.Rank), rank.LocalKvHeads, positions.Count, hostDecode);
            RotarySpec spec = RopeOf(layer);
            Rotate(q, rank.LocalHeads, KeyDim, positions, spec);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, spec);
        }

        private void HeadNorm(Tensor data, Tensor alpha, int heads, int tokens, bool onHost)
        {
            if (onHost)
            {
                RMSNormInPlaceCpu(data, alpha, heads, KeyDim, Config.Eps);
                HostWritten(data);
                return;
            }
            using Tensor rows = data.View(tokens * heads, KeyDim);
            Ops.RMSNorm(rows, rows, alpha, null, Config.Eps);
        }

        /// <summary>The head-wise output gate: each head's attention row times sigmoid(attn_gate · h) for that head (the rank's
        /// heads under tensor parallelism: attn_gate shards by head blocks).</summary>
        protected override Tensor GateAttentionOutput(RankScope scope, int layer, Tensor input, Tensor attn)
        {
            int tokens = (int)attn.Sizes[0], heads = scope.LocalHeads, d = ValueDim;
            float[] logits;
            using (Tensor g = Linear(scope, input, LayerName(layer, "attn_gate.weight")))
                logits = g.GetElementsAsFloat(tokens * heads);
            var widened = new float[(long)tokens * heads * d];
            for (int i = 0; i < tokens * heads; i++)
                Array.Fill(widened, logits[i], i * d, d);
            using var gate = new Tensor(scope.Allocator, DType.Float32, tokens, heads * d);
            gate.SetElementsAsFloat(widened);
            HostWritten(gate);
            Tensor dense = attn.IsContiguous() ? attn : Ops.NewContiguous(attn);
            Ops.SigmoidMul(dense, dense, gate);
            return dense;
        }
    }
}
