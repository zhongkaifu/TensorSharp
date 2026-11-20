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
    /// Muse-Glimmer (GGUF <c>muse-glimmer</c> / <c>muse_glimmer</c>) on the shared decoder runtime, which provides every
    /// placement, continuous batching, the prefix cache, checkpoints and speculation. The recipe (llama.cpp
    /// <c>src/models/muse-glimmer.cpp</c>; the legacy <see cref="MuseGlimmerModel"/> and its fused graph
    /// <c>ggml_ops_muse_glimmer.cpp</c>):
    /// <list type="bullet">
    /// <item>a dense stack of GQA attention layers, sliding-window or full by <c>attention.sliding_window_pattern</c>: a scalar
    /// period P (layer l slides when l % P &lt; P - 1, llama.cpp's <c>set_swa_pattern</c>) or a per-layer bool array. The 30B's
    /// 52 layers are 39 sliding-window layers (window 2048, keys (p - 2048, p]) and 13 full ones (l = 3, 7, ..., 51);</item>
    /// <item>attention: per-head Q/K RMSNorm (the Q weight carries <c>qk_scale_factor</c>), then a NORM (GPT-J pair) rotary on
    /// the sliding-window layers only: the full layers are NoPE. An elementwise output gate,
    /// <c>attn * sigmoid(attn_gate · attn_norm(x))</c> over the whole query width, before <c>attn_output</c>;</item>
    /// <item>sandwich norms: <c>post_attention_norm</c> and <c>post_ffw_norm</c> on each block's raw output before its residual
    /// add, at a hardcoded epsilon of 1e-8 (<see cref="PostNormEps"/>); the pre-norms at the model's epsilon;</item>
    /// <item>the embedded rows (media rows already injected) take an unweighted RMSNorm before the first layer
    /// (<see cref="EmitStreamInput"/>, <see cref="ApplyStreamInput"/>);</item>
    /// <item>a dense SwiGLU feed-forward; the head (<c>output</c>, or <c>token_embd</c> tied) scaled by <c>logit_scale</c>, then a
    /// tanh softcap at <c>final_logit_softcapping</c> (<see cref="LogitScale"/>, <see cref="FinalLogitSoftcap"/>).</item>
    /// </list>
    /// Until the switch (docs/design/muse-glimmer-unified.md stage S8) the descriptor builds the legacy model: this family is
    /// reached through <see cref="MuseGlimmerArchitecture.CreateRuntimeModel"/> (tests, the probe, benchmarks) only.
    /// </summary>
    internal sealed partial class MuseGlimmerRuntimeModel : StandardDecoderModel
    {
        /// <summary>The post norms' epsilon, hardcoded in llama.cpp's graph (and legacy's), not the model's.</summary>
        internal const float PostNormEpsilon = 1e-8f;

        private const int DefaultSwaPatternPeriod = 4;
        private const int DefaultSlidingWindow = 2048;
        private const float DefaultFinalLogitSoftcap = 20.0f;
        private const float DefaultLogitScale = 1.0f;

        private static readonly string[] RequiredLayerWeights =
        {
            "attn_norm.weight", "attn_gate.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
            "post_attention_norm.weight", "ffn_norm.weight", "ffn_down.weight", "post_ffw_norm.weight",
        };

        private readonly bool[] _isSwa;
        private readonly int _window;
        private readonly int _headDim;
        private readonly float _logitScale;
        private readonly float _softcap;
        private readonly RotarySpec _rope;

        /// <summary>The weightless input norm's ones, one per device the stream enters on (per-op path).</summary>
        private readonly Dictionary<IAllocator, Tensor> _inputNormOnes = new(ReferenceEqualityComparer.Instance);

        internal MuseGlimmerRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "muse-glimmer";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            int layers = Config.NumLayers;

            _headDim = Config.HeadDim;
            int valueDim = Config.ValueLength > 0 ? Config.ValueLength : _headDim;
            if (valueDim != _headDim)
                throw new NotSupportedException($"{arch}: key width {_headDim} != value width {valueDim}.");

            _window = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", DefaultSlidingWindow);
            Config.SlidingWindow = _window;
            _softcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", DefaultFinalLogitSoftcap);
            _logitScale = _gguf.GetFloat32($"{arch}.logit_scale", DefaultLogitScale);
            _isSwa = ResolveSwaPattern(arch, layers);

            // NORM (GPT-J pair) rotary over every dim of a head (llama.cpp's n_rot; the 30B writes none: its head width).
            int ropeDims = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_headDim);
            _rope = new RotarySpec(ropeDims, neoX: false, Config.RopeBase, 1.0f / Config.RopeScale);
            ParseTokenizer();

            int sliding = _isSwa.Count(s => s);
            Console.WriteLine($"Model: {arch} (shared decoder runtime), Layers={layers} ({sliding} sliding-window, {layers - sliding} full NoPE), " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={_headDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  RoPE (NORM, after QK-norm, sliding-window layers only) base={Config.RopeBase} scale={Config.RopeScale} dims={ropeDims}; " +
                $"window={_window}; LogitScale={_logitScale}; Softcap={_softcap}; post-norm eps={PostNormEpsilon}");

            InitializeDecoder(_headDim, _headDim, string.Create(CultureInfo.InvariantCulture,
                $"muse-glimmer|{Config.NumHeads}|{Config.NumKVHeads}|{_headDim}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{ropeDims}|" +
                $"{_window}|{LayerBits(_isSwa)}|{_logitScale:R}|{_softcap:R}|{PostNormEpsilon:R}"));
            try
            {
                ValidateLayerWeights();
            }
            catch
            {
                Dispose();
                throw;
            }
            // Legacy's decline under --tp (design D5): the drafter borrows the head and embedding the ranks shard. The host's
            // attach then refuses it as the model's own decline (SpeculativeDraftHeadLoader: the server exits 2, the CLI warns).
            // A whole-layer split drafts (its last stage holds the head).
            bool tensorParallel = tpDegree > 1 || (tpGroup is not ILayerSplitGroup && GlobalTpDegree > 1);
            if (tensorParallel && !string.IsNullOrWhiteSpace(draftModelPath))
                Console.WriteLine("  DFlash speculative decoding is not supported under tensor parallelism; ignoring the drafter.");
            else
                AttachDraftModel(draftModelPath);
        }

        /// <summary>
        /// <c>{arch}.attention.sliding_window_pattern</c> as legacy reads it: a per-layer bool array (at least one entry a
        /// layer), else a scalar period P (llama.cpp's <c>set_swa_pattern(P)</c>: layer l slides when l % P &lt; P - 1); a
        /// missing key or a period of 0 is 4.
        /// </summary>
        private bool[] ResolveSwaPattern(string arch, int layers)
        {
            var swa = new bool[layers];
            string key = $"{arch}.attention.sliding_window_pattern";
            bool[] perLayer = _gguf.GetBoolArray(key);
            if (perLayer != null && perLayer.Length >= layers)
            {
                Array.Copy(perLayer, swa, layers);
                return swa;
            }
            int period = (int)_gguf.GetUint32(key, DefaultSwaPatternPeriod);
            if (period == 0) period = DefaultSwaPatternPeriod;
            for (int l = 0; l < layers; l++)
                swa[l] = l % period < period - 1;
            return swa;
        }

        private static string LayerBits(bool[] flags) => new(flags.Select(f => f ? '1' : '0').ToArray());

        /// <summary>Tests: whether <paramref name="layer"/> is a sliding-window (rotary) layer.</summary>
        internal bool IsSlidingWindowLayerForTests(int layer) => _isSwa[layer];

        // ---------------------------------------------------------------- the runtime's declarations

        protected override int SlidingWindow(int layer) => _isSwa[layer] && _window > 0 ? _window : 0;

        protected override string MixerPostNormWeight(int layer) => LayerName(layer, "post_attention_norm.weight");

        protected override string FfnPostNormWeight(int layer) => LayerName(layer, "post_ffw_norm.weight");

        protected override float PostNormEps(int layer, BlockSite site) => PostNormEpsilon;

        protected override float LogitScale => _logitScale;

        protected override float FinalLogitSoftcap => _softcap;

        /// <summary>
        /// The default draft window on ggml_metal and MLX: 7 drafts (an 8-row verify), for the DFlash drafter and n-gram drafts alike (the
        /// window is the trunk's verify economics). Measured on an M5 Pro (UD-IQ2_XXS, the 1.63 GB dflash-kquant drafter, greedy,
        /// five prompts x 512 tokens): a verify costs 76-80 ms at any width from 4 to 8 rows (past one row the IQ2/IQ3 products
        /// take the matrix kernel), 135-147 ms at 9 (n-gram's default 8 drafts) and 152-160 ms at 16, against 46 ms for a plain
        /// step, and a block 25-29 ms whatever its width: the drafts an 8-row verify carries are nearly free, wider ones dear. With
        /// the cost governor: 7 drafts 34.1 tok/s in all (35.0 with the GGML builder's argmax top-1), 5 drafts 31.6, 7 at the block
        /// default's gate 32.8, against 21.7 plain (legacy at its own default 30.7); ungoverned, before the prompt-follow fix, 3 / 5 / 7 / 15 drafts 24.8 / 27.3 / 27.9 /
        /// 17.3. MLX the same: a verify 80-91 ms from 4 to 8 rows, a block 23-34 ms; 7 drafts without the gate 34.9 tok/s in all,
        /// the speculator's 8 at the gate 33.4, 3 without it 26.5, against 21.1 plain. Elsewhere the speculator's default (8) until
        /// measured there (docs/design/muse-glimmer-unified.md §5.4).
        /// </summary>
        public override int SpecPreferredDraftWindow => DraftsWideOnThisBackend ? 7 : base.SpecPreferredDraftWindow;

        /// <summary>No confidence gate for the drafter on ggml_metal and MLX, where a verify row is nearly free (see
        /// <see cref="SpecPreferredDraftWindow"/>): 7 drafts without it 34.1 / 34.9 tok/s, with the block default's 0.35 32.8 on
        /// ggml_metal, lower on every prompt. The cost governor parks the drafter where its drafts lose (prose on ggml_metal: 21.6
        /// tok/s against 21.9 plain).</summary>
        public override float SpecPreferredBlockGate => HasDFlash && DraftsWideOnThisBackend ? 0f : base.SpecPreferredBlockGate;

        /// <summary>The backends whose verify costs about the same at 2-8 rows here (measured: ggml_metal and MLX on an M5 Pro).</summary>
        private bool DraftsWideOnThisBackend => _backend is BackendType.GgmlMetal or BackendType.Mlx;

        protected override string FingerprintExtras =>
            string.Create(CultureInfo.InvariantCulture,
                $"|muse-glimmer|swa={_window}:{LayerBits(_isSwa)}|rope={_rope.Dims}@{_rope.FreqBase:R}/norm|post={PostNormEpsilon:R}");

        /// <summary>The Q/K norms are per-head vectors every rank reads whole.</summary>
        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
            => new[] { LayerName(layer, "attn_q_norm.weight"), LayerName(layer, "attn_k_norm.weight") };

        /// <summary>The output gate's rows follow the query's (one per query dim, head after head): they shard by the query's
        /// head blocks.</summary>
        protected override IEnumerable<string> HeadParallelLayerWeights(int layer)
            => new[] { LayerName(layer, "attn_gate.weight") };

        /// <summary>
        /// What a tensor-parallel degree must also meet, as legacy checked it (MuseGlimmerModel.TensorParallel.cs): the
        /// row-parallel projections' rows split into whole quantization blocks a rank. The KV heads (2 on the 30B, so --tp is at
        /// most 2) are the runtime's own check, whose refusal already names them ("2 KV heads must each divide by N").
        /// </summary>
        protected override IEnumerable<string> ShardingProblems(int degree)
        {
            foreach (string suffix in new[] { "attn_output.weight", "ffn_down.weight" })
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!_gguf.Tensors.TryGetValue($"blk.{l}.{suffix}", out GgufTensorInfo info) || info.Shape.Length < 1) continue;
                    long block = GgufFile.GetBlockSize(info.Type), width = (long)info.Shape[0];
                    if (block <= 1 || width % block != 0 || (width / block) % degree == 0) continue;
                    yield return $"the {width / block} quantization blocks of a {suffix[..^".weight".Length]} row ({info.Type}, layer {l})";
                    break;
                }
            }
        }

        /// <summary>Every owned layer carries the weights its blocks read: refused at load, not at the first forward.</summary>
        private void ValidateLayerWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                foreach (string suffix in RequiredLayerWeights)
                {
                    string name = LayerName(l, suffix);
                    if (!HasWeight(name))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing '{name}'.");
                }
            }
        }

        public override void Dispose()
        {
            DisposeVisionTower();
            foreach (Tensor t in _inputNormOnes.Values) t.Dispose();
            _inputNormOnes.Clear();
            base.Dispose();
        }

        // ---------------------------------------------------------------- the per-op path

        /// <summary>The weightless input RMSNorm (llama.cpp's <c>build_norm(inpL, nullptr, ...)</c>) over the embedded rows, in
        /// place: an RMSNorm by ones, as legacy runs it.</summary>
        protected override void ApplyStreamInput(Tensor embedded)
        {
            int hidden = Config.HiddenSize;
            IAllocator allocator = embedded.Storage.Allocator;
            if (!_inputNormOnes.TryGetValue(allocator, out Tensor ones))
            {
                ones = new Tensor(allocator, DType.Float32, hidden);
                Ops.Fill(ones, 1f);
                _inputNormOnes[allocator] = ones;
            }
            long rows = embedded.ElementCount() / hidden;
            using Tensor view = embedded.View(rows, hidden);
            Ops.RMSNorm(view, view, ones, null, Config.Eps);
        }

        /// <summary>Per-head Q/K RMSNorm, then the NORM rotary on the sliding-window layers (the full layers are NoPE).</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            bool hostDecode = positions.Count == 1 && positions.IsContiguous && HostDecodeOps;
            HeadNorm(q, Replica(_weights[LayerName(layer, "attn_q_norm.weight")], rank.Stage, rank.Rank), rank.LocalHeads, positions.Count, hostDecode);
            HeadNorm(k, Replica(_weights[LayerName(layer, "attn_k_norm.weight")], rank.Stage, rank.Rank), rank.LocalKvHeads, positions.Count, hostDecode);
            if (!_isSwa[layer]) return;
            Rotate(q, rank.LocalHeads, KeyDim, positions, _rope);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, _rope);
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

        /// <summary>The elementwise output gate: the attention result [tokens, LocalHeads * ValueDim] times
        /// sigmoid(attn_gate · input), one gate a query dim (the rank's heads' rows: attn_gate shards by head blocks).</summary>
        protected override Tensor GateAttentionOutput(RankScope scope, int layer, Tensor input, Tensor attn)
        {
            using Tensor gate = Linear(scope, input, LayerName(layer, "attn_gate.weight"));
            Tensor dense = attn.IsContiguous() ? attn : Ops.NewContiguous(attn);
            Ops.SigmoidMul(dense, dense, gate);
            return dense;
        }
    }
}
