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
    /// <c>src/models/muse-glimmer.cpp</c>, which the legacy model and its fused native graph, deleted in stage S9 of
    /// docs/design/muse-glimmer-unified.md, followed too):
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
    /// The descriptor builds it (<see cref="MuseGlimmerArchitecture.Descriptor"/>, since stage S8 of docs/design/muse-glimmer-unified.md).
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
        /// The rows a prompt step writes into a sliding-window ring on MLX (rings of align256(window + rows) cells: 3072 for the
        /// 30B's 2048-token window): the runtime's 512 elsewhere. MLX's graph executor does not overlap a step's host work with
        /// the previous step's GPU work, so each step costs host time a wider step amortizes. Measured on an M5 Pro (48 GB),
        /// UD-IQ2_XXS, the probe's bench, ABBA in one binary at 41e0d628 (docs/design/muse-glimmer-unified.md §5.8), 1024 against
        /// 512 rows: prefill 5.0% faster at 4K and 5.4% at 16K, within noise at 512 (two rounds, 0.956 and 1.032: one step either
        /// way); decode 0.997-1.001; peak wired 17.1 / 19.0 GB at 4K / 16K against 16.7 / 18.0, under legacy's 18.1 / 19.9.
        /// 2048 rows against 1024: prefill +2.7% (4K) and +1.4% (16K), but decode 0.9% slower and 18.9-20.9 GB wired, past
        /// legacy's peak. ggml_metal keeps 512 (2048 rows measured 0.99-1.00 there).
        /// </summary>
        internal const int MlxRingStepRows = 1024;

        /// <summary>
        /// The rows a prompt step writes into a sliding-window ring on direct CUDA, one device: 1024 (3072-cell rings). The
        /// executor's prompt-size products of the 30B's IQ2, IQ3 and Q4_K weights dequantize each weight to F16 once a call for
        /// a cuBLAS GEMM, a cost a step pays whatever its rows, so a prompt in 1024-row steps pays it half as often. Measured on
        /// an RTX PRO 4500 Blackwell (32 GB), against 512 rows (docs/design/muse-glimmer-unified.md §5.8): the probe's prefill
        /// 1.25x at 4K and 1.22x at 16K (UD-IQ2_XXS; 1.26x / 1.23x on UD-Q4_K_XL), decode 0.995-1.004x; time to first token at
        /// 4K 0.82-0.86x in the CLI and the server. Every scenario's peak VRAM within legacy's +5% (S7's bound): the server holding
        /// a 4K prompt 15,816 MiB against legacy's 15,106-15,369 and 21,520 against 20,746, the CLI and the probe under
        /// legacy's. 2048 rows prefilled 1.39x / 1.32x but its server peaked 5.6-7.5% over legacy's: the step's activations
        /// (about 0.8 GB more, held in the pool's cache after the step) and 59 MiB more ring a kept 4K conversation. Placements
        /// over several devices keep the shared rule (two GPUs were not measured); ggml_cuda and ggml_vulkan keep 512.
        /// </summary>
        internal const int CudaRingStepRows = 1024;

        protected override int RingStepRowsForPlacement => _backend switch
        {
            BackendType.Mlx => MlxRingStepRows,
            BackendType.Cuda when Mesh.Stages == 1 && Mesh.TensorParallel == 1 => CudaRingStepRows,
            _ => base.RingStepRowsForPlacement,
        };

        /// <summary>
        /// The default draft window, by the trunk's verify economics. Measured on an M5 Pro (UD-IQ2_XXS, the 1.63 GB dflash-kquant
        /// drafter, greedy, five prompts x 512 tokens; docs/design/muse-glimmer-unified.md §5.4):
        /// <list type="bullet">
        /// <item>ggml_metal: 7 drafts (an 8-row verify), for the drafter and n-gram drafts alike. A verify costs 76-80 ms from 4 to 8
        /// rows (past one row the IQ2/IQ3 products take the matrix kernel), 135-147 ms at 9 and 152-160 ms at 16, a plain step
        /// 46 ms, a block 25-29 ms at any width. With the governor 7 drafts 34.1 tok/s in all (35.0 with the GGML builder's argmax
        /// top-1), 5 drafts 31.6, 7 at the block gate 0.35 32.8, plain 21.7, legacy at its default 30.7; ungoverned, before the
        /// prompt-follow fix, 3 / 5 / 7 / 15 drafts 24.8 / 27.3 / 27.9 / 17.3. n-gram at 7 drafts 31.6 against 27.5 at 8.</item>
        /// <item>MLX: the drafter's whole block (15 drafts), whose verify costs 79-88 ms at any width up to 16 rows: 39.8 tok/s
        /// in all against 37.0 at 8 drafts and 35.0 at 7 (all ungated), 33.4 at the speculator's 8 with the gate, plain 21.1,
        /// legacy's DFlash 31.7. n-gram keeps the speculator's 8 there (31.7 tok/s in all against 30.8 at 7).</item>
        /// <item>ggml_cuda: the drafter's whole block (15 drafts) at the block gate 0.35 (RTX PRO 4500, sm_120; governed):
        /// <list type="bullet">
        /// <item>UD-IQ2_XXS: 127.1 tok/s in all against 121.0 at the speculator's 8 (plain 61.0, legacy's DFlash 88.0).</item>
        /// <item>UD-Q4_K_XL: 146.4 against 130.5 (plain 47.5, legacy 94.0).</item>
        /// <item>At or above the 8-draft default on every prompt but one, IQ2's essay (-0.9%).</item>
        /// <item>The gate keeps it: without it 15 drafts is +18% on Q4 but -6% on IQ2, whose essay falls to plain.</item>
        /// <item>The verify costs 22.8 / 27.8 / 37.0 / 31.0 ms at 4 / 6 / 8 / 16 rows on IQ2 (8 rows is the dearest shape there:
        /// 7 drafts ungated is the worst arm, 86.5), and 21.8-24.9 ms at 4-16 rows on Q4.</item>
        /// </list></item>
        /// <item>Direct CUDA: the same, not worse there. UD-IQ2_XXS runs at 21.8 tok/s against 20.6-20.8 at 8 drafts (plain
        /// 21.0); its IQ verify rows cost 124-348 ms against a 47-ms plain step, so the drafter barely pays.</item>
        /// </list>
        /// Elsewhere the speculator's default (8) until measured there.
        /// </summary>
        public override int SpecPreferredDraftWindow => _backend switch
        {
            BackendType.GgmlMetal => 7,
            BackendType.Mlx or BackendType.GgmlCuda or BackendType.Cuda when HasDFlash => DFlashSettings.MaxDraftTokens,
            _ => base.SpecPreferredDraftWindow,
        };

        /// <summary>No confidence gate for the drafter on ggml_metal and MLX, where a verify row is nearly free (see
        /// <see cref="SpecPreferredDraftWindow"/>): on ggml_metal 7 drafts without it 34.1 tok/s, with the block default's 0.35
        /// 32.8, lower on every prompt; on MLX 8 drafts without it 37.0, with it 33.4. The cost governor parks the drafter where
        /// its drafts lose (prose on ggml_metal: 21.6 tok/s against 21.9 plain).</summary>
        public override float SpecPreferredBlockGate => HasDFlash && _backend is BackendType.GgmlMetal or BackendType.Mlx
            ? 0f : base.SpecPreferredBlockGate;

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
        /// What a tensor-parallel degree must also meet, as the legacy model checked it: the
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
