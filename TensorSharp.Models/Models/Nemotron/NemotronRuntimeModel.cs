// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace TensorSharp.Models
{
    /// <summary>
    /// Nemotron-H (GGUF <c>nemotron_h</c>, <c>nemotron_h_moe</c>, <c>nemotron_h_omni</c>: Nemotron-H 8B/47B, Nemotron 3 Nano / Omni,
    /// Nemotron 3.5 Lightning, Bolt and Super VL) on the shared decoder runtime, which provides every placement (one device,
    /// <c>--tp</c>, whole-layer stages and their hybrid), continuous batching, the prefix cache, checkpoints and speculation;
    /// its steps run as whole-forward graphs on every executor (<see cref="StandardDecoderModel.RequiresGraphs"/>). The recipe
    /// (llama.cpp <c>src/models/nemotron-h.cpp</c>):
    /// <list type="bullet">
    /// <item>one block a layer, chosen by <c>attention.head_count_kv[l]</c> and <c>feed_forward_length[l]</c>: Mamba-2 (both 0),
    /// attention (KV heads, no feed-forward) or a feed-forward (a width); each block is <c>attn_norm</c> then the block, added to
    /// the stream (<see cref="StandardDecoderModel.LayerHasMixer"/>, <see cref="StandardDecoderModel.LayerHasFeedForward"/>);</item>
    /// <item>Mamba-2: <c>ssm_in</c> to [z | xBC | dt], a causal depthwise conv (<c>ssm.conv_kernel</c>, with its bias) and SiLU over
    /// xBC, ggml's selective scan with dt + <c>ssm_dt.bias</c> and <c>ssm_a</c>, the <c>ssm_d</c> skip, silu(z) gate, a grouped
    /// RMSNorm (<c>ssm.group_count</c> groups, <c>ssm_norm</c>) and <c>ssm_out</c>;</item>
    /// <item>attention: GQA without positional encoding at <c>attention.scale</c> (or 1/sqrt(head_dim));</item>
    /// <item>dense feed-forward: <c>ffn_up</c>, ReLU^2, <c>ffn_down</c>;</item>
    /// <item>routed experts: sigmoid router (<c>ffn_gate_inp</c>) whose <c>exp_probs_b</c> bias picks the top k (each layer's own
    /// expert count and top-k on Bolt), the unbiased probabilities as weights (renormalized when <c>expert_weights_norm</c>,
    /// times <c>expert_weights_scale</c>); ReLU^2 experts (<c>ffn_up_exps</c> / <c>ffn_down_exps</c>), optionally inside a latent
    /// bottleneck (<c>ffn_latent_in</c> / <c>ffn_latent_out</c>, or <c>ffn_latent_down</c> / <c>ffn_latent_up</c>), NVFP4 per-expert
    /// scales (<c>ffn_up_exps.scale</c> before ReLU^2, <c>ffn_down_exps.scale</c> through the route weights), plus a ReLU^2 shared
    /// expert (<c>ffn_up_shexp</c> / <c>ffn_down_shexp</c>);</item>
    /// <item>the head: <c>output_norm</c> and <c>output</c> (or the tied embedding).</item>
    /// </list>
    /// Under tensor parallelism a rank owns a contiguous block of Mamba-2 groups and the heads that read them, so its scan, conv
    /// and grouped norm are self-contained and only <c>ssm_out</c>'s partial is reduced, like attention's output projection.
    /// The Nemotron 3.5 MTP head (a <c>nextn</c> block shipped as its own GGUF, <c>--draft-model</c>) drafts as the runtime's
    /// draft layer; DFlash / DSpark drafters attach through the runtime.
    /// </summary>
    public sealed partial class NemotronRuntimeModel : StandardDecoderModel
    {
        /// <summary>A layer's block.</summary>
        internal enum Block { Mamba2, Attention, FeedForward }

        private readonly Block[] _block;
        private readonly int[] _layerKv;
        private readonly int[] _layerFfn;
        private readonly int _attentionKv;
        private readonly float _attentionScale;

        // Mamba-2 geometry (global).
        private readonly int _convKernel;
        private readonly int _dInner;
        private readonly int _dState;
        private readonly int _nHead;
        private readonly int _nGroup;
        private readonly int _ssmHeadDim;

        // Routed experts (per layer: Bolt's experts and top-k differ by layer).
        private readonly int[] _layerExperts;
        private readonly int[] _layerExpertsUsed;
        private readonly bool _expertWeightsNorm;
        private readonly float _expertWeightsScale;

        /// <summary>The draft (MTP) layer, past the decoder stack: 1 with a separate MTP head, else 0.</summary>
        private readonly int _draftLayers;
        private GgufFile _separateDraftHead;

        /// <summary>Per (layer, rank) Mamba-2 constants in the rank's local channel order, on the rank's device.</summary>
        private Mamba2Constants[][] _mamba;

        public NemotronRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "nemotron_h";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _convKernel = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");
            _dInner = (int)_gguf.GetUint32($"{arch}.ssm.inner_size");
            _dState = (int)_gguf.GetUint32($"{arch}.ssm.state_size");
            _nHead = (int)_gguf.GetUint32($"{arch}.ssm.time_step_rank");
            _nGroup = (int)_gguf.GetUint32($"{arch}.ssm.group_count");
            _attentionScale = _gguf.GetFloat32($"{arch}.attention.scale", 0f);

            // A trunk GGUF that carries its NextN block counts it in block_count; the separate MTP head brings its own.
            int nextn = (int)_gguf.GetUint32($"{arch}.nextn_predict_layers", 0);
            Config.NumLayers -= nextn;
            int layers = Config.NumLayers;
            uint[] kv = _gguf.GetUint32Array($"{arch}.attention.head_count_kv");
            uint[] ffn = _gguf.GetUint32Array($"{arch}.feed_forward_length");
            uint[] heads = _gguf.GetUint32Array($"{arch}.attention.head_count");
            _block = new Block[layers];
            _layerKv = new int[layers];
            _layerFfn = new int[layers];
            for (int l = 0; l < layers; l++)
            {
                _layerKv[l] = kv != null && l < kv.Length ? (int)kv[l] : Config.NumKVHeads;
                _layerFfn[l] = ffn != null && l < ffn.Length ? (int)ffn[l] : 0;
                _block[l] = _layerFfn[l] > 0 ? Block.FeedForward : _layerKv[l] > 0 ? Block.Attention : Block.Mamba2;
            }
            // A stack with Mamba-2 layers carries their geometry (one without, a test's attention + MoE layout, needs none).
            if (Array.IndexOf(_block, Block.Mamba2) >= 0
                && (_nHead <= 0 || _nGroup <= 0 || _dInner % _nHead != 0 || _nHead % _nGroup != 0 || _convKernel < 1 || _dState <= 0))
                throw new InvalidDataException($"{arch}: the Mamba-2 geometry (inner {_dInner}, {_nHead} heads, {_nGroup} groups, " +
                    $"state {_dState}, conv {_convKernel}) is inconsistent.");
            _ssmHeadDim = _nHead > 0 ? _dInner / _nHead : 0;
            // The attention layers' geometry is the model's (per-layer arrays read as their first entry, a Mamba-2 layer's 0).
            int firstAttention = Array.IndexOf(_block, Block.Attention);
            _attentionKv = firstAttention >= 0 ? _layerKv[firstAttention] : Math.Max(1, Config.NumKVHeads);
            if (heads != null && firstAttention >= 0 && firstAttention < heads.Length && heads[firstAttention] > 0)
                Config.NumHeads = (int)heads[firstAttention];
            Config.NumKVHeads = _attentionKv;
            for (int l = 0; l < layers; l++)
                if (_block[l] == Block.Attention && (_layerKv[l] != _attentionKv || (heads != null && l < heads.Length && heads[l] != Config.NumHeads)))
                    throw new NotSupportedException($"{arch}: attention layers of different head counts (layer {l}) are not supported.");
            int headDim = Config.KeyLength > 0 ? Config.KeyLength : Config.HiddenSize / Math.Max(1, Config.NumHeads);
            int valueDim = Config.ValueLength > 0 ? Config.ValueLength : headDim;
            Config.IntermediateSize = _layerFfn.DefaultIfEmpty(0).Max();

            // Routed experts: each layer's count from its stacked tensor's shape, its top-k from a per-layer array.
            _expertWeightsNorm = _gguf.GetBool($"{arch}.expert_weights_norm", false);
            _expertWeightsScale = _gguf.GetFloat32($"{arch}.expert_weights_scale", 1.0f);
            _layerExperts = new int[layers + 1];
            _layerExpertsUsed = new int[layers + 1];
            ResolveExpertLayout(arch, layers);

            // The NextN/MTP block drafts on one device, the last of this process's whole-layer stages, or (unsharded) on
            // rank 0 of a tensor-parallel group: the trunk's own block, or one shipped as its own GGUF (--draft-model).
            bool drafts = tpGroup == null || tpGroup is ILayerSplitGroup || tpGroup.NodeCount == 1;
            _draftLayers = drafts ? Math.Min(nextn, 1) : 0;
            if (_draftLayers > 0)
                ResolveDraftExperts(_gguf, arch);
            else if (drafts && !string.IsNullOrWhiteSpace(draftModelPath) && !SpeculativeDraftHeadLoader.IsDFlashDrafter(draftModelPath)
                && OpenMtpHead(draftModelPath, arch) is { } head)
            {
                _draftLayers = 1;
                _separateDraftHead = head;
                _loadingDraftHead = true;
                try
                {
                    LoadAdditionalWeights(head);
                }
                finally
                {
                    _loadingDraftHead = false;
                }
                ResolveDraftExperts(head, arch);
            }
            DrafterReadsTrunkEmbedding = _draftLayers > 0 || !string.IsNullOrWhiteSpace(draftModelPath);
            ParseTokenizer();

            int mamba = _block.Count(b => b == Block.Mamba2), attention = _block.Count(b => b == Block.Attention);
            Console.WriteLine($"Model: {arch} (shared decoder runtime), Layers={layers} ({mamba} Mamba-2, {attention} attention, " +
                $"{layers - mamba - attention} feed-forward{(HasExperts ? ", MoE" : ", dense")}), Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={_attentionKv}, HeadDim={headDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  Mamba-2: inner={_dInner}, state={_dState}, heads={_nHead} x {_ssmHeadDim}, groups={_nGroup}, conv={_convKernel}" +
                (_attentionScale != 0 ? $"; attention scale {_attentionScale}" : "") +
                (_draftLayers > 0 ? $"; MTP head {Path.GetFileName(draftModelPath)}" : ""));
            if (HasExperts) Console.WriteLine("  " + DescribeExperts());

            InitializeDecoder(headDim, valueDim, string.Create(CultureInfo.InvariantCulture,
                $"nemotron_h|{Config.NumHeads}|{_attentionKv}|{headDim}|{valueDim}|{Config.Eps:R}|{_attentionScale:R}|{_convKernel}|{_dInner}|" +
                $"{_dState}|{_nHead}|{_nGroup}|{LayerKinds()}|{string.Join(",", _layerExpertsUsed)}|{_expertWeightsNorm}|{_expertWeightsScale:R}"));
            try
            {
                ValidateLayerWeights();
                BuildMamba2Constants();
            }
            catch
            {
                Dispose();
                throw;
            }
            if (!string.IsNullOrWhiteSpace(draftModelPath) && SpeculativeDraftHeadLoader.IsDFlashDrafter(draftModelPath))
                AttachDraftModel(draftModelPath);
        }

        private bool HasExperts => Array.Exists(_layerExperts, e => e > 0);

        private string LayerKinds() => new(_block.Select(b => b switch { Block.Mamba2 => 'M', Block.Attention => '*', _ => '-' }).ToArray());

        // ---------------------------------------------------------------- the runtime's declarations

        protected override bool RequiresGraphs => true;
        protected override bool AttentionIsPositionless => true;
        protected override bool RecurrentMixerLowers => true;
        /// <summary>The Mamba-2 mixer traces one rank's groups (its sharded projections, per-rank constants and states).</summary>
        protected override bool RanksLower => true;
        protected override int DraftLayerCount => _draftLayers;

        protected override bool LayerHasMixer(int layer) => _block[layer] != Block.FeedForward;
        protected override bool LayerHasFeedForward(int layer) => _block[layer] == Block.FeedForward;
        protected override bool IsRecurrentLayer(int layer) => _block[layer] == Block.Mamba2;
        protected override int LayerKvHeads(int layer) => _attentionKv;

        protected override float AttentionScale(int layer)
            => _attentionScale != 0 ? _attentionScale : 1.0f / MathF.Sqrt(KeyDimOf(layer));

        /// <summary>Every trunk block reads <c>attn_norm</c>; the MTP block's feed-forward its <c>post_attention_norm</c>.</summary>
        protected override string FfnNormWeight(int layer)
            => layer < Config.NumLayers ? LayerName(layer, "attn_norm.weight") : LayerName(layer, "post_attention_norm.weight");

        protected override string FingerprintExtras => _block == null ? "|nemotron_h"
            : string.Create(CultureInfo.InvariantCulture, $"|nemotron_h|layers={LayerKinds()}|ssm={_dInner}/{_dState}/{_nHead}/{_nGroup}/{_convKernel}");

        /// <summary>Not implemented: Nemotron-H's blocks run in whole-forward graphs only.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
            => throw new NotSupportedException($"{Config.Architecture} runs whole-forward graphs only.");

        /// <summary>A separate MTP head's tensors are loading: only its block (the file's embedding and head are the trunk's).</summary>
        private bool _loadingDraftHead;

        /// <summary>Decoder layers and the draft layer (a trunk's NextN block it does not draft with is not loaded); from a
        /// separate MTP head only its block.</summary>
        protected override bool ShouldLoadWeight(GgufTensorInfo info)
        {
            bool inLayer = LayerSplitPlan.TryGetLayer(info.Name, out int layer);
            if (_loadingDraftHead) return inLayer && layer == Config.NumLayers && base.ShouldLoadWeight(info);
            return !(inLayer && layer >= Config.NumLayers + _draftLayers) && base.ShouldLoadWeight(info);
        }

        /// <summary>
        /// The Mamba-2 coefficients the graph reads as F32 load as F32 whatever a converter stored them as (the conv taps and
        /// bias, the decay, the skip, dt's bias, the grouped norm's gammas); so do the router bias and the per-expert scales.
        /// Projections stay compressed.
        /// </summary>
        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            string n = info.Name;
            if (n.EndsWith(".ssm_conv1d.weight", StringComparison.Ordinal) || n.EndsWith(".ssm_norm.weight", StringComparison.Ordinal)
                || n.EndsWith(".ssm_a", StringComparison.Ordinal) || n.EndsWith(".ssm_d", StringComparison.Ordinal)
                || n.EndsWith(".ssm_dt.bias", StringComparison.Ordinal) || n.EndsWith(".exp_probs_b.bias", StringComparison.Ordinal)
                || n.EndsWith("_exps.scale", StringComparison.Ordinal))
                return false;
            return base.IsQuantizedLinearWeight(info);
        }

        protected override IEnumerable<string> ShardingProblems(int degree)
        {
            if (Array.IndexOf(_block, Block.Mamba2) >= 0 && _nGroup % degree != 0) yield return $"{_nGroup} Mamba-2 groups";
            for (int l = 0; l < Config.NumLayers; l++)
                if (_block[l] == Block.FeedForward && _layerExperts[l] == 0 && _layerFfn[l] % degree != 0)
                {
                    yield return $"layer {l}'s feed-forward width {_layerFfn[l]}";
                    yield break;
                }
        }

        /// <summary>A Mamba-2 layer's projections for the bound stage group: <c>ssm_in</c> keeps each rank's rows (its heads' z,
        /// x and dt, its groups' B and C), <c>ssm_out</c> its heads' input columns.</summary>
        protected override void ShardRecurrentLayer(int layer)
        {
            int degree = ShardDegree;
            ShardOutputRows(LayerName(layer, "ssm_in.weight"), r => InProjectionRowsOf(r, degree));
            // A rank's heads are contiguous: their input columns are one block, cut on whole quant blocks (per-head groups narrower
            // than a block would be re-encoded or expanded to F32).
            ShardInputColumnGroups(LayerName(layer, "ssm_out.weight"), r => new[] { r }, _dInner / degree);
        }

        /// <summary>The Mamba-2 heads of global rank <paramref name="rank"/>: the heads of its contiguous block of groups.</summary>
        private int[] HeadsOf(int rank, int degree)
        {
            int width = _nHead / degree;
            return Enumerable.Range(rank * width, width).ToArray();
        }

        private int[] GroupsOf(int rank, int degree)
        {
            int width = _nGroup / degree;
            return Enumerable.Range(rank * width, width).ToArray();
        }

        /// <summary>The channels of xBC (and the conv) a rank keeps, in its local order: its heads' x, its groups' B, then C.</summary>
        private int[] ConvChannelsOf(int rank, int degree)
        {
            var rows = new List<int>();
            foreach (int h in HeadsOf(rank, degree))
                for (int i = 0; i < _ssmHeadDim; i++) rows.Add(h * _ssmHeadDim + i);
            foreach (int g in GroupsOf(rank, degree))
                for (int i = 0; i < _dState; i++) rows.Add(_dInner + g * _dState + i);
            foreach (int g in GroupsOf(rank, degree))
                for (int i = 0; i < _dState; i++) rows.Add(_dInner + _nGroup * _dState + g * _dState + i);
            return rows.ToArray();
        }

        /// <summary>The rows of <c>ssm_in</c> [z | xBC | dt] a rank keeps, in its local order: z, xBC (<see cref="ConvChannelsOf"/>),
        /// dt.</summary>
        private int[] InProjectionRowsOf(int rank, int degree)
        {
            var rows = new List<int>();
            foreach (int h in HeadsOf(rank, degree))
                for (int i = 0; i < _ssmHeadDim; i++) rows.Add(h * _ssmHeadDim + i);
            foreach (int c in ConvChannelsOf(rank, degree)) rows.Add(_dInner + c);
            int dt = 2 * _dInner + 2 * _nGroup * _dState;
            foreach (int h in HeadsOf(rank, degree)) rows.Add(dt + h);
            return rows.ToArray();
        }

        // ---------------------------------------------------------------- text LoRA

        // The projections a text LoRA adapter may adapt: every product the graphs run through GraphStep.MatMul (attn_q/k/v
        // through the fused attn_qkv). Routed experts run as MulMatId and are never adapted.
        private static readonly HashSet<string> s_textLoraProjections = new(StringComparer.Ordinal)
        {
            "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output", "ffn_up", "ffn_down", "ssm_in", "ssm_out", "ffn_up_shexp",
            "ffn_down_shexp", "ffn_latent_in", "ffn_latent_out", "ffn_latent_down", "ffn_latent_up", "ffn_gate_inp",
        };

        protected override bool SupportsTextLora => true;

        /// <summary>Every sidecar scale applies under tensor parallelism: a projection's on its shards (GraphStep.MatMul reads each
        /// shard's), the per-expert up and down vectors per selected expert, which a slice of the expert's width shares.</summary>
        protected override bool SupportsTensorParallelWeightScales => true;

        protected override bool AppliesTextLoraInGraphs => true;

        protected override string TextLoraWeightRefusal(string weightName)
            => weightName == "output.weight" || IsDecoderLayerProjection(weightName, s_textLoraProjections)
                ? null
                : "is not one of the projections Nemotron applies a LoRA delta to";

        /// <summary>An adapter may name the latent projections either way (<c>ffn_latent_in</c> / <c>_out</c>, or the Super / Bolt
        /// GGUFs' <c>ffn_latent_down</c> / <c>_up</c>): the model's own name is the one its file has.</summary>
        protected override string NormalizeTextLoraTarget(string target)
        {
            static string Swap(string name, string from, string to) => name.EndsWith(from, StringComparison.Ordinal) ? name[..^from.Length] + to : null;
            string other = Swap(target, ".ffn_latent_in.weight", ".ffn_latent_down.weight") ?? Swap(target, ".ffn_latent_out.weight", ".ffn_latent_up.weight")
                ?? Swap(target, ".ffn_latent_down.weight", ".ffn_latent_in.weight") ?? Swap(target, ".ffn_latent_up.weight", ".ffn_latent_out.weight");
            return other != null && !HasWeight(target) && HasWeight(other) ? other : target;
        }

        // ---------------------------------------------------------------- the MTP head

        /// <summary>A Nemotron MTP head GGUF (<c>nextn_predict_layers</c>, its <c>blk.{NumLayers}.nextn.*</c> block), validated and
        /// open, or null when <paramref name="path"/> is another drafter.</summary>
        private GgufFile OpenMtpHead(string path, string arch)
        {
            var file = new GgufFile(path);
            try
            {
                string headArch = file.GetString("general.architecture");
                string eh = $"blk.{Config.NumLayers}.nextn.eh_proj.weight";
                if (!file.Tensors.ContainsKey(eh))
                {
                    file.Dispose();
                    return null;
                }
                if (!string.Equals(headArch, arch, StringComparison.Ordinal) || file.GetUint32($"{headArch}.nextn_predict_layers", 0) < 1)
                    throw new InvalidDataException($"'{Path.GetFileName(path)}' is a {headArch} NextN head, not one of this {arch} trunk.");
                if ((int)file.GetUint32($"{headArch}.embedding_length") != Config.HiddenSize)
                    throw new InvalidDataException($"'{Path.GetFileName(path)}' drafts {file.GetUint32($"{headArch}.embedding_length")}-wide rows, the trunk's are {Config.HiddenSize}.");
                return file;
            }
            catch
            {
                file.Dispose();
                throw;
            }
        }

        private void ResolveDraftExperts(GgufFile head, string arch)
        {
            int layer = Config.NumLayers;
            _layerExperts[layer] = StackedExpertCount(head, layer);
            _layerExpertsUsed[layer] = _layerExperts[layer] > 0 ? (int)head.GetUint32($"{arch}.expert_used_count", (uint)_layerExpertsUsed.Max()) : 0;
        }

        // ---------------------------------------------------------------- routed experts

        private void ResolveExpertLayout(string arch, int layers)
        {
            uint[] usedArray = _gguf.GetUint32Array($"{arch}.expert_used_count");
            int usedScalar = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            for (int l = 0; l < layers; l++)
            {
                if (_block[l] != Block.FeedForward) continue;
                int experts = StackedExpertCount(_gguf, l);
                if (experts == 0) continue;
                long used = usedArray != null && usedArray.Length > 1 ? (l < usedArray.Length ? usedArray[l] : 0) : usedScalar;
                if (used < 1 || used > experts)
                    throw new InvalidDataException(
                        $"Nemotron MoE layer {l}: expert_used_count {used} is outside 1..{experts} (the layer's expert count), so no route " +
                        "can be selected. The GGUF's MoE metadata is inconsistent.");
                _layerExperts[l] = experts;
                _layerExpertsUsed[l] = (int)used;
            }
        }

        /// <summary>The experts of <paramref name="layer"/>'s stacked up projection in <paramref name="file"/> (0: a dense layer).</summary>
        private static int StackedExpertCount(GgufFile file, int layer)
            => file.Tensors.TryGetValue($"blk.{layer}.ffn_up_exps.weight", out GgufTensorInfo up) && up.Shape.Length >= 3 ? (int)up.Shape[2] : 0;

        /// <summary>The load line's MoE summary: uniform, or ranges (Bolt's per-layer experts and top-k).</summary>
        private string DescribeExperts()
        {
            var moe = Enumerable.Range(0, Config.NumLayers).Where(l => _layerExperts[l] > 0).ToArray();
            string Range(IEnumerable<int> values) { int min = values.Min(), max = values.Max(); return min == max ? $"{min}" : $"{min}-{max}"; }
            bool latent = moe.Any(l => HasWeight(LatentInName(l)) || _gguf.Tensors.ContainsKey(LatentInName(l)) || _gguf.Tensors.ContainsKey(LayerName(l, "ffn_latent_down.weight")));
            bool shared = moe.Any(l => _gguf.Tensors.ContainsKey(LayerName(l, "ffn_up_shexp.weight")));
            return $"MoE: {moe.Length} layers, {Range(moe.Select(l => _layerExperts[l]))} experts, top-{Range(moe.Select(l => _layerExpertsUsed[l]))}" +
                   $"{(latent ? ", latent" : "")}{(shared ? ", shared expert" : "")}; weights {(_expertWeightsNorm ? "renormalized, " : "")}x{_expertWeightsScale}";
        }

        private string LatentInName(int layer)
            => _gguf.Tensors.ContainsKey(LayerName(layer, "ffn_latent_down.weight")) ? LayerName(layer, "ffn_latent_down.weight") : LayerName(layer, "ffn_latent_in.weight");

        private string LatentOutName(int layer)
            => _gguf.Tensors.ContainsKey(LayerName(layer, "ffn_latent_up.weight")) ? LayerName(layer, "ffn_latent_up.weight") : LayerName(layer, "ffn_latent_out.weight");

        // ---------------------------------------------------------------- validation and constants

        /// <summary>Every owned layer carries the weights its block reads: refused at load, not at the first forward.</summary>
        private void ValidateLayerWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                var required = new List<string> { "attn_norm.weight" };
                switch (_block[l])
                {
                    case Block.Mamba2:
                        required.AddRange(new[] { "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_out.weight" });
                        if (Mesh.TensorParallel == 1) required.Add("ssm_in.weight");
                        break;
                    case Block.Attention:
                        required.Add("attn_output.weight");
                        break;
                    default:
                        required.Add(_layerExperts[l] > 0 ? "ffn_gate_inp.weight" : "ffn_down.weight");
                        break;
                }
                foreach (string suffix in required)
                    if (!HasWeight(LayerName(l, suffix)))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} ({_block[l]}) is missing '{LayerName(l, suffix)}'.");
                if (_block[l] == Block.Attention && !HasWeight(LayerName(l, "attn_qkv.weight")) && !HasWeight(LayerName(l, "attn_q.weight")))
                    throw new InvalidOperationException($"{Config.Architecture} layer {l} has no query projection.");
            }
        }

        /// <summary>One rank's Mamba-2 constants of one layer, F32 on the rank's device in its local order.</summary>
        private sealed class Mamba2Constants
        {
            public Tensor ConvWeight;   // [channels, kernel]
            public Tensor ConvBias;     // [channels] or null
            public Tensor DtBias;       // [heads]
            public Tensor A;            // [heads]
            public Tensor D;            // [heads] or null
            public Tensor Norm;         // [groups * groupWidth] or null
        }

        private void BuildMamba2Constants()
        {
            int layers = Config.NumLayers, ranks = Mesh.TensorParallel, degree = ShardDegree, offset = ShardRankOffset;
            int channels = _dInner + 2 * _nGroup * _dState;
            _mamba = new Mamba2Constants[layers][];
            for (int l = 0; l < layers; l++)
            {
                if (_block[l] != Block.Mamba2 || !OwnsLayer(l)) continue;
                float[] conv = Floats(LayerName(l, "ssm_conv1d.weight"), (long)channels * _convKernel);
                float[] convBias = TryFloats(LayerName(l, "ssm_conv1d.bias"), channels);
                float[] dt = Floats(LayerName(l, "ssm_dt.bias"), _nHead);
                float[] a = Floats(LayerName(l, "ssm_a"), _nHead);
                float[] d = TryFloats(LayerName(l, "ssm_d"), _nHead);
                float[] norm = TryFloats(LayerName(l, "ssm_norm.weight"), _dInner);
                int stage = Mesh.StageOf(l);
                _mamba[l] = new Mamba2Constants[ranks];
                for (int r = 0; r < ranks; r++)
                {
                    IAllocator alloc = Mesh.Allocator(stage, r);
                    int[] ch = ConvChannelsOf(offset + r, degree);
                    int[] hs = HeadsOf(offset + r, degree);
                    var w = new float[ch.Length * _convKernel];
                    for (int c = 0; c < ch.Length; c++)
                        Array.Copy(conv, (long)ch[c] * _convKernel, w, (long)c * _convKernel, _convKernel);
                    var k = new Mamba2Constants
                    {
                        ConvWeight = Upload(alloc, w, ch.Length, _convKernel),
                        ConvBias = convBias == null ? null : Upload(alloc, ch.Select(c => convBias[c]).ToArray()),
                        DtBias = Upload(alloc, hs.Select(h => dt[h]).ToArray()),
                        A = Upload(alloc, hs.Select(h => a[h]).ToArray()),
                        D = d == null ? null : Upload(alloc, hs.Select(h => d[h]).ToArray()),
                    };
                    if (norm != null)
                    {
                        // The grouped norm's gammas of the rank's heads (its groups'), in order.
                        var gn = new float[hs.Length * _ssmHeadDim];
                        for (int j = 0; j < hs.Length; j++)
                            Array.Copy(norm, (long)hs[j] * _ssmHeadDim, gn, (long)j * _ssmHeadDim, _ssmHeadDim);
                        k.Norm = Upload(alloc, gn);
                    }
                    _mamba[l][r] = k;
                }
            }
        }

        private float[] Floats(string name, long count)
            => TryFloats(name, count) ?? throw new InvalidOperationException($"{Config.Architecture}: '{name}' is missing.");

        /// <summary>The first <paramref name="count"/> values of F32 weight <paramref name="name"/> (null when absent); a weight
        /// sharded at load (none of these) is read from its source.</summary>
        private float[] TryFloats(string name, long count)
        {
            if (!_weights.TryGetValue(name, out Tensor t)) return null;
            if (t.ElementCount() < count)
                throw new InvalidDataException($"{Config.Architecture}: '{name}' holds {t.ElementCount()} values, {count} expected.");
            return t.GetElementsAsFloat(checked((int)count));
        }

        private Tensor Upload(IAllocator allocator, float[] values, params long[] sizes)
        {
            var t = sizes.Length == 0 ? new Tensor(allocator, DType.Float32, values.Length) : new Tensor(allocator, DType.Float32, sizes);
            t.SetElementsAsFloat(values);
            HostWritten(t);
            return t;
        }

        protected override RecurrentState CreateRecurrentState(RankScope scope, int layer)
        {
            int degree = ShardDegree;
            int heads = _nHead / degree, groups = _nGroup / degree;
            int channels = heads * _ssmHeadDim + 2 * groups * _dState;
            return new Mamba2State(scope.Allocator, ExecutorFor(scope.Allocator), t => HostPointer(t), _convKernel - 1, channels, heads,
                _ssmHeadDim, _dState);
        }

        private static unsafe IntPtr HostPointer(Tensor tensor) => (IntPtr)GetFloatPtr(tensor);

        /// <summary>A verify window keeps every Mamba-2 layer's state after each of its rows and its conv inputs (up to eight
        /// rows: an MTP or DFlash window).</summary>
        protected override long SpeculationReserveBytes
        {
            get
            {
                if (_draftLayers == 0) return 0;
                const int rows = 8;
                long state = (long)_nHead * _ssmHeadDim * _dState, channels = _dInner + 2L * _nGroup * _dState;
                long perLayer = (rows * state + channels * (_convKernel - 1 + rows)) * sizeof(float);
                return _block.Count(b => b == Block.Mamba2) * perLayer;
            }
        }

        public override void Dispose()
        {
            if (_mamba != null)
                foreach (Mamba2Constants[] layer in _mamba)
                    if (layer != null)
                        foreach (Mamba2Constants k in layer)
                        {
                            k?.ConvWeight?.Dispose();
                            k?.ConvBias?.Dispose();
                            k?.DtBias?.Dispose();
                            k?.A?.Dispose();
                            k?.D?.Dispose();
                            k?.Norm?.Dispose();
                        }
            _mamba = null;
            DisposeMultimodal();
            _separateDraftHead?.Dispose();
            _separateDraftHead = null;
            base.Dispose();
        }

        // ---------------------------------------------------------------- tests

        internal Block BlockOfLayerForTests(int layer) => _block[layer];

        internal (int Experts, int Used) ExpertLayoutForTests(int layer) => (_layerExperts[layer], _layerExpertsUsed[layer]);
    }
}
