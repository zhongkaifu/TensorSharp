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

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen 3.5 / 3.6 (GGUF <c>qwen35</c>, <c>qwen35moe</c>) on the shared decoder runtime: a
    /// hybrid stack of Gated DeltaNet (recurrent) layers and gated full-attention layers, with
    /// dense SwiGLU or routed-expert FFNs (softmax top-K, renormalized, plus a sigmoid-gated
    /// shared expert). The runtime provides every placement (tensor parallelism, layer split,
    /// hybrid), continuous batching and expert offload; this class is the recipe:
    /// <list type="bullet">
    /// <item>attention: <c>attn_q</c> emits <c>[q_h | gate_h]</c> per head (the runtime's gated
    /// attention), per-head Q/K RMSNorm BEFORE a partial NeoX rotary (<c>rope.dimension_count</c>
    /// of the head), output multiplied by <c>sigmoid(gate)</c>;</item>
    /// <item>Gated DeltaNet: <c>attn_qkv</c> through a causal depthwise conv (kernel
    /// <c>ssm.conv_kernel</c>) and SiLU, per-head L2-normalized q/k (value head h reads key head
    /// h mod nK), the delta rule <c>S = S*exp(softplus(alpha+dt)*A); S += k (x) (v - S k)*sigmoid(beta)</c>,
    /// output <c>rmsnorm(S q) * ssm_norm * silu(z)</c> through <c>ssm_out</c>.</item>
    /// </list>
    /// Under tensor parallelism rank r owns a contiguous block of key heads and the value heads
    /// that read them (block-cyclic), so its recurrence is self-contained and only
    /// <c>ssm_out</c>'s partial sum is reduced, like attention's output projection.
    /// </summary>
    public partial class Qwen35RuntimeModel : StandardDecoderModel
    {
        private readonly bool[] _isRecurrent;
        private readonly int _ropeDim;
        private readonly RotarySpec _rotary;

        // Gated DeltaNet geometry (global).
        private readonly int _numKHeads;
        private readonly int _numVHeads;
        private readonly int _headKDim;
        private readonly int _headVDim;
        private readonly int _convKernel;

        // Per (layer, rank) host constants of the recurrence, in the rank's local channel order.
        private float[][][] _convWT;   // [layer][rank] = [kernel, localQkv] (tap-major)
        private float[][][] _dtBias;   // [layer][rank] = [localV]
        private float[][][] _ssmA;     // [layer][rank] = [localV]
        private float[][] _ssmNorm;    // [layer] = [headVDim]

        public Qwen35RuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen35";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            if (_gguf.Metadata.TryGetValue($"{arch}.attention.head_count_kv", out var kvHeads))
                Config.NumKVHeads = Convert.ToInt32(kvHeads, CultureInfo.InvariantCulture);

            int ssmInner = (int)_gguf.GetUint32($"{arch}.ssm.inner_size");
            _headKDim = (int)_gguf.GetUint32($"{arch}.ssm.state_size");
            _numKHeads = (int)_gguf.GetUint32($"{arch}.ssm.group_count");
            _numVHeads = (int)_gguf.GetUint32($"{arch}.ssm.time_step_rank");
            _convKernel = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");
            _headVDim = ssmInner / _numVHeads;
            if (_headKDim != _headVDim)
                throw new NotSupportedException($"{arch}: Gated DeltaNet with key width {_headKDim} != value width {_headVDim} is not supported.");
            if (_numVHeads % _numKHeads != 0)
                throw new NotSupportedException($"{arch}: {_numVHeads} value heads do not group over {_numKHeads} key heads.");

            // NextN / MTP draft blocks are counted in block_count but are not decoder layers.
            int nextn = (int)_gguf.GetUint32($"{arch}.nextn_predict_layers", 0);
            Config.NumLayers -= nextn;
            // The NextN/MTP block drafts on one device, the last of this process's whole-layer stages, (unsharded) on
            // rank 0 of a tensor-parallel group, or on the last stage of a multi-node pipeline, which the driver runs.
            bool drafts = tpGroup == null || tpGroup is ILayerSplitGroup || tpGroup.NodeCount == 1;
            _draftLayers = drafts ? Math.Min(nextn, 1) : 0;
            // A family whose MTP head ships as its own file (--draft-model): its layer loads as the draft layer.
            if (_draftLayers == 0 && drafts && OpenSeparateDraftHead(draftModelPath) is { } head)
            {
                _draftLayers = 1;
                _separateDraftHead = head;
                LoadAdditionalWeights(head);
            }
            // A multi-node pipeline drafts on its last stage: a drafter that reads the trunk's embedding needs it there.
            DrafterReadsTrunkEmbedding = (_draftLayers > 0 && !_gguf.Tensors.ContainsKey(LayerName(Config.NumLayers, "nextn.embed_tokens.weight")))
                || !string.IsNullOrWhiteSpace(draftModelPath);

            int interval = (int)_gguf.GetUint32($"{arch}.full_attention_interval", 4);
            _isRecurrent = new bool[Config.NumLayers];
            string[] layerTypes = _gguf.GetStringArray($"{arch}.layer_types");
            uint[] recurrentLayers = _gguf.GetUint32Array($"{arch}.attention.recurrent_layers");   // qwen4exp's per-layer flags
            for (int l = 0; l < Config.NumLayers; l++)
                _isRecurrent[l] = layerTypes != null && layerTypes.Length == Config.NumLayers
                    ? string.Equals(layerTypes[l], "linear_attention", StringComparison.OrdinalIgnoreCase)
                    : recurrentLayers != null && recurrentLayers.Length >= Config.NumLayers
                        ? recurrentLayers[l] != 0
                        : (l + 1) % interval != 0;

            int headDim = Config.HeadDim;
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)headDim);
            // Qwen-VL's interleaved M-RoPE: rotated pairs per (T, H, W, extra) axis.
            _ropeSections = _gguf.GetInt32Array($"{arch}.rope.dimension_sections") is { Length: >= 3 } sections ? sections : null;
            _rotary = new RotarySpec(_ropeDim, neoX: true, Config.RopeBase, 1.0f / Config.RopeScale);
            ParseTokenizer();

            int recurrent = 0;
            foreach (bool r in _isRecurrent) if (r) recurrent++;
            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers} ({Config.NumLayers - recurrent} attention, {recurrent} Gated DeltaNet), " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={headDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  Gated DeltaNet: kHeads={_numKHeads}, vHeads={_numVHeads}, dim={_headKDim}, conv={_convKernel}; " +
                $"RoPE dims={_ropeDim} base={Config.RopeBase} (NeoX, after QK-norm)");

            // What the shared runtime does not implement for this family is refused, not
            // approximated: per-tensor NVFP4 scale sidecars on routed experts (the runtime's expert
            // kernels do not apply them; projections do). Bonsai2's PRISM rotations run where GGML
            // builds the graphs (PrepareLoadedWeights registers them); the load refuses them on every
            // other backend and under tensor parallelism (ModelBase.ReadBonsaiMetadata).

            // A drafter named besides a separate MTP head attaches after the expert plan, into the same memory.
            if (_separateDraftHead == null && !string.IsNullOrWhiteSpace(draftModelPath))
                _pendingDraftBytes = ExpertPlacementPlanner.PendingDFlashDraftBytes(draftModelPath);
            InitializeDecoder(headDim, headDim, string.Create(CultureInfo.InvariantCulture,
                $"qwen35|{Config.NumHeads}|{Config.NumKVHeads}|{headDim}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{_ropeDim}|" +
                $"{_numKHeads}|{_numVHeads}|{_headKDim}|{_convKernel}|{interval}"));
            if (HasSidecarWeightScales && HasRoutedExpertLayers)
            {
                Dispose();
                throw new NotSupportedException(
                    $"{arch}: per-tensor weight scales (NVFP4 sidecars) on routed experts are not applied by the shared " +
                    "decoder runtime; this checkpoint cannot run with --layer-split or host experts on cuda/mlx.");
            }
            BuildRecurrentConstants();
            if (TensorRecurrence)
                BuildDeviceConstants();
            if (_separateDraftHead == null) AttachDraftModel(draftModelPath);
            _pendingDraftBytes = 0;
        }

        /// <summary>A draft head shipped as its own GGUF (validated and open, its layer blk.{NumLayers}), or null:
        /// <c>--draft-model</c> then names a DFlash / DFlash2 / DSpine drafter. Called before the caches exist.</summary>
        protected virtual GgufFile OpenSeparateDraftHead(string draftModelPath) => null;

        private GgufFile _separateDraftHead;

        /// <summary>
        /// The requests the shared runtime serves for this family (see Qwen35Architecture):
        /// whole-layer stages, local or multi-node, alone or with --tp; routed experts on the host
        /// for cuda and mlx, where Qwen35Model keeps them in device buffers; and one device on
        /// ggml_cuda, ggml_metal, ggml_cpu or MLX (<see cref="ServesOneDevice"/>).
        /// </summary>
        internal static bool Serves(Architecture.ModelCreateContext context)
            => context.LayerSplitDegree > 1
               || context.TpGroup is ILayerSplitGroup
               || (MoeCpuOffloadConfig.IsEnabled && context.Backend is BackendType.Cuda or BackendType.Mlx)
               || ServesOneDevice(context);

        /// <summary>
        /// One device on ggml_cuda, ggml_metal, ggml_cpu or MLX, for a checkpoint without routed experts (with them on
        /// the GGML devices, below): the runtime matches Qwen35Model there (Qwen3.8-27B UD-IQ3_XXS, prefill / decode tok/s: ggml_metal 388.7 /
        /// 21.0 vs 393.3 / 21.0 on an M5 Pro; ggml_cuda measured on an RTX PRO 6000 in the commit adding this
        /// route; MLX decode 19.8 vs 19.6-20.0, MTP 29.8 vs 29.3-29.6, DFlash2 29.3-29.8 vs 27.5-27.7,
        /// Qwen3.5-0.8B Q8_0 prompt 12,130 vs 12,290) or beats it (ggml_cpu on the M5 Pro, prefill / decode:
        /// Qwen3.5-0.8B Q8_0 3.6K tokens 3.2 vs 9.6 s / 99-111 vs 20 tok/s; Qwen3.8-27B UD-IQ3_XXS 55 tokens 6.0
        /// vs 7.2-8.2 s / 6.2-6.8 vs 3.2-3.8 tok/s). Bonsai2's PRISM rotations included: the GGML graphs rotate
        /// the rotated products' inputs (MLX refuses them at load, as Qwen35Model does). Routed experts too, on the
        /// GGML devices (Qwen3.6-35B-A3B UD-Q4_K_M, runtime vs Qwen35Model: one A40 ggml_cuda, a 3.6K-token prompt
        /// then 512 tokens 762-804 vs 813-871 ms / 106.7-108.1 vs 104.0 tok/s, perplexity over 4,088 tokens 13.10 vs
        /// 13.13; ggml_metal on an M5 Pro 2.17 vs 2.21-2.23 s / 66.7-66.8 vs 66.8-67.0 tok/s, 13.13 vs 13.09; ggml_cpu
        /// 55-token prompt 0.57-0.59 vs 1.24-2.04 s, decode 37-46 vs 6.4-7.4 tok/s, 12.86 vs 12.96) and on MLX, whose
        /// lone prompts step 512 tokens at a time (Qwen3.6-35B-A3B UD-IQ2_XXS, M5 Pro, a 3.6K-token prompt then 512
        /// tokens: 4.48-4.65 vs 5.27-5.29 s / 38.1 vs 34.0 tok/s, peak wired 16 vs 16-17 GB; perplexity over 1,022
        /// tokens 14.73 vs 14.72). Qwen35Model keeps --tp (its per-rank fused graphs) and the other backends (direct
        /// CUDA and ggml_vulkan are not at parity yet).
        /// </summary>
        internal static bool ServesOneDevice(Architecture.ModelCreateContext context)
            => context.Backend is BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu or BackendType.Mlx
               && context.TpDegree <= 1 && context.TpGroup == null && context.LayerSplitDegree <= 1
               && context.Probe != null;

        protected override bool IsRecurrentLayer(int layer) => _isRecurrent[layer];

        /// <summary>One device or whole-layer stages: the in-projection packed into one product where
        /// the four sources share a quantized type (rows move as they are), before residency releases
        /// the host copies. Mixed types stay four products (the graph lowering runs them; under tensor
        /// parallelism ShardRecurrentLayer packs per rank instead). Then Bonsai2's PRISM transforms go on
        /// the rotated weights, fused QKV and gate/up included (the GGML graph builder rotates their inputs).</summary>
        protected override void PrepareLoadedWeights()
        {
            if (TensorRecurrence && Mesh.TensorParallel == 1)
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    // A packed in-projection would carry no transform: PRISM rotates qkv and z, never beta and alpha.
                    if (!_isRecurrent[l] || !OwnsLayer(l) || !InProjectionSharesType(l) || RotatesAny(InProjSources(l))) continue;
                    int layer = l;
                    PackInProjection(layer, 1, 0, 1, _ => AllocatorForWeight($"blk.{layer}.attn_qkv.weight"));
                }
            }
            RegisterBonsaiWeightTransforms(_headVDim, _numKHeads, _numVHeads);
        }

        private bool RotatesAny(string[] names)
        {
            if (BonsaiHadamard == null) return false;
            foreach (string name in names)
                if (BonsaiHadamard.WeightNames.Contains(name) || BonsaiHadamard.InverseWeightNames.Contains(name)) return true;
            return false;
        }

        protected unsafe IntPtr HostPointer(Tensor tensor) => (IntPtr)GetFloatPtr(tensor);

        protected override bool GatedAttentionOutput => true;

        protected override string FfnNormWeight(int layer) => LayerName(layer, "post_attention_norm.weight");

        protected override string FingerprintExtras => $"|qwen35|rope={_ropeDim}";

        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
            => _isRecurrent[layer]
                ? Array.Empty<string>()
                : new[] { $"blk.{layer}.attn_q_norm.weight", $"blk.{layer}.attn_k_norm.weight" };

        protected override IEnumerable<string> ShardingProblems(int degree)
        {
            if (_numKHeads % degree != 0)
                yield return $"{_numKHeads} Gated DeltaNet key heads";
        }

        /// <summary>The MTP draft block (layers past the decoder stack) is not loaded.</summary>
        private readonly int _draftLayers;

        protected override int DraftLayerCount => _draftLayers;

        /// <summary>Decoder layers and, on one device, the NextN/MTP draft layer.</summary>
        protected override bool ShouldLoadWeight(GgufTensorInfo info)
            => !(LayerSplitPlan.TryGetLayer(info.Name, out int layer) && layer >= Config.NumLayers + _draftLayers)
               && base.ShouldLoadWeight(info);

        // ---------------------------------------------------------------- attention

        /// <summary>Per-head Q/K RMSNorm, then the partial NeoX rotary.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            string p = $"blk.{layer}.";
            bool hostDecode = positions.Count == 1 && positions.IsContiguous && HostDecodeOps;
            HeadNorm(q, Replica(_weights[p + "attn_q_norm.weight"], rank.Stage, rank.Rank), rank.LocalHeads, positions.Count, hostDecode);
            HeadNorm(k, Replica(_weights[p + "attn_k_norm.weight"], rank.Stage, rank.Rank), rank.LocalKvHeads, positions.Count, hostDecode);
            Rotate(q, rank.LocalHeads, KeyDim, positions, _rotary);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, _rotary);
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

        // ---------------------------------------------------------------- Gated DeltaNet

        private int QkDim => _numKHeads * _headKDim;
        private int VDim => _numVHeads * _headVDim;

        /// <summary>Global key heads of rank r: a contiguous block.</summary>
        private int[] KeyHeadsOf(int globalRank, int degree)
        {
            int width = _numKHeads / degree;
            var heads = new int[width];
            for (int i = 0; i < width; i++) heads[i] = globalRank * width + i;
            return heads;
        }

        /// <summary>Global value heads of rank r: those whose key head (h mod nK) it owns, ascending.</summary>
        private int[] ValueHeadsOf(int globalRank, int degree)
        {
            int width = _numKHeads / degree;
            int first = globalRank * width;
            var heads = new List<int>();
            for (int h = 0; h < _numVHeads; h++)
            {
                int kh = h % _numKHeads;
                if (kh >= first && kh < first + width) heads.Add(h);
            }
            return heads.ToArray();
        }

        /// <summary>Channels of attn_qkv / ssm_conv1d a rank keeps: its q heads, k heads, v heads.</summary>
        private int[] QkvChannelsOf(int globalRank, int degree)
        {
            var rows = new List<int>();
            foreach (int kh in KeyHeadsOf(globalRank, degree))
                for (int d = 0; d < _headKDim; d++) rows.Add(kh * _headKDim + d);
            foreach (int kh in KeyHeadsOf(globalRank, degree))
                for (int d = 0; d < _headKDim; d++) rows.Add(QkDim + kh * _headKDim + d);
            foreach (int vh in ValueHeadsOf(globalRank, degree))
                for (int d = 0; d < _headVDim; d++) rows.Add(2 * QkDim + vh * _headVDim + d);
            return rows.ToArray();
        }

        private int[] ValueChannelsOf(int globalRank, int degree)
        {
            var rows = new List<int>();
            foreach (int vh in ValueHeadsOf(globalRank, degree))
                for (int d = 0; d < _headVDim; d++) rows.Add(vh * _headVDim + d);
            return rows.ToArray();
        }

        protected override void ShardRecurrentLayer(int layer)
        {
            string p = $"blk.{layer}.";
            int degree = ShardDegree;
            // GGML's per-op tensor parallelism runs the packed in-projection per rank; per-rank graphs read the
            // projections by name (row shards, as on every other executor).
            if (DeviceRecurrence && !RankGraphs)
            {
                PackInProjection(layer, TpDegree, TpRankOffset, degree, r => _tpGroup.GetAllocator(r));
                ShardInputColumnGroups(p + "ssm_out.weight", r => ValueHeadsOf(r, degree), _headVDim);
                return;
            }
            ShardOutputRows(p + "attn_qkv.weight", r => QkvChannelsOf(r, degree));
            ShardOutputRows(p + "attn_gate.weight", r => ValueChannelsOf(r, degree));
            ShardOutputRows(p + "ssm_beta.weight", r => ValueHeadsOf(r, degree));
            ShardOutputRows(p + "ssm_alpha.weight", r => ValueHeadsOf(r, degree));
            ShardInputColumnGroups(p + "ssm_out.weight", r => ValueHeadsOf(r, degree), _headVDim);
        }

        /// <summary>The recurrence's small per-head constants, gathered per rank in its local order.</summary>
        private void BuildRecurrentConstants()
        {
            int layers = Config.NumLayers;
            int ranks = Mesh.TensorParallel;
            int degree = ShardDegree;
            int offset = ShardRankOffset;
            _convWT = new float[layers][][];
            _dtBias = new float[layers][][];
            _ssmA = new float[layers][][];
            _ssmNorm = new float[layers][];
            int qkvDim = 2 * QkDim + VDim;
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l] || !OwnsLayer(l)) continue;
                string p = $"blk.{l}.";
                float[] conv = _weights[p + "ssm_conv1d.weight"].GetElementsAsFloat(qkvDim * _convKernel);
                float[] dt = _weights[p + "ssm_dt.bias"].GetElementsAsFloat(_numVHeads);
                float[] a = _weights[p + "ssm_a"].GetElementsAsFloat(_numVHeads);
                _ssmNorm[l] = _weights[p + "ssm_norm.weight"].GetElementsAsFloat(_headVDim);
                _convWT[l] = new float[ranks][];
                _dtBias[l] = new float[ranks][];
                _ssmA[l] = new float[ranks][];
                for (int r = 0; r < ranks; r++)
                {
                    int[] channels = QkvChannelsOf(offset + r, degree);
                    int[] vHeads = ValueHeadsOf(offset + r, degree);
                    // Stored [qkvDim, kernel] (one filter per channel); kept tap-major so a tap
                    // reads its channels contiguously.
                    var wt = new float[_convKernel * channels.Length];
                    for (int c = 0; c < channels.Length; c++)
                        for (int ki = 0; ki < _convKernel; ki++)
                            wt[ki * channels.Length + c] = conv[channels[c] * _convKernel + ki];
                    _convWT[l][r] = wt;
                    _dtBias[l][r] = new float[vHeads.Length];
                    _ssmA[l][r] = new float[vHeads.Length];
                    for (int j = 0; j < vHeads.Length; j++)
                    {
                        _dtBias[l][r][j] = dt[vHeads[j]];
                        _ssmA[l][r][j] = a[vHeads[j]];
                    }
                }
            }
        }

        /// <summary>One rank's Gated DeltaNet state for one sequence: the conv ring of the last
        /// kernel-1 inputs and the per-value-head delta-rule state [vHeads, Dv, Dk].</summary>
        private sealed class GdnState : RecurrentState
        {
            public GdnState(int convRows, int channels, int vHeads, int dv, int dk)
            {
                ConvRows = convRows;
                Channels = channels;
                Conv = new float[Math.Max(1, convRows) * channels];
                Ssm = new float[vHeads * dv * dk];
            }

            public int ConvRows { get; }
            public int Channels { get; }
            public float[] Conv { get; }
            public float[] Ssm { get; }
            public int WriteIndex { get; set; }

            public override void Reset()
            {
                Array.Clear(Conv);
                Array.Clear(Ssm);
                WriteIndex = 0;
            }

            // A per-op verify window (a multi-node pipeline's): the state before it, and the window's recorded
            // recurrence, run again over the kept rows when the window is cut.
            private float[] _verifyConv, _verifySsm;
            private int _verifyWriteIndex, _verifyWindow;
            private Action<int> _verifyReplay;

            public bool Verifying { get; private set; }

            protected internal override bool SupportsVerify => true;

            protected internal override void BeginVerify()
            {
                _verifyConv = (float[])Conv.Clone();
                _verifySsm = (float[])Ssm.Clone();
                _verifyWriteIndex = WriteIndex;
                _verifyReplay = null;
                _verifyWindow = 0;
                Verifying = true;
            }

            /// <summary>The window's <paramref name="rows"/> ran; <paramref name="replay"/> runs the recurrence over its
            /// first rows again.</summary>
            public void RecordVerifyWindow(int rows, Action<int> replay)
            {
                _verifyWindow = rows;
                _verifyReplay = replay;
            }

            protected internal override void CommitVerify(int kept)
            {
                if (!Verifying) return;
                Verifying = false;
                try
                {
                    if (_verifyReplay == null || kept >= _verifyWindow) return;
                    Array.Copy(_verifyConv, Conv, Conv.Length);
                    Array.Copy(_verifySsm, Ssm, Ssm.Length);
                    WriteIndex = _verifyWriteIndex;
                    if (kept > 0) _verifyReplay(kept);
                }
                finally
                {
                    _verifyConv = _verifySsm = null;
                    _verifyReplay = null;
                    _verifyWindow = 0;
                }
            }
        }

        protected override RecurrentState CreateRecurrentState(RankScope scope, int layer)
        {
            int vHeads = _numVHeads / ShardDegree;
            int kHeads = _numKHeads / ShardDegree;
            int channels = 2 * kHeads * _headKDim + vHeads * _headVDim;
            if (TensorRecurrence)
                return CreateDeviceGdnState(scope, layer, _convKernel - 1, channels, vHeads, _headVDim, _headKDim);
            return new GdnState(_convKernel - 1, channels, vHeads, _headVDim, _headKDim);
        }

        /// <summary>A layer's tensor state (a family adds what its layer keeps besides the recurrence).</summary>
        protected virtual DeviceGdnState CreateDeviceGdnState(RankScope scope, int layer, int convRows, int channels, int vHeads, int dv, int dk)
            => new(scope.Allocator, ExecutorFor(scope.Allocator), HostPointer, convRows, channels, vHeads, dv, dk);

        /// <summary>The recurrence's output gate: sigmoid(z) (Qwen3.8-Flash-Next) instead of silu(z).</summary>
        protected virtual bool MixerSigmoidGate => false;

        protected override Tensor RecurrentMixer(RankScope scope, Tensor normed, int layer, RecurrentState state)
        {
            if (state is DeviceGdnState device)
                return DeviceGatedDeltaNet(scope, normed, layer, device);
            string p = $"blk.{layer}.";
            var gdn = (GdnState)state;
            int tokens = (int)normed.Sizes[0];
            int kHeads = _numKHeads / ShardDegree;
            int vHeads = _numVHeads / ShardDegree;
            int channels = gdn.Channels;
            int vWidth = vHeads * _headVDim;

            float[] qkv, z, beta, alpha;
            using (Tensor t = Linear(scope, normed, p + "attn_qkv.weight")) qkv = t.GetElementsAsFloat(tokens * channels);
            using (Tensor t = Linear(scope, normed, p + "attn_gate.weight")) z = t.GetElementsAsFloat(tokens * vWidth);
            using (Tensor t = Linear(scope, normed, p + "ssm_beta.weight")) beta = t.GetElementsAsFloat(tokens * vHeads);
            using (Tensor t = Linear(scope, normed, p + "ssm_alpha.weight")) alpha = t.GetElementsAsFloat(tokens * vHeads);

            var gated = new float[tokens * vWidth];
            if (gdn.Verifying)
            {
                // A cut verify window runs the recurrence again over its kept rows from the state before it (CommitVerify).
                int rank = scope.Rank;
                gdn.RecordVerifyWindow(tokens, kept => GatedDeltaNet(gdn, qkv, z, beta, alpha, new float[kept * vWidth], kept, kHeads, vHeads,
                    _convWT[layer][rank], _dtBias[layer][rank], _ssmA[layer][rank], _ssmNorm[layer]));
            }
            GatedDeltaNet(gdn, qkv, z, beta, alpha, gated, tokens, kHeads, vHeads,
                _convWT[layer][scope.Rank], _dtBias[layer][scope.Rank], _ssmA[layer][scope.Rank], _ssmNorm[layer]);

            using var gatedT = new Tensor(scope.Allocator, DType.Float32, tokens, vWidth);
            gatedT.SetElementsAsFloat(gated);
            HostWritten(gatedT);
            return Linear(scope, gatedT, p + "ssm_out.weight");
        }

        /// <summary>The token-by-token recurrence on the host (rank-local heads).</summary>
        private void GatedDeltaNet(GdnState s, float[] qkv, float[] z, float[] beta, float[] alpha, float[] gated,
            int tokens, int kHeads, int vHeads, float[] convWT, float[] dtBias, float[] a, float[] norm)
        {
            int channels = s.Channels;
            int dk = _headKDim, dv = _headVDim;
            int qk = kHeads * dk;
            int convRows = s.ConvRows;
            float eps = Config.Eps;
            float qScale = 1.0f / MathF.Sqrt(dv);
            var conv = new float[channels];
            var q = new float[qk];
            var k = new float[qk];

            for (int t = 0; t < tokens; t++)
            {
                int row = t * channels;
                // Causal depthwise conv over [ring (oldest first) | this input], then SiLU.
                for (int c = 0; c < channels; c++)
                {
                    float acc = 0;
                    for (int ki = 0; ki < convRows; ki++)
                        acc += s.Conv[((s.WriteIndex + ki) % convRows) * channels + c] * convWT[ki * channels + c];
                    acc += qkv[row + c] * convWT[convRows * channels + c];
                    conv[c] = TensorComputePrimitives.SiLU(acc);
                }
                if (convRows > 0)
                {
                    Array.Copy(qkv, row, s.Conv, s.WriteIndex * channels, channels);
                    s.WriteIndex = (s.WriteIndex + 1) % convRows;
                }

                Array.Copy(conv, 0, q, 0, qk);
                Array.Copy(conv, qk, k, 0, qk);
                for (int h = 0; h < kHeads; h++)
                {
                    L2Normalize(q, h * dk, dk, eps);
                    L2Normalize(k, h * dk, dk, eps);
                }
                for (int i = 0; i < qk; i++) q[i] *= qScale;

                int vBase = 2 * qk;
                int zRow = t * vHeads * dv;
                Parallel.For(0, vHeads, j =>
                {
                    int kh = j % kHeads;
                    int qOff = kh * dk;
                    float decay = MathF.Exp(TensorComputePrimitives.Softplus(alpha[t * vHeads + j] + dtBias[j]) * a[j]);
                    float b = TensorComputePrimitives.Sigmoid(beta[t * vHeads + j]);
                    int stateBase = j * dv * dk;
                    var core = new float[dv];
                    for (int i = 0; i < dv * dk; i++) s.Ssm[stateBase + i] *= decay;
                    for (int r = 0; r < dv; r++)
                    {
                        int sr = stateBase + r * dk;
                        float mem = 0;
                        for (int d = 0; d < dk; d++) mem += s.Ssm[sr + d] * k[qOff + d];
                        float delta = (conv[vBase + j * dv + r] - mem) * b;
                        float o = 0;
                        for (int d = 0; d < dk; d++)
                        {
                            s.Ssm[sr + d] += k[qOff + d] * delta;
                            o += s.Ssm[sr + d] * q[qOff + d];
                        }
                        core[r] = o;
                    }
                    float ss = 0;
                    for (int r = 0; r < dv; r++) ss += core[r] * core[r];
                    float inv = 1.0f / MathF.Sqrt(ss / dv + eps);
                    int outBase = zRow + j * dv;
                    for (int r = 0; r < dv; r++)
                        gated[outBase + r] = core[r] * inv * norm[r] * TensorComputePrimitives.SiLU(z[outBase + r]);
                });
            }
        }

        private static void L2Normalize(float[] v, int offset, int length, float eps)
        {
            float ss = 0;
            for (int i = 0; i < length; i++) ss += v[offset + i] * v[offset + i];
            float inv = 1.0f / MathF.Sqrt(ss + eps);
            for (int i = 0; i < length; i++) v[offset + i] *= inv;
        }
    }
}
