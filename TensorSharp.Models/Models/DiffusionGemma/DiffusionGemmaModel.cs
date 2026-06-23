// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// DiffusionGemma (architecture key <c>diffusion-gemma</c>) — a block text-diffusion
    /// Mixture-of-Experts language model built on a Gemma-4 backbone.
    ///
    /// This is fundamentally different from the autoregressive Gemma 4 model:
    ///  - A single <b>no-cache, bidirectional</b> forward is run over a concatenated
    ///    <c>[prompt | canvas]</c> sequence. Split point <c>P = n_tokens - canvas_length</c>.
    ///  - Prompt queries are causal (and sliding-window clipped on local layers) and never
    ///    attend to the canvas; canvas queries are bidirectional over prompt + canvas.
    ///  - Input embeddings are region-aware: prompt = <c>embed*sqrt(n_embd)</c>,
    ///    canvas = <c>rms_norm_noscale(embed*sqrt(n_embd) [+ self-conditioning])</c>.
    ///  - Each layer applies a region-aware per-layer scalar: prompt uses the encoder scalar,
    ///    canvas uses the decoder scalar.
    ///
    /// Generation is performed by an iterative denoising loop (the EntropyBound sampler) which
    /// lives in <see cref="DiffusionGemmaSampler"/>; this class exposes the per-step forward.
    ///
    /// The Gemma-4 backbone shared with the AR model:
    ///  - Per-head Q/K RMSNorm, unweighted V RMSNorm, NeoX RoPE.
    ///  - Attention scale = 1.0 (the learnable Q/K norms absorb the 1/sqrt(d) factor).
    ///  - 5 sliding (local) layers : 1 global layer pattern (local head dim 256, global 512).
    ///  - Global layers omit the V projection (V == raw K projection).
    ///  - Dense gated-GELU MLP (shared expert) + 128-expert top-8 softmax MoE, summed per layer.
    ///  - Embedding scaling by sqrt(hidden), tied lm-head, final logit softcapping.
    /// </summary>
    public sealed class DiffusionGemmaModel : ModelBase
    {
        // ---- architecture configuration ----
        private bool[] _isLocal;            // per-layer: true = sliding-window (local), false = global
        private int[] _kvHeads;             // per-layer KV head count
        private int[] _headDim;             // per-layer head dim (local 256, global 512)
        private bool[] _hasVProj;           // per-layer: whether attn_v.weight exists (false on global layers)
        private int _localHeadDim;
        private int _globalHeadDim;
        private int _slidingWindow;
        private float _ropeLocalBase;
        private float _ropeGlobalBase;
        private int _denseFfn;              // dense MLP intermediate (feed_forward_length)
        private int _expertFfn;            // per-expert intermediate (expert_feed_forward_length)
        private int _numExperts;
        private int _numExpertsUsed;
        private float _finalLogitSoftcap;
        private int _canvasLength;
        private int _maskTokenId;

        // precomputed RoPE base frequencies per layer type (NeoX). Global folds in rope_freqs (NTK).
        private float[] _ropeFreqsLocal;    // length localHeadDim/2
        private float[] _ropeFreqsGlobal;   // length globalHeadDim/2
        // raw rope_freqs.weight (NOT folded), used by the on-device GGML freq-factors RoPE kernel.
        private Tensor _ropeFreqsRawTensor;

        // ---- on-device glue-op caches (rebuilt only when N/P change, i.e. once per block) ----
        // RoPE integer positions [N*numHeads], keyed by (numHeads, startPos).
        private readonly Dictionary<long, Tensor> _ropePosCache = new();
        private int _ropePosN = -1;
        // global cos/sin tables (folded freqs) for the non-GGML RoPE paths (mlx fused / cuda raw).
        private int _cosSinGlobalN = -1;
        private float[] _cosGlobalHost, _sinGlobalHost;
        private Tensor _cosGlobalTensor, _sinGlobalTensor;
        // additive attention masks [maskHeads,N,N] (0 = attend, large-negative = masked), per layer type.
        private int _maskN = -1, _maskP = -1;
        private Tensor _maskLocal, _maskGlobal;
        private const float MaskNeg = -1e30f;
        private const int NeoXRopeMode = 2;

        // host cos/sin tables for the CPU raw-pointer RoPE path, cached by N.
        private int _cosSinHostN = -1;
        private float[] _cosLocalHost, _sinLocalHost;

        // GPU backends keep the whole forward device-resident (Ops-based glue); CPU backends use the
        // hand-tuned SIMD raw-pointer glue, which is faster on the CPU and avoids host<->device churn
        // that doesn't exist there anyway.
        private bool _useDeviceGlue;

        // ---- prompt-KV caching (PKV) — the headline llama.cpp/vLLM diffusion-gemma optimization ----
        // The prompt's per-layer K/V do not depend on the canvas (prompt is causal and never attends to
        // the canvas), so they are identical across every denoising step. PrefillPrompt computes them
        // once per block and stores them here (head-first [kvHeads, P, hd]); DecodeCanvas then processes
        // only the canvas each step and prepends the cached prompt K/V. GPU path only.
        private Tensor[] _promptK, _promptV;
        private int _promptLen = -1;
        private bool _pkvEnabled;
        // decode-phase additive masks [maskHeads, C, P+C] (null when no masking is needed, i.e. when the
        // prompt fits within the sliding window — the common case).
        private int _decodeMaskP = -1, _decodeMaskC = -1;
        private Tensor _decodeMaskLocal, _decodeMaskGlobal;

        // per-layer output scalars
        private float[] _decScale;          // layer_output_scale (canvas / decoder)
        private float[] _encScale;          // enc_layer_output_scale (prompt / encoder)

        // fused-MoE stacked expert weights (one ggml_mul_mat_id dispatch per layer instead of 128 matmuls)
        private StackedExpertWeights[] _stackedGateUp;   // pre-fused [n_embd, 2*expertFfn, numExperts]
        private StackedExpertWeights[] _stackedDown;     // [expertFfn, n_embd, numExperts]
        private float[][] _perExpertScale;               // per-layer ffn_down_exps.scale
        private bool _fusedMoeAvailable;

        // MLX fused MoE via mlx_gather_qmm: one batched (expert-sorted) gather_qmm (gate_up) + GEGLU + one
        // gather_qmm (down) over stacked MLX-affine experts — the MLX analogue of GGML's mul_mat_id.
        // Correct (self-checked at runtime: cosine 1.0 vs the per-expert path), but measured NOT faster than
        // the per-expert affine path for this model's MoE shape: 256 canvas tokens × top-8 over 128 experts
        // leaves only ~16 tokens/expert, so the per-expert path already issues efficient grouped GEMMs and
        // the MoE is small-GEMM compute-bound — gather_qmm's sort/unsort overhead offsets its dispatch
        // savings (M=1 ~3900 ms, sorted ~3635 ms, per-expert affine ~3450 ms / decode step). Hence OFF by
        // default. Opt in with TS_MLX_MOE_GATHER_QMM=1 (it can win for larger canvases / fewer experts where
        // per-expert batches are bigger). The one-time self-check disables it permanently if it ever diverges.
        private readonly bool _moeGatherQmmEnabled = Environment.GetEnvironmentVariable("TS_MLX_MOE_GATHER_QMM") == "1";
        private bool _moeGatherQmmOk = true;
        private bool _moeGatherQmmChecked;

        // MLX FULLY-ON-DEVICE fused MoE: routing (top-K) AND the expert FFN run on-device with NO host read,
        // so the entire decode forward stays device-resident — the MLX port of GGML's fused single-graph
        // decode. CORRECT (self-check cosine 1.0) but measured NOT faster (~4030 vs ~3450 ms/step for the
        // per-expert affine path), because unlike GGML — where fusion into ONE C++ graph / Metal command
        // buffer removes ALL per-op host/native overhead (its 4x: per-op 2650 -> fused 640 ms) — MLX still
        // builds its graph op-by-op through the worker thread, so that overhead remains even when the forward
        // collapses to one lazy eval; and a device-resident router forces the M=1 (GEMV) gather_qmm whose
        // penalty offsets the (small) per-layer-sync saving. So OFF by default; opt in with
        // TS_MLX_FUSED_DEVICE_MOE=1. Matching GGML on MLX needs mlx_compile of the whole forward or a custom
        // Metal megakernel (i.e. reimplementing GGML's kernel) — not a cheap port. One-time self-check.
        private readonly bool _moeFusedDeviceEnabled = Environment.GetEnvironmentVariable("TS_MLX_FUSED_DEVICE_MOE") == "1";
        private bool _moeFusedDeviceOk = true;
        private bool _moeFusedDeviceChecked;
        private Tensor _moeLhsGateConst, _moeLhsArangeConst;   // host-built once: [N*K] gather_qmm lhs indices
        private int _moeConstNK = -1;

        // cached ones tensors for unweighted RMSNorm, keyed by dim
        private readonly Dictionary<int, Tensor> _onesByDim = new();

        // self-conditioning (optional)
        private bool _scEnabled;

        // reusable output buffer for canvas logits
        private float[] _canvasLogits;

        // timing
        private readonly Stopwatch _swForward = new();
        private long _tEmbed, _tAttn, _tMoe, _tDense, _tLmHead, _tSc, _tRope, _tMoeRoute, _tMoeFfn;
        private long _tScTopK, _tScDevice;   // self-conditioning split: host top-K vs on-device gather+MLP
        // When set (DIFFUSION_PROFILE=1) each timed section is drained before its timer stops so the
        // per-section attribution is accurate (otherwise async GGML/Metal work inflates the section that
        // happens to force the next host read).
        private readonly bool _profile = Environment.GetEnvironmentVariable("DIFFUSION_PROFILE") == "1";
        private readonly bool _stepTime = Environment.GetEnvironmentVariable("DIFFUSION_STEPTIME") == "1";
        private long _stepT0;
        // Fused decode-layer kernel: whole layer (attention + dense + MoE) in one GGML graph dispatch.
        // Correct (output matches the per-op path) but per-LAYER fusion alone doesn't speed up the decode
        // because `hidden` still round-trips host memory between layers, serialising them. The model-wide
        // single graph (hidden stays on-device across all layers) is the real throughput win. Opt-in for now
        // via DIFFUSION_FUSED_DECODE=1 so the default stays on the proven per-op path.
        // Default ON for GGML: fuse all transformer layers into one graph per decode step (correct + ~1.7x).
        // Disable via DIFFUSION_NO_FUSED_DECODE=1. The further lm_head fusion (~3-4x) is separately opt-in.
        private readonly bool _fusedDecodeEnabled = Environment.GetEnvironmentVariable("DIFFUSION_NO_FUSED_DECODE") != "1";
        private bool _fusedDecodeOk = true;   // flipped false if the kernel ever rejects the layout
        // Fused lm_head tail (output_norm + lm_head + softcap in one dispatch). Default ON for GGML; the
        // separate small graph keeps it correct (unlike folding lm_head into the layer graph).
        private readonly bool _fusedLmHeadTailDisabled = Environment.GetEnvironmentVariable("DIFFUSION_NO_FUSED_LMHEAD_TAIL") == "1";
        private bool _fusedLmHeadTailOk = true;
        // Segmented (per-layer) fused decode: run each layer as its own fused graph so layers whose
        // weights are NOT device-resident stream through one bounded, reused staging buffer instead of
        // all coexisting in VRAM. Selected automatically by PrepareCudaWeightResidency when the model
        // does not fit; override with DIFFUSION_SEGMENTED_DECODE=1/0.
        private bool _segmentedDecode;
        // On-device diffusion sampling (CUDA only): the fused lm_head kernel computes argmax + entropy +
        // multinomial sample + top-K (for self-conditioning) directly on the device logits, so only the
        // small per-position outputs come back instead of the full [vocab,C] block — eliminating the
        // ~268 MB device->host logits download AND the host's two full-vocab CPU sweeps (sampler + SC
        // top-K) per step. The canonical llama.cpp diffusion-sampling.cu + device-SC optimization.
        // Default ON for GgmlCuda; disable with DIFFUSION_NO_DEVICE_SAMPLE=1. Flips off if the kernel
        // ever rejects the layout (then the host logits + host sampler path is used).
        private readonly bool _deviceSampleEnabled = Environment.GetEnvironmentVariable("DIFFUSION_NO_DEVICE_SAMPLE") != "1";
        private readonly bool _deviceSampleForce = Environment.GetEnvironmentVariable("DIFFUSION_DEVICE_SAMPLE_FORCE") == "1";
        private bool _deviceSampleOk = true;
        // Reusable host buffer for the fused lm_head logits (268 MB at vocab 256K, C 256). A fresh
        // array per step costs a large-object-heap allocation + zeroing every step; one pooled buffer
        // is safe because each step's logits are fully consumed (sampling + self-conditioning read)
        // before the next step's lm_head overwrites it — the same contract the batched scheduler
        // already documents for the shared readback buffer. Allocated on the pinned object heap and
        // cudaHostRegister'ed so the per-step 268 MB device->host logits download takes the fast DMA
        // path instead of the pageable one.
        private float[] _fusedLogitsBuffer;
        // Host regions page-locked for fast per-step DMA (streamed weights + the logits buffer).
        // cudaHostRegister'ed memory MUST be unregistered before it is unmapped/freed (the streamed
        // weights live in the GGUF mmap), so Dispose undoes these.
        private readonly List<IntPtr> _pinnedHostRegions = new();
        // Streamed (non-resident) weights re-homed into page-locked PRIVATE host copies: Windows
        // cannot cudaHostRegister file-mapped (GGUF mmap) pages, so the bytes are copied once into
        // an owned allocation that CAN be page-locked — per-step uploads then run at DMA speed
        // (~2x pageable mmap throughput). Maps original weight pointer -> pinned copy; the fused
        // decode arg builder substitutes these. Freed (after unregistering) in Dispose.
        private readonly Dictionary<IntPtr, IntPtr> _pinnedStreamCopies = new();
        // Batched-decode lm_head memory cap: batch the [B*C, vocab] lm_head into one weight-read while the
        // logits tensor fits under this many bytes, else fall back to a per-sequence lm_head (one [C, vocab]
        // transient — the per-seq weight re-read costs ~1ms, negligible vs the per-step compute). Default 300
        // MB so a single canvas's logits (~262 MB at vocab 256K, C 256) batches but a B>=2 batch falls back —
        // this is what keeps batched decode inside the tight Metal headroom on a 24 GB machine running a 16.8
        // GB model. Tunable via DIFFUSION_LMHEAD_BATCH_CAP_MB.
        private static readonly long LmHeadBatchByteCap =
            (long)(int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_LMHEAD_BATCH_CAP_MB"), out int capMb) && capMb > 0 ? capMb : 300) * 1024 * 1024;
        private void ProfSync(Tensor t) { if (_profile && t != null) _ = t.GetElementsAsFloat(1); }

        public int CanvasLength => _canvasLength;
        public int MaskTokenId => _maskTokenId;
        public int VocabSize => Config.VocabSize;
        public bool SelfConditioningEnabled { get => _scEnabled; set => _scEnabled = value; }
        /// <summary>Whether prompt-KV caching is active: the sampler calls <see cref="PrefillPrompt"/>
        /// once per block then <see cref="DecodeCanvas"/> per step. Only available on the device-glue
        /// (GPU) backends; the setter is a no-op on CPU backends.</summary>
        public bool SupportsPromptKvCache
        {
            get => _pkvEnabled;
            set => _pkvEnabled = value && _useDeviceGlue;
        }

        /// <summary>Whether the per-step argmax/entropy/multinomial/top-K sampling runs on-device (CUDA),
        /// returning only the small per-canvas-position outputs instead of the full [vocab,C] logits. Gated
        /// to the GgmlCuda backend (the sample kernel reads the device logits pointer).
        ///
        /// IMPORTANT — only enabled when the model FITS in VRAM (<c>!_segmentedDecode</c>). When the model
        /// oversubscribes VRAM (segmented decode), two things make the on-device sampler a net loss and the
        /// host path the better choice: (1) the lm_head's [vocab,C] logits tensor gets paged out of VRAM by
        /// the Windows/WDDM working-set manager (VRAM is full of weights), so the kernel re-reads it over
        /// PCIe; (2) the streaming-bound decode keeps the GPU mostly idle, so its SM/memory clock stays at
        /// its idle state and any device-resident read runs ~15x below peak bandwidth — slower than the
        /// host's copy-engine readback (which uses the clock-independent copy engine) + the SIMD CPU sampler.
        /// When the model fits, the logits stay resident, the forward keeps the GPU busy (clock ramped), and
        /// the kernel is a large win (only C ints/floats cross PCIe instead of ~268 MB). Measured on a
        /// VRAM-oversubscribed RTX 3080 Laptop: device sampling was ~33% slower, hence the gate. Override
        /// the residency gate for experiments with DIFFUSION_DEVICE_SAMPLE_FORCE=1.</summary>
        public bool SupportsDeviceSampling =>
            _deviceSampleEnabled && _deviceSampleOk && _backend == BackendType.GgmlCuda
            && _fusedLmHeadTailOk && !_fusedLmHeadTailDisabled
            && (!_segmentedDecode || _deviceSampleForce);

        /// <summary>The number of top-K tokens the device sampler returns for self-conditioning (0 when SC
        /// is disabled, so the kernel skips the top-K pass).</summary>
        public int SelfCondTopK => _scEnabled ? Math.Min(ScTopK, Config.VocabSize) : 0;

        public DiffusionGemmaModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();
            ParseDiffusionConfig();

            ParseTokenizer();

            // MLX: DiffusionGemma always runs the MULTI-ROW regime (it denoises a C=256 canvas every
            // step), where MLX's raw GGUF K-quant Metal kernels are poorly tuned (~150 GFLOP/s — they are
            // written for rows==1 autoregressive decode). Preload K-quant weights into MLX-native AFFINE
            // form instead, so every matmul (attention projections, dense FFN, per-expert MoE, lm_head)
            // runs on Apple's fast built-in mlx_quantized_matmul. The affine repack is LOSSLESS for
            // K-quant (per-32 group scale=d*scaleByte, bias=-dmin*minByte is exactly ggml's dequant), so
            // accuracy is unchanged. Opt out with TS_MLX_KQUANT_AFFINE=0.
            // NOTE: this is intentionally not restored after load. MLX weights are created lazily on first
            // matmul (not eagerly in LoadWeights), and the per-matmul path keys off the *created* weight's
            // Mode — so the flag must stay set through the first forward for the K-quants to materialize in
            // affine form. The only side effect is that a subsequently-loaded autoregressive MLX model in the
            // same process would also get affine K-quants (lossless; its rows==1 decode may marginally prefer
            // the raw custom kernels). For the common single-model process this is a no-op. Force off with
            // TS_MLX_KQUANT_AFFINE=0.
            if (backend == BackendType.Mlx &&
                Environment.GetEnvironmentVariable("TS_MLX_KQUANT_AFFINE") != "0")
            {
                MlxQuantizedOps.PreferAffineKQuant = true;
            }

            LoadWeights();

            LoadLayerScalars();
            CacheStackedExpertWeights();
            PrecomputeRoPE();

            _useDeviceGlue = _backend is BackendType.GgmlMetal or BackendType.GgmlCuda
                or BackendType.Mlx or BackendType.Cuda;
            // Prompt-KV caching needs the device-resident attention path (it stores/concats K/V tensors).
            _pkvEnabled = _useDeviceGlue && Environment.GetEnvironmentVariable("DIFFUSION_NO_PKV") != "1";

            PrepareCudaWeightResidency();

            // Self-conditioning (matches the reference sampler) materially improves quality on longer
            // outputs and converges in far fewer steps. The top-K soft-embedding makes its per-step cost
            // negligible (~tens of ms), so it is enabled by default. Disable with DIFFUSION_NO_SC=1.
            _scEnabled = Environment.GetEnvironmentVariable("DIFFUSION_NO_SC") != "1";

            Console.WriteLine($"DiffusionGemma ready: canvas_length={_canvasLength}, experts={_numExperts}/{_numExpertsUsed}, " +
                $"softcap={_finalLogitSoftcap}, mask_token={_maskTokenId}, self_conditioning={_scEnabled}, " +
                $"device_glue={_useDeviceGlue}, prompt_kv_cache={_pkvEnabled}");
        }

        private void ParseDiffusionConfig()
        {
            string arch = Config.Architecture;

            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 1024);
            Config.SlidingWindow = _slidingWindow;

            _globalHeadDim = (int)_gguf.GetUint32($"{arch}.attention.key_length", 512);
            _localHeadDim = (int)_gguf.GetUint32($"{arch}.attention.key_length_swa", 256);

            _ropeGlobalBase = Config.RopeBase;
            _ropeLocalBase = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", 10000f);

            _denseFfn = Config.IntermediateSize > 0 ? Config.IntermediateSize
                : (int)_gguf.GetUint32($"{arch}.feed_forward_length", 0);
            _expertFfn = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);

            _finalLogitSoftcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", 0f);
            _canvasLength = (int)_gguf.GetUint32("diffusion.canvas_length", 256);
            _maskTokenId = (int)_gguf.GetUint32("tokenizer.ggml.mask_token_id", 0);

            // sliding-window pattern: true = local (SWA), false = global.
            bool[] swaPattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            int[] kvArr = _gguf.GetInt32Array($"{arch}.attention.head_count_kv");

            int L = Config.NumLayers;
            _isLocal = new bool[L];
            _kvHeads = new int[L];
            _headDim = new int[L];
            _hasVProj = new bool[L];
            for (int i = 0; i < L; i++)
            {
                bool local = swaPattern != null && i < swaPattern.Length ? swaPattern[i] : true;
                _isLocal[i] = local;
                _headDim[i] = local ? _localHeadDim : _globalHeadDim;
                if (kvArr != null && i < kvArr.Length)
                    _kvHeads[i] = kvArr[i];
                else
                    _kvHeads[i] = Config.NumKVHeads;
            }

            Console.WriteLine($"Model: {arch}, Layers={L}, Hidden={Config.HiddenSize}, Heads={Config.NumHeads}");
            Console.WriteLine($"Head dims: local={_localHeadDim} global={_globalHeadDim}, RoPE local={_ropeLocalBase} global={_ropeGlobalBase}");
            Console.WriteLine($"Sliding window={_slidingWindow}, dense FFN={_denseFfn}, expert FFN={_expertFfn}");
        }

        private void LoadLayerScalars()
        {
            int L = Config.NumLayers;
            _decScale = new float[L];
            _encScale = new float[L];
            for (int l = 0; l < L; l++)
            {
                _decScale[l] = _weights.TryGetValue($"blk.{l}.layer_output_scale.weight", out var d) ? d.GetElementAsFloat(0) : 1f;
                _encScale[l] = _weights.TryGetValue($"blk.{l}.enc_layer_output_scale.weight", out var e) ? e.GetElementAsFloat(0) : 1f;
                // detect missing V projection (global layers): V == raw K
                _hasVProj[l] = _weights.ContainsKey($"blk.{l}.attn_v.weight") || _quantWeights.ContainsKey($"blk.{l}.attn_v.weight");
            }
        }

        private void CacheStackedExpertWeights()
        {
            int L = Config.NumLayers;
            _stackedGateUp = new StackedExpertWeights[L];
            _stackedDown = new StackedExpertWeights[L];
            _perExpertScale = new float[L][];
            int ok = 0;
            for (int l = 0; l < L; l++)
            {
                string prefix = $"blk.{l}";
                _stackedExpertWeights.TryGetValue($"{prefix}.ffn_gate_up_exps.weight", out _stackedGateUp[l]);
                _stackedExpertWeights.TryGetValue($"{prefix}.ffn_down_exps.weight", out _stackedDown[l]);
                if (_weights.TryGetValue($"{prefix}.ffn_down_exps.scale", out var scaleT))
                {
                    var scales = new float[_numExperts];
                    for (int e = 0; e < _numExperts; e++) scales[e] = scaleT.GetElementAsFloat(e);
                    _perExpertScale[l] = scales;
                }
                if (_stackedGateUp[l] != null && _stackedDown[l] != null) ok++;
            }
            _fusedMoeAvailable = IsGgmlBackend && ok == L;
            Console.WriteLine($"  Fused MoE FFN kernel available on {ok}/{L} layers (enabled={_fusedMoeAvailable}).");
        }

        /// <summary>
        /// CUDA VRAM residency plan. The fused decode reads EVERY weight every denoising step, and
        /// ggml's allocator model means a monolithic whole-model graph needs all of them device-resident
        /// at once — when the model is larger than VRAM, Windows WDDM transparently pages the
        /// oversubscribed working set in and out of system RAM on every command submission (measured
        /// ~4.9 s/step for a ~150 ms compute on a 16 GB GPU with a 16 GB model). The fix mirrors
        /// llama.cpp's partial-offload discipline: never oversubscribe. We preload weights device-side
        /// in priority order (lm_head/embedding first — it is also read by every step's lm_head tail —
        /// then per-layer attention/dense weights, then MoE expert stacks by layer) until a budget of
        /// free-VRAM-minus-headroom is reached, cap incidental device copies (prompt K/V, masks) with
        /// the device-copy budget, and switch the decode to the SEGMENTED per-layer fused path so the
        /// non-resident remainder streams through one bounded reused staging buffer (PCIe-speed
        /// streaming, ~no paging) instead of joining the whole-model graph's working set.
        /// No-op when everything fits (the whole-model fused graph stays the default) or off-CUDA.
        /// </summary>
        private void PrepareCudaWeightResidency()
        {
            if (_backend != BackendType.GgmlCuda)
                return;
            EnsureQuantBackendAvailable();   // memory query / preloads must hit the CUDA backend
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeBytes, out _))
                return;

            long headroomMb = long.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_VRAM_HEADROOM_MB"), out long hm) && hm > 0 ? hm : 2048;
            long copyBudgetMb = long.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_DEVICE_COPY_BUDGET_MB"), out long cm) && cm > 0 ? cm : 768;
            // Opt-in: re-home streamed weights into page-locked private copies (costs RAM equal to
            // the streamed bytes). Measured neutral on Windows/WDDM (the pageable mmap upload already
            // ran near DMA speed once the file cache was warm), so default off; may pay off on Linux.
            bool pinStreamed = Environment.GetEnvironmentVariable("DIFFUSION_PIN_STREAMED") == "1";
            long preloadBudget = freeBytes - headroomMb * 1024 * 1024;

            // Priority order: tied lm_head/embedding (read by the per-step lm_head tail), per-layer
            // non-expert weights (attention + dense MLP), then the per-layer expert stacks — the bulk
            // of the model; whatever does not fit streams per step.
            var priority = new List<(IntPtr key, IntPtr host, int type, long ne0, long ne1, long bytes)>();
            void AddQuant(string name)
            {
                if (_quantWeights.TryGetValue(name, out var qw) && qw.HasHostData)
                    priority.Add((qw.CacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes));
            }
            AddQuant("token_embd.weight");
            int L = Config.NumLayers;
            for (int l = 0; l < L; l++)
            {
                AddQuant($"blk.{l}.attn_q.weight");
                AddQuant($"blk.{l}.attn_k.weight");
                AddQuant($"blk.{l}.attn_v.weight");
                AddQuant($"blk.{l}.attn_output.weight");
                AddQuant($"blk.{l}.ffn_gate.weight");
                AddQuant($"blk.{l}.ffn_up.weight");
                AddQuant($"blk.{l}.ffn_down.weight");
            }
            for (int l = 0; l < L; l++)
            {
                var gu = _stackedGateUp[l];
                var dn = _stackedDown[l];
                if (gu != null) priority.Add((gu.Data, gu.Data, gu.GgmlType, gu.PerExpertNe0, gu.PerExpertNe1 * _numExperts, gu.TotalRawBytes));
                if (dn != null) priority.Add((dn.Data, dn.Data, dn.GgmlType, dn.PerExpertNe0, dn.PerExpertNe1 * _numExperts, dn.TotalRawBytes));
            }

            long preloadedBytes = 0;
            int preloadedCount = 0;
            long streamedBytes = 0;
            int streamedCount = 0;
            long pinnedBytes = 0;
            foreach (var w in priority)
            {
                if (preloadedBytes + w.bytes <= preloadBudget)
                {
                    try
                    {
                        GgmlBasicOps.PreloadQuantizedWeight(w.key, w.host, w.type, w.ne0, w.ne1, w.bytes);
                        preloadedBytes += w.bytes;
                        preloadedCount++;
                        continue;
                    }
                    catch (Exception)
                    {
                        // Device allocation failed despite the budget (fragmentation); treat the rest
                        // as streamed.
                    }
                }
                streamedBytes += w.bytes;
                streamedCount++;
                // Re-home the streamed weight into a page-locked private copy so its per-step PCIe
                // upload uses the fast DMA path (~2x pageable mmap throughput; the mmap itself cannot
                // be page-locked on Windows). Costs RAM equal to the streamed bytes, so opt out with
                // DIFFUSION_PIN_STREAMED=0. Unregistered + freed in Dispose.
                if (pinStreamed)
                {
                    IntPtr copy = IntPtr.Zero;
                    try
                    {
                        copy = GgmlBasicOps.AlignedAlloc(w.bytes);
                        if (copy != IntPtr.Zero)
                        {
                            unsafe { Buffer.MemoryCopy((void*)w.host, (void*)copy, w.bytes, w.bytes); }
                            if (GgmlBasicOps.TryRegisterPinnedHostBuffer(copy, w.bytes))
                            {
                                _pinnedHostRegions.Add(copy);
                                _pinnedStreamCopies[w.key] = copy;
                                pinnedBytes += w.bytes;
                            }
                            else
                            {
                                GgmlBasicOps.AlignedFree(copy);
                            }
                        }
                    }
                    catch (Exception)
                    {
                        if (copy != IntPtr.Zero && !_pinnedHostRegions.Contains(copy))
                            GgmlBasicOps.AlignedFree(copy);
                    }
                }
            }

            bool everythingFits = streamedCount == 0;
            string segEnv = Environment.GetEnvironmentVariable("DIFFUSION_SEGMENTED_DECODE");
            _segmentedDecode = segEnv == "1" || (segEnv != "0" && !everythingFits);
            if (!everythingFits)
            {
                // Cap incidental device copies (prompt K/V, decode masks, activations bound by per-op
                // kernels) so they cannot push VRAM past physical either. When everything fits there is
                // no oversubscription risk, so the legacy unlimited behaviour is kept.
                GgmlBasicOps.SetDeviceCopyBudget(copyBudgetMb * 1024 * 1024);
            }
            Console.WriteLine(
                $"  CUDA weight residency: preloaded {preloadedBytes / 1024 / 1024} MB / {preloadedCount} tensors " +
                $"(free VRAM {freeBytes / 1024 / 1024} MB, headroom {headroomMb} MB); " +
                (everythingFits
                    ? "model fully resident."
                    : $"streaming {streamedBytes / 1024 / 1024} MB / {streamedCount} tensors per step " +
                      $"({pinnedBytes / 1024 / 1024} MB page-locked); segmented decode={(_segmentedDecode ? "on" : "off")}, device-copy budget {copyBudgetMb} MB."));
        }

        private void PrecomputeRoPE()
        {
            int localHalf = _localHeadDim / 2;
            _ropeFreqsLocal = new float[localHalf];
            for (int i = 0; i < localHalf; i++)
                _ropeFreqsLocal[i] = (float)(1.0 / Math.Pow(_ropeLocalBase, 2.0 * i / _localHeadDim));

            int globalHalf = _globalHeadDim / 2;
            _ropeFreqsGlobal = new float[globalHalf];
            _weights.TryGetValue("rope_freqs.weight", out var ft);
            _ropeFreqsRawTensor = ft;   // raw factors for the on-device GGML freq-factors kernel
            float[] freqFactors = ft != null ? TensorToFloatArray(ft) : null;
            for (int i = 0; i < globalHalf; i++)
            {
                double freq = 1.0 / Math.Pow(_ropeGlobalBase, 2.0 * i / _globalHeadDim);
                if (freqFactors != null && i < freqFactors.Length)
                    freq /= freqFactors[i];
                _ropeFreqsGlobal[i] = (float)freq;
            }
        }

        private Tensor GetOnes(int dim)
        {
            if (!_onesByDim.TryGetValue(dim, out var t))
            {
                t = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(t, 1f);
                _onesByDim[dim] = t;
            }
            return t;
        }

        // ===================================================================================
        //  Core per-step forward: runs the bidirectional [prompt|canvas] graph and returns the
        //  canvas logits [C, vocab] (after final softcap) as a flat float[C*vocab].
        //  scPrevLogits/scUse/prevTempInv drive self-conditioning (scPrevLogits is [C*vocab] raw
        //  logits from the previous step; scUse is the {0,1} gate; pass null/0 to disable SC).
        // ===================================================================================
        public unsafe float[] ForwardCanvas(int[] tokens, int promptLen,
            float[] scPrevLogits = null, float scUse = 0f, float prevTempInv = 1f)
        {
            _swForward.Start();
            int N = tokens.Length;
            int P = promptLen;
            int C = N - P;
            int D = Config.HiddenSize;
            float eps = Config.Eps;

            // 1) embeddings, region-aware
            long ts = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);          // [N, D]
            Ops.Mul(hidden, hidden, MathF.Sqrt(D));     // embed_scale = sqrt(n_embd)
            EmbedCanvasRegion(hidden, P, C, scPrevLogits, scUse, prevTempInv);
            _tEmbed += Stopwatch.GetTimestamp() - ts;

            // 2) transformer stack (all glue ops are on-device; per-N caches are rebuilt lazily)
            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, N, P, C);
            }

            // 4) final norm + tied lm-head over canvas positions only
            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor canvasHidden;
            using (var view = normed.Narrow(0, P, C))
                canvasHidden = Ops.NewContiguous(view);
            normed.Dispose();

            ts = Stopwatch.GetTimestamp();
            Tensor logits = LinearForward(canvasHidden, "token_embd.weight");   // [C, vocab]
            canvasHidden.Dispose();
            _tLmHead += Stopwatch.GetTimestamp() - ts;

            if (_finalLogitSoftcap > 0f)
            {
                Ops.Mul(logits, logits, 1f / _finalLogitSoftcap);
                Ops.Tanh(logits, logits);
                Ops.Mul(logits, logits, _finalLogitSoftcap);
            }

            int total = C * Config.VocabSize;
            float[] result = ReadbackLogits(logits, total);
            logits.Dispose();

            _swForward.Stop();
            return result;
        }

        /// <summary>One device->host read of the canvas logits. On GGML the storage is host-mapped so a
        /// pointer copy into the reusable buffer is cheapest; on cuda/mlx GetElementsAsFloat is the single
        /// read-only sync (GetFloatPtr would also force a host->device re-upload).</summary>
        private unsafe float[] ReadbackLogits(Tensor logits, int total)
        {
            if (IsGgmlBackend)
            {
                if (_canvasLogits == null || _canvasLogits.Length != total)
                    _canvasLogits = new float[total];
                float* lp = GetFloatPtr(logits);
                fixed (float* dst = _canvasLogits)
                    Buffer.MemoryCopy(lp, dst, (long)total * 4, (long)total * 4);
                return _canvasLogits;
            }
            return logits.GetElementsAsFloat(total);
        }

        /// <summary>Replace the canvas rows of <paramref name="hidden"/> with
        /// rms_norm_noscale(embed*sqrt [+ self-conditioning signal]).</summary>
        private unsafe void EmbedCanvasRegion(Tensor hidden, int P, int C,
            float[] scPrevLogits, float scUse, float prevTempInv)
        {
            int D = Config.HiddenSize;
            float eps = Config.Eps;
            using var canvasView = hidden.Narrow(0, P, C);   // [C, D] contiguous block

            if (_scEnabled && scPrevLogits != null && scUse != 0f)
            {
                long ts = Stopwatch.GetTimestamp();
                using var scSignal = ComputeSelfConditioning(scPrevLogits, C, prevTempInv); // [C, D]
                Ops.Mul(scSignal, scSignal, scUse);
                Ops.Add(canvasView, canvasView, scSignal);
                _tSc += Stopwatch.GetTimestamp() - ts;
            }

            Ops.RMSNorm(canvasView, canvasView, GetOnes(D), null, eps);
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int N, int P, int C)
        {
            string prefix = $"blk.{layer}";
            float eps = Config.Eps;

            long ts = Stopwatch.GetTimestamp();
            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");
            Tensor attnOut = Attention(attnNormed, layer, prefix, N, P);

            // post-attention norm + residual
            Ops.RMSNorm(attnOut, attnOut, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();
            ProfSync(attnOut);
            _tAttn += Stopwatch.GetTimestamp() - ts;

            attnOut = FeedForward(attnOut, layer, prefix, N);

            // region-aware per-layer scalar
            ApplyRegionScalar(attnOut, layer, P, C);
            return attnOut;
        }

        /// <summary>Dense gated-GELU MLP (shared expert) + 128-expert MoE, summed, post-norm + residual.
        /// Shared by the unified / prefill / decode forwards. Returns attnOut + post_ffw_norm(dense+moe).</summary>
        private Tensor FeedForward(Tensor attnOut, int layer, string prefix, int N)
        {
            float eps = Config.Eps;
            long ts = Stopwatch.GetTimestamp();
            Tensor mlpOut = DenseMlp(attnOut, prefix, N);
            Ops.RMSNorm(mlpOut, mlpOut, _weights[$"{prefix}.post_ffw_norm_1.weight"], null, eps);
            ProfSync(mlpOut);
            _tDense += Stopwatch.GetTimestamp() - ts;

            ts = Stopwatch.GetTimestamp();
            using (Tensor moeOut = MoEForward(attnOut, layer, prefix, N))
            {
                Ops.RMSNorm(moeOut, moeOut, _weights[$"{prefix}.post_ffw_norm_2.weight"], null, eps);
                Ops.Add(mlpOut, mlpOut, moeOut);
            }
            ProfSync(mlpOut);
            _tMoe += Stopwatch.GetTimestamp() - ts;

            Ops.RMSNorm(mlpOut, mlpOut, _weights[$"{prefix}.post_ffw_norm.weight"], null, eps);
            Ops.Add(attnOut, attnOut, mlpOut);
            mlpOut.Dispose();
            return attnOut;
        }

        private void ApplyRegionScalar(Tensor x, int layer, int P, int C)
        {
            float enc = _encScale[layer];
            float dec = _decScale[layer];
            if (P > 0 && enc != 1f)
            {
                using var prompt = x.Narrow(0, 0, P);
                Ops.Mul(prompt, prompt, enc);
            }
            if (C > 0 && dec != 1f)
            {
                using var canvas = x.Narrow(0, P, C);
                Ops.Mul(canvas, canvas, dec);
            }
        }

        private Tensor DenseMlp(Tensor input, string prefix, int N)
        {
            using var normed = RMSNormOp(input, $"{prefix}.ffn_norm.weight");
            Tensor gate = LinearForward(normed, $"{prefix}.ffn_gate.weight");
            using (Tensor up = LinearForward(normed, $"{prefix}.ffn_up.weight"))
                Ops.GELUMul(gate, gate, up);   // gate = gelu(gate) * up
            Tensor down = LinearForward(gate, $"{prefix}.ffn_down.weight");
            gate.Dispose();
            return down;
        }

        // ===================================================================================
        //  Attention: region-aware bidirectional, no KV cache. Q/K per-head RMSNorm, unweighted
        //  V RMSNorm, NeoX RoPE, attention scale = 1.0. Fully on-device (backend Ops): the scores are
        //  computed with a batched matmul, the region-aware mask is added as a precomputed additive
        //  bias tensor, then softmax and the value matmul. This keeps the whole attention block on the
        //  GPU for ggml_metal / ggml_cuda / mlx / cuda (no host round-trips).
        // ===================================================================================
        private Tensor Attention(Tensor input, int layer, string prefix, int N, int P)
        {
            bool local = _isLocal[layer];
            int hd = _headDim[layer];
            int qHeads = Config.NumHeads;
            int kvHeads = _kvHeads[layer];
            int groupSize = qHeads / kvHeads;
            float eps = Config.Eps;

            Tensor q = LinearForward(input, $"{prefix}.attn_q.weight");   // [N, qHeads*hd]
            Tensor k = LinearForward(input, $"{prefix}.attn_k.weight");   // [N, kvHeads*hd]
            Tensor v;
            if (_hasVProj[layer])
            {
                v = LinearForward(input, $"{prefix}.attn_v.weight");
            }
            else
            {
                v = new Tensor(_allocator, DType.Float32, k.Sizes);       // global layers: V == raw K
                Ops.Copy(v, k);
            }

            // per-head Q/K norm (with weight), unweighted V norm
            using (var qr = q.View(N * qHeads, hd))
                Ops.RMSNorm(qr, qr, _weights[$"{prefix}.attn_q_norm.weight"], null, eps);
            using (var kr = k.View(N * kvHeads, hd))
                Ops.RMSNorm(kr, kr, _weights[$"{prefix}.attn_k_norm.weight"], null, eps);
            using (var vr = v.View(N * kvHeads, hd))
                Ops.RMSNorm(vr, vr, GetOnes(hd), null, eps);

            if (!_useDeviceGlue)
            {
                // CPU fast path: SIMD raw-pointer RoPE + region-aware attention (host-resident).
                var (cos, sin) = GetCosSinHost(N, local);
                ApplyNeoXRoPERaw(q, qHeads, hd, N, cos, sin);
                ApplyNeoXRoPERaw(k, kvHeads, hd, N, cos, sin);
                Tensor cpuResult = new Tensor(_allocator, DType.Float32, N, qHeads * hd);
                AttentionRegionAware(q, k, v, cpuResult, N, P, qHeads, kvHeads, hd, local);
                q.Dispose(); k.Dispose(); v.Dispose();
                using (cpuResult)
                    return LinearForward(cpuResult, $"{prefix}.attn_output.weight");
            }

            // GPU path: everything stays device-resident.
            // NeoX RoPE on Q and K (in place, on-device); V is not rotated
            ApplyRoPE(q, qHeads, hd, N, 0, local);
            ApplyRoPE(k, kvHeads, hd, N, 0, local);

            // head-first [heads, N, hd]
            using var qH = ReshapeToHeads(q, qHeads, N, hd);
            using var kH = ReshapeToHeads(k, kvHeads, N, hd);
            using var vH = ReshapeToHeads(v, kvHeads, N, hd);
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor mask = GetAttentionMask(N, P, local);
            using var result = AttnCoreHeadFirst(qH, kH, vH, mask, N, N, qHeads, kvHeads, hd);
            return LinearForward(result, $"{prefix}.attn_output.weight");
        }

        /// <summary>Shared on-device attention core: expand KV heads, batched Q·Kᵀ (scale 1.0), add the
        /// additive mask (null = none), softmax, batched ·V, and reshape back to flat [qLen, qHeads*hd].</summary>
        private Tensor AttnCoreHeadFirst(Tensor qH, Tensor kHfull, Tensor vHfull, Tensor mask,
            int qLen, int kvLen, int qHeads, int kvHeads, int hd)
        {
            int groupSize = qHeads / kvHeads;
            using var kExp = ExpandKVHeads(kHfull, groupSize, kvLen);   // [qHeads, kvLen, hd]
            using var vExp = ExpandKVHeads(vHfull, groupSize, kvLen);
            using var kT = kExp.Transpose(1, 2);                        // [qHeads, hd, kvLen]
            using var scores = new Tensor(_allocator, DType.Float32, qHeads, qLen, kvLen);
            Ops.AddmmBatch(scores, 0f, scores, 1f, qH, kT);
            if (mask != null) Ops.Add(scores, scores, mask);
            Ops.Softmax(scores, scores);
            using var attnOut = new Tensor(_allocator, DType.Float32, qHeads, qLen, hd);
            Ops.AddmmBatch(attnOut, 0f, attnOut, 1f, scores, vExp);
            return ReshapeFromHeads(attnOut, qHeads, qLen, hd);          // [qLen, qHeads*hd]
        }

        /// <summary>SIMD raw-pointer region-aware attention for CPU backends (the contiguous allowed-key
        /// window per query lets us score only the unmasked interval). Scale 1.0.</summary>
        private unsafe void AttentionRegionAware(Tensor qT, Tensor kT, Tensor vT, Tensor outT,
            int N, int P, int qHeads, int kvHeads, int hd, bool local)
        {
            float* q = GetFloatPtr(qT);
            float* k = GetFloatPtr(kT);
            float* v = GetFloatPtr(vT);
            float* o = GetFloatPtr(outT);
            int groupSize = qHeads / kvHeads;
            int swa = _slidingWindow;

            Parallel.For(0, qHeads, h =>
            {
                int kvHead = h / groupSize;
                float[] scores = new float[N];
                for (int qi = 0; qi < N; qi++)
                {
                    bool qCanvas = qi >= P;
                    AllowedRange(qi, qCanvas, P, N, local, swa, out int klo, out int khi);
                    if (khi <= klo) khi = klo + 1;

                    float* qVec = q + ((long)qi * qHeads + h) * hd;
                    float maxScore = float.NegativeInfinity;
                    for (int kj = klo; kj < khi; kj++)
                    {
                        float dot = VecDot(qVec, k + ((long)kj * kvHeads + kvHead) * hd, hd);
                        scores[kj] = dot;
                        if (dot > maxScore) maxScore = dot;
                    }
                    float sum = 0f;
                    for (int kj = klo; kj < khi; kj++)
                    {
                        float e = MathF.Exp(scores[kj] - maxScore);
                        scores[kj] = e;
                        sum += e;
                    }
                    float inv = sum > 0f ? 1f / sum : 0f;
                    float* oVec = o + ((long)qi * qHeads + h) * hd;
                    VecZero(oVec, hd);
                    for (int kj = klo; kj < khi; kj++)
                        VecScaleAdd(oVec, v + ((long)kj * kvHeads + kvHead) * hd, scores[kj] * inv, hd);
                }
            });
            InvalidateTensorDeviceCache(outT);
        }

        private (float[] cos, float[] sin) GetCosSinHost(int N, bool isLocal)
        {
            if (_cosSinHostN != N)
            {
                BuildCosSin(N, _ropeFreqsLocal, out _cosLocalHost, out _sinLocalHost);
                BuildCosSin(N, _ropeFreqsGlobal, out _cosGlobalHost, out _sinGlobalHost);
                _cosSinHostN = N;
            }
            return isLocal ? (_cosLocalHost, _sinLocalHost) : (_cosGlobalHost, _sinGlobalHost);
        }

        // ---- on-device NeoX RoPE -----------------------------------------------------------
        /// <summary>In-place NeoX RoPE on a flat [N, numHeads*hd] tensor, on-device. Local layers use a
        /// single base; global layers use proportional/NTK RoPE (per-frequency <c>rope_freqs</c>). GGML
        /// backends use the native freq-factors kernel; mlx uses a fused cos/sin kernel; other backends
        /// fall back to a cos/sin host kernel.</summary>
        private void ApplyRoPE(Tensor data, int numHeads, int hd, int N, int startPos, bool isLocal)
        {
            Tensor pos = GetRoPEPositions(N, numHeads, startPos);
            using (var view = data.View(1, N, numHeads, hd))
            {
                if (isLocal)
                {
                    Ops.RoPEEx(view, view, pos, hd, NeoXRopeMode, 0, _ropeLocalBase, 1f, 0f, 1f, 0f, 0f);
                    return;
                }
                if (IsGgmlBackend && _ropeFreqsRawTensor != null)
                {
                    GgmlBasicOps.RoPEExWithFreqFactors(view, view, pos, _ropeFreqsRawTensor,
                        hd, NeoXRopeMode, 0, _ropeGlobalBase, 1f, 0f, 1f, 0f, 0f);
                    return;
                }
            }

            // non-GGML global: proportional RoPE via cos/sin tables (freqs already fold in rope_freqs)
            int half = hd / 2;
            var (cosT, sinT) = GetCosSinGlobalTensors(N, half);
            if (_backend == BackendType.Mlx &&
                MlxFusedOps.TryNeoXRoPEFlatInPlace(data, cosT, sinT, numHeads, N, hd, half))
                return;

            ApplyNeoXRoPERaw(data, numHeads, hd, N, _cosGlobalHost, _sinGlobalHost);
        }

        private Tensor GetRoPEPositions(int N, int numHeads, int startPos)
        {
            if (_ropePosN != N)
            {
                foreach (var t in _ropePosCache.Values) t?.Dispose();
                _ropePosCache.Clear();
                _ropePosN = N;
            }
            long key = ((long)numHeads << 32) | (uint)startPos;
            if (!_ropePosCache.TryGetValue(key, out var pos))
            {
                int[] positions = new int[N * numHeads];
                for (int s = 0; s < N; s++)
                    for (int h = 0; h < numHeads; h++)
                        positions[s * numHeads + h] = startPos + s;
                pos = CreateIntTensor(positions, N * numHeads);
                _ropePosCache[key] = pos;
            }
            return pos;
        }

        private (Tensor cos, Tensor sin) GetCosSinGlobalTensors(int N, int half)
        {
            if (_cosSinGlobalN != N)
            {
                BuildCosSin(N, _ropeFreqsGlobal, out _cosGlobalHost, out _sinGlobalHost);
                _cosGlobalTensor?.Dispose();
                _sinGlobalTensor?.Dispose();
                _cosGlobalTensor = CreateFloatTensor(_cosGlobalHost, N * half);
                _sinGlobalTensor = CreateFloatTensor(_sinGlobalHost, N * half);
                _cosSinGlobalN = N;
            }
            return (_cosGlobalTensor, _sinGlobalTensor);
        }

        // ---- region-aware additive attention mask -----------------------------------------
        /// <summary>Returns a cached additive attention mask [maskHeads, N, N] (0 where the key is
        /// attended, large-negative where masked) for the given layer type. maskHeads is 1 (broadcast)
        /// on GGML/MLX and qHeads on cuda (which doesn't broadcast element-wise adds).</summary>
        private Tensor GetAttentionMask(int N, int P, bool local)
        {
            if (_maskN != N || _maskP != P)
            {
                _maskLocal?.Dispose(); _maskGlobal?.Dispose();
                _maskLocal = null; _maskGlobal = null;
                _maskN = N; _maskP = P;
            }
            if (local && _maskLocal != null) return _maskLocal;
            if (!local && _maskGlobal != null) return _maskGlobal;

            int qHeads = Config.NumHeads;
            int maskHeads = _backend == BackendType.Cuda ? qHeads : 1;
            var data = new float[(long)maskHeads * N * N];
            for (int qi = 0; qi < N; qi++)
            {
                bool qCanvas = qi >= P;
                AllowedRange(qi, qCanvas, P, N, local, _slidingWindow, out int klo, out int khi);
                if (khi <= klo) khi = klo + 1;
                long rowBase = (long)qi * N;
                for (int kj = 0; kj < N; kj++)
                {
                    float val = (kj >= klo && kj < khi) ? 0f : MaskNeg;
                    for (int h = 0; h < maskHeads; h++)
                        data[(long)h * N * N + rowBase + kj] = val;
                }
            }
            var mask = CreateFloatTensor(data, maskHeads, N, N);
            if (local) _maskLocal = mask; else _maskGlobal = mask;
            return mask;
        }

        // ===================================================================================
        //  Prompt-KV caching (PKV): PrefillPrompt computes the prompt's per-layer K/V once; DecodeCanvas
        //  then processes only the canvas each denoising step, reading the cached prompt K/V. This is the
        //  canonical llama.cpp/vLLM diffusion-gemma optimization — it removes the prompt's projection /
        //  attention / dense-MLP / MoE-matmul work from every step (computed once instead of S times),
        //  which is a large saving for long prompts (system prompt + chat history). GPU path only.
        // ===================================================================================

        /// <summary>Project + per-head-norm + RoPE (positions startPos..startPos+seqLen-1) and reshape Q/K/V
        /// to head-first [heads, seqLen, hd]. Shared by the unified, prefill and decode forwards.</summary>
        private void ComputeQKVHeadFirst(Tensor input, int layer, string prefix, int seqLen, int startPos,
            out Tensor qH, out Tensor kH, out Tensor vH)
        {
            bool local = _isLocal[layer];
            int hd = _headDim[layer];
            int qHeads = Config.NumHeads;
            int kvHeads = _kvHeads[layer];
            float eps = Config.Eps;

            Tensor q = LinearForward(input, $"{prefix}.attn_q.weight");
            Tensor k = LinearForward(input, $"{prefix}.attn_k.weight");
            Tensor v;
            if (_hasVProj[layer]) v = LinearForward(input, $"{prefix}.attn_v.weight");
            else { v = new Tensor(_allocator, DType.Float32, k.Sizes); Ops.Copy(v, k); }

            using (var qr = q.View(seqLen * qHeads, hd))
                Ops.RMSNorm(qr, qr, _weights[$"{prefix}.attn_q_norm.weight"], null, eps);
            using (var kr = k.View(seqLen * kvHeads, hd))
                Ops.RMSNorm(kr, kr, _weights[$"{prefix}.attn_k_norm.weight"], null, eps);
            using (var vr = v.View(seqLen * kvHeads, hd))
                Ops.RMSNorm(vr, vr, GetOnes(hd), null, eps);

            ApplyRoPE(q, qHeads, hd, seqLen, startPos, local);
            ApplyRoPE(k, kvHeads, hd, seqLen, startPos, local);

            qH = ReshapeToHeads(q, qHeads, seqLen, hd); q.Dispose();
            kH = ReshapeToHeads(k, kvHeads, seqLen, hd); k.Dispose();
            vH = ReshapeToHeads(v, kvHeads, seqLen, hd); v.Dispose();
        }

        /// <summary>Run the prompt through every layer once and cache each layer's prompt K/V (head-first).
        /// The prompt uses scaled-embedding input, causal attention, and the encoder per-layer scalar.</summary>
        public void PrefillPrompt(int[] promptTokens)
        {
            if (!_pkvEnabled)
                throw new InvalidOperationException("Prompt-KV caching is not enabled for this backend.");

            AllocPromptStore();
            long pt0 = Stopwatch.GetTimestamp();
            _promptLen = PrefillPromptInto(promptTokens, _promptK, _promptV);
            if (_stepTime) { _ = _promptK[Config.NumLayers - 1].GetElementsAsFloat(1); long now = Stopwatch.GetTimestamp(); double ms = (now - pt0) * 1000.0 / Stopwatch.Frequency; Console.Error.WriteLine($"[prefill] {_promptLen} tokens in {ms:F1} ms = {_promptLen * 1000.0 / ms:F1} tok/s"); }
        }

        /// <summary>Core prompt prefill: run the prompt through every layer once and store each layer's
        /// prompt K/V (head-first [kvHeads, P, hd]) into <paramref name="outK"/>/<paramref name="outV"/>.
        /// Shared by the single-request (<see cref="PrefillPrompt"/>) and batched (<see cref="PrefillSeq"/>)
        /// paths. The caller owns/releases the K/V tensors. Returns the prompt length P.</summary>
        private int PrefillPromptInto(int[] promptTokens, Tensor[] outK, Tensor[] outV)
        {
            int P = promptTokens.Length;
            int D = Config.HiddenSize;
            float eps = Config.Eps;

            Tensor hidden = Embedding(promptTokens);
            Ops.Mul(hidden, hidden, MathF.Sqrt(D));   // prompt = embed*sqrt(n_embd) (no rms-norm, no SC)

            for (int l = 0; l < Config.NumLayers; l++)
            {
                string prefix = $"blk.{l}";
                bool local = _isLocal[l];
                int hd = _headDim[l];
                int qHeads = Config.NumHeads;
                int kvHeads = _kvHeads[l];

                using var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");
                ComputeQKVHeadFirst(normed, l, prefix, P, 0, out var qH, out var kH, out var vH);

                // cache this layer's prompt K/V for the denoising steps
                outK[l] = Ops.NewContiguous(kH);
                outV[l] = Ops.NewContiguous(vH);

                Tensor mask = GetAttentionMask(P, P, local);   // prompt-causal (SWA-clipped on local layers)
                Tensor attnOut;
                using (qH) using (kH) using (vH)
                using (var attnRes = AttnCoreHeadFirst(qH, kH, vH, mask, P, P, qHeads, kvHeads, hd))
                    attnOut = LinearForward(attnRes, $"{prefix}.attn_output.weight");

                Ops.RMSNorm(attnOut, attnOut, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
                Ops.Add(attnOut, attnOut, hidden);
                hidden.Dispose();

                attnOut = FeedForward(attnOut, l, prefix, P);
                if (_encScale[l] != 1f) Ops.Mul(attnOut, attnOut, _encScale[l]);   // encoder scalar
                hidden = attnOut;
            }
            hidden.Dispose();   // prompt hidden states aren't needed; only the cached K/V are
            return P;
        }

        /// <summary>Denoise one canvas step: process only the canvas (positions P..P+C-1), reading the
        /// cached prompt K/V, and return the canvas logits [C, vocab]. Uses the canvas embedding
        /// (rms-norm + self-conditioning) and the decoder per-layer scalar.</summary>
        public unsafe float[] DecodeCanvas(int[] canvasTokens, float[] scPrevLogits, float scUse, float prevTempInv)
            => DecodeCanvasCore(_promptK, _promptV, _promptLen, canvasTokens, scPrevLogits, scUse, prevTempInv);

        /// <summary>Core single-canvas decode shared by the instance-cached single-request path
        /// (<see cref="DecodeCanvas"/>) and the batched scheduler's B==1 fast path: the prompt K/V are
        /// supplied by the caller (<paramref name="pk"/>/<paramref name="pv"/>, length NumLayers) rather
        /// than read from the instance fields, so the fused decode kernel can be reused per-sequence.</summary>
        private unsafe float[] DecodeCanvasCore(Tensor[] pk, Tensor[] pv, int P,
            int[] canvasTokens, float[] scPrevLogits, float scUse, float prevTempInv)
        {
            _stepT0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            int C = canvasTokens.Length;
            int D = Config.HiddenSize;
            float eps = Config.Eps;

            long ts = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(canvasTokens);
            Ops.Mul(hidden, hidden, MathF.Sqrt(D));
            if (_scEnabled && scPrevLogits != null && scUse != 0f)
            {
                long tsc = Stopwatch.GetTimestamp();
                using var scSignal = ComputeSelfConditioning(scPrevLogits, C, prevTempInv);
                Ops.Mul(scSignal, scSignal, scUse);
                Ops.Add(hidden, hidden, scSignal);
                _tSc += Stopwatch.GetTimestamp() - tsc;
            }
            Ops.RMSNorm(hidden, hidden, GetOnes(D), null, eps);   // canvas = rms_norm_noscale(embed [+ SC])
            _tEmbed += Stopwatch.GetTimestamp() - ts;

            // Fast path: all transformer layers (attention + dense + MoE) in one fused GGML graph with the
            // canvas hidden staying on-device across all layers (the throughput win). C# does the lm_head tail.
            bool fusedLayers = false;
            long tLayers0 = Stopwatch.GetTimestamp();
            if (IsGgmlBackend && _fusedDecodeEnabled && _fusedDecodeOk)
            {
                bool ok = _segmentedDecode
                    ? TryFusedModelLayersSegmented(hidden, C, P, pk, pv)
                    : TryFusedModelLayers(hidden, C, P, pk, pv);
                if (ok) fusedLayers = true;
                else _fusedDecodeOk = false;
            }
            long tLayers1 = Stopwatch.GetTimestamp();

            for (int l = 0; !fusedLayers && l < Config.NumLayers; l++)
            {
                string prefix = $"blk.{l}";
                bool local = _isLocal[l];
                int hd = _headDim[l];
                int qHeads = Config.NumHeads;
                int kvHeads = _kvHeads[l];

                long tA = Stopwatch.GetTimestamp();
                using var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");
                ComputeQKVHeadFirst(normed, l, prefix, C, P, out var qH, out var kH, out var vH);

                Tensor attnOut;
                using (qH) using (kH) using (vH)
                using (var kFull = Ops.Concat(null, 1, pk[l], kH))   // [kvHeads, P+C, hd]
                using (var vFull = Ops.Concat(null, 1, pv[l], vH))
                {
                    Tensor dmask = GetDecodeMask(P, C, local);
                    using var attnRes = AttnCoreHeadFirst(qH, kFull, vFull, dmask, C, P + C, qHeads, kvHeads, hd);
                    attnOut = LinearForward(attnRes, $"{prefix}.attn_output.weight");
                }

                Ops.RMSNorm(attnOut, attnOut, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
                Ops.Add(attnOut, attnOut, hidden);
                hidden.Dispose();
                _tAttn += Stopwatch.GetTimestamp() - tA;

                attnOut = FeedForward(attnOut, l, prefix, C);
                if (_decScale[l] != 1f) Ops.Mul(attnOut, attnOut, _decScale[l]);   // decoder scalar
                hidden = attnOut;
            }

            // Fused lm_head tail (output_norm + lm_head + softcap in one dispatch) - separate small graph,
            // correct, and cheaper than the per-op RMSNorm + AddmmQuant + softcap + readback chain.
            if (IsGgmlBackend && !_fusedLmHeadTailDisabled && _fusedLmHeadTailOk)
            {
                float[] flLogits = TryFusedLmHead(hidden, C);
                if (flLogits != null)
                {
                    hidden.Dispose();
                    _swForward.Stop();
                    if (_stepTime)
                    {
                        long now = Stopwatch.GetTimestamp();
                        double f = 1000.0 / Stopwatch.Frequency;
                        Console.Error.WriteLine($"[decode] {(now - _stepT0) * f:F1} ms (layers={(tLayers1 - tLayers0) * f:F1} lmhead={(now - tLayers1) * f:F1})");
                    }
                    return flLogits;
                }
                _fusedLmHeadTailOk = false;
            }

            Tensor normedOut = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            ts = Stopwatch.GetTimestamp();
            Tensor logits = LinearForward(normedOut, "token_embd.weight");   // [C, vocab]
            normedOut.Dispose();
            if (_finalLogitSoftcap > 0f)
            {
                Ops.Mul(logits, logits, 1f / _finalLogitSoftcap);
                Ops.Tanh(logits, logits);
                Ops.Mul(logits, logits, _finalLogitSoftcap);
            }
            _tLmHead += Stopwatch.GetTimestamp() - ts;

            float[] result = ReadbackLogits(logits, C * Config.VocabSize);
            logits.Dispose();
            _swForward.Stop();
            if (_stepTime) { long now = Stopwatch.GetTimestamp(); Console.Error.WriteLine($"[decode] {(now - _stepT0) * 1000.0 / Stopwatch.Frequency:F1} ms"); }

            return result;
        }

        // ===================================================================================
        //  On-device sampled decode (CUDA): run the layers + lm_head, then compute argmax/entropy/
        //  multinomial/top-K on the device logits in one kernel — only the small per-position outputs
        //  cross PCIe. Self-conditioning reads the PREVIOUS step's device-computed top-K (no host logits).
        //  Returns false (caller falls back to the logits path) if any device step is unavailable.
        // ===================================================================================

        /// <summary>Instance-cached single-canvas decode + on-device sample (prompt-KV path).
        /// <paramref name="scPrevTopTokens"/>/<paramref name="scPrevTopProbs"/> are the PREVIOUS step's
        /// device-computed top-K (for self-conditioning; null/scUse=0 on the first step). <paramref name="u"/>
        /// is the pre-drawn per-position uniform for the multinomial. Fills the caller-owned output arrays.</summary>
        public unsafe bool DecodeCanvasSampled(int[] canvasTokens,
            int[] scPrevTopTokens, float[] scPrevTopProbs, float scUse, float tempInv, float[] u, int K,
            int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
            => DecodeCanvasSampledCore(_promptK, _promptV, _promptLen, canvasTokens, scPrevTopTokens, scPrevTopProbs,
                scUse, tempInv, u, K, argmaxOut, entropyOut, sampledOut, topTokensOut, topProbsOut);

        /// <summary>Per-sequence single-canvas decode + on-device sample (batched scheduler B==1 fast path).</summary>
        public unsafe bool DecodeCanvasSampledSeq(DiffusionSeqState seq, int[] canvasTokens,
            int[] scPrevTopTokens, float[] scPrevTopProbs, float scUse, float tempInv, float[] u, int K,
            int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
            => DecodeCanvasSampledCore(seq.PromptK, seq.PromptV, seq.PromptLen, canvasTokens, scPrevTopTokens, scPrevTopProbs,
                scUse, tempInv, u, K, argmaxOut, entropyOut, sampledOut, topTokensOut, topProbsOut);

        private unsafe bool DecodeCanvasSampledCore(Tensor[] pk, Tensor[] pv, int P,
            int[] canvasTokens, int[] scPrevTopTokens, float[] scPrevTopProbs, float scUse, float tempInv,
            float[] u, int K, int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
        {
            if (!SupportsDeviceSampling) return false;
            // device sampling needs the fused/segmented layer path (keeps hidden on-device for the tail)
            if (!(IsGgmlBackend && _fusedDecodeEnabled && _fusedDecodeOk)) return false;

            _stepT0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            int C = canvasTokens.Length;
            int D = Config.HiddenSize;
            float eps = Config.Eps;

            long ts = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(canvasTokens);
            Ops.Mul(hidden, hidden, MathF.Sqrt(D));
            if (_scEnabled && scPrevTopTokens != null && scUse != 0f)
            {
                long tsc = Stopwatch.GetTimestamp();
                using var scSignal = ComputeSelfConditioningFromTopK(scPrevTopTokens, scPrevTopProbs, C, K);
                Ops.Mul(scSignal, scSignal, scUse);
                Ops.Add(hidden, hidden, scSignal);
                _tSc += Stopwatch.GetTimestamp() - tsc;
            }
            Ops.RMSNorm(hidden, hidden, GetOnes(D), null, eps);
            _tEmbed += Stopwatch.GetTimestamp() - ts;

            long tLayers0 = Stopwatch.GetTimestamp();
            bool ok = _segmentedDecode
                ? TryFusedModelLayersSegmented(hidden, C, P, pk, pv)
                : TryFusedModelLayers(hidden, C, P, pk, pv);
            if (!ok) { _fusedDecodeOk = false; hidden.Dispose(); _swForward.Stop(); return false; }
            long tLayers1 = Stopwatch.GetTimestamp();

            bool sampled = TryFusedLmHeadSampleTail(hidden, C, tempInv, u, K,
                argmaxOut, entropyOut, sampledOut, topTokensOut, topProbsOut);
            hidden.Dispose();
            _swForward.Stop();
            if (!sampled) { _deviceSampleOk = false; return false; }
            if (_stepTime)
            {
                long now = Stopwatch.GetTimestamp();
                double f = 1000.0 / Stopwatch.Frequency;
                Console.Error.WriteLine($"[decode-sampled] {(now - _stepT0) * f:F1} ms (layers={(tLayers1 - tLayers0) * f:F1} lmhead+sample={(now - tLayers1) * f:F1})");
            }
            return true;
        }

        /// <summary>Fused lm_head + on-device sample tail: output_norm + lm_head produce device logits, then
        /// the CUDA kernel computes argmax/entropy/multinomial/top-K directly on them. No 268 MB readback.</summary>
        private unsafe bool TryFusedLmHeadSampleTail(Tensor hidden, int C, float tempInv, float[] u, int K,
            int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
        {
            try
            {
                if (!_quantWeights.TryGetValue("token_embd.weight", out var lmHead)) return false;
                if (!_weights.ContainsKey("output_norm.weight")) return false;
                int vocab = Config.VocabSize;
                bool wantTopK = K > 0 && topTokensOut != null && topProbsOut != null;
                fixed (float* uPtr = u)
                fixed (int* amPtr = argmaxOut)
                fixed (float* enPtr = entropyOut)
                fixed (int* smPtr = sampledOut)
                fixed (int* ttPtr = wantTopK ? topTokensOut : null)
                fixed (float* tpPtr = wantTopK ? topProbsOut : null)
                {
                    return GgmlBasicOps.TryDiffusionLmHeadSample(
                        (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, C,
                        WPtr("output_norm.weight"),
                        PinnedOr(lmHead.CacheKey), lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                        vocab, Config.Eps, _finalLogitSoftcap,
                        tempInv, (IntPtr)uPtr, wantTopK ? K : 0,
                        (IntPtr)amPtr, (IntPtr)enPtr, (IntPtr)smPtr, (IntPtr)ttPtr, (IntPtr)tpPtr);
                }
            }
            catch (Exception)
            {
                return false;
            }
        }

        // ===================================================================================
        //  Batched (multi-sequence) decode — the parallel-request throughput path.
        //  Multiple in-flight requests' canvases are processed in ONE forward per denoising step:
        //  the weight-bound work (embedding, dense MLP, 128-expert MoE, lm_head) runs once over all
        //  B*C canvas tokens (near-free extra throughput on the GPU since the experts/weights are read
        //  once for B canvases), while the only per-sequence work — attention over each sequence's own
        //  cached prompt K/V + its own canvas — loops over the B sequences. Per-sequence RoPE positions
        //  (canvas token i of sequence b lives at absolute position P_b+i) are supplied via an explicit
        //  positions tensor. Uses the per-op device path (the fused single-graph decode kernel is for the
        //  single-canvas case); B==1 still takes the fused fast path via DecodeCanvasSeq.
        //  GGML/device-glue backends only.
        // ===================================================================================

        /// <summary>Allocate a fresh per-sequence decode state (its own prompt K/V store). The caller must
        /// <see cref="DisposeSeqState"/> it when the request finishes.</summary>
        public DiffusionSeqState CreateSeqState()
        {
            return new DiffusionSeqState(Config.NumLayers);
        }

        /// <summary>Free a per-sequence state's cached prompt K/V device buffers (invalidate-then-dispose,
        /// matching <see cref="ReleasePromptKvTensor"/>) and reset its length.</summary>
        public void DisposeSeqState(DiffusionSeqState seq)
        {
            if (seq == null) return;
            if (seq.PromptK != null)
                for (int l = 0; l < seq.PromptK.Length; l++) ReleasePromptKvTensor(ref seq.PromptK[l]);
            if (seq.PromptV != null)
                for (int l = 0; l < seq.PromptV.Length; l++) ReleasePromptKvTensor(ref seq.PromptV[l]);
            seq.PromptLen = -1;
        }

        /// <summary>Prefill one sequence's prompt into its own K/V store (re-runnable per block; releases the
        /// previous block's K/V first, like <see cref="AllocPromptStore"/>).</summary>
        public void PrefillSeq(DiffusionSeqState seq, int[] promptTokens)
        {
            if (!_pkvEnabled)
                throw new InvalidOperationException("Prompt-KV caching is not enabled for this backend.");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ReleasePromptKvTensor(ref seq.PromptK[l]);
                ReleasePromptKvTensor(ref seq.PromptV[l]);
            }
            seq.PromptLen = PrefillPromptInto(promptTokens, seq.PromptK, seq.PromptV);
        }

        /// <summary>Single-canvas decode for a request held in a <see cref="DiffusionSeqState"/> (the batched
        /// scheduler's B==1 fast path — keeps the fused decode kernel for solo requests).</summary>
        public float[] DecodeCanvasSeq(DiffusionSeqState seq, int[] canvasTokens,
            float[] scPrevLogits, float scUse, float prevTempInv)
            => DecodeCanvasCore(seq.PromptK, seq.PromptV, seq.PromptLen, canvasTokens, scPrevLogits, scUse, prevTempInv);

        /// <summary>Decode B canvases in one batched forward, returning each sequence's canvas logits
        /// [C, vocab]. <paramref name="canvases"/>/<paramref name="scPrev"/>/<paramref name="scUse"/>/
        /// <paramref name="prevTempInv"/> are per-sequence (length B). Per-op device path; the heavy matmuls
        /// run once over all B*C rows, attention loops per sequence over its own prompt K/V.</summary>
        public unsafe float[][] DecodeCanvasBatched(DiffusionSeqState[] seqs, int[][] canvases,
            float[][] scPrev, float[] scUse, float[] prevTempInv)
        {
            int B = seqs.Length;
            int C = _canvasLength;
            int D = Config.HiddenSize;
            int vocab = Config.VocabSize;
            float eps = Config.Eps;
            int N = B * C;
            int qHeadsAll = Config.NumHeads;

            int[] P = new int[B];
            for (int b = 0; b < B; b++) P[b] = seqs[b].PromptLen;

            // 1) batched embedding of every canvas, region-aware (canvas = rms_norm_noscale(embed [+ SC])).
            int[] allTokens = new int[N];
            for (int b = 0; b < B; b++) Array.Copy(canvases[b], 0, allTokens, b * C, C);
            Tensor hidden = Embedding(allTokens);          // [N, D]
            Ops.Mul(hidden, hidden, MathF.Sqrt(D));
            if (_scEnabled)
            {
                for (int b = 0; b < B; b++)
                {
                    if (scPrev == null || scPrev[b] == null || scUse[b] == 0f) continue;
                    using var scSignal = ComputeSelfConditioning(scPrev[b], C, prevTempInv[b]);   // [C, D]
                    Ops.Mul(scSignal, scSignal, scUse[b]);
                    using var slice = hidden.Narrow(0, (long)b * C, C);
                    Ops.Add(slice, slice, scSignal);
                }
            }
            Ops.RMSNorm(hidden, hidden, GetOnes(D), null, eps);

            // per-call caches (avoid thrashing the instance dimension-keyed caches): RoPE positions keyed by
            // head count, and decode masks keyed by (P, layerType). Disposed at the end of this call.
            var posCache = new Dictionary<int, Tensor>();
            var maskCache = new Dictionary<long, Tensor>();
            Tensor RoPEPositions(int numHeads)
            {
                if (!posCache.TryGetValue(numHeads, out var t)) { t = BuildCanvasRoPEPositions(B, C, P, numHeads); posCache[numHeads] = t; }
                return t;
            }
            Tensor DecodeMask(int promptLen, bool local)
            {
                long key = ((long)promptLen << 1) | (local ? 1L : 0L);
                if (!maskCache.TryGetValue(key, out var t)) { t = BuildDecodeMaskTensor(promptLen, C, local); maskCache[key] = t; }
                return t;   // may be null (no masking needed)
            }

            try
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    string prefix = $"blk.{l}";
                    bool local = _isLocal[l];
                    int hd = _headDim[l];
                    int qHeads = Config.NumHeads;
                    int kvHeads = _kvHeads[l];

                    using var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");        // [N, D]
                    Tensor q = LinearForward(normed, $"{prefix}.attn_q.weight");               // [N, qHeads*hd]
                    Tensor k = LinearForward(normed, $"{prefix}.attn_k.weight");               // [N, kvHeads*hd]
                    Tensor v;
                    if (_hasVProj[l]) v = LinearForward(normed, $"{prefix}.attn_v.weight");
                    else { v = new Tensor(_allocator, DType.Float32, k.Sizes); Ops.Copy(v, k); }

                    using (var qr = q.View((long)N * qHeads, hd))
                        Ops.RMSNorm(qr, qr, _weights[$"{prefix}.attn_q_norm.weight"], null, eps);
                    using (var kr = k.View((long)N * kvHeads, hd))
                        Ops.RMSNorm(kr, kr, _weights[$"{prefix}.attn_k_norm.weight"], null, eps);
                    using (var vr = v.View((long)N * kvHeads, hd))
                        Ops.RMSNorm(vr, vr, GetOnes(hd), null, eps);

                    // NeoX RoPE on Q and K with per-sequence canvas positions (one dispatch each).
                    ApplyRoPEWithPos(q, qHeads, hd, N, RoPEPositions(qHeads), local);
                    ApplyRoPEWithPos(k, kvHeads, hd, N, RoPEPositions(kvHeads), local);

                    // per-sequence attention over (cached prompt K/V_b | fresh canvas K/V_b)
                    var attnOutFull = new Tensor(_allocator, DType.Float32, N, qHeads * hd);
                    for (int b = 0; b < B; b++)
                    {
                        using var qH = SeqSliceToHeads(q, b, C, qHeads, hd);
                        using var kCanvas = SeqSliceToHeads(k, b, C, kvHeads, hd);
                        using var vCanvas = SeqSliceToHeads(v, b, C, kvHeads, hd);
                        using var kFull = Ops.Concat(null, 1, seqs[b].PromptK[l], kCanvas);   // [kvHeads, P_b+C, hd]
                        using var vFull = Ops.Concat(null, 1, seqs[b].PromptV[l], vCanvas);
                        Tensor dmask = DecodeMask(P[b], local);
                        using var attnRes = AttnCoreHeadFirst(qH, kFull, vFull, dmask, C, P[b] + C, qHeads, kvHeads, hd);
                        using var dst = attnOutFull.Narrow(0, (long)b * C, C);
                        Ops.Copy(dst, attnRes);
                    }
                    q.Dispose(); k.Dispose(); v.Dispose();
                    InvalidateTensorDeviceCache(attnOutFull);

                    Tensor attnOut = LinearForward(attnOutFull, $"{prefix}.attn_output.weight");   // [N, D]
                    attnOutFull.Dispose();
                    Ops.RMSNorm(attnOut, attnOut, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
                    Ops.Add(attnOut, attnOut, hidden);
                    hidden.Dispose();

                    attnOut = FeedForward(attnOut, l, prefix, N);   // dense + MoE batched over all N rows
                    if (_decScale[l] != 1f) Ops.Mul(attnOut, attnOut, _decScale[l]);   // decoder scalar (all rows are canvas)
                    hidden = attnOut;
                }

                // output_norm batched over all rows. The lm_head (the big tied [D, vocab] matmul) reads its
                // weight once per dispatch, so batching all B*C rows into one dispatch reads that weight ONCE
                // for the whole batch (a real throughput win for this large-vocab model). The cost is a
                // [B*C, vocab] logits tensor; when that would exceed the memory cap we fall back to a
                // per-sequence lm_head (one weight read each, but only a [C, vocab] transient).
                Tensor normedOut = RMSNormOp(hidden, "output_norm.weight");   // [N, D]
                hidden.Dispose();

                var results = new float[B][];
                long logitsBytes = (long)N * vocab * sizeof(float);
                if (logitsBytes <= LmHeadBatchByteCap)
                {
                    Tensor logits = LinearForward(normedOut, "token_embd.weight");   // [N, vocab]
                    normedOut.Dispose();
                    if (_finalLogitSoftcap > 0f)
                    {
                        Ops.Mul(logits, logits, 1f / _finalLogitSoftcap);
                        Ops.Tanh(logits, logits);
                        Ops.Mul(logits, logits, _finalLogitSoftcap);
                    }
                    for (int b = 0; b < B; b++)
                    {
                        using var slice = logits.Narrow(0, (long)b * C, C);   // [C, vocab]
                        using var sliceC = Ops.NewContiguous(slice);
                        results[b] = ReadbackFresh(sliceC, C * vocab);
                    }
                    logits.Dispose();
                }
                else
                {
                    for (int b = 0; b < B; b++)
                    {
                        using var hSlice = normedOut.Narrow(0, (long)b * C, C);   // [C, D]
                        using var hSliceC = Ops.NewContiguous(hSlice);
                        Tensor logits = LinearForward(hSliceC, "token_embd.weight");   // [C, vocab]
                        if (_finalLogitSoftcap > 0f)
                        {
                            Ops.Mul(logits, logits, 1f / _finalLogitSoftcap);
                            Ops.Tanh(logits, logits);
                            Ops.Mul(logits, logits, _finalLogitSoftcap);
                        }
                        results[b] = ReadbackFresh(logits, C * vocab);
                        logits.Dispose();
                    }
                    normedOut.Dispose();
                }
                return results;
            }
            finally
            {
                foreach (var t in posCache.Values) t?.Dispose();
                foreach (var t in maskCache.Values) t?.Dispose();
            }
        }

        /// <summary>Narrow the b-th canvas (rows [b*C, b*C+C)) out of a batched [N, heads*hd] tensor and
        /// reshape to head-first [heads, C, hd].</summary>
        private Tensor SeqSliceToHeads(Tensor batched, int b, int C, int heads, int hd)
        {
            using var slice = batched.Narrow(0, (long)b * C, C);     // [C, heads*hd] (contiguous block)
            using var sliceC = Ops.NewContiguous(slice);
            return ReshapeToHeads(sliceC, heads, C, hd);
        }

        /// <summary>Build the per-sequence canvas RoPE positions tensor [N*numHeads]: canvas token i of
        /// sequence b is at absolute position P_b+i (so it rotates as the (P_b+i)-th position, matching the
        /// unbatched decode where the canvas starts at the prompt length).</summary>
        private Tensor BuildCanvasRoPEPositions(int B, int C, int[] P, int numHeads)
        {
            int N = B * C;
            int[] positions = new int[(long)N * numHeads];
            for (int b = 0; b < B; b++)
                for (int i = 0; i < C; i++)
                {
                    int pos = P[b] + i;
                    int baseOff = (b * C + i) * numHeads;
                    for (int h = 0; h < numHeads; h++) positions[baseOff + h] = pos;
                }
            return CreateIntTensor(positions, (long)N * numHeads);
        }

        /// <summary>In-place NeoX RoPE on a flat [N, numHeads*hd] tensor using an explicit positions tensor
        /// (GGML/device-glue backends). Local layers use a single base; global layers use the freq-factors
        /// (NTK) kernel.</summary>
        private void ApplyRoPEWithPos(Tensor data, int numHeads, int hd, int N, Tensor pos, bool local)
        {
            using var view = data.View(1, N, numHeads, hd);
            if (local)
            {
                Ops.RoPEEx(view, view, pos, hd, NeoXRopeMode, 0, _ropeLocalBase, 1f, 0f, 1f, 0f, 0f);
                return;
            }
            if (IsGgmlBackend && _ropeFreqsRawTensor != null)
            {
                GgmlBasicOps.RoPEExWithFreqFactors(view, view, pos, _ropeFreqsRawTensor,
                    hd, NeoXRopeMode, 0, _ropeGlobalBase, 1f, 0f, 1f, 0f, 0f);
                return;
            }
            // Non-GGML global RoPE in batched mode is not supported (the scheduler restricts batching to
            // GGML backends); fall back would require per-position cos/sin tables.
            throw new NotSupportedException("Batched diffusion decode requires a GGML backend for global RoPE.");
        }

        /// <summary>Build a fresh decode-phase additive mask [maskHeads, C, P+C] (no instance-field caching),
        /// or null when every key is attended (global layers, or local layers whose prompt fits the window).</summary>
        private Tensor BuildDecodeMaskTensor(int P, int C, bool local)
        {
            int klo = local ? Math.Max(0, P - _slidingWindow + 1) : 0;
            if (klo == 0) return null;
            int qHeads = Config.NumHeads;
            int maskHeads = _backend == BackendType.Cuda ? qHeads : 1;
            int kvLen = P + C;
            var data = new float[(long)maskHeads * C * kvLen];
            for (int qi = 0; qi < C; qi++)
            {
                long rowBase = (long)qi * kvLen;
                for (int kj = 0; kj < kvLen; kj++)
                {
                    float val = kj >= klo ? 0f : MaskNeg;
                    for (int h = 0; h < maskHeads; h++)
                        data[(long)h * C * kvLen + rowBase + kj] = val;
                }
            }
            return CreateFloatTensor(data, maskHeads, C, kvLen);
        }

        /// <summary>Read a logits tensor [.., total] into a freshly-allocated host array (batched decode needs
        /// one array per sequence, so it can't share the single reusable <c>_canvasLogits</c> buffer).</summary>
        private unsafe float[] ReadbackFresh(Tensor logits, int total)
        {
            if (IsGgmlBackend)
            {
                var arr = new float[total];
                float* lp = GetFloatPtr(logits);
                fixed (float* dst = arr)
                    Buffer.MemoryCopy(lp, dst, (long)total * 4, (long)total * 4);
                return arr;
            }
            return logits.GetElementsAsFloat(total);
        }

        private unsafe IntPtr WPtr(string name) => (IntPtr)GetFloatPtr(_weights[name]);

        /// <summary>Swap a streamed weight's pointer for its page-locked private copy when one exists
        /// (see <see cref="PrepareCudaWeightResidency"/>); identity for resident weights.</summary>
        private IntPtr PinnedOr(IntPtr weightPtr)
            => _pinnedStreamCopies.TryGetValue(weightPtr, out var pinned) ? pinned : weightPtr;

        /// <summary>Run a whole diffusion decode layer (attention over cached prompt K/V + canvas, dense MLP,
        /// 128-expert MoE) as a single fused GGML graph dispatch. Updates <paramref name="hidden"/> in place.
        /// Returns false (caller uses the per-op path) if any weight is missing or the backend rejects the
        /// kernel (e.g. flash-attn unsupported for this head_dim).</summary>
        private unsafe bool TryBuildLayerArgs(int layer, string prefix, int C, int P,
            bool local, int hd, int kvHeads, Tensor[] pk, Tensor[] pv, out DiffusionDecodeLayerArgs args)
        {
            args = default;
            try
            {
                if (pk == null || pk[layer] == null) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.attn_q.weight", out var qw)) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.attn_k.weight", out var kw)) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.attn_output.weight", out var ow)) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.ffn_gate.weight", out var gatew)) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.ffn_up.weight", out var upw)) return false;
                if (!_quantWeights.TryGetValue($"{prefix}.ffn_down.weight", out var downw)) return false;
                if (!_weights.TryGetValue($"{prefix}.ffn_gate_inp.weight", out var gateInpW)) return false;
                var gateUp = _stackedGateUp[layer];
                var down = _stackedDown[layer];
                if (gateUp == null || down == null) return false;

                bool hasV = _hasVProj[layer];
                IntPtr vwPtr = IntPtr.Zero; int vwType = 0; long vwNe0 = 0, vwNe1 = 0, vwBytes = 0;
                if (hasV)
                {
                    if (!_quantWeights.TryGetValue($"{prefix}.attn_v.weight", out var vw)) return false;
                    vwPtr = vw.CacheKey; vwType = vw.GgmlType; vwNe0 = vw.Ne0; vwNe1 = vw.Ne1; vwBytes = vw.RawBytes;
                }

                _weights.TryGetValue($"{prefix}.ffn_gate_inp.scale", out var gateInpScale);
                _weights.TryGetValue($"{prefix}.ffn_down_exps.scale", out var downExpsScale);

                args = new DiffusionDecodeLayerArgs
                {
                    Hidden = IntPtr.Zero,   // per-layer path sets this; ignored by model-wide
                    AttnNormW = WPtr($"{prefix}.attn_norm.weight"),
                    QW = PinnedOr(qw.CacheKey), KW = PinnedOr(kw.CacheKey), VW = PinnedOr(vwPtr),
                    QNormW = WPtr($"{prefix}.attn_q_norm.weight"),
                    KNormW = WPtr($"{prefix}.attn_k_norm.weight"),
                    OW = PinnedOr(ow.CacheKey),
                    PostAttnNormW = WPtr($"{prefix}.post_attention_norm.weight"),
                    PromptK = (IntPtr)GetFloatPtr(pk[layer]),
                    PromptV = (IntPtr)GetFloatPtr(pv[layer]),
                    FreqFactors = (!local && _ropeFreqsRawTensor != null) ? (IntPtr)GetFloatPtr(_ropeFreqsRawTensor) : IntPtr.Zero,
                    FfnNormW = WPtr($"{prefix}.ffn_norm.weight"),
                    GateW = PinnedOr(gatew.CacheKey), UpW = PinnedOr(upw.CacheKey), DownW = PinnedOr(downw.CacheKey),
                    PostFfwNorm1W = WPtr($"{prefix}.post_ffw_norm_1.weight"),
                    GateInpW = (IntPtr)GetFloatPtr(gateInpW),
                    GateInpScale = gateInpScale != null ? (IntPtr)GetFloatPtr(gateInpScale) : IntPtr.Zero,
                    PreFfwNorm2W = WPtr($"{prefix}.pre_ffw_norm_2.weight"),
                    GateUpExps = PinnedOr(gateUp.Data), DownExps = PinnedOr(down.Data),
                    DownExpsScale = downExpsScale != null ? (IntPtr)GetFloatPtr(downExpsScale) : IntPtr.Zero,
                    PostFfwNorm2W = WPtr($"{prefix}.post_ffw_norm_2.weight"),
                    PostFfwNormW = WPtr($"{prefix}.post_ffw_norm.weight"),

                    QNe0 = qw.Ne0, QNe1 = qw.Ne1, QBytes = qw.RawBytes,
                    KNe0 = kw.Ne0, KNe1 = kw.Ne1, KBytes = kw.RawBytes,
                    VNe0 = vwNe0, VNe1 = vwNe1, VBytes = vwBytes,
                    ONe0 = ow.Ne0, ONe1 = ow.Ne1, OBytes = ow.RawBytes,
                    GateNe0 = gatew.Ne0, GateNe1 = gatew.Ne1, GateBytes = gatew.RawBytes,
                    UpNe0 = upw.Ne0, UpNe1 = upw.Ne1, UpBytes = upw.RawBytes,
                    DownNe0 = downw.Ne0, DownNe1 = downw.Ne1, DownBytes = downw.RawBytes,
                    GueNe0 = gateUp.PerExpertNe0, GueNe1 = gateUp.PerExpertNe1, GueBytes = gateUp.TotalRawBytes,
                    DeNe0 = down.PerExpertNe0, DeNe1 = down.PerExpertNe1, DeBytes = down.TotalRawBytes,

                    StructBytes = Marshal.SizeOf<DiffusionDecodeLayerArgs>(),
                    HiddenSize = Config.HiddenSize,
                    CanvasLen = C, PromptLen = P,
                    NumHeads = Config.NumHeads, NumKvHeads = kvHeads, HeadDim = hd,
                    IsLocal = local ? 1 : 0, HasVProj = hasV ? 1 : 0,
                    SlidingWindow = _slidingWindow, RopeNDims = hd,
                    NumExperts = _numExperts, NumExpertsUsed = _numExpertsUsed,
                    FreqFactorsLen = (!local && _ropeFreqsRawTensor != null) ? (int)_ropeFreqsRawTensor.ElementCount() : 0,
                    QType = qw.GgmlType, KType = kw.GgmlType, VType = vwType, OType = ow.GgmlType,
                    GateType = gatew.GgmlType, UpType = upw.GgmlType, DownType = downw.GgmlType,
                    GueType = gateUp.GgmlType, DeType = down.GgmlType,

                    Eps = Config.Eps,
                    RopeBase = local ? _ropeLocalBase : _ropeGlobalBase,
                    InvSqrtHidden = 1f / MathF.Sqrt(Config.HiddenSize),
                    DecScale = _decScale[layer],
                };

                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>Run all transformer layers (attention over cached prompt K/V + canvas, dense MLP, MoE) as
        /// ONE fused GGML graph, keeping the canvas hidden on-device across all layers. Updates
        /// <paramref name="hidden"/> in place with the final layer output; C# then applies output_norm +
        /// lm_head. Returns false (caller falls back to the per-op layer loop) on any failure.</summary>
        /// <summary>Fused lm_head tail: output_norm + lm_head + final-logit softcap in one GGML graph
        /// (separate small graph, so no layer-graph interference). Returns canvas logits [C*vocab], or null
        /// (caller falls back to the per-op tail) on failure.</summary>
        private unsafe float[] TryFusedLmHead(Tensor hidden, int C)
        {
            try
            {
                if (!_quantWeights.TryGetValue("token_embd.weight", out var lmHead)) return null;
                if (!_weights.ContainsKey("output_norm.weight")) return null;
                int vocab = Config.VocabSize;
                long total = (long)C * vocab;
                // Pooled buffer: each step's logits are fully consumed (sampling + self-conditioning)
                // before the next step's lm_head overwrites it, so one reusable buffer replaces a
                // 268 MB LOH allocation per step. (Not cudaHostRegister'ed: ggml registers read-only,
                // which is for upload sources, and the device WRITES this buffer.)
                if (_fusedLogitsBuffer == null || _fusedLogitsBuffer.LongLength != total)
                    _fusedLogitsBuffer = GC.AllocateArray<float>(checked((int)total), pinned: true);
                float[] logits = _fusedLogitsBuffer;
                // The kernel outputs RAW logits (no softcap): the in-graph softcap on the 268 MB logits
                // tensor produces wrong results on Metal (both in-place and fresh-tensor variants), while
                // host softcap is correct. Apply the final-logit softcap on the host below.
                fixed (float* logitsPtr = logits)
                {
                    bool ok = GgmlBasicOps.TryDiffusionLmHead(
                        (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, C,
                        WPtr("output_norm.weight"),
                        PinnedOr(lmHead.CacheKey), lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                        (IntPtr)logitsPtr, vocab, Config.Eps, 0f);
                    if (!ok) return null;
                }
                if (_finalLogitSoftcap > 0f)
                {
                    float sc = _finalLogitSoftcap, inv = 1f / sc;
                    System.Threading.Tasks.Parallel.For(0, C, c =>
                    {
                        var row = logits.AsSpan(c * vocab, vocab);
                        TensorPrimitives.Multiply(row, inv, row);
                        TensorPrimitives.Tanh(row, row);
                        TensorPrimitives.Multiply(row, sc, row);
                    });
                }
                return logits;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private unsafe bool TryFusedModelLayers(Tensor hidden, int C, int P, Tensor[] pk, Tensor[] pv)
        {
            try
            {
                int L = Config.NumLayers;
                var layers = new DiffusionDecodeLayerArgs[L];
                for (int l = 0; l < L; l++)
                {
                    if (!TryBuildLayerArgs(l, $"blk.{l}", C, P, _isLocal[l], _headDim[l], _kvHeads[l], pk, pv, out layers[l]))
                        return false;
                }

                bool ok = GgmlBasicOps.TryDiffusionModelDecode(
                    layers, L,
                    (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, C, P,
                    IntPtr.Zero, IntPtr.Zero, 0, 0L, 0L, 0L,   // lm_head done per-op in C#; kernel outputs the final hidden
                    IntPtr.Zero, 0, 0f);
                if (!ok) return false;
                InvalidateTensorDeviceCache(hidden);   // hidden now holds the final layer output
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>Segmented fused decode: one fused graph dispatch PER LAYER instead of one
        /// whole-model graph. Numerically identical (the per-layer kernel computes the same layer
        /// math), but each layer's non-resident weights stream through one bounded, reused staging
        /// buffer (alloc_ctx_tensors_reuse) rather than the whole model's spill coexisting in VRAM —
        /// the mode <see cref="PrepareCudaWeightResidency"/> selects when the model is larger than
        /// VRAM. Costs a small per-layer hidden host round-trip ([H, C] ≈ a few MB).
        /// Returns false only on layer-0 rejection (clean fallback to the per-op path); a mid-model
        /// failure cannot fall back (hidden is partially transformed) and throws instead.</summary>
        private unsafe bool TryFusedModelLayersSegmented(Tensor hidden, int C, int P, Tensor[] pk, Tensor[] pv)
        {
            int L = Config.NumLayers;
            var args = new DiffusionDecodeLayerArgs[L];
            for (int l = 0; l < L; l++)
            {
                if (!TryBuildLayerArgs(l, $"blk.{l}", C, P, _isLocal[l], _headDim[l], _kvHeads[l], pk, pv, out args[l]))
                    return false;
            }
            IntPtr hiddenPtr = (IntPtr)GetFloatPtr(hidden);
            for (int l = 0; l < L; l++)
            {
                args[l].Hidden = hiddenPtr;
                if (!GgmlBasicOps.TryDiffusionDecodeLayer(in args[l]))
                {
                    if (l == 0) return false;   // hidden untouched -> caller can fall back cleanly
                    throw new InvalidOperationException(
                        $"Segmented diffusion decode failed at layer {l} after earlier layers ran; cannot fall back mid-model.");
                }
            }
            InvalidateTensorDeviceCache(hidden);
            return true;
        }

        private void AllocPromptStore()
        {
            if (_promptK == null) { _promptK = new Tensor[Config.NumLayers]; _promptV = new Tensor[Config.NumLayers]; }
            for (int l = 0; l < Config.NumLayers; l++)
            {
                // The fused/per-layer GGML decode binds the prompt K/V as *cacheable* device-local
                // copies keyed by their host pointer (try_get_cacheable_tensor_buffer, USAGE_COMPUTE in
                // ggml_ops_diffusion.cpp). Every block reallocates the prompt K/V at a new (growing)
                // size and host address, so the previous block's device buffers are orphaned in
                // g_host_buffer_cache: disposing the tensor only frees the host pool block, never the
                // cached MTLBuffer. With block-autoregressive generation (MaxBlocks = maxTokens/canvas)
                // this leaks dozens of K/V device buffers per turn and exhausts the Metal command-buffer
                // memory budget within a couple of turns (kIOGPUCommandBufferCallbackErrorOutOfMemory).
                // Invalidate the cache entry (frees the device buffer) *before* disposing the tensor,
                // while its host pointer is still valid as the cache key.
                ReleasePromptKvTensor(ref _promptK[l]);
                ReleasePromptKvTensor(ref _promptV[l]);
            }
        }

        /// <summary>Free a cached prompt K/V tensor: drop its GGML device-buffer-cache entry (so the
        /// device-local copy bound during decode is reclaimed, not orphaned) and then dispose the host
        /// storage. Order matters — the cache is keyed by the host pointer, which is only valid until
        /// dispose returns it to the pool / unmaps it.</summary>
        private void ReleasePromptKvTensor(ref Tensor t)
        {
            if (t == null) return;
            // Drain any in-flight GPU work that may still reference this tensor's device-local copy
            // before we free that buffer. invalidate_cached_buffer -> ggml_backend_buffer_free is an
            // *eager* free; under Metal async-compute a pending command buffer can still hold the
            // MTLBuffer (see the eager-free hazard documented in ggml_ops_core.cpp::sync_cached_buffer_to_host).
            // In the denoise loop the prior step's logits readback already drains the queue, so this is a
            // cheap no-op in practice (single atomic check), but it keeps the free correct if that readback
            // path ever changes. No-op on non-GGML backends.
            t.Storage.EnsureHostReadable();
            InvalidateTensorDeviceCache(t);
            t.Dispose();
            t = null;
        }

        /// <summary>Decode-phase additive mask [maskHeads, C, P+C], or null when every key is attended
        /// (global layers, or local layers whose prompt fits within the sliding window — the common case).</summary>
        private Tensor GetDecodeMask(int P, int C, bool local)
        {
            int klo = local ? Math.Max(0, P - _slidingWindow + 1) : 0;
            if (klo == 0) return null;   // all keys allowed -> no masking

            if (_decodeMaskP != P || _decodeMaskC != C)
            {
                _decodeMaskLocal?.Dispose(); _decodeMaskGlobal?.Dispose();
                _decodeMaskLocal = null; _decodeMaskGlobal = null;
                _decodeMaskP = P; _decodeMaskC = C;
            }
            if (local && _decodeMaskLocal != null) return _decodeMaskLocal;
            if (!local && _decodeMaskGlobal != null) return _decodeMaskGlobal;

            int qHeads = Config.NumHeads;
            int maskHeads = _backend == BackendType.Cuda ? qHeads : 1;
            int kvLen = P + C;
            var data = new float[(long)maskHeads * C * kvLen];
            for (int qi = 0; qi < C; qi++)
            {
                long rowBase = (long)qi * kvLen;
                for (int kj = 0; kj < kvLen; kj++)
                {
                    float val = kj >= klo ? 0f : MaskNeg;
                    for (int h = 0; h < maskHeads; h++)
                        data[(long)h * C * kvLen + rowBase + kj] = val;
                }
            }
            var mask = CreateFloatTensor(data, maskHeads, C, kvLen);
            if (local) _decodeMaskLocal = mask; else _decodeMaskGlobal = mask;
            return mask;
        }

        /// <summary>The region-aware DiffusionGemma mask always admits a contiguous key interval
        /// [klo, khi) per query. Mirrors the per-(q,k) rule from the llama.cpp reference:
        ///  - prompt query: causal over the prompt (SWA-clipped on local layers), never the canvas;
        ///  - canvas query (global): all positions; canvas query (local): all canvas + the last
        ///    (swa-1) prompt positions.</summary>
        private static void AllowedRange(int qi, bool qCanvas, int P, int N, bool local, int swa,
            out int klo, out int khi)
        {
            if (qCanvas)
            {
                klo = local ? Math.Max(0, P - swa + 1) : 0;
                khi = N;
            }
            else
            {
                klo = local ? Math.Max(0, qi - swa + 1) : 0;
                khi = qi + 1;
            }
        }

        // ===================================================================================
        //  MoE: 128-expert top-8 softmax routing + per-expert gated-GELU FFN, batched by expert.
        // ===================================================================================
        private unsafe Tensor MoEForward(Tensor attnOut, int layer, string prefix, int N)
        {
            int D = Config.HiddenSize;
            using var moeInput = RMSNormOp(attnOut, $"{prefix}.pre_ffw_norm_2.weight");  // [N, D]
            var output = new Tensor(_allocator, DType.Float32, N, D);

            // MLX FULLY-ON-DEVICE fused MoE (on-device routing + gather_qmm FFN, NO host read) — keeps the
            // whole decode forward device-resident so MLX evaluates all layers as one lazy graph (the MLX
            // port of GGML's fused single-graph decode, whose ~4x comes from killing the per-layer host
            // round-trip). Self-checked once vs the per-expert path; falls back permanently on divergence.
            if (_backend == BackendType.Mlx && _moeFusedDeviceEnabled && _moeFusedDeviceOk)
            {
                if (!_moeFusedDeviceChecked)
                {
                    _moeFusedDeviceChecked = true;
                    var (rwChk, seChk) = MoERoute(attnOut, prefix, N);
                    using var fdRef = new Tensor(_allocator, DType.Float32, N, D);
                    bool refOk = TryMoEMlx(moeInput, fdRef, seChk, rwChk, layer, prefix, N, D);
                    bool fOk = refOk && TryMoEFusedOnDeviceMlx(attnOut, moeInput, output, layer, N, D);
                    double cos = fOk ? CosineSimilarity(output, fdRef, (long)N * D) : 0;
                    _moeFusedDeviceOk = fOk && cos >= 0.999;
                    Console.WriteLine($"  [MLX fused on-device MoE] self-check cosine={cos:F6} -> {(_moeFusedDeviceOk ? "ENABLED" : "DISABLED (per-expert fallback)")}");
                    if (_moeFusedDeviceOk) return output;
                    Ops.Copy(output, fdRef);
                    return output;
                }
                if (TryMoEFusedOnDeviceMlx(attnOut, moeInput, output, layer, N, D))
                    return output;
                _moeFusedDeviceOk = false;
            }

            long tr = Stopwatch.GetTimestamp();
            (float[] routingWeights, int[] selectedExperts) = MoERoute(attnOut, prefix, N);
            _tMoeRoute += Stopwatch.GetTimestamp() - tr;

            // Fast path: single fused ggml_mul_mat_id dispatch over all experts (GGML backends).
            long tf = Stopwatch.GetTimestamp();
            if (_fusedMoeAvailable && TryFusedMoE(moeInput, output, selectedExperts, routingWeights, layer, N, D))
            {
                ProfSync(output);
                _tMoeFfn += Stopwatch.GetTimestamp() - tf;
                return output;
            }

            // MLX fast path: ONE fused gather_qmm over the layer's stacked experts (vs 128 per-expert
            // matmuls). Self-checked once against the per-expert path; falls back permanently if it
            // rejects or diverges.
            if (_backend == BackendType.Mlx && _moeGatherQmmEnabled && _moeGatherQmmOk)
            {
                if (!_moeGatherQmmChecked)
                {
                    _moeGatherQmmChecked = true;
                    using var reference = new Tensor(_allocator, DType.Float32, N, D);
                    bool refOk = TryMoEMlx(moeInput, reference, selectedExperts, routingWeights, layer, prefix, N, D);
                    bool fusedOk = refOk && TryMoEGatherQmmMlx(moeInput, output, selectedExperts, routingWeights, layer, N, D);
                    double cos = fusedOk ? CosineSimilarity(output, reference, (long)N * D) : 0;
                    _moeGatherQmmOk = fusedOk && cos >= 0.999;
                    Console.WriteLine($"  [MLX gather_qmm MoE] self-check cosine={cos:F6} -> {(_moeGatherQmmOk ? "ENABLED" : "DISABLED (per-expert fallback)")}");
                    if (_moeGatherQmmOk) return output;
                    Ops.Copy(output, reference);   // use the verified per-expert result this step
                    return output;
                }
                if (TryMoEGatherQmmMlx(moeInput, output, selectedExperts, routingWeights, layer, N, D))
                    return output;
                _moeGatherQmmOk = false;
            }

            // MLX device path: per-expert gather -> FFN -> weighted scatter-add, all on-device.
            if (_backend == BackendType.Mlx &&
                TryMoEMlx(moeInput, output, selectedExperts, routingWeights, layer, prefix, N, D))
                return output;

            // Host fallback (direct cuda + any backend the above paths reject): correct everywhere
            // (storage marks host writes dirty so the device re-reads), at the cost of per-layer syncs.
            Ops.Fill(output, 0f);
            float* inPtr = GetFloatPtr(moeInput);
            float* outPtr = GetFloatPtr(output);

            // per-expert post-down scale
            float[] perExpertScale = null;
            if (_weights.TryGetValue($"{prefix}.ffn_down_exps.scale", out var scaleT))
            {
                perExpertScale = new float[_numExperts];
                for (int e = 0; e < _numExperts; e++) perExpertScale[e] = scaleT.GetElementAsFloat(e);
            }

            // group tokens by expert
            var batches = new List<(int token, float weight)>[_numExperts];
            for (int e = 0; e < _numExperts; e++) batches[e] = new List<(int, float)>();
            for (int s = 0; s < N; s++)
                for (int u = 0; u < _numExpertsUsed; u++)
                {
                    int e = selectedExperts[s * _numExpertsUsed + u];
                    batches[e].Add((s, routingWeights[s * _numExpertsUsed + u]));
                }

            int rowBytes = D * sizeof(float);
            for (int e = 0; e < _numExperts; e++)
            {
                var batch = batches[e];
                if (batch.Count == 0) continue;
                int cnt = batch.Count;

                var batchInput = new Tensor(_allocator, DType.Float32, cnt, D);
                float* bp = GetFloatPtr(batchInput);
                for (int b = 0; b < cnt; b++)
                    Buffer.MemoryCopy(inPtr + (long)batch[b].token * D, bp + (long)b * D, rowBytes, rowBytes);

                Tensor expertOut = ExpertFFN(batchInput, prefix, e, cnt);
                batchInput.Dispose();

                if (perExpertScale != null && perExpertScale[e] != 1f)
                    Ops.Mul(expertOut, expertOut, perExpertScale[e]);

                float* ep = GetFloatPtr(expertOut);
                for (int b = 0; b < cnt; b++)
                {
                    int token = batch[b].token;
                    float w = batch[b].weight;
                    float* src = ep + (long)b * D;
                    float* dst = outPtr + (long)token * D;
                    for (int d = 0; d < D; d++) dst[d] += w * src[d];
                }
                expertOut.Dispose();
            }
            InvalidateTensorDeviceCache(output);
            return output;
        }

        /// <summary>Single-dispatch MoE FFN over all experts via the native ggml_mul_mat_id kernel.
        /// The per-expert post-down scale is folded into the routing weights. Returns false (and the
        /// caller falls back to the per-expert loop) if the kernel rejects the layout.</summary>
        private bool TryFusedMoE(Tensor moeInput, Tensor output, int[] selectedExperts,
            float[] routingWeights, int layer, int N, int D)
        {
            var gateUp = _stackedGateUp[layer];
            var down = _stackedDown[layer];
            int nFf = (int)(gateUp.PerExpertNe1 / 2);   // pre-fused gate_up -> ne1 = 2*expertFfn
            int nUsed = _numExpertsUsed;

            // fold per-expert scale into the routing weights (kernel applies one scalar per route)
            float[] kernelWeights = routingWeights;
            float[] scale = _perExpertScale[layer];
            if (scale != null)
            {
                kernelWeights = new float[N * nUsed];
                for (int s = 0; s < N; s++)
                    for (int k = 0; k < nUsed; k++)
                    {
                        int idx = s * nUsed + k;
                        kernelWeights[idx] = routingWeights[idx] * scale[selectedExperts[idx]];
                    }
            }

            try
            {
                GgmlBasicOps.MoEFFNPrefill(
                    moeInput, output,
                    N, D, nFf, _numExperts, nUsed,
                    selectedExperts, kernelWeights,
                    gateUp.Data, gateUp.GgmlType, gateUp.PerExpertNe0, gateUp.PerExpertNe1, gateUp.TotalRawBytes,
                    IntPtr.Zero, 0, 0L, 0L, 0L,             // up == null -> fused gate_up, kernel splits internally
                    down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, down.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.GEGLUSplit);
                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        /// <summary>On-device MLX MoE: group tokens by expert, then per expert gather the assigned rows,
        /// run the gated-GELU expert FFN, and weighted-scatter-add back — using MLX's device gather/scatter
        /// kernels so no rows ever round-trip through host memory. The per-expert post-down scale is folded
        /// into the scatter weights. Returns false (caller uses the host fallback) if a kernel rejects the
        /// layout.</summary>
        private bool TryMoEMlx(Tensor moeInput, Tensor output, int[] selectedExperts,
            float[] routingWeights, int layer, string prefix, int N, int D)
        {
            float[] perExpertScale = _perExpertScale[layer];
            var batchTokens = new List<int>[_numExperts];
            var batchWeights = new List<float>[_numExperts];
            for (int e = 0; e < _numExperts; e++) { batchTokens[e] = new List<int>(); batchWeights[e] = new List<float>(); }
            for (int s = 0; s < N; s++)
                for (int u = 0; u < _numExpertsUsed; u++)
                {
                    int e = selectedExperts[s * _numExpertsUsed + u];
                    float w = routingWeights[s * _numExpertsUsed + u];
                    if (perExpertScale != null) w *= perExpertScale[e];
                    batchTokens[e].Add(s);
                    batchWeights[e].Add(w);
                }

            Ops.Fill(output, 0f);
            for (int e = 0; e < _numExperts; e++)
            {
                int cnt = batchTokens[e].Count;
                if (cnt == 0) continue;
                using var rowIndices = CreateIntTensor(batchTokens[e].ToArray(), cnt);
                using var routeWeights = CreateFloatTensor(batchWeights[e].ToArray(), cnt);
                using var batchInput = new Tensor(_allocator, DType.Float32, cnt, D);
                if (!MlxFusedOps.TryGatherRows(batchInput, moeInput, rowIndices))
                    return false;
                Tensor expertOut = ExpertFFN(batchInput, prefix, e, cnt);   // [cnt, D]
                bool ok = MlxFusedOps.TryScatterAddWeightedRows(output, expertOut, rowIndices, routeWeights);
                expertOut.Dispose();
                if (!ok) return false;
            }
            return true;
        }

        /// <summary>Fused MLX MoE FFN via mlx_gather_qmm over the layer's stacked experts. Computes, for
        /// each of the N canvas tokens routed to its K experts: gate_up = gather_qmm(x, gateUpStack)[N*K,2ff],
        /// geglu = gelu(gate)*up [N*K,ff], down = gather_qmm(geglu, downStack)[N*K,D], then the routing-weighted
        /// sum over K (with the per-expert down scale folded in) via one batched matmul -> [N, D]. Numerically
        /// equivalent to the per-expert path (same affine expert weights, just batched). Returns false to fall
        /// back if any storage/type is unsupported or a kernel rejects the layout.</summary>
        private unsafe bool TryMoEGatherQmmMlx(Tensor moeInput, Tensor output,
            int[] selectedExperts, float[] routingWeights, int layer, int N, int D)
        {
            var gateUp = _stackedGateUp[layer];
            var down = _stackedDown[layer];
            if (gateUp == null || down == null) return false;
            int E = _numExperts;
            int K = _numExpertsUsed;
            int ff = _expertFfn;
            int NK = N * K;

            // Sort the (token, expert) pairs by expert so each expert's weight is loaded ONCE and serves its
            // consecutive rows as a GEMM (sorted_indices=true) — the key to beating the per-expert path,
            // which already groups ~N*K/E tokens per expert. Without sorting, gather_qmm reloads the expert
            // weight per row (GEMV-bound) and is slower. Sort is on the host (NK is tiny, ~2048).
            int[] order = new int[NK];
            for (int i = 0; i < NK; i++) order[i] = i;
            Array.Sort(order, (a, b) => selectedExperts[a].CompareTo(selectedExperts[b]));   // stable enough
            int[] expertsSorted = new int[NK];   // rhs (sorted, non-decreasing) for both projections
            int[] tokenSorted = new int[NK];     // lhs for gate_up: the token of each sorted pair
            int[] invOrder = new int[NK];        // unsort map: invOrder[originalPair] = sorted position
            int[] arangeNK = new int[NK];        // lhs for down: each sorted geglu row uses itself
            for (int i = 0; i < NK; i++)
            {
                int p = order[i];
                expertsSorted[i] = selectedExperts[p];
                tokenSorted[i] = p / K;
                invOrder[p] = i;
                arangeNK[i] = i;
            }

            // routing weight per ORIGINAL (n,k), with the per-expert post-down scale folded in (matches TryMoEMlx).
            float[] perExpertScale = _perExpertScale[layer];
            float[] w = routingWeights;
            if (perExpertScale != null)
            {
                w = new float[NK];
                for (int nk = 0; nk < NK; nk++) w[nk] = routingWeights[nk] * perExpertScale[selectedExperts[nk]];
            }

            using var tokenSortedT = CreateIntTensor(tokenSorted, NK);
            using var expertsSortedT = CreateIntTensor(expertsSorted, NK);
            using var arangeT = CreateIntTensor(arangeNK, NK);
            using var invOrderT = CreateIntTensor(invOrder, NK);
            using var input3 = moeInput.View(N, 1, D);                       // [N,1,in] (M=1)

            // gate_up: sorted rows -> grouped GEMM per expert. Output in SORTED order.
            using var gateUpSorted = new Tensor(_allocator, DType.Float32, NK, 2 * ff);
            if (!MlxQuantizedOps.TryGatherQmm(gateUpSorted, input3, tokenSortedT, expertsSortedT,
                    gateUp.Data, gateUp.Data, gateUp.GgmlType, gateUp.PerExpertNe0, gateUp.PerExpertNe1, E, gateUp.TotalRawBytes,
                    sortedIndices: true))
                return false;

            using var gegluSorted = new Tensor(_allocator, DType.Float32, NK, ff);
            if (!MlxFusedOps.TryGeluMulSplit(gegluSorted, gateUpSorted, ff)) return false;

            using var gegluSorted3 = gegluSorted.View(NK, 1, ff);
            using var downSorted = new Tensor(_allocator, DType.Float32, NK, D);
            if (!MlxQuantizedOps.TryGatherQmm(downSorted, gegluSorted3, arangeT, expertsSortedT,
                    down.Data, down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, E, down.TotalRawBytes,
                    sortedIndices: true))
                return false;

            // Unsort the down output back to original (n,k) order: downOrig[p] = downSorted[invOrder[p]].
            using var downOrig = new Tensor(_allocator, DType.Float32, NK, D);
            if (!MlxFusedOps.TryGatherRows(downOrig, downSorted, invOrderT)) return false;

            // routing-weighted sum over the K experts: out[N,1,D] = w[N,1,K] @ down[N,K,D].
            using var wT = CreateFloatTensor(w, N, 1, K);
            using var downB = downOrig.View(N, K, D);
            using var outB = output.View(N, 1, D);
            Ops.AddmmBatch(outB, 0f, outB, 1f, wT, downB);
            InvalidateTensorDeviceCache(output);
            return true;
        }

        /// <summary>FULLY-ON-DEVICE MLX MoE: routes (top-K) AND runs the expert FFN with NO host read, so
        /// the decode forward stays device-resident and MLX evaluates the whole layer stack as one lazy
        /// graph. Router: rms_norm_noscale(attnOut)·(1/sqrt(D))·gate_inp_scale → gate_inp matmul → on-device
        /// batched top-K (argpartition + take_along_axis + softmax) → device idx[N,K] + weights[N,K]. FFN:
        /// gather_qmm(gate_up) → GEGLU → gather_qmm(down, per-expert scale folded into the stacked affine)
        /// → routing-weighted sum. Equivalent to the per-expert path (self-checked). Returns false to fall
        /// back if any op rejects the layout.</summary>
        private unsafe bool TryMoEFusedOnDeviceMlx(Tensor attnOut, Tensor moeInput, Tensor output, int layer, int N, int D)
        {
            var gateUp = _stackedGateUp[layer];
            var down = _stackedDown[layer];
            if (gateUp == null || down == null) return false;
            int E = _numExperts, K = _numExpertsUsed, ff = _expertFfn, NK = N * K;
            float eps = Config.Eps;
            string prefix = $"blk.{layer}";

            // ---- on-device router (mirrors MoERoute's device pre-processing; top-K stays on device) ----
            using var routerNormed = Ops.NewContiguous(attnOut);
            Ops.RMSNorm(routerNormed, routerNormed, GetOnes(D), null, eps);   // unweighted rms_norm
            Ops.Mul(routerNormed, routerNormed, 1f / MathF.Sqrt(D));
            if (_weights.TryGetValue($"{prefix}.ffn_gate_inp.scale", out var gscale))
                Ops.Mul(routerNormed, routerNormed, gscale);
            using var scores = LinearForward(routerNormed, $"{prefix}.ffn_gate_inp.weight");   // [N, E]
            using var idx = new Tensor(_allocator, DType.Int32, N, K);
            using var weights = new Tensor(_allocator, DType.Float32, N, K);
            if (!MlxFusedOps.TryBatchedMoeRouterTopK(scores, idx, weights)) return false;

            // ---- on-device FFN via gather_qmm over the stacked experts ----
            var (lhsGate, lhsArange) = GetMoEConstIndices(N, K);
            using var idxFlat = idx.View(NK);                       // [N*K] int32 (rhs = chosen expert per pair)
            using var input3 = moeInput.View(N, 1, D);              // [N,1,in] (M=1)
            using var gateUpOut = new Tensor(_allocator, DType.Float32, NK, 2 * ff);
            if (!MlxQuantizedOps.TryGatherQmm(gateUpOut, input3, lhsGate, idxFlat,
                    gateUp.Data, gateUp.Data, gateUp.GgmlType, gateUp.PerExpertNe0, gateUp.PerExpertNe1, E, gateUp.TotalRawBytes))
                return false;
            using var geglu = new Tensor(_allocator, DType.Float32, NK, ff);
            if (!MlxFusedOps.TryGeluMulSplit(geglu, gateUpOut, ff)) return false;
            using var geglu3 = geglu.View(NK, 1, ff);
            using var downOut = new Tensor(_allocator, DType.Float32, NK, D);
            if (!MlxQuantizedOps.TryGatherQmm(downOut, geglu3, lhsArange, idxFlat,
                    down.Data, down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, E, down.TotalRawBytes,
                    perExpertScale: _perExpertScale[layer]))   // per-expert down scale folded into the stacked affine
                return false;

            // routing-weighted sum over the K experts: out[N,1,D] = w[N,1,K] @ down[N,K,D] (w device, no host read).
            using var wView = weights.View(N, 1, K);
            using var downB = downOut.View(N, K, D);
            using var outB = output.View(N, 1, D);
            Ops.AddmmBatch(outB, 0f, outB, 1f, wView, downB);
            InvalidateTensorDeviceCache(output);
            return true;
        }

        /// <summary>Cached host-built constant gather_qmm lhs index tensors [N*K]: lhsGate[nk]=nk/K (the
        /// token n that pair nk belongs to, for the shared gate_up input) and lhsArange[nk]=nk (each down
        /// row uses its own GEGLU output). Constant (routing-independent), so built once and reused.</summary>
        private (Tensor lhsGate, Tensor lhsArange) GetMoEConstIndices(int N, int K)
        {
            int NK = N * K;
            if (_moeConstNK != NK)
            {
                _moeLhsGateConst?.Dispose(); _moeLhsArangeConst?.Dispose();
                int[] lhsGate = new int[NK];
                int[] lhsArange = new int[NK];
                for (int nk = 0; nk < NK; nk++) { lhsGate[nk] = nk / K; lhsArange[nk] = nk; }
                _moeLhsGateConst = CreateIntTensor(lhsGate, NK);
                _moeLhsArangeConst = CreateIntTensor(lhsArange, NK);
                _moeConstNK = NK;
            }
            return (_moeLhsGateConst, _moeLhsArangeConst);
        }

        /// <summary>Cosine similarity between the first <paramref name="n"/> elements of two tensors (host
        /// read). Used by the one-time gather_qmm MoE self-check.</summary>
        private double CosineSimilarity(Tensor a, Tensor b, long n)
        {
            float[] av = a.GetElementsAsFloat((int)n);
            float[] bv = b.GetElementsAsFloat((int)n);
            double dot = 0, na = 0, nb = 0;
            for (long i = 0; i < n; i++) { double x = av[i], y = bv[i]; dot += x * y; na += x * x; nb += y * y; }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        }

        private Tensor ExpertFFN(Tensor input, string prefix, int expert, int cnt)
        {
            // gate_up are fused in this checkpoint: ffn_gate_up_exps.{e}.weight -> [cnt, 2*expertFfn]
            Tensor gateUp = LinearForward(input, $"{prefix}.ffn_gate_up_exps.{expert}.weight");
            int half = _expertFfn;
            Tensor gate, up;
            using (var gv = gateUp.Narrow(1, 0, half)) gate = Ops.NewContiguous(gv);
            using (var uv = gateUp.Narrow(1, half, half)) up = Ops.NewContiguous(uv);
            gateUp.Dispose();
            Ops.GELUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, $"{prefix}.ffn_down_exps.{expert}.weight");
            gate.Dispose();
            return down;
        }

        private unsafe (float[] routingWeights, int[] selectedExperts) MoERoute(Tensor input, string prefix, int N)
        {
            int D = Config.HiddenSize;
            float eps = Config.Eps;

            using var normed = Ops.NewContiguous(input);
            Ops.RMSNorm(normed, normed, GetOnes(D), null, eps);       // unweighted RMSNorm
            Ops.Mul(normed, normed, 1f / MathF.Sqrt(D));
            if (_weights.TryGetValue($"{prefix}.ffn_gate_inp.scale", out var scale))
                Ops.Mul(normed, normed, scale);                       // per-dim learned scale

            int E = _numExperts;
            int K = _numExpertsUsed;
            float[] scoresArr;
            using (var scores = LinearForward(normed, $"{prefix}.ffn_gate_inp.weight"))  // [N, numExperts]
                scoresArr = scores.GetElementsAsFloat(N * E);   // one read-only device->host copy

            var routingWeights = new float[N * K];
            var selectedExperts = new int[N * K];

            fixed (float* sp = scoresArr)
            for (int s = 0; s < N; s++)
            {
                float* row = sp + (long)s * E;
                // softmax over all experts
                float max = float.NegativeInfinity;
                for (int i = 0; i < E; i++) if (row[i] > max) max = row[i];
                float sum = 0f;
                for (int i = 0; i < E; i++) { float ex = MathF.Exp(row[i] - max); row[i] = ex; sum += ex; }
                float invSum = sum > 0f ? 1f / sum : 0f;
                for (int i = 0; i < E; i++) row[i] *= invSum;

                // top-K selection by probability
                int off = s * K;
                float selSum = 0f;
                for (int kk = 0; kk < K; kk++)
                {
                    int best = -1; float bestV = float.NegativeInfinity;
                    for (int i = 0; i < E; i++)
                    {
                        float val = row[i];
                        if (val > bestV) { bestV = val; best = i; }
                    }
                    selectedExperts[off + kk] = best;
                    routingWeights[off + kk] = bestV;
                    selSum += bestV;
                    row[best] = float.NegativeInfinity; // exclude from next pick
                }
                // renormalize over selected
                if (selSum > 0f)
                {
                    float inv = 1f / selSum;
                    for (int kk = 0; kk < K; kk++) routingWeights[off + kk] *= inv;
                }
            }
            return (routingWeights, selectedExperts);
        }

        // ===================================================================================
        //  Self-conditioning: soft_emb = (softmax(prev_logits*temp_inv) @ token_embd) * sqrt(D),
        //  then a gated-GELU MLP. Returns the additive signal [C, D].
        //
        //  The softmax is sharply concentrated (temp_inv in [1.25, 2.5]) so we use a top-K
        //  soft-embedding: only the K highest-probability tokens contribute meaningfully. This
        //  turns the otherwise dense [C, vocab] x [vocab, D] matmul (189 GFLOP/step + a 3 GB
        //  dense embedding table) into C*K embedding-row lookups + a weighted sum, a ~1000x
        //  reduction with negligible numerical impact (the tail mass is ~0 and is re-normalized
        //  away by the subsequent RMSNorm).
        //
        //  K = 32 (was 128): the dominant SC cost is the on-device gather of the C*K embedding
        //  rows (each row a Q4_K dequant), NOT the host top-K (~7 ms). Measured on Metal
        //  (diffusiongemma-26B-A4B, C=256, vocab=262144): K=128 -> ~1065 ms/decode-step, K=32 ->
        //  ~660 ms, K=16/8 -> ~634 ms (the SC-off floor is ~580 ms). A tempInv>=1.25 softmax puts
        //  >0.9999 of its mass in the top 32 tokens, so K=32 is numerically indistinguishable from
        //  K=128 (verified: Paris / FF7 coherence tests unchanged) while running ~1.6x faster.
        //  Tunable via DIFFUSION_SC_TOPK for experiments.
        // ===================================================================================
        private static readonly int ScTopK =
            int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_SC_TOPK"), out int _scK) && _scK > 0 ? _scK : 32;

        private unsafe Tensor ComputeSelfConditioning(float[] prevLogits, int C, float prevTempInv)
        {
            int D = Config.HiddenSize;
            int vocab = Config.VocabSize;
            int K = Math.Min(ScTopK, vocab);

            // per canvas position: top-K tokens and their softmax(logits*temp_inv) weights (over top-K)
            int[] topTokens = new int[C * K];
            float[] topProbs = new float[C * K];
            long _scT0 = Stopwatch.GetTimestamp();
            Parallel.For(0, C, c =>
            {
                long baseOff = (long)c * vocab;
                int outBase = c * K;
                TopK(prevLogits, baseOff, vocab, K, topTokens, topProbs, outBase);

                // softmax over the K selected logits (numerically stable), in temp_inv space
                float max = float.NegativeInfinity;
                for (int k = 0; k < K; k++)
                {
                    float z = topProbs[outBase + k] * prevTempInv;
                    topProbs[outBase + k] = z;
                    if (z > max) max = z;
                }
                float sum = 0f;
                for (int k = 0; k < K; k++)
                {
                    float e = MathF.Exp(topProbs[outBase + k] - max);
                    topProbs[outBase + k] = e;
                    sum += e;
                }
                float inv = sum > 0f ? 1f / sum : 0f;
                for (int k = 0; k < K; k++) topProbs[outBase + k] *= inv;
            });
            _tScTopK += Stopwatch.GetTimestamp() - _scT0;
            return ComputeSelfConditioningFromTopK(topTokens, topProbs, C, K);
        }

        /// <summary>Self-conditioning soft-embedding from PRECOMPUTED per-position top-K tokens
        /// (<paramref name="topTokens"/> [C*K]) and their softmax-over-top-K weights
        /// (<paramref name="topProbs"/> [C*K], already in the previous step's temp_inv space). This is the
        /// device-sampling path: the kernel returns the top-K, so the host top-K sweep over the full logits
        /// is skipped. Gathers the C*K embedding rows, weighted-sums them, and applies the gated-GELU MLP.</summary>
        private unsafe Tensor ComputeSelfConditioningFromTopK(int[] topTokens, float[] topProbs, int C, int K)
        {
            int D = Config.HiddenSize;
            long _scT1 = Stopwatch.GetTimestamp();

            // gather the C*K selected embedding rows, then per-position weighted sum as a batched matmul:
            // soft[c] = probs[c, 1, K] @ rows[c, K, D] -> [c, 1, D]. Fully on-device.
            using Tensor rows = Embedding(topTokens);          // [C*K, D]
            using var rows3 = rows.View(C, K, D);
            using var probsT = CreateFloatTensor(topProbs, C, 1, K);
            using var soft = new Tensor(_allocator, DType.Float32, C, 1, D);
            Ops.AddmmBatch(soft, 0f, soft, 1f, probsT, rows3);
            Ops.Mul(soft, soft, MathF.Sqrt(D));
            using var soft2 = soft.View(C, D);

            // gated-GELU MLP: down( gelu(gate(pre_norm(soft))) * up(pre_norm(soft)) )
            using var normed = RMSNormOp(soft2, "self_cond_pre_norm.weight");
            Tensor gate = LinearForward(normed, "self_cond_gate.weight");
            using (Tensor up = LinearForward(normed, "self_cond_up.weight"))
                Ops.GELUMul(gate, gate, up);
            Tensor sig = LinearForward(gate, "self_cond_down.weight");
            gate.Dispose();
            if (_profile) ProfSync(sig);
            _tScDevice += Stopwatch.GetTimestamp() - _scT1;
            return sig;
        }

        /// <summary>Select the top-K values of logits[baseOff .. baseOff+n) via a size-K min-heap.
        /// Writes the token indices to <paramref name="outTokens"/> and raw logits to
        /// <paramref name="outVals"/> starting at <paramref name="outBase"/> (unordered).</summary>
        private static void TopK(float[] logits, long baseOff, int n, int K,
            int[] outTokens, float[] outVals, int outBase)
        {
            // heap stored in outVals/outTokens[outBase .. outBase+K); heap[0] is the minimum.
            int filled = 0;
            for (int v = 0; v < n; v++)
            {
                float val = logits[baseOff + v];
                if (filled < K)
                {
                    int i = outBase + filled;
                    outVals[i] = val; outTokens[i] = v;
                    filled++;
                    if (filled == K) BuildMinHeap(outVals, outTokens, outBase, K);
                }
                else if (val > outVals[outBase])
                {
                    outVals[outBase] = val; outTokens[outBase] = v;
                    SiftDown(outVals, outTokens, outBase, K, 0);
                }
            }
        }

        private static void BuildMinHeap(float[] vals, int[] toks, int b, int K)
        {
            for (int i = K / 2 - 1; i >= 0; i--) SiftDown(vals, toks, b, K, i);
        }

        private static void SiftDown(float[] vals, int[] toks, int b, int K, int i)
        {
            while (true)
            {
                int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
                if (l < K && vals[b + l] < vals[b + smallest]) smallest = l;
                if (r < K && vals[b + r] < vals[b + smallest]) smallest = r;
                if (smallest == i) break;
                (vals[b + i], vals[b + smallest]) = (vals[b + smallest], vals[b + i]);
                (toks[b + i], toks[b + smallest]) = (toks[b + smallest], toks[b + i]);
                i = smallest;
            }
        }

        // ===================================================================================
        //  RoPE helpers (NeoX): pairs (x[j], x[j+halfDim]). cos/sin tables are [N, halfDim].
        // ===================================================================================
        private static void BuildCosSin(int N, float[] freqs, out float[] cos, out float[] sin)
        {
            int half = freqs.Length;
            cos = new float[N * half];
            sin = new float[N * half];
            for (int p = 0; p < N; p++)
            {
                int baseOff = p * half;
                for (int j = 0; j < half; j++)
                {
                    float angle = p * freqs[j];
                    cos[baseOff + j] = MathF.Cos(angle);
                    sin[baseOff + j] = MathF.Sin(angle);
                }
            }
        }

        /// <summary>Host cos/sin NeoX RoPE fallback (used only by non-GGML, non-MLX backends e.g. direct
        /// cuda). Modifies <paramref name="data"/> in place; the storage marks itself host-dirty so the
        /// next device op re-uploads.</summary>
        private unsafe void ApplyNeoXRoPERaw(Tensor data, int numHeads, int headDim, int N, float[] cos, float[] sin)
        {
            float* ptr = GetFloatPtr(data);
            int half = headDim / 2;
            Parallel.For(0, N, n =>
            {
                int cbase = n * half;
                float* basePtr = ptr + (long)n * numHeads * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = basePtr + (long)h * headDim;
                    for (int j = 0; j < half; j++)
                    {
                        float c = cos[cbase + j];
                        float s = sin[cbase + j];
                        float x0 = head[j];
                        float x1 = head[j + half];
                        head[j] = x0 * c - x1 * s;
                        head[j + half] = x0 * s + x1 * c;
                    }
                }
            });
            InvalidateTensorDeviceCache(data);
        }

        // ===================================================================================
        //  IModelArchitecture contract
        // ===================================================================================
        public override float[] Forward(int[] tokens)
        {
            throw new NotSupportedException(
                "DiffusionGemma is a diffusion model and does not support autoregressive Forward(). " +
                "Use DiffusionGemmaSampler.Generate() / the diffusion run mode.");
        }

        public override void ResetKVCache()
        {
            // no KV cache in the unified diffusion forward
        }

        public override void WarmUpKernels()
        {
            // Warm the native quantized matmul / norm / MoE kernels with a tiny [prompt|canvas]
            // forward (1 prompt token + 4 canvas positions). Autoregressive Forward() is unsupported.
            try
            {
                int bos = Tokenizer != null ? Tokenizer.BosTokenId : 0;
                if (bos < 0) bos = 0;
                int[] tokens = { bos, _maskTokenId, _maskTokenId, _maskTokenId, _maskTokenId };
                ForwardCanvas(tokens, promptLen: 1);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DiffusionGemma warmup skipped: {ex.Message}");
            }
            _swForward.Reset();
        }

        public void PrintForwardTiming()
        {
            double f = 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"  ForwardCanvas total: {_swForward.ElapsedMilliseconds} ms");
            Console.WriteLine($"    embed={_tEmbed * f:F0}ms  attn={_tAttn * f:F0}ms  denseMLP={_tDense * f:F0}ms  " +
                $"MoE={_tMoe * f:F0}ms (route={_tMoeRoute * f:F0}ms ffn={_tMoeFfn * f:F0}ms)  " +
                $"lm_head={_tLmHead * f:F0}ms  selfCond={_tSc * f:F0}ms (topK_host={_tScTopK * f:F0}ms device={_tScDevice * f:F0}ms)");
        }

        public override void Dispose()
        {
            foreach (var t in _onesByDim.Values) t?.Dispose();
            _onesByDim.Clear();
            foreach (var t in _ropePosCache.Values) t?.Dispose();
            _ropePosCache.Clear();
            _cosGlobalTensor?.Dispose();
            _sinGlobalTensor?.Dispose();
            _maskLocal?.Dispose();
            _maskGlobal?.Dispose();
            _decodeMaskLocal?.Dispose();
            _decodeMaskGlobal?.Dispose();
            _moeLhsGateConst?.Dispose();
            _moeLhsArangeConst?.Dispose();
            // Reclaim the GGML device-buffer-cache entries for the prompt K/V (see AllocPromptStore /
            // ReleasePromptKvTensor) before disposing their host storage, so the cached device-local
            // copies are freed rather than orphaned.
            if (_promptK != null) for (int l = 0; l < _promptK.Length; l++) ReleasePromptKvTensor(ref _promptK[l]);
            if (_promptV != null) for (int l = 0; l < _promptV.Length; l++) ReleasePromptKvTensor(ref _promptV[l]);
            // Unregister page-locked regions, then free the private streamed-weight copies (the
            // logits buffer in _pinnedHostRegions is GC-owned and must NOT be freed here).
            foreach (var region in _pinnedHostRegions) GgmlBasicOps.UnregisterPinnedHostBuffer(region);
            _pinnedHostRegions.Clear();
            foreach (var copy in _pinnedStreamCopies.Values) GgmlBasicOps.AlignedFree(copy);
            _pinnedStreamCopies.Clear();
            base.Dispose();
        }
    }

    /// <summary>Per-sequence decode state for batched (multi-request) diffusion generation: each in-flight
    /// request owns its own prompt K/V store (head-first [kvHeads, P, hd] per layer) so several canvases can
    /// be denoised together in one batched forward without sharing the model's instance-field K/V cache.
    /// Created via <see cref="DiffusionGemmaModel.CreateSeqState"/>; freed via
    /// <see cref="DiffusionGemmaModel.DisposeSeqState"/>.</summary>
    public sealed class DiffusionSeqState
    {
        internal readonly Tensor[] PromptK;
        internal readonly Tensor[] PromptV;
        internal int PromptLen = -1;

        internal DiffusionSeqState(int numLayers)
        {
            PromptK = new Tensor[numLayers];
            PromptV = new Tensor[numLayers];
        }
    }
}
