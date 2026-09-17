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
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Muse-Glimmer (arch string "muse-glimmer") - a dense decoder with:
    /// - Interleaved sliding-window attention. The pattern period comes from
    ///   "muse-glimmer.attention.sliding_window_pattern": with period P layer l is a
    ///   sliding-window layer when (l % P) &lt; (P - 1), so every P-th layer is a full
    ///   causal layer. For the 30B (52 layers, P = 4) that is 39 SWA + 13 full.
    /// - RoPE on the sliding-window layers only; the full layers are NoPE.
    ///   RoPE is ggml's NORM (interleaved-pair) flavour, not NeoX.
    /// - Per-head RMSNorm on Q and K. The Q norm weight carries the model's
    ///   qk_scale_factor (folded in at conversion time), the K norm weight is ones.
    /// - An attention output gate: sigmoid(W_gate @ attn_norm(x)) multiplies the
    ///   attention output before o_proj (same shape as afmoe / Qwen3.5).
    /// - Four RMSNorms per layer. The pre-attention and pre-FFN norms use the
    ///   model's f_norm_rms_eps; the post-attention and post-FFN norms use a
    ///   hardcoded 1e-8 (see llama.cpp src/models/muse-glimmer.cpp).
    /// - An unweighted RMSNorm on the input embeddings (no learned scale).
    /// - Dense SwiGLU FFN (no MoE).
    /// - Output multiplier ("muse-glimmer.logit_scale") applied to the logits,
    ///   followed by tanh softcapping at "muse-glimmer.final_logit_softcapping".
    /// - Optional vision tower (mmproj, projector type "muse-glimmer").
    /// </summary>
    public partial class MuseGlimmerModel : ModelBase
    {
        /// <summary>Hardcoded epsilon of the post-attention / post-FFN norms.</summary>
        private const float PostNormEps = 1e-8f;

        private const int DefaultSwaPatternPeriod = 4;
        private const int DefaultSlidingWindow = 2048;
        private const float DefaultFinalLogitSoftcap = 20.0f;
        private const float DefaultLogitScale = 1.0f;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _kvCacheCapacity;

        private float[] _ropeFreqs;
        private int _slidingWindow;
        private int _headDim;
        private float _finalLogitScale;
        private float _finalLogitSoftcap;
        private bool _hasTiedOutput;

        /// <summary>is_swa[l]; the complement is a full causal (NoPE) layer.</summary>
        private bool[] _isSwaLayer;

        /// <summary>Cached ones vector for the weightless input-embedding RMSNorm.</summary>
        private Tensor _onesForEmbNorm;

        private string[][] _layerWeightNames;

        private int[] _cachedSWAMaskWidths;
        private int _cachedSWAMaskQueryLen;
        private int _cachedSWAMaskStartPos = -1;

        private MuseGlimmerVisionEncoder _visionEncoder;
        private readonly List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        /// <summary>
        /// On MLX, kick the accumulated lazy graph every N layers so the GPU
        /// starts the front of the stack while the host queues the back.
        /// </summary>
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();

        /// <summary>
        /// On MLX, force the K/V cache into a materialised array every N cache writes.
        /// Each decode step appends a lazy slice_update node, so without this the chain
        /// every later attention read has to walk grows linearly in decode-step count.
        /// </summary>
        private static readonly int MlxKvMaterializeInterval = ResolveMlxKvMaterializeInterval();

        private static int ResolveMlxKvMaterializeInterval()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_MUSE_GLIMMER_KV_MATERIALIZE")
                         ?? Environment.GetEnvironmentVariable("TS_MLX_KV_MATERIALIZE_INTERVAL");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v;
            return 16;
        }

        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS")
                         ?? Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v > 0 ? v : int.MaxValue;   // 0 disables the periodic kick
            return 4;
        }

        /// <param name="draftModelPath">Optional DFlash drafter GGUF (general.architecture
        /// "dflash"). When supplied, speculative decoding is armed after the target
        /// weights are in place.</param>
        public MuseGlimmerModel(string ggufPath, BackendType backend, int tpDegree = 1,
            ITensorParallelGroup tpGroup = null, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "muse-glimmer";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _headDim = Config.HeadDim;
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", DefaultSlidingWindow);
            Config.SlidingWindow = _slidingWindow;
            _finalLogitSoftcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", DefaultFinalLogitSoftcap);
            _finalLogitScale = _gguf.GetFloat32($"{arch}.logit_scale", DefaultLogitScale);

            ResolveSwaPattern(arch);

            int swaCount = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_isSwaLayer[l]) swaCount++;

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"HeadDim={_headDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase} scale={Config.RopeScale} (sliding-window layers only; full layers are NoPE)");
            Console.WriteLine($"Sliding window={_slidingWindow}, LogitScale={_finalLogitScale}, Softcap={_finalLogitSoftcap}");
            Console.WriteLine($"Layer types: {swaCount} sliding-window, {Config.NumLayers - swaCount} full causal");

            ParseTokenizer();
            LoadWeights();

            _hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight");
            if (_hasTiedOutput)
                Console.WriteLine("  Output tied to token_embd.weight");

            FuseGateUpWeights();

            if (IsTensorParallel)
            {
                ValidateMuseGlimmerTpConstraints();
                ShardMuseGlimmerWeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            PrecomputeConstants();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            if (IsTensorParallel)
            {
                InitTpKVCache(initialCacheLength, maxContextLength);
                // Needs the per-rank KV caches and the per-rank device preload,
                // so it goes last. No-op outside tensor parallelism.
                BuildMuseGlimmerTpDecodeArrays();
            }
            else
            {
                InitKVCache(initialCacheLength, maxContextLength);
            }

            // --draft-model, else TS_MUSE_GLIMMER_DFLASH. The drafter's speculative
            // path runs the SINGLE-GPU fused kernel (for its residual capture) and
            // grows the single-GPU KV cache through EnsureCacheCapacity; neither
            // exists under tensor parallelism, so it is not offered there.
            string dflashPath = ResolveDFlashPath(draftModelPath);
            if (!string.IsNullOrWhiteSpace(dflashPath) && IsTensorParallel)
                Console.WriteLine("  DFlash speculative decoding is not supported under tensor parallelism; ignoring the drafter.");
            else if (!string.IsNullOrWhiteSpace(dflashPath))
                LoadDFlashDraftWeights(dflashPath);
        }

        /// <summary>
        /// "{arch}.attention.sliding_window_pattern" is written either as a scalar
        /// period (llama.cpp's set_swa_pattern) or as a per-layer bool array. Both
        /// spellings are accepted; a missing key falls back to period 4.
        /// </summary>
        private void ResolveSwaPattern(string arch)
        {
            int numLayers = Config.NumLayers;
            _isSwaLayer = new bool[numLayers];

            string key = $"{arch}.attention.sliding_window_pattern";
            bool[] perLayer = _gguf.GetBoolArray(key);
            if (perLayer != null && perLayer.Length >= numLayers)
            {
                Array.Copy(perLayer, _isSwaLayer, numLayers);
                return;
            }

            uint period = _gguf.GetUint32(key, DefaultSwaPatternPeriod);
            if (period == 0)
                period = DefaultSwaPatternPeriod;

            // llama_hparams::set_swa_pattern(period, dense_first = false):
            //   is_swa[l] = (l % period) < (period - 1)
            for (int l = 0; l < numLayers; l++)
                _isSwaLayer[l] = (l % (int)period) < (int)(period - 1);
        }

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                _layerWeightNames[l] = new[]
                {
                    p + "attn_norm.weight",             // 0
                    p + "attn_q.weight",                // 1
                    p + "attn_k.weight",                // 2
                    p + "attn_v.weight",                // 3
                    p + "attn_gate.weight",             // 4
                    p + "attn_q_norm.weight",           // 5
                    p + "attn_k_norm.weight",           // 6
                    p + "attn_output.weight",           // 7
                    p + "post_attention_norm.weight",   // 8
                    p + "ffn_norm.weight",              // 9
                    p + "ffn_gate_up.weight",           // 10
                    p + "ffn_down.weight",              // 11
                    p + "post_ffw_norm.weight",         // 12
                };
            }

            int halfDim = _headDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / _headDim);
        }

        /// <summary>
        /// Zero a freshly allocated KV tensor UNCONDITIONALLY.
        ///
        /// <see cref="ModelBase.InitializeCacheTensor"/> deliberately skips the fill on
        /// GgmlCuda/MLX because the per-op attention only ever reads rows it wrote.
        /// The fused kernel does not have that property: it reads a window padded up
        /// to a 256-row stride so the decode graph's shape (and therefore its CUDA
        /// capture) stays stable, and the padding rows are masked with -inf. Adding
        /// -inf to an uninitialized NaN still yields NaN, so those rows must be
        /// finite. One fill at allocation is the cheap way to guarantee it.
        /// </summary>
        private void ZeroCacheTensor(Tensor tensor)
        {
            if (tensor != null)
                Ops.Fill(tensor, 0f);
        }

        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            // Size the ring from the MAX context, never the initial allocation. The ring
            // size is published through MaxReusablePrefixTokens and KVStateFingerprint,
            // both of which the scheduler samples ONCE at engine construction. Deriving it
            // from the initial allocation made it change the first time the cache grew
            // (2048 tokens initially, so ResolveSwaRows returned 0 and the model advertised
            // unbounded prefix reuse; after the first grow past 4352 the layers became
            // rings but the scheduler still believed reuse was unbounded, and would inject
            // a pooled snapshot the wrapped rows can no longer represent). Fixing it at
            // construction keeps the advertised cap truthful for the model's whole life,
            // and is also what arms live-cache continuation for long conversations.
            _kvSwaRows = ResolveSwaRows(maxSeqLen);
            int swaLayers = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool ring = _kvSwaRows > 0 && _isSwaLayer[l];
                // A ring layer is allocated at its final row count immediately: the ring
                // never grows, so this is the only allocation it will ever need.
                int rows = ring ? _kvSwaRows : initialSeqLen;
                if (ring) swaLayers++;
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                ZeroCacheTensor(_kvCacheK[l]);
                ZeroCacheTensor(_kvCacheV[l]);
            }
            if (_kvSwaRows > 0)
            {
                long sized = (long)(Config.NumLayers - swaLayers) * initialSeqLen + (long)swaLayers * _kvSwaRows;
                long uniform = (long)Config.NumLayers * initialSeqLen;
                long bytes = sized * Config.NumKVHeads * _headDim * 2 * kvDtype.Size();
                Console.WriteLine(
                    $"  Muse-Glimmer KV cache: {swaLayers} sliding-window layers ring at {_kvSwaRows} rows, " +
                    $"{Config.NumLayers - swaLayers} full layers at {initialSeqLen} " +
                    $"({bytes / (1024 * 1024)} MB, {sized * 100 / uniform}% of a uniform cache).");
            }
            _cacheSeqLen = 0;
        }

        /// <summary>
        /// Rows allocated to each sliding-window layer, or 0 when every layer is
        /// sized for the full context.
        ///
        /// 39 of the 52 layers never look back further than n_swa (2048), so sizing
        /// them for a 64K context wastes most of the cache: uniform F16 KV at 64K is
        /// 3.5 GB, of which llama.cpp allocates 954 MB by giving its SWA layers their
        /// own small cache (llama-kv-cache-iswa.cpp sizes them
        /// min(n_ctx, n_swa + n_ubatch)). On a 16 GB card that difference is the whole
        /// headroom: a 64K text-only run peaked at 16059 MiB of 16384 and lost 22% of
        /// its decode rate to the resulting memory pressure.
        ///
        /// The rows are a RING indexed by (position % rows). Only the fused kernel
        /// reads that layout - see RequireUniformCache.
        /// </summary>
        private int _kvSwaRows;

        private static readonly bool SwaRingEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_RING"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Ring size for the SWA layers at a given capacity, or 0 to keep them
        /// full-sized.
        ///
        /// It has to hold the window plus a whole prefill chunk, otherwise a chunk
        /// would overwrite rows its own earlier rows still need. The trailing +1 is
        /// the same margin <see cref="DFlashConfig.RingRows"/> uses (n_swa +
        /// block_size + 1): sized without it, a 4651-token prompt at chunk 2048
        /// diverged from llama.cpp on the first decode step, and 4352 rows
        /// (pad(2048 + 2048 + 1, 256)) is the smallest size that reproduces
        /// llama.cpp exactly.
        /// </summary>
        private int ResolveSwaRows(int capacity)
        {
            // Either fused kernel can read the ring; only the per-op path cannot
            // (see RequireUniformCache). CanUseTpFusedForward is the TP twin of
            // CanUseFusedForward, which stays false under --tp on purpose.
            //
            // GgmlCpu keeps uniform caches: a ring is read WHOLE (its slots are
            // not in position order), and ggml-cpu's flash-attention evaluates
            // every KV column, masked or not - so 39 of 52 layers would pay the
            // full 4352 ring rows of attention at ANY depth, where the uniform
            // moving span costs pad256(window + chunk) only once the context is
            // actually that long. The GPU backends keep the ring: their fixed
            // graph shape is what preserves the persistent graph (and the CUDA
            // capture), and their flash kernels skip fully-masked blocks.
            if (!SwaRingEnabled || !(CanUseFusedForward || CanUseTpFusedForward) || _slidingWindow <= 0)
                return 0;
            // The ring is correct on one GPU and WRONG under tensor parallelism,
            // and the difference is the KV head count. A ring layer is read
            // whole (window == rows), so kv_window_needs_cuda_flash_attn_copy
            // takes its "the window IS the cache" early return and never
            // materialises it - which is right when a rank owns every KV head.
            // Split two ways this model holds ONE (32 Q heads over 2 KV heads),
            // and ggml-cuda's flash-attention picks a different kernel for a
            // ne[2] == 1 K/V; the same single-KV-head hazard that
            // ggml_ops_transformer_common.h already documents for truncated
            // windows. Measured: --tp 2 answers "Compute 17 plus 25" with
            // fluent nonsense on the ring and correctly without it, byte-for-byte
            // deterministic either way, on every AllReduce transport and with
            // CUDA graph capture disabled.
            //
            // Uniform caches cost KV memory the ring was introduced to save, so
            // this is a correctness-first fallback, not the end state: the ring
            // should come back for TP once the ne[2] == 1 flash-attention path is
            // pinned down. One GPU is untouched and keeps the ring.
            if (IsTensorParallel)
                return 0;
            if (_backend == BackendType.GgmlCpu)
                return 0;
            int need = _slidingWindow + Math.Max(PrefillChunkTokens, 1) + 1;
            string raw = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_ROWS");
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int forced) && forced > 0)
                need = forced;
            int rows = ((need + 255) / 256) * 256;
            return rows < capacity ? rows : 0;
        }

        /// <summary>Rows the given layer's cache actually has.</summary>
        private int KvRows(int layer)
            => _kvSwaRows > 0 && _isSwaLayer[layer] ? _kvSwaRows : _kvCacheCapacity;

        /// <summary>Cache row holding absolute position <paramref name="pos"/>.</summary>
        private int KvRow(int layer, int pos)
            => _kvSwaRows > 0 && _isSwaLayer[layer] ? pos % _kvSwaRows : pos;

        /// <summary>
        /// The per-op attention path treats a cache row as an absolute position, which
        /// a ring row is not. The ring is only ever armed when the fused kernel is
        /// available, so reaching here means the fused forward declined at run time;
        /// failing loudly beats returning quietly wrong logits.
        /// </summary>
        private void RequireUniformCache(int layer)
        {
            if (_kvSwaRows > 0 && _isSwaLayer[layer])
            {
                throw new InvalidOperationException(
                    "Muse-Glimmer sliding-window layers are using a ring KV cache, which only the fused " +
                    "kernel can read, but the fused forward declined and the per-op path was reached. " +
                    "Set TS_MUSE_GLIMMER_SWA_RING=0 to allocate every layer for the full context instead " +
                    "(costs ~2.5 GB more KV at 64K).");
            }
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            DType kvDtype = _kvCacheDtype.ToDType();
            // The ring is sized from the max context at construction and never changes,
            // so a grow only ever re-allocates the FULL-attention layers, and every row
            // index it holds keeps its meaning (a full layer addresses linearly). Keeping
            // the ring size fixed is what makes MaxReusablePrefixTokens and
            // KVStateFingerprint - both sampled ONCE by the scheduler - stay truthful.
            int newSwaRows = _kvSwaRows;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (newSwaRows > 0 && _isSwaLayer[l])
                    continue;   // ring layers are already at their final size

                var newK = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, _headDim);
                var newV = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, _headDim);
                ZeroCacheTensor(newK);
                ZeroCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    int keep = Math.Min(_cacheSeqLen, _kvCacheCapacity);
                    using (var srcK = _kvCacheK[l].Narrow(1, 0, keep))
                    using (var dstK = newK.Narrow(1, 0, keep))
                        Ops.Copy(dstK, srcK);
                    using (var srcV = _kvCacheV[l].Narrow(1, 0, keep))
                    using (var dstV = newV.Narrow(1, 0, keep))
                        Ops.Copy(dstV, srcV);
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            // The captured decode graph baked the old KV device addresses and cache
            // size; replaying it against the reallocated buffers would hang.
            ResetFusedDecodeCache();
            _fusedProbed = false;
            if (newSwaRows > 0)
            {
                int swaLayers = 0;
                for (int l = 0; l < Config.NumLayers; l++) if (_isSwaLayer[l]) swaLayers++;
                long sized = (long)(Config.NumLayers - swaLayers) * newCapacity + (long)swaLayers * newSwaRows;
                long uniform = (long)Config.NumLayers * newCapacity;
                Console.WriteLine(
                    $"Expanded Muse-Glimmer attention cache to {newCapacity} tokens " +
                    $"({swaLayers} sliding-window layers ring at {newSwaRows} rows, " +
                    $"{sized * 100 / uniform}% of a uniform cache).");
            }
            else
            {
                Console.WriteLine($"Expanded Muse-Glimmer attention cache to {newCapacity} tokens.");
            }
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            _ringWrittenTokens = 0;
            _cachedSWAMaskStartPos = -1;
            ResetFusedDecodeCache();
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            ResetTpKVCache();
            if (_kvCacheK == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
        }

        /// <summary>
        /// Whether a rewind from <paramref name="cachedTokenCount"/> to
        /// <paramref name="targetTokenCount"/> leaves every sliding-window row the next
        /// query needs intact in a ring of <paramref name="ringRows"/> rows.
        ///
        /// <para>A ring layer stores position <c>p</c> at row <c>p % rows</c>, so once the
        /// sequence passes <c>rows</c> tokens each new position overwrites the oldest one.
        /// Rewinding moves only the head: the rows positions below the target lost to the
        /// wrap do not come back, and a query at the target still attends a whole window
        /// behind it. The window behind the target survives while
        /// <c>written - target &lt;= rows - window - 1</c> (the ring is window + chunk + 1
        /// rows, so the 16-token live-cache rewind always fits whenever the prefill chunk
        /// is at least 16). A ring that never wrapped (<c>written &lt;= rows</c>) still
        /// holds every row, and dropping to zero or to the head reads nothing old.</para>
        ///
        /// <para><c>written</c> is the furthest head this cache reached since it was last
        /// emptied, not the current head: an earlier accepted rewind can bring the head
        /// back below <c>rows</c> while the rows its wrap overwrote stay overwritten, so
        /// a second rewind judged by the current head alone would pass as "never
        /// wrapped" and read them.</para>
        ///
        /// <para>Muse-Glimmer had no such guard, so a deeper rewind on a wrapped ring
        /// continued from rows another position had overwritten (radix design P23, M0c).</para>
        /// </summary>
        internal static bool RingRewindIsExact(int cachedTokenCount, int targetTokenCount, int ringRows, int slidingWindow,
            int writtenTokenCount = 0)
        {
            if (targetTokenCount < 0 || targetTokenCount > cachedTokenCount) return false;
            if (targetTokenCount == cachedTokenCount || targetTokenCount == 0) return true;
            int written = Math.Max(cachedTokenCount, writtenTokenCount);
            if (ringRows <= 0 || written <= ringRows) return true;
            return written - targetTokenCount <= ringRows - slidingWindow - 1;
        }

        /// <summary>The furthest head this cache has reached since it was last emptied
        /// (reset, or truncated to zero). Only a rewind lowers the head, so it is folded
        /// in there; see <see cref="RingRewindIsExact"/>.</summary>
        private int _ringWrittenTokens;

        private void NoteHeadBeforeRewind(int tokenCount)
            => _ringWrittenTokens = tokenCount == 0 ? 0 : Math.Max(_ringWrittenTokens, _cacheSeqLen);

        public override bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
            => base.CanTruncateKVCache(cachedTokenCount, targetTokenCount)
                && RingRewindIsExact(cachedTokenCount, targetTokenCount, _kvSwaRows, _slidingWindow, _ringWrittenTokens);

        /// <summary>
        /// Refuses (changing nothing) a rewind the ring cannot honour; see
        /// <see cref="RingRewindIsExact"/>. The caller then resets and re-prefills.
        /// </summary>
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            if (!CanTruncateKVCache(_cacheSeqLen, tokenCount)) return false;
            NoteHeadBeforeRewind(tokenCount);
            ApplyKVCacheTruncation(tokenCount);
            return true;
        }

        /// <summary>The non-refusable form throws on a rewind the ring cannot honour,
        /// rather than continuing from overwritten rows (Gemma 4 keeps the same contract).</summary>
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!TryTruncateKVCacheCore(tokenCount))
                throw new InvalidOperationException(
                    $"Muse-Glimmer cannot truncate its KV cache from {_cacheSeqLen} to {tokenCount}: its " +
                    $"{_kvSwaRows}-row sliding-window ring no longer holds the window behind that position. " +
                    "Use TryTruncateKVCache and re-prefill when it declines.");
        }

        private void ApplyKVCacheTruncation(int tokenCount)
        {
            // The fused kernel writes K/V on-device and leaves the host mirror stale
            // (_fusedKvDirty). The invalidation below drops the device copies so the
            // next forward re-uploads from host - so the device rows MUST come back
            // to the host first, or the re-upload restores the stale mirror and every
            // position before the truncation point turns to garbage. Measured exactly
            // that on ggml_metal: a 3-turn CLI conversation planned
            // "Partial reuse: keeping 156/173", and the model then confabulated both
            // recalled facts ("name was..." with no name, colour "#FF0000 red" for
            // teal) while staying perfectly fluent - the 17 fresh tokens were the
            // only intact context it had. Gemma 4's TruncateKVCacheCore does the same
            // host sync first, for the same reason.
            EnsureFusedKvHostSynchronized();
            base.TruncateKVCacheCore(tokenCount);
            _cachedSWAMaskStartPos = -1;
            ResetFusedDecodeCache();
            // The parked per-rank graphs bake the start position this rewind just
            // moved. The per-rank KV buffers themselves are NOT invalidated: the
            // device copy is authoritative under TP.
            ResetMuseGlimmerTpGraphs();
            SyncMuseGlimmerTpKvCacheToHost();
            if (_kvCacheK == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override bool SupportsKVStateSnapshot
            => IsTensorParallel
                ? _tpKvCacheK != null && _tpKvCacheV != null
                : _kvCacheK != null && _kvCacheV != null;

        /// <summary>
        /// Under tensor parallelism the K/V for one layer is split across ranks, so the
        /// snapshot walks the per-rank tensors as if they were <c>layers * tp</c> layers,
        /// LAYER-MAJOR and rank-minor. That works because a pooled block is an opaque
        /// byte blob: any stable flattening round-trips, and
        /// <see cref="KvBlockTransfer"/> validates and copies each tensor against its
        /// OWN row count through that tensor's own host mirror - it never routes through
        /// an op, so mixing tensors from different rank allocators is safe.
        ///
        /// Without this, snapshots would be unavailable under --tp, and because
        /// Muse-Glimmer is not an <c>IBatchedPagedModel</c> that would make
        /// <c>InferenceEngineHost.TryGetEngine</c> return null - i.e. the server would
        /// refuse to serve the model at all under tensor parallelism.
        /// </summary>
        private Tensor[] _tpKvFlatK;
        private Tensor[] _tpKvFlatV;

        private void EnsureTpKvFlatViews()
        {
            int tp = TpDegree;
            int need = Config.NumLayers * tp;
            if (_tpKvFlatK != null && _tpKvFlatK.Length == need)
                return;
            _tpKvFlatK = new Tensor[need];
            _tpKvFlatV = new Tensor[need];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int r = 0; r < tp; r++)
                {
                    _tpKvFlatK[l * tp + r] = _tpKvCacheK[l][r];
                    _tpKvFlatV[l * tp + r] = _tpKvCacheV[l][r];
                }
            }
        }

        /// <summary>Cache tensors the snapshot path should address, flattened when
        /// tensor parallelism splits each layer across ranks.</summary>
        private (Tensor[] k, Tensor[] v) SnapshotCaches()
        {
            if (!IsTensorParallel)
                return (_kvCacheK, _kvCacheV);
            EnsureTpKvFlatViews();
            return (_tpKvFlatK, _tpKvFlatV);
        }

        /// <summary>
        /// The POOLED snapshot path is capped at the sliding-window ring size, for the
        /// same reason Gemma 4 caps at its window: restoring more positions than a ring
        /// layer physically holds would have to reconstruct rows that the wrap already
        /// overwrote. Below the cap a restore is byte-identical to a real prefill (no
        /// row wraps, so every layer addresses linearly).
        ///
        /// Reuse BEYOND the cap still happens, and correctly, through live-cache
        /// continuation (BatchExecutor.ComputeLiveContinuationLcp), which keeps the
        /// model's actual cache between same-session turns instead of rebuilding it.
        /// That path is only armed for models that report a finite cap, so publishing
        /// one here is what turns multi-turn reuse on for Muse-Glimmer.
        /// </summary>
        public override int MaxReusablePrefixTokens
            => _kvSwaRows > 0 ? _kvSwaRows : int.MaxValue;

        public override string KVStateFingerprint =>
            $"muse-glimmer|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}" +
            $"|hd={_headDim}|swa={_slidingWindow}|rows={_kvSwaRows}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
        {
            var (k, v) = SnapshotCaches();
            return KvBlockTransfer.ComputeBlockByteSize(k, v, tokenCount);
        }

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            // Either fused kernel writes K/V on-device; the snapshot walks the caches
            // through host pointers, so the device rows have to come back first or we
            // would capture the host mirror's stale (zero-filled) bytes.
            EnsureFusedKvHostSynchronized();
            SyncMuseGlimmerTpKvCacheToHost();
            var (k, v) = SnapshotCaches();
            return KvBlockTransfer.Extract(
                _allocator, k, v, _cacheSeqLen,
                startToken, tokenCount, destination, _kvSwaRows);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            // Drop the dirty flag first: the host copy is about to become
            // authoritative, so a later sync must not overwrite it with device rows.
            SyncMuseGlimmerTpKvCacheToHost();
            if (IsTensorParallel)
                EnsureTpCacheCapacity(destToken + tokenCount);
            else
                EnsureCacheCapacity(destToken + tokenCount);

            var (k, v) = SnapshotCaches();
            if (!KvBlockTransfer.Inject(
                    _allocator, k, v, _cacheSeqLen,
                    destToken, tokenCount, source, _kvSwaRows))
            {
                return false;
            }
            // The host copy is now authoritative for [0, destToken+tokenCount); the
            // device copies are dropped below so the next forward re-uploads it.
            _fusedKvDirty = false;
            _cacheSeqLen = destToken + tokenCount;
            _cachedSWAMaskStartPos = -1;
            // Dropping the device copies moves the KV buffers the captured decode
            // graph baked in; replaying that capture would hit freed memory.
            ResetFusedDecodeCache();
            ResetMuseGlimmerTpGraphs();
            for (int i = 0; i < k.Length; i++)
            {
                InvalidateTensorDeviceCache(k[i]);
                InvalidateTensorDeviceCache(v[i]);
            }
            return true;
        }

        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new MuseGlimmerVisionEncoder(mmProjPath, _allocator);
            _visionEncoder.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public MuseGlimmerVisionEncoder VisionEncoder => _visionEncoder;

        internal bool HasPendingVisionEmbeddings => _pendingVisionEmbeddingsList.Count > 0;

        /// <summary>
        /// Largest number of tokens submitted in one forward. llama.cpp splits a
        /// prompt into <c>n_ubatch</c>-sized chunks for the same reason: the compute
        /// graph's activations scale with the row count, so a single 16K-token graph
        /// needs tens of GB even though 16K tokens of KV cost under a GB. 0 disables
        /// chunking.
        /// </summary>
        private static readonly int PrefillChunkTokens = ResolvePrefillChunk();

        private static int ResolvePrefillChunk()
        {
            string raw = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return 2048;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (IsTensorParallel)
                return ForwardTP(tokens);

            // A multimodal prompt chunks too: ForwardChunked re-slices the pending vision
            // spans onto each piece. A 4096-token image plus its text would otherwise be
            // one forward, which the sliding-window ring cannot hold (see the throw below)
            // and whose activations dwarf the chunked ones.
            if (PrefillChunkTokens > 0 && tokens.Length > PrefillChunkTokens)
                return ForwardChunked(tokens, PrefillChunkTokens);

            // The SWA ring holds sliding_window + chunk + 1 rows, so a forward wider
            // than (rows - sliding_window) would alias two live positions onto one slot
            // and silently corrupt all 39 sliding-window layers. Chunking keeps text
            // prompts under the limit by construction; a multimodal prompt skips
            // chunking, so it is the one way to get here oversized.
            if (_kvSwaRows > 0 && tokens.Length > _kvSwaRows - _slidingWindow)
            {
                throw new InvalidOperationException(
                    $"Muse-Glimmer forward of {tokens.Length} tokens exceeds what the sliding-window " +
                    $"ring can hold in one pass ({_kvSwaRows - _slidingWindow} rows: ring {_kvSwaRows} " +
                    $"minus window {_slidingWindow}). Raise TS_MUSE_GLIMMER_PREFILL_CHUNK (the ring is " +
                    "sized from it) or set TS_MUSE_GLIMMER_SWA_RING=0.");
            }

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);
            TraceLayerSetRows(seqLen, startPos);

            // Text-only fast path: the fused graph does the embedding gather and the
            // weightless input norm itself, so nothing is materialised on the host.
            if (!HasPendingVisionEmbeddings && FusedInGraphEmbedAvailable)
            {
                float[] embedded = TryFusedForward(null, seqLen, startPos, null, null, tokens);
                if (embedded != null)
                {
                    TraceLogits(embedded);
                    TraceLayerDone();
                    _logitsBuffer = embedded;
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _forwardSw.Stop();
                    return _logitsBuffer;
                }
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            // Multimodal rows replace the token embeddings BEFORE the input norm:
            // llama.cpp feeds projected image embeddings straight into build_inp_embd
            // and applies the weightless RMSNorm to the merged stream.
            DrainPendingVisionEmbeddings(hidden, startPos);

            ApplyInputEmbeddingNorm(hidden, seqLen);

            // Whole-model fused graph (one GGML submission for all 52 layers, the
            // final norm, the LM head, the logit scale and the softcap).
            float[] fused = TryFusedForward(hidden, seqLen, startPos, null, null);
            if (fused != null)
            {
                TraceLogits(fused);
                TraceLayerDone();
                hidden.Dispose();
                _logitsBuffer = fused;
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return _logitsBuffer;
            }

            EnsureFusedKvHostSynchronized();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                TraceLayerResidual(hidden, l);
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }
            TraceLayerResidual(hidden, Config.NumLayers);

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var lastRow = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(lastRow);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            t0 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, _hasTiedOutput ? "token_embd.weight" : "output.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            lastHidden.Dispose();

            ApplyOutputTransform(logitsTensor);

            t0 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
            logitsTensor.Dispose();
            TraceLogits(_logitsBuffer);
            TraceLayerDone();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        protected override float[] ForwardRefillCore(int[] tokens)
        {
            // The chunk loop below runs PrefillWithoutLogits, which is the
            // single-GPU path and reads the (null) non-TP KV caches. ForwardTP
            // does its own chunking, so hand it the whole prompt instead.
            if (IsTensorParallel)
                return ForwardCore(tokens);

            if (tokens == null || tokens.Length <= 1)
                return ForwardCore(tokens);

            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;
            if (tokens.Length <= chunkSize)
                return ForwardCore(tokens);

            // The pending vision spans are positioned against THIS token array, so the
            // chunk loop has to re-slice them exactly as ForwardChunked does. Without
            // this, chunk 0 would receive a span whose rows run past its end and
            // InjectVisionEmbeddings' Narrow would throw (or, for a span starting after
            // the first chunk, the image would be dropped entirely).
            List<(Tensor embeddings, int position)> spans = null;
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                spans = new List<(Tensor, int)>(_pendingVisionEmbeddingsList);
                _pendingVisionEmbeddingsList.Clear();
            }

            try
            {
                for (int pos = 0; pos < lastIdx; pos += chunkSize)
                {
                    int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                    var chunk = new int[chunkLen];
                    Array.Copy(tokens, pos, chunk, 0, chunkLen);
                    RequeueVisionSpansForWindow(spans, pos, chunkLen);
                    PrefillWithoutLogits(chunk);
                }
                RequeueVisionSpansForWindow(spans, lastIdx, 1);
                return ForwardCore(new[] { tokens[lastIdx] });
            }
            finally
            {
                if (spans != null)
                    foreach (var (embeddings, _) in spans)
                        embeddings?.Dispose();
            }
        }

        /// <summary>
        /// Push the sub-range of each span in <paramref name="spans"/> that falls inside
        /// the window [<paramref name="windowStart"/>, +<paramref name="windowLen"/>) onto
        /// the pending list, re-based to the window's own coordinates. The pushed tensors
        /// are standalone copies because <see cref="DrainPendingVisionEmbeddings"/> disposes
        /// whatever it injects.
        /// </summary>
        private void RequeueVisionSpansForWindow(
            List<(Tensor embeddings, int position)> spans, int windowStart, int windowLen)
        {
            if (spans == null)
                return;
            foreach (var (embeddings, position) in spans)
            {
                int rows = (int)embeddings.Sizes[0];
                int overlapStart = Math.Max(windowStart, position);
                int overlapEnd = Math.Min(windowStart + windowLen, position + rows);
                if (overlapStart >= overlapEnd)
                    continue;
                using var view = embeddings.Narrow(0, overlapStart - position, overlapEnd - overlapStart);
                _pendingVisionEmbeddingsList.Add((Ops.NewContiguous(view), overlapStart - windowStart));
            }
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            if (!HasPendingVisionEmbeddings && FusedInGraphEmbedAvailable
                && TryFusedForward(null, seqLen, startPos, null, null, tokens) != null)
            {
                _cacheSeqLen += seqLen;
                _forwardSw.Stop();
                return;
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            DrainPendingVisionEmbeddings(hidden, startPos);
            ApplyInputEmbeddingNorm(hidden, seqLen);

            // The fused kernel always folds the LM head, so a logits-free prefill
            // chunk simply discards them - still far cheaper than the per-op chain.
            if (TryFusedForward(hidden, seqLen, startPos, null, null) != null)
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardSw.Stop();
                return;
            }

            EnsureFusedKvHostSynchronized();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        /// <summary>
        /// Feeds a long prompt through in <paramref name="chunk"/>-token pieces.
        /// Each piece advances <c>_cacheSeqLen</c>, so the next one sees the previous
        /// one's keys and values exactly as if the whole prompt had been submitted at
        /// once; only the final piece's logits are wanted, and those are what the
        /// last call returns.
        /// </summary>
        private float[] ForwardChunked(int[] tokens, int chunk)
        {
            // Take ownership of the pending spans: their positions are relative to THIS
            // call's token array, so each chunk needs the sub-range that lands inside it,
            // re-based to the chunk's own coordinates. Handing the whole span to the first
            // chunk (or dropping it) is what would silently strand image rows.
            List<(Tensor embeddings, int position)> spans = null;
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                spans = new List<(Tensor, int)>(_pendingVisionEmbeddingsList);
                _pendingVisionEmbeddingsList.Clear();
            }

            try
            {
                float[] last = null;
                for (int off = 0; off < tokens.Length; off += chunk)
                {
                    int n = Math.Min(chunk, tokens.Length - off);
                    int[] piece = new int[n];
                    Array.Copy(tokens, off, piece, 0, n);

                    RequeueVisionSpansForWindow(spans, off, n);
                    last = ForwardCore(piece);
                }
                return last;
            }
            finally
            {
                if (spans != null)
                    foreach (var (embeddings, _) in spans)
                        embeddings?.Dispose();
            }
        }

        private void DrainPendingVisionEmbeddings(Tensor hidden, int startPos)
        {
            if (_pendingVisionEmbeddingsList.Count == 0)
                return;
            foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
            {
                InjectVisionEmbeddings(hidden, embeddings, position, startPos);
                embeddings.Dispose();
            }
            _pendingVisionEmbeddingsList.Clear();
        }

        private void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos, int startPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            using var target = hidden.Narrow(0, insertPos, numVisionTokens);
            if (ReferenceEquals(target.Allocator, visionEmbeddings.Allocator))
            {
                Ops.Copy(target, visionEmbeddings);
            }
            else
            {
                float[] host = visionEmbeddings.GetElementsAsFloat((int)visionEmbeddings.ElementCount());
                using var staged = new Tensor(hidden.Allocator, visionEmbeddings.ElementType, visionEmbeddings.Sizes);
                staged.SetElementsAsFloat(host);
                Ops.Copy(target, staged);
            }
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {startPos + insertPos})");
        }

        /// <summary>
        /// Weightless RMSNorm over the input embeddings (llama.cpp:
        /// build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)).
        /// </summary>
        private void ApplyInputEmbeddingNorm(Tensor hidden, int seqLen)
        {
            int dim = Config.HiddenSize;
            if (_onesForEmbNorm == null || (int)_onesForEmbNorm.Sizes[0] != dim)
            {
                _onesForEmbNorm?.Dispose();
                _onesForEmbNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForEmbNorm, 1f);
            }
            using var reshaped = hidden.View(seqLen, dim);
            Ops.RMSNorm(reshaped, reshaped, _onesForEmbNorm, null, Config.Eps);
        }

        /// <summary>logits *= logit_scale, then tanh softcapping.</summary>
        private void ApplyOutputTransform(Tensor logits)
        {
            if (_finalLogitScale != 1.0f && _finalLogitScale > 0f)
                Ops.Mul(logits, logits, _finalLogitScale);

            if (_finalLogitSoftcap > 0f)
            {
                float cap = _finalLogitSoftcap;
                Ops.Mul(logits, logits, 1f / cap);
                Ops.Tanh(logits, logits);
                Ops.Mul(logits, logits, cap);
            }
        }

        /// <summary>RMSNorm with an explicit epsilon (the post norms use 1e-8, not Config.Eps).</summary>
        private Tensor RMSNormWithEps(Tensor input, string weightName, float eps)
        {
            long t0 = Stopwatch.GetTimestamp();
            var alpha = _weights[weightName];
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);
            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;
            Tensor result = Ops.RMSNorm(null, src, alpha, null, eps);
            input2d?.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] names = _layerWeightNames[layer];

            // Pre-attention norm; the same tensor feeds Q/K/V and the output gate.
            using var attnNormed = RMSNormOp(hidden, names[0]);

            using var attnOut = Attention(attnNormed, layer, names, seqLen, startPos);

            // post_attention_norm: rms_norm(eps = 1e-8) * weight
            using var postAttnNormed = RMSNormWithEps(attnOut, names[8], PostNormEps);

            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            // ffn_norm + gate/up + SiLU-mul + down in a single GGML graph when the
            // backend supports it (actType 0 = SwiGLU). The fused rms_norm uses the
            // same weight and Config.Eps as RMSNormOp, so the result is numerically
            // identical to the unfused chain; it just keeps the [tokens, 2*n_ff]
            // intermediate device-resident instead of round-tripping it per op.
            // post_ffw_norm still runs on the small returned tensor.
            //
            // Prefill only. Measured on Muse-Glimmer-30B / ggml_cuda / RTX 3080:
            // prefill 31.5 -> 55.5 tok/s, but single-row decode 4.3 -> 3.9 tok/s -
            // at one row the fused graph's build/bind cost outweighs keeping the
            // [1, 2*n_ff] intermediate resident.
            Tensor ffnOut = seqLen > 1
                ? TryFusedDenseFFNProject(postAttnNormed, names[9], names[10], names[11], actType: 0)
                : null;
            if (ffnOut == null)
            {
                using var ffnNormed = RMSNormOp(postAttnNormed, names[9]);
                ffnOut = FFN(ffnNormed, names[10], names[11], seqLen);
            }

            using var _ffnOut = ffnOut;
            using var postFfnNormed = RMSNormWithEps(ffnOut, names[12], PostNormEps);

            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);
            return result;
        }

        private Tensor Attention(Tensor input, int layer, string[] names, int seqLen, int startPos)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool useRope = _isSwaLayer[layer];

            Tensor q = LinearForward(input, names[1]);
            Tensor k = LinearForward(input, names[2]);
            Tensor v = LinearForward(input, names[3]);
            Tensor gate = LinearForward(input, names[4]);

            // Per-head QK RMSNorm. The Q norm weight absorbs qk_scale_factor.
            if (seqLen == 1)
            {
                RMSNormInPlace(q, _weights[names[5]], Config.NumHeads, _headDim, Config.Eps);
                RMSNormInPlace(k, _weights[names[6]], Config.NumKVHeads, _headDim, Config.Eps);
            }
            else
            {
                q = ApplyBatchRMSNorm(q, names[5], Config.NumHeads, seqLen, _headDim);
                k = ApplyBatchRMSNorm(k, names[6], Config.NumKVHeads, seqLen, _headDim);
            }

            // RoPE (NORM / interleaved pairs) on the sliding-window layers only.
            //
            // The single-row host RoPE below is a GGML/CUDA optimisation - it trades a
            // RoPEEx dispatch for a scalar loop over host memory. On MLX that trade
            // inverts: Q and K are device-authoritative (they were just produced by a
            // quantized matmul and an RMSNorm), so GetFloatPtr forces an mlx_eval plus a
            // device->host copy and marks the storage host-dirty, which makes the very
            // next device op free and re-upload it. Qwen 3.5 routes MLX and direct CUDA
            // to the tensor RoPE at seqLen == 1 for exactly this reason.
            if (useRope)
            {
                if (seqLen == 1 && _backend != BackendType.Mlx)
                {
                    ApplyNormRoPEDecode(q, Config.NumHeads, _headDim, startPos, _ropeFreqs);
                    ApplyNormRoPEDecode(k, Config.NumKVHeads, _headDim, startPos, _ropeFreqs);
                }
                else
                {
                    q = ApplyRoPEPrefill(q, Config.NumHeads, _headDim, seqLen, startPos);
                    k = ApplyRoPEPrefill(k, Config.NumKVHeads, _headDim, seqLen, startPos);
                }
            }

            float qScale = 1f / MathF.Sqrt(_headDim);
            Ops.Mul(q, q, qScale);

            int totalSeqLen = startPos + seqLen;
            Tensor attn;

            if (seqLen == 1)
            {
                RequireUniformCache(layer);
                CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                    Config.NumKVHeads, _headDim, startPos);

                // MLX builds one lazy slice_update node per decode step, so an
                // un-materialised cache makes every later attention read walk a chain
                // that grows linearly in decode-step count - fine for the first few
                // hundred tokens, catastrophic past ~500. Gemma 4 breaks the chain on
                // this same schedule.
                if (_backend == BackendType.Mlx && MlxKvMaterializeInterval > 0
                    && ((startPos + 1 + layer) % MlxKvMaterializeInterval) == 0)
                {
                    MlxFusedOps.TryMaterialize(_kvCacheK[layer]);
                    MlxFusedOps.TryMaterialize(_kvCacheV[layer]);
                }

                int attendLen = _isSwaLayer[layer] ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
                int attendStart = totalSeqLen - attendLen;

                attn = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * _headDim);

                // On MLX the host loop below is the single most expensive thing in the
                // token: GetHalfPointer(kCache)/(vCache) each force an mlx_eval plus a
                // device->host copy of the WHOLE cache and then mark it host-dirty, so
                // the next write frees and re-uploads it - 104 full-cache transfers each
                // way per token across 52 layers. TryDecodeAttention keeps all of it on
                // device. Q was already scaled by 1/sqrt(headDim) above, hence scale: 1f.
                // RequireUniformCache above guarantees a linear cache here, so the
                // circular mode is never needed (the ring is only armed for the fused
                // GGML kernel).
                bool usedMlx = _backend == BackendType.Mlx &&
                    MlxFusedOps.TryDecodeAttention(attn, q, _kvCacheK[layer], _kvCacheV[layer],
                        Config.NumHeads, Config.NumKVHeads, _headDim,
                        attendStart, attendLen, KvRows(layer), circular: false, scale: 1f);
                if (!usedMlx)
                {
                    AttentionDecodeWithWindow(q, _kvCacheK[layer], _kvCacheV[layer], attn,
                        Config.NumHeads, Config.NumKVHeads, _headDim, _headDim,
                        attendStart, totalSeqLen);
                }
            }
            else
            {
                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, _headDim);
                Tensor kHeads = ReshapeToHeads(k, Config.NumKVHeads, seqLen, _headDim);
                Tensor vHeads = ReshapeToHeads(v, Config.NumKVHeads, seqLen, _headDim);

                RequireUniformCache(layer);
                CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                kHeads.Dispose();
                vHeads.Dispose();

                // MLX fast path: SDPA straight off the device-resident cache. It skips
                // BOTH host round-trips the generic chain below pays per layer - the
                // 16x GQA broadcast that ExpandKVHeadsF16 rebuilds on the host, and the
                // sliding-window band that ApplyCausalMask fills on the host (which
                // drags the whole [32, seqLen, totalSeqLen] score matrix down and back).
                attn = _backend == BackendType.Mlx
                    ? TryMlxPrefillAttention(qHeads, layer, seqLen, totalSeqLen)
                    : null;

                if (attn != null)
                {
                    qHeads.Dispose();
                }
                else
                {
                    int groupSize = Config.NumHeads / Config.NumKVHeads;
                    Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    ApplyCausalMask(scores, seqLen, totalSeqLen, _isSwaLayer[layer] ? _slidingWindow : 0);
                    Ops.Softmax(scores, scores);

                    var attnHeads = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, _headDim);
                    Ops.AddmmBatch(attnHeads, 0, attnHeads, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    attn = ReshapeFromHeads(attnHeads, Config.NumHeads, seqLen, _headDim);
                    attnHeads.Dispose();
                }
            }

            q.Dispose();
            k.Dispose();
            v.Dispose();

            // Attention output gate: attn * sigmoid(gate), applied before o_proj.
            Ops.SigmoidMul(attn, attn, gate);
            gate.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            using (attn)
            {
                return LinearForward(attn, names[7]);
            }
        }

        /// <summary>
        /// Multi-row attention through MLX's fused SDPA, reading the live K/V window
        /// straight off the device-resident cache. Every prefill chunk stays on
        /// device: the first chunk uses the plain "causal" string mask, later chunks
        /// a device-built causal(+SWA) band mask, and sliding-window layers narrow
        /// the cache read to the window itself so a 16K prefill does not score 39
        /// layers against out-of-window keys. Returns null when MLX declines the
        /// shape and the caller falls back to the generic materialised-score chain
        /// (host GQA expand + host-masked scores).
        /// </summary>
        private Tensor TryMlxPrefillAttention(Tensor qHeads, int layer, int seqLen, int totalSeqLen)
        {
            int qStart = totalSeqLen - seqLen;
            // The earliest query (position qStart) still sees keys back to
            // qStart - window + 1, so the windowed read starts there, never later.
            int kvStart = _isSwaLayer[layer] && _slidingWindow > 0
                ? Math.Max(0, qStart - _slidingWindow + 1)
                : 0;
            var result = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * _headDim);
            using var kLive = _kvCacheK[layer].Narrow(1, kvStart, totalSeqLen - kvStart);
            using var vLive = _kvCacheV[layer].Narrow(1, kvStart, totalSeqLen - kvStart);
            // Q is pre-scaled by 1/sqrt(headDim), hence scale: 1f.
            if (MlxFusedOps.TryPrefillAttentionBanded(
                    result, qHeads, kLive, vLive,
                    Config.NumHeads, Config.NumKVHeads, _headDim,
                    seqLen, totalSeqLen - kvStart,
                    qStart: qStart,
                    kvStart: kvStart,
                    windowSize: _isSwaLayer[layer] ? _slidingWindow : 0,
                    scale: 1f))
            {
                return result;
            }

            result.Dispose();
            return null;
        }

        private Tensor ApplyBatchRMSNorm(Tensor data, string weightName, int numHeads, int seqLen, int headDim)
        {
            var alpha = _weights[weightName];
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor flat = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return flat;
        }

        /// <summary>
        /// ggml NORM-flavour RoPE: rotates adjacent pairs (x[2i], x[2i+1]).
        /// Decode fast path (single row) to avoid a RoPEEx dispatch.
        /// </summary>
        private unsafe void ApplyNormRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int pairCount = headDim / 2;

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int i = 0, pair = 0; i < pairCount; i++, pair += 2)
                {
                    float angle = position * freqs[i];
                    float cos = MathF.Cos(angle);
                    float sin = MathF.Sin(angle);
                    float x0 = head[pair];
                    float x1 = head[pair + 1];
                    head[pair] = x0 * cos - x1 * sin;
                    head[pair + 1] = x1 * cos + x0 * sin;
                }
            }
            InvalidateTensorDeviceCache(data);
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            // mode 0 == GGML_ROPE_TYPE_NORM (interleaved pairs).
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 0, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>
        /// Single-row decode attention over the half-open key range
        /// [attendStart, totalSeqLen). Full-causal layers pass attendStart = 0;
        /// sliding-window layers pass totalSeqLen - min(totalSeqLen, n_swa), which is
        /// llama.cpp's STANDARD SWA predicate (a key at p0 is visible to the query at
        /// p1 iff p1 - p0 &lt; n_swa).
        ///
        /// <see cref="ApplyModelAlignedKvCacheDefault"/> picks F16 caches for quantized
        /// models, so all three cache dtypes are handled: F32 and F16 walk the cache in
        /// place, block-quantized caches dequantize the live window first (same
        /// deep-fallback shape as <see cref="AttentionDecodePureCS"/>).
        /// </summary>
        private unsafe void AttentionDecodeWithWindow(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int attendStart, int totalSeqLen)
        {
            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                // Dequantize [0, totalSeqLen) into compact F32 (no GQA broadcast: the
                // loop below reads per KV head), then re-enter. The compact copy's
                // Sizes[1] == totalSeqLen doubles as its row stride.
                using Tensor kF32 = ExpandKVHeads(kCache, 1, totalSeqLen);
                using Tensor vF32 = ExpandKVHeads(vCache, 1, totalSeqLen);
                AttentionDecodeWithWindow(q, kF32, vF32, result,
                    numHeads, numKVHeads, keyDim, valDim, attendStart, totalSeqLen);
                return;
            }

            bool half = kCache.ElementType == DType.Float16;
            if (half != (vCache.ElementType == DType.Float16))
            {
                throw new NotSupportedException(
                    $"Muse-Glimmer decode attention needs matching K/V cache dtypes, got {kCache.ElementType}/{vCache.ElementType}.");
            }

            float* qPtr = GetFloatPtr(q);
            float* rPtr = GetFloatPtr(result);
            float* kPtrF = half ? null : GetFloatPtr(kCache);
            float* vPtrF = half ? null : GetFloatPtr(vCache);
            ushort* kPtrH = half ? TensorComputePrimitives.GetHalfPointer(kCache) : null;
            ushort* vPtrH = half ? TensorComputePrimitives.GetHalfPointer(vCache) : null;

            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;
            int attendLen = totalSeqLen - attendStart;

            nint qN = (nint)qPtr, rN = (nint)rPtr;
            nint kN = half ? (nint)kPtrH : (nint)kPtrF;
            nint vN = half ? (nint)vPtrH : (nint)vPtrF;

            void Head(int h)
            {
                float* qHead = (float*)qN + h * keyDim;
                int kvHead = h / groupSize;
                long kBase = (long)kvHead * maxSeqLen * keyDim + (long)attendStart * keyDim;
                long vBase = (long)kvHead * maxSeqLen * valDim + (long)attendStart * valDim;

                float* scores = stackalloc float[attendLen];

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    float s = half
                        ? TensorComputePrimitives.DotF32F16(qHead, (ushort*)kN + kBase + (long)t * keyDim, keyDim)
                        : VecDot(qHead, (float*)kN + kBase + (long)t * keyDim, keyDim);
                    scores[t] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0;
                for (int t = 0; t < attendLen; t++)
                {
                    float e = MathF.Exp(scores[t] - maxScore);
                    scores[t] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int t = 0; t < attendLen; t++)
                    scores[t] *= invSum;

                float* rHead = (float*)rN + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < attendLen; t++)
                {
                    if (half)
                        TensorComputePrimitives.ScaleAddF16(rHead, (ushort*)vN + vBase + (long)t * valDim, scores[t], valDim);
                    else
                        VecScaleAdd(rHead, (float*)vN + vBase + (long)t * valDim, scores[t], valDim);
                }
            }

            if (numHeads > 1 && (long)numHeads * attendLen >= 4096)
                System.Threading.Tasks.Parallel.For(0, numHeads, Head);
            else
                for (int h = 0; h < numHeads; h++) Head(h);

            InvalidateTensorDeviceCache(result);
        }

        /// <summary>
        /// Causal mask plus, for sliding-window layers, the llama.cpp STANDARD SWA
        /// predicate: a key at p0 is masked from a query at p1 when p1 - p0 &gt;= n_swa.
        /// </summary>
        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize)
        {
            int startPos = totalKVLen - queryLen;
            Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);

            if (windowSize <= 0)
                return;

            if (_cachedSWAMaskWidths == null ||
                _cachedSWAMaskQueryLen != queryLen ||
                _cachedSWAMaskStartPos != startPos)
            {
                _cachedSWAMaskWidths = new int[queryLen];
                _cachedSWAMaskQueryLen = queryLen;
                _cachedSWAMaskStartPos = startPos;
                for (int qi = 0; qi < queryLen; qi++)
                    _cachedSWAMaskWidths[qi] = Math.Max(0, startPos + qi - windowSize + 1);
            }

            float* sPtr = GetFloatPtr(scores);
            int numHeads = (int)scores.Sizes[0];
            int rowStride = queryLen * totalKVLen;

            for (int h = 0; h < numHeads; h++)
            {
                float* headScores = sPtr + (long)h * rowStride;
                for (int qi = 0; qi < queryLen; qi++)
                {
                    int width = _cachedSWAMaskWidths[qi];
                    if (width > 0)
                        new Span<float>(headScores + (long)qi * totalKVLen, width).Fill(float.NegativeInfinity);
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            _onesForEmbNorm?.Dispose();
            DisposeDFlashDrafter();
            DisposeMuseGlimmerTpState();
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            base.Dispose();
        }
    }
}
