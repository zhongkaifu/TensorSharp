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
using System.Diagnostics;
using TensorSharp;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Muse-Glimmer's MLX pipelined greedy decode: the whole 52-layer decode
    /// step is built as ONE MLX graph, inside ONE MLX worker round-trip, with
    /// ZERO host syncs. The predicted token is returned as a <c>[1]</c> int32
    /// DEVICE tensor, so the caller can submit step N+1 before it reads step
    /// N's token.
    ///
    /// Ported from Qwen35Model's SubmitGreedyDecodeStep (since removed); see
    /// <c>ModelBase.SupportsPipelinedGreedy</c> for the contract and
    /// <c>TensorSharp.Cli/Program.cs</c> for the driving loop.
    ///
    /// =====================================================================
    /// What this bought, measured on M5 Pro / Muse-Glimmer-30B-UD-IQ2_XXS
    /// (62-token prompt, --top-k 1, 64 decode tokens, all weights resident)
    /// =====================================================================
    ///
    /// The per-op path decoded at 3.6 tok/s (278 ms/token) and its timing
    /// report blamed "Norm 45.9% / Attention 36.1%". That attribution is a
    /// LAZY-EVAL ARTIFACT: MLX only queues work, so whichever op happens to
    /// force the drain gets charged for everything queued behind it. Running
    /// the same build with TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS=0 and
    /// TS_MLX_MUSE_GLIMMER_KV_MATERIALIZE=0 (no intermediate evals at all)
    /// left the token time UNCHANGED at 281 ms while moving 98.1% of it onto
    /// the single logits drain — proof that the graph BUILD is ~6 ms/token
    /// and the other ~275 ms is device execution.
    ///
    /// A `sample(1)` profile of the per-op run agrees: the host thread is
    /// 98% blocked in __psynch_cvwait, the MLX worker is 47% in
    /// IOSurfaceSharedEvent::waitUntilSignaledValue (GPU wait) and 38% in
    /// eval_impl's condition wait, and NO thread in the process is CPU-busy.
    /// The token is GPU-bound, not hand-off-bound.
    ///
    /// So this path removes what is actually removable on the host side:
    ///   * ~1150 MLX worker hand-offs per token collapse to 1 (see
    ///     MlxFusedOps.RunDecodeGraphBatched);
    ///   * the 202048-float logits tensor never crosses to the host — argmax
    ///     and the next embedding lookup both run on device;
    ///   * the remaining host work (graph build) overlaps the previous step's
    ///     GPU execution instead of serializing behind it.
    ///
    /// What it does NOT fix, and what still separates MLX from ggml_metal's
    /// 47 ms/token here, is the throughput of the quantized matmul kernels
    /// themselves. This GGUF is a mixed Unsloth-Dynamic quant: IQ2_S 41% of
    /// the weight bytes (ffn_gate/ffn_up), IQ3_XXS 25% (ffn_down), Q6_K 21%
    /// (token_embd + LM head), IQ2_XXS 7%, IQ3_S 6%. A decode token has to
    /// stream ~9.6 GB of weights; ggml_metal's 47 ms/token is ~205 GB/s, i.e.
    /// the memory roofline, while MLX's ~275 ms is ~35 GB/s. The MLX decode
    /// kernels for those types (MlxNative.Iq2SMatmulSource and friends) call
    /// their per-ELEMENT dequant helper once per weight value, so each output
    /// value re-loads the super-block header and re-does the codebook lookup
    /// that ggml's Metal kernels amortize over a whole 32-value sub-block.
    /// Closing that gap is a kernel rewrite in MlxNative, not a scheduling
    /// change here.
    /// </summary>
    public partial class MuseGlimmerModel
    {
        // ====================================================================
        // Cached weight handles
        // ====================================================================
        //
        // Every projection and norm the decode step touches is resolved ONCE
        // into a flat array, exactly as Qwen 3.5 does with its _attnQkvQW[] /
        // _postAttnNormW[] tables. The per-op path re-hashes 13 weight-name
        // strings per layer per token (676 dictionary probes) and calls
        // QuantizedWeight.EnsureDeviceCacheKey on every matmul; neither shows
        // up in a GPU-bound profile, but both are pure waste inside a loop
        // that is supposed to do nothing but queue kernels.

        private bool _mlxDecodeProbed;
        private bool _mlxDecodeReady;

        private Tensor[] _mgAttnNormW;
        private Tensor[] _mgQNormW;
        private Tensor[] _mgKNormW;
        private Tensor[] _mgPostAttnNormW;
        private Tensor[] _mgFfnNormW;
        private Tensor[] _mgPostFfwNormW;

        private QuantizedWeight[] _mgQQW, _mgKQW, _mgVQW, _mgGateQW, _mgOutQW, _mgGateUpQW, _mgDownQW;
        private Tensor[] _mgQF32, _mgKF32, _mgVF32, _mgGateF32, _mgOutF32, _mgGateUpF32, _mgDownF32;

        private Tensor _mgOutputNormW;
        private QuantizedWeight _mgLmHeadQW;
        private Tensor _mgLmHeadF32;
        private QuantizedWeight _mgTokenEmbdQW;

        /// <summary>
        /// Input embedding for the NEXT pipelined step, produced on device from
        /// the current step's argmax. Null before the first step and after
        /// <see cref="ResetPipelinedGreedyState"/>.
        /// </summary>
        private Tensor _pipelineNextInputDevice;

        /// <summary>Wall time spent submitting pipelined decode steps.</summary>
        private long _mgPipelinedSubmitTicks;
        private int _mgPipelinedSteps;

        /// <summary>
        /// TS_MUSE_GLIMMER_MLX_STAGE_PROFILE=1 inserts a BLOCKING mlx_eval after
        /// each stage of the decode step and charges the wait to that stage.
        /// This is the only honest way to attribute time on a lazy backend, and
        /// it is what produced the IQ2_S / IQ3_XXS / Q6_K numbers in the class
        /// comment. It serializes the whole token, so it roughly doubles decode
        /// latency — diagnostics only, never on by default.
        /// </summary>
        private static readonly bool MlxStageProfileEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_MLX_STAGE_PROFILE"), "1", StringComparison.Ordinal);

        private long _mgStageQkvTicks, _mgStageAttnTicks, _mgStageOutProjTicks;
        private long _mgStageFfnTicks, _mgStageNormTicks, _mgStageHeadTicks;

        // ====================================================================
        // Contract
        // ====================================================================

        /// <summary>
        /// Pipelined greedy decode is MLX-only: it needs a device-side argmax
        /// and a device-side embedding gather, neither of which the GGML or
        /// CUDA paths expose here (they have their own whole-model fused graph
        /// instead, see MuseGlimmerModel.Fused.cs).
        ///
        /// Also refused when:
        ///   * tensor parallelism is active — the TP decode arrays own the KV
        ///     caches and the collectives make a "no host sync" step impossible;
        ///   * vision embeddings are still pending — those have to be spliced
        ///     into a multi-row forward before any decode step runs;
        ///   * the sliding-window KV ring is armed — a ring row is not an
        ///     absolute position and only the fused GGML kernel can read it
        ///     (see <see cref="RequireUniformCache"/>). It is never armed on
        ///     MLX today; the guard keeps that from becoming silent corruption.
        /// </summary>
        public override bool SupportsPipelinedGreedy =>
            IsPipelinedGreedyEligible(
                _backend,
                IsTensorParallel,
                HasPendingVisionEmbeddings,
                _kvSwaRows)
            && EnsureMlxDecodeWeightCache();

        /// <summary>
        /// The backend/state half of <see cref="SupportsPipelinedGreedy"/>,
        /// split out so it can be unit-tested without a 10 GB GGUF (the weight
        /// half needs real weights and is covered by the CLI parity runs).
        /// </summary>
        internal static bool IsPipelinedGreedyEligible(
            BackendType backend, bool isTensorParallel, bool hasPendingVisionEmbeddings, int kvSwaRows)
        {
            return backend == BackendType.Mlx
                && !isTensorParallel
                && !hasPendingVisionEmbeddings
                && kvSwaRows == 0;
        }

        /// <summary>
        /// Resolve every weight the decode step needs. Returns false — leaving
        /// the model on the per-op path — if anything is missing, so a partially
        /// converted GGUF degrades instead of throwing mid-generation.
        /// </summary>
        private bool EnsureMlxDecodeWeightCache()
        {
            if (_mlxDecodeProbed)
                return _mlxDecodeReady;

            _mlxDecodeProbed = true;
            _mlxDecodeReady = false;

            if (_backend != BackendType.Mlx || IsTensorParallel || _layerWeightNames == null)
                return false;

            int n = Config.NumLayers;
            _mgAttnNormW = new Tensor[n];
            _mgQNormW = new Tensor[n];
            _mgKNormW = new Tensor[n];
            _mgPostAttnNormW = new Tensor[n];
            _mgFfnNormW = new Tensor[n];
            _mgPostFfwNormW = new Tensor[n];
            _mgQQW = new QuantizedWeight[n];
            _mgKQW = new QuantizedWeight[n];
            _mgVQW = new QuantizedWeight[n];
            _mgGateQW = new QuantizedWeight[n];
            _mgOutQW = new QuantizedWeight[n];
            _mgGateUpQW = new QuantizedWeight[n];
            _mgDownQW = new QuantizedWeight[n];
            _mgQF32 = new Tensor[n];
            _mgKF32 = new Tensor[n];
            _mgVF32 = new Tensor[n];
            _mgGateF32 = new Tensor[n];
            _mgOutF32 = new Tensor[n];
            _mgGateUpF32 = new Tensor[n];
            _mgDownF32 = new Tensor[n];

            for (int l = 0; l < n; l++)
            {
                string[] names = _layerWeightNames[l];

                if (!_weights.TryGetValue(names[0], out _mgAttnNormW[l])
                    || !_weights.TryGetValue(names[5], out _mgQNormW[l])
                    || !_weights.TryGetValue(names[6], out _mgKNormW[l])
                    || !_weights.TryGetValue(names[8], out _mgPostAttnNormW[l])
                    || !_weights.TryGetValue(names[9], out _mgFfnNormW[l])
                    || !_weights.TryGetValue(names[12], out _mgPostFfwNormW[l]))
                {
                    return false;
                }

                if (!ResolveProjection(names[1], out _mgQQW[l], out _mgQF32[l])
                    || !ResolveProjection(names[2], out _mgKQW[l], out _mgKF32[l])
                    || !ResolveProjection(names[3], out _mgVQW[l], out _mgVF32[l])
                    || !ResolveProjection(names[4], out _mgGateQW[l], out _mgGateF32[l])
                    || !ResolveProjection(names[7], out _mgOutQW[l], out _mgOutF32[l])
                    || !ResolveProjection(names[10], out _mgGateUpQW[l], out _mgGateUpF32[l])
                    || !ResolveProjection(names[11], out _mgDownQW[l], out _mgDownF32[l]))
                {
                    return false;
                }

                // The device cache key has to exist BEFORE the first matmul:
                // resolving it lazily inside the hot loop is exactly the kind
                // of per-op-per-token work this cache exists to remove.
                _mgQQW[l]?.EnsureDeviceCacheKey();
                _mgKQW[l]?.EnsureDeviceCacheKey();
                _mgVQW[l]?.EnsureDeviceCacheKey();
                _mgGateQW[l]?.EnsureDeviceCacheKey();
                _mgOutQW[l]?.EnsureDeviceCacheKey();
                _mgGateUpQW[l]?.EnsureDeviceCacheKey();
                _mgDownQW[l]?.EnsureDeviceCacheKey();
            }

            if (!_weights.TryGetValue("output_norm.weight", out _mgOutputNormW) || _mgOutputNormW == null)
                return false;

            string lmHeadName = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            _quantWeights.TryGetValue(lmHeadName, out _mgLmHeadQW);
            _weights.TryGetValue(lmHeadName, out _mgLmHeadF32);
            if (_mgLmHeadQW == null && _mgLmHeadF32 == null)
                return false;
            _mgLmHeadQW?.EnsureDeviceCacheKey();

            // Optional: without it the next-token embedding falls back to the
            // host gather, which costs one sync per step (correct, just not
            // pipelined).
            _quantWeights.TryGetValue("token_embd.weight", out _mgTokenEmbdQW);
            _mgTokenEmbdQW?.EnsureDeviceCacheKey();

            _mlxDecodeReady = true;
            return true;
        }

        private bool ResolveProjection(string name, out QuantizedWeight qw, out Tensor f32)
        {
            _quantWeights.TryGetValue(name, out qw);
            _weights.TryGetValue(name, out f32);
            return qw != null || f32 != null;
        }

        // ====================================================================
        // The step
        // ====================================================================

        /// <summary>
        /// Queue one decode step and return its predicted token as a
        /// <c>[1]</c> int32 DEVICE tensor. The caller syncs it with
        /// <c>Tensor.GetElementsAsInt(1)</c> and disposes it — after having
        /// submitted the following step, which is the whole point.
        ///
        /// <paramref name="firstTokenForBegin"/> is the token sampled from the
        /// prefill logits on the first call after a prefill; every later call
        /// passes null and consumes the embedding this method already produced
        /// on device.
        /// </summary>
        public override Tensor SubmitGreedyDecodeStep(int? firstTokenForBegin)
        {
            if (!EnsureMlxDecodeWeightCache())
            {
                throw new NotSupportedException(
                    "MuseGlimmerModel.SubmitGreedyDecodeStep requires the MLX backend with every "
                    + "projection and norm resolved; check SupportsPipelinedGreedy before calling.");
            }

            _forwardSw.Start();
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + 1);

            long submitStart = Stopwatch.GetTimestamp();

            // ONE worker round-trip for the entire token. Everything below runs
            // on the MLX worker thread, where the ~1150 nested Invokes a
            // 52-layer step issues short-circuit to direct calls.
            Tensor deviceToken = MlxFusedOps.RunDecodeGraphBatched(
                () => SubmitGreedyDecodeStepCore(firstTokenForBegin, startPos));

            _mgPipelinedSubmitTicks += Stopwatch.GetTimestamp() - submitStart;
            _mgPipelinedSteps++;

            _cacheSeqLen += 1;
            _forwardCount++;
            _forwardSw.Stop();
            return deviceToken;
        }

        private Tensor SubmitGreedyDecodeStepCore(int? firstTokenForBegin, int startPos)
        {
            Tensor hidden;
            if (firstTokenForBegin.HasValue)
            {
                // Begin path: the caller sampled this token from the prefill
                // logits on the host, so the embedding gather starts there too.
                _pipelineNextInputDevice?.Dispose();
                _pipelineNextInputDevice = null;

                long embT0 = Stopwatch.GetTimestamp();
                hidden = Embedding(new[] { firstTokenForBegin.Value });
                _embTicks += Stopwatch.GetTimestamp() - embT0;
            }
            else if (_pipelineNextInputDevice != null)
            {
                hidden = _pipelineNextInputDevice;
                _pipelineNextInputDevice = null;
            }
            else
            {
                throw new InvalidOperationException(
                    "SubmitGreedyDecodeStep: no cached input embedding and no firstTokenForBegin provided.");
            }

            // Weightless RMSNorm over the input embedding, matching
            // ForwardCore (llama.cpp build_norm(inpL, nullptr, nullptr, RMS, -1)).
            ApplyInputEmbeddingNorm(hidden, 1);

            // One positions tensor per step rather than per RoPE call: the
            // position is the same for every layer, and the per-op path was
            // allocating + uploading 78 of these per token (2 per sliding-window
            // layer). They are NOT cached across steps — MLX may still hold the
            // previous step's graph, and rewriting a live host buffer under it
            // is how you get a silently wrong rotation.
            Tensor ropePosQ = null;
            Tensor ropePosK = null;
            try
            {
                for (int layer = 0; layer < Config.NumLayers; layer++)
                    hidden = MlxDecodeLayer(hidden, layer, startPos, ref ropePosQ, ref ropePosK);
            }
            finally
            {
                ropePosQ?.Dispose();
                if (!ReferenceEquals(ropePosK, ropePosQ))
                    ropePosK?.Dispose();
            }

            // Tail: output_norm -> LM head -> logit_scale -> tanh softcap.
            long lmT0 = Stopwatch.GetTimestamp();
            Tensor normed = RMSNormCached(hidden, _mgOutputNormW, Config.Eps);
            hidden.Dispose();
            Tensor logits = LinearCached(normed, _mgLmHeadQW, _mgLmHeadF32);
            normed.Dispose();

            // Kept even though argmax is invariant under both transforms
            // (logit_scale > 0 and tanh are monotonic): four elementwise passes
            // over 202048 floats is ~6 MB of traffic against a ~9.6 GB token,
            // and keeping them means this path samples EXACTLY what Forward
            // does, ties included.
            ApplyOutputTransform(logits);
            _lmHeadTicks += Stopwatch.GetTimestamp() - lmT0;
            StageProfile(logits, ref _mgStageHeadTicks);

            // Device argmax -> [1] int32. This is what replaces the 202048-float
            // host copy the per-op path pays every token.
            var deviceToken = new Tensor(_allocator, DType.Int32, 1);
            if (!MlxFusedOps.TryArgMaxLastAxis(deviceToken, logits))
            {
                // Host fallback: correct, but it syncs, so the pipeline
                // degenerates to the old behaviour for this step.
                long copyT0 = Stopwatch.GetTimestamp();
                float[] host = logits.GetElementsAsFloat((int)logits.ElementCount());
                _logitsCopyTicks += Stopwatch.GetTimestamp() - copyT0;
                deviceToken.SetElementsAsInt(new[] { SampleGreedy(host) });
            }
            logits.Dispose();

            // Queue the NEXT step's input embedding from the device token, so
            // the next call has no host work to do at all.
            _pipelineNextInputDevice = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
            if (!TryLookupNextEmbeddingOnDevice(_pipelineNextInputDevice, deviceToken))
            {
                int hostTok = deviceToken.GetElementsAsInt(1)[0];
                _pipelineNextInputDevice.Dispose();
                _pipelineNextInputDevice = Embedding(new[] { hostTok });
            }
            else
            {
                // Hand the graph to Metal now rather than waiting for the
                // caller's host read of deviceToken to drain it, so the GPU
                // starts this token while the host queues the next one.
                MlxFusedOps.TryAsyncEvaluate(_pipelineNextInputDevice);
            }

            return deviceToken;
        }

        /// <summary>
        /// One Muse-Glimmer decode layer, node-for-node identical to
        /// <c>TransformerBlock</c> + <c>Attention</c> at seqLen == 1 on MLX:
        ///
        ///   attn_norm(eps)            -> Q / K / V / gate projections
        ///   per-head QK RMSNorm(eps)  -> NORM-flavour RoPE on SWA layers only
        ///   Q *= 1/sqrt(head_dim)     -> KV write -> windowed SDPA
        ///   attn *= sigmoid(gate)     -> o_proj
        ///   post_attention_norm(1e-8) -> += residual
        ///   ffn_norm(eps) -> gate_up -> SwiGLU -> down
        ///   post_ffw_norm(1e-8)       -> += residual
        ///
        /// The two epsilons are NOT the same: the pre-norms use the model's
        /// f_norm_rms_eps and the post-norms a hardcoded 1e-8 (llama.cpp
        /// src/models/muse-glimmer.cpp). Swapping them is a silent quality
        /// regression, not a crash.
        /// </summary>
        private Tensor MlxDecodeLayer(Tensor hidden, int layer, int startPos, ref Tensor ropePosQ, ref Tensor ropePosK)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = _headDim;

            long normT0 = Stopwatch.GetTimestamp();
            Tensor attnNormed = RMSNormCached(hidden, _mgAttnNormW[layer], Config.Eps);
            _normTicks += Stopwatch.GetTimestamp() - normT0;

            long attnT0 = Stopwatch.GetTimestamp();
            Tensor q = LinearCached(attnNormed, _mgQQW[layer], _mgQF32[layer]);
            Tensor k = LinearCached(attnNormed, _mgKQW[layer], _mgKF32[layer]);
            Tensor v = LinearCached(attnNormed, _mgVQW[layer], _mgVF32[layer]);
            Tensor gate = LinearCached(attnNormed, _mgGateQW[layer], _mgGateF32[layer]);
            attnNormed.Dispose();

            // Per-head QK RMSNorm. The Q norm weight carries the model's
            // qk_scale_factor (folded in at conversion time).
            RMSNormInPlace(q, _mgQNormW[layer], numHeads, headDim, Config.Eps);
            RMSNormInPlace(k, _mgKNormW[layer], numKVHeads, headDim, Config.Eps);

            // RoPE (ggml NORM flavour, interleaved pairs) on sliding-window
            // layers only; the full-causal layers are NoPE. The tensor RoPE is
            // used even at one row because Q and K are device-authoritative
            // here and the host scalar loop would force an eval plus a
            // round-trip (same reasoning as Attention's MLX branch).
            if (_isSwaLayer[layer])
            {
                ropePosQ ??= CreateDecodeRopePositions(numHeads, startPos);
                ropePosK ??= numKVHeads == numHeads ? ropePosQ : CreateDecodeRopePositions(numKVHeads, startPos);
                q = ApplyDecodeRoPE(q, numHeads, headDim, ropePosQ);
                k = ApplyDecodeRoPE(k, numKVHeads, headDim, ropePosK);
            }

            Ops.Mul(q, q, 1f / MathF.Sqrt(headDim));
            StageProfile(q, ref _mgStageQkvTicks);

            int totalSeqLen = startPos + 1;
            CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v, numKVHeads, headDim, startPos);

            // No periodic TryMaterialize here (the per-op path breaks the lazy
            // slice_update chain every 16 writes). The pipelined loop host-reads
            // one token per step, and this step's own attention reads the cache
            // it just wrote, so every KV write is inside the graph that read
            // drains — the chain can never grow past a single step.
            int attendLen = _isSwaLayer[layer] ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
            int attendStart = totalSeqLen - attendLen;

            var attn = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);
            // Q is already scaled above, hence scale: 1f. RequireUniformCache is
            // guaranteed by SupportsPipelinedGreedy's _kvSwaRows == 0 check, so
            // the circular ring layout is never in play.
            if (!MlxFusedOps.TryDecodeAttention(attn, q, _kvCacheK[layer], _kvCacheV[layer],
                    numHeads, numKVHeads, headDim, attendStart, attendLen, KvRows(layer),
                    circular: false, scale: 1f))
            {
                AttentionDecodeWithWindow(q, _kvCacheK[layer], _kvCacheV[layer], attn,
                    numHeads, numKVHeads, headDim, headDim, attendStart, totalSeqLen);
            }
            StageProfile(attn, ref _mgStageAttnTicks);

            q.Dispose();
            k.Dispose();
            v.Dispose();

            // Attention output gate: attn * sigmoid(gate), before o_proj.
            Ops.SigmoidMul(attn, attn, gate);
            gate.Dispose();

            Tensor attnOut = LinearCached(attn, _mgOutQW[layer], _mgOutF32[layer]);
            attn.Dispose();
            _attnTicks += Stopwatch.GetTimestamp() - attnT0;
            StageProfile(attnOut, ref _mgStageOutProjTicks);

            long postT0 = Stopwatch.GetTimestamp();
            Tensor residual = RMSNormCached(attnOut, _mgPostAttnNormW[layer], PostNormEps);
            attnOut.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - postT0;

            Ops.Add(residual, residual, hidden);
            hidden.Dispose();
            StageProfile(residual, ref _mgStageNormTicks);

            long ffnNormT0 = Stopwatch.GetTimestamp();
            Tensor ffnNormed = RMSNormCached(residual, _mgFfnNormW[layer], Config.Eps);
            _normTicks += Stopwatch.GetTimestamp() - ffnNormT0;

            long ffnT0 = Stopwatch.GetTimestamp();
            Tensor gateUp = LinearCached(ffnNormed, _mgGateUpQW[layer], _mgGateUpF32[layer]);
            ffnNormed.Dispose();

            int halfDim = Config.IntermediateSize > 0 ? Config.IntermediateSize : (int)(gateUp.Sizes[1] / 2);
            // Same shape as ModelBase.FFN at seqLen == 1: narrow views, no copy,
            // in-place SiLU-mul into the gate half.
            Tensor swiGlu = gateUp.Narrow(1, 0, halfDim);
            using (Tensor up = gateUp.Narrow(1, halfDim, halfDim))
                Ops.SiLUMul(swiGlu, swiGlu, up);

            Tensor down = LinearCached(swiGlu, _mgDownQW[layer], _mgDownF32[layer]);
            swiGlu.Dispose();
            gateUp.Dispose();
            _linearTicks += Stopwatch.GetTimestamp() - ffnT0;
            StageProfile(down, ref _mgStageFfnTicks);

            long postFfnT0 = Stopwatch.GetTimestamp();
            Tensor ffnOut = RMSNormCached(down, _mgPostFfwNormW[layer], PostNormEps);
            down.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - postFfnT0;

            Ops.Add(residual, residual, ffnOut);
            ffnOut.Dispose();
            return residual;
        }

        // ====================================================================
        // Cached primitives (the Qwen 3.5 RMSNormOpCached / LinearForwardCached
        // pair, minus the dictionary lookup and the GGML branches this path
        // can never take)
        // ====================================================================

        /// <summary>
        /// RMSNorm against a pre-resolved alpha and an explicit epsilon.
        /// Arithmetic is identical to <see cref="ModelBase.RMSNormOp"/> /
        /// <see cref="RMSNormWithEps"/>; only the name lookup is gone.
        /// </summary>
        private Tensor RMSNormCached(Tensor input, Tensor alpha, float eps)
        {
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);

            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;
            Tensor result = Ops.RMSNorm(null, src, alpha, null, eps);
            input2d?.Dispose();
            return result;
        }

        /// <summary>
        /// Linear against pre-resolved weight handles. The quantized branch goes
        /// straight to the MLX device matmul; the F32 branch keeps the generic
        /// Addmm so a mixed GGUF still works.
        /// </summary>
        private Tensor LinearCached(Tensor input, QuantizedWeight qw, Tensor wF32)
        {
            long t0 = Stopwatch.GetTimestamp();
            Tensor result;

            if (qw != null)
            {
                int rows = (int)input.Sizes[0];
                result = new Tensor(_allocator, DType.Float32, rows, (int)qw.Ne1);
                if (!MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                        result, input, qw.CacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                {
                    // Host row-dequant fallback (unsupported quant type). Costs
                    // a sync per call, which is why load-time reporting warns
                    // about fallback weights.
                    AddmmQuantManaged(result, input, qw);
                }
            }
            else
            {
                int rows = (int)input.Sizes[0];
                result = new Tensor(_allocator, DType.Float32, rows, (int)wF32.Sizes[0]);
                using var wT = wF32.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// [count] int32 positions, all equal to <paramref name="position"/> —
        /// what <see cref="ModelBase"/>'s RoPEEx wants for a single-row batch
        /// (one entry per (row, head) pair, and a decode step has one row).
        /// </summary>
        private Tensor CreateDecodeRopePositions(int count, int position)
        {
            int[] positions = new int[count];
            for (int i = 0; i < count; i++)
                positions[i] = position;
            return CreateIntTensorOn(_allocator, positions, count);
        }

        /// <summary>
        /// Single-row RoPE, identical to <see cref="ApplyRoPEPrefill"/> at
        /// seqLen == 1 (mode 0 == GGML_ROPE_TYPE_NORM, interleaved pairs), with
        /// the positions tensor hoisted out to the caller.
        /// </summary>
        private Tensor ApplyDecodeRoPE(Tensor data, int numHeads, int headDim, Tensor positions)
        {
            using Tensor reshaped = data.View(1, 1, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, positions, headDim, 0, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(1, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>
        /// Gather the token_embd row for a <c>[1]</c> int32 DEVICE token into
        /// <paramref name="outEmb"/>, with no host round trip. Returns false
        /// when the embedding table is not device-resident, so the caller can
        /// fall back to the host gather.
        /// </summary>
        private bool TryLookupNextEmbeddingOnDevice(Tensor outEmb, Tensor deviceTokenInt)
        {
            if (_backend != BackendType.Mlx || _mgTokenEmbdQW == null)
                return false;

            return MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outEmb,
                _mgTokenEmbdQW.CacheKey,
                _mgTokenEmbdQW.Data,
                _mgTokenEmbdQW.GgmlType,
                _mgTokenEmbdQW.Ne0,
                _mgTokenEmbdQW.Ne1,
                _mgTokenEmbdQW.RawBytes,
                deviceTokenInt);
        }

        /// <summary>Release the pipelined-decode carry-over state.</summary>
        public override void ResetPipelinedGreedyState()
        {
            _pipelineNextInputDevice?.Dispose();
            _pipelineNextInputDevice = null;
        }

        /// <summary>
        /// Diagnostics only: force the queued graph to complete and charge the
        /// wait to <paramref name="bucket"/>. No-op unless
        /// TS_MUSE_GLIMMER_MLX_STAGE_PROFILE=1.
        /// </summary>
        private void StageProfile(Tensor stageOutput, ref long bucket)
        {
            if (!MlxStageProfileEnabled || stageOutput == null)
                return;

            long t0 = Stopwatch.GetTimestamp();
            MlxFusedOps.TryEvaluate(stageOutput);
            bucket += Stopwatch.GetTimestamp() - t0;
        }

        // ====================================================================
        // Honest timing
        // ====================================================================

        /// <summary>
        /// MLX is a LAZY backend: an op returns as soon as its node is in the
        /// graph, so a per-op stopwatch measures graph BUILD time and whichever
        /// op happens to force the drain absorbs everything queued behind it.
        /// Reporting those numbers as "Norm: 70.9%" invites a whole afternoon
        /// of optimizing RMSNorm for a token that is 98% GPU wait.
        ///
        /// The labels below say what the numbers actually are. Use
        /// TS_MUSE_GLIMMER_MLX_STAGE_PROFILE=1 for a real per-stage GPU
        /// attribution.
        /// </summary>
        public override void PrintTimingStats()
        {
            if (_backend != BackendType.Mlx || _forwardCount == 0)
            {
                base.PrintTimingStats();
                return;
            }

            double totalMs = _forwardSw.Elapsed.TotalMilliseconds;
            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double linearMs = _linearTicks * msPerTick;
            double attnMs = _attnTicks * msPerTick;
            double normMs = _normTicks * msPerTick;
            double embMs = _embTicks * msPerTick;
            double lmHeadMs = _lmHeadTicks * msPerTick;
            double logitsCopyMs = _logitsCopyTicks * msPerTick;
            double submitMs = _mgPipelinedSubmitTicks * msPerTick;
            double otherMs = totalMs - linearMs - attnMs - normMs;

            Console.WriteLine($"Timing ({_forwardCount} forward calls, {totalMs:F0} ms total, {totalMs / _forwardCount:F0} ms/token):");
            Console.WriteLine("  (MLX is lazy: the buckets below are graph-build time plus whatever MLX");
            Console.WriteLine("   back-pressure or drain happened to land on that op, NOT that op's GPU cost.");
            Console.WriteLine("   Set TS_MUSE_GLIMMER_MLX_STAGE_PROFILE=1 for real per-stage GPU time.)");
            Console.WriteLine($"  Linear graph build: {linearMs:F0} ms ({100 * linearMs / totalMs:F1}%)");
            Console.WriteLine($"  Attention CPU/ops:  {attnMs:F0} ms ({100 * attnMs / totalMs:F1}%)");
            Console.WriteLine($"  Norm graph build:   {normMs:F0} ms ({100 * normMs / totalMs:F1}%)");
            Console.WriteLine($"  (LM head build:     {lmHeadMs:F0} ms, included in Linear)");
            Console.WriteLine($"  (Embedding:         {embMs:F0} ms, in Other)");
            Console.WriteLine($"  (Logits host copy:  {logitsCopyMs:F0} ms — 0 on the pipelined path, argmax runs on device)");
            Console.WriteLine($"  Other/GPU wait:     {otherMs:F0} ms ({100 * otherMs / totalMs:F1}%)");

            if (_mgPipelinedSteps > 0)
            {
                Console.WriteLine(
                    $"  Pipelined decode:   {_mgPipelinedSteps} steps, {submitMs:F0} ms in submit "
                    + $"({submitMs / _mgPipelinedSteps:F1} ms/step: build + MLX back-pressure, 1 worker round-trip each)");
            }

            if (MlxStageProfileEnabled)
            {
                Console.WriteLine("  Per-stage GPU time (TS_MUSE_GLIMMER_MLX_STAGE_PROFILE=1, blocking eval per stage):");
                Console.WriteLine($"    QKV+gate proj:    {_mgStageQkvTicks * msPerTick:F0} ms");
                Console.WriteLine($"    Attention:        {_mgStageAttnTicks * msPerTick:F0} ms");
                Console.WriteLine($"    o_proj:           {_mgStageOutProjTicks * msPerTick:F0} ms");
                Console.WriteLine($"    post-attn norm:   {_mgStageNormTicks * msPerTick:F0} ms");
                Console.WriteLine($"    FFN (gu/act/dn):  {_mgStageFfnTicks * msPerTick:F0} ms");
                Console.WriteLine($"    final norm+head:  {_mgStageHeadTicks * msPerTick:F0} ms");
            }
        }
    }
}
