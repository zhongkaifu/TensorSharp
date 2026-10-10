// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GLM-5.3-Flash (glm5next) in pure managed code.
//
// Until now this architecture ran through the native GGML executor ONLY - the
// constructor threw "there is no managed per-op fallback for the KDA/mHC layers
// yet" on any non-GGML backend, so it was the one supported family with no
// 100% C# path. Three things had to exist for that path:
//
//   1. KDA (Kimi Delta Attention) linear attention, on 34 of the 45 trunk
//      layers. It is the gated delta rule again, but with a per-CHANNEL decay
//      rather than Qwen's per-head one, its q/k/v pushed through one short
//      convolution, and a low-rank sigmoid output gate.
//   2. Sinkhorn hyper-connections (mHC) around EVERY residual crossing: the
//      hidden state is not one vector per token but `hc` parallel streams, and
//      each sublayer reads a learned mixture of them and writes back a learned
//      recombination. The mixing matrix is doubly-stochastic, produced by
//      Sinkhorn iteration.
//   3. The trunk's MLA layers are NoPE (no rotary at all) and their DSA indexer
//      scores POOLS of cells rather than cells.
//
// Ported from ggml_ops_glm_dsa.cpp (build_kda / build_hc_pre / build_hc_post)
// and ggml's own ggml_compute_forward_gated_delta_net and
// ggml_compute_forward_dsv4_hc_comb, which are the authorities for the two
// recurrences. Tag names in Trace() calls match the native tracer's so
// `TS_GLM_DEBUG=1` output can be diffed tag-by-tag against llama.cpp.
using System;
using System.Diagnostics;
using System.Threading.Tasks;

using TensorSharp;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        // ---- glm5next hyper-parameters -------------------------------------
        private bool _g5n;
        private int _kdaHeadDim;        // 128
        private int _kdaHeads;          // n_head on KDA layers
        private float _kdaGateLowerBound = -5.0f;
        private int _dConv;             // short conv kernel (4)
        private int _indexerKpool;      // cells per indexer pool (4)
        private int _hcMult;            // hyper-connection streams (4)
        private int _hcSinkhorn = 20;
        private float _hcEps = 1e-6f;
        private bool[] _layerIsRecurrent;

        // Per-layer KDA state, all managed:
        //   conv ring : [(dConv-1) * 3*dInner]
        //   ssm state : [heads * hd * hd], stored TRANSPOSED as ggml does -
        //               s[j*hd + i] == S[i][j] - so row j is contiguous.
        private float[][] _kdaConvState;
        private int[] _kdaConvWrite;
        private float[][] _kdaSsmState;

        private bool IsGlm5Next => _g5n;

        /// <summary>Read the glm5next-only metadata. Mirrors the native loader's
        /// g5n block; every key here is absent on GLM-5.2.</summary>
        private void ParseGlm5NextConfig(string arch, int numTrunkLayers)
        {
            _g5n = GlmDsaArchitecture.IsGlm5Next(arch);
            if (!_g5n)
                return;

            _kdaHeadDim = (int)_gguf.GetUint32($"{arch}.kda.head_dim");
            _dConv = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");
            _indexerKpool = (int)_gguf.GetUint32($"{arch}.attention.indexer.kpool");
            _hcMult = (int)_gguf.GetUint32($"{arch}.hyper_connection.count");
            _kdaGateLowerBound = _gguf.GetFloat32($"{arch}.kda.gate_lower_bound", -5.0f);
            _hcSinkhorn = (int)_gguf.GetUint32($"{arch}.hyper_connection.sinkhorn_iterations", 20);
            _hcEps = _gguf.GetFloat32($"{arch}.hyper_connection.epsilon", 1e-6f);
            _kdaHeads = Config.NumHeads;

            // Layer type comes from the per-layer head_count_kv array: 0 marks a
            // KDA layer, non-zero an MLA+DSA one. There is no separate flag.
            _layerIsRecurrent = new bool[numTrunkLayers];
            uint[] kvh = _gguf.GetUint32Array($"{arch}.attention.head_count_kv");
            if (kvh == null || kvh.Length == 0)
                throw new NotSupportedException(
                    "glm5next needs the per-layer attention.head_count_kv array to tell KDA layers from MLA ones.");
            for (int il = 0; il < numTrunkLayers && il < kvh.Length; il++)
                _layerIsRecurrent[il] = kvh[il] == 0;

            if (_kdaHeadDim <= 0 || _dConv <= 1 || _indexerKpool <= 0 || _hcMult <= 0)
                throw new NotSupportedException(
                    $"glm5next metadata is incomplete (kda.head_dim={_kdaHeadDim}, ssm.conv_kernel={_dConv}, " +
                    $"indexer.kpool={_indexerKpool}, hyper_connection.count={_hcMult}).");

            AllocateKdaState(numTrunkLayers);
        }

        private void AllocateKdaState(int numTrunkLayers)
        {
            int dInner = _kdaHeadDim * _kdaHeads;
            _kdaConvState = new float[numTrunkLayers][];
            _kdaConvWrite = new int[numTrunkLayers];
            _kdaSsmState = new float[numTrunkLayers][];
            for (int il = 0; il < numTrunkLayers; il++)
            {
                if (!_layerIsRecurrent[il]) continue;
                _kdaConvState[il] = new float[(_dConv - 1) * 3 * dInner];
                _kdaSsmState[il] = new float[(long)_kdaHeads * _kdaHeadDim * _kdaHeadDim];
            }
        }

        /// <summary>Zero the KDA recurrent state. Unlike an attention KV cache it
        /// cannot be rewound to an earlier position, which is why
        /// <see cref="SupportsKVCacheTruncation"/> is false for this family.</summary>
        private void ResetKdaState()
        {
            if (_kdaConvState == null) return;
            for (int il = 0; il < _kdaConvState.Length; il++)
            {
                if (_kdaConvState[il] != null) Array.Clear(_kdaConvState[il]);
                if (_kdaSsmState[il] != null) Array.Clear(_kdaSsmState[il]);
                _kdaConvWrite[il] = 0;
            }
        }

        // ====================================================================
        // The glm5next decoder block and forward
        // ====================================================================

        /// <summary>
        /// One glm5next trunk layer. Unlike GLM-5.2's plain pre-norm residual, the
        /// state here is `hc` parallel streams and BOTH halves are wrapped in a
        /// hyper-connection: the sublayer reads a learned mixture of the streams
        /// and its output is written back through a learned recombination.
        ///
        ///     residual = streams
        ///     cur      = hc_pre(streams)        -> [nt, n_embd]
        ///     cur      = rms(cur, attn_norm)
        ///     attn     = KDA(cur)  or  MLA+DSA(cur)
        ///     streams  = hc_post(attn, residual)
        ///     ... and the same again around the FFN.
        /// </summary>
        private Tensor Glm5NextDecoderBlock(Tensor streams, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerNames[layer];

            Tensor residual = streams;
            Tensor cur = Glm5NextHcPre(streams, layer, ffn: false, seqLen, out float[] post, out float[] comb);
            Tensor normed = RMSNormOp(cur, wn[WN_ATTN_NORM]);
            cur.Dispose();
            Trace("attn_norm", layer, normed);

            Tensor attnOut = _layerIsRecurrent[layer]
                ? KdaLayer(normed, layer, seqLen)
                : Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();
            Trace("attn_out", layer, attnOut);

            Tensor afterAttn = Glm5NextHcPost(attnOut, residual, post, comb, seqLen);
            attnOut.Dispose();
            residual.Dispose();
            Trace("ffn_inp", layer, afterAttn);

            residual = afterAttn;
            cur = Glm5NextHcPre(afterAttn, layer, ffn: true, seqLen, out post, out comb);
            Tensor normed2 = RMSNormOp(cur, wn[WN_FFN_NORM]);
            cur.Dispose();
            Trace("ffn_norm", layer, normed2);

            Tensor ffnOut = layer < _numDenseLead
                ? DenseFFN(normed2, wn, seqLen)
                : MoEBlock(normed2, layer, wn, seqLen);
            normed2.Dispose();
            Trace("ffn_out", layer, ffnOut);

            Tensor outStreams = Glm5NextHcPost(ffnOut, residual, post, comb, seqLen);
            ffnOut.Dispose();
            residual.Dispose();
            Trace("l_out", layer, outStreams);
            return outStreams;
        }

        /// <summary>
        /// glm5next's forward. Structurally the same as the GLM-5.2 one, with the
        /// trunk state carried as hyper-connection streams and an unweighted mean
        /// taken before the output norm.
        /// </summary>
        private float[] ForwardCoreGlm5Next(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor embd = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;
            Trace("inp_embd", -1, embd);

            Tensor streams = Glm5NextToStreams(embd, seqLen);
            embd.Dispose();

            _sharedTopKCount = 0;
            for (int layer = 0; layer < _numTrunkLayers; layer++)
                streams = Glm5NextDecoderBlock(streams, layer, seqLen, startPos);

            Tensor pooled = Glm5NextHcMean(streams, seqLen);
            streams.Dispose();

            Tensor normed = RMSNormOp(pooled, "output_norm.weight");
            pooled.Dispose();
            Trace("result_norm", -1, normed);

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight")
                                  ?? LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // ====================================================================
        // Sinkhorn hyper-connections
        // ====================================================================

        /// <summary>
        /// The per-token mixing matrices. `mixes` is [nt, (2+hc)*hc]; the first hc
        /// rows are the PRE weights (how the sublayer reads the streams), the next
        /// hc the POST weights (how its output is broadcast back), and the
        /// remaining hc*hc the raw combination logits that Sinkhorn turns into a
        /// doubly-stochastic stream-to-stream matrix.
        ///
        /// Layout note carried over from ggml: comb is indexed [idst + hc*isrc].
        /// </summary>
        private unsafe void HyperConnectMixes(
            float* mixes, float* hcScale, float* hcBase, int nt,
            float* preOut, float* postOut, float* combOut)
        {
            int hc = _hcMult;
            int mixDim = (2 + hc) * hc;
            int combOffset = 2 * hc;
            float scalePre = hcScale[0], scalePost = hcScale[1], scaleComb = hcScale[2];

            for (int t = 0; t < nt; t++)
            {
                float* m = mixes + (long)t * mixDim;
                float* pre = preOut + (long)t * hc;
                float* post = postOut + (long)t * hc;
                float* comb = combOut + (long)t * hc * hc;

                // pre  = sigmoid(x*scale + base) + eps
                // post = 2 * sigmoid(x*scale + base)
                for (int i = 0; i < hc; i++)
                {
                    pre[i] = Sigmoid(m[i] * scalePre + hcBase[i]) + _hcEps;
                    post[i] = 2.0f * Sigmoid(m[hc + i] * scalePost + hcBase[hc + i]);
                }

                // comb: softmax over dst within each src, +eps, then Sinkhorn.
                for (int isrc = 0; isrc < hc; isrc++)
                {
                    float max = float.NegativeInfinity;
                    for (int idst = 0; idst < hc; idst++)
                    {
                        int idx = idst + hc * isrc;
                        float v = m[combOffset + idx] * scaleComb + hcBase[combOffset + idx];
                        comb[idx] = v;
                        if (v > max) max = v;
                    }
                    float sum = 0f;
                    for (int idst = 0; idst < hc; idst++)
                    {
                        int idx = idst + hc * isrc;
                        float e = MathF.Exp(comb[idx] - max);
                        comb[idx] = e;
                        sum += e;
                    }
                    float inv = 1.0f / sum;
                    for (int idst = 0; idst < hc; idst++)
                        comb[idst + hc * isrc] = comb[idst + hc * isrc] * inv + _hcEps;
                }

                SinkhornNormalizeCols(comb, hc, _hcEps);
                for (int i = 1; i < _hcSinkhorn; i++)
                {
                    SinkhornNormalizeRows(comb, hc, _hcEps);
                    SinkhornNormalizeCols(comb, hc, _hcEps);
                }
            }
        }

        private static unsafe void SinkhornNormalizeCols(float* comb, int hc, float eps)
        {
            for (int idst = 0; idst < hc; idst++)
            {
                float sum = eps;
                for (int isrc = 0; isrc < hc; isrc++) sum += comb[idst + hc * isrc];
                float inv = 1.0f / sum;
                for (int isrc = 0; isrc < hc; isrc++) comb[idst + hc * isrc] *= inv;
            }
        }

        private static unsafe void SinkhornNormalizeRows(float* comb, int hc, float eps)
        {
            for (int isrc = 0; isrc < hc; isrc++)
            {
                float sum = eps;
                for (int idst = 0; idst < hc; idst++) sum += comb[idst + hc * isrc];
                float inv = 1.0f / sum;
                for (int idst = 0; idst < hc; idst++) comb[idst + hc * isrc] *= inv;
            }
        }

        private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

        /// <summary>
        /// Weightless RMS norm over each token's flattened stream vector, which is
        /// what ggml_rms_norm does before the mix projection (there is no gain
        /// tensor on this one).
        /// </summary>
        private unsafe Tensor HcFlatRmsNorm(Tensor flat, int nt, int dim)
        {
            var outT = new Tensor(_allocator, DType.Float32, nt, dim);
            float* src = GetFloatPtr(flat);
            float* dst = GetFloatPtr(outT);
            float eps = Config.Eps;
            Parallel.For(0, nt, t =>
            {
                float* s = src + (long)t * dim;
                float* o = dst + (long)t * dim;
                float sum = 0f;
                for (int i = 0; i < dim; i++) sum += s[i] * s[i];
                float inv = 1.0f / MathF.Sqrt(sum / dim + eps);
                for (int i = 0; i < dim; i++) o[i] = s[i] * inv;
            });
            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        /// <summary>
        /// Read the hyper-connection streams down to one vector per token, and
        /// produce the POST broadcast weights and the doubly-stochastic COMB
        /// matrix that the matching <see cref="Glm5NextHcPost"/> needs.
        ///
        /// streams is [nt, hc, n_embd]; the result is [nt, n_embd].
        /// </summary>
        private unsafe Tensor Glm5NextHcPre(
            Tensor streams, int il, bool ffn, int seqLen, out float[] post, out float[] comb)
        {
            int hc = _hcMult;
            int nEmbd = Config.HiddenSize;
            string p = $"blk.{il}." + (ffn ? "hc_ffn" : "hc_attn");

            using Tensor flat = streams.View(seqLen, hc * nEmbd);
            using Tensor normed = HcFlatRmsNorm(flat, seqLen, hc * nEmbd);
            using Tensor mixes = LinearForward(normed, $"{p}_fn.weight");

            var pre = new float[(long)seqLen * hc];
            post = new float[(long)seqLen * hc];
            comb = new float[(long)seqLen * hc * hc];

            var outT = new Tensor(_allocator, DType.Float32, seqLen, nEmbd);
            float* mixP = GetFloatPtr(mixes);
            float* scaleP = GetFloatPtr(_weights[$"{p}_scale.weight"]);
            float* baseP = GetFloatPtr(_weights[$"{p}_base.weight"]);
            float* streamP = GetFloatPtr(streams);
            float* outP = GetFloatPtr(outT);

            fixed (float* prePinned = pre, postPinned = post, combPinned = comb)
            {
                HyperConnectMixes(mixP, scaleP, baseP, seqLen, prePinned, postPinned, combPinned);

                float* preLocal = prePinned;
                Parallel.For(0, seqLen, t =>
                {
                    float* o = outP + (long)t * nEmbd;
                    float* w = preLocal + (long)t * hc;
                    float* x = streamP + (long)t * hc * nEmbd;
                    for (int e = 0; e < nEmbd; e++) o[e] = 0f;
                    for (int c = 0; c < hc; c++)
                    {
                        float wc = w[c];
                        float* xc = x + (long)c * nEmbd;
                        for (int e = 0; e < nEmbd; e++) o[e] += xc[e] * wc;
                    }
                });
            }

            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        /// <summary>
        /// Write a sublayer's output back into the streams: a rank-1 broadcast of
        /// that output weighted by POST, plus the previous streams recombined
        /// through the doubly-stochastic COMB matrix.
        ///
        /// x is [nt, n_embd]; residual and the result are [nt, hc, n_embd].
        /// </summary>
        private unsafe Tensor Glm5NextHcPost(
            Tensor x, Tensor residual, float[] post, float[] comb, int seqLen)
        {
            int hc = _hcMult;
            int nEmbd = Config.HiddenSize;
            var outT = new Tensor(_allocator, DType.Float32, seqLen, hc, nEmbd);

            float* xP = GetFloatPtr(x);
            float* rP = GetFloatPtr(residual);
            float* oP = GetFloatPtr(outT);

            fixed (float* postPinned = post, combPinned = comb)
            {
                float* postLocal = postPinned;
                float* combLocal = combPinned;
                Parallel.For(0, seqLen, t =>
                {
                    float* xt = xP + (long)t * nEmbd;
                    float* rt = rP + (long)t * hc * nEmbd;
                    float* ot = oP + (long)t * hc * nEmbd;
                    float* pt = postLocal + (long)t * hc;
                    float* ct = combLocal + (long)t * hc * hc;

                    for (int cdst = 0; cdst < hc; cdst++)
                    {
                        float* o = ot + (long)cdst * nEmbd;
                        float pw = pt[cdst];
                        for (int e = 0; e < nEmbd; e++) o[e] = xt[e] * pw;
                        for (int csrc = 0; csrc < hc; csrc++)
                        {
                            // comb is indexed [idst + hc*isrc], as in ggml.
                            float w = ct[cdst + hc * csrc];
                            float* r = rt + (long)csrc * nEmbd;
                            for (int e = 0; e < nEmbd; e++) o[e] += r[e] * w;
                        }
                    }
                });
            }

            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        /// <summary>Replicate the token embedding across the hyper-connection
        /// streams: from the very first layer the trunk state is [nt, hc, n_embd],
        /// not [nt, n_embd].</summary>
        private unsafe Tensor Glm5NextToStreams(Tensor embd, int seqLen)
        {
            int hc = _hcMult;
            int nEmbd = Config.HiddenSize;
            var outT = new Tensor(_allocator, DType.Float32, seqLen, hc, nEmbd);
            float* src = GetFloatPtr(embd);
            float* dst = GetFloatPtr(outT);
            for (int t = 0; t < seqLen; t++)
                for (int c = 0; c < hc; c++)
                    Buffer.MemoryCopy(src + (long)t * nEmbd, dst + ((long)t * hc + c) * nEmbd,
                        nEmbd * 4L, nEmbd * 4L);
            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        /// <summary>The trunk head is the UNWEIGHTED mean over streams - glm5next
        /// does not use DeepSeek V4's learned gated readout.</summary>
        private unsafe Tensor Glm5NextHcMean(Tensor streams, int seqLen)
        {
            int hc = _hcMult;
            int nEmbd = Config.HiddenSize;
            var outT = new Tensor(_allocator, DType.Float32, seqLen, nEmbd);
            float* src = GetFloatPtr(streams);
            float* dst = GetFloatPtr(outT);
            float inv = 1.0f / hc;
            Parallel.For(0, seqLen, t =>
            {
                float* o = dst + (long)t * nEmbd;
                float* x = src + (long)t * hc * nEmbd;
                for (int e = 0; e < nEmbd; e++) o[e] = 0f;
                for (int c = 0; c < hc; c++)
                {
                    float* xc = x + (long)c * nEmbd;
                    for (int e = 0; e < nEmbd; e++) o[e] += xc[e];
                }
                for (int e = 0; e < nEmbd; e++) o[e] *= inv;
            });
            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        // ====================================================================
        // KDA linear attention
        // ====================================================================

        /// <summary>
        /// One KDA layer. `cur` is [nt, n_embd] (already hc-mixed and RMS-normed by
        /// the caller); returns [nt, n_embd].
        ///
        /// Differences from Qwen's Gated DeltaNet that matter, all of them silent
        /// if got wrong:
        ///   * ONE convolution over concatenated q|k|v, with SiLU applied to the
        ///     conv OUTPUT (not to the projections).
        ///   * the decay gate is per CHANNEL, so the state is scaled column-wise
        ///     rather than by a scalar.
        ///   * q is NOT pre-scaled; the 1/sqrt(head_dim) factor lands on the
        ///     attention output, matching ggml_gated_delta_net.
        ///   * the output gate is a low-rank projection through sigmoid, and the
        ///     RMS norm weight is shared by every head.
        /// </summary>
        private unsafe Tensor KdaLayer(Tensor cur, int il, int seqLen)
        {
            int hd = _kdaHeadDim;
            int H = _kdaHeads;
            int dInner = hd * H;
            int dc = _dConv;
            string p = $"blk.{il}";

            // Projections: [nt, dInner] each.
            Tensor qp = LinearForward(cur, $"{p}.attn_q.weight");
            Tensor kp = LinearForward(cur, $"{p}.attn_k.weight");
            Tensor vp = LinearForward(cur, $"{p}.attn_v.weight");

            // Low-rank forget gate and the per-head beta, both read the LAYER
            // INPUT, not the convolved q/k/v.
            Tensor fa = LinearForward(cur, $"{p}.ssm_f_a.weight");
            Tensor fRaw = LinearForward(fa, $"{p}.ssm_f_b.weight");   // [nt, dInner]
            fa.Dispose();
            Tensor ga = LinearForward(cur, $"{p}.ssm_g_a.weight");
            Tensor gRaw = LinearForward(ga, $"{p}.ssm_g_b.weight");   // [nt, dInner]
            ga.Dispose();
            Tensor betaRaw = LinearForward(cur, $"{p}.ssm_beta.weight"); // [nt, H]

            var outT = new Tensor(_allocator, DType.Float32, seqLen, dInner);

            float* qP = GetFloatPtr(qp);
            float* kP = GetFloatPtr(kp);
            float* vP = GetFloatPtr(vp);
            float* fP = GetFloatPtr(fRaw);
            float* gP = GetFloatPtr(gRaw);
            float* bP = GetFloatPtr(betaRaw);
            float* outP = GetFloatPtr(outT);

            float* convQ = GetFloatPtr(_weights[$"{p}.ssm_conv1d_q.weight"]);
            float* convK = GetFloatPtr(_weights[$"{p}.ssm_conv1d_k.weight"]);
            float* convV = GetFloatPtr(_weights[$"{p}.ssm_conv1d_v.weight"]);
            float* dtB = GetFloatPtr(_weights[$"{p}.ssm_dt.bias"]);
            float* aVec = GetFloatPtr(_weights[$"{p}.ssm_a"]);
            float* oNorm = GetFloatPtr(_weights[$"{p}.ssm_norm.weight"]);

            float[] convState = _kdaConvState[il];
            float[] ssm = _kdaSsmState[il];

            float outScale = 1.0f / MathF.Sqrt(hd);
            float eps = Config.Eps;

            var qkv = new float[3 * dInner];
            var conv = new float[3 * dInner];

            fixed (float* convStateP = convState, ssmPinned = ssm, qkvP = qkv, convP = conv)
            {
                // A `fixed` local cannot be captured by a lambda, so take a plain
                // copy for the per-head parallel body.
                float* ssmP = ssmPinned;
                for (int t = 0; t < seqLen; t++)
                {
                    // Pack this token's q|k|v the way the stacked conv kernel expects.
                    Buffer.MemoryCopy(qP + (long)t * dInner, qkvP, dInner * 4L, dInner * 4L);
                    Buffer.MemoryCopy(kP + (long)t * dInner, qkvP + dInner, dInner * 4L, dInner * 4L);
                    Buffer.MemoryCopy(vP + (long)t * dInner, qkvP + 2L * dInner, dInner * 4L, dInner * 4L);

                    KdaConvStep(qkvP, convP, convStateP, convQ, convK, convV,
                        dInner, dc, ref _kdaConvWrite[il]);

                    float* qc = convP;
                    float* kc = convP + dInner;
                    float* vc = convP + 2L * dInner;

                    // L2 norm per head. 1e-6 is the reference's own constant here,
                    // deliberately NOT the model's rms eps.
                    L2NormPerHead(qc, H, hd, 1e-6f);
                    L2NormPerHead(kc, H, hd, 1e-6f);

                    float* fRow = fP + (long)t * dInner;
                    float* gRow = gP + (long)t * dInner;
                    float* bRow = bP + (long)t * H;
                    float* oRow = outP + (long)t * dInner;

                    Parallel.For(0, H, h =>
                    {
                        float* s = ssmP + (long)h * hd * hd;
                        float* qh = qc + (long)h * hd;
                        float* kh = kc + (long)h * hd;
                        float* vh = vc + (long)h * hd;
                        float* fh = fRow + (long)h * hd;
                        float* gh = gRow + (long)h * hd;
                        float* oh = oRow + (long)h * hd;

                        float beta = Sigmoid(bRow[h]);
                        float aH = aVec[h];

                        // Per-channel decay. ssm_a holds -exp(A_log), so
                        // exp(A_log)*y == -(y * ssm_a); the gate is then
                        // gate_lb * sigmoid(that), and the recurrence uses exp(gate).
                        //
                        // The state is stored transposed (row j == column j of S),
                        // so scaling "column i of every row" is an elementwise
                        // multiply of each ROW by the decay vector - which is both
                        // what ggml does and the only cache-friendly direction.
                        float* decay = stackalloc float[hd];
                        for (int i = 0; i < hd; i++)
                        {
                            float y = (fh[i] + dtB[(long)h * hd + i]) * aH;
                            decay[i] = MathF.Exp(_kdaGateLowerBound * Sigmoid(-y));
                        }
                        for (int j = 0; j < hd; j++)
                        {
                            float* row = s + (long)j * hd;
                            for (int i = 0; i < hd; i++) row[i] *= decay[i];
                        }

                        // delta[j] = (v[j] - dot(row j, k)) * beta
                        // row j += k * delta[j];  out[j] = dot(row j, q) * scale
                        for (int j = 0; j < hd; j++)
                        {
                            float* row = s + (long)j * hd;
                            float dot = VecDot(row, kh, hd);
                            float delta = (vh[j] - dot) * beta;
                            VecScaleAdd(row, kh, delta, hd);
                            oh[j] = VecDot(row, qh, hd) * outScale;
                        }

                        // Shared-weight RMS norm over the head, then a plain
                        // sigmoid gate (not SiLU).
                        float rms = 1.0f / MathF.Sqrt((VecSumSq(oh, hd) / hd) + eps);
                        for (int i = 0; i < hd; i++)
                            oh[i] = oh[i] * rms * oNorm[i] * Sigmoid(gh[i]);
                    });
                }
            }

            qp.Dispose(); kp.Dispose(); vp.Dispose();
            fRaw.Dispose(); gRaw.Dispose(); betaRaw.Dispose();

            InvalidateTensorDeviceCache(outT);
            Tensor proj = LinearForward(outT, $"{p}.attn_output.weight");
            outT.Dispose();
            return proj;
        }

        /// <summary>
        /// One step of the stacked short convolution over [q|k|v], followed by
        /// SiLU. The three kernels are stored separately in the file and stacked
        /// here; the ring holds the previous dc-1 inputs.
        /// </summary>
        private static unsafe void KdaConvStep(
            float* qkv, float* outBuf, float* state,
            float* convQ, float* convK, float* convV,
            int dInner, int dc, ref int writeIdx)
        {
            int convDim = dc - 1;
            int total = 3 * dInner;

            for (int c = 0; c < total; c++) outBuf[c] = 0f;

            // Taps 0..dc-2 read the ring, oldest first.
            for (int ki = 0; ki < convDim; ki++)
            {
                int slot = (writeIdx + ki) % convDim;
                float* sp = state + (long)slot * total;
                for (int c = 0; c < total; c++)
                {
                    float w = ConvWeight(convQ, convK, convV, dInner, dc, c, ki);
                    outBuf[c] += sp[c] * w;
                }
            }
            // Last tap is the current token.
            for (int c = 0; c < total; c++)
            {
                float w = ConvWeight(convQ, convK, convV, dInner, dc, c, convDim);
                outBuf[c] += qkv[c] * w;
            }

            // SiLU on the conv output.
            for (int c = 0; c < total; c++)
                outBuf[c] = outBuf[c] / (1.0f + MathF.Exp(-outBuf[c]));

            if (convDim > 0)
            {
                Buffer.MemoryCopy(qkv, state + (long)writeIdx * total, total * 4L, total * 4L);
                writeIdx = (writeIdx + 1) % convDim;
            }
        }

        /// <summary>Kernel tap for channel c. The GGUF stores each of the three
        /// convolutions as [dc, d_inner], and the stacked layout puts q first,
        /// then k, then v.</summary>
        private static unsafe float ConvWeight(
            float* convQ, float* convK, float* convV, int dInner, int dc, int channel, int tap)
        {
            int which = channel / dInner;
            int ch = channel - which * dInner;
            float* w = which == 0 ? convQ : which == 1 ? convK : convV;
            return w[(long)ch * dc + tap];
        }

        private static unsafe void L2NormPerHead(float* x, int heads, int dim, float eps)
        {
            for (int h = 0; h < heads; h++)
            {
                float* v = x + (long)h * dim;
                float sum = 0f;
                for (int i = 0; i < dim; i++) sum += v[i] * v[i];
                float inv = 1.0f / MathF.Sqrt(sum + eps);
                for (int i = 0; i < dim; i++) v[i] *= inv;
            }
        }
    }
}
