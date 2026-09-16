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
using TensorSharp.Core;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // Transposed conv kernels, [tap][channel], built once per layer: the GGUF
        // stores [kernel, channels] and the step wants a contiguous row per tap.
        private float[][] _gdnConvWT;
        private int[] _gdnConvWriteIdx;

        // Scratch, sized on first use.
        private float[] _gdnConvOut, _gdnQ, _gdnK, _gdnV, _gdnQx, _gdnKx, _gdnDelta, _gdnCore;


        /// <summary>
        /// Gated DeltaNet, the same recurrence Qwen 3.5 runs, with two differences
        /// that matter: the input arrives already normed (the hyper-connection mixer
        /// did it) and the output gate is a SIGMOID rather than a SiLU.
        ///
        /// Deliberately a self-contained per-token loop rather than a call into
        /// Qwen35Model's chunked path: that one is welded to its own state arrays and
        /// weight caches, and correctness here is worth more than sharing the fused
        /// kernel, which this can adopt later.
        /// </summary>
        private unsafe Tensor GdnLayer(Tensor cur, int il, int seqLen)
        {
            int valueDim = _headVDim * _numVHeads;

            EnsureGdnScratch();
            EnsureGdnConvWeights(il);

            long tp0 = Stopwatch.GetTimestamp();
            Tensor qkv = LinearForward(cur, $"blk.{il}.attn_qkv.weight");     // [T, convDim]
            Tensor z = LinearForward(cur, $"blk.{il}.attn_gate.weight");      // [T, valueDim]
            Tensor betaT = LinearForward(cur, $"blk.{il}.ssm_beta.weight");   // [T, numVHeads]
            Tensor alphaT = LinearForward(cur, $"blk.{il}.ssm_alpha.weight"); // [T, numVHeads]
            Q4eGdnProjTicks += Stopwatch.GetTimestamp() - tp0;

            var gated = new Tensor(_allocator, DType.Float32, seqLen, valueDim);

            // The conv, the q/k norm+tiling and the alpha/beta gate arithmetic stay on
            // the host - the fused kernel wants them pre-computed - but the recurrence
            // itself, which was 41% of a decode token as a C# loop, goes to the GPU.
            EnsureChunkedStaging(seqLen);
            long tp1 = Stopwatch.GetTimestamp();
            PrepareGdnInputs(qkv, z, betaT, alphaT, il, seqLen);
            Q4eGdnPrepTicks += Stopwatch.GetTimestamp() - tp1;

            using Tensor qv = _gdnQBuf.Narrow(0, 0, seqLen);
            using Tensor kv = _gdnKBuf.Narrow(0, 0, seqLen);
            using Tensor vv = _gdnVBuf.Narrow(0, 0, seqLen);
            using Tensor zv = _gdnZBuf.Narrow(0, 0, seqLen);
            using Tensor av = _gdnAlphaBuf.Narrow(0, 0, seqLen);
            using Tensor bv = _gdnBetaBuf.Narrow(0, 0, seqLen);

            // The kernel reads [T, H, D]; `gated` is the same memory laid out [T, H*D].
            using Tensor gated3 = gated.View(seqLen, _numVHeads, _headVDim);
            long tp2 = Stopwatch.GetTimestamp();
            if (!IsGgmlBackend)
            {
                // Pure-C# backend: there is no native kernel to call. Without this
                // qwen4exp aborted on its first GDN layer with "Native GGML
                // gated_delta_net_chunked failed", so the architecture had no
                // managed backend at all.
                GatedDeltaNetRecurrenceManaged(il, seqLen, gated3);
                Q4eGdnKernelTicks += Stopwatch.GetTimestamp() - tp2;
                qkv.Dispose(); z.Dispose(); betaT.Dispose(); alphaT.Dispose();
                InvalidateTensorDeviceCache(gated);
                long tp3m = Stopwatch.GetTimestamp();
                Tensor outProjM = LinearForward(gated, $"blk.{il}.ssm_out.weight");
                Q4eGdnOutTicks += Stopwatch.GetTimestamp() - tp3m;
                gated.Dispose();
                return outProjM;
            }
            GgmlBasicOps.GatedDeltaNetChunked(
                qv, kv, vv, zv, av, bv, _gdnStateT[il], gated3,
                (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_dt.bias"]),
                (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_a"]),
                (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_norm.weight"]),
                // Fixed 64. Sizing the chunk to the batch looked appealing - a 1-token
                // decode is padded into a 64-wide chunk - but it measured within noise
                // and an arbitrary chunk (seqLen 58) aborts the kernel, which assumes
                // the tuned width.
                GdnChunkSize, Config.Eps,
                // qwen4exp gates the delta-net output with a SIGMOID where Qwen 3.5
                // uses SiLU. Same kernel, one flag.
                gateMode: 1);
            Q4eGdnKernelTicks += Stopwatch.GetTimestamp() - tp2;

            qkv.Dispose(); z.Dispose(); betaT.Dispose(); alphaT.Dispose();
            InvalidateTensorDeviceCache(gated);

            long tp3 = Stopwatch.GetTimestamp();
            Tensor outProj = LinearForward(gated, $"blk.{il}.ssm_out.weight");
            Q4eGdnOutTicks += Stopwatch.GetTimestamp() - tp3;
            gated.Dispose();
            return outProj;
        }

        /// <summary>
        /// The gated delta rule, in managed code, for backends with no native kernel.
        ///
        /// PrepareGdnInputs has already done the parts the native kernel documents as
        /// host-side AND the parts it does itself: conv1d + SiLU, the per-head L2 norm
        /// of Q/K, the K-head tiling, the 1/sqrt(D) scale on Q, the alpha gate
        /// (softplus(alpha + dt_bias) * ssm_a, with ssm_a pre-negated so it decays) and
        /// the sigmoid on beta. So all that is left is the recurrence:
        ///
        ///     S      *= exp(gate)
        ///     delta   = (v - S k) * beta
        ///     S      += delta k^T
        ///     core    = S q
        ///     out     = rmsnorm(core) * ssm_norm_w * sigmoid(z)
        ///
        /// qwen4exp gates with SIGMOID where Qwen 3.5 uses SiLU - the same one-flag
        /// difference the native kernel takes as gate_mode.
        ///
        /// Heads are independent, so they run in parallel; the sequence is not, so t
        /// stays serial.
        /// </summary>
        private unsafe void GatedDeltaNetRecurrenceManaged(int il, int seqLen, Tensor gatedOut)
        {
            int headK = _headKDim;
            int headV = _headVDim;
            int heads = _numVHeads;
            int statePerHead = headV * headK;
            float eps = Config.Eps;

            float* q = GetFloatPtr(_gdnQBuf);
            float* k = GetFloatPtr(_gdnKBuf);
            float* v = GetFloatPtr(_gdnVBuf);
            float* z = GetFloatPtr(_gdnZBuf);
            float* a = GetFloatPtr(_gdnAlphaBuf);
            float* b = GetFloatPtr(_gdnBetaBuf);
            float* state = GetFloatPtr(_gdnStateT[il]);
            float* ssmNorm = GetFloatPtr(_weights[$"blk.{il}.ssm_norm.weight"]);
            float* outP = GetFloatPtr(gatedOut);

            // One delta/core scratch row per head so the parallel loop never shares.
            EnsureGdnRecurrenceScratch(heads * headV);
            fixed (float* deltaAll = _gdnDelta, coreAll = _gdnCore)
            {
                float* deltaBase = deltaAll;
                float* coreBase = coreAll;
                for (int t = 0; t < seqLen; t++)
                {
                    float* qT = q + (long)t * heads * headK;
                    float* kT = k + (long)t * heads * headK;
                    float* vT = v + (long)t * heads * headV;
                    float* zT = z + (long)t * heads * headV;
                    float* aT = a + (long)t * heads;
                    float* bT = b + (long)t * heads;
                    float* oT = outP + (long)t * heads * headV;

                    System.Threading.Tasks.Parallel.For(0, heads, h =>
                    {
                        float* s = state + (long)h * statePerHead;
                        float* qh = qT + (long)h * headK;
                        float* kh = kT + (long)h * headK;
                        float* vh = vT + (long)h * headV;
                        float* zh = zT + (long)h * headV;
                        float* oh = oT + (long)h * headV;
                        float* delta = deltaBase + (long)h * headV;
                        float* core = coreBase + (long)h * headV;

                        VecScale(s, MathF.Exp(aT[h]), statePerHead);
                        float beta = bT[h];

                        for (int row = 0; row < headV; row++)
                            delta[row] = (vh[row] - VecDot(s + (long)row * headK, kh, headK)) * beta;

                        for (int row = 0; row < headV; row++)
                        {
                            float* sRow = s + (long)row * headK;
                            VecScaleAdd(sRow, kh, delta[row], headK);
                            core[row] = VecDot(sRow, qh, headK);
                        }

                        float rmsInv = 1.0f / MathF.Sqrt((VecSumSq(core, headV) / headV) + eps);
                        for (int i = 0; i < headV; i++)
                        {
                            float gate = 1.0f / (1.0f + MathF.Exp(-zh[i]));   // sigmoid, not SiLU
                            oh[i] = core[i] * rmsInv * ssmNorm[i] * gate;
                        }
                    });
                }
            }

            InvalidateTensorDeviceCache(_gdnStateT[il]);
        }

        private void EnsureGdnRecurrenceScratch(int needed)
        {
            if (_gdnDelta == null || _gdnDelta.Length < needed)
            {
                _gdnDelta = new float[needed];
                _gdnCore = new float[needed];
            }
        }

        private const int GdnChunkSize = 64;

        // Staging for the fused kernel: [T, H, D] q/k/v/z and [T, H] alpha/beta, grown
        // to the largest sequence seen so far and sub-viewed per call.
        private Tensor _gdnQBuf, _gdnKBuf, _gdnVBuf, _gdnZBuf, _gdnAlphaBuf, _gdnBetaBuf;
        private Tensor[] _gdnStateT;
        private int _gdnStagingRows;

        private void EnsureChunkedStaging(int seqLen)
        {
            if (_gdnStagingRows >= seqLen && _gdnQBuf != null) return;
            _gdnQBuf?.Dispose(); _gdnKBuf?.Dispose(); _gdnVBuf?.Dispose();
            _gdnZBuf?.Dispose(); _gdnAlphaBuf?.Dispose(); _gdnBetaBuf?.Dispose();
            int rows = Math.Max(seqLen, 1);
            _gdnQBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads, _headKDim);
            _gdnKBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads, _headKDim);
            _gdnVBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads, _headVDim);
            _gdnZBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads, _headVDim);
            _gdnAlphaBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads);
            _gdnBetaBuf = new Tensor(_allocator, DType.Float32, rows, _numVHeads);
            _gdnStagingRows = rows;
        }

        /// <summary>
        /// Host-side front of the recurrence: the causal depthwise conv over the ring
        /// history, the q/k L2 norm and head tiling, and the alpha/beta gate arithmetic
        /// the fused kernel expects pre-computed.
        /// </summary>
        private unsafe void PrepareGdnInputs(Tensor qkv, Tensor z, Tensor betaT, Tensor alphaT,
            int il, int seqLen)
        {
            int keyDim = _headKDim * _numKHeads;
            int valueDim = _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            float eps = Config.Eps;
            float qScale = 1.0f / MathF.Sqrt(_headVDim);

            float* qkvP = GetFloatPtr(qkv);
            float* zP = GetFloatPtr(z);
            float* betaP = GetFloatPtr(betaT);
            float* alphaP = GetFloatPtr(alphaT);
            float* dtBias = GetFloatPtr(_weights[$"blk.{il}.ssm_dt.bias"]);
            float* aLog = GetFloatPtr(_weights[$"blk.{il}.ssm_a"]);

            float* qOut = GetFloatPtr(_gdnQBuf);
            float* kOut = GetFloatPtr(_gdnKBuf);
            float* vOut = GetFloatPtr(_gdnVBuf);
            float* zOut = GetFloatPtr(_gdnZBuf);
            float* aOut = GetFloatPtr(_gdnAlphaBuf);
            float* bOut = GetFloatPtr(_gdnBetaBuf);

            float[] convState = _gdnConvState[il];
            float[] convWT = _gdnConvWT[il];

            fixed (float* convOut = _gdnConvOut, qBuf = _gdnQ, kBuf = _gdnK, vBuf = _gdnV,
                          convStatePtr = convState, convWPtr = convWT)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    float* xin = qkvP + (long)t * _convDim;
                    int writeIdx = _gdnConvWriteIdx[il];

                    for (int c = 0; c < _convDim; c++) convOut[c] = 0f;
                    for (int ki = 0; ki < convDim; ki++)
                    {
                        int slot = (writeIdx + ki) % convDim;
                        float* sp = convStatePtr + (long)slot * _convDim;
                        float* wp = convWPtr + (long)ki * _convDim;
                        for (int c = 0; c < _convDim; c++) convOut[c] += sp[c] * wp[c];
                    }
                    {
                        float* wp = convWPtr + (long)convDim * _convDim;
                        for (int c = 0; c < _convDim; c++) convOut[c] += xin[c] * wp[c];
                    }
                    for (int c = 0; c < _convDim; c++)
                        convOut[c] = convOut[c] / (1.0f + MathF.Exp(-convOut[c]));

                    if (convDim > 0)
                    {
                        Buffer.MemoryCopy(xin, convStatePtr + (long)writeIdx * _convDim,
                            _convDim * 4L, _convDim * 4L);
                        _gdnConvWriteIdx[il] = (writeIdx + 1) % convDim;
                    }

                    Buffer.MemoryCopy(convOut, qBuf, keyDim * 4L, keyDim * 4L);
                    Buffer.MemoryCopy(convOut + keyDim, kBuf, keyDim * 4L, keyDim * 4L);
                    Buffer.MemoryCopy(convOut + 2 * keyDim, vBuf, valueDim * 4L, valueDim * 4L);

                    // L2 norm per K head, then TILE the K heads across the V heads:
                    // ggml_repeat tiles rather than interleaves, so head h reads
                    // h % num_k_heads. Getting this backwards is silent nonsense.
                    L2NormHeads(qBuf, _numKHeads, _headKDim, eps);
                    L2NormHeads(kBuf, _numKHeads, _headKDim, eps);

                    float* qRow = qOut + (long)t * _numVHeads * _headKDim;
                    float* kRow = kOut + (long)t * _numVHeads * _headKDim;
                    for (int h = 0; h < _numVHeads; h++)
                    {
                        int src = (h % _numKHeads) * _headKDim;
                        float* qd = qRow + (long)h * _headKDim;
                        Buffer.MemoryCopy(qBuf + src, qd, _headKDim * 4L, _headKDim * 4L);
                        Buffer.MemoryCopy(kBuf + src, kRow + (long)h * _headKDim,
                            _headKDim * 4L, _headKDim * 4L);
                        for (int i = 0; i < _headKDim; i++) qd[i] *= qScale;
                    }

                    Buffer.MemoryCopy(vBuf, vOut + (long)t * valueDim, valueDim * 4L, valueDim * 4L);
                    Buffer.MemoryCopy(zP + (long)t * valueDim, zOut + (long)t * valueDim,
                        valueDim * 4L, valueDim * 4L);

                    float* aRow = aOut + (long)t * _numVHeads;
                    float* bRow = bOut + (long)t * _numVHeads;
                    float* alphaRow = alphaP + (long)t * _numVHeads;
                    float* betaRow = betaP + (long)t * _numVHeads;
                    for (int h = 0; h < _numVHeads; h++)
                    {
                        // ssm_a ships pre-negated, so this decays.
                        aRow[h] = Softplus(alphaRow[h] + dtBias[h]) * aLog[h];
                        bRow[h] = 1.0f / (1.0f + MathF.Exp(-betaRow[h]));
                    }
                }
            }

            InvalidateTensorDeviceCache(_gdnQBuf);
            InvalidateTensorDeviceCache(_gdnKBuf);
            InvalidateTensorDeviceCache(_gdnVBuf);
            InvalidateTensorDeviceCache(_gdnZBuf);
            InvalidateTensorDeviceCache(_gdnAlphaBuf);
            InvalidateTensorDeviceCache(_gdnBetaBuf);
        }

        private static float Softplus(float x)
            => x > 20f ? x : MathF.Log(1.0f + MathF.Exp(x));

        private static unsafe void L2NormHeads(float* x, int heads, int dim, float eps)
        {
            for (int h = 0; h < heads; h++)
            {
                float* p = x + (long)h * dim;
                double ss = 0;
                for (int i = 0; i < dim; i++) ss += (double)p[i] * p[i];
                float inv = (float)(1.0 / Math.Sqrt(ss + eps));
                for (int i = 0; i < dim; i++) p[i] *= inv;
            }
        }

        private void EnsureGdnScratch()
        {
            // The fused kernel updates the recurrent state in place, so it lives in a
            // Tensor rather than the float[] the per-token loop used. A holder
            // swap may have installed pre-seeded tensors whose addresses key the
            // native state entries - reuse them.
            _gdnStateT ??= new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                if (_gdnStateT[l] != null) continue;
                _gdnStateT[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                Ops.Fill(_gdnStateT[l], 0f);
            }
            // These belong to the active holder. Shared scratch may already exist
            // when restoring the primary holder before its first forward.
            _gdnConvWriteIdx ??= new int[Config.NumLayers];
            if (_gdnConvOut != null) return;
            int keyDim = _headKDim * _numKHeads;
            int valueDim = _headVDim * _numVHeads;
            _gdnConvOut = new float[_convDim];
            _gdnQ = new float[keyDim];
            _gdnK = new float[keyDim];
            _gdnV = new float[valueDim];
            _gdnQx = new float[(long)_numVHeads * _headKDim];
            _gdnKx = new float[(long)_numVHeads * _headKDim];
            _gdnDelta = new float[_headVDim];
            _gdnCore = new float[_headVDim];
        }

        private unsafe void EnsureGdnConvWeights(int il)
        {
            _gdnConvWT ??= new float[Config.NumLayers][];
            if (_gdnConvWT[il] != null) return;

            Tensor w = _weights[$"blk.{il}.ssm_conv1d.weight"];
            var wt = new float[(long)_convKernel * _convDim];
            float* src = GetFloatPtr(w);
            // GGUF ships [kernel, channels]; the step wants one contiguous row per tap.
            for (int c = 0; c < _convDim; c++)
                for (int k = 0; k < _convKernel; k++)
                    wt[(long)k * _convDim + c] = src[(long)c * _convKernel + k];
            _gdnConvWT[il] = wt;
        }

        /// <summary>
        /// Full attention: a joint Q|gate projection (per head, interleaved), Q/K RMS
        /// norm, partial IMRoPE, then a sigmoid gate on the attention output.
        ///
        /// This per-operation fallback is for dense configurations. QSA runs in
        /// the token span, which also owns its persistent raw indexer cache.
        /// </summary>
        private unsafe Tensor AttentionLayer(Tensor cur, int il, int seqLen, int startPos)
        {
            int nHead = Config.NumHeads;
            int nKvHead = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int totalLen = startPos + seqLen;

            long ta0 = Stopwatch.GetTimestamp();
            Tensor qgFull = LinearForward(cur, $"blk.{il}.attn_q.weight");  // [T, nHead*2*headDim]
            var q = new Tensor(_allocator, DType.Float32, seqLen, nHead * headDim);
            var gate = new Tensor(_allocator, DType.Float32, seqLen, nHead * headDim);
            {
                float* src = GetFloatPtr(qgFull);
                float* qp = GetFloatPtr(q);
                float* gp = GetFloatPtr(gate);
                int stride = 2 * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    float* s = src + (long)t * nHead * stride;
                    float* qd = qp + (long)t * nHead * headDim;
                    float* gd = gp + (long)t * nHead * headDim;
                    for (int h = 0; h < nHead; h++)
                    {
                        Buffer.MemoryCopy(s + (long)h * stride, qd + (long)h * headDim, headDim * 4L, headDim * 4L);
                        Buffer.MemoryCopy(s + (long)h * stride + headDim, gd + (long)h * headDim, headDim * 4L, headDim * 4L);
                    }
                }
                InvalidateTensorDeviceCache(q);
                InvalidateTensorDeviceCache(gate);
            }
            qgFull.Dispose();

            Tensor k = LinearForward(cur, $"blk.{il}.attn_k.weight");
            Tensor v = LinearForward(cur, $"blk.{il}.attn_v.weight");
            long ta1 = Stopwatch.GetTimestamp(); Q4eAttnProjTicks += ta1 - ta0;

            RmsNormHeads(q, $"blk.{il}.attn_q_norm.weight", nHead, headDim, seqLen);
            RmsNormHeads(k, $"blk.{il}.attn_k_norm.weight", nKvHead, headDim, seqLen);
            long ta2 = Stopwatch.GetTimestamp(); Q4eAttnNormTicks += ta2 - ta1;

            q = ApplyRope(q, nHead, headDim, seqLen, startPos);
            k = ApplyRope(k, nKvHead, headDim, seqLen, startPos);
            long ta3 = Stopwatch.GetTimestamp(); Q4eAttnRopeTicks += ta3 - ta2;

            Tensor attn = RunAttention(q, k, v, il, seqLen, startPos, totalLen);
            long ta4 = Stopwatch.GetTimestamp(); Q4eAttnCoreTicks += ta4 - ta3;
            _attnTicks += ta4 - ta3;

            // Gate the attention output before the output projection. Two dispatches
            // for 6144 elements; on a dispatch-bound decode step that is pure launch
            // overhead, so small batches do it on the host (see HostElementwiseMaxRows).
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.Sigmoid(gate, gate);
                Ops.Mul(attn, attn, gate);
            }
            else
            {
                float* gp = GetFloatPtr(gate);
                float* ap = GetFloatPtr(attn);
                long count = (long)seqLen * nHead * headDim;
                for (long i = 0; i < count; i++)
                    ap[i] *= 1.0f / (1.0f + MathF.Exp(-gp[i]));
                InvalidateTensorDeviceCache(attn);
            }
            gate.Dispose();

            Tensor outProj = LinearForward(attn, $"blk.{il}.attn_output.weight");
            attn.Dispose();
            Q4eAttnOutTicks += Stopwatch.GetTimestamp() - ta4;
            return outProj;
        }

        private Tensor RunAttention(Tensor q, Tensor k, Tensor v, int il,
            int seqLen, int startPos, int totalLen)
        {
            int nHead = Config.NumHeads;
            int nKvHead = Config.NumKVHeads;
            int headDim = Config.HeadDim;

            if (seqLen == 1)
            {
                var res = new Tensor(_allocator, DType.Float32, 1, nHead * headDim);

                // Prefer the device kernel: it appends K/V and attends without ever
                // pulling the cache to the host. AttentionDecodePureCS walks the cache
                // as a host buffer, which on this model (head_dim 256, 12 attention
                // layers) was 2.75 ms per layer - 33 ms of a 106 ms decode token, the
                // single largest cost after the recurrence.
                if (IsGgmlBackend && TryFlashDecode(q, k, v, res, il, startPos))
                {
                    q.Dispose(); k.Dispose(); v.Dispose();
                    return res;
                }

                CopyToCacheDecode(_kCache[il], k, _vCache[il], v, nKvHead, headDim, startPos);
                k.Dispose(); v.Dispose();
                AttentionDecodePureCS(q, _kCache[il], _vCache[il], res,
                    nHead, nKvHead, headDim, totalLen, _attnScale);
                q.Dispose();
                return res;
            }

            Tensor qH = ReshapeToHeads(q, nHead, seqLen, headDim); q.Dispose();
            Tensor kH = ReshapeToHeads(k, nKvHead, seqLen, headDim); k.Dispose();
            Tensor vH = ReshapeToHeads(v, nKvHead, seqLen, headDim); v.Dispose();

            CopyToCache(_kCache[il], kH, startPos, seqLen);
            CopyToCache(_vCache[il], vH, startPos, seqLen);
            kH.Dispose(); vH.Dispose();

            int group = nHead / nKvHead;
            Tensor kExp = ExpandKVHeads(_kCache[il], group, totalLen);
            Tensor vExp = ExpandKVHeads(_vCache[il], group, totalLen);

            using var kT = kExp.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, nHead, seqLen, totalLen);
            Ops.AddmmBatch(scores, 0, scores, _attnScale, qH, kT);
            qH.Dispose(); kExp.Dispose();

            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(scores, sinks: null,
                    numHeads: nHead, seqLen: seqLen, kvLen: totalLen,
                    maskStartPos: startPos, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, nHead, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExp);
            scores.Dispose(); vExp.Dispose();

            Tensor flat = ReshapeFromHeads(attnOut, nHead, seqLen, headDim);
            attnOut.Dispose();
            return flat;
        }

        /// <summary>Device-side single-token attention: appends K/V to the cache and
        /// attends over [0, position]. False on any shape or dtype the kernel declines,
        /// which falls back to the host walk.</summary>
        private bool TryFlashDecode(Tensor q, Tensor k, Tensor v, Tensor output, int il, int position)
        {
            Tensor kCache = _kCache[il], vCache = _vCache[il];
            if (kCache == null || vCache == null || kCache.ElementType != vCache.ElementType)
                return false;
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                return false;

            try
            {
                GgmlBasicOps.FlashAttnDecode(q, k, v, kCache, vCache, output,
                    Config.NumHeads, Config.NumKVHeads, Config.HeadDim,
                    _kvCacheCapacity, position, _attnScale);
                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private unsafe void RmsNormHeads(Tensor data, string weightName, int heads, int dim, int seqLen)
        {
            float* p = GetFloatPtr(data);
            float* w = GetFloatPtr(_weights[weightName]);
            float eps = Config.Eps;
            for (int t = 0; t < seqLen; t++)
            {
                float* row = p + (long)t * heads * dim;
                for (int h = 0; h < heads; h++)
                {
                    float* x = row + (long)h * dim;
                    double ss = 0;
                    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
                    float inv = (float)(1.0 / Math.Sqrt(ss / dim + eps));
                    for (int i = 0; i < dim; i++) x[i] = x[i] * inv * w[i];
                }
            }
            InvalidateTensorDeviceCache(data);
        }



        /// <summary>
        /// Partial rotary over the first <c>rope.dimension_count</c> of each head.
        /// The file asks for IMRoPE, which interleaves the t/h/w position components
        /// across pairs; for text every component is the same position, so it reduces
        /// exactly to NEOX RoPE. Vision will need the real thing.
        /// </summary>
        private unsafe Tensor ApplyRope(Tensor data, int heads, int headDim, int seqLen, int startPos)
        {
            // Partial NEOX rotary over the first _ropeDimCount dims. For a decode step
            // this is 6144 (Q) or 512 (K) elements - far cheaper on the host than a
            // GGML dispatch, and it is 24 of them per token across the 12 attention
            // layers.
            if (seqLen <= HostElementwiseMaxRows)
            {
                int half = _ropeDimCount / 2;
                float* p = GetFloatPtr(data);
                // Same frequency scale the GGML path (below) and both fused kernels
                // pass. Omitting it here made this fast path rotate on a different
                // convention from every other path the moment a GGUF declares
                // rope.scaling.factor != 1, so K rows written before and after a
                // fallback would disagree. No-op for factor == 1 (today's qwen4exp
                // GGUFs), which is why it went unnoticed.
                float freqScale = 1.0f / Config.RopeScale;
                for (int t = 0; t < seqLen; t++)
                {
                    int pos = startPos + t;
                    float* row = p + (long)t * heads * headDim;
                    for (int h = 0; h < heads; h++)
                    {
                        float* x = row + (long)h * headDim;
                        for (int i = 0; i < half; i++)
                        {
                            float theta = freqScale * pos / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDimCount);
                            float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
                            float a = x[i], b = x[i + half];
                            x[i] = a * cos - b * sin;
                            x[i + half] = a * sin + b * cos;
                        }
                    }
                }
                InvalidateTensorDeviceCache(data);
                return data;
            }

            int rows = seqLen * heads;
            var positions = new int[rows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                    positions[s * heads + h] = startPos + s;

            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, positions, rows);
            using var reshaped = data.View(1, seqLen, heads, headDim);
            Tensor rotated = Ops.RoPEEx(null, reshaped, posTensor, _ropeDimCount, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale, 0.0f, 1.0f, 0.0f, 0.0f);
            data.Dispose();

            Tensor flat = rotated.View(seqLen, heads * headDim);
            rotated.Dispose();
            return flat;
        }
    }
}
