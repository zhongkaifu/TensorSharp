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
        // KV for the full-attention layers only; the GDN layers carry recurrent state
        // instead. Indexed by absolute layer, null on a recurrent one.
        private Tensor[] _kCache;
        private Tensor[] _vCache;

        // Indexer keys, cached RAW: the pooling that produces a block key happens
        // before the norm and the rotation, so a cached post-rotation key could not be
        // re-pooled at a different block boundary.
        private Tensor[] _idxKCache;

        // GDN state, host side. conv is a [convKernel-1, convDim] ring per layer and
        // delta is [numVHeads, headKDim, headVDim].
        private float[][] _gdnConvState;

        // PLE depthwise conv history: (kernel-1)*ngram rows of hcDim.
        private float[] _pleConvState;

        private int _convDim;

        // Rows currently allocated in the attention caches; grows on demand up to
        // _maxContextLength.
        private int _kvCacheCapacity;
        private int _initialKvCacheCapacity;

        // Per-phase tick counters. Cheap (one timestamp per phase per layer) and the
        // only way to tell a slow recurrence from a slow MoE without a profiler.
        // Reported by TS_Q4E_PROFILE=1.
        internal long Q4ePleTicks, Q4eHcTicks, Q4eGdnTicks, Q4eAttnTicks, Q4eMoeTicks, Q4eHeadTicks, Q4eSpanTicks;
        internal long Q4eGdnProjTicks, Q4eGdnPrepTicks, Q4eGdnKernelTicks, Q4eGdnOutTicks;
        internal long Q4eAttnProjTicks, Q4eAttnNormTicks, Q4eAttnRopeTicks, Q4eAttnCoreTicks, Q4eAttnOutTicks;
        private static readonly bool _q4eProfile =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_PROFILE"), "1", StringComparison.Ordinal);

        private void ReportQ4eProfile(int tokens)
        {
            if (!_q4eProfile) return;
            double f = 1000.0 / Stopwatch.Frequency;
            Console.WriteLine(
                $"[q4e-profile] tokens={tokens} ple={Q4ePleTicks * f:F0}ms hc={Q4eHcTicks * f:F0}ms " +
                $"gdn={Q4eGdnTicks * f:F0}ms attn={Q4eAttnTicks * f:F0}ms moe={Q4eMoeTicks * f:F0}ms " +
                $"span={Q4eSpanTicks * f:F0}ms " +
                $"head={Q4eHeadTicks * f:F0}ms");
            Console.WriteLine(
                $"[q4e-profile]   gdn: proj={Q4eGdnProjTicks * f:F0} prep={Q4eGdnPrepTicks * f:F0} " +
                $"kernel={Q4eGdnKernelTicks * f:F0} out={Q4eGdnOutTicks * f:F0}");
            Console.WriteLine(
                $"[q4e-profile]   attn: proj={Q4eAttnProjTicks * f:F0} norm={Q4eAttnNormTicks * f:F0} " +
                $"rope={Q4eAttnRopeTicks * f:F0} core={Q4eAttnCoreTicks * f:F0} out={Q4eAttnOutTicks * f:F0}");
        }

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            _initialKvCacheCapacity = initialSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();

            int nLayer = Config.NumLayers;
            _kCache = new Tensor[nLayer];
            _vCache = new Tensor[nLayer];
            _idxKCache = new Tensor[nLayer];
            _gdnConvState = new float[nLayer][];

            int keyDim = _headKDim * _numKHeads;
            int valueDim = _headVDim * _numVHeads;
            _convDim = keyDim * 2 + valueDim;

            for (int l = 0; l < nLayer; l++)
            {
                if (_isRecurrent[l])
                {
                    _gdnConvState[l] = new float[(long)(_convKernel - 1) * _convDim];
                    continue;
                }

                _kCache[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                _vCache[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                InitializeCacheTensor(_kCache[l]);
                InitializeCacheTensor(_vCache[l]);

                if (UsesQsa(l))
                {
                    _idxKCache[l] = new Tensor(_allocator, kvDtype, 1, initialSeqLen, _indexerHeadDim);
                    InitializeQsaCache(_idxKCache[l]);
                }
            }

            if (_pleHeads > 0)
                // PINNED: the span kernel seeds its device conv state from this address.
                _pleConvState = GC.AllocateArray<float>(
                    checked((int)((long)(_pleConvKernel - 1) * _pleNgram * _hcDim)), pinned: true);

            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
            {
                throw new InvalidOperationException(
                    $"Sequence length {requiredSeqLen} exceeds the configured context length {_maxContextLength}.");
            }

            int newCapacity = Math.Min(_maxContextLength, Math.Max(requiredSeqLen, _kvCacheCapacity * 2));
            // The fused path writes the KV cache on the DEVICE; the host mirror
            // is stale until synced. Growing from the stale mirror would carry
            // garbage forward for every past position.
            EnsureKvCacheHostSynchronized();
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                GrowCache(ref _kCache[l], Config.NumKVHeads, newCapacity, Config.HeadDim, kvDtype);
                GrowCache(ref _vCache[l], Config.NumKVHeads, newCapacity, Config.HeadDim, kvDtype);
                if (_idxKCache[l] != null)
                    GrowQsaCache(l, newCapacity, kvDtype);
            }
            _kvCacheCapacity = newCapacity;
            // The KV tensors just moved: the pinned attention descriptors hold the
            // old storage pointers and byte counts, and every graph keyed on them -
            // per-layer and span alike - is stale. Dropping the array forces a refill
            // with the new pointers and, through the graph keys, a rebuild.
            _attnArgs = null;
            _qsaArgs = null;
        }

        private void GrowCache(ref Tensor cache, int heads, int newLen, int dim, DType dtype)
        {
            var grown = new Tensor(_allocator, dtype, heads, newLen, dim);
            // Zero UNCONDITIONALLY. InitializeCacheTensor skips the fill on
            // GgmlCuda/Mlx, and the native span kernel's own memset is guarded by
            // `position == 0`, which a mid-sequence grow never satisfies - so the
            // rows past _cacheSeqLen would keep whatever the memory pool handed
            // back. The flash window is padded to a 256 stride past n_kv and those
            // rows ARE read: a masked column contributes nothing only if its K row
            // is finite, and an Inf there plus the -inf mask is NaN, which takes
            // the whole softmax row. Fires on any conversation past the initial
            // 8192-token capacity - e.g. a --max-tokens 20000 answer.
            Ops.Fill(grown, 0f);
            if (_cacheSeqLen > 0)
            {
                using var src = cache.Narrow(1, 0, _cacheSeqLen);
                using var dst = grown.Narrow(1, 0, _cacheSeqLen);
                Ops.Copy(dst, src);
            }
            // Evict the old tensor's device-resident copy before its host
            // pointer is freed: a recycled pinned address must never re-find
            // the smaller stale device slab.
            InvalidateTensorDeviceCache(cache);
            cache.Dispose();
            cache = grown;
        }

        /// <summary>Download the device-resident KV copies back into the host
        /// mirrors. The fused kernels write KV on the device only; anything that
        /// reads or copies the host arrays afterwards must sync first.</summary>
        private void EnsureKvCacheHostSynchronized()
        {
            if (!_kvCacheHostStale || !IsGgmlBackend || _kCache == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kCache[l] != null) SyncTensorHostCache(_kCache[l]);
                if (_vCache[l] != null) SyncTensorHostCache(_vCache[l]);
                if (_idxKCache[l] != null) SyncQsaCache(l);
            }
            _kvCacheHostStale = false;
        }

        private unsafe void InvalidateActiveSeqState()
        {
            if (_idxKCache != null)
                foreach (var t in _idxKCache)
                    if (t != null) GgmlBasicOps.Qwen4ExpInvalidateSeqState(TensorComputePrimitives.GetStoragePointer(t));
            if (_gdnConvStateT != null)
                foreach (var t in _gdnConvStateT)
                    if (t != null) GgmlBasicOps.Qwen4ExpInvalidateSeqState((IntPtr)GetFloatPtr(t));
            if (_pleConvState != null && _pleConvState.Length > 0)
                GgmlBasicOps.Qwen4ExpInvalidateSeqState(
                    System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(_pleConvState, 0));
        }

        protected override void ResetKVCacheCore()
        {
            ReleaseMtpState(ActiveSpecOwner);
            ReleaseSpecSnapshot();
            _qsaPositionCount = 0;
            unchecked { ++_specResetVersion; }
            _cacheSeqLen = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_gdnConvState[l] != null) Array.Clear(_gdnConvState[l]);
                // The delta-net state lives in a device tensor now (the fused kernel
                // updates it in place), so clearing the old host array is not enough.
                if (_gdnStateT != null && _gdnStateT[l] != null)
                {
                    Ops.Fill(_gdnStateT[l], 0f);
                    InvalidateTensorDeviceCache(_gdnStateT[l]);
                }
                // The fused recurrent kernel keeps the conv history in its own
                // [d_conv-1, conv_dim] tensor rather than the host ring.
                if (_gdnConvStateT != null && _gdnConvStateT[l] != null)
                {
                    Ops.Fill(_gdnConvStateT[l], 0f);
                    InvalidateTensorDeviceCache(_gdnConvStateT[l]);
                }
                if (_gdnConvWriteIdx != null) _gdnConvWriteIdx[l] = 0;
            }
            if (_pleConvState != null) Array.Clear(_pleConvState);
            ResetPleHistory();
            _mropeCacheGap = 0;

            // The fused kernels keep the conv and recurrent state in their OWN device
            // buffers inside a persisted graph, so zeroing the host copies above does
            // not reach them. Dropping the graphs makes the next call rebuild and
            // re-upload from the (now zeroed) host state. Without this the kernel
            // warmup's state leaked into the first real generation, which read as
            // fluent-looking noise while every per-layer check passed.
            if (IsGgmlBackend && (_ffnArgs != null || _gdnArgs != null))
            {
                // Re-arm the seed upload for THIS conversation's recurrent-state
                // entries (their host copies were just zeroed above); other
                // sequences' entries stay device-authoritative. The graph reset
                // below then forces the rebuild that performs the re-seed.
                InvalidateActiveSeqState();
                GgmlBasicOps.Qwen4ExpResetFfnCache();
            }
            _specStateFailed = false;
            // Host seeds zeroed and every entry re-armed: the host is the truth again.
            _deviceStateAuthoritative = false;
            // Reset invalidates QSA's native entry. There are no live KV rows
            // to export when a larger first prompt immediately grows capacity.
            _kvCacheHostStale = false;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (_specStateFailed)
                throw new InvalidOperationException("qwen4exp: the active conversation has failed speculative state; reset it before forwarding.");
            try
            {
                return ForwardCoreInner(tokens);
            }
            catch
            {
                // A thrown forward must not leak this request's pending
                // multimodal state into the NEXT sequence the engine runs (the
                // per-sequence executor catches per-request errors and moves on).
                if (HasQsa) _specStateFailed = true;
                _pendingMRoPEPositions = null;
                foreach (var (emb, _) in _visionEmbeddingsList) emb?.Dispose();
                _visionEmbeddingsList.Clear();
                throw;
            }
        }

        private float[] ForwardCoreInner(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);
            PrepareQsaHistory(startPos, seqLen);

            long tEmb = Stopwatch.GetTimestamp();
            Tensor embd = Embedding(tokens);           // [T, n_embd]
            InjectVisionEmbeddings(embd, seqLen);
            _embTicks += Stopwatch.GetTimestamp() - tEmb;
            PhaseLog(seqLen, "embedding", tEmb);

            // The wide residual starts as hc identical copies of the embedding.
            long tBr = Stopwatch.GetTimestamp();
            Tensor res = BroadcastToStreams(embd, seqLen);
            embd.Dispose();
            PhaseLog(seqLen, "broadcast", tBr);

            // Hand the residual to the device once; the fused halves chain through it
            // and only the layers still running op-by-op pull it back.
            _resOnDevice = false;

            // The whole token as (almost) one graph: every PLE-free run of layers -
            // both halves of each - chained into a single persisted GGML graph, so a
            // decode token is 2 graph launches instead of 96 and the residual crosses
            // the bus twice instead of 192 times. Anything the span declines falls
            // back to the per-layer loop below.
            long tSpan = Stopwatch.GetTimestamp();
            bool spanDone = TryFusedTokenSpans(res, tokens, seqLen, startPos);
            if (!spanDone && (_specForwardActive || HasQsa))
            {
                res.Dispose();
                throw new InvalidOperationException("qwen4exp: speculative token span failed; reset this conversation before retrying.");
            }
            if (spanDone) Q4eSpanTicks += Stopwatch.GetTimestamp() - tSpan;
            if (!spanDone && LayerSplitDegree > 1)
            {
                // The op-by-op fallback allocates from _allocator (rank 0) and reads
                // the HOST recurrent-state arrays. Under a layer split most layers'
                // weights, KV device copies and GDN/PLE state live on another GPU, so
                // running it would silently compute against the wrong device's state -
                // the same class of failure as the mid-sequence QSA fallback that used
                // to collapse generation. Refuse instead.
                throw new NotSupportedException(
                    "qwen4exp: the token-span path declined while a layer split is active. "
                    + "The per-layer fallback is single-GPU only, so it cannot run here. "
                    + "Re-run without --tp to use the fallback.");
            }
            if (!spanDone && _pendingMRoPEPositions != null)
            {
                // The fallback paths rotate with scalar positions only; running an
                // image prompt through them would be silently wrong.
                throw new NotSupportedException(
                    "qwen4exp image prompts need the token-span path "
                    + "(TS_Q4E_TOKEN_GRAPH=0 and image inputs cannot be combined).");
            }

            if (!spanDone && _ffnArgs != null && !_fusedFfnUnsupported)
                ResidualToDevice(res, seqLen);

            for (int il = spanDone ? Config.NumLayers : 0; il < Config.NumLayers; il++)
            {
                long t0 = Stopwatch.GetTimestamp();
                if (_isPle[il])
                {
                    ResidualToHost(res, seqLen);
                    PleLayer(res, tokens, seqLen, startPos, il);
                    ResidualToDevice(res, seqLen);
                }
                long t1 = Stopwatch.GetTimestamp(); Q4ePleTicks += t1 - t0;

                // The recurrent half - mixer, projections, causal conv, the delta-net
                // recurrence and the scatter - is one graph when the kernel takes the
                // shape. That is ~9 GGML submissions collapsed into 1 on 36 of the 48
                // layers, on a path that is dispatch bound.
                // Verify once for a single-token step and once for a multi-token one:
                // the conv history and the multi-token recurrence only exercise the
                // second, and a kernel can be right for one and wrong for the other.
                if (_isRecurrent[il] && _gdnVerify
                    && ((seqLen == 1 && !_gdnVerified1) || (seqLen > 1 && !_gdnVerifiedN)))
                {
                    VerifyGdnBlock(res, il, seqLen);
                    if (seqLen == 1) _gdnVerified1 = true; else _gdnVerifiedN = true;
                }

                if (_isRecurrent[il] && TryFusedGdnBlock(res, il, seqLen))
                {
                    long tg = Stopwatch.GetTimestamp(); Q4eGdnTicks += tg - t1;
                    if (TryFusedFfnBlock(res, il, seqLen))
                    {
                        Q4eMoeTicks += Stopwatch.GetTimestamp() - tg;
                        continue;
                    }
                    // FFN fell back; run the op-by-op FFN half below with a fresh mixer.
                    ResidualToHost(res, seqLen);
                    Tensor injF;
                    Tensor curF = HcMix(res, seqLen,
                        $"blk.{il}.hc_ffn_norm.weight", $"blk.{il}.hc_ffn_down.weight",
                        $"blk.{il}.hc_ffn_up.weight", $"blk.{il}.hc_ffn_inject.weight", out injF);
                    Tensor ffnF = MoeFfn(curF, il, seqLen);
                    curF.Dispose();
                    HcCombine(res, ffnF, injF, seqLen);
                    ffnF.Dispose();
                    injF.Dispose();
                    ResidualToDevice(res, seqLen);
                    continue;
                }

                // The full-attention half is one graph too when the kernel takes the
                // shape: mixer, query|gate, norms, rotary, KV append, gated attention
                // and the scatter.
                if (!_isRecurrent[il] && TryFusedAttnBlock(res, il, seqLen, startPos))
                {
                    Q4eAttnTicks += Stopwatch.GetTimestamp() - t1;
                    if (TryFusedFfnBlock(res, il, seqLen))
                    {
                        Q4eMoeTicks += Stopwatch.GetTimestamp() - t1;
                        continue;
                    }
                    ResidualToHost(res, seqLen);
                    Tensor injA;
                    Tensor curA = HcMix(res, seqLen,
                        $"blk.{il}.hc_ffn_norm.weight", $"blk.{il}.hc_ffn_down.weight",
                        $"blk.{il}.hc_ffn_up.weight", $"blk.{il}.hc_ffn_inject.weight", out injA);
                    Tensor ffnA = MoeFfn(curA, il, seqLen);
                    curA.Dispose();
                    HcCombine(res, ffnA, injA, seqLen);
                    ffnA.Dispose();
                    injA.Dispose();
                    ResidualToDevice(res, seqLen);
                    continue;
                }

                ResidualToHost(res, seqLen);
                Tensor inject;
                Tensor cur = HcMix(res, seqLen,
                    $"blk.{il}.hc_attn_norm.weight", $"blk.{il}.hc_attn_down.weight",
                    $"blk.{il}.hc_attn_up.weight", $"blk.{il}.hc_attn_inject.weight", out inject);
                long t2 = Stopwatch.GetTimestamp(); Q4eHcTicks += t2 - t1;

                Tensor blockOut = _isRecurrent[il]
                    ? GdnLayer(cur, il, seqLen)
                    : AttentionLayer(cur, il, seqLen, startPos);
                cur.Dispose();
                long t3 = Stopwatch.GetTimestamp();
                if (_isRecurrent[il]) Q4eGdnTicks += t3 - t2; else Q4eAttnTicks += t3 - t2;

                HcCombine(res, blockOut, inject, seqLen);
                blockOut.Dispose();
                inject.Dispose();
                _ = t3;
                if (_ffnArgs != null && !_fusedFfnUnsupported)
                    ResidualToDevice(res, seqLen);

                // The mixer, the experts and the scatter are one graph when the
                // fused kernel takes the shape - 8 GGML submissions collapsed into
                // 1, 48 times a token, on a path that is dispatch bound.
                if (TryFusedFfnBlock(res, il, seqLen))
                {
                    Q4eMoeTicks += Stopwatch.GetTimestamp() - t3;
                    continue;
                }

                ResidualToHost(res, seqLen);
                cur = HcMix(res, seqLen,
                    $"blk.{il}.hc_ffn_norm.weight", $"blk.{il}.hc_ffn_down.weight",
                    $"blk.{il}.hc_ffn_up.weight", $"blk.{il}.hc_ffn_inject.weight", out inject);
                long t4 = Stopwatch.GetTimestamp(); Q4eHcTicks += t4 - t3;

                Tensor ffnOut = MoeFfn(cur, il, seqLen);
                cur.Dispose();
                long t5 = Stopwatch.GetTimestamp(); Q4eMoeTicks += t5 - t4;

                HcCombine(res, ffnOut, inject, seqLen);
                ffnOut.Dispose();
                inject.Dispose();
                Q4eHcTicks += Stopwatch.GetTimestamp() - t5;
            }

            ResidualToHost(res, seqLen);

            if (_pendingMRoPEPositions != null)
                UpdateMropeGap(startPos, seqLen);
            _pendingMRoPEPositions = null;

            if (_spanLogitsValid)
            {
                if (_specForwardActive)
                    CopySpecLastLogits(seqLen);
                // The last span already produced the logits; the residual is dead.
                res.Dispose();
                _logitsBuffer = _spanLogits;
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                ReportQ4eProfile(seqLen);
                return _logitsBuffer;
            }

            // The final mixer IS the output norm - qwen4exp ships no separate one.
            Tensor normed = HcMix(res, seqLen,
                "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight",
                null, out _);
            res.Dispose();

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

            long tHead = Stopwatch.GetTimestamp();
            Tensor logits = LinearForward(lastHidden, "output.weight")
                            ?? LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - tHead;
            Q4eHeadTicks += Stopwatch.GetTimestamp() - tHead;
            lastHidden.Dispose();

            _logitsBuffer = TensorToFloatArray(logits);
            logits.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            ReportQ4eProfile(seqLen);
            return _logitsBuffer;
        }

        /// <summary>[T, n_embd] -> [T, hc*n_embd], every stream a copy.</summary>
        private unsafe Tensor BroadcastToStreams(Tensor x, int seqLen)
        {
            int n = Config.HiddenSize;
            var res = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            long srcA = (long)GetFloatPtr(x);
            long dstA = (long)GetFloatPtr(res);
            int hc = _hc, hcDim = _hcDim;
            void CopyToken(int t)
            {
                float* s = (float*)srcA + (long)t * n;
                float* d = (float*)dstA + (long)t * hcDim;
                for (int c = 0; c < hc; c++)
                    Buffer.MemoryCopy(s, d + (long)c * n, n * 4L, n * 4L);
            }
            if (seqLen >= 16) System.Threading.Tasks.Parallel.For(0, seqLen, CopyToken);
            else for (int t = 0; t < seqLen; t++) CopyToken(t);
            InvalidateTensorDeviceCache(res);
            return res;
        }

        /// <summary>
        /// Read the wide residual: grouped RMS norm over each stream, a low-rank
        /// sigmoid gate, then collapse the streams by their mean. Also produces the
        /// per-stream scatter weights the matching <see cref="HcCombine"/> needs.
        /// </summary>
        private unsafe Tensor HcMix(Tensor res, int seqLen,
            string normName, string downName, string upName, string injectName, out Tensor inject)
        {
            int n = Config.HiddenSize;

            // xn = groupRmsNorm(res) * w_norm. The converter folded the gammas to
            // (1 + w), so this is a plain multiply rather than an affine.
            var xn = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            {
                float* src = GetFloatPtr(res);
                float* dst = GetFloatPtr(xn);
                float* w = GetFloatPtr(_weights[normName]);
                float eps = Config.Eps;
                for (int t = 0; t < seqLen; t++)
                {
                    float* r = src + (long)t * _hcDim;
                    float* o = dst + (long)t * _hcDim;
                    for (int c = 0; c < _hc; c++)
                    {
                        float* rc = r + (long)c * n;
                        float* oc = o + (long)c * n;
                        double ss = 0;
                        for (int i = 0; i < n; i++) ss += (double)rc[i] * rc[i];
                        float inv = (float)(1.0 / Math.Sqrt(ss / n + eps));
                        int b = c * n;
                        for (int i = 0; i < n; i++) oc[i] = rc[i] * inv * w[b + i];
                    }
                }
                InvalidateTensorDeviceCache(xn);
            }

            // gate = sigmoid(W_up @ silu(W_down @ xn / hc))
            //
            // The scale+SiLU on [T, 320] and the sigmoid on [T, 10240] are three more
            // GGML dispatches for a few thousand elements. At 96 mixers per token that
            // was 288 dispatches of pure launch overhead, on a path that is already
            // dispatch-bound, so they run on the host instead.
            Tensor lo = LinearForward(xn, downName);                 // [T, hcLowRank]
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.Mul(lo, lo, 1.0f / _hc);
                Ops.SiLU(lo, lo);
            }
            else
            {
                float* p = GetFloatPtr(lo);
                long count = (long)seqLen * _hcLowRank;
                float inv = 1.0f / _hc;
                for (long i = 0; i < count; i++)
                {
                    float x = p[i] * inv;
                    p[i] = x / (1.0f + MathF.Exp(-x));               // silu
                }
                InvalidateTensorDeviceCache(lo);
            }
            Tensor gate = LinearForward(lo, upName);                 // [T, hcDim]
            lo.Dispose();
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.Sigmoid(gate, gate);
            }
            else
            {
                float* p = GetFloatPtr(gate);
                long count = (long)seqLen * _hcDim;
                for (long i = 0; i < count; i++)
                    p[i] = 1.0f / (1.0f + MathF.Exp(-p[i]));
                InvalidateTensorDeviceCache(gate);
            }

            // mixed = mean over the streams of (xn * gate)
            var mixed = new Tensor(_allocator, DType.Float32, seqLen, n);
            {
                float* xp = GetFloatPtr(xn);
                float* gp = GetFloatPtr(gate);
                float* mp = GetFloatPtr(mixed);
                float invHc = 1.0f / _hc;
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = xp + (long)t * _hcDim;
                    float* g = gp + (long)t * _hcDim;
                    float* m = mp + (long)t * n;
                    for (int i = 0; i < n; i++) m[i] = x[i] * g[i];
                    for (int c = 1; c < _hc; c++)
                    {
                        int b = c * n;
                        for (int i = 0; i < n; i++) m[i] += x[b + i] * g[b + i];
                    }
                    for (int i = 0; i < n; i++) m[i] *= invHc;
                }
                InvalidateTensorDeviceCache(mixed);
            }
            gate.Dispose();

            inject = injectName != null ? LinearForward(xn, injectName) : null;  // [T, hc]
            xn.Dispose();
            return mixed;
        }

        /// <summary>
        /// Write a block's output back into every residual stream, each scaled by its
        /// own gate. 2*sigmoid centres the weights on 1, so an untrained injection
        /// reproduces a plain residual add.
        /// </summary>
        private unsafe void HcCombine(Tensor res, Tensor blockOut, Tensor inject, int seqLen)
        {
            int n = Config.HiddenSize;
            float* rp = GetFloatPtr(res);
            float* bp = GetFloatPtr(blockOut);
            float* ip = GetFloatPtr(inject);
            float invHc = 1.0f / _hc;
            for (int t = 0; t < seqLen; t++)
            {
                float* r = rp + (long)t * _hcDim;
                float* b = bp + (long)t * n;
                float* inj = ip + (long)t * _hc;
                for (int c = 0; c < _hc; c++)
                {
                    float w = 2.0f / (1.0f + MathF.Exp(-inj[c] * invHc));
                    float* rc = r + (long)c * n;
                    for (int i = 0; i < n; i++) rc[i] += b[i] * w;
                }
            }
            InvalidateTensorDeviceCache(res);
        }

        /// <summary>
        /// 512-expert MoE: softmax over all experts, top-k, renormalise, plus a shared
        /// expert behind its own scalar sigmoid gate.
        /// </summary>
        private unsafe Tensor MoeFfn(Tensor input, int il, int seqLen)
        {
            int n = Config.HiddenSize;
            if (_fusedGateUpExperts)
            {
                throw new NotSupportedException(
                    "This qwen4exp GGUF stacks gate and up into blk.N.ffn_gate_up_exps; "
                    + "only the separate ffn_gate_exps / ffn_up_exps layout is wired up so far.");
            }
            var gateStack = _stackedExpertWeights[$"blk.{il}.ffn_gate_exps.weight"];
            var upStack = _stackedExpertWeights[$"blk.{il}.ffn_up_exps.weight"];
            var downStack = _stackedExpertWeights[$"blk.{il}.ffn_down_exps.weight"];

            Tensor routerLogits = LinearForward(input, $"blk.{il}.ffn_gate_inp.weight"); // [T, nExperts]

            // Host top-k routing for every token at once: softmax over all 512 experts,
            // keep the best n_expert_used, renormalise (llama.cpp's norm_w).
            int K = _numExpertsUsed;
            if (_moeSelExperts == null || _moeSelExperts.Length < (long)seqLen * K)
            {
                _moeSelExperts = new int[(long)seqLen * K];
                _moeRouteWts = new float[(long)seqLen * K];
            }
            var probs = _moeProbs ??= new float[_numExperts];

            float* rl = GetFloatPtr(routerLogits);
            for (int t = 0; t < seqLen; t++)
            {
                float* row = rl + (long)t * _numExperts;
                float max = float.NegativeInfinity;
                for (int e = 0; e < _numExperts; e++) if (row[e] > max) max = row[e];
                float sum = 0f;
                for (int e = 0; e < _numExperts; e++) { probs[e] = MathF.Exp(row[e] - max); sum += probs[e]; }
                float invSum = 1.0f / sum;
                for (int e = 0; e < _numExperts; e++) probs[e] *= invSum;

                float wSum = 0f;
                for (int k = 0; k < K; k++)
                {
                    int best = -1; float bestV = float.NegativeInfinity;
                    for (int e = 0; e < _numExperts; e++)
                        if (probs[e] > bestV) { bestV = probs[e]; best = e; }
                    probs[best] = float.NegativeInfinity;
                    _moeSelExperts[t * K + k] = best;
                    _moeRouteWts[t * K + k] = bestV;
                    wSum += bestV;
                }
                float invW = 1.0f / wSum;
                for (int k = 0; k < K; k++) _moeRouteWts[t * K + k] *= invW;
            }
            routerLogits.Dispose();

            var result = new Tensor(_allocator, DType.Float32, seqLen, n);
            int nFf = (int)gateStack.PerExpertNe1;

            if (IsGgmlBackend)
            {
                // ONE ggml_mul_mat_id over the whole batch. The per-token call this
                // replaced issued seqLen * n_layers native graph builds - 49k of them
                // for a 1024-token prefill - and was the dominant prefill cost once the
                // recurrence moved to the GPU.
                GgmlBasicOps.MoEFFNPrefill(
                    input, result, seqLen, n, nFf,
                    gateStack.NumExperts, K, _moeSelExperts, _moeRouteWts,
                    gateStack.Data, gateStack.GgmlType, gateStack.PerExpertNe0, gateStack.PerExpertNe1, gateStack.TotalRawBytes,
                    upStack.Data, upStack.GgmlType, upStack.PerExpertNe0, upStack.PerExpertNe1, upStack.TotalRawBytes,
                    downStack.Data, downStack.GgmlType, downStack.PerExpertNe0, downStack.PerExpertNe1, downStack.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.SwiGLUSplit);
            }
            else
            {
                // Portable path for the direct-CUDA and pure-C# backends, which have no
                // stacked-expert kernel: one expert at a time through the ordinary
                // quantized linear, accumulated by route weight.
                var ids = new int[K];
                var wts = new float[K];
                for (int t = 0; t < seqLen; t++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        ids[k] = _moeSelExperts[t * K + k];
                        wts[k] = _moeRouteWts[t * K + k];
                    }
                    using var tokenIn = input.Narrow(0, t, 1);
                    using var tokenOut = result.Narrow(0, t, 1);
                    MoeExpertsPortable(tokenOut, tokenIn, il, ids, wts);
                }
            }
            InvalidateTensorDeviceCache(result);

            // Shared expert, batched over all tokens, behind its own sigmoid gate.
            Tensor sg = LinearForward(input, $"blk.{il}.ffn_gate_shexp.weight");
            Tensor su = LinearForward(input, $"blk.{il}.ffn_up_shexp.weight");
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.SiLUMul(sg, sg, su);
            }
            else
            {
                // silu(gate) * up on [T, 640] - host, for the same reason as the mixer.
                float* gp = GetFloatPtr(sg);
                float* up = GetFloatPtr(su);
                long count = (long)seqLen * sg.Sizes[1];
                for (long i = 0; i < count; i++)
                    gp[i] = (gp[i] / (1.0f + MathF.Exp(-gp[i]))) * up[i];
                InvalidateTensorDeviceCache(sg);
            }
            su.Dispose();
            Tensor sd = LinearForward(sg, $"blk.{il}.ffn_down_shexp.weight");
            sg.Dispose();

            // The gate is ONE scalar per token and its weight is a bare [n_embd] vector
            // rather than a matrix - LinearForward would read that as a 2560-wide
            // projection - so the dot product is explicit.
            Tensor gateVec = _weights[$"blk.{il}.ffn_gate_inp_shexp.weight"];
            {
                float* gv = GetFloatPtr(gateVec);
                float* xp = GetFloatPtr(input);
                float* sp = GetFloatPtr(sd);
                float* rp = GetFloatPtr(result);
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = xp + (long)t * n;
                    double dot = 0;
                    for (int i = 0; i < n; i++) dot += (double)x[i] * gv[i];
                    float g = 1.0f / (1.0f + MathF.Exp(-(float)dot));
                    float* sv = sp + (long)t * n;
                    float* r = rp + (long)t * n;
                    for (int i = 0; i < n; i++) r[i] += sv[i] * g;
                }
                InvalidateTensorDeviceCache(result);
            }
            sd.Dispose();
            return result;
        }

        /// <summary>
        /// Above this many rows the small elementwise steps go back to GGML.
        ///
        /// A decode step is dispatch-bound - ~850 GGML launches per token, each costing
        /// far more than the arithmetic - so running a 320- or 10240-element sigmoid on
        /// the host removes three launches per mixer, 288 per token. A 1024-row prefill
        /// is the opposite: the same code becomes 10.5 M host exp() calls per mixer and
        /// cost 2.7 s of prefill when it was applied unconditionally.
        /// </summary>
        private const int HostElementwiseMaxRows = 8;

        // TS_Q4E_GDN_VERIFY=1 runs the fused recurrent half and the op-by-op one on
        // the SAME input and state, and reports how far apart they are. Diagnostic
        // only: it runs once, on the first recurrent layer of the first forward,
        // where the state is still zero so both paths start from the same place.
        private static readonly bool _gdnVerify =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_GDN_VERIFY"), "1", StringComparison.Ordinal);
        private bool _gdnVerified1, _gdnVerifiedN;

        private unsafe void VerifyGdnBlock(Tensor res, int il, int seqLen)
        {
            long n = (long)seqLen * _hcDim;
            var input = new float[n];
            var fused = new float[n];
            float* rp = GetFloatPtr(res);
            for (long i = 0; i < n; i++) input[i] = rp[i];

            void ZeroState()
            {
                EnsureGdnScratch();
                if (_gdnStateT?[il] != null) { Ops.Fill(_gdnStateT[il], 0f); InvalidateTensorDeviceCache(_gdnStateT[il]); }
                if (_gdnConvStateT?[il] != null) { Ops.Fill(_gdnConvStateT[il], 0f); InvalidateTensorDeviceCache(_gdnConvStateT[il]); }
                if (_gdnConvState?[il] != null) Array.Clear(_gdnConvState[il]);
                if (_gdnConvWriteIdx != null) _gdnConvWriteIdx[il] = 0;
            }

            // TWO consecutive calls: the second one is the only thing that exercises
            // the state written back by the first, which a single call cannot check.
            const int Iters = 2;

            // ---- fused ----
            ZeroState();
            for (int it = 0; it < Iters; it++)
            {
                for (long i = 0; i < n; i++) rp[i] = input[i];
                InvalidateTensorDeviceCache(res);
                if (!TryFusedGdnBlock(res, il, seqLen))
                { Console.WriteLine("[q4e-verify] fused GDN declined the shape."); return; }
            }
            for (long i = 0; i < n; i++) fused[i] = rp[i];

            // ---- op-by-op, from the same input and state ----
            ZeroState();
            for (int it = 0; it < Iters; it++)
            {
                for (long i = 0; i < n; i++) rp[i] = input[i];
                InvalidateTensorDeviceCache(res);

                Tensor inj;
                Tensor cur = HcMix(res, seqLen,
                    $"blk.{il}.hc_attn_norm.weight", $"blk.{il}.hc_attn_down.weight",
                    $"blk.{il}.hc_attn_up.weight", $"blk.{il}.hc_attn_inject.weight", out inj);
                Tensor blockOut = GdnLayer(cur, il, seqLen);
                cur.Dispose();
                HcCombine(res, blockOut, inj, seqLen);
                blockOut.Dispose();
                inj.Dispose();
            }

            // Compare the block's CONTRIBUTION (res_after - res_before), not the
            // residual: the residual is dominated by the pass-through, so a large
            // error in what the block adds hides inside it.
            double maxAbs = 0, refMax = 0, contribMax = 0, contribErr = 0;
            long worst = -1;
            for (long i = 0; i < n; i++)
            {
                double refC = rp[i] - input[i];
                double fusC = fused[i] - input[i];
                double d = Math.Abs(fusC - refC);
                if (d > contribErr) { contribErr = d; worst = i; }
                contribMax = Math.Max(contribMax, Math.Abs(refC));
                maxAbs = Math.Max(maxAbs, Math.Abs(fused[i] - rp[i]));
                refMax = Math.Max(refMax, Math.Abs(rp[i]));
            }
            Console.WriteLine($"[q4e-verify] layer {il} seqLen {seqLen}: " +
                $"residual maxAbs={maxAbs:E3} rel={(refMax > 0 ? maxAbs / refMax : 0):E3} | " +
                $"CONTRIB maxAbs={contribErr:E3} rel={(contribMax > 0 ? contribErr / contribMax : 0):E3} " +
                $"contribMax={contribMax:E3}" +
                (worst >= 0 ? $" | fusedC={fused[worst] - input[worst]:E3} refC={rp[worst] - input[worst]:E3}" : ""));

            // Leave the state as the op-by-op path left it: that one is the reference.
            ZeroState();
            for (long i = 0; i < n; i++) rp[i] = input[i];
            InvalidateTensorDeviceCache(res);
        }

        // TS_Q4E_FUSED_ATTN=0 falls back to the op-by-op attention half.
        private static readonly bool _fusedAttnEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_FUSED_ATTN"), "0", StringComparison.Ordinal);
        private bool _fusedAttnUnsupported;
        private Qwen4ExpAttnArgs[] _attnArgs;
        private ushort[] _attnMask;

        // Must mirror kQwen4ExpKvStride and q4e_flash_attn_ok in ggml_ops_qwen4exp.cpp.
        private const int KvStride = 256;
        private static readonly bool FlashAttnEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_FLASH_ATTN"), "0", StringComparison.Ordinal);

        private static bool UseFlashAttn(DType kvType, int headDim)
        {
            if (!FlashAttnEnabled || kvType != DType.Float16) return false;
            return headDim is 64 or 80 or 96 or 112 or 128 or 256;
        }

        private unsafe bool TryFusedAttnBlock(Tensor res, int il, int seqLen, int startPos)
        {
            if (!_fusedAttnEnabled || _fusedAttnUnsupported || !IsGgmlBackend)
                return false;
            // No QSA-budget guard here either - it computes the same dense
            // attention the fallback would, and declining mid-sequence loses the
            // device-resident recurrent state. See TryFusedTokenSpans.
            int totalLen = startPos + seqLen;

            try
            {
                if (!EnsureAttnArgs()) return false;
                BuildAttnMask(totalLen, seqLen, startPos, _kCache[il].ElementType);

                bool ok;
                fixed (ushort* maskPtr = _attnMask)
                {
                    ok = GgmlBasicOps.Qwen4ExpAttnBlock(ref _attnArgs[il],
                        (IntPtr)GetFloatPtr(res), (IntPtr)maskPtr,
                        Config.HiddenSize, _hc, _hcLowRank, seqLen,
                        Config.HeadDim, Config.NumHeads, Config.NumKVHeads,
                        _kvCacheCapacity, totalLen, startPos,
                        _ropeDimCount, Config.RopeBase, 1.0f / Config.RopeScale, _attnScale,
                        Config.Eps, cacheSlot: il, resResident: _resOnDevice);
                }
                if (!ok) { _fusedAttnUnsupported = true; return false; }
                if (!_resOnDevice) InvalidateTensorDeviceCache(res);
                _kvCacheHostStale = true;
                _deviceStateAuthoritative = true;
                return true;
            }
            catch (Exception ex)
            {
                _fusedAttnUnsupported = true;
                WarnFusedPathDisabled("attention", ex);
                return false;
            }
        }

        // The fused attention kernel writes the KV cache on the device; the host
        // mirror is behind until something syncs it.
        private bool _kvCacheHostStale;

        // True once a fused forward has seeded the native per-sequence state
        // entries (GDN conv+ssm, PLE conv, QSA raw keys) of the ACTIVE holder from
        // its host seeds. From then on those device entries are the truth and the
        // host seeds are stale; a reset zeroes the seeds and re-arms the upload,
        // which is the only way back to host-authoritative. Travels with the
        // holder (PerSeqCache) and decides what a copy reads (RetainedCache).
        private bool _deviceStateAuthoritative;

        private unsafe bool TryFillAttnArgs(int il, ref Qwen4ExpAttnArgs a)
        {
            if (!TryResolveQuant($"blk.{il}.hc_attn_down.weight", out IntPtr hd, out int hdT, out long hdB)
                || !TryResolveQuant($"blk.{il}.hc_attn_up.weight", out IntPtr hu, out int huT, out long huB)
                || !TryResolveQuant($"blk.{il}.hc_attn_inject.weight", out IntPtr hj, out int hjT, out long hjB)
                || !TryResolveQuant($"blk.{il}.attn_q.weight", out IntPtr wq, out int wqT, out long wqB)
                || !TryResolveQuant($"blk.{il}.attn_k.weight", out IntPtr wk, out int wkT, out long wkB)
                || !TryResolveQuant($"blk.{il}.attn_v.weight", out IntPtr wv, out int wvT, out long wvB)
                || !TryResolveQuant($"blk.{il}.attn_output.weight", out IntPtr wo, out int woT, out long woB))
            {
                return false;
            }
            if (_kCache[il] == null || _vCache[il] == null)
                return false;

            a.HcNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.hc_attn_norm.weight"]);
            a.HcDown = hd; a.HcDownType = hdT; a.HcDownBytes = hdB;
            a.HcUp = hu; a.HcUpType = huT; a.HcUpBytes = huB;
            a.HcInject = hj; a.HcInjectType = hjT; a.HcInjectBytes = hjB;
            a.Wq = wq; a.WqType = wqT; a.WqBytes = wqB;
            a.Wk = wk; a.WkType = wkT; a.WkBytes = wkB;
            a.Wv = wv; a.WvType = wvT; a.WvBytes = wvB;
            a.Wo = wo; a.WoType = woT; a.WoBytes = woB;
            a.QNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.attn_q_norm.weight"]);
            a.KNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.attn_k_norm.weight"]);
            a.KCache = GetStorageBasePtrOf(_kCache[il]);
            a.VCache = GetStorageBasePtrOf(_vCache[il]);
            a.KvType = FusedGraphKvTypeId(_kCache[il].ElementType);
            a.KvBytes = _kCache[il].ElementCount() * KvElementSize(_kCache[il].ElementType);
            return a.KCache != IntPtr.Zero && a.VCache != IntPtr.Zero && a.KvType >= 0;
        }

        private static int FusedGraphKvTypeId(DType t) => t switch
        {
            DType.Float32 => 0,     // GGML_TYPE_F32
            DType.Float16 => 1,     // GGML_TYPE_F16
            _ => -1,
        };

        private static long KvElementSize(DType t) => t == DType.Float32 ? 4 : 2;

        /// <summary>The KV cache's backing device pointer, which is what the fused
        /// kernel binds so both paths write the same copy.</summary>
        private static IntPtr GetStorageBasePtrOf(Tensor t)
            => TensorComputePrimitives.GetStorageBasePointer(t);

        // ------------------------------------------------------------------
        // The whole token as (almost) one graph. TS_Q4E_TOKEN_GRAPH=0 falls back to
        // the per-layer fused kernels; those in turn fall back op-by-op.
        // ------------------------------------------------------------------
        // ON by default: a decode token is two graph launches. The long hunt that
        // once kept this off ended at a single root cause - gallocr frees an
        // uploaded leaf weight after its last consumer unless it carries the OUTPUT
        // flag, so the first compute overwrote the small weights (dt/a/ssm-norm,
        // the attention q/k norms) and every replay read decayed garbage. With the
        // flag in place every configuration that used to degenerate verifies clean.
        private static readonly bool _tokenGraphEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_TOKEN_GRAPH"), "0", StringComparison.Ordinal);
        // TS_Q4E_DRIVER_TRACE=1 prints the residual L2 after every span and
        // attention call the driver makes. Diagnosis only.
        private static readonly bool _driverTrace =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_DRIVER_TRACE"), "1", StringComparison.Ordinal);

        // TS_Q4E_PHASE=1 prints host-side wall times for a prefill forward.
        private static readonly bool _phaseLog =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_PHASE"), "1", StringComparison.Ordinal);

        private static void PhaseLog(int seqLen, string what, long t0)
        {
            if (!_phaseLog || seqLen <= 1) return;
            double ms = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
            Console.Error.WriteLine($"[q4e-cs] T={seqLen} {what}={ms:F1}ms");
        }

        private unsafe void DriverTrace(Tensor res, int seqLen, string what)
        {
            if (!_driverTrace) return;
            float* rp = GetFloatPtr(res);
            long n = (long)seqLen * _hcDim;
            double n2 = 0;
            for (long i = 0; i < n; i++) n2 += (double)rp[i] * rp[i];
            Console.Error.WriteLine($"[q4e-drv] {what} l2={Math.Sqrt(n2):E9}");
        }

        // Fired from the permanent kill-switch catches below. Each latch is
        // checked on entry and set exactly once, so this prints once per path.
        private static void WarnFusedPathDisabled(string what, Exception ex)
        {
            Console.Error.WriteLine(
                $"[q4e] fused {what} path disabled after error: {ex.Message}; " +
                "continuing on the per-op path (slower). Reported once.");
        }

        // TS_Q4E_SPAN_ATTN=0 cuts the spans at attention layers and runs those
        // halves through the per-layer kernel - the hybrid that served while
        // attention-in-span was misdiagnosed as a numeric amplifier. The actual
        // culprit was the q/k norm weights (1KB gallocr leafs) being freed and
        // overwritten; with them protected, attention chains into the span.
        private static readonly bool _spanAttnEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_SPAN_ATTN"), "0", StringComparison.Ordinal);
        private bool _tokenGraphUnsupported;
        private byte[] _layerKinds;
        // The final mixer + LM head riding the last span. _spanLogits holds the
        // downloaded [vocab] row when the head was fused this forward.
        // The PLE block inside the span: only the hash and the table gather stay on
        // the host; the gathered rows ride in as a graph input.
        private Qwen4ExpPleArgs[] _pleArgs;
        private bool _pleArgsFailed;
        private float[] _pleConvWT;
        private float[] _pleEmbBuf;
        private int _pleLayerIndex = -1;

        private unsafe bool EnsurePleArgs()
        {
            if (_pleArgs != null) return true;
            if (_pleArgsFailed) return false;
            int il = -1;
            for (int l = 0; l < Config.NumLayers; l++) if (_isPle[l]) { il = l; break; }
            if (il < 0) { _pleArgsFailed = true; return false; }
            if (!TryResolveQuant($"blk.{il}.ple_key.weight", out IntPtr kw, out int kwT, out long kwB)
                || !TryResolveQuant($"blk.{il}.ple_value.weight", out IntPtr vw, out int vwT, out long vwB)
                || !_weights.ContainsKey($"blk.{il}.ple_norm_key.weight")
                || !_weights.ContainsKey($"blk.{il}.ple_norm_query.weight")
                || !_weights.ContainsKey($"blk.{il}.ple_norm_conv.weight")
                || !_weights.TryGetValue($"blk.{il}.ple_conv1d.weight", out Tensor convW))
            {
                _pleArgsFailed = true;
                return false;
            }

            int kern = _pleConvKernel;
            // ple_conv1d is [channels, kern] with the taps fastest; the graph wants a
            // contiguous per-channel column per tap, so transpose once. PINNED: the
            // kernel binds this address for the graph's lifetime.
            _pleConvWT = GC.AllocateArray<float>(_hcDim * kern, pinned: true);
            float* wp = GetFloatPtr(convW);
            for (int c = 0; c < _hcDim; c++)
                for (int kk = 0; kk < kern; kk++)
                    _pleConvWT[(long)kk * _hcDim + c] = wp[(long)c * kern + kk];

            var args = GC.AllocateArray<Qwen4ExpPleArgs>(1, pinned: true);
            args[0].KeyW = kw; args[0].KeyType = kwT; args[0].KeyBytes = kwB;
            args[0].ValueW = vw; args[0].ValueType = vwT; args[0].ValueBytes = vwB;
            args[0].NormKey = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ple_norm_key.weight"]);
            args[0].NormQuery = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ple_norm_query.weight"]);
            args[0].NormConv = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ple_norm_conv.weight"]);
            fixed (float* ct = _pleConvWT) args[0].Conv1dT = (IntPtr)ct;   // pinned array
            fixed (float* cs = _pleConvState) args[0].ConvState = (IntPtr)cs; // pinned array
            args[0].Kern = kern;
            args[0].Dil = _pleNgram;
            _pleLayerIndex = il;
            _pleArgs = args;
            return true;
        }

        private Qwen4ExpHeadArgs[] _headArgs;
        private bool _headArgsFailed;
        private float[] _spanLogits;
        private bool _spanLogitsValid;

        private unsafe bool EnsureHeadArgs()
        {
            if (_headArgs != null) return true;
            if (_headArgsFailed) return false;
            if (!TryResolveQuant("output_hc_down.weight", out IntPtr hd, out int hdT, out long hdB)
                || !TryResolveQuant("output_hc_up.weight", out IntPtr hu, out int huT, out long huB)
                || !_weights.ContainsKey("output_hc_norm.weight")
                || (!TryResolveQuant("output.weight", out IntPtr wh, out int whT, out long whB)
                    && !TryResolveQuant("token_embd.weight", out wh, out whT, out whB)))
            {
                _headArgsFailed = true;
                return false;
            }
            // PINNED: the kernel keys the span graph on the descriptor address.
            var args = GC.AllocateArray<Qwen4ExpHeadArgs>(1, pinned: true);
            args[0].HcNorm = (IntPtr)GetFloatPtr(_weights["output_hc_norm.weight"]);
            args[0].HcDown = hd; args[0].HcDownType = hdT; args[0].HcDownBytes = hdB;
            args[0].HcUp = hu; args[0].HcUpType = huT; args[0].HcUpBytes = huB;
            args[0].Head = wh; args[0].HeadType = whT; args[0].HeadBytes = whB;
            args[0].Vocab = Config.VocabSize;
            _spanLogits = GC.AllocateArray<float>(Config.VocabSize, pinned: true);
            _headArgs = args;
            return true;
        }

        private unsafe bool TryFusedTokenSpans(Tensor res, int[] tokens, int seqLen, int startPos)
        {
            if (!_tokenGraphEnabled || _tokenGraphUnsupported || !IsGgmlBackend
                || _fusedGateUpExperts
                || !_fusedFfnEnabled || _fusedFfnUnsupported
                || !_fusedGdnEnabled || _fusedGdnUnsupported
                || !_fusedAttnEnabled || _fusedAttnUnsupported
                || _gdnMaxLayers >= 0 || _gdnVerify)
            {
                return false;
            }


            // QSA stays inside the same span as recurrent state, including the
            // first token crossing its sparse width. A fallback cannot reseed it.
            int totalLen = startPos + seqLen;

            try
            {
                if (!EnsureFfnArgs() || !EnsureGdnArgs() || !EnsureAttnArgs() || !EnsureQsaArgs())
                {
                    _tokenGraphUnsupported = true;
                    return false;
                }
                if (_layerKinds == null)
                {
                    // PINNED: its address rides in the span's graph key.
                    _layerKinds = GC.AllocateArray<byte>(Config.NumLayers, pinned: true);
                    for (int l = 0; l < Config.NumLayers; l++)
                        _layerKinds[l] = _isRecurrent[l] ? (byte)1 : (byte)0;
                }

                int firstAttn = Array.IndexOf(_layerKinds, (byte)0);
                long tMask = Stopwatch.GetTimestamp();
                BuildAttnMask(totalLen, seqLen, startPos,
                        firstAttn >= 0 ? _kCache[firstAttn].ElementType : DType.Float16);
                PhaseLog(seqLen, "mask", tMask);

                bool ranAnything = false;
                _spanLogitsValid = false;
                bool fuseHead = EnsureHeadArgs();
                bool fusePle = EnsurePleArgs();

                // Both of the remaining per-layer cuts run OUTSIDE a span, and only a
                // span carries the device: the per-layer attention kernel and the host
                // PleLayer would execute on whatever rank the thread happens to hold
                // (always 0) while their layer's weights, KV device copy and recurrent
                // state live on another GPU. Neither is reachable in a default run -
                // the attention cut needs TS_Q4E_SPAN_ATTN=0 and the PLE cut needs the
                // PLE descriptors to fail - so refuse the combination rather than add
                // an untested placement path.
                if (LayerSplitDegree > 1 && (!_spanAttnEnabled || !fusePle))
                {
                    _tokenGraphUnsupported = true;
                    throw new NotSupportedException(
                        "qwen4exp: a layer split needs every layer inside a token span, but "
                        + (!_spanAttnEnabled
                            ? "TS_Q4E_SPAN_ATTN=0 cuts attention out of the span"
                            : "the PLE block could not be built into the span")
                        + ". These per-layer paths are single-GPU only. Re-run without --tp, "
                        + "or without TS_Q4E_SPAN_ATTN=0.");
                }

                // Image prompts carry a (T,H,W) IMRoPE position table; the span
                // kernel takes it as-is and text forwards pass nothing.
                bool useMrope = _pendingMRoPEPositions != null
                    && _pendingMRoPEPositions.Length >= 3 * seqLen
                    && EnsureMropeSections();
                if (useMrope)
                {
                    if (_mropePosPinned == null || _mropePosPinned.Length < 3 * seqLen)
                        _mropePosPinned = GC.AllocateArray<int>(3 * seqLen, pinned: true);
                    Array.Copy(_pendingMRoPEPositions, _mropePosPinned, 3 * seqLen);
                }
                if (fusePle)
                {
                    // The hash mutates the n-gram history, so it runs exactly once and
                    // in call order, same as the host path did.
                    long tG = Stopwatch.GetTimestamp();
                    int[] pleRows = ComputePleRows(tokens, startPos);
                    long need = (long)seqLen * Config.HiddenSize;
                    if (_pleEmbBuf == null || _pleEmbBuf.Length < need)
                        _pleEmbBuf = GC.AllocateArray<float>(checked((int)need), pinned: true);
                    fixed (float* eb = _pleEmbBuf)
                        GatherPleRowsRaw(eb, pleRows, seqLen);
                    PhaseLog(seqLen, "ple.gather", tG);
                }
                fixed (Qwen4ExpFfnArgs* fp = _ffnArgs)
                fixed (Qwen4ExpGdnArgs* gp = _gdnArgs)
                fixed (Qwen4ExpAttnArgs* ap = _attnArgs)
                fixed (Qwen4ExpQsaArgs* qp = _qsaArgs)
                fixed (int* qpositions = _qsaPositions)
                fixed (byte* kp = _layerKinds)
                fixed (ushort* mp = _attnMask)
                {
                    int begin = 0, spanIdx = 0;
                    bool beginFfnOnly = false;
                    for (int il = 0; il <= Config.NumLayers; il++)
                    {
                        // The host-side PLE layer cuts the token into spans - unless
                        // the PLE block rides inside the span, which is the default.
                        // So does every attention layer unless TS_Q4E_SPAN_ATTN=1.
                        bool attnCut = il < Config.NumLayers && !_isRecurrent[il] && !_spanAttnEnabled;
                        // LAYER SPLIT: a span runs entirely on one GPU, so the token
                        // is also cut wherever the owning device changes. The residual
                        // crosses the seam through the host buffer it is already
                        // passed in - the same hand-off a PLE cut uses - so a seam
                        // costs one 40 KB round trip per decode token and nothing else.
                        bool deviceCut = il < Config.NumLayers && il > begin
                            && DeviceForLayer(il) != DeviceForLayer(begin);
                        bool cut = il == Config.NumLayers || (_isPle[il] && !fusePle) || attnCut || deviceCut;
                        if (!cut) continue;
                        if (il > begin)
                        {
                            // The last span carries the final mixer + LM head and
                            // hands back logits instead of the residual.
                            bool last = il == Config.NumLayers && fuseHead;
                            IntPtr headPtr = IntPtr.Zero, logitsPtr = IntPtr.Zero;
                            if (last)
                            {
                                fixed (Qwen4ExpHeadArgs* hp = _headArgs)
                                fixed (float* lp = _spanLogits)
                                { headPtr = (IntPtr)hp; logitsPtr = _specForwardActive ? _specLogitsOutput : (IntPtr)lp; }
                            }
                            IntPtr mropePtr = IntPtr.Zero, sectPtr = IntPtr.Zero;
                            if (useMrope)
                            {
                                fixed (int* mp2 = _mropePosPinned)
                                fixed (int* sp2 = _mropeSectionsPinned)
                                { mropePtr = (IntPtr)mp2; sectPtr = (IntPtr)sp2; }
                            }
                            IntPtr plePtr = IntPtr.Zero, pleEmbPtr = IntPtr.Zero;
                            int pleLayerArg = -1;
                            if (fusePle && _pleLayerIndex >= begin && _pleLayerIndex < il)
                            {
                                fixed (Qwen4ExpPleArgs* pp = _pleArgs)
                                fixed (float* eb = _pleEmbBuf)
                                { plePtr = (IntPtr)pp; pleEmbPtr = (IntPtr)eb; }
                                pleLayerArg = _pleLayerIndex;
                            }
                            long tSpan = Stopwatch.GetTimestamp();
                            bool ok = GgmlBasicOps.Qwen4ExpTokenSpan(
                                (IntPtr)fp, (IntPtr)gp, (IntPtr)ap, (IntPtr)kp,
                                begin, il,
                                (IntPtr)GetFloatPtr(res), (IntPtr)mp,
                                Config.HiddenSize, _hc, _hcLowRank, seqLen,
                                _headKDim, _headVDim, _numKHeads, _numVHeads, _convKernel,
                                Config.HeadDim, Config.NumHeads, Config.NumKVHeads,
                                _kvCacheCapacity, totalLen, startPos,
                                _ropeDimCount, Config.RopeBase, 1.0f / Config.RopeScale, _attnScale,
                                _numExperts, _numExpertsUsed, _expertFf, _sharedFf,
                                Config.Eps, cacheSlot: spanIdx + _seqSlotBase, firstFfnOnly: beginFfnOnly,
                                head: headPtr, logitsOut: logitsPtr,
                                ple: plePtr, pleLayer: pleLayerArg, pleEmb: pleEmbPtr,
                                mropePos: mropePtr, mropeSections: sectPtr,
                                ropePosition: useMrope ? -1 : startPos - _mropeCacheGap,
                                device: DeviceForLayer(begin),
                                hiddenOut: last && _specForwardActive ? _specHiddenOutput : IntPtr.Zero,
                                logitsRows: last && _specForwardActive && _specAllLogitsRows ? seqLen : 1,
                                qsa: (IntPtr)qp, qsaPositions: (IntPtr)qpositions, qsaPositionCount: _qsaPositionCount);
                            if (ok && last) _spanLogitsValid = true;
                            if (!ok)
                            {
                                _tokenGraphUnsupported = true;
                                if (ranAnything)
                                {
                                    // Layers [0, begin) already advanced the KV cache
                                    // and the recurrent state; re-running them would
                                    // apply the token twice. Fail loudly instead of
                                    // quietly double-stepping the model; the next
                                    // forward takes the per-layer fallback.
                                    throw new InvalidOperationException(
                                        "qwen4exp token span failed mid-token; the per-layer " +
                                        "fallback takes over on the next forward.");
                                }
                                return false;
                            }
                            ranAnything = true;
                            PhaseLog(seqLen, $"span[{begin},{il})", tSpan);
                            DriverTrace(res, seqLen, $"span{spanIdx} [{begin},{il}) T={seqLen} pos={startPos}");
                            spanIdx++;
                            InvalidateTensorDeviceCache(res);
                        }
                        // `&& !fusePle` MUST match the cut condition above. Without it,
                        // ANY other reason to cut at a PLE layer - a device boundary, or
                        // attnCut - ran the HOST PleLayer even though the PLE block is
                        // also built into the spans, so PLE executed twice and its
                        // n-gram conv history advanced twice for one token.
                        if (il < Config.NumLayers && _isPle[il] && !fusePle)
                        {
                            long tPle = Stopwatch.GetTimestamp();
                            PleLayer(res, tokens, seqLen, startPos, il);
                            PhaseLog(seqLen, "ple", tPle);
                            begin = il;
                            beginFfnOnly = false;
                        }
                        else if (attnCut)
                        {
                            // The attention HALF through the proven per-layer kernel;
                            // the FFN half of this layer rides at the head of the
                            // next span instead of costing its own launch.
                            if (!TryFusedAttnBlock(res, il, seqLen, startPos))
                            {
                                _tokenGraphUnsupported = true;
                                if (ranAnything)
                                {
                                    throw new InvalidOperationException(
                                        "qwen4exp token span: the per-layer attention fallback " +
                                        "failed mid-token; the next forward runs per-layer.");
                                }
                                return false;
                            }
                            ranAnything = true;
                            DriverTrace(res, seqLen, $"attn{il} T={seqLen} pos={startPos}");
                            begin = il;
                            beginFfnOnly = true;
                        }
                        else if (deviceCut)
                        {
                            // Pure layer-split seam: no host work here at all. The span
                            // just ended wrote the residual back to its host buffer and
                            // the InvalidateTensorDeviceCache above dropped the device
                            // copy, so the next span re-uploads it onto ITS gpu.
                            begin = il;
                            beginFfnOnly = false;
                        }
                    }
                }
                _kvCacheHostStale = true;
                _deviceStateAuthoritative = true;
                return true;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (Exception ex)
            {
                _tokenGraphUnsupported = true;
                WarnFusedPathDisabled("token-graph", ex);
                return false;
            }
        }

        private bool EnsureAttnArgs()
        {
            if (_attnArgs != null) return true;
            // PINNED: the kernel keys its cached graph on the descriptor address.
            var args = GC.AllocateArray<Qwen4ExpAttnArgs>(Config.NumLayers, pinned: true);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                if (!TryFillAttnArgs(l, ref args[l])) { _fusedAttnUnsupported = true; return false; }
            }
            _attnArgs = args;
            return true;
        }

        private bool EnsureGdnArgs()
        {
            if (_gdnArgs != null) return true;
            // PINNED for the same reason as the FFN descriptors: the kernel keys its
            // cached graph on the descriptor address.
            var args = GC.AllocateArray<Qwen4ExpGdnArgs>(Config.NumLayers, pinned: true);
            // A per-sequence holder swap may have installed pre-seeded state
            // tensors; their HOST addresses key the native device-state entries,
            // so they must be reused, never re-allocated.
            _gdnConvStateT ??= new Tensor[Config.NumLayers];
            EnsureGdnScratch();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                // The kernel wants the conv history as [d_conv-1, conv_dim] rather
                // than the host ring the op-by-op path walks.
                if (_gdnConvStateT[l] == null)
                {
                    _gdnConvStateT[l] = new Tensor(_allocator, DType.Float32, _convKernel - 1, _convDim);
                    Ops.Fill(_gdnConvStateT[l], 0f);
                }
                if (!TryFillGdnArgs(l, ref args[l])) { _fusedGdnUnsupported = true; return false; }
            }
            _gdnArgs = args;
            return true;
        }

        private bool EnsureFfnArgs()
        {
            if (_ffnArgs != null) return true;
            // PINNED: the kernel keys its cached graph on the descriptor's address, so
            // a moving array would look like a different layer every token and rebuild
            // the graph each time - exactly the cost the cache exists to remove.
            var args = GC.AllocateArray<Qwen4ExpFfnArgs>(Config.NumLayers, pinned: true);
            for (int l = 0; l < Config.NumLayers; l++)
                if (!TryFillFfnArgs(l, ref args[l])) { _fusedFfnUnsupported = true; return false; }
            _ffnArgs = args;
            return true;
        }

        /// <summary>Build the causal F16 mask at the width the kernel will read
        /// ([n_kv_pad, T]) and return that padded width. Must agree with the kernel
        /// on the padding - same predicate, same stride - because it reads exactly
        /// n_kv_pad*T entries.</summary>
        private unsafe int BuildAttnMask(int totalLen, int seqLen, int startPos, DType kvType)
        {
            int padded = totalLen;
            if (UseFlashAttn(kvType, Config.HeadDim))
            {
                padded = ((totalLen + KvStride - 1) / KvStride) * KvStride;
                if (padded > _kvCacheCapacity) padded = _kvCacheCapacity;
            }
            long need = (long)padded * seqLen;
            if (_attnMask == null || _attnMask.Length < need)
                _attnMask = new ushort[need];
            const ushort NegInfF16 = 0xFC00;
            bool qsaMask = HasQsa && _qsaPositions != null;
            // ONE fixed block around the whole fill. Pinning per row ends the pin at
            // that statement and leaves the writes going through a pointer into an
            // array the GC is free to move.
            fixed (ushort* m = _attnMask)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    int limit = startPos + t;
                    ushort* row = m + (long)t * padded;
                    for (int j = 0; j < padded; j++)
                    {
                        // QSA media cells use the same T,H,W total order as the
                        // pooled-key plan. Text reduces to the ordinary mask.
                        bool visible = qsaMask
                            ? j < totalLen && CompareQsaPositions(_qsaPositions, j, limit) <= 0
                            : j <= limit;
                        row[j] = visible ? (ushort)0 : NegInfF16;
                    }
                }
            }
            return padded;
        }

        // TS_Q4E_FUSED_GDN=0 falls back to the op-by-op recurrent half.
        // Keeping the residual in the kernels' device buffer across a layer, so the
        // fused halves chain without a host round trip each call.
        //
        // OFF by default: it measures ~18% on decode (73.3 vs 62.0 t/s) but produces
        // wrong output, and that does not buy a correctness risk.
        //
        // The earlier note here blamed the hand-off with layers still running op-by-op.
        // That is wrong. Bisected:
        //
        //   fused GDN + op-by-op FFN, resident   -> correct
        //   op-by-op GDN + fused FFN, resident   -> correct
        //   fused GDN + fused FFN,    resident   -> garbage
        //
        // So each kernel's residency is right on its own; it is the two persisted
        // graphs alternating that breaks. Forcing the residual down to host memory and
        // back between the two halves does NOT fix it, which rules out both the shared
        // device buffer hand-off and the residual values themselves - by then the data
        // has made a full round trip through the host and is provably correct. Also
        // ruled out: the ggml-cuda graph-uid stamp (TS_Q4E_GRAPH_UID=0 is byte-identical
        // to =1 here). Something the two kernels share besides the residual is being
        // disturbed; the shared g_q4e_res_buf binding and ggml-cuda's single captured
        // graph slot are the two candidates left.
        //
        // TS_Q4E_RES_RESIDENT=1 re-enables it for debugging.
        private static readonly bool _resResidentEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_RES_RESIDENT"), "1", StringComparison.Ordinal);
        private bool _resOnDevice;

        /// <summary>Bring the residual back to the host so op-by-op code can read it.</summary>
        private unsafe void ResidualToHost(Tensor res, int seqLen)
        {
            if (!_resOnDevice) return;
            long bytes = (long)seqLen * _hcDim * sizeof(float);
            if (GgmlBasicOps.Qwen4ExpResDownload((IntPtr)GetFloatPtr(res), bytes))
                InvalidateTensorDeviceCache(res);
            _resOnDevice = false;
        }

        /// <summary>Hand the residual to the device so the fused kernels can chain.</summary>
        private unsafe bool ResidualToDevice(Tensor res, int seqLen)
        {
            if (_resOnDevice) return true;
            // Never under a layer split: g_q4e_res is per-device, so a residual left
            // resident on one GPU is invisible to the next span on another. The
            // fallback that calls this is already refused under a split; this is the
            // second lock on the same door.
            if (!_resResidentEnabled || !IsGgmlBackend || LayerSplitDegree > 1) return false;
            long bytes = (long)seqLen * _hcDim * sizeof(float);
            _resOnDevice = GgmlBasicOps.Qwen4ExpResUpload((IntPtr)GetFloatPtr(res), bytes);
            return _resOnDevice;
        }

        private static readonly bool _fusedGdnEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_FUSED_GDN"), "0", StringComparison.Ordinal);
        private bool _fusedGdnUnsupported;
        private static readonly int _gdnMaxLayers =
            int.TryParse(Environment.GetEnvironmentVariable("TS_Q4E_GDN_MAX_LAYERS"), out int gm) ? gm : -1;
        private Qwen4ExpGdnArgs[] _gdnArgs;
        private Tensor[] _gdnConvStateT;

        private unsafe bool TryFusedGdnBlock(Tensor res, int il, int seqLen)
        {
            if (!_fusedGdnEnabled || _fusedGdnUnsupported || !IsGgmlBackend)
                return false;
            // Bisect handle: TS_Q4E_GDN_MAX_LAYERS=N fuses only layers below N and
            // leaves the rest op-by-op, which separates a per-layer error from one
            // that only appears once it compounds across 36 of them.
            if (_gdnMaxLayers >= 0 && il >= _gdnMaxLayers)
                return false;

            try
            {
                if (!EnsureGdnArgs()) return false;
                if (_gdnConvStateT[il] == null) return false;

                bool ok = GgmlBasicOps.Qwen4ExpGdnBlock(ref _gdnArgs[il],
                    (IntPtr)GetFloatPtr(res),
                    Config.HiddenSize, _hc, _hcLowRank, seqLen,
                    _headKDim, _headVDim, _numKHeads, _numVHeads, _convKernel,
                    Config.Eps, cacheSlot: il, resResident: _resOnDevice);
                if (!ok) { _fusedGdnUnsupported = true; return false; }
                if (!_resOnDevice) InvalidateTensorDeviceCache(res);
                _deviceStateAuthoritative = true;
                return true;
            }
            catch (Exception ex)
            {
                _fusedGdnUnsupported = true;
                WarnFusedPathDisabled("GDN", ex);
                return false;
            }
        }

        private unsafe bool TryFillGdnArgs(int il, ref Qwen4ExpGdnArgs a)
        {
            if (!TryResolveQuant($"blk.{il}.hc_attn_down.weight", out IntPtr hd, out int hdT, out long hdB)
                || !TryResolveQuant($"blk.{il}.hc_attn_up.weight", out IntPtr hu, out int huT, out long huB)
                || !TryResolveQuant($"blk.{il}.hc_attn_inject.weight", out IntPtr hj, out int hjT, out long hjB)
                || !TryResolveQuant($"blk.{il}.attn_qkv.weight", out IntPtr qv, out int qvT, out long qvB)
                || !TryResolveQuant($"blk.{il}.attn_gate.weight", out IntPtr gt, out int gtT, out long gtB)
                || !TryResolveQuant($"blk.{il}.ssm_beta.weight", out IntPtr bt, out int btT, out long btB)
                || !TryResolveQuant($"blk.{il}.ssm_alpha.weight", out IntPtr al, out int alT, out long alB)
                || !TryResolveQuant($"blk.{il}.ssm_out.weight", out IntPtr op, out int opT, out long opB))
            {
                return false;
            }
            // The conv kernel has to be F32: the graph multiplies it against F32
            // activations without a cast.
            if (!_weights.TryGetValue($"blk.{il}.ssm_conv1d.weight", out Tensor convW))
                return false;

            a.HcNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.hc_attn_norm.weight"]);
            a.HcDown = hd; a.HcDownType = hdT; a.HcDownBytes = hdB;
            a.HcUp = hu; a.HcUpType = huT; a.HcUpBytes = huB;
            a.HcInject = hj; a.HcInjectType = hjT; a.HcInjectBytes = hjB;
            a.Qkv = qv; a.QkvType = qvT; a.QkvBytes = qvB;
            a.Gate = gt; a.GateType = gtT; a.GateBytes = gtB;
            a.Beta = bt; a.BetaType = btT; a.BetaBytes = btB;
            a.Alpha = al; a.AlphaType = alT; a.AlphaBytes = alB;
            a.OutProj = op; a.OutProjType = opT; a.OutProjBytes = opB;
            a.Conv1d = (IntPtr)GetFloatPtr(convW);
            a.SsmDt = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_dt.bias"]);
            a.SsmA = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_a"]);
            a.SsmNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ssm_norm.weight"]);
            a.ConvState = (IntPtr)GetFloatPtr(_gdnConvStateT[il]);
            a.SsmState = (IntPtr)GetFloatPtr(_gdnStateT[il]);
            return true;
        }

        // TS_Q4E_FUSED_FFN=0 falls back to the op-by-op mixer + MoE + scatter,
        // which is also the automatic fallback for any shape the kernel declines.
        private static readonly bool _fusedFfnEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_FUSED_FFN"), "0", StringComparison.Ordinal);
        private bool _fusedFfnUnsupported;
        private Qwen4ExpFfnArgs[] _ffnArgs;

        private unsafe bool TryFusedFfnBlock(Tensor res, int il, int seqLen)
        {
            if (!_fusedFfnEnabled || _fusedFfnUnsupported || !IsGgmlBackend || _fusedGateUpExperts)
                return false;

            try
            {
                if (!EnsureFfnArgs()) return false;

                bool ok = GgmlBasicOps.Qwen4ExpFfnBlock(ref _ffnArgs[il],
                    (IntPtr)GetFloatPtr(res),
                    Config.HiddenSize, _hc, _hcLowRank, seqLen,
                    _numExperts, _numExpertsUsed, _expertFf, _sharedFf, Config.Eps,
                    cacheSlot: il, resResident: _resOnDevice);
                if (!ok)
                {
                    _fusedFfnUnsupported = true;
                    return false;
                }
                if (!_resOnDevice) InvalidateTensorDeviceCache(res);
                return true;
            }
            catch (Exception ex)
            {
                _fusedFfnUnsupported = true;
                WarnFusedPathDisabled("FFN", ex);
                return false;
            }
        }

        private unsafe bool TryFillFfnArgs(int il, ref Qwen4ExpFfnArgs a)
        {
            if (!_stackedExpertWeights.TryGetValue($"blk.{il}.ffn_gate_exps.weight", out var g)
                || !_stackedExpertWeights.TryGetValue($"blk.{il}.ffn_up_exps.weight", out var u)
                || !_stackedExpertWeights.TryGetValue($"blk.{il}.ffn_down_exps.weight", out var d))
            {
                return false;
            }

            if (!TryResolveQuant($"blk.{il}.hc_ffn_down.weight", out IntPtr hd, out int hdT, out long hdB)
                || !TryResolveQuant($"blk.{il}.hc_ffn_up.weight", out IntPtr hu, out int huT, out long huB)
                || !TryResolveQuant($"blk.{il}.hc_ffn_inject.weight", out IntPtr hj, out int hjT, out long hjB)
                || !TryResolveQuant($"blk.{il}.ffn_gate_inp.weight", out IntPtr rt, out int rtT, out long rtB)
                || !TryResolveQuant($"blk.{il}.ffn_gate_shexp.weight", out IntPtr sg, out int sgT, out long sgB)
                || !TryResolveQuant($"blk.{il}.ffn_up_shexp.weight", out IntPtr su, out int suT, out long suB)
                || !TryResolveQuant($"blk.{il}.ffn_down_shexp.weight", out IntPtr sd, out int sdT, out long sdB))
            {
                return false;
            }

            a.HcNorm = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.hc_ffn_norm.weight"]);
            a.HcDown = hd; a.HcDownType = hdT; a.HcDownBytes = hdB;
            a.HcUp = hu; a.HcUpType = huT; a.HcUpBytes = huB;
            a.HcInject = hj; a.HcInjectType = hjT; a.HcInjectBytes = hjB;
            a.Router = rt; a.RouterType = rtT; a.RouterBytes = rtB;
            a.GateExps = g.Data; a.GateExpsType = g.GgmlType; a.GateExpsBytes = g.TotalRawBytes;
            a.UpExps = u.Data; a.UpExpsType = u.GgmlType; a.UpExpsBytes = u.TotalRawBytes;
            a.DownExps = d.Data; a.DownExpsType = d.GgmlType; a.DownExpsBytes = d.TotalRawBytes;
            a.ShGateInp = (IntPtr)GetFloatPtr(_weights[$"blk.{il}.ffn_gate_inp_shexp.weight"]);
            a.ShGate = sg; a.ShGateType = sgT; a.ShGateBytes = sgB;
            a.ShUp = su; a.ShUpType = suT; a.ShUpBytes = suB;
            a.ShDown = sd; a.ShDownType = sdT; a.ShDownBytes = sdB;
            return true;
        }

        /// <summary>Resolve a weight to (device cache key, ggml type, raw bytes),
        /// from either its quantized or its F32 form.</summary>
        private unsafe bool TryResolveQuant(string name, out IntPtr ptr, out int type, out long bytes)
        {
            if (_quantWeights.TryGetValue(name, out var qw))
            {
                ptr = qw.CacheKey; type = qw.GgmlType; bytes = qw.RawBytes;
                return true;
            }
            if (_weights.TryGetValue(name, out var w))
            {
                ptr = (IntPtr)GetFloatPtr(w);
                type = 0;                                   // GGML_TYPE_F32
                bytes = w.ElementCount() * sizeof(float);
                return true;
            }
            ptr = IntPtr.Zero; type = 0; bytes = 0;
            return false;
        }

        private int[] _moeSelExperts;
        private float[] _moeRouteWts;
        private float[] _moeProbs;

        // Per-expert non-owning views over the stacked tensors, built on first use.
        // Only the backends without a stacked-expert kernel need them.
        private QuantizedWeight[][] _expertGateView, _expertUpView, _expertDownView;
        private Tensor _moeScratchGate, _moeScratchUp, _moeScratchDown;

        private void MoeExpertsPortable(Tensor outRow, Tensor inRow, int il, int[] expertIds, float[] routeW)
        {
            EnsureExpertViews(il);
            int n = Config.HiddenSize;
            int ff = (int)_expertGateView[il][expertIds[0]].Ne1;

            _moeScratchGate ??= new Tensor(_allocator, DType.Float32, 1, ff);
            _moeScratchUp ??= new Tensor(_allocator, DType.Float32, 1, ff);
            _moeScratchDown ??= new Tensor(_allocator, DType.Float32, 1, n);

            Ops.Fill(outRow, 0f);
            for (int k = 0; k < expertIds.Length; k++)
            {
                int e = expertIds[k];
                AddmmQuantManaged(_moeScratchGate, inRow, _expertGateView[il][e]);
                AddmmQuantManaged(_moeScratchUp, inRow, _expertUpView[il][e]);
                Ops.SiLUMul(_moeScratchGate, _moeScratchGate, _moeScratchUp);
                AddmmQuantManaged(_moeScratchDown, _moeScratchGate, _expertDownView[il][e]);
                Ops.AddMulV(outRow, outRow, _moeScratchDown, routeW[k]);
            }
        }

        private void EnsureExpertViews(int il)
        {
            _expertGateView ??= new QuantizedWeight[Config.NumLayers][];
            _expertUpView ??= new QuantizedWeight[Config.NumLayers][];
            _expertDownView ??= new QuantizedWeight[Config.NumLayers][];
            if (_expertGateView[il] != null) return;

            var g = _stackedExpertWeights[$"blk.{il}.ffn_gate_exps.weight"];
            var u = _stackedExpertWeights[$"blk.{il}.ffn_up_exps.weight"];
            var d = _stackedExpertWeights[$"blk.{il}.ffn_down_exps.weight"];
            _expertGateView[il] = new QuantizedWeight[_numExperts];
            _expertUpView[il] = new QuantizedWeight[_numExperts];
            _expertDownView[il] = new QuantizedWeight[_numExperts];
            for (int e = 0; e < _numExperts; e++)
            {
                _expertGateView[il][e] = QuantizedWeight.CreateExpertView(g, e);
                _expertUpView[il][e] = QuantizedWeight.CreateExpertView(u, e);
                _expertDownView[il][e] = QuantizedWeight.CreateExpertView(d, e);
            }
        }

        public override void Dispose()
        {
            DisposeMtpHead();
            ReleaseSpecSnapshot();
            DisposeAllFusedHolders();
            if (IsGgmlBackend)
                GgmlBasicOps.Qwen4ExpReleaseAllSeqState();
            if (_kCache != null)
                foreach (var t in _kCache) t?.Dispose();
            if (_vCache != null)
                foreach (var t in _vCache) t?.Dispose();
            if (_idxKCache != null)
                foreach (var t in _idxKCache) t?.Dispose();
            base.Dispose();
        }
    }
}
