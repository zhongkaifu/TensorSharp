// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Qwen3.5/3.6 NextN/MTP (multi-token prediction) draft head.
//
// Qwen3.6 GGUFs ship one extra decoder block past the main stack (blk.N where
// N == trunk layer count) flagged by `{arch}.nextn_predict_layers`. The block
// is a standard full-attention Qwen3.5 decoder block (dense FFN on 27B, MoE
// FFN on 35B-A3B) plus four NextN-specific tensors:
//   nextn.eh_proj          [2*hidden, hidden]  input projection
//   nextn.enorm            [hidden]            RMS norm over the token embedding
//   nextn.hnorm            [hidden]            RMS norm over the trunk hidden state
//   nextn.shared_head_norm [hidden]            final norm before the LM head
// (nextn.embed_tokens / nextn.shared_head_head are optional and absent in the
// stock GGUFs; we fall back to the trunk token embedding / LM head.)
//
// The MTP step consumes (token x_p, trunk hidden h_{p-1}) at position p and
// produces logits predicting x_{p+1} plus its own hidden state used to chain
// further draft steps. This mirrors llama.cpp's graph_mtp (src/models/qwen35.cpp)
// and vLLM's Qwen3_5MultiTokenPredictor (qwen3_5_mtp.py).
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    // IMtpBatchedSpeculativeModel (extends IMtpSpeculativeModel) is the
    // Runtime-side contract BatchExecutor drives for engine-path speculation;
    // every member is implemented below or inherited from ModelBase
    // (CacheSeqLen, MaxContextLength).
    public partial class Qwen35Model : IMtpBatchedSpeculativeModel
    {
        // NextN/MTP weights (cached once at load; null when the GGUF has no MTP block).
        private QuantizedWeight _mtpEhProjQW;
        private Tensor _mtpEhProjF32;
        private Tensor _mtpEnormW;
        private Tensor _mtpHnormW;
        private Tensor _mtpHeadNormW;       // nextn.shared_head_norm (falls back to output_norm)
        private QuantizedWeight _mtpEmbdQW; // optional nextn.embed_tokens
        private Tensor _mtpEmbdF32;
        private QuantizedWeight _mtpHeadQW; // optional nextn.shared_head_head
        private Tensor _mtpHeadF32;

        // Recurrent-state snapshot used to roll the trunk back when a verify
        // batch is partially rejected (GDN state cannot be truncated in place).
        private byte[][] _mtpGdnSnapshot;

        /// <summary>
        /// True when the loaded GGUF contains a usable NextN/MTP draft block.
        /// </summary>
        public bool HasMtp { get; private set; }

        /// <summary>Trunk layer count (excludes NextN/MTP blocks).</summary>
        public int NumTrunkLayers => Config.NumLayers;

        private void CacheMtpWeights()
        {
            if (_numNextnLayers <= 0 || _mtpLayerIdx < 0)
                return;

            string p = $"blk.{_mtpLayerIdx}.";
            _quantWeights.TryGetValue(p + "nextn.eh_proj.weight", out _mtpEhProjQW);
            _weights.TryGetValue(p + "nextn.eh_proj.weight", out _mtpEhProjF32);
            _weights.TryGetValue(p + "nextn.enorm.weight", out _mtpEnormW);
            _weights.TryGetValue(p + "nextn.hnorm.weight", out _mtpHnormW);
            _weights.TryGetValue(p + "nextn.shared_head_norm.weight", out _mtpHeadNormW);
            _quantWeights.TryGetValue(p + "nextn.embed_tokens.weight", out _mtpEmbdQW);
            _weights.TryGetValue(p + "nextn.embed_tokens.weight", out _mtpEmbdF32);
            _quantWeights.TryGetValue(p + "nextn.shared_head_head.weight", out _mtpHeadQW);
            _weights.TryGetValue(p + "nextn.shared_head_head.weight", out _mtpHeadF32);

            bool hasProj = _mtpEhProjQW != null || _mtpEhProjF32 != null;
            bool hasAttn = _attnQkvQW[_mtpLayerIdx] != null || _attnQkvF32[_mtpLayerIdx] != null
                || _attnQQW[_mtpLayerIdx] != null || _attnQF32[_mtpLayerIdx] != null;
            HasMtp = _numNextnLayers == 1 && hasProj && _mtpEnormW != null && _mtpHnormW != null
                && hasAttn && _attnNormW[_mtpLayerIdx] != null && _postAttnNormW[_mtpLayerIdx] != null;

            if (_numNextnLayers > 0 && !HasMtp)
                Console.WriteLine("  NextN/MTP block present but incomplete; MTP drafting disabled.");
            else if (HasMtp)
                Console.WriteLine($"  NextN/MTP draft head ready (layer {_mtpLayerIdx}, " +
                    $"moe={( _isMoeLayer != null && _isMoeLayer[_mtpLayerIdx] ? "yes" : "no")}, " +
                    $"ownHead={(_mtpHeadQW != null || _mtpHeadF32 != null ? "yes" : "no")})");
        }

        /// <summary>
        /// Token embedding lookup for the MTP block: prefers nextn.embed_tokens
        /// when shipped, otherwise reuses the trunk token embedding.
        /// </summary>
        private Tensor MtpEmbedding(int[] tokens)
        {
            if (_mtpEmbdQW != null && _mtpEmbdQW.HasHostData)
            {
                var result = new Tensor(_allocator, DType.Float32, tokens.Length, Config.HiddenSize);
                PopulateQuantizedRows(result, _mtpEmbdQW, tokens);
                return result;
            }
            return Embedding(tokens);
        }

        /// <summary>
        /// Shared MTP core: projects (token, previous trunk hidden) pairs into the
        /// MTP decoder block and runs it (updating the MTP block's KV cache rows at
        /// [startPos, startPos+n)). Returns the block output [n, hidden] BEFORE the
        /// shared head norm.
        /// <paramref name="hRows"/> holds n rows of post-final-norm trunk hidden
        /// states; row k must be the hidden state of the token PRECEDING tokens[k].
        /// </summary>
        private unsafe Tensor MtpForwardCore(int[] tokens, float[] hRows, int startPos)
        {
            int n = tokens.Length;
            int hidden = Config.HiddenSize;
            EnsureCacheCapacity(startPos + n);

            Tensor emb = MtpEmbedding(tokens);
            Tensor eNorm = RMSNormOpCached(emb, _mtpEnormW);
            emb.Dispose();

            // hRows may be a reusable buffer larger than n*hidden; copy exactly
            // the rows we need (SetElementsAsFloat would write value.Length
            // elements and overrun the tensor allocation).
            var h = new Tensor(_allocator, DType.Float32, n, hidden);
            fixed (float* src = hRows)
            {
                float* dst = GetFloatPtr(h);
                Buffer.MemoryCopy(src, dst, (long)n * hidden * 4, (long)n * hidden * 4);
            }
            InvalidateTensorDeviceCache(h);
            Tensor hNorm = RMSNormOpCached(h, _mtpHnormW);
            h.Dispose();

            // concat([e_norm, h_norm], featureDim) -> eh_proj -> [n, hidden]
            var cat = new Tensor(_allocator, DType.Float32, n, 2L * hidden);
            using (var dstE = cat.Narrow(1, 0, hidden))
                Ops.Copy(dstE, eNorm);
            using (var dstH = cat.Narrow(1, hidden, hidden))
                Ops.Copy(dstH, hNorm);
            eNorm.Dispose();
            hNorm.Dispose();

            Tensor x = LinearForwardCached(cat, _mtpEhProjQW, _mtpEhProjF32);
            cat.Dispose();

            // Full decoder block (attention + FFN/MoE with residuals); reuses the
            // trunk machinery — the MTP layer's weights/KV live at _mtpLayerIdx.
            x = AttentionBlock(x, _mtpLayerIdx, n, startPos);
            return x;
        }

        /// <summary>
        /// One MTP draft step: consume (token, hPrev) at <paramref name="pos"/>,
        /// fill <paramref name="logitsOut"/> (vocab floats) with next-token logits
        /// and <paramref name="hOut"/> (hidden floats) with the MTP hidden state
        /// used to chain the next draft step.
        /// </summary>
        public unsafe void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            if (!HasMtp)
                throw new InvalidOperationException("Model has no NextN/MTP draft block.");
            EnterSpecSession();

            Tensor x = MtpForwardCore(new[] { token }, hPrev, pos);

            Tensor headNorm = _mtpHeadNormW ?? _finalNormW;
            Tensor hn = RMSNormOpCached(x, headNorm);
            x.Dispose();

            fixed (float* dst = hOut)
            {
                float* src = GetFloatPtr(hn);
                Buffer.MemoryCopy(src, dst, (long)hOut.Length * 4, (long)Config.HiddenSize * 4);
            }

            // nextn.shared_head_head when shipped, otherwise the trunk LM head.
            bool hasOwnHead = _mtpHeadQW != null || _mtpHeadF32 != null;
            QuantizedWeight headQW = hasOwnHead ? _mtpHeadQW : _lmHeadQW;
            Tensor headF32 = hasOwnHead ? _mtpHeadF32 : _lmHeadF32;
            Tensor logitsT = LinearForwardCached(hn, headQW, headF32);
            hn.Dispose();

            fixed (float* dst = logitsOut)
            {
                float* src = GetFloatPtr(logitsT);
                Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
            }
            logitsT.Dispose();
        }

        /// <summary>
        /// MTP catch-up pass (llama.cpp's draft-mtp process()): replays verified
        /// trunk tokens through the MTP block so its KV cache stays in sync with
        /// exact trunk hidden states. Logits are not needed — only the KV side
        /// effects matter.
        /// </summary>
        public void MtpCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (!HasMtp)
                throw new InvalidOperationException("Model has no NextN/MTP draft block.");
            EnterSpecSession();
            Tensor x = MtpForwardCore(tokens, hRows, startPos);
            x.Dispose();
        }

        /// <summary>
        /// Trunk forward for speculative decoding. Identical math to Forward()
        /// but additionally captures the post-final-norm hidden state of every
        /// row into <paramref name="hAllOut"/> (n*hidden floats; llama.cpp's
        /// h_nextn) and, when <paramref name="allLogitsRows"/> is set, computes
        /// LM-head logits for every row into <paramref name="logitsOut"/>
        /// (n*vocab floats) instead of only the last row.
        /// Advances the KV caches and _cacheSeqLen exactly like Forward().
        /// </summary>
        // SpecForward layer-type timing (speculative-path profiling; cheap
        // enough to keep always-on: one timestamp per layer per pass).
        public long SpecAttnLayerTicks { get; private set; }
        public long SpecRecurrentLayerTicks { get; private set; }
        public long SpecLmHeadTicks { get; private set; }
        public void ResetSpecLayerTimings()
        {
            SpecAttnLayerTicks = SpecRecurrentLayerTicks = SpecLmHeadTicks = 0;
        }

        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            EnterSpecSession();
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int hiddenSize = Config.HiddenSize;
            EnsureCacheCapacity(startPos + seqLen);

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            // Fast path: run the whole trunk over the N tokens as ONE fused GGML
            // graph (TSGgml_Qwen35ModelVerify) instead of the op-by-op layer loop.
            // Writes hAllOut (post-norm hidden) + logitsOut directly; advances KV +
            // GDN state by N. Env-gated TS_QWEN35_FUSED_VERIFY (default off).
            if (TryFusedVerifyTrunk(hidden, startPos, seqLen, hAllOut, logitsOut, allLogitsRows))
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return;
            }

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                long tl = Stopwatch.GetTimestamp();
                if (_isRecurrent[layer])
                {
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                    SpecRecurrentLayerTicks += Stopwatch.GetTimestamp() - tl;
                }
                else
                {
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
                    SpecAttnLayerTicks += Stopwatch.GetTimestamp() - tl;
                }
                TryEvaluateMlxLayerBoundary(hidden, layer, seqLen);
            }

            // Final norm over ALL rows (the MTP draft head consumes per-row
            // post-norm hidden states, llama.cpp's t_h_nextn).
            Tensor normed = RMSNormOpCached(hidden, _finalNormW);
            hidden.Dispose();

            if (hAllOut != null)
            {
                fixed (float* dst = hAllOut)
                {
                    float* src = GetFloatPtr(normed);
                    Buffer.MemoryCopy(src, dst, (long)hAllOut.Length * 4, (long)seqLen * hiddenSize * 4);
                }
            }

            long t2 = Stopwatch.GetTimestamp();
            if (allLogitsRows)
            {
                Tensor logitsT = LinearForwardCached(normed, _lmHeadQW, _lmHeadF32);
                normed.Dispose();
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)seqLen * Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }
            else
            {
                Tensor lastRow;
                if (seqLen > 1)
                {
                    using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                    lastRow = Ops.NewContiguous(narrowed);
                    normed.Dispose();
                }
                else
                {
                    lastRow = normed;
                }
                Tensor logitsT = LinearForwardCached(lastRow, _lmHeadQW, _lmHeadF32);
                lastRow.Dispose();
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            SpecLmHeadTicks += Stopwatch.GetTimestamp() - t2;

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
        }

        private float[] _fvLogitsAllBuf;

        /// <summary>Fused trunk verify fast path for <see cref="SpecForward"/>: the
        /// kernel always produces all-N logit rows, so route them to logitsOut
        /// directly when the caller wants all rows, else stage in a reusable buffer
        /// and copy the last row. Returns false (op-by-op fallback) when disabled or
        /// the kernel declines the shape.</summary>
        private bool TryFusedVerifyTrunk(Tensor hidden, int startPos, int seqLen,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (!_fusedVerifyEnabled)
                return false;
            int vocab = Config.VocabSize;
            float[] allLogits;
            if (allLogitsRows && logitsOut != null && logitsOut.Length >= (long)seqLen * vocab)
            {
                allLogits = logitsOut;
            }
            else
            {
                if (_fvLogitsAllBuf == null || _fvLogitsAllBuf.Length < (long)seqLen * vocab)
                    _fvLogitsAllBuf = new float[(long)seqLen * vocab];
                allLogits = _fvLogitsAllBuf;
            }
            if (!TryFullModelVerify(hidden, startPos, seqLen, hAllOut, allLogits))
                return false;
            if (!allLogitsRows && logitsOut != null)
                Array.Copy(allLogits, (long)(seqLen - 1) * vocab, logitsOut, 0, vocab);
            return true;
        }

        /// <summary>
        /// Grow the KV caches up front to cover a full speculative window.
        /// EnsureCacheCapacity's growth path only preserves rows below
        /// _cacheSeqLen, so growing mid-draft would drop the MTP rows written
        /// past the trunk position; callers pre-grow before drafting instead.
        /// </summary>
        public void MtpEnsureCapacity(int requiredSeqLen) => EnsureCacheCapacity(requiredSeqLen);

        /// <summary>
        /// Snapshot the GDN recurrent state of every trunk layer. Taken right
        /// before a speculative verify batch so a partial rejection can roll the
        /// recurrent state back (attention KV needs only a position rewind).
        /// </summary>
        public void MtpSnapshotRecurrentState()
        {
            // Direct-CUDA fast path: snapshot the GDN state device-to-device
            // (async cuMemcpyDtoD on the stream) instead of draining it to host
            // bytes. The host path does an EnsureHostReadable DtoH per recurrent
            // layer (48 syncs) every verify step, but the snapshot is only ever
            // consumed on a partial-rejection rollback (~1 in this model) -- so
            // those DtoH stalls were almost entirely wasted on a sync-bound
            // backend.
            if (_backend == BackendType.Cuda)
            {
                MtpSnapshotRecurrentStateCudaDevice();
                return;
            }

            _mtpGdnSnapshot ??= new byte[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                long bytes = GdnLayerStateBytes(l);
                if (_mtpGdnSnapshot[l] == null || _mtpGdnSnapshot[l].Length != bytes)
                    _mtpGdnSnapshot[l] = new byte[bytes];
                if (!CopyGdnStateOut(l, _mtpGdnSnapshot[l], out _))
                    throw new InvalidOperationException($"Failed to snapshot GDN state for layer {l}.");
            }
        }

        /// <summary>Restore the GDN recurrent state captured by <see cref="MtpSnapshotRecurrentState"/>.</summary>
        public void MtpRestoreRecurrentState()
        {
            if (_backend == BackendType.Cuda)
            {
                MtpRestoreRecurrentStateCudaDevice();
                return;
            }

            if (_mtpGdnSnapshot == null)
                throw new InvalidOperationException("No GDN snapshot to restore.");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                if (!CopyGdnStateIn(l, _mtpGdnSnapshot[l], out _))
                    throw new InvalidOperationException($"Failed to restore GDN state for layer {l}.");
            }
        }

        // Reusable device-resident GDN snapshot buffers for the CUDA linear-trunk
        // path (conv ring buffer + SSM/delta state + conv ring write index).
        private Tensor[] _mtpGdnConvDevSnap;
        private Tensor[] _mtpGdnDeltaDevSnap;
        private int[] _mtpGdnConvIdxDevSnap;

        private void MtpSnapshotRecurrentStateCudaDevice()
        {
            int layers = Config.NumLayers;
            _mtpGdnDeltaDevSnap ??= new Tensor[layers];
            _mtpGdnConvDevSnap ??= new Tensor[layers];
            _mtpGdnConvIdxDevSnap ??= new int[layers];
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                Tensor delta = _deltaStateTensor[l];
                _mtpGdnDeltaDevSnap[l] ??= new Tensor(_allocator, delta.ElementType, _numVHeads, _headVDim, _headKDim);
                Ops.Copy(_mtpGdnDeltaDevSnap[l], delta);

                Tensor conv = _cudaGdnConvStateTensor?[l];
                if (conv != null)
                {
                    _mtpGdnConvDevSnap[l] ??= new Tensor(_allocator, conv.ElementType, conv.Sizes[0], conv.Sizes[1]);
                    Ops.Copy(_mtpGdnConvDevSnap[l], conv);
                }
                _mtpGdnConvIdxDevSnap[l] = _convStateWriteIdx[l];
            }
        }

        private void MtpRestoreRecurrentStateCudaDevice()
        {
            if (_mtpGdnDeltaDevSnap == null)
                throw new InvalidOperationException("No GDN snapshot to restore.");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                // Ops.Copy is device-to-device and MarkDeviceModified(), so the
                // recurrence kernel reads the restored device state directly (the
                // stale host mirror is never re-uploaded over it).
                Ops.Copy(_deltaStateTensor[l], _mtpGdnDeltaDevSnap[l]);
                if (_cudaGdnConvStateTensor?[l] != null && _mtpGdnConvDevSnap[l] != null)
                    Ops.Copy(_cudaGdnConvStateTensor[l], _mtpGdnConvDevSnap[l]);
                _convStateWriteIdx[l] = _mtpGdnConvIdxDevSnap[l];
            }
        }

        /// <summary>
        /// Rewind the attention KV position counter after rejected speculative
        /// tokens. Rows past <paramref name="length"/> are dead weight that the
        /// next forward simply overwrites (the causal mask never reads past the
        /// current position), so no data movement is needed.
        /// </summary>
        public void MtpRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            _cacheSeqLen = length;
        }

        // ====================================================================
        // Batched-trunk speculative decoding (IMtpBatchedSpeculativeModel):
        // trunk passes run through ForwardBatch (paged KV via the sequence's
        // block table, per-slot GDN state) so speculation rides the same
        // kernels as the non-speculative batched baseline. The MTP draft head
        // above is unchanged — it runs on the linear cache at _mtpLayerIdx,
        // which is private to the speculative context.
        // ====================================================================

        // Per-slot GDN snapshot used to roll back a partially-rejected verify
        // batch: per recurrent layer, ONE slot's conv ring buffer + write
        // index + SSM state + init flag.
        private float[][] _mtpSlotConvSnapshot;
        private int[] _mtpSlotConvIdxSnapshot;
        private float[][] _mtpSlotSsmSnapshot;
        private bool[] _mtpSlotInitSnapshot;
        private int _mtpSlotSnapshotSlot = -1;

        /// <summary>
        /// MTP speculation is only a throughput WIN when the model's STANDARD
        /// decode is slow enough that drafting + verifying K tokens saves more
        /// than the speculation machinery costs. On <c>ggml_cuda</c> the standard
        /// Qwen3.6 decode IS the fused, CUDA-graph-captured whole-model decode
        /// (<see cref="TryFullModelDecode"/>): one graph replay per token, fully
        /// device-resident GDN/KV state, zero host orchestration — ~73 tok/s
        /// (~13.7 ms/token) on 35B-A3B IQ2_XXS, already at the memory-bandwidth
        /// floor (only ~3B of 35B params are active, so each decode token is cheap).
        ///
        /// <c>--mtp-spec</c> is an EXPLICIT operator opt-in, so we honor it whenever
        /// the model actually has an MTP/NextN head — even on ggml_cuda where the
        /// captured decode (~73 tok/s) may still beat speculation. (Earlier this gated
        /// OFF on ggml_cuda because the op-by-op verify made MTP ~34x slower; the
        /// fused multi-token verify <see cref="TryFullModelVerify"/>,
        /// <c>TS_QWEN35_FUSED_VERIFY=1</c>, cuts the verify ~20x and is the path that
        /// makes spec competitive — long context / large drafts.) The user asked that
        /// the flag be respected regardless, so don't second-guess it here.
        /// </summary>
        public bool MtpSpeculationProfitable => HasMtp;

        /// <summary>Batched spec trunk needs the GGML batched paged path (the
        /// MLX backend keeps GDN state inside opaque per-slot MLX caches the
        /// snapshot/restore below cannot capture). When the fused multi-token
        /// verify (<see cref="TryFullModelVerify"/>) is enabled we route spec to the
        /// LINEAR trunk instead (SpecForward), whose KV/GDN state the fused verify
        /// reads/writes; the batched paged trunk uses a different (paged) store.</summary>
        public bool SupportsBatchedSpecTrunk => HasMtp && IsGgmlBackend && IsBatchedPathEnabled() && !_fusedVerifyEnabled;

        public void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            EnterSpecSession();
            ArgumentNullException.ThrowIfNull(seq);
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Tokens must not be empty.", nameof(tokens));
            // The batched path reads the sequence's committed length for its
            // attention extents, so every spec pass must start exactly there
            // (the executor advances the sequence only after the step).
            if (startPos != seq.NumComputedTokens)
                throw new InvalidOperationException(
                    $"SpecForwardBatched at position {startPos} but sequence has {seq.NumComputedTokens} computed tokens.");

            int n = tokens.Length;
            var bt = seq.BlockTable;
            if (bt.CapacityTokens < startPos + n)
                throw new InvalidOperationException(
                    $"Block table covers {bt.CapacityTokens} tokens but the spec pass needs {startPos + n}.");

            var positions = new System.Collections.Generic.List<int>(n);
            var slotMapping = new System.Collections.Generic.List<int>(n);
            for (int i = 0; i < n; i++)
            {
                int pos = startPos + i;
                positions.Add(pos);
                int blockIdx = pos / bt.BlockSize;
                slotMapping.Add(bt.Blocks[blockIdx].Id * bt.BlockSize + pos % bt.BlockSize);
            }
            var table = new int[bt.NumBlocks];
            for (int b = 0; b < bt.NumBlocks; b++)
                table[b] = bt.Blocks[b].Id;

            var ctx = new BatchedForwardContext
            {
                Sequences = new System.Collections.Generic.List<SequenceState> { seq },
                NumScheduledTokens = new System.Collections.Generic.List<int> { n },
                QueryStartLoc = new System.Collections.Generic.List<int> { 0, n },
                Positions = positions,
                SlotMapping = slotMapping,
                BlockTables = new[] { table },
                MaxQueryLen = n,
                MaxSeqLen = startPos + n,
                OverrideFlatTokens = tokens,
                CaptureHiddenAll = hAllOut,
                CaptureLogitsAll = allLogitsRows ? logitsOut : null,
            };

            var perSeq = ForwardBatch(ctx);
            if (!allLogitsRows && logitsOut != null)
                Array.Copy(perSeq[0], logitsOut, Config.VocabSize);
        }

        public unsafe void MtpSnapshotRecurrentStateSlots(SequenceState seq)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (_q35GdnSlotConvBuf == null)
                throw new InvalidOperationException(
                    "Batched GDN slot state not initialized (no batched forward has run for this sequence yet).");

            int slot = seq.BlockTable.Blocks[0].Id;
            int layers = Config.NumLayers;
            _mtpSlotConvSnapshot ??= new float[layers][];
            _mtpSlotSsmSnapshot ??= new float[layers][];
            _mtpSlotConvIdxSnapshot ??= new int[layers];
            _mtpSlotInitSnapshot ??= new bool[layers];

            int ssmLen = _numVHeads * _headVDim * _headKDim;
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                EnsureGdnSlotAllocated(l, slot);
                float[] conv = _q35GdnSlotConvBuf[l][slot];
                if (_mtpSlotConvSnapshot[l] == null || _mtpSlotConvSnapshot[l].Length != conv.Length)
                    _mtpSlotConvSnapshot[l] = new float[conv.Length];
                Array.Copy(conv, _mtpSlotConvSnapshot[l], conv.Length);
                _mtpSlotConvIdxSnapshot[l] = _q35GdnSlotConvWriteIdx[l][slot];
                _mtpSlotInitSnapshot[l] = _q35GdnSlotInit[l][slot];
                // Pointer copy into a reused buffer: GetElementsAsFloat would
                // allocate a fresh ~3 MB array per layer per verify step
                // (gigabytes of GC churn per request — measured 92 ms/step
                // vs ~12 ms for the raw copy).
                if (_mtpSlotSsmSnapshot[l] == null || _mtpSlotSsmSnapshot[l].Length != ssmLen)
                    _mtpSlotSsmSnapshot[l] = new float[ssmLen];
                float* src = GetFloatPtr(_q35GdnSlotSsmTensor[l][slot]);
                fixed (float* dst = _mtpSlotSsmSnapshot[l])
                    Buffer.MemoryCopy(src, dst, (long)ssmLen * 4, (long)ssmLen * 4);
            }
            _mtpSlotSnapshotSlot = slot;
        }

        public unsafe void MtpRestoreRecurrentStateSlots(SequenceState seq)
        {
            ArgumentNullException.ThrowIfNull(seq);
            int slot = seq.BlockTable.Blocks[0].Id;
            if (_mtpSlotSnapshotSlot != slot)
                throw new InvalidOperationException(
                    $"No recurrent-state snapshot for slot {slot} (snapshot holds slot {_mtpSlotSnapshotSlot}).");

            int ssmLen = _numVHeads * _headVDim * _headKDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                Array.Copy(_mtpSlotConvSnapshot[l], _q35GdnSlotConvBuf[l][slot], _mtpSlotConvSnapshot[l].Length);
                _q35GdnSlotConvWriteIdx[l][slot] = _mtpSlotConvIdxSnapshot[l];
                _q35GdnSlotInit[l][slot] = _mtpSlotInitSnapshot[l];
                Tensor ssm = _q35GdnSlotSsmTensor[l][slot];
                float* dst = GetFloatPtr(ssm);
                fixed (float* src = _mtpSlotSsmSnapshot[l])
                    Buffer.MemoryCopy(src, dst, (long)ssmLen * 4, (long)ssmLen * 4);
                InvalidateTensorDeviceCache(ssm);
            }
        }
    }
}
