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
    // IMtpSpeculativeModel is the Runtime-side contract BatchExecutor drives
    // for engine-path speculation; every member is implemented below or
    // inherited from ModelBase (CacheSeqLen, MaxContextLength).
    public partial class Qwen35Model : IMtpSpeculativeModel
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
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int hiddenSize = Config.HiddenSize;
            EnsureCacheCapacity(startPos + seqLen);

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

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
    }
}
