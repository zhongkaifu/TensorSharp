// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// UMT5-XXL text encoder for Wan video generation. Tokenizes with the GGUF's
// SentencePiece vocabulary (T5 unigram pieces) and runs the 24-layer encoder as
// ONE native ggml graph (TSGgml_WanT5Encode); the big projections stay in their
// GGUF quantization (resident-cached by mmap pointer), the norms and per-layer
// relative-attention bias tables go in as F32.
//
// Semantics mirror the Wan pipeline exactly (diffusers WanPipeline /
// stable-diffusion.cpp T5CLIPEmbedder with zero_out_masked=true): tokens are
// truncated/padded to text_len=512 with EOS appended and pad id 0, the encoder
// attends only to the real tokens (additive -inf mask), and the hidden states at
// padded positions are zeroed before the DiT consumes all 512 positions unmasked.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanTextEncoder : IWanTextEncoder, IGgmlHostMemoryOwner
    {
        public const int TextLen = 512;

        private readonly GgufFile _gguf;
        internal GgufFile Gguf => _gguf;
        private readonly UnigramTokenizer _tokenizer;
        private readonly bool _addSpacePrefix;
        private readonly int _eosId;
        private readonly int _padId;

        public int Dim { get; }
        public int Layers { get; }
        public int Heads { get; }
        public int HeadDim { get; }
        public int Ff { get; }

        // Stable F32 copies for small tensors the native kernel reads as F32
        // (norm gains, relative-attention bias) when the GGUF stores them
        // in another dtype. Keyed by tensor name; freed on Dispose.
        private readonly Dictionary<string, IntPtr> _f32Copies = new();
        private readonly List<IntPtr> _f32Owned = new();

        public WanTextEncoder(string tePath)
        {
            _gguf = new GgufFile(tePath);

            string arch = _gguf.GetString("general.architecture") ?? "";
            if (arch != "t5encoder" && arch != "t5")
                throw new InvalidDataException(
                    $"Wan text encoder '{Path.GetFileName(tePath)}' has architecture '{arch}'; expected a " +
                    "umt5-xxl encoder GGUF (general.architecture = t5encoder), e.g. " +
                    "city96/umt5-xxl-encoder-gguf.");

            Dim = (int)_gguf.GetUint32("t5encoder.embedding_length", 4096);
            Layers = (int)_gguf.GetUint32("t5encoder.block_count", 24);
            Heads = (int)_gguf.GetUint32("t5encoder.attention.head_count", 64);
            HeadDim = (int)_gguf.GetUint32("t5encoder.attention.key_length", 64);
            Ff = (int)_gguf.GetUint32("t5encoder.feed_forward_length", 10240);

            var vocab = _gguf.GetStringArray("tokenizer.ggml.tokens")
                ?? throw new InvalidDataException("UMT5 GGUF has no tokenizer vocabulary.");
            var scores = _gguf.GetFloatArray("tokenizer.ggml.scores");
            var types = _gguf.GetInt32Array("tokenizer.ggml.token_type");
            _eosId = (int)_gguf.GetUint32("tokenizer.ggml.eos_token_id", 1);
            _padId = (int)_gguf.GetUint32("tokenizer.ggml.padding_token_id", 0);
            _addSpacePrefix = _gguf.GetBool("tokenizer.ggml.add_space_prefix", true);
            int unkId = (int)_gguf.GetUint32("tokenizer.ggml.unknown_token_id", 2);
            _tokenizer = new UnigramTokenizer(vocab, scores, types, unkId);
        }

        /// <summary>Tokenize + truncate/pad to <see cref="TextLen"/>. Returns the padded
        /// token ids and the number of real tokens (prompt + EOS).</summary>
        internal int[] PrepareTokens(string text, out int realLen)
        {
            // T5's SentencePiece normalization adds a dummy space prefix so the first
            // word gets its "▁word" piece.
            string t = text ?? string.Empty;
            if (_addSpacePrefix && t.Length > 0 && !t.StartsWith(" ", StringComparison.Ordinal))
                t = " " + t;
            List<int> ids = _tokenizer.Encode(t);
            if (ids.Count > TextLen - 1)
                ids.RemoveRange(TextLen - 1, ids.Count - (TextLen - 1));
            ids.Add(_eosId);
            realLen = ids.Count;
            var padded = new int[TextLen];
            for (int i = 0; i < TextLen; i++) padded[i] = i < ids.Count ? ids[i] : _padId;
            return padded;
        }

        /// <summary>Encode a prompt to the Wan conditioning: [TextLen, Dim] hidden states,
        /// zero at padded positions.</summary>
        public unsafe float[] Encode(string text)
        {
            int[] tokens = PrepareTokens(text, out int realLen);
            int n = TextLen;

            string dumpDir = Environment.GetEnvironmentVariable("TS_WAN_DEBUG_DUMP");
            if (!string.IsNullOrEmpty(dumpDir))
            {
                Directory.CreateDirectory(dumpDir);
                var bytes = new byte[tokens.Length * 4];
                Buffer.BlockCopy(tokens, 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(Path.Combine(dumpDir, $"tokens_{realLen}.bin"), bytes);
                Console.WriteLine($"  [wan-debug] tokens ({realLen} real): {string.Join(",", tokens.AsSpan(0, Math.Min(realLen, 32)).ToArray())}");
            }

            int[] buckets = ComputeRelativeBuckets(n);
            var mask = new float[n];
            for (int i = realLen; i < n; i++) mask[i] = float.NegativeInfinity;

            var layers = new WanT5LayerW[Layers];
            for (int l = 0; l < Layers; l++)
            {
                string p = $"enc.blk.{l}.";
                layers[l] = new WanT5LayerW
                {
                    Q = W($"{p}attn_q.weight"),
                    K = W($"{p}attn_k.weight"),
                    V = W($"{p}attn_v.weight"),
                    O = W($"{p}attn_o.weight"),
                    Gate = W($"{p}ffn_gate.weight"),
                    Up = W($"{p}ffn_up.weight"),
                    Down = W($"{p}ffn_down.weight"),
                    AttnNorm = F32Ptr($"{p}attn_norm.weight"),
                    FfnNorm = F32Ptr($"{p}ffn_norm.weight"),
                    RelB = F32Ptr($"{p}attn_rel_b.weight"),
                };
            }

            var outHidden = new float[(long)n * Dim];
            var handles = new List<GCHandle>();
            IntPtr Pin<T>(T[] a) where T : struct
            {
                var h = GCHandle.Alloc(a, GCHandleType.Pinned);
                handles.Add(h);
                return h.AddrOfPinnedObject();
            }
            try
            {
                var d = new WanT5EncodeArgs
                {
                    Tokens = Pin(tokens),
                    RelBucket = Pin(buckets),
                    AttnMask = Pin(mask),
                    TokEmbd = W("token_embd.weight"),
                    Layers = Pin(layers),
                    FinalNorm = F32Ptr("enc.output_norm.weight"),
                    Out = Pin(outHidden),
                    NTokens = n,
                    NumLayers = Layers,
                    Dim = Dim,
                    Ff = Ff,
                    Heads = Heads,
                    HeadDim = HeadDim,
                    Eps = 1e-6f,
                    StructBytes = Marshal.SizeOf<WanT5EncodeArgs>(),
                };
                if (!GgmlBasicOps.TryWanT5Encode(in d))
                    throw new InvalidOperationException("Wan UMT5 encoder forward failed (see native error above).");
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }

            // zero_out_masked: the DiT cross-attends over all 512 positions with no
            // mask, so padded positions must carry exact zeros.
            Array.Clear(outHidden, realLen * Dim, (n - realLen) * Dim);
            return outHidden;
        }

        // T5 bidirectional relative position buckets (num_buckets=32, max_distance=128);
        // bucket[q*n + k] for relative_position = k - q. Port of transformers
        // T5Attention._relative_position_bucket.
        internal static int[] ComputeRelativeBuckets(int n, int numBuckets = 32, int maxDistance = 128)
        {
            var b = new int[n * n];
            int half = numBuckets / 2;
            int maxExact = half / 2;
            for (int q = 0; q < n; q++)
            {
                for (int k = 0; k < n; k++)
                {
                    int rel = k - q;
                    int bucket = rel > 0 ? half : 0;
                    int a = Math.Abs(rel);
                    int v;
                    if (a < maxExact)
                        v = a;
                    else
                    {
                        double logPos = Math.Log((double)a / maxExact);
                        double logBase = Math.Log((double)maxDistance / maxExact);
                        v = maxExact + (int)(logPos / logBase * (half - maxExact));
                        v = Math.Min(v, half - 1);
                    }
                    b[q * n + k] = bucket + v;
                }
            }
            return b;
        }

        private WanW W(string name)
        {
            var info = _gguf.Tensors[name];
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr p))
                throw new InvalidDataException($"UMT5 tensor '{name}' has no mmap pointer.");
            long ne0 = (long)info.Shape[0];
            long ne1 = 1;
            for (int i = 1; i < info.Shape.Length; i++) ne1 *= (long)info.Shape[i];
            return new WanW
            {
                W = p,
                Type = (int)info.Type,
                Ne0 = ne0,
                Ne1 = ne1,
                Bytes = _gguf.GetTensorByteCount(info),
            };
        }

        // Stable F32 pointer for small tensors: zero-copy mmap for F32, else a
        // one-time dequantized copy (the native side resident-caches by pointer,
        // so the address must not move).
        private IntPtr F32Ptr(string name)
        {
            lock (_f32Copies)
            {
                if (_f32Copies.TryGetValue(name, out IntPtr cached)) return cached;
                var info = _gguf.Tensors[name];
                IntPtr result;
                if (info.Type == GgmlTensorType.F32)
                {
                    _gguf.TryGetTensorDataPointer(info, out result);
                }
                else
                {
                    // Non-F32 (F16/BF16/quantized): proper dequant, not a raw byte copy.
                    long nel = info.NumElements;
                    long srcBytes = _gguf.GetTensorByteCount(info);
                    result = Marshal.AllocHGlobal((IntPtr)(nel * sizeof(float)));
                    _f32Owned.Add(result);
                    IntPtr tmp = Marshal.AllocHGlobal((IntPtr)srcBytes);
                    try
                    {
                        _gguf.ReadTensorDataToNative(info, tmp, srcBytes);
                        NativeDequant.DequantizeToFloat32Native((int)info.Type, tmp, result, nel);
                    }
                    finally { Marshal.FreeHGlobal(tmp); }
                }
                _f32Copies[name] = result;
                return result;
            }
        }

        /// <summary>This encoder's GGUF mapping and the F32 copies it made.</summary>
        void IGgmlHostMemoryOwner.CollectHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys)
        {
            GgmlOwnerRelease.AddMappings(ranges, _gguf);
            keys.AddRange(_f32Owned);
        }

        public void Dispose()
        {
            foreach (var p in _f32Owned)
            {
                GgmlBasicOps.InvalidateHostBuffer(p);   // see WanVae.Dispose
                Marshal.FreeHGlobal(p);
            }
            _f32Owned.Clear();
            _f32Copies.Clear();
            _gguf?.Dispose();
        }
    }

    /// <summary>
    /// SentencePiece unigram tokenizer (Viterbi over piece log-probabilities) for the
    /// T5 vocabulary. The shared <see cref="SentencePieceTokenizer"/> approximates
    /// segmentation with greedy score merging, which diverges from the reference
    /// tokenization on real prompts (e.g. "fluffy" -&gt; wrong pieces); the conditioning
    /// encoder needs the exact ids the model was trained with, so this implements the
    /// true unigram DP the HuggingFace T5 tokenizer uses.
    /// </summary>
    internal sealed class UnigramTokenizer
    {
        private const int TokenTypeNormal = 1;

        private readonly Dictionary<string, (int Id, float Score)> _pieces;
        private readonly int _maxPieceChars;
        private readonly int _unkId;
        private readonly float _unkScore;

        public UnigramTokenizer(string[] vocab, float[] scores, int[] tokenTypes, int unkId)
        {
            _pieces = new Dictionary<string, (int, float)>(vocab.Length, StringComparer.Ordinal);
            float minScore = 0f;
            int maxLen = 1;
            for (int i = 0; i < vocab.Length; i++)
            {
                // Only NORMAL pieces participate in segmentation (control/unknown/user
                // tokens would otherwise swallow text; llama.cpp's UGM does the same).
                if (tokenTypes != null && i < tokenTypes.Length && tokenTypes[i] != TokenTypeNormal)
                    continue;
                float score = scores != null && i < scores.Length ? scores[i] : 0f;
                if (!_pieces.ContainsKey(vocab[i]))
                    _pieces[vocab[i]] = (i, score);
                if (vocab[i].Length > maxLen) maxLen = vocab[i].Length;
                if (score < minScore) minScore = score;
            }
            _maxPieceChars = Math.Min(maxLen, 64);
            _unkId = unkId;
            _unkScore = minScore - 10f;
        }

        /// <summary>Tokenize (no specials added). Whitespace becomes U+2581 first.</summary>
        public List<int> Encode(string text)
        {
            string s = (text ?? string.Empty).Replace(' ', '▁');
            int n = s.Length;
            var ids = new List<int>();
            if (n == 0) return ids;

            // Viterbi: best[i] = highest total score of a segmentation of s[0..i).
            var best = new double[n + 1];
            var bestTok = new int[n + 1];
            var bestPrev = new int[n + 1];
            for (int i = 1; i <= n; i++) { best[i] = double.NegativeInfinity; bestTok[i] = -1; }

            for (int i = 0; i < n; i++)
            {
                if (double.IsNegativeInfinity(best[i])) continue;
                int maxEnd = Math.Min(n, i + _maxPieceChars);
                bool matched = false;
                for (int j = i + 1; j <= maxEnd; j++)
                {
                    // Do not split a surrogate pair.
                    if (j < n && char.IsLowSurrogate(s[j])) continue;
                    if (_pieces.TryGetValue(s.Substring(i, j - i), out var piece))
                    {
                        matched = true;
                        double sc = best[i] + piece.Score;
                        if (sc > best[j]) { best[j] = sc; bestTok[j] = piece.Id; bestPrev[j] = i; }
                    }
                }
                if (!matched)
                {
                    // Unknown character: consume one rune as UNK with a strong penalty
                    // (adjacent UNKs merge below, matching SentencePiece behavior).
                    int j = i + 1;
                    if (j < n && char.IsLowSurrogate(s[j])) j++;
                    double sc = best[i] + _unkScore;
                    if (sc > best[j]) { best[j] = sc; bestTok[j] = _unkId; bestPrev[j] = i; }
                }
            }

            // Backtrack (best[n] is always reachable via the UNK fallback).
            var rev = new List<int>();
            for (int i = n; i > 0; i = bestPrev[i])
                rev.Add(bestTok[i]);
            rev.Reverse();
            foreach (int id in rev)
            {
                if (id == _unkId && ids.Count > 0 && ids[^1] == _unkId) continue;   // merge UNK runs
                ids.Add(id);
            }
            return ids;
        }
    }
}
