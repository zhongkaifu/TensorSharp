// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen2.5-VL-7B text encoder for Qwen-Image-Edit. We only need a single forward
// pass over the (already-tokenized) prompt to produce the 3584-dim conditioning
// hidden states the DiT consumes — no generation, no KV cache, no logits.
//
// This is a standard Qwen2.5 decoder trunk: RMSNorm -> GQA attention (separate
// q/k/v projections WITH bias, no q/k-norm, NeoX RoPE theta=1e6, 28 q-heads / 4
// kv-heads / head_dim 128) -> RMSNorm -> SwiGLU MLP -> final RMSNorm. For
// text-only input (Stage 2) Qwen2.5-VL's M-RoPE degenerates to 1D RoPE, so the
// vision position deltas are irrelevant here (added with the vision tower in
// Stage 4). It is its own ModelBase instance over the text-encoder GGUF so it
// reuses the fast quantized matmul / ggml primitives.
// ============================================================================
using System;
using System.Threading.Tasks;
using TensorSharp.Core;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Image conditioning injected into the text encoder at the image-pad span.</summary>
    internal sealed class ImageCond
    {
        public int Start;          // index of first image-pad token
        public int Count;          // number of image-pad tokens (= llm_h * llm_w)
        public int GridH, GridW;   // vision patch grid (before 2x merge)
        public float[] Embeds;     // [Count, 3584] merged vision embeddings
    }

    internal sealed class QwenImageTextEncoder : ModelBase
    {
        private readonly int _numHeads, _numKVHeads, _headDim, _numLayers;
        private readonly float _ropeBase, _eps;

        public int HiddenSize => Config.HiddenSize;

        public QwenImageTextEncoder(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") ?? "qwen2vl" };
            ParseBaseConfig();
            _numHeads = Config.NumHeads;
            _numKVHeads = Config.NumKVHeads;
            _headDim = Config.HeadDim;
            _numLayers = Config.NumLayers;
            _ropeBase = Config.RopeBase > 0 ? Config.RopeBase : 1000000f;
            _eps = Config.Eps > 0 ? Config.Eps : 1e-6f;
            ParseTokenizer();
            EnsureQuantBackendAvailable();
            LoadWeights();
        }

        /// <summary>
        /// Run the trunk over <paramref name="tokens"/> and return the post-final-norm hidden
        /// states as a row-major <c>[seqLen, hidden]</c> float array (matches
        /// transformers <c>outputs.hidden_states[-1]</c>). Caller drops the template prefix.
        /// </summary>
        // M-RoPE 3D positions [3*seq] = (t[seq], h[seq], w[seq]); for text-only all three equal.
        private int[] _mropePos;
        private static readonly int[] MRopeSection = { 16, 24, 24 };   // sums to head_dim/2 = 64

        /// <summary>Text-only conditioning (M-RoPE degenerates to 1D RoPE).</summary>
        public float[] EncodeHidden(int[] tokens) => EncodeHidden(tokens, null);

        /// <summary>
        /// Image-grounded conditioning: <paramref name="img"/> (if non-null) replaces the
        /// <c>&lt;|image_pad|&gt;</c> token embeddings with the vision encoder's merged embeds and
        /// applies 3D M-RoPE positions for the image span.
        /// </summary>
        public unsafe float[] EncodeHidden(int[] tokens, ImageCond img)
        {
            int seq = tokens.Length;
            _mropePos = BuildPositions(tokens.Length, img);

            Tensor hidden = Embedding(tokens);             // [seq, hidden]
            if (img != null)
            {
                // overwrite the image-pad rows with the merged vision embeddings
                float* hp = GetFloatPtr(hidden);
                int H = Config.HiddenSize;
                for (int i = 0; i < img.Count; i++)
                    fixed (float* src = &img.Embeds[(long)i * H])
                        Buffer.MemoryCopy(src, hp + (long)(img.Start + i) * H, (long)H * 4, (long)H * 4);
                InvalidateTensorDeviceCache(hidden);
            }

            for (int layer = 0; layer < _numLayers; layer++)
            {
                string p = $"blk.{layer}";
                using (Tensor normed = RMSNormOp(hidden, $"{p}.attn_norm.weight"))
                using (Tensor attnOut = Attention(normed, p, seq))
                {
                    Tensor res = Ops.Add(hidden, hidden, attnOut);
                    if (!ReferenceEquals(res, hidden)) { hidden.Dispose(); hidden = res; }
                }
                using (Tensor normed2 = RMSNormOp(hidden, $"{p}.ffn_norm.weight"))
                using (Tensor ffnOut = SwiGluFfn(normed2, p))
                {
                    Tensor res = Ops.Add(hidden, hidden, ffnOut);
                    if (!ReferenceEquals(res, hidden)) { hidden.Dispose(); hidden = res; }
                }
            }
            using (Tensor finalNorm = RMSNormOp(hidden, "output_norm.weight"))
            {
                hidden.Dispose();
                return TensorToHostFloat(finalNorm, (long)seq * Config.HiddenSize);
            }
        }

        // get_rope_index: text tokens get sequential positions (all 3 equal); image tokens get
        // (t=cur, h=cur+row, w=cur+col) over the llm grid; after image, cur += max(llm_h,llm_w).
        private int[] BuildPositions(int seq, ImageCond img)
        {
            var pos = new int[3 * seq];   // [t(0..seq), h(seq..2seq), w(2seq..3seq)]
            int cur = 0, s = 0;
            int imgStart = img?.Start ?? -1, imgEnd = img != null ? img.Start + img.Count : -1;
            while (s < seq)
            {
                if (img != null && s == imgStart)
                {
                    int lh = img.GridH / 2, lw = img.GridW / 2;
                    for (int r = 0; r < lh; r++)
                        for (int c = 0; c < lw; c++)
                        {
                            pos[s] = cur; pos[seq + s] = cur + r; pos[2 * seq + s] = cur + c; s++;
                        }
                    cur += Math.Max(lh, lw);
                }
                else
                {
                    pos[s] = cur; pos[seq + s] = cur; pos[2 * seq + s] = cur; cur++; s++;
                }
            }
            return pos;
        }

        private Tensor Attention(Tensor input, string prefix, int seq)
        {
            int qDim = _numHeads * _headDim, kvDim = _numKVHeads * _headDim;
            float scale = 1.0f / MathF.Sqrt(_headDim);

            Tensor q = LinearWithBias(input, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
            Tensor k = LinearWithBias(input, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
            Tensor v = LinearWithBias(input, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");

            ApplyMRoPE(q, _numHeads, seq);
            ApplyMRoPE(k, _numKVHeads, seq);

            Tensor qHeads = ReshapeToHeads(q, _numHeads, seq, _headDim); q.Dispose();
            Tensor kHeads = ReshapeToHeads(k, _numKVHeads, seq, _headDim); k.Dispose();
            Tensor vHeads = ReshapeToHeads(v, _numKVHeads, seq, _headDim); v.Dispose();

            int groupSize = _numHeads / _numKVHeads;
            using Tensor kExp = ExpandKVHeads(kHeads, groupSize, seq); kHeads.Dispose();
            using Tensor vExp = ExpandKVHeads(vHeads, groupSize, seq); vHeads.Dispose();

            using Tensor kT = kExp.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, seq, seq);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();

            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(scores, sinks: null,
                    numHeads: _numHeads, seqLen: seq, kvLen: seq,
                    maskStartPos: 0, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seq, 0, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, _numHeads, seq, _headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExp);
            scores.Dispose();

            using Tensor flat = ReshapeFromHeads(attnOut, _numHeads, seq, _headDim);
            attnOut.Dispose();
            return LinearForward(flat, $"{prefix}.attn_output.weight");
        }

        private Tensor SwiGluFfn(Tensor input, string prefix)
        {
            Tensor gate = LinearForward(input, $"{prefix}.ffn_gate.weight");
            using (Tensor up = LinearForward(input, $"{prefix}.ffn_up.weight"))
            {
                SiluMulInPlace(gate, up);
            }
            Tensor down = LinearForward(gate, $"{prefix}.ffn_down.weight");
            gate.Dispose();
            return down;
        }

        // Multimodal RoPE (rotate_half / NeoX) over [seq, numHeads*headDim]. Each of the
        // headDim/2 frequency indices belongs to a t/h/w section (mrope_section 16/24/24);
        // the rotation angle uses that section's 3D position component. Text tokens have all
        // three components equal, so this is identical to standard 1D RoPE.
        private unsafe void ApplyMRoPE(Tensor data, int numHeads, int seq)
        {
            int half = _headDim / 2;     // 64
            int[] pos = _mropePos;       // [3*seq]
            // section id per freq index i: 0=t (i<16), 1=h (i<40), 2=w (else)
            float* p = GetFloatPtr(data);
            Parallel.For(0, seq, s =>
            {
                for (int hh = 0; hh < numHeads; hh++)
                {
                    float* head = p + (long)s * (numHeads * _headDim) + (long)hh * _headDim;
                    int acc = 0, sec = 0;
                    for (int i = 0; i < half; i++)
                    {
                        if (i >= acc + MRopeSection[sec]) { acc += MRopeSection[sec]; sec++; }
                        int comp = sec;                                  // 0/1/2 -> t/h/w
                        int position = pos[comp * seq + s];
                        float freq = (float)Math.Pow(_ropeBase, -2.0 * i / _headDim);
                        float ang = position * freq;
                        float c = MathF.Cos(ang), sn = MathF.Sin(ang);
                        float x1 = head[i], x2 = head[half + i];
                        head[i] = x1 * c - x2 * sn;
                        head[half + i] = x2 * c + x1 * sn;
                    }
                }
            });
            InvalidateTensorDeviceCache(data);
        }

        private unsafe Tensor LinearWithBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = LinearForward(input, weightName);
            if (_weights.TryGetValue(biasName, out var bias))
            {
                int seq = (int)result.Sizes[0], outDim = (int)result.Sizes[1];
                float* rPtr = GetFloatPtr(result);
                float* bPtr = GetFloatPtr(bias);
                int dim = Math.Min(outDim, (int)bias.ElementCount());
                Parallel.For(0, seq, s =>
                {
                    float* row = rPtr + (long)s * outDim;
                    for (int d = 0; d < dim; d++) row[d] += bPtr[d];
                });
            }
            return result;
        }

        private unsafe void SiluMulInPlace(Tensor gate, Tensor up)
        {
            int n = (int)gate.ElementCount();
            float* g = GetFloatPtr(gate);
            float* u = GetFloatPtr(up);
            Parallel.For(0, n, i =>
            {
                float x = g[i];
                g[i] = (x / (1f + MathF.Exp(-x))) * u[i];
            });
        }

        private unsafe float[] TensorToHostFloat(Tensor t, long count)
        {
            var dst = new float[count];
            float* p = GetFloatPtr(t);
            fixed (float* d = dst)
                Buffer.MemoryCopy(p, d, count * sizeof(float), count * sizeof(float));
            return dst;
        }

        public override float[] Forward(int[] tokens) =>
            throw new NotSupportedException("Use EncodeHidden().");
        public override void ResetKVCache() { }
    }
}
