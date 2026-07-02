// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen2.5-VL vision tower + qwen2.5vl_merger, loaded from the companion mmproj GGUF
// (clip arch). Produces the merged image embeddings (out_hidden_size 3584) that the
// text encoder injects at the <|image_pad|> positions for image-grounded prompts.
//
// Arch (verified vs transformers Qwen2_5_VL): 32 RMSNorm blocks (hidden 1280, 16
// heads, SwiGLU ffn 3420), Conv3d(temporal 2,14,14) patch embed, 2D rotate-half
// RoPE, WINDOW attention (window 112 -> 4 merged-patch windows) except blocks
// {7,15,23,31} which are full attention; window_index reorders patches in groups of
// spatial_merge_unit=4; merger = RMSNorm -> view(4*1280=5120) -> Linear -> GELU ->
// Linear(3584), then the window reorder is reversed.
//
// Big matmuls use ggml LinearForward (mmproj weights are BF16); the elementwise /
// reorder / window-mask pieces are host-pointer kernels, like QwenImageTextEncoder.
// ============================================================================
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Core;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class QwenImageVisionEncoder : ModelBase
    {
        public const int Emb = 1280, Heads = 16, HeadDim = 80, Blocks = 32;
        public const int Merge = 2, MergeUnit = 4, WindowMerger = 4; // 112/2/14
        public const int OutDim = QwenImageModel.DitTextDim;          // 3584
        private const float Eps = 1e-6f, Theta = 10000f;
        private static readonly int[] FullAtt = { 7, 15, 23, 31 };

        private float[] _patchEmbdCombined; // [Emb, 1176] = [oc, ic*2*14*14...]

        public QwenImageVisionEncoder(string mmprojPath, BackendType backend) : base(mmprojPath, backend)
        {
            Config = new ModelConfig { Architecture = "clip", HiddenSize = Emb };
            EnsureQuantBackendAvailable();
            LoadWeights();
        }

        public override float[] Forward(int[] tokens) => throw new NotSupportedException();
        public override void ResetKVCache() { }

        /// <summary>
        /// Encode patchified pixel values into merged image embeddings.
        /// </summary>
        /// <param name="pixelValues">[seq, 1176] flattened patches (C,T,Ph,Pw), seq = gridH*gridW.</param>
        /// <returns>[seq/4, 3584] merged embeddings (in original, un-windowed order).</returns>
        public float[] Encode(float[] pixelValues, int gridH, int gridW)
        {
            int seq = gridH * gridW;
            int[] win = BuildWindowIndex(gridH, gridW, out int[] cuWin);

            // patch embed (host matmul: [seq,1176] x [1176,Emb])
            float[] embeddedHost = PatchEmbed(pixelValues, seq);
            // window reorder (groups of 4)
            float[] reordered = ReorderGroups(embeddedHost, seq, Emb, win, MergeUnit);

            // 2D RoPE cos/sin per token (after reorder)
            BuildRope(gridH, gridW, win, out float[] cos, out float[] sin);   // [seq, HeadDim]

            // Fused whole-trunk path (TSGgml_QwenTeTrunk): all 32 blocks in ONE device graph
            // (window/full attention via an additive mask). Falls back to the per-op loop.
            float[] groupedHost = TryFusedTrunk(reordered, seq, cos, sin, cuWin);
            if (groupedHost == null)
            {
                Tensor x = HostToTensor(reordered, seq, Emb);
                int[] cuFull = { 0, seq };
                for (int L = 0; L < Blocks; L++)
                {
                    string p = $"v.blk.{L}";
                    int[] cu = Array.IndexOf(FullAtt, L) >= 0 ? cuFull : cuWin;
                    using (Tensor n1 = RMSNormOp(x, $"{p}.ln1.weight"))
                    using (Tensor attn = Attention(n1, p, seq, cu, cos, sin))
                        AddInPlace(x, attn, seq, Emb);
                    using (Tensor n2 = RMSNormOp(x, $"{p}.ln2.weight"))
                    using (Tensor ffn = SwiGlu(n2, p))
                        AddInPlace(x, ffn, seq, Emb);
                }
                using Tensor normed = RMSNormOp(x, "v.post_ln.weight");
                x.Dispose();
                groupedHost = TensorToHost(normed, (long)seq * Emb); // already grouped contiguous (4 consecutive rows)
            }

            // merger: group 4 -> [seq/4, 5120] -> mm.0 -> GELU -> mm.2 -> [seq/4, 3584]
            int merged = seq / MergeUnit;
            using Tensor grouped = HostToTensor(groupedHost, merged, Emb * MergeUnit);
            using Tensor h0 = LinearBias(grouped, "mm.0.weight", "mm.0.bias");
            GeluInPlace(h0);
            using Tensor outT = LinearBias(h0, "mm.2.weight", "mm.2.bias");
            float[] mergedHost = TensorToHost(outT, (long)merged * OutDim);

            // reverse the window reorder (operates on merged tokens, 1 per group)
            int[] revWin = ArgSort(win);
            var result = new float[(long)merged * OutDim];
            for (int i = 0; i < merged; i++)
                Array.Copy(mergedHost, (long)revWin[i] * OutDim, result, (long)i * OutDim, OutDim);
            return result;
        }

        // ---- attention with window/full block-diagonal mask ----
        private Tensor Attention(Tensor input, string prefix, int seq, int[] cu, float[] cos, float[] sin)
        {
            Tensor q = LinearBias(input, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
            Tensor k = LinearBias(input, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
            Tensor v = LinearBias(input, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");
            ApplyRope(q, seq, cos, sin);
            ApplyRope(k, seq, cos, sin);

            Tensor qH = ReshapeToHeads(q, Heads, seq, HeadDim); q.Dispose();
            Tensor kH = ReshapeToHeads(k, Heads, seq, HeadDim); k.Dispose();
            Tensor vH = ReshapeToHeads(v, Heads, seq, HeadDim); v.Dispose();
            using Tensor kT = kH.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, Heads, seq, seq);
            Ops.AddmmBatch(scores, 0, scores, 1.0f / MathF.Sqrt(HeadDim), qH, kT);
            qH.Dispose(); kH.Dispose();
            ApplyBlockMaskSoftmax(scores, Heads, seq, cu);
            var attn = new Tensor(_allocator, DType.Float32, Heads, seq, HeadDim);
            Ops.AddmmBatch(attn, 0, attn, 1.0f, scores, vH);
            scores.Dispose(); vH.Dispose();
            using Tensor flat = ReshapeFromHeads(attn, Heads, seq, HeadDim); attn.Dispose();
            return LinearBias(flat, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
        }

        private Tensor SwiGlu(Tensor input, string prefix)
        {
            Tensor gate = LinearBias(input, $"{prefix}.ffn_gate.weight", $"{prefix}.ffn_gate.bias");
            using (Tensor up = LinearBias(input, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias"))
                SiluMul(gate, up);
            Tensor down = LinearBias(gate, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
            gate.Dispose();
            return down;
        }

        // ---- host kernels ----
        private unsafe float[] PatchEmbed(float[] pixel, int seq)
        {
            _patchEmbdCombined ??= BuildPatchEmbdWeight();
            int inDim = 1176;
            var outp = new float[(long)seq * Emb];
            var w = _patchEmbdCombined;
            Parallel.For(0, seq, s =>
            {
                long pb = (long)s * inDim;
                for (int oc = 0; oc < Emb; oc++)
                {
                    long wb = (long)oc * inDim;
                    float acc = 0;
                    for (int i = 0; i < inDim; i++) acc += pixel[pb + i] * w[wb + i];
                    outp[(long)s * Emb + oc] = acc;
                }
            });
            return outp;
        }

        private float[] BuildPatchEmbdWeight()
        {
            // v.patch_embd.weight ggml [14,14,3,1280] -> data order (kw,kh,ic,oc); numpy [oc,ic,kh,kw].
            // Combined weight [oc, ic, t, kh, kw] flattened to [oc, 1176] matching pixel [ic,t,kh,kw].
            var w0 = ReadF32("v.patch_embd.weight");    // length 14*14*3*1280, ggml ne0=kw fastest
            var w1 = ReadF32("v.patch_embd.weight.1");
            var comb = new float[(long)Emb * 1176];
            // ggml layout index: ((oc*3 + ic)*14 + kh)*14 + kw  == numpy [oc,ic,kh,kw] (kw fastest)
            for (int oc = 0; oc < Emb; oc++)
                for (int ic = 0; ic < 3; ic++)
                    for (int kh = 0; kh < 14; kh++)
                        for (int kw = 0; kw < 14; kw++)
                        {
                            long src = (((long)oc * 3 + ic) * 14 + kh) * 14 + kw;
                            long dst0 = ((long)oc * 1176) + ((((long)ic * 2 + 0) * 14 + kh) * 14 + kw);
                            long dst1 = ((long)oc * 1176) + ((((long)ic * 2 + 1) * 14 + kh) * 14 + kw);
                            comb[dst0] = w0[src];
                            comb[dst1] = w1[src];
                        }
            return comb;
        }

        private float[] ReadF32(string name)
        {
            var info = _gguf.Tensors[name];
            long n = info.NumElements;
            byte[] raw = _gguf.ReadTensorData(info);
            var dst = new float[n];
            GgmlGgufTensorDequant.DequantizeToFloat32((int)info.Type, raw, 0, dst, 0, n);
            return dst;
        }

        // reorder rows in groups of `unit` by window index (win has seq/unit entries)
        private static float[] ReorderGroups(float[] src, int seq, int dim, int[] win, int unit)
        {
            int groups = seq / unit;
            var outp = new float[(long)seq * dim];
            for (int g = 0; g < groups; g++)
            {
                int srcG = win[g];
                for (int u = 0; u < unit; u++)
                    Array.Copy(src, (long)(srcG * unit + u) * dim, outp, (long)(g * unit + u) * dim, dim);
            }
            return outp;
        }

        private int[] BuildWindowIndex(int h, int w, out int[] cuWin)
        {
            int lh = h / Merge, lw = w / Merge;
            int ph = (WindowMerger - lh % WindowMerger) % WindowMerger;
            int pw = (WindowMerger - lw % WindowMerger) % WindowMerger;
            int nwh = (lh + ph) / WindowMerger, nww = (lw + pw) / WindowMerger;
            var index = new int[lh, lw];
            for (int i = 0; i < lh; i++) for (int j = 0; j < lw; j++) index[i, j] = i * lw + j;
            var win = new System.Collections.Generic.List<int>();
            var cu = new System.Collections.Generic.List<int> { 0 };
            int running = 0;
            for (int wy = 0; wy < nwh; wy++)
                for (int wx = 0; wx < nww; wx++)
                {
                    int cnt = 0;
                    for (int iy = 0; iy < WindowMerger; iy++)
                        for (int ix = 0; ix < WindowMerger; ix++)
                        {
                            int gy = wy * WindowMerger + iy, gx = wx * WindowMerger + ix;
                            if (gy < lh && gx < lw) { win.Add(index[gy, gx]); cnt++; }
                        }
                    running += cnt * MergeUnit;
                    cu.Add(running);
                }
            // unique-consecutive
            var cuList = new System.Collections.Generic.List<int> { cu[0] };
            for (int i = 1; i < cu.Count; i++) if (cu[i] != cuList[cuList.Count - 1]) cuList.Add(cu[i]);
            cuWin = cuList.ToArray();
            return win.ToArray();
        }

        // 2D RoPE: per (merged-group, sub) token, h/w grid positions; reordered by window groups.
        private void BuildRope(int h, int w, int[] win, out float[] cos, out float[] sin)
        {
            int seq = h * w, lh = h / Merge, lw = w / Merge;
            int half = HeadDim / 2;            // 40
            int freqN = half / 2;              // 20
            var inv = new float[freqN];
            for (int i = 0; i < freqN; i++) inv[i] = (float)(1.0 / Math.Pow(Theta, (2.0 * i) / half));

            // position_ids per token in merge order: [seq,2] (hpos,wpos)
            var hpos = new int[seq]; var wpos = new int[seq];
            int t = 0;
            for (int by = 0; by < lh; by++)
                for (int bx = 0; bx < lw; bx++)
                    for (int sy = 0; sy < Merge; sy++)
                        for (int sx = 0; sx < Merge; sx++)
                        { hpos[t] = by * Merge + sy; wpos[t] = bx * Merge + sx; t++; }

            cos = new float[(long)seq * HeadDim];
            sin = new float[(long)seq * HeadDim];
            // build per token then reorder by window groups
            var cosTmp = new float[(long)seq * HeadDim];
            var sinTmp = new float[(long)seq * HeadDim];
            for (int s = 0; s < seq; s++)
            {
                // rope[40] = [h*inv(20), w*inv(20)]; emb[80] = cat(rope,rope)
                for (int i = 0; i < freqN; i++)
                {
                    float ah = hpos[s] * inv[i], aw = wpos[s] * inv[i];
                    long b = (long)s * HeadDim;
                    cosTmp[b + i] = MathF.Cos(ah); sinTmp[b + i] = MathF.Sin(ah);
                    cosTmp[b + freqN + i] = MathF.Cos(aw); sinTmp[b + freqN + i] = MathF.Sin(aw);
                    cosTmp[b + half + i] = cosTmp[b + i]; sinTmp[b + half + i] = sinTmp[b + i];
                    cosTmp[b + half + freqN + i] = cosTmp[b + freqN + i]; sinTmp[b + half + freqN + i] = sinTmp[b + freqN + i];
                }
            }
            cos = ReorderGroups(cosTmp, seq, HeadDim, win, MergeUnit);
            sin = ReorderGroups(sinTmp, seq, HeadDim, win, MergeUnit);
        }

        private unsafe void ApplyRope(Tensor qkv, int seq, float[] cos, float[] sin)
        {
            int half = HeadDim / 2;
            float* p = GetFloatPtr(qkv);
            Parallel.For(0, seq, s =>
            {
                for (int hh = 0; hh < Heads; hh++)
                {
                    float* head = p + (long)s * (Heads * HeadDim) + (long)hh * HeadDim;
                    long b = (long)s * HeadDim;
                    for (int i = 0; i < half; i++)
                    {
                        float c1 = cos[b + i], s1 = sin[b + i];
                        float c2 = cos[b + half + i], s2 = sin[b + half + i];
                        float x1 = head[i], x2 = head[half + i];
                        // rotate_half: out[i] = x1*cos[i] - x2*sin[i]; out[i+half] = x2*cos[i+half] + x1*sin[i+half]
                        head[i] = x1 * c1 - x2 * s1;
                        head[half + i] = x2 * c2 + x1 * s2;
                    }
                }
            });
            InvalidateTensorDeviceCache(qkv);
        }

        // softmax over each query row, masked to its cu-segment (block-diagonal).
        private unsafe void ApplyBlockMaskSoftmax(Tensor scores, int heads, int seq, int[] cu)
        {
            // segment id per token
            var seg = new int[seq];
            for (int i = 0; i < cu.Length - 1; i++)
                for (int t = cu[i]; t < cu[i + 1]; t++) seg[t] = i;
            float* p = GetFloatPtr(scores);
            Parallel.For(0, heads * seq, hs =>
            {
                int h = hs / seq, qi = hs % seq;
                float* row = p + ((long)h * seq + qi) * seq;
                int sq = seg[qi];
                float mx = float.NegativeInfinity;
                for (int j = 0; j < seq; j++) { if (seg[j] != sq) row[j] = float.NegativeInfinity; else if (row[j] > mx) mx = row[j]; }
                float sum = 0;
                for (int j = 0; j < seq; j++) { if (seg[j] == sq) { float e = MathF.Exp(row[j] - mx); row[j] = e; sum += e; } else row[j] = 0; }
                float inv = 1f / sum;
                for (int j = 0; j < seq; j++) row[j] *= inv;
            });
            InvalidateTensorDeviceCache(scores);
        }

        private static int[] ArgSort(int[] a)
        {
            var idx = new int[a.Length];
            for (int i = 0; i < a.Length; i++) idx[i] = i;
            Array.Sort(idx, (x, y) => a[x].CompareTo(a[y]));
            return idx;
        }

        // ---- fused whole-trunk path (TSGgml_QwenTeTrunk) ------------------------------
        // The per-op loop pays ~10 device<->host round-trips per block (RoPE, bias adds,
        // block-mask softmax, SiLU as host loops) x 32 blocks. The fused path runs the
        // whole tower as one device graph; the window/full attention pattern becomes an
        // additive [seq, seq] mask (0 inside the token's window segment, -inf outside).
        // Returns the post-v.post_ln hidden states [seq, Emb], or null to fall back.
        // TS_QWEN_TE_FUSED=0 disables (shared knob with the text-encoder trunk).
        private static readonly bool FusedTrunkOn =
            Environment.GetEnvironmentVariable("TS_QWEN_TE_FUSED") != "0";
        private QwenTeLayerW[] _fusedLayers;
        private readonly System.Collections.Generic.List<IntPtr> _fusedAllocs = new();
        private IntPtr _fusedFinalNorm;
        private bool _fusedFailed;

        private unsafe float[] TryFusedTrunk(float[] reordered, int seq, float[] cos, float[] sin, int[] cuWin)
        {
            if (!FusedTrunkOn || !IsGgmlBackend || _fusedFailed) return null;
            if (_fusedLayers == null && !BuildFusedLayers()) { _fusedFailed = true; return null; }

            // additive window mask from cu_seqlens
            var seg = new int[seq];
            for (int i = 0; i < cuWin.Length - 1; i++)
                for (int t = cuWin[i]; t < cuWin[i + 1]; t++) seg[t] = i;
            var mask = new float[(long)seq * seq];
            Parallel.For(0, seq, q =>
            {
                long b = (long)q * seq;
                int sq = seg[q];
                for (int j = 0; j < seq; j++) mask[b + j] = seg[j] == sq ? 0f : float.NegativeInfinity;
            });

            var outArr = new float[(long)seq * Emb];
            bool ok;
            fixed (float* xp = reordered, cp = cos, sp = sin, mp = mask, op = outArr)
            fixed (QwenTeLayerW* lp = _fusedLayers)
            {
                var a = new QwenTeTrunkArgs
                {
                    X = (IntPtr)xp, Out = (IntPtr)op, CosF = (IntPtr)cp, SinF = (IntPtr)sp,
                    WinMask = (IntPtr)mp, FinalNorm = _fusedFinalNorm,
                    Layers = (IntPtr)lp, NumLayers = Blocks,
                    StructBytes = Marshal.SizeOf<QwenTeTrunkArgs>(),
                    Hidden = Emb, Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, Seq = seq,
                    Eps = Config.Eps,
                };
                ok = GgmlBasicOps.TryQwenTeTrunk(in a);
            }
            if (!ok)
            {
                Console.WriteLine("  [venc-fused] trunk kernel unavailable; using the per-op path.");
                _fusedFailed = true;
                return null;
            }
            return outArr;
        }

        private unsafe bool BuildFusedLayers()
        {
            try
            {
                QImgAttnW W(string name, string biasName)
                {
                    var info = _gguf.Tensors[name];
                    _gguf.TryGetTensorDataPointer(info, out IntPtr p);
                    var w = new QImgAttnW
                    {
                        W = p,
                        Type = (int)info.Type,
                        Ne0 = (long)info.Shape[0],
                        Ne1 = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
                        Bytes = _gguf.GetTensorByteCount(info),
                    };
                    if (biasName != null && _gguf.Tensors.ContainsKey(biasName))
                        w.B = F32Stable(biasName);
                    return w;
                }

                var layers = new QwenTeLayerW[Blocks];
                for (int l = 0; l < Blocks; l++)
                {
                    string p = $"v.blk.{l}";
                    layers[l] = new QwenTeLayerW
                    {
                        Ln1 = F32Stable($"{p}.ln1.weight"),
                        Ln2 = F32Stable($"{p}.ln2.weight"),
                        Q = W($"{p}.attn_q.weight", $"{p}.attn_q.bias"),
                        K = W($"{p}.attn_k.weight", $"{p}.attn_k.bias"),
                        V = W($"{p}.attn_v.weight", $"{p}.attn_v.bias"),
                        O = W($"{p}.attn_out.weight", $"{p}.attn_out.bias"),
                        Gate = W($"{p}.ffn_gate.weight", $"{p}.ffn_gate.bias"),
                        Up = W($"{p}.ffn_up.weight", $"{p}.ffn_up.bias"),
                        Down = W($"{p}.ffn_down.weight", $"{p}.ffn_down.bias"),
                        MaskKind = Array.IndexOf(FullAtt, l) >= 0 ? 0 : 2,
                    };
                }
                _fusedFinalNorm = F32Stable("v.post_ln.weight");
                _fusedLayers = layers;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [venc-fused] layer table build failed ({ex.Message}); using the per-op path.");
                return false;
            }
        }

        private unsafe IntPtr F32Stable(string name)
        {
            var info = _gguf.Tensors[name];
            long n = info.NumElements;
            var host = new float[n];
            byte[] raw = _gguf.ReadTensorData(info);
            GgmlGgufTensorDequant.DequantizeToFloat32((int)info.Type, raw, 0, host, 0, n);
            IntPtr p = Marshal.AllocHGlobal((IntPtr)(n * sizeof(float)));
            Marshal.Copy(host, 0, p, (int)n);
            _fusedAllocs.Add(p);
            return p;
        }

        public override void Dispose()
        {
            base.Dispose();
            foreach (var p in _fusedAllocs) Marshal.FreeHGlobal(p);
            _fusedAllocs.Clear();
            _fusedLayers = null;
        }

        // ---- shared small helpers (mirror QwenImageTextEncoder) ----
        private unsafe Tensor LinearBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = LinearForward(input, weightName);
            if (_weights.TryGetValue(biasName, out var bias))
            {
                int rows = (int)result.Sizes[0], outDim = (int)result.Sizes[1];
                float* r = GetFloatPtr(result); float* bp = GetFloatPtr(bias);
                int dim = Math.Min(outDim, (int)bias.ElementCount());
                Parallel.For(0, rows, s => { float* row = r + (long)s * outDim; for (int d = 0; d < dim; d++) row[d] += bp[d]; });
            }
            return result;
        }
        private unsafe void SiluMul(Tensor gate, Tensor up)
        {
            int n = (int)gate.ElementCount(); float* g = GetFloatPtr(gate); float* u = GetFloatPtr(up);
            Parallel.For(0, n, i => { float x = g[i]; g[i] = (x / (1f + MathF.Exp(-x))) * u[i]; });
            InvalidateTensorDeviceCache(gate);
        }
        private unsafe void GeluInPlace(Tensor t)
        {
            int n = (int)t.ElementCount(); float* p = GetFloatPtr(t);
            Parallel.For(0, n, i => { float x = p[i]; p[i] = 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x))); });
            InvalidateTensorDeviceCache(t);
        }
        private unsafe void AddInPlace(Tensor a, Tensor b, int rows, int dim)
        {
            float* ap = GetFloatPtr(a); float* bp = GetFloatPtr(b);
            Parallel.For(0, rows, s => { long o = (long)s * dim; for (int d = 0; d < dim; d++) ap[o + d] += bp[o + d]; });
            InvalidateTensorDeviceCache(a);
        }
        private unsafe Tensor HostToTensor(float[] data, int rows, int cols)
        {
            var t = new Tensor(_allocator, DType.Float32, rows, cols);
            float* p = GetFloatPtr(t);
            fixed (float* s = data) Buffer.MemoryCopy(s, p, (long)rows * cols * 4, (long)data.Length * 4);
            InvalidateTensorDeviceCache(t);
            return t;
        }
        private unsafe float[] TensorToHost(Tensor t, long count)
        {
            var dst = new float[count]; float* p = GetFloatPtr(t);
            fixed (float* d = dst) Buffer.MemoryCopy(p, d, count * 4, count * 4);
            return dst;
        }
    }
}
