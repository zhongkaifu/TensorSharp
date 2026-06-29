// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen-Image MMDiT (QwenImageTransformer2DModel): 60 joint (img+txt) double-stream
// blocks. One forward = one denoising-step velocity prediction. Mirrors diffusers
// transformer_qwenimage.py. The big projections are ggml quantized matmuls
// (LinearForward over the Q4_0 DiT GGUF — dequantizing the whole 11.8GB model to
// f32 is infeasible); the elementwise pieces (AdaLN modulation, QK-RMSNorm,
// interleaved multi-axis RoPE, GELU) are host-pointer kernels over the (GgmlCpu)
// activation tensors, the same pattern as QwenImageTextEncoder.
// ============================================================================
using System;
using System.Threading.Tasks;
using TensorSharp.Core;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageDiT : ModelBase
    {
        private const int Dim = QwenImageModel.DitHiddenSize;   // 3072
        private const int NumHeads = QwenImageModel.DitNumHeads; // 24
        private const int HeadDim = QwenImageModel.DitHeadDim;   // 128
        private const int NumLayers = QwenImageModel.DitNumLayers; // 60
        private const int InCh = QwenImageModel.DitInChannels;  // 64
        private const int TxtDim = QwenImageModel.DitTextDim;    // 3584
        private const float Eps = 1e-6f;
        private static readonly int[] AxesDim = { 16, 56, 56 };  // complex halves 8/28/28 = 64
        private const float RopeTheta = 10000f;

        public QwenImageDiT(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = "qwen_image", HiddenSize = Dim, NumLayers = NumLayers };
            EnsureQuantBackendAvailable();
            LoadWeights();
        }

        public override float[] Forward(int[] tokens) => throw new NotSupportedException("Use Predict().");
        public override void ResetKVCache() { }

        // The native per-block path (TS_QWEN_DIT_NATIVE=1) streams each block's weights
        // from the GGUF mmap into small device slots per call, so the ~7 GB whole-model
        // CUDA preload is pure wasted VRAM there (and double-allocates against the capture
        // scratch, causing the 16 GB spill/OOM). Skip it on the native path; the managed
        // path still needs the resident weights.
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !NativeBlockOn;

        private static readonly bool DebugOn = Environment.GetEnvironmentVariable("TS_QIMG_DEBUG") == "1";
        private unsafe void Dbg(string name, Tensor t)
        {
            if (!DebugOn) return;
            int n = (int)t.ElementCount();
            var h = TensorToHost(t, n);
            Console.WriteLine($"  [dit] {name,-12} {Stat(h, n)}");
        }
        internal static string Stat(float[] a, int n)
        {
            double mn = double.MaxValue, mx = double.MinValue, sum = 0; int nan = 0;
            for (int i = 0; i < n; i++)
            {
                float v = a[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
                if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
            }
            return $"min={mn:E2} max={mx:E2} mean={sum / Math.Max(1, n - nan):E2} nan/inf={nan}";
        }

        /// <summary>
        /// Predict the FlowMatch velocity for the image tokens.
        /// </summary>
        /// <param name="imgTokens">packed latent tokens [imgSeq, 64] (noise tokens first, then reference).</param>
        /// <param name="textCond">conditioning [txtSeq, 3584] from the text encoder.</param>
        /// <param name="timestep01">timestep in [0,1].</param>
        /// <param name="modulateIndex">per image token, 0 = noise/generated, 1 = reference (t=0 modulation); null = all 0.</param>
        /// <param name="rope">precomputed interleaved RoPE cos/sin for [txt, img] tokens.</param>
        /// <returns>velocity [imgSeq, 64].</returns>
        public float[] Predict(float[] imgTokens, int imgSeq, float[] textCond, int txtSeq,
            float timestep01, int[] modulateIndex, DitRope rope,
            int stepIndex = -1, int totalSteps = 0, int cfgBranch = 0, int genTokens = 0)
        {
            // img_in: Linear(64 -> 3072) (+bias)
            using Tensor imgT = HostToTensor(imgTokens, imgSeq, InCh);
            Tensor img = LinearBias(imgT, "img_in.weight", "img_in.bias");   // [imgSeq, 3072]

            // txt_norm (RMSNorm 3584) then txt_in: Linear(3584 -> 3072) (+bias)
            using Tensor txtT = HostToTensor(textCond, txtSeq, TxtDim);
            using Tensor txtNormed = RMSNormOp(txtT, "txt_norm.weight");
            Tensor txt = LinearBias(txtNormed, "txt_in.weight", "txt_in.bias"); // [txtSeq, 3072]

            // time_text_embed: sinusoidal(256) -> linear_1 -> SiLU -> linear_2  => temb[2,3072] (t and 0)
            Tensor temb = TimeEmbed(timestep01);   // [2, 3072]

            var blkSw = TimingOn ? System.Diagnostics.Stopwatch.StartNew() : null;
            if (NativeBlockOn)
            {
                // Stage 7: run all 60 blocks via the fused native kernels on host arrays
                // (one device round-trip per sub-layer instead of ~16 per block).
                float[] imgHost = TensorToHost(img, (long)imgSeq * Dim); img.Dispose();
                float[] txtHost = TensorToHost(txt, (long)txtSeq * Dim); txt.Dispose();

                // Whole-model fast path: run the blocks in ONE resident-weight graph (in-graph
                // modulation, single dispatch + sync instead of ~180 per-block CPU<->GPU syncs).
                // Integrated WITH the First-Block-Cache: on a cache-active step, block 0 is run
                // per-block (for the residual decision), then blocks 1..N-1 go through this kernel
                // (or are skipped on a cache hit). RunBlocks dispatches the per-layer work.
                bool cacheActive = CacheEnabled && stepIndex >= 0 && totalSteps > 1;
                // Run blocks [start, NumLayers) — whole-model graph when enabled, else per-block.
                void RunBlocks(int start)
                {
                    if (WholeModelOn)
                    {
                        var swW = TimingOn ? System.Diagnostics.Stopwatch.StartNew() : null;
                        bool ok = TryWholeBlocks(imgHost, imgSeq, txtHost, txtSeq, temb, modulateIndex, rope, start, NumLayers - start);
                        if (swW != null) { swW.Stop(); Console.WriteLine($"  [dit-timing] whole-model {NumLayers - start}-block graph imgSeq={imgSeq} txtSeq={txtSeq}: {swW.Elapsed.TotalMilliseconds:F0}ms"); }
                        if (ok) return;
                    }
                    for (int layer = start; layer < NumLayers; layer++)
                        RunNativeLayer(layer, imgHost, imgSeq, txtHost, txtSeq, temb, modulateIndex, rope);
                }

                if (!cacheActive)
                {
                    RunBlocks(0);
                }
                else
                {
                    long imgLen = (long)imgSeq * Dim;
                    int genLen = (int)Math.Min(imgLen, (long)(genTokens > 0 ? genTokens : imgSeq) * Dim);
                    var state = _cache[cfgBranch & 1];

                    // hidden (generated region) before block 0, to form the block-0 residual
                    var imgBeforeGen = new float[genLen];
                    Array.Copy(imgHost, 0, imgBeforeGen, 0, genLen);

                    RunNativeLayer(0, imgHost, imgSeq, txtHost, txtSeq, temb, modulateIndex, rope);

                    var firstResidualGen = new float[genLen];
                    for (int i = 0; i < genLen; i++) firstResidualGen[i] = imgHost[i] - imgBeforeGen[i];

                    if (DecideUseCache(state, stepIndex, firstResidualGen))
                    {
                        // reuse: img_final = img_after_block0 + cached(blocks 1..N-1 residual)
                        var rem = state.RemainingResidual;
                        for (long i = 0; i < imgLen; i++) imgHost[i] += rem[i];
                        _cacheSkipped++;
                    }
                    else
                    {
                        var imgAfter0 = new float[imgLen];
                        Array.Copy(imgHost, imgAfter0, imgLen);
                        RunBlocks(1);   // blocks 1..N-1 (whole-model graph when enabled)
                        var rem = state.RemainingResidual ?? new float[imgLen];
                        for (long i = 0; i < imgLen; i++) rem[i] = imgHost[i] - imgAfter0[i];
                        state.RemainingResidual = rem;
                        _cacheComputed++;
                    }
                }

                img = HostToTensor(imgHost, imgSeq, Dim);
                txt = HostToTensor(txtHost, txtSeq, Dim);
            }
            else
            {
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    Block(ref img, ref txt, temb, imgSeq, txtSeq, layer, modulateIndex, rope);
                    if (DebugOn) Dbg($"img@blk{layer}", img);
                }
            }
            if (blkSw != null)
            {
                blkSw.Stop();
                string mode = NativeBlockOn ? (FusedBlockOn ? "fused" : "3-call") : "managed";
                Console.WriteLine($"  [dit-timing] {mode} {NumLayers}-block loop imgSeq={imgSeq} txtSeq={txtSeq}: {blkSw.Elapsed.TotalMilliseconds:F0}ms");
            }
            temb.Dispose();
            txt.Dispose();

            // norm_out (AdaLN continuous) on img with temb[0], then proj_out -> [imgSeq, 64]
            Tensor normed = AdaLayerNormOut(img, timestep01); img.Dispose();
            Tensor outT = LinearBias(normed, "proj_out.weight", "proj_out.bias"); normed.Dispose();
            float[] velocity = TensorToHost(outT, (long)imgSeq * InCh);
            outT.Dispose();
            if (DebugOn) { var st = Stat(velocity, velocity.Length); Console.WriteLine($"  [dit] velocity {st}"); }
            return velocity;
        }

        // Run one block of the native per-block path (the body of the block loop),
        // updating imgHost/txtHost in place. Factored out so First-Block-Cache can run
        // block 0 alone, decide, then optionally run the remaining blocks.
        private void RunNativeLayer(int layer, float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            Tensor temb, int[] modulateIndex, DitRope rope)
        {
            string bn = $"transformer_blocks.{layer}";
            float[] imgMod = ModParams(temb, $"{bn}.img_mod.1.weight", $"{bn}.img_mod.1.bias");
            float[] txtMod = ModParams(temb, $"{bn}.txt_mod.1.weight", $"{bn}.txt_mod.1.bias");
            if (FusedBlockOn)
            {
                NativeBlock(imgHost, imgSeq, txtHost, txtSeq, imgMod, txtMod, modulateIndex, rope, layer);
            }
            else
            {
                NativeAttnSubLayer(imgHost, imgSeq, txtHost, txtSeq, imgMod, txtMod, modulateIndex, rope, layer);
                NativeMlpSubLayer(imgHost, imgSeq, imgMod, modulateIndex, $"{bn}.img_mlp");
                NativeMlpSubLayer(txtHost, txtSeq, txtMod, null, $"{bn}.txt_mlp");
            }
        }

        // ---- one double-stream block ---------------------------------------
        private void Block(ref Tensor img, ref Tensor txt, Tensor temb, int imgSeq, int txtSeq,
            int layer, int[] modulateIndex, DitRope rope)
        {
            string b = $"transformer_blocks.{layer}";

            // modulation params: img_mod / txt_mod = SiLU -> Linear(3072 -> 18432) (+bias)
            float[] imgMod = ModParams(temb, $"{b}.img_mod.1.weight", $"{b}.img_mod.1.bias"); // [2,18432]
            float[] txtMod = ModParams(temb, $"{b}.txt_mod.1.weight", $"{b}.txt_mod.1.bias"); // [2,18432]

            // ---- attention sub-layer ----
            using Tensor imgN1 = LayerNormNoAffine(img, imgSeq);
            using Tensor txtN1 = LayerNormNoAffine(txt, txtSeq);
            // modulate norm1: x*(1+scale)+shift, gate kept for residual
            using Tensor imgMod1 = Modulate(imgN1, imgSeq, imgMod, 0, modulateIndex, out float[] imgGate1);
            using Tensor txtMod1 = Modulate(txtN1, txtSeq, txtMod, 0, null, out float[] txtGate1);

            (Tensor imgAttn, Tensor txtAttn) = JointAttention(imgMod1, txtMod1, imgSeq, txtSeq, b, rope);
            // gated residual
            GatedAddInPlace(img, imgAttn, imgSeq, imgGate1, modulateIndex); imgAttn.Dispose();
            GatedAddInPlace(txt, txtAttn, txtSeq, txtGate1, null); txtAttn.Dispose();

            // ---- MLP sub-layer ----
            using Tensor imgN2 = LayerNormNoAffine(img, imgSeq);
            using Tensor txtN2 = LayerNormNoAffine(txt, txtSeq);
            using Tensor imgMod2 = Modulate(imgN2, imgSeq, imgMod, 1, modulateIndex, out float[] imgGate2);
            using Tensor txtMod2 = Modulate(txtN2, txtSeq, txtMod, 1, null, out float[] txtGate2);

            using (Tensor imgMlp = GeGluMlp(imgMod2, $"{b}.img_mlp"))
                GatedAddInPlace(img, imgMlp, imgSeq, imgGate2, modulateIndex);
            using (Tensor txtMlp = GeGluMlp(txtMod2, $"{b}.txt_mlp"))
                GatedAddInPlace(txt, txtMlp, txtSeq, txtGate2, null);
        }

        // joint bidirectional attention over concat[txt, img]
        private (Tensor img, Tensor txt) JointAttention(Tensor imgMod, Tensor txtMod, int imgSeq, int txtSeq,
            string b, DitRope rope)
        {
            // projections (+bias)
            Tensor iq = LinearBias(imgMod, $"{b}.attn.to_q.weight", $"{b}.attn.to_q.bias");
            Tensor ik = LinearBias(imgMod, $"{b}.attn.to_k.weight", $"{b}.attn.to_k.bias");
            Tensor iv = LinearBias(imgMod, $"{b}.attn.to_v.weight", $"{b}.attn.to_v.bias");
            Tensor tq = LinearBias(txtMod, $"{b}.attn.add_q_proj.weight", $"{b}.attn.add_q_proj.bias");
            Tensor tk = LinearBias(txtMod, $"{b}.attn.add_k_proj.weight", $"{b}.attn.add_k_proj.bias");
            Tensor tv = LinearBias(txtMod, $"{b}.attn.add_v_proj.weight", $"{b}.attn.add_v_proj.bias");

            // QK RMSNorm per head + interleaved RoPE
            QkNormHead(iq, imgSeq, $"{b}.attn.norm_q.weight");
            QkNormHead(ik, imgSeq, $"{b}.attn.norm_k.weight");
            QkNormHead(tq, txtSeq, $"{b}.attn.norm_added_q.weight");
            QkNormHead(tk, txtSeq, $"{b}.attn.norm_added_k.weight");
            ApplyRope(iq, imgSeq, rope.ImgCos, rope.ImgSin);
            ApplyRope(ik, imgSeq, rope.ImgCos, rope.ImgSin);
            ApplyRope(tq, txtSeq, rope.TxtCos, rope.TxtSin);
            ApplyRope(tk, txtSeq, rope.TxtCos, rope.TxtSin);

            // concat [txt, img]
            int total = txtSeq + imgSeq;
            using Tensor q = ConcatRows(tq, txtSeq, iq, imgSeq); tq.Dispose(); iq.Dispose();
            using Tensor k = ConcatRows(tk, txtSeq, ik, imgSeq); tk.Dispose(); ik.Dispose();
            using Tensor v = ConcatRows(tv, txtSeq, iv, imgSeq); tv.Dispose(); iv.Dispose();

            Tensor qH = ReshapeToHeads(q, NumHeads, total, HeadDim);
            Tensor kH = ReshapeToHeads(k, NumHeads, total, HeadDim);
            Tensor vH = ReshapeToHeads(v, NumHeads, total, HeadDim);
            float scale = 1.0f / MathF.Sqrt(HeadDim);
            using Tensor kT = kH.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, NumHeads, total, total);
            Ops.AddmmBatch(scores, 0, scores, scale, qH, kT);
            qH.Dispose(); kH.Dispose();
            Ops.Softmax(scores, scores);   // bidirectional, no mask (no text padding)
            var attn = new Tensor(_allocator, DType.Float32, NumHeads, total, HeadDim);
            Ops.AddmmBatch(attn, 0, attn, 1.0f, scores, vH);
            scores.Dispose(); vH.Dispose();
            using Tensor flat = ReshapeFromHeads(attn, NumHeads, total, HeadDim); attn.Dispose(); // [total, 3072]

            // split [txt, img]
            using Tensor txtPart = Ops.NewContiguous(flat.Narrow(0, 0, txtSeq));
            using Tensor imgPart = Ops.NewContiguous(flat.Narrow(0, txtSeq, imgSeq));
            Tensor imgOut = LinearBias(imgPart, $"{b}.attn.to_out.0.weight", $"{b}.attn.to_out.0.bias");
            Tensor txtOut = LinearBias(txtPart, $"{b}.attn.to_add_out.weight", $"{b}.attn.to_add_out.bias");
            return (imgOut, txtOut);
        }

        // GELU MLP: net.0.proj (Linear 3072->12288) -> GELU -> net.2 (Linear 12288->3072)
        private Tensor GeGluMlp(Tensor x, string prefix)
        {
            Tensor h = LinearBias(x, $"{prefix}.net.0.proj.weight", $"{prefix}.net.0.proj.bias");
            GeluInPlace(h);
            Tensor outp = LinearBias(h, $"{prefix}.net.2.weight", $"{prefix}.net.2.bias");
            h.Dispose();
            return outp;
        }

        // ---- host-pointer kernels ------------------------------------------

        private unsafe Tensor LinearBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = ScaledLinear(input, weightName);
            if (_weights.TryGetValue(biasName, out var bias))
            {
                int rows = (int)result.Sizes[0], outDim = (int)result.Sizes[1];
                float* r = GetFloatPtr(result); float* bp = GetFloatPtr(bias);
                int dim = Math.Min(outDim, (int)bias.ElementCount());
                Parallel.For(0, rows, s => { float* row = r + (long)s * outDim; for (int d = 0; d < dim; d++) row[d] += bp[d]; });
            }
            return result;
        }

        // ggml quantizes activations on-the-fly to q8_1, whose per-block sum is stored in
        // FP16; large activations (this DiT's residual stream legitimately grows to ~1e8 in
        // the late blocks) overflow that FP16 sum -> inf. Since matmul is linear, scale the
        // input down by k, run the quantized matmul, then scale the result back up by k.
        private unsafe Tensor ScaledLinear(Tensor input, string weightName)
        {
            const float Threshold = 1024f;
            int n = (int)input.ElementCount();
            float* ip = GetFloatPtr(input);
            float amax = 0f;
            for (int i = 0; i < n; i++) { float a = MathF.Abs(ip[i]); if (a > amax) amax = a; }
            if (amax <= Threshold || amax == 0f) return LinearForward(input, weightName);

            float k = amax / Threshold, invk = 1f / k;
            using Tensor scaled = CloneTensor(input);
            float* sp = GetFloatPtr(scaled);
            Parallel.For(0, n, i => sp[i] *= invk);
            InvalidateTensorDeviceCache(scaled);
            Tensor r = LinearForward(scaled, weightName);
            int rn = (int)r.ElementCount();
            float* rp = GetFloatPtr(r);
            Parallel.For(0, rn, i => rp[i] *= k);
            InvalidateTensorDeviceCache(r);
            return r;
        }

        // LayerNorm with NO affine (elementwise_affine=False): (x-mean)/sqrt(var+eps) over features.
        private unsafe Tensor LayerNormNoAffine(Tensor x, int rows)
        {
            var outp = new Tensor(_allocator, DType.Float32, rows, Dim);
            float* xp = GetFloatPtr(x); float* op = GetFloatPtr(outp);
            Parallel.For(0, rows, s =>
            {
                float* xr = xp + (long)s * Dim; float* orow = op + (long)s * Dim;
                double mean = 0; for (int d = 0; d < Dim; d++) mean += xr[d]; mean /= Dim;
                double var = 0; for (int d = 0; d < Dim; d++) { double v = xr[d] - mean; var += v * v; } var /= Dim;
                float inv = (float)(1.0 / Math.Sqrt(var + Eps));
                for (int d = 0; d < Dim; d++) orow[d] = (float)((xr[d] - mean) * inv);
            });
            return outp;
        }

        // modulate: out = x*(1+scale)+shift, returns gate. modParams = [2, 18432]; half=0 -> mod1, half=1 -> mod2.
        private unsafe Tensor Modulate(Tensor x, int rows, float[] modParams, int half, int[] modulateIndex, out float[] gate)
        {
            // modParams layout per row: [mod1(9216) | mod2(9216)]; each = [shift(3072)|scale(3072)|gate(3072)]
            int baseOff = half * 3 * Dim;   // within the 18432 row, offset of this half
            var outp = new Tensor(_allocator, DType.Float32, rows, Dim);
            gate = new float[(long)rows * Dim];
            float* xp = GetFloatPtr(x); float* op = GetFloatPtr(outp);
            var g = gate;
            Parallel.For(0, rows, s =>
            {
                int idx = modulateIndex != null ? modulateIndex[s] : 0;   // 0 or 1
                long mbase = (long)idx * 18432 + baseOff;
                float* xr = xp + (long)s * Dim; float* orow = op + (long)s * Dim;
                for (int d = 0; d < Dim; d++)
                {
                    float shift = modParams[mbase + d];
                    float scale = modParams[mbase + Dim + d];
                    orow[d] = xr[d] * (1f + scale) + shift;
                    g[(long)s * Dim + d] = modParams[mbase + 2 * Dim + d];
                }
            });
            return outp;
        }

        // residual += gate * sublayer
        private unsafe void GatedAddInPlace(Tensor residual, Tensor sublayer, int rows, float[] gate, int[] modulateIndex)
        {
            float* rp = GetFloatPtr(residual); float* sp = GetFloatPtr(sublayer);
            Parallel.For(0, rows, s =>
            {
                long o = (long)s * Dim;
                for (int d = 0; d < Dim; d++) rp[o + d] += gate[o + d] * sp[o + d];
            });
            InvalidateTensorDeviceCache(residual);
        }

        // RMSNorm each head's HeadDim vector with a [HeadDim] weight (QK norm).
        private unsafe void QkNormHead(Tensor qkv, int rows, string weightName)
        {
            var w = _weights[weightName];
            float* p = GetFloatPtr(qkv); float* wp = GetFloatPtr(w);
            Parallel.For(0, rows, s =>
            {
                for (int h = 0; h < NumHeads; h++)
                {
                    float* head = p + (long)s * Dim + (long)h * HeadDim;
                    double ss = 0; for (int d = 0; d < HeadDim; d++) ss += (double)head[d] * head[d];
                    float inv = (float)(1.0 / Math.Sqrt(ss / HeadDim + Eps));
                    for (int d = 0; d < HeadDim; d++) head[d] = head[d] * inv * wp[d];
                }
            });
            InvalidateTensorDeviceCache(qkv);
        }

        // Interleaved RoPE: per token, 64 complex pairs (2i,2i+1) rotated by cos/sin[token,i], shared across heads.
        private unsafe void ApplyRope(Tensor qkv, int rows, float[] cos, float[] sin)
        {
            int half = HeadDim / 2;
            float* p = GetFloatPtr(qkv);
            Parallel.For(0, rows, s =>
            {
                for (int h = 0; h < NumHeads; h++)
                {
                    float* head = p + (long)s * Dim + (long)h * HeadDim;
                    for (int i = 0; i < half; i++)
                    {
                        float c = cos[(long)s * half + i], sn = sin[(long)s * half + i];
                        float a = head[2 * i], bb = head[2 * i + 1];
                        head[2 * i] = a * c - bb * sn;
                        head[2 * i + 1] = a * sn + bb * c;
                    }
                }
            });
            InvalidateTensorDeviceCache(qkv);
        }

        private unsafe void GeluInPlace(Tensor t)
        {
            int n = (int)t.ElementCount();
            float* p = GetFloatPtr(t);
            Parallel.For(0, n, i =>
            {
                float x = p[i];
                // tanh approximation (diffusers GELU(approximate="tanh"))
                p[i] = 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            });
            InvalidateTensorDeviceCache(t);
        }

        // time_text_embed -> temb[2,3072] (row0 = embed(t), row1 = embed(0))
        private unsafe Tensor TimeEmbed(float timestep01)
        {
            var proj = new float[2 * 256];
            FillSinusoid(proj, 0, timestep01 * 1000f);
            FillSinusoid(proj, 256, 0f);
            using Tensor projT = HostToTensor(proj, 2, 256);
            Tensor h = LinearBias(projT, "time_text_embed.timestep_embedder.linear_1.weight",
                                  "time_text_embed.timestep_embedder.linear_1.bias");
            SiluInPlace(h);
            Tensor temb = LinearBias(h, "time_text_embed.timestep_embedder.linear_2.weight",
                                     "time_text_embed.timestep_embedder.linear_2.bias");
            h.Dispose();
            return temb;   // [2, 3072]
        }

        // sinusoidal timestep embedding (diffusers get_timestep_embedding, dim 256, no scale/shift, cos-then-sin? )
        private static void FillSinusoid(float[] dst, int off, float t)
        {
            const int dim = 256; int half = dim / 2;
            const float maxPeriod = 10000f;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * i / half);
                float a = t * freq;
                dst[off + i] = MathF.Cos(a);
                dst[off + half + i] = MathF.Sin(a);
            }
        }

        // AdaLayerNormContinuous: norm_out.linear = SiLU(temb_row0) -> Linear(3072 -> 6144) => scale,shift
        private unsafe Tensor AdaLayerNormOut(Tensor img, float timestep01)
        {
            int rows = (int)img.Sizes[0];
            // recompute temb row0 only
            var proj = new float[256]; FillSinusoid(proj, 0, timestep01 * 1000f);
            using Tensor projT = HostToTensor(proj, 1, 256);
            using Tensor h = LinearBias(projT, "time_text_embed.timestep_embedder.linear_1.weight",
                                  "time_text_embed.timestep_embedder.linear_1.bias");
            SiluInPlace(h);
            using Tensor temb = LinearBias(h, "time_text_embed.timestep_embedder.linear_2.weight",
                                     "time_text_embed.timestep_embedder.linear_2.bias"); // [1,3072]
            using Tensor tembAct = CloneTensor(temb); SiluInPlace(tembAct);
            using Tensor sc = LinearBias(tembAct, "norm_out.linear.weight", "norm_out.linear.bias"); // [1, 6144]
            using Tensor normed = LayerNormNoAffine(img, rows);
            var outp = new Tensor(_allocator, DType.Float32, rows, Dim);
            float* np_ = GetFloatPtr(normed); float* op = GetFloatPtr(outp); float* scp = GetFloatPtr(sc);
            // AdaLayerNormContinuous: x*(1+scale)+shift, with [scale, shift] = chunk(2) (scale first)
            Parallel.For(0, rows, s =>
            {
                float* nr = np_ + (long)s * Dim; float* orow = op + (long)s * Dim;
                for (int d = 0; d < Dim; d++)
                    orow[d] = nr[d] * (1f + scp[d]) + scp[Dim + d];
            });
            return outp;
        }

        // mod params: SiLU(temb) -> Linear(weight,bias) -> host [2,18432]
        private float[] ModParams(Tensor temb, string weightName, string biasName)
        {
            using Tensor act = CloneTensor(temb);
            SiluInPlace(act);
            using Tensor mod = LinearBias(act, weightName, biasName);  // [2, 18432]
            return TensorToHost(mod, (long)mod.Sizes[0] * mod.Sizes[1]);
        }

        // ---- small tensor helpers ----
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
        private unsafe Tensor CloneTensor(Tensor t)
        {
            int rows = (int)t.Sizes[0], cols = (int)t.Sizes[1];
            var c = new Tensor(_allocator, DType.Float32, rows, cols);
            float* sp = GetFloatPtr(t); float* dp = GetFloatPtr(c);
            Buffer.MemoryCopy(sp, dp, (long)rows * cols * 4, (long)rows * cols * 4);
            InvalidateTensorDeviceCache(c);
            return c;
        }
        private unsafe void SiluInPlace(Tensor t)
        {
            int n = (int)t.ElementCount(); float* p = GetFloatPtr(t);
            Parallel.For(0, n, i => { float x = p[i]; p[i] = x / (1f + MathF.Exp(-x)); });
            InvalidateTensorDeviceCache(t);
        }
        private Tensor ConcatRows(Tensor a, int aRows, Tensor b, int bRows)
        {
            var outp = new Tensor(_allocator, DType.Float32, aRows + bRows, Dim);
            using (var top = outp.Narrow(0, 0, aRows)) Ops.Copy(top, a);
            using (var bot = outp.Narrow(0, aRows, bRows)) Ops.Copy(bot, b);
            return outp;
        }
    }

    /// <summary>Precomputed interleaved RoPE cos/sin tables for the DiT (img + txt streams).</summary>
    internal sealed class DitRope
    {
        public float[] ImgCos, ImgSin;   // [imgSeq * 64]
        public float[] TxtCos, TxtSin;   // [txtSeq * 64]

        private const float Theta = 10000f;
        // axis complex-half sizes: 16/2, 56/2, 56/2 = 8, 28, 28 (sum 64 = head_dim/2)
        private static readonly int[] AxisDim = { 16, 56, 56 };

        // freq for axis dim D, complex index j: theta^(-2j/D)
        private static float Freq(int D, int j) => (float)Math.Pow(Theta, -2.0 * j / D);

        /// <summary>
        /// Build interleaved multi-axis RoPE tables. img_shapes are (frame, h, w) in latent-patch
        /// units, ordered [generated, reference...]; tokens are concatenated in that order.
        /// </summary>
        public static DitRope Build((int f, int h, int w)[] imgShapes, int txtSeq)
        {
            int imgSeq = 0, maxVidIndex = 0;
            foreach (var s in imgShapes)
            {
                imgSeq += s.f * s.h * s.w;
                maxVidIndex = Math.Max(maxVidIndex, Math.Max(s.h / 2, s.w / 2));
            }

            var imgCos = new float[(long)imgSeq * 64];
            var imgSin = new float[(long)imgSeq * 64];
            int tok = 0;
            for (int idx = 0; idx < imgShapes.Length; idx++)
            {
                var (f, h, w) = imgShapes[idx];
                for (int fi = 0; fi < f; fi++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                        {
                            long o = (long)tok * 64;
                            int col = 0;
                            // frame axis (8 complex), position = image idx (+ fi for video)
                            WriteAxis(imgCos, imgSin, o, ref col, AxisDim[0], idx + fi);
                            // height axis (28), centered position
                            WriteAxis(imgCos, imgSin, o, ref col, AxisDim[1], hi - (h - h / 2));
                            // width axis (28), centered position
                            WriteAxis(imgCos, imgSin, o, ref col, AxisDim[2], wi - (w - w / 2));
                            tok++;
                        }
            }

            var txtCos = new float[(long)txtSeq * 64];
            var txtSin = new float[(long)txtSeq * 64];
            for (int t = 0; t < txtSeq; t++)
            {
                long o = (long)t * 64;
                int col = 0, pos = maxVidIndex + t;
                WriteAxis(txtCos, txtSin, o, ref col, AxisDim[0], pos);
                WriteAxis(txtCos, txtSin, o, ref col, AxisDim[1], pos);
                WriteAxis(txtCos, txtSin, o, ref col, AxisDim[2], pos);
            }

            return new DitRope { ImgCos = imgCos, ImgSin = imgSin, TxtCos = txtCos, TxtSin = txtSin };
        }

        private static void WriteAxis(float[] cos, float[] sin, long baseOff, ref int col, int D, int pos)
        {
            int n = D / 2;
            for (int j = 0; j < n; j++)
            {
                float ang = pos * Freq(D, j);
                cos[baseOff + col] = MathF.Cos(ang);
                sin[baseOff + col] = MathF.Sin(ang);
                col++;
            }
        }
    }
}
