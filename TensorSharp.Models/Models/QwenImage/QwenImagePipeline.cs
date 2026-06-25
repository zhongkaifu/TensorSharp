// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Orchestrates the Qwen-Image-Edit denoising pipeline: preprocess the input image,
    /// VAE-encode it to a (normalized, packed) reference latent, build text conditioning,
    /// run the FlowMatch-Euler DiT denoise loop (true-CFG, reference-latent concatenation),
    /// then VAE-decode the result back to pixels.
    /// </summary>
    internal sealed class QwenImagePipeline : IDisposable
    {
        // VAE latent normalization (diffusers AutoencoderKLQwenImage config).
        private static readonly float[] LatentsMean = {
            -0.7571f,-0.7089f,-0.9113f,0.1075f,-0.1745f,0.9653f,-0.1517f,1.5508f,
            0.4134f,-0.0715f,0.5517f,-0.3632f,-0.1922f,-0.9497f,0.2503f,-0.2921f };
        private static readonly float[] LatentsStd = {
            2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
            3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.916f };
        private const int C = QwenImageModel.VaeLatentChannels; // 16

        private readonly QwenImageModel _model;
        private QwenImageVae _vae;
        private QwenImageTextEncoder _te;
        private QwenImageDiT _dit;
        private QwenImageVisionEncoder _vision;

        public QwenImagePipeline(QwenImageModel model) { _model = model; }

        private QwenImageVae Vae => _vae ??= new QwenImageVae(_model);
        private QwenImageTextEncoder Te => _te ??= new QwenImageTextEncoder(_model.TePath, _model.Backend);
        private QwenImageDiT Dit => _dit ??= new QwenImageDiT(_model.DitGgufPath, _model.Backend);
        private QwenImageVisionEncoder Vision => _vision ??= new QwenImageVisionEncoder(_model.MmprojPath, _model.Backend);

        // Vision grounding (Stage 4): the Qwen2.5-VL vision tower + M-RoPE image conditioning are
        // VERIFIED correct against the real transformers model (block-0 cosine 0.99997; window_index,
        // M-RoPE section mapping, position_ids, tokenizer all match exactly). The remaining cosine
        // gap vs transformers is Q4_K-LLM quantization + fp32-vs-bf16 precision (same as the
        // text-only path), not a bug. This is the model's INTENDED path (the DiT was trained with
        // image-grounded conditioning), so it's on by default when the mmproj is present.
        // CAVEAT: the 264-token image conditioning makes each CPU denoise step ~128s and the grounded
        // generation needs more steps for clean output — use the CUDA path (Stage 7) for practical
        // full-quality edits, or TS_QWEN_IMAGE_NO_VISION=1 for the faster (ungrounded) text-only path.
        private bool UseVision =>
            _model.MmprojPath != null &&
            Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_NO_VISION") != "1";

        // The reference (condition) image carried into Edit so EncodePrompt can ground on it.
        private RgbImage _conditionImage;

        public RgbImage Edit(string prompt, RgbImage input, QwenImageParams p)
        {
            var phase = System.Diagnostics.Stopwatch.StartNew();
            void Phase(string n) { Console.WriteLine($"  [pipe-timing] {n}: {phase.Elapsed.TotalMilliseconds:F0}ms"); phase.Restart(); }

            _conditionImage = input;   // vision grounding uses the original (resized to 384^2 internally)
            // 1. preprocess (aspect-preserving, dims multiple of 16). Clamp the target area
            // to what the DiT attention will fit in device VRAM (avoids the 16 GB OOM/shared-
            // VRAM spill at 1 MP) unless the caller pinned explicit Width/Height.
            long area = (p.Width > 0 && p.Height > 0) ? p.TargetArea : ClampAreaToVram(p.TargetArea);
            RgbImage img = (p.Width > 0 && p.Height > 0) ? ImageIO.Resize(input, p.Width, p.Height)
                                                         : ImageIO.ResizeToArea(input, area);

            // 2. VAE encode reference -> normalize -> pack
            VaeLatent refLat = Vae.Encode(img);
            Phase($"VAE encode ({img.Width}x{img.Height})");
            int Hl = refLat.Height, Wl = refLat.Width;          // latent dims (H/8)
            NormalizeLatent(refLat.Data, Hl, Wl);
            float[] refPacked = Pack(refLat.Data, Hl, Wl);       // [refSeq, 64]
            int hp = Hl / 2, wp = Wl / 2, seq = hp * wp;

            // 3. text conditioning (prompt + negative for CFG)
            float[] cond = EncodePrompt(prompt, out int txtSeq);
            bool doCfg = p.CfgScale > 1f;
            float[] negCond = null; int negTxt = 0;
            if (doCfg) negCond = EncodePrompt(p.NegativePrompt ?? " ", out negTxt);
            Phase($"text encode (txtSeq={txtSeq})");

            // The text + vision encoders are finished for this request. On CUDA they hold
            // ~3.6 GB (Qwen2.5-VL-7B ~2.3 GB + mmproj ~1.3 GB) that is dead weight through
            // the entire denoise loop — leaving it resident pushes the DiT (7 GB) + its
            // O(n^2) attention scratch past 16 GB, spilling into shared VRAM (slow) and
            // then OOM at high resolution. Free it now; the lazy properties reload it on
            // the next Edit. Keep the VAE (small, needed again for the final decode).
            FreeEncoders();

            // 4. noise latents (packed) + reference concatenation layout
            var rng = new GaussianRng(p.Seed);
            float[] noise = new float[(long)C * Hl * Wl];
            for (long i = 0; i < noise.Length; i++) noise[i] = rng.Next();
            float[] latents = Pack(noise, Hl, Wl);               // [seq, 64], evolves

            int imgSeq = seq + seq;                              // gen + ref
            var modulateIndex = new int[imgSeq];
            for (int i = seq; i < imgSeq; i++) modulateIndex[i] = 1;

            // img_shapes: generated then reference (both same grid here)
            var imgShapes = new (int f, int h, int w)[] { (1, hp, wp), (1, hp, wp) };
            DitRope rope = DitRope.Build(imgShapes, txtSeq);
            DitRope ropeNeg = doCfg ? DitRope.Build(imgShapes, negTxt) : null;

            // 5. scheduler
            var sched = new QwenImageScheduler(p.Steps, seq);

            // 6. denoise loop
            var imgTokens = new float[(long)imgSeq * 64];
            Array.Copy(refPacked, 0, imgTokens, (long)seq * 64, (long)seq * 64);  // ref part fixed
            for (int step = 0; step < sched.Steps; step++)
            {
                Array.Copy(latents, 0, imgTokens, 0, (long)seq * 64);             // gen part = current latents
                float t01 = sched.Sigmas[step];   // FlowMatch: timestep == sigma
                float[] vel = Dit.Predict(imgTokens, imgSeq, cond, txtSeq, t01, modulateIndex, rope);
                // keep only the generated region
                var v = new float[(long)seq * 64];
                Array.Copy(vel, 0, v, 0, v.Length);

                if (doCfg)
                {
                    float[] velNeg = Dit.Predict(imgTokens, imgSeq, negCond, negTxt, t01, modulateIndex, ropeNeg);
                    // true-CFG: comb = neg + scale*(cond-neg), then renorm to cond norm (per token row)
                    TrueCfg(v, velNeg, seq, p.CfgScale);
                }
                sched.Step(latents, v, step);
                if (Environment.GetEnvironmentVariable("TS_QIMG_DEBUG") == "1")
                    Console.WriteLine($"  [pipe] step {step + 1}/{sched.Steps} sigma={t01:F3} vel {QwenImageDiT.Stat(v, v.Length)} | latent {QwenImageDiT.Stat(latents, latents.Length)}");
                else
                    Console.Write($"\r  denoise step {step + 1}/{sched.Steps}   ");
            }
            Console.WriteLine();
            Phase($"denoise ({sched.Steps} steps)");

            // 7. unpack -> denormalize -> VAE decode
            float[] outLatent = Unpack(latents, Hl, Wl);
            DenormalizeLatent(outLatent, Hl, Wl);
            RgbImage result = Vae.Decode(new VaeLatent(C, Hl, Wl, outLatent));
            Phase("VAE decode");
            return result;
        }

        private float[] EncodePrompt(string prompt, out int condSeq)
        {
            int[] tokens;
            ImageCond imgCond = null;
            if (UseVision && _conditionImage != null)
            {
                string templated = QwenImagePrompt.BuildWithImage(prompt);
                int[] baseTokens = Te.Tokenizer.Encode(templated, addSpecial: false).ToArray();
                float[] pv = QwenImageVisionProcessor.Preprocess(_conditionImage, out int gridH, out int gridW);
                int n = (gridH / 2) * (gridW / 2);
                float[] embeds = Vision.Encode(pv, gridH, gridW);    // [n, 3584]
                tokens = ExpandImagePad(baseTokens, QwenImagePrompt.ImagePadTokenId, n, out int imgStart);
                if (imgStart >= 0)
                    imgCond = new ImageCond { Start = imgStart, Count = n, GridH = gridH, GridW = gridW, Embeds = embeds };
            }
            else
            {
                tokens = Te.Tokenizer.Encode(QwenImagePrompt.Build(prompt), addSpecial: false).ToArray();
            }

            float[] full = Te.EncodeHidden(tokens, imgCond);   // [seq, 3584]
            int seq = tokens.Length;
            int hidden = Te.HiddenSize;
            int drop = Math.Min(QwenImagePrompt.DropIdx, seq - 1);
            condSeq = seq - drop;
            var cond = new float[(long)condSeq * hidden];
            Array.Copy(full, (long)drop * hidden, cond, 0, cond.Length);
            return cond;
        }

        // Replace the single <|image_pad|> placeholder with `n` copies (one per merged vision patch).
        private static int[] ExpandImagePad(int[] tokens, int padId, int n, out int start)
        {
            int idx = Array.IndexOf(tokens, padId);
            start = idx;
            if (idx < 0) return tokens;
            var outp = new int[tokens.Length - 1 + n];
            Array.Copy(tokens, 0, outp, 0, idx);
            for (int i = 0; i < n; i++) outp[idx + i] = padId;
            Array.Copy(tokens, idx + 1, outp, idx + n, tokens.Length - idx - 1);
            return outp;
        }

        // true-CFG over packed velocity rows [seq, 64]
        private static void TrueCfg(float[] cond, float[] neg, int rows, float scale)
        {
            for (int s = 0; s < rows; s++)
            {
                long o = (long)s * 64;
                double condNorm = 0, combNorm = 0;
                var comb = new float[64];
                for (int d = 0; d < 64; d++)
                {
                    float c = cond[o + d], n = neg[o + d];
                    float cb = n + scale * (c - n);
                    comb[d] = cb;
                    condNorm += (double)c * c; combNorm += (double)cb * cb;
                }
                float r = (float)(Math.Sqrt(condNorm) / (Math.Sqrt(combNorm) + 1e-12));
                for (int d = 0; d < 64; d++) cond[o + d] = comb[d] * r;
            }
        }

        // pack [C,H,W] -> [(H/2)*(W/2), C*4], token (hi,wi), channel c*4+ph*2+pw
        private static float[] Pack(float[] lat, int H, int W)
        {
            int hp = H / 2, wp = W / 2, hw = H * W;
            var outp = new float[(long)hp * wp * C * 4];
            for (int hi = 0; hi < hp; hi++)
                for (int wi = 0; wi < wp; wi++)
                {
                    long tok = (long)(hi * wp + wi) * (C * 4);
                    for (int c = 0; c < C; c++)
                        for (int ph = 0; ph < 2; ph++)
                            for (int pw = 0; pw < 2; pw++)
                                outp[tok + c * 4 + ph * 2 + pw] = lat[(long)c * hw + (hi * 2 + ph) * W + (wi * 2 + pw)];
                }
            return outp;
        }

        private static float[] Unpack(float[] packed, int H, int W)
        {
            int hp = H / 2, wp = W / 2, hw = H * W;
            var lat = new float[(long)C * hw];
            for (int hi = 0; hi < hp; hi++)
                for (int wi = 0; wi < wp; wi++)
                {
                    long tok = (long)(hi * wp + wi) * (C * 4);
                    for (int c = 0; c < C; c++)
                        for (int ph = 0; ph < 2; ph++)
                            for (int pw = 0; pw < 2; pw++)
                                lat[(long)c * hw + (hi * 2 + ph) * W + (wi * 2 + pw)] = packed[tok + c * 4 + ph * 2 + pw];
                }
            return lat;
        }

        private static void NormalizeLatent(float[] lat, int H, int W)
        {
            int hw = H * W;
            for (int c = 0; c < C; c++)
                for (int i = 0; i < hw; i++)
                    lat[(long)c * hw + i] = (lat[(long)c * hw + i] - LatentsMean[c]) / LatentsStd[c];
        }
        private static void DenormalizeLatent(float[] lat, int H, int W)
        {
            int hw = H * W;
            for (int c = 0; c < C; c++)
                for (int i = 0; i < hw; i++)
                    lat[(long)c * hw + i] = lat[(long)c * hw + i] * LatentsStd[c] + LatentsMean[c];
        }

        // Clamp the output area so the DiT's O(n^2) attention scratch fits device VRAM.
        // Peak DiT VRAM ~= reserve (weights/VAE/activations) + 2*T^2*heads*4 bytes
        // (attention scores + probs), where total tokens T ~= area/128 + txt. Solve for
        // the largest area whose scores fit the remaining budget. Only relevant on the
        // GGML CUDA backend; a no-op elsewhere or when the request already fits.
        private long ClampAreaToVram(long requestedArea)
        {
            if (_model.Backend != BackendType.GgmlCuda) return requestedArea;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out _, out long total) || total <= 0) return requestedArea;

            // Flash-attention (native default) never materializes the [total,total] scores,
            // so the DiT memory grows ~linearly with the token count instead of O(n^2). Use
            // a much looser linear budget there; only the explicit-scores path needs the
            // tight quadratic clamp.
            // Flash is default-on now that the ggml-cuda flash path is fixed (see
            // qi_flash_enabled): O(n) attention memory, so the looser linear VRAM budget
            // applies (higher resolution fits). TS_QWEN_DIT_FLASH=0 forces the
            // explicit-scores path with its tighter quadratic budget.
            bool nativeFlash =
                Environment.GetEnvironmentVariable("TS_QWEN_DIT_NATIVE") != "0" &&
                Environment.GetEnvironmentVariable("TS_QWEN_DIT_FLASH") != "0";

            long budget = (long)(total * 0.85);
            long maxArea;
            if (nativeFlash)
            {
                // Flash removes the DiT scores, but the GPU VAE conv's im2col scratch grows
                // with the output area (it peaked ~15.9 GB encoding a full 1 MP image on 16 GB)
                // and is the real high-res limit. Keep a conservative linear budget so the
                // default never oversubscribes VRAM (push higher with explicit --width/--height).
                const long reserve = 6L * 1024 * 1024 * 1024;
                const long perToken = 1700L * 1024;             // ~1.7 MB/token (VAE im2col-dominated)
                long maxT = Math.Max(256, (budget - reserve) / perToken);
                maxArea = (maxT - 400) * 128;
            }
            else
            {
                const long reserve = 12L * 1024 * 1024 * 1024;  // explicit-scores path (16 GB-calibrated: 512^2 fits ~11.6 GB clean)
                const int heads = QwenImageModel.DitNumHeads;   // 24
                long scoresBudget = Math.Max(512L * 1024 * 1024, budget - reserve);
                double maxT = Math.Sqrt(scoresBudget / (8.0 * heads));   // 2*T^2*heads*4 <= scoresBudget
                maxArea = (long)Math.Max(256, maxT - 400) * 128;
            }

            if (requestedArea > maxArea)
            {
                Console.WriteLine($"  [pipe] target area {requestedArea} exceeds the {total / 1024 / 1024} MiB VRAM budget; " +
                                  $"clamping to {maxArea} px (set explicit --width/--height to override).");
                return maxArea;
            }
            return requestedArea;
        }

        // Release the text + vision encoders (reclaim their CUDA VRAM mid-request); the
        // lazy Te/Vision properties re-create them on demand for the next request.
        private void FreeEncoders()
        {
            _te?.Dispose(); _te = null;
            _vision?.Dispose(); _vision = null;
        }

        public void Dispose()
        {
            _vae?.Dispose();
            _te?.Dispose();
            _dit?.Dispose();
            _vision?.Dispose();
        }
    }

    /// <summary>Seeded Gaussian (Box-Muller) RNG for the initial noise latent.</summary>
    internal sealed class GaussianRng
    {
        private readonly Random _r;
        private double _spare; private bool _hasSpare;
        public GaussianRng(long seed) { _r = new Random((int)(seed ^ (seed >> 32))); }
        public float Next()
        {
            if (_hasSpare) { _hasSpare = false; return (float)_spare; }
            double u, v, s;
            do { u = _r.NextDouble() * 2 - 1; v = _r.NextDouble() * 2 - 1; s = u * u + v * v; }
            while (s >= 1 || s == 0);
            double m = Math.Sqrt(-2.0 * Math.Log(s) / s);
            _spare = v * m; _hasSpare = true;
            return (float)(u * m);
        }
    }
}
