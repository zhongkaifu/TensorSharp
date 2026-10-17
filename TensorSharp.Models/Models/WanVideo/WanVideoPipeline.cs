// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan video pipeline: UMT5 conditioning (cond + uncond) -> FlowMatch denoise
// with classifier-free guidance over the Wan DiT -> causal 3D VAE decode to RGB
// frames. Mirrors the official Wan sampling recipes (diffusers WanPipeline /
// WanImageToVideoPipeline):
//   - Wan 2.1 T2V: flow shift 3/5/8 recipe, cfg 6.0, 16 fps.
//   - Wan 2.2 TI2V-5B: cfg 5.0, shift 5.0, 24 fps; image-to-video replaces the
//     first latent frame with the VAE-encoded image each step and modulates its
//     tokens at timestep 0 (diffusers expand_timesteps).
//   - Wan 2.2 A14B: two DiT experts switched at a timestep boundary (0.9 i2v /
//     0.875 t2v) with per-expert cfg; image-to-video concatenates a 4-channel
//     first-frame mask + the VAE-encoded [image, zeros...] clip (36-ch input).
using System;
using System.Diagnostics;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>A generated video: decoded frames plus the intended playback rate.</summary>
    public sealed class GeneratedVideo
    {
        public RgbImage[] Frames { get; init; }
        public int Fps { get; init; }
        public long Seed { get; init; }

        /// <summary>The decoded audio track, on models that generate audio jointly with
        /// the video (MiniMax-H3). Null for video-only models such as Wan.</summary>
        public GeneratedVideoAudio Audio { get; init; }
    }

    internal sealed class WanVideoPipeline : IDisposable
    {
        // The official Wan default negative prompt (used by the reference repo,
        // diffusers examples and stable-diffusion.cpp docs alike; unchanged in 2.2).
        public const string DefaultNegativePrompt =
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，" +
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，" +
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

        private readonly WanVideoModel _model;
        private readonly bool _direct;   // BackendType.Cuda / Cpu: direct tensor path
        private DirectContext _dctx;
        private IWanTextEncoder _te;
        private IWanDit _dit;       // single-model variants (and the A14B high-noise expert)
        private IWanDit _ditLow;    // A14B low-noise expert
        private WanVaeBase _vae;

        public WanVideoPipeline(WanVideoModel model)
        {
            _model = model;
            _direct = model.Backend is BackendType.Cuda or BackendType.Cpu;
        }

        private DirectContext Dctx => _dctx ??= new DirectContext(_model.DirectAllocator);
        private IWanTextEncoder Te => _te ??= _direct
            ? new WanDirectT5(Dctx, _model.TePath)
            : new WanTextEncoder(_model.TePath);
        private IWanDit Dit => _dit ??= _direct
            ? new WanDirectDiT(Dctx, _model.HighNoiseGguf ?? _model.DitGguf)
            : new WanDiT(_model.HighNoiseGguf ?? _model.DitGguf);
        private IWanDit DitLow => _ditLow ??= _direct
            ? new WanDirectDiT(Dctx, _model.LowNoiseGguf)
            : new WanDiT(_model.LowNoiseGguf);
        private WanVaeBase Vae => _vae ??= _direct
            ? new WanDirectVae(Dctx, _model.VaePath)
            : new WanVae(_model.VaePath);

        public GeneratedVideo Generate(string prompt, VideoGenerationParams p)
        {
            try
            {
                return GenerateCore(prompt, p ?? new VideoGenerationParams());
            }
            finally
            {
                // Hand back ALL device residency at request end (DiT weights, captured
                // graphs, reuse gallocr) so a following request — or another model on
                // the same server — starts from clean VRAM. Model objects stay loaded.
                ReleaseDeviceMemory();
            }
        }

        private void ReleaseDeviceMemory()
        {
            if (_model.Backend is BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlMetal)
            {
                // Hand back the finished stage's device residency (weights, reuse
                // gallocr). Vulkan in particular cannot fit the UMT5 + DiT + VAE
                // graph all resident on a 16 GB card.
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
            else if (_model.Backend is BackendType.Cuda && _dctx != null)
            {
                // Evict the resident quantized weights of the finished stage (the
                // UMT5's ~6 GB, the DiT's blocks) and hand the pool's cached
                // transients back — the next stage's tensor shapes are disjoint
                // (denoise [seq, dim] rows vs VAE [T, H, W, C] planes).
                TensorSharp.Cuda.CudaQuantizedOps.ClearDeviceCache(_dctx.CudaAllocator);
                _dctx.CudaAllocator.TrimPool();
            }
        }

        /// <summary>The frames a Wan checkpoint can pin: at most a first frame, which an I2V-A14B checkpoint
        /// requires and a T2V-only checkpoint refuses. A last frame is refused rather than silently dropped.</summary>
        public static void ValidateConditioning(VideoGenerationParams p, bool hasImage, bool supportsImage, bool requiresImage)
        {
            ArgumentNullException.ThrowIfNull(p);
            if (p.EndImage != null || !string.IsNullOrWhiteSpace(p.EndImagePath))
                throw new NotSupportedException(
                    "Wan takes at most a first-frame image; it cannot pin the last frame. " +
                    "MiniMax-H3 FL2VA or LTX 2.5 can end a clip on an image.");
            if (hasImage && !supportsImage)
                throw new NotSupportedException(
                    "This Wan checkpoint is text-to-video only; image-to-video needs a Wan 2.2 model " +
                    "(Wan2.2-TI2V-5B or Wan2.2-I2V-A14B).");
            if (requiresImage && !hasImage)
                throw new ArgumentException(
                    "Wan2.2-I2V-A14B is an image-to-video model: supply a conditioning image " +
                    "(CLI --image, API \"image\"/\"imagePath\").");
        }

        private GeneratedVideo GenerateCore(string prompt, VideoGenerationParams p)
        {
            // On the GGML backends the networks run through the native whole-graph
            // kernels; make sure that backend is the active one. The direct
            // Cuda/Cpu path runs on the model's allocator instead.
            if (!_direct)
            {
                GgmlBasicOps.EnsureBackendAvailable(_model.Backend switch
                {
                    BackendType.GgmlCuda => GgmlBackendType.Cuda,
                    BackendType.GgmlMetal => GgmlBackendType.Metal,
                    BackendType.GgmlVulkan => GgmlBackendType.Vulkan,
                    _ => GgmlBackendType.Cpu,
                });
            }

            var total = Stopwatch.StartNew();
            var phase = Stopwatch.StartNew();
            void Phase(string n) { Console.WriteLine($"  [wan-timing] {n}: {phase.Elapsed.TotalMilliseconds:F0}ms"); phase.Restart(); }

            // One DiT pass over a 720p/121-frame latent is minutes of uninterrupted
            // GPU work inside a single native call. Without a heartbeat the console
            // and the Web UI go silent for that whole time, which reads as a hang.
            using var beat = new WanHeartbeat(p, total);

            WanVariant variant = _model.Variant;
            bool ti2v = variant == WanVariant.TI2V;
            bool a14b = variant == WanVariant.A14B;
            bool dualExpert = a14b && _model.LowNoiseGguf != null && _model.HighNoiseGguf != _model.LowNoiseGguf;
            int zc = _model.LatentChannels;                   // 16 / 48
            int spatial = _model.VaeSpatialScale;             // 8 / 16
            int grid = spatial * WanDiT.PatchH;               // pixel grid: 16 / 32

            // ---- conditioning image (image-to-video) ----
            RgbImage image = p.ResolveImage();
            bool i2v = image != null;
            ValidateConditioning(p, i2v, _model.SupportsImage, _model.RequiresImage);

            // ---- resolve parameters ----
            // Default resolution follows each model's official recipe: TI2V-5B is a
            // 720p model (1280x704 area — the diffusers / Wan-repo default), the
            // others default to the 480p recipe (832x480 area).
            int defW = ti2v ? 1280 : 832, defH = ti2v ? 704 : 480;
            int width, height;
            if (i2v && (p.Width <= 0 || p.Height <= 0))
            {
                // Fit the default target area to the conditioning image's aspect ratio.
                long area = (long)(p.Width > 0 ? p.Width : defW) * (p.Height > 0 ? p.Height : defH);
                double ar = (double)image.Width / image.Height;
                height = SnapDim((int)Math.Round(Math.Sqrt(area / ar)), grid);
                width = SnapDim((int)Math.Round(Math.Sqrt(area / ar) * ar), grid);
            }
            else
            {
                width = SnapDim(p.Width > 0 ? p.Width : defW, grid);
                height = SnapDim(p.Height > 0 ? p.Height : defH, grid);
            }
            int frames = p.Frames > 0 ? SnapFrames(p.Frames) : (ti2v ? 49 : 33);
            // A step-distilled checkpoint replaces the whole sampling recipe: it is
            // trained to land in a handful of steps AND without classifier-free
            // guidance, so it costs one pass per step instead of two. Applying the
            // base recipe to one wastes 25x the work and looks worse.
            int distilled = _model.DistilledSteps;
            int steps = p.Steps > 0 ? p.Steps
                : distilled > 0 ? distilled
                : ti2v ? 50 : a14b ? 40 : 30;
            float cfg = p.CfgScale > 0 ? p.CfgScale
                : distilled > 0 ? 1.0f
                : ti2v ? 5.0f
                : a14b ? (i2v ? 3.5f : 4.0f)
                : 6.0f;
            float cfg2 = p.CfgScale2 > 0 ? p.CfgScale2
                : distilled > 0 ? 1.0f
                : a14b ? (i2v ? 3.5f : 3.0f)
                : cfg;
            // FlowMatch shift. Wan 2.2 recipes: 5.0 (12.0 for A14B T2V). Wan 2.1: 8.0
            // for the 1.3B model's video runs, else 3.0 at <= 480p / 5.0 above; stills
            // keep the low shift (a high shift softens texture on stills).
            float shift = p.FlowShift > 0 ? p.FlowShift
                : ti2v ? 5.0f
                : a14b ? (i2v ? 5.0f : 12.0f)
                : Dit.Dim <= 1536 && frames > 1 ? 8.0f
                : Math.Min(width, height) > 480 ? 5.0f : 3.0f;
            int fps = p.Fps > 0 ? p.Fps : ti2v ? 24 : 16;
            long seed = p.Seed >= 0 ? p.Seed : Random.Shared.NextInt64() & 0x7fffffff;
            string negPrompt = p.NegativePrompt ?? DefaultNegativePrompt;
            bool useCfg = cfg > 1.0f || (dualExpert && cfg2 > 1.0f);
            bool useEuler = string.Equals(p.Sampler, "euler", StringComparison.OrdinalIgnoreCase);
            float boundaryT = dualExpert
                ? (p.BoundaryRatio > 0 ? p.BoundaryRatio : i2v ? 0.9f : 0.875f) * 1000f
                : -1f;

            int lw = width / spatial;
            int lh = height / spatial;
            int lt = 1 + (frames - 1) / WanVae.TemporalScale;
            int hLen = lh / WanDiT.PatchH, wLen = lw / WanDiT.PatchW;
            int seq = lt * hLen * wLen;
            int hw = hLen * wLen;                 // tokens per latent frame
            int zTok = zc * WanDiT.PatchH * WanDiT.PatchW;   // latent features per token

            string family = ti2v ? "2.2-TI2V" : a14b ? "2.2-A14B" : "2.1";
            // Wan is trained at 480p (832x480 area) and 720p (1280x704/720). Far below
            // that the DiT is out of distribution and produces soft, unstable video no
            // matter how many steps are spent — warn instead of failing silently.
            if ((long)width * height < 300_000)
                Console.WriteLine($"  [wan] WARNING: {width}x{height} is far below Wan's training resolutions " +
                                  $"(480p: 832x480, 720p: 1280x704{(ti2v ? ", the TI2V-5B native recipe" : "")}); " +
                                  "expect soft, distorted results. Prefer generating at a supported resolution " +
                                  "(e.g. --width 480 --height 704 for portrait) and downscaling afterwards.");
            // Guidance-delta cache (opt-in; see VideoGenerationParams.CfgCacheStride).
            int cfgStride = useCfg ? Math.Max(0, p.CfgCacheStride) : 0;
            int passes = steps;
            if (useCfg)
                for (int i = 0; i < steps; i++)
                    if (UsesUncondPass(i, steps, cfgStride)) passes++;
            Console.WriteLine($"Wan {family} {(i2v ? "I2V" : "T2V")}: {width}x{height}x{frames}f " +
                              $"({lt}x{hLen}x{wLen} = {seq} tokens), {steps} steps, " +
                              $"{(useEuler ? "euler" : "unipc")}, cfg {cfg}{(dualExpert ? $"/{cfg2}" : "")}, " +
                              $"shift {shift}, seed {seed}");
            // Past a few thousand tokens the quadratic self-attention term dominates a
            // pass, and a pass becomes minutes long — say so up front, because the
            // request that produced it looks identical to a cheap one.
            if (seq > 8000)
                Console.WriteLine($"  [wan] large request: {passes} DiT passes over {seq} tokens. Self-attention " +
                                  $"costs O(tokens^2) and dominates at this size, so halving the frame count or " +
                                  $"the frame area makes it ~4x cheaper. A per-pass timing and an ETA follow the " +
                                  $"first pass; progress ticks every 30s (TS_WAN_HEARTBEAT_S).");

            // ---- 1. text conditioning (both prompts), then free the TE's VRAM ----
            beat.Set("text-encode", 0, steps, "UMT5-XXL");
            float[] ctxCond = Te.Encode(prompt);
            float[] ctxNeg = useCfg ? Te.Encode(negPrompt) : null;
            // The UMT5 weights (several GB resident) are not needed again this
            // request; release them so the VAE encoder + DiT get the VRAM.
            ReleaseDeviceMemory();
            Phase("text-encode");

            // ---- 2. image conditioning (I2V): VAE-encode the first frame ----
            // TI2V: the image latent replaces latent frame 0; its tokens are modulated
            // at timestep 0 (condTok, seq0 = hw). A14B: 4 mask channels + the encoded
            // [image, zeros...] clip concat behind the noise channels (condTok, 80/token).
            float[] condTok = null;
            int inTok = ti2v || !i2v ? zTok : (zc + 4 + zc) * WanDiT.PatchH * WanDiT.PatchW;
            if (i2v)
            {
                beat.Set("image-encode", 0, steps, "VAE encode");
                var resized = QwenImage.ImageIO.Resize(image, width, height);
                if (ti2v)
                {
                    float[] pixels = ToPlanarSigned(resized, 1, height, width);
                    float[] condLat = Vae.Encode(pixels, 1, height, width);   // [48, 1, lh, lw]
                    condTok = new float[(long)zTok * hw];
                    Patchify(condLat, condTok, 1, lh, lw, zc, zTok, 0);
                }
                else
                {
                    // pixel clip = [image, zeros x (frames-1)] (diffusers WanImageToVideoPipeline)
                    float[] pixels = ToPlanarSigned(resized, frames, height, width);
                    float[] imgLat = Vae.Encode(pixels, frames, height, width);  // [16, lt, lh, lw]
                    int condC = 4 + zc;
                    var cond = new float[(long)condC * lt * lh * lw];
                    // mask channels: latent frame 0 = 1 (the conditioned first frame), rest 0
                    long lhw = (long)lh * lw;
                    for (int c = 0; c < 4; c++)
                        for (long i = 0; i < lhw; i++)
                            cond[(long)c * lt * lhw + i] = 1f;
                    Array.Copy(imgLat, 0, cond, 4L * lt * lhw, imgLat.LongLength);
                    condTok = new float[(long)(condC * WanDiT.PatchH * WanDiT.PatchW) * seq];
                    Patchify(cond, condTok, lt, lh, lw, condC, condC * WanDiT.PatchH * WanDiT.PatchW, 0);
                }
                ReleaseDeviceMemory();   // free the VAE encoder's VRAM before the DiT loads
                Phase("image-encode");
            }

            // ---- 3. init noise + denoise loop ----
            var rope = WanRope.Build(lt, hLen, wLen);
            WanScheduler euler = useEuler ? new WanScheduler(steps, shift) : null;
            WanUniPCScheduler unipc = useEuler ? null : new WanUniPCScheduler(steps, shift);
            float[] timesteps = useEuler ? euler.Timesteps : unipc.Timesteps;
            var x = new float[(long)zTok * seq];
            var rng = new GaussianRng(seed);
            // Noise is drawn in the [zc, t, h, w] latent order (matching the reference
            // pipelines), then patchified — the denoise itself runs in token space.
            var noise = new float[(long)zc * lt * lh * lw];
            for (long i = 0; i < noise.LongLength; i++) noise[i] = rng.Next();
            Patchify(noise, x, lt, lh, lw, zc, zTok, 0);
            Phase("init");

            // TS_WAN_DEBUG_DUMP=<dir>: dump the first step's DiT inputs/outputs for
            // numeric comparison against a reference implementation.
            string dumpDir = Environment.GetEnvironmentVariable("TS_WAN_DEBUG_DUMP");
            if (!string.IsNullOrEmpty(dumpDir))
            {
                System.IO.Directory.CreateDirectory(dumpDir);
                DumpF32(System.IO.Path.Combine(dumpDir, "ctx_cond.bin"), ctxCond);
                if (ctxNeg != null) DumpF32(System.IO.Path.Combine(dumpDir, "ctx_neg.bin"), ctxNeg);
                DumpF32(System.IO.Path.Combine(dumpDir, "noise_latent.bin"), noise);
                DumpF32(System.IO.Path.Combine(dumpDir, "x0_tokens.bin"), x);
                if (condTok != null) DumpF32(System.IO.Path.Combine(dumpDir, "cond_tokens.bin"), condTok);
                System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "meta.txt"),
                    $"seq={seq} lt={lt} hLen={hLen} wLen={wLen} t0={timesteps[0]} inTok={inTok} " +
                    $"zTok={zTok} variant={variant} i2v={i2v}");
            }

            // A14B I2V: the model input is [noise 64 | mask+image 80] per token; the
            // conditioning features are constant across steps.
            float[] xFull = !ti2v && i2v ? new float[(long)inTok * seq] : null;
            if (xFull != null)
            {
                int condW = inTok - zTok;
                System.Threading.Tasks.Parallel.For(0, seq, tok =>
                {
                    Array.Copy(condTok, (long)tok * condW, xFull, (long)tok * inTok + zTok, condW);
                });
            }

            bool usedLowExpert = false;
            int passCount = 0, cfgCacheHits = 0;
            double passSeconds = 0;
            float[] dCache = cfgStride > 1 ? new float[(long)zTok * seq] : null;
            if (dCache != null)
                Console.WriteLine($"  [wan] guidance cache: unconditional pass every {cfgStride} steps " +
                                  $"after a {CfgCacheWarmup}-step warm-up (approximate; cfgCacheStride=1 disables)");
            for (int i = 0; i < steps; i++)
            {
                float t = timesteps[i];
                IWanDit cur = Dit;
                float curCfg = cfg;
                if (dualExpert && t < boundaryT)
                {
                    if (!usedLowExpert)
                    {
                        // One-time expert switch: hand the high-noise expert's resident
                        // weights and captured graphs back before the low expert loads.
                        usedLowExpert = true;
                        ReleaseDeviceMemory();
                        Console.WriteLine($"  [wan] switching to the low-noise expert at t={t:F1}");
                    }
                    cur = DitLow;
                    curCfg = cfg2;
                }

                // Build the model input for this step.
                float[] xIn = x;
                int seq0 = 0;
                if (ti2v && i2v)
                {
                    // frame-0 tokens run the clean image latent at timestep 0
                    xIn = (float[])x.Clone();
                    Array.Copy(condTok, 0, xIn, 0, condTok.Length);
                    seq0 = hw;
                }
                else if (xFull != null)
                {
                    System.Threading.Tasks.Parallel.For(0, seq, tok =>
                    {
                        Array.Copy(x, (long)tok * zTok, xFull, (long)tok * inTok, zTok);
                    });
                    xIn = xFull;
                }

                var pass = Stopwatch.StartNew();
                beat.Set("denoise", i + 1, steps, "cond pass");
                float[] vCond = cur.Predict(xIn, ctxCond, t, rope, seq, seq0);
                double condS = pass.Elapsed.TotalSeconds;
                passCount++;
                passSeconds += condS;
                if (i == 0 && !string.IsNullOrEmpty(dumpDir))
                {
                    DumpF32(System.IO.Path.Combine(dumpDir, "v0_cond.bin"), vCond);
                    Console.WriteLine($"  [wan-debug] ctx {Stat(ctxCond)} | x {Stat(xIn)} | v {Stat(vCond)}");
                }
                float[] v = vCond;
                double negS = 0;
                if (useCfg)
                {
                    // v = v_cond + (cfg-1) * d, d = v_cond - v_uncond. With the guidance
                    // cache on, d is only recomputed every cfgStride steps (always over
                    // the warm-up and on the final step, where the result is most
                    // sensitive); in between the cached d is reused and the unconditional
                    // pass is skipped entirely.
                    bool computeNeg = dCache == null || UsesUncondPass(i, steps, cfgStride);
                    if (computeNeg)
                    {
                        pass.Restart();
                        beat.Set("denoise", i + 1, steps, "uncond pass");
                        float[] vNeg = cur.Predict(xIn, ctxNeg, t, rope, seq, seq0);
                        negS = pass.Elapsed.TotalSeconds;
                        passCount++;
                        passSeconds += negS;
                        if (dCache != null)
                            for (long j = 0; j < v.LongLength; j++) dCache[j] = vCond[j] - vNeg[j];
                        for (long j = 0; j < v.LongLength; j++)
                            v[j] = vNeg[j] + curCfg * (vCond[j] - vNeg[j]);
                    }
                    else
                    {
                        cfgCacheHits++;
                        for (long j = 0; j < v.LongLength; j++)
                            v[j] = vCond[j] + (curCfg - 1f) * dCache[j];
                    }
                }
                if (euler != null) euler.Step(x, v, i);
                else x = unipc.Step(x, v);
                p.OnStep?.Invoke(i + 1, steps);

                // ETA from the mean pass so far. The first pass carries the one-time
                // weight upload and graph build, so it over-estimates slightly; the
                // running mean corrects itself from step 2 on.
                double eta = passCount > 0 ? passSeconds / passCount * (passes - passCount) : -1;
                Console.WriteLine($"  step {i + 1}/{steps} t={t:F1} ({phase.Elapsed.TotalSeconds:F1}s" +
                                  (useCfg ? $": cond {condS:F1}s + uncond {negS:F1}s" : "") +
                                  $") — {FormatEta(eta)} left");
                beat.Report("denoise", i + 1, steps, "step done", eta, heartbeat: false);
                phase.Restart();
            }
            Phase("denoise-last-step");
            Console.WriteLine($"  [wan] denoise: {passCount} DiT passes, {passSeconds / Math.Max(1, passCount):F1}s mean" +
                              (cfgCacheHits > 0 ? $" ({cfgCacheHits} unconditional passes served from the guidance cache)" : ""));

            // TI2V I2V: pin the conditioning frame into the final latent.
            if (ti2v && i2v)
                Array.Copy(condTok, 0, x, 0, condTok.Length);

            // ---- 4. free the DiT's device residency, then VAE decode ----
            ReleaseDeviceMemory();
            var latent = new float[(long)zc * lt * lh * lw];
            Unpatchify(x, latent, lt, lh, lw, zc, zTok);
            if (!string.IsNullOrEmpty(dumpDir))
            {
                DumpF32(System.IO.Path.Combine(dumpDir, "latent_final.bin"), latent);
                Console.WriteLine($"  [wan-debug] final latent {Stat(latent)}");
            }
            beat.Set("vae-decode", steps, steps, $"{frames} frames at {width}x{height}");
            RgbImage[] framesOut = Vae.Decode(latent, lt, lh, lw, (done, bands) =>
                beat.Set("vae-decode", steps, steps,
                         $"{frames} frames at {width}x{height}, band {Math.Min(done + 1, bands)}/{bands}"));
            Phase("vae-decode");
            AssertFramesAreNotDegenerate(framesOut);
            beat.Report("done", steps, steps, "complete", 0, heartbeat: false);

            Console.WriteLine($"  [wan-timing] total: {total.Elapsed.TotalSeconds:F1}s");
            return new GeneratedVideo { Frames = framesOut, Fps = fps, Seed = seed };
        }

        // RgbImage [0,1] HWC -> planar [3, t, H, W] in [-1,1]; frame 0 is the image,
        // frames 1..t-1 are zeros (the "gray" -1..1 midpoint the reference pipelines
        // feed the VAE for the unobserved future frames).
        private static float[] ToPlanarSigned(RgbImage img, int t, int H, int W)
        {
            var dst = new float[3L * t * H * W];
            long hwPix = (long)H * W;
            var src = img.Pixels;
            System.Threading.Tasks.Parallel.For(0, H, y =>
            {
                for (int xpx = 0; xpx < W; xpx++)
                {
                    long q = (long)y * W + xpx;
                    long pIdx = q * 3;
                    dst[q] = src[pIdx] * 2f - 1f;
                    dst[(long)t * hwPix + q] = src[pIdx + 1] * 2f - 1f;
                    dst[2L * t * hwPix + q] = src[pIdx + 2] * 2f - 1f;
                }
            });
            return dst;
        }

        private static void DumpF32(string path, float[] data)
        {
            var bytes = new byte[data.LongLength * 4];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            System.IO.File.WriteAllBytes(path, bytes);
        }

        /// <summary>
        /// Fail loudly when the decode produced a uniformly flat (usually black)
        /// video. A backend that miscomputes the VAE's conv GEMMs — the Metal 4
        /// tensor-API mul_mm defect is the known one — yields exactly this, and it
        /// is otherwise silent: the request "succeeds" after however many minutes
        /// the generation took and writes an unusable MP4. Two frames are enough to
        /// catch it and the scan is microseconds next to the decode.
        /// </summary>
        internal static void AssertFramesAreNotDegenerate(RgbImage[] frames)
        {
            if (frames == null || frames.Length == 0) return;
            // Spread the samples: a legitimately flat frame (a fade to or from black)
            // must not condemn the video, while the corruption flattens all of them.
            int n = frames.Length;
            foreach (int idx in new[] { 0, n / 3, 2 * n / 3, n - 1 })
            {
                var px = frames[idx].Pixels;
                float min = float.MaxValue, max = float.MinValue;
                long bad = 0;
                foreach (float v in px)
                {
                    if (float.IsNaN(v) || float.IsInfinity(v)) { bad++; continue; }
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
                if (bad == 0 && max - min > 1e-3f) return;   // a real picture; done
            }
            throw new InvalidOperationException(
                "The Wan VAE decoded a uniformly flat video (every sampled frame is a single " +
                "colour, usually black). That is a backend numerics failure, not a sampling " +
                "problem. On Metal the known cause is the Metal 4 tensor-API mul_mm defect in " +
                "the VAE's conv GEMMs: run with TS_WAN_METAL_TENSOR_API=0 (or check that " +
                "GGML_METAL_TENSOR_DISABLE is not being overridden).");
        }

        /// <summary>Steps whose guidance delta is recomputed rather than reused: the
        /// first <see cref="CfgCacheWarmup"/> steps (where the frame's structure is
        /// decided), the last step, and every <paramref name="stride"/>-th step in
        /// between. A stride of 0/1 means "every step" — the cache is off.</summary>
        internal static bool UsesUncondPass(int step, int steps, int stride)
        {
            if (stride <= 1) return true;
            if (step < CfgCacheWarmup || step == steps - 1) return true;
            return (step - CfgCacheWarmup) % stride == 0;
        }

        /// <summary>Leading steps that always run both passes when the guidance cache is on.</summary>
        internal const int CfgCacheWarmup = 3;

        /// <summary>"1h 04m" / "7m 12s" / "43s"; "?" before the first pass finishes.</summary>
        internal static string FormatEta(double seconds)
        {
            if (seconds < 0) return "ETA unknown";
            if (seconds >= 3600) return $"~{(int)(seconds / 3600)}h {(int)(seconds % 3600 / 60):00}m";
            if (seconds >= 60) return $"~{(int)(seconds / 60)}m {(int)(seconds % 60):00}s";
            return $"~{seconds:F0}s";
        }

        private static string Stat(float[] a)
        {
            double mn = double.MaxValue, mx = double.MinValue, sum = 0, sumsq = 0; long nan = 0;
            foreach (float v in a)
            {
                if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
                if (v < mn) mn = v; if (v > mx) mx = v; sum += v; sumsq += (double)v * v;
            }
            long n = a.LongLength - nan;
            double mean = sum / Math.Max(1, n);
            double std = Math.Sqrt(Math.Max(0, sumsq / Math.Max(1, n) - mean * mean));
            return $"min={mn:E2} max={mx:E2} mean={mean:E3} std={std:E3} nan={nan}";
        }

        // Round spatial dims to the DiT/VAE grid (16 px for Wan 2.1: 8x VAE * 2x patch;
        // 32 px for Wan2.2-TI2V: 16x VAE * 2x patch).
        internal static int SnapDim(int v, int grid = 16) => Math.Max(grid, (v + grid / 2) / grid * grid);
        // Snap the frame count to the VAE's temporal grid (4k + 1).
        internal static int SnapFrames(int f) => Math.Max(1, (f + 2) / 4 * 4 + 1);

        // [channels, t, lh, lw] planar latent -> per-token features written into
        // tokens[tok * tokenStride + featureOffset ...]; token (tt,hh,ww), feature
        // index kw + 2*kh + 4*c (matches the flattened patch_embedding kernel layout).
        internal static void Patchify(float[] latent, float[] tokens, int lt, int lh, int lw,
                                      int channels, int tokenStride, int channelOffset)
        {
            int hLen = lh / 2, wLen = lw / 2;
            long hw = (long)lh * lw;
            for (int tt = 0; tt < lt; tt++)
                for (int hh = 0; hh < hLen; hh++)
                    for (int ww = 0; ww < wLen; ww++)
                    {
                        long tok = ((long)tt * hLen + hh) * wLen + ww;
                        long o = tok * tokenStride + (long)channelOffset * 4;
                        for (int c = 0; c < channels; c++)
                        {
                            long src = ((long)c * lt + tt) * hw;
                            for (int kh = 0; kh < 2; kh++)
                            {
                                long row = src + (long)(hh * 2 + kh) * lw + ww * 2;
                                tokens[o + kh * 2 + c * 4] = latent[row];
                                tokens[o + kh * 2 + c * 4 + 1] = latent[row + 1];
                            }
                        }
                    }
        }

        internal static void Unpatchify(float[] tokens, float[] latent, int lt, int lh, int lw,
                                        int channels, int tokenStride)
        {
            int hLen = lh / 2, wLen = lw / 2;
            long hw = (long)lh * lw;
            for (int tt = 0; tt < lt; tt++)
                for (int hh = 0; hh < hLen; hh++)
                    for (int ww = 0; ww < wLen; ww++)
                    {
                        long tok = ((long)tt * hLen + hh) * wLen + ww;
                        long o = tok * tokenStride;
                        for (int c = 0; c < channels; c++)
                        {
                            long dst = ((long)c * lt + tt) * hw;
                            for (int kh = 0; kh < 2; kh++)
                            {
                                long row = dst + (long)(hh * 2 + kh) * lw + ww * 2;
                                latent[row] = tokens[o + kh * 2 + c * 4];
                                latent[row + 1] = tokens[o + kh * 2 + c * 4 + 1];
                            }
                        }
                    }
        }

        public void Dispose()
        {
            _te?.Dispose();
            _dit?.Dispose();
            _ditLow?.Dispose();
            _vae?.Dispose();
            _dctx?.Dispose();
        }
    }

    /// <summary>
    /// Periodic "still working" ticks for the phases that spend minutes inside one
    /// blocking native call (a DiT pass, a VAE band). Ticks go to the console and,
    /// when the caller supplied one, to <see cref="VideoGenerationParams.OnProgress"/> —
    /// the Web UI turns them into a live status line. TS_WAN_HEARTBEAT_S=0 silences
    /// the ticks (the phase-transition reports still fire).
    /// </summary>
    internal sealed class WanHeartbeat : IDisposable
    {
        private readonly VideoGenerationParams _p;
        private readonly Stopwatch _total;
        private readonly System.Threading.Timer _timer;
        private readonly object _lock = new();
        private string _phase = "starting", _detail = "";
        private int _step, _totalSteps;
        private double _eta = -1;
        private Stopwatch _since = Stopwatch.StartNew();

        public WanHeartbeat(VideoGenerationParams p, Stopwatch total)
        {
            _p = p;
            _total = total;
            int seconds = 30;
            string env = Environment.GetEnvironmentVariable("TS_WAN_HEARTBEAT_S");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v)) seconds = v;
            if (seconds <= 0) return;
            var period = TimeSpan.FromSeconds(seconds);
            _timer = new System.Threading.Timer(_ => Tick(), null, period, period);
        }

        /// <summary>Enter a phase and report it at once; the next ticks describe it. A host's
        /// status line then follows the run instead of trailing it by up to a tick.</summary>
        public void Set(string phase, int step, int totalSteps, string detail)
        {
            double eta;
            lock (_lock)
            {
                _phase = phase; _step = step; _totalSteps = totalSteps; _detail = detail;
                _since = Stopwatch.StartNew();
                eta = _eta;
            }
            Report(phase, step, totalSteps, detail, eta, heartbeat: false);
        }

        public void Report(string phase, int step, int totalSteps, string detail, double eta, bool heartbeat)
        {
            lock (_lock) { _eta = eta; }
            _p?.OnProgress?.Invoke(new VideoGenerationProgress
            {
                Phase = phase, Step = step, TotalSteps = totalSteps, Detail = detail,
                ElapsedSeconds = _total.Elapsed.TotalSeconds, EtaSeconds = eta, Heartbeat = heartbeat,
            });
        }

        private void Tick()
        {
            string phase, detail; int step, totalSteps; double inPhase, eta;
            lock (_lock)
            {
                phase = _phase; detail = _detail; step = _step; totalSteps = _totalSteps;
                inPhase = _since.Elapsed.TotalSeconds; eta = _eta;
            }
            Console.WriteLine($"  [wan] …{phase}" +
                              (totalSteps > 0 && phase == "denoise" ? $" step {step}/{totalSteps}" : "") +
                              $" ({detail}) {inPhase:F0}s in this pass, " +
                              $"{_total.Elapsed.TotalSeconds:F0}s total" +
                              (eta > 0 ? $", {WanVideoPipeline.FormatEta(eta)} left" : ""));
            Report(phase, step, totalSteps, detail, eta, heartbeat: true);
        }

        public void Dispose() => _timer?.Dispose();
    }

    /// <summary>Seeded standard-normal sampler (xoshiro-seeded Box-Muller).</summary>
    internal sealed class GaussianRng
    {
        private readonly Random _rng;
        private double _spare;
        private bool _hasSpare;

        public GaussianRng(long seed) { _rng = new Random(unchecked((int)(seed ^ (seed >> 32)))); }

        public float Next()
        {
            if (_hasSpare) { _hasSpare = false; return (float)_spare; }
            double u, v, s;
            do
            {
                u = _rng.NextDouble() * 2.0 - 1.0;
                v = _rng.NextDouble() * 2.0 - 1.0;
                s = u * u + v * v;
            } while (s >= 1.0 || s == 0.0);
            double m = Math.Sqrt(-2.0 * Math.Log(s) / s);
            _spare = v * m;
            _hasSpare = true;
            return (float)(u * m);
        }
    }
}
