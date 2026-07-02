// Qwen-Image-Edit benchmark + correctness harness.
//
// Modes:
//   vae-verify <ref_dir>   Load the diffusers oracle's exact input (input_rgb.npy), run the
//                          managed reference VAE, and diff latent.npy / decoded_rgb.npy.
//   vae-roundtrip <img> <out.png> [size]
//                          VAE encode->decode an image (no reference), report PSNR + timings.
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0) { Usage(); return 1; }
        switch (args[0])
        {
            case "vae-verify": return VaeVerify(args);
            case "vae-roundtrip": return VaeRoundtrip(args);
            case "te-verify": return TeVerify(args);
            case "vis-verify": return VisVerify(args);
            case "te-img-verify": return TeImgVerify(args);
            case "tok-check": return TokCheck(args);
            case "dit-mlp-verify": return DitMlpVerify(args);
            case "dit-block-verify": return DitBlockVerify(args);
            case "dit-cfg-verify": return DitCfgVerify(args);
            case "dit-forward": return DitForward(args);
            case "edit": return Edit(args);
            case "edit2": return Edit2(args);
            case "conv-tile-test": return ConvTileTest(args);
            default: Usage(); return 1;
        }
    }

    private static void Usage()
    {
        Console.WriteLine("QwenImageEditBench <mode> ...");
        Console.WriteLine("  vae-verify <ref_dir>");
        Console.WriteLine("  vae-roundtrip <img> <out.png> [size]");
        Console.WriteLine("  te-verify <ref_dir>");
        Console.WriteLine("  edit <dit.gguf> <in.png> \"<prompt>\" <out.png> [steps] [cfg] [area]");
    }

    // Verify the band-tiled VAE conv (TryGpuConv2dMaybeTiled) == the un-tiled conv.
    private static int ConvTileTest(string[] args)
    {
        var backend = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cpu") switch
        {
            "ggml_cuda" => TensorSharp.GGML.GgmlBackendType.Cuda,
            "ggml_metal" => TensorSharp.GGML.GgmlBackendType.Metal,
            _ => TensorSharp.GGML.GgmlBackendType.Cpu,
        };
        TensorSharp.GGML.GgmlBasicOps.EnsureBackendAvailable(backend);
        double worst = VaeReferenceMath.ConvTileSelfTest();
        bool ok = worst < 1e-4;
        Console.WriteLine($"CONV-TILE-TEST: worst relL2={worst:E3} -> {(ok ? "PASS" : "FAIL")}");
        return ok ? 0 : 2;
    }

    private static int Edit(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q4_0.gguf";
        string inPath = args.Length > 2 ? args[2] : "C:/Works/test.jpg";
        string prompt = args.Length > 3 ? args[3] : "Change the background to a sunny beach.";
        string outPath = args.Length > 4 ? args[4] : "C:/Works/TensorSharp/tools/_edit_out.png";
        int steps = args.Length > 5 ? int.Parse(args[5]) : 8;
        float cfg = args.Length > 6 ? float.Parse(args[6]) : 4.0f;
        long area = args.Length > 7 ? long.Parse(args[7]) : 256L * 256;
        int w = args.Length > 8 ? int.Parse(args[8]) : 0;   // explicit width (bypasses VRAM area clamp)
        int h = args.Length > 9 ? int.Parse(args[9]) : 0;   // explicit height

        Environment.SetEnvironmentVariable("MAX_CONTEXT", "4096");
        var backend = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cpu") switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.Cuda,
            _ => BackendType.GgmlCpu,
        };
        Console.WriteLine($"[bench] backend={backend}");
        var model = (TensorSharp.Models.QwenImage.QwenImageModel)
            TensorSharp.Models.ModelBase.Create(dit, backend);
        try
        {
            var input = ImageIO.Load(inPath);
            var p = new QwenImageParams { Steps = steps, CfgScale = cfg, TargetArea = area, Seed = 42, Width = w, Height = h };
            var sw = Stopwatch.StartNew();
            var outImg = model.EditImage(prompt, input, p);
            double ms = sw.Elapsed.TotalMilliseconds;
            ImageIO.SavePng(outPath, outImg);
            Console.WriteLine($"edit {outImg.Width}x{outImg.Height} steps={steps} cfg={cfg} -> {outPath}  ({ms / 1000:F1}s, {ms / steps:F0}ms/step)");
            return 0;
        }
        finally { model.Dispose(); }
    }

    // Isolated DiT-forward perf bench: drive QwenImageDiT.Predict directly at a target
    // token count (no VAE / text-encoder), so a denoise forward can be A/B'd in seconds.
    //   dit-forward <dit.gguf> <hp> <wp> <txtSeq> <iters> [cfgPairs]
    // imgSeq = 2*hp*wp (generated + reference). 576x432 -> hp=36 wp=27 -> imgSeq=1944.
    private static int DitForward(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q2_K.gguf";
        int hp = args.Length > 2 ? int.Parse(args[2]) : 36;
        int wp = args.Length > 3 ? int.Parse(args[3]) : 27;
        int txtSeq = args.Length > 4 ? int.Parse(args[4]) : 284;
        int iters = args.Length > 5 ? int.Parse(args[5]) : 6;
        bool cfgPairs = args.Length > 6 && args[6] == "cfg";
        int negTxt = txtSeq - 5 > 0 ? txtSeq - 5 : txtSeq;     // different len, like a real neg prompt

        Environment.SetEnvironmentVariable("MAX_CONTEXT", "4096");
        var backend = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cuda") switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.Cuda,
            _ => BackendType.GgmlCpu,
        };
        int seq = hp * wp, imgSeq = 2 * seq;
        Console.WriteLine($"[dit-forward] backend={backend} hp={hp} wp={wp} imgSeq={imgSeq} txtSeq={txtSeq} iters={iters} cfgPairs={cfgPairs}");

        var enc = new TensorSharp.Models.QwenImage.QwenImageDiT(dit, backend);
        try
        {
            var rng = new Random(5);
            float[] img = Rand(rng, imgSeq * 64, 0.5f);
            float[] cond = Rand(rng, txtSeq * 3584, 0.5f);
            float[] negCond = Rand(rng, negTxt * 3584, 0.5f);
            var modIndex = new int[imgSeq];
            for (int i = seq; i < imgSeq; i++) modIndex[i] = 1;
            var shapes = new (int f, int h, int w)[] { (1, hp, wp), (1, hp, wp) };
            var rope = TensorSharp.Models.QwenImage.DitRope.Build(shapes, txtSeq);
            var ropeNeg = TensorSharp.Models.QwenImage.DitRope.Build(shapes, negTxt);

            Func<float[]> one = () =>
            {
                if (cfgPairs && enc.UseCfgBatch && (imgSeq + Math.Max(txtSeq, negTxt)) <= enc.CfgBatchMaxTokens)
                {
                    var (vc, vn) = enc.PredictCfg(img, imgSeq, cond, txtSeq, rope, negCond, negTxt, ropeNeg, 0.5f, modIndex, -1, 0, seq);
                    return vc;
                }
                else
                {
                    float[] v = enc.Predict(img, imgSeq, cond, txtSeq, 0.5f, modIndex, rope);
                    if (cfgPairs) enc.Predict(img, imgSeq, negCond, negTxt, 0.5f, modIndex, ropeNeg);
                    return v;
                }
            };

            // Correctness alongside perf: inputs are identical every call, so every velocity
            // must be finite and IDENTICAL across calls — replay corruption (the historical
            // flash-under-capture NaN appeared from the first captured replay on) shows up as
            // NaN or iteration-to-iteration drift here.
            int nanTotal = 0;
            double driftMax = 0;
            float[] first = null;
            Func<float[], string> check = v =>
            {
                int nan = 0;
                for (int i = 0; i < v.Length; i++) if (float.IsNaN(v[i]) || float.IsInfinity(v[i])) nan++;
                nanTotal += nan;
                double d = 0;
                if (first == null) first = (float[])v.Clone();
                else for (int i = 0; i < v.Length; i++) d = Math.Max(d, Math.Abs(v[i] - first[i]));
                driftMax = Math.Max(driftMax, d);
                return $"nan={nan} maxdiff-vs-first={d:E2}";
            };

            // warmup (cold graph build + capture)
            var wsw = Stopwatch.StartNew(); var v0 = one(); Console.WriteLine($"  warmup1: {wsw.Elapsed.TotalMilliseconds:F0}ms  {check(v0)}");
            wsw.Restart(); var v1 = one(); Console.WriteLine($"  warmup2: {wsw.Elapsed.TotalMilliseconds:F0}ms  {check(v1)}");

            var times = new double[iters];
            for (int i = 0; i < iters; i++)
            {
                var sw = Stopwatch.StartNew();
                float[] v = one();
                times[i] = sw.Elapsed.TotalMilliseconds;
                Console.WriteLine($"  iter{i}: {times[i]:F0}ms  {check(v)}");
            }
            Array.Sort(times);
            double median = times[iters / 2], min = times[0];
            double sum = 0; foreach (var t in times) sum += t;
            Console.WriteLine($"  per-forward{(cfgPairs ? "-pair" : "")}: median={median:F0}ms min={min:F0}ms mean={sum / iters:F0}ms  (all: {string.Join(",", Array.ConvertAll(times, t => t.ToString("F0")))})");

            // Cross-process reference compare: TS_QIBENCH_DUMP=<path> writes the first velocity;
            // TS_QIBENCH_REF=<path> compares against a previously dumped one (e.g. captured-flash
            // vs TS_QWEN_DIT_WHOLE_CAPTURE=0 / TS_QWEN_DIT_FLASH=0 references).
            string dump = Environment.GetEnvironmentVariable("TS_QIBENCH_DUMP");
            string refp = Environment.GetEnvironmentVariable("TS_QIBENCH_REF");
            if (dump != null)
            {
                var bytes = new byte[first.Length * 4];
                Buffer.BlockCopy(first, 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(dump, bytes);
                Console.WriteLine($"  dumped velocity -> {dump}");
            }
            if (refp != null && File.Exists(refp))
            {
                var bytes = File.ReadAllBytes(refp);
                var rf = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, rf, 0, bytes.Length);
                Console.WriteLine($"  vs-ref: cosine={Cosine(first, rf):F6} relL2={RelL2(first, rf):E3}");
            }

            bool pass = nanTotal == 0 && driftMax == 0;
            Console.WriteLine(pass ? "DIT-FORWARD: PASS (finite, replay-stable)"
                                   : $"DIT-FORWARD: FAIL (nan={nanTotal} driftMax={driftMax:E2})");
            return pass ? 0 : 2;
        }
        finally { enc.Dispose(); }
    }

    // Run TWO edits on the SAME model instance (pipeline persists) to exercise the server's
    // cross-request path: per-edit FreeEncoders -> global cache clear -> capture-ring reset ->
    // next edit rebinds the DiT weights + rebuilds/recaptures. Verifies edit #2 doesn't crash
    // or corrupt (NaN/black) after the ring was reset.
    private static int Edit2(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q2_K.gguf";
        string inPath = args.Length > 2 ? args[2] : "C:/Works/test.jpg";
        string prompt = args.Length > 3 ? args[3] : "Change the background to a sunny beach.";
        string outBase = args.Length > 4 ? args[4] : "C:/Works/TensorSharp/tools/_edit2";
        int steps = args.Length > 5 ? int.Parse(args[5]) : 4;
        float cfg = args.Length > 6 ? float.Parse(args[6]) : 4.0f;
        long area = args.Length > 7 ? long.Parse(args[7]) : 256L * 256;

        Environment.SetEnvironmentVariable("MAX_CONTEXT", "4096");
        var backend = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cuda") switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            _ => BackendType.GgmlCpu,
        };
        var model = (TensorSharp.Models.QwenImage.QwenImageModel)TensorSharp.Models.ModelBase.Create(dit, backend);
        try
        {
            var input = ImageIO.Load(inPath);
            int rc = 0;
            for (int e = 1; e <= 2; e++)
            {
                var p = new QwenImageParams { Steps = steps, CfgScale = cfg, TargetArea = area, Seed = 42 };
                var sw = Stopwatch.StartNew();
                var outImg = model.EditImage(prompt, input, p);
                double ms = sw.Elapsed.TotalMilliseconds;
                string outPath = $"{outBase}_{e}.png";
                ImageIO.SavePng(outPath, outImg);
                // sanity: non-degenerate image (real content, not black/NaN)
                double sum = 0, sq = 0; int n = outImg.Pixels.Length;
                bool finite = true;
                for (int i = 0; i < n; i++) { float v = outImg.Pixels[i]; if (float.IsNaN(v) || float.IsInfinity(v)) finite = false; sum += v; sq += (double)v * v; }
                double mean = sum / n, std = Math.Sqrt(sq / n - mean * mean);
                bool ok = finite && std > 0.02;
                Console.WriteLine($"edit#{e} {outImg.Width}x{outImg.Height} {ms / 1000:F1}s mean={mean:F3} std={std:F3} finite={finite} -> {(ok ? "OK" : "DEGENERATE")} ({outPath})");
                if (!ok) rc = 2;
            }
            Console.WriteLine(rc == 0 ? "EDIT2: PASS (cross-edit reset+rebuild works)" : "EDIT2: FAIL");
            return rc;
        }
        finally { model.Dispose(); }
    }

    private static int TeVerify(string[] args)
    {
        string refDir = args.Length > 1 ? args[1] : "C:/Works/TensorSharp/tools/_te_ref";
        string tePath = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE")
            ?? "C:/Works/models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf";
        const int DropIdx = 64;

        long[] tok64 = Npy.LoadInt64(Path.Combine(refDir, "tokens.npy"), out _);
        int[] tokens = Array.ConvertAll(tok64, x => (int)x);
        var refCond = Npy.Load(Path.Combine(refDir, "hidden_cond.npy"), out var condShape); // [seq-drop, hidden]
        int hidden = condShape[1];

        var enc = new TensorSharp.Models.QwenImage.QwenImageTextEncoder(tePath, BackendType.GgmlCpu);
        try
        {
            var sw = Stopwatch.StartNew();
            float[] full = enc.EncodeHidden(tokens);   // [seq*hidden]
            double ms = sw.Elapsed.TotalMilliseconds;
            int seq = tokens.Length;
            // drop first DropIdx rows
            int condRows = seq - DropIdx;
            var cond = new float[(long)condRows * hidden];
            Array.Copy(full, (long)DropIdx * hidden, cond, 0, cond.Length);

            double err = RelL2(cond, refCond);
            double cos = Cosine(cond, refCond);
            Console.WriteLine($"text-encoder seq={seq} hidden={hidden} cond rows={condRows} (vs ref [{string.Join(',', condShape)}])");
            Console.WriteLine($"  hidden relL2={err:E3}  cosine={cos:F6}  time={ms:F0}ms");
            // Managed encoder uses the Q4_K GGUF vs the fp16 oracle, so exact match is not
            // expected; high cosine is the forward-logic correctness signal (a logic bug
            // collapses cosine, quantization-only noise stays ~0.99+).
            bool ok = cos > 0.99 && err < 0.15;
            Console.WriteLine(ok ? "TE-VERIFY: PASS" : "TE-VERIFY: FAIL (above tolerance)");
            return ok ? 0 : 2;
        }
        finally { enc.Dispose(); }
    }

    private static int VisVerify(string[] args)
    {
        string refDir = args.Length > 1 ? args[1] : "C:/Works/TensorSharp/tools/_vis_ref";
        string mmproj = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ")
            ?? "C:/Works/models/qwen-image-te-mmproj-BF16.gguf";
        var pv = Npy.Load(Path.Combine(refDir, "pixel_values.npy"), out var pvShape);   // [seq,1176]
        long[] grid = Npy.LoadInt64(Path.Combine(refDir, "grid_thw.npy"), out _);
        int gridH = (int)grid[1], gridW = (int)grid[2];
        var refMerged = Npy.Load(Path.Combine(refDir, "merged.npy"), out var mShape);     // [M,3584]

        var enc = new TensorSharp.Models.QwenImage.QwenImageVisionEncoder(mmproj, BackendType.GgmlCpu);
        try
        {
            var sw = Stopwatch.StartNew();
            float[] merged = enc.Encode(pv, gridH, gridW);
            double ms = sw.Elapsed.TotalMilliseconds;
            double err = RelL2(merged, refMerged);
            double cos = Cosine(merged, refMerged);
            Console.WriteLine($"vision grid {gridH}x{gridW} merged rows={merged.Length / TensorSharp.Models.QwenImage.QwenImageVisionEncoder.OutDim} (ref [{string.Join(',', mShape)}])");
            Console.WriteLine($"  merged relL2={err:E3}  cosine={cos:F6}  time={ms:F0}ms");
            bool ok = cos > 0.99 && err < 0.15;
            Console.WriteLine(ok ? "VIS-VERIFY: PASS" : "VIS-VERIFY: FAIL");
            return ok ? 0 : 2;
        }
        finally { enc.Dispose(); }
    }

    private static int TeImgVerify(string[] args)
    {
        string refDir = args.Length > 1 ? args[1] : "C:/Works/TensorSharp/tools/_cond_ref";
        string tePath = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE")
            ?? "C:/Works/models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf";
        string mmproj = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ")
            ?? "C:/Works/models/qwen-image-te-mmproj-BF16.gguf";
        const int Drop = 64;

        long[] tok64 = Npy.LoadInt64(Path.Combine(refDir, "tokens.npy"), out _);
        int[] tokens = Array.ConvertAll(tok64, x => (int)x);
        var pv = Npy.Load(Path.Combine(refDir, "pixel_values.npy"), out _);
        long[] grid = Npy.LoadInt64(Path.Combine(refDir, "grid_thw.npy"), out _);
        int gH = (int)grid[1], gW = (int)grid[2];
        long[] istart = Npy.LoadInt64(Path.Combine(refDir, "img_start.npy"), out _);
        int imgStart = (int)istart[0];
        var refCond = Npy.Load(Path.Combine(refDir, "cond.npy"), out var condShape);
        int hidden = condShape[1];

        var vis = new TensorSharp.Models.QwenImage.QwenImageVisionEncoder(mmproj, BackendType.GgmlCpu);
        var te = new TensorSharp.Models.QwenImage.QwenImageTextEncoder(tePath, BackendType.GgmlCpu);
        try
        {
            int n = (gH / 2) * (gW / 2);
            // To isolate the LLM+M-RoPE from vision-embed fp error, use the oracle's exact embeds
            // when TS_USE_REF_EMBEDS=1; otherwise run our own (verified) vision encoder.
            float[] embeds;
            string refEmbedsPath = Path.Combine(refDir, "merged_embeds.npy");
            if (Environment.GetEnvironmentVariable("TS_USE_REF_EMBEDS") == "1" && File.Exists(refEmbedsPath))
                embeds = Npy.Load(refEmbedsPath, out _);
            else
                embeds = vis.Encode(pv, gH, gW);
            var img = new TensorSharp.Models.QwenImage.ImageCond { Start = imgStart, Count = n, GridH = gH, GridW = gW, Embeds = embeds };
            var sw = Stopwatch.StartNew();
            float[] full = te.EncodeHidden(tokens, img);
            double ms = sw.Elapsed.TotalMilliseconds;
            int seq = tokens.Length, condRows = seq - Drop;
            var cond = new float[(long)condRows * hidden];
            Array.Copy(full, (long)Drop * hidden, cond, 0, cond.Length);
            double err = RelL2(cond, refCond), cos = Cosine(cond, refCond);
            Console.WriteLine($"image-grounded cond seq={seq} imgStart={imgStart} n_img={n} rows={condRows} (ref [{string.Join(',', condShape)}])");
            Console.WriteLine($"  cond relL2={err:E3}  cosine={cos:F6}  time={ms:F0}ms");
            // split: image-region rows vs text-region rows (cond starts at token Drop=64)
            int imgRow0 = imgStart - Drop, imgRowN = imgRow0 + n;
            var imgC = Slice(cond, imgRow0, imgRowN, hidden); var imgR = Slice(refCond, imgRow0, imgRowN, hidden);
            var txtC = SliceComplement(cond, imgRow0, imgRowN, condRows, hidden); var txtR = SliceComplement(refCond, imgRow0, imgRowN, condRows, hidden);
            Console.WriteLine($"  image-region cosine={Cosine(imgC, imgR):F6} relL2={RelL2(imgC, imgR):E3}");
            Console.WriteLine($"  text-region  cosine={Cosine(txtC, txtR):F6} relL2={RelL2(txtC, txtR):E3}");
            bool ok = cos > 0.99 && err < 0.15;
            Console.WriteLine(ok ? "TE-IMG-VERIFY: PASS" : "TE-IMG-VERIFY: FAIL");
            return ok ? 0 : 2;
        }
        finally { vis.Dispose(); te.Dispose(); }
    }

    private static int DitBlockVerify(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q4_0.gguf";
        int dim = 3072, layer = 0;
        var shapes = new (int f, int h, int w)[] { (1, 4, 4), (1, 4, 4) };
        int imgSeq = 16 + 16, txtSeq = 16;
        var rope = TensorSharp.Models.QwenImage.DitRope.Build(shapes, txtSeq);
        var backend = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cpu") switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.Cuda,
            _ => BackendType.GgmlCpu,
        };
        Console.WriteLine($"[dit-block-verify] backend={backend}");
        float amp = float.TryParse(Environment.GetEnvironmentVariable("TS_QIBENCH_AMP"), out var a) ? a : 1f;
        var enc = new TensorSharp.Models.QwenImage.QwenImageDiT(dit, backend);
        try
        {
            var rng = new Random(3);
            float[] img = Rand(rng, imgSeq * dim, amp), txt = Rand(rng, txtSeq * dim, amp);
            float[] imgMod = Rand(rng, 2 * 18432, 0.1f), txtMod = Rand(rng, 2 * 18432, 0.1f);
            var modIndex = new int[imgSeq];
            for (int i = imgSeq / 2; i < imgSeq; i++) modIndex[i] = 1;

            // ----- attention sub-layer -----
            float[] imgM = (float[])img.Clone(), txtM = (float[])txt.Clone();
            float[] imgN = (float[])img.Clone(), txtN = (float[])txt.Clone();
            enc.ManagedAttnSubLayer(imgM, imgSeq, txtM, txtSeq, imgMod, txtMod, modIndex, rope, layer);
            bool okA = enc.NativeAttnSubLayer(imgN, imgSeq, txtN, txtSeq, imgMod, txtMod, modIndex, rope, layer);
            Console.WriteLine($"attn native ok={okA}");
            Console.WriteLine($"  attn img cosine={Cosine(imgN, imgM):F6} relL2={RelL2(imgN, imgM):E3}");
            Console.WriteLine($"  attn txt cosine={Cosine(txtN, txtM):F6} relL2={RelL2(txtN, txtM):E3}");

            // ----- MLP sub-layer (img stream) -----
            float[] xM = (float[])img.Clone(), xN = (float[])img.Clone();
            enc.ManagedMlpSubLayer(xM, imgSeq, imgMod, modIndex, $"transformer_blocks.{layer}.img_mlp");
            bool okM = enc.NativeMlpSubLayer(xN, imgSeq, imgMod, modIndex, $"transformer_blocks.{layer}.img_mlp");
            Console.WriteLine($"mlp native ok={okM}");
            Console.WriteLine($"  mlp img cosine={Cosine(xN, xM):F6} relL2={RelL2(xN, xM):E3}");

            // ----- WHOLE fused block (attn + both MLP streams in one native graph) -----
            // managed reference = chain the verified attn + 2 mlp sub-layers.
            float[] biM = (float[])img.Clone(), btM = (float[])txt.Clone();
            enc.ManagedAttnSubLayer(biM, imgSeq, btM, txtSeq, imgMod, txtMod, modIndex, rope, layer);
            enc.ManagedMlpSubLayer(biM, imgSeq, imgMod, modIndex, $"transformer_blocks.{layer}.img_mlp");
            enc.ManagedMlpSubLayer(btM, txtSeq, txtMod, null, $"transformer_blocks.{layer}.txt_mlp");
            float[] biN = (float[])img.Clone(), btN = (float[])txt.Clone();
            bool okB = enc.NativeBlock(biN, imgSeq, btN, txtSeq, imgMod, txtMod, modIndex, rope, layer);
            Console.WriteLine($"fused block native ok={okB}");
            Console.WriteLine($"  block img cosine={Cosine(biN, biM):F6} relL2={RelL2(biN, biM):E3}");
            Console.WriteLine($"  block txt cosine={Cosine(btN, btM):F6} relL2={RelL2(btN, btM):E3}");

            bool pass = okA && okM && okB
                && Cosine(imgN, imgM) > 0.999 && Cosine(txtN, txtM) > 0.999 && Cosine(xN, xM) > 0.999
                && Cosine(biN, biM) > 0.999 && Cosine(btN, btM) > 0.999;
            Console.WriteLine(pass ? "DIT-BLOCK-VERIFY: PASS" : "DIT-BLOCK-VERIFY: FAIL");
            return pass ? 0 : 2;
        }
        finally { enc.Dispose(); }
    }

    // Verify the CFG-batched block kernel (TSGgml_QwenImageBlockCfg) produces the same
    // outputs as running the two single-branch NativeBlock kernels separately. The branches
    // use DIFFERENT txt lengths (as in true-CFG: prompt vs negative prompt).
    private static int DitCfgVerify(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q4_0.gguf";
        int dim = 3072, layer = 0;
        var shapes = new (int f, int h, int w)[] { (1, 4, 4), (1, 4, 4) };
        int imgSeq = 16 + 16, txtSeqC = 20, txtSeqN = 16;     // different lengths per branch
        var ropeC = TensorSharp.Models.QwenImage.DitRope.Build(shapes, txtSeqC);
        var ropeN = TensorSharp.Models.QwenImage.DitRope.Build(shapes, txtSeqN);
        var enc = new TensorSharp.Models.QwenImage.QwenImageDiT(dit, BackendType.GgmlCpu);
        try
        {
            var rng = new Random(7);
            float[] img = Rand(rng, imgSeq * dim, 1f);            // identical img_in for both branches
            float[] txtC = Rand(rng, txtSeqC * dim, 1f), txtN = Rand(rng, txtSeqN * dim, 1f);
            float[] imgMod = Rand(rng, 2 * 18432, 0.1f), txtMod = Rand(rng, 2 * 18432, 0.1f);
            var modIndex = new int[imgSeq];
            for (int i = imgSeq / 2; i < imgSeq; i++) modIndex[i] = 1;

            // single-branch reference (the verified NativeBlock), one per branch
            float[] iCs = (float[])img.Clone(), tCs = (float[])txtC.Clone();
            enc.NativeBlock(iCs, imgSeq, tCs, txtSeqC, imgMod, txtMod, modIndex, ropeC, layer);
            float[] iNs = (float[])img.Clone(), tNs = (float[])txtN.Clone();
            enc.NativeBlock(iNs, imgSeq, tNs, txtSeqN, imgMod, txtMod, modIndex, ropeN, layer);

            // combined CFG-batched kernel
            float[] iCc = (float[])img.Clone(), tCc = (float[])txtC.Clone();
            float[] iNc = (float[])img.Clone(), tNc = (float[])txtN.Clone();
            bool ok = enc.NativeBlockCfg(iCc, imgSeq, tCc, txtSeqC, ropeC, iNc, tNc, txtSeqN, ropeN, imgMod, txtMod, modIndex, layer);
            Console.WriteLine($"cfg-batch native ok={ok}");
            double cic = Cosine(iCc, iCs), ctc = Cosine(tCc, tCs), cin = Cosine(iNc, iNs), ctn = Cosine(tNc, tNs);
            Console.WriteLine($"  cond img cosine={cic:F6} relL2={RelL2(iCc, iCs):E3}");
            Console.WriteLine($"  cond txt cosine={ctc:F6} relL2={RelL2(tCc, tCs):E3}");
            Console.WriteLine($"  neg  img cosine={cin:F6} relL2={RelL2(iNc, iNs):E3}");
            Console.WriteLine($"  neg  txt cosine={ctn:F6} relL2={RelL2(tNc, tNs):E3}");
            bool pass = ok && cic > 0.9999 && ctc > 0.9999 && cin > 0.9999 && ctn > 0.9999;
            Console.WriteLine(pass ? "DIT-CFG-VERIFY: PASS" : "DIT-CFG-VERIFY: FAIL");
            return pass ? 0 : 2;
        }
        finally { enc.Dispose(); }
    }

    private static float[] Rand(Random r, int n, float amp)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((r.NextDouble() * 2 - 1) * amp);
        return a;
    }

    private static unsafe int DitMlpVerify(string[] args)
    {
        string dit = args.Length > 1 ? args[1] : "C:/Works/models/qwen-image-edit-2511-Q4_0.gguf";
        int dim = 3072, ff = 12288, seq = 16;
        TensorSharp.GGML.GgmlBasicOps.EnsureBackendAvailable(TensorSharp.GGML.GgmlBackendType.Cpu);
        using var gguf = new TensorSharp.Runtime.GgufFile(dit);
        string b = "transformer_blocks.0.img_mlp";

        (IntPtr ptr, int type, long ne0, long ne1, long bytes) W(string name)
        {
            var info = gguf.Tensors[name];
            gguf.TryGetTensorDataPointer(info, out IntPtr p);
            return (p, (int)info.Type, (long)info.Shape[0], info.Shape.Length > 1 ? (long)info.Shape[1] : 1, gguf.GetTensorByteCount(info));
        }
        float[] Deq(string name)
        {
            var info = gguf.Tensors[name]; long n = info.NumElements; var d = new float[n];
            byte[] raw = gguf.ReadTensorData(info);
            TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32((int)info.Type, raw, 0, d, 0, n);
            return d;
        }

        var n0 = W($"{b}.net.0.proj.weight"); var n0b = Deq($"{b}.net.0.proj.bias");
        var n2 = W($"{b}.net.2.weight"); var n2b = Deq($"{b}.net.2.bias");
        float[] n0f = Deq($"{b}.net.0.proj.weight"), n2f = Deq($"{b}.net.2.weight");

        var rng = new Random(1);
        float[] x = new float[dim * seq], sc = new float[dim * seq], sh = new float[dim * seq], gt = new float[dim * seq];
        for (int i = 0; i < x.Length; i++) { x[i] = (float)(rng.NextDouble() * 2 - 1); sc[i] = (float)(rng.NextDouble() * 0.2 - 0.1); sh[i] = (float)(rng.NextDouble() * 0.2 - 0.1); gt[i] = (float)(rng.NextDouble() * 0.2 - 0.1); }
        float[] scale1 = new float[dim * seq];
        for (int i = 0; i < scale1.Length; i++) scale1[i] = 1f + sc[i];

        // managed reference
        float[] refOut = new float[dim * seq];
        for (int s = 0; s < seq; s++)
        {
            double mean = 0; for (int c = 0; c < dim; c++) mean += x[s * dim + c]; mean /= dim;
            double var = 0; for (int c = 0; c < dim; c++) { double v = x[s * dim + c] - mean; var += v * v; } var /= dim;
            float inv = (float)(1.0 / Math.Sqrt(var + 1e-6));
            var mod = new float[dim];
            for (int c = 0; c < dim; c++) mod[c] = (float)((x[s * dim + c] - mean) * inv) * scale1[s * dim + c] + sh[s * dim + c];
            var h = new float[ff];
            System.Threading.Tasks.Parallel.For(0, ff, oc => { double a = n0b[oc]; long wb = (long)oc * dim; for (int c = 0; c < dim; c++) a += mod[c] * n0f[wb + c]; double hv = a; h[oc] = (float)(0.5 * hv * (1 + Math.Tanh(0.7978845608 * (hv + 0.044715 * hv * hv * hv)))); });
            System.Threading.Tasks.Parallel.For(0, dim, c => { double a = n2b[c]; long wb = (long)c * ff; for (int oc = 0; oc < ff; oc++) a += h[oc] * n2f[wb + oc]; refOut[s * dim + c] = x[s * dim + c] + gt[s * dim + c] * (float)a; });
        }

        // native kernel
        bool ok;
        fixed (float* xp = x, scp = scale1, shp = sh, gtp = gt, n0bp = n0b, n2bp = n2b)
        {
            var dsc = new TensorSharp.GGML.QwenImageModMlpArgs
            {
                X = (IntPtr)xp, ScalePlus1 = (IntPtr)scp, Shift = (IntPtr)shp, Gate = (IntPtr)gtp,
                Net0W = n0.ptr, Net0Type = n0.type, Net0Ne0 = n0.ne0, Net0Ne1 = n0.ne1, Net0Bytes = n0.bytes, Net0B = (IntPtr)n0bp,
                Net2W = n2.ptr, Net2Type = n2.type, Net2Ne0 = n2.ne0, Net2Ne1 = n2.ne1, Net2Bytes = n2.bytes, Net2B = (IntPtr)n2bp,
                StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<TensorSharp.GGML.QwenImageModMlpArgs>(),
                Dim = dim, Ff = ff, Seq = seq, Eps = 1e-6f,
            };
            var sw = Stopwatch.StartNew();
            ok = TensorSharp.GGML.GgmlBasicOps.TryQwenImageModMlp(in dsc);
            Console.WriteLine($"native kernel ok={ok} ({sw.Elapsed.TotalMilliseconds:F1}ms)  (x overwritten in place)");
        }
        if (!ok) { Console.WriteLine("DIT-MLP-VERIFY: FAIL (native returned false)"); return 2; }
        double err = RelL2(x, refOut), cos = Cosine(x, refOut);
        Console.WriteLine($"  native vs managed-ref: relL2={err:E3} cosine={cos:F6}");
        bool pass = cos > 0.999 && err < 5e-2;
        Console.WriteLine(pass ? "DIT-MLP-VERIFY: PASS" : "DIT-MLP-VERIFY: FAIL");
        return pass ? 0 : 2;
    }

    private static int TokCheck(string[] args)
    {
        string tePath = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE")
            ?? "C:/Works/models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf";
        string prompt = args.Length > 1 ? args[1] : "add a pair of sunglasses to the person";
        var te = new TensorSharp.Models.QwenImage.QwenImageTextEncoder(tePath, BackendType.GgmlCpu);
        try
        {
            // text-only template
            string t0 = TensorSharp.Models.QwenImage.QwenImagePrompt.Build(prompt);
            int[] tok0 = te.Tokenizer.Encode(t0, addSpecial: false).ToArray();
            // image template (base, single image_pad)
            string t1 = TensorSharp.Models.QwenImage.QwenImagePrompt.BuildWithImage(prompt);
            int[] tok1 = te.Tokenizer.Encode(t1, addSpecial: false).ToArray();
            int padIdx = Array.IndexOf(tok1, TensorSharp.Models.QwenImage.QwenImagePrompt.ImagePadTokenId);
            Console.WriteLine($"text-only tokens={tok0.Length}  (HF gave 77 for the beach prompt)");
            Console.WriteLine($"image-template base tokens={tok1.Length}  image_pad index={padIdx}  (HF: base=85, idx=70, prefix=64)");
            Console.WriteLine($"  first 8: [{string.Join(',', tok1[..Math.Min(8, tok1.Length)])}]");
            Console.WriteLine($"  around pad (idx {padIdx}): [{string.Join(',', tok1[Math.Max(0, padIdx - 2)..Math.Min(tok1.Length, padIdx + 3)])}]");
            return 0;
        }
        finally { te.Dispose(); }
    }

    private static float[] Slice(float[] a, int r0, int r1, int dim)
    {
        var o = new float[(long)(r1 - r0) * dim];
        Array.Copy(a, (long)r0 * dim, o, 0, o.Length); return o;
    }
    private static float[] SliceComplement(float[] a, int r0, int r1, int rows, int dim)
    {
        var o = new float[(long)(rows - (r1 - r0)) * dim]; long k = 0;
        for (int r = 0; r < rows; r++) if (r < r0 || r >= r1) { Array.Copy(a, (long)r * dim, o, k, dim); k += dim; }
        return o;
    }

    private static double Cosine(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length); double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    private static string VaePath() =>
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_VAE") ?? "C:/Works/models/qwen_image_vae.gguf";

    private static (VaeWeights w, IDisposable g) OpenVae()
    {
        // Accept both the converted F32 GGUF and the original .safetensors (BF16->F32
        // upcast on read) — the same sources QwenImageVae itself resolves.
        string path = VaePath();
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            var st = new TensorSharp.Runtime.SafetensorsFile(path);
            return (VaeWeights.Load(st), st);
        }
        var g = new GgufFile(path);
        return (VaeWeights.Load(g), g);
    }

    private static int VaeVerify(string[] args)
    {
        string refDir = args.Length > 1 ? args[1] : "C:/Works/TensorSharp/tools/_vae_ref";
        var inHwc = Npy.Load(Path.Combine(refDir, "input_rgb.npy"), out var inShape); // H,W,3
        int H = inShape[0], W = inShape[1];
        var img = new RgbImage(W, H, inHwc);

        // The pipeline's QwenImageVae ctor inits the GGML backend; vae-verify uses the
        // static math directly, so init CPU ggml here for the device-conv path.
        if (TensorSharp.Models.QwenImage.VaeReferenceMath.UseGpuConv)
            TensorSharp.GGML.GgmlBasicOps.EnsureBackendAvailable(TensorSharp.GGML.GgmlBackendType.Cpu);

        var (w, g) = OpenVae();
        try
        {
            var sw = Stopwatch.StartNew();
            var latent = VaeReferenceMath.Encode(w, img);
            double encMs = sw.Elapsed.TotalMilliseconds; sw.Restart();
            var dec = VaeReferenceMath.Decode(w, latent);
            double decMs = sw.Elapsed.TotalMilliseconds;

            var refLat = Npy.Load(Path.Combine(refDir, "latent.npy"), out var latShape);   // C,H,W
            var refDec = Npy.Load(Path.Combine(refDir, "decoded_rgb.npy"), out var decShape); // H,W,3

            // managed latent is planar CHW already
            double latErr = RelL2(latent.Data, refLat);
            double latMax = MaxAbs(latent.Data, refLat);
            // managed decoded -> HWC for compare
            double decErr = RelL2(dec.Pixels, refDec);
            double decPsnr = Psnr(dec.Pixels, refDec);

            Console.WriteLine($"latent  shape C{latent.Channels} {latent.Height}x{latent.Width} vs ref [{string.Join(',', latShape)}]");
            Console.WriteLine($"  latent  relL2={latErr:E3}  maxAbs={latMax:E3}");
            Console.WriteLine($"  decoded relL2={decErr:E3}  PSNR(vs ref)={decPsnr:F2} dB");
            Console.WriteLine($"  timings encode={encMs:F0}ms decode={decMs:F0}ms");

            bool ok = latErr < 2e-2 && decErr < 3e-2;
            Console.WriteLine(ok ? "VAE-VERIFY: PASS" : "VAE-VERIFY: FAIL (above tolerance)");
            return ok ? 0 : 2;
        }
        finally { g.Dispose(); }
    }

    private static int VaeRoundtrip(string[] args)
    {
        string imgPath = args.Length > 1 ? args[1] : "C:/Works/test.jpg";
        string outPath = args.Length > 2 ? args[2] : "C:/Works/TensorSharp/tools/_vae_rt.png";
        int size = args.Length > 3 ? int.Parse(args[3]) : 128;
        var img = ImageIO.ResizeToArea(ImageIO.Load(imgPath), (long)size * size);

        // vae-roundtrip drives the static math directly (no QwenImageVae ctor to init the
        // backend), so honor TS_QWEN_BACKEND for device-conv/fused-graph perf runs.
        if (TensorSharp.Models.QwenImage.VaeReferenceMath.UseGpuConv)
        {
            var be = (Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cpu") switch
            {
                "ggml_cuda" => TensorSharp.GGML.GgmlBackendType.Cuda,
                "ggml_metal" => TensorSharp.GGML.GgmlBackendType.Metal,
                _ => TensorSharp.GGML.GgmlBackendType.Cpu,
            };
            TensorSharp.GGML.GgmlBasicOps.EnsureBackendAvailable(be);
        }

        var (w, g) = OpenVae();
        try
        {
            var sw = Stopwatch.StartNew();
            var latent = VaeReferenceMath.Encode(w, img);
            double encMs = sw.Elapsed.TotalMilliseconds; sw.Restart();
            var dec = VaeReferenceMath.Decode(w, latent);
            double decMs = sw.Elapsed.TotalMilliseconds;
            ImageIO.SavePng(outPath, dec);
            double psnr = Psnr(dec.Pixels, img.Pixels);
            Console.WriteLine($"roundtrip {img.Width}x{img.Height} latent C{latent.Channels} {latent.Height}x{latent.Width}");
            Console.WriteLine($"  self-PSNR={psnr:F2} dB  encode={encMs:F0}ms decode={decMs:F0}ms  -> {outPath}");
            return 0;
        }
        finally { g.Dispose(); }
    }

    // ---- metrics ----
    private static double RelL2(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length); double num = 0, den = 0;
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-12));
    }
    private static double MaxAbs(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length); double m = 0;
        for (int i = 0; i < n; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
    private static double Psnr(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length); double mse = 0;
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; mse += d * d; }
        mse /= n;
        return mse <= 1e-12 ? 99 : 10 * Math.Log10(1.0 / mse);
    }
}

// Minimal .npy (float32, C order) reader.
internal static class Npy
{
    public static float[] Load(string path, out int[] shape)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var magic = br.ReadBytes(6);
        if (magic[0] != 0x93) throw new InvalidDataException("not a npy file");
        br.ReadByte(); br.ReadByte(); // version
        int hlen = br.ReadUInt16();
        string header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(hlen));
        // parse shape tuple
        int si = header.IndexOf("'shape':", StringComparison.Ordinal);
        int lp = header.IndexOf('(', si), rp = header.IndexOf(')', lp);
        var dims = header.Substring(lp + 1, rp - lp - 1).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        shape = Array.ConvertAll(dims, s => int.Parse(s));
        bool f4 = header.Contains("<f4");
        bool f8 = header.Contains("<f8");
        long count = 1; foreach (var d in shape) count *= d;
        var data = new float[count];
        if (f4) { for (long i = 0; i < count; i++) data[i] = br.ReadSingle(); }
        else if (f8) { for (long i = 0; i < count; i++) data[i] = (float)br.ReadDouble(); }
        else throw new InvalidDataException("unsupported npy dtype (need <f4/<f8): " + header);
        return data;
    }

    public static long[] LoadInt64(string path, out int[] shape)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var magic = br.ReadBytes(6);
        if (magic[0] != 0x93) throw new InvalidDataException("not a npy file");
        br.ReadByte(); br.ReadByte();
        int hlen = br.ReadUInt16();
        string header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(hlen));
        int si = header.IndexOf("'shape':", StringComparison.Ordinal);
        int lp = header.IndexOf('(', si), rp = header.IndexOf(')', lp);
        var dims = header.Substring(lp + 1, rp - lp - 1).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        shape = Array.ConvertAll(dims, s => int.Parse(s));
        long count = 1; foreach (var d in shape) count *= d;
        var data = new long[count];
        bool i8 = header.Contains("<i8");
        bool i4 = header.Contains("<i4");
        if (i8) { for (long i = 0; i < count; i++) data[i] = br.ReadInt64(); }
        else if (i4) { for (long i = 0; i < count; i++) data[i] = br.ReadInt32(); }
        else throw new InvalidDataException("unsupported npy int dtype: " + header);
        return data;
    }
}
