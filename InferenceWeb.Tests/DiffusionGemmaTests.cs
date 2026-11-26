// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Correctness + performance tests for the DiffusionGemma block-diffusion MoE model
// (architecture key "diffusion-gemma") against a real GGUF. Opt-in via TS_TEST_MODEL_DIR.
//
// These tests exercise the full denoising pipeline (DiffusionGemmaModel.ForwardCanvas +
// DiffusionGemmaSampler EntropyBound sampler). They are skipped cleanly when the model file
// is not available so CI without the 16 GB checkpoint stays green.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class DiffusionGemmaTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    // One pattern for BOTH the [ModelFact] skip gate and TryLoad's file pick. They used to be written
    // out separately, and the gate's list omitted the unhyphenated spelling that the released weights
    // actually ship with (diffusiongemma-26B-A4B-it-Q4_K_M.gguf). Every test in this class therefore
    // reported "no matching GGUF" and skipped even with TS_TEST_MODEL_DIR correctly set, so the whole
    // suite was silently green while a real prompt-conditioning regression sat in the model.
    private const string GgufPattern =
        "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private readonly ITestOutputHelper _output;
    public DiffusionGemmaTests(ITestOutputHelper output) { _output = output; }

    private static readonly IPromptRenderer Renderer = new GgufPromptRenderer();

    private BackendType _loadedBackend;

    private DiffusionGemmaModel TryLoad()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return null;
        }
        // Same helper (and same pattern) the [ModelFact] gate uses, so a test can never be reported as
        // runnable and then fail to find its weights, or vice versa.
        string modelPath = TestGates.FindGguf(dir, GgufPattern);
        if (modelPath == null)
        {
            _output.WriteLine("No diffusion-gemma GGUF available; skipping");
            return null;
        }
        BackendType backend = TestBackend();
        _output.WriteLine($"[diffusion-gemma] loading {Path.GetFileName(modelPath)} on {backend}");
        _loadedBackend = backend;
        var model = (DiffusionGemmaModel)ModelBase.Create(modelPath, backend);
        return model;
    }

    /// <summary>The backend these tests load: the GGML backend this process pins
    /// (TS_TEST_GGML_BACKEND). TS_TEST_BACKEND overrides it (e.g. ggmlcuda on a Windows/Linux CUDA box, cpu
    /// for the pure-C# backend), mirroring TS_REPRO_BACKEND elsewhere.</summary>
    internal static BackendType TestBackend()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        string backendEnv = Environment.GetEnvironmentVariable("TS_TEST_BACKEND");
        if (!string.IsNullOrWhiteSpace(backendEnv))
        {
            backend = backendEnv.ToLowerInvariant() switch
            {
                "cpu" => BackendType.Cpu,
                "cuda" => BackendType.Cuda,
                "ggmlcpu" => BackendType.GgmlCpu,
                "ggmlcuda" => BackendType.GgmlCuda,
                "ggmlmetal" => BackendType.GgmlMetal,
                "mlx" => BackendType.Mlx,
                _ => backend,
            };
        }
        return backend;
    }

    private int[] RenderPrompt(DiffusionGemmaModel model, string text)
    {
        var messages = new System.Collections.Generic.List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = text }
        };
        string rendered = Renderer.Render(model.Config.ChatTemplate, messages,
            addGenerationPrompt: true, architecture: model.Config.Architecture);
        return model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void ForwardCanvas_ProducesFiniteLogits_AndArgmaxIsValid()
    {
        using var model = TryLoad();
        if (model == null) return;

        // [prompt | canvas] with a tiny canvas; verify the forward produces finite,
        // correctly-shaped canvas logits.
        int bos = model.Tokenizer.BosTokenId;
        int P = 1;
        int C = 8;
        var tokens = new int[P + C];
        tokens[0] = bos < 0 ? 0 : bos;
        for (int i = 0; i < C; i++) tokens[P + i] = model.MaskTokenId;

        float[] logits = model.ForwardCanvas(tokens, P);
        Assert.Equal((long)C * model.VocabSize, logits.LongLength);

        // every canvas position must have a finite argmax in-range
        for (int c = 0; c < C; c++)
        {
            long baseOff = (long)c * model.VocabSize;
            float max = float.NegativeInfinity;
            int amax = -1;
            for (int v = 0; v < model.VocabSize; v++)
            {
                float z = logits[baseOff + v];
                Assert.False(float.IsNaN(z), $"NaN logit at canvas {c}, vocab {v}");
                if (z > max) { max = z; amax = v; }
            }
            Assert.InRange(amax, 0, model.VocabSize - 1);
        }
        _output.WriteLine("[diffusion-gemma] ForwardCanvas produced finite, valid logits.");
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void CapitalOfFrance_Generates_Paris()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "What is the capital of France? Answer in one short sentence.");
        var sampler = new DiffusionGemmaSampler(model);
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, Seed = 0, MaxBlocks = 1 };

        var sw = Stopwatch.StartNew();
        int steps = 0;
        var generated = sampler.Generate(prompt, p, (blk, step, total, _) => steps++);
        sw.Stop();

        string text = model.Tokenizer.Decode(generated);
        _output.WriteLine($"[diffusion-gemma] prompt_tokens={prompt.Length} steps={steps} " +
            $"time={sw.Elapsed.TotalSeconds:F1}s ms/step={sw.Elapsed.TotalMilliseconds / Math.Max(1, steps):F0}");
        _output.WriteLine($"[diffusion-gemma] output: {text}");

        Assert.True(generated.Count > 0, "no tokens generated");
        Assert.Contains("Paris", text, StringComparison.OrdinalIgnoreCase);
    }

    // Regression guard for the Metal async-compute (lazy-sync) corruption. GgmlContext turns on
    // lazy-sync for Metal: a per-op kernel returns without waiting on its command buffer. Reads are
    // safe (GetFloatPtr drains), but there is no host-WRITE barrier, so a CPU write followed by a
    // device kernel could dispatch against a stale device buffer. Every per-op path in this model
    // mixes the two (MoERoute's host top-K, the per-expert MoE's Buffer.MemoryCopy, the embedding /
    // mask / self-conditioning helpers), which made the forward NON-REPRODUCIBLE: two identical
    // ForwardCanvas calls differed on ~all 67M logits (cosine as low as 0.77) on an M5 Pro.
    //
    // A forward pass is a pure function of its inputs, so the strongest possible assertion is
    // BITWISE equality across two identical calls. This fails loudly on any reintroduction of an
    // unsynchronised host-write/device-read pair, and it needs no golden data.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void ForwardCanvas_IsBitwiseReproducible()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "Describe the video game Final Fantasy VII in detail.");
        int P = prompt.Length, C = model.CanvasLength;
        var tokens = new int[P + C];
        Array.Copy(prompt, tokens, P);
        for (int i = 0; i < C; i++) tokens[P + i] = model.MaskTokenId;

        // ForwardCanvas returns a reusable internal buffer, so each result must be cloned before the
        // next call overwrites it.
        float[] a = (float[])model.ForwardCanvas(tokens, P).Clone();
        float[] b = (float[])model.ForwardCanvas(tokens, P).Clone();

        long differing = 0;
        double maxAbs = 0;
        for (long i = 0; i < a.LongLength; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) differing++;
            double d = Math.Abs((double)a[i] - b[i]);
            if (d > maxAbs) maxAbs = d;
        }
        _output.WriteLine($"[diffusion-gemma][repro] differing={differing}/{a.LongLength} maxAbsDiff={maxAbs:F6}");
        Assert.True(differing == 0,
            $"ForwardCanvas is not reproducible: {differing}/{a.LongLength} logits differ (maxAbsDiff {maxAbs:F4}) " +
            "between two identical calls — an unsynchronised host-write/device-read (Metal async compute)");
    }

    // The same reproducibility guarantee for the prompt-KV path, which is what the server actually
    // runs. PrefillPrompt walks the per-op stack ONCE and freezes the result as the prompt K/V for
    // every denoising step of the turn, so a corrupted prefill permanently poisons the model's view
    // of the prompt — this is why the async-compute race produced fluent but prompt-irrelevant text
    // rather than mere step-to-step noise.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void PrefillAndDecode_AreBitwiseReproducible()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][repro] CPU backend (no PKV); skipping");
            return;
        }

        var prompt = RenderPrompt(model, "请详细介绍最终幻想7");
        int C = model.CanvasLength;

        // The canvas MUST change between steps, and self-conditioning MUST be fed, or this test is
        // blind: with a constant all-mask canvas every step re-uploads identical bytes, so a stale
        // device buffer still holds the right data and a real race passes unnoticed. The sampler
        // re-noises rejected positions every step, which is what makes the per-step
        // Embedding(canvasTokens) a host-write the next device kernel must observe.
        float[] Run()
        {
            model.PrefillPrompt(prompt);
            var canvas = new int[C];
            for (int i = 0; i < C; i++) canvas[i] = model.MaskTokenId;
            float[] logits = null, sc = null;
            for (int step = 0; step < 4; step++)
            {
                logits = model.DecodeCanvas(canvas, sc, step == 0 ? 0f : 1f, 1.25f);
                sc = logits;                       // alias, exactly as DenoiseBlock does
                // deterministic re-noise so both runs see the identical token stream
                uint h = (uint)(step * 2654435761u + 1);
                for (int i = 0; i < C; i++)
                {
                    h ^= h << 13; h ^= h >> 17; h ^= h << 5;
                    if ((h & 3) == 0) canvas[i] = (int)(h % (uint)model.VocabSize);
                }
            }
            return (float[])logits.Clone();
        }

        float[] a = Run();
        float[] b = Run();
        long differing = 0;
        for (long i = 0; i < a.LongLength; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) differing++;

        _output.WriteLine($"[diffusion-gemma][repro] prefill+4 decode steps differing={differing}/{a.LongLength}");
        Assert.True(differing == 0,
            $"prefill+decode is not reproducible: {differing}/{a.LongLength} logits differ between two " +
            "identical runs — the frozen prompt K/V or a per-step canvas upload is being corrupted");
    }

    // End-to-end mirror of the reported failure: a CJK prompt asking for a detailed description of
    // Final Fantasy 7 came back as fluent Chinese that answered as though the user had typed only
    // "请介绍" ("please introduce") — the prompt's actual subject was gone. Exercises the default
    // server configuration (prompt-KV cache + fused decode) and asserts the answer is on-topic.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void Generate_ChinesePrompt_IsOnTopic()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "请详细介绍最终幻想7");
        var sampler = new DiffusionGemmaSampler(model);
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, Seed = 0, MaxBlocks = 1 };

        var generated = sampler.Generate(prompt, p);
        string text = model.Tokenizer.Decode(generated);
        _output.WriteLine($"[diffusion-gemma][cjk] tokens={generated.Count} text: {text}");

        Assert.True(generated.Count >= 24, $"answer collapsed too early ({generated.Count} tokens)");
        // The subject must actually appear: the model names the game in Chinese, in Latin, or by
        // its abbreviation. The pre-fix output contained none of these.
        bool onTopic =
            text.Contains("最终幻想") || text.Contains("太空戰士") ||
            text.Contains("Final Fantasy", StringComparison.OrdinalIgnoreCase) ||
            text.Contains("FF7", StringComparison.OrdinalIgnoreCase) ||
            text.Contains("FFVII", StringComparison.OrdinalIgnoreCase);
        Assert.True(onTopic,
            "answer never mentions Final Fantasy VII — the prompt's conditional content was lost: " + text);
        // The pre-fix answer complained about a "repeated instruction" instead of answering.
        Assert.False(text.Contains("重复的指令") || text.Contains("重複的指令"),
            "model reported a repeated/garbled instruction — the prompt reached it corrupted: " + text);
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void Benchmark_StepThroughput()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "Explain in two sentences why the sky is blue.");
        var sampler = new DiffusionGemmaSampler(model);
        var p = FixedStepParams(6);

        var sw = Stopwatch.StartNew();
        int steps = 0;
        var generated = sampler.Generate(prompt, p, (blk, step, total, _) => steps++);
        sw.Stop();

        double msPerStep = sw.Elapsed.TotalMilliseconds / Math.Max(1, steps);
        _output.WriteLine($"[diffusion-gemma][bench] canvas={model.CanvasLength} steps={steps} " +
            $"total={sw.Elapsed.TotalSeconds:F2}s ms/step={msPerStep:F0} pkv={model.SupportsPromptKvCache}");
        model.PrintForwardTiming();

        Assert.Equal(6, steps);
        Assert.True(msPerStep > 0);
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void PromptKvCache_LogitsMatchUnified()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][pkv] CPU backend (no PKV); skipping logits-equivalence test");
            return;
        }

        // Prompt-KV caching must be numerically equivalent to the unified [prompt|canvas] forward (the
        // prompt's K/V don't depend on the canvas). Run one forward of each over the SAME [prompt|canvas]
        // (a fixed mask-token canvas, no self-conditioning) and compare the canvas logits directly.
        var prompt = RenderPrompt(model, "List three primary colors.");
        int P = prompt.Length;
        int C = model.CanvasLength;
        int vocab = model.VocabSize;
        var canvas = new int[C];
        for (int i = 0; i < C; i++) canvas[i] = model.MaskTokenId;
        var full = new int[P + C];
        Array.Copy(prompt, full, P);
        Array.Copy(canvas, 0, full, P, C);

        model.SupportsPromptKvCache = false;
        float[] uLogits = (float[])model.ForwardCanvas(full, P).Clone();   // clone: buffer is reused
        model.SupportsPromptKvCache = true;
        model.PrefillPrompt(prompt);
        float[] pLogits = model.DecodeCanvas(canvas, null, 0f, 1f);

        long n = (long)C * vocab;
        double dot = 0, nu = 0, np = 0, maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double a = uLogits[i], b = pLogits[i];
            dot += a * b; nu += a * a; np += b * b;
            double d = Math.Abs(a - b); if (d > maxAbs) maxAbs = d;
        }
        double cosine = dot / (Math.Sqrt(nu) * Math.Sqrt(np) + 1e-12);
        _output.WriteLine($"[diffusion-gemma][pkv] logits cosine={cosine:F6} maxAbsDiff={maxAbs:F4} (P={P}, C={C})");

        // PKV is mathematically equivalent to the unified forward (the prompt K/V are canvas-independent).
        // The residual difference is FP op-ordering (batched [P|C] vs split prefill/decode) amplified by
        // the MoE's discrete top-8 expert selection (a ~1e-6 router-score nudge can flip one expert),
        // which is inherent to any MoE; the logit vectors stay highly aligned (>0.99 cosine).
        Assert.True(cosine >= 0.99, $"PKV logits diverged from unified (cosine {cosine:F6}) — likely a bug");
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void Benchmark_PromptKvCache_SpeedupOnLongPrompt()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][pkv] backend has no device glue (CPU); PKV not applicable, skipping");
            return;
        }

        // Long prompt (system-style context + question) so the prompt dominates the [prompt|canvas]
        // sequence — the regime where prompt-KV caching pays off (it turns the unified O(N^2) attention
        // into O(C*N) and removes the prompt's per-step projection/dense/MoE work).
        string ctx = string.Concat(Enumerable.Repeat(
            "You are a meticulous senior engineer who explains concepts precisely and weighs trade-offs " +
            "while always considering performance, correctness, and maintainability. ", 50));
        var prompt = RenderPrompt(model, ctx + "Given the above, answer concisely: what is the capital of France?");
        var sampler = new DiffusionGemmaSampler(model);
        const int steps = 5;

        // Warm up once (kernel/codegen) so the comparison isn't skewed by first-call costs.
        sampler.Generate(prompt, FixedStepParams(2), null);

        double Run(bool pkv)
        {
            model.SupportsPromptKvCache = pkv;
            int n = 0;
            var sw = Stopwatch.StartNew();
            sampler.Generate(prompt, FixedStepParams(steps), (b, s, t, _) => n++);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / Math.Max(1, n);
            _output.WriteLine($"[diffusion-gemma][pkv] pkv={pkv} promptTokens={prompt.Length} steps={n} " +
                $"total={sw.Elapsed.TotalSeconds:F2}s ms/step={ms:F0}");
            return sw.Elapsed.TotalMilliseconds;
        }

        double off = Run(false);
        double on = Run(true);
        model.SupportsPromptKvCache = true;   // restore default
        _output.WriteLine($"[diffusion-gemma][pkv] speedup = {off / on:F2}x (off={off:F0}ms on={on:F0}ms)");

        // On a long prompt PKV should be clearly faster than the unified path.
        Assert.True(on < off, $"PKV ({on:F0}ms) not faster than unified ({off:F0}ms) on a long prompt");
    }

    private static DiffusionEbParams FixedStepParams(int steps) => new DiffusionEbParams
    {
        MaxDenoisingSteps = steps,
        Seed = 0,
        MaxBlocks = 1,
        ConfidenceThreshold = -1f,        // never satisfy the confidence stop
        StabilityThreshold = int.MaxValue, // never satisfy the stability stop
    };

    private static int MaxConsecutiveRepeat(System.Collections.Generic.IReadOnlyList<int> ids)
    {
        int best = ids.Count > 0 ? 1 : 0, run = 1;
        for (int i = 1; i < ids.Count; i++)
        {
            run = ids[i] == ids[i - 1] ? run + 1 : 1;
            if (run > best) best = run;
        }
        return best;
    }

    // Regression guard for the Metal fused-lm_head async-download race: on a long-answer prompt the canvas
    // used to decode the first few tokens correctly then collapse into a repetition tail (e.g. "**，**，**，")
    // because the host softcap/readback raced the in-flight device->host logits blit. A healthy answer is
    // substantial and token-diverse. Exercises the default (fused decode + fused lm_head + PKV) path on the
    // GPU backend. Skipped without TS_TEST_MODEL_DIR.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void Generate_LongAnswer_IsCoherent_NotDegenerate()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "Describe the video game Final Fantasy VII in two or three sentences.");
        var sampler = new DiffusionGemmaSampler(model);
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, Seed = 0, MaxBlocks = 1 };

        var generated = sampler.Generate(prompt, p);
        string text = model.Tokenizer.Decode(generated);
        int maxRun = MaxConsecutiveRepeat(generated);
        double distinctRatio = generated.Count > 0 ? generated.Distinct().Count() / (double)generated.Count : 0;

        // The race corrupted the *tail* of the canvas (the prefix denoised before the stale read mattered),
        // so the strongest signal is a single token dominating the trailing region. Measure the most
        // frequent token over the last K positions — a healthy answer keeps this well under half.
        int k = Math.Min(24, generated.Count);
        double tailTopFreq = k == 0 ? 1.0
            : generated.Skip(generated.Count - k).GroupBy(t => t).Max(g => g.Count()) / (double)k;
        _output.WriteLine($"[diffusion-gemma][regression] tokens={generated.Count} maxConsecRepeat={maxRun} " +
            $"distinctRatio={distinctRatio:F2} tailTopFreq={tailTopFreq:F2}");
        _output.WriteLine($"[diffusion-gemma][regression] text: {text}");

        Assert.True(generated.Count >= 24, $"answer collapsed too early ({generated.Count} tokens) — likely the garbage tail");
        Assert.True(maxRun <= 5, $"degenerate repetition: a token repeats {maxRun}x consecutively (garbage tail)");
        Assert.True(distinctRatio >= 0.5, $"low token diversity {distinctRatio:F2} — the answer degenerated into repetition");
        Assert.True(tailTopFreq <= 0.4, $"one token is {tailTopFreq:P0} of the answer tail — the canvas tail degenerated into repetition");
    }

    // Regression guard for the Metal OOM-on-second-prompt bug. The fused/per-layer diffusion decode
    // binds the prompt K/V as *cacheable* device-local copies keyed by their host pointer
    // (try_get_cacheable_tensor_buffer, USAGE_COMPUTE). Block-autoregressive generation reallocates the
    // prompt K/V on every block (AllocPromptStore), so without an explicit cache invalidation on dispose,
    // each block orphaned numLayers*2 device buffers in g_host_buffer_cache — a per-block GPU memory leak
    // that exhausted the Metal command-buffer budget after a couple of turns
    // (kIOGPUCommandBufferCallbackErrorOutOfMemory). The fix (ReleasePromptKvTensor) frees the cached
    // device copy before disposing each K/V tensor. This test reallocates the prompt K/V many times and
    // asserts the resident device-copy bytes stay bounded (≈ one prefill's K/V), not growing per prefill.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void PromptKvCache_DeviceCopiesDoNotLeakAcrossPrefills()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "List three primary colors and briefly explain additive color mixing.");
        int C = model.CanvasLength;
        var canvas = new int[C];
        for (int i = 0; i < C; i++) canvas[i] = model.MaskTokenId;

        // One prefill + decode to create and cache this prompt's K/V device copies (the decode bind is
        // what populates g_host_buffer_cache with the DeviceCopy entries).
        void PrefillAndDecodeOnce()
        {
            model.PrefillPrompt(prompt);
            _ = model.DecodeCanvas(canvas, null, 0f, 1f);
        }

        PrefillAndDecodeOnce();
        long baseline = GgmlBasicOps.DeviceCopyCacheResidentBytes();
        _output.WriteLine($"[diffusion-gemma][leak] resident device-copy after 1 prefill = {baseline / (1024.0 * 1024.0):F1} MB");

        // A leaking build grows resident bytes ~linearly with the prefill count; the fixed build frees the
        // previous prefill's device copies in AllocPromptStore so resident stays ≈ one prefill's worth.
        const int reps = 16;
        for (int i = 0; i < reps; i++) PrefillAndDecodeOnce();
        long after = GgmlBasicOps.DeviceCopyCacheResidentBytes();
        _output.WriteLine($"[diffusion-gemma][leak] resident device-copy after {reps + 1} prefills = {after / (1024.0 * 1024.0):F1} MB " +
            $"(growth = {(after - baseline) / (1024.0 * 1024.0):F1} MB; a leak would be ~{reps}x baseline)");

        // Allow generous slack for allocator rounding, but a real leak (reps extra K/V sets) is many ×.
        Assert.True(after <= baseline * 3 / 2,
            $"prompt K/V device copies leaked across prefills: {after / (1024.0 * 1024.0):F1} MB resident after {reps + 1} prefills " +
            $"vs {baseline / (1024.0 * 1024.0):F1} MB after 1 (expected ≈ constant)");
    }

    // End-to-end mirror of the reported failure: generate repeatedly on ONE model instance (as a chat
    // server does turn after turn). Pre-fix this OOM'd on the Metal backend on the 2nd/3rd turn because
    // every block of every turn leaked its prompt K/V device copies. Asserts generation keeps succeeding
    // and resident device memory stays bounded across turns. Uses a fixed prompt so the per-turn resident
    // footprint is constant (isolates the leak from the natural growth of an accumulating chat history).
    [ModelFact(EnvModelDir, GgufPattern)]
    public void MultiTurn_Generation_DoesNotLeakDeviceMemory()
    {
        using var model = TryLoad();
        if (model == null) return;
        Assert.True(model.SupportsPromptKvCache, $"prompt-KV cache is off on {_loadedBackend}");

        var prompt = RenderPrompt(model, "Describe the video game Final Fantasy VII in two or three sentences.");
        var sampler = new DiffusionGemmaSampler(model);
        // Force several prefills per turn (MaxBlocks > 1) so each turn exercises the per-block K/V realloc.
        var p = new DiffusionEbParams { MaxDenoisingSteps = 6, Seed = 0, MaxBlocks = 3, ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue };

        long firstTurnResident = 0;
        long lastTurnResident = 0;
        const int turns = 4;
        for (int turn = 0; turn < turns; turn++)
        {
            var generated = sampler.Generate(prompt, p);   // must not throw (pre-fix: OOM on a later turn)
            Assert.True(generated.Count > 0, $"turn {turn} produced no tokens");
            long resident = GgmlBasicOps.DeviceCopyCacheResidentBytes();
            if (turn == 0) firstTurnResident = resident;
            lastTurnResident = resident;
            _output.WriteLine($"[diffusion-gemma][leak] turn {turn}: tokens={generated.Count} " +
                $"resident device-copy = {resident / (1024.0 * 1024.0):F1} MB");
        }

        // Constant prompt ⇒ identical per-turn footprint after the fix; a leak grows it every turn.
        Assert.True(lastTurnResident <= firstTurnResident * 3 / 2,
            $"device memory grew across turns ({firstTurnResident / (1024.0 * 1024.0):F1} MB → " +
            $"{lastTurnResident / (1024.0 * 1024.0):F1} MB over {turns} turns) — the per-block K/V leak regressed");
    }

    // ---- Batched (parallel-request) decode -------------------------------------------------

    // Correctness gate for the batched throughput path: a sequence's canvas logits must be identical
    // whether it is decoded alone or batched with another sequence (each canvas row's attention/FFN/lm_head
    // depends only on that row + its own prompt K/V, so batching must not perturb it). Uses fixed mask-token
    // canvases with self-conditioning off so the forward is deterministic. Skipped without a GPU/PKV backend.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void BatchedDecode_LogitsMatchSolo()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][batched] CPU backend (no PKV); skipping batched-decode equivalence test");
            return;
        }

        var promptA = RenderPrompt(model, "List three primary colors.");
        var promptB = RenderPrompt(model, "Explain in one sentence why the sky is blue.");
        int C = model.CanvasLength;
        int vocab = model.VocabSize;
        var canvasA = new int[C];
        var canvasB = new int[C];
        for (int i = 0; i < C; i++) { canvasA[i] = model.MaskTokenId; canvasB[i] = model.MaskTokenId; }

        bool prevSc = model.SelfConditioningEnabled;
        model.SelfConditioningEnabled = false;   // deterministic forward (no self-conditioning signal)
        DiffusionSeqState seqA = null, seqB = null;
        try
        {
            seqA = model.CreateSeqState();
            seqB = model.CreateSeqState();
            model.PrefillSeq(seqA, promptA);
            model.PrefillSeq(seqB, promptB);

            // seqA decoded ALONE (batch of one, per-op path) vs seqA decoded BATCHED with seqB.
            float[] solo = (float[])model.DecodeCanvasBatched(
                new[] { seqA }, new[] { canvasA }, new float[1][], new[] { 0f }, new[] { 1f })[0].Clone();
            float[][] batched = model.DecodeCanvasBatched(
                new[] { seqA, seqB }, new[] { canvasA, canvasB }, new float[2][], new[] { 0f, 0f }, new[] { 1f, 1f });
            float[] batchedA = batched[0];

            long n = (long)C * vocab;
            double dot = 0, na = 0, nb = 0, maxAbs = 0;
            for (long i = 0; i < n; i++)
            {
                double a = solo[i], b = batchedA[i];
                dot += a * b; na += a * a; nb += b * b;
                double d = Math.Abs(a - b); if (d > maxAbs) maxAbs = d;
            }
            double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            _output.WriteLine($"[diffusion-gemma][batched] seqA solo-vs-batched logits cosine={cosine:F6} maxAbsDiff={maxAbs:F5} (C={C})");

            // Each sequence's rows read only its own cache, so seqA's logits are its solo logits up to the products' arithmetic.
            // The pure-C# cpu backend's products are row-independent: batched is solo bit for bit (measured on the 26B: every
            // arrangement, max |d| 0). The GGML backends pick other kernels for 512 rows than for 256 (ggml-cuda's MMQ tiles and
            // stream-k, ggml-cpu's expert groups and F32 router GEMM), and the mask-token canvas's 128-expert routing amplifies
            // those last bits into other top-8 picks: cosine 0.9964-0.9966 on ggml_cpu whether seqA is batched with an identical
            // copy of itself or with seqB, 0.9757 on an A40's ggml_cuda, where the legacy model measured 0.9722. The bar there is
            // that spread, not a divergence of the arithmetic.
            if (_loadedBackend == BackendType.Cpu)
            {
                Assert.Equal(0d, maxAbs);
                return;
            }
            Assert.True(cosine >= 0.95, $"batched seqA logits diverged from solo (cosine {cosine:F6}) — batching corrupted the forward");
        }
        finally
        {
            model.DisposeSeqState(seqA);
            model.DisposeSeqState(seqB);
            model.SelfConditioningEnabled = prevSc;
        }
    }

    // End-to-end mirror of the reported bug: two prompts generated together in ONE batched run must BOTH
    // produce a correct, non-empty answer (pre-fix, a second parallel request produced nothing). Drives the
    // batched sampler the way the server scheduler does — one block at a time over the active set.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void BatchedGeneration_TwoPrompts_BothProduceOutput()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][batched] CPU backend; skipping parallel-generation test");
            return;
        }

        var sampler = new DiffusionGemmaSampler(model);
        var promptA = RenderPrompt(model, "What is the capital of France? Answer in one short sentence.");
        var promptB = RenderPrompt(model, "Name the largest planet in our solar system. Answer in one short sentence.");
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, Seed = 0, MaxBlocks = 1 };

        var (textA, textB) = GenerateTwoBatched(model, sampler, promptA, promptB, p);
        _output.WriteLine($"[diffusion-gemma][batched] A (France): {textA}");
        _output.WriteLine($"[diffusion-gemma][batched] B (planet): {textB}");

        Assert.False(string.IsNullOrWhiteSpace(textA), "batched prompt A produced no output");
        Assert.False(string.IsNullOrWhiteSpace(textB), "batched prompt B produced no output (the reported parallel-request bug)");
        Assert.Contains("Paris", textA, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Jupiter", textB, StringComparison.OrdinalIgnoreCase);
    }

    // Regression guard for the CPU-backend server crash: the batched scheduler path (RunBlockBatched)
    // used to call PrefillSeq unconditionally, which throws "Prompt-KV caching is not enabled for this
    // backend" on the non-PKV (cpu / ggml_cpu) backends — so ANY web-UI / API chat request crashed.
    // The fix routes non-PKV sequences through the unified [prefix|canvas] ForwardCanvas, the same
    // fallback DenoiseBlock uses. Since both paths share the RNG, step temperatures, DenoiseStep and
    // TrimCanvas, a single request driven through RunBlockBatched must produce TOKEN-IDENTICAL output
    // to the single-request Generate() path — on every backend. Runs a few fixed steps so it is cheap
    // enough for the CPU backends.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void BatchedScheduler_SingleRequest_MatchesGenerate_AllBackends()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "What is the capital of France? Answer in one short sentence.");
        var sampler = new DiffusionGemmaSampler(model);
        var p = FixedStepParams(3);

        var solo = sampler.Generate(prompt, p);

        var run = new DiffusionSeqRun(prompt, p, model.CreateSeqState(), CancellationToken.None, null);
        try
        {
            int guard = 0;
            while (!run.Done && guard++ < 16)
                sampler.RunBlockBatched(new List<DiffusionSeqRun> { run });
        }
        finally
        {
            model.DisposeSeqState(run.State);
        }

        _output.WriteLine($"[diffusion-gemma][batched] pkv={model.SupportsPromptKvCache} " +
            $"solo={solo.Count} tokens, batched={run.Response.Count} tokens");
        Assert.Equal(solo, run.Response);
    }

    /// <summary>Drive two prompts through the batched sampler to completion (block-synchronous, as the
    /// server's DiffusionBatchScheduler does) and return each decoded answer.</summary>
    private (string, string) GenerateTwoBatched(DiffusionGemmaModel model, DiffusionGemmaSampler sampler,
        int[] a, int[] b, DiffusionEbParams p)
    {
        var runs = new List<DiffusionSeqRun>
        {
            new DiffusionSeqRun(a, p, model.CreateSeqState(), CancellationToken.None, null),
            new DiffusionSeqRun(b, p, model.CreateSeqState(), CancellationToken.None, null),
        };
        try
        {
            var pending = new List<DiffusionSeqRun>(runs);
            int guard = 0;
            while (pending.Count > 0 && guard++ < 4096)
            {
                sampler.RunBlockBatched(pending);
                pending.RemoveAll(r => r.Done);
            }
            return (model.Tokenizer.Decode(runs[0].Response), model.Tokenizer.Decode(runs[1].Response));
        }
        finally
        {
            foreach (var r in runs) model.DisposeSeqState(r.State);
        }
    }

    // Throughput of two concurrent requests' canvases: each request's own step in turn (the scheduler's default,
    // on-device sampled where the backend samples on the device) against both canvases in one step
    // (DecodeCanvasBatched, host logits). Reported, not ordered: which wins depends on how far one canvas fills the device.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void Benchmark_BatchedDecodeThroughput()
    {
        using var model = TryLoad();
        if (model == null) return;

        var prompt = RenderPrompt(model, "Explain in two sentences why the sky is blue.");
        int C = model.CanvasLength;
        var canvas = new int[C];
        for (int i = 0; i < C; i++) canvas[i] = model.MaskTokenId;

        bool prevSc = model.SelfConditioningEnabled;
        model.SelfConditioningEnabled = false;
        DiffusionSeqState seqA = null, seqB = null;
        double sliceMs, batchedMs;
        try
        {
            seqA = model.CreateSeqState();
            seqB = model.CreateSeqState();
            model.PrefillSeq(seqA, prompt);
            model.PrefillSeq(seqB, prompt);
            const int steps = 6;
            bool device = model.SupportsDeviceSampling;
            int[] am = new int[C], sm = new int[C];
            float[] en = new float[C], u = Enumerable.Repeat(0.5f, C).ToArray();
            void Step(DiffusionSeqState seq)
            {
                if (device) Assert.True(model.DecodeCanvasSampledSeq(seq, canvas, null, null, 0f, 1.25f, u, 0, am, en, sm, null, null));
                else model.DecodeCanvasSeq(seq, canvas, null, 0f, 1f);
            }

            Step(seqA);   // warmup
            Step(seqB);
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < steps; s++) { Step(seqA); Step(seqB); }
            sliceMs = sw.Elapsed.TotalMilliseconds / steps;

            model.DecodeCanvasBatched(new[] { seqA, seqB }, new[] { canvas, canvas }, new float[2][], new[] { 0f, 0f }, new[] { 1f, 1f });   // warmup
            sw.Restart();
            for (int s = 0; s < steps; s++)
                model.DecodeCanvasBatched(new[] { seqA, seqB }, new[] { canvas, canvas }, new float[2][], new[] { 0f, 0f }, new[] { 1f, 1f });
            batchedMs = sw.Elapsed.TotalMilliseconds / steps;
            _output.WriteLine($"[diffusion-gemma][bench] 2 requests, each its own step{(device ? " (on-device sampled)" : "")}: {sliceMs:F0} ms a round " +
                $"-> {2 * C / (sliceMs / 1000.0):F0} canvas-tok/s");
            _output.WriteLine($"[diffusion-gemma][bench] 2 requests in one batched step (host logits): {batchedMs:F0} ms " +
                $"-> {2 * C / (batchedMs / 1000.0):F0} canvas-tok/s");
        }
        finally
        {
            model.DisposeSeqState(seqA);
            model.DisposeSeqState(seqB);
            model.SelfConditioningEnabled = prevSc;
        }
        Assert.True(sliceMs > 0 && batchedMs > 0);
    }
}
