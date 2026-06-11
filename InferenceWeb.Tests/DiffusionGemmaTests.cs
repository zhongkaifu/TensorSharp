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

    private readonly ITestOutputHelper _output;
    public DiffusionGemmaTests(ITestOutputHelper output) { _output = output; }

    private static readonly IPromptRenderer Renderer = new GgufPromptRenderer();

    private DiffusionGemmaModel TryLoad()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return null;
        }
        string modelPath = Directory.GetFiles(dir, "*.gguf").FirstOrDefault(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("diffusion") && n.Contains("gemma") && !n.Contains("mmproj");
        });
        if (modelPath == null)
        {
            _output.WriteLine("No diffusion-gemma GGUF available; skipping");
            return null;
        }
        // Exercise the GPU path on macOS (ggml_metal), CPU elsewhere.
        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
        _output.WriteLine($"[diffusion-gemma] loading {Path.GetFileName(modelPath)} on {backend}");
        var model = (DiffusionGemmaModel)ModelBase.Create(modelPath, backend);
        return model;
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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
    [Fact]
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
    [Fact]
    public void PromptKvCache_DeviceCopiesDoNotLeakAcrossPrefills()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][leak] CPU backend (no device K/V copies); skipping");
            return;
        }

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

        Assert.True(baseline > 0, "expected the prompt K/V to be cached as device copies on the GPU backend");
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
    [Fact]
    public void MultiTurn_Generation_DoesNotLeakDeviceMemory()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][leak] CPU backend; skipping multi-turn device-memory test");
            return;
        }

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

        Assert.True(firstTurnResident > 0, "expected device-copy K/V to be resident after the first turn");
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
    [Fact]
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

            // Batching is row-independent, so seqA's logits should match to within MoE FP op-ordering noise.
            Assert.True(cosine >= 0.9999, $"batched seqA logits diverged from solo (cosine {cosine:F6}) — batching corrupted the forward");
            Assert.True(maxAbs <= 0.5, $"batched seqA logits diverged from solo (maxAbsDiff {maxAbs:F4})");
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
    [Fact]
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

    // Throughput benchmark comparing the two ways to serve concurrent requests on ONE GPU:
    //   (1) FUSED time-slice (the scheduler default): the fast fused single-canvas kernel run once per
    //       request each step. Aggregate canvas-tok/s = C / fused_ms_per_step.
    //   (2) PER-OP true-batched forward (DIFFUSION_BATCHED_FORWARD): all canvases in one per-op forward.
    // This 128-expert MoE is compute-bound by a single 256-token canvas, so true batching does NOT multiply
    // throughput; worse, the per-op batched forward is markedly slower per canvas than the fused kernel, so
    // the fused time-slice wins. This benchmark proves that ordering (so the scheduler default is correct)
    // and reports the numbers. Skipped without a GPU/PKV backend.
    [Fact]
    public void Benchmark_BatchedDecodeThroughput()
    {
        using var model = TryLoad();
        if (model == null) return;
        if (!model.SupportsPromptKvCache)
        {
            _output.WriteLine("[diffusion-gemma][bench] CPU backend; skipping throughput benchmark");
            return;
        }

        var prompt = RenderPrompt(model, "Explain in two sentences why the sky is blue.");
        int C = model.CanvasLength;
        var canvas = new int[C];
        for (int i = 0; i < C; i++) canvas[i] = model.MaskTokenId;

        bool prevSc = model.SelfConditioningEnabled;
        model.SelfConditioningEnabled = false;
        DiffusionSeqState seqA = null, seqB = null;
        double fusedMs, batchedMs;
        try
        {
            seqA = model.CreateSeqState();
            seqB = model.CreateSeqState();
            model.PrefillSeq(seqA, prompt);
            model.PrefillSeq(seqB, prompt);

            const int steps = 6;

            // (1) fused single-canvas decode — the per-request fast path the scheduler time-slices.
            model.DecodeCanvasSeq(seqA, canvas, null, 0f, 1f);   // warmup
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < steps; s++) model.DecodeCanvasSeq(seqA, canvas, null, 0f, 1f);
            sw.Stop();
            fusedMs = sw.Elapsed.TotalMilliseconds / steps;

            // (2) per-op true-batched forward over 2 canvases.
            model.DecodeCanvasBatched(new[] { seqA, seqB }, new[] { canvas, canvas }, new float[2][], new[] { 0f, 0f }, new[] { 1f, 1f });   // warmup
            sw.Restart();
            for (int s = 0; s < steps; s++)
                model.DecodeCanvasBatched(new[] { seqA, seqB }, new[] { canvas, canvas }, new float[2][], new[] { 0f, 0f }, new[] { 1f, 1f });
            sw.Stop();
            batchedMs = sw.Elapsed.TotalMilliseconds / steps;
        }
        finally
        {
            model.DisposeSeqState(seqA);
            model.DisposeSeqState(seqB);
            model.SelfConditioningEnabled = prevSc;
        }

        double fusedAggTokS = C / (fusedMs / 1000.0);            // fused time-slice aggregate (flat in N)
        double sliceTwoMs = 2 * fusedMs;                         // 2 requests via fused time-slice / step-round
        double batchedAggTokS = (2 * C) / (batchedMs / 1000.0);  // per-op batched aggregate over 2 canvases
        _output.WriteLine($"[diffusion-gemma][bench] fused single-canvas: {fusedMs:F0} ms/step → {fusedAggTokS:F0} canvas-tok/s per request");
        _output.WriteLine($"[diffusion-gemma][bench] 2 requests — fused time-slice: {sliceTwoMs:F0} ms/step-round → {fusedAggTokS:F0} canvas-tok/s aggregate");
        _output.WriteLine($"[diffusion-gemma][bench] 2 requests — per-op batched : {batchedMs:F0} ms/step → {batchedAggTokS:F0} canvas-tok/s aggregate");
        _output.WriteLine($"[diffusion-gemma][bench] fused time-slice delivers {batchedMs / sliceTwoMs:F2}x the aggregate throughput of per-op batching");

        // The scheduler default (fused time-slice) must serve 2 concurrent requests at higher aggregate
        // throughput than the per-op true-batched forward on this compute-bound model — i.e. two fused
        // single-canvas decodes are cheaper than one 2-canvas per-op batched forward.
        Assert.True(fusedMs > 0 && batchedMs > 0);
        Assert.True(sliceTwoMs < batchedMs,
            $"fused time-slice for 2 requests ({sliceTwoMs:F0} ms) was not faster than per-op batched ({batchedMs:F0} ms) — the scheduler default may be wrong for this hardware");
    }
}
