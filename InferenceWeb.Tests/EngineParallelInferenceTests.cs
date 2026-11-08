// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end tests for the continuous-batching InferenceEngine driving real
// GGUF models. Loading and running a real model is expensive (multi-second
// startup, multi-second per request), so these tests are opt-in: set the
// environment variable TS_TEST_MODEL_DIR to a directory containing the
// required GGUFs (Gemma 4, Qwen 3.6, Nemotron 3) and they will run. The media
// tests also need the projector next to the model and the fixtures
// eng/make-test-media.sh writes (TS_TEST_MEDIA_DIR, default ~/work/models/testmedia).
// A test whose model, projector or media file is missing is reported as skipped,
// with the reason, at discovery (ParallelModelFact) - never as a pass. The model
// runs on the lane's pinned GGML backend (TS_TEST_GGML_BACKEND).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Trait("Requires", "Models")]
public class EngineParallelInferenceTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    // The E4B projector under its llama.cpp name (mmproj-gemma-4-E4B-it-BF16.gguf,
    // ...-Q8_0.gguf) or a renamed copy (gemma-4-E4B-mmproj-F16.gguf). FindFirst takes
    // the first match in name order, so a folder with several quantizations uses BF16.
    // Declared before Manifests: static initializers run in textual order.
    private static readonly string[] Gemma4E4BProjectors = { "mmproj-gemma-4-E4B", "gemma-4-E4B-mmproj", "gemma-4-mmproj" };

    private static readonly Dictionary<string, ModelManifest> Manifests = new(StringComparer.OrdinalIgnoreCase)
    {
        ["gemma4"] = new ModelManifest(
            ModelPatterns: new[] { "gemma-4-E4B-it-Q8_0" },
            // The "-assistant" variant and llama.cpp's "mmproj-<model>" projector
            // ship under the same name; exclude both (the projector is a clip GGUF).
            ExcludePatterns: new[] { "assistant", "mmproj" },
            MmprojPatterns: Gemma4E4BProjectors),
        // The exact E4B build from the q4_0 KV-cache reuse bug report
        // (gemma-4-E4B-it-uncensored-Q8_0.gguf). Matches any E4B "it" GGUF;
        // excludes the tiny assistant/mmproj sidecars.
        ["gemma4-e4b"] = new ModelManifest(
            ModelPatterns: new[] { "gemma-4-E4B-it" },
            ExcludePatterns: new[] { "assistant", "mmproj" },
            MmprojPatterns: Gemma4E4BProjectors),
        // The user-reported repro runs the 12B QAT build on ggml_cuda. Matches the
        // exact files from the bug report; set TS_TEST_MODEL_DIR to their folder and
        // TS_TEST_GGML_BACKEND=cuda to exercise the concurrent fused-decode path.
        ["gemma4-12b"] = new ModelManifest(
            ModelPatterns: new[] { "gemma-4-12B-it-qat", "gemma-4-12B-it" },
            ExcludePatterns: new[] { "mmproj", "MTP", "assistant" },
            MmprojPatterns: new[] { "gemma-4-12B-mmproj" }),
        ["qwen36"] = new ModelManifest(
            // 27B (dense) before 35B-A3B (MoE) - the dense build runs faster
            // on CPU/Metal and is what the user requested first.
            ModelPatterns: new[] { "Qwen3.6-27B-IQ4_XS", "Qwen3.6-27B-", "Qwen3.6-35B-A3B" },
            ExcludePatterns: new[] { "mmproj" },
            MmprojPatterns: new[] { "Qwen3.6-27B-mmproj", "Qwen3.6-35B-A3B-mmproj" }),
        // 35B-A3B MoE variant. Same Qwen3.5 architecture as "qwen36" but with
        // sparse MoE FFN (256 experts, top-8). Used to exercise the Phase 3
        // batched MoE dispatch in Qwen35Model.BatchedForward.cs.
        ["qwen36moe"] = new ModelManifest(
            ModelPatterns: new[] { "Qwen3.6-35B-A3B-UD-IQ2_XXS", "Qwen3.6-35B-A3B" },
            ExcludePatterns: new[] { "mmproj" },
            MmprojPatterns: new[] { "Qwen3.6-35B-A3B-mmproj" }),
        // Qwen3.8-27B (the qwen35 architecture), the family of issue #338.
        ["qwen38"] = new ModelManifest(
            ModelPatterns: new[] { "Qwen3.8-27B-UD-IQ3_XXS", "Qwen3.8-27B-" },
            ExcludePatterns: new[] { "mmproj", "DFlash", "MTP" },
            MmprojPatterns: null),
        ["nemotron3"] = new ModelManifest(
            ModelPatterns: new[] { "Nemotron-3-Nano-Omni" },
            ExcludePatterns: new[] { "mmproj" },
            MmprojPatterns: new[] { "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mmproj" }),
        // Mistral 3 (Ministral-3-14B) is the first production model that
        // implements IBatchedPagedModel, so parallel requests on this entry
        // exercise the true batched-paged-attention path through the engine
        // (BatchExecutor.ExecuteStepBatched) rather than the per-sequence
        // KV-swap fallback. Worth comparing the wall-clock numbers against
        // the other models in this table.
        ["mistral3"] = new ModelManifest(
            ModelPatterns: new[] { "Ministral-3-14B-Instruct", "Mistral-3" },
            ExcludePatterns: new[] { "mmproj" },
            MmprojPatterns: new[] { "Ministral-3-14B-Instruct-2512-BF16-mmproj", "Mistral-3-mmproj" }),
    };

    // The fixtures eng/make-test-media.sh writes: the repo banner (image.png), a spoken
    // pangram (sample.wav) and a slow zoom on the banner (sample.mp4). Each media test
    // asserts content only a model that received the media can produce - the banner's
    // title, the pangram's "fox" - as the test matrix's media cells do.
    private const string EnvMediaDir = "TS_TEST_MEDIA_DIR";

    private static string MediaPath(string media)
    {
        string dir = Environment.GetEnvironmentVariable(EnvMediaDir)
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "work", "models", "testmedia");
        return Path.Combine(dir, media switch
        {
            "image" => "image.png",
            "audio" => "sample.wav",
            "video" => "sample.mp4",
            _ => throw new ArgumentOutOfRangeException(nameof(media), media, null),
        });
    }

    // The banner's title as a model that saw image.png or sample.mp4 writes it
    // ("TensorSharp", "Tensor Sharp", "TENSORSHARP").
    private static bool NamesTheBannerTitle(string text)
        => (text ?? string.Empty).Replace(" ", string.Empty).Contains("tensorsharp", StringComparison.OrdinalIgnoreCase);

    private readonly ITestOutputHelper _output;
    public EngineParallelInferenceTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// A real-model test of this class: skipped at discovery, with the reason, unless
    /// TS_TEST_MODEL_DIR holds the model the manifest entry <c>key</c> names (and, for a
    /// media test, its projector and the media fixture). The skip used to be a silent
    /// return inside the test, so a lane without the model reported a pass - which is
    /// how a broken 12B concurrent-reuse repro went unnoticed.
    /// </summary>
    private sealed class ParallelModelFactAttribute : FactAttribute
    {
        public ParallelModelFactAttribute(string key, string media = null)
            => Skip = SkipReason(key, media);
    }

    private static string SkipReason(string key, string media)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            return $"Requires model weights ({EnvModelDir} not set or missing).";
        if (!Manifests.TryGetValue(key, out var manifest))
            return $"No model manifest for '{key}'.";
        if (FindFirst(dir, manifest.ModelPatterns, manifest.ExcludePatterns) == null)
            return $"Requires model weights (no '{string.Join("' / '", manifest.ModelPatterns)}' GGUF under {EnvModelDir}).";
        if (media == null)
            return null;
        if (manifest.MmprojPatterns == null || FindFirst(dir, manifest.MmprojPatterns, null) == null)
            return $"Requires the '{key}' projector under {EnvModelDir}.";
        string file = MediaPath(media);
        return File.Exists(file)
            ? null
            : $"Requires the {media} fixture {file} (eng/make-test-media.sh writes it; {EnvMediaDir} names its folder).";
    }

    [ParallelModelFact("gemma4")] public Task Gemma4_FiveTextPromptsParallel() => RunTextParallel("gemma4", numRequests: 5);
    // Repro for the user-reported bug: two long-generation parallel
    // requests (where each output >> Gemma 4's 512-token SWA window) plus
    // a third request submitted while the first two are still mid-decode.
    // Pre-fix: request #2 produced garbled tokens (per-seq KV swap reading
    // from a wrapped circular cache) and request #3 hit
    // "Continuous-batching engine is unavailable" (TryGetEngine returned
    // null once SupportsKVStateSnapshot flipped to false on cache wrap).
    // Post-fix: ForwardBatch is the default Gemma 4 path so KV swap +
    // snapshot are never needed.
    [ParallelModelFact("gemma4")] public Task Gemma4_ThreeLongGenerationsParallel() => RunLongGenerationParallel("gemma4");
    // Repro for the user-reported "vision encoder + parallel text aborts
    // the process" crash. Pre-fix, the chat pipeline's vision-encoder GGML
    // ops on the request thread raced the engine worker's batched-forward
    // GGML ops on the GPU, and ggml_metal_synchronize would abort with a
    // command-buffer-status=1/2 fatal error. Post-fix, both call sites
    // take ModelBase.GpuComputeLock so they serialise.
    [ParallelModelFact("gemma4", "image")] public Task Gemma4_ImageAndTextSimultaneous() => RunImageAndTextSimultaneous("gemma4");
    [ParallelModelFact("qwen36")] public Task Qwen36_FiveTextPromptsParallel() => RunTextParallel("qwen36", numRequests: 5);
    // Phase 3 verification: same flow on the 35B-A3B MoE GGUF — exercises the
    // batched MoE FFN dispatch added in Qwen35Model.BatchedForward.cs.
    // Tiny budget (2 prompts × 6 new tokens) because IQ2_XXS dequantization +
    // 256-expert routing per layer is slow on CPU/Metal; the test exists to
    // verify the code path runs, not for performance comparison.
    [ParallelModelFact("qwen36moe")] public Task Qwen36MoE_FiveTextPromptsParallel() => RunTextParallel("qwen36moe", numRequests: 2, maxNewTokensOverride: 6);
    [ParallelModelFact("nemotron3")] public Task Nemotron3_FiveTextPromptsParallel() => RunTextParallel("nemotron3", numRequests: 5);
    // Mistral 3 runs through the TRUE batched-paged path (ExecuteStepBatched),
    // not the per-sequence KV swap fallback. Smaller maxNewTokens budget
    // because the managed paged-attention kernel is unoptimised C#; raise
    // once the GPU/native kernel lands.
    [ParallelModelFact("mistral3")] public Task Mistral3_FourTextPromptsParallelBatched() => RunTextParallel("mistral3", numRequests: 4, maxNewTokensOverride: 8);
    // Long-context variant: pads each prompt to ~512 tokens of repeated
    // text so the per-layer paged attention compute actually dominates
    // over the gather + GPU-launch overhead. This is where the GPU paged
    // kernel is expected to outperform the managed scalar kernel.
    [ParallelModelFact("mistral3")] public Task Mistral3_FourLongPromptsParallelBatched() => RunLongContextParallel("mistral3");

    [ParallelModelFact("gemma4")] public Task Gemma4_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("gemma4");
    // Direct repro for the low-kvReusePercent bug: a long first turn (whose
    // K/V state wraps the SWA circular cache) followed by a same-session
    // follow-up. Pre-fix, prefix-cache reuse capped at one sliding window
    // (~10-20% for long histories); post-fix the per-seq extract/inject use
    // modular SWA addressing so the whole history is reusable.
    [ParallelModelFact("gemma4")] public Task Gemma4_SwaPrefixCacheReuseAcrossLongTurn() => RunSwaPrefixReuseRepro("gemma4");
    // Direct repro for the user-reported "KV cache reuse ratio is 0 after a
    // concurrent round" bug. Two conversations run in PARALLEL (so they take the
    // per-sequence FUSED concurrent-decode path, which keeps each request's K/V in
    // its own holder and writes nothing to the shared prefix pool), then two
    // "请继续" follow-ups run in parallel. Pre-fix: the follow-ups found an empty
    // pool (and live-cache continuation is single-stream only) so reuse == 0 and
    // each re-prefilled the whole conversation. Post-fix: the finished holders are
    // retained and re-adopted, so the follow-ups reuse the whole prefix.
    [ParallelModelFact("gemma4-12b")] public Task Gemma4_ConcurrentSwaPrefixReuseAcrossTurns() => RunConcurrentSwaPrefixReuseRepro("gemma4-12b");
    // Real-model repro for the reported bug: with --kv-cache-dtype q4_0 a
    // same-session follow-up ("请继续") after a long first turn reused 0 KV
    // tokens (re-prefilling the whole conversation), while f16 reused nearly
    // all of it. Both variants must now reuse the full prefix via live-cache
    // continuation. Needs TS_TEST_MODEL_DIR=<dir with gemma-4-E4B-it*.gguf>. Gemma 4
    // declines a block-quantized cache and loads f16, so the q4_0 and q8_0 rows check
    // that the fallback keeps long-turn reuse.
    [ParallelModelFact("gemma4-e4b")] public Task Gemma4E4B_Q4_0_LongTurnPrefixReuse() => RunKvDtypeLongPromptReuseRepro("gemma4-e4b", KvCacheDtype.Q4_0);
    [ParallelModelFact("gemma4-e4b")] public Task Gemma4E4B_F16_LongTurnPrefixReuse() => RunKvDtypeLongPromptReuseRepro("gemma4-e4b", KvCacheDtype.F16);
    [ParallelModelFact("gemma4-e4b")] public Task Gemma4E4B_Q8_0_LongTurnPrefixReuse() => RunKvDtypeLongPromptReuseRepro("gemma4-e4b", KvCacheDtype.Q8_0);
    [ParallelModelFact("qwen36")] public Task Qwen36_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("qwen36");
    [ParallelModelFact("nemotron3")] public Task Nemotron3_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("nemotron3");

    [ParallelModelFact("gemma4", "image")] public Task Gemma4_ImagePromptThenTextParallel() => RunMixedMultimodal("gemma4");
    [ParallelModelFact("qwen36", "image")] public Task Qwen36_ImagePromptThenTextParallel() => RunMixedMultimodal("qwen36");
    [ParallelModelFact("nemotron3", "image")] public Task Nemotron3_ImagePromptThenTextParallel() => RunMixedMultimodal("nemotron3");

    [ParallelModelFact("gemma4", "audio")] public Task Gemma4_AudioPrompt() => RunAudioSmoke("gemma4");
    [ParallelModelFact("nemotron3", "audio")] public Task Nemotron3_AudioPrompt() => RunAudioSmoke("nemotron3");

    [ParallelModelFact("gemma4", "video")] public Task Gemma4_VideoPrompt() => RunVideoSmoke("gemma4");
    [ParallelModelFact("nemotron3", "video")] public Task Nemotron3_VideoPrompt() => RunVideoSmoke("nemotron3");

    // ---- core flows ----

    private async Task RunTextParallel(string key, int numRequests, int maxNewTokensOverride = 24)
    {
        var ctx = Load(key);
        using (ctx)
        {
            // Report which path the executor will actually use. The
            // TS_SCHED_DISABLE_BATCHED env var force-disables the batched
            // path even when the model implements IBatchedPagedModel.
            bool modelSupportsBatched = ctx.Model is TensorSharp.Runtime.Scheduling.IBatchedPagedModel;
            string disableRaw = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
            bool forceDisabled = !string.IsNullOrEmpty(disableRaw) && disableRaw != "0"
                && disableRaw.ToLowerInvariant() != "false";
            string path = (modelSupportsBatched && !forceDisabled)
                ? "batched (IBatchedPagedModel)"
                : "per-sequence KV swap";
            _output.WriteLine($"[{key}] dispatch path = {path}" +
                (forceDisabled && modelSupportsBatched ? " [forced via TS_SCHED_DISABLE_BATCHED]" : ""));

            var prompts = MakeDiverseTextPrompts(numRequests);
            var sw = Stopwatch.StartNew();
            var results = await SubmitAndGather(ctx, prompts, maxNewTokens: maxNewTokensOverride);
            sw.Stop();

            Assert.Equal(prompts.Count, results.Count);
            foreach (var r in results)
                Assert.True(r.OutputTokenCount > 0, $"{r.RequestId}: empty output");
            LogBench(ctx, "text-parallel", numRequests, results, sw.Elapsed);
        }
    }

    private async Task RunLongContextParallel(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            bool modelSupportsBatched = ctx.Model is TensorSharp.Runtime.Scheduling.IBatchedPagedModel;
            string disableRaw = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
            bool forceDisabled = !string.IsNullOrEmpty(disableRaw) && disableRaw != "0"
                && disableRaw.ToLowerInvariant() != "false";
            string path = (modelSupportsBatched && !forceDisabled)
                ? "batched (IBatchedPagedModel)"
                : "per-sequence KV swap";
            _output.WriteLine($"[{key}] long-context dispatch path = {path}");

            // Build four ~500-token prompts by padding with filler. The
            // per-sequence attention then walks ~500 tokens of K/V per
            // layer, which dwarfs the tensor-launch overhead.
            string filler = string.Concat(System.Linq.Enumerable.Repeat(
                "The quick brown fox jumps over the lazy dog. ", 80));
            var prompts = new List<string>();
            string[] questions =
            {
                "Q: What is two plus two?\nA:",
                "Q: Name a primary color.\nA:",
                "Q: Capital of France?\nA:",
                "Q: Year of the moon landing?\nA:",
            };
            for (int i = 0; i < 4; i++)
                prompts.Add(filler + questions[i]);

            var sw = Stopwatch.StartNew();
            var results = await SubmitAndGather(ctx, prompts, maxNewTokens: 6);
            sw.Stop();

            Assert.Equal(4, results.Count);
            foreach (var r in results) Assert.True(r.OutputTokenCount > 0);
            LogBench(ctx, "long-context-parallel", 4, results, sw.Elapsed);
        }
    }


    private async Task RunLongGenerationParallel(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            // The bug repro requires generations that EXCEED Gemma 4's
            // 512-token SWA window so the legacy per-seq KV-swap path
            // exercises the wrapped-cache code path (where it produced
            // garbage). Ask the model for a long answer to get there.
            // 720 tokens > 512 sliding-window comfortably.
            const int maxNewTokens = 720;
            var prompts = new List<string>
            {
                "请详细介绍最终幻想7。",
                "请详细介绍时间简史。",
                "请详细介绍量子力学。",
            };

            // Greedy sampling so the comparison is deterministic - any
            // garbling we see is a state bug, not random sampling noise.
            var sampling = SamplingConfig.Greedy;

            var sw = Stopwatch.StartNew();
            var results = await SubmitAndGather(ctx, prompts, maxNewTokens, sampling);
            sw.Stop();

            Assert.Equal(prompts.Count, results.Count);
            // Report ALL three first, then assert - so we can see whether
            // the degenerate output is one-off or affects every sequence.
            var failures = new List<string>();
            foreach (var r in results)
            {
                if (r.OutputTokenCount == 0)
                {
                    failures.Add($"{r.RequestId}: empty output (engine probably became unavailable mid-batch - the pre-fix failure mode)");
                    continue;
                }
                string txt = r.OutputText ?? string.Empty;
                int replCount = 0;
                foreach (var ch in txt) if (ch == '�') replCount++;
                int longestRepeat = LongestImmediateRepeat(txt);
                _output.WriteLine(
                    $"[{key}] {r.RequestId}: {r.OutputTokenCount} tok, " +
                    $"replChars={replCount}, longestImmediateRepeat={longestRepeat}, " +
                    $"preview=\"{Truncate(txt, 120)}\"");
                if (replCount >= 10)
                    failures.Add($"{r.RequestId}: {replCount} UTF-8 replacement chars - output is byte-corrupted");
                if (longestRepeat >= 20)
                    failures.Add($"{r.RequestId}: longest immediate repeat = {longestRepeat} - output looks degenerate (state corruption?)");
            }
            Assert.True(failures.Count == 0, string.Join("\n", failures));
            LogBench(ctx, "long-generation-parallel", prompts.Count, results, sw.Elapsed);
        }
    }

    private async Task RunImageAndTextSimultaneous(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            Assert.False(string.IsNullOrEmpty(ctx.MmprojPath), $"[{key}] the projector did not load");
            string imagePath = MediaPath("image");

            // Fire two text requests and one image request simultaneously.
            // The image request runs the vision encoder on the request
            // thread while the engine worker chews through the text
            // requests on the GPU - that's the exact concurrency that
            // crashed pre-fix. The image-request assertion below also
            // covers the per-request injector bucketing fix: if image
            // embeddings get consumed by the wrong sequence's Forward()
            // (the pre-fix model-shared-state bug), the image response
            // cannot name the banner's title.
            var tasks = new List<Task<RequestResult>>
            {
                SubmitTextPrompt(ctx, "请详细介绍最终幻想7。", maxNewTokens: 64, "text-ff7"),
                SubmitTextPrompt(ctx, "请详细介绍时间简史。", maxNewTokens: 64, "text-history"),
                SubmitImagePrompt(ctx, imagePath, "What is the large title written in this image? Answer with the title only.", maxNewTokens: 24),
            };
            var results = await Task.WhenAll(tasks);

            RequestResult imageResult = null;
            foreach (var r in results)
            {
                _output.WriteLine($"[{key}] {r.RequestId}: {r.OutputTokenCount} tok, " +
                    $"preview=\"{Truncate(r.OutputText, 80)}\"");
                Assert.True(r.OutputTokenCount > 0,
                    $"{r.RequestId}: empty output (process probably crashed mid-batch - the pre-fix failure mode)");
                if (r.RequestId.StartsWith("img-", StringComparison.Ordinal)) imageResult = r;
            }

            // Verify the model actually read the banner. Pre-fix the engine
            // path never queued vision embeddings into the model, so the
            // model saw only placeholder tokens and could not name the title.
            Assert.NotNull(imageResult);
            Assert.True(NamesTheBannerTitle(imageResult.OutputText),
                $"image request output \"{imageResult.OutputText}\" doesn't name the banner's title (TensorSharp) - " +
                "vision embeddings probably weren't injected (engine-path multimodal bug).");
        }
    }

    // Real-model driver for the q4_0 KV-cache reuse bug. A long first turn
    // (whose K/V wraps the SWA circular cache) followed by a same-session
    // "请继续" that must reuse the whole prefix via live-cache continuation,
    // run under a chosen KV-cache dtype. Reports reuse% and the follow-up's
    // wall time (the "high performance" check: reuse means the follow-up
    // re-prefills only the short new suffix, not the whole conversation).
    private async Task RunKvDtypeLongPromptReuseRepro(string key, TensorSharp.Models.KvCacheDtype kvDtype)
    {
        // The model captures KvCacheDtypeConfig.Current at construction, so
        // set it BEFORE loading (mirrors the server's --kv-cache-dtype flag), and
        // put the process-wide choice back afterwards for the tests that follow.
        var restoreDtype = TensorSharp.Models.KvCacheDtypeConfig.Current;
        bool restoreExplicit = TensorSharp.Models.KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            TensorSharp.Models.KvCacheDtypeConfig.Set(kvDtype);
            await RunKvDtypeLongPromptReuseReproCore(key, kvDtype);
        }
        finally
        {
            TensorSharp.Models.KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit);
        }
    }

    private async Task RunKvDtypeLongPromptReuseReproCore(string key, TensorSharp.Models.KvCacheDtype kvDtype)
    {
        var ctx = Load(key);
        using (ctx)
        {
            // Gemma 4 declines a block-quantized K/V cache on every attention path
            // (Gemma4Model.SupportsBlockQuantizedKvCache: its circular SWA helpers
            // are float-only) and loads f16 instead, with a warning. Its q4_0 and
            // q8_0 rows therefore check that fallback and that long-turn reuse
            // still holds under it.
            bool declinesBlockQuantized = ctx.Model is Gemma4Model
                && (kvDtype == TensorSharp.Models.KvCacheDtype.Q8_0 || kvDtype == TensorSharp.Models.KvCacheDtype.Q4_0);
            Assert.Equal(declinesBlockQuantized ? TensorSharp.Models.KvCacheDtype.F16 : kvDtype, ctx.Model.KvCacheDtype);
            int window = ctx.Model.Config?.SlidingWindow ?? 0;
            _output.WriteLine($"[{key}] kvDtype={ctx.Model.KvCacheDtype.ToShortString()} slidingWindow={window}");
            var sampling = SamplingConfig.Greedy;

            // Build a long first-turn user message (~2.4k tokens of deterministic
            // filler) so turn-1 K/V comfortably exceeds the SWA window and only the
            // conversation's retained end state (not window-capped pages) can reuse it.
            var sb = new System.Text.StringBuilder();
            sb.Append("请阅读下面这段材料，然后用中文简要总结它的主要内容。\n\n");
            for (int i = 0; i < 120; i++)
                sb.Append($"第{i + 1}段：在一个遥远的星系里，探险者们记录下了行星的轨道、恒星的亮度以及气候的缓慢变化，并把这些观测整理成册。 ");
            string longText = sb.ToString();

            var turn1History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = longText },
            };
            var sw1 = Stopwatch.StartNew();
            var (t1, t1Reply) = await SubmitWithHistoryCapturingTokens(ctx, turn1History, maxNewTokens: 40, "kvq-turn1", sampling, "kvq");
            sw1.Stop();
            _output.WriteLine($"[{key}] turn1: prompt={t1.PromptTokenCount} reused={t1.PrefixCacheReusedTokens} out={t1.OutputTokenCount} wall={sw1.Elapsed.TotalMilliseconds:F0}ms");
            Assert.True(t1.PromptTokenCount > window,
                $"turn1 prompt {t1.PromptTokenCount} must exceed the SWA window {window} to exercise the bug");

            // Turn 2: same session, "请继续" appended after turn 1's output. Splice
            // turn 1's raw output tokens so the re-render produces an exact token
            // prefix match (as the production chat pipeline does).
            var turn2History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = longText },
                t1Reply,
                new ChatMessage { Role = "user", Content = "请继续" },
            };
            var sw2 = Stopwatch.StartNew();
            var (t2, _) = await SubmitWithHistoryCapturingTokens(ctx, turn2History, maxNewTokens: 40, "kvq-turn2", sampling, "kvq");
            sw2.Stop();

            double t2Pct = t2.PromptTokenCount > 0 ? 100.0 * t2.PrefixCacheReusedTokens / t2.PromptTokenCount : 0;
            _output.WriteLine($"[{key}] turn2: prompt={t2.PromptTokenCount} reused={t2.PrefixCacheReusedTokens} ({t2Pct:F1}%) out={t2.OutputTokenCount} wall={sw2.Elapsed.TotalMilliseconds:F0}ms");
            _output.WriteLine($"[{key}] turn2 sample: {Truncate(t2.OutputText, 160)}");

            // Output sanity: a re-prefill loop or corruption would show as long
            // immediate repeats.
            Assert.True(LongestImmediateRepeat(t2.OutputText ?? string.Empty) < 20,
                $"turn2 output looks degenerate: {Truncate(t2.OutputText, 200)}");
            // The fix: the follow-up reuses (almost) the entire conversation
            // prefix instead of the reported 0. Pre-fix q4_0/q8_0 gave ~0% here
            // while f16 gave ~full; post-fix all three reuse the full prefix.
            Assert.True(t2Pct >= 70.0,
                $"[{kvDtype.ToShortString()}] turn2 reuse {t2Pct:F1}% too low (the q4_0 reuse-0 bug). " +
                $"reused={t2.PrefixCacheReusedTokens}/{t2.PromptTokenCount}");
        }
    }

    private async Task RunSwaPrefixReuseRepro(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            var sampling = SamplingConfig.Greedy;

            // Turn 1: long generation that drives the SWA cache past one window
            // (Gemma 4 default SWA = 512 tokens, so 600 new tokens guarantees wrap).
            var turn1History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
            };
            var (t1, t1Reply) = await SubmitWithHistoryCapturingTokens(ctx, turn1History, maxNewTokens: 600, "turn1", sampling, "ff7");
            _output.WriteLine($"[{key}] turn1: prompt={t1.PromptTokenCount} reused={t1.PrefixCacheReusedTokens} out={t1.OutputTokenCount}");

            // Turn 2: same session, follow-up "还有吗" appended after turn 1's output.
            // Splice turn 1's raw output tokens into the assistant message so the
            // chat template re-render produces an exact token prefix match (the
            // production chat pipeline does this via ChatHistoryPreparer).
            var turn2History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
                t1Reply,
                new ChatMessage { Role = "user", Content = "还有吗" },
            };
            var (t2, t2Reply) = await SubmitWithHistoryCapturingTokens(ctx, turn2History, maxNewTokens: 200, "turn2", sampling, "ff7");
            _output.WriteLine($"[{key}] turn2: prompt={t2.PromptTokenCount} reused={t2.PrefixCacheReusedTokens} out={t2.OutputTokenCount}");

            // Turn 3: "请继续"
            var turn3History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
                t1Reply,
                new ChatMessage { Role = "user", Content = "还有吗" },
                t2Reply,
                new ChatMessage { Role = "user", Content = "请继续" },
            };
            var (t3, _) = await SubmitWithHistoryCapturingTokens(ctx, turn3History, maxNewTokens: 200, "turn3", sampling, "ff7");
            _output.WriteLine($"[{key}] turn3: prompt={t3.PromptTokenCount} reused={t3.PrefixCacheReusedTokens} out={t3.OutputTokenCount}");

            // Pre-fix: t2.PrefixCacheReusedTokens ≈ slidingWindow (512), giving 10-30% reuse for a long history.
            // Post-fix: t2's prefix should match almost all of turn1's prompt+assistant block.
            double t2Pct = 100.0 * t2.PrefixCacheReusedTokens / t2.PromptTokenCount;
            double t3Pct = 100.0 * t3.PrefixCacheReusedTokens / t3.PromptTokenCount;
            _output.WriteLine($"[{key}] reuse pct: t2={t2Pct:F1}% t3={t3Pct:F1}%");
            _output.WriteLine($"[{key}] t2 sample: {Truncate(t2.OutputText, 200)}");
            _output.WriteLine($"[{key}] t3 sample: {Truncate(t3.OutputText, 200)}");
            // Output sanity: a token-repeat loop would manifest as long
            // immediate repeats, which is how state-corruption bugs surface.
            Assert.True(LongestImmediateRepeat(t2.OutputText ?? string.Empty) < 20,
                $"turn2 output looks degenerate: {Truncate(t2.OutputText, 200)}");
            Assert.True(LongestImmediateRepeat(t3.OutputText ?? string.Empty) < 20,
                $"turn3 output looks degenerate: {Truncate(t3.OutputText, 200)}");
            Assert.True(t2Pct >= 70.0, $"turn2 reuse {t2Pct:F1}% too low (expected ≥70% after SWA fix)");
            Assert.True(t3Pct >= 70.0, $"turn3 reuse {t3Pct:F1}% too low (expected ≥70% after SWA fix)");
        }
    }

    private async Task RunConcurrentSwaPrefixReuseRepro(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            var sampling = SamplingConfig.Greedy;
            // Round-1 generation must exceed Gemma 4's SWA window (512) for the bug to
            // bite; keep that as the default but allow a smaller value for a quick
            // end-to-end smoke (TS_REPRO_ROUND1_TOKENS / TS_REPRO_ROUND2_TOKENS).
            int round1Tokens = EnvInt("TS_REPRO_ROUND1_TOKENS", 600);
            int round2Tokens = EnvInt("TS_REPRO_ROUND2_TOKENS", 200);

            // ---- Round 1: two distinct conversations, IN PARALLEL. ----
            // Long generation (> Gemma 4's 512-token SWA window) so the only thing
            // that can reuse the prefix later is each conversation's retained end
            // state. Running both at once forces the per-seq fused path.
            var histA1 = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
            };
            var histB1 = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍时间简史" },
            };
            var sw1 = Stopwatch.StartNew();
            var t1Task = SubmitWithHistoryCapturingTokens(ctx, histA1, maxNewTokens: round1Tokens, "A1", sampling, "A");
            var t1bTask = SubmitWithHistoryCapturingTokens(ctx, histB1, maxNewTokens: round1Tokens, "B1", sampling, "B");
            await Task.WhenAll(t1Task, t1bTask);
            sw1.Stop();
            var (a1, a1Reply) = t1Task.Result;
            var (b1, b1Reply) = t1bTask.Result;
            _output.WriteLine($"[{key}] round1 parallel: A prompt={a1.PromptTokenCount} out={a1.OutputTokenCount} | " +
                $"B prompt={b1.PromptTokenCount} out={b1.OutputTokenCount} | wall={sw1.Elapsed.TotalMilliseconds:F0}ms");

            // ---- Round 2: two "请继续" follow-ups, IN PARALLEL. ----
            var histA2 = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
                a1Reply,
                new ChatMessage { Role = "user", Content = "请继续" },
            };
            var histB2 = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍时间简史" },
                b1Reply,
                new ChatMessage { Role = "user", Content = "请继续" },
            };
            var sw2 = Stopwatch.StartNew();
            var t2Task = SubmitWithHistoryCapturingTokens(ctx, histA2, maxNewTokens: round2Tokens, "A2", sampling, "A");
            var t2bTask = SubmitWithHistoryCapturingTokens(ctx, histB2, maxNewTokens: round2Tokens, "B2", sampling, "B");
            await Task.WhenAll(t2Task, t2bTask);
            sw2.Stop();
            var (a2, _) = t2Task.Result;
            var (b2, _) = t2bTask.Result;

            double a2Pct = 100.0 * a2.PrefixCacheReusedTokens / a2.PromptTokenCount;
            double b2Pct = 100.0 * b2.PrefixCacheReusedTokens / b2.PromptTokenCount;
            _output.WriteLine($"[{key}] round2 parallel follow-ups: " +
                $"A reused={a2.PrefixCacheReusedTokens}/{a2.PromptTokenCount} ({a2Pct:F1}%) | " +
                $"B reused={b2.PrefixCacheReusedTokens}/{b2.PromptTokenCount} ({b2Pct:F1}%) | " +
                $"wall={sw2.Elapsed.TotalMilliseconds:F0}ms");
            _output.WriteLine($"[{key}] A2 sample: {Truncate(a2.OutputText, 160)}");
            _output.WriteLine($"[{key}] B2 sample: {Truncate(b2.OutputText, 160)}");

            // Output must stay coherent (a corrupt KV continuation degenerates into
            // long immediate repeats).
            Assert.True(LongestImmediateRepeat(a2.OutputText ?? string.Empty) < 20,
                $"A2 output looks degenerate: {Truncate(a2.OutputText, 200)}");
            Assert.True(LongestImmediateRepeat(b2.OutputText ?? string.Empty) < 20,
                $"B2 output looks degenerate: {Truncate(b2.OutputText, 200)}");

            // The core regression assertion: reuse is no longer 0 for parallel
            // follow-ups. Both should reuse nearly the whole conversation prefix.
            Assert.True(a2.PrefixCacheReusedTokens > 0,
                "A2 KV-cache reuse is 0 (the reported bug) — retained-fused continuation didn't engage.");
            Assert.True(b2.PrefixCacheReusedTokens > 0,
                "B2 KV-cache reuse is 0 (the reported bug) — retained-fused continuation didn't engage.");
            Assert.True(a2Pct >= 70.0, $"A2 reuse {a2Pct:F1}% too low (expected ≥70% after the fix)");
            Assert.True(b2Pct >= 70.0, $"B2 reuse {b2Pct:F1}% too low (expected ≥70% after the fix)");
        }
    }

    // A turn of a multi-turn conversation, run as the chat pipeline runs one: the Radix
    // cache reuses a finished turn only for a later request of the same conversation
    // (cacheScope; unscoped requests share only a public prefix), and the reply comes
    // back as the pipeline records it - the raw tokens the cache holds plus the framing
    // the generation prompt ended with (Gemma 4's empty thought channel, the boundary
    // whitespace), which the template does not re-emit for a past turn. A reply without
    // that framing diverges from the cache right after the turn header, and continuing
    // the conversation's retained state would mean rewinding the whole answer.
    private async Task<(RequestResult result, ChatMessage reply)> SubmitWithHistoryCapturingTokens(
        EngineContext ctx, List<ChatMessage> history, int maxNewTokens, string reqId,
        SamplingConfig sampling, string conversation)
    {
        string arch = ctx.Model.Config?.Architecture ?? string.Empty;
        var tokens = ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer, ctx.Model.Config?.ChatTemplate, history, arch,
            addGenerationPrompt: true, out _, out string trailingWhitespace, tools: null, enableThinking: false);
        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, sampling ?? SamplingConfig.Default,
            cacheScope: conversation);
        var handle = ctx.Engine.SubmitRequest(seq);
        var outputTokens = await DrainHandle(handle);
        var completion = await handle.Completion;
        var result = new RequestResult
        {
            RequestId = reqId,
            OutputTokenCount = outputTokens.Count,
            OutputText = ctx.Model.Tokenizer.Decode(outputTokens),
            PrefixCacheReusedTokens = completion.PrefixCacheReusedTokens,
            PromptTokenCount = completion.PromptTokenCount,
        };
        var reply = new ChatMessage
        {
            Role = "assistant",
            Content = result.OutputText,
            RawOutputTokens = outputTokens,
            RawPromptTrailingWhitespace = trailingWhitespace,
            RawGenerationSuffix = TensorSharp.Server.ChatGenerationPipeline.RecordedGenerationSuffix(
                ctx.Model.Tokenizer, tokens, arch, enableThinking: false),
        };
        return (result, reply);
    }

    private async Task RunPrefixCacheBench(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            const string system = "You are a helpful, concise assistant. Answer in one sentence.\n\n";
            var sw1 = Stopwatch.StartNew();
            var first = (await SubmitAndGather(ctx, new[] { system + "User: What is two plus two?\nAssistant:" }, 16)).Single();
            sw1.Stop();

            var sw2 = Stopwatch.StartNew();
            var second = (await SubmitAndGather(ctx, new[] { system + "User: What is the capital of France?\nAssistant:" }, 16)).Single();
            sw2.Stop();

            _output.WriteLine($"[{key}] first request: prompt={first.PromptTokenCount} tokens, wall={sw1.Elapsed.TotalMilliseconds:F0}ms");
            _output.WriteLine($"[{key}] second request: prompt={second.PromptTokenCount} tokens, reused={second.PrefixCacheReusedTokens} tokens, wall={sw2.Elapsed.TotalMilliseconds:F0}ms");
            Assert.True(first.OutputTokenCount > 0);
            Assert.True(second.OutputTokenCount > 0);
        }
    }

    private async Task RunMixedMultimodal(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            Assert.False(string.IsNullOrEmpty(ctx.MmprojPath), $"[{key}] the projector did not load");
            string imagePath = MediaPath("image");

            var sw = Stopwatch.StartNew();
            var imageResp = await SubmitImagePrompt(ctx, imagePath, "What is the large title written in this picture? Answer with the title only.", maxNewTokens: 24);
            _output.WriteLine($"[{key}] image: {imageResp.OutputTokenCount} out tokens in {sw.Elapsed.TotalMilliseconds:F0}ms — \"{Truncate(imageResp.OutputText, 80)}\"");
            Assert.True(imageResp.OutputTokenCount > 0);
            // Only a model that sees the vision content (not just the bare
            // <|image_pad|> placeholders) can read the banner's title. This
            // assertion catches regressions where the image embeddings or
            // MRoPE positions get dropped silently.
            Assert.True(NamesTheBannerTitle(imageResp.OutputText),
                $"{key} image response \"{imageResp.OutputText}\" doesn't name the banner's title (TensorSharp) - " +
                "vision embeddings or MRoPE positions probably weren't injected correctly.");

            // Now submit text prompts concurrently with no multimodal state pollution.
            sw.Restart();
            var textResults = await SubmitAndGather(ctx, MakeDiverseTextPrompts(3), maxNewTokens: 16);
            Assert.All(textResults, r => Assert.True(r.OutputTokenCount > 0));
            LogBench(ctx, "image+text", 4, textResults.Concat(new[] { imageResp }).ToList(), sw.Elapsed);
        }
    }

    private async Task RunAudioSmoke(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            Assert.False(string.IsNullOrEmpty(ctx.MmprojPath), $"[{key}] the projector did not load");
            string audioPath = MediaPath("audio");
            var resp = await SubmitAudioPrompt(ctx, audioPath, "Transcribe the first sentence of this audio clip.", maxNewTokens: 32);
            _output.WriteLine($"[{key}] audio: {resp.OutputTokenCount} out tokens — \"{Truncate(resp.OutputText, 80)}\"");
            // sample.wav speaks "The quick brown fox jumps over the lazy dog near the river bank".
            Assert.True(resp.OutputText?.Contains("fox", StringComparison.OrdinalIgnoreCase) == true,
                $"[{key}] audio output \"{resp.OutputText}\" doesn't transcribe the pangram - the clip probably never reached the encoder.");
        }
    }

    private async Task RunVideoSmoke(string key)
    {
        var ctx = Load(key);
        using (ctx)
        {
            Assert.False(string.IsNullOrEmpty(ctx.MmprojPath), $"[{key}] the projector did not load");
            string videoPath = MediaPath("video");

            // A video goes in as its frames, the way the CLI and the upload endpoint
            // send it: PNGs sampled at 1 fps in ImagePaths, with IsVideo=true. The
            // injector decodes images only; handed the .mp4 itself it throws.
            List<string> frames = MediaHelper.ExtractVideoFrames(videoPath);
            Assert.NotEmpty(frames);
            try
            {
                var history = new List<ChatMessage>
                {
                    new ChatMessage
                    {
                        Role = "user",
                        Content = "What is the large title shown in this video? Answer with the title only.",
                        ImagePaths = frames,
                        IsVideo = true,
                    }
                };

                var resp = await SubmitWithHistory(ctx, history, maxNewTokens: 20, "video");
                _output.WriteLine($"[{key}] video ({frames.Count} frames): {resp.OutputTokenCount} out — \"{Truncate(resp.OutputText, 80)}\"");
                Assert.True(NamesTheBannerTitle(resp.OutputText),
                    $"[{key}] video output \"{resp.OutputText}\" doesn't name the banner's title (TensorSharp) - the frames probably never reached the encoder.");
            }
            finally
            {
                try { Directory.Delete(Path.GetDirectoryName(frames[0])!, recursive: true); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    // ---- request helpers ----

    // Find the longest run of repeated identical substrings of length 2-8.
    // "abcabcabc" → 3, "ToToToToTo" → 5, "abcdef" → 1.
    private static int LongestImmediateRepeat(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int best = 1;
        for (int unit = 2; unit <= 8 && unit * 2 <= s.Length; unit++)
        {
            int run = 1;
            for (int i = 0; i + unit * 2 <= s.Length; i++)
            {
                bool match = true;
                for (int k = 0; k < unit; k++)
                    if (s[i + k] != s[i + unit + k]) { match = false; break; }
                if (match)
                {
                    run++;
                    if (run > best) best = run;
                    i += unit - 1; // skip ahead so we count non-overlapping repeats
                }
                else run = 1;
            }
        }
        return best;
    }

    private static List<string> MakeDiverseTextPrompts(int n)
    {
        string[] templates =
        {
            "Q: What is two plus two? Answer with one sentence.\nA:",
            "Q: Write a single-sentence haiku about computers.\nA:",
            "Q: Name three colors. Reply with a comma-separated list.\nA:",
            "Q: What is the boiling point of water at sea level?\nA:",
            "Q: Translate 'good morning' to Spanish.\nA:",
            "Q: Who wrote Hamlet?\nA:",
            "Q: What is the chemical symbol for gold?\nA:",
            "Q: In which year did the first moon landing happen?\nA:",
        };
        var prompts = new List<string>(n);
        for (int i = 0; i < n; i++)
            prompts.Add(templates[i % templates.Length]);
        return prompts;
    }

    private async Task<List<RequestResult>> SubmitAndGather(
        EngineContext ctx, IEnumerable<string> prompts, int maxNewTokens,
        SamplingConfig sampling = null)
    {
        var tasks = new List<Task<RequestResult>>();
        int i = 0;
        foreach (var p in prompts)
            tasks.Add(SubmitTextPrompt(ctx, p, maxNewTokens, $"r{i++}", sampling));
        return (await Task.WhenAll(tasks)).ToList();
    }

    private async Task<RequestResult> SubmitTextPrompt(
        EngineContext ctx, string prompt, int maxNewTokens, string reqId,
        SamplingConfig sampling = null)
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        return await SubmitWithHistory(ctx, history, maxNewTokens, reqId, sampling);
    }

    private async Task<RequestResult> SubmitImagePrompt(
        EngineContext ctx, string imagePath, string prompt, int maxNewTokens)
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = prompt, ImagePaths = new List<string> { imagePath } }
        };
        return await SubmitWithHistory(ctx, history, maxNewTokens, "img-" + Path.GetFileNameWithoutExtension(imagePath));
    }

    private async Task<RequestResult> SubmitAudioPrompt(
        EngineContext ctx, string audioPath, string prompt, int maxNewTokens)
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = prompt, AudioPaths = new List<string> { audioPath } }
        };
        return await SubmitWithHistory(ctx, history, maxNewTokens, "audio");
    }

    private async Task<RequestResult> SubmitWithHistory(
        EngineContext ctx, List<ChatMessage> history, int maxNewTokens, string reqId,
        SamplingConfig sampling = null)
    {
        bool hasMultimodal = history.Exists(m =>
            (m.ImagePaths != null && m.ImagePaths.Count > 0) ||
            (m.AudioPaths != null && m.AudioPaths.Count > 0));

        List<int> tokens;
        if (hasMultimodal)
        {
            // Multimodal prep drives the vision/audio encoder, which runs
            // many GGML ops. Take the same model-level lock the engine
            // worker uses so we serialise against batched forward (GGML
            // backends are not thread-safe; concurrent compute aborts the
            // process). This mirrors what ChatGenerationPipeline does in
            // production. Bucket prepared embeddings by request id so
            // concurrent requests don't share the injector's state.
            lock (ctx.Model.GpuComputeLock)
            {
                tokens = RenderTokens(ctx, history);
                tokens = ctx.Model.MultimodalInjector.ProcessPromptTokens(history, tokens, reqId);
            }
        }
        else
        {
            tokens = RenderTokens(ctx, history);
        }

        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, sampling ?? SamplingConfig.Default);
        try
        {
            var handle = ctx.Engine.SubmitRequest(seq);
            var outputTokens = await DrainHandle(handle);
            var completion = await handle.Completion;

            return new RequestResult
            {
                RequestId = reqId,
                OutputTokenCount = outputTokens.Count,
                OutputText = ctx.Model.Tokenizer.Decode(outputTokens),
                PrefixCacheReusedTokens = completion.PrefixCacheReusedTokens,
                PromptTokenCount = completion.PromptTokenCount,
            };
        }
        finally
        {
            if (hasMultimodal)
                ctx.Model.MultimodalInjector.ClearPreparedPromptState(reqId);
        }
    }

    private static async Task<List<int>> DrainHandle(InferenceRequestHandle handle)
    {
        var outs = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
                outs.Add(tok);
        }
        catch (Exception) { /* error reflected in Completion */ }
        return outs;
    }

    private static List<int> RenderTokens(EngineContext ctx, List<ChatMessage> history)
    {
        return ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer,
            ctx.Model.Config?.ChatTemplate,
            history,
            ctx.Model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true,
            tools: null,
            enableThinking: false);
    }

    private static int EnvInt(string name, int fallback)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(v) && int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    private static string Truncate(string s, int maxLen)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        s = s.Replace("\n", " ").Replace("\r", " ").Trim();
        return s.Length <= maxLen ? s : s.Substring(0, maxLen) + "...";
    }

    /// <summary>Load the manifest entry's model; <see cref="ParallelModelFactAttribute"/> has
    /// already skipped the test when it is absent. A model that is present but does not
    /// load is a failure, not a skip.</summary>
    private EngineContext Load(string key)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        var manifest = Manifests[key];
        string modelPath = FindFirst(dir, manifest.ModelPatterns, manifest.ExcludePatterns);
        Assert.NotNull(modelPath);
        string mmproj = manifest.MmprojPatterns != null ? FindFirst(dir, manifest.MmprojPatterns, null) : null;
        _output.WriteLine($"[{key}] loading {Path.GetFileName(modelPath)} (mmproj={Path.GetFileName(mmproj ?? "")})");
        return new EngineContext(modelPath, mmproj);
    }

    private static string FindFirst(string dir, IEnumerable<string> patterns, string[] excludePatterns)
    {
        var files = Directory.GetFiles(dir, "*.gguf");
        Array.Sort(files, StringComparer.Ordinal);
        foreach (var p in patterns)
        {
            foreach (var f in files)
            {
                string name = Path.GetFileName(f);
                if (!name.Contains(p, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (excludePatterns != null)
                {
                    bool excluded = false;
                    foreach (var ex in excludePatterns)
                    {
                        if (name.Contains(ex, StringComparison.OrdinalIgnoreCase))
                        {
                            excluded = true;
                            break;
                        }
                    }
                    if (excluded) continue;
                }
                return f;
            }
        }
        return null;
    }

    private void LogBench(EngineContext ctx, string label, int n, IReadOnlyList<RequestResult> results, TimeSpan wall)
    {
        int totalTokens = 0, totalPrompt = 0, totalReused = 0;
        foreach (var r in results)
        {
            totalTokens += r.OutputTokenCount;
            totalPrompt += r.PromptTokenCount;
            totalReused += r.PrefixCacheReusedTokens;
        }
        double sec = wall.TotalSeconds;
        double tps = sec > 0 ? totalTokens / sec : 0;
        var stats = ctx.Engine.PoolStats;
        _output.WriteLine(
            $"[{ctx.Model.Config?.Architecture ?? "?"}] {label} " +
            $"n={n} promptTokens={totalPrompt} outTokens={totalTokens} reused={totalReused} " +
            $"wall={sec:F2}s tps={tps:F1} " +
            $"poolFree={stats.freeBlocks}/{stats.totalBlocks}");
    }

    private sealed record ModelManifest(
        string[] ModelPatterns,
        string[] MmprojPatterns,
        string[] ExcludePatterns = null);

    // ---- Issue #338: reasoning that would end without an answer ------------------------------------

    /// <summary>
    /// A stop drawn inside the reasoning closes it - Qwen's hand-over sentence, then &lt;/think&gt; -
    /// and the model answers, on a real model through the engine's own draw paths. A real stop
    /// mid-reasoning (3 of 53 Aider turns on the NVFP4 build) is not reproducible, so the policy's stop
    /// test also matches the reasoning's second full stop, a token the model is certain to draw.
    /// </summary>
    [ParallelModelFact("qwen38")]
    public async Task Qwen38_AStopInsideReasoning_ClosesIt_ThenTheModelAnswers()
    {
        using var ctx = Load("qwen38");
        var tokenizer = ctx.Model.Tokenizer;
        int stops = 0;
        SamplingConfig sampling = ReasoningPolicy(ctx, "What is 17 * 23? Think it through, then give the number.",
            null, out List<int> prompt, stop => t => stop(t) || (tokenizer.Decode(new List<int> { t }).Contains('.') && ++stops == 2));
        var (text, finish) = await Generate(ctx, prompt, sampling, 768);
        _output.WriteLine(text);
        int close = text.IndexOf("Considering the limited time", StringComparison.Ordinal);
        Assert.True(close >= 0, "the reasoning was not closed with the hand-over text");
        int end = text.IndexOf("</think>", close, StringComparison.Ordinal);
        Assert.True(end > close, "no </think> after the hand-over text");
        Assert.Contains("391", text.Substring(end));
        Assert.Equal("eos", finish);
    }

    /// <summary>
    /// Issue #338's loop on a real model: the reasoning is primed with the counting lines the issue
    /// logged (<c>Need maybe tests expect to_string for 1:1533 -> 02:33. Good.</c>, the number going up
    /// by one a line), the model carries the count on, and the run is closed at
    /// <see cref="ReasoningLoopDetector.MinLines"/> generated lines; the model then answers.
    /// </summary>
    [ParallelModelFact("qwen38")]
    public async Task Qwen38_ACountingLoopInTheReasoning_IsClosed_ThenTheModelAnswers()
    {
        using var ctx = Load("qwen38");
        var primed = new System.Text.StringBuilder("The clock normalizes minutes past the hour. Need check the tests.\n");
        for (int k = 0; k < 12; k++)
        {
            int total = 60 + 1500 + k;
            primed.Append($"Need maybe tests expect to_string for 1:{1500 + k} -> {total / 60 % 24:00}:{total % 60:00}. Good.\n");
        }
        SamplingConfig sampling = ReasoningPolicy(ctx,
            "A 24-hour clock is set to 1:00 and then 1533 minutes are added (it is written 1:1533). What time does it show? Answer as HH:MM.",
            primed.ToString(), out List<int> prompt, stop => stop);
        var (text, finish) = await Generate(ctx, prompt, sampling, 2048);
        _output.WriteLine(text);
        int close = text.IndexOf("Considering the limited time", StringComparison.Ordinal);
        Assert.True(close >= 0, "the counting loop was not closed");
        int counted = System.Text.RegularExpressions.Regex.Matches(text.Substring(0, close), @"to_string for 1:\d+ ->").Count;
        Assert.InRange(counted, ReasoningLoopDetector.MinLines - 2, ReasoningLoopDetector.MinLines + 2);
        string answer = text.Substring(text.IndexOf("</think>", close, StringComparison.Ordinal));
        Assert.Contains("02:33", answer);
        Assert.Equal("eos", finish);
    }

    private static SamplingConfig ReasoningPolicy(EngineContext ctx, string question, string primedReasoning,
        out List<int> prompt, Func<Func<int, bool>, Func<int, bool>> stopTest)
    {
        string arch = ctx.Model.Config?.Architecture ?? string.Empty;
        var history = new List<ChatMessage> { new() { Role = "user", Content = question } };
        prompt = ctx.Renderer.RenderToTokens(ctx.Model.Tokenizer, ctx.Model.Config?.ChatTemplate, history, arch,
            addGenerationPrompt: true, out _, out _, tools: null, enableThinking: true);
        // Reasoning the model is made to have written already: the generation prompt opened <think>.
        if (primedReasoning != null)
            prompt.AddRange(ctx.Model.Tokenizer.Encode(primedReasoning, addSpecial: false));
        // The server's policy for this request (no cap), with the stop test the caller wraps.
        SamplingConfig config = TensorSharp.Server.ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy,
            ctx.Model.Tokenizer, arch, 0, out bool installed, enableThinking: true, promptTokens: prompt);
        Assert.True(installed);
        var b = config.ThinkingBudget!;
        config.ThinkingBudget = new ThinkingTokenBudget(b.TokenLimit, b.EndTokenId, b.CloseOnRepetition, b.OpenTokenId,
            b.OpenAtStart, b.SuppressUnopenedEnd, b.CloseAtBoundary, b.ClosingTokenIds, stopTest(b.CloseOnStop!));
        return config;
    }

    private static async Task<(string Text, string Finish)> Generate(EngineContext ctx, List<int> prompt,
        SamplingConfig sampling, int maxNewTokens)
    {
        var seq = new SequenceState("reasoning-" + Guid.NewGuid().ToString("N"), prompt, maxNewTokens, ctx.BlockSize, sampling);
        var handle = ctx.Engine.SubmitRequest(seq);
        var output = await DrainHandle(handle);
        var completion = await handle.Completion;
        return (ctx.Model.Tokenizer.Decode(output), completion.FinishReason);
    }

    private sealed class RequestResult
    {
        public string RequestId;
        public int OutputTokenCount;
        public string OutputText;
        public int PrefixCacheReusedTokens;
        public int PromptTokenCount;
    }

    private sealed class EngineContext : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public string MmprojPath { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }
        public object MultimodalLock { get; } = new();

        /// <summary>The longest conversation these tests build (the long-turn reuse
        /// repros reach ~4.9k tokens).</summary>
        private const int MinimumContext = 8192;

        public EngineContext(string modelPath, string mmprojPath)
        {
            // A lane's small explicit MAX_CONTEXT (1024 in the model gates) would end
            // the long-turn repros with "exceeds configured max context"; raise it for
            // the load only, and leave an unset or larger value alone.
            using (var env = new EnvScope())
            {
                if (int.TryParse(Environment.GetEnvironmentVariable("MAX_CONTEXT"), out int configured)
                    && configured > 0 && configured < MinimumContext)
                    env.Set("MAX_CONTEXT", MinimumContext.ToString(System.Globalization.CultureInfo.InvariantCulture));
                Model = TensorSharp.Models.ModelBase.Create(modelPath, TestGates.PinnedGgmlBackend);
            }
            if (!string.IsNullOrEmpty(mmprojPath) && File.Exists(mmprojPath))
            {
                Model.MultimodalInjector.LoadProjectors(mmprojPath);
                MmprojPath = mmprojPath;
            }
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 2048,
                MaxNumRunningSequences = 8,
                MaxPrefillChunkSize = 512,
                NumBlocks = 64,
                BlockSize = BlockSize,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = BlockSize,
            };
            Engine = new InferenceEngine(Model, cfg, NullLogger.Instance);
        }

        public void Dispose()
        {
            Engine.Dispose();
            Model.Dispose();
        }
    }
}
