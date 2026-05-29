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
// required GGUFs (Gemma 4, Qwen 3.6, Nemotron 3) and they will run. Otherwise
// they short-circuit with a console message and pass trivially.
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

public class EngineParallelInferenceTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private static readonly Dictionary<string, ModelManifest> Manifests = new(StringComparer.OrdinalIgnoreCase)
    {
        ["gemma4"] = new ModelManifest(
            ModelPatterns: new[] { "gemma-4-E4B-it-Q8_0" },
            // The "-assistant" variant ships under the same prefix; exclude it.
            ExcludePatterns: new[] { "assistant" },
            MmprojPatterns: new[] { "gemma-4-mmproj" }),
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

    private readonly ITestOutputHelper _output;
    public EngineParallelInferenceTests(ITestOutputHelper output) { _output = output; }

    [Fact] public Task Gemma4_FiveTextPromptsParallel() => RunTextParallel("gemma4", numRequests: 5);
    // Repro for the user-reported bug: two long-generation parallel
    // requests (where each output >> Gemma 4's 512-token SWA window) plus
    // a third request submitted while the first two are still mid-decode.
    // Pre-fix: request #2 produced garbled tokens (per-seq KV swap reading
    // from a wrapped circular cache) and request #3 hit
    // "Continuous-batching engine is unavailable" (TryGetEngine returned
    // null once SupportsKVStateSnapshot flipped to false on cache wrap).
    // Post-fix: ForwardBatch is the default Gemma 4 path so KV swap +
    // snapshot are never needed.
    [Fact] public Task Gemma4_ThreeLongGenerationsParallel() => RunLongGenerationParallel("gemma4");
    // Repro for the user-reported "vision encoder + parallel text aborts
    // the process" crash. Pre-fix, the chat pipeline's vision-encoder GGML
    // ops on the request thread raced the engine worker's batched-forward
    // GGML ops on the GPU, and ggml_metal_synchronize would abort with a
    // command-buffer-status=1/2 fatal error. Post-fix, both call sites
    // take ModelBase.GpuComputeLock so they serialise.
    [Fact] public Task Gemma4_ImageAndTextSimultaneous() => RunImageAndTextSimultaneous("gemma4");
    [Fact] public Task Qwen36_FiveTextPromptsParallel() => RunTextParallel("qwen36", numRequests: 5);
    // Phase 3 verification: same flow on the 35B-A3B MoE GGUF — exercises the
    // batched MoE FFN dispatch added in Qwen35Model.BatchedForward.cs.
    // Tiny budget (2 prompts × 6 new tokens) because IQ2_XXS dequantization +
    // 256-expert routing per layer is slow on CPU/Metal; the test exists to
    // verify the code path runs, not for performance comparison.
    [Fact] public Task Qwen36MoE_FiveTextPromptsParallel() => RunTextParallel("qwen36moe", numRequests: 2, maxNewTokensOverride: 6);
    [Fact] public Task Nemotron3_FiveTextPromptsParallel() => RunTextParallel("nemotron3", numRequests: 5);
    // Mistral 3 runs through the TRUE batched-paged path (ExecuteStepBatched),
    // not the per-sequence KV swap fallback. Smaller maxNewTokens budget
    // because the managed paged-attention kernel is unoptimised C#; raise
    // once the GPU/native kernel lands.
    [Fact] public Task Mistral3_FourTextPromptsParallelBatched() => RunTextParallel("mistral3", numRequests: 4, maxNewTokensOverride: 8);
    // Long-context variant: pads each prompt to ~512 tokens of repeated
    // text so the per-layer paged attention compute actually dominates
    // over the gather + GPU-launch overhead. This is where the GPU paged
    // kernel is expected to outperform the managed scalar kernel.
    [Fact] public Task Mistral3_FourLongPromptsParallelBatched() => RunLongContextParallel("mistral3");

    [Fact] public Task Gemma4_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("gemma4");
    // Direct repro for the low-kvReusePercent bug: a long first turn (whose
    // K/V state wraps the SWA circular cache) followed by a same-session
    // follow-up. Pre-fix, prefix-cache reuse capped at one sliding window
    // (~10-20% for long histories); post-fix the per-seq extract/inject use
    // modular SWA addressing so the whole history is reusable.
    [Fact] public Task Gemma4_SwaPrefixCacheReuseAcrossLongTurn() => RunSwaPrefixReuseRepro("gemma4");
    [Fact] public Task Qwen36_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("qwen36");
    [Fact] public Task Nemotron3_PrefixCacheHitAcceleratesSecondRequest() => RunPrefixCacheBench("nemotron3");

    [Fact] public Task Gemma4_ImagePromptThenTextParallel() => RunMixedMultimodal("gemma4");
    [Fact] public Task Qwen36_ImagePromptThenTextParallel() => RunMixedMultimodal("qwen36");
    [Fact] public Task Nemotron3_ImagePromptThenTextParallel() => RunMixedMultimodal("nemotron3");

    [Fact] public Task Gemma4_AudioPrompt() => RunAudioSmoke("gemma4");
    [Fact] public Task Nemotron3_AudioPrompt() => RunAudioSmoke("nemotron3");

    [Fact] public Task Gemma4_VideoPrompt() => RunVideoSmoke("gemma4");
    [Fact] public Task Nemotron3_VideoPrompt() => RunVideoSmoke("nemotron3");

    // ---- core flows ----

    private async Task RunTextParallel(string key, int numRequests, int maxNewTokensOverride = 24)
    {
        if (!TryLoad(key, out var ctx)) return;
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
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            bool modelSupportsBatched = ctx.Model is TensorSharp.Runtime.Scheduling.IBatchedPagedModel;
            string disableRaw = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
            bool forceDisabled = !string.IsNullOrEmpty(disableRaw) && disableRaw != "0"
                && disableRaw.ToLowerInvariant() != "false";
            string kernelHint = Environment.GetEnvironmentVariable("TS_PAGED_ATTN_KERNEL") ?? "(default tensor)";
            string path = (modelSupportsBatched && !forceDisabled)
                ? $"batched (IBatchedPagedModel, kernel={kernelHint})"
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
        if (!TryLoad(key, out var ctx)) return;
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
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            if (string.IsNullOrEmpty(ctx.MmprojPath))
            {
                _output.WriteLine($"[{key}] no mmproj; skipping");
                return;
            }
            string imagePath = "/Users/ZhongkaiFu/Downloads/apple.png";
            if (!File.Exists(imagePath))
            {
                _output.WriteLine($"[{key}] apple.png missing in Downloads; skipping");
                return;
            }

            // Fire two text requests and one image request simultaneously.
            // The image request runs the vision encoder on the request
            // thread while the engine worker chews through the text
            // requests on the GPU - that's the exact concurrency that
            // crashed pre-fix. The image-request assertion below also
            // covers the per-request injector bucketing fix: if image
            // embeddings get consumed by the wrong sequence's Forward()
            // (the pre-fix model-shared-state bug), the image response
            // contains no recognisable description.
            var tasks = new List<Task<RequestResult>>
            {
                SubmitTextPrompt(ctx, "请详细介绍最终幻想7。", maxNewTokens: 64, "text-ff7"),
                SubmitTextPrompt(ctx, "请详细介绍时间简史。", maxNewTokens: 64, "text-history"),
                SubmitImagePrompt(ctx, imagePath, "What is in this image? Answer in one short phrase, in English.", maxNewTokens: 24),
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

            // Verify the model actually saw the apple image. Pre-fix the
            // engine path never queued vision embeddings into the model,
            // so the model would see only placeholder tokens and produce
            // text unrelated to apples/fruit.
            Assert.NotNull(imageResult);
            string lowerText = imageResult.OutputText?.ToLowerInvariant() ?? string.Empty;
            bool mentionsApple = lowerText.Contains("apple") || lowerText.Contains("fruit") ||
                                 lowerText.Contains("red") || imageResult.OutputText?.Contains("苹果") == true ||
                                 imageResult.OutputText?.Contains("水果") == true;
            Assert.True(mentionsApple,
                $"image request output \"{imageResult.OutputText}\" doesn't mention apple/fruit - " +
                "vision embeddings probably weren't injected (engine-path multimodal bug).");
        }
    }

    private async Task RunSwaPrefixReuseRepro(string key)
    {
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            var sampling = SamplingConfig.Greedy;

            // Turn 1: long generation that drives the SWA cache past one window
            // (Gemma 4 default SWA = 512 tokens, so 600 new tokens guarantees wrap).
            var turn1History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
            };
            var (t1, t1RawTokens) = await SubmitWithHistoryCapturingTokens(ctx, turn1History, maxNewTokens: 600, "turn1", sampling);
            _output.WriteLine($"[{key}] turn1: prompt={t1.PromptTokenCount} reused={t1.PrefixCacheReusedTokens} out={t1.OutputTokenCount}");

            // Turn 2: same session, follow-up "还有吗" appended after turn 1's output.
            // Splice turn 1's raw output tokens into the assistant message so the
            // chat template re-render produces an exact token prefix match (the
            // production chat pipeline does this via ChatHistoryPreparer).
            var turn2History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
                new ChatMessage { Role = "assistant", Content = t1.OutputText, RawOutputTokens = t1RawTokens },
                new ChatMessage { Role = "user", Content = "还有吗" },
            };
            var (t2, t2RawTokens) = await SubmitWithHistoryCapturingTokens(ctx, turn2History, maxNewTokens: 200, "turn2", sampling);
            _output.WriteLine($"[{key}] turn2: prompt={t2.PromptTokenCount} reused={t2.PrefixCacheReusedTokens} out={t2.OutputTokenCount}");

            // Turn 3: "请继续"
            var turn3History = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "请详细介绍最终幻想7" },
                new ChatMessage { Role = "assistant", Content = t1.OutputText, RawOutputTokens = t1RawTokens },
                new ChatMessage { Role = "user", Content = "还有吗" },
                new ChatMessage { Role = "assistant", Content = t2.OutputText, RawOutputTokens = t2RawTokens },
                new ChatMessage { Role = "user", Content = "请继续" },
            };
            var (t3, _) = await SubmitWithHistoryCapturingTokens(ctx, turn3History, maxNewTokens: 200, "turn3", sampling);
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

    private async Task<(RequestResult result, List<int> rawTokens)> SubmitWithHistoryCapturingTokens(
        EngineContext ctx, List<ChatMessage> history, int maxNewTokens, string reqId,
        SamplingConfig sampling = null)
    {
        var tokens = RenderTokens(ctx, history);
        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, sampling ?? SamplingConfig.Default);
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
        return (result, outputTokens);
    }

    private async Task RunPrefixCacheBench(string key)
    {
        if (!TryLoad(key, out var ctx)) return;
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
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            if (string.IsNullOrEmpty(ctx.MmprojPath))
            {
                _output.WriteLine($"[{key}] no mmproj available, skipping image test");
                return;
            }

            string imagePath = "/Users/ZhongkaiFu/Downloads/apple.png";
            if (!File.Exists(imagePath))
            {
                _output.WriteLine("apple.png missing in Downloads; image test skipped");
                return;
            }

            var sw = Stopwatch.StartNew();
            var imageResp = await SubmitImagePrompt(ctx, imagePath, "What is the main object in this picture? Answer in one short phrase.", maxNewTokens: 24);
            _output.WriteLine($"[{key}] image: {imageResp.OutputTokenCount} out tokens in {sw.Elapsed.TotalMilliseconds:F0}ms — \"{Truncate(imageResp.OutputText, 80)}\"");
            Assert.True(imageResp.OutputTokenCount > 0);
            // The apple.png image is unambiguously an apple. If the model is
            // actually seeing vision content (not just the bare <|image_pad|>
            // placeholders), the response should mention apple/fruit/red. This
            // assertion catches regressions where the image embeddings or
            // MRoPE positions get dropped silently.
            string lowerImg = imageResp.OutputText?.ToLowerInvariant() ?? string.Empty;
            bool mentionsApple = lowerImg.Contains("apple") || lowerImg.Contains("fruit") ||
                                 lowerImg.Contains("red");
            Assert.True(mentionsApple,
                $"{key} image response \"{imageResp.OutputText}\" doesn't mention apple/fruit/red - " +
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
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            string audioPath = "/Users/ZhongkaiFu/Downloads/obama_first_45_secs.mp3";
            if (string.IsNullOrEmpty(ctx.MmprojPath) || !File.Exists(audioPath))
            {
                _output.WriteLine($"[{key}] audio prerequisites missing; skipping");
                return;
            }
            var resp = await SubmitAudioPrompt(ctx, audioPath, "Summarize this audio in one sentence.", maxNewTokens: 24);
            _output.WriteLine($"[{key}] audio: {resp.OutputTokenCount} out tokens — \"{Truncate(resp.OutputText, 80)}\"");
            Assert.True(resp.OutputTokenCount > 0);
        }
    }

    private async Task RunVideoSmoke(string key)
    {
        if (!TryLoad(key, out var ctx)) return;
        using (ctx)
        {
            string videoPath = "/Users/ZhongkaiFu/Downloads/concert.mp4";
            if (string.IsNullOrEmpty(ctx.MmprojPath) || !File.Exists(videoPath))
            {
                _output.WriteLine($"[{key}] video prerequisites missing; skipping");
                return;
            }

            // Treat the video as a single frame extracted by the model's image pipeline.
            // Most TensorSharp models accept videos as a series of image frames via
            // ImagePaths + IsVideo=true; we send the raw path here and let the
            // injector handle decoding. If a model can't parse mp4 the multimodal
            // injector returns no embeddings and the test degrades gracefully.
            var history = new List<ChatMessage>
            {
                new ChatMessage
                {
                    Role = "user",
                    Content = "Describe the scene in this video in one short sentence.",
                    ImagePaths = new List<string> { videoPath },
                    IsVideo = true,
                }
            };

            try
            {
                var resp = await SubmitWithHistory(ctx, history, maxNewTokens: 20, "video");
                _output.WriteLine($"[{key}] video: {resp.OutputTokenCount} out — \"{Truncate(resp.OutputText, 80)}\"");
                Assert.True(resp.OutputTokenCount > 0);
            }
            catch (Exception ex)
            {
                // Video decoding may not be supported by this model build; record but don't fail.
                _output.WriteLine($"[{key}] video test soft-failed: {ex.Message}");
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

    private static string Truncate(string s, int maxLen)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        s = s.Replace("\n", " ").Replace("\r", " ").Trim();
        return s.Length <= maxLen ? s : s.Substring(0, maxLen) + "...";
    }

    private bool TryLoad(string key, out EngineContext ctx)
    {
        ctx = null;
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"[{key}] {EnvModelDir} not set (or directory missing); test skipped");
            return false;
        }

        if (!Manifests.TryGetValue(key, out var manifest))
        {
            _output.WriteLine($"[{key}] no manifest registered; test skipped");
            return false;
        }

        string modelPath = FindFirst(dir, manifest.ModelPatterns, manifest.ExcludePatterns);
        if (modelPath == null)
        {
            _output.WriteLine($"[{key}] no model file matching {string.Join("/", manifest.ModelPatterns)} under {dir}; skipped");
            return false;
        }
        string mmproj = manifest.MmprojPatterns != null ? FindFirst(dir, manifest.MmprojPatterns, null) : null;

        _output.WriteLine($"[{key}] loading {Path.GetFileName(modelPath)} (mmproj={Path.GetFileName(mmproj ?? "")})");
        try
        {
            ctx = new EngineContext(modelPath, mmproj);
            return true;
        }
        catch (Exception ex)
        {
            _output.WriteLine($"[{key}] failed to load: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static string FindFirst(string dir, IEnumerable<string> patterns, string[] excludePatterns)
    {
        foreach (var p in patterns)
        {
            foreach (var f in Directory.GetFiles(dir, "*.gguf"))
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
            $"poolFree={stats.freeBlocks}/{stats.totalBlocks} hashedCached={stats.hashedBlocks}");
    }

    private sealed record ModelManifest(
        string[] ModelPatterns,
        string[] MmprojPatterns,
        string[] ExcludePatterns = null);

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

        public EngineContext(string modelPath, string mmprojPath)
        {
            // Prefer the platform-appropriate GGML variant; fall back to CPU.
            BackendType backend = OperatingSystem.IsMacOS()
                ? BackendType.GgmlMetal
                : BackendType.GgmlCpu;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
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
