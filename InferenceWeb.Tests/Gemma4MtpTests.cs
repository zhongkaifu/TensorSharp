// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 4 MTP (gemma4-assistant draft head) speculative decoding: correctness +
// performance. The draft head ships as a SEPARATE GGUF and is attached to the
// target via Gemma4Model.LoadMtpDraftWeights().
//
// Greedy speculative decoding is verification-gated, so the output stream must
// match plain greedy decoding except where batched-vs-sequential kernel ordering
// flips a near-tie argmax. A high acceptance rate is the signal that the draft
// head itself is wired correctly (a broken draft still yields correct output via
// verification, just with ~0 acceptance and no speedup).
//
// Opt-in via TS_GMTP_E2E=1. Model resolution:
//   TS_GMTP_TARGET / TS_GMTP_DRAFT  — explicit .gguf paths
//   TS_GMTP_DIR                     — directory holding both GGUFs
//   C:\Works\models\gemma_mtp       — default
// Backend via TS_GMTP_BACKEND: ggml_cpu (default) | ggml_cuda | cpu | cuda.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4MtpTests
{
    private readonly ITestOutputHelper _output;
    public Gemma4MtpTests(ITestOutputHelper output) { _output = output; }

    private const string DefaultModelDir = @"C:\Works\models\gemma_mtp";

    [Fact]
    public void Gemma4Mtp_SpeculativeGreedy_MatchesBaselineGreedy()
    {
        if (!TryResolveModels(out string targetPath, out string draftPath))
        { _output.WriteLine("[gmtp] opt-in not set or models missing; skipping"); return; }

        int maxNew = EnvInt("TS_GMTP_NEW_TOKENS", 32);
        int maxDraft = EnvInt("TS_GMTP_DRAFT_N", 4);

        _output.WriteLine($"[gmtp] target={Path.GetFileName(targetPath)} draft={Path.GetFileName(draftPath)} backend={ResolveBackend()}");
        using var model = (Gemma4Model)ModelBase.Create(targetPath, ResolveBackend());
        model.LoadMtpDraftWeights(draftPath);
        Xunit.Assert.True(model.HasMtp, "model should expose a Gemma 4 MTP draft head");

        string prompt = "The capital of France is Paris. The capital of Japan is Tokyo. " +
                        "The capital of Italy is";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: true).ToArray();
        _output.WriteLine($"[gmtp] prompt tokens: {tokens.Length}");

        // Baseline plain greedy.
        model.ResetKVCache();
        var baseline = new List<int>();
        float[] logits = model.ForwardRefill(tokens);
        int t = Argmax(logits);
        baseline.Add(t);
        for (int i = 1; i < maxNew; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            baseline.Add(t);
        }

        // Speculative greedy.
        var spec = new MtpSpeculativeDecoder(model, maxDraft);
        List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);

        _output.WriteLine($"[gmtp] baseline: \"{Trim(model.Tokenizer.Decode(baseline))}\"");
        _output.WriteLine($"[gmtp] spec:     \"{Trim(model.Tokenizer.Decode(specTokens))}\"");
        _output.WriteLine($"[gmtp] baseline tokens: [{string.Join(",", baseline)}]");
        _output.WriteLine($"[gmtp] spec tokens:     [{string.Join(",", specTokens)}]");
        _output.WriteLine($"[gmtp] drafted={spec.TokensDrafted} accepted={spec.TokensAccepted} " +
            $"({spec.AcceptanceRate:P0}), verifySteps={spec.VerifySteps}, plainSteps={spec.PlainSteps}, " +
            $"rollbacks={spec.RollbackSteps}");
        double verifyPer = spec.Stats.VerifyMs / Math.Max(1, spec.VerifySteps);
        _output.WriteLine($"[gmtp] spec decode {spec.LastDecodeSeconds:F2}s = {(specTokens.Count - 1) / Math.Max(0.001, spec.LastDecodeSeconds):F1} tok/s | " +
            $"verify {spec.Stats.VerifyMs:F0}ms (~{verifyPer:F0} ms/step), draft {spec.Stats.DraftMs:F0}ms, plain {spec.Stats.PlainMs:F0}ms");

        int compareLen = Math.Min(baseline.Count, specTokens.Count);
        int matchPrefix = 0;
        while (matchPrefix < compareLen && baseline[matchPrefix] == specTokens[matchPrefix])
            matchPrefix++;
        _output.WriteLine($"[gmtp] prefix match {matchPrefix}/{compareLen}");

        Xunit.Assert.Equal(maxNew, specTokens.Count);
        Xunit.Assert.True(matchPrefix >= compareLen / 2,
            $"spec/baseline prefix match {matchPrefix}/{compareLen} below 50% — structural divergence suspected");
        Xunit.Assert.True(spec.TokensDrafted > 0, "draft head never produced a candidate");
        // A correctly-wired draft head agrees with the target on most tokens of a
        // simple factual continuation; ~0 acceptance means the draft is broken.
        Xunit.Assert.True(spec.AcceptanceRate > 0.30,
            $"draft acceptance {spec.AcceptanceRate:P0} too low — draft head likely mis-wired");
    }

    // Long-prompt prefill OOM repro / fix-validation on the STANDALONE linear-cache
    // path (no paged-block ceiling, so the whole long prompt actually prefills).
    // With MTP a long prompt used to route its prefill through the fused verify
    // kernel, which materializes a [totalSeqLen, seqLen, numHeads] global-attention
    // score tensor (tens of GB) and CUDA-OOMs ("allocating 23104.00 MiB ... out of
    // memory"). The fix routes the prefill through the same fast chunked flash path
    // ForwardRefill uses. This test prefills + generates with MTP and asserts it
    // completes (no OOM), that prefill time matches the non-MTP baseline (the fix is
    // high-performance, not just a slow fallback), and that the greedy output tracks
    // the baseline for a healthy prefix.
    //   TS_GMTP_E2E=1 TS_GMTP_LONGPROMPT=1 TS_GMTP_BACKEND=ggml_cuda
    //   TS_GMTP_PROMPT_FILE=<long.txt> [TS_GMTP_NEW_TOKENS=48] [TS_GMTP_DRAFT_N=6]
    [Fact]
    public void Gemma4Mtp_LongPromptPrefill_NoOom()
    {
        if (!TryResolveModels(out string targetPath, out string draftPath) ||
            Environment.GetEnvironmentVariable("TS_GMTP_LONGPROMPT") != "1")
        { _output.WriteLine("[gmtp-long] set TS_GMTP_LONGPROMPT=1 (and TS_GMTP_E2E=1) to run; skipping"); return; }

        int maxNew = EnvInt("TS_GMTP_NEW_TOKENS", 48);
        int maxDraft = EnvInt("TS_GMTP_DRAFT_N", 6);

        _output.WriteLine($"[gmtp-long] target={Path.GetFileName(targetPath)} backend={ResolveBackend()}");
        using var model = (Gemma4Model)ModelBase.Create(targetPath, ResolveBackend());
        model.LoadMtpDraftWeights(draftPath);
        Xunit.Assert.True(model.HasMtp);

        string prompt = Environment.GetEnvironmentVariable("TS_GMTP_PROMPT");
        string promptFile = Environment.GetEnvironmentVariable("TS_GMTP_PROMPT_FILE");
        if (string.IsNullOrEmpty(prompt) && !string.IsNullOrEmpty(promptFile) && File.Exists(promptFile))
            prompt = File.ReadAllText(promptFile);
        Xunit.Assert.False(string.IsNullOrEmpty(prompt), "set TS_GMTP_PROMPT_FILE to a long prompt");
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: true).ToArray();
        _output.WriteLine($"[gmtp-long] prompt={tokens.Length} tok, gen={maxNew}, draft window={maxDraft}");

        // Baseline: the non-MTP path the user runs with MTP disabled (chunked flash
        // ForwardRefill prefill + fused single-token decode). This always worked.
        model.ResetKVCache();
        var swBasePrefill = Stopwatch.StartNew();
        float[] logits = model.ForwardRefill(tokens);
        swBasePrefill.Stop();
        var baseline = new List<int>();
        int tb = Argmax(logits);
        baseline.Add(tb);
        for (int i = 1; i < maxNew; i++) { logits = model.Forward(new[] { tb }); tb = Argmax(logits); baseline.Add(tb); }
        _output.WriteLine($"[gmtp-long] baseline (MTP off): prefill {swBasePrefill.Elapsed.TotalSeconds:F2}s, " +
            $"first tokens [{string.Join(",", baseline.Take(8))}]");

        // Spec: MTP through the standalone linear-cache decoder. Pre-fix this OOM'd
        // during the prompt prefill; post-fix it must complete. Match the prefill
        // chunk size to the baseline's ForwardRefill chunk (the server's default,
        // 1024) so the prefill-time comparison is apples-to-apples.
        var spec = new MtpSpeculativeDecoder(model, maxDraft)
        {
            PrefillChunkSize = EnvInt("TS_GMTP_PREFILL_CHUNK", 1024),
        };
        List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);
        _output.WriteLine($"[gmtp-long] spec   (MTP on):  prefill {spec.LastPrefillSeconds:F2}s, decode " +
            $"{spec.LastDecodeSeconds:F2}s, acceptance {spec.AcceptanceRate:P0} ({spec.TokensAccepted}/{spec.TokensDrafted})");
        _output.WriteLine($"[gmtp-long] spec first tokens [{string.Join(",", specTokens.Take(8))}]");

        int cmp = Math.Min(baseline.Count, specTokens.Count);
        int match = 0;
        while (match < cmp && baseline[match] == specTokens[match]) match++;
        _output.WriteLine($"[gmtp-long] baseline/spec prefix match {match}/{cmp}");
        double prefillRatio = spec.LastPrefillSeconds / Math.Max(1e-6, swBasePrefill.Elapsed.TotalSeconds);
        _output.WriteLine($"[gmtp-long] prefill time ratio spec/baseline {prefillRatio:F2}x");

        // The whole point: MTP must speculate (drafted>0) and complete without OOM.
        Xunit.Assert.True(spec.TokensDrafted > 0, "spec never drafted — MTP not engaged");
        Xunit.Assert.True(specTokens.Count >= Math.Min(8, maxNew), "spec generation did not progress");
        // The fix reuses the SAME prefill machinery as the baseline, so the first
        // generated token (identical prefill + identical fused last-token decode)
        // must match exactly; later tokens may diverge on verify-kernel FP near-ties.
        Xunit.Assert.True(match >= 1, $"first token diverged (baseline {baseline[0]} vs spec {specTokens[0]})");
        // Prefill must be in the same ballpark as the non-MTP path (not a 3x-slower
        // per-op fallback). Generous bound to absorb warmup / measurement noise.
        Xunit.Assert.True(prefillRatio < 2.0,
            $"MTP prefill {spec.LastPrefillSeconds:F2}s vs baseline {swBasePrefill.Elapsed.TotalSeconds:F2}s — too slow");
    }

    [Fact]
    public void Gemma4Mtp_PerfBench_SpecVsBaseline()
    {
        if (!TryResolveModels(out string targetPath, out string draftPath) ||
            Environment.GetEnvironmentVariable("TS_GMTP_BENCH") != "1")
        { _output.WriteLine("[gmtp-bench] opt-in not set; skipping"); return; }

        int maxNew = EnvInt("TS_GMTP_NEW_TOKENS", 64);
        int maxDraft = EnvInt("TS_GMTP_DRAFT_N", 4);

        _output.WriteLine($"[gmtp-bench] target={Path.GetFileName(targetPath)} backend={ResolveBackend()}");
        using var model = (Gemma4Model)ModelBase.Create(targetPath, ResolveBackend());
        model.LoadMtpDraftWeights(draftPath);
        Xunit.Assert.True(model.HasMtp);

        string prompt = "Write a short story about a robot learning to paint. " +
                        "Once upon a time, in a small workshop at the edge of the city,";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: true).ToArray();

        // Warmup.
        model.ResetKVCache();
        model.ForwardRefill(tokens);
        model.Forward(new[] { tokens[^1] });

        // Baseline greedy.
        model.ResetKVCache();
        var baseline = new List<int>();
        var swPrefill = Stopwatch.StartNew();
        float[] logits = model.ForwardRefill(tokens);
        swPrefill.Stop();
        var swDecode = Stopwatch.StartNew();
        int t = Argmax(logits);
        baseline.Add(t);
        for (int i = 1; i < maxNew; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            baseline.Add(t);
        }
        swDecode.Stop();
        double baseTps = (maxNew - 1) / swDecode.Elapsed.TotalSeconds;

        // Per-op baseline (TS_GEMMA4_FORCE_UNFUSED): the trunk's per-token path
        // WITHOUT the fused single-token decode kernel — the same kernel family the
        // multi-token verify is forced onto (Gemma 4 has no fused multi-token
        // forward). Comparing spec to this isolates the speculative algorithm's
        // contribution from the fused-decode advantage the production decode enjoys.
        Environment.SetEnvironmentVariable("TS_GEMMA4_FORCE_UNFUSED", "1");
        model.ResetKVCache();
        logits = model.ForwardRefill(tokens);
        var swUnfused = Stopwatch.StartNew();
        t = Argmax(logits);
        for (int i = 1; i < maxNew; i++) { logits = model.Forward(new[] { t }); t = Argmax(logits); }
        swUnfused.Stop();
        Environment.SetEnvironmentVariable("TS_GEMMA4_FORCE_UNFUSED", null);
        double baseUnfusedTps = (maxNew - 1) / swUnfused.Elapsed.TotalSeconds;

        // Speculative greedy.
        var spec = new MtpSpeculativeDecoder(model, maxDraft);
        string pminEnv = Environment.GetEnvironmentVariable("TS_GMTP_PMIN");
        if (!string.IsNullOrEmpty(pminEnv) && float.TryParse(pminEnv, out float pmin))
            spec.MinDraftProb = pmin;
        List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);
        double specTps = (specTokens.Count - 1) / spec.LastDecodeSeconds;

        _output.WriteLine($"[gmtp-bench] prompt={tokens.Length} tok, gen={maxNew} tok, draft window={maxDraft}");
        _output.WriteLine($"[gmtp-bench] baseline (fused decode): decode {swDecode.Elapsed.TotalSeconds:F2}s = {baseTps:F2} tok/s");
        _output.WriteLine($"[gmtp-bench] baseline (per-op decode):  {baseUnfusedTps:F2} tok/s");
        _output.WriteLine($"[gmtp-bench] spec:     decode {spec.LastDecodeSeconds:F2}s = {specTps:F2} tok/s");
        _output.WriteLine($"[gmtp-bench] speedup vs fused decode:  {specTps / baseTps:F2}x  (production single-seq baseline)");
        _output.WriteLine($"[gmtp-bench] speedup vs per-op decode: {specTps / baseUnfusedTps:F2}x  (isolates the speculative algorithm)");
        _output.WriteLine($"[gmtp-bench] acceptance {spec.AcceptanceRate:P0} " +
            $"({spec.TokensAccepted}/{spec.TokensDrafted}), verifySteps={spec.VerifySteps}, " +
            $"plainSteps={spec.PlainSteps}, rollbacks={spec.RollbackSteps}");
        var st = spec.Stats;
        _output.WriteLine($"[gmtp-bench] phase ms: draft={st.DraftMs:F0} verify={st.VerifyMs:F0} " +
            $"plain={st.PlainMs:F0} rollback={st.RollbackMs:F0} catchup={st.CatchUpMs:F0} snapshot={st.SnapshotMs:F0}");
        _output.WriteLine($"[gmtp-bench] baseline: \"{Trim(model.Tokenizer.Decode(baseline))}\"");
        _output.WriteLine($"[gmtp-bench] spec:     \"{Trim(model.Tokenizer.Decode(specTokens))}\"");
    }

    [Fact]
    public void Gemma4Mtp_EngineBench_FusedTrunk()
    {
        // The PRODUCTION path: MTP speculation through the engine. A solo sequence
        // runs the linear trunk driving the fused single-graph verify + draft
        // kernels (NativeGemma4ModelVerify / NativeGemma4DraftStep). MTP-OFF is the
        // same engine without speculation (the production single-seq baseline).
        if (!TryResolveModels(out string targetPath, out string draftPath) ||
            Environment.GetEnvironmentVariable("TS_GMTP_ENGINE") != "1")
        { _output.WriteLine("[gmtp-engine] set TS_GMTP_ENGINE=1 to run; skipping"); return; }

        int maxNew = EnvInt("TS_GMTP_NEW_TOKENS", 96);
        int maxDraft = EnvInt("TS_GMTP_DRAFT_N", 6);

        _output.WriteLine($"[gmtp-engine] target={Path.GetFileName(targetPath)} backend={ResolveBackend()}");
        using var model = (Gemma4Model)ModelBase.Create(targetPath, ResolveBackend());
        model.LoadMtpDraftWeights(draftPath);
        Xunit.Assert.True(model.HasMtp);

        string prompt = Environment.GetEnvironmentVariable("TS_GMTP_PROMPT");
        string promptFile = Environment.GetEnvironmentVariable("TS_GMTP_PROMPT_FILE");
        if (string.IsNullOrEmpty(prompt) && !string.IsNullOrEmpty(promptFile) && File.Exists(promptFile))
            prompt = File.ReadAllText(promptFile);
        if (string.IsNullOrEmpty(prompt))
            prompt = "The capital of France is Paris. The capital of Japan is Tokyo. " +
                        "The capital of Italy is";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: true).ToArray();

        // Match the server chat defaults (temp 0.8, repPen 1.1) when requested, so
        // acceptance/throughput reflect a real interactive session rather than greedy.
        var greedy = Environment.GetEnvironmentVariable("TS_GMTP_CHAT") == "1"
            ? new SamplingConfig { Temperature = 0.8f, TopK = 40, TopP = 0.95f, MinP = 0f, RepetitionPenalty = 1.1f }
            : new SamplingConfig { Temperature = 0f, TopK = 0, TopP = 1.0f, MinP = 0f, RepetitionPenalty = 1.0f };

        (double tps, List<int> outTokens, double accept, long drafted) Run(bool mtp)
        {
            var cfg = new SchedulerConfig
            {
                MtpSpeculativeEnabled = mtp,
                MtpMaxDraftTokens = maxDraft,
                MtpMinDraftProb = EnvFloat("TS_GMTP_PMIN", 0.6f),
                NumBlocks = 64,
                BlockSize = 256,
                MaxNumBatchedTokens = 1024,
                MaxNumRunningSequences = 1,
                MaxPrefillChunkSize = 512,
                // Mirror the server (prefix caching on) when requested so the
                // bench exercises the real finish path — a speculative batch
                // that hits EOS mid-window leaves committed > total, which the
                // scheduler's prefix-cache block registration must tolerate.
                EnablePrefixCaching = Environment.GetEnvironmentVariable("TS_GMTP_PREFIX_CACHE") == "1",
                DecodeQuantumTokens = 1,
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            // Warmup request (kernel compile, paged-buffer alloc).
            var warm = new SequenceState("warm", tokens, 4, cfg.BlockSize, greedy);
            engine.SubmitRequest(warm).Completion.GetAwaiter().GetResult();

            var seq = new SequenceState("bench", tokens, maxNew, cfg.BlockSize, greedy);
            var sw = Stopwatch.StartNew();
            var completion = engine.SubmitRequest(seq).Completion.GetAwaiter().GetResult();
            sw.Stop();
            double secs = sw.Elapsed.TotalSeconds;
            var stats = seq.SpecStats;
            if (mtp && stats != null)
                _output.WriteLine($"[gmtp-engine] phase ms: draft={stats.DraftMs:F0} verify={stats.VerifyMs:F0} " +
                    $"plain={stats.PlainMs:F0} rollback={stats.RollbackMs:F0} | verifySteps={stats.VerifySteps} plainSteps={stats.PlainSteps}");
            return (completion.OutputTokenCount / secs, new List<int>(seq.OutputTokens),
                    stats?.AcceptanceRate ?? 0, stats?.TokensDrafted ?? 0);
        }

        var off = Run(false);
        var on = Run(true);

        _output.WriteLine($"[gmtp-engine] prompt={tokens.Length} tok, gen={maxNew} tok, draft window={maxDraft}");
        _output.WriteLine($"[gmtp-engine] MTP off: {off.tps:F2} tok/s ({off.outTokens.Count} tokens)");
        _output.WriteLine($"[gmtp-engine] MTP on:  {on.tps:F2} tok/s ({on.outTokens.Count} tokens), " +
            $"acceptance {on.accept:P0}, drafted {on.drafted}");
        _output.WriteLine($"[gmtp-engine] speedup: {on.tps / off.tps:F2}x");
        _output.WriteLine($"[gmtp-engine] off: \"{Trim(model.Tokenizer.Decode(off.outTokens))}\"");
        _output.WriteLine($"[gmtp-engine] on:  \"{Trim(model.Tokenizer.Decode(on.outTokens))}\"");

        // On a backend that can drive the accelerated MTP path, speculation must
        // engage (drafted>0). On a backend that cannot (e.g. --backend cuda, the
        // pure-C# CUDA path with no fused multi-token kernels), MtpSpeculationProfitable
        // is false: the engine intentionally serves standard decode (drafted==0) so
        // speculation never regresses throughput below MTP-off.
        if (model.MtpSpeculationProfitable)
        {
            Xunit.Assert.True(on.drafted > 0, "spec trunk never drafted — MTP not engaged through the engine");
        }
        else
        {
            Xunit.Assert.True(on.drafted == 0,
                $"MTP should be gated off on this backend but drafted {on.drafted}");
            Xunit.Assert.True(on.tps > off.tps * 0.9,
                $"gated MTP-on must not regress vs MTP-off (on {on.tps:F1} < off {off.tps:F1} tok/s)");
        }

        int cmp = Math.Min(off.outTokens.Count, on.outTokens.Count);
        int match = 0;
        while (match < cmp && off.outTokens[match] == on.outTokens[match]) match++;
        _output.WriteLine($"[gmtp-engine] off/on prefix match {match}/{cmp}");
    }

    // Long-generation Metal-OOM repro / fix-validation. Drives the production
    // engine spec path for a large token count while sampling device-copy cache
    // residency + process footprint, so we can see whether unified memory grows
    // cumulatively (leak) or is flat (pressure) and confirm the run completes
    // without kIOGPUCommandBufferCallbackErrorOutOfMemory. Opt-in:
    //   TS_GMTP_E2E=1 TS_GMTP_OOM=1 TS_GMTP_BACKEND=ggml_metal
    //   [TS_GMTP_NEW_TOKENS=1500] [TS_GMTP_PROMPT=...] [TS_GMTP_DRAFT_N=8]
    [Fact]
    public void Gemma4Mtp_LongGen_NoOom()
    {
        if (!TryResolveModels(out string targetPath, out string draftPath) ||
            Environment.GetEnvironmentVariable("TS_GMTP_OOM") != "1")
        { _output.WriteLine("[gmtp-oom] set TS_GMTP_OOM=1 (and TS_GMTP_E2E=1) to run; skipping"); return; }

        int maxNew = EnvInt("TS_GMTP_NEW_TOKENS", 1500);
        int maxDraft = EnvInt("TS_GMTP_DRAFT_N", 8);

        _output.WriteLine($"[gmtp-oom] target={Path.GetFileName(targetPath)} backend={ResolveBackend()} maxNew={maxNew} maxDraft={maxDraft}");
        using var model = (Gemma4Model)ModelBase.Create(targetPath, ResolveBackend());
        model.LoadMtpDraftWeights(draftPath);
        Xunit.Assert.True(model.HasMtp);

        // Faithful repro: the reported OOM ran with --mmproj loaded, which holds
        // ~1.2 GB resident even for text-only chat. Load it when TS_GMTP_MMPROJ
        // points at the projector GGUF so the headroom matches the server.
        string mmproj = Environment.GetEnvironmentVariable("TS_GMTP_MMPROJ");
        if (!string.IsNullOrEmpty(mmproj) && File.Exists(mmproj))
        {
            model.MultimodalInjector.LoadProjectors(mmproj);
            _output.WriteLine($"[gmtp-oom] loaded mmproj {Path.GetFileName(mmproj)}");
        }

        string prompt = Environment.GetEnvironmentVariable("TS_GMTP_PROMPT");
        if (string.IsNullOrEmpty(prompt))
            prompt = "请详细介绍最终幻想7";
        // Render through the GGUF chat template exactly like the server, so the
        // workload mirrors the report (a raw prompt derails MTP differently).
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            model.Config.ChatTemplate,
            new List<ChatMessage> { new ChatMessage { Role = "user", Content = prompt } },
            addGenerationPrompt: true, architecture: model.Config.Architecture);
        int[] tokens = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
        _output.WriteLine($"[gmtp-oom] prompt tokens={tokens.Length} (chat-templated)");

        // Match the server chat defaults so the workload mirrors the report.
        // TS_GMTP_GREEDY=1 forces deterministic argmax for a stable coherence check.
        var sampling = Environment.GetEnvironmentVariable("TS_GMTP_GREEDY") == "1"
            ? new SamplingConfig { Temperature = 0f, TopK = 0, TopP = 1.0f, MinP = 0f, RepetitionPenalty = 1.0f }
            : new SamplingConfig { Temperature = 0.8f, TopK = 40, TopP = 0.95f, MinP = 0f, RepetitionPenalty = 1.1f };

        var cfg = new SchedulerConfig
        {
            MtpSpeculativeEnabled = true,
            MtpMaxDraftTokens = maxDraft,
            MtpMinDraftProb = EnvFloat("TS_GMTP_PMIN", 0.75f),
            NumBlocks = 64,
            BlockSize = 256,
            MaxNumBatchedTokens = 1024,
            MaxNumRunningSequences = 1,
            MaxPrefillChunkSize = 512,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        long Mb(long b) => b / 1024 / 1024;
        void Snap(string tag, int produced)
        {
            long dev = GgmlBasicOps.DeviceCopyCacheResidentBytes();
            long ws = Process.GetCurrentProcess().WorkingSet64;
            string metal = GgmlBasicOps.TryGetBackendMemory(out long free, out long total)
                ? $" metalResident={Mb(total - free)}MB/{Mb(total)}MB" : "";
            _output.WriteLine($"[gmtp-oom] {tag} produced={produced} deviceCopyResident={Mb(dev)}MB workingSet={Mb(ws)}MB gc={Mb(GC.GetTotalMemory(false))}MB{metal}");
        }

        var seq = new SequenceState("oom", tokens, maxNew, cfg.BlockSize, sampling);
        var handle = engine.SubmitRequest(seq);

        var stop = new ManualResetEventSlim(false);
        var sampler = new Thread(() =>
        {
            int last = -1;
            while (!stop.Wait(500))
            {
                int n = seq.OutputTokens.Count;
                if (n != last) { Snap("progress", n); last = n; }
            }
        }) { IsBackground = true };
        sampler.Start();

        var sw = Stopwatch.StartNew();
        Exception failure = null;
        int outCount = 0;
        try
        {
            var completion = handle.Completion.GetAwaiter().GetResult();
            outCount = completion.OutputTokenCount;
        }
        catch (Exception ex) { failure = ex; }
        sw.Stop();
        stop.Set(); sampler.Join();

        Snap("final", seq.OutputTokens.Count);
        var stats = seq.SpecStats;
        if (stats != null)
            _output.WriteLine($"[gmtp-oom] phase ms: draft={stats.DraftMs:F0} verify={stats.VerifyMs:F0} plain={stats.PlainMs:F0} rollback={stats.RollbackMs:F0} | verifySteps={stats.VerifySteps} plainSteps={stats.PlainSteps} rollbacks={stats.RollbackSteps} acceptance={stats.AcceptanceRate:P0} drafted={stats.TokensDrafted}");
        _output.WriteLine($"[gmtp-oom] produced {seq.OutputTokens.Count} tokens in {sw.Elapsed.TotalSeconds:F1}s = {seq.OutputTokens.Count / Math.Max(0.001, sw.Elapsed.TotalSeconds):F2} tok/s");
        _output.WriteLine($"[gmtp-oom] text: \"{Trim(model.Tokenizer.Decode(seq.OutputTokens), 400)}\"");

        if (failure != null)
        {
            _output.WriteLine($"[gmtp-oom] FAILED: {failure.GetType().Name}: {failure.Message}");
            throw failure;
        }
        Xunit.Assert.True(seq.OutputTokens.Count >= Math.Min(maxNew, 64),
            $"expected to generate the requested tokens without OOM, got {seq.OutputTokens.Count}");
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    private static int EnvInt(string name, int fallback)
    {
        string s = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
    }

    private static float EnvFloat(string name, float fallback)
    {
        string s = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(s) && float.TryParse(s, out float v) && v > 0 ? v : fallback;
    }

    private static BackendType ResolveBackend() =>
        (Environment.GetEnvironmentVariable("TS_GMTP_BACKEND") ?? "ggml_cpu").ToLowerInvariant() switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.Cuda,
            "cpu" => BackendType.Cpu,
            _ => BackendType.GgmlCpu,
        };

    private static bool TryResolveModels(out string targetPath, out string draftPath)
    {
        targetPath = null;
        draftPath = null;
        if (Environment.GetEnvironmentVariable("TS_GMTP_E2E") != "1")
            return false;

        string explicitTarget = Environment.GetEnvironmentVariable("TS_GMTP_TARGET");
        string explicitDraft = Environment.GetEnvironmentVariable("TS_GMTP_DRAFT");
        if (!string.IsNullOrEmpty(explicitTarget) && File.Exists(explicitTarget) &&
            !string.IsNullOrEmpty(explicitDraft) && File.Exists(explicitDraft))
        {
            targetPath = explicitTarget;
            draftPath = explicitDraft;
            return true;
        }

        string dir = Environment.GetEnvironmentVariable("TS_GMTP_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            dir = DefaultModelDir;
        if (!Directory.Exists(dir))
            return false;

        var ggufs = Directory.GetFiles(dir, "*.gguf")
            .Where(p => !Path.GetFileName(p).ToLowerInvariant().Contains("mmproj"))
            .ToList();
        string draft = ggufs.FirstOrDefault(p => Path.GetFileName(p).ToUpperInvariant().Contains("MTP"));
        string target = ggufs.FirstOrDefault(p => p != draft);
        draftPath = draft;
        targetPath = target;
        return targetPath != null && draftPath != null;
    }

    private static string Trim(string s, int len = 200)
        => string.IsNullOrEmpty(s) ? string.Empty
           : (s.Length <= len ? s : s.Substring(0, len) + "...").Replace("\n", "\\n");
}
