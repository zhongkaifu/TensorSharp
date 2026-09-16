// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Private extension of the pinned AgentTurnBench source; never compiled into a server.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace AgentTurnBench;

internal sealed partial class Bench
{
    private readonly List<object> _mtpInputs = new();
    private string MtpMode => Environment.GetEnvironmentVariable("QWEN_MTP_MODE") ?? throw new InvalidOperationException("QWEN_MTP_MODE is required");
    private bool MtpLong => Environment.GetEnvironmentVariable("QWEN_MTP_TIER") == "long-qsa-unqualified";
    private SpeculationOptions MtpOptions => MtpMode == "plain" ? SpeculationOptions.Disabled : SpecOptions(SpeculatorRegistry.Auto);

    internal static void RequireMtpHead(ModelBase model, Options options)
    {
        string mode = Environment.GetEnvironmentVariable("QWEN_MTP_MODE");
        string tier = Environment.GetEnvironmentVariable("QWEN_MTP_TIER");
        if (mode is not ("plain" or "mtp") || tier is not ("dense" or "long-qsa-unqualified"))
            throw new InvalidOperationException("Explicit mode plain|mtp and tier dense|long-qsa-unqualified required");
        if (model.Config.Architecture != "qwen4exp" || model is not IDraftHead { HasDraftHead: true } head
            || !head.DraftHeadKind.ToString().Equals("mtp", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("The real shared learned MTP head must attach in BOTH modes; fallback is a failure");
        if (options.Warmup != 0 || options.MeasurePasses != 1 || options.Out == null || options.Scenarios.Count != 1)
            throw new InvalidOperationException("Owner schedules separate warmed process pairs; use --warmup 0 --measure-passes 1 --scenarios qwen-mtp --out");
        if (model.MaxContextLength != 65536 || Environment.GetEnvironmentVariable("TS_SPEC_DRAFT") != "3")
            throw new InvalidOperationException("Expected context 65536 and explicit TS_SPEC_DRAFT=3");
        Console.WriteLine($"[qwen-mtp-followup] mode={mode} tier={tier} attached_head={head.DraftHeadKind} release_qualified=false");
    }

    private static string TokenSha(List<int> tokens)
    {
        byte[] bytes = new byte[tokens.Count * 4];
        for (int i = 0; i < tokens.Count; i++) System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4, 4), tokens[i]);
        return Convert.ToHexStringLower(SHA256.HashData(bytes));
    }

    private void RecordInput(string id, List<int> prompt, int maxNew, bool media = false)
    {
        // Reserve the verifier anchor plus its three proposals. Never relabel a
        // prompt as a dense control merely because the user text is short.
        int upper = checked(prompt.Count + maxNew + 4);
        if (prompt.Count < 1 || upper > _model.MaxContextLength || (!MtpLong && upper > 2051))
            throw new InvalidOperationException($"{id}: rendered/injected prompt {prompt.Count} + output {maxNew} + verify reserve 4 exceeds tier limit");
        _mtpInputs.Add(new { id, prompt_tokens = prompt.Count, max_new_tokens = maxNew, verify_reserve = 4,
            maximum_position_exclusive = upper, tokens = prompt.ToArray(), tokens_i32_sha256 = TokenSha(prompt),
            mode = MtpMode, tier = MtpLong ? "long-qsa-unqualified" : "dense", media });
    }

    private async Task<Row> MtpRequest(InferenceEngine engine, string id, List<int> prompt, int maxNew,
        SamplingConfig config = null, int shared = 0, bool media = false)
    {
        RecordInput(id, prompt, maxNew, media);
        var row = await RunAsync(engine, "qwen-mtp", id, prompt, maxNew, config ?? SamplingConfig.Greedy,
            expectBatched: true, sharedPrefix: shared, requestId: id);
        if (row.Tokens.Count == 0 || row.Finish is not ("eos" or "length"))
        { Failures++; Note(row, "FAIL: missing visible tokens or incomplete/error finish"); }
        return row;
    }

    private async Task QwenMtpAsync()
    {
        RequireMtpHead(_model, _o);
        try
        {
            // Same complete request warm-up in each process, retained separately
            // and excluded by the comparator from the measured 5% gate.
            using (var warm = NewEngine(MtpOptions))
                await MtpRequest(warm, "warmup", Render(Corpus.MinimalSystemPrompt, "Count from one to twenty in words."), 96);
            string file = Corpus.CodeText(_model.Tokenizer, MtpLong ? 7800 : 450);
            using (var engine = NewEngine(MtpOptions))
                await MtpRequest(engine, "short", Render(Corpus.MinimalSystemPrompt, "Explain a hash collision in two sentences."), 128);
            using (var engine = NewEngine(MtpOptions))
                await MtpRequest(engine, "copy", Render(Corpus.MinimalSystemPrompt,
                    $"Repeat this file exactly, then describe it in one sentence:\n```csharp\n{file}```"), 192);
            using (var engine = NewEngine(MtpOptions))
            {
                var config = SamplingConfig.Greedy.Clone();
                config.Grammar = GrammarLibrary.NewConstraint(GrammarLibrary.ForJsonObject(_model.Tokenizer), _model.Tokenizer);
                var row = await MtpRequest(engine, "json", Render(Corpus.MinimalSystemPrompt,
                    "Return only a JSON object with keys name, lines, and summary describing this file:\n" + file), 192, config);
                string output = _model.Tokenizer.Decode(row.Tokens.Where(t => !_model.Tokenizer.IsEos(t)).ToList());
                try { using var doc = JsonDocument.Parse(output); if (doc.RootElement.ValueKind != JsonValueKind.Object) throw new JsonException("Expected object"); }
                catch (JsonException) { Failures++; Note(row, "FAIL: complete JSON object required, including at length finish"); }
            }
            await MtpRetained(file);
            await MtpConcurrent(file);
            if (!MtpLong) { await MtpImage(); await MtpImage(orderedFrames: true); }
        }
        finally
        {
            File.WriteAllText(_o.Out + ".inputs.json", JsonSerializer.Serialize(new {
                mode = MtpMode, tier = MtpLong ? "long-qsa-unqualified" : "dense", release_qualified = false,
                actual_head = (_model as IDraftHead)?.DraftHeadKind.ToString(), inputs = _mtpInputs,
                limitations = new[] { "Continuous scheduler path, not HTTP/parser validation; run the separate API catalog.",
                    "Per-request prompt IDs include all generation-history changes; mismatch blocks the matched-workload gate.",
                    "Concurrent request rows preserve individual timing/counters; engine-step attribution is unavailable.",
                    "Long tier retains the pre-QSA correctness limitation and cannot qualify release." }
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
    }

    private async Task MtpRetained(string file)
    {
        using var engine = NewEngine(MtpOptions);
        // Exceed the scheduler's 256-token prefix-cache block so a sub-block
        // prompt cannot explain refusal. This does not imply model support for
        // retained checkpoints; only observed reuse can establish that coverage.
        string system = Corpus.MinimalSystemPrompt + "\n" + Corpus.CodeText(_model.Tokenizer, 384);
        var systemOnly = _renderer.RenderToTokens(_model.Tokenizer, _model.Config.ChatTemplate,
            new List<ChatMessage> { new() { Role = "system", Content = system } }, _arch, false);
        var history = new List<ChatMessage> { new() { Role = "system", Content = system },
            new() { Role = "user", Content = "Summarize the purpose of the code in the system message in one sentence." } };
        var a = RenderHistory(history, out var boundary);
        int shared = Lcp(systemOnly, a);
        if (shared < engine.PoolStats.blockSize) throw new InvalidOperationException("Shared-prefix fixture does not contain one complete cache block");
        var row = await MtpRequest(engine, "retained-A", a, 96, shared: shared);
        await MtpRequest(engine, "retained-B", Render(system, "Give a one-sentence explanation of why unit tests are useful."), 96, shared: shared);
        history.Add(AssistantTurn(row, boundary));
        history.Add(new() { Role = "user", Content = "Now repeat these first lines exactly:\n" + Corpus.CodeText(_model.Tokenizer, 150) });
        await MtpRequest(engine, "retained-A-followup", RenderHistory(history, out _), 160, shared: shared);
    }

    private async Task MtpConcurrent(string file)
    {
        using var engine = NewEngine(MtpOptions);
        var prompts = Enumerable.Range(0, 4).Select(i => Render(Corpus.MinimalSystemPrompt,
            $"Request {i}: repeat the following file exactly:\n" + file)).ToArray();
        long start = Stopwatch.GetTimestamp();
        var pending = new List<Task<Run>>();
        for (int i = 0; i < 4; i++)
        {
            string id = $"parallel4-i{i}"; RecordInput(id, prompts[i], 192);
            pending.Add(SubmitAsync(engine, id, prompts[i], 192, SamplingConfig.Greedy, 0, start));
        }
        Run[] runs = await Task.WhenAll(pending);
        double wall = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        var aggregate = ConcurrentDecodeMetrics.Calculate(runs.Select(r => new ConcurrentDelivery(r.SubmissionOffsetMs, r.TokenTimesMs)).ToArray(), wall);
        for (int i = 0; i < 4; i++)
        {
            Run r = runs[i]; var counters = ToTimeline(r).Speculation;
            double decodeMs = r.TotalMs - r.TtftMs;
            var row = new Row("qwen-mtp", r.Id, prompts[i].Count, r.Reused, 0, 0, 0, r.TtftMs,
                r.TtftMs > 0 ? (prompts[i].Count - r.Reused) * 1000 / r.TtftMs : 0,
                decodeMs > 0 ? Math.Max(0, r.Tokens.Count - 1) * 1000 / decodeMs : 0, r.Tokens.Count, r.TotalMs,
                r.Finish, counters.Drafted, counters.Accepted, counters.VerifySteps, counters.PlainSteps, counters.Rollbacks,
                "Engine-step attribution unavailable for simultaneous requests; aggregate delivery window retained.")
            { Tokens = r.Tokens, TokenTimesMs = r.TokenTimesMs, StartedUnixMilliseconds = r.StartedUnixMilliseconds,
                RequestTimelines = new() { ToTimeline(r) }, ConcurrentDecode = aggregate };
            Add(row);
            if (r.Error != null || r.Tokens.Count == 0 || r.Finish is not ("eos" or "length")) Failures++;
        }
        await MtpRequest(engine, "after-parallel4", Render(Corpus.MinimalSystemPrompt, "Explain a linked list in two sentences."), 96);
    }

    private async Task MtpImage(bool orderedFrames = false)
    {
        if (_o.MmProj == null || _o.Image == null) throw new InvalidOperationException("Image and mmproj are required for the dense suite");
        using var engine = NewEngine(MtpOptions);
        string id = orderedFrames ? "ordered-frames" : "image";
        var images = orderedFrames
            ? new List<string> { Path.Combine(Path.GetDirectoryName(_o.Image), "frame-0.png"), Path.Combine(Path.GetDirectoryName(_o.Image), "frame-1.png") }
            : new List<string> { _o.Image };
        foreach (string image in images) if (!File.Exists(image)) throw new FileNotFoundException("Required frame/image fixture", image);
        var history = new List<ChatMessage> { new() { Role = "system", Content = Corpus.MinimalSystemPrompt },
            new() { Role = "user", Content = orderedFrames
                ? "These two images are sampled frames in chronological order, at 0 and 1 seconds. List their visible numbers in that order, then describe the colors."
                : "Describe the colored card and all text visible in this image in two sentences.", ImagePaths = images } };
        var prompt = RenderHistory(history, out var boundary);
        prompt = _model.MultimodalInjector.ProcessPromptTokens(history, prompt, id);
        var first = await MtpRequest(engine, id, prompt, 128, media: true);
        history.Add(AssistantTurn(first, boundary));
        history.Add(new() { Role = "user", Content = orderedFrames
            ? "Using those same two images, which number appeared at 1 second? Answer it, then explain the ordering."
            : "Based on the same image, repeat its text, then explain its dominant color. Do not invent missing content." });
        var followup = RenderHistory(history, out _);
        followup = _model.MultimodalInjector.ProcessPromptTokens(history, followup, id + "-followup");
        await MtpRequest(engine, id + "-followup", followup, 128, media: true);
    }
}
