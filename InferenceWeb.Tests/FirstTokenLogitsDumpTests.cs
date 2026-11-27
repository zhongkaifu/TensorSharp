// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// TS_DUMP_LOGITS must hold the logits that choose the FIRST GENERATED TOKEN - the last prompt row
/// after the whole prompt ran - whatever path ran it, and never a warm-up or decode forward.
///
/// The stand-in model's logits are a function of (cache length after the forward, last token), so
/// every forward of a run - each prompt chunk, each decode step - produces a distinct vector, and the
/// prompt row of a P-token prompt is <c>Row(P, prompt[P-1])</c> however the prompt was split. A dump
/// taken from the first forward (the old semantics) is a different vector whenever the prompt ran as
/// more than one forward or not through <c>ModelBase.Forward</c> at all.
///
/// The dump is process-wide state, so these tests run alone (the engine-environment collection).
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class FirstTokenLogitsDumpTests : IDisposable
{
    private const int Vocab = 128;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"ts-first-token-dump-{Guid.NewGuid():N}");
    private readonly string _gguf;
    private readonly string _dump;

    public FirstTokenLogitsDumpTests()
    {
        Directory.CreateDirectory(_dir);
        _gguf = Path.Combine(_dir, "probe.gguf");
        _dump = Path.Combine(_dir, "first-token.f32");
        BackendFailureWarmupTests.WriteProbeGguf(_gguf);
        FirstTokenLogitsDump.ResetForTests(_dump);
    }

    public void Dispose()
    {
        FirstTokenLogitsDump.ResetForTests(FirstTokenLogitsDump.EnvironmentPath);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private static readonly int[] Prompt = "Hello wo".Select(c => (int)c).ToArray();   // 8 tokens

    // ------------------------------------------------------------------ engine paths

    [Fact]
    public void ChunkedPrompt_DumpsTheLastPromptRow_NotTheFirstChunk()
    {
        using var model = new RowModel(_gguf);
        var generated = RunEngine(model, Config(chunk: 3), ("req-a", Prompt, 4));

        float[] dumped = ReadDump();
        Assert.Equal(new[] { 3, 3, 2 }, model.PromptForwardSizes());
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), dumped);
        // The first chunk's row is a different vector: the old first-forward dump held it.
        Assert.NotEqual(model.Forwards[0].Logits, dumped);
        // The dumped argmax is the token the engine generated first.
        Assert.Equal(generated["req-a"][0], ArgMax(dumped));

        using var sidecar = ReadSidecar();
        var root = sidecar.RootElement;
        Assert.Equal(FirstTokenLogitsDump.Semantics, root.GetProperty("semantics").GetString());
        Assert.Equal("engine", root.GetProperty("source").GetString());
        Assert.Equal("req-a", root.GetProperty("request_id").GetString());
        Assert.Equal(Prompt.Length, root.GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(FirstTokenLogitsDump.Fingerprint(Prompt), root.GetProperty("prompt_fingerprint").GetString());
        Assert.Equal(0, root.GetProperty("prefix_reused_tokens").GetInt32());
        Assert.Equal(3, root.GetProperty("prefill_forwards").GetInt32());
        Assert.Equal(new[] { 3, 3, 2 }, root.GetProperty("prefill_steps").EnumerateArray()
            .Select(step => step.GetProperty("tokens").GetInt32()).ToArray());
        Assert.Equal("PerSequence", root.GetProperty("final_prefill_path").GetString());
        Assert.Equal(generated["req-a"][0], root.GetProperty("first_token").GetInt32());
        Assert.Equal("sampled", root.GetProperty("first_token_source").GetString());
        Assert.Equal(ArgMax(dumped), root.GetProperty("argmax").GetInt32());
        Assert.Equal(Vocab, root.GetProperty("count").GetInt32());
        // The sidecar names the exact bytes beside it, so a stale sidecar can be told apart.
        Assert.Equal(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(_dump))).ToLowerInvariant(),
            root.GetProperty("vector_sha256").GetString());
        Assert.Equal(0, root.GetProperty("media_spans").GetInt32());
    }

    [Fact]
    public void DecodeForwardsAndLaterRequests_DoNotReplaceTheDump()
    {
        using var model = new RowModel(_gguf);
        var config = Config(chunk: 64);
        var first = RunEngine(model, config, ("first", Prompt, 5));
        int forwardsAfterFirst = model.Forwards.Count;
        int[] second = "Another prompt".Select(c => (int)c).ToArray();
        RunEngine(model, config, ("second", second, 3));

        Assert.True(forwardsAfterFirst >= 1 + 4, "the first request must have decoded after its prompt");
        Assert.True(model.Forwards.Count > forwardsAfterFirst);
        float[] dumped = ReadDump();
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), dumped);
        Assert.Equal(first["first"][0], ArgMax(dumped));
        using var sidecar = ReadSidecar();
        Assert.Equal("first", sidecar.RootElement.GetProperty("request_id").GetString());
    }

    /// <summary>The Nemotron <c>--tp 2</c> / Qwen3.8 <c>cuda</c> case: one configuration runs the
    /// prompt in one forward, the other splits it across forwards. The dumps must be the same row.</summary>
    [Fact]
    public void OnePromptForwardAndChunkedPerSequence_DumpTheSameRow()
    {
        string dumpA = Path.Combine(_dir, "single.f32"), dumpB = Path.Combine(_dir, "split.f32");

        FirstTokenLogitsDump.ResetForTests(dumpA);
        using (var single = new RowModel(_gguf))
        {
            RunEngine(single, Config(chunk: 64), ("single", Prompt, 3));
            Assert.Equal(new[] { Prompt.Length }, single.PromptForwardSizes());
        }

        FirstTokenLogitsDump.ResetForTests(dumpB);
        using (var split = new RowModel(_gguf))
        {
            RunEngine(split, Config(chunk: 3), ("split", Prompt, 3));
            Assert.Equal(new[] { 3, 3, 2 }, split.PromptForwardSizes());
        }

        float[] a = ReadDump(dumpA), b = ReadDump(dumpB);
        Assert.Equal(a, b);
        using var sa = ReadSidecar(dumpA);
        using var sb = ReadSidecar(dumpB);
        Assert.Equal("PerSequence", sa.RootElement.GetProperty("final_prefill_path").GetString());
        Assert.Equal("PerSequence", sb.RootElement.GetProperty("final_prefill_path").GetString());
        Assert.Equal(sa.RootElement.GetProperty("prompt_fingerprint").GetString(),
            sb.RootElement.GetProperty("prompt_fingerprint").GetString());
    }

    [Fact]
    public void WarmUpForwardsAndWarmUpRequests_AreNotDumped()
    {
        using var model = new RowModel(_gguf);
        model.WarmUpKernels();   // ModelBase's own throwaway decode + prefill forwards
        Assert.True(model.Forwards.Count >= 2, "the warm-up should have run forwards");
        Assert.False(File.Exists(_dump), "a warm-up forward was dumped");

        var config = Config(chunk: 64);
        int[] warmPrompt = "hi".Select(c => (int)c).ToArray();
        using (FirstTokenLogitsDump.Suppress())
            RunEngine(model, config, ("warm-up", warmPrompt, 2));
        Assert.False(File.Exists(_dump), "a suppressed warm-up request was dumped");

        RunEngine(model, config, ("real", Prompt, 2));
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), ReadDump());
        using var sidecar = ReadSidecar();
        Assert.Equal("real", sidecar.RootElement.GetProperty("request_id").GetString());
    }

    // ------------------------------------------------------------------ the real warm-up call sites

    /// <summary>The CLI's <c>--warmup-runs</c> loop (Program.RunWarmupPasses) runs whole requests
    /// before the timed one; none of them may be the dump.</summary>
    [Fact]
    public void CliWarmupRuns_AreNotDumped_TheTimedRunIs()
    {
        using var model = new RowModel(_gguf);
        using var session = new TensorSharp.Cli.CliInferenceSession(model, Config(chunk: 64), NullLogger.Instance);
        int[] warm = "hi there".Select(c => (int)c).ToArray();
        int pass = 0;
        TensorSharp.Cli.Program.RunWarmupPasses(model, session, runs: 2, maxTokens: 8,
            decodeTokens => session.Generate(warm, decodeTokens, SamplingConfig.Greedy, requestId: $"warm-up-{++pass}"));
        Assert.Equal(2, pass);
        Assert.False(File.Exists(_dump), "a CLI --warmup-runs request was dumped");

        var timed = session.Generate(Prompt, 3, SamplingConfig.Greedy, requestId: "timed");
        float[] dumped = ReadDump();
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), dumped);
        Assert.Equal(timed.Tokens[0], ArgMax(dumped));
        using var sidecar = ReadSidecar();
        Assert.Equal("timed", sidecar.RootElement.GetProperty("request_id").GetString());
    }

    /// <summary>The server / TensorAgent prefix-cache warm-up (PrefixCacheWarmup.RunAsync) is a real
    /// one-token chat request; it must not be the dump, and the first real request must be.</summary>
    [Fact]
    public async Task PrefixCacheWarmupRequest_IsNotDumped_TheFirstRealRequestIs()
    {
        using var model = new RowModel(_gguf);
        using var engine = new InferenceEngine(model, Config(chunk: 64), NullLogger.Instance);
        int[] warm = "hi there".Select(c => (int)c).ToArray();
        var result = await PrefixCacheWarmup.RunAsync(
            (body, ct) => SubmitAndFinish(engine, "warm-up", warm, ct), "warm-up-session", think: false, NullLogger.Instance);
        Assert.True(result.Warmed, result.Detail);
        Assert.False(File.Exists(_dump), "the prefix-cache warm-up request was dumped");

        var real = await Drain(Submit(engine, "real", Prompt, 3));
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), ReadDump());
        using var sidecar = ReadSidecar();
        Assert.Equal("real", sidecar.RootElement.GetProperty("request_id").GetString());
        Assert.Equal(real[0], sidecar.RootElement.GetProperty("first_token").GetInt32());
    }

    /// <summary>TensorAgent cancels its background warm-up when a real turn arrives, so RunAsync can
    /// return (closing its suppression scope) while the warm-up request is still in the engine. The
    /// exclusion belongs to the request, so the warm-up's late first token is still not dumped.</summary>
    [Fact]
    public async Task WarmupStillInTheEngineAfterItsCallerReturned_IsNotDumped()
    {
        using var model = new RowModel(_gguf);
        using var engine = new InferenceEngine(model, Config(chunk: 64), NullLogger.Instance);
        using var gate = new ManualResetEventSlim(false);
        model.Gate = gate;   // every forward waits: the warm-up cannot sample before RunAsync returns
        int[] warm = "hi there".Select(c => (int)c).ToArray();
        InferenceRequestHandle warmHandle = null;
        var result = await PrefixCacheWarmup.RunAsync(
            (body, ct) => SubmitAndLeave(engine, "warm-up", warm, h => warmHandle = h), "warm-up-session",
            think: false, NullLogger.Instance);
        Assert.True(result.Warmed, result.Detail);
        Assert.NotNull(warmHandle);
        Assert.False(FirstTokenLogitsDump.IsSuppressedHere);

        gate.Set();
        await Drain(warmHandle);
        Assert.False(File.Exists(_dump), "a warm-up that sampled after its caller's scope closed was dumped");

        await Drain(Submit(engine, "real", Prompt, 2));
        using var sidecar = ReadSidecar();
        Assert.Equal("real", sidecar.RootElement.GetProperty("request_id").GetString());
    }

    /// <summary>A real request that arrives from another flow while a warm-up's scope is open is not
    /// a warm-up: it is dumped (the warm-up, sampled in the same steps, is not).</summary>
    [Fact]
    public async Task RealRequestFromAnotherFlowDuringAWarmup_IsDumped()
    {
        using var model = new RowModel(_gguf);
        using var engine = new InferenceEngine(model, Config(chunk: 64), NullLogger.Instance);
        using var gate = new ManualResetEventSlim(false);
        model.Gate = gate;
        int[] warm = "hi there".Select(c => (int)c).ToArray();
        var submitReal = new SemaphoreSlim(0);
        InferenceRequestHandle realHandle = null;
        // Started before the warm-up's scope opens: its own flow, as a separate HTTP request would be.
        Task real = Task.Run(async () =>
        {
            await submitReal.WaitAsync();
            Assert.False(FirstTokenLogitsDump.IsSuppressedHere);
            realHandle = Submit(engine, "real", Prompt, 2);
        });

        var result = await PrefixCacheWarmup.RunAsync((body, ct) => WarmupAlongsideReal(), "warm-up-session",
            think: false, NullLogger.Instance);
        Assert.True(result.Warmed, result.Detail);
        await Drain(realHandle);
        float[] dumped = ReadDump();
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), dumped);
        using var sidecar = ReadSidecar();
        Assert.Equal("real", sidecar.RootElement.GetProperty("request_id").GetString());

        async IAsyncEnumerable<object> WarmupAlongsideReal()
        {
            Assert.True(FirstTokenLogitsDump.IsSuppressedHere);
            var warmHandle = Submit(engine, "warm-up", warm, 1);
            submitReal.Release();
            await real;                 // the real request is in the engine before anything samples
            gate.Set();
            await Drain(warmHandle);
            yield return new { done = true };
        }
    }

    /// <summary>Samplers mask the logits buffer in place (an unopened thinking end, a grammar); the
    /// dump is the raw row the model produced, written before the sampler touches it.</summary>
    [Fact]
    public async Task SamplerMasks_AreNotInTheDump()
    {
        using var model = new RowModel(_gguf);
        float[] row = RowModel.Row(Prompt.Length, Prompt[^1]);
        int end = ArgMax(row);   // mask the model's own favourite
        var sampling = SamplingConfig.Greedy;
        sampling.ThinkingBudget = new ThinkingTokenBudget(int.MaxValue, end, openTokenId: 2, openAtStart: false,
            suppressUnopenedEnd: true);
        using var engine = new InferenceEngine(model, Config(chunk: 3), NullLogger.Instance);
        var handle = engine.SubmitRequest(new SequenceState("masked", Prompt.ToList(), 2, engine.PoolStats.blockSize, sampling));
        var tokens = await Drain(handle);

        Assert.NotEqual(end, tokens[0]);   // the sampler did mask it
        float[] dumped = ReadDump();
        Assert.All(dumped, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(row, dumped);
        using var sidecar = ReadSidecar();
        Assert.Equal(end, sidecar.RootElement.GetProperty("argmax").GetInt32());
        Assert.Equal(tokens[0], sidecar.RootElement.GetProperty("first_token").GetInt32());
    }

    /// <summary>An earlier run's pair is removed when a process arms the dump (so a run that dies
    /// before its first token leaves nothing), once: a later engine never removes this run's dump.</summary>
    [Fact]
    public void StaleOutputs_AreClearedOnce_WhenTheFirstEngineStarts()
    {
        File.WriteAllBytes(_dump, new byte[8]);
        File.WriteAllText(_dump + FirstTokenLogitsDump.SidecarSuffix, "{\"semantics\": \"first-generated-token\"}");
        using var model = new RowModel(_gguf);
        using (new InferenceEngine(model, Config(chunk: 64), NullLogger.Instance)) { }
        Assert.False(File.Exists(_dump), "a stale vector survived");
        Assert.False(File.Exists(_dump + FirstTokenLogitsDump.SidecarSuffix), "a stale sidecar survived");

        RunEngine(model, Config(chunk: 64), ("real", Prompt, 2));
        RunEngine(model, Config(chunk: 64), ("later", Prompt, 2));   // another engine in this process
        Assert.Equal(RowModel.Row(Prompt.Length, Prompt[^1]), ReadDump());
        using var sidecar = ReadSidecar();
        Assert.Equal("real", sidecar.RootElement.GetProperty("request_id").GetString());
    }

    /// <summary>The prompt never runs through <c>ModelBase.Forward</c>: two arrivals prefill in one
    /// batched fused call and decode in one batched fused call. The old dump saw no forward at all
    /// here; the sampler still picks the first token from the prompt row.</summary>
    [Fact]
    public void BatchedFusedPrefillAndDecode_DumpThePromptRowOfTheFirstSampledRequest()
    {
        using var model = new RowModel(_gguf) { PerSequenceFused = true };
        var cfg = Config(chunk: 64);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(cfg, pool, NullLogger.Instance, supportsCrossSequenceKvReuse: false);
        var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

        int[] other = "Different".Select(c => (int)c).ToArray();
        var seqs = new[]
        {
            NewSequence(pool, "fused-a", Prompt, cfg.BlockSize),
            NewSequence(pool, "fused-b", other, cfg.BlockSize),
        };

        var prefill = new SchedulerOutput();
        foreach (var seq in seqs)
            prefill.ScheduledWork.Add(new ScheduledSequenceWork(seq, seq.PromptTokens.Count, isNewAdmission: true, isPrefill: true));
        Assert.All(executor.ExecuteStep(prefill), r => Assert.Null(r.Error));
        Assert.Equal(1, model.BatchedPrefillCalls);
        Assert.False(File.Exists(_dump));

        var decode = new SchedulerOutput();
        foreach (var seq in seqs)
            decode.ScheduledWork.Add(new ScheduledSequenceWork(seq, 1, isNewAdmission: false, isPrefill: false));
        var decoded = executor.ExecuteStep(decode);
        Assert.All(decoded, r => Assert.Null(r.Error));
        Assert.Equal(1, model.BatchedDecodeCalls);
        Assert.Empty(model.Forwards);   // nothing ran through ModelBase.Forward

        using var sidecar = ReadSidecar();
        string id = sidecar.RootElement.GetProperty("request_id").GetString();
        var owner = seqs.Single(s => s.RequestId == id);
        float[] dumped = ReadDump();
        Assert.Equal(RowModel.Row(owner.PromptTokens.Count, owner.PromptTokens[^1]), dumped);
        Assert.Equal(owner.OutputTokens[0], ArgMax(dumped));
        Assert.Equal(owner.OutputTokens[0], sidecar.RootElement.GetProperty("first_token").GetInt32());
        Assert.Equal("PerSequenceFused", sidecar.RootElement.GetProperty("final_prefill_path").GetString());
    }

    // ------------------------------------------------------------------ CLI direct loops

    /// <summary><c>--benchmark --bench-chunked</c> runs the prompt through <c>ForwardRefill</c>, which
    /// the old dump never saw: it wrote the first DECODE forward instead.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Benchmark_DumpsRunOnesPromptRow_ForForwardAndForwardRefill(bool chunked)
    {
        using var model = new RowModel(_gguf) { RefillChunk = 3 };
        TensorSharp.Cli.Program.RunBenchmark(model, prefillTokens: 8, decodeTokens: 3, runs: 2, chunked: chunked);

        Assert.Equal(chunked ? new[] { 3, 3, 2 } : new[] { 8 }, model.Forwards.Take(chunked ? 3 : 1).Select(f => f.Tokens).ToArray());
        int[] prompt = model.FirstPromptTokens;
        float[] dumped = ReadDump();
        Assert.Equal(RowModel.Row(prompt.Length, prompt[^1]), dumped);
        using var sidecar = ReadSidecar();
        Assert.Equal("cli-benchmark", sidecar.RootElement.GetProperty("source").GetString());
        Assert.Equal(chunked ? "ForwardRefill" : "Forward", sidecar.RootElement.GetProperty("final_prefill_path").GetString());
        Assert.Equal(8, sidecar.RootElement.GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(ArgMax(dumped), sidecar.RootElement.GetProperty("first_token").GetInt32());
        // The benchmark's first greedy token is the first decode forward's input.
        Assert.Equal(ArgMax(dumped), model.Forwards[chunked ? 3 : 1].LastToken);
    }

    // ------------------------------------------------------------------ the dump itself

    [Fact]
    public void Unset_WritesNothing_AndIsNotArmed()
    {
        FirstTokenLogitsDump.ResetForTests(null);
        Assert.False(FirstTokenLogitsDump.IsArmed);
        Assert.Null(FirstTokenLogitsDump.TryCapture(new float[] { 1, 2 }, new FirstTokenLogitsInfo()));
        using var model = new RowModel(_gguf);
        RunEngine(model, Config(chunk: 3), ("unset", Prompt, 2));
        Assert.Empty(Directory.GetFiles(_dir, "*.f32*"));
    }

    [Fact]
    public void Capture_IsOncePerProcess_AndSuppressionNests()
    {
        using (FirstTokenLogitsDump.Suppress())
        {
            using (FirstTokenLogitsDump.Suppress()) { }
            Assert.True(FirstTokenLogitsDump.IsSuppressedHere);
            Assert.Null(FirstTokenLogitsDump.TryCapture(new float[] { 9, 1 }, new FirstTokenLogitsInfo()));
            Assert.True(FirstTokenLogitsDump.IsArmed);   // still owed: suppression is per flow / request
        }
        Assert.False(FirstTokenLogitsDump.IsSuppressedHere);
        Assert.True(FirstTokenLogitsDump.IsArmed);

        var capture = FirstTokenLogitsDump.TryCapture(new float[] { 1, 5, 2 }, new FirstTokenLogitsInfo { Source = "test", PromptTokens = 4 });
        Assert.NotNull(capture);
        Assert.Equal(1, capture.Argmax);
        Assert.False(FirstTokenLogitsDump.IsArmed);
        Assert.Null(FirstTokenLogitsDump.TryCapture(new float[] { 7, 7, 7 }, new FirstTokenLogitsInfo()));
        Assert.Equal(new float[] { 1, 5, 2 }, ReadDump());

        using (var before = ReadSidecar())
            Assert.Equal(JsonValueKind.Null, before.RootElement.GetProperty("first_token").ValueKind);
        capture.Complete(1, "argmax");
        using var after = ReadSidecar();
        Assert.Equal(1, after.RootElement.GetProperty("first_token").GetInt32());
        Assert.Equal("argmax", after.RootElement.GetProperty("first_token_source").GetString());
    }

    // ------------------------------------------------------------------ helpers

    private static SchedulerConfig Config(int chunk) => new()
    {
        BlockSize = 4, NumBlocks = 256, MaxNumRunningSequences = 2,
        MaxNumBatchedTokens = Math.Max(chunk, 16), MaxPrefillChunkSize = chunk, SoloPrefillChunkSize = chunk,
        EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
    };

    private static Dictionary<string, List<int>> RunEngine(RowModel model, SchedulerConfig config,
        params (string Id, int[] Prompt, int MaxTokens)[] requests)
    {
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var handles = requests.Select(r => (r.Id, Handle: engine.SubmitRequest(
            new SequenceState(r.Id, r.Prompt.ToList(), r.MaxTokens, engine.PoolStats.blockSize, SamplingConfig.Greedy)))).ToList();
        var tokens = new Dictionary<string, List<int>>();
        foreach (var (id, handle) in handles)
        {
            var list = new List<int>();
            while (handle.Tokens.WaitToReadAsync().AsTask().GetAwaiter().GetResult())
                while (handle.Tokens.TryRead(out int t)) list.Add(t);
            var completion = handle.Completion.GetAwaiter().GetResult();
            Assert.True(completion.Status != SequenceStatus.FinishedError, $"{id} failed");
            tokens[id] = list;
        }
        return tokens;
    }

    private static InferenceRequestHandle Submit(InferenceEngine engine, string id, int[] prompt, int maxTokens)
        => engine.SubmitRequest(new SequenceState(id, prompt.ToList(), maxTokens, engine.PoolStats.blockSize, SamplingConfig.Greedy));

    private static async Task<List<int>> Drain(InferenceRequestHandle handle)
    {
        var list = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync())
            list.Add(token);
        var completion = await handle.Completion;
        Assert.True(completion.Status != SequenceStatus.FinishedError, $"{completion.Status}");
        return list;
    }

    /// <summary>A stand-in for the host's chat stream: one request, run to completion, then done.</summary>
    private static async IAsyncEnumerable<object> SubmitAndFinish(InferenceEngine engine, string id, int[] prompt,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Drain(Submit(engine, id, prompt, 1));
        cancellationToken.ThrowIfCancellationRequested();
        yield return new { done = true };
    }

    /// <summary>A chat stream that ends while its request is still in the engine (the cancelled
    /// background warm-up).</summary>
    private static async IAsyncEnumerable<object> SubmitAndLeave(InferenceEngine engine, string id, int[] prompt,
        Action<InferenceRequestHandle> submitted)
    {
        submitted(Submit(engine, id, prompt, 1));
        await Task.Yield();
        yield return new { done = true };
    }

    private static SequenceState NewSequence(BlockPool pool, string id, int[] prompt, int blockSize)
    {
        var seq = new SequenceState(id, prompt.ToList(), 4, blockSize, SamplingConfig.Greedy);
        foreach (var block in pool.AllocateNew((prompt.Length + 4 + blockSize - 1) / blockSize)!)
            seq.BlockTable.AppendBlock(block);
        seq.Status = SequenceStatus.Running;
        return seq;
    }

    private float[] ReadDump(string path = null)
    {
        byte[] bytes = File.ReadAllBytes(path ?? _dump);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private JsonDocument ReadSidecar(string path = null)
        => JsonDocument.Parse(File.ReadAllText((path ?? _dump) + FirstTokenLogitsDump.SidecarSuffix));

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>Stand-in whose every forward yields <see cref="Row"/>(cache length after it, last
    /// token), through a REUSED output buffer as the real models do. Modes: plain per-sequence
    /// (PerSequence plan) and per-request fused holders with batched fused prefill/decode
    /// (PerSequenceFused).</summary>
    private sealed class RowModel : ModelBase, IBatchedPagedModel
    {
        private readonly float[] _buffer = new float[Vocab];
        private readonly Dictionary<string, int> _holders = new(StringComparer.Ordinal);
        private string _active;
        private int _primaryLength;

        public bool PerSequenceFused { get; init; }
        /// <summary>ForwardRefill runs the prompt in pieces of this size (0: one forward).</summary>
        public int RefillChunk { get; init; }
        public List<(int Tokens, int LengthAfter, int LastToken, bool Prompt, float[] Logits)> Forwards { get; } = new();
        public int[] FirstPromptTokens { get; private set; }
        public int BatchedPrefillCalls, BatchedDecodeCalls;
        /// <summary>When set, every forward waits for it (to hold a request inside the engine).</summary>
        public ManualResetEventSlim Gate { get; set; }

        public RowModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = Vocab, NumLayers = 1 };
            Tokenizer = new CharTokenizer();
        }

        public static float[] Row(int length, int lastToken)
        {
            var row = new float[Vocab];
            for (int i = 0; i < Vocab; i++)
                row[i] = (float)Math.Sin(0.37 * i + 1.3 * length + 0.11 * lastToken);
            row['A' + (length * 7 + lastToken) % 26] = 8f + 0.01f * length;
            return row;
        }

        public int[] PromptForwardSizes() => Forwards.Where(f => f.Prompt).Select(f => f.Tokens).ToArray();

        private int Length
        {
            get => _active != null ? _holders[_active] : _primaryLength;
            set { if (_active != null) _holders[_active] = value; else _primaryLength = value; }
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (Gate != null && !Gate.Wait(TimeSpan.FromSeconds(60)))
                throw new TimeoutException("the test never opened the forward gate");
            FirstPromptTokens ??= tokens.Length > 1 ? tokens : null;
            Length += tokens.Length;
            float[] row = Row(Length, tokens[^1]);
            Forwards.Add((tokens.Length, Length, tokens[^1], ForwardIsPrompt || tokens.Length > 1, row));
            Array.Copy(row, _buffer, Vocab);
            return _buffer;
        }

        protected override float[] ForwardRefillCore(int[] tokens)
        {
            FirstPromptTokens ??= tokens;
            if (RefillChunk <= 0) return ForwardCore(tokens);
            float[] last = null;
            for (int start = 0; start < tokens.Length; start += RefillChunk)
                last = ForwardCore(tokens.Skip(start).Take(Math.Min(RefillChunk, tokens.Length - start)).ToArray());
            return last;
        }

        protected override void ResetKVCacheCore() => Length = 0;

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "first-token-dump-probe";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) { Length = start + count; return true; }

        // IBatchedPagedModel: no ForwardBatch, so a solo request plans PerSequence.
        public bool BatchedForwardAvailable => false;
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx) => throw new NotSupportedException();

        public bool SupportsPerSequenceFusedForward => PerSequenceFused;
        public bool HasFusedSequenceCache(string id) => id != null && _holders.ContainsKey(id);
        public bool BindSequenceCache(string id)
        {
            bool fresh = _holders.TryAdd(id, 0);
            _active = id;
            return fresh;
        }
        public void AdoptPrimaryCacheToFused(string id) { }
        public void RestorePrimaryCache() => _active = null;
        public void OnSequenceReleased(string id)
        {
            _holders.Remove(id);
            if (_active == id) _active = null;
        }

        public bool TryForwardBatchedFusedPrefill(IReadOnlyList<string> ids, IReadOnlyList<int[]> tokens, int[] positions, float[][] outLogits)
        {
            if (!PerSequenceFused) return false;
            BatchedPrefillCalls++;
            for (int i = 0; i < ids.Count; i++)
            {
                int length = positions[i] + tokens[i].Length;
                _holders[ids[i]] = length;
                outLogits[i] = Row(length, tokens[i][^1]);
            }
            return true;
        }

        public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> ids, int[] tokens, int[] positions, float[][] outLogits)
        {
            if (!PerSequenceFused) return false;
            BatchedDecodeCalls++;
            for (int i = 0; i < ids.Count; i++)
            {
                int length = positions[i] + 1;
                _holders[ids[i]] = length;
                outLogits[i] = Row(length, tokens[i]);
            }
            return true;
        }
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, FirstTokenLogitsDumpTests.Vocab).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => FirstTokenLogitsDumpTests.Vocab;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
