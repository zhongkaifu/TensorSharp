// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Serving regressions found on Nemotron-H 8B/47B Reasoning-128K (campaign 2026-09-16).
//
//   1. Host/device recurrent-state split. The native Mamba2 decode kernel keeps the
//      conv/SSM state on the device between tokens and never drained it, so every
//      reader of the host arrays afterwards - a multi-token forward continuing the
//      sequence, a per-block snapshot, the migration into the batched slot pool when a
//      second request arrives - resumed from the state as it was before the first
//      decoded token. Concurrent requests produced different greedy output from the
//      same requests served alone ("101", "1000000...", loops).
//   2. Long-prompt prefill materialized one [heads, chunk, context] score tensor per
//      attention layer: 37 GB at 4,096 x 37k on the 8B, cudaMalloc OOM, HTTP 500.
//   3. Growing the model-owned paged K/V pool replaced every layer's buffer before
//      copying from it, so a sequence decoding in that step attended over zeros.
//   4. A sequence forwarded alone borrowed the model's logits buffer; when another
//      request took ownership first, the owner sampled from that request's logits
//      (engine-level, also on the per-sequence path).
//   5. Cached Mamba2 prefill graphs (GBs each at 4,096 tokens) were never evicted.
//
// Opt-in (needs the weights and a GPU backend for the state split to exist at all):
//   TS_TEST_NEMOTRON_H_DIR=/workspace/models/nemotron-h8b TS_TEST_GGML_BACKEND=cuda
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class NemotronHModelFixture : IDisposable
{
    public const string EnvDir = "TS_TEST_NEMOTRON_H_DIR";
    public const string GgufPattern = "nemotron-h";

    private ModelBase _model;

    public ModelBase Model
    {
        get
        {
            if (_model != null) return _model;
            string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvDir), GgufPattern);
            Assert.False(string.IsNullOrEmpty(path), $"no *{GgufPattern}* GGUF under {EnvDir}");
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("MAX_CONTEXT")))
                Environment.SetEnvironmentVariable("MAX_CONTEXT", "65536");
            var backend = TestGates.PinnedGgmlBackend;
            // KV_CACHE_DTYPE (q8_0, q4_0, f16, ...) runs the lane on that K/V cache, as the
            // server's --kv-cache-dtype would; the process-wide choice is put back after load
            // (the model keeps its own copy).
            KvCacheDtype restoreDtype = KvCacheDtypeConfig.Current;
            bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
            try
            {
                if (KvCacheDtypeConfig.TryParse(Environment.GetEnvironmentVariable("KV_CACHE_DTYPE"), out KvCacheDtype dtype))
                    KvCacheDtypeConfig.Set(dtype);
                _model = ModelBase.Create(path, backend);
            }
            finally
            {
                KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit);
            }
            return _model;
        }
    }

    public void Dispose() => _model?.Dispose();
}

public class NemotronHServingRegressionTests : IClassFixture<NemotronHModelFixture>
{
    private readonly NemotronHModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public NemotronHServingRegressionTests(NemotronHModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    // Short prompts with clear answers. Batched and single-sequence forwards run
    // different kernels (paged F32 K/V against the F16 cache, batch-size dependent
    // quantized matmul): ForwardBatch tracks the single-sequence logits to max|dlogit|
    // 0.3-1.4 on the 8B, so only near-ties may flip; a state bug moves far more.
    private static readonly string[] Prompts =
    {
        "[validation short-c4-r0-i0]\nWhat is 17 + 25? Reply with only the integer.",
        "[validation short-c4-r0-i1]\nName the capital of France in one word.",
        "[validation short-c4-r0-i2]\nWhat is the chemical symbol for gold? Reply with only the symbol.",
        "[validation short-c4-r0-i3]\nList three prime numbers separated by commas.",
    };

    [Fact]
    public void QueryRowsPerChunk_KeepsScoresWithinBudget()
    {
        long budget = 1L << 30;
        int rows = NemotronModel.PrefillQueryRowsPerChunk(32, 4096, 62561, budget);
        Assert.InRange(rows, 1, 4096);
        Assert.True((long)rows * 32 * 62561 * sizeof(float) <= budget);
        Assert.Equal(512, NemotronModel.PrefillQueryRowsPerChunk(32, 512, 1024, budget));
        Assert.Equal(1, NemotronModel.PrefillQueryRowsPerChunk(64, 4096, 1 << 20, 1));
    }

    /// <summary>A multi-token forward that continues a sequence after native decode
    /// steps must see the state those steps produced. Compared against prefilling the
    /// same tokens from scratch.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void ContinuationAfterDecode_MatchesFreshPrefill()
    {
        var model = _fixture.Model;
        int[] prompt = model.Tokenizer.Encode("The three primary colors of light are", addSpecial: true).ToArray();
        int[] tail = model.Tokenizer.Encode(" In summary, the answer is", addSpecial: false).ToArray();

        model.ResetKVCache();
        float[] logits = model.Forward(prompt);
        var generated = new List<int>();
        for (int i = 0; i < 12; i++)
        {
            int tok = ArgMax(logits);
            generated.Add(tok);
            logits = model.Forward(new[] { tok }); // single-token decode: the native kernel on GPU backends
        }
        float[] continued = (float[])model.Forward(tail).Clone();

        model.ResetKVCache();
        float[] fresh = (float[])model.Forward(prompt.Concat(generated).Concat(tail).ToArray()).Clone();
        model.ResetKVCache();

        double maxDiff = MaxAbsDiff(continued, fresh);
        _output.WriteLine($"generated={string.Join(",", generated)} argmax continued={ArgMax(continued)} fresh={ArgMax(fresh)} maxAbsDiff={maxDiff:F4}");
        Assert.Equal(ArgMax(fresh), ArgMax(continued));
        Assert.True(maxDiff < 1.0, $"continued-vs-fresh logits differ by {maxDiff:F3}: the continuation read stale recurrent state.");
    }

    /// <summary>The per-sequence path (<c>--no-continuous-batching</c>) serves concurrent
    /// requests one sequence per step through the same kernels as a request served
    /// alone, swapping each sequence's K/V and recurrent state in and out. Its greedy
    /// output must therefore equal the serial output token for token.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public Task ConcurrentGreedy_PerSequencePath_MatchesSerialExactly() => RunConcurrentGreedy(batchedPath: false);

    /// <summary>The batched path runs different kernels than a request served alone
    /// (paged F32 attention, batch-composition dependent quantized matmul), so a token
    /// whose top two candidates are within that numeric noise may legitimately flip.
    /// Every token it chooses must still be a near-top choice when the same history is
    /// replayed through the single-sequence forward; a sequence reading another's state,
    /// or state from the wrong step, chooses tokens far below the top.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public Task ConcurrentGreedy_BatchedPath_ChoosesOnlyNearTopTokens() => RunConcurrentGreedy(batchedPath: true);

    // Largest serial-forward logit gap a batched choice may have. ForwardBatch tracks the
    // single-sequence logits to max|dlogit| 0.3-1.4 on the 8B, so a flip needs a gap
    // below about twice that.
    private const float NearTopLogitGap = 3.0f;

    private async Task RunConcurrentGreedy(bool batchedPath)
    {
        var model = _fixture.Model;
        string previous = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", batchedPath ? "0" : "1");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        int[][] promptTokens = Prompts.Select(p => Render(model, renderer, p)).ToArray();
        const int maxNew = 32;
        var outputs = new List<(string Label, int Prompt, List<int> Tokens)>();
        List<int>[] serial = new List<int>[Prompts.Length];
        try
        {
            model.ResetKVCache();
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 512,
                NumBlocks = 64,
                BlockSize = 256,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = 256,
            };
            using (var engine = new InferenceEngine(model, cfg, NullLogger.Instance))
            {
                for (int i = 0; i < Prompts.Length; i++)
                    serial[i] = await Generate(engine, promptTokens[i], $"serial-{i}", maxNew, _ => { });

                var burst = await Task.WhenAll(Enumerable.Range(0, Prompts.Length).Select(i =>
                    Generate(engine, promptTokens[i], $"burst-{i}", maxNew, _ => { })));
                for (int i = 0; i < Prompts.Length; i++) outputs.Add(($"burst[{i}]", i, burst[i]));

                // Staggered: the first request is decoding alone when the others arrive,
                // so its state has to move (into the batched slot pool, or out to a
                // snapshot on the per-sequence path).
                var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                var first = Generate(engine, promptTokens[0], "stagger-0", maxNew,
                    n => { if (n == 6) started.TrySetResult(); });
                await Task.WhenAny(started.Task, first);
                var rest = Enumerable.Range(1, Prompts.Length - 1).Select(i =>
                    Generate(engine, promptTokens[i], $"stagger-{i}", maxNew, _ => { })).ToList();
                outputs.Add(("staggered[0]", 0, await first));
                var restOut = await Task.WhenAll(rest);
                for (int i = 1; i < Prompts.Length; i++) outputs.Add(($"staggered[{i}]", i, restOut[i - 1]));

                // Worker pool: four clients drain every prompt twice, so a new request
                // arrives whenever one finishes while the others are still decoding - the
                // arrival pattern of an HTTP benchmark at concurrency 4 - and the second
                // wave lands on the recurrent-state slots the first wave released.
                const int poolRequests = 12;
                var pooled = new List<int>[poolRequests];
                int next = -1;
                await Task.WhenAll(Enumerable.Range(0, 4).Select(async w =>
                {
                    int i;
                    while ((i = Interlocked.Increment(ref next)) < poolRequests)
                        pooled[i] = await Generate(engine, promptTokens[i % Prompts.Length], $"pool-{w}-{i}", maxNew, _ => { });
                }));
                for (int k = 0; k < poolRequests; k++) outputs.Add(($"pooled[{k}]", k % Prompts.Length, pooled[k]));
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", previous);
        }

        var failures = new List<string>();
        double worstGap = 0;
        for (int i = 0; i < Prompts.Length; i++)
            _output.WriteLine($"[{i}] serial      : {Show(Decode(model, serial[i]))}");
        foreach (var (label, p, tokens) in outputs)
        {
            bool same = serial[p].SequenceEqual(tokens);
            _output.WriteLine($"[{p}] {label,-12}: {(same ? "same  " : "DIFF  ")}{Show(Decode(model, tokens))}");
            if (same) continue;
            if (!batchedPath)
            {
                failures.Add($"{label} diverges from serial at token {FirstDiff(serial[p], tokens)}");
                continue;
            }

            // Replay the concurrent history through the single-sequence forward.
            model.ResetKVCache();
            float[] logits = model.Forward(promptTokens[p]);
            for (int t = 0; t < tokens.Count; t++)
            {
                float top = logits.Max();
                float gap = top - logits[tokens[t]];
                worstGap = Math.Max(worstGap, gap);
                if (gap > NearTopLogitGap)
                {
                    failures.Add($"{label} token {t} ({Show(model.Tokenizer.Decode(new List<int> { tokens[t] }))}) is {gap:F2} logits below the single-sequence top choice");
                    break;
                }
                if (t + 1 < tokens.Count)
                    logits = model.Forward(new[] { tokens[t] });
            }
            model.ResetKVCache();
        }
        _output.WriteLine($"largest serial-forward logit gap of a concurrent choice: {worstGap:F2}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private static int[] Render(ModelBase model, KVCachePromptRenderer renderer, string prompt) =>
        renderer.RenderToTokens(model.Tokenizer, model.Config?.ChatTemplate,
            new List<ChatMessage> { new() { Role = "user", Content = prompt } },
            model.Config?.Architecture ?? string.Empty, addGenerationPrompt: true, tools: null, enableThinking: false).ToArray();

    /// <summary>A prompt far past the point where one materialized score tensor per chunk
    /// stops fitting on the device prefills without an allocation failure.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void LongPromptPrefill_StaysBounded()
    {
        var model = _fixture.Model;
        var sb = new StringBuilder();
        int n = 0;
        int[] tokens;
        do
        {
            for (int i = 0; i < 400; i++, n++)
                sb.Append("Record ").Append(n).Append(": the item code is cedar-").Append(n * 17 + 31).Append(". Keep the records in order.\n");
            tokens = model.Tokenizer.Encode(sb.ToString(), addSpecial: true).ToArray();
        } while (tokens.Length < 49152);
        tokens = tokens.Take(49152).ToArray();

        model.ResetKVCache();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] logits = null;
        for (int start = 0; start < tokens.Length; start += 4096)
            logits = model.Forward(tokens.Skip(start).Take(4096).ToArray());
        sw.Stop();
        model.ResetKVCache();

        _output.WriteLine($"prefilled {tokens.Length} tokens in {sw.Elapsed.TotalSeconds:F1}s ({tokens.Length / sw.Elapsed.TotalSeconds:F0} tok/s)");
        Assert.NotNull(logits);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>Mamba2 prefill graphs are cached per chunk length, and every graph
    /// intermediate lives in the cached buffer (2.7 GB at 4,096 tokens on the 8B, 5.5 GB
    /// on the 47B). Prompts whose chunks come in many lengths must not leave all of those
    /// graphs resident: unbounded, a 32k prompt on the 47B left the device full and the
    /// next concurrent decode step failed its allocations (HTTP 500).</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void PrefillOfManyChunkLengths_DoesNotAccumulateDeviceMemory()
    {
        var model = _fixture.Model;
        var sb = new StringBuilder();
        for (int i = 0; i < 1200; i++)
            sb.Append("Entry ").Append(i).Append(": the lantern code is ember-").Append(i * 13 + 7).Append(".\n");
        int[] tokens = model.Tokenizer.Encode(sb.ToString(), addSpecial: true).ToArray();
        Assert.True(tokens.Length >= 4096, $"need 4096 tokens, have {tokens.Length}");

        model.ResetKVCache();
        model.Forward(tokens.Take(4096).ToArray()); // first long prefill sizes every pool
        model.ResetKVCache();
        if (!TensorSharp.GGML.GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeBefore, out long total) || total <= 0)
        {
            _output.WriteLine("backend reports no device memory; nothing to measure");
            return;
        }

        foreach (int len in new[] { 3968, 3840, 3712, 3584, 3456, 3328 })
        {
            model.ResetKVCache();
            model.Forward(tokens.Take(len).ToArray());
        }
        model.ResetKVCache();

        Assert.True(TensorSharp.GGML.GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeAfter, out _));
        long grownMiB = (freeBefore - freeAfter) >> 20;
        _output.WriteLine($"device free before={freeBefore >> 20} MiB after={freeAfter >> 20} MiB grown={grownMiB} MiB");
        Assert.True(grownMiB < 2048, $"six prefills of distinct chunk lengths left {grownMiB} MiB more resident on the device");
    }

    /// <summary>The fused prefill kernel and the query-chunked materialized fallback
    /// compute the same attention.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void FusedAndChunkedMaterializedPrefill_Agree()
    {
        var model = _fixture.Model;
        var text = string.Join("\n", Enumerable.Range(0, 120).Select(i => $"Line {i}: the tide rises at hour {i % 24}."));
        int[] tokens = model.Tokenizer.Encode(text, addSpecial: true).ToArray();

        float[] Prefill()
        {
            model.ResetKVCache();
            float[] last = null;
            for (int start = 0; start < tokens.Length; start += 512)
                last = (float[])model.Forward(tokens.Skip(start).Take(512).ToArray()).Clone();
            model.ResetKVCache();
            return last;
        }

        float[] fused = Prefill();
        float[] chunked;
        string previousBudget = Environment.GetEnvironmentVariable("TS_NEMOTRON_ATTN_SCORE_BUDGET_MB");
        try
        {
            NemotronModel.MaterializedPrefillAttentionForTest = true;
            Environment.SetEnvironmentVariable("TS_NEMOTRON_ATTN_SCORE_BUDGET_MB", "1");
            chunked = Prefill();
        }
        finally
        {
            NemotronModel.MaterializedPrefillAttentionForTest = false;
            Environment.SetEnvironmentVariable("TS_NEMOTRON_ATTN_SCORE_BUDGET_MB", previousBudget);
        }

        double maxDiff = MaxAbsDiff(fused, chunked);
        _output.WriteLine($"tokens={tokens.Length} argmax fused={ArgMax(fused)} chunked={ArgMax(chunked)} maxAbsDiff={maxDiff:F4}");
        Assert.Equal(ArgMax(fused), ArgMax(chunked));
        Assert.True(maxDiff < 1.5, $"fused vs chunked-materialized prefill logits differ by {maxDiff:F3}");
    }

    /// <summary>ForwardBatch over four sequences (prefill, then decode steps) tracks the
    /// single-sequence forward of each sequence on its own tokens. The two paths use
    /// different kernels (paged F32 K/V against the F16 cache, batch-size dependent
    /// quantized matmul), which measured max|dlogit| 0.3-1.4 on the 8B; a sequence reading
    /// another's state, or state from the wrong step, is off by far more. A q4_0 lane checks
    /// the batched choices instead (see the comment at the reference loop).</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void ForwardBatch_FourSequences_MatchesPerSequenceForward()
    {
        var model = _fixture.Model;
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        const int blockSize = 16;
        const int decodeSteps = 8;
        int n = 4;

        var prompts = Prompts.Take(n).Select(p => renderer.RenderToTokens(model.Tokenizer, model.Config?.ChatTemplate,
            new List<ChatMessage> { new() { Role = "user", Content = p } },
            model.Config?.Architecture ?? string.Empty, addGenerationPrompt: true, tools: null, enableThinking: false).ToArray()).ToArray();

        model.ResetKVCache();
        var pool = new TensorSharp.Runtime.Paged.BlockPool(numBlocks: 64, blockSize: blockSize,
            blockByteSize: model.ComputeKVBlockByteSize(blockSize));
        var seqs = new SequenceState[n];
        for (int s = 0; s < n; s++)
        {
            seqs[s] = new SequenceState($"fb-{s}-{Guid.NewGuid():N}", prompts[s], decodeSteps + 1, blockSize, SamplingConfig.Greedy);
            int blocks = (prompts[s].Length + decodeSteps + blockSize) / blockSize + 1;
            foreach (var b in pool.AllocateNew(blocks)) seqs[s].BlockTable.AppendBlock(b);
        }

        // Batched: one prefill step for all four, then decode steps feeding each row's argmax.
        var batchedLogits = new List<float[]>[n];
        var chosen = new List<int>[n];
        for (int s = 0; s < n; s++) { batchedLogits[s] = new List<float[]>(); chosen[s] = new List<int>(); }
        try
        {
            var step = BuildContext(seqs, prompts, blockSize);
            var outs = batched.ForwardBatch(step);
            for (int s = 0; s < n; s++)
            {
                batchedLogits[s].Add((float[])outs[s].Clone());
                seqs[s].AdvanceComputedTokens(prompts[s].Length);
            }
            for (int d = 0; d < decodeSteps; d++)
            {
                var toks = new int[n][];
                for (int s = 0; s < n; s++)
                {
                    int tok = ArgMax(batchedLogits[s][^1]);
                    chosen[s].Add(tok);
                    toks[s] = new[] { tok };
                }
                outs = batched.ForwardBatch(BuildContext(seqs, toks, blockSize));
                for (int s = 0; s < n; s++)
                {
                    batchedLogits[s].Add((float[])outs[s].Clone());
                    seqs[s].AdvanceComputedTokens(1);
                }
            }
        }
        finally
        {
            foreach (var s in seqs) batched.OnSequenceReleased(s.RequestId);
        }

        // Reference: each sequence alone through Forward, on the tokens the batch chose.
        // The logit bound is the batched kernels' own drift (up to 1.4 on f16 and q8_0).
        // On a q4_0 cache the reference attends over 4-bit K/V - a step is 1/8 of its
        // block's largest value, against 1/127 for q8_0 - while ForwardBatch reads float32
        // paged K/V, so the reference alone sits 2-5 logits off (max|dlogit| 5.14 measured,
        // argmax unchanged) and no logit bound tells that from a state bug. That lane
        // checks the choice instead: each batched argmax is the reference's, or a token the
        // reference ranks within NearTopLogitGap of its top, which a sequence attending over
        // another's state, or the wrong step's, does not manage.
        bool quantizationBound = model.KvCacheDtype == KvCacheDtype.Q4_0;
        _output.WriteLine($"kv={model.KvCacheDtype.ToShortString()}: " +
            (quantizationBound ? "near-top choice check" : "max|dlogit| <= 4.0"));
        var failures = new List<string>();
        for (int s = 0; s < n; s++)
        {
            model.ResetKVCache();
            var reference = new List<float[]> { (float[])model.Forward(prompts[s]).Clone() };
            foreach (int tok in chosen[s])
                reference.Add((float[])model.Forward(new[] { tok }).Clone());

            var line = new StringBuilder($"[seq {s}] ");
            for (int t = 0; t < reference.Count; t++)
            {
                double diff = MaxAbsDiff(reference[t], batchedLogits[s][t]);
                int choice = ArgMax(batchedLogits[s][t]);
                bool top = ArgMax(reference[t]) == choice;
                line.Append($"{t}:{diff:F2}{(top ? "" : "!")} ");
                if (!quantizationBound && diff > 4.0)
                    failures.Add($"seq {s} step {t}: argmax {(top ? "same" : "differs")}, max|dlogit|={diff:F2}");
                float gap = reference[t][ArgMax(reference[t])] - reference[t][choice];
                if (quantizationBound && gap > NearTopLogitGap)
                    failures.Add($"seq {s} step {t}: batched chose a token the reference ranks {gap:F2} below its top");
            }
            _output.WriteLine(line.ToString());
        }
        model.ResetKVCache();
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>A sequence decoding in the batched step that grows the model-owned paged
    /// K/V pool (another request's prefill brings a block id past its capacity) keeps its
    /// attention history. The grow used to replace every layer's buffer before copying
    /// from it, so that sequence attended over zeros: max|dlogit| 6.95 against the
    /// single-sequence forward and a wrong greedy token, where the batched kernels
    /// otherwise stay within about 1.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void ForwardBatch_PagedPoolGrowth_KeepsLiveSequenceHistory()
    {
        var model = _fixture.Model;
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        int[][] prompts = Prompts.Select(p => Render(model, renderer, p)).ToArray();
        const int blockSize = 16;
        // Block ids far past anything the pool has held, so the second step must grow it.
        const int farBlockId = 1024;

        model.ResetKVCache();
        int token = ArgMax(model.Forward(prompts[0]));
        float[] reference = (float[])model.Forward(new[] { token }).Clone();
        model.ResetKVCache();

        var seqs = prompts.Select((p, i) => new SequenceState($"grow-{i}-{Guid.NewGuid():N}", p, 4, blockSize, SamplingConfig.Greedy)).ToArray();
        int nextId = farBlockId;
        for (int s = 0; s < seqs.Length; s++)
        {
            int blocks = (prompts[s].Length + 1 + blockSize - 1) / blockSize;
            for (int b = 0; b < blocks; b++)
                seqs[s].BlockTable.AppendBlock(new TensorSharp.Runtime.Paged.KvBlock(s == 0 ? b : nextId++));
        }

        float[] batchedDecode;
        try
        {
            batched.ForwardBatch(BuildContext(new[] { seqs[0] }, new[] { prompts[0] }, blockSize));
            seqs[0].AdvanceComputedTokens(prompts[0].Length);

            var tokens = new int[seqs.Length][];
            tokens[0] = new[] { token };
            for (int s = 1; s < seqs.Length; s++) tokens[s] = prompts[s];
            batchedDecode = batched.ForwardBatch(BuildContext(seqs, tokens, blockSize))[0];
        }
        finally
        {
            foreach (var seq in seqs) batched.OnSequenceReleased(seq.RequestId);
            model.ResetKVCache();
        }

        double diff = MaxAbsDiff(reference, batchedDecode);
        _output.WriteLine($"decode after pool grow: max|dlogit|={diff:F2} argmax single={ArgMax(reference)} batched={ArgMax(batchedDecode)}");
        Assert.True(diff < 3.0, $"the decoding sequence lost its K/V history when the paged pool grew (max|dlogit| {diff:F2})");
        Assert.Equal(ArgMax(reference), ArgMax(batchedDecode));
    }

    /// <summary>Radix pages on the real weights, under whatever K/V dtype the lane loads
    /// (KV_CACHE_DTYPE). A Nemotron-H page carries the 24 Mamba2 layers' state (~99 MiB on
    /// the 8B) only where it is a restore point; every other page holds K/V rows alone. A
    /// new chat sharing the first request's public prefix resumes from the restore point -
    /// rebuilt from K/V-only pages plus the one carrying the state - and decodes exactly
    /// what a cold prefill of its prompt decodes. The radix budget charges the real slabs.
    /// A q8_0 / q4_0 cache used to keep every request off this path entirely (no linear
    /// K/V migration, so no N=1 fused route and no pages).</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public async Task SharedPrefixPages_CarryStateOnlyAtRestorePoints_AndResumeExactly()
    {
        var model = _fixture.Model;
        var caps = ExecutionCapabilities.FromModel(model);
        _output.WriteLine($"kv={model.KvCacheDtype.ToShortString()} {caps.Describe()}");
        if (!LaneMigratesQuantizedKv(model))
        {
            Assert.False(caps.SupportsLinearKvMigration, "a quantized cache migrated where the N=1 decode cannot append to it");
            _output.WriteLine("this lane keeps a quantized cache on the batched route: no N=1 path, no pages");
            return;
        }
        Assert.True(caps.SupportsLinearKvMigration, "the N=1 fused path is unavailable on this K/V dtype");
        // And the prefix cache sees the batched route's end states, as it does for f16.
        var prefixCaps = Assert.IsAssignableFrom<TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel>(model)
            .GetPrefixCacheCapabilities();
        _output.WriteLine($"prefix cache: endState={prefixCaps.EndState} pagedEndStates={prefixCaps.PagedEndStates} pages={prefixCaps.Pages}");
        Assert.Equal(TensorSharp.Runtime.Scheduling.PrefixCache.EndStateSupport.DonateOnly, prefixCaps.EndState);
        Assert.True(prefixCaps.PagedEndStates);

        var sb = new StringBuilder();
        for (int i = 0; i < 60; i++)
            sb.Append("Rule ").Append(i).Append(": the archive keeps ledger ").Append(i * 7 + 3).Append(" on shelf ").Append(i % 9).Append(".\n");
        int[] shared = model.Tokenizer.Encode(sb.ToString(), addSpecial: true).ToArray();
        Assert.True(shared.Length > 3 * 256, $"need more than 768 shared tokens, have {shared.Length}");
        int[] Ask(string q) => shared.Concat(model.Tokenizer.Encode("\nQuestion: " + q + "\nAnswer:", addSpecial: false)).ToArray();
        int[] promptA = Ask("Which shelf holds ledger 52?");
        int[] promptB = Ask("What does rule 12 keep?");
        const int maxNew = 24;
        SchedulerConfig Config(bool prefixCaching) => new()
        {
            MaxNumBatchedTokens = 4096,
            MaxNumRunningSequences = 4,
            NumBlocks = 64,
            BlockSize = 256,
            EnablePrefixCaching = prefixCaching,
            StopRepetition = false,
        };

        long fullPage = model.ComputeKVBlockByteSize(256);
        long kvPage = model.ComputeKVBlockByteSizeWithoutRecurrentState(256);
        _output.WriteLine($"page bytes: full={fullPage >> 20} MiB, K/V-only={kvPage >> 10} KiB");
        Assert.True(kvPage < fullPage / 4);

        List<int> warmB;
        int reused;
        model.ResetKVCache();
        using (var engine = new InferenceEngine(model, Config(prefixCaching: true), NullLogger.Instance))
        {
            await RunScoped(engine, "a", promptA, maxNew, "chat-a", shared.Length);
            var tree = engine.RadixCache!.Tree;
            int full = 0, kvOnly = 0;
            long slabs = 0;
            for (int id = 0; id < engine.Pool.NumBlocks; id++)
            {
                var block = engine.Pool.GetBlock(id);
                if (!tree.TryGetBlockOwner(block, out _)) continue;
                long len = engine.Pool.Storage.SlabLength(id);
                slabs += len;
                if (len == fullPage && block.IsRestorablePrefixEnd) full++;
                else if (len == kvPage && !block.IsRestorablePrefixEnd) kvOnly++;
                else Assert.Fail($"page {id}: {len} bytes, restorable={block.IsRestorablePrefixEnd}");
            }
            _output.WriteLine($"pages: {full} with state, {kvOnly} K/V-only; charged {tree.Cached.HostKv >> 20} MiB of {slabs >> 20} MiB slabs");
            Assert.True(full >= 1 && kvOnly >= 1);
            Assert.Equal(slabs, tree.Cached.HostKv);

            var b = await RunScoped(engine, "b", promptB, maxNew, "chat-b", shared.Length);
            warmB = b.OutputTokens.ToList();
            reused = b.PrefixCacheReusedTokens;
        }

        model.ResetKVCache();
        List<int> coldB;
        using (var cold = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance))
            coldB = (await RunScoped(cold, "b-cold", promptB, maxNew, "chat-cold", 0)).OutputTokens.ToList();
        model.ResetKVCache();

        _output.WriteLine($"B reused {reused} of {promptB.Length} prompt tokens");
        _output.WriteLine($"warm: {Show(Decode(model, warmB))}");
        _output.WriteLine($"cold: {Show(Decode(model, coldB))}");
        Assert.True(reused >= 256 && reused % 256 == 0, $"B reused {reused} tokens");
        Assert.Equal(coldB, warmB);
    }

    /// <summary>The linear-to-paged migration a second request triggers moves the solo
    /// sequence's K/V into the batched path's float32 paged arrays. On a q8_0 / q4_0 cache
    /// that read dequantizes (it used to decline, which kept every request of such a cache
    /// off the N=1 fused path); a batched decode on the migrated history then tracks the
    /// decode the linear cache itself gives, as the batched kernels do for f16.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public void MigratedLinearHistory_DecodesLikeTheLinearCache()
    {
        var model = _fixture.Model;
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        if (!LaneMigratesQuantizedKv(model))
        {
            Assert.False(batched.SupportsLinearKVMigration, "a quantized cache migrated where the N=1 decode cannot append to it");
            _output.WriteLine($"kv={model.KvCacheDtype.ToShortString()}: this lane keeps a quantized cache on the batched route");
            return;
        }
        Assert.True(batched.SupportsLinearKVMigration, $"no linear migration on a {model.KvCacheDtype.ToShortString()} cache");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        int[] prompt = Render(model, renderer, Prompts[0] + " Explain the steps in two sentences.");
        const int blockSize = 16;

        model.ResetKVCache();
        int token = ArgMax(model.Forward(prompt));
        float[] linearDecode = (float[])model.Forward(new[] { token }).Clone();

        model.ResetKVCache();
        model.Forward(prompt);
        var seq = new SequenceState($"mig-{Guid.NewGuid():N}", prompt, 4, blockSize, SamplingConfig.Greedy);
        for (int b = 0; b < (prompt.Length + 1 + blockSize - 1) / blockSize; b++)
            seq.BlockTable.AppendBlock(new TensorSharp.Runtime.Paged.KvBlock(b));
        seq.AdvanceComputedTokens(prompt.Length);
        float[] batchedDecode;
        try
        {
            Assert.True(batched.TryMigrateLinearKVToPaged(seq, blockSize), "migration refused");
            batchedDecode = batched.ForwardBatch(BuildContext(new[] { seq }, new[] { new[] { token } }, blockSize))[0];
        }
        finally
        {
            batched.OnSequenceReleased(seq.RequestId);
            model.ResetKVCache();
        }

        double diff = MaxAbsDiff(linearDecode, batchedDecode);
        _output.WriteLine($"kv={model.KvCacheDtype.ToShortString()} prompt={prompt.Length} decode after migration: " +
            $"max|dlogit|={diff:F2} argmax linear={ArgMax(linearDecode)} batched={ArgMax(batchedDecode)}");
        Assert.True(diff < 3.0, $"the migrated history decodes differently (max|dlogit| {diff:F2})");
        Assert.Equal(ArgMax(linearDecode), ArgMax(batchedDecode));
    }

    /// <summary>Whether this lane migrates its K/V cache to the batched route: a float cache
    /// always does, a q8_0 / q4_0 one only where the N=1 decode is validated to append to and
    /// read it in place (ggml-metal). ggml-cpu aborts on that append, and ggml-cuda and
    /// ggml-vulkan have not run it with this model, so there a quantized cache keeps the
    /// batched route, by design.</summary>
    private static bool LaneMigratesQuantizedKv(ModelBase model)
        => !model.KvCacheDtype.IsBlockQuantized()
           || TestGates.PinnedGgmlBackend is BackendType.GgmlMetal;

    private static async Task<SequenceState> RunScoped(InferenceEngine engine, string id, int[] prompt, int maxNew,
        string scope, int publicTokens)
    {
        var seq = new SequenceState(id, prompt, maxNew, 256, SamplingConfig.Greedy,
            sharedPrefixTokens: publicTokens, cacheScope: scope);
        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromMinutes(5));
        Assert.NotEqual(SequenceStatus.FinishedError, completion.Status);
        return seq;
    }

    private static BatchedForwardContext BuildContext(SequenceState[] seqs, int[][] tokens, int blockSize)
    {
        var ctx = new BatchedForwardContext
        {
            Sequences = seqs.ToList(),
            NumScheduledTokens = tokens.Select(t => t.Length).ToList(),
            QueryStartLoc = new List<int> { 0 },
            Positions = new List<int>(),
            SlotMapping = new List<int>(),
            BlockTables = new int[seqs.Length][],
            OverrideFlatTokens = tokens.SelectMany(t => t).ToArray(),
        };
        int total = 0;
        for (int s = 0; s < seqs.Length; s++)
        {
            int start = seqs[s].NumComputedTokens;
            var ids = new int[seqs[s].BlockTable.NumBlocks];
            for (int b = 0; b < ids.Length; b++) ids[b] = seqs[s].BlockTable.Blocks[b].Id;
            ctx.BlockTables[s] = ids;
            for (int i = 0; i < tokens[s].Length; i++)
            {
                int pos = start + i;
                ctx.Positions.Add(pos);
                ctx.SlotMapping.Add(ids[pos / blockSize] * blockSize + pos % blockSize);
            }
            total += tokens[s].Length;
            ctx.QueryStartLoc.Add(total);
            ctx.MaxQueryLen = Math.Max(ctx.MaxQueryLen, tokens[s].Length);
            ctx.MaxSeqLen = Math.Max(ctx.MaxSeqLen, start + tokens[s].Length);
        }
        return ctx;
    }

    private static async Task<List<int>> Generate(InferenceEngine engine, int[] tokens, string reqId, int maxNew, Action<int> onToken)
    {
        var seq = new SequenceState(reqId, tokens, maxNew, 256, SamplingConfig.Greedy);
        var handle = engine.SubmitRequest(seq);
        var outs = new List<int>();
        await foreach (var tok in handle.Tokens.ReadAllAsync())
        {
            outs.Add(tok);
            onToken(outs.Count);
        }
        await handle.Completion;
        return outs;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static int FirstDiff(List<int> a, List<int> b)
    {
        int n = Math.Min(a.Count, b.Count);
        for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
        return n;
    }

    private static string Decode(ModelBase model, List<int> t) => model.Tokenizer.Decode(t);
    private static string Show(string s) => (s.Length <= 160 ? s : s.Substring(0, 160) + "...").Replace("\n", "\\n");
}
