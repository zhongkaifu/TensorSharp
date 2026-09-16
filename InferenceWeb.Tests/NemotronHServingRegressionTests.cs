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
            var backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").ToLowerInvariant() switch
            {
                "cuda" => BackendType.GgmlCuda,
                "metal" => BackendType.GgmlMetal,
                _ => BackendType.GgmlCpu,
            };
            _model = ModelBase.Create(path, backend);
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

    // Batched and single-sequence forwards run different kernels (paged F32 K/V against
    // the F16 cache, batch-size dependent quantized matmul): ForwardBatch tracks the
    // single-sequence logits to max|dlogit| 0.3-1.4 on the 8B, so a prompt whose top two
    // candidates are that close can legitimately differ. These four have clear answers
    // (measured: identical at every arrival pattern); a state bug changes them.
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

    /// <summary>Greedy output of requests served together - arriving at once, and
    /// arriving while the first one is already decoding - matches serving each alone.</summary>
    [ModelFact(NemotronHModelFixture.EnvDir, NemotronHModelFixture.GgufPattern)]
    public async Task ConcurrentGreedy_MatchesSerial()
    {
        var model = _fixture.Model;
        model.ResetKVCache();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        const int maxNew = 32;
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
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var serial = new List<int>[Prompts.Length];
        for (int i = 0; i < Prompts.Length; i++)
            serial[i] = await Generate(engine, model, renderer, Prompts[i], $"serial-{i}", maxNew, _ => { });

        var burst = await Task.WhenAll(Prompts.Take(4).Select((p, i) =>
            Generate(engine, model, renderer, p, $"burst-{i}", maxNew, _ => { })));

        // Staggered: the first request is decoding on the single-sequence path when the
        // others arrive, so its state has to move into the batched slot pool.
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var first = Generate(engine, model, renderer, Prompts[0], "stagger-0", maxNew,
            n => { if (n == 6) started.TrySetResult(); });
        await Task.WhenAny(started.Task, first);
        var rest = Prompts.Skip(1).Take(3).Select((p, i) =>
            Generate(engine, model, renderer, p, $"stagger-{i + 1}", maxNew, _ => { })).ToList();
        var staggered = new List<List<int>> { await first };
        staggered.AddRange(await Task.WhenAll(rest));

        // Worker pool: four clients drain all prompts, so a new request arrives whenever
        // one finishes while the others are still decoding in the batched path - the
        // arrival pattern of an HTTP benchmark at concurrency 4.
        // Every prompt twice, so the second wave lands on the Mamba2 slots and attention
        // blocks the first wave released.
        const int poolRequests = 12;
        var pooled = new List<int>[poolRequests];
        int next = -1;
        await Task.WhenAll(Enumerable.Range(0, 4).Select(async w =>
        {
            int i;
            while ((i = Interlocked.Increment(ref next)) < poolRequests)
                pooled[i] = await Generate(engine, model, renderer, Prompts[i % Prompts.Length], $"pool-{w}-{i}", maxNew, _ => { });
        }));

        var failures = new List<string>();
        for (int i = 0; i < Prompts.Length; i++)
        {
            string s = Decode(model, serial[i]);
            _output.WriteLine($"[{i}] serial   : {Show(s)}");
            for (int k = i; k < poolRequests; k += Prompts.Length)
            {
                _output.WriteLine($"[{i}] pooled#{k,-2}: {Show(Decode(model, pooled[k]))}");
                if (!serial[i].SequenceEqual(pooled[k])) failures.Add($"pooled[{k}] diverges at token {FirstDiff(serial[i], pooled[k])}");
            }
            _output.WriteLine($"[{i}] burst    : {Show(Decode(model, burst[i]))}");
            _output.WriteLine($"[{i}] staggered: {Show(Decode(model, staggered[i]))}");
            if (!serial[i].SequenceEqual(burst[i])) failures.Add($"burst[{i}] diverges at token {FirstDiff(serial[i], burst[i])}");
            if (!serial[i].SequenceEqual(staggered[i])) failures.Add($"staggered[{i}] diverges at token {FirstDiff(serial[i], staggered[i])}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

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
    /// another's state, or state from the wrong step, is off by far more.</summary>
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
                bool top = ArgMax(reference[t]) == ArgMax(batchedLogits[s][t]);
                line.Append($"{t}:{diff:F2}{(top ? "" : "!")} ");
                if (diff > 4.0)
                    failures.Add($"seq {s} step {t}: argmax {(top ? "same" : "differs")}, max|dlogit|={diff:F2}");
            }
            _output.WriteLine(line.ToString());
        }
        model.ResetKVCache();
        Assert.True(failures.Count == 0, string.Join("; ", failures));
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

    private static async Task<List<int>> Generate(InferenceEngine engine, ModelBase model, KVCachePromptRenderer renderer,
        string prompt, string reqId, int maxNew, Action<int> onToken)
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        var tokens = renderer.RenderToTokens(model.Tokenizer, model.Config?.ChatTemplate, history,
            model.Config?.Architecture ?? string.Empty, addGenerationPrompt: true, tools: null, enableThinking: false);
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
