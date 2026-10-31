// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The measurements behind the direct-CUDA model-level tests (DirectCudaPrefillModelTests): one
// model's logits along a teacher-forced trajectory (single-pass or chunked prefill, then decode
// steps), a K/V + recurrent-state snapshot restore, a 16-row verify window against sequential
// decoding, and the continuous-batching engine's solo and concurrent greedy streams. They return
// numbers; the tests decide what is acceptable.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

internal static class DirectCudaModelChecks
{
    /// <summary>A deterministic prompt of <paramref name="length"/> byte tokens (ids 3..252, so
    /// no BOS/EOS), varied enough that attention and routing see different rows.</summary>
    public static int[] Prompt(int length, int salt = 0)
        => Enumerable.Range(0, length).Select(i => 3 + (i * 37 + i / 7 + salt * 101) % 250).ToArray();

    /// <summary>Tokens fed after the prompt, one per decode step (teacher forcing: both sides of a
    /// comparison see the same inputs, wherever their argmax would go).</summary>
    public static int[] Forced(int steps, int salt = 0)
        => Enumerable.Range(0, steps).Select(s => 3 + (s * 53 + salt * 17) % 250).ToArray();

    /// <summary>
    /// Logits after the prompt (fed in <paramref name="chunks"/> consecutive forwards, or in one
    /// when null) and after each forced decode token. Starts from an empty cache.
    /// </summary>
    public static List<float[]> Trajectory(ModelBase model, int[] prompt, int[] forced, int[] chunks = null)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { PrefillLogits(model, prompt, chunks) };
        foreach (int token in forced)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows;
    }

    /// <summary>The last prompt row's logits, feeding the prompt in the given chunk lengths (the
    /// remainder, if any, as a final chunk).</summary>
    public static float[] PrefillLogits(ModelBase model, int[] prompt, int[] chunks = null)
    {
        if (chunks == null || chunks.Length == 0)
            return (float[])model.ForwardPrompt(prompt).Clone();
        float[] logits = null;
        int at = 0;
        foreach (int length in chunks.Append(prompt.Length))
        {
            int take = Math.Min(length, prompt.Length - at);
            if (take <= 0) continue;
            logits = (float[])model.ForwardPrompt(prompt[at..(at + take)]).Clone();
            at += take;
        }
        return logits;
    }

    public sealed record GdnRingState(int Ring, int[] Phases);

    /// <summary>
    /// A Qwen3.5 model's Gated DeltaNet conv ring bookkeeping (the private host write index of each
    /// recurrent layer, and the ring size conv_kernel - 1), read by reflection; null for other
    /// models. On direct CUDA every path that writes the device ring advances the index by the
    /// tokens it wrote (TryRunCudaNativeGdnPacked, the decode/prefill graph replays), so after a
    /// reset and N tokens every layer's phase is N mod ring.
    /// </summary>
    public static GdnRingState GdnRingPhases(ModelBase model)
    {
        if (model is not Qwen35Model)
            return null;
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        Type type = typeof(Qwen35Model);
        var phases = (int[])type.GetField("_convStateWriteIdx", flags)?.GetValue(model);
        var recurrent = (bool[])type.GetField("_isRecurrent", flags)?.GetValue(model);
        object kernel = type.GetField("_convKernel", flags)?.GetValue(model);
        if (phases == null || recurrent == null || kernel is not int convKernel || convKernel <= 1)
            throw new InvalidOperationException("Qwen35Model's GDN ring fields (_convStateWriteIdx, _isRecurrent, _convKernel) were renamed: update GdnRingPhases");
        return new GdnRingState(convKernel - 1, phases.Where((_, layer) => recurrent[layer]).ToArray());
    }

    public sealed record SnapshotResult(int BlockBytes, float[][] Straight, float[][] Restored);

    /// <summary>
    /// Prefill <paramref name="prefix"/>, extract it as one K/V (+ recurrent state) block, keep
    /// going with <paramref name="suffix"/> and <paramref name="forced"/> (the straight run); then
    /// reset, inject the block and replay the same suffix and steps (the restored run). Null when
    /// the model refuses the snapshot.
    /// </summary>
    public static SnapshotResult SnapshotRestore(ModelBase model, int[] prefix, int[] suffix, int[] forced)
    {
        model.ResetKVCache();
        model.ForwardPrompt(prefix);
        if (!model.SupportsKVStateSnapshot)
            return null;
        long size = model.ComputeKVBlockByteSize(prefix.Length);
        if (size <= 0)
            return null;
        var block = new byte[size];
        if (!model.TryExtractKVBlock(0, prefix.Length, block))
            return null;
        float[][] straight = Continue(model, suffix, forced);

        model.ResetKVCache();
        if (!model.TryInjectKVBlock(0, prefix.Length, block))
            throw new InvalidOperationException("the model extracted a block it then refused to inject into an empty cache");
        float[][] restored = Continue(model, suffix, forced);
        return new SnapshotResult((int)size, straight, restored);
    }

    private static float[][] Continue(ModelBase model, int[] suffix, int[] forced)
    {
        var rows = new List<float[]> { (float[])model.ForwardPrompt(suffix).Clone() };
        foreach (int token in forced)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows.ToArray();
    }

    public sealed record VerifyResult(bool AllRows, float[][] Window, float[][] Sequential);

    /// <summary>
    /// After <paramref name="prefix"/>, a <paramref name="window"/>-row verify forward: every row's
    /// logits from <see cref="ISpeculativeTarget.SpecForward"/> when the model is a speculative
    /// target (AllRows), else the one logits row a multi-token Forward returns; against the same
    /// tokens fed one decode step at a time from a fresh prefill of the same prefix.
    /// </summary>
    public static VerifyResult VerifyWindow(ModelBase model, int[] prefix, int[] window)
    {
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        model.ForwardPrompt(prefix);
        float[][] rows;
        bool allRows = model is ISpeculativeTarget;
        if (model is ISpeculativeTarget target)
        {
            target.SpecEnsureCapacity(prefix.Length + window.Length);
            var flat = new float[window.Length * vocab];
            target.SpecForward(window, null, flat, allLogitsRows: true);
            rows = Enumerable.Range(0, window.Length).Select(r => flat.AsSpan(r * vocab, vocab).ToArray()).ToArray();
        }
        else
        {
            rows = new[] { (float[])model.Forward(window).Clone() };
        }

        model.ResetKVCache();
        model.ForwardPrompt(prefix);
        var sequential = window.Select(t => (float[])model.Forward(new[] { t }).Clone()).ToArray();
        if (!allRows)
            sequential = new[] { sequential[^1] };
        return new VerifyResult(allRows, rows, sequential);
    }

    public sealed record EngineStreams(List<int>[] Solo, List<int>[] Concurrent);

    /// <summary>
    /// Greedy streams of <paramref name="prompts"/> through the continuous-batching engine: each
    /// prompt alone, then all of them submitted together (the batched path where the model has
    /// one, TS_SCHED_DISABLE_BATCHED unset). Prefix caching is off so every request prefills.
    /// </summary>
    public static async Task<EngineStreams> SoloAndConcurrent(ModelBase model, int[][] prompts, int newTokens)
    {
        const int blockSize = 16;
        var config = new SchedulerConfig
        {
            MaxNumBatchedTokens = 2048, MaxNumRunningSequences = Math.Max(4, prompts.Length), MaxPrefillChunkSize = 512,
            SoloPrefillChunkSize = 512, NumBlocks = 256, BlockSize = blockSize, EnablePrefixCaching = false,
            DecodeQuantumTokens = blockSize,
        };
        int serial = 0;
        async Task<List<int>> Run(InferenceEngine engine, int[] prompt)
        {
            var seq = new SequenceState($"dc-{System.Threading.Interlocked.Increment(ref serial)}-{Guid.NewGuid():N}",
                prompt, newTokens, blockSize, SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(seq);
            var tokens = new List<int>();
            await foreach (int token in handle.Tokens.ReadAllAsync())
                tokens.Add(token);
            await handle.Completion;
            return tokens;
        }

        model.ResetKVCache();
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var solo = new List<int>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            solo[i] = await Run(engine, prompts[i]);
        var concurrent = await Task.WhenAll(prompts.Select(p => Run(engine, p)));
        return new EngineStreams(solo, concurrent);
    }

    /// <summary>
    /// Teacher-forced rows of <paramref name="prompts"/> run TOGETHER through the model's batched
    /// entry point (IBatchedPagedModel.ForwardBatch, what the continuous-batching engine runs for
    /// two or more sequences, and for one on families without a single-sequence fused path): one
    /// prefill batch holding every prompt, then one decode batch per forced token. Block ids come
    /// from one fresh pool per call, so the sequences' recurrent-state slots (a sequence's first
    /// block id) are distinct within a call and reused by the next call, as the engine reuses
    /// freed blocks. Row 0 of each result is the prompt's last row.
    /// </summary>
    public static List<float[]>[] BatchedTrajectories(ModelBase model, int[][] prompts, int[][] forced)
    {
        const int blockSize = 16;
        var batched = (IBatchedPagedModel)model;
        var pool = new BlockPool(numBlocks: 256, blockSize: blockSize, blockByteSize: 1);
        var sequences = new SequenceState[prompts.Length];
        var rows = new List<float[]>[prompts.Length];
        for (int s = 0; s < prompts.Length; s++)
        {
            sequences[s] = new SequenceState($"fb-{s}-{Guid.NewGuid():N}", prompts[s], 256, blockSize, SamplingConfig.Greedy);
            int blocks = (prompts[s].Length + forced[s].Length + blockSize) / blockSize + 1;
            foreach (var block in pool.AllocateNew(blocks))
                sequences[s].BlockTable.AppendBlock(block);
            rows[s] = new List<float[]>();
        }

        BatchedForwardContext Context(int[] counts)
        {
            var queryStart = new List<int> { 0 };
            var positions = new List<int>();
            var slotMapping = new List<int>();
            var tables = new int[sequences.Length][];
            var tokens = new List<int>();
            int total = 0, maxQuery = 0, maxSeq = 0;
            for (int s = 0; s < sequences.Length; s++)
            {
                SequenceState state = sequences[s];
                int start = state.NumComputedTokens;
                var ids = new int[state.BlockTable.NumBlocks];
                for (int b = 0; b < ids.Length; b++)
                    ids[b] = state.BlockTable.Blocks[b].Id;
                tables[s] = ids;
                for (int t = 0; t < counts[s]; t++)
                {
                    int pos = start + t;
                    positions.Add(pos);
                    slotMapping.Add(ids[pos / blockSize] * blockSize + pos % blockSize);
                    tokens.Add(state.TokenAt(pos));
                }
                total += counts[s];
                queryStart.Add(total);
                maxQuery = Math.Max(maxQuery, counts[s]);
                maxSeq = Math.Max(maxSeq, start + counts[s]);
            }
            return new BatchedForwardContext
            {
                Sequences = sequences.ToList(),
                NumScheduledTokens = counts.ToList(),
                QueryStartLoc = queryStart,
                Positions = positions,
                SlotMapping = slotMapping,
                BlockTables = tables,
                MaxQueryLen = maxQuery,
                MaxSeqLen = maxSeq,
                OverrideFlatTokens = tokens.ToArray(),
            };
        }

        var logits = batched.ForwardBatch(Context(prompts.Select(p => p.Length).ToArray()));
        for (int s = 0; s < sequences.Length; s++)
        {
            rows[s].Add((float[])logits[s].Clone());
            sequences[s].AdvanceComputedTokens(prompts[s].Length);
        }
        int steps = forced.Min(f => f.Length);
        for (int step = 0; step < steps; step++)
        {
            for (int s = 0; s < sequences.Length; s++)
                sequences[s].AppendOutputToken(forced[s][step]);
            logits = batched.ForwardBatch(Context(sequences.Select(_ => 1).ToArray()));
            for (int s = 0; s < sequences.Length; s++)
            {
                rows[s].Add((float[])logits[s].Clone());
                sequences[s].AdvanceComputedTokens(1);
            }
        }
        return rows;
    }

    /// <summary>Teacher-forced logits for <paramref name="prompt"/> followed by the first
    /// <paramref name="steps"/> tokens of <paramref name="stream"/>: row k is what the model saw
    /// before emitting stream[k].</summary>
    public static float[] LogitsBefore(ModelBase model, int[] prompt, IReadOnlyList<int> stream, int step)
    {
        model.ResetKVCache();
        float[] logits = (float[])model.ForwardPrompt(prompt).Clone();
        for (int k = 0; k < step; k++)
            logits = (float[])model.Forward(new[] { stream[k] }).Clone();
        return logits;
    }

    // ---- metrics --------------------------------------------------------------------------

    /// <summary>||a - b|| / ||b||.</summary>
    public static double RelL2(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("length mismatch");
        double err = 0, norm = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            err += d * d;
            norm += (double)b[i] * b[i];
        }
        return Math.Sqrt(err / Math.Max(norm, 1e-300));
    }

    public static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>Gap between the largest and second-largest logit.</summary>
    public static double Margin(float[] v)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float x in v)
        {
            if (x > first) { second = first; first = x; }
            else if (x > second) second = x;
        }
        return first - second;
    }

    public static double Rms(float[] v) => Math.Sqrt(v.Sum(x => (double)x * x) / v.Length);

    public static double Quantile(IEnumerable<double> values, double q)
    {
        double[] s = values.OrderBy(v => v).ToArray();
        if (s.Length == 0) return double.NaN;
        double pos = q * (s.Length - 1);
        int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
        return s[lo] + (s[hi] - s[lo]) * (pos - lo);
    }

    public static bool BitIdentical(float[] a, float[] b)
        => a.Length == b.Length
           && System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.AsSpan()));

    public static string Fmt(IEnumerable<double> values) => string.Join(" ", values.Select(v => v.ToString("E2")));
}
