// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Gemma-4 dense batched forward (IBatchedPagedModel.ForwardBatch: what continuous batching
// runs for two or more sequences, and for one) against the single-sequence path on the same
// tokens. Every dense Gemma-4 (E2B, E4B, 12B, 31B) takes it; the 26B-A4B MoE refuses it and
// runs per sequence. Their global layers use proportional RoPE: rope_freqs.weight
// divides every frequency pair past the first quarter by 1e30, so those pairs must not rotate.
// The batched path applied that only on GGML storages; on direct CUDA, the managed CPU backend
// and MLX it rotated every pair with the plain base frequencies, and on gemma-4-E4B its logits
// were 0.17-0.85 relative L2 from the single-sequence ones (a sequence alone included), so
// concurrent greedy streams left their solo twins far from any near-tie. The synthetic file
// (Gemma4DenseSyntheticModelBuilder) keeps what selects those routes: local and global layers
// with 256/512 head dims, E4B's rope_freqs layout (64 ones, 192 x 1e30 of 256), a sliding window
// the decode crosses, per-layer embeddings, shared-KV layers and per-layer output scales.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Gemma4DenseBatchedForwardParityTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "g4dense-batched-" + Guid.NewGuid().ToString("N"));

    public Gemma4DenseBatchedForwardParityTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void ManagedCpu_ForwardBatch_TracksTheSingleSequencePath() => Check(BackendType.Cpu);

    [CudaFact]
    public void DirectCuda_ForwardBatch_TracksTheSingleSequencePath() => Check(BackendType.Cuda);

    /// <summary>The MLX branch of the fix (MlxFusedOps.TryNeoXRoPEFlatInPlace). Not yet measured on Apple
    /// hardware, so it gets the wider GGML band; the bug's 5.2e-2..0.42 still fails it.</summary>
    [MlxFact]
    public void Mlx_ForwardBatch_TracksTheSingleSequencePath() => Check(BackendType.Mlx);

    /// <summary>Control: GGML storages already applied the factors (RoPEExWithFreqFactors).</summary>
    [Fact]
    public void PinnedGgml_ForwardBatch_TracksTheSingleSequencePath() => Check(TestGates.PinnedGgmlBackend);

    // Two prompts of different lengths, so the sequences of a batch sit at different positions,
    // then six teacher-forced decode steps: the first sequence's steps cross the 208-token sliding
    // window (its ring wraps and the window starts to mask). The prompts stay inside the window:
    // a prefill chunk longer than the window is written into the single-sequence ring with a
    // race (CopyToCacheCircular writes rows s and s + window to one slot in parallel; a separate
    // open defect), which would move the reference rather than test the batched path.
    private static readonly int[][] Prompts = { Tokens(205, seed: 3), Tokens(120, seed: 5) };
    private static readonly int[][] Forced = { new[] { 40, 41, 42, 43, 44, 45 }, new[] { 90, 91, 92, 93, 94, 95 } };

    // Measured (laptop RTX 3080): with the fix the managed CPU and direct CUDA rows are within
    // 9.2e-3, ggml_cpu within 1.0e-2 and ggml-cuda (TS_TEST_GGML_BACKEND=cuda) within 2.4e-2 (the
    // single-sequence path keeps an F16 KV cache, the batched path F32 pages); without it the
    // managed CPU and direct CUDA rows are 5.2e-2..0.42 off and the GGML rows unchanged. The GGML
    // control gets more room because ggml-cuda's own band sits close to 0.025.
    private static double Tolerance(BackendType backend) => backend is BackendType.Cpu or BackendType.Cuda ? 0.025 : 0.04;

    private void Check(BackendType backend)
    {
        string path = new Gemma4DenseSyntheticModelBuilder().Write(Path.Combine(_dir, "gemma4-dense.gguf"));
        using var model = ModelBase.Create(path, backend);
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.True(batched.BatchedForwardAvailable, "the dense Gemma-4 file must take the batched path");

        // The single-sequence reference runs per op (no fused prefill, no captured decode graph),
        // so it is the plain math the batched path must match; the direct-CUDA decode-graph replay
        // has open defects of its own (a freed V-norm weight read on replay).
        ((Gemma4Model)model).ForceUnfused = true;
        var single = new List<float[]>[Prompts.Length];
        for (int s = 0; s < Prompts.Length; s++)
        {
            model.ResetKVCache();
            single[s] = new List<float[]> { (float[])model.ForwardPrompt(Prompts[s]).Clone() };
            foreach (int t in Forced[s])
                single[s].Add((float[])model.Forward(new[] { t }).Clone());
        }

        List<float[]>[] together = BatchedRows(model, batched, Prompts, Forced);
        double tolerance = Tolerance(backend);
        var failures = new List<string>();
        for (int s = 0; s < Prompts.Length; s++)
        {
            List<float[]> alone = BatchedRows(model, batched, new[] { Prompts[s] }, new[] { Forced[s] })[0];
            foreach (var (label, rows) in new[] { ("alone", alone), ("together", together[s]) })
            {
                for (int r = 0; r < rows.Count; r++)
                {
                    double rel = RelL2(rows[r], single[s][r]);
                    _output.WriteLine($"{backend} sequence {s} ({Prompts[s].Length} tokens) {label} row {r}: relL2 {rel:E2}");
                    if (!(rel <= tolerance))
                        failures.Add($"sequence {s} {label} row {r}: relL2 {rel:E3} > {tolerance}");
                }
            }
        }
        Assert.True(failures.Count == 0, $"{backend}: ForwardBatch left the single-sequence path:\n" + string.Join("\n", failures));
    }

    /// <summary>The prompts prefilled together in one ForwardBatch, then decoded together on the
    /// forced tokens, one ForwardBatch per step; the logits row of each step per sequence.</summary>
    private static List<float[]>[] BatchedRows(ModelBase model, IBatchedPagedModel batched, int[][] prompts, int[][] forced)
    {
        const int blockSize = 16;
        var pool = new BlockPool(numBlocks: 64, blockSize: blockSize, blockByteSize: 1);
        var seqs = new SequenceState[prompts.Length];
        var rows = new List<float[]>[prompts.Length];
        for (int s = 0; s < prompts.Length; s++)
        {
            seqs[s] = new SequenceState($"g4dense-{s}-{Guid.NewGuid():N}", prompts[s], 16, blockSize, SamplingConfig.Greedy);
            foreach (var block in pool.AllocateNew((prompts[s].Length + forced[s].Length) / blockSize + 1)!)
                seqs[s].BlockTable.AppendBlock(block);
            rows[s] = new List<float[]>();
        }

        BatchedForwardContext Context(int[] counts)
        {
            var queryStart = new List<int> { 0 };
            var positions = new List<int>();
            var slots = new List<int>();
            var tables = new int[seqs.Length][];
            var tokens = new List<int>();
            int total = 0, maxQuery = 0, maxSeq = 0;
            for (int s = 0; s < seqs.Length; s++)
            {
                int start = seqs[s].NumComputedTokens;
                var ids = new int[seqs[s].BlockTable.NumBlocks];
                for (int k = 0; k < ids.Length; k++) ids[k] = seqs[s].BlockTable.Blocks[k].Id;
                tables[s] = ids;
                for (int t = 0; t < counts[s]; t++)
                {
                    int pos = start + t;
                    positions.Add(pos);
                    slots.Add(ids[pos / blockSize] * blockSize + pos % blockSize);
                    tokens.Add(seqs[s].TokenAt(pos));
                }
                total += counts[s];
                queryStart.Add(total);
                maxQuery = Math.Max(maxQuery, counts[s]);
                maxSeq = Math.Max(maxSeq, start + counts[s]);
            }
            return new BatchedForwardContext
            {
                Sequences = seqs.ToList(), NumScheduledTokens = counts.ToList(), QueryStartLoc = queryStart,
                Positions = positions, SlotMapping = slots, BlockTables = tables, MaxQueryLen = maxQuery,
                MaxSeqLen = maxSeq, OverrideFlatTokens = tokens.ToArray(),
            };
        }

        model.ResetKVCache();
        var logits = batched.ForwardBatch(Context(prompts.Select(p => p.Length).ToArray()));
        for (int s = 0; s < seqs.Length; s++)
        {
            rows[s].Add((float[])logits[s].Clone());
            seqs[s].AdvanceComputedTokens(prompts[s].Length);
        }
        for (int step = 0; step < forced[0].Length; step++)
        {
            for (int s = 0; s < seqs.Length; s++) seqs[s].AppendOutputToken(forced[s][step]);
            logits = batched.ForwardBatch(Context(seqs.Select(_ => 1).ToArray()));
            for (int s = 0; s < seqs.Length; s++)
            {
                rows[s].Add((float[])logits[s].Clone());
                seqs[s].AdvanceComputedTokens(1);
            }
        }
        return rows;
    }

    private static int[] Tokens(int count, int seed)
    {
        var tokens = new int[count];
        tokens[0] = 1;   // BOS
        uint state = (uint)seed * 2654435761u;
        for (int i = 1; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            tokens[i] = 3 + (int)(state % 250);
        }
        return tokens;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-300));
    }
}
