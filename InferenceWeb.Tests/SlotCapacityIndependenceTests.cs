// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: a slot's capacity depends on its request's history (a pooled or donated slot, growth headroom, the capacity
// classes of slots sized to the device's room). A request's arithmetic must not: the same prompt in three chunks, greedy decode
// rows, an 8-row verify window and the rows after it give the same bits in a slot of any capacity - on the managed CPU executor,
// and on direct CUDA over its graphs, its per-op path (a LoRA adapter's, a layer split's: the GQA decode kernels) and a q8_0
// cache (prompt and decode rows one at a time there, the per-op path having no verify window), for the dense Qwen3.5 layout (group 4, d 256), Qwen3.8 Flash Next's (QSA selecting past its budget) and an attention-only family's (Mistral 3).
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class SlotCapacityIndependenceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-slot-capacity-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public SlotCapacityIndependenceTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int Context = 16384;

    private string Fixture(string family) => family switch
    {
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, ContextLength = Context }
            .Write(Path.Combine(_dir, "q35-dense.gguf")),
        "qwen4exp" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, contextLength: Context, qsa: true),
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", ContextLength = Context }
            .Write(Path.Combine(_dir, "mistral3.gguf")),
        _ => throw new ArgumentException(family),
    };

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>A request with <paramref name="reserve"/> tokens reserved: its prompt in three chunks, six greedy decode rows,
    /// an 8-row verify window (three rows kept) and three decode rows after it; every row's logits, and its slot's capacity.</summary>
    private static List<float[]> Run(StandardDecoderModel model, int promptLength, int reserve, out int capacity, bool verify = true)
    {
        var unified = (IUnifiedBatchModel)model;
        var batched = (IBatchedSpeculativeTarget)model;
        int vocab = model.Config.VocabSize;
        int[] prompt = Enumerable.Range(0, promptLength).Select(i => 3 + (i * 37 + i / 7) % 250).ToArray();
        Assert.Equal(0, unified.EnsureSequenceSlot("r", reserve));
        capacity = model.SlotCapacityForTests("r") ?? 0;
        var rows = new List<float[]>();
        var logits = new float[1][];
        int third = promptLength / 3;
        foreach (var (from, count) in new[] { (0, third), (third, third), (2 * third, promptLength - 2 * third) })
        {
            Assert.True(unified.TryForwardUnified(new[] { "r" }, new[] { prompt[from..(from + count)] }, new[] { from }, logits));
            rows.Add((float[])logits[0].Clone());
        }
        int length = promptLength, next = ArgMax(logits[0]);
        void Decode(int steps)
        {
            for (int t = 0; t < steps; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { "r" }, new[] { new[] { next } }, new[] { length }, logits));
                rows.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
        }
        Decode(6);
        if (verify)
        {
            int[] window = Enumerable.Range(0, 8).Select(i => i == 0 ? next : 3 + (next * 13 + i * 29) % 250).ToArray();
            var windowRows = new float[8 * vocab];
            Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { "r" }, new[] { length }, 8, window, windowRows, null));
            for (int i = 0; i < 8; i++) rows.Add(windowRows.AsSpan(i * vocab, vocab).ToArray());
            batched.CommitBatchedSpecVerify(new[] { "r" }, new[] { length }, new[] { 3 });
            length += 3;
            next = ArgMax(windowRows.AsSpan(2 * vocab, vocab).ToArray());
        }
        Decode(verify ? 3 : 11);
        model.OnSequenceReleased("r");
        return rows;
    }

    private void AssertSameRows(List<float[]> a, List<float[]> b, string what)
    {
        Assert.Equal(a.Count, b.Count);
        int apart = 0;
        for (int i = 0; i < a.Count; i++)
            if (!a[i].AsSpan().SequenceEqual(b[i])) apart++;
        _output.WriteLine($"{what}: {apart} of {a.Count} rows apart");
        Assert.Equal(0, apart);
    }

    private void Check(BackendType backend, string family, int promptLength, int smallReserve, int largeReserve, bool perOp = false)
    {
        string path = Fixture(family);
        List<float[]> Rows(int reserve, out int capacity)
        {
            using var model = (StandardDecoderModel)ModelBase.Create(path, backend);
            if (perOp) model.GraphLoweringEnabled = false;
            // A verify window runs on the graphs only: the per-op path decodes its rows one at a time instead.
            return Run(model, promptLength, reserve, out capacity, verify: !perOp);
        }
        List<float[]> small = Rows(smallReserve, out int smallCapacity);
        List<float[]> large = Rows(largeReserve, out int largeCapacity);
        _output.WriteLine($"{family} on {backend}{(perOp ? " per-op" : "")}: slots of {smallCapacity:N0} and {largeCapacity:N0} cells");
        Assert.True(largeCapacity > smallCapacity, $"the slots took {smallCapacity} and {largeCapacity} cells");
        AssertSameRows(small, large, $"{family} {smallCapacity} against {largeCapacity} cells");
    }

    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    [InlineData("mistral3")]
    public void RowsAreTheSameInSlotsOfAnyCapacity_OnTheManagedExecutor(string family)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        Check(BackendType.Cpu, family, promptLength: 300, smallReserve: 364, largeReserve: 9300);
    }

    /// <summary>On direct CUDA past 4,096 keys, where the GQA decode kernels' partitions had followed the capacity
    /// (6,144 against 16,384 cells: 96 against 256 keys a partition for the per-op path's rows).</summary>
    [CudaTheory]
    [InlineData("qwen35", "f16", false)]
    [InlineData("qwen35", "f16", true)]
    [InlineData("qwen35", "q8_0", false)]
    [InlineData("qwen4exp", "f16", false)]   // Qwen3.8 Flash Next runs on the graphs only (no per-op path)
    [InlineData("mistral3", "f16", false)]
    [InlineData("mistral3", "f16", true)]
    public void RowsAreTheSameInSlotsOfAnyCapacity_OnDirectCuda(string family, string kv, bool perOp)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(kv == "q8_0" ? KvCacheDtype.Q8_0 : KvCacheDtype.F16);
        Check(BackendType.Cuda, family, promptLength: 4600, smallReserve: 4664, largeReserve: 13600, perOp);
    }
}
