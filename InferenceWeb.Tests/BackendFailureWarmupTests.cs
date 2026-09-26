// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// What the model does with a GPU backend that has died (issue #226).
///
/// A Metal command buffer that runs out of memory latches ggml-metal's sticky error,
/// and every later graph is refused. The prefill warm-up caught the exception that
/// reported it, printed "Prefill warmup skipped" and let the host start; the first
/// request then failed on its embedding lookup with "Native GGML get_rows_quant
/// failed", and the report blamed the cheapest op in the forward. The model now
/// refuses to run on a dead backend, names the failure whichever op tripped over
/// it, and the warm-up lets it through.
///
/// Nothing short of a real out-of-memory command buffer latches the native flag,
/// so the probe model stands in for it through <see cref="ModelBase.TryGetBackendFailure"/>.
/// </summary>
public sealed class BackendFailureWarmupTests : IDisposable
{
    private const string OutOfMemory =
        "ggml_metal_synchronize: error: command buffer 0 failed with status 5 | " +
        "error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)";

    private readonly string _path = Path.Combine(Path.GetTempPath(), $"backend-failure-probe-{Guid.NewGuid():N}.gguf");

    public BackendFailureWarmupTests() => WriteProbeGguf(_path);

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [Fact]
    public void WarmUpStopsWhenThePrefillWarmupKillsTheBackend()
    {
        // The decode warm-up is forward 1; the prefill warm-up's refill is forward 2.
        using var model = new DyingBackendModel(_path) { DiesOnForward = 2, ThrowsWhenDying = true };

        var ex = Assert.Throws<InvalidOperationException>(() => model.WarmUpKernels());

        Assert.Contains("cannot recover in this process", ex.Message);
        Assert.Contains("kIOGPUCommandBufferCallbackErrorOutOfMemory", ex.Message);
        // An out-of-memory says what it means and what to change.
        Assert.Contains("The GPU ran out of memory", ex.Message);
        // The op that tripped over the dead backend is kept, as the symptom.
        Assert.NotNull(ex.InnerException);
        Assert.Contains("get_rows_quant", ex.InnerException!.Message);
        // Only the decode warm-up's reset ran: resetting a dead backend would fail
        // and replace the exception that says why.
        Assert.Equal(1, model.Resets);
    }

    [Fact]
    public void WarmUpStopsWhenTheBackendDiesUnderAForwardThatReportedSuccess()
    {
        // The op that drains a dead command buffer returns success; only the
        // latched flag tells.
        using var model = new DyingBackendModel(_path) { DiesOnForward = 2, ThrowsWhenDying = false };

        var ex = Assert.Throws<InvalidOperationException>(() => model.WarmUpKernels());

        Assert.Contains("cannot recover in this process", ex.Message);
        Assert.Contains("Insufficient Memory", ex.Message);
        Assert.Null(ex.InnerException);
    }

    [Fact]
    public void AFailedPrefillWarmupOnAHealthyBackendIsStillOnlySkipped()
    {
        // A model that dislikes a dummy refill must never block startup.
        using var model = new DyingBackendModel(_path) { RefusesRefill = true };

        model.WarmUpKernels();

        Assert.False(model.Failed);
        Assert.Equal(2, model.Resets);
    }

    [Fact]
    public void AForwardOnAnAlreadyDeadBackendNamesTheFailureWithoutRunning()
    {
        using var model = new DyingBackendModel(_path) { Failed = true };

        var forward = Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1 }));
        var refill = Assert.Throws<InvalidOperationException>(() => model.ForwardRefill(new[] { 1, 1, 1 }));

        Assert.Contains("cannot recover in this process", forward.Message);
        Assert.Contains("Insufficient Memory", forward.Message);
        Assert.Contains("cannot recover in this process", refill.Message);
        Assert.Equal(0, model.Forwards);
    }

    [Fact]
    public void OnlyAnOutOfMemoryFailureCarriesTheMemoryAdvice()
    {
        using var model = new DyingBackendModel(_path)
        {
            Failed = true,
            FailureText = "ggml_metal_synchronize: error: command buffer 0 failed with status 5 | " +
                "error: Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)",
        };

        var ex = Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1 }));

        Assert.Contains("cannot recover in this process", ex.Message);
        Assert.DoesNotContain("ran out of memory", ex.Message);
        Assert.True(ModelBase.ReportsOutOfMemory(OutOfMemory));
        Assert.False(ModelBase.ReportsOutOfMemory(model.FailureText));
        Assert.False(ModelBase.ReportsOutOfMemory(null));
    }

    [Fact]
    public void AHealthyBackendRunsAndReportsNothing()
    {
        using var model = new DyingBackendModel(_path);

        Assert.Equal(DyingBackendModel.VocabSize, model.Forward(new[] { 1 }).Length);
        Assert.Equal(DyingBackendModel.VocabSize, model.ForwardRefill(new[] { 1, 1 }).Length);
        Assert.Equal(2, model.Forwards);
    }

    [Fact]
    [Trait("Category", "Bench")]
    public void TheEntryCheckCostsNothingNextToAForward()
    {
        // The real probe is one P/Invoke reading one atomic. Bound the managed side
        // of the new entry/exit checks on the healthy path, which every token pays.
        using var model = new DyingBackendModel(_path);
        int[] token = { 1 };
        for (int i = 0; i < 10_000; i++)
            model.Forward(token);

        const int iterations = 200_000;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            model.Forward(token);
        clock.Stop();

        double nsPerForward = clock.Elapsed.TotalMilliseconds * 1e6 / iterations;
        Console.WriteLine($"[bench] healthy Forward overhead (probe model): {nsPerForward:F1} ns/forward");
        // A real decode step is ~10-100 ms; 5 us would still be under 0.05%.
        Assert.True(nsPerForward < 5_000, $"{nsPerForward:F1} ns per forward");
    }

    private sealed class DyingBackendModel : ModelBase
    {
        public const int VocabSize = 8;
        private readonly float[] _logits = new float[VocabSize];

        public DyingBackendModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = VocabSize };
        }

        /// <summary>1-based forward on which the GPU dies; 0 never.</summary>
        public int DiesOnForward { get; init; }
        /// <summary>Whether the dying forward throws (a fused path failing over to a
        /// per-op op the dead backend refuses) or returns as if it succeeded.</summary>
        public bool ThrowsWhenDying { get; init; }
        public bool RefusesRefill { get; init; }
        public bool Failed { get; set; }
        public string FailureText { get; init; } = OutOfMemory;
        public int Forwards { get; private set; }
        public int Resets { get; private set; }

        internal override bool TryGetBackendFailure(out string detail)
        {
            detail = Failed ? FailureText : null;
            return Failed;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            Forwards++;
            if (RefusesRefill && tokens.Length > 1)
                throw new NotSupportedException("this model does not take a dummy refill");
            if (Forwards == DiesOnForward)
            {
                Failed = true;
                if (ThrowsWhenDying)
                    throw new InvalidOperationException(
                        "Native GGML get_rows_quant failed. ggml: ggml_metal_graph_compute: backend is in error state " +
                        "from a previous command buffer failure - recreate the backend to recover");
            }
            return _logits;
        }

        protected override void ResetKVCacheCore() => Resets++;
    }

    internal static void WriteProbeGguf(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);
        writer.Write(0UL); // tensors
        writer.Write(1UL); // metadata entries
        byte[] key = System.Text.Encoding.UTF8.GetBytes("general.architecture");
        writer.Write((ulong)key.Length);
        writer.Write(key);
        writer.Write((uint)GgufValueType.String);
        byte[] value = System.Text.Encoding.UTF8.GetBytes("probe");
        writer.Write((ulong)value.Length);
        writer.Write(value);
        writer.Write(new byte[(32 - stream.Position % 32) % 32]);
    }
}
