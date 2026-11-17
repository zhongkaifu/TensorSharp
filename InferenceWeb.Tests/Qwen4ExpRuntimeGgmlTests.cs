// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.8-Flash-Next (qwen4exp) on the shared runtime's GGML backends, on the tiny checkpoint of
/// <see cref="Qwen4ExpSyntheticModelBuilder"/>: the fixture runs whole, a one-row speculative capture, verify windows
/// of changing width and the MTP head's draft. These tests used to sit beside the removed direct-CUDA engine's tests
/// (the fixture was that engine's oracle); <see cref="Qwen4ExpRuntimeTests"/> holds the runtime to its goldens and to
/// --backend cuda.
/// </summary>
public sealed class Qwen4ExpRuntimeGgmlTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen4exp-ggml-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Qwen4ExpRuntimeGgmlTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>F16 caches for the models this test constructs on its own flow (the process-wide choice is shared).</summary>
    private static IDisposable F16Cache() => KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);

    private string Fixture(int indexerTopK = 16)
        => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"qwen4exp-{indexerTopK}.gguf"), indexerTopK);

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();
    private static readonly int[] Steps = { 17, 203, 99, 4, 150, 61 };

    /// <summary>The prompt's logits, then each decode step's.</summary>
    private static float[][] Run(ModelBase model)
    {
        var rows = new float[Steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(Prompt).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        return rows;
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheFixture_RunsOnTheRuntime_Cpu() => Smoke(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheFixture_RunsOnTheRuntime_Metal() => Smoke(BackendType.GgmlMetal);

    /// <summary>The fixture runs whole on the runtime's graphs: the in-graph PLE block and QSA past its 19-cell width.</summary>
    private void Smoke(BackendType backend)
    {
        using IDisposable kv = F16Cache();
        string path = Fixture();
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpRuntimeModel>(model);
        float[][] rows = Run(model);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(Qwen4ExpSyntheticModelBuilder.Vocab, rows[i].Length);
            Assert.All(rows[i], v => Assert.True(float.IsFinite(v)));
            double norm = Math.Sqrt(rows[i].Sum(v => (double)v * v));
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: argmax {ArgMax(rows[i])} |logits| {norm:F3}");
            Assert.True(norm > 1e-3, "degenerate logits");
            if (i > 0)
                Assert.True(RelativeError(rows[i], rows[i - 1]) > 1e-3, "the step did not change the logits");
        }
    }

    /// <summary>A prompt chunk older than a block drafter's ring asks for its last row's features only (a one-row
    /// buffer from SpeculativeExecution.PrefillStep); it gets exactly the last row of the full capture.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void ASpeculativePromptChunk_CanAskForItsLastRowOnly_Cpu()
    {
        using IDisposable kv = F16Cache();
        string path = Fixture(indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, BackendType.GgmlCpu);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        var all = new float[Prompt.Length * feat];
        spec.SpecForward(Prompt, all, new float[vocab], false);
        model.ResetKVCache();
        var last = new float[feat];
        spec.SpecForward(Prompt, last, new float[vocab], false);
        Assert.True(last.AsSpan().SequenceEqual(all.AsSpan((Prompt.Length - 1) * feat, feat)),
            "the one-row capture is not the full capture's last row");
    }

    /// <summary>The shared runtime verifies a window as one batched forward on every GGML backend, which rounds its
    /// activations unlike its decode steps (ggml_cpu: 1.4e-4 measured; Qwen4ExpModel's one-token row kernels were exact).
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Cpu() => ChangingWidths(BackendType.GgmlCpu, 2e-3);

    [GgmlFact(BackendType.GgmlMetal)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Metal() => ChangingWidths(BackendType.GgmlMetal, 2e-2);

    [GgmlFact(BackendType.GgmlCuda)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Cuda() => ChangingWidths(BackendType.GgmlCuda, 2e-2);

    /// <summary>Verify windows of changing width, some partly accepted (committed from the window's own per-row states),
    /// against a plain decode of the same tokens. Each width keeps its own persisted graph, and a wider window grows the
    /// buffers the narrower graphs captured into: those graphs must rebuild, never replay against a freed buffer.</summary>
    private void ChangingWidths(BackendType backend, double bound)
    {
        using IDisposable kv = F16Cache();
        string path = Fixture(indexerTopK: 2048);
        int[] seq = Enumerable.Range(0, 15).Select(i => (i * 29 + 5) % 250).ToArray();
        (int Width, int Keep)[] windows = { (2, 2), (4, 2), (2, 1), (4, 4), (3, 2), (4, 1), (2, 2) };
        using ModelBase model = ModelBase.Create(path, backend);
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        var plain = seq.Select(t => (float[])model.Forward(new[] { t }).Clone()).ToArray();

        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        int pos = 0;
        foreach (var (width, keep) in windows)
        {
            int[] window = seq[pos..(pos + width)];
            var all = new float[width * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            spec.SpecOnVerifyAccepted(keep - 1, width - 1);
            // A fully accepted window keeps the verify's state as it is (the executor does not rewind).
            if (keep < width)
            {
                Assert.True(spec.SpecVerifyPersistsAcceptedKv, $"window {width}/{keep} at {pos}: the verify did not commit its prefix");
                spec.SpecRewindCache(Prompt.Length + pos + keep);
            }
            for (int r = 0; r < keep; r++)
            {
                double err = RelativeError(all.AsSpan(r * vocab, vocab).ToArray(), plain[pos + r]);
                Assert.True(err <= bound, $"window {width}/{keep} at {pos}: row {r} differs from its decode by {err:E2}");
            }
            pos += keep;
        }
        double next = RelativeError(model.Forward(new[] { seq[pos] }), plain[pos]);
        _output.WriteLine($"after {windows.Length} windows: next decode relative error {next:E2}");
        Assert.True(next <= bound, $"the committed state differs from a plain decode by {next:E2}");
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheMtpFixture_DraftsOnTheRuntimeHead_Cpu() => MtpSmoke(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheMtpFixture_DraftsOnTheRuntimeHead_Metal() => MtpSmoke(BackendType.GgmlMetal);

    /// <summary>The synthetic NextN file passes the published head's validation and the runtime's head drafts from it.</summary>
    private void MtpSmoke(BackendType backend)
    {
        using IDisposable kv = F16Cache();
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, backend, draftModelPath: mtp);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
        Assert.True(head.HasDraftHead);
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        var hidden = new float[Prompt.Length * feat];
        spec.SpecForward(Prompt, hidden, new float[Prompt.Length * vocab], true);
        var rows = new float[Prompt.Length * feat];
        Array.Copy(hidden, 0, rows, feat, (Prompt.Length - 1) * feat);
        head.DraftCatchUp(Prompt, rows, 0);
        var logits = new float[vocab];
        var streams = new float[feat];
        head.DraftStep(Steps[0], hidden.AsSpan((Prompt.Length - 1) * feat, feat).ToArray(), Prompt.Length, logits, streams);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.All(streams, v => Assert.True(float.IsFinite(v)));
        _output.WriteLine($"draft argmax {ArgMax(logits)}, |streams| {Math.Sqrt(streams.Sum(v => (double)v * v)):F3}");
        Assert.True(streams.Any(v => v != 0f), "degenerate draft streams");
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private static double RelativeError(float[] actual, float[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
