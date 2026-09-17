// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The contract conformance script (DESIGN §14.2 M2) against REAL weights: every
// reuse the family's IPrefixCacheModel offers must continue token for token like
// a cold prefill of the same tokens on the same model and backend.
//
// Opt-in, one model per process, under the Mac model lock:
//   TS_TEST_MODEL_DIR=<dir holding the GGUF>  TS_TEST_GGML_BACKEND=metal
//   flock /tmp/ts-mac-model.lock dotnet test ... --filter "FullyQualifiedName~PrefixCacheModelConformanceTests.Gemma4"
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests.PrefixCache;

[Trait("Category", "PrefixCacheModel")]
[Collection("PrefixCacheModelConformance")]
public sealed class PrefixCacheModelConformanceTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    // One pattern per family, shared by the gate and the loader (memory note
    // diffusiongemma-test-gate-was-skipping: a gate and a loader that disagree pass vacuously).
    private const string Gemma4E4B = "gemma-4-e4b-it-q8_0";
    internal const string Qwen35_9B = "qwen3.5-9b-q8_0";
    // A40: the whole Qwen 3.8 Flash Next GGUF directory, loaded as a layer split over the GPUs in
    // CUDA_VISIBLE_DEVICES with TENSORSHARP_TP_DEGREE set to their count.
    private const string EnvQwen38Dir = "TS_TEST_QWEN38_DIR";
    // The first shard of a split GGUF (the loader follows the rest).
    private const string Qwen38FlashNext = "qwen3.8-flash-next-ud-q2_k_xl-00001";
    // The small generated DeepSeek V4.1 GGUF (eng/validation/prepare-dsv41-managed-fixture.py), on CPU.
    private const string EnvDsv41FixtureDir = "TS_TEST_DSV41_FIXTURE_DIR";
    private const string Dsv41Fixture = "deepseek41-fixture";
    // The full DeepSeek V4.1 Flash checkpoint. Not run while V4.1 GPU runs are on hold.
    private const string EnvDsv41ModelDir = "TS_TEST_DSV41_MODEL_DIR";
    private const string Dsv41Flash = "deepseek-v4.1-flash";

    private readonly ITestOutputHelper _output;

    public PrefixCacheModelConformanceTests(ITestOutputHelper output) { _output = output; }

    [ModelFact(EnvModelDir, Gemma4E4B)]
    public void Gemma4_E4B_PassesTheConformanceScript_WithTheHolderPoolAttached()
    {
        using var model = (Gemma4Model)Load(Gemma4E4B);
        // Tree mode's configuration: the pool recycles every released holder, so the script's later
        // cold runs and fresh binds run on recycled holders.
        model.AttachPrefixCache(new RecordingPayloadSink());
        ConformanceSubject subject = TextSubject(model, "gemma4-e4b") with
        {
            PayloadDeviceDirty = model.IsRetainedHostDirty,
            // The fused decode writes K/V on the device and marks the host copy stale.
            ExpectDirtyDonations = true,
        };
        ConformanceReport report = PrefixCacheConformanceScript.Run(subject);

        Assert.Contains("capture", report.Ran);
        Assert.Contains("clone x2", report.Ran);
        Assert.Contains("settle then clone", report.Ran);
        Assert.Contains("truncate refused (wrapped window)", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("truncate in range", StringComparison.Ordinal));
        Assert.Contains("primary conversion", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("export/import", StringComparison.Ordinal));
        Assert.True(model.PooledHolderCount > 0, "the attached pool parked no released holder");

        // A recycled holder (the logical reset the primary gets between requests) decodes exactly.
        int pooled = model.PooledHolderCount;
        Assert.True(model.BindSequenceCache("pooled-holder"));
        Assert.Equal(pooled - 1, model.PooledHolderCount);
        int[] prompt = subject.SharedPrefix.Concat(subject.Suffix).ToArray();
        List<int> fromPool = Greedy(model, model.Forward(prompt), subject.DecodeTokens);
        model.OnSequenceReleased("pooled-holder");
        model.TrimIdleMemory();
        Assert.Equal(0, model.PooledHolderCount);
        Assert.True(model.BindSequenceCache("fresh-allocation"));
        Assert.Equal(Greedy(model, model.Forward(prompt), subject.DecodeTokens), fromPool);
        model.OnSequenceReleased("fresh-allocation");
    }

    [ModelFact(EnvModelDir, Qwen35_9B)]
    public void Qwen35_9B_PassesTheConformanceScript()
    {
        using var model = (Qwen35Model)Load(Qwen35_9B);
        model.AttachPrefixCache(new RecordingPayloadSink());
        ConformanceSubject subject = TextSubject(model, "qwen35-9b") with
        {
            PayloadDeviceDirty = model.IsRetainedDeviceAuthoritative,
            // A decoded holder's GDN state is device-resident (and its K/V dirty) on Metal.
            ExpectDirtyDonations = true,
        };
        ConformanceReport report = PrefixCacheConformanceScript.Run(subject);

        Assert.Contains("capture", report.Ran);
        Assert.Contains("clone x2", report.Ran);
        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains("settle then clone", report.Ran);
        Assert.Contains("truncate refused (Truncation=None)", report.Ran);
        Assert.Contains("primary conversion", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("batched release", StringComparison.Ordinal));
        Assert.Contains(report.Ran, r => r.StartsWith("export/import", StringComparison.Ordinal));
    }

    [ModelFact(EnvQwen38Dir, Qwen38FlashNext)]
    public void Qwen38_FlashNext_PassesTheConformanceScript()
    {
        using var model = (Qwen4ExpModel)LoadFrom(EnvQwen38Dir, Qwen38FlashNext);
        model.AttachPrefixCache(new RecordingPayloadSink());
        ConformanceReport report = PrefixCacheConformanceScript.Run(TextSubject(model, "qwen38-flash-next"));

        Assert.Contains("capture", report.Ran);
        Assert.Contains("clone x2", report.Ran);
        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains("settle then clone", report.Ran);
        Assert.Contains("truncate refused (Truncation=None)", report.Ran);
        Assert.Contains("primary conversion", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("batched release", StringComparison.Ordinal));
    }

    [ModelFact(EnvDsv41FixtureDir, Dsv41Fixture, GgmlBackend = BackendType.GgmlCpu)]
    public void DeepSeek41_ManagedFixture_PassesTheConformanceScript_OnCpu()
    {
        using var env = new ScopedEnvironment(new Dictionary<string, string>
        {
            // The managed fixture's environment (DeepSeekNativeRetentionFixtureTests), retention on.
            ["TS_DSV41_RETAINED_CACHE"] = "1", ["TS_DSV41_RETAINED_CACHE_MB"] = "2048", ["MAX_CONTEXT"] = "512",
            ["TS_DSV4_UBATCH"] = "3", ["TS_DSV4_THREADS"] = "2", ["TS_DSV41_ENGRAM_THREADS"] = "2",
            ["TS_DSV4_FA"] = "0", ["TS_DSV41_TP"] = "0",
        });
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvDsv41FixtureDir), Dsv41Fixture);
        Assert.True(path != null, "the gate admitted the fixture but the loader found none");
        using var model = new DeepSeek4Model(path, BackendType.GgmlCpu);
        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        ConformanceReport report = PrefixCacheConformanceScript.Run(new ConformanceSubject
        {
            Name = "deepseek41-fixture",
            Model = model,
            SharedPrefix = new[] { 0, 15, 32, 64, 128, 13, 254, 18, 7, 99, 42, 3, 77, 150, 201, 33 },
            Suffix = new[] { 9, 21, 85, 60, 11, 140, 2, 58 },
            DecodeTokens = 8,
            DonateAfter = 3,
            RewindTokens = 4,
            ExpectedReadiness = PrefixCacheMode.Legacy,
            // Native slots: the managed side does not measure their bytes (M5e).
            PayloadBytesKnown = false,
            Log = _output.WriteLine,
        });

        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("truncate in range", StringComparison.Ordinal));
        Assert.Contains("primary conversion", report.Ran);
        Assert.Contains(report.Ran, r => r.StartsWith("batched release", StringComparison.Ordinal));
        _output.WriteLine($"reclaims reported through the sink: {sink.Reports.Count}");
    }

    /// <summary>The same script on the full DeepSeek V4.1 Flash checkpoint (CUDA). Pending: V4.1 GPU runs are
    /// on hold, so M2 ran the fixture above instead; gated on its own variable so no lane loads it by accident.</summary>
    [ModelFact(EnvDsv41ModelDir, Dsv41Flash)]
    public void DeepSeek41_Flash_PassesTheConformanceScript()
    {
        using var env = new ScopedEnvironment(new Dictionary<string, string> { ["TS_DSV41_RETAINED_CACHE"] = "1" });
        using var model = (DeepSeek4Model)LoadFrom(EnvDsv41ModelDir, Dsv41Flash);
        model.AttachPrefixCache(new RecordingPayloadSink());
        ConformanceReport report = PrefixCacheConformanceScript.Run(
            TextSubject(model, "deepseek41-flash") with { PayloadBytesKnown = false });
        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains("primary conversion", report.Ran);
    }

    // ------------------------------------------------------------------ helpers

    internal ModelBase Load(string pattern) => LoadFrom(EnvModelDir, pattern);

    private ModelBase LoadFrom(string envVar, string pattern)
    {
        string dir = Environment.GetEnvironmentVariable(envVar);
        string path = TestGates.FindGguf(dir, pattern);
        Assert.True(path != null, $"the gate admitted '{pattern}' but no GGUF under {dir} matches it");
        _output.WriteLine($"loading {path} on {TestGates.PinnedGgmlBackend}");
        return ModelBase.Create(path, TestGates.PinnedGgmlBackend);
    }

    /// <summary>A ~600-token system prompt (past Gemma 4's 512-token window) and a ~60-token first message.</summary>
    private ConformanceSubject TextSubject(ModelBase model, string name, int decodeTokens = 8) => new()
    {
        Name = name,
        Model = model,
        SharedPrefix = model.Tokenizer.Encode(SharedPrefixText(), addSpecial: true).ToArray(),
        Suffix = model.Tokenizer.Encode(
            " First question, and please take it seriously: name three long rivers in Europe, say which" +
            " countries each one flows through, and add one memorable fact about each of them.", addSpecial: false).ToArray(),
        DecodeTokens = decodeTokens,
        DonateAfter = 3,
        RewindTokens = 4,
        ExpectedReadiness = PrefixCacheMode.Legacy,
        Log = _output.WriteLine,
    };

    internal static string SharedPrefixText()
    {
        var sb = new StringBuilder();
        sb.Append("You are a careful assistant running on a phone. Answer briefly and precisely. ");
        for (int i = 0; i < 40; i++)
        {
            sb.Append("Rule ").Append(i + 1).Append(": when the user asks about geography, prefer well-known facts, ")
              .Append("cite the continent, and keep each sentence under twenty words. ");
        }
        sb.Append("Tools: none are available in this session. ");
        return sb.ToString();
    }

    private static List<int> Greedy(ModelBase model, float[] logits, int n)
    {
        var output = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (i > 0) logits = model.Forward(new[] { output[^1] });
            int best = 0;
            for (int v = 1; v < logits.Length; v++)
                if (logits[v] > logits[best]) best = v;
            output.Add(best);
        }
        return output;
    }
}

/// <summary>Sets environment variables for one test and restores them afterwards.</summary>
internal sealed class ScopedEnvironment : IDisposable
{
    private readonly Dictionary<string, string> _previous = new();

    internal ScopedEnvironment(IReadOnlyDictionary<string, string> values)
    {
        foreach (var (key, value) in values)
        {
            _previous[key] = Environment.GetEnvironmentVariable(key);
            Environment.SetEnvironmentVariable(key, value);
        }
    }

    public void Dispose()
    {
        foreach (var (key, value) in _previous)
            Environment.SetEnvironmentVariable(key, value);
    }
}

[CollectionDefinition("PrefixCacheModelConformance", DisableParallelization = true)]
public sealed class PrefixCacheModelConformanceCollection { }
