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

    // ------------------------------------------------------------------ helpers

    private ModelBase Load(string pattern)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
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

[CollectionDefinition("PrefixCacheModelConformance", DisableParallelization = true)]
public sealed class PrefixCacheModelConformanceCollection { }
