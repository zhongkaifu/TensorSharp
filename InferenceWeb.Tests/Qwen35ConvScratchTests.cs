// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The whole-model decode keeps each cache's GDN conv state in an unmanaged scratch
// (_fdConvScratch), swapped with the cache. It used to be allocated only when the
// decode descriptors were first built, for whichever cache was active then. A cache
// that had never decoded before that moment - the model's original primary cache,
// saved aside when a per-request holder was bound first - kept a null scratch, and
// its first decode after being restored wrote the reseeded conv state through a null
// pointer (NullReferenceException in TryFullModelDecodeCore). On the engine that is a
// fresh model whose first scheduled step is a concurrent one, followed later by a
// solo decode on the primary cache.
//
//   TS_TEST_MODEL_DIR=~/work/models/Qwen TS_TEST_GGML_BACKEND=metal|cuda
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen35ConvScratchTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string ModelPattern = "qwen3.5-9b-iq4_xs|qwen3.5-9b-q8_0";
    private readonly ITestOutputHelper _output;

    public Qwen35ConvScratchTests(ITestOutputHelper output) { _output = output; }

    [ModelFact(EnvModelDir, ModelPattern)]
    public void APrimaryCacheThatNeverDecoded_DecodesAfterAHolderDid()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = dir == null ? null : TestGates.FindGguf(dir, ModelPattern);
        if (modelPath == null) { _output.WriteLine("no qwen3.5-9b model; skipping"); return; }
        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant() switch
        {
            "metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.GgmlCuda,
            _ => BackendType.GgmlCpu,
        };
        if (backend is not (BackendType.GgmlMetal or BackendType.GgmlCuda))
        {
            _output.WriteLine($"{backend} has no whole-model decode; not applicable");
            return;
        }

        using var model = ModelBase.Create(modelPath, backend);
        var paged = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        int[] prompt = model.Tokenizer.Encode(
            "The lighthouse keeper counted the ships each evening and wrote their names in a ledger.",
            addSpecial: false).Take(16).ToArray();

        // A per-request holder is bound before the primary cache ever decoded, and the
        // holder's decode is the first one the model runs.
        Assert.True(paged.BindSequenceCache("first-holder"));
        float[] holderPrefill = (float[])model.Forward(prompt).Clone();
        int next = ArgMax(holderPrefill);
        float[] holderDecode = (float[])model.Forward(new[] { next }).Clone();

        // Back to the primary cache: the same prompt and token decode exactly as they did
        // on the holder.
        paged.RestorePrimaryCache();
        model.ResetKVCache();
        float[] primaryPrefill = (float[])model.Forward(prompt).Clone();
        Assert.Equal(next, ArgMax(primaryPrefill));
        float[] primaryDecode = (float[])model.Forward(new[] { next }).Clone();

        float worst = 0;
        for (int i = 0; i < holderDecode.Length; i++) worst = Math.Max(worst, Math.Abs(holderDecode[i] - primaryDecode[i]));
        _output.WriteLine($"primary decode vs holder decode: max |dlogit| {worst:G4}");
        Assert.Equal(ArgMax(holderDecode), ArgMax(primaryDecode));
        Assert.True(worst < 1e-2f, $"the primary's decode differs from the holder's by {worst}");
        paged.OnSequenceReleased("first-holder");
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
