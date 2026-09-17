// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The primary (single-stream) cache decoding for the first time AFTER a per-request
// holder already decoded: the whole-model fused decode allocated its conv scratch
// only on the model's first fused decode, and a holder brings its own, so the
// primary was left with none and its first decode wrote through a null pointer.
//
// Opt-in: TS_TEST_MODEL_DIR=<dir with Qwen3.5-9B-Q8_0.gguf> TS_TEST_GGML_BACKEND=metal
// (or cuda), under the model lock.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen35PrimaryDecodeAfterHolderTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string Qwen35_9B = "qwen3.5-9b-q8_0";
    private readonly ITestOutputHelper _output;

    public Qwen35PrimaryDecodeAfterHolderTests(ITestOutputHelper output) { _output = output; }

    [ModelFact(EnvModelDir, Qwen35_9B)]
    public void PrimaryDecodes_WhenAHolderRanTheModelsFirstFusedDecode()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir), Qwen35_9B);
        Assert.True(path != null, "the gate admitted the model but the loader found none");
        using var model = (Qwen35Model)ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        int[] prompt = model.Tokenizer.Encode("The three longest rivers in Europe are", addSpecial: true).ToArray();

        // The model's very first fused decode runs on a per-request holder.
        Assert.True(model.BindSequenceCache("holder"));
        List<int> fromHolder = Greedy(model, model.Forward(prompt), 6);
        model.OnSequenceReleased("holder");
        _output.WriteLine($"holder: {string.Join(",", fromHolder)}");

        // Then the primary decodes the same prompt.
        model.RestorePrimaryCache();
        model.ResetKVCache();
        List<int> fromPrimary = Greedy(model, model.Forward(prompt), 6);
        _output.WriteLine($"primary: {string.Join(",", fromPrimary)}");
        Assert.Equal(fromHolder, fromPrimary);
    }

    private static List<int> Greedy(ModelBase model, float[] logits, int n)
    {
        var output = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (i > 0) logits = model.Forward(new[] { output[^1] });
            output.Add(Array.IndexOf(logits, logits.Max()));
        }
        return output;
    }
}
