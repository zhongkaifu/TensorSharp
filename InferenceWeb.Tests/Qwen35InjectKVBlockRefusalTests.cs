// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen35Model.TryInjectKVBlock on the real model: every payload it refuses leaves the
// model exactly as it was. A refused block is where the executor's inject stops and the
// sequence resumes (an inject shortfall is a miss), so a refusal that had already
// overwritten GDN recurrent state in place would resume from a state that belongs to
// neither block. The trailing M-RoPE delta used to be checked only after every layer
// was written, and a delta that puts the next token at a negative position was not
// checked at all.
//
//   TS_TEST_MODEL_DIR=~/work/models/Qwen TS_TEST_GGML_BACKEND=metal|cuda|cpu
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen35InjectKVBlockRefusalTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string ModelPattern = "qwen3.5-9b-iq4_xs|qwen3.5-9b-q8_0";
    private readonly ITestOutputHelper _output;

    public Qwen35InjectKVBlockRefusalTests(ITestOutputHelper output) { _output = output; }

    [ModelFact(EnvModelDir, ModelPattern)]
    public void RefusedPayloads_LeaveTheModelUntouched_AndAGoodOneStillInjects()
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

        using var model = ModelBase.Create(modelPath, backend);
        var q35 = Assert.IsType<Qwen35Model>(model);
        _output.WriteLine($"model {System.IO.Path.GetFileName(modelPath)} on {backend}");
        int[] tokens = model.Tokenizer.Encode(
            "The lighthouse keeper counted the ships each evening and wrote their names in a ledger.",
            addSpecial: false).Take(17).ToArray();
        Assert.Equal(17, tokens.Length);
        const int n = 8;

        float[] Forward(int[] t) => (float[])model.Forward(t).Clone();
        byte[] Extract(int start, int count)
        {
            var bytes = new byte[model.ComputeKVBlockByteSize(count)];
            Assert.True(model.TryExtractKVBlock(start, count, bytes), $"extract [{start}, {start + count})");
            return bytes;
        }

        // The block for tokens [8, 16): their attention rows, the recurrent state after
        // token 16 and the delta. And the logits a straight run gives token 17.
        model.ResetKVCache();
        Forward(tokens[..(2 * n)]);
        byte[] block = Extract(n, n);
        float[] straight = Forward(new[] { tokens[2 * n] });

        // The model holds tokens [0, 8).
        model.ResetKVCache();
        Forward(tokens[..n]);
        byte[] before = Extract(0, n);
        int deltaBefore = q35.ActiveRopePositionDelta;

        var refusals = new List<(string Name, int Dest, byte[] Payload)>
        {
            ("truncated by one byte", n, block[..^1]),
            ("missing the M-RoPE delta", n, block[..^sizeof(int)]),
            ("one byte too long", n, block.Concat(new byte[1]).ToArray()),
            ("empty", n, Array.Empty<byte>()),
            ("a delta that puts the next token at a negative position", n, WithDelta(block, -(2 * n + 1))),
            ("not appended at the cache end", n - 1, block),
            ("past the cache end", n + 1, block),
        };
        foreach (var (name, dest, payload) in refusals)
        {
            Assert.False(model.TryInjectKVBlock(dest, n, payload), $"a payload {name} was accepted");
            Assert.True(before.AsSpan().SequenceEqual(Extract(0, n)), $"a payload {name} changed the model's state");
            Assert.Equal(deltaBefore, q35.ActiveRopePositionDelta);
            // Still exactly 8 tokens held.
            Assert.False(model.TryExtractKVBlock(0, n + 1, new byte[model.ComputeKVBlockByteSize(n + 1)]),
                $"a payload {name} changed the model's cached length");
            _output.WriteLine($"refused, state unchanged: {name}");
        }

        // The genuine block is still accepted after all of that and continues exactly.
        Assert.True(model.TryInjectKVBlock(n, n, block), "the genuine block was refused");
        Assert.True(block.AsSpan().SequenceEqual(Extract(n, n)), "the injected block does not read back");
        float[] injected = Forward(new[] { tokens[2 * n] });
        float worst = 0;
        for (int i = 0; i < straight.Length; i++) worst = Math.Max(worst, Math.Abs(straight[i] - injected[i]));
        _output.WriteLine($"after inject, next-token max |dlogit| vs a straight run: {worst:G4}");
        Assert.Equal(ArgMax(straight), ArgMax(injected));
    }

    private static byte[] WithDelta(byte[] block, int delta)
    {
        var copy = (byte[])block.Clone();
        BitConverter.GetBytes(delta).CopyTo(copy, copy.Length - sizeof(int));
        return copy;
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
