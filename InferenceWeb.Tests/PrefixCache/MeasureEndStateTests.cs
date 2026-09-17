// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What a Qwen 3.5 end state reports it holds must be what it holds (DESIGN §6.4.1):
// the tree's byte budgets charge MeasureEndState, and the model's own holder pool
// budgets IdleHolderBytes. The two count independently and must agree within 1%,
// or the tree and the pool would disagree about the same bytes.
//
// Opt-in: TS_TEST_MODEL_DIR=<dir with Qwen3.5-9B-Q8_0.gguf> TS_TEST_GGML_BACKEND=metal,
// under the Mac model lock.
using System;
using System.Linq;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests.PrefixCache;

[Trait("Category", "PrefixCacheModel")]
[Collection("PrefixCacheModelConformance")]
public sealed class MeasureEndStateTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;

    public MeasureEndStateTests(ITestOutputHelper output) { _output = output; }

    [ModelFact(EnvModelDir, PrefixCacheContractConformanceTests.Qwen35_9B)]
    public void Qwen35_MeasureEndState_MatchesIdleHolderBytesWithinOnePercent()
    {
        using var model = (Qwen35Model)new PrefixCacheContractConformanceTests(_output).Load(PrefixCacheContractConformanceTests.Qwen35_9B);
        IPrefixCacheModel pcm = model;
        int[] prefix = model.Tokenizer.Encode(PrefixCacheContractConformanceTests.SharedPrefixText(), addSpecial: true).ToArray();
        int[] suffix = model.Tokenizer.Encode(" Name three rivers in Europe.", addSpecial: false).ToArray();

        // A capture: host-authoritative, never bound.
        Assert.True(model.BindSequenceCache("measured"));
        model.Forward(prefix);
        Assert.True(pcm.TryCaptureCopy("measured", "pc:measure:1", out PayloadFootprint captured));
        AssertWithinOnePercent("capture", captured, model.RetainedIdleHolderBytes("pc:measure:1"));
        Assert.Equal(prefix.Length, captured.Tokens);
        Assert.Equal(0, captured.Bytes.DeviceKv);
        Assert.True(captured.Bytes.HostKv > 0 && captured.Bytes.StateSnapshot > 0, "a hybrid end state has both K/V and recurrent bytes");

        // A donation of a holder that decoded on the device.
        float[] logits = model.Forward(suffix);
        for (int i = 0; i < 3; i++) logits = model.Forward(new[] { Array.IndexOf(logits, logits.Max()) });
        int length = prefix.Length + suffix.Length + 3;
        Assert.True(pcm.TryCaptureDonate("measured", "pc:measure:2", length, out PayloadFootprint donated));
        AssertWithinOnePercent("donation", donated, model.RetainedIdleHolderBytes("pc:measure:2"));
        Assert.Equal(length, donated.Tokens);
        if (TestGates.PinnedGgmlBackend != BackendType.GgmlCpu)
            Assert.True(donated.Bytes.DeviceKv > 0, "a holder bound on a GPU backend charges its device mirrors");

        // The clone estimate is what a clone allocates.
        ResourceVector estimate = pcm.EstimateCloneBytes("pc:measure:1", prefix.Length);
        Assert.True(pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:measure:1", "clone", prefix.Length, prefix.Length)));
        long cloned = model.PrivateIdleHolderBytes("clone");
        long estimated = estimate.HostKv + estimate.StateSnapshot;
        _output.WriteLine($"clone: estimated {estimated:N0} bytes, allocated {cloned:N0}");
        Assert.True(Math.Abs(estimated - cloned) <= cloned / 100, $"clone estimate {estimated} is not within 1% of the {cloned} bytes the clone holds");

        model.OnSequenceReleased("clone");
        pcm.ReleasePayloads(new[] { "pc:measure:1", "pc:measure:2" }, ReleaseReason.Reset);
        Assert.Empty(model.RetainedPayloadKeys);
    }

    private void AssertWithinOnePercent(string what, PayloadFootprint footprint, long idleHolderBytes)
    {
        long host = footprint.Bytes.HostKv + footprint.Bytes.StateSnapshot;
        _output.WriteLine($"{what}: MeasureEndState host {host:N0} (K/V {footprint.Bytes.HostKv:N0}, state {footprint.Bytes.StateSnapshot:N0}, device {footprint.Bytes.DeviceKv:N0}); IdleHolderBytes {idleHolderBytes:N0}");
        Assert.True(idleHolderBytes > 0, $"{what}: no retained holder was measured");
        Assert.True(Math.Abs(host - idleHolderBytes) <= idleHolderBytes / 100,
            $"{what}: MeasureEndState reports {host} host bytes, IdleHolderBytes {idleHolderBytes} (more than 1% apart)");
    }
}
