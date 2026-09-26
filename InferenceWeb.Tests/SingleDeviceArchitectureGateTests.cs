// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Families whose factories never pass --tp on (DiffusionGemma, Wan, MiniMax-H3)
// used to keep the descriptor's default TensorParallel mode, so the shared gate
// stepped aside: --tp N was dropped without a word and a distributed group was
// handed to a model that would never issue its collectives. Declaring them
// SingleDevice routes both cases through the gate's warning and refusal.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Models.Architecture;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class SingleDeviceArchitectureGateTests
{
    public static TheoryData<string> SingleDeviceFamilies => new()
    {
        "diffusion-gemma",
        "wan",
        "minimax-h3",
    };

    private static ModelArchitectureDescriptor Arch(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var descriptor),
            $"architecture '{id}' is not registered");
        return descriptor;
    }

    [Theory]
    [MemberData(nameof(SingleDeviceFamilies))]
    public void FamilyWithoutMultiGpuPath_IsDeclaredSingleDevice(string id)
    {
        var arch = Arch(id);
        Assert.Equal(MultiGpuMode.SingleDevice, arch.MultiGpu);
        Assert.Contains(id, arch.MultiGpuLimitation, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [MemberData(nameof(SingleDeviceFamilies))]
    public void LocalTp_IsIgnoredWithAWarning(string id)
    {
        var arch = Arch(id);
        ITensorParallelGroup group = null;
        var captured = new StringWriter();
        TextWriter saved = Console.Error;
        int tp;
        int layerSplit;
        try
        {
            Console.SetError(captured);
            tp = TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
                arch, BackendType.GgmlCuda, 2, ref group, out layerSplit);
        }
        finally { Console.SetError(saved); }

        Assert.Equal(1, tp);
        Assert.Equal(1, layerSplit);
        Assert.Null(group);
        string warning = captured.ToString();
        Assert.Contains("--tp 2 ignored", warning, StringComparison.Ordinal);
        Assert.Contains(id, warning, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [MemberData(nameof(SingleDeviceFamilies))]
    public void DistributedGroup_IsRefused(string id)
    {
        ITensorParallelGroup group = new StubTpGroup();
        var error = Assert.Throws<NotSupportedException>(() =>
            TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
                Arch(id), BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains(id, error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("--tp-node-id/--tp-peers", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(SingleDeviceFamilies))]
    public void NoTpRequested_StaysSilent(string id)
    {
        var arch = Arch(id);
        ITensorParallelGroup group = null;
        var captured = new StringWriter();
        TextWriter saved = Console.Error;
        try
        {
            Console.SetError(captured);
            Assert.Equal(1, TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
                arch, BackendType.GgmlCuda, 1, ref group, out int layerSplit));
            Assert.Equal(1, layerSplit);
        }
        finally { Console.SetError(saved); }

        Assert.Equal(string.Empty, captured.ToString());
    }

    /// <summary>Minimal live group: the gate only reads whether one exists.</summary>
    private sealed class StubTpGroup : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
