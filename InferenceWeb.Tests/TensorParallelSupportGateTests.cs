// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Explicit tensor-parallel and layer-placement requests are checked against
// each architecture's descriptor. Unsupported requests must fail, never turn
// into a different execution mode or silently leave extra GPUs idle.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models.Architecture;
using Xunit;

namespace InferenceWeb.Tests;

public class TensorParallelSupportGateTests
{
    private static ModelArchitectureDescriptor Arch(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var descriptor),
            $"architecture '{id}' is not registered");
        return descriptor;
    }

    private static int Resolve(ModelArchitectureDescriptor arch, BackendType backend, int tpDegree,
        ref ITensorParallelGroup group, out int layerSplit)
        => TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            arch, backend, tpDegree, ref group, out layerSplit);

    private static int Resolve(string arch, BackendType backend, int tpDegree,
        ref ITensorParallelGroup group, out int layerSplit)
        => Resolve(Arch(arch), backend, tpDegree, ref group, out layerSplit);

    [Fact]
    public void LayerSplitArchitecture_RequiresExplicitLayerSplit()
    {
        ITensorParallelGroup group = null;
        var error = Assert.Throws<NotSupportedException>(() =>
            Resolve("deepseek4", BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("--layer-split", error.Message);
        int tp = TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            Arch("deepseek4"), BackendType.GgmlCuda, 1, ref group, out int layerSplit, 2);

        // No tensor-parallel group: IsTensorParallel gates weight sharding and the
        // AllReduce machinery, none of which a layer split uses.
        Assert.Equal(1, tp);
        Assert.Null(group);
        // ...but both GPUs are used, by layers.
        Assert.Equal(2, layerSplit);
    }

    [Fact]
    public void LayerSplit_OnlyOnBackendsThatHaveSeveralDevices()
    {
        ITensorParallelGroup group = null;
        Assert.Throws<NotSupportedException>(() => TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            Arch("qwen4exp"), BackendType.GgmlCpu, 1, ref group, out _, 2));
    }

    [Fact]
    public void DistributedTpOnUnsupportedArchitecture_Throws()
    {
        // A distributed group cannot be downgraded on one node: the other nodes
        // would still be waiting on collectives this rank will never issue. A
        // different placement mode cannot satisfy the requested collectives.
        ITensorParallelGroup group = new StubTpGroup();
        var ex = Assert.Throws<NotSupportedException>(
            () => Resolve("deepseek4", BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("deepseek4", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    [InlineData("gemma4")]
    [InlineData("muse-glimmer")]
    [InlineData("muse-spark")]
    [InlineData("minimax-m3")]
    [InlineData("glm-dsa")]
    [InlineData("glm5next")]
    public void TpCapableArchitectures_AreUntouched(string arch)
    {
        ITensorParallelGroup group = null;
        Assert.Equal(2, Resolve(arch, BackendType.GgmlCuda, 2, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
    }

    [Fact]
    public void MuseSparkRetainsExplicitLayerSplitAndRejectsDistributedTp()
    {
        ITensorParallelGroup group = null;
        Assert.Equal(1, TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            Arch("muse-spark"), BackendType.GgmlCuda, 1, ref group, out int split, 5));
        Assert.Equal(5, split);
        Assert.Null(group);
        group = new StubTpGroup();
        var error = Assert.Throws<NotSupportedException>(() =>
            Resolve("muse-spark", BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("muse-spark", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OlderGlmNativeTensorParallelism_RejectsDistributedGroupBeforeModelConstruction()
    {
        var context = new ModelCreateContext(
            "/model-is-not-opened.gguf", BackendType.GgmlCuda, probe: null,
            tpDegree: 2, tpGroup: new StubTpGroup());

        var error = Assert.Throws<NotSupportedException>(() => Arch("glm-dsa").Factory(context));
        Assert.Contains("local/single-process", error.Message, StringComparison.Ordinal);
        Assert.Contains("--tp-node-id/--tp-peers", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeepSeek41NativeExecutor_RequiresExistingNestedTransportBeforeSidecarOrWeights()
    {
        var context = new ModelCreateContext(
            "/model-is-not-opened.gguf", BackendType.GgmlCuda, probe: null,
            tpDegree: 2, tpGroup: new StubTpGroup());

        var error = Assert.Throws<NotSupportedException>(() => Arch("deepseek41").ApplyNativeTunables(context));
        Assert.Contains("nested group", error.Message, StringComparison.Ordinal);
        Assert.Contains("cross-node AllReduce", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoTpRequested_IsAlwaysAPassthrough()
    {
        // The gate must not fire on ordinary single-GPU runs of the very
        // architectures it knows about.
        ITensorParallelGroup group = null;
        Assert.Equal(1, Resolve("qwen4exp", BackendType.GgmlCuda, 1, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
        Assert.Null(group);
    }

    [Fact]
    public void ArchitectureWithoutAServingContract_IsRejectedAtRegistration()
    {
        // The old gate was opt-in: a family that said nothing silently got tensor
        // parallelism. The contract is mandatory - the compiler requires the property, and
        // registration refuses a null one - so a new family must say what it supports.
        var silent = new ModelArchitectureDescriptor
        {
            Id = "brand-new-arch",
            Aliases = new[] { "brand-new-arch" },
            Factory = _ => throw new NotSupportedException("not constructed by this test"),
            Serving = null!,
        };
        var ex = Assert.Throws<InvalidOperationException>(() => ModelArchitectureRegistry.Register(silent));
        Assert.Contains("serving contract", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryGapExplainsItself()
    {
        // The reason is the whole value of a gap - it is what tells the operator why the
        // second GPU is idle, or why --n-cpu-moe is refused on this backend.
        foreach (var arch in ModelArchitectureRegistry.All)
        {
            foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
            {
                MechanismSupport support = arch.Serving[mechanism];
                if (support.Kind == SupportKind.NotApplicable)
                    Assert.False(string.IsNullOrWhiteSpace(support.Reason), $"'{arch.Id}' {mechanism}: no reason.");
                else if (support.GapsWithin(arch.Serving.GapUniverse(mechanism)).Any())
                    Assert.False(string.IsNullOrWhiteSpace(support.GapReason), $"'{arch.Id}' {mechanism}: gap without a reason.");
            }
        }
    }

    [Fact]
    public void DeclaringAGapWithoutAReasonIsRejectedAtRegistration()
    {
        // Validate() is what makes "every gap explains itself" true by construction.
        var silent = new ModelArchitectureDescriptor
        {
            Id = "silent-arch",
            Aliases = new[] { "silent-arch" },
            Factory = _ => throw new NotSupportedException(),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.On(ServingContract.AllBackends.ToArray()),
                TensorParallel = MechanismSupport.On(BackendType.GgmlCuda),   // cuda and ggml_vulkan left unexplained
                LayerSplit = MechanismSupport.None("no layer split"),
                HybridParallel = MechanismSupport.None("no hybrid"),
                ExpertOffload = MechanismSupport.NotApplicable("dense"),
            },
        };
        var ex = Assert.Throws<InvalidOperationException>(() => ModelArchitectureRegistry.Register(silent));
        Assert.Contains("does not say why", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryBuiltInArchitectureIsWellFormed()
    {
        var all = ModelArchitectureRegistry.All;
        Assert.NotEmpty(all);
        foreach (var arch in all)
        {
            Assert.NotNull(arch.Factory);
            Assert.Contains(arch.Id, arch.Aliases, StringComparer.OrdinalIgnoreCase);
            foreach (string alias in arch.Aliases)
                Assert.Same(arch, Arch(alias));
        }

        // Aliases are the routing key; two families claiming one would silently shadow.
        var aliases = all.SelectMany(a => a.Aliases).Select(a => a.ToLowerInvariant()).ToList();
        Assert.Equal(aliases.Count, aliases.Distinct().Count());
    }

    [Fact]
    public void NemotronCpuTensorParallel_IsNotUserReachable()
    {
        // NemotronModel accepts BackendType.Cpu with a caller-supplied group as a
        // model-free test seam (NemotronExpertScaleTests). Users must not reach it:
        // --tp on cpu, with or without a group, is refused before any model loads.
        ITensorParallelGroup group = null;
        var ex = Assert.Throws<NotSupportedException>(() => Resolve("nemotron_h", BackendType.Cpu, 2, ref group, out _));
        Assert.Contains("multi-device GPU backend", ex.Message);
        group = new MuseSparkTensorParallelTests.CpuGroup(2);
        Assert.Throws<NotSupportedException>(() => Resolve("nemotron_h", BackendType.Cpu, 2, ref group, out _));
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
