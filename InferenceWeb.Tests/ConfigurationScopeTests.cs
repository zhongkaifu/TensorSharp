// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// D04: configuration has a scope and an effective time. Engine-step options are one snapshot a step, changed only at step
/// boundaries; a model's placement and MoE offload are its load-time snapshot; the CLI and environment are input channels
/// resolved once by the host (flags first), never written back for other modules to read; process-only settings report
/// conflicts.
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class ConfigurationScopeTests : IDisposable
{
    private readonly EnvScope _env = new();
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-config-scope-" + Guid.NewGuid().ToString("N"));

    public ConfigurationScopeTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        _env.Dispose();
        MoeCpuOffloadConfig.Reset();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void AFixedSourceChangesOnlyAtAStepBoundaryAfterAnUpdate()
    {
        var source = ExecutionOptionsSource.Fixed(ExecutionOptions.Default);
        long version = source.Version;
        _env.Set("TS_SCHED_DISABLE_BATCHED", "1");
        Assert.False(source.BeginStep().BatchedPathDisabled);   // the environment is not a channel for a fixed source
        Assert.Equal(version, source.Version);

        source.Update(ExecutionOptions.Default with { PrefixCheckpointBudget = 9 });
        Assert.Equal(4, source.Current.PrefixCheckpointBudget);   // the step running now keeps its snapshot
        Assert.Equal(9, source.BeginStep().PrefixCheckpointBudget);
        Assert.Equal(9, source.Current.PrefixCheckpointBudget);
        Assert.Equal(version + 1, source.Version);
        source.BeginStep();
        Assert.Equal(version + 1, source.Version);   // an unchanged boundary keeps the version
    }

    [Fact]
    public void TheCompatibilitySourceReadsTheEnvironmentAtStepBoundariesOnly()
    {
        _env.Set("TS_PREFIX_CHECKPOINTS_MAX", "3");
        var source = ExecutionOptionsSource.FollowEnvironment();
        Assert.True(source.FollowsEnvironment);
        Assert.Equal(3, source.Current.PrefixCheckpointBudget);
        _env.Set("TS_PREFIX_CHECKPOINTS_MAX", "7");
        Assert.Equal(3, source.Current.PrefixCheckpointBudget);   // mid-step: the step's snapshot
        Assert.Equal(7, source.BeginStep().PrefixCheckpointBudget);

        source.Update(ExecutionOptions.Default);   // an explicit update stops following the environment
        Assert.False(source.FollowsEnvironment);
        source.BeginStep();
        _env.Set("TS_PREFIX_CHECKPOINTS_MAX", "1");
        Assert.Equal(4, source.BeginStep().PrefixCheckpointBudget);
    }

    [Fact]
    public async System.Threading.Tasks.Task AnEnginesExplicitOptionsHoldAndAnUpdateLandsAtTheNextStep()
    {
        var model = new UnifiedOnlyOracle();
        var options = ExecutionOptions.Default with { PrefixCheckpointBudget = 2 };
        using var engine = new InferenceEngine(model, new SchedulerConfig
        {
            BlockSize = 8, NumBlocks = 64, MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 16, EnablePrefixCaching = false,
            StopRepetition = false, ExecutionOptions = options,
        });
        Assert.Equal(options, engine.ExecutionOptions);
        _env.Set("TS_PREFIX_CHECKPOINTS_MAX", "11");
        long version = engine.ExecutionOptionsVersion;
        var first = new SequenceState("one", new System.Collections.Generic.List<int> { 1, 2, 3, 4, 5 }, 4, 8, TensorSharp.Runtime.SamplingConfig.Greedy);
        await engine.SubmitRequest(first).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Equal(2, engine.ExecutionOptions.PrefixCheckpointBudget);   // the environment did not reach it
        Assert.Equal(version, engine.ExecutionOptionsVersion);

        engine.UpdateExecutionOptions(options with { PrefixCheckpointBudget = 5 });
        var second = new SequenceState("two", new System.Collections.Generic.List<int> { 6, 7, 8 }, 4, 8, TensorSharp.Runtime.SamplingConfig.Greedy);
        await engine.SubmitRequest(second).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Equal(5, engine.ExecutionOptions.PrefixCheckpointBudget);
        Assert.True(engine.ExecutionOptionsVersion > version);
    }

    [Fact]
    public void TheHostsPlacementIsResolvedOnceWithFlagsFirstAndNotWrittenBack()
    {
        _env.Set(ModelPlacementOptions.TpDegreeVariable, "4");
        _env.Set(ModelPlacementOptions.LayerSplitDegreeVariable, null);
        _env.Set(ModelPlacementOptions.TpDevicesVariable, "1,3");
        var parsed = ModelParallelismOptions.Parse(new[] { "--tp=2" });
        Assert.Equal(2, parsed.Placement.TpDegree);                       // the flag wins
        Assert.Equal("1,3", parsed.Placement.TpDevices);                  // the environment fills what no flag names
        Assert.Equal(new[] { 1, 3 }, parsed.Placement.DeviceIds(2, collectives: true));
        Assert.False(parsed.Placement.LayerSplitRequested);
        Assert.Equal("4", Environment.GetEnvironmentVariable(ModelPlacementOptions.TpDegreeVariable));   // nothing written back

        var fromEnvironment = ModelParallelismOptions.Parse(Array.Empty<string>());
        Assert.Equal(4, fromEnvironment.Placement.TpDegree);
        Assert.False(fromEnvironment.HasCliOverrides);
    }

    [Fact]
    public void AnExplicitPlacementIsTheModelsSnapshotWhateverTheEnvironmentSays()
    {
        string path = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }.Write(Path.Combine(_dir, "m.gguf"));
        _env.Set(ModelPlacementOptions.TpDegreeVariable, "2");   // would ask the legacy overload for two ranks
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu, ModelPlacementOptions.Default);
        Assert.Equal(1, model.Placement.TpDegree);
        Assert.Equal(1, model.Placement.LayerSplitDegree);
        _env.Set(ModelPlacementOptions.TpDevicesVariable, "5,6");
        Assert.Null(model.Placement.TpDevices);   // fixed at load
    }

    [Fact]
    public void AModelsMoeOffloadIsItsLoadTimeSnapshot()
    {
        string path = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }.Write(Path.Combine(_dir, "m.gguf"));
        MoeCpuOffloadConfig.SetLayers(2);
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu, ModelPlacementOptions.Default);
        MoeCpuOffloadConfig.SetAllLayers();
        Assert.Equal(new MoeCpuOffloadPolicy(2, false, true), model.MoeOffload);
        Assert.True(model.MoeOffload.IsLayerOnCpu(1));
        Assert.False(model.MoeOffload.IsLayerOnCpu(2));
        Assert.True(MoeCpuOffloadConfig.Current.AllLayers);   // the next load takes the new default
    }

    [Fact]
    public void AProcessSettingTakenOnceReportsAConflictingClaim()
    {
        string setting = "test.setting." + Guid.NewGuid().ToString("N");
        try
        {
            ProcessSettings.Claim(setting, "a", "the first owner");
            ProcessSettings.Claim(setting, "a", "a second owner of the same value");
            var conflict = Assert.Throws<ProcessSettingConflictException>(() => ProcessSettings.Claim(setting, "b", "a later owner"));
            Assert.Equal("a", conflict.HeldValue);
            Assert.Equal("the first owner", conflict.HeldOwner);
            Assert.Contains("cannot be isolated per model", conflict.Message);
            Assert.False(ProcessSettings.TryClaim(setting, "b", "a later owner", out var reported));
            Assert.NotNull(reported);
            Assert.True(ProcessSettings.TryGet(setting, out string value, out _));
            Assert.Equal("a", value);
        }
        finally
        {
            ProcessSettings.ReleaseForTests(setting);
        }
    }
}
