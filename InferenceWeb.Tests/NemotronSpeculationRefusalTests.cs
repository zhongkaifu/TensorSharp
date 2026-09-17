// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.CompilerServices;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Nemotron-H refuses speculative decoding for correctness: its multi-token
/// verify and single-token decode run different kernels, so a speculative
/// stream diverged from plain greedy decoding (every solo DSpark request in the
/// 2026-09-16 campaign). These tests pin the refusal at each layer that could
/// otherwise arm speculation, plus the drafter attention-sink math that was
/// wrong regardless.
/// </summary>
public sealed class NemotronSpeculationRefusalTests : IDisposable
{
    private readonly EnvScope _env = new();
    private readonly string _dir;

    public NemotronSpeculationRefusalTests()
    {
        _env.ClearSpeculationVars();
        _dir = Path.Combine(Path.GetTempPath(), $"ts-nemo-spec-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    private static NemotronModel UninitializedNemotron() =>
        (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));

    [Fact]
    public void NemotronTrunk_DeclaresACorrectnessRefusal()
    {
        ISpeculativeTarget target = UninitializedNemotron();

        Assert.False(string.IsNullOrWhiteSpace(target.SpeculationRefusal));
        Assert.Contains("plain greedy decoding", target.SpeculationRefusal);
    }

    [Fact]
    public void SpeculatorRegistry_RefusesEveryAlgorithmOnARefusingTrunk()
    {
        ISpeculativeTarget target = UninitializedNemotron();

        foreach (string name in new[] { SpeculatorRegistry.Auto, SpeculatorRegistry.NGram, SpeculatorRegistry.Block })
        {
            var speculator = SpeculatorRegistry.Create(target,
                new SpeculationOptions { Enabled = true, SpeculatorName = name }, out string reason);

            Assert.Null(speculator);
            Assert.Equal(target.SpeculationRefusal, reason);
        }
    }

    [Fact]
    public void Planner_RefusalOutranksProfitability_AndServesStandardDecode()
    {
        var caps = new ExecutionCapabilities
        {
            SupportsBatchedPagedAttention = true,
            BatchedForwardAvailable = true,
            SupportsLinearKvMigration = true,
            SupportsKvStateSnapshot = true,
            SupportsCrossSequenceKvReuse = true,
            SupportsSpeculativeTrunk = true,
            HasDraftHead = true,
            SpeculationProfitable = true,
            SpeculationRefusal = NemotronModel.SpeculationRefusalReason,
        };
        var config = new SchedulerConfig { Speculation = new SpeculationOptions { Enabled = true } };

        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, config, new ExecutionStepFeatures { SequenceCount = 1 });

        Assert.NotEqual(ExecutionPathKind.SpeculativePerSequence, plan.Selected);
        Assert.DoesNotContain(ExecutionPathKind.SpeculativePerSequence, plan.Candidates);
        Assert.Equal(NemotronModel.SpeculationRefusalReason, plan.SpeculationRefusal);
        Assert.False(plan.SpeculationUnprofitable);
        var rejection = Assert.Single(plan.Rejections, r => r.Path == ExecutionPathKind.SpeculativePerSequence);
        Assert.Contains(NemotronModel.SpeculationRefusalReason, rejection.Reason);

        string report = ExecutionPlanner.BuildCapabilityReport(caps, ExecutionOptions.Default, config);
        Assert.Contains("refused by the model", report);
    }

    [Fact]
    public void DFlashDrafter_IsRefusedBeforeAnyFileIsRead()
    {
        var model = UninitializedNemotron();

        // The path does not exist: the refusal must win over the file check, so
        // nothing about a drafter is ever loaded onto this trunk.
        var ex = Assert.Throws<NotSupportedException>(
            () => model.LoadDFlashDraftWeights(Path.Combine(_dir, "missing-dspark.gguf")));
        Assert.Equal(NemotronModel.SpeculationRefusalReason, ex.Message);
        Assert.False(model.HasDFlash);
    }

    [Fact]
    public void DraftHeadLoader_ReportsTheRefusalForAConfiguredDrafter()
    {
        string draft = Path.Combine(_dir, "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark.gguf");
        File.WriteAllBytes(draft, new byte[] { 0 });
        _env.Set(SpeculationEnvVars.DraftModel, draft);

        bool attached = SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(UninitializedNemotron(), out string error);

        Assert.False(attached);
        Assert.Contains("is not attached", error);
        Assert.Contains(NemotronModel.SpeculationRefusalReason, error);
    }

    // ----- DFlash/DSpark attention sinks -----

    [Fact]
    public void DFlashSinkSoftmax_IsASoftmaxWithOneExtraLogitPerQueryHead()
    {
        // kvHeads=2, groupSize=2 (4 query heads), block b=2, 3 key columns.
        const int kvHeads = 2, groupSize = 2, b = 2, total = 3;
        var rnd = new Random(7);
        var scores = new float[kvHeads * groupSize * b * total];
        for (int i = 0; i < scores.Length; i++)
            scores[i] = (float)(rnd.NextDouble() * 6 - 3);
        scores[5] = float.NegativeInfinity;                  // a masked column
        float[] sinks = { 0.5f, -1f, 2f, 0f };
        var input = (float[])scores.Clone();

        ModelBase.DFlashSinkSoftmaxRows(scores, sinks, b, groupSize, kvHeads, total);

        for (int g = 0; g < kvHeads; g++)
        {
            for (int j = 0; j < groupSize * b; j++)
            {
                int row = (g * groupSize * b + j) * total;
                float sink = sinks[g * groupSize + j / b];
                double denom = Math.Exp(sink);
                for (int c = 0; c < total; c++)
                    denom += Math.Exp(input[row + c]);
                double mass = 0;
                for (int c = 0; c < total; c++)
                {
                    double expected = Math.Exp(input[row + c]) / denom;
                    Assert.Equal(expected, scores[row + c], 5);
                    mass += scores[row + c];
                }
                // The sink absorbs probability mass; the keys never get MORE than 1.
                Assert.Equal(1.0 - Math.Exp(sink) / denom, mass, 5);
            }
        }
        Assert.Equal(0f, scores[5]);
    }
}
