// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class Glm5NextSpeculationEligibilityTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm5next-spec-" + Guid.NewGuid().ToString("N"));
    public Glm5NextSpeculationEligibilityTests() => Directory.CreateDirectory(_directory);
    public void Dispose() { try { Directory.Delete(_directory, true); } catch (IOException) { } }

    private string Fixture() => GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
        Path.Combine(_directory, "kda.gguf"), numHeads: 4, quantizeAttentionOutput: false);

    private static EnvScope EnvironmentForFixture()
    {
        var env = new EnvScope();
        env.ClearSpeculationVars();
        env.Set("MAX_CONTEXT", "256");
        env.Set("TS_GLM_NATIVE", null);
        return env;
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void ActualKdaCheckpoint_DeclinesNgramDespiteRegistryNotRequiringAHead(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        using var model = ModelBase.Create(Fixture(), backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        Assert.Equal("glm5next", model.Config.Architecture);
        Assert.False(SpeculatorRegistry.RequiresDraftHead(SpeculatorRegistry.NGram));
        using var algorithm = SpeculatorRegistry.Create(target, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3,
        }, out string decline);
        Assert.NotNull(algorithm); // A missing NextN head cannot protect this path.
        Assert.Null(decline);
        Assert.False(target.SpeculationProfitable);
        Assert.False(target.SpecVerifyPersistsAcceptedKv);
        var caps = ExecutionCapabilities.FromModel(model);
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default,
            new SchedulerConfig { Speculation = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram } },
            new ExecutionStepFeatures { SequenceCount = 1 });
        Assert.DoesNotContain(ExecutionPathKind.SpeculativePerSequence, plan.Candidates);
        Assert.DoesNotContain(ExecutionPathKind.SpeculativeBatchedTrunk, plan.Candidates);
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void DirectSpeculativeCalls_RefuseBeforeMutatingActualKdaContinuation(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        using var model = ModelBase.Create(Fixture(), backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        int[] prompt = {65, 66, 65, 66, 65};
        model.ForwardRefill(prompt);
        float[] expected = (float[])model.Forward(new[] {67}).Clone();
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        float[] logits = Enumerable.Repeat(float.NaN, 2 * model.Config.VocabSize).ToArray();
        var forward = Assert.Throws<NotSupportedException>(() => target.SpecForward(new[] {67, 68}, null, logits, true));
        Assert.Contains("KDA recurrent-state rollback", forward.Message);
        Assert.All(logits, value => Assert.True(float.IsNaN(value)));
        Assert.Throws<NotSupportedException>(() => target.SpecRewindCache(3));
        float[] actual = model.Forward(new[] {67});
        Assert.Equal(expected, actual); // Entire vocabulary, not only argmax.
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public async Task SchedulerRequestedNgram_UsesPlainActualKdaAndPreservesTokens(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        string fixture = Fixture();
        async Task<(int[] tokens, string finish, SequenceState sequence)> Run(bool speculate)
        {
            using var model = ModelBase.Create(fixture, backend);
            var config = new SchedulerConfig
            {
                MaxNumBatchedTokens = 32, MaxNumRunningSequences = 1,
                MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 16,
                NumBlocks = 16, BlockSize = 16, EnablePrefixCaching = false,
                Speculation = new SpeculationOptions { Enabled = speculate, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3 },
            };
            using var engine = new InferenceEngine(model, config, NullLogger.Instance);
            var sequence = new SequenceState("kda-" + speculate,
                new List<int> {65, 66, 65, 66, 65, 66, 65, 66}, 12, 16, SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(sequence);
            var tokens = new List<int>();
            await foreach (int token in handle.Tokens.ReadAllAsync()) tokens.Add(token);
            var completion = await handle.Completion;
            Assert.NotEqual("error", completion.FinishReason);
            Assert.NotEmpty(tokens);
            return (tokens.ToArray(), completion.FinishReason, sequence);
        }
        var plain = await Run(false);
        var requested = await Run(true);
        Assert.Equal(plain.tokens, requested.tokens);
        Assert.Equal(plain.finish, requested.finish);
        Assert.Null(requested.sequence.SpecStats);
    }

    [Fact]
    public void OrdinaryGlmDsa_RetainsItsSpeculativeTrunk()
    {
        using var env = EnvironmentForFixture();
        string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(_directory, "glm-dsa.gguf"));
        using var model = ModelBase.Create(path, BackendType.Cpu);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        Assert.Equal("glm-dsa", model.Config.Architecture);
        Assert.True(target.SpeculationProfitable);
        Assert.True(target.SpecVerifyPersistsAcceptedKv);
    }
}
