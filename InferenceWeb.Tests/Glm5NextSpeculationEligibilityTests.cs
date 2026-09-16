// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Whether GLM-5.3-Flash (glm5next) is ELIGIBLE for speculation, and what the
// engine does with a request that asks for it. The rollback itself - the KDA
// recurrent-state snapshot a partially rejected window needs - is proven in
// Glm5NextSpeculativeRollbackTests; this file pins the contract the executor
// plans against: the trunk is profitable, its verify does NOT persist the
// accepted prefix (so the executor restores and re-forwards), and a served
// request produces exactly the plain stream with speculation actually armed.
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
    public void ActualKdaCheckpoint_IsEligibleForNgram_OnTheRecurrentContract(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        using var model = ModelBase.Create(Fixture(), backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        Assert.Equal("glm5next", model.Config.Architecture);
        // No NextN head is built for glm5next, so `auto` declines and only the
        // weight-free algorithm arms.
        Assert.Equal(DraftHeadKind.None, Assert.IsAssignableFrom<IDraftHead>(model).DraftHeadKind);
        Assert.False(SpeculatorRegistry.RequiresDraftHead(SpeculatorRegistry.NGram));
        using var algorithm = SpeculatorRegistry.Create(target, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3,
        }, out string decline);
        Assert.NotNull(algorithm);
        Assert.Null(decline);

        // The recurrent contract: profitable, but the verify's writes are NOT the
        // accepted prefix's final state - the executor must restore the snapshot
        // and re-forward (the Qwen 3.5 / Qwen 3.8 route), never just rewind.
        Assert.True(target.SpeculationProfitable);
        Assert.False(target.SpecVerifyPersistsAcceptedKv);
        Assert.Equal(3, target.SpecPreferredDraftWindow);

        var caps = ExecutionCapabilities.FromModel(model);
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default,
            new SchedulerConfig { Speculation = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram } },
            new ExecutionStepFeatures { SequenceCount = 1 });
        Assert.Contains(ExecutionPathKind.SpeculativePerSequence, plan.Candidates);
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void ArbitraryRewind_IsRefusedWithoutMutatingTheContinuation(BackendType backend)
    {
        // A KDA state cannot be rewound to an arbitrary position: the only exact
        // rewind is back to a restored snapshot. Anything else must be refused
        // loudly and leave the trunk exactly where it was.
        using var env = EnvironmentForFixture();
        using var model = ModelBase.Create(Fixture(), backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        int[] prompt = {65, 66, 65, 66, 65};
        model.ForwardRefill(prompt);
        float[] expected = (float[])model.Forward(new[] {67}).Clone();
        model.ResetKVCache();
        model.ForwardRefill(prompt);

        var refused = Assert.Throws<NotSupportedException>(() => target.SpecRewindCache(prompt.Length - 1));
        Assert.Contains("cannot be rewound", refused.Message);
        Assert.Equal(prompt.Length, target.CacheSeqLen);
        // A rewind to where the trunk already is drops nothing and is allowed.
        target.SpecRewindCache(prompt.Length);
        Assert.Equal(prompt.Length, target.CacheSeqLen);
        // A restore with no snapshot behind it is a protocol error, not a silent no-op.
        Assert.Throws<InvalidOperationException>(() => target.SpecRestoreRecurrentState());

        float[] actual = model.Forward(new[] {67});
        Assert.Equal(expected, actual); // Entire vocabulary, not only argmax.
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public async Task SchedulerRequestedNgram_ArmsAndPreservesThePlainStream(BackendType backend)
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
                Speculation = new SpeculationOptions
                {
                    Enabled = speculate, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3,
                },
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
        Assert.Null(plain.sequence.SpecStats);
        // Speculation actually armed for the served request (it used to be
        // declined outright on this architecture).
        Assert.NotNull(requested.sequence.SpecStats);
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
        Assert.Equal(0, target.SpecPreferredDraftWindow);
    }
}
