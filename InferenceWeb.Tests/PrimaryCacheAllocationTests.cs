// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: where recorded steps bind slots (the CUDA executor), a unified model's primary cache no longer takes the whole
// MAX_CONTEXT window at load. The managed CPU executor runs the policy through the model's flow-scoped seams (tests do not set
// MAX_CONTEXT or TS_KV_INITIAL_TOKENS in the process environment).
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class PrimaryCacheAllocationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-primary-cache-" + Guid.NewGuid().ToString("N"));

    public PrimaryCacheAllocationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture() => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }
        .Write(Path.Combine(_dir, "q35-dense.gguf"));

    /// <summary>Constructs the model with an explicit window (what MAX_CONTEXT=4096 answers: the whole window) and, with
    /// <paramref name="slotBound"/>, the slot-bound policies, as the CUDA executor's model takes them.</summary>
    private static Qwen35RuntimeModel Create(string path, bool slotBound)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        int? initial = StandardDecoderModel.InitialCacheLengthForTests.Value;
        bool policies = StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value;
        StandardDecoderModel.InitialCacheLengthForTests.Value = 4096;
        StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value = slotBound;
        try
        {
            return new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        }
        finally
        {
            StandardDecoderModel.InitialCacheLengthForTests.Value = initial;
            StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value = policies;
        }
    }

    /// <summary>With an explicit window, the primary of a model whose requests run in slots of their own starts at a fresh
    /// slot's size, and it never grows while a served conversation runs: the request's slot holds its rows. Elsewhere the
    /// explicit window is still allocated whole at load.</summary>
    [Fact]
    public void ThePrimaryStartsAtAFreshSlotWhereRecordedStepsBindSlots()
    {
        using (Qwen35RuntimeModel whole = Create(Fixture(), slotBound: false))
        {
            Assert.Equal(4096, whole.MaxContextLength);
            Assert.Equal(4096, whole.PrimaryCacheCapacityForTests);
        }

        using Qwen35RuntimeModel model = Create(Fixture(), slotBound: true);
        Assert.Equal(4096, model.MaxContextLength);
        int fresh = model.PrimaryCacheCapacityForTests;
        Assert.InRange(fresh, 1, 1024);   // a fresh slot: 512 cells here, 1,024 where the CUDA executor records steps

        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 700).Select(i => 5 + i * 13 % 240).ToArray();
        var logits = new float[1][];
        Assert.Equal(0, unified.EnsureSequenceSlot("long", 1500));
        Assert.True(unified.TryForwardUnified(new[] { "long" }, new[] { prompt }, new[] { 0 }, logits));
        int length = prompt.Length;
        for (int t = 0; t < 4; t++)
        {
            int next = Array.IndexOf(logits[0], logits[0].Max());
            Assert.True(unified.TryForwardUnified(new[] { "long" }, new[] { new[] { next } }, new[] { length++ }, logits));
            Assert.Equal(fresh, model.PrimaryCacheCapacityForTests);
        }
        Assert.True(model.RetainSequenceCacheAs("long", "kept"));
        Assert.Equal(fresh, model.PrimaryCacheCapacityForTests);
    }
}
