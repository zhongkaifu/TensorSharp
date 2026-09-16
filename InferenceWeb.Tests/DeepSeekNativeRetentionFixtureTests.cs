// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Managed ownership against the small generated CPU GGUF fixture.
/// This is not a publisher-checkpoint quality or performance test.</summary>
public sealed class DeepSeekNativeRetentionFixtureTests
{
    [ModelTheory("TS_TEST_DSV41_FIXTURE_DIR", "deepseek41-fixture")]
    [InlineData("io")]
    [InlineData("oom")]
    [InlineData("disposed")]
    public void FailingPostCommitDiagnosticCannotUndoNativeOwnership(string failure)
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_DSV41_FIXTURE_DIR"), "deepseek41-fixture");
        string[] keys = { "TS_DSV41_RETAINED_CACHE", "TS_DSV41_RETAINED_CACHE_MB", "MAX_CONTEXT",
            "TS_DSV4_UBATCH", "TS_DSV4_THREADS", "TS_DSV41_ENGRAM_THREADS", "TS_DSV4_FA", "TS_DSV41_TP" };
        string[] values = { "1", "2048", "512", "3", "2", "2", "0", "0" };
        var old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], values[i]);
            using var model = new DeepSeek4Model(path, BackendType.GgmlCpu);
            int[] prefix = { 0, 15, 32, 64, 128, 13, 254, 18 }, suffix = { 9, 21, 85 };
            model.Forward(prefix);
            var expected = model.Forward(suffix);
            model.ResetKVCache();
            model.Forward(prefix);
            model.AdoptPrimaryCacheToFused("first");
            WithFailedStderr(failure, () => Assert.True(model.RetainSequenceCache("first")));
            Assert.False(model.HasFusedSequenceCache("first"));
            Assert.True(model.CanReuseRetainedPrefix("first", prefix.Length, prefix.Length));
            WithFailedStderr(failure, () => Assert.True(model.TryRebindRetainedCache("first", "follow")));
            Assert.True(model.HasFusedSequenceCache("follow"));
            Assert.False(model.CanReuseRetainedPrefix("first", prefix.Length, prefix.Length));
            Assert.False(model.BindSequenceCache("follow"));
            Assert.Equal(expected, model.Forward(suffix));
            Assert.True(model.RetainSequenceCache("follow"));
            WithFailedStderr(failure, () => model.DiscardRetainedCache("follow"));
            Assert.False(model.CanReuseRetainedPrefix("follow", prefix.Length + suffix.Length, 0));
            Assert.True(model.CanReuseLivePrefix(0, 0));
        }
        finally
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], old[i]);
        }
    }

    private static void WithFailedStderr(string failure, Action action)
    {
        TextWriter previous = Console.Error;
        using var writer = new FailingWriter(failure);
        try { Console.SetError(writer); action(); }
        finally { Console.SetError(previous); }
        Assert.Equal(1, writer.Attempts);
    }

    private sealed class FailingWriter(string failure) : TextWriter
    {
        public int Attempts;
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;
        public override void WriteLine(string value)
        {
            Attempts++;
            if (failure == "oom") throw new OutOfMemoryException("controlled diagnostic formatting allocation");
            if (failure == "disposed") throw new ObjectDisposedException("controlled stderr sink");
            throw new IOException("controlled stderr write failure");
        }
    }

    [ModelFact("TS_TEST_DSV41_FIXTURE_DIR", "deepseek41-fixture")]
    public void RetainedNativeHolderPreservesContinuationAndCanBeReclaimedWithoutASparePrimary()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_DSV41_FIXTURE_DIR"), "deepseek41-fixture");
        string[] keys = { "TS_DSV41_RETAINED_CACHE", "TS_DSV41_RETAINED_CACHE_MB", "MAX_CONTEXT",
            "TS_DSV4_UBATCH", "TS_DSV4_THREADS", "TS_DSV41_ENGRAM_THREADS", "TS_DSV4_FA", "TS_DSV41_TP" };
        string[] values = { "1", "2048", "512", "3", "2", "2", "0", "0" };
        var old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], values[i]);
            using var model = new DeepSeek4Model(path, BackendType.GgmlCpu);
            Assert.True(model.SupportsRetainedFusedCache);
            int[] prefix = { 0, 15, 32, 64, 128, 13, 254, 18 };
            int[] suffix = { 9, 21, 85 };
            model.Forward(prefix);
            var expected = model.Forward(suffix);
            model.ResetKVCache();
            model.Forward(prefix);
            model.AdoptPrimaryCacheToFused("first");
            Assert.True(model.RetainSequenceCache("first"));
            model.OnSequenceReleased("first");
            Assert.True(model.CanReuseRetainedPrefix("first", prefix.Length, prefix.Length));
            Assert.False(model.CanReuseRetainedPrefix("first", prefix.Length + 1, prefix.Length));
            Assert.True(model.TryRebindRetainedCache("first", "follow"));
            Assert.False(model.BindSequenceCache("follow"));
            Assert.Equal(expected, model.Forward(suffix));
            Assert.True(model.RetainSequenceCache("follow"));
            model.OnSequenceReleased("follow");
            model.RestorePrimaryCache(); // Reclaims selected retained storage; no spare primary was allocated.
            Assert.False(model.CanReuseRetainedPrefix("follow", prefix.Length + suffix.Length, prefix.Length));
            Assert.True(model.CanReuseLivePrefix(0, 0));
            var unrelated = model.Forward(suffix);
            model.ResetKVCache();
            Assert.Equal(unrelated, model.Forward(suffix));
        }
        finally
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], old[i]);
        }
    }
}
