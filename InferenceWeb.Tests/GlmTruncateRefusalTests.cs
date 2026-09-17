// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// M0c (radix design P23): a KV rewind the GLM native executor refuses must be
// REPORTED as a refusal.
//
// GlmDsaModel overrode only the void TruncateKVCacheCore and ignored
// TSGgml_GlmRewind's result, so ModelBase.TryTruncateKVCache - the form every
// caller with a re-prefill fallback uses - answered true while the native head
// stayed where it was. The caller then decoded the rest of the turn at positions
// it believed it had dropped. The native side refuses a target past the head, any
// glm5next rewind other than to 0 or to the head (the KDA recurrence cannot go
// back), and any rewind on a slot whose KDA restore failed.
//
// These use the synthetic checkpoints from GlmDsaSyntheticModelBuilder on the ggml
// CPU backend, which is the native executor (TS_GLM_NATIVE unset), so they need no
// model download.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GlmTruncateRefusalTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm-truncate-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();

    public GlmTruncateRefusalTests()
    {
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
        _env.Set("MAX_CONTEXT", "256");
        _env.Set("TS_GLM_NATIVE", null);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray();

    private ModelBase LoadGlmDsa() =>
        ModelBase.Create(GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "dsa.gguf")), BackendType.GgmlCpu);

    private ModelBase LoadGlm5Next() =>
        ModelBase.Create(GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "next.gguf"), numHeads: 4, quantizeAttentionOutput: false, numLayers: 2), BackendType.GgmlCpu);

    [Fact]
    public void GlmDsa_NativeRefusal_IsReportedAndLeavesTheHead()
    {
        using var model = LoadGlmDsa();
        Assert.Contains("exec=native", model.KVStateFingerprint);
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        Assert.Equal(Prompt.Length, model.CacheSeqLen);

        // Past the head: the native rewind refuses, and so must TryTruncateKVCache.
        Assert.False(model.TryTruncateKVCache(Prompt.Length + 4));
        Assert.Equal(Prompt.Length, model.CacheSeqLen);
    }

    [Fact]
    public void GlmDsa_AcceptedRewind_MovesTheHead_AndContinuesExactly()
    {
        using var model = LoadGlmDsa();
        model.ResetKVCache();
        float[] cold = (float[])model.ForwardRefill(Prompt).Clone();

        model.ResetKVCache();
        model.ForwardRefill(Prompt.Concat(new[] { 70, 71, 72 }).ToArray());
        Assert.True(model.TryTruncateKVCache(Prompt.Length - 4));
        Assert.Equal(Prompt.Length - 4, model.CacheSeqLen);

        // The MLA and indexer rows are per position, so re-forwarding the dropped
        // tail reproduces the cold prefill exactly (whole vocabulary).
        float[] resumed = model.Forward(Prompt.Skip(Prompt.Length - 4).ToArray());
        Assert.Equal(Prompt.Length, model.CacheSeqLen);
        Assert.Equal(cold, resumed);
    }

    [Fact]
    public void Glm5Next_NativeRefusesAMidSequenceRewind_AndTheTrunkIsUntouched()
    {
        using var model = LoadGlm5Next();
        Assert.Equal("glm5next", model.Config.Architecture);
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        float[] expected = (float[])model.Forward(new[] { 70 }).Clone();

        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        Assert.False(model.TryTruncateKVCache(Prompt.Length - 3));
        Assert.Equal(Prompt.Length, model.CacheSeqLen);

        // A refused rewind changed nothing: the continuation is the uninterrupted one.
        Assert.Equal(expected, model.Forward(new[] { 70 }));
    }

    [Fact]
    public void Glm5Next_RewindToTheHeadOrToZero_IsStillAccepted()
    {
        using var model = LoadGlm5Next();
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        Assert.True(model.TryTruncateKVCache(Prompt.Length));
        Assert.Equal(Prompt.Length, model.CacheSeqLen);
        Assert.True(model.TryTruncateKVCache(0));
        Assert.Equal(0, model.CacheSeqLen);
    }

    [Fact]
    public void Glm_VoidTruncate_ThrowsOnARefusal_InsteadOfReturningWithAStaleHead()
    {
        using var model = LoadGlm5Next();
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        var refused = Assert.Throws<InvalidOperationException>(() => model.TruncateKVCache(Prompt.Length - 3));
        Assert.Contains("TryTruncateKVCache", refused.Message);
        Assert.Equal(Prompt.Length, model.CacheSeqLen);
    }
}
