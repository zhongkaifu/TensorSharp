// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// qwen4exp's verify keeps every row's recurrent state so a partial acceptance commits the state after the kept row
/// (TSGgml_Qwen4ExpCommitVerifyRow) instead of restoring and re-forwarding. Those row buffers were keyed by the
/// holder's state and never freed with it: each request that speculated on a holder of its own left ~0.9 GB behind on
/// Flash Next, and the Metal working set ran out under Strata's concurrency rounds. One set now serves every holder (a
/// verify's rows are read only by the commit right after it). Two holders verify and commit in turn: the second adds
/// no row buffer bytes, and each continues exactly as decoding its kept tokens one by one would (exact row kernels).
/// </summary>
public sealed class Qwen4ExpVerifyRowBufferTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-q4e-rows-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();
    private readonly NativeEnvScope _nativeEnv = new NativeEnvScope();

    public Qwen4ExpVerifyRowBufferTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
        // Row-by-row verify kernels: a verify row is then bit-identical to decoding that token alone.
        _nativeEnv.Set("TS_Q4E_VERIFY_ROW_KERNELS", "1");
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _nativeEnv.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] PromptA = Enumerable.Range(0, 12).Select(i => (i * 37 + 11) % 250).ToArray();
    private static readonly int[] PromptB = Enumerable.Range(0, 9).Select(i => (i * 53 + 7) % 250).ToArray();
    private static readonly int[] WindowA = { 17, 203, 99, 4 };    // anchor + three drafts
    private static readonly int[] WindowB = { 150, 61, 8, 222 };
    private const int Accepted = 1;                                  // keep the anchor and one draft
    private const int Next = 79;

    private static float[] Plain(Qwen4ExpModel model, int[] prompt, int[] window)
    {
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        for (int i = 0; i <= Accepted; i++) model.Forward(new[] { window[i] });
        return (float[])model.Forward(new[] { Next }).Clone();
    }

    private static void VerifyAndCommit(Qwen4ExpModel model, IBatchedPagedModel fused, string holder, int[] window)
    {
        fused.BindSequenceCache(holder);
        int start = model.CacheSeqLen;
        model.SpecSnapshotRecurrentState();
        var hidden = new float[window.Length * model.SpecFeatureSize];
        var logits = new float[window.Length * model.Config.VocabSize];
        model.SpecForward(window, hidden, logits, allLogitsRows: true);
        model.SpecOnVerifyAccepted(Accepted, window.Length - 1);
        model.SpecRewindCache(start + Accepted + 1);
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void TwoHoldersVerifyingInTurn_ShareOneRowBufferSet_AndContinueAsPlainDecoding()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "rows.gguf"));
        float[] plainA, plainB;
        using (var reference = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlMetal))
        {
            plainA = Plain(reference, PromptA, WindowA);
            plainB = Plain(reference, PromptB, WindowB);
        }

        using var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlMetal);
        var fused = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        fused.BindSequenceCache("a");
        model.ForwardPrompt(PromptA);
        fused.BindSequenceCache("b");
        model.ForwardPrompt(PromptB);

        long before = GgmlBasicOps.Qwen4ExpRowBufferBytes();
        VerifyAndCommit(model, fused, "a", WindowA);
        long afterA = GgmlBasicOps.Qwen4ExpRowBufferBytes();
        VerifyAndCommit(model, fused, "b", WindowB);
        long afterB = GgmlBasicOps.Qwen4ExpRowBufferBytes();
        _output.WriteLine($"row buffer bytes: before {before}, after holder a {afterA}, after holder b {afterB}");
        Assert.True(afterA > before || before > 0, "the verify kept no rows (the commit path did not run)");
        Assert.Equal(afterA, afterB);

        fused.BindSequenceCache("a");
        float[] specA = (float[])model.Forward(new[] { Next }).Clone();
        fused.BindSequenceCache("b");
        float[] specB = (float[])model.Forward(new[] { Next }).Clone();
        Assert.True(specA.AsSpan().SequenceEqual(plainA), "holder a: the committed row's state differs from decoding its kept tokens");
        Assert.True(specB.AsSpan().SequenceEqual(plainB), "holder b: the committed row's state differs from decoding its kept tokens");
    }
}
