// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// An explicit MAX_CONTEXT allocates the whole window up front, which every concurrent request's holder paid
/// again: on Flash Next at 32768 each live conversation took ~1.6 GB of a Mac's memory for a 29-token prompt,
/// and Strata's 4-at-once rounds ran 20-50% slower per step under the pressure. On Metal a request's holder now
/// starts at <see cref="Qwen4ExpModel.MetalFreshHolderTokens"/> rows and grows with its request; the primary
/// keeps the whole window, an explicit TS_KV_INITIAL_TOKENS sizes both, and a holder that grew answers bit for
/// bit as the whole-window primary does.
/// </summary>
public sealed class Qwen4ExpFreshHolderCapacityTests : IDisposable
{
    private const int Context = 8192;
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-q4e-holder-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Qwen4ExpFreshHolderCapacityTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", Context.ToString());
        _env.Set("TS_KV_INITIAL_TOKENS", null);
        _env.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // Longer than a fresh holder, so its first chunk has to grow it.
    private static readonly int[] Prompt = Enumerable.Range(0, 3000).Select(i => (i * 37 + 11) % 250).ToArray();
    private const int Next = 79;

    private string Model() => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "holder.gguf"), contextLength: Context);

    [GgmlFact(BackendType.GgmlMetal)]
    public void AFreshHolderStartsSmall_GrowsWithItsPrompt_AndAnswersAsTheWholeWindowPrimary()
    {
        string path = Model();
        float[] primary;
        using (var reference = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlMetal))
        {
            Assert.Equal(Context, reference.KvCacheCapacity);
            reference.ResetKVCache();
            reference.ForwardRefill(Prompt);
            primary = (float[])reference.Forward(new[] { Next }).Clone();
        }

        using var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlMetal);
        var fused = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.Equal(Context, model.KvCacheCapacity);
        fused.BindSequenceCache("a");
        Assert.Equal(Qwen4ExpModel.MetalFreshHolderTokens, model.KvCacheCapacity);
        model.ForwardPrompt(Prompt);
        Assert.InRange(model.KvCacheCapacity, Prompt.Length, Context);
        float[] holder = (float[])model.Forward(new[] { Next }).Clone();

        double maxDiff = 0;
        for (int i = 0; i < holder.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(holder[i] - primary[i]));
        _output.WriteLine($"grew to {model.KvCacheCapacity} rows; max |holder - primary| = {maxDiff:E3}");
        Assert.True(holder.AsSpan().SequenceEqual(primary), $"the grown holder answers {maxDiff:E3} away from the whole-window primary");
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void AnExplicitInitialSize_SizesTheHoldersToo()
    {
        _env.Set("TS_KV_INITIAL_TOKENS", "4096");
        using var model = (Qwen4ExpModel)ModelBase.Create(Model(), BackendType.GgmlMetal);
        Assert.Equal(4096, model.KvCacheCapacity);
        ((IBatchedPagedModel)model).BindSequenceCache("a");
        Assert.Equal(4096, model.KvCacheCapacity);
    }
}
