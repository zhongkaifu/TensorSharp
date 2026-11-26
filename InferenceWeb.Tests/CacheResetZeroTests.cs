// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A reset of the shared decoder runtime's cache on a backend that zeroes reset caches (the pure-C# CPU backend) zeroes only
/// the rows its prompts wrote, and leaves the cache as zero as a fresh one: the next prompt answers bit for bit as on a fresh
/// model. Large CPU buffers come zeroed from the C allocator, so a full-context cache is not filled at load either.
/// </summary>
public sealed class CacheResetZeroTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("cache-reset-zero").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static int[] Prompt(int n, int salt) => Enumerable.Range(0, n).Select(i => 30 + (i * 13 + salt * 7) % 160).ToArray();

    private static bool AllZero(Tensor t)
    {
        var bytes = new byte[t.Storage.ByteLength];
        Marshal.Copy(TensorComputePrimitives.GetStoragePointer(t), bytes, 0, bytes.Length);
        return Array.TrueForAll(bytes, b => b == 0);
    }

    private static Tensor[] Caches(StandardDecoderModel model)
        => model.KeyCachesForTests.Concat(model.ValueCachesForTests)
            .Where(layer => layer != null).SelectMany(layer => layer).Where(t => t != null).ToArray();

    [Theory]
    [InlineData(KvCacheDtype.F32)]
    [InlineData(KvCacheDtype.F16)]
    public void CpuReset_ZeroesTheWrittenRows_AndTheNextPromptAnswersAsOnAFreshModel(KvCacheDtype cacheType)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(cacheType);
        string path = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }
            .Write(Path.Combine(_dir, "mistral3.gguf"));
        int[] first = Prompt(41, 1), second = Prompt(23, 2);

        float[] fresh;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cpu))
            fresh = (float[])model.ForwardRefill(second).Clone();

        using ModelBase reused = ModelBase.Create(path, BackendType.Cpu);
        var runtime = Assert.IsAssignableFrom<StandardDecoderModel>(reused);
        reused.ForwardRefill(first);
        Tensor[] caches = Caches(runtime);
        Assert.NotEmpty(caches);
        Assert.All(caches, t => Assert.True(t.Sizes[1] > first.Length, "the cache holds cells past the prompt's rows"));
        Assert.Contains(caches, t => !AllZero(t));

        reused.ResetKVCache();
        Assert.All(Caches(runtime), t => Assert.True(AllZero(t), "a reset cache is zero"));
        float[] again = reused.ForwardRefill(second);
        Assert.True(again.AsSpan().SequenceEqual(fresh), "the prompt after a reset answers as on a fresh model");
    }

    [Fact]
    public void LargeCpuBuffersComeZeroed_SmallOnesAreNotPromised()
    {
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        long large = CpuStorage.ZeroedAllocationBytes / sizeof(float);
        using (var t = new Tensor(cpu, DType.Float32, large))
        {
            Assert.True(t.Storage.TakeAllocationZeroGuarantee());
            Assert.False(t.Storage.TakeAllocationZeroGuarantee());   // consumed: later writes do not renew it
            var sample = new float[4096];
            Marshal.Copy(TensorComputePrimitives.GetStoragePointer(t) + (nint)((large - sample.Length) * sizeof(float)), sample, 0, sample.Length);
            Assert.All(sample, v => Assert.Equal(0f, v));
        }
        using (var t = new Tensor(cpu, DType.Float32, 1024))
            Assert.False(t.Storage.TakeAllocationZeroGuarantee());
    }
}
