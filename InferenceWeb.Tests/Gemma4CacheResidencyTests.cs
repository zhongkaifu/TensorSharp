// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4CacheResidencyTests
{
    private readonly ITestOutputHelper _output;

    public Gemma4CacheResidencyTests(ITestOutputHelper output) => _output = output;

    [CudaFact("TS_TEST_MODEL_DIR", "gemma-4-e4b", GgmlBackend = BackendType.GgmlCuda)]
    public void InitializeResidentCache_DoesNotRestorePreviousDeviceContents()
    {
        string path = TestGates.FindGguf(
            Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gemma-4-e4b");
        Assert.False(string.IsNullOrEmpty(path));
        KvCacheDtype previousDtype = KvCacheDtypeConfig.Current;
        try
        {
            KvCacheDtypeConfig.Set(KvCacheDtype.F16);
            using var model = Assert.IsType<Gemma4Model>(ModelBase.Create(path, BackendType.GgmlCuda));
            Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype);

            int[] tokens = model.Tokenizer.Encode(string.Join("\n", Enumerable.Range(0, 20).Select(i =>
                $"Record {i}: cedar is the item name and {17 * i + 31} is its code.")),
                addSpecial: true).Take(65).ToArray();
            Assert.Equal(65, tokens.Length);
            model.Forward(tokens.Take(64).ToArray());
            model.Forward(new[] { tokens[64] });

            MethodInfo initialize = typeof(Gemma4Model).GetMethod(
                "InitGemma4CacheTensor", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Gemma4 cache initializer not found.");

            foreach (string fieldName in new[] { "_kvCacheK", "_kvCacheV" })
            {
                FieldInfo field = typeof(Gemma4Model).GetField(
                    fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
                    ?? throw new InvalidOperationException($"Gemma4 field {fieldName} not found.");
                var tensors = Assert.IsType<Tensor[]>(field.GetValue(model));
                Tensor cache = tensors.First(t => t != null);
                Assert.Equal(DType.Float16, cache.ElementType);
                Assert.Equal(0, cache.StorageOffset);
                IntPtr pointer = cache.Storage.PtrAtElement(0);

                GgmlBasicOps.SyncHostBuffer(pointer, cache.Storage.ByteLength);
                short[] original = ReadHalfBits(cache);
                Assert.Contains(original, bits => bits != 0);
                Assert.All(original, bits => Assert.True(
                    System.Half.IsFinite(BitConverter.UInt16BitsToHalf(unchecked((ushort)bits))),
                    $"{fieldName} contains a non-finite value before initialization."));

                // Prove these bytes have a live device copy. A host-only backend
                // or an unbound tensor must not make this regression pass without
                // exercising native residency: the deliberate host clear below
                // must be undone by downloading the existing device contents.
                var zeros = new short[original.Length];
                Marshal.Copy(zeros, 0, pointer, zeros.Length);
                GgmlBasicOps.SyncHostBuffer(pointer, cache.Storage.ByteLength);
                Assert.Equal(original, ReadHalfBits(cache));

                // A pooled allocation can reuse this exact host address while
                // its old device copy remains cached. Reinitialize the occupied
                // address directly to make that state deterministic, without
                // depending on the allocator's best-fit choice or prior tests.
                initialize.Invoke(model, new object[] { cache });
                Assert.Equal(pointer, cache.Storage.PtrAtElement(0));
                Assert.Equal(zeros, ReadHalfBits(cache));
                GgmlBasicOps.SyncHostBuffer(pointer, cache.Storage.ByteLength);
                short[] afterSync = ReadHalfBits(cache);
                int restoredValues = afterSync.Count(bits => bits != 0);
                _output.WriteLine(
                    $"{fieldName}: dtype=F16 bytes={cache.Storage.ByteLength} " +
                    $"original_nonzero={original.Count(bits => bits != 0)} restored_after_initialize={restoredValues}");
                Assert.True(restoredValues == 0,
                    $"{fieldName}: synchronization restored {restoredValues} stale device values " +
                    "after the cache initializer cleared the same storage.");

                // Clearing an already-cleared cache must retain the same contract.
                initialize.Invoke(model, new object[] { cache });
                GgmlBasicOps.SyncHostBuffer(pointer, cache.Storage.ByteLength);
                Assert.Equal(zeros, ReadHalfBits(cache));
            }
        }
        finally
        {
            KvCacheDtypeConfig.Set(previousDtype);
        }
    }

    private static short[] ReadHalfBits(Tensor tensor)
    {
        tensor.Storage.EnsureHostReadable();
        var bits = new short[checked((int)(tensor.Storage.ByteLength / sizeof(short)))];
        Marshal.Copy(tensor.Storage.PtrAtElement(0), bits, 0, bits.Length);
        return bits;
    }
}
