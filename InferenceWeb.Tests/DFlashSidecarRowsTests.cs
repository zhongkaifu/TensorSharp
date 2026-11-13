// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class DFlashSidecarRowsTests
{
    [Theory]
    [InlineData(1f, 1)]
    [InlineData(.25f, 1)]
    [InlineData(.25f, 3)]
    [InlineData(-.5f, 3)]
    public void PerOpDraftEmbeddingAndSelectorGather_ApplySidecarOnce(float scale, int count)
    {
        // Actual Q8_0 rows, with an exact binary block scale. The oracle reads
        // the integer codes directly and is independent of row dequantization.
        const int width = 32, vocabulary = 3;
        var raw = new byte[vocabulary * 34];
        for (int row = 0; row < vocabulary; row++)
        {
            ushort blockScale = BitConverter.HalfToUInt16Bits((System.Half).125f);
            raw[row * 34] = (byte)blockScale;
            raw[row * 34 + 1] = (byte)(blockScale >> 8);
            for (int column = 0; column < width; column++)
                raw[row * 34 + 2 + column] = unchecked((byte)(sbyte)(column - 16 + row * 3));
        }
        using var weight = new QuantizedWeight(raw, ggmlType: 8, width, vocabulary) { Scale = scale };
        var model = (Qwen35RuntimeModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35RuntimeModel));
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        Set(model, "_allocator", allocator);
        Set(model, "_backend", BackendType.Cpu);
        Set(model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
        Set(model, "<Config>k__BackingField", new ModelConfig { HiddenSize = width, VocabSize = vocabulary });
        Set(model, "_weights", new Dictionary<string, Tensor>());
        Set(model, "_quantWeights", new Dictionary<string, QuantizedWeight>
        {
            ["token_embd.weight"] = weight,
            ["dflash.selector_predecessor.weight"] = weight,
            ["dflash.selector_successor.weight"] = weight,
        });
        try
        {
            int[] ids = count == 1 ? new[] { 2 } : new[] { 2, 0, 2 };
            using var embedding = (Tensor)Invoke(model, "DFlashEmbedding", new object[] { ids });
            float[] actual = embedding.GetElementsAsFloat(count * width);
            for (int row = 0; row < count; row++)
                for (int col = 0; col < width; col++)
                    Assert.Equal((col - 16 + ids[row] * 3) * .125f * scale, actual[row * width + col]);

            foreach (string table in new[] { "selector_predecessor", "selector_successor" })
            {
                // Also exercise the selector's partial-count branch.
                int selected = Math.Min(count, 2);
                var rows = (float[])Invoke(model, "DFlashGatherCodebook",
                    new object[] { "dflash." + table + ".weight", ids, selected, width });
                Assert.Equal(selected * width, rows.Length);
                for (int i = 0; i < rows.Length; i++)
                    Assert.Equal(actual[i], rows[i]);
            }
            // Repeated calls must not scale the shared weight bytes in place.
            using var again = (Tensor)Invoke(model, "DFlashEmbedding", new object[] { ids });
            Assert.Equal(actual, again.GetElementsAsFloat(count * width));
        }
        finally
        {
            // This isolated arithmetic fixture does not own a loaded model.
            GC.SuppressFinalize(model);
        }
    }

    private static void Set(object model, string field, object value) =>
        typeof(ModelBase).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);

    private static object Invoke(object model, string name, object[] args) =>
        typeof(ModelBase).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, args)!;
}
