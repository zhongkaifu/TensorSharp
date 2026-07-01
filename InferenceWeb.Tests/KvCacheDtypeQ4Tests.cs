// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit coverage for the Q4_0 live KV-cache storage dtype (the most aggressive
/// tier, ~0.5625 bytes/elem). These exercise the pure config + byte-sizing layer
/// the rest of the engine threads through - no GPU required. End-to-end decode
/// quality through ggml's CUDA cpy / flash_attn_ext is validated separately on
/// hardware.
/// </summary>
public class KvCacheDtypeQ4Tests
{
    [Theory]
    [InlineData("q4_0")]
    [InlineData("q4")]
    [InlineData("int4")]
    [InlineData("Q4_0")]
    public void TryParse_AcceptsQ4Aliases(string value)
    {
        Assert.True(KvCacheDtypeConfig.TryParse(value, out var dtype));
        Assert.Equal(KvCacheDtype.Q4_0, dtype);
    }

    [Fact]
    public void Q4_0_MapsToExpectedGgmlTypeAndDType()
    {
        // GGML_TYPE_Q4_0 == 2 in ggml.h; the native kernels receive this id.
        Assert.Equal(2, KvCacheDtype.Q4_0.GgmlType());
        Assert.Equal(DType.Q4_0, KvCacheDtype.Q4_0.ToDType());
        Assert.Equal("q4_0", KvCacheDtype.Q4_0.ToShortString());
    }

    [Fact]
    public void Q4_0_IsBlockQuantized()
    {
        // The model-level guards route block-quantized caches down the native
        // flash path; Q4_0 must report true so it follows Q8_0's routing exactly.
        Assert.True(KvCacheDtype.Q4_0.IsBlockQuantized());
    }

    [Theory]
    [InlineData(32, 18)]    // exactly one 32-element block
    [InlineData(64, 36)]    // two blocks
    [InlineData(33, 36)]    // rounds up to two blocks
    [InlineData(1, 18)]     // partial block still costs a whole block
    public void Q4_0Bytes_RoundsUpToBlockBoundary(long elements, long expectedBytes)
    {
        Assert.Equal(expectedBytes, DTypeExtensions.Q4_0Bytes(elements));
        Assert.Equal(expectedBytes, KvCacheDtype.Q4_0.ByteLengthFor(elements));
        Assert.Equal(expectedBytes, DType.Q4_0.ByteLengthFor(elements));
    }

    [Fact]
    public void Q4_0_IsHalfOfQ8_0_AndEighthOfF32()
    {
        // 1M elements: F32=4MB-equivalent, Q8_0~1.0625x, Q4_0~0.5625x.
        const long elems = 1L << 20;
        long f32 = KvCacheDtype.F32.ByteLengthFor(elems);
        long f16 = KvCacheDtype.F16.ByteLengthFor(elems);
        long q8 = KvCacheDtype.Q8_0.ByteLengthFor(elems);
        long q4 = KvCacheDtype.Q4_0.ByteLengthFor(elems);

        // Q4_0 is strictly the smallest tier.
        Assert.True(q4 < q8, $"Q4_0 ({q4}) should be smaller than Q8_0 ({q8})");
        Assert.True(q4 < f16, $"Q4_0 ({q4}) should be smaller than F16 ({f16})");

        // Q4_0 block is 18 bytes vs Q8_0's 34 -> ~0.53x; allow a small band.
        double q4VsQ8 = (double)q4 / q8;
        Assert.InRange(q4VsQ8, 0.50, 0.55);

        // Q4_0 vs F32 (4 bytes/elem) -> ~0.1406x (1/7.1).
        double q4VsF32 = (double)q4 / f32;
        Assert.InRange(q4VsF32, 0.13, 0.15);
    }

    [Fact]
    public void Q4_0_BudgetExample_131kContext_IsTiny()
    {
        // Sanity-check the README claim: a Gemma-4-E4B-shaped cache (e.g. ~42
        // layers * 4 KV heads * 256 head_dim * 2 (K+V)) at 131072 tokens lands
        // well under 1 GB at Q4_0 - the whole point of the tier.
        const long layers = 42, kvHeads = 4, headDim = 256, ctx = 131072;
        long elems = layers * kvHeads * headDim * ctx * 2; // K + V
        long q4Bytes = KvCacheDtype.Q4_0.ByteLengthFor(elems);
        long f32Bytes = KvCacheDtype.F32.ByteLengthFor(elems);

        double q4Gb = q4Bytes / 1024.0 / 1024.0 / 1024.0;
        double f32Gb = f32Bytes / 1024.0 / 1024.0 / 1024.0;
        Assert.True(q4Gb < f32Gb / 6.0,
            $"Q4_0 ({q4Gb:F2} GB) should be <1/6 of F32 ({f32Gb:F2} GB)");
    }
}
