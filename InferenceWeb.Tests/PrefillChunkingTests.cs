// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using TensorSharp;

namespace InferenceWeb.Tests;

/// <summary>
/// Sanity checks for <see cref="PrefillChunking"/> — the shared chunked-prefill
/// policy that both the server pipeline and the CLI interactive session route
/// through. The actual chunked forward loops are covered by the integration
/// tests that drive real model loads; these tests pin the policy itself so a
/// future change can't silently shrink the chunk size to 0.
/// </summary>
public class PrefillChunkingTests
{
    [Fact]
    public void ResolveChunkSize_ZeroTokens_Returns_Zero()
    {
        Assert.Equal(0, PrefillChunking.ResolveChunkSize(BackendType.Cpu, 0));
        Assert.Equal(0, PrefillChunking.ResolveChunkSize(BackendType.GgmlCuda, -1));
    }

    [Fact]
    public void ResolveChunkSize_GgmlCuda_UsesLargerCap()
    {
        Assert.Equal(5120, PrefillChunking.ResolveChunkSize(BackendType.GgmlCuda, 8192));
    }

    [Fact]
    public void ResolveChunkSize_OtherBackends_CapAtTwoK()
    {
        Assert.Equal(2048, PrefillChunking.ResolveChunkSize(BackendType.Cpu, 8192));
        Assert.Equal(2048, PrefillChunking.ResolveChunkSize(BackendType.Mlx, 8192));
        Assert.Equal(2048, PrefillChunking.ResolveChunkSize(BackendType.GgmlMetal, 8192));
    }

    [Fact]
    public void ResolveChunkSize_ShortPrompts_NotInflated()
    {
        // A 500-token prompt must never request a chunk larger than itself,
        // otherwise the chunked loop would pad past the buffer end.
        Assert.Equal(500, PrefillChunking.ResolveChunkSize(BackendType.Cpu, 500));
        Assert.Equal(500, PrefillChunking.ResolveChunkSize(BackendType.GgmlCuda, 500));
    }

    [Fact]
    public void ShouldChunk_True_OnlyWhenChunkSmallerThanPrompt()
    {
        Assert.False(PrefillChunking.ShouldChunk(BackendType.Cpu, 100, out int small));
        Assert.Equal(100, small);

        Assert.True(PrefillChunking.ShouldChunk(BackendType.Cpu, 5000, out int big));
        Assert.Equal(2048, big);
    }
}
