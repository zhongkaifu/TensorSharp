using System;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3NativeGraphAbiTests
{
    [Fact]
    public void DitDescriptor_MatchesNativeGraphArenaLayout()
    {
        // The paired native static_assert validates the C++ side independently.
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(552, Marshal.SizeOf<H3DitForwardArgs>());
        Assert.Equal(528, Marshal.OffsetOf<H3DitForwardArgs>(nameof(H3DitForwardArgs.GraphArena)).ToInt32());
        Assert.Equal(536, Marshal.OffsetOf<H3DitForwardArgs>(nameof(H3DitForwardArgs.GraphArenaBytes)).ToInt32());
        Assert.Equal(544, Marshal.OffsetOf<H3DitForwardArgs>(nameof(H3DitForwardArgs.Adapter)).ToInt32());
    }

    [Fact]
    public void BlockWeights_MatchTheNativeLayout()
    {
        Assert.Equal(272, Marshal.SizeOf<H3DitBlockW>());
        Assert.Equal(224, Marshal.SizeOf<H3RefinerBlockW>());
    }

    [Fact]
    public void LoraAdapter_MatchesGgmlH3LoraHeader()
    {
        // TSGH3Lora / TSGH3BlockLora / TSGH3DitAdapter in ggml_h3_lora.h, whose static_asserts
        // pin the same numbers on the native side.
        Assert.Equal(40, Marshal.SizeOf<H3Lora>());
        Assert.Equal(0, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Down)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Up)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Type)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Rank)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.In)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Out)).ToInt32());

        Assert.Equal(240, Marshal.SizeOf<H3BlockLora>());
        string[] parts = { nameof(H3BlockLora.Q), nameof(H3BlockLora.K), nameof(H3BlockLora.V),
                           nameof(H3BlockLora.Out), nameof(H3BlockLora.Fc1), nameof(H3BlockLora.Fc2) };
        for (int i = 0; i < parts.Length; i++)
            Assert.Equal(i * 40, Marshal.OffsetOf<H3BlockLora>(parts[i]).ToInt32());

        Assert.Equal(32, Marshal.SizeOf<H3DitAdapter>());
        Assert.Equal(0, Marshal.OffsetOf<H3DitAdapter>(nameof(H3DitAdapter.StructBytes)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<H3DitAdapter>(nameof(H3DitAdapter.NumBlocks)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<H3DitAdapter>(nameof(H3DitAdapter.NumRefinerBlocks)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<H3DitAdapter>(nameof(H3DitAdapter.Blocks)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<H3DitAdapter>(nameof(H3DitAdapter.Refiner)).ToInt32());
    }
}
