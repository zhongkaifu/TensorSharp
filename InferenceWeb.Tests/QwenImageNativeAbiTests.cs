// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the managed layouts of the structs shared with the native Qwen-Image-2.1 kernels.
/// TSGgml_QwenTeTrunk checks only its descriptor's size (a mismatch silently selects the
/// several-times-slower per-op text encoder); the per-layer weight array is not checked
/// at all, so a one-sided layout change would mis-stride every layer. The native side
/// holds the same numbers in static_asserts (ggml_ops_qwen_image.cpp).
/// </summary>
public sealed class QwenImageNativeAbiTests
{
    [Fact]
    public void TextEncoderTrunkStructsMatchTheNativeLayout()
    {
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(48, Marshal.SizeOf<QImgAttnW>());           // TSGImgAttnW
        Assert.Equal(368, Marshal.SizeOf<QwenTeLayerW>());       // TSGTeLayerW
        Assert.Equal(88, Marshal.SizeOf<QwenTeTrunkArgs>());     // TSGgmlQwenTeTrunkDesc

        Assert.Equal(0, (int)Marshal.OffsetOf<QImgAttnW>(nameof(QImgAttnW.W)));
        Assert.Equal(8, (int)Marshal.OffsetOf<QImgAttnW>(nameof(QImgAttnW.Type)));
        Assert.Equal(16, (int)Marshal.OffsetOf<QImgAttnW>(nameof(QImgAttnW.Ne0)));
        Assert.Equal(40, (int)Marshal.OffsetOf<QImgAttnW>(nameof(QImgAttnW.B)));

        Assert.Equal(16, (int)Marshal.OffsetOf<QwenTeLayerW>(nameof(QwenTeLayerW.Q)));
        Assert.Equal(352, (int)Marshal.OffsetOf<QwenTeLayerW>(nameof(QwenTeLayerW.QNorm)));
        Assert.Equal(360, (int)Marshal.OffsetOf<QwenTeLayerW>(nameof(QwenTeLayerW.KNorm)));

        Assert.Equal(32, (int)Marshal.OffsetOf<QwenTeTrunkArgs>(nameof(QwenTeTrunkArgs.Layers)));
        Assert.Equal(44, (int)Marshal.OffsetOf<QwenTeTrunkArgs>(nameof(QwenTeTrunkArgs.StructBytes)));
        Assert.Equal(68, (int)Marshal.OffsetOf<QwenTeTrunkArgs>(nameof(QwenTeTrunkArgs.Eps)));
        Assert.Equal(72, (int)Marshal.OffsetOf<QwenTeTrunkArgs>(nameof(QwenTeTrunkArgs.DeepStack)));
        Assert.Equal(80, (int)Marshal.OffsetOf<QwenTeTrunkArgs>(nameof(QwenTeTrunkArgs.DeepStackCount)));
    }

    [Fact]
    public void VaeStructsKeepTheirNativeLayout()
    {
        // Mirrored by ggml_ops_qwen_image.cpp (static_asserts) and by the native test
        // tests/qwen_image21_vae_shortcut_test.cpp. The op and weight arrays are not size-checked.
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(64, Marshal.SizeOf<QwenVaeOp>());           // TSGVaeOp: 16 x int32
        Assert.Equal(16, Marshal.SizeOf<QwenVaeWeightRef>());    // TSGVaeWeightRef
        Assert.Equal(112, Marshal.SizeOf<Conv2dArgs>());         // TSGgmlConv2dDesc
    }
}
