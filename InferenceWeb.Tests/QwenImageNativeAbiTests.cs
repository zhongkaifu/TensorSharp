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
    public void DiTForwardStructsKeepTheirNativeLayout()
    {
        // Mirrored by static_asserts in ggml_ops_qwen_image21.h. The native forward
        // rejects a descriptor whose StructBytes differs, so a one-sided change fails
        // loudly there; the prefix-cache fields must also land where native reads them.
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(40, Marshal.SizeOf<QwenImage21Weight>());      // TSGQi21Weight
        Assert.Equal(296, Marshal.SizeOf<QwenImage21Block>());      // TSGQi21Block
        Assert.Equal(16, Marshal.SizeOf<QwenImage21Segment>());     // TSGQi21Segment
        Assert.Equal(472, Marshal.SizeOf<QwenImage21ForwardArgs>()); // TSGQi21Desc
        Assert.Equal(376, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.Blocks)));
        Assert.Equal(440, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.Eps)));
        Assert.Equal(448, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.PrefixCacheKey)));
        Assert.Equal(456, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.PrefixCacheType)));
        Assert.Equal(460, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.TpRanks)));
        Assert.Equal(464, (int)Marshal.OffsetOf<QwenImage21ForwardArgs>(nameof(QwenImage21ForwardArgs.Adapter)));
        Assert.Equal(24, Marshal.SizeOf<QwenImage21PrefixCacheInfo>()); // TSGQi21PrefixCacheInfo
        Assert.Equal(16, (int)Marshal.OffsetOf<QwenImage21PrefixCacheInfo>(nameof(QwenImage21PrefixCacheInfo.Bytes)));
        // Enum values are part of the ABI (TSGQi21PrefixCacheType / TSGQi21ForwardPath).
        Assert.Equal(4, (int)QwenImage21PrefixCacheType.Q8_0V);
        Assert.Equal(3, (int)QwenImage21ForwardPath.Cached);
        Assert.Equal(4, (int)QwenImage21ForwardPath.Declined);
    }

    [Fact]
    public void LoraAdapterStructsKeepTheirNativeLayout()
    {
        // Mirrored by static_asserts in ggml_ops_qwen_image21.h (TSGQi21Lora,
        // TSGQi21BlockLora, TSGQi21Adapter). The adapter carries its own StructBytes;
        // the per-layer block array is not size-checked natively, so a one-sided change
        // would mis-stride every block's updates.
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(48, Marshal.SizeOf<QwenImage21Lora>());            // TSGQi21Lora
        Assert.Equal(0, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.Down)));
        Assert.Equal(8, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.Up)));
        Assert.Equal(16, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.RowScale)));
        Assert.Equal(24, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.Type)));
        Assert.Equal(28, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.Rank)));
        Assert.Equal(32, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.In)));
        Assert.Equal(40, (int)Marshal.OffsetOf<QwenImage21Lora>(nameof(QwenImage21Lora.Out)));

        Assert.Equal(336, Marshal.SizeOf<QwenImage21BlockLora>());      // TSGQi21BlockLora
        Assert.Equal(0, (int)Marshal.OffsetOf<QwenImage21BlockLora>(nameof(QwenImage21BlockLora.Q)));
        Assert.Equal(144, (int)Marshal.OffsetOf<QwenImage21BlockLora>(nameof(QwenImage21BlockLora.Out)));
        Assert.Equal(288, (int)Marshal.OffsetOf<QwenImage21BlockLora>(nameof(QwenImage21BlockLora.Down)));

        Assert.Equal(416, Marshal.SizeOf<QwenImage21Adapter>());        // TSGQi21Adapter
        Assert.Equal(0, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.StructBytes)));
        Assert.Equal(4, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.NumLayers)));
        Assert.Equal(8, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.ImageIn)));
        Assert.Equal(344, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.ProjOut)));
        Assert.Equal(392, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.Blocks)));
        Assert.Equal(400, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.OutputHead)));
        Assert.Equal(408, (int)Marshal.OffsetOf<QwenImage21Adapter>(nameof(QwenImage21Adapter.OutputHeadType)));
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
