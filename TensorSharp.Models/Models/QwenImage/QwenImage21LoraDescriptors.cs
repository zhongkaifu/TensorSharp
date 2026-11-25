// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The descriptors a QwenImage21LoraSet builds in native memory (one adapter a tensor-parallel rank) and the denoiser's graph
// emitter reads. They began as the retired native forward's ABI and keep its layout: the factors they point at are the set's
// own allocations, which the bound denoiser registers with the GGML host-memory registry.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>A LoRA update of one projection, applied unmerged: y = RowScale * (W x) + Up (Down x). Down is ggml [In, Rank]
    /// (PyTorch lora_A), Up is ggml [Rank, Out] (PyTorch lora_B) with every scale folded in.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct QwenImage21Lora
    {
        public IntPtr Down, Up, RowScale;
        /// <summary>GGML type of both factors: 1 = F16, 0 = F32.</summary>
        public int Type, Rank;
        public long In, Out;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct QwenImage21BlockLora
    {
        public QwenImage21Lora Q, K, V, Out, Gate, Up, Down;
    }

    /// <summary>A LoRA plug-in's changes to the transformer.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct QwenImage21Adapter
    {
        public int StructBytes, NumLayers;
        public QwenImage21Lora ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
        public IntPtr Blocks;
        /// <summary>The step's replacement of proj_out ([dim, channels]), or zero.</summary>
        public IntPtr OutputHead;
        public int OutputHeadType, Reserved;
    }
}
