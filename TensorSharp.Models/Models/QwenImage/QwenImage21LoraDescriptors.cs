// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The updates a QwenImage21LoraSet describes for each tensor-parallel rank, which the denoiser's graph emitter reads. The
// factors they point at are the set's own native allocations (GGML binds them as host buffers), which the bound denoiser
// registers with the GGML host-memory registry.
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>A LoRA update of one projection, applied unmerged: y = RowScale * (W x) + Up (Down x). Down is ggml [In, Rank]
    /// (PyTorch lora_A), Up is ggml [Rank, Out] (PyTorch lora_B) with every scale folded in.</summary>
    internal struct QwenImage21Lora
    {
        public IntPtr Down, Up, RowScale;
        /// <summary>GGML type of both factors: 1 = F16, 0 = F32.</summary>
        public int Type, Rank;
        public long In, Out;
    }

    internal struct QwenImage21BlockLora
    {
        public QwenImage21Lora Q, K, V, Out, Gate, Up, Down;
    }

    /// <summary>A LoRA plug-in's changes to the transformer on one tensor-parallel rank: the projections outside the blocks
    /// and each block's.</summary>
    internal sealed class QwenImage21Adapter
    {
        public QwenImage21Lora ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
        public QwenImage21BlockLora[] Blocks = Array.Empty<QwenImage21BlockLora>();
    }
}
