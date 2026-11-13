// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Models
{
    public sealed partial class HunyuanDenseModel
    {
        protected override RotarySpec GraphRotary => _rotary;

        /// <summary>NeoX rotary, then per-head Q/K RMSNorm (Hunyuan normalizes after rotating).</summary>
        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.RmsNorm(step.Rope(q, _rotary), LayerName(layer, "attn_q_norm.weight"));
            k = step.RmsNorm(step.Rope(k, _rotary), LayerName(layer, "attn_k_norm.weight"));
        }
    }
}
