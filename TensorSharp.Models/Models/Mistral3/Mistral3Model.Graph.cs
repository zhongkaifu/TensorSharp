// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.GGML;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public partial class Mistral3Model
    {
        protected override RotarySpec GraphRotary => _rotary;

        /// <summary>The YaRN rotary, then (with an original context) the per-position query scale
        /// <c>1 + beta * ln(1 + floor(pos / n_ctx_orig))</c>.</summary>
        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.Rope(q, _rotary);
            k = step.Rope(k, _rotary);
            if (_ropeOrigCtx <= 0) return;
            int[] positions = step.PositionValues;
            var scale = new float[positions.Length];
            bool any = false;
            for (int t = 0; t < positions.Length; t++)
            {
                float interval = MathF.Floor((float)positions[t] / _ropeOrigCtx);
                scale[t] = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                any |= MathF.Abs(scale[t] - 1.0f) >= 1e-7f;
            }
            if (any)
                q = step.Op(GraphOp.Mul, q, step.InputF32(scale, 1, 1, positions.Length));
        }
    }
}
