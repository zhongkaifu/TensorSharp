// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.MLX
{
    /// <summary>The Metal kernels of the graph executor's fused regions (TensorSharp.Graph.GraphFusion).</summary>
    internal static partial class MlxNative
    {
        // GraphFusion.DeltaNetPrep for one sequence. One threadgroup of D threads per (head, row): q heads,
        // then k heads, then v heads; thread i owns channel head * D + i. Inputs (2-D or 1-D views, read
        // through their strides): window [Channels, ConvRows] (time innermost), qkv [rows, Channels], conv_w
        // [Channels, Width], alpha and beta [rows, VHeads], dt and a [VHeads]. Outputs are contiguous: conv_in
        // [Channels, ConvRows + rows] (time innermost), q and k L2-normed per head (x * rsqrt(sum x^2 + eps)),
        // v, g = softplus(alpha + dt) * a with ggml's softplus, beta = sigmoid(beta).
        private const string DeltaNetPrepSource = @"
threadgroup float partial[32];
const uint i = thread_position_in_threadgroup.x;
const uint head = threadgroup_position_in_grid.x;
const uint t = threadgroup_position_in_grid.y;
const int rows = qkv_shape[0];
const int span = ConvRows + rows;
const uint c = head * D + i;
const int64_t qr = qkv_strides[0], qc = qkv_strides[1], wc = window_strides[0], wt = window_strides[1];
float x = 0.0f;
for (int j = 0; j < Width; j++) {
    int p = (int)t + j;
    float in = p < ConvRows ? window[c * wc + p * wt] : qkv[(p - ConvRows) * qr + c * qc];
    x += in * conv_w[c * conv_w_strides[0] + j * conv_w_strides[1]];
}
conv_in[c * span + ConvRows + t] = qkv[t * qr + c * qc];
if (t == 0) {
    for (int p = 0; p < ConvRows; p++) conv_in[c * span + p] = window[c * wc + p * wt];
}
float y = x / (1.0f + metal::exp(-x));
if (head < 2 * KHeads) {
    float sq = simd_sum(y * y);
    if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < D / 32; s++) total += partial[s];
    float yn = y * metal::rsqrt(total + as_type<float>(EpsBits));
    if (head < KHeads) q_out[(t * KHeads + head) * D + i] = yn;
    else k_out[(t * KHeads + head - KHeads) * D + i] = yn;
} else {
    const uint vh = head - 2 * KHeads;
    v_out[(t * VHeads + vh) * D + i] = y;
    if (i == 0) {
        float s = alpha[t * alpha_strides[0] + vh * alpha_strides[1]] + dt[vh * dt_strides[0]];
        float sp = s > 20.0f ? s : metal::log(1.0f + metal::exp(s));
        g_out[t * VHeads + vh] = sp * a[vh * a_strides[0]];
        beta_out[t * VHeads + vh] = 1.0f / (1.0f + metal::exp(-beta[t * beta_strides[0] + vh * beta_strides[1]]));
    }
}
";

        // GraphFusion.GatedRmsNorm: one threadgroup of D threads per row of y [n, D]:
        // y * rsqrt(mean(y^2) + eps) * w * silu(z), or sigmoid(z) with Sigmoid.
        private const string GatedRmsNormSource = @"
threadgroup float partial[32];
const uint i = thread_position_in_threadgroup.x;
const uint r = threadgroup_position_in_grid.x;
const float v = y[r * D + i];
float sq = simd_sum(v * v);
if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (uint s = 0; s < D / 32; s++) total += partial[s];
const float zz = z[r * D + i];
out[r * D + i] = v * metal::rsqrt(total / D + as_type<float>(EpsBits)) * w[i] * (Sigmoid ? 1.0f / (1.0f + metal::exp(-zz)) : zz / (1.0f + metal::exp(-zz)));
";

        // ggml_gated_delta_net for one row a sequence (decode): MlxNative's T=1 kernel with ggml's inputs, the
        // decay as log (exp(g) here) and the output scaled by 1/sqrt(Dk) (ggml scales q; the output is linear
        // in q). One simdgroup per (sequence * head, value row); state [n, Hv, Dv, Dk].
        private const string GatedDeltaDecodeSource = @"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx % Hk;
constexpr int n_per_t = Dk / 32;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;
auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
auto v_row = v + b_idx * Hv * Dv + hv_idx * Dv;
auto y_row = y + b_idx * Hv * Dv + hv_idx * Dv;
const float g_scalar = metal::exp(g[b_idx * Hv + hv_idx]);
const float beta_scalar = beta[b_idx * Hv + hv_idx];
const float v_scalar = v_row[dv_idx];
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) state[i] = i_state[n_per_t * dk_idx + i];
float kv_mem = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = state[i] * g_scalar;
  kv_mem = fma(state[i], k_[s_idx], kv_mem);
}
kv_mem = simd_sum(kv_mem);
const float delta = (v_scalar - kv_mem) * beta_scalar;
float out_val = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = fma(k_[s_idx], delta, state[i]);
  out_val = fma(state[i], q_[s_idx], out_val);
}
out_val = simd_sum(out_val);
if (thread_index_in_simdgroup == 0) y_row[dv_idx] = out_val * metal::rsqrt((float)Dk);
for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
";

        private static MlxFastMetalKernel deltaNetPrepKernel, gatedRmsNormKernel, gatedDeltaDecodeKernel;

        private static MlxFastMetalKernel GraphKernel(ref MlxFastMetalKernel slot, string name, string[] inputs, string[] outputs, string source,
            bool stridedInputs = false)
        {
            lock (fastKernelSync)
            {
                if (!slot.IsValid)
                    slot = CreateFastMetalKernel(name, inputs, outputs, source, string.Empty, ensureRowContiguous: !stridedInputs);
                if (!slot.IsValid)
                    throw new NotSupportedException($"Unable to initialize MLX kernel {name}.");
                return slot;
            }
        }

        private static MlxArray[] RunGraphKernel(MlxFastMetalKernel kernel, MlxArray[] inputs, (int[] Shape, string What)[] outputs,
            (string Name, int Value)[] template, int gridX, int gridY, int group, string what)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray ins = default, outs = default;
                try
                {
                    foreach (var (name, value) in template) AddTemplateInt(config, name, value);
                    foreach (var (shape, _) in outputs)
                        Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {what} output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, gridX, gridY, 1), $"configuring {what} grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, group, 1, 1), $"configuring {what} threadgroup");
                    ins = CreateVectorArray(inputs);
                    outs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outs, kernel, ins, config, DefaultStream()), $"running {what}");
                    if ((int)mlx_vector_array_size(outs) < outputs.Length)
                        throw new InvalidOperationException($"{what} produced {(int)mlx_vector_array_size(outs)} outputs.");
                    var result = new MlxArray[outputs.Length];
                    for (int o = 0; o < result.Length; o++)
                        Check(mlx_vector_array_get(out result[o], outs, (nuint)o), $"reading {what} output {o}");
                    return result;
                }
                finally
                {
                    if (ins.IsValid) _ = mlx_vector_array_free(ins);
                    if (outs.IsValid) _ = mlx_vector_array_free(outs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        /// <summary>GraphFusion.DeltaNetPrep (shapes above); returns conv_in, q, k, v, g, beta.</summary>
        internal static MlxArray[] DeltaNetPrep(MlxArray window, MlxArray qkv, MlxArray convW, MlxArray alpha, MlxArray beta,
            MlxArray dt, MlxArray a, int rows, int channels, int convRows, int width, int d, int kHeads, int vHeads, float eps)
        {
            MlxFastMetalKernel kernel = GraphKernel(ref deltaNetPrepKernel, "tensorsharp_graph_deltanet_prep",
                new[] { "window", "qkv", "conv_w", "alpha", "beta", "dt", "a" },
                new[] { "conv_in", "q_out", "k_out", "v_out", "g_out", "beta_out" }, DeltaNetPrepSource, stridedInputs: true);
            int heads = 2 * kHeads + vHeads;
            return RunGraphKernel(kernel, new[] { window, qkv, convW, alpha, beta, dt, a },
                new[]
                {
                    (new[] { channels, convRows + rows }, "conv_in"), (new[] { rows, kHeads, d }, "q"), (new[] { rows, kHeads, d }, "k"),
                    (new[] { rows, vHeads, d }, "v"), (new[] { rows, vHeads, 1 }, "g"), (new[] { rows, vHeads, 1 }, "beta"),
                },
                new[]
                {
                    ("Channels", channels), ("ConvRows", convRows), ("Width", width), ("D", d), ("KHeads", kHeads),
                    ("VHeads", vHeads), ("EpsBits", BitConverter.SingleToInt32Bits(eps)),
                },
                heads * d, rows, d, "the delta rule's prep");
        }

        /// <summary>ggml_gated_delta_net for one row of <paramref name="batch"/> sequences: q, k [batch, Hk, Dk],
        /// v [batch, Hv, Dv], g (log decay) and beta [batch, Hv], state [batch, Hv, Dv, Dk]; returns y
        /// [batch, Hv, Dv] and the next state.</summary>
        internal static (MlxArray Y, MlxArray State) GatedDeltaDecode(MlxArray q, MlxArray k, MlxArray v, MlxArray g, MlxArray beta,
            MlxArray state, int batch, int hk, int hv, int dk, int dv)
        {
            MlxFastMetalKernel kernel = GraphKernel(ref gatedDeltaDecodeKernel, "tensorsharp_graph_gated_delta_decode",
                new[] { "q", "k", "v", "g", "beta", "state_in" }, new[] { "y", "state_out" }, GatedDeltaDecodeSource);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray ins = default, outs = default;
                try
                {
                    AddTemplateInt(config, "Dk", dk);
                    AddTemplateInt(config, "Dv", dv);
                    AddTemplateInt(config, "Hk", hk);
                    AddTemplateInt(config, "Hv", hv);
                    int[] yShape = { batch, hv, dv }, sShape = { batch, hv, dv, dk };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, yShape, (nuint)yShape.Length, ToMlxDtype(DType.Float32)), "configuring gated-delta decode output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, sShape, (nuint)sShape.Length, ToMlxDtype(DType.Float32)), "configuring gated-delta decode state");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv), "configuring gated-delta decode grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, Math.Min(dv, 4), 1), "configuring gated-delta decode threadgroup");
                    ins = CreateVectorArray(q, k, v, g, beta, state);
                    outs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outs, kernel, ins, config, DefaultStream()), "running gated-delta decode");
                    Check(mlx_vector_array_get(out MlxArray y, outs, 0), "reading gated-delta decode output");
                    Check(mlx_vector_array_get(out MlxArray next, outs, 1), "reading gated-delta decode state");
                    return (y, next);
                }
                finally
                {
                    if (ins.IsValid) _ = mlx_vector_array_free(ins);
                    if (outs.IsValid) _ = mlx_vector_array_free(outs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        /// <summary>GraphFusion.GatedRmsNorm over <paramref name="n"/> rows of <paramref name="d"/>.</summary>
        internal static MlxArray GatedRmsNorm(MlxArray y, MlxArray z, MlxArray w, int n, int d, float eps, bool sigmoid = false)
        {
            MlxFastMetalKernel kernel = GraphKernel(ref gatedRmsNormKernel, "tensorsharp_graph_gated_rms_norm",
                new[] { "y", "z", "w" }, new[] { "out" }, GatedRmsNormSource);
            return RunGraphKernel(kernel, new[] { y, z, w }, new[] { (new[] { n, d }, "out") },
                new[] { ("D", d), ("EpsBits", BitConverter.SingleToInt32Bits(eps)), ("Sigmoid", sigmoid ? 1 : 0) }, n * d, 1, d, "the gated RMS norm")[0];
        }
    }
}
