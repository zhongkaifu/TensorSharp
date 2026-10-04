// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the root of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35VisionEncoder
    {
        // Restrict the new path to GGML CUDA. The switch
        // restores the previous host-erf/per-block implementation for A/B runs.
        private bool UseFusedVision21 => _qwenImage21 &&
            _allocator is GgmlAllocator allocator && allocator.Context.BackendType == GgmlBackendType.Cuda &&
            Environment.GetEnvironmentVariable("TS_QWEN21_VISION_FUSED") != "0";

        /// <summary>Qwen3-VL main image embedding and additions for the first language blocks.
        /// The deepstack projectors normalize merged (4*1152) features, unlike the final
        /// projector, whose normalization precedes spatial merging.</summary>
        internal Tensor[] EncodeWithDeepStack(float[] pixels, int height, int width)
        {
            var deepStack = new List<Tensor>();
            try
            {
                var main = EncodeCore(pixels, null, height, width, deepStack);
                deepStack.Insert(0, main);
                return deepStack.ToArray();
            }
            catch
            {
                foreach (var tensor in deepStack) tensor.Dispose();
                throw;
            }
        }

        private Tensor ProjectDeepStack(Tensor hidden, int layer, int patches)
        {
            int unit = _spatialMergeSize * _spatialMergeSize;
            string prefix = $"v.deepstack.{layer}";
            using var merged = hidden.View(patches / unit, _hiddenSize * unit);
            using var normalized = LayerNormOp(merged, prefix + ".norm.weight", prefix + ".norm.bias");
            using var fc1 = LinearForwardWithBias(normalized, prefix + ".fc1.weight", prefix + ".fc1.bias");
            ApplyVisionGelu(fc1);
            return LinearForwardWithBias(fc1, prefix + ".fc2.weight", prefix + ".fc2.bias");
        }
        private void ApplyVisionGelu(Tensor tensor)
        {
            if (!_qwenImage21) { Ops.GELU(tensor, tensor); return; }
            ApplyVisionGeluErf(tensor);
        }

        private void ApplyVisionMergerGelu(Tensor tensor)
        {
            if (!_mergerGeluErf) { ApplyVisionGelu(tensor); return; }
            ApplyVisionGeluErf(tensor);
        }

        private void ApplyVisionGeluErf(Tensor tensor)
        {
            if (UseFusedVision21 || (!_qwenImage21 && _useNativeAttention))
            {
                GgmlBasicOps.GELUErf(tensor, tensor);
                return;
            }
            if (UseCpuLinear && tensor.IsContiguous())
            {
                GeluErfInPlace(tensor);
                return;
            }
            // Ops.GELU uses the tanh approximation, so an erf activation needs
            // an explicit path on backends without the native erf unary op.
            var values = tensor.GetElementsAsFloat((int)tensor.ElementCount());
            System.Threading.Tasks.Parallel.For(0, values.Length, i =>
            {
                double x = values[i] / Math.Sqrt(2.0);
                double sign = x < 0 ? -1.0 : 1.0;
                x = Math.Abs(x);
                double t = 1.0 / (1.0 + 0.3275911 * x);
                double p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
                double erf = sign * (1.0 - p * Math.Exp(-x * x));
                values[i] = (float)(0.5 * values[i] * (1.0 + erf));
            });
            tensor.SetElementsAsFloat(values);
            if (_useNativeAttention)
                GgmlBasicOps.InvalidateHostBuffer(TensorComputePrimitives.GetStoragePointer(tensor));
        }

        /// <summary>
        /// The same Abramowitz-Stegun erf GELU as the host loop above, evaluated in double in
        /// place over host memory (pure-C# backend): four doubles per vector, chunked over the
        /// CPU pool, no copy out to a managed array and back. Only exp differs (vector vs
        /// Math.Exp, both within an ulp in double), invisible after the rounding to float.
        /// </summary>
        internal static unsafe void GeluErfInPlace(Tensor tensor)
        {
            long n = tensor.ElementCount();
            nint pL = (nint)TensorComputePrimitives.GetFloatPointer(tensor);
            const int Chunk = 16 * 1024;
            int chunks = (int)((n + Chunk - 1) / Chunk);
            QwenImage.CpuPackedGemm.ForEach(chunks, chunks > 1, c =>
            {
                float* p = (float*)pL + (long)c * Chunk;
                int count = (int)Math.Min(Chunk, n - (long)c * Chunk), i = 0;
                var sqrt2 = Vector256.Create(Math.Sqrt(2.0));
                var one = Vector256<double>.One;
                for (; i + 4 <= count; i += 4)
                {
                    Vector256<double> v = Vector256.WidenLower(Vector256.Create(Vector128.Load(p + i), Vector128<float>.Zero));
                    Vector256<double> x = v / sqrt2;
                    Vector256<double> sign = Vector256.ConditionalSelect(Vector256.LessThan(x, Vector256<double>.Zero),
                        Vector256.Create(-1.0), one);
                    x = Vector256.Abs(x);
                    Vector256<double> t = one / (one + Vector256.Create(0.3275911) * x);
                    Vector256<double> poly = ((((Vector256.Create(1.061405429) * t - Vector256.Create(1.453152027)) * t
                        + Vector256.Create(1.421413741)) * t - Vector256.Create(0.284496736)) * t + Vector256.Create(0.254829592)) * t;
                    Vector256<double> erf = sign * (one - poly * Vector256.Exp(-x * x));
                    Vector256<double> y = Vector256.Create(0.5) * v * (one + erf);
                    Vector128.Store(Vector256.Narrow(y, Vector256<double>.Zero).GetLower(), p + i);
                }
                for (; i < count; i++)
                {
                    double x = p[i] / Math.Sqrt(2.0);
                    double sign = x < 0 ? -1.0 : 1.0;
                    x = Math.Abs(x);
                    double t = 1.0 / (1.0 + 0.3275911 * x);
                    double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
                    double erf = sign * (1.0 - poly * Math.Exp(-x * x));
                    p[i] = (float)(0.5 * p[i] * (1.0 + erf));
                }
            });
        }
    }
}
