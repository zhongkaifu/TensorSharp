// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the root of this source tree.
using System;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Single-frame Qwen-Image-2.1 RGBA VAE: 64 normalized channels at 1/16 resolution.</summary>
    internal sealed class QwenImage21Vae : IDisposable, IGgmlHostMemoryOwner
    {
        private VaeWeights _weights;
        public QwenImage21Vae(IFloatTensorStore source, BackendType backend)
        {
            QwenImage21CompanionValidation.ValidateVae(source);
            _weights = VaeWeights.Load(source);
            // Vulkan convolutions run on the device too: the native F32 convolution
            // rescales inputs past the F16 range that Vulkan's matrix units accept.
            bool ggml = backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlMetal
                or BackendType.GgmlVulkan;
            VaeReferenceMath.UseGpuConv = ggml;
            // The whole-VAE graph on CUDA and Metal. On an M5 Pro at 2048x2048 it decodes in
            // 25 s instead of 85 s with a 45 GB instead of 64 GB peak footprint, at 97-100 dB
            // PSNR to the per-op path (1024x1024: 5 s instead of 13-15 s, where
            // stable-diffusion.cpp takes 7 s). Per convolution on ggml-cpu, and on Vulkan:
            // NVIDIA Vulkan multiplies through F16 cooperative-matrix operands and only the
            // per-convolution path rescales for that (the fused graph turns the decoder's
            // >65504 activations into NaN there).
            VaeReferenceMath.UseFusedGraph21 = backend is BackendType.GgmlCuda or BackendType.GgmlMetal;
            if (ggml) GgmlBasicOps.EnsureBackendAvailable(backend switch
            {
                BackendType.GgmlCuda => GgmlBackendType.Cuda,
                BackendType.GgmlMetal => GgmlBackendType.Metal,
                BackendType.GgmlVulkan => GgmlBackendType.Vulkan,
                _ => GgmlBackendType.Cpu,
            });
        }
        public VaeLatent Encode(RgbImage image) => VaeReferenceMath.Encode21(_weights, image);
        public RgbImage Decode(VaeLatent latent) => VaeReferenceMath.Decode21(_weights, latent);
        public void Dispose() { _weights?.FusedGraph?.Dispose(); _weights = null; }

        /// <summary>The fused graph's weight copies (the decoder's GGUF mapping is the model's).</summary>
        void IGgmlHostMemoryOwner.CollectHostMemory(System.Collections.Generic.List<(IntPtr Base, long Length)> ranges,
            System.Collections.Generic.List<IntPtr> keys) => (_weights?.FusedGraph as IGgmlHostMemoryOwner)?.CollectHostMemory(ranges, keys);
    }

    internal static partial class VaeReferenceMath
    {
        internal static bool UseFusedGraph21;

        private static QwenImageVaeGraph FusedGraph21(VaeWeights weights)
        {
            if (!UseGpuConv || !UseFusedGraph21 || weights.FusedGraphBuildFailed) return null;
            if (weights.FusedGraph == null)
            {
                weights.FusedGraph = QwenImageVaeGraph.TryBuild(weights);
                if (weights.FusedGraph == null) weights.FusedGraphBuildFailed = true;
            }
            return weights.FusedGraph;
        }

        internal static readonly float[] Qwen21Mean = { 0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f, 1.8873f, -1.7177f, -0.6510f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f, 2.0210f, 0.0472f, 0.9320f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f, -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f, -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f, -1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f, -0.4433f, 0.3764f, 0.5729f, -2.0595f, 1.0960f, -1.3260f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f, 0.3026f, 1.9373f, 1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f };
        internal static readonly float[] Qwen21Std = { 3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f, 3.0785f, 3.6500f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f, 3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f, 3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f, 3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.1490f, 3.7662f, 3.6746f, 3.4563f, 3.8161f };

        // Causal conv on T=1 (see LastTemporalSlice). The managed fast path convolves with the
        // layer's packed kernel; with upsample2x it reads x through the decoder's nearest 2x
        // upsample instead of materializing the 4x larger map.
        private static Feature Conv21(VaeWeights w, string prefix, Feature x, int pad, bool upsample2x = false)
        {
            var shape = w.Shape(prefix + ".weight");
            int oc = (int)shape[0], ic = (int)shape[1];
            int kh = (int)shape[^2], kw = (int)shape[^1];
            int kd = shape.Length == 5 ? (int)shape[2] : 1;
            float[] bias = w.Get(prefix + ".bias");
            if (FastCpu)
                return Conv2dCpu(x, w.PackedKernel(prefix + ".weight", oc, ic, kd, kh * kw), bias, oc, kh, kw,
                    1, 1, pad, pad, pad, pad, upsample2x);
            float[] weight = w.KernelSlice(prefix + ".weight", oc, ic, kd, kh, kw);
            return Conv2d(upsample2x ? NearestUpsample2x(x) : x, weight, oc, ic, kh, kw, bias, 1, 1, pad, pad, pad, pad);
        }

        // Norm then SiLU: one fused pass on the managed fast path (vectorized exp), the exact
        // scalar SiLU elsewhere. With inPlace (x is dead afterwards) the fast path overwrites x
        // instead of allocating a map; the result is either x or a new map, and the caller
        // releases accordingly.
        private static Feature NormSilu21(VaeWeights w, string gammaName, Feature x, bool inPlace = false)
        {
            if (FastCpu) return RmsNormChannelFast(x, w.Get(gammaName), silu: true, inPlace);
            var t = RmsNormChannel(x, w.Get(gammaName));
            SiluInPlace(t.D);
            return t;
        }

        // Every map below is released (VaeFeaturePool.Release) as soon as its last reader is
        // done; that is a no-op off the managed fast path. ownsInput says whether x may be
        // released here: the first block of a decoder stage reads the stage input, which the
        // stage still needs for its shortcut.
        private static Feature Residual21(VaeWeights w, string prefix, Feature x, bool ownsInput)
        {
            bool hasShortcut = w.Has(prefix + ".shortcut.weight");
            var residual = hasShortcut ? Conv21(w, prefix + ".shortcut", x, 0) : x;
            // With a shortcut conv, x has no reader after the first norm: normalize it in place.
            bool inputDead = ownsInput && hasShortcut;
            var t = NormSilu21(w, prefix + ".residual.0.gamma", x, inPlace: inputDead);
            var u = Conv21(w, prefix + ".residual.2", t, 1);
            if (!ReferenceEquals(t, x)) VaeFeaturePool.Release(t);
            if (inputDead) VaeFeaturePool.Release(x);
            var v = NormSilu21(w, prefix + ".residual.3.gamma", u, inPlace: true);
            var result = Conv21(w, prefix + ".residual.6", v, 1);
            VaeFeaturePool.Release(v);
            if (!ReferenceEquals(v, u)) VaeFeaturePool.Release(u);
            AddInPlace(result, residual);
            if (hasShortcut || ownsInput) VaeFeaturePool.Release(residual);
            return result;
        }

        private static Feature Mid21(VaeWeights w, string prefix, Feature x)
        {
            x = Residual21(w, prefix + ".0", x, ownsInput: true);
            x = AttentionBlock(w, prefix + ".1", x);
            return Residual21(w, prefix + ".2", x, ownsInput: true);
        }

        // Wan2.2's shortcut rearranges [C,T,H,W] -> [OC,group,T',H',W'] and
        // averages group. T=1 is front padded with zeros, including the factor-2
        // time stages: dropping this zero half changes the image encoder output.
        internal static Feature AverageDown21(Feature x, int outChannels, int timeFactor, int spatialFactor)
        {
            int group = x.C * timeFactor * spatialFactor * spatialFactor / outChannels;
            var y = new Feature(outChannels, x.H / spatialFactor, x.W / spatialFactor);
            int sf2 = spatialFactor * spatialFactor;
            // Output channels are independent and each keeps its sequential group order, so the
            // parallel pass is bit-identical to the serial one.
            CpuPackedGemm.ForEach(outChannels, (long)x.D.Length >= 1 << 16, oc =>
            {
                for (int g = 0; g < group; g++)
                {
                    int packed = oc * group + g;
                    int ic = packed / (timeFactor * sf2);
                    int t = packed / sf2 % timeFactor;
                    if (t != timeFactor - 1) continue;
                    int dy = packed / spatialFactor % spatialFactor, dx = packed % spatialFactor;
                    for (int h = 0; h < y.H; h++)
                    {
                        int dst = (oc * y.H + h) * y.W;
                        int src = (ic * x.H + h * spatialFactor + dy) * x.W + dx;
                        for (int v = 0; v < y.W; v++)
                            y.D[dst + v] += x.D[src + v * spatialFactor] / group;
                    }
                }
            }, CpuPool);
            return y;
        }

        // repeat_interleave each input channel, then unpack channels into T,H,W;
        // the first chunk retains only the final temporal sample.
        internal static Feature DuplicateUp21(Feature x, int outChannels, int timeFactor, int spatialFactor)
        {
            int repeats = outChannels * timeFactor * spatialFactor * spatialFactor / x.C;
            var y = Feature.Uninitialized(outChannels, x.H * spatialFactor, x.W * spatialFactor);
            CpuPackedGemm.ForEach(outChannels * y.H, (long)y.D.Length >= 1 << 16, row =>
            {
                int oc = row / y.H, h = row - oc * y.H;
                int dst = row * y.W;
                // Output column v = u*s + j reads input column u of channel ic(j): the source
                // channel depends only on v % s, so each output row interleaves s input rows.
                for (int j = 0; j < spatialFactor; j++)
                {
                    int packed = ((oc * timeFactor + timeFactor - 1) * spatialFactor + h % spatialFactor) * spatialFactor + j;
                    int src = (packed / repeats * x.H + h / spatialFactor) * x.W;
                    for (int u = 0; u < x.W; u++) y.D[dst + u * spatialFactor + j] = x.D[src + u];
                }
            }, CpuPool);
            return y;
        }

        /// <summary>
        /// a += DuplicateUp21(x, a.C, timeFactor, spatialFactor), without materializing the
        /// duplicated map (at 2048x2048 the decoder's last shortcut is 4.8 GB, twice the stage
        /// input it is read from). Every element is the same single float addition as
        /// AddInPlace(a, DuplicateUp21(...)), so the result is bit-identical.
        /// </summary>
        internal static Feature AddDuplicateUp21InPlace(Feature a, Feature x, int timeFactor, int spatialFactor)
        {
            int outChannels = a.C;
            if (a.H != x.H * spatialFactor || a.W != x.W * spatialFactor ||
                (long)outChannels * timeFactor * spatialFactor * spatialFactor % x.C != 0)
                throw new ArgumentException($"cannot add the {spatialFactor}x duplicate of [{x.C}, {x.H}, {x.W}] to [{a.C}, {a.H}, {a.W}]");
            int repeats = outChannels * timeFactor * spatialFactor * spatialFactor / x.C;
            // DuplicateUp21's index math: output column v = u*s + j reads input column u of the
            // channel selected by (oc, h % s, j), row h / s.
            CpuPackedGemm.ForEach(outChannels * a.H, (long)a.D.Length >= 1 << 16, row =>
            {
                int oc = row / a.H, h = row - oc * a.H;
                int dst = row * a.W;
                for (int j = 0; j < spatialFactor; j++)
                {
                    int packed = ((oc * timeFactor + timeFactor - 1) * spatialFactor + h % spatialFactor) * spatialFactor + j;
                    int src = (packed / repeats * x.H + h / spatialFactor) * x.W;
                    for (int u = 0; u < x.W; u++) a.D[dst + u * spatialFactor + j] += x.D[src + u];
                }
            }, CpuPool);
            return a;
        }

        internal static VaeLatent Encode21(VaeWeights w, RgbImage image)
        {
            if (image.Width % 16 != 0 || image.Height % 16 != 0)
                throw new ArgumentException("Qwen-Image-2.1 VAE dimensions must be multiples of 16.");
            int hw = image.Width * image.Height;
            var x = new Feature(4, image.Height, image.Width);
            for (int i = 0; i < hw; i++)
            {
                for (int c = 0; c < 3; c++) x.D[c * hw + i] = image.Pixels[i * 3 + c] * 2f - 1f;
                x.D[3 * hw + i] = image.Alpha == null ? 1f : image.Alpha[i] * 2f - 1f;
            }
            var fused = FusedGraph21(w);
            using var pool = VaeFeaturePool.Enter(FastCpu);
            if (fused != null && fused.TryEncode(x.D, x.H, x.W, out var moments, out int lh, out int lw))
                x = new Feature(128, lh, lw, moments);
            else
            {
                x = Consume(Conv21(w, "encoder.conv1", x, 1), x);
                for (int stage = 0; stage < 5; stage++)
                {
                    string prefix = $"encoder.downsamples.{stage}.downsamples";
                    // The stage's output width (96, 192, 384, 768, 768 in the released VAE).
                    int channels = (int)w.Shape(prefix + ".0.residual.6.weight")[0];
                    var shortcut = AverageDown21(x, channels, stage is >= 1 and <= 3 ? 2 : 1, stage < 4 ? 2 : 1);
                    x = Residual21(w, prefix + ".0", x, ownsInput: true);
                    x = Residual21(w, prefix + ".1", x, ownsInput: true);
                    if (stage < 4) x = Consume(Downsample(w, prefix + ".2", x, x.C), x);
                    x = AddInPlace(x, shortcut);
                    VaeFeaturePool.Release(shortcut);
                }
                x = Mid21(w, "encoder.middle", x);
                var normalized = NormSilu21(w, "encoder.head.0.gamma", x, inPlace: true);
                if (!ReferenceEquals(normalized, x)) VaeFeaturePool.Release(x);
                x = Consume(Conv21(w, "encoder.head.2", normalized, 1), normalized);
                x = Consume(Conv21(w, "conv1", x, 0), x);
            }
            int pixels = x.H * x.W;
            var latent = new float[64 * pixels];
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++) latent[c * pixels + i] = (x.D[c * pixels + i] - Qwen21Mean[c]) / Qwen21Std[c];
            VaeFeaturePool.Release(x);
            return new VaeLatent(64, x.H, x.W, latent);
        }

        /// <summary>Returns <paramref name="result"/> after releasing the map it was computed
        /// from (<paramref name="input"/> has no other reader).</summary>
        private static Feature Consume(Feature result, Feature input)
        {
            if (!ReferenceEquals(result, input)) VaeFeaturePool.Release(input);
            return result;
        }

        internal static RgbImage Decode21(VaeWeights w, VaeLatent latent)
        {
            if (latent.Channels != 64 || latent.Data.Length != 64L * latent.Height * latent.Width)
                throw new ArgumentException("Qwen-Image-2.1 VAE requires 64 latent channels.");
            int pixels = latent.Height * latent.Width;
            var x = new Feature(64, latent.Height, latent.Width);
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++) x.D[c * pixels + i] = latent.Data[c * pixels + i] * Qwen21Std[c] + Qwen21Mean[c];
            var fused = FusedGraph21(w);
            using var pool = VaeFeaturePool.Enter(FastCpu);
            if (fused != null && fused.TryDecode(x.D, x.H, x.W, out var rgba, out int oh, out int ow))
                x = new Feature(4, oh, ow, rgba);
            else
            {
                x = Consume(Conv21(w, "conv2", x, 0), x);
                x = Consume(Conv21(w, "decoder.conv1", x, 1), x);
                x = Mid21(w, "decoder.middle", x);
                for (int stage = 0; stage < 5; stage++)
                {
                    string prefix = $"decoder.upsamples.{stage}.upsamples";
                    // The stage shortcut duplicates the stage input up to the upsampled output.
                    // It is added straight from the input at the end (AddDuplicateUp21InPlace)
                    // instead of being materialized up front: the input is kept alive, half to a
                    // quarter the size of the duplicated map.
                    var stageInput = x;
                    bool upsample = stage < 4;
                    for (int j = 0; j < 3; j++) x = Residual21(w, prefix + $".{j}", x, ownsInput: j > 0 || !upsample);
                    if (upsample)
                    {
                        var up = Consume(Conv21(w, prefix + ".3.resample.1", x, 1, upsample2x: true), x);
                        x = AddDuplicateUp21InPlace(up, stageInput, stage < 3 ? 2 : 1, 2);
                        VaeFeaturePool.Release(stageInput);
                    }
                }
                var normalized = NormSilu21(w, "decoder.head.0.gamma", x, inPlace: true);
                if (!ReferenceEquals(normalized, x)) VaeFeaturePool.Release(x);
                x = Consume(Conv21(w, "decoder.head.2", normalized, 1), normalized);
            }
            if (Array.Exists(x.D, v => !float.IsFinite(v)))
                throw new InvalidOperationException("Qwen-Image-2.1 VAE produced non-finite RGBA values.");
            pixels = x.H * x.W;
            var rgb = new float[3 * pixels];
            var alpha = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                alpha[i] = Math.Clamp((x.D[3 * pixels + i] + 1f) * .5f, 0f, 1f);
                for (int c = 0; c < 3; c++) rgb[i * 3 + c] = Math.Clamp((x.D[c * pixels + i] + 1f) * .5f, 0f, 1f);
            }
            VaeFeaturePool.Release(x);
            return new RgbImage(x.W, x.H, rgb, alpha);
        }
    }
}
