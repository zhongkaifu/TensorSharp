// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the root of this source tree.
using System;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Single-frame Qwen-Image-2.1 RGBA VAE: 64 normalized channels at 1/16 resolution.</summary>
    internal sealed class QwenImage21Vae : IDisposable
    {
        private VaeWeights _weights;
        public QwenImage21Vae(QwenImageModel model)
        {
            QwenImage21CompanionValidation.ValidateVae(model.VaeWeightSource);
            _weights = VaeWeights.Load(model.VaeWeightSource);
            // Vulkan convolutions run on the device too: the native F32 convolution
            // rescales inputs past the F16 range that Vulkan's matrix units accept.
            bool ggml = model.Backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlMetal
                or BackendType.GgmlVulkan;
            VaeReferenceMath.UseGpuConv = ggml && Environment.GetEnvironmentVariable("TS_QWEN_VAE_GPU") != "0";
            // The whole-VAE graph, by default on CUDA and Metal. On an M5 Pro at 2048x2048 it
            // decodes in 25 s instead of 85 s with a 45 GB instead of 64 GB peak footprint,
            // at 97-100 dB PSNR to the per-op path (1024x1024: 5 s instead of 13-15 s, where
            // stable-diffusion.cpp takes 7 s). TS_QWEN21_VAE_FUSED=1 forces it on another GGML
            // backend, =0 turns it off.
            // Never on Vulkan: NVIDIA Vulkan multiplies through F16 cooperative-matrix
            // operands and only the per-convolution path rescales for that (the fused graph
            // turns the decoder's >65504 activations into NaN there).
            string fused = Environment.GetEnvironmentVariable("TS_QWEN21_VAE_FUSED");
            if (fused == "1" && model.Backend == BackendType.GgmlVulkan)
                Console.WriteLine("  [vae21] TS_QWEN21_VAE_FUSED=1 ignored on Vulkan: the whole-VAE graph overflows F16 " +
                    "cooperative-matrix operands there; decoding per convolution instead.");
            VaeReferenceMath.UseFusedGraph21 = model.Backend != BackendType.GgmlVulkan &&
                (fused == "1" || (fused != "0" && model.Backend is BackendType.GgmlCuda or BackendType.GgmlMetal));
            if (ggml) GgmlBasicOps.EnsureBackendAvailable(model.Backend switch
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
    }

    internal static partial class VaeReferenceMath
    {
        internal static bool UseFusedGraph21;
        private static readonly bool TraceVae21 = Environment.GetEnvironmentVariable("TS_QWEN21_VAE_TRACE") == "1";

        private static void Trace21(string name, float[] values)
        {
            if (!TraceVae21) return;
            int nonFinite = 0;
            float max = 0;
            foreach (float value in values)
                if (!float.IsFinite(value)) nonFinite++;
                else max = Math.Max(max, Math.Abs(value));
            Console.WriteLine($"  [vae21] {name}: count={values.Length} nonfinite={nonFinite} maxabs={max:G9}");
            if (nonFinite != 0) throw new InvalidOperationException($"Qwen-Image-2.1 VAE first non-finite tensor: {name}.");
        }

        private static QwenImageVaeGraph FusedGraph21(VaeWeights weights)
        {
            if (!UseGpuConv || !UseFusedGraph || !UseFusedGraph21 || weights.FusedGraphBuildFailed ||
                Environment.GetEnvironmentVariable("TS_QWEN21_VAE_FUSED") == "0") return null;
            if (weights.FusedGraph == null)
            {
                weights.FusedGraph = QwenImageVaeGraph.TryBuild(weights);
                if (weights.FusedGraph == null) weights.FusedGraphBuildFailed = true;
            }
            return weights.FusedGraph;
        }

        internal static readonly float[] Qwen21Mean = { 0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f, 1.8873f, -1.7177f, -0.6510f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f, 2.0210f, 0.0472f, 0.9320f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f, -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f, -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f, -1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f, -0.4433f, 0.3764f, 0.5729f, -2.0595f, 1.0960f, -1.3260f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f, 0.3026f, 1.9373f, 1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f };
        internal static readonly float[] Qwen21Std = { 3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f, 3.0785f, 3.6500f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f, 3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f, 3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f, 3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.1490f, 3.7662f, 3.6746f, 3.4563f, 3.8161f };

        private static Feature Conv21(VaeWeights w, string prefix, Feature x, int pad)
        {
            var shape = w.Shape(prefix + ".weight");
            int oc = (int)shape[0], ic = (int)shape[1];
            int kh = (int)shape[^2], kw = (int)shape[^1];
            int kd = shape.Length == 5 ? (int)shape[2] : 1;
            float[] weight = w.Get(prefix + ".weight"), bias = w.Get(prefix + ".bias");
            Trace21(prefix + ".input", x.D);
            Trace21(prefix + ".weight", weight);
            Trace21(prefix + ".bias", bias);
            var result = CausalConv3dT1(x, weight, oc, ic, kd, kh, kw, bias, pad);
            Trace21(prefix + ".output", result.D);
            return result;
        }

        private static Feature Residual21(VaeWeights w, string prefix, Feature x)
        {
            var residual = w.Has(prefix + ".shortcut.weight") ? Conv21(w, prefix + ".shortcut", x, 0) : x;
            var t = RmsNormChannel(x, w.Get(prefix + ".residual.0.gamma"));
            Trace21(prefix + ".residual.0.output", t.D);
            SiluInPlace(t.D);
            t = Conv21(w, prefix + ".residual.2", t, 1);
            t = RmsNormChannel(t, w.Get(prefix + ".residual.3.gamma"));
            Trace21(prefix + ".residual.3.output", t.D);
            SiluInPlace(t.D);
            t = Conv21(w, prefix + ".residual.6", t, 1);
            var result = AddInPlace(t, residual);
            Trace21(prefix + ".output", result.D);
            return result;
        }

        private static Feature Mid21(VaeWeights w, string prefix, Feature x)
        {
            x = Residual21(w, prefix + ".0", x);
            x = AttentionBlock(w, prefix + ".1", x);
            Trace21(prefix + ".1.output", x.D);
            return Residual21(w, prefix + ".2", x);
        }

        // Wan2.2's shortcut rearranges [C,T,H,W] -> [OC,group,T',H',W'] and
        // averages group. T=1 is front padded with zeros, including the factor-2
        // time stages: dropping this zero half changes the image encoder output.
        internal static Feature AverageDown21(Feature x, int outChannels, int timeFactor, int spatialFactor)
        {
            int group = x.C * timeFactor * spatialFactor * spatialFactor / outChannels;
            var y = new Feature(outChannels, x.H / spatialFactor, x.W / spatialFactor);
            int sf2 = spatialFactor * spatialFactor;
            for (int oc = 0; oc < outChannels; oc++)
                for (int g = 0; g < group; g++)
                {
                    int packed = oc * group + g;
                    int ic = packed / (timeFactor * sf2);
                    int t = packed / sf2 % timeFactor;
                    if (t != timeFactor - 1) continue;
                    int dy = packed / spatialFactor % spatialFactor, dx = packed % spatialFactor;
                    for (int h = 0; h < y.H; h++)
                        for (int v = 0; v < y.W; v++)
                            y.D[(oc * y.H + h) * y.W + v] += x.D[(ic * x.H + h * spatialFactor + dy) * x.W + v * spatialFactor + dx] / group;
                }
            return y;
        }

        // repeat_interleave each input channel, then unpack channels into T,H,W;
        // the first chunk retains only the final temporal sample.
        internal static Feature DuplicateUp21(Feature x, int outChannels, int timeFactor, int spatialFactor)
        {
            int repeats = outChannels * timeFactor * spatialFactor * spatialFactor / x.C;
            var y = new Feature(outChannels, x.H * spatialFactor, x.W * spatialFactor);
            for (int oc = 0; oc < outChannels; oc++)
                for (int h = 0; h < y.H; h++)
                    for (int v = 0; v < y.W; v++)
                    {
                        int packed = ((oc * timeFactor + timeFactor - 1) * spatialFactor + h % spatialFactor) * spatialFactor + v % spatialFactor;
                        int ic = packed / repeats;
                        y.D[(oc * y.H + h) * y.W + v] = x.D[(ic * x.H + h / spatialFactor) * x.W + v / spatialFactor];
                    }
            return y;
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
            Trace21("encoder.input", x.D);
            var fused = FusedGraph21(w);
            if (fused != null && fused.TryEncode(x.D, x.H, x.W, out var moments, out int lh, out int lw))
                x = new Feature(128, lh, lw, moments);
            else
            {
                x = Conv21(w, "encoder.conv1", x, 1);
                int[] channels = {96,192,384,768,768};
                for (int stage = 0; stage < 5; stage++)
                {
                    string prefix = $"encoder.downsamples.{stage}.downsamples";
                    var shortcut = AverageDown21(x, channels[stage], stage is >= 1 and <= 3 ? 2 : 1, stage < 4 ? 2 : 1);
                    x = Residual21(w, prefix + ".0", x);
                    x = Residual21(w, prefix + ".1", x);
                    if (stage < 4) x = Downsample(w, prefix + ".2", x, channels[stage]);
                    x = AddInPlace(x, shortcut);
                }
                x = Mid21(w, "encoder.middle", x);
                x = RmsNormChannel(x, w.Get("encoder.head.0.gamma"));
                SiluInPlace(x.D);
                x = Conv21(w, "encoder.head.2", x, 1);
                x = Conv21(w, "conv1", x, 0);
            }
            int pixels = x.H * x.W;
            var latent = new float[64 * pixels];
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++) latent[c * pixels + i] = (x.D[c * pixels + i] - Qwen21Mean[c]) / Qwen21Std[c];
            return new VaeLatent(64, x.H, x.W, latent);
        }

        internal static RgbImage Decode21(VaeWeights w, VaeLatent latent)
        {
            if (latent.Channels != 64 || latent.Data.Length != 64L * latent.Height * latent.Width)
                throw new ArgumentException("Qwen-Image-2.1 VAE requires 64 latent channels.");
            int pixels = latent.Height * latent.Width;
            var x = new Feature(64, latent.Height, latent.Width);
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++) x.D[c * pixels + i] = latent.Data[c * pixels + i] * Qwen21Std[c] + Qwen21Mean[c];
            Trace21("decoder.input", x.D);
            var fused = FusedGraph21(w);
            if (fused != null && fused.TryDecode(x.D, x.H, x.W, out var rgba, out int oh, out int ow))
                x = new Feature(4, oh, ow, rgba);
            else
            {
                x = Conv21(w, "conv2", x, 0);
                x = Conv21(w, "decoder.conv1", x, 1);
                x = Mid21(w, "decoder.middle", x);
                int[] channels = {1152,1152,576,288,144};
                for (int stage = 0; stage < 5; stage++)
                {
                    string prefix = $"decoder.upsamples.{stage}.upsamples";
                    var shortcut = stage < 4 ? DuplicateUp21(x, channels[stage], stage < 3 ? 2 : 1, 2) : null;
                    for (int j = 0; j < 3; j++) x = Residual21(w, prefix + $".{j}", x);
                    if (stage < 4)
                    {
                        x = Conv21(w, prefix + ".3.resample.1", NearestUpsample2x(x), 1);
                        x = AddInPlace(x, shortcut);
                    }
                }
                x = RmsNormChannel(x, w.Get("decoder.head.0.gamma"));
                SiluInPlace(x.D);
                x = Conv21(w, "decoder.head.2", x, 1);
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
            return new RgbImage(x.W, x.H, rgb, alpha);
        }
    }
}
