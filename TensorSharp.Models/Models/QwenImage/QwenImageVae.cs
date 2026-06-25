// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>A single-frame VAE latent: 16 channels at 1/8 spatial resolution, planar CHW.</summary>
    public sealed class VaeLatent
    {
        public int Channels { get; }
        public int Height { get; }
        public int Width { get; }
        /// <summary>Planar CHW, length <c>Channels*Height*Width</c>.</summary>
        public float[] Data { get; }

        public VaeLatent(int channels, int height, int width, float[] data)
        {
            Channels = channels; Height = height; Width = width; Data = data;
        }
    }

    /// <summary>
    /// Qwen-Image VAE (AutoencoderKLQwenImage, a WAN-style 3D causal-conv autoencoder):
    /// encodes an RGB image to a 16-channel latent and decodes a latent back to RGB.
    /// Weights load by name from the companion VAE GGUF (<see cref="QwenImageModel.VaeGguf"/>).
    /// </summary>
    internal sealed partial class QwenImageVae : IDisposable
    {
        private readonly QwenImageModel _model;

        public QwenImageVae(QwenImageModel model)
        {
            _model = model;

            // Run the conv stack on the device when the model is on a GGML backend
            // (the pure-C# VAE conv is the ~459 s/1 MP CPU-bound, GPU-idle phase).
            bool ggml = model.Backend is BackendType.GgmlCuda or BackendType.GgmlCpu or BackendType.GgmlMetal;
            VaeReferenceMath.UseGpuConv = ggml &&
                Environment.GetEnvironmentVariable("TS_QWEN_VAE_GPU") != "0";
            if (VaeReferenceMath.UseGpuConv)
            {
                GgmlBasicOps.EnsureBackendAvailable(model.Backend switch
                {
                    BackendType.GgmlCuda => GgmlBackendType.Cuda,
                    BackendType.GgmlMetal => GgmlBackendType.Metal,
                    _ => GgmlBackendType.Cpu,
                });
            }
        }

        public VaeLatent Encode(RgbImage image) => EncodeCore(image);

        public RgbImage Decode(VaeLatent latent) => DecodeCore(latent);

        public void Dispose() => DisposeCore();
    }
}
