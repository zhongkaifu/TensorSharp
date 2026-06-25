// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Managed CPU reference implementation of the Qwen-Image VAE (AutoencoderKLQwenImage,
// a WAN-style 3D causal-conv autoencoder). This is the *correctness oracle*: it is
// faithful to the diffusers/sglang reference but not optimized (plain array loops),
// so it is exercised on small images for verification. The performance path (native
// ggml graph / CUDA) is validated against this and against diffusers.
//
// Single-image simplification: Qwen-Image treats an image as a 1-frame "video".
// With one temporal chunk the WAN feature-cache temporal convolutions (time_conv)
// are skipped, so the network reduces to a 2D-conv pipeline with a degenerate
// time axis (T=1). Causal Conv3d therefore only front-pads the (length-1) time
// dim and behaves as a 2D conv. This file implements exactly that case.
// ============================================================================
using System;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageVae
    {
        // dim_mult (1,2,4,4), base_dim 96 -> channel stages 96,192,384,384; z_dim 16.
        private VaeWeights _w;

        private VaeLatent EncodeCore(RgbImage image)
        {
            EnsureWeights();
            return VaeReferenceMath.Encode(_w, image);
        }

        private RgbImage DecodeCore(VaeLatent latent)
        {
            EnsureWeights();
            return VaeReferenceMath.Decode(_w, latent);
        }

        private void EnsureWeights()
        {
            _w ??= VaeWeights.Load(_model.VaeGguf);
        }

        private void DisposeCore()
        {
            _w = null;
        }
    }
}
