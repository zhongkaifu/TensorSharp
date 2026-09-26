// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>Wan 2.1 / 2.2 video-generation architecture plug-in.</summary>
    internal static class WanVideoArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "wan",
            DisplayName = "Wan video",
            Aliases = new[] { "wan", "wan2.1", "wan2.2" },
            Factory = c => new WanVideoModel(c.GgufPath, c.Backend),
            ApplyNativeTunables = ApplyMetalTunables,
            // The factory never passes the degree or group on, so without this
            // declaration --tp N was dropped in silence and a distributed group
            // was left waiting on collectives this rank never issues.
            MultiGpu = MultiGpuMode.SingleDevice,
            MultiGpuLimitation =
                "wan has no tensor-parallel or layer-split path; it runs on one GPU and extra GPUs stay idle.",
        };

        /// <summary>Process-wide ggml-metal switch this family needs set before its
        /// weights load. No-op on every other backend.</summary>
        private static void ApplyMetalTunables(ModelCreateContext c)
        {
            if (c.Backend != BackendType.GgmlMetal)
                return;

            var probe = c.Probe;
            // Wan renders NaN - uniformly black frames - on Metal whenever ggml
            // routes mul_mm through the Metal 4 tensor API. Diagnosis (2026-08-13):
            // the corruption is confined to the VAE's conv GEMMs. The tensor-API
            // mul_mm intermittently misreads operand columns there on the FIRST
            // pass over a graph (M5, macOS 26.6) - buffer-layout dependent (32x32
            // latents failed while 33x33 packed differently and passed),
            // first-run-only (recomputes read back the previous pass's bytes and
            // look clean), and immune to every kill switch (fusion / concurrency /
            // graph-optimize off, even serialized 64-node slices with full syncs).
            // The same GEMMs are bit-correct in isolation and LLM/DiT-style GEMMs
            // never corrupt - an upstream ggml-metal/driver defect, and has_tensor
            // is a device-level property fixed at init, so the kernel choice
            // cannot be scoped per-op.
            //
            // Re-verified 2026-08-17 that this is NOT yet fixed, and how easy it is
            // to conclude otherwise: an isolated WanVideoBench decode of synthetic
            // latents at five shapes - including the 32x32 all-NaN repro - matched
            // the non-tensor decode to 91-93 dB PSNR with no NaN. The next full
            // 1088x832x121f generation with the tensor API on produced 121 BLACK
            // frames. Only an end-to-end video decides this.
            //
            // With the tensor API on, the VAE stays CORRECT because
            // ggml_ops_wan.cpp routes its convs through ggml_conv_2d_direct on
            // tensor-API devices (wan_vae_gemm_budget) - slower than im2col+GEMM,
            // but a FIXED per-video cost, while the tensor API's DiT speedup
            // scales with step count and model size. Measured at 480x480x9f,
            // 6 steps, M5 Pro:
            //   A14B I2V: 17.1 vs 30.2 s/step (1.77x); VAE enc+dec 135s vs 19s
            //             -> break-even ~9 steps; the 40-step recipe is ~33% faster
            //             with the tensor API (~13.7 vs ~20.5 min).
            //   TI2V-5B:  1.6 vs 2.9 s/step; VAE decode 179s vs 13s
            //             -> break-even ~128 steps; never wins.
            // So the default follows the DiT class: enabled for A14B/14B-class
            // models (patch_embedding oc >= 5120), disabled for smaller ones.
            // TS_WAN_METAL_TENSOR_API=1/0 forces either way. When upstream fixes
            // the tensor-API mul_mm, enable it everywhere and drop the direct-conv
            // carve-out - and confirm with a full video, not a decode probe.
            long dim = 0;
            if (probe.Tensors.TryGetValue("patch_embedding.weight", out var patch) && patch.Shape.Length >= 5)
                dim = (long)patch.Shape[4];

            string env = Environment.GetEnvironmentVariable("TS_WAN_METAL_TENSOR_API");
            bool enable = string.Equals(env, "1", StringComparison.Ordinal) ||
                          (!string.Equals(env, "0", StringComparison.Ordinal) && dim >= 5120);
            if (enable)
            {
                Console.WriteLine("  [wan] Metal 4 tensor API enabled (~1.8x faster DiT steps; the VAE runs " +
                                  "on the slower-but-correct direct-conv path; TS_WAN_METAL_TENSOR_API=0 opts out).");
                return;
            }

            try
            {
                if (TensorSharp.GGML.GgmlBasicOps.SetNativeEnvironmentVariable(
                        "GGML_METAL_TENSOR_DISABLE", "1", overwrite: false))
                {
                    Console.WriteLine("  [wan] Metal 4 tensor API disabled for this process (fastest correct " +
                                      "config for this model class; TS_WAN_METAL_TENSOR_API=1 opts in: faster " +
                                      "DiT steps but a slower direct-conv VAE).");
                }
            }
            catch (DllNotFoundException) { /* managed-only host; nothing to configure */ }
            catch (EntryPointNotFoundException) { /* older GgmlOps without the setter */ }
        }
    }
}
