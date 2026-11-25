// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Globalization;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Memory a Qwen-Image-2.1 request needs on the pure-C# <c>cpu</c> backend, estimated from the
    /// output size before any work starts, so a size the machine cannot hold is refused in
    /// milliseconds instead of failing (or paging for hours) in the VAE decode after the whole
    /// denoise has run.
    /// </summary>
    /// <remarks>
    /// <para>Two phases grow with the image. The <b>VAE decode</b> holds full-resolution feature
    /// maps: with every map released when read (<see cref="VaeFeaturePool"/>), the largest live set
    /// is stage 4's first residual block - its 288-channel input (normalized in place), the
    /// 144-channel shortcut and the 144-channel first convolution, 576 floats (2304 bytes) per
    /// output pixel - next to the packed F32 decoder weights. The <b>denoise</b> holds eight
    /// [tokens, 4096] F32 activations (joint, hidden, query, output, key, value, transposed keys,
    /// per-head values; 128 KiB per token) plus the file-mapped transformer weights, which must stay
    /// resident to run at speed.</para>
    /// <para>Measured on an i7-11800H (32 GB, Windows, Q4_K_M transformer, Pruna 5-step LoRA): a
    /// 1024x1024 request peaked at 3.7 GiB of commit (the decode: 2.25 GiB of live maps) and 8.0 GiB
    /// of working set (plus the mapped transformer); a 2048x2048 VAE encode+decode peaked at 10.5 GiB
    /// of commit, 9.0 GiB of it live maps. The estimate is 4.1 and 10.8 GiB there. A 2048x2048
    /// transformer step peaked at 8.3 GiB of working set (4.4 GiB commit plus the mapped
    /// transformer); its estimate is 8.4 GiB.</para>
    /// </remarks>
    internal static class QwenImage21CpuMemory
    {
        /// <summary>TS_QWEN_IMAGE_CPU_MEMORY_CHECK=0 skips the refusal (a machine with a large
        /// page file that is willing to page).</summary>
        internal const string CheckVariable = "TS_QWEN_IMAGE_CPU_MEMORY_CHECK";

        /// <summary>Largest live set of decoder feature maps per output pixel (see remarks).</summary>
        internal const long DecodeFeatureBytesPerPixel = (288 + 144 + 144) * sizeof(float);

        /// <summary>The packed F32 decoder (259M parameters) plus the encoder (79M) when an edit
        /// loaded it, plus the runtime, GC and pool slack measured on top of the live maps.</summary>
        internal const long DecodeOverheadBytes = 1850L << 20;

        /// <summary>Transformer activations per joint-sequence token (8 x 4096 F32).</summary>
        internal const long DenoiseBytesPerToken = 8L * 4096 * sizeof(float);

        /// <summary>Transformer scratch that does not grow with the image (the MLP chunk, the
        /// LoRA and prefix-cache buffers of a typical plug-in, the runtime).</summary>
        internal const long DenoiseOverheadBytes = 2560L << 20;

        /// <summary>Peak bytes of the managed VAE decode of a <paramref name="width"/> x
        /// <paramref name="height"/> image.</summary>
        internal static long EstimateDecodeBytes(int width, int height) =>
            DecodeFeatureBytesPerPixel * width * height + DecodeOverheadBytes;

        /// <summary>Peak bytes of the denoise: activations for the joint sequence plus the
        /// transformer's file-mapped weights (<paramref name="transformerFileBytes"/>).</summary>
        internal static long EstimateDenoiseBytes(int width, int height, int prefixTokens, long transformerFileBytes) =>
            DenoiseBytesPerToken * ((long)(width / 16) * (height / 16) + Math.Max(0, prefixTokens)) +
            DenoiseOverheadBytes + Math.Max(0, transformerFileBytes);

        /// <summary>The larger of the two phases.</summary>
        internal static long EstimatePeakBytes(int width, int height, int prefixTokens, long transformerFileBytes) =>
            Math.Max(EstimateDecodeBytes(width, height), EstimateDenoiseBytes(width, height, prefixTokens, transformerFileBytes));

        /// <summary>
        /// Null when a <paramref name="width"/> x <paramref name="height"/> request fits in
        /// <paramref name="totalMemoryBytes"/> (the machine's or container's physical memory), else
        /// the reason, with the largest square size that does fit, for an ArgumentException.
        /// </summary>
        internal static string Refusal(int width, int height, int prefixTokens, long transformerFileBytes, long totalMemoryBytes)
        {
            if (totalMemoryBytes <= 0) return null;   // unknown: nothing to compare with
            long needed = EstimatePeakBytes(width, height, prefixTokens, transformerFileBytes);
            if (needed <= totalMemoryBytes) return null;
            bool decodeBound = EstimateDecodeBytes(width, height) >= EstimateDenoiseBytes(width, height, prefixTokens, transformerFileBytes);
            int side = LargestSquareSide(prefixTokens, transformerFileBytes, totalMemoryBytes);
            string suggestion = side > 0
                ? $"Choose a smaller size (at most about {side}x{side}, or the same area at another aspect ratio) with --width/--height " +
                  "(width/height in an API request)"
                : "This machine cannot hold even a small image next to the transformer weights; use a smaller (more quantized) transformer GGUF";
            return string.Create(CultureInfo.InvariantCulture,
                $"Qwen-Image-2.1 at {width}x{height} on the pure-C# cpu backend needs about {Gib(needed):F1} GiB of memory " +
                $"({(decodeBound ? "the VAE decode's full-resolution feature maps" : "the transformer's activations and weights")}), " +
                $"but this machine has {Gib(totalMemoryBytes):F1} GiB. {suggestion}, or use a GPU backend. " +
                $"Set {CheckVariable}=0 to run anyway (it will page heavily or run out of memory).");
        }

        /// <summary>The largest multiple-of-32 square side whose estimate fits, or 0.</summary>
        internal static int LargestSquareSide(int prefixTokens, long transformerFileBytes, long totalMemoryBytes)
        {
            int best = 0;
            for (int side = 32; side <= 16384; side += 32)
            {
                if (EstimatePeakBytes(side, side, prefixTokens, transformerFileBytes) > totalMemoryBytes) break;
                best = side;
            }
            return best;
        }

        /// <summary>Physical memory the process may use (the machine's, or a container limit),
        /// or 0 when the runtime cannot tell.</summary>
        internal static long TotalMemoryBytes()
        {
            try { return Math.Max(0, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes); }
            catch (PlatformNotSupportedException) { return 0; }
        }

        /// <summary>Physical memory not in use right now, from a fresh GC memory-load sample (see
        /// QwenImage21Denoiser's free-memory rule for why the gen-0 collection), or 0 if unknown.</summary>
        internal static long FreeMemoryBytes()
        {
            try
            {
                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                var info = GC.GetGCMemoryInfo();
                return info.TotalAvailableMemoryBytes > 0 ? Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes) : 0;
            }
            catch (PlatformNotSupportedException) { return 0; }
        }

        internal static double Gib(long bytes) => bytes / (1024.0 * 1024 * 1024);
    }
}
