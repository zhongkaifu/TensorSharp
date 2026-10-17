// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Arithmetic and native storage shared by the LoRA loaders that pack factors for a native
// graph's unmerged update path (QwenImage21LoraSet, MiniMaxH3LoraSet).
using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace TensorSharp.Models
{
    /// <summary>
    /// Conditioning of LoRA factors before they are stored for a native graph: rank-component
    /// balancing, the F16 range check, scaling and the F16/F32 store.
    /// </summary>
    /// <remarks>
    /// Plain loops, not TensorPrimitives. On Mono, which runs the Mac and iOS apps, its generic
    /// vector operators ran hundreds of times slower than on CoreCLR. MEASURED in the Release
    /// Mac app: loading Viggle Turbo with Film Stills spent 497 thread-seconds in MaxMagnitude
    /// and 73 in Balance's norms, products and quotients, 31 s of wall time for what the CLI
    /// loads in 0.6 s. A loop like these is fast on both runtimes. The norms are summed in double
    /// in index order, so every runtime computes the same bits.
    /// </remarks>
    internal static class LoraFactorMath
    {
        /// <summary>The magnitude below which a factor is stored as F16; a group with any value
        /// at or above it stays F32 (F16's largest finite value is 65504).</summary>
        internal const float F16Limit = 60000f;

        /// <summary>
        /// Give every rank component equal down-row and up-column norms; the product is
        /// unchanged. <paramref name="down"/> is [<paramref name="rank"/>, <paramref name="input"/>]
        /// and <paramref name="up"/> [<paramref name="output"/>, <paramref name="rank"/>], both
        /// row-major; only those leading elements are read and written.
        /// </summary>
        internal static void Balance(float[] down, float[] up, int rank, long input, long output)
        {
            var upNorm = new double[rank];
            for (long o = 0; o < output; o++)
                for (int r = 0; r < rank; r++) { double v = up[o * rank + r]; upNorm[r] += v * v; }
            var divisors = new float[rank];
            for (int r = 0; r < rank; r++)
            {
                divisors[r] = 1f;
                Span<float> row = down.AsSpan((int)(r * input), (int)input);
                double downNorm = Norm(row);
                double upLength = Math.Sqrt(upNorm[r]);
                if (downNorm <= 0 || upLength <= 0) continue;
                float k = (float)Math.Sqrt(upLength / downNorm);
                Scale(row, k);
                divisors[r] = k;
            }
            // Row by row: the up factor is [out, rank], so a column at a time would stride.
            for (long o = 0; o < output; o++)
            {
                Span<float> row = up.AsSpan((int)(o * rank), rank);
                for (int r = 0; r < row.Length; r++) row[r] /= divisors[r];
            }
        }

        /// <summary>Whether every value is a number of magnitude below <paramref name="limit"/>; a NaN is not.</summary>
        internal static bool AllBelow(ReadOnlySpan<float> values, float limit)
        {
            foreach (float v in values)
                if (!(MathF.Abs(v) < limit)) return false;
            return true;
        }

        internal static void Scale(Span<float> values, float factor)
        {
            for (int i = 0; i < values.Length; i++) values[i] *= factor;
        }

        internal static double Norm(ReadOnlySpan<float> values)
        {
            double sum = 0;
            foreach (float v in values) sum += (double)v * v;
            return Math.Sqrt(sum);
        }

        /// <summary>Store <paramref name="values"/> at <paramref name="destination"/> as F16 or F32.</summary>
        internal static unsafe void Store(float[] values, IntPtr destination, bool f16)
        {
            if (f16) TensorPrimitives.ConvertToHalf(values, new Span<System.Half>((void*)destination, values.Length));
            else values.CopyTo(new Span<float>((void*)destination, values.Length));
        }

        /// <summary><see cref="Store"/> as a plain loop (a <c>(Half)</c> cast per value, the same
        /// round-to-nearest-even conversion), for the loaders that run on Mono.</summary>
        internal static unsafe void StoreLoop(ReadOnlySpan<float> values, IntPtr destination, bool f16)
        {
            if (f16)
            {
                var target = new Span<System.Half>((void*)destination, values.Length);
                for (int i = 0; i < values.Length; i++) target[i] = (System.Half)values[i];
            }
            else
            {
                values.CopyTo(new Span<float>((void*)destination, values.Length));
            }
        }
    }

    /// <summary>
    /// Page-aligned native allocations holding packed LoRA factors and their descriptors,
    /// freed together. 16 KiB alignment lets the GGML backends wrap a factor in place.
    /// </summary>
    internal sealed class LoraFactorBuffers : IDisposable
    {
        internal const long PageAlignment = 16384;

        private readonly List<IntPtr> _buffers = new();

        internal unsafe IntPtr Allocate(long bytes)
        {
            void* p = NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 1), (nuint)PageAlignment);
            if (p == null) throw new OutOfMemoryException($"Unable to allocate {bytes} bytes for LoRA factors.");
            var ptr = (IntPtr)p;
            lock (_buffers) _buffers.Add(ptr);
            return ptr;
        }

        internal unsafe IntPtr Copy(ReadOnlySpan<float> values)
        {
            IntPtr p = Allocate(values.Length * sizeof(float));
            values.CopyTo(new Span<float>((void*)p, values.Length));
            return p;
        }

        public void Dispose()
        {
            foreach (var p in _buffers) unsafe { NativeMemory.AlignedFree(p.ToPointer()); }
            _buffers.Clear();
        }
    }
}
