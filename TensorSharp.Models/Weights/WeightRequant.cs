// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>Load-time re-quantization of weights a model reads every token (for example Strata's Q8_0 read of
    /// Qwen3.8-Flash-Next's BF16 hyper-connection matrices). Shared by the GGML and direct-CUDA paths.</summary>
    public static class WeightRequant
    {
        public const int GgmlTypeBf16 = 30, GgmlTypeQ8_0 = 8;

        public const int GgmlTypeQ4_0 = 2, GgmlTypeQ2_0 = 42;

        /// <summary>
        /// Rows of any ggml-decodable type requantized to <paramref name="dstType"/> through F32 with ggml's own quantizer
        /// (ggml_quantize_chunk), in parallel chunks: <paramref name="readRows"/>(first, count, buffer) puts source rows
        /// in a buffer (a mapped file's rows are copied straight from it). Returns aligned native memory the caller frees
        /// with NativeMemory.AlignedFree, or zero when ggml cannot produce the type without an importance matrix or the
        /// rows are not whole blocks of it.
        /// </summary>
        public static unsafe IntPtr Requantize(Action<long, long, IntPtr> readRows, int srcType, long cols, long rows, int dstType,
            out long dstBytes)
        {
            dstBytes = 0;
            long srcRow = ManagedQuantizedOps.RowSize(srcType, (int)cols), dstRow = ManagedQuantizedOps.RowSize(dstType, (int)cols);
            if (srcRow <= 0 || dstRow <= 0 || rows <= 0)
                return IntPtr.Zero;
            long total = rows * dstRow;
            // Page-aligned (16 KiB on Apple silicon): a Metal backend can then wrap the rows without a copy.
            IntPtr dst = (IntPtr)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)total, 16384);
            const long ChunkRows = 640;
            long chunks = (rows + ChunkRows - 1) / ChunkRows;
            int failed = 0;
            Parallel.For(0, chunks, c =>
            {
                long r0 = c * ChunkRows, n = Math.Min(ChunkRows, rows - r0);
                byte* raw = (byte*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(n * srcRow));
                float* f = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(n * cols * 4));
                try
                {
                    readRows(r0, n, (IntPtr)raw);
                    TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32Native(srcType, (IntPtr)raw, (IntPtr)f, n * cols);
                    long w = TensorSharp.GGML.GgmlGgufTensorDequant.QuantizeFloat32RowsOrZero(dstType, (IntPtr)f, dst + (nint)(r0 * dstRow), n, cols);
                    if (w != n * dstRow)
                        System.Threading.Interlocked.Exchange(ref failed, 1);
                }
                finally
                {
                    System.Runtime.InteropServices.NativeMemory.Free(f);
                    System.Runtime.InteropServices.NativeMemory.Free(raw);
                }
            });
            if (failed != 0)
            {
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)dst);
                return IntPtr.Zero;
            }
            dstBytes = total;
            return dst;
        }

        /// <summary>
        /// TS_Q4E_MTP_EXPERTS: the ggml type Qwen3.8-Flash-Next's MTP draft block keeps its 512 routed experts in, or -1 to
        /// keep the file's. <c>q2_0</c> is Strata's read (its draft layer's experts are Q2_0, 708 MB): the shared Q8_0
        /// head's 2.5 GB of experts take a third of the memory, which the target's expert tiers or slots then hold.
        /// Drafts only: the verify decides every token.
        /// </summary>
        public static int MtpExpertType()
        {
            string v = Environment.GetEnvironmentVariable("TS_Q4E_MTP_EXPERTS");
            if (string.IsNullOrWhiteSpace(v))
                return -1;
            return v.Trim().ToLowerInvariant() switch
            {
                "q2_0" => GgmlTypeQ2_0,
                "q4_0" => GgmlTypeQ4_0,
                "q8_0" => GgmlTypeQ8_0,
                "file" => -1,
                _ => throw new ArgumentException($"TS_Q4E_MTP_EXPERTS takes q2_0, q4_0, q8_0 or file, not '{v}'."),
            };
        }

        /// <summary>Bytes of a Q8_0 matrix of <paramref name="rows"/> rows of <paramref name="cols"/> values (a multiple of 32).</summary>
        public static long Q8_0Bytes(long rows, long cols) => rows * (cols / 32) * 34;

        /// <summary>BF16 rows to Q8_0 rows, as ggml's quantize_row_q8_0_ref: per 32 values d = max|x| / 127,
        /// q = round(x / d) (half away from zero), d stored as F16. <paramref name="cols"/> must be a multiple of 32.</summary>
        public static unsafe void Bf16ToQ8_0(ushort* src, byte* dst, long rows, long cols)
        {
            if (cols % 32 != 0)
                throw new ArgumentException($"a Q8_0 row needs a multiple of 32 values, not {cols}", nameof(cols));
            long blocks = cols / 32;
            IntPtr s = (IntPtr)src, d = (IntPtr)dst;
            Parallel.For(0, rows, r => Row((ushort*)s + r * cols, (byte*)d + r * blocks * 34, blocks));
        }

        private static unsafe void Row(ushort* x, byte* y, long blocks)
        {
            float* v = stackalloc float[32];
            for (long b = 0; b < blocks; b++)
            {
                float amax = 0;
                for (int j = 0; j < 32; j++)
                {
                    v[j] = BitConverter.UInt32BitsToSingle((uint)x[b * 32 + j] << 16);
                    amax = Math.Max(amax, Math.Abs(v[j]));
                }
                float d = amax / 127f, id = d != 0 ? 1f / d : 0f;
                byte* blk = y + b * 34;
                *(System.Half*)blk = (System.Half)d;
                for (int j = 0; j < 32; j++)
                    ((sbyte*)blk)[2 + j] = (sbyte)MathF.Round(v[j] * id, MidpointRounding.AwayFromZero);
            }
        }
    }
}
