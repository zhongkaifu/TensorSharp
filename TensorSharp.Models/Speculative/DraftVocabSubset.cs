// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace TensorSharp.Models
{
    /// <summary>
    /// A draft vocabulary: the token ids a block drafter scores, in the format of Strata's
    /// data/draft_vocab*.bin (little-endian int32 token ids, each listed once). The drafter's
    /// per-token rows are gathered into subset order once; drafted subset indices map back to token ids.
    /// </summary>
    internal static class DraftVocabSubset
    {
        /// <summary>Reads and checks a subset file against a vocabulary of <paramref name="vocab"/> tokens.
        /// Returns null and a reason when the file cannot serve as a draft vocabulary.</summary>
        internal static int[] Load(string path, long vocab, int minTokens, out string problem)
        {
            byte[] raw;
            try { raw = File.ReadAllBytes(path); }
            catch (Exception ex) { problem = ex.Message; return null; }
            return Parse(raw, vocab, minTokens, out problem);
        }

        internal static int[] Parse(byte[] raw, long vocab, int minTokens, out string problem)
        {
            problem = null;
            if (raw == null || raw.Length < 4 || raw.Length % 4 != 0)
            {
                problem = "expected little-endian int32 token ids";
                return null;
            }
            int[] map = MemoryMarshal.Cast<byte, int>(raw).ToArray();
            if (!BitConverter.IsLittleEndian)
                for (int i = 0; i < map.Length; i++)
                    map[i] = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(map[i]);
            if (map.Length < minTokens)
            {
                problem = $"{map.Length} tokens, fewer than the {minTokens} the draft needs";
                return null;
            }
            // Order is free (Strata's default set is its English run followed by a second sorted run); the subset
            // index order only decides ties between equal logits. Every id must be in range and listed once.
            var seen = new bool[vocab];
            for (int i = 0; i < map.Length; i++)
            {
                if (map[i] < 0 || map[i] >= vocab)
                {
                    problem = $"entry {i} ({map[i]}) is out of range for {vocab} tokens";
                    return null;
                }
                if (seen[map[i]])
                {
                    problem = $"entry {i} ({map[i]}) is listed twice";
                    return null;
                }
                seen[map[i]] = true;
            }
            return map;
        }

        /// <summary>Rows <paramref name="map"/>[i] of a table of <paramref name="rowBytes"/>-byte rows, in map
        /// order, in page-aligned native memory (the native tensor cache keys device copies by this pointer).
        /// Free with <see cref="Free"/>.</summary>
        internal static unsafe IntPtr Gather(IntPtr src, long rowBytes, int[] map, out long bytes)
        {
            bytes = rowBytes * map.Length;
            var dst = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 16384);
            var from = (byte*)src;
            for (int i = 0; i < map.Length; i++)
                Buffer.MemoryCopy(from + map[i] * rowBytes, dst + i * rowBytes, rowBytes, rowBytes);
            Wire((IntPtr)dst, bytes);
            return (IntPtr)dst;
        }

        internal static unsafe void Free(IntPtr p, long bytes)
        {
            if (p == IntPtr.Zero)
                return;
            if (!OperatingSystem.IsWindows())
                munlock(p, (nuint)bytes);
            NativeMemory.AlignedFree((void*)p);
        }

        /// <summary>
        /// Wire the rows. On unified memory the GPU reads these pages in place (a page-aligned host buffer is
        /// wrapped, not copied), and macOS compresses unwired anonymous pages under memory pressure: measured on
        /// Flash Next on the 48 GiB Mac, the English subset's draft block ran 4.1-9.2 ms against the full
        /// vocabulary's 5.0, though its head and top-k took 1.7 ms less. The expert cache's wiring
        /// (TS_HOST_MOE_IO_WIRE) fixed the same stall. Best effort: a refused mlock leaves the rows usable.
        /// </summary>
        private static bool s_wireWarned;
        private static void Wire(IntPtr p, long bytes)
        {
            if (OperatingSystem.IsWindows() || mlock(p, (nuint)bytes) == 0 || s_wireWarned)
                return;
            s_wireWarned = true;
            Console.WriteLine($"  Draft vocabulary rows could not be wired (mlock errno {Marshal.GetLastPInvokeError()}); "
                + "under memory pressure the OS may compress them and drafts can stall.");
        }

        [DllImport("libc", SetLastError = true)]
        private static extern int mlock(IntPtr addr, nuint len);

        [DllImport("libc", SetLastError = true)]
        private static extern int munlock(IntPtr addr, nuint len);

        /// <summary>Subset indices to token ids, in place, for the first <paramref name="count"/> entries.
        /// Values outside the subset (a sentinel, say) are left as they are.</summary>
        internal static void MapToTokens(int[] map, int[] ids, int count)
        {
            if (map == null || ids == null)
                return;
            for (int i = 0; i < count && i < ids.Length; i++)
            {
                int v = ids[i];
                if ((uint)v < (uint)map.Length)
                    ids[i] = map[v];
            }
        }
    }
}
