// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Supplies DeepSeek V4.1's Engram rows to the direct-CUDA engine.
    ///
    /// <para>An Engram table is one row per n-gram hash bucket: hundreds of
    /// millions of rows, tens to hundreds of GiB, of which a token touches
    /// <see cref="HashColumns"/>. It is far too large to live in VRAM beside the
    /// model, so it stays a host mapping and only the selected rows cross the
    /// bus. Which rows those are depends on the embedded GGUF metadata and on
    /// the sequence's own token history, neither of which is the engine's
    /// business — so the executor that owns the GGUF does the hashing and the
    /// dequantization, and the engine only uploads what comes back.</para>
    /// </summary>
    public unsafe interface IDsv41EngramSource
    {
        /// <summary>Rows read per token per table: one per (lookback, head).</summary>
        int HashColumns { get; }

        /// <summary>Values per row.</summary>
        int HeadDim { get; }

        /// <summary>
        /// Extends one sequence's hash history by this ubatch, once, before any layer asks
        /// for rows. The history is indexed by absolute position, so a prompt split across
        /// ubatches hashes exactly as it would in one shot, and a rewind only has to shorten it.
        /// </summary>
        void BeginEngramUbatch(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos);

        /// <summary>
        /// The batched-decode form: row i is <paramref name="tokens"/>[i] at
        /// <paramref name="positions"/>[i] of the sequence whose history is
        /// <paramref name="histories"/>[i].
        /// </summary>
        void BeginEngramRows(Dsv41EngramHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions);

        /// <summary>
        /// Writes this ubatch's rows for one table, laid out
        /// [token][hash column][head dim] — <paramref name="count"/> tokens of
        /// <see cref="HashColumns"/> * <see cref="HeadDim"/> floats each.
        /// </summary>
        void GatherEngramRows(int engramIndex, int count, float* dst);

        /// <summary>
        /// Writes this ubatch's selected row indices into one table, laid out [token][hash column]:
        /// <paramref name="count"/> tokens of <see cref="HashColumns"/> ints each. The engine gathers
        /// the rows of a device-resident table itself from these.
        /// </summary>
        void CopyEngramRowIndices(int engramIndex, int count, int* dst);

        /// <summary>
        /// A forward of <paramref name="tokens"/> at <paramref name="startPos"/> of the sequence whose history is
        /// <paramref name="history"/> is about to run in ubatches of <paramref name="ubatch"/>: start reading the rows
        /// it will gather from this node's host-mapped tables, in the order its layers will ask for them. Advisory:
        /// it hashes a copy (the history is not touched) and only reads ahead; the gathers still read every row.
        /// </summary>
        void PrefetchEngramRows(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos, int ubatch);

        /// <summary>
        /// Stops the read-ahead <see cref="PrefetchEngramRows"/> started (a reset, or a forward that failed): its rows will
        /// not be gathered. Returns at once; no result depends on it.
        /// </summary>
        void CancelEngramPrefetch();

        /// <summary>
        /// Bytes <see cref="ExportEngramRows"/> writes for <paramref name="count"/> tokens of table
        /// <paramref name="engramIndex"/>: their row ids (int32, [token][hash column]) and then the raw quantized
        /// rows in the same order, padded to a whole number of 32-bit words.
        /// </summary>
        long EngramTransferBytes(int engramIndex, int count);

        /// <summary>
        /// On the node that owns table <paramref name="engramIndex"/> for another node's layer: the row ids and raw
        /// rows of the <paramref name="count"/> tokens the last <see cref="BeginEngramUbatch"/> hashed (a whole
        /// forward), laid out as <see cref="EngramTransferBytes"/> says.
        /// </summary>
        void ExportEngramRows(int engramIndex, int count, byte* dst);

        /// <summary>
        /// On the node whose layer reads table <paramref name="engramIndex"/> while another node owns it: the rows
        /// <see cref="ExportEngramRows"/> wrote there for the forward of <paramref name="count"/> tokens at
        /// <paramref name="startPos"/>. Each later <see cref="GatherEngramRows"/> of that table in the forward
        /// dequantizes from them (the same host function as a local gather, so the same floats), after checking their
        /// row ids against this node's own hashes.
        /// </summary>
        void ImportEngramRows(int engramIndex, int startPos, int count, byte* src);

        /// <summary>Stopwatch ticks the host has spent in <see cref="GatherEngramRows"/> and
        /// <see cref="ExportEngramRows"/> so far (TS_DSV4_PERF reports the share of each forward).</summary>
        long EngramReadTicks { get; }
    }

    /// <summary>One sequence's Engram hash history: its (mapped) token ids by absolute
    /// position. The engine keeps one per sequence slot.</summary>
    public sealed class Dsv41EngramHistory
    {
        public int[] Tokens;
        public int Length;
    }
}
