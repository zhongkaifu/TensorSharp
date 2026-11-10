// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    /// <summary>One weight of a DSpine draft as the native graph binds it (TsgDsWeight in ggml_ops_dspine.cpp): the
    /// host bytes (also the device-cache key), ggml dims (ne0 = row length) and ggml type.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct GgmlDSpineWeight
    {
        public IntPtr Data;
        public long Ne0;
        public long Ne1;
        public long Bytes;
        public int Type;
        public int Reserved;

        public GgmlDSpineWeight(IntPtr data, int type, long ne0, long ne1, long bytes)
        {
            Data = data;
            Type = type;
            Ne0 = ne0;
            Ne1 = ne1;
            Bytes = bytes;
            Reserved = 0;
        }
    }

    /// <summary>Geometry of a DSpine draft (TsgDsParams in ggml_ops_dspine.cpp).</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct GgmlDSpineParams
    {
        public int NumLayers, Hidden, NumHeads, NumKvHeads, HeadDim, Vocab;
        public int CodeSize, Rank, TopK, RingRows, RingDtype, SlidingWindow;
        public float Eps, BridgeEps, RopeBase, RopeFreqScale, KqScale;
        public int Reserved0, Reserved1, Reserved2;
    }

    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial IntPtr TSGgml_DSpineTablesCreate(GgmlDSpineParams* p, GgmlDSpineWeight* w, int nWeights, int tableType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial void TSGgml_DSpineTablesFree(IntPtr tables);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_DSpineTablesRead(IntPtr tables, int which, int row0, int n, float* output);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial void TSGgml_DSpineResetCaches();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_DSpineDraftBlock(GgmlDSpineParams* p, GgmlDSpineWeight* w, int nWeights,
            IntPtr tables, IntPtr* ringK, IntPtr* ringV, int* blockIds, int blockLen, int* positions, int* ringSlotPos,
            float* scoresOut, int* candOut, IntPtr probe);
    }

    public partial class GgmlBasicOps
    {
        /// <summary>Slots of the DSpine weight table: globals first, then <see cref="DSpinePerLayer"/> per layer.</summary>
        public const int DSpineTokEmbd = 0, DSpineLmHead = 1, DSpineOutNorm = 2, DSpineCodebook = 3, DSpineBridge = 4,
            DSpineGateBias = 5, DSpineGlobals = 8;

        public const int DSpineAttnNorm = 0, DSpineQ = 1, DSpineK = 2, DSpineV = 3, DSpineQNorm = 4, DSpineKNorm = 5,
            DSpineO = 6, DSpineFfnNorm = 7, DSpineGate = 8, DSpineUp = 9, DSpineDown = 10, DSpineRead = 11, DSpineWrite = 12,
            DSpinePerLayer = 16;

        /// <summary>Builds the DSpine refinement tables (TG [vocab, rank], TSV [vocab, 2 rank]) on the active backend in
        /// <paramref name="tableType"/> (ggml F32, F16 or BF16). Zero on failure (see <see cref="LastNativeError"/>),
        /// including a native library without DSpine.</summary>
        public static unsafe IntPtr DSpineTablesCreate(GgmlDSpineParams p, GgmlDSpineWeight[] weights, int tableType)
        {
            try
            {
                fixed (GgmlDSpineWeight* w = weights)
                    return GgmlNative.TSGgml_DSpineTablesCreate(&p, w, weights.Length, tableType);
            }
            catch (EntryPointNotFoundException)
            {
                return IntPtr.Zero;
            }
        }

        public static void DSpineTablesFree(IntPtr tables)
        {
            if (tables != IntPtr.Zero)
                GgmlNative.TSGgml_DSpineTablesFree(tables);
        }

        /// <summary>Rows of TG (<paramref name="which"/> = 0) or TSV (1) as F32, for tests.</summary>
        public static unsafe bool DSpineTablesRead(IntPtr tables, int which, int row0, int n, float[] output)
        {
            fixed (float* o = output)
                return GgmlNative.TSGgml_DSpineTablesRead(tables, which, row0, n, o) != 0;
        }

        public static void DSpineResetCaches()
        {
            try { GgmlNative.TSGgml_DSpineResetCaches(); }
            catch (EntryPointNotFoundException) { }
        }

        /// <summary>One DSpine draft of a <paramref name="blockLen"/>-row block over the ring the DFlash inject filled.
        /// Fills <paramref name="scoresOut"/> with the anchor row (top-k floats) and then one [k(pred), k(cand)] matrix
        /// per later row, and <paramref name="candOut"/> with [rows - 1][k] candidate ids: the layout
        /// ModelBase.DFlashWalkLattice reads.</summary>
        public static unsafe bool DSpineDraftBlock(GgmlDSpineParams p, GgmlDSpineWeight[] weights, IntPtr tables,
            IntPtr[] ringK, IntPtr[] ringV, int[] blockIds, int blockLen, int[] positions, int[] ringSlotPos,
            float[] scoresOut, int[] candOut)
        {
            try
            {
                fixed (GgmlDSpineWeight* w = weights)
                fixed (IntPtr* rk = ringK)
                fixed (IntPtr* rv = ringV)
                fixed (int* ids = blockIds)
                fixed (int* pos = positions)
                fixed (int* slots = ringSlotPos)
                fixed (float* scores = scoresOut)
                fixed (int* cand = candOut)
                    return GgmlNative.TSGgml_DSpineDraftBlock(&p, w, weights.Length, tables, rk, rv, ids, blockLen, pos, slots,
                        scores, cand, IntPtr.Zero) != 0;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
        }
    }
}
