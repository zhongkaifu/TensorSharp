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
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace InferenceEngine
{
    internal static class NativeDequant
    {
        private const string DllName = "GgmlOps";

        static NativeDequant()
        {
            NativeLibrary.SetDllImportResolver(typeof(NativeDequant).Assembly, ImportResolver);
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int TSGgml_DequantizeToF32(int ggmlType, IntPtr src, long numElements, IntPtr dst);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr TSGgml_RowSize(int ggmlType, long ne);

        public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            GCHandle hSrc = GCHandle.Alloc(src, GCHandleType.Pinned);
            GCHandle hDst = GCHandle.Alloc(dst, GCHandleType.Pinned);
            try
            {
                IntPtr pSrc = IntPtr.Add(hSrc.AddrOfPinnedObject(), srcOffset);
                IntPtr pDst = IntPtr.Add(hDst.AddrOfPinnedObject(), dstOffset * sizeof(float));
                int r = TSGgml_DequantizeToF32(ggmlType, pSrc, numElements, pDst);
                if (r != 0)
                    throw new InvalidOperationException($"Dequantization failed for type {ggmlType}, result={r}");
            }
            finally
            {
                if (hSrc.IsAllocated) hSrc.Free();
                if (hDst.IsAllocated) hDst.Free();
            }
        }

        public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            GCHandle hDst = GCHandle.Alloc(dst, GCHandleType.Pinned);
            try
            {
                IntPtr pDst = IntPtr.Add(hDst.AddrOfPinnedObject(), dstOffset * sizeof(float));
                int r = TSGgml_DequantizeToF32(ggmlType, src, numElements, pDst);
                if (r != 0)
                    throw new InvalidOperationException($"Dequantization failed for type {ggmlType}, result={r}");
            }
            finally
            {
                if (hDst.IsAllocated) hDst.Free();
            }
        }

        public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            int r = TSGgml_DequantizeToF32(ggmlType, src, numElements, dst);
            if (r != 0)
                throw new InvalidOperationException($"Dequantization failed for type {ggmlType}, result={r}");
        }

        public static long RowSize(int ggmlType, long ne)
        {
            return (long)TSGgml_RowSize(ggmlType, ne);
        }

        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, DllName, StringComparison.Ordinal))
                return IntPtr.Zero;

            foreach (string candidate in GetCandidatePaths())
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        private static string[] GetCandidatePaths()
        {
            string baseDir = AppContext.BaseDirectory;
            string nativeBuildDir = Path.Combine(
                Path.GetDirectoryName(Path.GetDirectoryName(baseDir.TrimEnd(Path.DirectorySeparatorChar))),
                "TensorSharp.GGML.Native", "build");

            return new[]
            {
                Path.Combine(baseDir, "libGgmlOps.dylib"),
                Path.Combine(nativeBuildDir, "libGgmlOps.dylib"),
                Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(baseDir.TrimEnd(Path.DirectorySeparatorChar))),
                    "..", "TensorSharp.GGML.Native", "build", "libGgmlOps.dylib"),
                "/Users/ZhongkaiFu/work/TensorSharp/TensorSharp.GGML.Native/build/libGgmlOps.dylib",
                "/Users/ZhongkaiFu/work/Seq2SeqSharp/TensorSharp.GGML.Native/build/libGgmlOps.dylib",
            };
        }
    }
}
