using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static partial class CudaDriverApi
    {
        // CUDA 12.8+ documented module enumeration/parameter APIs. These are
        // only resolved by the explicitly loaded W8 component, never old paths.
        [LibraryImport(LibName)]
        public static partial int cuModuleGetFunctionCount(out uint count, IntPtr module);
        [LibraryImport(LibName)]
        public static unsafe partial int cuModuleEnumerateFunctions(IntPtr* functions, uint count, IntPtr module);
        [LibraryImport(LibName)]
        public static partial int cuFuncGetName(out IntPtr name, IntPtr function);
        [LibraryImport(LibName)]
        public static partial int cuFuncLoad(IntPtr function);
        [LibraryImport(LibName)]
        public static partial int cuFuncGetParamInfo(IntPtr function, UIntPtr index, out UIntPtr offset, out UIntPtr size);
        [LibraryImport(LibName)]
        public static partial int cuOccupancyMaxActiveBlocksPerMultiprocessor(out int activeBlocks, IntPtr function, int blockSize, UIntPtr dynamicSharedBytes);

        // cuda.h: union value is 64 bytes/alignment 8, after the 4-byte enum
        // and 4-byte pad. All unspecified bytes are zero in the launch adapter.
        [StructLayout(LayoutKind.Explicit, Size = 72)]
        internal struct CudaLaunchAttribute
        {
            [FieldOffset(0)] public int Id;
            [FieldOffset(8)] public ulong ValueAlignment;
            [FieldOffset(8)] public int ProgrammaticStreamSerializationAllowed;
        }
        [StructLayout(LayoutKind.Explicit, Size = 56)]
        internal struct CudaLaunchConfig
        {
            [FieldOffset(0)] public uint GridDimX;
            [FieldOffset(4)] public uint GridDimY;
            [FieldOffset(8)] public uint GridDimZ;
            [FieldOffset(12)] public uint BlockDimX;
            [FieldOffset(16)] public uint BlockDimY;
            [FieldOffset(20)] public uint BlockDimZ;
            [FieldOffset(24)] public uint SharedMemBytes;
            [FieldOffset(32)] public IntPtr Stream;
            [FieldOffset(40)] public IntPtr Attributes;
            [FieldOffset(48)] public uint NumAttributes;
        }
        [LibraryImport(LibName)]
        public static partial int cuLaunchKernelEx(in CudaLaunchConfig config, IntPtr function, IntPtr kernelParams, IntPtr extra);
    }
}
