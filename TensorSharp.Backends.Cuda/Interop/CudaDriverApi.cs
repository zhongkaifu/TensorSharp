using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaDriverApi
    {
        private const string LibName = "cuda";

        [DllImport(LibName)]
        public static extern int cuInit(uint flags);

        [DllImport(LibName)]
        public static extern int cuDeviceGet(out int device, int ordinal);

        [DllImport(LibName)]
        public static extern int cuDeviceGetCount(out int count);

        [DllImport(LibName)]
        public static extern int cuDeviceGetName(byte[] name, int len, int device);

        [DllImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
        public static extern int cuDeviceTotalMem(out UIntPtr bytes, int device);

        [DllImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
        public static extern int cuMemGetInfo(out UIntPtr free, out UIntPtr total);

        [DllImport(LibName)]
        public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);

        [DllImport(LibName, EntryPoint = "cuCtxCreate_v2")]
        public static extern int cuCtxCreate(out IntPtr ctx, uint flags, int device);

        [DllImport(LibName, EntryPoint = "cuCtxDestroy_v2")]
        public static extern int cuCtxDestroy(IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuCtxSetCurrent(IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuCtxGetCurrent(out IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRelease(int device);

        [DllImport(LibName, EntryPoint = "cuMemAlloc_v2")]
        public static extern int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);

        [DllImport(LibName, EntryPoint = "cuMemFree_v2")]
        public static extern int cuMemFree(IntPtr devicePtr);

        [DllImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
        public static extern int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
        public static extern int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
        public static extern int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static extern int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemsetD8_v2")]
        public static extern int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);

        [DllImport(LibName)]
        public static extern int cuModuleLoadData(out IntPtr module, IntPtr image);

        [DllImport(LibName)]
        public static extern int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

        [DllImport(LibName)]
        public static extern int cuModuleUnload(IntPtr module);

        [DllImport(LibName)]
        public static extern int cuLaunchKernel(
            IntPtr function,
            uint gridDimX,
            uint gridDimY,
            uint gridDimZ,
            uint blockDimX,
            uint blockDimY,
            uint blockDimZ,
            uint sharedMemBytes,
            IntPtr stream,
            IntPtr kernelParams,
            IntPtr extra);

        [DllImport(LibName)]
        public static extern int cuStreamCreate(out IntPtr stream, uint flags);

        [DllImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
        public static extern int cuStreamDestroy(IntPtr stream);

        [DllImport(LibName)]
        public static extern int cuStreamSynchronize(IntPtr stream);

        [DllImport(LibName)]
        public static extern int cuGetErrorString(int error, out IntPtr str);

        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
        public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    }
}
