using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static partial class CudaDriverApi
    {
        private const string LibName = "cuda";

        [LibraryImport(LibName)]
        public static partial int cuInit(uint flags);

        /// <summary>The CUDA version the installed driver supports, 1000 * major + 10 * minor
        /// (13050 for a CUDA 13.5 driver). TensorSharp needs 13000 or newer.</summary>
        [LibraryImport(LibName)]
        public static partial int cuDriverGetVersion(out int driverVersion);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGet(out int device, int ordinal);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetCount(out int count);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetName(byte[] name, int len, int device);

        [LibraryImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
        public static partial int cuDeviceTotalMem(out UIntPtr bytes, int device);

        [LibraryImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
        public static partial int cuMemGetInfo(out UIntPtr free, out UIntPtr total);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetAttribute(out int value, int attribute, int device);

        [LibraryImport(LibName)]
        public static partial int cuCtxGetDevice(out int device);

        [LibraryImport(LibName)]
        public static partial int cuCtxSetCurrent(IntPtr ctx);

        [LibraryImport(LibName)]
        public static partial int cuCtxGetCurrent(out IntPtr ctx);

        [LibraryImport(LibName)]
        public static partial int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

        [LibraryImport(LibName)]
        public static partial int cuDevicePrimaryCtxRelease(int device);

        [LibraryImport(LibName, EntryPoint = "cuMemAlloc_v2")]
        public static partial int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);

        [LibraryImport(LibName, EntryPoint = "cuMemFree_v2")]
        public static partial int cuMemFree(IntPtr devicePtr);

        /// <summary>Page-locked (pinned) host allocation. A captured HtoD memcpy
        /// node whose source is pinned host memory re-reads the buffer on every
        /// graph launch, which is how decode-graph replays receive per-token
        /// parameters (see CudaDecodeDynParams).</summary>
        [LibraryImport(LibName, EntryPoint = "cuMemHostAlloc")]
        public static partial int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuMemFreeHost")]
        public static partial int cuMemFreeHost(IntPtr hostPtr);

        /// <summary>cuMemHostAlloc flags: PORTABLE (all contexts) and DEVICEMAP (kernels address it).</summary>
        public const uint CU_MEMHOSTALLOC_PORTABLE = 0x01, CU_MEMHOSTALLOC_DEVICEMAP = 0x02;

        [LibraryImport(LibName, EntryPoint = "cuMemHostGetDevicePointer_v2")]
        public static partial int cuMemHostGetDevicePointer(out IntPtr devicePtr, IntPtr hostPtr, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
        public static partial int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
        public static partial int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
        public static partial int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
        public static partial int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
        public static partial int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static partial int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        /// <summary>CUDA_MEMCPY2D: a pitched copy of <see cref="Height"/> rows of <see cref="WidthInBytes"/>
        /// bytes, field for field the driver's struct (natural alignment, so the enum fields are padded).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CudaMemcpy2D
        {
            public UIntPtr SrcXInBytes, SrcY;
            public int SrcMemoryType;
            public IntPtr SrcHost, SrcDevice, SrcArray;
            public UIntPtr SrcPitch;
            public UIntPtr DstXInBytes, DstY;
            public int DstMemoryType;
            public IntPtr DstHost, DstDevice, DstArray;
            public UIntPtr DstPitch;
            public UIntPtr WidthInBytes, Height;
        }

        /// <summary>CUmemorytype values for <see cref="CudaMemcpy2D"/>.</summary>
        public const int CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2;

        [LibraryImport(LibName, EntryPoint = "cuMemcpy2DAsync_v2")]
        public static partial int cuMemcpy2DAsync(in CudaMemcpy2D copy, IntPtr stream);

        /// <summary>Queues a host function on a stream (<c>void CUDA_CB fn(void* userData)</c>): it runs once the
        /// work queued before it has completed, and the work queued after it waits for it to return.</summary>
        [LibraryImport(LibName, EntryPoint = "cuLaunchHostFunc")]
        public static partial int cuLaunchHostFunc(IntPtr stream, IntPtr fn, IntPtr userData);

        [LibraryImport(LibName, EntryPoint = "cuMemsetD8_v2")]
        public static partial int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);

        [LibraryImport(LibName, EntryPoint = "cuMemsetD8Async")]
        public static partial int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemsetD32Async")]
        public static partial int cuMemsetD32Async(IntPtr dstDevice, uint value, UIntPtr count, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuModuleLoadData(out IntPtr module, IntPtr image);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        public static partial int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

        [LibraryImport(LibName)]
        public static partial int cuModuleUnload(IntPtr module);

        [LibraryImport(LibName)]
        public static partial int cuFuncSetAttribute(IntPtr function, int attribute, int value);

        // CUfunction_attribute: opt-in cap for dynamic shared memory per block
        // (default launches are limited to 48 KB without it).
        public const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

        [LibraryImport(LibName)]
        public static partial int cuFuncGetAttribute(out int value, int attribute, IntPtr function);

        // CUfunction_attribute: the virtual architecture (major * 10 + minor) the code the
        // driver loaded for a function was compiled from, i.e. its __CUDA_ARCH__ / 10: 86 for
        // sm_86 SASS, 75 for compute_75 PTX the driver JIT-compiled for a newer GPU.
        public const int CU_FUNC_ATTRIBUTE_PTX_VERSION = 5;

        // CUfunction_attribute: the architecture (major * 10 + minor) of the binary the driver
        // actually runs - the GPU's own sm for JIT-compiled PTX.
        public const int CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6;

        [LibraryImport(LibName)]
        public static partial int cuLaunchKernel(
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

        /// <summary>Encodes a TMA tensor map (CUtensorMap, 128 bytes, 64-byte aligned) for a tiled
        /// bulk copy. Data type 0 is UINT8; swizzle 0 none, 2 64-byte; the rest are passed as 0.</summary>
        [LibraryImport(LibName)]
        public static unsafe partial int cuTensorMapEncodeTiled(void* tensorMap, int dataType, uint rank,
            IntPtr globalAddress, ulong* globalDim, ulong* globalStrides, uint* boxDim, uint* elementStrides,
            int interleave, int swizzle, int l2Promotion, int oobFill);

        [LibraryImport(LibName)]
        public static partial int cuStreamCreate(out IntPtr stream, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
        public static partial int cuStreamDestroy(IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuStreamSynchronize(IntPtr stream);

        // ---- CUDA Graphs (stream capture) ----
        [LibraryImport(LibName, EntryPoint = "cuStreamBeginCapture_v2")]
        public static partial int cuStreamBeginCapture(IntPtr stream, int mode);

        [LibraryImport(LibName)]
        public static partial int cuStreamEndCapture(IntPtr stream, out IntPtr graph);

        [LibraryImport(LibName)]
        public static partial int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags);

        [LibraryImport(LibName)]
        public static partial int cuGraphLaunch(IntPtr graphExec, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuGraphExecDestroy(IntPtr graphExec);

        [LibraryImport(LibName)]
        public static partial int cuGraphDestroy(IntPtr graph);

        // CUstreamCaptureMode: 0 = GLOBAL, 1 = THREAD_LOCAL, 2 = RELAXED.
        public const int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
        // Relaxed capture permits API calls that are unsafe-by-default during
        // capture (notably cuMemAlloc for pool misses); the capture owner takes
        // responsibility for the referenced memory's lifetime.
        public const int CU_STREAM_CAPTURE_MODE_RELAXED = 2;

        [LibraryImport(LibName)]
        public static partial int cuGetErrorString(int error, out IntPtr str);

        // ---- CUDA Events ----
        [LibraryImport(LibName)]
        public static partial int cuEventCreate(out IntPtr phEvent, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuEventDestroy_v2")]
        public static partial int cuEventDestroy(IntPtr hEvent);

        [LibraryImport(LibName, EntryPoint = "cuEventRecord")]
        public static partial int cuEventRecord(IntPtr hEvent, IntPtr hStream);

        [LibraryImport(LibName)]
        public static partial int cuEventSynchronize(IntPtr hEvent);

        /// <summary>0 when the event's work completed, CUDA_ERROR_NOT_READY (600) while it runs.</summary>
        [LibraryImport(LibName)]
        public static partial int cuEventQuery(IntPtr hEvent);

        [LibraryImport(LibName)]
        public static partial int cuEventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

        [LibraryImport(LibName)]
        public static partial int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

        // ---- Peer-to-Peer (multi-GPU) ----
        [LibraryImport(LibName)]
        public static partial int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev);

        [LibraryImport(LibName)]
        public static partial int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags);

        [LibraryImport(LibName)]
        public static partial int cuCtxDisablePeerAccess(IntPtr peerContext);

        [LibraryImport(LibName)]
        public static partial int cuMemcpyPeerAsync(
            IntPtr dstDevice, IntPtr dstContext,
            IntPtr srcDevice, IntPtr srcContext,
            UIntPtr byteCount, IntPtr hStream);

        // CUstream flags
        public const uint CU_STREAM_DEFAULT = 0;
        public const uint CU_STREAM_NON_BLOCKING = 1;

        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
        public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    }
}
