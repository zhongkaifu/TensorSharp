#nullable enable annotations
using System;
using System.Runtime.CompilerServices;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaDevice
    {
        private CudaDevice(int ordinal, string name, long totalMemoryBytes, int ccMajor, int ccMinor, int multiprocessorCount)
        {
            Ordinal = ordinal;
            Name = name;
            TotalMemoryBytes = totalMemoryBytes;
            ComputeCapabilityMajor = ccMajor;
            ComputeCapabilityMinor = ccMinor;
            MultiprocessorCount = multiprocessorCount;
        }

        public int Ordinal { get; }
        public string Name { get; }
        public long TotalMemoryBytes { get; }
        public int ComputeCapabilityMajor { get; }
        public int ComputeCapabilityMinor { get; }
        public int MultiprocessorCount { get; }

        public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";
        public string ComputeCapability => $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

        public static bool IsAvailable()
        {
            if (VisibilityMaskHidesAllDevices(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
                return false;

            try
            {
                // The resolver keeps the driver backing its P/Invoke entry points
                // alive. Loading and immediately unloading it as an availability
                // probe can leave vendor notification callbacks pointing into an
                // unloaded module, even when no CUDA device is visible.
                return ProbeDeviceCount() > 0;
            }
            catch
            {
                return false;
            }
        }

        public static int GetDeviceCount()
        {
            if (VisibilityMaskHidesAllDevices(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
                return 0;

            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count;
        }

        public static CudaDevice GetDevice(int ordinal)
        {
            if (VisibilityMaskHidesAllDevices(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
                throw new InvalidOperationException("CUDA_VISIBLE_DEVICES hides every CUDA device.");

            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();

            byte[] nameBuffer = new byte[256];
            CudaDriverApi.cuDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
            int nullIndex = Array.IndexOf(nameBuffer, (byte)0);
            string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIndex >= 0 ? nullIndex : nameBuffer.Length).Trim();

            CudaDriverApi.cuDeviceTotalMem(out UIntPtr totalMemory, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMinor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int smCount, CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

            return new CudaDevice(ordinal, name, checked((long)totalMemory.ToUInt64()), ccMajor, ccMinor, smCount);
        }

        public override string ToString()
        {
            return $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, sm_{ComputeCapabilityMajor}{ComputeCapabilityMinor}, {MultiprocessorCount} SMs)";
        }

        // A partial list such as "0,-1" still exposes its leading device. Only
        // the explicit all-hidden forms bypass driver initialization.
        internal static bool VisibilityMaskHidesAllDevices(string? mask)
            => mask != null && (string.IsNullOrWhiteSpace(mask) || mask.Trim() == "-1");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int ProbeDeviceCount()
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count;
        }
    }
}
