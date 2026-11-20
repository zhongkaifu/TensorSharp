// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    /// <summary>
    /// The few NVML calls the CUDA allocator reads dedicated device memory with (issue #383): NVML ships with the driver
    /// (nvml.dll in System32 on Windows, libnvidia-ml.so.1 on Linux) and is loaded only where present, through
    /// <see cref="CudaLibraryResolver"/>; a missing library or entry point throws on the first call, which the caller takes
    /// as "no NVML".
    /// </summary>
    internal static partial class NvmlApi
    {
        internal const string LibName = "nvidia-ml";

        internal const int Success = 0;

        [StructLayout(LayoutKind.Sequential)]
        internal struct Memory
        {
            public ulong Total;
            public ulong Free;
            public ulong Used;
        }

        [LibraryImport(LibName, EntryPoint = "nvmlInit_v2")]
        internal static partial int nvmlInit();

        [LibraryImport(LibName, EntryPoint = "nvmlDeviceGetHandleByUUID", StringMarshalling = StringMarshalling.Utf8)]
        internal static partial int nvmlDeviceGetHandleByUUID(string uuid, out IntPtr device);

        [LibraryImport(LibName, EntryPoint = "nvmlDeviceGetHandleByPciBusId_v2", StringMarshalling = StringMarshalling.Utf8)]
        internal static partial int nvmlDeviceGetHandleByPciBusId(string pciBusId, out IntPtr device);

        [LibraryImport(LibName, EntryPoint = "nvmlDeviceGetMemoryInfo")]
        internal static partial int nvmlDeviceGetMemoryInfo(IntPtr device, out Memory memory);

        /// <summary>Where NVML is looked for: the driver's own copy first.</summary>
        internal static IEnumerable<string> LibraryCandidates(bool windows)
        {
            if (!windows)
            {
                yield return "libnvidia-ml.so.1";
                yield break;
            }
            string system = Environment.GetFolderPath(Environment.SpecialFolder.System);
            if (!string.IsNullOrEmpty(system)) yield return Path.Combine(system, "nvml.dll");
            yield return "nvml.dll";
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            if (!string.IsNullOrEmpty(programFiles)) yield return Path.Combine(programFiles, "NVIDIA Corporation", "NVSMI", "nvml.dll");
        }

        /// <summary>NVML's name for a GPU from the 16 bytes cuDeviceGetUuid gives (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).</summary>
        internal static string UuidString(ReadOnlySpan<byte> bytes, string prefix = "GPU-")
        {
            if (bytes.Length < 16) throw new ArgumentException("a CUDA UUID is 16 bytes", nameof(bytes));
            string hex = Convert.ToHexString(bytes[..16]).ToLowerInvariant();
            return $"{prefix}{hex[..8]}-{hex[8..12]}-{hex[12..16]}-{hex[16..20]}-{hex[20..32]}";
        }
    }
}
