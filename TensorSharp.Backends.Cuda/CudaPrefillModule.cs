// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The loader the direct-CUDA prefill kernel modules share (tensorsharp_flash_prefill,
// tensorsharp_moe_prefill, tensorsharp_ssm_prefill): locate the module's PTX (or the fatbin
// beside it), load it into the current context, and read the module's capability probe,
// ts_<module>_caps, which reports the source version the loaded image was compiled from and
// the architecture of the code the driver selected. A module whose version differs from the
// one its C# launchers were written against is stale - typically a committed compute_75 PTX
// that was not refreshed after a kernel change - and is refused, so every route stays on its
// existing fallback instead of launching kernels that are missing or take other arguments.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal abstract unsafe class CudaPrefillModule : IDisposable
    {
        /// <param name="module">The loaded image; this instance owns it.</param>
        /// <param name="capsKernel">The module's capability probe (ts_&lt;module&gt;_caps).</param>
        /// <param name="expectedVersion">TS_&lt;MODULE&gt;_VERSION as the C# launchers know it.</param>
        protected CudaPrefillModule(CudaModule module, string capsKernel, int expectedVersion)
        {
            Module = module ?? throw new ArgumentNullException(nameof(module));
            CapsKernel = capsKernel;
            ExpectedVersion = expectedVersion;
            (Version, ImageArch) = ReadCaps(module, capsKernel);
            CompiledArch = Version > 0 ? module.CompiledArchOf(capsKernel) : 0;
            CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
            DeviceCcMajor = ccMajor;
        }

        protected CudaModule Module { get; }

        /// <summary>The module's capability probe kernel.</summary>
        public string CapsKernel { get; }

        /// <summary>The source version the loaded image was compiled from (its probe's out[0]);
        /// 0 when the image has no probe.</summary>
        public int Version { get; }

        /// <summary>The version the launchers in this assembly were written against.</summary>
        public int ExpectedVersion { get; }

        /// <summary>The <c>__CUDA_ARCH__</c> / 10 the probe ran with: the SASS the driver chose from
        /// a fatbin, or the target a PTX was JIT-compiled from (75 for the committed baseline).</summary>
        public int ImageArch { get; }

        /// <summary><see cref="CudaModule.CompiledArchOf"/> for the probe: the driver's record of the
        /// same thing, which agrees with <see cref="ImageArch"/> on every image the build produces.</summary>
        public int CompiledArch { get; }

        /// <summary>Compute capability major of the device whose context loaded the module.</summary>
        public int DeviceCcMajor { get; }

        /// <summary>Whether the loaded image matches the launchers (see <see cref="ExpectedVersion"/>).</summary>
        public bool IsCurrent => Version == ExpectedVersion;

        /// <summary>Whether kernels whose bodies are guarded by <c>__CUDA_ARCH__ &gt;= 800</c> carry
        /// those bodies in this image on this device (<see cref="CudaKernels.ArchGuardedKernelsUsable"/>),
        /// by the lower of the two arch readings.</summary>
        public bool ArchGuardedKernelsUsable =>
            CudaKernels.ArchGuardedKernelsUsable(Math.Min(CompiledArch, ImageArch), DeviceCcMajor);

        /// <summary>Why this module is refused, or null when it is current.</summary>
        public string StaleReason => IsCurrent
            ? null
            : $"{CapsKernel} reports module version {Version}, but this build's launchers expect {ExpectedVersion}: " +
              "the loaded image is stale (refresh the committed PTX with nvcc 13.0, see DEVELOPMENT.md, or rebuild with nvcc).";

        /// <summary>
        /// Run <paramref name="capsKernel"/> once and read back (version, arch / 10). (0, 0) when the
        /// image has no such kernel (a module built before the probe existed). Requires the
        /// module's context to be current; synchronous.
        /// </summary>
        internal static (int Version, int Arch) ReadCaps(CudaModule module, string capsKernel)
        {
            IntPtr caps;
            try { caps = module.GetFunction(capsKernel); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { return (0, 0); } // CUDA_ERROR_NOT_FOUND
            CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(2 * sizeof(int))).ThrowOnError();
            try
            {
                int* host = stackalloc int[2] { 0, 0 };
                CudaDriverApi.cuMemcpyHtoD(probe, (IntPtr)host, new UIntPtr(2 * sizeof(int))).ThrowOnError();
                IntPtr p = probe;
                void** args = stackalloc void*[] { &p };
                CudaDriverApi.cuLaunchKernel(caps, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(2 * sizeof(int))).ThrowOnError();
                return (host[0], host[1] / 10);
            }
            finally
            {
                CudaDriverApi.cuMemFree(probe);
            }
        }

        /// <summary>Load <paramref name="ptxFileName"/> from where <see cref="CudaKernels.LocatePtxPath(string)"/>
        /// finds it (the fatbin beside it is preferred, see <see cref="CudaModule.LoadFromFile"/>).</summary>
        protected static CudaModule LoadModule(string ptxFileName)
        {
            string path = CudaKernels.LocatePtxPath(ptxFileName);
            if (path == null)
            {
                throw new InvalidOperationException(
                    ptxFileName + " could not be located next to the application or under a 'cuda_kernels' folder. " +
                    "Build TensorSharp.Backends.Cuda (the committed PTX baseline is copied there; nvcc 13+ compiles it).");
            }
            return CudaModule.LoadFromFile(path);
        }

        /// <summary>Wrap <paramref name="module"/> with <paramref name="create"/>, unloading it when that throws.</summary>
        protected static T Own<T>(CudaModule module, Func<CudaModule, T> create) where T : CudaPrefillModule
        {
            try
            {
                return create(module);
            }
            catch
            {
                module.Dispose();
                throw;
            }
        }

        /// <summary>
        /// <paramref name="create"/> over the located module when it loads and is current; otherwise
        /// false with the reason (no PTX, a load failure, a stale image). Requires a current context.
        /// </summary>
        protected static bool TryCreate<T>(string ptxFileName, Func<CudaModule, T> create, out T kernels, out string reason)
            where T : CudaPrefillModule
        {
            kernels = null;
            try
            {
                T loaded = Own(LoadModule(ptxFileName), create);
                if (!loaded.IsCurrent)
                {
                    reason = loaded.StaleReason;
                    loaded.Dispose();
                    return false;
                }
                kernels = loaded;
                reason = null;
                return true;
            }
            catch (Exception ex) when (ex is CudaException or InvalidOperationException or System.IO.IOException)
            {
                reason = ex.Message;
                return false;
            }
        }

        public void Dispose() => Module.Dispose();
    }
}
