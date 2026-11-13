// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Programmatic dependent launch (Hopper and newer). A decode step of a large MoE model is a few thousand small
    /// dependent kernels, and each kernel boundary costs the GPU a launch gap even inside a CUDA graph. Launched with the
    /// programmatic-serialization attribute, a kernel is scheduled while the one before it in the stream finishes: its
    /// first instruction, griddepcontrol.wait (TS_PDL_ENTRY in every exported kernel of tensorsharp_kernels.cu,
    /// tensorsharp_dsv4_kernels.cu, tensorsharp_glm_kernels.cu and tensorsharp_q4e_kernels.cu), holds it until the earlier
    /// grid has completed and its writes are visible, so the kernels compute exactly what they computed before. Graph
    /// capture keeps the dependency as a programmatic edge.
    ///
    /// Used only when the module's code was compiled for sm_90 or newer (the wait is compiled out below that, and the
    /// committed compute_75 PTX has none) and the device is sm_90 or newer. TS_CUDA_PDL=0 launches every kernel plainly.
    /// </summary>
    internal static unsafe class CudaPdl
    {
        private const int CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6;

        /// <summary>Whether PDL is allowed at all (TS_CUDA_PDL, on unless "0").</summary>
        public static readonly bool Requested = Environment.GetEnvironmentVariable("TS_CUDA_PDL") != "0";

        /// <summary>Whether a module compiled for <paramref name="compiledArch"/> on the current context's device launches
        /// with PDL.</summary>
        public static bool UsableFor(int compiledArch)
        {
            if (!Requested || compiledArch < 90)
                return false;
            try
            {
                CudaDriverApi.cuCtxGetDevice(out int device);
                CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, device);
                return ccMajor >= 9;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>cuLaunchKernel with the programmatic-serialization attribute.</summary>
        public static int Launch(IntPtr fn, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint sharedBytes, IntPtr stream,
            void** args)
        {
            var attr = new CudaDriverApi.CudaLaunchAttribute
            {
                Id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                ProgrammaticStreamSerializationAllowed = 1,
            };
            var config = new CudaDriverApi.CudaLaunchConfig
            {
                GridDimX = gx, GridDimY = gy, GridDimZ = gz,
                BlockDimX = bx, BlockDimY = by, BlockDimZ = bz,
                SharedMemBytes = sharedBytes,
                Stream = stream,
                Attributes = (IntPtr)(&attr),
                NumAttributes = 1,
            };
            return CudaDriverApi.cuLaunchKernelEx(in config, fn, (IntPtr)args, IntPtr.Zero);
        }
    }
}
