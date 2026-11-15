// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    /// <summary>CUDA_KERNEL_NODE_PARAMS_v2.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaKernelNodeParams
    {
        public IntPtr Function;
        public uint GridX, GridY, GridZ;
        public uint BlockX, BlockY, BlockZ;
        public uint SharedMemBytes;
        public IntPtr KernelParams;   // void**: a pointer to each argument's bytes
        public IntPtr Extra;
        public IntPtr Kernel;
        public IntPtr Context;
    }

    /// <summary>CUDA_MEMSET_NODE_PARAMS.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemsetNodeParams
    {
        public IntPtr Destination;
        public UIntPtr Pitch;
        public uint Value;
        public uint ElementSize;
        public UIntPtr Width;
        public UIntPtr Height;
    }

    /// <summary>CUDA_MEMCPY3D (memory types: 1 host, 2 device, 3 array, 4 unified).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemcpy3D
    {
        public UIntPtr SrcXInBytes, SrcY, SrcZ, SrcLod;
        public int SrcMemoryType;
        public IntPtr SrcHost, SrcDevice, SrcArray, Reserved0;
        public UIntPtr SrcPitch, SrcHeight;
        public UIntPtr DstXInBytes, DstY, DstZ, DstLod;
        public int DstMemoryType;
        public IntPtr DstHost, DstDevice, DstArray, Reserved1;
        public UIntPtr DstPitch, DstHeight;
        public UIntPtr WidthInBytes, Height, Depth;
    }

    internal static partial class CudaDriverApi
    {
        [LibraryImport(LibName)]
        public static partial int cuGraphGetNodes(IntPtr graph, [Out] IntPtr[] nodes, ref UIntPtr numNodes);

        /// <summary>CUgraphNodeType: 0 kernel, 1 memcpy, 2 memset, 3 host, 4 child graph, 5 empty, 6 event wait, 7 event record, ...</summary>
        [LibraryImport(LibName)]
        public static partial int cuGraphNodeGetType(IntPtr node, out int type);

        [LibraryImport(LibName, EntryPoint = "cuGraphKernelNodeGetParams_v2")]
        public static partial int cuGraphKernelNodeGetParams(IntPtr node, out CudaKernelNodeParams parameters);

        [LibraryImport(LibName, EntryPoint = "cuGraphExecKernelNodeSetParams_v2")]
        public static partial int cuGraphExecKernelNodeSetParams(IntPtr graphExec, IntPtr node, in CudaKernelNodeParams parameters);

        /// <summary>A library kernel's parameter <paramref name="index"/> (a node recorded from a runtime launch names its
        /// kernel this way, with no function).</summary>
        [LibraryImport(LibName)]
        public static partial int cuKernelGetParamInfo(IntPtr kernel, UIntPtr index, out UIntPtr offset, out UIntPtr size);

        [LibraryImport(LibName)]
        public static partial int cuGraphMemsetNodeGetParams(IntPtr node, out CudaMemsetNodeParams parameters);

        [LibraryImport(LibName)]
        public static partial int cuGraphExecMemsetNodeSetParams(IntPtr graphExec, IntPtr node, in CudaMemsetNodeParams parameters, IntPtr context);

        [LibraryImport(LibName)]
        public static partial int cuGraphMemcpyNodeGetParams(IntPtr node, out CudaMemcpy3D parameters);

        [LibraryImport(LibName)]
        public static partial int cuGraphExecMemcpyNodeSetParams(IntPtr graphExec, IntPtr node, in CudaMemcpy3D parameters, IntPtr context);
    }
}
