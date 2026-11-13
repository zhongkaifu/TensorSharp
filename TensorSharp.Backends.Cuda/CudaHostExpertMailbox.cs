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
    /// <summary>The doorbell's shared memory: pinned, device-mapped, one block.</summary>
    internal sealed unsafe class CudaHostExpertMailbox : IDisposable
    {
        public readonly int MaxRows, Hidden, NUsed;
        public IntPtr XDev, IdsDev, WDev, HeaderDev, SeqDev, FlagDev, CountDev, OutDev;
        public float* X, W, Out;
        public int* Ids, Header, Count;
        public uint* Seq, Flag;
        private IntPtr _host;

        public CudaHostExpertMailbox(int maxRows, int hidden, int nUsed)
        {
            MaxRows = maxRows;
            Hidden = hidden;
            NUsed = nUsed;
            // Each field on its own 128-byte line: the GPU writes the counter, the host the flag.
            static long Line(long b) => (b + 127) & ~127L;
            long oX = 0, oOut = oX + Line((long)maxRows * hidden * 4), oIds = oOut + Line((long)maxRows * hidden * 4);
            long oW = oIds + Line((long)maxRows * nUsed * 4), oHdr = oW + Line((long)maxRows * nUsed * 4);
            long oSeq = oHdr + 128, oFlag = oSeq + 128, oCount = oFlag + 128, total = oCount + 128;
            CudaDriverApi.cuMemHostAlloc(out _host, new UIntPtr((ulong)total),
                CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP).ThrowOnError();
            new Span<byte>((void*)_host, (int)total).Clear();
            CudaDriverApi.cuMemHostGetDevicePointer(out IntPtr dev, _host, 0).ThrowOnError();
            byte* h = (byte*)_host;
            X = (float*)(h + oX); Out = (float*)(h + oOut); Ids = (int*)(h + oIds); W = (float*)(h + oW);
            Header = (int*)(h + oHdr); Seq = (uint*)(h + oSeq); Flag = (uint*)(h + oFlag); Count = (int*)(h + oCount);
            XDev = dev + (nint)oX; OutDev = dev + (nint)oOut; IdsDev = dev + (nint)oIds; WDev = dev + (nint)oW;
            HeaderDev = dev + (nint)oHdr; SeqDev = dev + (nint)oSeq; FlagDev = dev + (nint)oFlag; CountDev = dev + (nint)oCount;
        }

        public void Dispose()
        {
            if (_host != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_host);
            _host = IntPtr.Zero;
        }
    }
}
