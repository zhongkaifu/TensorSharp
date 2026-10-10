// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Records a real CUDA completion event after submitted work. The caller
/// owns the context/stream until Completion finishes. Cancellation cannot prove that
/// a kernel stopped using its pointers; this fence is intentionally not cancellable.</summary>
public static class CudaExecutionFence
{
    public static Task Record(CudaContext context, IntPtr stream = default)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.MakeCurrent();
        CudaDriverApi.cuEventCreate(out var evt, 2 /* CU_EVENT_DISABLE_TIMING */).ThrowOnError();
        try { CudaDriverApi.cuEventRecord(evt, stream).ThrowOnError(); }
        catch { CudaDriverApi.cuEventDestroy(evt); throw; }
        return Task.Run(() =>
        {
            context.MakeCurrent();
            try { CudaDriverApi.cuEventSynchronize(evt).ThrowOnError(); }
            finally { CudaDriverApi.cuEventDestroy(evt).ThrowOnError(); }
        });
    }
}
