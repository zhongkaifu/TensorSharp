// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

/// <summary>Destination-row mapping applied to an original NInfer NVFP4 matrix.
/// It changes byte placement only; it never changes codes, scales or calibration.</summary>
public enum CudaNinferOriginalRowMapping
{
    Identity = 0,
    GateUpRows8 = 1,
    AttentionQueryGateKeyValue = 2
}

/// <summary>Optional access to the original NVFP4 object behind a virtual TS-plane source.
/// The object is adjacent-K E2M1 codes, m128x4 E4M3 scales, then the original FP32
/// weight-divisor trailer. The separate activation divisor is not part of this object.
/// Sources remain borrowed until the engine's synchronous load has finished.</summary>
public interface ICudaNinferOriginalProjectionSource : ICudaWeightSource
{
    long OriginalBytes { get; }
    int Rows { get; }
    int Columns { get; }
    CudaNinferOriginalRowMapping RowMapping { get; }

    /// <summary>Reads exactly a tensor-relative original range into caller-owned storage,
    /// or throws. Implementations verify the original reader/descriptor ownership and
    /// liveness, and use individual positional reads no larger than128MiB. A zero-length
    /// read still checks ownership and liveness. No conversion or quantization is applied.
    /// The caller supplies sufficient destination capacity for the declared byte count.</summary>
    void ReadOriginal(long offset, IntPtr dst, long bytes);
}
