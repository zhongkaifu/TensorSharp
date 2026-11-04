// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda
{
    /// <summary>Storage and arithmetic of an original NInfer target projection.
    /// This is independent of ggml types; no GGUF conversion or requantization occurs.</summary>
    public enum CudaNinferProjectionFormat { NvFp4Planes, BFloat16, W8G32 }

    public sealed class CudaNinferProjectionDesc
    {
        public ICudaWeightSource Source { get; }
        public long SourceOffset { get; }
        public long Bytes { get; }
        public int Rows { get; }
        public int Columns { get; }
        public CudaNinferProjectionFormat Format { get; }
        public float InputScaleDivisor { get; }
        public float WeightScaleDivisor { get; }
        public bool AllowA4 { get; }
        public CudaNinferProjectionDesc(ICudaWeightSource source, long sourceOffset, long bytes,
            int rows, int columns, CudaNinferProjectionFormat format,
            float inputScaleDivisor = 1, float weightScaleDivisor = 1, bool allowA4 = true)
        {
            ArgumentNullException.ThrowIfNull(source);
            if (sourceOffset < 0 || bytes <= 0 || sourceOffset > long.MaxValue - bytes || rows <= 0 || columns <= 0)
                throw new ArgumentOutOfRangeException(nameof(bytes));
            if (!float.IsFinite(inputScaleDivisor) || inputScaleDivisor <= 0
                || !float.IsFinite(weightScaleDivisor) || weightScaleDivisor <= 0)
                throw new ArgumentException("NInfer calibration divisors must be positive and finite.");
            if (format == CudaNinferProjectionFormat.NvFp4Planes
                && (!float.IsFinite(1.0f / weightScaleDivisor) || 1.0f / weightScaleDivisor <= 0
                    || !float.IsFinite(1.0f / (inputScaleDivisor * weightScaleDivisor))
                    || 1.0f / (inputScaleDivisor * weightScaleDivisor) <= 0))
                throw new ArgumentException("NInfer calibration cannot be represented by finite positive F32 projection factors.");
            long count = checked((long)rows * columns);
            long expected = format switch
            {
                CudaNinferProjectionFormat.NvFp4Planes when columns % 64 == 0 && rows % 128 == 0 => count / 2 + count / 16,
                CudaNinferProjectionFormat.BFloat16 => checked(count * 2),
                CudaNinferProjectionFormat.W8G32 when columns % 128 == 0 => count + count / 32 * 2,
                _ => throw new NotSupportedException("Unsupported NInfer matrix storage geometry.")
            };
            if (bytes != expected) throw new ArgumentException("NInfer matrix byte count differs from its storage geometry.");
            Source = source; SourceOffset = sourceOffset; Bytes = bytes; Rows = rows; Columns = columns;
            Format = format; InputScaleDivisor = inputScaleDivisor; WeightScaleDivisor = weightScaleDivisor; AllowA4 = allowA4;
        }
    }

    public sealed class CudaNinferLayerDesc
    {
        public CudaNinferProjectionDesc Input, AlphaBeta, Output, GateUp, Down;
    }

    /// <summary>The NInfer numerical recipe, consumed by the common Qwen CUDA model/runtime.
    /// Sources are borrowed only during the synchronous engine constructor.</summary>
    public sealed class CudaNinferModelDesc
    {
        public CudaNinferProjectionDesc Embedding;
        public CudaW8HeadWeightDesc Output;
        public CudaNinferLayerDesc[] Layers;
        public string ModulePath;
        public string Provenance;
    }
}
