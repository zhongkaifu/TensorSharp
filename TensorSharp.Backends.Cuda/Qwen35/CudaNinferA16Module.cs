// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Separate, explicit NInfer A16 arithmetic candidate. Constructing the ordinary
/// NInfer adapter never loads this module. It does not yet qualify a serving optimization.</summary>
internal sealed unsafe class CudaNinferA16Module : IDisposable
{
    internal const string EnvironmentVariable = "TS_Q35_NINFER_A16_PTX";
    private readonly CudaModule _module;
    private readonly IntPtr[,] _functions = new IntPtr[3, 3];
    private bool _disposed;
    internal string ImageSha256 => _module.ImageSha256;

    internal CudaNinferA16Module(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _module = CudaModule.LoadFromFile(Path.GetFullPath(path), identify: true);
        try
        {
            int[] widths = { 5120, 6144, 17408 };
            for (int width = 0; width < widths.Length; width++)
                for (int rows = 1; rows <= 3; rows++)
                {
                    IntPtr function = _module.GetFunction($"ts_ni_project_a16_candidate_k{widths[width]}_t{rows}");
                    CudaDriverApi.cuFuncLoad(function).ThrowOnError();
                    _functions[width, rows - 1] = function;
                }
        }
        catch { _module.Dispose(); throw; }
    }

    internal static CudaNinferA16Module? CreateExplicit(string? path)
    {
        if (path == null) return null;
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("The NInfer A16 candidate path cannot be blank.", nameof(path));
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath)) throw new FileNotFoundException("The explicit NInfer A16 candidate module does not exist.", fullPath);
        return new(fullPath);
    }

    internal readonly record struct ProjectionLayout(int WidthIndex, int InputStride, int OutputStride,
        long CodeBytes, long ScaleBytes, long InputBytes, long OutputBytes);

    internal static ProjectionLayout ValidateLayout(int outputRows, int columns, int rows,
        float inverseWeightDivisor, int inputStride = 0, int outputStride = 0)
    {
        int widthIndex = columns switch { 5120 => 0, 6144 => 1, 17408 => 2, _ => -1 };
        if (outputRows <= 0 || widthIndex < 0 || rows < 1 || rows > 3)
            throw new ArgumentOutOfRangeException(nameof(columns), "The candidate supports T=1..3 and K=5120/6144/17408.");
        if (!float.IsFinite(inverseWeightDivisor) || inverseWeightDivisor <= 0)
            throw new ArgumentOutOfRangeException(nameof(inverseWeightDivisor));
        inputStride = inputStride == 0 ? columns : inputStride;
        outputStride = outputStride == 0 ? outputRows : outputStride;
        if (inputStride < columns || outputStride < outputRows)
            throw new ArgumentOutOfRangeException(nameof(inputStride), "Projection strides cannot truncate a row.");
        return new(widthIndex, inputStride, outputStride,
            checked((long)outputRows * columns / 2), checked((long)outputRows * columns / 16),
            checked(((long)(rows - 1) * inputStride + columns) * sizeof(float)),
            checked(((long)(rows - 1) * outputStride + outputRows) * sizeof(float)));
    }

    /// <summary>Input F32 values must already be exactly BF16-representable. Output is the
    /// unrounded F32 original-schedule accumulator, for the caller's format epilogue.</summary>
    internal void Project(IntPtr codes, IntPtr scales, IntPtr input, IntPtr output,
        int outputRows, int columns, int rows, float inverseWeightDivisor, IntPtr stream,
        int inputStride = 0, int outputStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var layout = ValidateLayout(outputRows, columns, rows, inverseWeightDivisor, inputStride, outputStride);
        CheckPointer(codes, layout.CodeBytes); CheckPointer(scales, layout.ScaleBytes);
        CheckPointer(input, layout.InputBytes); CheckPointer(output, layout.OutputBytes);
        if (output == input || output == codes || output == scales)
            throw new ArgumentException("A16 output must use a separate destination.");
        inputStride = layout.InputStride; outputStride = layout.OutputStride;
        void** args = stackalloc void*[] { &codes, &scales, &input, &output, &outputRows, &columns,
            &rows, &inputStride, &outputStride, &inverseWeightDivisor };
        CudaDriverApi.cuLaunchKernel(_functions[layout.WidthIndex, rows - 1], checked((uint)(((long)outputRows + 15) / 16)),
            1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }

    private static void CheckPointer(IntPtr pointer, long bytes)
    {
        if (pointer.ToInt64() <= 0 || unchecked((ulong)pointer.ToInt64()) > (ulong)long.MaxValue - (ulong)bytes)
            throw new ArgumentException("A16 buffers require a nonzero, nonoverflowing device range.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true; _module.Dispose();
    }
}
