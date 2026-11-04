// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Standalone N64 AOT experiment; no serving hooks. Python, Torch and CuTe
/// are compilation dependencies only. Inputs are actual BF16 plus original
/// adjacent-K NVFP4 codes and compact M128x4 E4M3 scales. Output stays F32.
/// Caller owns complete device allocations, alpha validation and stream lifetime.</summary>
internal sealed unsafe class CudaNinferFlashInferA16N64Module : IDisposable
{
    internal const int OutputRows = 34816, Columns = 5120, TileN = 64;
    internal const int DynamicSmemBytesT1 = 13376, DynamicSmemBytesT8 = 17184;
    internal const long CodeBytes = (long)OutputRows * Columns / 2, ScaleBytes = (long)OutputRows * Columns / 16;
    internal const string Revision = "61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0";
    internal const string NumericalPolicy = "flashinfer61bc8-n64-raw-nv4-bf16-mma-f32-alpha-post-s1-w4-k128-st2-v1";
    internal const string OriginalSourceManifestSha256 = "1eaa7ccd7f7fb35ff917951c1910e8ebd6d42b423221768f2d00beed8149a68c";
    private readonly CudaModule?[] _modules = new CudaModule?[2];
    private readonly IntPtr[] _kernels = new IntPtr[2];
    private readonly IntPtr _context;
    private readonly uint _threads;
    private bool _disposed;
    internal string ManifestSha256 { get; }
    internal string ArithmeticPolicy { get; }
    internal string ImageSha256ForRows(int rows) => _modules[Plan(rows).Index]!.ImageSha256;

    internal readonly record struct Specialization(int Rows, string Cubin, string CubinSha256, string Kernel);
    internal readonly record struct ProjectionPlan(int Index, long InputBytes, long OutputBytes, int SharedBytes);
    internal readonly record struct Tactic(int Warps, int Stages, string Policy);

    internal static Tactic ValidateTactic(string? policy, int warps, int stages) => (policy, warps, stages) switch
    {
        (NumericalPolicy, 4, 2) => new(4, 2, NumericalPolicy),
        _ => throw new ArgumentException("Unknown or inconsistent explicit FlashInfer AOT tactic.")
    };

    internal static ProjectionPlan Plan(int rows) => rows switch
    {
        1 => new(0, 2L * Columns, 4L * OutputRows, DynamicSmemBytesT1),
        8 => new(1, 8L * 2 * Columns, 8L * 4 * OutputRows, DynamicSmemBytesT8),
        _ => throw new ArgumentOutOfRangeException(nameof(rows), "This standalone prototype contains only static T1 and T8; no padding, chunking or fallback.")
    };

    internal static Specialization[] ValidateManifest(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        var tactic = ValidateTactic(root.GetProperty("arithmeticPolicy").GetString(), root.GetProperty("warps").GetInt32(), root.GetProperty("stages").GetInt32());
        if (root.GetProperty("schema").GetString() != "tensorsharp-ninfer-flashinfer-a16-n64-aot/1"
            || root.GetProperty("status").GetString() != "complete"
            || root.GetProperty("arch").GetString() != "sm_120a"
            || root.GetProperty("outputDtype").GetString() != "Float32"
            || root.GetProperty("splits").GetInt32() != 1 || root.GetProperty("tileK").GetInt32() != 128
            || root.GetProperty("enablePdl").GetBoolean())
            throw new ArgumentException("Unsupported FlashInfer AOT compilation policy.");
        var verification = root.GetProperty("sourceVerification");
        if (verification.GetProperty("revision").GetString() != Revision
            || verification.GetProperty("manifestSha256").GetString() != OriginalSourceManifestSha256
            || verification.GetProperty("originalFilesVerified").GetInt32() != 4
            || verification.GetProperty("licenseFilesVerified").GetInt32() != 1
            || verification.GetProperty("modifiedUpstreamFiles").GetArrayLength() != 0)
            throw new ArgumentException("Unrecognized original source provenance.");
        var values = root.GetProperty("specializations");
        if (values.GetArrayLength() != 2) throw new ArgumentException("Both static AOT specializations are required.");
        var result = new Specialization[2];
        var seen = new bool[2];
        foreach (var value in values.EnumerateArray())
        {
            int rows = value.GetProperty("rows").GetInt32();
            var plan = Plan(rows);
            if (seen[plan.Index]) throw new ArgumentException("Duplicate static AOT specialization.");
            seen[plan.Index] = true;
            if (value.GetProperty("outputRows").GetInt32() != OutputRows || value.GetProperty("columns").GetInt32() != Columns
                || value.GetProperty("dynamicSmemBytes").GetInt32() != plan.SharedBytes
                || value.GetProperty("inputBytes").GetInt64() != plan.InputBytes
                || value.GetProperty("outputBytes").GetInt64() != plan.OutputBytes
                || value.GetProperty("codeBytes").GetInt64() != CodeBytes || value.GetProperty("scaleBytes").GetInt64() != ScaleBytes
                || value.GetProperty("alphaBytes").GetInt64() != 4)
                throw new ArgumentException("AOT specialization geometry differs from the pinned static kernel.");
            RequireVector(value.GetProperty("grid"), new[] { OutputRows / TileN, 1, 1 });
            RequireVector(value.GetProperty("block"), new[] { tactic.Warps * 32, 1, 1 });
            string[] order = { "inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32" };
            if (!value.GetProperty("argumentOrder").EnumerateArray().Select(x => x.GetString()).SequenceEqual(order))
                throw new ArgumentException("AOT pointer argument order changed.");
            var parameters = value.GetProperty("parameters");
            if (parameters.GetArrayLength() != 5) throw new ArgumentException("AOT device parameter count changed.");
            int index = 0;
            foreach (var parameter in parameters.EnumerateArray())
            {
                if (parameter.GetProperty("index").GetInt32() != index
                    || parameter.GetProperty("offset").GetInt32() != 8 * index || parameter.GetProperty("size").GetInt32() != 8)
                    throw new ArgumentException("AOT device parameter layout changed.");
                index++;
            }
            string image = value.GetProperty("cubin").GetString() ?? "";
            string sha = value.GetProperty("cubinSha256").GetString() ?? "";
            string kernel = value.GetProperty("kernel").GetString() ?? "";
            if (image.Length == 0 || Path.GetFileName(image) != image || image.Contains('/') || image.Contains('\\')
                || !image.EndsWith(".cubin", StringComparison.Ordinal) || sha.Length != 64 || !sha.All(Uri.IsHexDigit)
                || kernel.Length == 0 || kernel.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '_' or '.' or '$')))
                throw new ArgumentException("Invalid AOT image identity or symbol.");
            result[plan.Index] = new(rows, image, sha.ToLowerInvariant(), kernel);
        }
        return result;
    }

    private static void RequireVector(JsonElement actual, int[] expected)
    {
        if (!actual.EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(expected))
            throw new ArgumentException("Static AOT launch geometry changed.");
    }

    internal CudaNinferFlashInferA16N64Module(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        manifestPath = Path.GetFullPath(manifestPath);
        byte[] manifest = File.ReadAllBytes(manifestPath);
        ManifestSha256 = Convert.ToHexString(SHA256.HashData(manifest)).ToLowerInvariant();
        var specs = ValidateManifest(manifest);
        using (var metadata = JsonDocument.Parse(manifest))
        {
            var root = metadata.RootElement;
            var tactic = ValidateTactic(root.GetProperty("arithmeticPolicy").GetString(), root.GetProperty("warps").GetInt32(), root.GetProperty("stages").GetInt32());
            ArithmeticPolicy = tactic.Policy; _threads = checked((uint)tactic.Warps * 32);
        }
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero) throw new InvalidOperationException("Make the owning CUDA context current before loading AOT images.");
        try
        {
            for (int i = 0; i < specs.Length; i++)
            {
                string imagePath = Path.Combine(Path.GetDirectoryName(manifestPath)!, specs[i].Cubin);
                byte[] image = File.ReadAllBytes(imagePath);
                if (image.Length < 4 || image[0] != 0x7f || image[1] != 'E' || image[2] != 'L' || image[3] != 'F'
                    || Convert.ToHexString(SHA256.HashData(image)).ToLowerInvariant() != specs[i].CubinSha256)
                    throw new InvalidDataException("AOT CUBIN contents differ from the compilation manifest.");
                var module = CudaModule.LoadFromBytes(image, identify: true);
                _modules[i] = module;
                var symbols = module.EnumerateFunctions();
                if (symbols.Length != 1 || symbols[0].Name != specs[i].Kernel)
                    throw new InvalidOperationException("AOT image must contain exactly the recorded single kernel.");
                var kernel = module.GetFunction(specs[i].Kernel);
                CudaDriverApi.cuFuncLoad(kernel).ThrowOnError();
                for (uint p = 0; p < 5; p++)
                {
                    CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)p, out var offset, out var size).ThrowOnError();
                    if (offset.ToUInt64() != 8UL * p || size.ToUInt64() != 8)
                        throw new InvalidOperationException("The actual AOT device ABI differs from the static manifest.");
                }
                if (CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)5, out _, out _) != 1)
                    throw new InvalidOperationException("Unexpected extra AOT kernel parameter.");
                _kernels[i] = kernel;
            }
        }
        catch { Dispose(); throw; }
    }

    internal void ProjectRaw(IntPtr codes, IntPtr scales, IntPtr inputBf16, IntPtr outputF32,
        IntPtr alphaF32, int rows, IntPtr stream)
    {
        var plan = Plan(rows);
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.cuCtxGetCurrent(out var actual).ThrowOnError();
        if (actual != _context) throw new InvalidOperationException("The AOT module's owning CUDA context is not current.");
        CheckPointer(codes, CodeBytes, 16); CheckPointer(scales, ScaleBytes, 16);
        CheckPointer(inputBf16, plan.InputBytes, 16); CheckPointer(outputF32, plan.OutputBytes, 4); CheckPointer(alphaF32, 4, 4);
        Separate(outputF32, plan.OutputBytes, codes, CodeBytes); Separate(outputF32, plan.OutputBytes, scales, ScaleBytes);
        Separate(outputF32, plan.OutputBytes, inputBf16, plan.InputBytes); Separate(outputF32, plan.OutputBytes, alphaF32, 4);
        void** arguments = stackalloc void*[] { &inputBf16, &codes, &scales, &alphaF32, &outputF32 };
        CudaDriverApi.cuLaunchKernel(_kernels[plan.Index], OutputRows / TileN, 1, 1, _threads, 1, 1,
            checked((uint)plan.SharedBytes), stream, (IntPtr)arguments, IntPtr.Zero).ThrowOnError();
    }

    private static void CheckPointer(IntPtr value, long bytes, int alignment)
    {
        if (value.ToInt64() <= 0 || value.ToInt64() % alignment != 0 || value.ToInt64() > long.MaxValue - bytes)
            throw new ArgumentException("Invalid or overflowing AOT device span/alignment.");
    }
    private static void Separate(IntPtr a, long aBytes, IntPtr b, long bBytes)
    {
        if (a.ToInt64() < b.ToInt64() + bBytes && b.ToInt64() < a.ToInt64() + aBytes)
            throw new ArgumentException("AOT projection output cannot alias an input.");
    }
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var module in _modules) module?.Dispose();
    }
}
