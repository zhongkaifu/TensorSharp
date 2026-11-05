// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>AOT raw-layout W4A16 body projections. No implicit module discovery,
/// activation conversion, row permutation, requantization or serving hook.
/// The caller validates original finite scales/calibration before publishing
/// weights and owns complete device allocations and stream completion.</summary>
internal sealed unsafe class CudaNinferFlashInferBodyModule : IDisposable
{
    internal const int MaximumRows = 8, MaximumOutputRows = 34816, MaximumColumns = 17408, SpecializationCount = 32;
    internal const long MaximumRawWeightBytes = 100270080;
    internal const string Revision = "61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0";
    internal const string OriginalSourceManifestSha256 = "1eaa7ccd7f7fb35ff917951c1910e8ebd6d42b423221768f2d00beed8149a68c";
    internal const string N128Policy = "flashinfer61bc8-body-raw-nv4-bf16-mma-f32-alpha-post-n128-s1-w4-k128-st2-v1";
    internal const string N64Policy = "flashinfer61bc8-body-raw-nv4-bf16-mma-f32-alpha-post-n64-s1-w4-k128-st2-v1";
    private readonly CudaModule?[] _modules = new CudaModule?[SpecializationCount];
    private readonly IntPtr[] _kernels = new IntPtr[SpecializationCount];
    private readonly IntPtr _context;
    private readonly int _tileN;
    private bool _disposed;
    internal string ManifestSha256 { get; }
    internal string ImageSetSha256 { get; }
    internal string ArithmeticPolicy { get; }
    internal string Family { get; }

    internal readonly record struct ProjectionLayout(int Index, int ShapeIndex, int OutputRows, int Columns, int Rows,
        long CodeBytes, long ScaleBytes, long InputBytes, long OutputBytes);
    internal readonly record struct ImageSpec(ProjectionLayout Layout, string Cubin, string CubinSha256, string Kernel);
    internal readonly record struct ManifestContract(string Family, string Policy, int TileN, ImageSpec[] Images);

    internal static bool Supports(int outputRows, int columns, int rows) => rows is >= 1 and <= MaximumRows
        && ((outputRows, columns) is (34816, 5120) or (5120, 17408) or (5120, 6144) or (14336, 5120));

    internal static ProjectionLayout Plan(int outputRows, int columns, int rows)
    {
        if (!Supports(outputRows, columns, rows))
            throw new ArgumentOutOfRangeException(nameof(rows), "Only the four original body projection shapes and physical T1..8 are supported.");
        int shape = (outputRows, columns) switch { (34816, 5120) => 0, (5120, 17408) => 1, (5120, 6144) => 2, _ => 3 };
        return new(shape * MaximumRows + rows - 1, shape, outputRows, columns, rows,
            (long)outputRows * columns / 2, (long)outputRows * columns / 16,
            (long)rows * columns * 2, (long)rows * outputRows * 4);
    }

    internal static int SharedBytes(string family, int rows)
    {
        if (rows is < 1 or > MaximumRows) throw new ArgumentOutOfRangeException(nameof(rows));
        int tileN, liveRows;
        if (family == "n128") { tileN = 128; liveRows = 16; }
        else if (family == "n64") { tileN = 64; liveRows = Math.Min(rows + 1, 16); }
        else throw new ArgumentException("Unknown original kernel family.", nameof(family));
        return 2 * (liveRows * 68 * 4 + tileN * 20 * 4 + 1024);
    }

    internal static ManifestContract ValidateManifest(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        string family = root.GetProperty("family").GetString() ?? "";
        string policy = root.GetProperty("arithmeticPolicy").GetString() ?? "";
        int tileN = (family, policy) switch { ("n128", N128Policy) => 128, ("n64", N64Policy) => 64,
            _ => throw new ArgumentException("Unknown or mislabeled raw body arithmetic policy.") };
        if (root.GetProperty("schema").GetString() != "tensorsharp-ninfer-flashinfer-body-aot/1"
            || root.GetProperty("status").GetString() != "complete" || root.GetProperty("arch").GetString() != "sm_120a"
            || root.GetProperty("outputDtype").GetString() != "Float32" || root.GetProperty("splits").GetInt32() != 1
            || root.GetProperty("warps").GetInt32() != 4 || root.GetProperty("tileK").GetInt32() != 128
            || root.GetProperty("stages").GetInt32() != 2 || root.GetProperty("enablePdl").GetBoolean())
            throw new ArgumentException("Unsupported body AOT compilation contract.");
        var source = root.GetProperty("sourceVerification");
        if (source.GetProperty("revision").GetString() != Revision || source.GetProperty("manifestSha256").GetString() != OriginalSourceManifestSha256
            || source.GetProperty("originalFilesVerified").GetInt32() != 4 || source.GetProperty("licenseFilesVerified").GetInt32() != 1
            || source.GetProperty("modifiedUpstreamFiles").GetArrayLength() != 0)
            throw new ArgumentException("Unrecognized pinned original source provenance.");
        var specializations = root.GetProperty("specializations");
        if (specializations.GetArrayLength() != SpecializationCount) throw new ArgumentException("All32 distinct body specializations are mandatory.");
        var images = new ImageSpec[SpecializationCount]; var seen = new bool[SpecializationCount];
        foreach (var value in specializations.EnumerateArray())
        {
            var layout = Plan(value.GetProperty("outputRows").GetInt32(), value.GetProperty("columns").GetInt32(), value.GetProperty("rows").GetInt32());
            if (seen[layout.Index]) throw new ArgumentException("Duplicate body AOT specialization.");
            seen[layout.Index] = true;
            if (value.GetProperty("codeBytes").GetInt64() != layout.CodeBytes || value.GetProperty("scaleBytes").GetInt64() != layout.ScaleBytes
                || value.GetProperty("inputBytes").GetInt64() != layout.InputBytes || value.GetProperty("outputBytes").GetInt64() != layout.OutputBytes
                || value.GetProperty("alphaBytes").GetInt32() != 4 || value.GetProperty("dynamicSmemBytes").GetInt32() != SharedBytes(family, layout.Rows))
                throw new ArgumentException("Body buffer geometry or shared reservation differs from the fixed kernel.");
            if (!value.GetProperty("grid").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(new[] { layout.OutputRows / tileN, 1, 1 })
                || !value.GetProperty("block").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(new[] { 128, 1, 1 })
                || !value.GetProperty("argumentOrder").EnumerateArray().Select(x => x.GetString()).SequenceEqual(
                    new[] { "inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32" }))
                throw new ArgumentException("Body AOT launch or device argument order changed.");
            var parameters = value.GetProperty("parameters");
            if (parameters.GetArrayLength() != 5) throw new ArgumentException("Body AOT requires exactly five pointer parameters.");
            int index = 0;
            foreach (var parameter in parameters.EnumerateArray())
            {
                if (parameter.GetProperty("index").GetInt32() != index || parameter.GetProperty("offset").GetInt32() != index * 8 || parameter.GetProperty("size").GetInt32() != 8)
                    throw new ArgumentException("Body AOT device parameter layout changed.");
                index++;
            }
            string file = value.GetProperty("cubin").GetString() ?? "", sha = value.GetProperty("cubinSha256").GetString() ?? "", kernel = value.GetProperty("kernel").GetString() ?? "";
            if (file.Length == 0 || Path.GetFileName(file) != file || file.Contains('/') || file.Contains('\\') || !file.EndsWith(".cubin", StringComparison.Ordinal)
                || sha.Length != 64 || !sha.All(Uri.IsHexDigit) || kernel.Length == 0 || kernel.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '_' or '.' or '$')))
                throw new ArgumentException("Invalid body AOT image identity.");
            images[layout.Index] = new(layout, file, sha.ToLowerInvariant(), kernel);
        }
        if (seen.Any(x => !x) || images.Select(x => x.Cubin).Distinct(StringComparer.Ordinal).Count() != SpecializationCount)
            throw new ArgumentException("Incomplete or aliased body image coverage.");
        if (root.GetProperty("imageSetSha256").GetString() != ComputeImageSetSha256(policy, images))
            throw new ArgumentException("The canonical32-image body identity differs from the manifest.");
        return new(family, policy, tileN, images);
    }

    internal static string ComputeImageSetSha256(string policy, ImageSpec[] images)
    {
        var value = new StringBuilder(policy).Append('\n');
        foreach (var spec in images.OrderBy(x => x.Layout.Index))
            value.Append(spec.Layout.OutputRows).Append('x').Append(spec.Layout.Columns).Append("/t").Append(spec.Layout.Rows)
                .Append('=').Append(spec.CubinSha256.ToLowerInvariant()).Append('\n');
        return Hex(SHA256.HashData(Encoding.UTF8.GetBytes(value.ToString())));
    }

    internal CudaNinferFlashInferBodyModule(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        manifestPath = Path.GetFullPath(manifestPath);
        byte[] manifest = File.ReadAllBytes(manifestPath);
        ManifestSha256 = Hex(SHA256.HashData(manifest));
        var contract = ValidateManifest(manifest);
        Family = contract.Family; ArithmeticPolicy = contract.Policy; _tileN = contract.TileN;
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero) throw new InvalidOperationException("Make the owning CUDA context current before loading the body AOT images.");
        var canonical = new StringBuilder(ArithmeticPolicy).Append('\n');
        try
        {
            for (int i = 0; i < SpecializationCount; i++)
            {
                var spec = contract.Images[i]; byte[] image = File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(manifestPath)!, spec.Cubin));
                if (image.Length < 4 || image[0] != 0x7f || image[1] != 'E' || image[2] != 'L' || image[3] != 'F' || Hex(SHA256.HashData(image)) != spec.CubinSha256)
                    throw new InvalidDataException("Body CUBIN bytes differ from their pinned compilation manifest.");
                var module = CudaModule.LoadFromBytes(image, identify: true); _modules[i] = module;
                var symbols = module.EnumerateFunctions();
                if (symbols.Length != 1 || symbols[0].Name != spec.Kernel) throw new InvalidOperationException("Each body image requires exactly its single recorded kernel.");
                var kernel = module.GetFunction(spec.Kernel); CudaDriverApi.cuFuncLoad(kernel).ThrowOnError();
                for (uint parameter = 0; parameter < 5; parameter++)
                {
                    CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)parameter, out var offset, out var size).ThrowOnError();
                    if (offset.ToUInt64() != parameter * 8UL || size.ToUInt64() != 8) throw new InvalidOperationException("Actual body device parameter ABI changed.");
                }
                if (CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)5, out _, out _) != 1) throw new InvalidOperationException("Unexpected extra body kernel parameter.");
                _kernels[i] = kernel;
                canonical.Append(spec.Layout.OutputRows).Append('x').Append(spec.Layout.Columns).Append("/t").Append(spec.Layout.Rows)
                    .Append('=').Append(module.ImageSha256.ToLowerInvariant()).Append('\n');
            }
            ImageSetSha256 = Hex(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())));
        }
        catch { Dispose(); throw; }
    }

    internal string ImageSha256For(int outputRows, int columns, int rows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _modules[Plan(outputRows, columns, rows).Index]!.ImageSha256;
    }
    internal void ProjectRaw(IntPtr codes, IntPtr scales, IntPtr inputBf16, IntPtr outputF32, IntPtr alphaF32,
        int outputRows, int columns, int rows, IntPtr stream)
    {
        var plan = Plan(outputRows, columns, rows); ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.cuCtxGetCurrent(out var actual).ThrowOnError();
        if (actual != _context) throw new InvalidOperationException("The body module's owning CUDA context is not current.");
        CheckPointer(codes, plan.CodeBytes, 16); CheckPointer(scales, plan.ScaleBytes, 16); CheckPointer(inputBf16, plan.InputBytes, 16);
        CheckPointer(outputF32, plan.OutputBytes, 4); CheckPointer(alphaF32, 4, 4);
        Separate(outputF32, plan.OutputBytes, codes, plan.CodeBytes); Separate(outputF32, plan.OutputBytes, scales, plan.ScaleBytes);
        Separate(outputF32, plan.OutputBytes, inputBf16, plan.InputBytes); Separate(outputF32, plan.OutputBytes, alphaF32, 4);
        void** arguments = stackalloc void*[] { &inputBf16, &codes, &scales, &alphaF32, &outputF32 };
        CudaDriverApi.cuLaunchKernel(_kernels[plan.Index], checked((uint)(outputRows / _tileN)), 1, 1, 128, 1, 1,
            checked((uint)SharedBytes(Family, rows)), stream, (IntPtr)arguments, IntPtr.Zero).ThrowOnError();
    }
    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();
    private static void CheckPointer(IntPtr pointer, long bytes, int alignment)
    {
        if (pointer.ToInt64() <= 0 || pointer.ToInt64() % alignment != 0 || bytes <= 0 || pointer.ToInt64() > long.MaxValue - bytes)
            throw new ArgumentException("Invalid or overflowing body device buffer span/alignment.");
    }
    private static void Separate(IntPtr a, long aBytes, IntPtr b, long bBytes)
    {
        if (a.ToInt64() < b.ToInt64() + bBytes && b.ToInt64() < a.ToInt64() + aBytes) throw new ArgumentException("Body output must not alias any input.");
    }
    public void Dispose() { if (_disposed) return; _disposed = true; foreach (var module in _modules) module?.Dispose(); }
}
