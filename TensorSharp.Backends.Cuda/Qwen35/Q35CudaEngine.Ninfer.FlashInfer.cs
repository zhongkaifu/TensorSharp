// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferFlashInferEnvironment = "TS_Q35_NINFER_FI_MANIFEST";
    internal const string NinferFlashInferBridgeEnvironment = "TS_Q35_NINFER_FI_BRIDGE_PTX";
    internal const string NinferFlashInferStoragePolicy = "original-raw-singlecopy-m128x4-adjacentk-f32-scatter-existing-epilogues-t1t8-v1";
    internal const int NinferFlashInferMatrixCount = 199;
    internal const long NinferFlashInferPlaneScratchBytes = 100270080;
    private const long NinferFlashInferInputBytes = 8L * 17408 * 2;
    private const long NinferFlashInferOutputBytes = 8L * 34816 * 4;

    private sealed record NinferFlashInferConfiguration(string Manifest, string Bridge, string Repack);
    internal readonly record struct NinferFlashInferProjectionPlan(int OutputRows, int Columns,
        long CodeBytes, long ScaleBytes, long OriginalBytes, CudaNinferOriginalRowMapping Mapping);
    private sealed record NinferFlashInferResident(NinferFlashInferProjectionPlan Plan, IntPtr Alpha, NinferMatrix PlaneView);
    private NinferFlashInferConfiguration? _ninferFiConfiguration;
    private bool _ninferFiPreflighted, _ninferFiDisposed, _ninferFiLoadComplete;
    private readonly Dictionary<CudaNinferProjectionDesc, NinferFlashInferProjectionPlan> _ninferFiPlan = new();
    private readonly Dictionary<CudaNinferProjectionDesc, int> _ninferFiAlphaIndices = new();
    private readonly HashSet<CudaNinferProjectionDesc> _ninferFiLoadedDescriptors = new();
    private readonly Dictionary<NinferMatrix, NinferFlashInferResident> _ninferFiResident = new();
    private CudaNinferFlashInferBodyModule? _ninferFiBody;
    private CudaNinferFlashInferBridge? _ninferFiBridge;
    private CudaNinferRepackModule? _ninferFiRepack;
    private IncrementalHash? _ninferFiPayloadHash;
    private IntPtr _ninferFiInput, _ninferFiOutput, _ninferFiPlanes, _ninferFiAlphas;
    private string _ninferFiNamespace = string.Empty;
    private string _ninferFiArithmeticPolicy = string.Empty;
    private long _ninferFiOriginalBytes;
    private int _ninferFiLoaded;

    public bool NinferFlashInferEnabled => _ninferFiBody != null && !_ninferFiDisposed;
    public string NinferFlashInferPolicy => _ninferFiArithmeticPolicy;
    public string NinferFlashInferImageSetSha256 { get; private set; } = string.Empty;
    public string NinferFlashInferManifestSha256 { get; private set; } = string.Empty;
    public string NinferFlashInferBridgeImageSha256 { get; private set; } = string.Empty;
    public string NinferFlashInferRepackImageSha256 { get; private set; } = string.Empty;
    public string NinferFlashInferPayloadSha256 { get; private set; } = string.Empty;
    public int NinferFlashInferLoadedMatrices => _ninferFiLoaded;
    public long NinferFlashInferOriginalBytes => _ninferFiOriginalBytes;
    public long NinferFlashInferScratchBytes => NinferFlashInferEnabled
        ? NinferFlashInferPlaneScratchBytes + NinferFlashInferInputBytes + NinferFlashInferOutputBytes + NinferFlashInferMatrixCount * 4L : 0;
    private string NinferFlashInferNamespaceSuffix => _ninferFiNamespace;

    internal static string? ResolveNinferFlashInferPath(string? raw, string option, bool required)
    {
        if (raw == null && !required) return null;
        if (string.IsNullOrWhiteSpace(raw) || !Path.IsPathFullyQualified(raw) || raw != Path.GetFullPath(raw))
            throw new ArgumentException(option + " requires an explicit normalized absolute file path.");
        if (!File.Exists(raw)) throw new FileNotFoundException("The explicit FI dependency does not exist: " + option, raw);
        return raw;
    }

    internal static void ValidateNinferFlashInferFlags(bool enabled, bool hasNinfer, bool hasBridge, bool hasExactVerify)
    {
        if (!enabled)
        {
            if (hasBridge) throw new ArgumentException(NinferFlashInferBridgeEnvironment + " requires " + NinferFlashInferEnvironment);
            return;
        }
        if (!hasNinfer) throw new NotSupportedException("The explicit raw FI body requires the original NInfer target.");
        if (hasExactVerify) throw new NotSupportedException("The raw FI body and exact-verification policy cannot be selected together.");
        if (!hasBridge) throw new ArgumentException("The raw FI body requires its explicit owned placement bridge.");
    }

    /// <summary>Geometry admission is independent of module availability. GDN input,
    /// BF16 exceptions, alpha/beta, W8 and draft weights never enter this route.</summary>
    internal static bool TryPlanNinferFlashInferProjection(CudaNinferProjectionDesc descriptor,
        out NinferFlashInferProjectionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        plan = default;
        if (descriptor.Format != CudaNinferProjectionFormat.NvFp4Planes) return false;
        CudaNinferOriginalRowMapping expected;
        switch (descriptor.Rows, descriptor.Columns)
        {
            case (34816, 5120): expected = CudaNinferOriginalRowMapping.GateUpRows8; break;
            case (5120, 17408):
            case (5120, 6144): expected = CudaNinferOriginalRowMapping.Identity; break;
            case (14336, 5120): expected = CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue; break;
            default: return false;
        }
        if (descriptor.Source is not ICudaNinferOriginalProjectionSource original)
            throw new NotSupportedException("The raw FI body requires bounded access to original encoded projection bytes.");
        var geometry = CudaNinferRepackModule.ValidateGeometry(descriptor.Rows, descriptor.Columns, expected);
        if (descriptor.SourceOffset != 0 || original.Rows != descriptor.Rows || original.Columns != descriptor.Columns
            || original.RowMapping != expected || original.OriginalBytes != geometry.OriginalBytes
            || descriptor.Bytes != geometry.DestinationBytes)
            throw new ArgumentException("The raw FI projection source or row mapping differs from its target role.");
        plan = new(descriptor.Rows, descriptor.Columns, geometry.CodeBytes, geometry.ScaleBytes, geometry.OriginalBytes, expected);
        return true;
    }

    /// <summary>Snapshot the explicit policy and validate all borrowed sources before
    /// CUDA context/weight/scratch allocation. Root constructor calls this once.</summary>
    private void PreflightNinferFlashInfer()
    {
        ObjectDisposedException.ThrowIf(_ninferFiDisposed, this);
        if (_ninferFiPreflighted) throw new InvalidOperationException("The FI policy was already snapshotted.");
        string? raw = Environment.GetEnvironmentVariable(NinferFlashInferEnvironment);
        string? bridge = Environment.GetEnvironmentVariable(NinferFlashInferBridgeEnvironment);
        ValidateNinferFlashInferFlags(raw != null, _m.Ninfer != null, bridge != null,
            Environment.GetEnvironmentVariable(NinferExactVerifyEnvironment) != null);
        if (raw == null) { _ninferFiPreflighted = true; return; }
        var configuration = new NinferFlashInferConfiguration(
            ResolveNinferFlashInferPath(raw, NinferFlashInferEnvironment, true)!,
            ResolveNinferFlashInferPath(bridge, NinferFlashInferBridgeEnvironment, true)!,
            ResolveNinferFlashInferPath(Environment.GetEnvironmentVariable(NinferLoadRepackEnvironment), NinferLoadRepackEnvironment, true)!);
        var recipe = _m.Ninfer!;
        for (int layer = 0; layer < recipe.Layers.Length; layer++)
        {
            var weights = recipe.Layers[layer];
            // Preserve the same deterministic order as LoadNinferLayer. In
            // particular AlphaBeta and every recurrent Input are excluded.
            if ((layer + 1) % 4 == 0) Add(weights.Input);
            Add(weights.Output); Add(weights.GateUp); Add(weights.Down);
        }
        if (_ninferFiPlan.Count != NinferFlashInferMatrixCount
            || _ninferFiPlan.Values.Count(x => x.OutputRows == 34816) != 64
            || _ninferFiPlan.Values.Count(x => x.Columns == 17408) != 64
            || _ninferFiPlan.Values.Count(x => x.Columns == 6144) != 61
            || _ninferFiPlan.Values.Count(x => x.OutputRows == 14336) != 10)
            throw new NotSupportedException("The raw FI policy requires all199 projections of the validated Qwen3.8 target recipe.");
        _ninferFiConfiguration = configuration; _ninferFiPreflighted = true;
        void Add(CudaNinferProjectionDesc descriptor)
        {
            if (!TryPlanNinferFlashInferProjection(descriptor, out var plan)) return;
            if (!_ninferFiPlan.TryAdd(descriptor, plan)) throw new ArgumentException("The FI recipe aliases projection descriptors.");
            _ninferFiAlphaIndices.Add(descriptor, _ninferFiAlphaIndices.Count);
        }
    }

    private void InitializeNinferFlashInfer()
    {
        ObjectDisposedException.ThrowIf(_ninferFiDisposed, this);
        if (!_ninferFiPreflighted) throw new InvalidOperationException("Preflight the FI policy before any CUDA allocation.");
        if (_ninferFiConfiguration == null) return;
        if (_ninferFiBody != null) throw new InvalidOperationException("The FI modules were already initialized.");
        var configuration = _ninferFiConfiguration;
        CudaNinferFlashInferBodyModule? body = null;
        CudaNinferFlashInferBridge? bridge = null;
        CudaNinferRepackModule? repack = null;
        IncrementalHash? hash = null;
        try
        {
            body = new(configuration.Manifest); bridge = new(configuration.Bridge); repack = new(configuration.Repack);
            foreach (var plan in _ninferFiPlan.Values)
                for (int rows = 1; rows <= 8; rows++)
                    if (!CudaNinferFlashInferBodyModule.Supports(plan.OutputRows, plan.Columns, rows))
                        throw new NotSupportedException("The explicit FI image set is incomplete for the admitted model.");
            hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            _ninferFiInput = Alloc(NinferFlashInferInputBytes); _ninferFiOutput = Alloc(NinferFlashInferOutputBytes);
            _ninferFiPlanes = Alloc(NinferFlashInferPlaneScratchBytes);
            var alphas = new float[NinferFlashInferMatrixCount];
            foreach (var pair in _ninferFiAlphaIndices) alphas[pair.Value] = 1.0f / pair.Key.WeightScaleDivisor;
            _ninferFiAlphas = UploadF32(alphas);
            NinferFlashInferImageSetSha256 = body.ImageSetSha256; NinferFlashInferManifestSha256 = body.ManifestSha256;
            NinferFlashInferBridgeImageSha256 = bridge.ImageSha256; NinferFlashInferRepackImageSha256 = repack.ImageSha256;
            _ninferFiArithmeticPolicy = body.ArithmeticPolicy;
            _ninferFiBody = body; _ninferFiBridge = bridge; _ninferFiRepack = repack; _ninferFiPayloadHash = hash;
            body = null; bridge = null; repack = null; hash = null;
        }
        finally { hash?.Dispose(); repack?.Dispose(); bridge?.Dispose(); body?.Dispose(); }
    }

    private bool TryLoadNinferFlashInferMatrix(Loader loader, CudaNinferProjectionDesc descriptor, out NinferMatrix? matrix)
    {
        matrix = null;
        if (!_ninferFiPlan.TryGetValue(descriptor, out var plan)) return false;
        if (!NinferFlashInferEnabled || _ninferFiLoadComplete || _ninferFiPayloadHash == null)
            throw new InvalidOperationException("The admitted raw FI projection cannot use an uninitialized or completed loader.");
        if (_ninferFiLoadedDescriptors.Contains(descriptor))
            throw new InvalidOperationException("An admitted FI projection was already loaded.");
        var data = loader.NinferFlashInferRaw(descriptor, plan, _ninferFiPayloadHash);
        matrix = new() { Descriptor = descriptor, Data = data, Scales = data + (nint)plan.CodeBytes };
        var view = new NinferMatrix { Descriptor = descriptor, Data = _ninferFiPlanes, Scales = _ninferFiPlanes + (nint)plan.CodeBytes };
        _ninferFiResident.Add(matrix, new(plan, _ninferFiAlphas + (nint)(_ninferFiAlphaIndices[descriptor] * 4L), view));
        _ninferFiLoadedDescriptors.Add(descriptor);
        _ninferFiLoaded++;
        _ninferFiOriginalBytes = checked(_ninferFiOriginalBytes + plan.OriginalBytes);
        return true;
    }

    private void CompleteNinferFlashInferLoad()
    {
        if (_ninferFiConfiguration == null) return;
        if (!NinferFlashInferEnabled || _ninferFiLoadComplete || _ninferFiLoaded != NinferFlashInferMatrixCount
            || !_ninferFiLoadedDescriptors.SetEquals(_ninferFiPlan.Keys)
            || _ninferFiPayloadHash == null || _ninferFiBody == null)
            throw new InvalidOperationException("The raw FI model load is incomplete or duplicated.");
        NinferFlashInferPayloadSha256 = Convert.ToHexString(_ninferFiPayloadHash.GetHashAndReset()).ToLowerInvariant();
        _ninferFiPayloadHash.Dispose(); _ninferFiPayloadHash = null;
        _ninferFiNamespace = BuildNinferFlashInferStateSuffix(_ninferFiBody.ArithmeticPolicy,
            NinferFlashInferManifestSha256, NinferFlashInferImageSetSha256, NinferFlashInferBridgeImageSha256,
            NinferFlashInferRepackImageSha256, NinferFlashInferPayloadSha256);
        _ninferFiLoadComplete = true;
        Console.Error.WriteLine($"[q35-cuda] explicit raw FI body: {_ninferFiLoaded} matrices, {_ninferFiOriginalBytes} original bytes, policy={_ninferFiBody.ArithmeticPolicy}, images={NinferFlashInferImageSetSha256}");
    }

    internal static string BuildNinferFlashInferStateSuffix(string arithmeticPolicy, params string[] hashes)
    {
        if (string.IsNullOrWhiteSpace(arithmeticPolicy) || arithmeticPolicy.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '-' or '_'))
            || hashes.Length != 5 || hashes.Any(x => x == null || x.Length != 64 || x.Any(c => !Uri.IsHexDigit(c))))
            throw new ArgumentException("The raw FI namespace requires the complete arithmetic/layout/image/payload identity.");
        return "|ninfer-fi=" + NinferFlashInferStoragePolicy + ":" + arithmeticPolicy + ":" + string.Join(":", hashes.Select(x => x.ToLowerInvariant()));
    }

    /// <summary>Small-T policy only. No CPU reads, allocation or hidden epilogue;
    /// all scratch and original matrices are owned by this engine for graph life.</summary>
    private bool TryProjectNinferFlashInfer(NinferMatrix matrix, IntPtr input, IntPtr output, int rows)
    {
        if (!_ninferFiResident.TryGetValue(matrix, out var resident)) return false;
        ObjectDisposedException.ThrowIf(_ninferFiDisposed, this);
        if (!_ninferFiLoadComplete || _ninferFiBody == null || _ninferFiBridge == null)
            throw new InvalidOperationException("The raw FI body is not ready for execution.");
        var plan = resident.Plan;
        if (!CudaNinferFlashInferBodyModule.Supports(plan.OutputRows, plan.Columns, rows)) return false;
        _ninferFiBridge.Cast(input, _ninferFiInput, checked((long)rows * plan.Columns), Stream);
        bool mapped = plan.Mapping != CudaNinferOriginalRowMapping.Identity;
        IntPtr rawOutput = mapped ? _ninferFiOutput : output;
        _ninferFiBody.ProjectRaw(matrix.Data, matrix.Scales, _ninferFiInput, rawOutput, resident.Alpha,
            plan.OutputRows, plan.Columns, rows, Stream);
        if (mapped) _ninferFiBridge.Scatter(rawOutput, output, plan.OutputRows, rows, plan.Mapping, Stream);
        return true;
    }

    /// <summary>One lossless plane workspace is reused in stream order. The view
    /// never escapes the current projection, and resident raw metadata stays intact.
    /// Existing large-T A4/A16 math and rounding run unchanged on this exact layout.</summary>
    private NinferMatrix PrepareNinferFlashInferFallback(NinferMatrix matrix)
    {
        if (!_ninferFiResident.TryGetValue(matrix, out var resident)) return matrix;
        ObjectDisposedException.ThrowIf(_ninferFiDisposed, this);
        if (!_ninferFiLoadComplete || _ninferFiRepack == null)
            throw new InvalidOperationException("The resident raw matrix requires its live exact layout repacker.");
        var plan = resident.Plan;
        _ninferFiRepack.Repack(matrix.Data, _ninferFiPlanes, plan.OutputRows, plan.Columns, plan.Mapping, Stream);
        return resident.PlaneView;
    }

    private void DisposeNinferFlashInfer()
    {
        if (_ninferFiDisposed) return;
        _ninferFiDisposed = true;
        _ninferFiPayloadHash?.Dispose(); _ninferFiPayloadHash = null;
        _ninferFiRepack?.Dispose(); _ninferFiRepack = null;
        _ninferFiBridge?.Dispose(); _ninferFiBridge = null;
        _ninferFiBody?.Dispose(); _ninferFiBody = null;
        _ninferFiPlan.Clear(); _ninferFiAlphaIndices.Clear(); _ninferFiResident.Clear(); _ninferFiLoadedDescriptors.Clear();
        _ninferFiInput = _ninferFiOutput = _ninferFiPlanes = _ninferFiAlphas = IntPtr.Zero;
        // Device buffers belong only to _owned; the synchronized engine disposer
        // releases each once. Immutable namespace and provenance remain readable.
    }

    private sealed partial class Loader
    {
        public IntPtr NinferFlashInferRaw(CudaNinferProjectionDesc descriptor,
            NinferFlashInferProjectionPlan plan, IncrementalHash payloadHash)
        {
            var source = new CudaNinferValidatedOriginalSource(descriptor);
            if (source.Bytes != plan.OriginalBytes || source.Bytes > StagingBytes)
                throw new NotSupportedException("The original FI projection exceeds the bounded loader geometry.");
            // Validate all scales and exact divisor bits before a kernel can read
            // these bytes. There is a single final device copy, no resident planes.
            source.Read(0, _host, source.Bytes); source.RequireCompleteValidation();
            payloadHash.AppendData(new ReadOnlySpan<byte>((void*)_host, checked((int)source.Bytes)));
            IntPtr data = _e.Alloc(source.Bytes);
            CudaDriverApi.cuMemcpyHtoD(data, _host, (UIntPtr)source.Bytes).ThrowOnError();
            return data;
        }
    }
}
