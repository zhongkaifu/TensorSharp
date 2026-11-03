// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public sealed partial class Qwen35CudaExecutor
{
    internal const string NinferHeadEnvironment = "TS_Q35_NINFER_HEAD";
    internal const string W8HeadModuleEnvironment = "TS_Q35_W8_HEAD_PTX";
    internal const string NinferHeadScopeEnvironment = "TS_Q35_NINFER_HEAD_SCOPE";
    internal sealed record NinferHeadOptions(string ArtifactPath, string ModulePath, CudaW8HeadScope Scope = CudaW8HeadScope.TargetAndDraft);

    internal static void ValidateNinferHeadNativeRoute(bool native, string? artifact, string? module, string? scope = null)
    {
        if (!native && (artifact is not null || module is not null || scope is not null))
            throw new NotSupportedException("The NInfer output-head companion requires the native single-device CUDA Qwen35 route. It cannot be ignored by the managed, tensor-parallel, or TS_Q35_NATIVE=0 path.");
    }

    // Values are snapshotted exactly once at construction. Unset is the only default;
    // invalid/half-specified opt-ins never silently select the original head.
    internal static NinferHeadOptions? ParseNinferHeadOptions(string? artifact, string? module, string? scope = null)
    {
        if (artifact is null && module is null && scope is null) return null;
        if (string.IsNullOrWhiteSpace(artifact) || string.IsNullOrWhiteSpace(module))
            throw new ArgumentException("The experimental NInfer head requires both TS_Q35_NINFER_HEAD and TS_Q35_W8_HEAD_PTX paths.");
        if (scope != "draft")
            throw new ArgumentException("Set TS_Q35_NINFER_HEAD_SCOPE=draft to use the experimental draft head. Replacing the target head is not enabled by this configuration.", nameof(scope));
        static string Canonical(string path)
        {
            if (!Path.IsPathFullyQualified(path) || path != Path.GetFullPath(path))
                throw new ArgumentException("NInfer companion paths must be absolute and normalized.");
            return path;
        }
        return new(Canonical(artifact), Canonical(module), CudaW8HeadScope.DraftOnly);
    }

    internal static void ValidateNinferHeadBody(Q35CudaEngine.ModelDesc model, string? generalName = null, string? basename = null)
    {
        if (model.NLayer != 64 || model.NEmbd != 5120 || model.NFf != 17408 || model.NVocab != 248320
            || model.NHead != 24 || model.NKvHead != 4 || model.HeadDim != 256
            || model.GdnKHeads != 16 || model.GdnVHeads != 48 || model.DConv != 4)
            throw new NotSupportedException("The NInfer W8 companion head is restricted to the known Qwen3.8-27B body geometry.");
        foreach (string? identity in new[] { generalName, basename })
        {
            // Architecture=qwen35 is shared by3.5/3.8 and is NOT a family ID.
            // The actual target's general.name is an opaque40hex revision, so do
            // not invent a mandatory Qwen3.8 name whitelist or checkpoint claim.
            string normalized = identity is null ? string.Empty : string.Concat(identity.Where(char.IsLetterOrDigit)).ToLowerInvariant();
            if (normalized.Contains("qwen35", StringComparison.Ordinal) || normalized.Contains("qwen2", StringComparison.Ordinal)
                || normalized.StartsWith("llama", StringComparison.Ordinal) || normalized.StartsWith("gemma", StringComparison.Ordinal)
                || normalized.StartsWith("mistral", StringComparison.Ordinal))
                throw new NotSupportedException("GGUF metadata explicitly identifies a different model family from the NInfer companion recipe.");
        }
    }

    private Q35CudaEngine CreateEngineWithCompanionHead(int maxContext, int nUbatch)
    {
        var options = ParseNinferHeadOptions(Environment.GetEnvironmentVariable(NinferHeadEnvironment),
            Environment.GetEnvironmentVariable(W8HeadModuleEnvironment), Environment.GetEnvironmentVariable(NinferHeadScopeEnvironment));
        var model = BuildModelDesc(maxContext, nUbatch);
        if (options is null) return new Q35CudaEngine(model); // Original descriptor and engine sequence.
        return CreateEngineWithNinferHead(model, options, _gguf.First.GetString("general.name"),
            _gguf.First.GetString("general.basename"), _gguf.First.GetString("general.finetune"), _gguf.First.GetString("general.size_label"));
    }

    private static Q35CudaEngine CreateEngineWithNinferHead(Q35CudaEngine.ModelDesc model, NinferHeadOptions options,
        string? generalName, string? basename, string? finetune, string? sizeLabel)
    {
        ValidateNinferHeadBody(model, generalName, basename);
        var source = NinferHeadWeightSource.Open(options);
        return LoadNinferHeadOwned(source, () =>
        {
            // The engine's synchronous load owns all GPU bytes before returning.
            using var original = JsonDocument.Parse(source.Provenance);
            string provenance = JsonSerializer.Serialize(new
            {
                ggufBody = new { generalName, basename, finetune, sizeLabel, fullGeometryValidated = true, sameCheckpointAsNinferEstablished = false },
                headPolicy = new { scope = options.Scope.ToString(), targetQ6Preserved = options.Scope == CudaW8HeadScope.DraftOnly },
                ninferCompanion = original.RootElement
            });
            model.OutputW8 = new CudaW8HeadWeightDesc(source, 0, source.Bytes, 248320, 5120,
                options.ModulePath, provenance, scope: options.Scope);
            return new Q35CudaEngine(model);
        });
    }
    // This is the actual construction seam used above. It permits CPU assertions
    // for successful/throwing upload ownership without constructing a CUDA context.
    internal static T LoadNinferHeadOwned<T>(NinferHeadWeightSource source, Func<T> load)
    {
        using (source) return load();
    }
}

/// <summary>Caller-declared destination capacity belongs to the CUDA loader. This source
/// checks its own original reader-owned object range; it does not claim host-pointer
/// allocation ownership, map the artifact, or materialize a full tensor in CPU memory.</summary>
internal sealed unsafe class NinferHeadWeightSource : ICudaWeightSource, IDisposable
{
    internal const long ExpectedBytes = 1_350_860_800, CodeBytes = 1_271_398_400, ScaleBytes = 79_462_400;
    internal const int MaximumReadSpan = 8 << 20;
    private readonly object _gate = new();
    private readonly NinferArtifactReader _reader;
    private readonly NinferTensorDescriptor _head;
    private bool _disposed;
    internal long Bytes => checked((long)_head.ByteLength);
    internal string Provenance { get; }

    internal static NinferHeadWeightSource Open(Qwen35CudaExecutor.NinferHeadOptions options)
    {
        RequireRegularPath(options.ModulePath); RequireRegularPath(options.ArtifactPath);
        var reader = new NinferArtifactReader(options.ArtifactPath);
        try
        {
            // Strict906 target validation; no input calibration/resource/head payload
            // is read by this step. Original geometry is not disguised as a GGML type.
            var plan = NinferQwen38TargetPlan.Create(reader);
            var head = plan.OutputHead.Descriptor;
            ValidateMetadata(reader.Identity, head.Name, head.Format, head.Layout, head.Shape, head.Geometry, head.ByteLength);
            return new(reader, head);
        }
        catch { reader.Dispose(); throw; }
    }

    private static void RequireRegularPath(string path)
    {
        var file = new FileInfo(path);
        if (!file.Exists || file.LinkTarget is not null) throw new IOException("Regular NInfer head artifact/module required: " + path);
        for (string? parent = Path.GetDirectoryName(path); parent is not null; parent = Path.GetDirectoryName(parent))
            if (new DirectoryInfo(parent).LinkTarget is not null) throw new IOException("Aliased companion parent refused: " + parent);
    }

    internal static void ValidateMetadata(NinferArtifactIdentity identity, string name, NinferTensorFormat format,
        NinferTensorLayout layout, IReadOnlyList<ulong> shape, NinferTensorGeometry geometry, ulong bytes)
    {
        var expected = new NinferTensorGeometry((ulong)ExpectedBytes, 256, PaddedK: 5120, GroupSize: 32,
            LowPlaneBytes: (ulong)CodeBytes, HighPlaneOffset: (ulong)CodeBytes, HighPlaneBytes: 0,
            ScalePlaneOffset: (ulong)CodeBytes, ScalePlaneBytes: (ulong)ScaleBytes);
        if (identity != new NinferArtifactIdentity("qwen3.8-27b", "nvfp4full") || name != "text/output_head"
            || format != NinferTensorFormat.W8G32Float16Scale || layout != NinferTensorLayout.RowSplitK128V1
            || !shape.SequenceEqual(new ulong[] { 248320, 5120 }) || geometry != expected || bytes != (ulong)ExpectedBytes)
            throw new NotSupportedException("Only the original known-family W8G32_F16S row-split-k128-v1 output head is supported.");
    }

    // This internal byte-source entry also permits small real-reader CPU range tests.
    // Only Open validates full companion metadata and creates the production descriptor.
    internal NinferHeadWeightSource(NinferArtifactReader reader, NinferTensorDescriptor head)
    {
        ArgumentNullException.ThrowIfNull(reader); ArgumentNullException.ThrowIfNull(head);
        reader.ReadObjectRange(head, 0, Span<byte>.Empty); // Enforce original reader ownership + liveness.
        _reader = reader; _head = head;
        Provenance = JsonSerializer.Serialize(new
        {
            recipe = "EXPERIMENTAL GGUF body/Q4 embedding/draft/tokenizer + original NInfer W8 output head",
            artifactPath = reader.FullPath, artifactBytes = reader.FileBytes, identity = reader.Identity,
            directorySha256 = reader.DirectorySha256, prefixSha256 = reader.PrefixSha256,
            headName = head.Name, head.AbsoluteOffset, head.ByteLength, format = head.Format.ToString(), layout = head.Layout.ToString(),
            fullArtifactSha256Recomputed = false, headPayloadSha256 = "COMPUTED_BY_BACKEND_DURING_UPLOAD",
            scope = "Structural metadata only; finite scale payload/actual head math/model quality are separate gates."
        });
    }

    public void Read(long offset, IntPtr dst, long bytes)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (offset < 0 || bytes < 0 || offset > Bytes || bytes > Bytes - offset)
                throw new ArgumentOutOfRangeException(nameof(offset), "Read must stay inside the original output-head object.");
            if (bytes > 0 && (dst.ToInt64() <= 0 || unchecked((ulong)dst.ToInt64()) > (ulong)long.MaxValue - (ulong)bytes))
                throw new ArgumentException("Nonempty source reads need a nonzero, nonoverflowing host destination.", nameof(dst));
            long done = 0;
            // Positional RandomAccess under the reader; no shared cursor or full-head array.
            do
            {
                int count = (int)Math.Min(MaximumReadSpan, bytes - done);
                _reader.ReadObjectRange(_head, checked((ulong)(offset + done)), new Span<byte>((byte*)dst + done, count));
                done += count;
            } while (done < bytes);
        }
    }

    public void Dispose()
    {
        lock (_gate) { if (_disposed) return; _disposed = true; _reader.Dispose(); }
    }
}
