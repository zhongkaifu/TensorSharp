// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferHostEmbeddingEnvironment = "TS_Q35_NINFER_HOST_EMBEDDING";
    internal const long NinferHostEmbeddingMaximumPinnedBytes = 256L << 20;
    private CudaNinferHostEmbedding? _hostNinferEmbedding;
    private IntPtr _hostNinferEmbeddingStaging;
    private int _hostNinferEmbeddingRows;
    private long _hostNinferEmbeddingStagingBytes;
    private bool _hostNinferEmbeddingInitialized, _hostNinferEmbeddingDisposed, _hostNinferEmbeddingUploadPending;
    private string _hostNinferEmbeddingSuffix = string.Empty;
    private string NinferHostEmbeddingNamespaceSuffix => _hostNinferEmbeddingSuffix;
    public bool NinferHostEmbeddingEnabled => _hostNinferEmbedding != null;
    public long NinferHostEmbeddingPayloadBytes => _hostNinferEmbedding?.PayloadBytes ?? 0;
    public long NinferHostEmbeddingPinnedBytes => _hostNinferEmbeddingStagingBytes;
    public string? NinferHostEmbeddingPayloadSha256 => _hostNinferEmbedding?.PayloadSha256;

    internal static bool HostNinferEmbeddingRequested(string? value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(NinferHostEmbeddingEnvironment + " accepts only 0 or 1.")
    };

    internal static long HostNinferEmbeddingStagingBytes(int ubatch, int width, out int capacityRows)
    {
        if (ubatch <= 0 || width != 5120) throw new ArgumentOutOfRangeException(nameof(ubatch));
        // Prefill fits NUbatch; verification and native draft blocks can contain
        // up to MaxVerifyRows even when NUbatch is smaller.
        capacityRows = Math.Max(ubatch, MaxVerifyRows);
        long bytes = checked((long)capacityRows * width * sizeof(float));
        if (bytes > NinferHostEmbeddingMaximumPinnedBytes)
            throw new NotSupportedException("NInfer host embedding pinned staging exceeds its 256 MiB cap; reduce TS_Q35_UBATCH or disable the explicit host embedding option.");
        return bytes;
    }

    /// <summary>Called instead of GPU embedding-weight upload, while the original source
    /// is still open. False retains the existing device-weight load.</summary>
    private bool TryLoadNinferHostEmbedding()
    {
        ObjectDisposedException.ThrowIf(_hostNinferEmbeddingDisposed, this);
        if (_hostNinferEmbeddingInitialized) return _hostNinferEmbedding != null;
        bool requested = HostNinferEmbeddingRequested(Environment.GetEnvironmentVariable(NinferHostEmbeddingEnvironment));
        if (!requested) { _hostNinferEmbeddingInitialized = true; return false; }
        if (_m.Ninfer == null) throw new NotSupportedException(NinferHostEmbeddingEnvironment + " requires the complete NInfer target body.");
        var embedding = _m.Ninfer.Embedding;
        if (embedding.Format != CudaNinferProjectionFormat.W8G32 || embedding.Rows != 248320 || embedding.Columns != 5120)
            throw new NotSupportedException("NInfer host embedding requires the original 248320x5120 W8G32/F16-scale matrix.");
        long stagingBytes = HostNinferEmbeddingStagingBytes(_m.NUbatch, _m.NEmbd, out int capacityRows);
        CudaNinferHostEmbedding? host = null;
        IntPtr staging = IntPtr.Zero;
        try
        {
            host = new CudaNinferHostEmbedding(embedding);
            CudaDriverApi.cuMemHostAlloc(out staging, new UIntPtr(checked((ulong)stagingBytes)), 0).ThrowOnError();
            _hostNinferEmbeddingSuffix = "|ninfer-embedding=host-w8g32-f16s-bf16-v1:" + host.PayloadSha256;
            _hostNinferEmbedding = host;
            _hostNinferEmbeddingStaging = staging;
            _hostNinferEmbeddingRows = capacityRows;
            _hostNinferEmbeddingStagingBytes = stagingBytes;
            _hostNinferEmbeddingInitialized = true;
            host = null; staging = IntPtr.Zero;
            return true;
        }
        finally
        {
            if (staging != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(staging).ThrowOnError();
            host?.Dispose();
        }
    }

    /// <summary>Called immediately after RunGroups' metadata Sync. That Sync also drains
    /// the previous embedding upload before this single pinned staging buffer is reused.</summary>
    private bool TryHostNinferEmbedding(int rows)
    {
        ObjectDisposedException.ThrowIf(_hostNinferEmbeddingDisposed, this);
        var host = _hostNinferEmbedding;
        if (host == null) return false;
        if (rows < 1 || rows > _hostNinferEmbeddingRows)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (_pinned == IntPtr.Zero || _hostNinferEmbeddingStaging == IntPtr.Zero || _x == IntPtr.Zero)
            throw new InvalidOperationException("NInfer host embedding staging is unavailable.");
        int values = checked(rows * _m.NEmbd);
        host.GatherRows(new ReadOnlySpan<int>((void*)_pinned, rows), new Span<float>((void*)_hostNinferEmbeddingStaging, values));
        CudaDriverApi.cuMemcpyHtoDAsync(_x, _hostNinferEmbeddingStaging, new UIntPtr(checked((ulong)values * sizeof(float))), Stream).ThrowOnError();
        _hostNinferEmbeddingUploadPending = true;
        return true;
    }

    internal static int HostNinferDraftEmbeddingRows(ReadOnlySpan<int> anchors, int block,
        int maskToken, int vocabulary, int capacityRows)
    {
        if (anchors.Length < 1 || anchors.Length > 8 || block < 2 || block > 8 || vocabulary < 1)
            throw new ArgumentOutOfRangeException(nameof(block), "Original draft embedding requires1..8 lanes and2..8 rows per lane.");
        int rows = checked(anchors.Length * block);
        if (rows > 64 || capacityRows < rows)
            throw new ArgumentOutOfRangeException(nameof(capacityRows), "Original draft embedding exceeds the existing staging capacity.");
        if ((uint)maskToken >= (uint)vocabulary)
            throw new ArgumentOutOfRangeException(nameof(maskToken));
        foreach (int token in anchors)
            if ((uint)token >= (uint)vocabulary)
                throw new ArgumentOutOfRangeException(nameof(anchors), "Draft anchor is outside the physical vocabulary.");
        return rows;
    }

    /// <summary>Actual CPU gather for the original PrepareBlock IDs: each lane's
    /// anchor then its mask rows, including physically padded columns. The IDs
    /// do not depend on valid counts/positions. Uses only a bounded stack buffer.</summary>
    internal static int GatherHostNinferDraftEmbedding(CudaNinferHostEmbedding host,
        ReadOnlySpan<int> anchors, int block, int maskToken, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(host);
        int rows = HostNinferDraftEmbeddingRows(anchors, block, maskToken, host.Vocabulary, destination.Length / host.Width);
        Span<int> tokens = stackalloc int[64];
        for (int lane = 0; lane < anchors.Length; lane++)
        {
            tokens[lane * block] = anchors[lane];
            tokens.Slice(lane * block + 1, block - 1).Fill(maskToken);
        }
        host.GatherRows(tokens[..rows], destination);
        return rows;
    }

    /// <summary>Original native-draft embedding bridge. The caller retains
    /// the existing F32-to-BF16 cast after this stage. The common host payload and
    /// pinned staging are reused; no per-proposal host/device allocation occurs.</summary>
    private bool TryHostNinferDraftEmbedding(ReadOnlySpan<int> anchors, int block)
    {
        ObjectDisposedException.ThrowIf(_hostNinferEmbeddingDisposed, this);
        var host = _hostNinferEmbedding;
        if (host == null) return false;
        int rows = HostNinferDraftEmbeddingRows(anchors, block, CudaNinferDraftDesc.MaskToken,
            host.Vocabulary, _hostNinferEmbeddingRows);
        long bytes = checked((long)rows * _m.NEmbd * sizeof(float));
        if (!HasNinferDrafter || host.Width != 5120 || _m.NEmbd != 5120 || host.Vocabulary != _m.NVocab
            || _hostNinferEmbeddingStaging == IntPtr.Zero || _ndEmbeddingF32 == IntPtr.Zero
            || bytes > _hostNinferEmbeddingStagingBytes)
            throw new InvalidOperationException("Original native-draft host embedding staging is unavailable or undersized.");
        MakeCurrent();
        if (_hostNinferEmbeddingUploadPending)
        {
            // Propose normally drained this stream already. Still establish
            // completion here before overwriting a buffer an upload may read.
            Sync();
            _hostNinferEmbeddingUploadPending = false;
        }
        GatherHostNinferDraftEmbedding(host, anchors, block, CudaNinferDraftDesc.MaskToken,
            new Span<float>((void*)_hostNinferEmbeddingStaging, checked((int)(bytes / sizeof(float)))));
        // Conservatively retain the staging lifetime even if submission fails.
        _hostNinferEmbeddingUploadPending = true;
        CudaDriverApi.cuMemcpyHtoDAsync(_ndEmbeddingF32, _hostNinferEmbeddingStaging,
            new UIntPtr(checked((ulong)bytes)), Stream).ThrowOnError();
        return true;
    }

    private void DisposeNinferHostEmbedding()
    {
        if (_hostNinferEmbeddingDisposed) return;
        if (_hostNinferEmbeddingUploadPending)
            CudaDriverApi.cuStreamSynchronize(Stream).ThrowOnError();
        _hostNinferEmbeddingDisposed = true;
        _hostNinferEmbeddingUploadPending = false;
        var host = _hostNinferEmbedding;
        _hostNinferEmbedding = null;
        IntPtr staging = _hostNinferEmbeddingStaging;
        _hostNinferEmbeddingStaging = IntPtr.Zero;
        _hostNinferEmbeddingStagingBytes = 0;
        try
        {
            if (staging != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(staging).ThrowOnError();
        }
        finally { host?.Dispose(); }
    }
}
