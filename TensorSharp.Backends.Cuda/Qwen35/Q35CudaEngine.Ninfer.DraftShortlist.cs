// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Separate candidate kernel contract. Implementations preserve the pinned
/// original Q4 dispatch and remap arithmetic, and own their native module.</summary>
internal interface ICudaNinferDraftShortlistKernels : IDisposable
{
    string ImageSha256 { get; }
    void Project(IntPtr codes, IntPtr scales, IntPtr inputBf16, IntPtr outputBf16, int rows, IntPtr stream);
    void Remap(IntPtr candidateIds, IntPtr map, long count, IntPtr stream);
}

public sealed unsafe partial class Q35CudaEngine
{
    internal const int NinferShortlistRows = 131072, NinferShortlistColumns = 5120;
    internal const long NinferShortlistCodeBytes = (long)NinferShortlistRows * NinferShortlistColumns / 2;
    internal const long NinferShortlistScaleBytes = (long)NinferShortlistRows * NinferShortlistColumns / 32;
    internal const long NinferShortlistWeightBytes = NinferShortlistCodeBytes + NinferShortlistScaleBytes;
    internal const int NinferShortlistLoadChunkBytes = 128 << 20;
    private ICudaNinferDraftShortlistKernels _ninferDraftShortlistKernels;
    private IntPtr _ninferDraftShortlistWeights, _ninferDraftShortlistMap;
    private string _ninferDraftShortlistNamespace = string.Empty;
    private string NinferDraftShortlistNamespaceSuffix => _ninferDraftShortlistNamespace;
    private int NinferDraftProposalHeadRows => _ninferDraftShortlistKernels == null ? _m.NVocab : NinferShortlistRows;

    /// <summary>Explicit candidate attachment with an independent namespace and compact
    /// diagnostic row width. Component and model quality gates remain separate.
    /// Ownership of the returned kernel object transfers only inside this method.</summary>
    private void InitializeNinferDraftShortlist(CudaNinferDraftShortlistDesc descriptor,
        Func<ICudaNinferDraftShortlistKernels> createKernels)
    {
        ArgumentNullException.ThrowIfNull(descriptor); ArgumentNullException.ThrowIfNull(createKernels);
        if (_m.Ninfer == null || !_ninferVerificationEnabled || _d != null || _ninferDraftShortlistKernels != null
            || HasNinferDrafter || _ninferVerifiedSlots != null || HostBytesInUse != 0 || _active != null && _active.NPast != 0)
            throw new InvalidOperationException("Initialize the separate original shortlist only during a fresh NInfer draft attachment.");
        byte[] map = EncodeNinferShortlistMap(descriptor.TokenIds);
        string mapSha = Convert.ToHexString(SHA256.HashData(map)).ToLowerInvariant();
        MakeCurrent(); Sync();
        int ownedStart = _owned.Count; ICudaNinferDraftShortlistKernels kernels = null;
        try
        {
            kernels = createKernels() ?? throw new InvalidOperationException("Missing original shortlist kernels.");
            string moduleSha = NormalizeNinferShortlistSha(kernels.ImageSha256);
            IntPtr weights;
            string payloadSha;
            using (var loader = new Loader(this)) weights = loader.NinferDraftShortlist(descriptor, out payloadSha);
            IntPtr tokenMap = Alloc(map.Length);
            fixed (byte* p = map) CudaDriverApi.cuMemcpyHtoD(tokenMap, (IntPtr)p, (UIntPtr)map.Length).ThrowOnError();
            Sync();
            string identity = NinferShortlistStateSuffix(payloadSha, mapSha, moduleSha);
            // Publish only after every original scale, map entry and byte has been
            // validated/uploaded. No converted weights or source lifetime survives.
            _ninferDraftShortlistWeights = weights; _ninferDraftShortlistMap = tokenMap;
            _ninferDraftShortlistNamespace = identity;
            _ninferDraftShortlistKernels = kernels; kernels = null;
        }
        catch
        {
            Sync(); kernels?.Dispose();
            for (int i = _owned.Count - 1; i >= ownedStart; i--)
            { CudaDriverApi.cuMemFree(_owned[i]); _owned.RemoveAt(i); }
            throw;
        }
    }

    /// <summary>UNWIRED proposal-head stage. Top16 ranks original shortlist rows;
    /// only afterwards are the candidate IDs mapped for the unchanged selector.
    /// Unary BF16 shortlist values are never remapped, merged or requantized.</summary>
    private bool TryNinferDraftShortlistHead(IntPtr proposalHidden, int columns)
    {
        if (_ninferDraftShortlistKernels == null) return false;
        if (!HasNinferDrafter || _ninferDraftDesc.HeadPolicy != CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist
            || columns < 1 || columns > NinferDraftHeadMaximumRows)
            throw new InvalidOperationException("The original shortlist requires its attached draft policy and complete supported proposal shape.");
        _ninferDraftShortlistKernels.Project(_ninferDraftShortlistWeights,
            _ninferDraftShortlistWeights + (nint)NinferShortlistCodeBytes, proposalHidden, _ndLogits, columns, Stream);
        _ninferDraftModule.Top16(_ndLogits, _ndCandidate, _ndUnary, NinferShortlistRows, columns, _ndTopScratch, Stream);
        _ninferDraftShortlistKernels.Remap(_ndCandidate, _ninferDraftShortlistMap, 16L * columns, Stream);
        return true;
    }

    private void DisposeNinferDraftShortlist()
    {
        var kernels = _ninferDraftShortlistKernels; _ninferDraftShortlistKernels = null;
        _ninferDraftShortlistWeights = _ninferDraftShortlistMap = IntPtr.Zero;
        _ninferDraftShortlistNamespace = string.Empty;
        kernels?.Dispose(); // Weight/map allocations have the common _owned lifetime.
    }

    internal static byte[] EncodeNinferShortlistMap(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        if (ids.Count != NinferShortlistRows) throw new ArgumentException("Original shortlist map must contain131072 entries.");
        byte[] bytes = new byte[NinferShortlistRows * 4];
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i]; if ((uint)id >= 248320) throw new InvalidDataException("Shortlist map token is outside the physical target vocabulary.");
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4, 4), id);
        }
        // Duplicate IDs are original row semantics and must remain present.
        return bytes;
    }
    private static string NormalizeNinferShortlistSha(string sha)
    {
        if (sha?.Length != 64) throw new ArgumentException("Complete shortlist payload/map/module SHA256 identities are required.");
        foreach (char c in sha) if (!Uri.IsHexDigit(c)) throw new ArgumentException("Invalid shortlist SHA256 identity.");
        return sha.ToLowerInvariant();
    }
    internal static string NinferShortlistStateSuffix(string payloadSha, string mapSha, string moduleSha)
        => "|ninfer-dflash-head=original-q4g64-f16s-row-split-k128-e600-v1:"
            + NormalizeNinferShortlistSha(payloadSha) + ":" + NormalizeNinferShortlistSha(mapSha) + ":" + NormalizeNinferShortlistSha(moduleSha);

    /// <summary>Single-loader streaming validation. Arbitrary chunk boundaries may split
    /// a scale. A failed or incomplete original payload can never be published.</summary>
    internal sealed class NinferShortlistPayloadValidator
    {
        internal long BytesRead { get; private set; }
        internal long ValidatedScales { get; private set; }
        private bool _failed, _hasLow;
        private byte _low;

        internal void Accept(long offset, ReadOnlySpan<byte> bytes)
        {
            if (_failed) throw new InvalidOperationException("Original shortlist validation already failed.");
            try
            {
                if (offset != BytesRead || offset < 0 || bytes.Length > NinferShortlistLoadChunkBytes
                    || offset > NinferShortlistWeightBytes - bytes.Length)
                    throw new ArgumentOutOfRangeException(nameof(offset), "Contiguous bounded original shortlist reads are required.");
                int first = checked((int)Math.Max(0, Math.Min(bytes.Length, NinferShortlistCodeBytes - offset)));
                foreach (byte b in bytes[first..])
                {
                    if (!_hasLow) { _low = b; _hasLow = true; }
                    else
                    {
                        ushort scale = (ushort)(_low | b << 8);
                        if ((scale & 0x7c00) == 0x7c00) throw new InvalidDataException("Nonfinite original shortlist F16 scale.");
                        _hasLow = false; ValidatedScales++;
                    }
                }
                BytesRead = checked(offset + bytes.Length);
            }
            catch { _failed = true; throw; }
        }
        internal void RequireComplete()
        {
            if (_failed || BytesRead != NinferShortlistWeightBytes || _hasLow || ValidatedScales != NinferShortlistScaleBytes / 2)
            {
                _failed = true;
                throw new InvalidOperationException("Every original shortlist byte and finite F16 scale must be validated before use.");
            }
        }
    }

    private sealed partial class Loader
    {
        internal IntPtr NinferDraftShortlist(CudaNinferDraftShortlistDesc descriptor, out string payloadSha)
        {
            if (descriptor.Bytes != NinferShortlistWeightBytes || StagingBytes != NinferShortlistLoadChunkBytes)
                throw new InvalidDataException("Original shortlist payload geometry differs.");
            IntPtr device = _e.Alloc(descriptor.Bytes);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var validator = new NinferShortlistPayloadValidator();
            for (long offset = 0; offset < descriptor.Bytes; offset += StagingBytes)
            {
                int count = checked((int)Math.Min(StagingBytes, descriptor.Bytes - offset));
                descriptor.Source.Read(offset, _host, count);
                var bytes = new ReadOnlySpan<byte>((void*)_host, count);
                validator.Accept(offset, bytes); hash.AppendData(bytes);
                CudaDriverApi.cuMemcpyHtoD(device + (nint)offset, _host, (UIntPtr)count).ThrowOnError();
            }
            validator.RequireComplete(); payloadSha = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
            return device;
        }
    }
}
