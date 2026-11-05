// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Original DFlash2 host execution. Attachment is explicit; model/runtime and slot hooks are separate gates.</summary>
public sealed unsafe partial class Q35CudaEngine
{
    private const int NdHidden = 5120, NdFeatureWidth = 25600, NdFeatureBytes = NdFeatureWidth * 2;
    private const int NdMaxLanes = 8, NdMaxRows = 64, NdAppendRows = 2048;
    private const int NdAnchors = 0, NdBases = 8, NdValid = 16, NdIds = 24, NdPositions = 88;
    private const int NdZeroLane = 152, NdCounts = 153, NdAppendPositions = 161, NdMetaInts = NdAppendPositions + NdAppendRows;
    private CudaNinferDraftDesc _ninferDraftDesc;
    private CudaNinferDraftModule _ninferDraftModule;
    private NinferDraftLayer[] _ninferDraftLayers;
    private NinferDraftMatrix _ndFeature, _ndSelectorHidden, _ndSelectorPredecessor, _ndSelectorSuccessor;
    private IntPtr _ndContextNorm, _ndFinalNorm, _ndFrequencies, _ndMeta;
    private IntPtr _ndFeatures, _ndEncoded, _ndContext, _ndQkv, _ndKey, _ndValue, _ndKeyNormed, _ndPendingFeature;
    private IntPtr _ndResidual, _ndNorm, _ndDynamic, _ndConvInput, _ndProjected, _ndConvOutput, _ndEmbeddingF32;
    private IntPtr _ndQueryRaw, _ndQuery, _ndAttention, _ndIntermediate, _ndPacked, _ndFinal, _ndLogits;
    private IntPtr _ndCandidate, _ndUnary, _ndPredecessors, _ndSelectorBf16, _ndSelectorF32, _ndScores, _ndDrafts, _ndTopScratch;
    private IntPtr _ndAttentionPartial, _ndAttentionMaxima, _ndAttentionDenominators, _ndDiagnosticF32;
    private readonly int[] _ndHostMeta = new int[NdMetaInts];
    private readonly int[] _ndHostDrafts = new int[NdMaxLanes * 7];
    private readonly ConditionalWeakTable<Slot, NinferDraftState> _ndStates = new();
    private string _ninferDraftNamespace = string.Empty;
    public bool HasNinferDrafter => _ninferDraftDesc != null && _ninferDraftModule != null;
    public int NinferDraftMaximumLanes => HasNinferDrafter ? Math.Min(Math.Min(NdMaxLanes, RowCap / _ninferDraftDesc.BlockSize), CudaW8G32HeadModule.MaximumRows / _ninferDraftDesc.DraftTokens) : 0;
    private string NinferDraftNamespaceSuffix => _ninferDraftNamespace;
    public string NinferDraftPayloadSha256 { get; private set; }
    public string NinferDraftModuleSha256 => _ninferDraftModule?.ImageSha256;
    public bool NinferDraftUsesShortlist => _ninferDraftShortlistKernels != null;
    public int NinferDraftHeadRows => NinferDraftProposalHeadRows;

    private sealed class NinferDraftMatrix
    {
        internal IntPtr Codes, Scales;
        internal int Rows, Columns;
        internal float InverseDivisor;
    }
    private sealed class NinferDraftLayer
    {
        internal NinferDraftMatrix Qkv, Output, AttentionConv, GateUp, Down, MlpConv;
        internal IntPtr InputNorm, QueryNorm, KeyNorm, PostNorm, AttentionBase, MlpBase;
    }
    private sealed class NinferDraftState
    {
        internal int Frontier, FirstPosition;
        internal byte[] Pending;
        internal bool Faulted;
    }

    /// <summary>Opaque, owned host metadata accompanying the slot's raw ring bytes.
    /// Frontier and a possible raw anchor column are inseparable from a restore.</summary>
    internal sealed class NinferDraftSlotSnapshot
    {
        internal string Namespace { get; }
        internal int TargetPosition { get; }
        internal int Frontier { get; }
        internal int FirstPosition { get; }
        private readonly byte[] _pending;
        internal ReadOnlyMemory<byte> Pending => _pending ?? ReadOnlyMemory<byte>.Empty;
        internal NinferDraftSlotSnapshot(string identity, int target, int frontier, int first, ReadOnlySpan<byte> pending)
        {
            Namespace = identity; TargetPosition = target; Frontier = frontier; FirstPosition = first;
            _pending = pending.IsEmpty ? null : pending.ToArray();
            ValidateNinferDraftSnapshot(this, target, identity);
        }
    }

    internal static void ValidateNinferDraftSnapshot(NinferDraftSlotSnapshot snapshot, int targetPosition, string identity)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (string.IsNullOrWhiteSpace(identity) || snapshot.Namespace != identity || snapshot.TargetPosition != targetPosition
            || targetPosition < 0 || snapshot.Frontier < 0 || snapshot.FirstPosition < Math.Max(0, snapshot.Frontier - 2048) || snapshot.FirstPosition > snapshot.Frontier
            || (snapshot.Pending.IsEmpty ? snapshot.Frontier != targetPosition : snapshot.Frontier != targetPosition - 1)
            || (!snapshot.Pending.IsEmpty && snapshot.Pending.Length != NdFeatureBytes))
            throw new InvalidOperationException("NInfer draft ring frontier/pending column does not describe the restored target checkpoint.");
        foreach (ushort bits in MemoryMarshal.Cast<byte, ushort>(snapshot.Pending.Span))
            if ((bits & 0x7f80) == 0x7f80) throw new InvalidDataException("Nonfinite pending original BF16 feature.");
    }

    private NinferDraftState NinferDraftStateFor(Slot slot)
    {
        ArgumentNullException.ThrowIfNull(slot);
        if (!_ndStates.TryGetValue(slot, out var state))
        {
            if (slot.NPast != 0) throw new InvalidOperationException("A restored NInfer draft slot requires its original frontier/pending metadata.");
            state = new(); _ndStates.Add(slot, state);
        }
        if (state.Faulted) throw new InvalidOperationException("A failed NInfer draft slot must be reset or restored before reuse.");
        return state;
    }
    private void ResetNinferDraftSlot(Slot slot)
    {
        if (!HasNinferDrafter) return;
        _ndStates.Remove(slot); _ndStates.Add(slot, new());
    }
    private void ForgetNinferDraftSlot(Slot slot) => _ndStates.Remove(slot);
    private NinferDraftSlotSnapshot CaptureNinferDraftSlotState(Slot slot)
    {
        if (!HasNinferDrafter) return null;
        var state = NinferDraftStateFor(slot);
        return new(_ninferDraftNamespace, slot.NPast, state.Frontier, state.FirstPosition, state.Pending);
    }
    private void RestoreNinferDraftSlotState(Slot slot, NinferDraftSlotSnapshot snapshot)
    {
        if (!HasNinferDrafter)
        {
            if (snapshot != null) throw new InvalidOperationException("Cannot restore NInfer draft metadata without its drafter.");
            return;
        }
        ValidateNinferDraftSnapshot(snapshot, slot.NPast, _ninferDraftNamespace);
        _ndStates.Remove(slot); _ndStates.Add(slot, new() { Frontier = snapshot.Frontier, FirstPosition = snapshot.FirstPosition,
            Pending = snapshot.Pending.IsEmpty ? null : snapshot.Pending.ToArray() });
    }
    private void CopyNinferDraftSlotState(Slot source, Slot destination, int tokens)
    {
        if (!HasNinferDrafter) return;
        if (Math.Min(tokens, source.NPast) != source.NPast) throw new NotSupportedException("NInfer draft/GDN checkpoints cannot rewind a longer slot to a shorter prefix.");
        RestoreNinferDraftSlotState(destination, CaptureNinferDraftSlotState(source));
    }

    /// <summary>Called while descriptor sources are open. No model factory enables this implicitly.</summary>
    public void AttachNinferDrafter(CudaNinferDraftDesc descriptor, string modulePath, string shortlistModulePath = null)
    {
        ArgumentNullException.ThrowIfNull(descriptor); ArgumentException.ThrowIfNullOrWhiteSpace(modulePath);
        if (_m.Ninfer == null || !_ninferVerificationEnabled || _d != null || HasNinferDrafter
            || RowCap < descriptor.BlockSize)
            throw new NotSupportedException("Original NInfer drafting requires its explicit verification route and supported target geometry.");
        bool shortlist = descriptor.HeadPolicy == CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist;
        if (shortlist != (descriptor.Shortlist != null) || shortlist != (shortlistModulePath != null))
            throw new ArgumentException("Original shortlist descriptors require their explicit original Q4 module; full heads do not accept one.");
        if (_outputW8Module == null || _outputW8Weights == IntPtr.Zero || _ninferVerifiedSlots != null
            || HostBytesInUse != 0 || _active != null && _active.NPast != 0)
            throw new InvalidOperationException("Attach the original drafter to a fresh NInfer engine with its loaded target head and no pending verification.");
        MakeCurrent(); Sync(); InvalidateVerifyGraphs(); InvalidateNinferDecodeGraphs();
        int ownedStart = _owned.Count; CudaNinferDraftModule module = null;
        try
        {
            module = new(modulePath);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            using (var loader = new Loader(this))
            {
                _ndFeature = loader.NinferDraftRaw(descriptor.FeatureProjection, hash);
                _ndContextNorm = UploadNinferDraftBf16(descriptor.ContextNorm, hash); _ndFinalNorm = UploadNinferDraftBf16(descriptor.FinalNorm, hash);
                _ninferDraftLayers = new NinferDraftLayer[5];
                for (int i = 0; i < 5; i++)
                {
                    var d = descriptor.Layers[i];
                    _ninferDraftLayers[i] = new() { Qkv = loader.NinferDraftRaw(d.QueryKeyValue, hash), Output = loader.NinferDraftRaw(d.AttentionOutput, hash),
                        AttentionConv = loader.NinferDraftRaw(d.AttentionConvProjection, hash), GateUp = loader.NinferDraftRaw(d.GateUp, hash),
                        Down = loader.NinferDraftRaw(d.Down, hash), MlpConv = loader.NinferDraftRaw(d.MlpConvProjection, hash),
                        InputNorm = UploadNinferDraftBf16(d.InputNorm, hash), QueryNorm = UploadNinferDraftBf16(d.QueryNorm, hash),
                        KeyNorm = UploadNinferDraftBf16(d.KeyNorm, hash), PostNorm = UploadNinferDraftBf16(d.PostAttentionNorm, hash),
                        AttentionBase = UploadNinferDraftBf16(d.AttentionConvBase, hash), MlpBase = UploadNinferDraftBf16(d.MlpConvBase, hash) };
                }
                _ndSelectorHidden = loader.NinferDraftRaw(descriptor.SelectorHidden, hash);
                _ndSelectorPredecessor = loader.NinferDraftRaw(descriptor.SelectorPredecessor, hash);
                _ndSelectorSuccessor = loader.NinferDraftRaw(descriptor.SelectorSuccessor, hash);
            }
            AllocateNinferDraftScratch(descriptor.BlockSize);
            if (shortlist)
                InitializeNinferDraftShortlist(descriptor.Shortlist, () => new CudaNinferDraftShortlistModule(shortlistModulePath));
            double[] frequencies = CudaNinferDraftModule.RopeFrequencies(); _ndFrequencies = Alloc(frequencies.Length * 8L);
            fixed (double* p = frequencies) CudaDriverApi.cuMemcpyHtoD(_ndFrequencies, (IntPtr)p, new UIntPtr(512)).ThrowOnError();
            Array.Clear(_ndHostMeta); fixed (int* p = _ndHostMeta) Upload(_ndMeta, p, _ndHostMeta.Length);
            NinferDraftPayloadSha256 = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
            _ninferDraftNamespace = $"|ninfer-dflash2=original-a16-bf16-ring2048-{(shortlist ? "q4shortlist" : "fullhead")}-k{descriptor.DraftTokens}-pending-anchor-v1:{module.ImageSha256}:{NinferDraftPayloadSha256}" + NinferDraftShortlistNamespaceSuffix;
            ReleaseFreeArenas(); _ninferDraftModule = module; module = null; _ninferDraftDesc = descriptor;
            InvalidateHostStateLayout();
            // Root's common ring-layout hooks select the original BF16 geometry.
            if (_active != null) { if (_active.RingBase == IntPtr.Zero) CreateRing(_active); ResetNinferDraftSlot(_active); }
            Sync();
        }
        catch
        {
            Sync(); module?.Dispose(); _ninferDraftModule?.Dispose(); _ninferDraftModule = null; _ninferDraftDesc = null;
            DisposeNinferDraftShortlist();
            for (int i = _owned.Count - 1; i >= ownedStart; i--) { CudaDriverApi.cuMemFree(_owned[i]); _owned.RemoveAt(i); }
            _ninferDraftNamespace = string.Empty; throw;
        }
    }

    private IntPtr UploadNinferDraftBf16(ReadOnlyMemory<float> values, IncrementalHash hash)
    {
        byte[] bytes = new byte[checked(values.Length * 2)]; var words = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
        for (int i = 0; i < values.Length; i++)
        {
            int bits = BitConverter.SingleToInt32Bits(values.Span[i]);
            if (!float.IsFinite(values.Span[i]) || (bits & 65535) != 0) throw new InvalidDataException("Draft parameters must retain their exact original BF16 bits.");
            words[i] = (ushort)((uint)bits >> 16);
        }
        hash.AppendData(bytes); IntPtr device = Alloc(bytes.Length);
        fixed (byte* p = bytes) CudaDriverApi.cuMemcpyHtoD(device, (IntPtr)p, (UIntPtr)bytes.Length).ThrowOnError(); return device;
    }
    private sealed partial class Loader
    {
        public NinferDraftMatrix NinferDraftRaw(CudaNinferProjectionDesc descriptor, IncrementalHash hash)
        {
            if (descriptor.AllowA4 || descriptor.InputScaleDivisor != 1) throw new NotSupportedException("Original DFlash projections are A16-only without activation-divisor sites.");
            var source = new CudaNinferValidatedOriginalSource(descriptor);
            if (source.Bytes > StagingBytes) throw new NotSupportedException("Original draft object exceeds the existing bounded staging buffer.");
            source.Read(0, _host, source.Bytes); source.RequireCompleteValidation();
            hash.AppendData(new ReadOnlySpan<byte>((void*)_host, checked((int)source.Bytes)));
            IntPtr device = _e.Alloc(source.Bytes); CudaDriverApi.cuMemcpyHtoD(device, _host, (UIntPtr)source.Bytes).ThrowOnError();
            return new() { Codes = device, Scales = device + (nint)((long)descriptor.Rows * descriptor.Columns / 2),
                Rows = descriptor.Rows, Columns = descriptor.Columns, InverseDivisor = 1f / descriptor.WeightScaleDivisor };
        }
    }

    private void AllocateNinferDraftScratch(int block)
    {
        _ndMeta = Alloc(NdMetaInts * 4L); _ndFeatures = Alloc((long)RowCap * NdFeatureBytes); _ndPendingFeature = Alloc(NdFeatureBytes);
        _ndEncoded = Alloc(NdAppendRows * NdHidden * 2L); _ndContext = Alloc(NdAppendRows * NdHidden * 2L);
        _ndQkv = Alloc(NdAppendRows * 6144L * 2); _ndKey = Alloc(NdAppendRows * 1024L * 2); _ndValue = Alloc(NdAppendRows * 1024L * 2); _ndKeyNormed = Alloc(NdAppendRows * 1024L * 2);
        long e = NdMaxRows * NdHidden * 2L;
        _ndResidual = Alloc(e); _ndNorm = Alloc(e); _ndConvInput = Alloc(e); _ndProjected = Alloc(e); _ndConvOutput = Alloc(e);
        _ndEmbeddingF32 = Alloc(e * 2); _ndDynamic = Alloc(NdMaxRows * 1280L * 2);
        _ndQueryRaw = Alloc(NdMaxRows * 4096L * 2); _ndQuery = Alloc(NdMaxRows * 4096L * 2); _ndAttention = Alloc(NdMaxRows * 4096L * 2);
        _ndIntermediate = Alloc(NdMaxRows * 17408L * 2); _ndPacked = Alloc(e); _ndFinal = Alloc(e);
        _ndLogits = Alloc(NdMaxLanes * 7L * _m.NVocab * 2); _ndCandidate = Alloc(NdMaxLanes * 7L * 16 * 4); _ndUnary = Alloc(NdMaxLanes * 7L * 16 * 4);
        _ndPredecessors = Alloc(NdMaxLanes * 7L * 16 * 4); _ndSelectorBf16 = Alloc(NdMaxLanes * 7L * 256 * 2); _ndSelectorF32 = Alloc(NdMaxLanes * 7L * 256 * 4);
        _ndScores = Alloc(NdMaxLanes * 7L * 256 * 4); _ndDrafts = Alloc(NdMaxLanes * 7L * 4);
        _ndTopScratch = Alloc(CudaNinferDraftModule.Top16WorkspaceBytes(_m.NVocab, NdMaxLanes * 7));
        var attention = CudaNinferDraftModule.ValidateSwa(_m.NCtx, block, 1, 1);
        _ndAttentionPartial = Alloc(attention.PartialBytes); _ndAttentionMaxima = Alloc(attention.StatBytes); _ndAttentionDenominators = Alloc(attention.StatBytes);
    }

    private static void NinferDraftCopyRows(IntPtr source, int sourceWidth, int first, IntPtr destination, int width, int rows, IntPtr stream)
    {
        if (first < 0 || width < 1 || rows < 1 || first + width > sourceWidth) throw new ArgumentOutOfRangeException(nameof(width));
        var copy = new CudaDriverApi.CudaMemcpy2D { SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
            SrcDevice = source + (nint)(first * 2L), SrcPitch = (UIntPtr)(sourceWidth * 2L), DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
            DstDevice = destination, DstPitch = (UIntPtr)(width * 2L), WidthInBytes = (UIntPtr)(width * 2L), Height = (UIntPtr)rows };
        CudaDriverApi.cuMemcpy2DAsync(in copy, stream).ThrowOnError();
    }
    private void NinferDraftProject(NinferDraftMatrix matrix, IntPtr input, IntPtr output, int rows)
    {
        // Wide context appends run on the tensor cores (ignis' A16 MMA route); the
        // per-32-row CUDA-core loop remains for proposals and short appends.
        if (_ninferDraftModule.TryProjectMma(matrix.Codes, matrix.Scales, input, output, matrix.Rows, matrix.Columns,
                rows, matrix.InverseDivisor, Stream))
            return;
        for (int first = 0; first < rows; first += 32)
            _ninferDraftModule.ProjectRaw(matrix.Codes, matrix.Scales, input + (nint)((long)first * matrix.Columns * 2),
                output + (nint)((long)first * matrix.Rows * 2), matrix.Rows, matrix.Columns, Math.Min(32, rows - first), matrix.InverseDivisor, Stream);
    }
    internal static int ValidateNinferDraftHeadShape(int draftTokens, int lanes)
    {
        if (draftTokens < 1 || draftTokens > 7 || lanes < 1 || lanes > 8) throw new ArgumentOutOfRangeException(nameof(draftTokens));
        int columns = draftTokens * lanes;
        if (columns > CudaW8G32HeadModule.MaximumRows) throw new NotSupportedException("The exact original small-T full head supports at most33 proposal rows.");
        return columns;
    }
    internal static bool NinferDraftCarriesAnchor(int physicalWindow, int kept)
    {
        if (physicalWindow < 1 || physicalWindow > 8 || kept < 0 || kept > physicalWindow) throw new ArgumentOutOfRangeException(nameof(physicalWindow));
        return physicalWindow == 1 && kept == 1;
    }
    internal static (int First, int Rows) NinferDraftPrefillTail(int realRows, int firstTapped)
    {
        if (realRows < 0 || firstTapped < 0 || firstTapped > realRows) throw new ArgumentOutOfRangeException(nameof(firstTapped));
        int first = Math.Max(firstTapped, realRows - 2048); return (first, realRows - first);
    }
    private void CaptureNinferDraftTargetTap(int layer, int tapFirst, int rows)
    {
        if (!HasNinferDrafter || tapFirst >= rows) return;
        int tap = layer switch { 6 => 0, 20 => 1, 34 => 2, 48 => 3, 62 => 4, _ => -1 }; if (tap < 0) return;
        if (tapFirst < 0 || rows > RowCap) throw new ArgumentOutOfRangeException(nameof(tapFirst));
        _ninferDraftModule.CopyTargetTap(_x + (nint)((long)tapFirst * NdHidden * 4), _ndFeatures + (nint)((long)tapFirst * NdFeatureBytes), rows - tapFirst, tap, Stream);
    }

    private void UploadNinferDraftAppendMetadata(int physicalRows)
    {
        fixed (int* p = &_ndHostMeta[NdCounts]) Upload(_ndMeta + (nint)(NdCounts * 4L), p, NdMaxLanes + physicalRows);
    }
    private void RequireNinferDraftRing(Slot slot)
    {
        if (slot.RopeDelta != 0 || slot.RingK == null || slot.RingV == null || slot.RingK.Length != 5 || slot.RingV.Length != 5)
            throw new InvalidOperationException("Original text DFlash requires this slot's five BF16 2048-position K/V rings.");
        for (int i = 0; i < 5; i++) if (slot.RingK[i] == IntPtr.Zero || slot.RingV[i] == IntPtr.Zero)
            throw new InvalidOperationException("Missing original draft slot ring.");
    }

    // The context is encoded ONCE, and the same encoded rows feed every layer's
    // independent QKV projection. No proposal convolution/MLP is run here.
    // The full physical row shape reaches FC/QKV; counts only gate the ring copy.
    private void AppendNinferDraftContext(IntPtr features, int width, Slot[] slots, int firstSlot, int lanes)
    {
        int rows = checked(width * lanes);
        if (rows < 1 || rows > NdAppendRows || width > 2048 || lanes > 8) throw new ArgumentOutOfRangeException(nameof(width));
        UploadNinferDraftAppendMetadata(rows);
        NinferDraftProject(_ndFeature, features, _ndEncoded, rows);
        _ninferDraftModule.PlainRms(_ndEncoded, _ndContextNorm, _ndContext, NdHidden, rows, 1e-6f, Stream);
        for (int layer = 0; layer < 5; layer++)
        {
            var weights = _ninferDraftLayers[layer];
            NinferDraftProject(weights.Qkv, _ndContext, _ndQkv, rows);
            NinferDraftCopyRows(_ndQkv, 6144, 4096, _ndKey, 1024, rows, Stream);
            NinferDraftCopyRows(_ndQkv, 6144, 5120, _ndValue, 1024, rows, Stream);
            _ninferDraftModule.PlainRms(_ndKey, weights.KeyNorm, _ndKeyNormed, 128, rows * 8, 1e-6f, Stream);
            _ninferDraftModule.QkRope(IntPtr.Zero, _ndKeyNormed, _ndMeta + (nint)(NdAppendPositions * 4L), _ndFrequencies, rows, Stream);
            for (int q = 0; q < lanes; q++)
            {
                if (_ndHostMeta[NdCounts + q] == 0) continue;
                var slot = slots[firstSlot + q]; RequireNinferDraftRing(slot);
                nint offset = (nint)((long)q * width * 1024 * 2);
                _ninferDraftModule.RingAppend(_ndKeyNormed + offset, _ndValue + offset,
                    _ndMeta + (nint)((NdAppendPositions + q * width) * 4L), _ndMeta + (nint)((NdCounts + q) * 4L),
                    _ndMeta + (nint)(NdZeroLane * 4L), slot.RingK[layer], slot.RingV[layer], width, 1, 1, Stream);
            }
        }
    }
    private static void AdvanceNinferDraftFrontier(NinferDraftState state, int first, int count)
    {
        if (count == 0) return;
        if (state.Pending != null || first < state.Frontier) throw new InvalidOperationException("Original draft append would overwrite a mismatched frontier.");
        if (first > state.Frontier) state.FirstPosition = first;
        state.Frontier = checked(first + count); state.FirstPosition = Math.Max(state.FirstPosition, state.Frontier - 2048);
    }
    private void FlushNinferDraftPending(Slot slot, int targetBefore)
    {
        var state = NinferDraftStateFor(slot);
        if (state.Pending == null) return;
        if (state.Frontier != targetBefore - 1 || state.Pending.Length != NdFeatureBytes)
            throw new InvalidOperationException("A carried draft anchor does not precede the target checkpoint by one token.");
        fixed (byte* p = state.Pending) CudaDriverApi.cuMemcpyHtoD(_ndPendingFeature, (IntPtr)p, (UIntPtr)NdFeatureBytes).ThrowOnError();
        Array.Clear(_ndHostMeta, NdCounts, NdMaxLanes); _ndHostMeta[NdCounts] = 1; _ndHostMeta[NdAppendPositions] = state.Frontier;
        // One column is intentional: the original pending-anchor FC/QKV takes
        // the T1/four-chain schedule, unlike normal kept1 in a wider window.
        var one = new[] { slot }; AppendNinferDraftContext(_ndPendingFeature, 1, one, 0, 1); Sync();
        state.Pending = null; AdvanceNinferDraftFrontier(state, state.Frontier, 1);
    }

    private void InjectNinferDraftRun(RowGroup group, int rowOffset, int slotOffset, int tapFirst, int[] beforePositions)
    {
        if (!HasNinferDrafter) return;
        if (group.Verify || beforePositions.Length != group.Slots.Length) throw new ArgumentException("Ordinary original draft injection requires its pre-forward positions.");
        try
        {
            for (int q = 0; q < group.Slots.Length; q++)
            {
                var slot = group.Slots[q]; int count = group.Counts == null ? group.RowsPerSeq : group.Counts[q];
                if (count < 0 || count > group.RowsPerSeq || slot.NPast != beforePositions[q] + count) throw new InvalidOperationException("Ordinary target prefix and draft append positions differ.");
                if (count == 0) continue;
                RequireNinferDraftRing(slot); var state = NinferDraftStateFor(slot); FlushNinferDraftPending(slot, beforePositions[q]);
                var tail = NinferDraftPrefillTail(count, Math.Clamp(tapFirst - rowOffset - q * group.RowsPerSeq, 0, count));
                int laneFirst = tail.First, take = tail.Rows;
                if (take == 0) continue;
                int first = checked(beforePositions[q] + laneFirst);
                if (first < state.Frontier) throw new InvalidOperationException("Ordinary draft feature span precedes the encoded frontier.");
                Array.Clear(_ndHostMeta, NdCounts, NdMaxLanes); _ndHostMeta[NdCounts] = take;
                for (int i = 0; i < take; i++) _ndHostMeta[NdAppendPositions + i] = first + i;
                IntPtr features = _ndFeatures + (nint)((long)(rowOffset + q * group.RowsPerSeq + laneFirst) * NdFeatureBytes);
                AppendNinferDraftContext(features, take, group.Slots, q, 1); Sync();
                AdvanceNinferDraftFrontier(state, first, take);
            }
        }
        catch { foreach (var slot in group.Slots) if (_ndStates.TryGetValue(slot, out var state)) state.Faulted = true; throw; }
    }

    private void InjectNinferDraftKept(Slot[] slots, int window, int[] kept, int[] beforePositions)
    {
        if (!HasNinferDrafter) return;
        if (slots == null || kept == null || beforePositions == null || slots.Length < 1 || slots.Length > 8
            || kept.Length != slots.Length || beforePositions.Length != slots.Length || window < 1 || window > 8
            || slots.Length * window > MaxVerifyRows)
            throw new ArgumentException("Original draft commit requires one complete bounded verified window.");
        try
        {
            bool any = false;
            for (int q = 0; q < slots.Length; q++)
            {
                if (kept[q] < 0 || kept[q] > window || slots[q].NPast != beforePositions[q] + kept[q]) throw new InvalidOperationException("Committed target and draft positions differ.");
                RequireNinferDraftRing(slots[q]); NinferDraftStateFor(slots[q]);
                if (kept[q] > 0) { FlushNinferDraftPending(slots[q], beforePositions[q]); any = true; }
            }
            if (!any) return;
            if (window == 1)
            {
                // Current target API has no per-lane padded extent: window1 is
                // the source's extent0 case. A wider-window kept1 is NOT this case.
                Sync();
                for (int q = 0; q < slots.Length; q++) if (NinferDraftCarriesAnchor(window, kept[q]))
                {
                    var state = NinferDraftStateFor(slots[q]);
                    if (state.Frontier != beforePositions[q] || state.Pending != null) throw new InvalidOperationException("The carried anchor must start exactly at the draft frontier.");
                    byte[] pending = new byte[NdFeatureBytes];
                    fixed (byte* p = pending) CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _ndFeatures + (nint)((long)q * NdFeatureBytes), (UIntPtr)NdFeatureBytes).ThrowOnError();
                    state.Pending = pending;
                }
                return;
            }
            Array.Clear(_ndHostMeta, NdCounts, NdMaxLanes);
            for (int q = 0; q < slots.Length; q++)
            {
                _ndHostMeta[NdCounts + q] = kept[q];
                for (int t = 0; t < window; t++) _ndHostMeta[NdAppendPositions + q * window + t] = beforePositions[q] + t;
                if (kept[q] > 0 && beforePositions[q] < NinferDraftStateFor(slots[q]).Frontier) throw new InvalidOperationException("Verified draft context precedes its encoded frontier.");
            }
            AppendNinferDraftContext(_ndFeatures, window, slots, 0, slots.Length); Sync();
            for (int q = 0; q < slots.Length; q++) if (kept[q] > 0) AdvanceNinferDraftFrontier(NinferDraftStateFor(slots[q]), beforePositions[q], kept[q]);
        }
        catch { foreach (var slot in slots) if (_ndStates.TryGetValue(slot, out var state)) state.Faulted = true; throw; }
    }

    /// <summary>Original full-head deterministic proposals. Confidence1 only preserves
    /// the fixed-window admission API; it is not a probability or selector softmax.</summary>
    public void ProposeNinfer(Slot[] slots, int[] anchors, int[][] tokensOut, float[][] confidenceOut, float[] fullHeadOut = null)
    {
        if (NinferDraftUsesShortlist && fullHeadOut != null)
            throw new NotSupportedException("The original shortlist has a compact head, not full-target logits. Use the explicit compact-head diagnostic API.");
        ProposeNinferCore(slots, anchors, tokensOut, confidenceOut, fullHeadOut);
    }

    /// <summary>Validation-only compact original-head output, with NinferDraftHeadRows values per proposal row.</summary>
    internal void ProposeNinferHeadDiagnostic(Slot[] slots, int[] anchors, int[][] tokensOut, float[][] confidenceOut, float[] headOut)
        => ProposeNinferCore(slots, anchors, tokensOut, confidenceOut, headOut);

    private void ProposeNinferCore(Slot[] slots, int[] anchors, int[][] tokensOut, float[][] confidenceOut, float[] fullHeadOut)
    {
        if (!HasNinferDrafter) throw new InvalidOperationException("No original NInfer drafter is attached.");
        int lanes = slots?.Length ?? 0, draft = _ninferDraftDesc.DraftTokens, block = draft + 1, rows = checked(lanes * block), columns = checked(lanes * draft);
        if (lanes < 1 || lanes > 8 || anchors == null || anchors.Length != lanes || tokensOut == null || confidenceOut == null
            || tokensOut.Length != lanes || confidenceOut.Length != lanes || rows > MaxVerifyRows || rows > RowCap)
            throw new ArgumentException("Invalid original draft proposal batch.");
        // The current original W8 module has T1..33. The source switches its
        // head MMA route above33, so silently chunking would change the recipe.
        ValidateNinferDraftHeadShape(draft, lanes);
        int headRows = NinferDraftProposalHeadRows;
        if (fullHeadOut != null && fullHeadOut.LongLength < (long)columns * headRows) throw new ArgumentException("Short proposal-head diagnostic buffer.");
        for (int q = 0; q < lanes; q++)
        {
            if (slots[q] == null || (uint)anchors[q] >= (uint)_m.NVocab || tokensOut[q] == null || tokensOut[q].Length < draft
                || confidenceOut[q] == null || confidenceOut[q].Length < draft || slots[q].NPast < 0 || slots[q].NPast >= _m.NCtx)
                throw new ArgumentException("Invalid original draft lane metadata/output.");
            for (int p = 0; p < q; p++) if (ReferenceEquals(slots[q], slots[p])) throw new ArgumentException("Duplicate original draft slots.");
        }
        if (_ninferVerifiedSlots != null) throw new InvalidOperationException("A pending target verification must be committed before proposing again.");
        MakeCurrent();
        try
        {
            for (int q = 0; q < lanes; q++)
            {
                var slot = slots[q]; RequireNinferDraftRing(slot); FlushNinferDraftPending(slot, slot.NPast); var state = NinferDraftStateFor(slot);
                if (state.Frontier != slot.NPast || state.FirstPosition > Math.Max(0, slot.NPast - 2047))
                    throw new InvalidOperationException("The original draft ring has a gap in the target's committed context; restore its complete checkpoint or replay the retained tail.");
                _ndHostMeta[NdAnchors + q] = anchors[q]; _ndHostMeta[NdBases + q] = slot.NPast; _ndHostMeta[NdValid + q] = Math.Min(block, _m.NCtx - slot.NPast);
            }
            fixed (int* p = _ndHostMeta) Upload(_ndMeta, p, NdCounts);
            IntPtr anchorsDevice = _ndMeta + (nint)(NdAnchors * 4L), baseDevice = _ndMeta + (nint)(NdBases * 4L), validDevice = _ndMeta + (nint)(NdValid * 4L);
            IntPtr ids = _ndMeta + (nint)(NdIds * 4L), positions = _ndMeta + (nint)(NdPositions * 4L);
            _ninferDraftModule.PrepareBlock(anchorsDevice, baseDevice, validDevice, ids, positions, block, lanes, Stream);
            if (!TryHostNinferDraftEmbedding(_ndHostMeta.AsSpan(NdAnchors, lanes), block))
                _ninfer.EmbedW8(_tokEmbd, ids, _ndEmbeddingF32, rows, NdHidden, _m.NVocab, Stream);
            _ninfer.CastToBf16(_ndEmbeddingF32, _ndResidual, (long)rows * NdHidden, Stream);
            for (int l = 0; l < 5; l++)
            {
                var layer = _ninferDraftLayers[l];
                _ninferDraftModule.PlainRms(_ndResidual, layer.InputNorm, _ndNorm, NdHidden, rows, 1e-6f, Stream);
                NinferDraftProject(layer.AttentionConv, _ndNorm, _ndDynamic, rows);
                _ninferDraftModule.DynamicConv(_ndNorm, _ndDynamic, layer.AttentionBase, _ndConvInput, NdHidden, block, lanes, 0, Stream);
                NinferDraftProject(layer.Qkv, _ndConvInput, _ndQkv, rows);
                NinferDraftCopyRows(_ndQkv, 6144, 0, _ndQueryRaw, 4096, rows, Stream);
                NinferDraftCopyRows(_ndQkv, 6144, 4096, _ndKey, 1024, rows, Stream);
                NinferDraftCopyRows(_ndQkv, 6144, 5120, _ndValue, 1024, rows, Stream);
                _ninferDraftModule.PlainRms(_ndQueryRaw, layer.QueryNorm, _ndQuery, 128, rows * 32, 1e-6f, Stream);
                _ninferDraftModule.PlainRms(_ndKey, layer.KeyNorm, _ndKeyNormed, 128, rows * 8, 1e-6f, Stream);
                _ninferDraftModule.QkRope(_ndQuery, _ndKeyNormed, positions, _ndFrequencies, rows, Stream);
                for (int q = 0; q < lanes; q++)
                    _ninferDraftModule.Swa(_ndQuery + (nint)((long)q * block * 4096 * 2), _ndKeyNormed + (nint)((long)q * block * 1024 * 2),
                        _ndValue + (nint)((long)q * block * 1024 * 2), positions + (nint)((long)q * block * 4), validDevice + (nint)(q * 4L),
                        _ndMeta + (nint)(NdZeroLane * 4L), slots[q].RingK[l], slots[q].RingV[l], _ndAttentionPartial, _ndAttentionMaxima, _ndAttentionDenominators,
                        _ndAttention + (nint)((long)q * block * 4096 * 2), _m.NCtx, block, 1, 1, Stream);
                NinferDraftProject(layer.Output, _ndAttention, _ndProjected, rows);
                _ninferDraftModule.DynamicConv(_ndProjected, _ndDynamic, layer.AttentionBase, _ndConvOutput, NdHidden, block, lanes, 1, Stream);
                _ninferDraftModule.ResidualAdd(_ndConvOutput, _ndResidual, (long)rows * NdHidden, Stream);
                _ninferDraftModule.PlainRms(_ndResidual, layer.PostNorm, _ndNorm, NdHidden, rows, 1e-6f, Stream);
                NinferDraftProject(layer.MlpConv, _ndNorm, _ndDynamic, rows);
                _ninferDraftModule.DynamicConv(_ndNorm, _ndDynamic, layer.MlpBase, _ndConvInput, NdHidden, block, lanes, 0, Stream);
                for (int first = 0; first < rows; first += 16)
                    _ninferDraftModule.SwiGluRaw(layer.GateUp.Codes, layer.GateUp.Scales, _ndConvInput + (nint)((long)first * NdHidden * 2),
                        _ndIntermediate + (nint)((long)first * 17408 * 2), Math.Min(16, rows - first), layer.GateUp.InverseDivisor, Stream);
                NinferDraftProject(layer.Down, _ndIntermediate, _ndProjected, rows);
                _ninferDraftModule.DynamicConv(_ndProjected, _ndDynamic, layer.MlpBase, _ndConvOutput, NdHidden, block, lanes, 1, Stream);
                _ninferDraftModule.ResidualAdd(_ndConvOutput, _ndResidual, (long)rows * NdHidden, Stream);
            }
            _ninferDraftModule.PackDraftHidden(_ndResidual, _ndPacked, block, lanes, Stream);
            _ninferDraftModule.PlainRms(_ndPacked, _ndFinalNorm, _ndFinal, NdHidden, columns, 1e-6f, Stream);
            if (!TryNinferDraftShortlistHead(_ndFinal, columns))
            {
                _outputW8Module.LaunchBf16(_outputW8Weights, _outputW8.Bytes, _ndFinal, (long)columns * NdHidden * 2,
                    _ndLogits, (long)columns * _m.NVocab * 2, columns, Stream);
                _ninferDraftModule.Top16(_ndLogits, _ndCandidate, _ndUnary, _m.NVocab, columns, _ndTopScratch, Stream);
            }
            _ninferDraftModule.SelectorPredecessors(_ndCandidate, anchorsDevice, _ndPredecessors, draft, lanes, Stream);
            NinferDraftProject(_ndSelectorHidden, _ndFinal, _ndSelectorBf16, columns);
            _ninferDraftModule.CastBf16ToF32(_ndSelectorBf16, _ndSelectorF32, (long)columns * 256, Stream);
            _ninferDraftModule.SelectorScoresRaw(_ndCandidate, _ndPredecessors, _ndUnary, _ndSelectorF32, _ndSelectorSuccessor.Codes,
                _ndSelectorSuccessor.Scales, _ndSelectorSuccessor.InverseDivisor, _ndSelectorPredecessor.Codes, _ndSelectorPredecessor.Scales,
                _ndSelectorPredecessor.InverseDivisor, _ndScores, _m.NVocab, draft, lanes, Stream);
            _ninferDraftModule.SelectorWalk(_ndScores, _ndCandidate, _ndDrafts, draft, lanes, Stream);
            if (fullHeadOut != null)
            {
                if (_ndDiagnosticF32 == IntPtr.Zero) _ndDiagnosticF32 = Alloc(NdMaxLanes * 7L * _m.NVocab * 4);
                _ninferDraftModule.CastBf16ToF32(_ndLogits, _ndDiagnosticF32, (long)columns * headRows, Stream);
            }
            Sync();
            fixed (int* p = _ndHostDrafts) CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _ndDrafts, (UIntPtr)(columns * 4L)).ThrowOnError();
            if (fullHeadOut != null) fixed (float* p = fullHeadOut) CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _ndDiagnosticF32, (UIntPtr)((long)columns * headRows * 4)).ThrowOnError();
            for (int q = 0; q < lanes; q++) for (int t = 0; t < draft; t++)
            { tokensOut[q][t] = _ndHostDrafts[q * draft + t]; confidenceOut[q][t] = 1f; }
        }
        catch { foreach (var slot in slots) if (_ndStates.TryGetValue(slot, out var state)) state.Faulted = true; throw; }
    }

    private void DisposeNinferDraft()
    {
        var module = _ninferDraftModule; _ninferDraftModule = null; _ninferDraftDesc = null;
        module?.Dispose(); _ndStates.Clear();
        DisposeNinferDraftShortlist();
    }
}
