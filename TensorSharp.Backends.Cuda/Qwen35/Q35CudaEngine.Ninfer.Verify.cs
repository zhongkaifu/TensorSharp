// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferVerificationEnvironment = "TS_Q35_NINFER_VERIFY";
    private readonly bool _ninferVerificationEnabled = NinferVerificationRequested(Environment.GetEnvironmentVariable(NinferVerificationEnvironment));
    private Slot[] _ninferVerifiedSlots;
    private int[] _ninferVerifiedPositions, _ninferVerifiedVersions;
    private int _ninferVerifiedWindow;

    internal static bool NinferVerificationRequested(string value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(NinferVerificationEnvironment + " accepts only 0 or 1.")
    };

    private void RequireNinferVerification()
    {
        if (_m.Ninfer != null && !_ninferVerificationEnabled)
            throw new NotSupportedException("NInfer target verification requires the explicit " + NinferVerificationEnvironment + "=1 candidate option.");
    }

    private void RequireNoPendingNinferVerification()
    {
        if (_m.Ninfer != null && _ninferVerifiedSlots != null)
            throw new InvalidOperationException("Commit or discard the pending NInfer verification before another target layer pass overwrites its captured state.");
    }

    private void ValidateNinferVerifyRequest(Slot[] slots, int window, int[] tokens, int[] drawn, float[] logits)
    {
        if (_m.Ninfer == null) return;
        if (slots == null || tokens == null || drawn == null || slots.Length < 1 || slots.Length > MaxSeqs
            || window < 1 || (long)slots.Length * window > MaxVerifyRows)
            throw new ArgumentException("Invalid NInfer verification geometry.");
        int rows = slots.Length * window;
        if (tokens.Length < rows || drawn.Length < rows) throw new ArgumentException("Incomplete NInfer verification buffers.");
        if (logits != null && logits.LongLength < (long)rows * _m.NVocab)
            throw new ArgumentException("Incomplete NInfer full-head verification buffer.");
        ValidateNinferDraftVerifyGeometry(slots.Length, window);
        ValidateNinferExactVerifyRequest(slots.Length, window);
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i] == null) throw new ArgumentException("Null NInfer verification slot.");
            for (int j = 0; j < i; j++) if (ReferenceEquals(slots[i], slots[j])) throw new ArgumentException("Duplicate verification slots.");
        }
        for (int i = 0; i < rows; i++)
            if ((uint)tokens[i] >= (uint)_m.NVocab) throw new ArgumentException("NInfer verification token is outside the vocabulary.");
    }

    private void ValidateNinferDraftVerifyGeometry(int lanes, int window)
    {
        if (!HasNinferDrafter) return;
        if (lanes < 1 || lanes > 8)
            throw new ArgumentOutOfRangeException(nameof(lanes), "Original NInfer draft context commit supports at most eight lanes.");
        if (window < 1 || window > 8 || (long)lanes * window > NdMaxRows)
            throw new ArgumentOutOfRangeException(nameof(window), "Original NInfer draft context commit supports at most an eight-token physical window.");
    }

    private void ForgetNinferVerification(Slot slot = null)
    {
        if (_ninferVerifiedSlots == null) return;
        if (slot != null && Array.IndexOf(_ninferVerifiedSlots, slot) < 0) return;
        _ninferVerifiedSlots = null;
        _ninferVerifiedPositions = _ninferVerifiedVersions = null;
        _ninferVerifiedWindow = 0;
    }

    private void RecordNinferVerification(Slot[] slots, int window)
    {
        if (_m.Ninfer == null) return;
        var copy = (Slot[])slots.Clone();
        var positions = new int[slots.Length];
        var versions = new int[slots.Length];
        for (int i = 0; i < slots.Length; i++)
        {
            positions[i] = slots[i].NPast;
            versions[i] = slots[i].Version;
        }
        _ninferVerifiedSlots = copy;
        _ninferVerifiedPositions = positions;
        _ninferVerifiedVersions = versions;
        _ninferVerifiedWindow = window;
    }

    private void CommitNinfer(Slot[] slots, int window, int[] kept)
    {
        RequireNinferVerification();
        if (slots == null || kept == null || slots.Length == 0 || slots.Length != kept.Length
            || _ninferVerifiedSlots == null || slots.Length != _ninferVerifiedSlots.Length || window != _ninferVerifiedWindow)
            throw new InvalidOperationException("NInfer commit requires the most recent complete verification of these slots and window.");
        for (int i = 0; i < slots.Length; i++)
            if (!ReferenceEquals(slots[i], _ninferVerifiedSlots[i]) || slots[i].NPast != _ninferVerifiedPositions[i]
                || slots[i].Version != _ninferVerifiedVersions[i] || kept[i] < 0 || kept[i] > window)
                throw new InvalidOperationException("NInfer commit slots, positions, cache ownership or kept counts differ from their verified checkpoint.");
        ValidateNinferDraftVerifyGeometry(slots.Length, window);
        ValidateNinferExactVerifyRequest(slots.Length, window);
        MakeCurrent();
        BindSlots(slots);
        var before = (int[])_ninferVerifiedPositions.Clone();
        // Consume before submitting work: a partially failed commit must never be retried.
        ForgetNinferVerification();
        fixed (int* counts = kept) Upload(_keptDev, counts, kept.Length);
        _ninfer.GdnConvState(_vRaw, ConvDim, (long)MaxVerifyRows * ConvDim, window, _keptDev, slots.Length,
            ConvDim, _m.DConv, _tabConv, MaxSeqs, _nGdn, Stream);
        _ninfer.GdnReplay(_vQkv, (long)MaxVerifyRows * ConvDim, ConvDim, _m.GdnKHeads, _m.GdnVHeads,
            window, _keptDev, _vG, _vBeta, (long)MaxVerifyRows * _m.GdnVHeads, _tabState, MaxSeqs, slots.Length, _nGdn, Stream);
        for (int i = 0; i < slots.Length; i++) slots[i].NPast += kept[i];
        InjectNinferDraftKept(slots, window, kept, before);
        Sync();
        _servingTrace?.CommitCompleted(slots, window, kept, before);
    }

    /// <summary>Independent ordinary counted prefill for the verification probe. Keeping
    /// the complete physical window preserves projection/attention arithmetic while
    /// advancing only the real prefix. No head is selected, including for zero counts.</summary>
    internal void NinferValidationPrefillPadded(Slot[] slots, int window, int[] fullTokens, int[] counts, float[] allRowsLogits = null)
    {
        RequireNinferVerification();
        if (_m.Ninfer == null || slots == null || counts == null || fullTokens == null || slots.Length < 1
            || slots.Length > MaxSeqs || slots.Length != counts.Length || window < 1
            || (long)window * slots.Length > RowCap || fullTokens.Length != window * slots.Length)
            throw new ArgumentException("Invalid complete NInfer padded prefill geometry.");
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i] == null || counts[i] < 0 || counts[i] > window) throw new ArgumentException("Invalid NInfer padded prefix.");
            for (int j = 0; j < i; j++) if (ReferenceEquals(slots[i], slots[j])) throw new ArgumentException("Duplicate padded slots.");
        }
        if (allRowsLogits != null)
        {
            if ((long)window * slots.Length > MaxVerifyRows || allRowsLogits.LongLength < (long)window * slots.Length * _m.NVocab)
                throw new ArgumentException("Incomplete all-row ordinary verification-control head buffer.");
            foreach (int count in counts)
                if (count != window) throw new ArgumentException("Every physical control row must be real when comparing all-row heads.");
        }
        MakeCurrent();
        ForgetNinferVerification();
        EnsureRoom(slots, window);
        RunRows(slots, window, fullTokens, 0, null, fullTokens.Length, verify: false, tapFirst: 0, counts: counts);
        if (allRowsLogits != null)
        {
            int rows = window * slots.Length;
            FinalNorm(_x, _block, _outputNorm, _xn, rows, _m.NEmbd, _m.Eps, Stream);
            Head(_xn, rows, allRowsLogits, null);
        }
        Sync();
    }
}
