// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : ISpeculativeTarget
    {
        // The MTP head consumes all HC streams BEFORE the final mixer. Collapsing
        // this to HiddenSize loses information required by the trained head. A DFlash
        // drafter reads its own row instead: the hc-stream means of its target layers.
        public int SpecFeatureSize => HasDFlash ? _dflash.FeatureSize : _hcDim;
        // DFlash prefills its ring from the trunk's per-row features in whole chunks.
        public int SpecPrefillChunkSize => HasDFlash ? DFlashPrefillChunkSize : 0;
        // DFlash2 drafts a block of 7 per pass; like Strata's Mac engine, verify windows of up to 6
        // drafts, ended where the product of the drafter's confidences falls below the gate.
        public int SpecPreferredDraftWindow => HasDFlash ? DFlashWindow() : 3;
        public float SpecPreferredBlockGate => HasDFlash ? DFlashGate() : float.NaN;
        // Each verify row routes its own ten experts, so a draft the MTP head is unsure of costs more than it saves:
        // on ggml_metal (M5 Pro, ISTA Q2_0) a 0.5 gate decoded long answers at 29.9-31.0 tok/s against 25.4-26.8 with
        // the per-token default (0.15), and accepted 90/72/79% of its drafts against 88/66/70% (Strata's --spec-min-p).
        public float SpecPreferredDraftGate => 0.5f;
        // Verification includes the pending anchor. CPU/CUDA preserve decode
        // arithmetic only through eight rows, including for explicit windows.
        public int SpecMaxDraftTokens => ResolveSpecMaxDraftTokens(_backend);
        internal static int ResolveSpecMaxDraftTokens(BackendType backend)
            => backend is BackendType.GgmlCpu or BackendType.GgmlCuda ? 7 : int.MaxValue;
        // True right after a partial acceptance was committed from the verify's own
        // per-row states (SpecOnVerifyAccepted): the scheduler then only rewinds.
        public bool SpecVerifyPersistsAcceptedKv => _specRowCommitted;

        // ---- verify row commit (NInfer's ReplaySSM idea) ------------------------
        // A partial acceptance used to restore the pre-verify recurrent state and
        // re-forward the kept rows through all 48 layers (~60 ms on Flash Next,
        // ggml_metal). The verify span now keeps every row's GDN conv/SSM state and
        // PLE conv history (upstream ggml_gated_delta_net's per-row snapshots), and
        // the accepted row is copied into the live state instead: with memcpy where
        // the state buffers are host-visible (ggml_metal, ggml_cpu), as one ggml copy
        // graph on the device otherwise (ggml_cuda, ggml_vulkan). TS_Q4E_VERIFY_ROW_COMMIT=0
        // keeps the restore-and-re-forward path.
        private static bool VerifyRowCommitEnabled()
            => Environment.GetEnvironmentVariable("TS_Q4E_VERIFY_ROW_COMMIT") is not "0";
        private bool _specCaptureArmed, _specRowCommitted;
        private int _specCapturedRows, _specVerifyStart, _specCommitPosition;
        private int[] _specVerifyTokens;
        public bool SpecPlainStepUsesForward => true;
        // SpecForward delegates to Forward's image injection and captures each
        // chunk's MRoPE positions before Forward consumes them for MTP catch-up.
        public bool SpecSupportsMultimodalPrefill => !HasDistributedQwen;
        public bool SpecTrunkFollowsBoundCache => !HasDistributedQwen;

        /// <summary>On ggml_metal with expert tiers that read their misses into window slots (unified memory), a step
        /// is bound by the experts it reads: four requests batched decoded 30-35 tokens/s in all on a 48 GB M5 Pro
        /// against 40-44 for one speculating alone, so concurrent requests take turns (BatchExecutor).</summary>
        public bool SpecTurnsPreferred => _backend == BackendType.GgmlMetal && _tierWindowSlots > 0;
        public bool SpeculationProfitable => !HasDistributedQwen && !_specStateFailed && IsGgmlBackend
            && !_tokenGraphUnsupported && !_fusedGateUpExperts
            && !_fusedFfnUnsupported && !_fusedGdnUnsupported && !_fusedAttnUnsupported
            && !_gdnVerify;

        private bool _specForwardActive;
        private bool _specAllLogitsRows;
        private IntPtr _specHiddenOutput;
        private IntPtr _specLogitsOutput;
        // A speculative forward over a prompt (more rows than any verify window), which may run
        // as several spans when one does not fit (ForwardInSpans).
        private bool _specPromptRows;
        private IntPtr _specNativeSnapshot;
        private object _specSnapshotOwner;
        private object _specSnapshotAttnArgs, _specSnapshotGdnArgs, _specSnapshotPleArgs;
        private SpecMetadata _specMetadata;
        private int _specResetVersion;
        private bool _specRecurrentRestored;
        private bool _specStateFailed;

        // Only metadata changes on the host in the span path. GDN conv/SSM and
        // PLE conv bytes remain in their existing device buffers; the native
        // snapshot copies those device-to-device after flushing an arena slot.
        internal sealed class SpecMetadata
        {
            internal object Owner;
            internal int ResetVersion;
            internal int Position;
            internal int PleNextPosition;
            internal int MropeGap;
            internal int[] PleHistory;
            internal bool DeviceStateAuthoritative;
        }

        private object ActiveSpecOwner => (object)_gdnConvStateT ?? _kCache;

        private SpecMetadata CaptureSpecMetadata() => new SpecMetadata
        {
            Owner = ActiveSpecOwner,
            ResetVersion = _specResetVersion,
            Position = _cacheSeqLen,
            PleNextPosition = _pleNextPos,
            MropeGap = _mropeCacheGap,
            PleHistory = _pleHistory?.ToArray() ?? Array.Empty<int>(),
            DeviceStateAuthoritative = _deviceStateAuthoritative,
        };

        private void ValidateSpecMetadata(SpecMetadata metadata)
        {
            if (metadata == null || !ReferenceEquals(metadata.Owner, ActiveSpecOwner)
                || metadata.ResetVersion != _specResetVersion)
                throw new InvalidOperationException("qwen4exp: speculative snapshot no longer belongs to the active conversation.");
        }

        private void RestoreSpecMetadata(SpecMetadata metadata)
        {
            ValidateSpecMetadata(metadata);
            _pleHistory ??= new List<int>();
            _pleHistory.EnsureCapacity(metadata.PleHistory.Length);
            _pleHistory.Clear();
            _pleHistory.AddRange(metadata.PleHistory);
            _pleNextPos = metadata.PleNextPosition;
            _mropeCacheGap = metadata.MropeGap;
            _deviceStateAuthoritative = metadata.DeviceStateAuthoritative;
        }

        private unsafe void CopySpecLastLogits(int tokenCount)
        {
            int offset = _specAllLogitsRows ? checked((tokenCount - 1) * Config.VocabSize) : 0;
            new ReadOnlySpan<float>((float*)_specLogitsOutput + offset, Config.VocabSize)
                .CopyTo(_spanLogits);
        }

        // A prompt chunk older than a block drafter's ring asks for its LAST row's features only (a one-row buffer,
        // SpeculativeExecution.PrefillStep): the span writes every row, so it writes them here and the last is copied out.
        private float[] _specLastRowScratch;

        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            int feat = SpecFeatureSize;
            if (hAllOut != null && tokens.Length > 1 && hAllOut.Length >= feat && hAllOut.Length < checked(tokens.Length * feat))
            {
                long need = checked((long)tokens.Length * feat);
                if (_specLastRowScratch == null || _specLastRowScratch.LongLength < need)
                    _specLastRowScratch = new float[need];
                SpecForward(tokens, _specLastRowScratch, logitsOut, allLogitsRows);
                Array.Copy(_specLastRowScratch, (long)(tokens.Length - 1) * feat, hAllOut, 0, feat);
                return;
            }
            RefuseUnmirroredQwenOperation("speculative execution and rollback");
            ArgumentNullException.ThrowIfNull(logitsOut);
            if (tokens.Length == 0 || logitsOut.Length < checked(Config.VocabSize * (allLogitsRows ? tokens.Length : 1))
                || (hAllOut != null && hAllOut.Length < checked(tokens.Length * feat)))
                throw new ArgumentException("qwen4exp: speculative output buffers do not cover the requested rows.");
            if (!SpeculationProfitable || !EnsureHeadArgs() || !EnsurePleArgs())
                throw new NotSupportedException("qwen4exp speculation requires the complete GGML token-span path, including PLE and the output head.");
            if (_specForwardActive)
                throw new InvalidOperationException("qwen4exp: a speculative forward is already active.");
            MtpPositionRange mtpPositions = null;
            object mtpOwner = null;
            if (_mtpReady)
            {
                // EnsureGdnArgs stabilizes the holder identity before publishing.
                if (!EnsureGdnArgs()) throw new InvalidOperationException("qwen4exp: cannot initialize speculative state.");
                mtpOwner = ActiveSpecOwner;
                mtpPositions = PrepareMtpPositions(mtpOwner, tokens.Length);
            }
            fixed (float* hidden = hAllOut)
            fixed (float* logits = logitsOut)
            {
                _specForwardActive = true;
                _specPromptRows = tokens.Length > 8;
                _specAllLogitsRows = allLogitsRows;
                _specHiddenOutput = (IntPtr)hidden;
                _specLogitsOutput = (IntPtr)logits;
                bool capture = _specCaptureArmed && tokens.Length >= 2 && tokens.Length <= 8
                    && (_backend is BackendType.GgmlMetal or BackendType.GgmlCpu or BackendType.GgmlCuda
                        or BackendType.GgmlVulkan) && VerifyRowCommitEnabled();
                _specCaptureArmed = false;
                _specCapturedRows = 0;
                int verifyStart = _cacheSeqLen;
                if (capture) GgmlBasicOps.Qwen4ExpCaptureVerifyRows(true);
                try
                {
                    Forward(tokens);
                    if (capture)
                    {
                        _specCapturedRows = tokens.Length;
                        _specVerifyTokens = (int[])tokens.Clone();
                        _specVerifyStart = verifyStart;
                    }
                    if (mtpPositions != null) PublishMtpPositions(_mtpPositions[mtpOwner], mtpPositions);
                }
                catch { _specStateFailed = true; throw; }
                finally
                {
                    if (capture) GgmlBasicOps.Qwen4ExpCaptureVerifyRows(false);
                    _specForwardActive = false;
                    _specPromptRows = false;
                    _specAllLogitsRows = false;
                    _specHiddenOutput = IntPtr.Zero;
                    _specLogitsOutput = IntPtr.Zero;
                }
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen)
        {
            RefuseUnmirroredQwenOperation("speculative execution and rollback");
            if (requiredSeqLen < _cacheSeqLen || requiredSeqLen > MaxContextLength)
                throw new ArgumentOutOfRangeException(nameof(requiredSeqLen));
            EnsureCacheCapacity(requiredSeqLen);
            if (_mtpReady) EnsureMtpState(requiredSeqLen);
        }

        public unsafe void SpecSnapshotRecurrentState()
        {
            RefuseUnmirroredQwenOperation("speculative execution and rollback");
            FlushArenaCache(_kCache);
            _specMetadata = null; // a partial capture must never restore an older window
            _specRecurrentRestored = false;
            if (!SpeculationProfitable || !EnsureGdnArgs() || !EnsureAttnArgs() || !EnsurePleArgs())
                throw new NotSupportedException("qwen4exp: recurrent snapshots require the complete token-span state family.");
            var metadata = CaptureSpecMetadata();
            if (_specNativeSnapshot == IntPtr.Zero || !ReferenceEquals(_specSnapshotOwner, ActiveSpecOwner)
                || !ReferenceEquals(_specSnapshotAttnArgs, _attnArgs)
                || !ReferenceEquals(_specSnapshotGdnArgs, _gdnArgs)
                || !ReferenceEquals(_specSnapshotPleArgs, _pleArgs))
            {
                ReleaseSpecSnapshot();
                var keys = new List<IntPtr>();
                var devices = new List<int>();
                for (int layer = 0; layer < Config.NumLayers; ++layer)
                {
                    if (!_isRecurrent[layer]) continue;
                    keys.Add((IntPtr)GetFloatPtr(_gdnConvStateT[layer]));
                    devices.Add(DeviceForLayer(layer));
                }
                if (_pleConvState != null && _pleConvState.Length != 0)
                {
                    keys.Add(Marshal.UnsafeAddrOfPinnedArrayElement(_pleConvState, 0));
                    devices.Add(DeviceForLayer(_pleLayerIndex));
                }
                if (IsTensorParallel)
                {
                    // Replicated recurrence runs on every rank. Rollback must restore
                    // every copy before the next FFN collective joins their outputs.
                    int singleRankCount = keys.Count;
                    for (int rank = 1; rank < TpDegree; ++rank)
                        for (int i = 0; i < singleRankCount; ++i)
                        { keys.Add(keys[i]); devices.Add(rank); }
                }
                fixed (Qwen4ExpAttnArgs* attn = _attnArgs)
                fixed (Qwen4ExpGdnArgs* gdn = _gdnArgs)
                fixed (Qwen4ExpPleArgs* ple = _pleArgs)
                    _specNativeSnapshot = GgmlBasicOps.Qwen4ExpStateSnapshotCreate(
                        keys.ToArray(), devices.ToArray(), (IntPtr)attn, (IntPtr)gdn, (IntPtr)ple);
                if (_specNativeSnapshot == IntPtr.Zero)
                    throw new InvalidOperationException("qwen4exp: failed to allocate a recurrent snapshot.");
                _specSnapshotOwner = ActiveSpecOwner;
                _specSnapshotAttnArgs = _attnArgs;
                _specSnapshotGdnArgs = _gdnArgs;
                _specSnapshotPleArgs = _pleArgs;
            }
            if (!GgmlBasicOps.Qwen4ExpStateSnapshotCapture(_specNativeSnapshot))
            {
                _specStateFailed = true;
                throw new InvalidOperationException("qwen4exp: recurrent snapshot capture failed.");
            }
            _specMetadata = metadata;
            _specCaptureArmed = true;
            _specRowCommitted = false;
        }

        /// <summary><paramref name="verifyRows"/> counts the drafts verified and
        /// <paramref name="acceptedRows"/> the drafts accepted; the span ran the anchor
        /// plus the drafts, so the state to keep is the one after row
        /// <paramref name="acceptedRows"/> of <paramref name="verifyRows"/> + 1.</summary>
        public void SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            _specRowCommitted = false;
            int captured = _specCapturedRows;
            _specCapturedRows = 0;
            // Full acceptance keeps the verify's final state; nothing to commit.
            if (captured != verifyRows + 1 || verifyRows < 1 || acceptedRows < 0 || acceptedRows >= verifyRows
                || _specMetadata == null || _specVerifyTokens == null)
                return;
            if (!GgmlBasicOps.Qwen4ExpCommitVerifyRow(acceptedRows, captured))
                return;                       // live state untouched: the restore path still applies
            // Host metadata back to the pre-verify state, then advanced over the kept
            // rows exactly as their forward would have (the PLE n-gram history).
            bool deviceAuthoritative = _deviceStateAuthoritative;
            RestoreSpecMetadata(_specMetadata);
            _deviceStateAuthoritative = deviceAuthoritative;
            if (_pleHeads > 0)
                ComputePleRows(_specVerifyTokens.AsSpan(0, acceptedRows + 1).ToArray(), _specVerifyStart);
            _specCommitPosition = _specVerifyStart + acceptedRows + 1;
            _specRowCommitted = true;
        }

        public void SpecRestoreRecurrentState()
        {
            RefuseUnmirroredQwenOperation("speculative execution and rollback");
            FlushArenaCache(_kCache);
            ValidateSpecMetadata(_specMetadata);
            // Reserve before native ownership changes; metadata publication after
            // a successful restore cannot fail from growing the history list.
            _pleHistory ??= new List<int>();
            _pleHistory.EnsureCapacity(_specMetadata.PleHistory.Length);
            if (!GgmlBasicOps.Qwen4ExpStateSnapshotRestore(_specNativeSnapshot))
            {
                _specMetadata = null;
                _specStateFailed = true;
                throw new InvalidOperationException("qwen4exp: recurrent restore failed; reset the conversation before reuse.");
            }
            RestoreSpecMetadata(_specMetadata);
            _specRecurrentRestored = true;
        }

        public void SpecRewindCache(int length)
        {
            RefuseUnmirroredQwenOperation("speculative execution and rollback");
            ValidateSpecMetadata(_specMetadata);
            if (_specRowCommitted)
            {
                if (length != _specCommitPosition || length > _cacheSeqLen)
                    throw new InvalidOperationException("qwen4exp: a committed verify row must be kept exactly.");
                _cacheSeqLen = length;
                if (HasQsa) _qsaPositionCount = length;
                _specRowCommitted = false;
                return;
            }
            if (!_specRecurrentRestored || length != _specMetadata.Position || length > _cacheSeqLen)
                throw new InvalidOperationException("qwen4exp: recurrent rewind must return to the captured position, then replay accepted tokens.");
            _cacheSeqLen = length;
            if (HasQsa) _qsaPositionCount = length;
            _specRecurrentRestored = false;
        }

        private void ReleaseSpecSnapshot()
        {
            _specMetadata = null;
            _specRecurrentRestored = false;
            if (_specNativeSnapshot != IntPtr.Zero)
            {
                GgmlBasicOps.Qwen4ExpStateSnapshotFree(_specNativeSnapshot);
                _specNativeSnapshot = IntPtr.Zero;
            }
            _specSnapshotOwner = null;
            _specSnapshotAttnArgs = _specSnapshotGdnArgs = _specSnapshotPleArgs = null;
        }
    }
}
