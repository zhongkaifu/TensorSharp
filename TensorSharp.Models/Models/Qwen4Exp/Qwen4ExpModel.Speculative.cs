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
        // The drafter consumes all HC streams BEFORE the final mixer. Collapsing
        // this to HiddenSize loses information required by the trained head.
        public int SpecFeatureSize => _hcDim;
        public int SpecPreferredDraftWindow => 3;
        public bool SpecVerifyPersistsAcceptedKv => false;
        public bool SpecPlainStepUsesForward => true;
        public bool SpecTrunkFollowsBoundCache => true;
        public bool SpeculationProfitable => !_specStateFailed && IsGgmlBackend && _tokenGraphEnabled
            && !_tokenGraphUnsupported && _spanAttnEnabled && !_fusedGateUpExperts
            && _fusedFfnEnabled && !_fusedFfnUnsupported
            && _fusedGdnEnabled && !_fusedGdnUnsupported
            && _fusedAttnEnabled && !_fusedAttnUnsupported
            && _gdnMaxLayers < 0 && !_gdnVerify
            && (_specApiAvailable ??= GgmlBasicOps.Qwen4ExpSpecApiAvailable())
            && (!HasQsa || (_qsaApiAvailable ??= GgmlBasicOps.Qwen4ExpQsaApiAvailable()));

        private bool _specForwardActive;
        private bool _specAllLogitsRows;
        private IntPtr _specHiddenOutput;
        private IntPtr _specLogitsOutput;
        private IntPtr _specNativeSnapshot;
        private object _specSnapshotOwner;
        private object _specSnapshotAttnArgs, _specSnapshotGdnArgs, _specSnapshotPleArgs;
        private SpecMetadata _specMetadata;
        private int _specResetVersion;
        private bool? _specApiAvailable;
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

        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(logitsOut);
            if (tokens.Length == 0 || logitsOut.Length < checked(Config.VocabSize * (allLogitsRows ? tokens.Length : 1))
                || (hAllOut != null && hAllOut.Length < checked(tokens.Length * SpecFeatureSize)))
                throw new ArgumentException("qwen4exp: speculative output buffers do not cover the requested rows.");
            if (!SpeculationProfitable || !EnsureHeadArgs() || !EnsurePleArgs())
                throw new NotSupportedException("qwen4exp speculation requires the complete GGML token-span path, including PLE and the output head.");
            if (_specForwardActive)
                throw new InvalidOperationException("qwen4exp: a speculative forward is already active.");
            MtpPositionRange mtpPositions = null;
            object mtpOwner = null;
            if (HasDraftHead)
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
                _specAllLogitsRows = allLogitsRows;
                _specHiddenOutput = (IntPtr)hidden;
                _specLogitsOutput = (IntPtr)logits;
                try
                {
                    Forward(tokens);
                    if (mtpPositions != null) PublishMtpPositions(_mtpPositions[mtpOwner], mtpPositions);
                }
                catch { _specStateFailed = true; throw; }
                finally
                {
                    _specForwardActive = false;
                    _specAllLogitsRows = false;
                    _specHiddenOutput = IntPtr.Zero;
                    _specLogitsOutput = IntPtr.Zero;
                }
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen < _cacheSeqLen || requiredSeqLen > MaxContextLength)
                throw new ArgumentOutOfRangeException(nameof(requiredSeqLen));
            EnsureCacheCapacity(requiredSeqLen);
            if (HasDraftHead) EnsureMtpState(requiredSeqLen);
        }

        public unsafe void SpecSnapshotRecurrentState()
        {
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
        }

        public void SpecRestoreRecurrentState()
        {
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
            ValidateSpecMetadata(_specMetadata);
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
