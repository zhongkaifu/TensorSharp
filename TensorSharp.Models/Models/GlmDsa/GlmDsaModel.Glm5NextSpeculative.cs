// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Speculative decoding on GLM-5.3-Flash (glm5next): the recurrent-state
// rollback that KDA needs.
//
// GLM-5.2's speculative path is a position rewind - MLA rows and indexer keys
// are per-position, so dropping a rejected tail is exact and free. glm5next's
// 34 KDA layers each carry a recurrent state (the short-conv tail over q|k|v
// and the per-head delta-net state) that a verify batch advances by the WHOLE
// window, and no position arithmetic brings it back. So on this architecture
// the trunk honours the recurrent contract instead (the one Qwen 3.5's
// GatedDeltaNet and Qwen 3.8's trunk use, SpecVerifyPersistsAcceptedKv=false):
//
//   1. SpecSnapshotRecurrentState  - before the verify, copy every KDA state
//                                    and remember the position P;
//   2. SpecForward([t, d1..dK])    - the verify, advancing state and position
//                                    to P+K+1;
//   3. on a partial rejection, SpecRestoreRecurrentState + SpecRewindCache(P),
//      after which the caller re-forwards the accepted prefix, so the state
//      equals a plain decode of exactly those tokens.
//
// The hyper-connection streams are per-token (they are the residual stream,
// not a carried state), the MLA rows are per-position, and the pooled indexer
// keys are per-cell, so the KDA state is the only thing a rollback has to
// restore; the rest is rewritten by the re-forward before anything reads it.
//
// On the managed per-op path the state is host arrays and the snapshot is a
// copy. On the native executor it lives per slot on the device that owns each
// layer, and the copy is device-to-device inside ggml_ops_glm_dsa.cpp
// (TSGgml_GlmKdaStateCapture / Restore); the managed side only keeps the
// position and the "restored" flag that make SpecRewindCache exact.
//
// GLM-5.3-Flash's own NextN block is NOT built (llama.cpp asserts that graph
// unimplemented too, and the native loader prints a notice), so what this
// enables is the weight-free n-gram speculator: `--spec --spec-type ngram`.
using System;
using TensorSharp.Core;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>The checkpoint architecture is glm5next. Unlike
        /// <see cref="IsGlm5Next"/> (set by the managed config parser, which the
        /// native constructor returns before), this is true on BOTH execution
        /// paths.</summary>
        private bool IsGlm5NextArch => string.Equals(Config.Architecture, "glm5next", StringComparison.Ordinal);

        // Managed-path snapshot: one copy per recurrent layer, plus the ring
        // write index and the position they belong to.
        private float[][] _g5nSnapConv;
        private float[][] _g5nSnapSsm;
        private int[] _g5nSnapConvWrite;

        /// <summary>Position the live snapshot was taken at; -1 when none.</summary>
        private int _g5nSnapPos = -1;
        private bool _g5nSnapValid;
        /// <summary>Set by a restore, cleared by the rewind that follows it: the
        /// only moment a rewind below the live position is exact.</summary>
        private bool _g5nSnapRestored;

        private bool? _g5nNativeSnapshotApi;

        /// <summary>
        /// Whether this glm5next instance can roll a verify batch back. The
        /// managed path always can; the native executor needs a library that
        /// exports the KDA snapshot API (an older libGgmlOps loads and forwards
        /// glm5next fine but cannot undo a rejected window, so speculation is
        /// declined up front rather than failing mid-verify).
        /// </summary>
        private bool Glm5NextRollbackAvailable
            => !UsesNativeExecutor || (_g5nNativeSnapshotApi ??= GgmlGlmNative.KdaStateApiAvailable());

        private void Glm5NextInvalidateSnapshot()
        {
            _g5nSnapValid = false;
            _g5nSnapRestored = false;
            _g5nSnapPos = -1;
        }

        private void Glm5NextSnapshotRecurrentState()
        {
            Glm5NextInvalidateSnapshot();
            if (!Glm5NextRollbackAvailable)
                throw new NotSupportedException(
                    "GLM-5.3-Flash speculative decoding needs the native KDA snapshot API " +
                    "(TSGgml_GlmKdaStateCapture), which this libGgmlOps does not export. Rebuild the native library.");

            if (UsesNativeExecutor)
            {
                lock (_nativeSync)
                {
                    if (!GgmlGlmNative.KdaStateCapture(_native))
                        throw new InvalidOperationException("glm5next: native KDA state capture failed (see stderr).");
                    _g5nSnapPos = GgmlGlmNative.NPast(_native);
                }
            }
            else
            {
                int layers = _kdaConvState.Length;
                _g5nSnapConv ??= new float[layers][];
                _g5nSnapSsm ??= new float[layers][];
                _g5nSnapConvWrite ??= new int[layers];
                for (int il = 0; il < layers; il++)
                {
                    float[] conv = _kdaConvState[il];
                    float[] ssm = _kdaSsmState[il];
                    if (conv == null || ssm == null)
                        continue;
                    if (_g5nSnapConv[il] == null || _g5nSnapConv[il].Length != conv.Length)
                        _g5nSnapConv[il] = new float[conv.Length];
                    if (_g5nSnapSsm[il] == null || _g5nSnapSsm[il].Length != ssm.Length)
                        _g5nSnapSsm[il] = new float[ssm.Length];
                    Array.Copy(conv, _g5nSnapConv[il], conv.Length);
                    Array.Copy(ssm, _g5nSnapSsm[il], ssm.Length);
                    _g5nSnapConvWrite[il] = _kdaConvWrite[il];
                }
                _g5nSnapPos = _cacheSeqLen;
            }
            _g5nSnapValid = true;
        }

        private void Glm5NextRestoreRecurrentState()
        {
            if (!_g5nSnapValid)
                throw new InvalidOperationException(
                    "glm5next: no KDA recurrent snapshot to restore (SpecSnapshotRecurrentState was not called, " +
                    "or the cache was reset since).");

            if (UsesNativeExecutor)
            {
                lock (_nativeSync)
                {
                    int pos = GgmlGlmNative.KdaStateRestore(_native);
                    if (pos < 0 || pos != _g5nSnapPos)
                    {
                        Glm5NextInvalidateSnapshot();
                        throw new InvalidOperationException(
                            "glm5next: native KDA state restore failed; reset the conversation before reuse (see stderr).");
                    }
                    // The native restore already rewound the slot; mirror it so
                    // CacheSeqLen never disagrees with n_past.
                    _cacheSeqLen = pos;
                }
            }
            else
            {
                for (int il = 0; il < _kdaConvState.Length; il++)
                {
                    if (_g5nSnapConv[il] == null || _kdaConvState[il] == null)
                        continue;
                    Array.Copy(_g5nSnapConv[il], _kdaConvState[il], _g5nSnapConv[il].Length);
                    Array.Copy(_g5nSnapSsm[il], _kdaSsmState[il], _g5nSnapSsm[il].Length);
                    _kdaConvWrite[il] = _g5nSnapConvWrite[il];
                }
            }
            _g5nSnapRestored = true;
        }

        /// <summary>
        /// The KDA recurrence cannot be rewound to an arbitrary position. A
        /// rewind is exact in exactly two cases: it is a no-op (the position is
        /// already there - a fully accepted window, or the native restore that
        /// moved it), or it returns to the position a restored snapshot was
        /// taken at. Anything else is refused loudly: silently keeping the
        /// advanced state would condition every later token on the rejected
        /// tail, which no output check would ever attribute to this line.
        /// </summary>
        private void Glm5NextRewindCache(int length)
        {
            if (length > _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} is past the trunk position {_cacheSeqLen}.");
            if (length == _cacheSeqLen)
                return;
            if (!_g5nSnapValid || !_g5nSnapRestored || length != _g5nSnapPos)
                throw new NotSupportedException(
                    $"glm5next: the KDA recurrent state cannot be rewound from {_cacheSeqLen} to {length}. " +
                    "A rewind is exact only back to the position a restored recurrent snapshot was taken at " +
                    "(SpecSnapshotRecurrentState, then SpecRestoreRecurrentState, then SpecRewindCache); " +
                    "for any other position reset the cache and re-prefill.");

            if (UsesNativeExecutor)
            {
                // The restore already parked n_past at the captured position, so
                // this is the native rewind's no-op case; a refusal here would
                // mean the two sides disagree, which must not pass silently.
                RewindNative(length);
                if (_cacheSeqLen != length)
                    throw new InvalidOperationException(
                        $"glm5next: native rewind to {length} refused (n_past is {_cacheSeqLen}).");
            }
            else
            {
                _cacheSeqLen = length;
            }
            _g5nSnapRestored = false;
        }

        /// <summary>
        /// The managed glm5next trunk forward that a speculative verify needs:
        /// identical to <see cref="ForwardCoreGlm5Next"/> in every state it
        /// leaves behind (same layers, same KDA update, same MLA rows, same
        /// position advance), but it norms EVERY row so the per-token hidden
        /// states can be captured and, when asked, runs the LM head on all of
        /// them. RMS norm and the stream mean are row-wise, so the last row's
        /// logits are bit-identical to the plain forward's.
        /// </summary>
        private void SpecForwardGlm5Next(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            int seqLen = tokens.Length;
            int hidden = Config.HiddenSize;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor embd = Embedding(tokens);
            Trace("inp_embd", -1, embd);
            Tensor streams = Glm5NextToStreams(embd, seqLen);
            embd.Dispose();

            _sharedTopKCount = 0;
            for (int layer = 0; layer < _numTrunkLayers; layer++)
                streams = Glm5NextDecoderBlock(streams, layer, seqLen, startPos);

            if (hAllOut == null && logitsOut == null)
            {
                // A prefill chunk: only the caches and the recurrent state matter.
                streams.Dispose();
                _cacheSeqLen += seqLen;
                return;
            }

            Tensor pooled = Glm5NextHcMean(streams, seqLen);
            streams.Dispose();
            Tensor normed = RMSNormOp(pooled, "output_norm.weight");
            pooled.Dispose();
            Trace("result_norm", -1, normed);

            if (hAllOut != null)
                CopyRowsToBuffer(normed, hAllOut, seqLen, hidden);

            if (logitsOut != null)
            {
                Tensor headIn;
                if (allLogitsRows || seqLen == 1)
                {
                    headIn = normed.CopyRef();
                }
                else
                {
                    using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                    headIn = Ops.NewContiguous(narrowed);
                }
                Tensor logitsTensor = LinearForward(headIn, "output.weight")
                                      ?? LinearForward(headIn, "token_embd.weight");
                headIn.Dispose();
                CopyRowsToBuffer(logitsTensor, logitsOut, allLogitsRows ? seqLen : 1, Config.VocabSize);
                logitsTensor.Dispose();
            }
            normed.Dispose();

            _cacheSeqLen += seqLen;
        }
    }
}
