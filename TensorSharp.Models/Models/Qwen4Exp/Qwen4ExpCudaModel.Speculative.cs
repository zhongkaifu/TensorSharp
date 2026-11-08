// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next on the direct CUDA engine as a speculative target: a verify window runs as one ubatch with every
// row's logits (Q4eCudaEngine.ForwardVerify), and the accepted prefix is committed by replaying the recurrences over
// it (CommitVerify), so the speculator needs no replay forward. The drafter reads the trunk's hidden streams before
// the output mixer (SpecFeatureSize = hc * E, as on the GGML backends): the shared-target MTP block of --draft-model
// runs on the engine (Q4eCudaEngine.Mtp.cs), and weight-free drafters (n-gram, prompt lookup) need nothing.
using System;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public sealed partial class Qwen4ExpCudaModel : IDraftHead
    {
        private bool _specRowCommitted, _specRestored;
        private int _specCommitPosition = -1, _specWindowStart = -1;

        /// <summary>The trunk's hidden streams before the output mixer: what the MTP block joins with each token.</summary>
        public int SpecFeatureSize => _exec?.HiddenWidth ?? Config.HiddenSize;
        // As on the GGML backends: the verify replays the accepted prefix of every recurrence, so a short window keeps
        // the rejected tail cheap.
        int ISpeculativeTarget.SpecPreferredDraftWindow => 3;
        // As on the GGML backends (Qwen4ExpModel.SpecPreferredDraftGate): on the RTX 5090 with the adaptive expert cache,
        // a 0.5 gate decoded long answers at 136-163 tok/s against 122-155 with the per-token default (0.15).
        float ISpeculativeTarget.SpecPreferredDraftGate => 0.5f;
        int ISpeculativeTarget.SpecMaxDraftTokens => TensorSharp.Cuda.Q4eCudaEngine.VerifyMaxRows - 1;
        int ISpeculativeTarget.SpecSoloMaxDraftTokens => TensorSharp.Cuda.Q4eCudaEngine.VerifyMaxRows - 1;
        bool ISpeculativeTarget.SpecVerifyPersistsAcceptedKv => _specRowCommitted;

        void ISpeculativeTarget.SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            lock (_sync)
            {
                int n = tokens.Length, vocab = Config.VocabSize;
                int window = TensorSharp.Cuda.Q4eCudaEngine.VerifyMaxRows;
                if (n > window && allLogitsRows)
                    throw new NotSupportedException(
                        $"qwen4exp on --backend cuda verifies at most {window} rows with every row's logits; the speculator asked for {n}.");
                if (n > window)
                {
                    // A prompt chunk: the plain forward, last row's logits, and the rows' hidden streams (every row's,
                    // or the last row's when hAllOut holds one).
                    if (_logitsBuffer == null || _logitsBuffer.Length != vocab)
                        _logitsBuffer = new float[vocab];
                    _exec.Forward(tokens, _logitsBuffer, hAllOut);
                    Array.Copy(_logitsBuffer, logitsOut, vocab);
                    _cacheSeqLen = _exec.NPast;
                    return;
                }
                _specWindowStart = _exec.NPast;
                float[] all = allLogitsRows ? logitsOut : new float[(long)n * vocab];
                _exec.ForwardVerify(tokens, all, hAllOut);
                if (!allLogitsRows)
                    Array.Copy(all, (long)(n - 1) * vocab, logitsOut, 0, vocab);
                _cacheSeqLen = _exec.NPast;
            }
        }

        void ISpeculativeTarget.SpecEnsureCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen > MaxContextLength)
                throw new InvalidOperationException($"qwen4exp: a speculative window needs {requiredSeqLen} positions; the context holds {MaxContextLength}.");
        }

        /// <summary>The verify keeps its own copy of the recurrent states (Q4eCudaEngine.Verify.cs).</summary>
        void ISpeculativeTarget.SpecSnapshotRecurrentState()
        {
            _specRowCommitted = false;
            _specRestored = false;
        }

        void ISpeculativeTarget.SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            lock (_sync)
            {
                _specRowCommitted = false;
                if (_specWindowStart < 0 || acceptedRows < 0 || acceptedRows > verifyRows)
                    return;
                // The window is the last committed token and verifyRows drafts: keep it and the accepted drafts.
                if (!_exec.CommitVerify(acceptedRows + 1))
                    return;
                _specCommitPosition = _specWindowStart + acceptedRows + 1;
                _cacheSeqLen = _exec.NPast;
                _specRowCommitted = true;
            }
        }

        void ISpeculativeTarget.SpecRestoreRecurrentState()
        {
            lock (_sync)
            {
                if (!_exec.RestoreVerify())
                    throw new InvalidOperationException("qwen4exp: no verify window to restore on --backend cuda.");
                _cacheSeqLen = _exec.NPast;
                _specRestored = true;
            }
        }

        void ISpeculativeTarget.SpecRewindCache(int length)
        {
            lock (_sync)
            {
                if (_specRowCommitted)
                {
                    if (length != _specCommitPosition)
                        throw new InvalidOperationException("qwen4exp: a committed verify row must be kept exactly.");
                    _specRowCommitted = false;
                    return;
                }
                if (!_specRestored || length != _exec.NPast)
                    throw new InvalidOperationException("qwen4exp: a rewind on --backend cuda returns to the restored window start only.");
                _specRestored = false;
                _cacheSeqLen = length;
            }
        }

        // ---- the MTP draft block (--draft-model) ----

        public bool HasDraftHead => _exec?.HasMtp == true;
        public DraftHeadKind DraftHeadKind => HasDraftHead ? DraftHeadKind.PerToken : DraftHeadKind.None;
        public bool SupportsFusedCatchUpStep => HasDraftHead;
        // The block keeps its own KV: it cannot draft after positions it never replayed.
        public bool DraftHeadResumesAfterGap => false;

        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => RunMtp(new[] { token }, hPrev, pos, logitsOut, hOut);

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
            => RunMtp(tokens, hRows, startPos, null, null);

        public void DraftCatchUpAndStep(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
            => RunMtp(tokens, hRows, startPos, logitsOut, hOut);

        private void RunMtp(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (hRows == null)
                throw new ArgumentNullException(nameof(hRows), "the MTP block needs the trunk's hidden streams of every row it replays");
            if (tokens.Length == 0)
                return;
            lock (_sync)
                _exec.MtpForward(tokens, hRows, startPos, logitsOut, hOut);
        }
    }
}
