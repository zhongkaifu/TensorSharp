// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The trunk backend a speculative execution drives: prompt/verify/plain
    /// forwards plus recurrent-state snapshot/rollback, implemented by
    /// <see cref="LinearSpecTrunk"/> over the model's live linear cache (the
    /// standalone decoder and the per-sequence engine route).
    ///
    /// The draft/verify loop only ever asks for "forward these tokens" and
    /// "undo back to here".
    /// </summary>
    public interface ISpecTrunk
    {
        /// <summary>Forward <paramref name="tokens"/> at the trunk's current
        /// position, capturing per-row hidden states and logits like
        /// <see cref="ISpeculativeTarget.SpecForward"/>. Advances the trunk
        /// by <c>tokens.Length</c>.</summary>
        void Forward(int[] tokens, float[]? hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Optional plain-greedy verify, with the same state effects as
        /// <see cref="Forward"/>. False means nothing was forwarded or written.</summary>
        bool TryForwardGreedyVerify(int[] tokens, float[]? hAllOut, int[] drawnOut) => false;

        /// <summary>
        /// One plain single-token step for a speculator that needs no hidden state:
        /// forward <paramref name="token"/> at the trunk's position, fill the
        /// next-token logits, advance by one. The default is the generic
        /// <see cref="Forward"/>; a trunk whose ordinary decode is cheaper than its
        /// speculative forward (a fused decode kernel with the LM head folded in,
        /// against a per-op norm + head + softcap tail) overrides it, because a
        /// PARKED speculator runs nothing but plain steps and was paying that
        /// tail on every one of them: measured 14% (Gemma 4 E4B) to 19% (E2B)
        /// slower than the non-speculative decode while drafting nothing.
        /// </summary>
        /// <param name="parked">True when the governor has parked speculation, so
        /// this is one of a long run of plain steps; a trunk whose cheap plain step
        /// costs a state-family switch (<see cref="ISpeculativeTarget.SpecPlainStepCostsFamilySwitch"/>)
        /// takes it only then.</param>
        void ForwardPlain(int token, float[] logitsOut, bool parked)
            => Forward(new[] { token }, null, logitsOut, allLogitsRows: false);

        /// <summary>True when <see cref="ForwardPlain"/> is genuinely cheaper than a
        /// one-row <see cref="Forward"/> on this trunk (the model's own decode).</summary>
        bool HasCheapPlainStep => false;

        /// <summary>Snapshot recurrent state before a verify batch.</summary>
        void SnapshotRecurrentState();

        /// <summary>
        /// The accept count for the verify that just ran, handed to the trunk
        /// BEFORE any rollback decision and on EVERY step - full acceptance
        /// included. A trunk that defers part of its post-verify bookkeeping until
        /// it knows how much was accepted settles it here; Qwen 3.5/3.8 uses it to
        /// pick the recurrent-state snapshot for the accepted prefix, which is what
        /// lets <see cref="TryCommitVerifiedPrefix"/> then succeed on a recurrent
        /// model at all. Default: nothing to do.
        /// </summary>
        void OnVerifyAccepted(int acceptedRows, int verifyRows) { }

        /// <summary>Roll the trunk back to <paramref name="position"/>
        /// committed tokens: restore the recurrent snapshot and rewind any
        /// attention-KV bookkeeping.</summary>
        void Rollback(int position);

        /// <summary>
        /// Fast partial-acceptance commit: when the trunk's verify already wrote
        /// reusable KV for the accepted prefix (no recurrent state to replay), keep
        /// those writes and just set the live position to <paramref name="newPosition"/>
        /// (= committed + accepted + 1), skipping the redundant kept-prefix
        /// re-forward. Returns false when the trunk cannot do this (caller falls back
        /// to <see cref="Rollback"/> + re-forward). Default: not supported.
        /// </summary>
        bool TryCommitVerifiedPrefix(int newPosition) => false;
    }

    /// <summary>Linear-cache trunk: forwards through
    /// <see cref="ISpeculativeTarget.SpecForward"/> on the model's single
    /// live KV cache.</summary>
    public sealed class LinearSpecTrunk : ISpecTrunk
    {
        private readonly ISpeculativeTarget _model;

        public LinearSpecTrunk(ISpeculativeTarget model)
            => _model = model ?? throw new ArgumentNullException(nameof(model));

        public void Forward(int[] tokens, float[]? hAllOut, float[] logitsOut, bool allLogitsRows)
            => _model.SpecForward(tokens, hAllOut, logitsOut, allLogitsRows);

        public bool TryForwardGreedyVerify(int[] tokens, float[]? hAllOut, int[] drawnOut)
            => _model.TrySpecForwardGreedyVerify(tokens, hAllOut, drawnOut);

        private readonly int[] _plainToken = new int[1];

        /// <summary>The model's own decode step - the same fused kernel the
        /// non-speculative engine path runs - which on the linear cache advances
        /// exactly like <see cref="ISpeculativeTarget.SpecForward"/> does.</summary>
        public bool HasCheapPlainStep => _model.SpecPlainStepUsesForward;

        public void ForwardPlain(int token, float[] logitsOut, bool parked)
        {
            _plainToken[0] = token;
            if (!_model.SpecPlainStepUsesForward || (!parked && _model.SpecPlainStepCostsFamilySwitch))
            {
                _model.SpecForward(_plainToken, null, logitsOut, allLogitsRows: false);
                return;
            }
            float[] logits = _model.Forward(_plainToken);
            Array.Copy(logits, logitsOut, Math.Min(logits.Length, logitsOut.Length));
        }

        public void SnapshotRecurrentState() => _model.SpecSnapshotRecurrentState();

        public void OnVerifyAccepted(int acceptedRows, int verifyRows)
            => _model.SpecOnVerifyAccepted(acceptedRows, verifyRows);

        public void Rollback(int position)
        {
            _model.SpecRestoreRecurrentState();
            _model.SpecRewindCache(position);
        }

        public bool TryCommitVerifiedPrefix(int newPosition)
        {
            if (!_model.SpecVerifyPersistsAcceptedKv)
                return false;
            // The verify wrote correct KV for all batch tokens; the accepted prefix's
            // KV is already live. Just drop the rejected tail by rewinding the position
            // (rejected slots are overwritten by later writes and never read past the
            // live position). No recurrent state to restore for such models.
            _model.SpecRewindCache(newPosition);
            return true;
        }
    }
}
