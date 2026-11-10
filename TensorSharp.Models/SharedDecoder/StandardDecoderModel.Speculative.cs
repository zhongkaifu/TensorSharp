// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    /// <summary>
    /// Speculative decoding for every family on the runtime, once. The trunk side is a step
    /// graph over the request's slot that returns every row's logits and the hidden rows a
    /// drafter reads (the residuals entering DFlash's target layers, or the final-normed rows);
    /// a verify window of a hybrid stack keeps its recurrent state before and after each row, so
    /// the accept count selects the accepted prefix's state with no second forward and attention
    /// needs only a position rewind. Drafting is <see cref="ModelBase"/>'s generic DFlash /
    /// DFlash2 / DSpine drafter; a family adds nothing but its recurrent snapshot commit.
    /// </summary>
    public abstract partial class StandardDecoderModel : ISpeculativeModel
    {
        private int _verifyRows;   // a verify whose recurrent state awaits its accept count

        /// <summary>A family's cached graphs that bind the slots' buffers (dropped with the step graphs).</summary>
        protected virtual void OnGraphCacheCleared() { }

        /// <summary>A verify (or a prompt chunk through SpecForward) left the active request's recurrent state
        /// awaiting an accept count: settle it as fully accepted before anything else uses that state.</summary>
        private void SettlePendingVerify()
        {
            if (_verifyRows <= 1) return;
            int rows = _verifyRows;
            _verifyRows = 0;
            CommitRecurrentPrefix(rows, rows);
        }

        /// <summary>Set the active request's recurrent states to those after <paramref name="kept"/>
        /// rows of the last <paramref name="rows"/>-row verify (0: before it). Hybrid stacks only.</summary>
        protected virtual void CommitRecurrentPrefix(int kept, int rows)
            => throw new NotSupportedException($"{Config.Architecture} keeps no recurrent snapshots.");

        /// <summary>The active request's recurrent state of (layer, rank 0).</summary>
        protected RecurrentState ActiveRecurrentState(int layer) => _recurrentPrimary?[layer]?[0];

        /// <summary>A family moved buffers a cached graph binds: drop the cached graphs.</summary>
        protected void NoteGraphBindingsChanged() => ClearGraphCache();

        /// <summary>Attach a DFlash / DFlash2 / DSpine drafter after the trunk loaded.</summary>
        protected void AttachDraftModel(string draftModelPath)
        {
            if (string.IsNullOrEmpty(draftModelPath)) return;
            if (GraphUnsupportedReason() is { } reason)
            {
                Console.WriteLine($"  Draft model '{System.IO.Path.GetFileName(draftModelPath)}' not attached: speculative decoding runs on whole-forward graphs ({reason}).");
                return;
            }
            LoadDFlashDraftWeights(draftModelPath);
        }

        // ---------------------------------------------------------------- the trunk

        /// <summary>Speculation verifies on whole-forward graphs; elsewhere the engine decodes plainly.</summary>
        public string SpeculationRefusal => GraphUnsupportedReason() is { } reason
            ? $"speculative decoding runs on whole-forward graphs ({reason})"
            : null;

        public bool SpeculationProfitable => SpeculationRefusal == null;

        public int SpecFeatureSize => HasDFlash ? DFlashSettings.FeatureSize : Config.HiddenSize;

        public int SpecPrefillChunkSize => HasDFlash ? DFlashPrefillChunkSize : 0;

        /// <summary>The bound slot is the active cache, so a verify runs on whichever request is bound.</summary>
        public bool SpecTrunkFollowsBoundCache => true;

        /// <summary>A verify writes every row's K/V and keeps every row's recurrent state.</summary>
        public bool SpecVerifyPersistsAcceptedKv => true;

        public void SpecEnsureCapacity(int requiredSeqLen) => EnsureCacheCapacity(requiredSeqLen);

        /// <summary>
        /// Draft window for a hybrid stack, where a verify carries per-row recurrent states. With a
        /// block drafter: 5 on ggml_metal, where an 8-row verify costs about one row and the 0.35
        /// gate tuned on sparse-MoE trunks cost speed (M5 Pro, Qwen3.8-27B UD-IQ3_XXS + DFlash2,
        /// 1,224 greedy tokens: 5 / no gate 28.9 tok/s, 7 / 0 28.5, 7 / 0.35 24.8, 3 / 0.35 26.1,
        /// plain 20.5); 7 on ggml_cuda, the most rows its matvec kernel serves (RTX Spark N1X,
        /// coding workload: 7 beat 5 and 3 on every turn); 3 elsewhere and for other drafters.
        /// Attention-only stacks take the speculator's own default.
        /// </summary>
        public int SpecPreferredDraftWindow => !_anyRecurrent ? 0
            : !HasDFlash ? 3
            : _backend == BackendType.GgmlCuda ? 7
            : _backend == BackendType.GgmlMetal ? 5 : 3;

        /// <summary>No confidence gate for a block drafter on a hybrid stack on ggml_metal (see
        /// <see cref="SpecPreferredDraftWindow"/>).</summary>
        public float SpecPreferredBlockGate => _anyRecurrent && HasDFlash && _backend == BackendType.GgmlMetal ? 0f : float.NaN;

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            SettlePendingVerify();
            int rows = tokens.Length, start = _cacheSeqLen, hiddenSize = Config.HiddenSize;
            EnsureCacheCapacity(checked(start + rows));
            bool dflash = HasDFlash;
            var request = new GraphRequest
            {
                AllRowLogits = allLogitsRows,
                LastRowLogits = !allLogitsRows,
                CaptureLayers = dflash && hAllOut != null ? DFlashSettings.TargetLayerIds : Array.Empty<int>(),
                CaptureFinal = !dflash && hAllOut != null,
                RecurrentSnapshots = _anyRecurrent && rows > 1 && rows <= SmallChunkRows,
            };
            Tensor hidden = Embedding(tokens);
            GraphResult result;
            _forwardSw.Start();
            try
            {
                if (!TryRunGraphStep(hidden, new[] { SnapshotActive() }, new[] { 0 }, new[] { rows }, new[] { start }, request, out result))
                    throw new NotSupportedException(SpeculationRefusal ?? $"{Config.Architecture}: this verify cannot run as a graph.");
            }
            finally
            {
                hidden.Dispose();
                _forwardSw.Stop();
            }
            _cacheDeviceDirty = true;
            _cacheSeqLen += rows;
            _verifyRows = request.RecurrentSnapshots ? rows : 0;

            int vocab = Config.VocabSize;
            long logitFloats = allLogitsRows ? (long)rows * vocab : vocab;
            Array.Copy(result.Logits, 0, logitsOut, 0, Math.Min(logitFloats, logitsOut.LongLength));

            if (hAllOut == null || hAllOut.LongLength == 0) return;
            if (dflash)
            {
                // Row-major [rows][target layers][hidden]; a buffer too small for every row takes the last.
                int feat = DFlashSettings.FeatureSize;
                bool all = hAllOut.LongLength >= (long)rows * feat;
                if (!all && hAllOut.LongLength < feat)
                    throw new ArgumentException($"DFlash hidden capture buffer holds {hAllOut.LongLength} floats; one feature row needs {feat}.", nameof(hAllOut));
                for (int c = 0; c < result.Captures.Length; c++)
                {
                    float[] layer = result.Captures[c];
                    if (all)
                        for (int r = 0; r < rows; r++)
                            Array.Copy(layer, (long)r * hiddenSize, hAllOut, (long)r * feat + (long)c * hiddenSize, hiddenSize);
                    else
                        Array.Copy(layer, (long)(rows - 1) * hiddenSize, hAllOut, (long)c * hiddenSize, hiddenSize);
                }
            }
            else
            {
                long floats = Math.Min(hAllOut.LongLength, (long)rows * hiddenSize);
                if (floats >= (long)rows * hiddenSize)
                    Array.Copy(result.Final, 0, hAllOut, 0, floats);
                else
                    Array.Copy(result.Final, (long)(rows - 1) * hiddenSize, hAllOut, 0, hiddenSize);
            }
        }

        /// <summary>The verify keeps its own snapshots; nothing to take before it.</summary>
        public void SpecSnapshotRecurrentState() { }

        public void SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            if (_verifyRows <= 1) return;
            int rows = _verifyRows;
            int kept = acceptedRows + 1;
            if (kept < 1 || kept > rows)
                throw new InvalidOperationException($"Accepted {acceptedRows} rows of a {rows}-row verify.");
            _verifyRows = 0;
            CommitRecurrentPrefix(kept, rows);
        }

        /// <summary>Back to the state before the last verify: until its commit the live state is that state.</summary>
        public void SpecRestoreRecurrentState()
        {
            if (_verifyRows > 1)
            {
                _verifyRows = 0;
                return;
            }
            if (_anyRecurrent)
                throw new InvalidOperationException($"{Config.Architecture}: the verify was already committed; its pre-window state is gone.");
        }

        /// <summary>Attention rows past <paramref name="length"/> are dead (overwritten by the next writes).</summary>
        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(length), $"rewind to {length} of {_cacheSeqLen}");
            _cacheSeqLen = length;
        }

        /// <summary>With a drafter attached, also warm each verify width (and its commit), so the first
        /// request does not compile the kernels and build the graphs a verify of each width needs.</summary>
        public override void WarmUpKernels()
        {
            base.WarmUpKernels();
            if (!HasDFlash || SpeculationRefusal != null) return;
            int widest = Math.Min(SmallChunkRows, Math.Max(2, Math.Max(SpecPreferredDraftWindow, 1) + 1));
            int safeToken = Config.VocabSize > 1 ? 1 : 0;
            var started = System.Diagnostics.Stopwatch.StartNew();
            var logits = new float[(long)widest * Config.VocabSize];
            var features = new float[(long)widest * SpecFeatureSize];
            for (int rows = 2; rows <= widest; rows++)
            {
                ResetKVCache();
                SpecForward(new[] { safeToken }, null, logits, allLogitsRows: false);
                var window = new int[rows];
                Array.Fill(window, safeToken);
                SpecForward(window, features, logits, allLogitsRows: true);
                SpecOnVerifyAccepted(rows - 2, rows);
            }
            ResetKVCache();
            Console.WriteLine($"    Verify warmup (2-{widest} rows): {started.Elapsed.TotalMilliseconds:F1} ms");
        }

        // ---------------------------------------------------------------- the draft head

        public DraftHeadKind DraftHeadKind => HasDFlash ? DraftHeadKind.Block : DraftHeadKind.None;

        public int DraftBlockSize => HasDFlash ? DFlashSettings.MaxDraftTokens : 0;

        public bool DraftBlockIsCausal => HasDFlash && DFlashSettings.CausalBlock;

        public int DraftContextRows => HasDFlash && DFlashRingRows > 0 ? DFlashRingRows : int.MaxValue;

        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => DFlashPropose(lastToken, hPrev, position, draftOut, confOut);

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (!HasDFlash) throw new InvalidOperationException($"{Config.Architecture} has no draft head.");
            DFlashCommit(tokens, hRows, startPos);
        }
    }
}
