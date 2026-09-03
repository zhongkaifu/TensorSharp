// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ---------------------------------------------------------------------------
// Nemotron-3.5 Lightning as a DSpark / DFlash speculation TARGET.
//
// The drafter itself is shared - see ModelBase.DFlash*.cs. What lives here is
// the half only this trunk can provide: the per-layer residuals the drafter's
// encoder was trained on (eagle_aux_hidden_state_layer_ids = the 6 trunk layers
// the DSpark module reads), tapped out of a forward that otherwise runs exactly
// like the non-speculative one, plus the rollback half of the speculative
// contract that a hybrid Mamba2 trunk needs: the conv/SSM state snapshot.
//
// The GGUF names those layers in dflash.target_layers (already +1-shifted by
// the converter, so id L means "the residual ENTERING 0-based layer L" == "the
// output of layer L-1"), and the drafter's fc projects the concatenation of
// them - exactly the eagle_aux_hidden_state_layer_ids the NVFP4-DSpark
// checkpoint ships.
//
// The model is recurrent (23 of the 52 layers are Mamba2), so partial
// rejection cannot be a bare position rewind: the verify batch advances the
// conv/SSM state for every row, and the executor's restore-and-re-forward
// rollback needs the pre-verify state (SpecVerifyPersistsAcceptedKv stays
// false, as on Qwen 3.5/3.8's GatedDeltaNet trunk).
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class NemotronModel : ModelBase, ISpeculativeModel
    {
        /// <summary>
        /// Path of a DFlash/DSpark drafter GGUF, if one was configured
        /// (<c>--draft-model</c> / <c>TS_NEMOTRON_DFLASH</c>) AND it really is
        /// one. The same flag also names an MTP-only file for other
        /// architectures, so the architecture string decides rather than the
        /// extension.
        /// </summary>
        internal static string ResolveNemotronDFlashPath(string explicitPath)
        {
            string path = !string.IsNullOrWhiteSpace(explicitPath)
                ? explicitPath
                : Environment.GetEnvironmentVariable("TS_NEMOTRON_DFLASH");
            if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
                return null;
            try
            {
                using var probe = new GgufFile(path);
                string arch = probe.GetString("general.architecture") ?? string.Empty;
                return string.Equals(arch, DFlashConfig.ArchName, StringComparison.Ordinal) ? path : null;
            }
            catch
            {
                // A file we cannot open is not a drafter; the loader below will say
                // so if the operator really meant it.
                return null;
            }
        }

        /// <summary>
        /// Attach a DFlash/DSpark drafter to this trunk. Called from the
        /// constructor after the trunk weights and caches exist, because the
        /// drafter's tensors are merged into the same dictionaries and its ring
        /// is allocated from the same allocator.
        /// </summary>
        private void TryLoadNemotronDFlash(string draftModelPath)
        {
            string path = ResolveNemotronDFlashPath(draftModelPath);
            if (path == null)
                return;

            if (IsTensorParallel)
            {
                // The drafter borrows the trunk's LM head and token embedding,
                // both sharded under TP. Refuse rather than draft from a shard.
                Console.WriteLine("  DFlash/DSpark speculative decoding is not supported under tensor parallelism; ignoring the drafter.");
                return;
            }

            try
            {
                LoadDFlashDraftWeights(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DFlash drafter '{System.IO.Path.GetFileName(path)}' could not be attached: {ex.Message}");
                return;
            }

            Console.WriteLine($"  DSpark drafter attached; a DFlash block drafter will propose tokens for this Nemotron trunk.");
        }

        // ====================================================================
        // IDraftHead - the shared DFlash block machinery, thinly wrapped
        // ====================================================================

        /// <summary>Block (semi-autoregressive) drafter when a DFlash/DSpark
        /// GGUF is attached; nothing otherwise.</summary>
        public DraftHeadKind DraftHeadKind => HasDFlash ? DraftHeadKind.Block : DraftHeadKind.None;

        /// <summary>Tokens a block proposes: block_size - 1 for a plain DFlash
        /// drafter, the whole block width for a Markov (DSpark) drafter.</summary>
        public int DraftBlockSize => HasDFlash ? _dflash.MaxDraftTokens : 0;

        /// <summary>One whole-block draft. See <c>ModelBase.DFlashPropose</c>.</summary>
        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => DFlashPropose(lastToken, hPrev, position, draftOut, confOut);

        /// <summary>Replay committed trunk tokens through the drafter so its KV
        /// ring tracks exact trunk hidden states (llama.cpp's draft-mtp
        /// process() for block drafters).</summary>
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
            => DFlashCommit(tokens, hRows, startPos);

        // ====================================================================
        // ISpeculativeTarget
        // ====================================================================

        /// <summary>One encoder row is the concatenation of the target layers'
        /// residuals (6 x HiddenSize for the Nemotron DSpark module).</summary>
        public int SpecFeatureSize => HasDFlash ? _dflash.FeatureSize : Config.HiddenSize;

        public int SpecPrefillChunkSize => HasDFlash ? DFlashPrefillChunkSize : 0;

        /// <summary>Recurrent trunk: a wide verify batch runs the chunked SSM
        /// scan and a partial rejection restores (then re-advances) the conv/SSM
        /// state, so the marginal draft token costs more than it does on a dense
        /// trunk. 3 is where the DSpark block pays off; the operator can widen
        /// it explicitly with --spec-draft.</summary>
        public int SpecPreferredDraftWindow => HasMamba2Layers ? 3 : 0;

        /// <summary>The op-by-op verify advances the conv/SSM state for all rows
        /// and leaves no per-row snapshots, so partial acceptance must restore
        /// and re-forward the kept prefix.</summary>
        public bool SpecVerifyPersistsAcceptedKv => false;

        private bool HasMamba2Layers
        {
            get
            {
                if (_layerTypes == null)
                    return false;
                foreach (LayerType t in _layerTypes)
                    if (t == LayerType.Mamba2)
                        return true;
                return false;
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen) => EnsureCacheCapacity(requiredSeqLen);

        /// <summary>
        /// Trunk forward for speculative decoding: the ordinary per-op layer
        /// loop (identical math to <see cref="ForwardCore"/>) plus, when a DFlash
        /// drafter is attached, a capture of each dflash.target_layers residual
        /// into <paramref name="hAllOut"/> (n x <see cref="SpecFeatureSize"/>
        /// floats; a buffer too small for one row per token holds only the last
        /// row, written to row 0) and, when <paramref name="allLogitsRows"/> is
        /// set, LM-head logits for every row instead of only the last.
        /// Advances the KV caches and _cacheSeqLen exactly like ForwardCore().
        /// </summary>
        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor hidden = Embedding(tokens);

            bool captureAll = false, captureLast = false;
            if (HasDFlash && hAllOut != null && hAllOut.Length > 0)
            {
                int feat = _dflash.FeatureSize;
                captureAll = hAllOut.LongLength >= (long)seqLen * feat;
                captureLast = !captureAll && hAllOut.LongLength >= feat;
            }

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                int slot = _dflashCaptureSlot != null && (captureAll || captureLast)
                    ? _dflashCaptureSlot[layer]
                    : -1;
                if (slot >= 0)
                    DFlashCaptureFeature(hidden, slot, seqLen, hAllOut, captureLast);

                switch (_layerTypes[layer])
                {
                    case LayerType.Mamba2:
                        hidden = Mamba2Block(hidden, layer, seqLen, isDecode: false);
                        break;
                    case LayerType.Attention:
                        hidden = AttentionBlock(hidden, layer, seqLen, startPos, isDecode: false);
                        break;
                    case LayerType.FFN:
                        hidden = FFNBlock(hidden, layer, seqLen, isDecode: false);
                        break;
                }
            }
            _forwardSw.Stop();

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            if (allLogitsRows)
            {
                Tensor logitsT = LinearForward(normed, "output.weight")
                    ?? LinearForward(normed, "token_embd.weight");
                normed.Dispose();
                if (logitsT == null)
                    throw new InvalidOperationException("Nemotron SpecForward: no LM head weight (output.weight / token_embd.weight).");
                if (logitsOut == null || logitsOut.LongLength < (long)seqLen * Config.VocabSize)
                    throw new InvalidOperationException("Nemotron SpecForward: logitsOut too small for all logits rows.");
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, logitsOut.LongLength * 4, (long)seqLen * Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }
            else
            {
                Tensor lastRow;
                if (seqLen > 1)
                {
                    using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                    lastRow = Ops.NewContiguous(narrowed);
                    normed.Dispose();
                }
                else
                {
                    lastRow = normed;
                }
                Tensor logitsT = LinearForward(lastRow, "output.weight")
                    ?? LinearForward(lastRow, "token_embd.weight");
                lastRow.Dispose();
                if (logitsT == null || logitsOut == null)
                    throw new InvalidOperationException("Nemotron SpecForward: no LM head weight (output.weight / token_embd.weight).");
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
        }

        // ====================================================================
        // rollback half - Mamba2 conv/SSM state snapshot
        // ====================================================================

        private float[][] _specConvSnap;
        private float[][] _specSsmSnap;

        /// <summary>
        /// Snapshot the Mamba2 conv/SSM state of every trunk layer. Taken right
        /// before a speculative verify batch so a partial rejection can roll the
        /// recurrent state back (attention KV needs only a position rewind, and
        /// the kept-prefix re-forward rewrites it).
        /// </summary>
        public void SpecSnapshotRecurrentState()
        {
            if (!HasMamba2Layers)
                return;
            int numLayers = Config.NumLayers;
            _specConvSnap ??= new float[numLayers][];
            _specSsmSnap ??= new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Mamba2)
                    continue;
                float[] c = _convState[l], s = _ssmState[l];
                if (c == null || s == null)
                    continue;
                if (_specConvSnap[l] == null || _specConvSnap[l].Length != c.Length)
                    _specConvSnap[l] = new float[c.Length];
                if (_specSsmSnap[l] == null || _specSsmSnap[l].Length != s.Length)
                    _specSsmSnap[l] = new float[s.Length];
                Array.Copy(c, _specConvSnap[l], c.Length);
                Array.Copy(s, _specSsmSnap[l], s.Length);
            }
        }

        /// <summary>Restore the Mamba2 conv/SSM state captured by
        /// <see cref="SpecSnapshotRecurrentState"/>. The native-decode flags are
        /// cleared too so the next single-token decode re-seeds its device state
        /// from the restored host arrays (same discipline as ResetStateForNewSession).</summary>
        public void SpecRestoreRecurrentState()
        {
            if (_specConvSnap == null || _specSsmSnap == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Mamba2)
                    continue;
                if (_specConvSnap[l] != null && _convState[l] != null)
                    Array.Copy(_specConvSnap[l], _convState[l], _specConvSnap[l].Length);
                if (_specSsmSnap[l] != null && _ssmState[l] != null)
                    Array.Copy(_specSsmSnap[l], _ssmState[l], _specSsmSnap[l].Length);
            }
            if (_mamba2NativeDecodeStateInitialized != null)
                Array.Clear(_mamba2NativeDecodeStateInitialized);
        }

        /// <summary>
        /// Rewind the attention KV position counter after rejected speculative
        /// tokens. Rows past <paramref name="length"/> are dead weight that the
        /// next forward simply overwrites (the causal mask never reads past the
        /// current position), so no data movement is needed.
        /// </summary>
        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            }
            _cacheSeqLen = length;
        }
    }
}