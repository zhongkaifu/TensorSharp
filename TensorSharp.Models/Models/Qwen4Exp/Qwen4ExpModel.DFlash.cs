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
// Qwen3.8-Flash-Next (qwen4exp) as a DFlash / DFlash2 speculation TARGET.
//
// The drafter is shared (ModelBase.DFlash*.cs). What only this trunk can provide is
// the feature row its encoder was trained on. incoai/Qwen3.8-Flash-Next-DFlash2 reads
// the COMPLETED outputs of 0-based layers 5, 15, 25, 35 and 45, each averaged over the
// four hyper-connection streams (5 x 2560 = 12800 floats per position). The GGUF names
// them dflash.target_layers = [6, 16, 26, 36, 46], already shifted by one ("the
// residual entering layer L"), so the tap after layer L - 1 is the same quantity.
//
// The taps run inside the native token span (TSGgml_Qwen4ExpSetFeatureTaps): each is
// one mean over the [hc, n_embd] residual after that layer's FFN half, written into the
// caller's feature rows. A verify, a plain step and a prompt chunk all capture them, so
// the executor commits every accepted position into the drafter's ring.
// ---------------------------------------------------------------------------
using System;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IPerRequestDraftState
    {
        /// <summary>Each request drafts over a DFlash ring of its own (ModelBase.DFlash.Rings.cs, 81 MB on Flash Next's
        /// DFlash2), so requests that take turns at speculating keep their drafting context. The MTP head's state is
        /// already per holder (Qwen4ExpModel.Mtp.cs).</summary>
        public void BindDraftState(string requestId) => BindDFlashRing(requestId);

        public void ReleaseDraftState(string requestId) => ReleaseDFlashRing(requestId);

        /// <summary>Completed-output layer ids the native span taps, in feature-row
        /// order (dflash.target_layers - 1). Null without a DFlash drafter.</summary>
        private int[] _dflashTapLayers;

        /// <summary>True when <paramref name="path"/> names a DFlash/DFlash2 drafter
        /// (general.architecture = dflash) rather than a shared MTP head.</summary>
        internal static bool IsDFlashDraftFile(string path)
            => Qwen35Model.ResolveQwen35DFlashPath(path) != null;

        /// <summary>Attach a DFlash drafter. Called last in the constructor, because its
        /// tensors join the trunk's weight dictionaries and its ring comes off the same
        /// allocator. Declines (with a message) rather than failing the load: a drafter
        /// is an optimization.</summary>
        private void TryLoadQwen4ExpDFlash(string path)
        {
            try
            {
                string reason = DFlashTargetRefusal(path);
                if (reason != null)
                {
                    Console.WriteLine($"  DFlash drafter '{Path.GetFileName(path)}' not attached: {reason}");
                    return;
                }
                LoadDFlashDraftWeights(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DFlash drafter '{Path.GetFileName(path)}' could not be attached: {ex.Message}");
                return;
            }
            if (!HasDFlash)
                return;
            _dflashTapLayers = _dflash.TargetLayerIds.Select(l => l - 1).ToArray();
            Console.WriteLine($"  DFlash drafter taps: hc-stream mean after layers [{string.Join(",", _dflashTapLayers)}]"
                + $" ({_dflash.FeatureSize} floats per position)");
        }

        private string DFlashTargetRefusal(string path)
        {
            if (IsTensorParallel || HasDistributedQwen)
                return "tensor parallelism and distributed stages shard the LM head and token embedding the drafter borrows.";
            if (!IsGgmlBackend)
                return "the feature taps run inside the GGML token span.";
            using var probe = new GgufFile(path);
            var cfg = DFlashConfig.FromGguf(probe);
            if (cfg.TargetLayerIds.Any(l => l < 1 || l > Config.NumLayers))
                return $"its target layers [{string.Join(",", cfg.TargetLayerIds)}] are not completed outputs of this trunk's "
                    + $"{Config.NumLayers} layers.";
            return null;
        }

        /// <summary>Number of DRAFTS a block produces.</summary>
        public int DraftBlockSize => HasDFlash ? _dflash.MaxDraftTokens : 0;

        /// <summary>The DFlash ring: older context rows are overwritten before any draft.</summary>
        public int DraftContextRows => HasDFlash && DFlashRingRows > 0 ? DFlashRingRows : int.MaxValue;

        /// <summary>One whole-block draft (see ModelBase.DFlashPropose).</summary>
        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => HasDFlash ? DFlashPropose(lastToken, hPrev, position, draftOut, confOut) : 0;

        /// <summary>Drafts per verify window with a DFlash drafter (--spec-draft overrides).</summary>
        private int DFlashWindow() => Math.Min(6, Math.Max(1, _dflash.MaxDraftTokens));

        /// <summary>The cumulative-confidence gate for DFlash blocks (--spec-pmin overrides). Strata's Mac engine ends its
        /// windows at 0.7; on ggml_metal (M5 Pro, ISTA Q2_0, DFlash2 Q4_K_M, window 6, greedy) 0.5 decoded the benchmark's
        /// code and prose answers at 47.3 / 30.8 tok/s against 46.4 / 28.0 at 0.7.</summary>
        private static float DFlashGate() => 0.5f;

        /// <summary>Points the next token-span calls' DFlash taps at
        /// <paramref name="rows"/> (or turns them off with zero). Per span call, because a
        /// prompt that runs as several spans moves its row pointer per span.</summary>
        private void SetDFlashTaps(IntPtr rows)
        {
            if (_dflashTapLayers == null)
                return;
            if (!GgmlBasicOps.Qwen4ExpSetFeatureTaps(rows == IntPtr.Zero ? null : _dflashTapLayers, rows,
                    rows == IntPtr.Zero ? 0 : _dflash.FeatureSize))
                throw new InvalidOperationException(GgmlBasicOps.LastNativeError("qwen4exp: cannot set the DFlash feature taps."));
        }
    }
}
