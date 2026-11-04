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
using System.Collections.Generic;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Block speculation through a semi-autoregressive drafter that proposes a
    /// WHOLE window in one pass, with a confidence head scoring each position:
    /// DeepSeek V4's DSpark, Muse-Glimmer's DFlash (block diffusion). One
    /// drafter pass per decode step instead of K, which is what makes a large
    /// drafter affordable next to a sparse-MoE trunk.
    ///
    /// The gate is CUMULATIVE rather than per-position, and that is not a
    /// detail: draft position i only pays off when the whole prefix before it
    /// is also accepted, so the PRODUCT of the per-position acceptance
    /// probabilities is the expected value of adding it, and
    /// <see cref="MinDraftProb"/> is the point where that value stops covering
    /// the extra verify row. Per-position gating - what the reference runtimes
    /// do - keeps positions whose prefix has already gone unlikely.
    /// </summary>
    public sealed class BlockDraftSpeculator : ISpeculator
    {
        /// <summary>
        /// Default gate for a BLOCK drafter. It thresholds a different quantity to
        /// <see cref="DraftHeadSpeculator.DefaultGate"/>:
        /// different quantity: the CUMULATIVE prefix probability (the product of
        /// the confidence head's per-position estimates), which decays with every
        /// position. 0.35 is the break-even point where an extra verify row stops
        /// paying for itself on a sparse-MoE trunk; a per-token-sized 0.75 here
        /// truncates almost every block to nothing.
        /// </summary>
        public const float DefaultGate = 0.35f;

        private readonly IDraftHead _head;
        internal IDraftHead HeadIdentity => _head;
        private readonly int[] _blockTokens;
        private readonly float[] _blockConf;

        public BlockDraftSpeculator(IDraftHead head, int blockSize, int maxDraftTokens)
        {
            _head = head ?? throw new ArgumentNullException(nameof(head));
            if (blockSize < 1)
                throw new ArgumentOutOfRangeException(nameof(blockSize));
            if (maxDraftTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(maxDraftTokens));
            BlockSize = blockSize;
            // A block drafter's window can never exceed the block it was trained
            // to emit.
            MaxDraftTokens = Math.Min(maxDraftTokens, blockSize);
            _blockTokens = new int[blockSize];
            _blockConf = new float[blockSize];
        }

        /// <summary>Tokens the drafter was trained to emit per pass.</summary>
        public int BlockSize { get; }

        public string Name => SpeculatorRegistry.Block;

        /// <summary>
        /// A block drafter reads its own sliding KV ring, which is refilled by the
        /// freshly forwarded suffix and by every token the trunk commits, so an adopted
        /// prefix costs it a shorter drafting context for the first block or two and
        /// nothing after that. Refusing to arm here is what made DFlash look useless
        /// from a chat's second turn onward.
        /// </summary>
        public bool CanArmAfterPrefixReuse => true;

        public string Describe() => $"block({BlockSize})";

        public int MaxDraftTokens { get; }

        public float MinDraftProb { get; set; } = DefaultGate;

        public float DefaultMinDraftProb => DefaultGate;

        public bool NeedsHiddenState => true;

        public bool HandlesOwnPrefill => _head.DraftSelfCatchUp;

        public int PrefillContextRows => _head.DraftContextRows;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int n = _head.DraftBlock(ctx.LastToken, ctx.CarryHidden, ctx.Position, _blockTokens, _blockConf);
            return SelectDrafts(_blockTokens, _blockConf, n, ctx.MaxTokens, draftOut);
        }

        /// <summary>The cumulative-confidence gate over one drafted block: keeps the
        /// longest prefix (at most <paramref name="maxTokens"/>) whose product of
        /// confidences stays at or above <see cref="MinDraftProb"/>. Shared by
        /// <see cref="Propose"/> and drafts made for several sequences at once.</summary>
        public int SelectDrafts(int[] blockTokens, float[] blockConf, int n, int maxTokens, List<int> draftOut)
        {
            double cumulative = 1.0;
            int limit = Math.Min(n, maxTokens);
            for (int i = 0; i < limit; i++)
            {
                cumulative *= blockConf[i];
                if (cumulative < MinDraftProb)
                    break;
                draftOut.Add(blockTokens[i]);
            }
            return draftOut.Count;
        }

        public void Commit(int[] tokens, float[]? hRows, int startPos)
            => _head.DraftCatchUp(tokens, hRows, startPos);

        public void Reset() { }

        public void Dispose() { }
    }
}
