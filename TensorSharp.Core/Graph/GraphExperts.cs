// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>
    /// A layer's stacked routed experts of which only some live on the executor's device: the model's
    /// <see cref="IGraphExpertHost"/> serves the others. <see cref="GraphOp.MoeHostExperts"/> carries these as its
    /// expert sources, so an executor learns which routes it may run itself without knowing the model, and every
    /// backend shares one expert-offload implementation (host compute, SSD streaming, routing profile, tiers).
    /// </summary>
    public interface IGraphHostedExperts : IGraphExpertWeights
    {
        /// <summary>The model's expert service for this layer's experts.</summary>
        IGraphExpertHost Host { get; }
    }

    /// <summary>
    /// The model side of expert offload, backend-neutral: which experts an executor's device holds (the layer's
    /// accelerator tier), the routed experts the host computes for the routes it does not hold, and the routing the
    /// tiers are ranked by. An executor computes the routes of resident experts with its own kernels and hands the
    /// rest here (see <see cref="TensorGraphExecutor"/>'s <see cref="GraphOp.MoeHostExperts"/>).
    /// </summary>
    /// <summary>How much a step's routes count in an adaptive tier's recent routing.</summary>
    public static class GraphExpertAdapt
    {
        /// <summary>A prompt chunk's selections weigh as this many rows' worth of decode selections (the Qwen3.8 engines'
        /// weighting: after a long prompt the answer routes much as the prompt did, but one prompt must not wipe the tier).</summary>
        public const float PromptRows = 128f;

        /// <summary>The weight of each route of a step of <paramref name="rows"/> rows.</summary>
        public static float Weight(int rows) => rows <= 16 ? 1f : Math.Min(1f, PromptRows / rows);
    }

    public interface IGraphExpertHost
    {
        /// <summary>Whether the executor computes <paramref name="expert"/> of <paramref name="layer"/> on its device:
        /// a member of the layer's tier, kept resident there.</summary>
        bool IsResident(int layer, int expert);

        /// <summary>Whether a step of <paramref name="rows"/> rows runs every route on the device, the experts its
        /// device does not hold streamed in for the step (a prompt chunk: reading each selected expert once beats
        /// computing many rows of it on the host).</summary>
        bool StreamsStep(int layer, int rows);

        /// <summary>Count a step's routes ([rows, K] expert ids) for the routing profile and the tier policy (a prompt
        /// chunk's routes weigh <paramref name="adaptWeight"/> each in the adaptive tier's recent counts).</summary>
        void Record(int layer, ReadOnlySpan<int> ids, float adaptWeight = 1f);

        /// <summary>
        /// Adaptive tiers (on by default: TS_MOE_TIER_ADAPT, as ggml's tiers and the engines have them): at most
        /// <paramref name="budget"/> trades, largest gains first across the layers, each a layer's expert outside its tier
        /// routed more than twice as often lately as the tier member it would replace, and at least twice; the recent counts
        /// decay once a call (so an executor calls it once a round). The executor that can move them
        /// does, then reports each with <see cref="CommitSwap"/>. Empty when adaptation is off.
        /// </summary>
        IReadOnlyList<(int Layer, int Out, int In)> ProposeSwaps(int budget);

        /// <summary>The executor moved <paramref name="layer"/>'s expert <paramref name="into"/> into the device slot of
        /// <paramref name="outOf"/>: the tier follows.</summary>
        void CommitSwap(int layer, int outOf, int into);

        /// <summary>
        /// The route-tail skip of <paramref name="layer"/> (approximate, on by default: TS_MOE_ROUTE_TAIL_SKIP, Strata's
        /// STRATA_ROUTE_TAIL_SKIP): in a step the host computes, an expert the device does not hold that every row routes at
        /// this 0-based rank or lower (in the row's descending top-K) is left out, its weight dropped, not renormalized.
        /// 0 computes every route (a layer whose tier holds most of its experts, or the switch at 0).
        /// </summary>
        int TailSkipRank(int layer);

        /// <summary>The most experts the host serves in any one layer (outside the layer's tier): what an executor reserves
        /// to stage one layer's host experts for a prompt chunk. Fixed for a load: an adaptive trade swaps one expert for
        /// another.</summary>
        int MostHostExperts { get; }

        /// <summary>
        /// The routed experts of <paramref name="rows"/> rows on the host: <paramref name="x"/> [rows, hidden],
        /// <paramref name="ids"/> and <paramref name="weights"/> [rows, K]; a route whose <paramref name="skip"/> entry
        /// is nonzero is the device's and adds nothing here. Writes the weighted sum [rows, hidden] to
        /// <paramref name="output"/>. False when the host kernels do not cover the experts' type or shape.
        /// </summary>
        bool Compute(int layer, ReadOnlySpan<float> x, ReadOnlySpan<int> ids, ReadOnlySpan<float> weights,
            ReadOnlySpan<byte> skip, int rows, Span<float> output);

        /// <summary>Where <paramref name="layer"/>'s host experts live when they are packed in page-locked host memory the
        /// device can address (<see cref="HostExpertCopies"/>), or null: each at its place in its stack's mapping.</summary>
        HostExpertCopies Copies(int layer) => null;
    }

    /// <summary>
    /// One layer's host experts packed in page-locked host memory the device can address, as the removed direct Qwen3.8 engine
    /// kept them: per matrix (gate, up, down) a stack of places of one expert each. The host computes
    /// its experts from here and the device reads them here (a prompt chunk's staging, a decode step's misses, a trade's
    /// copies), so nothing is page-locked while a model serves. A trade puts the tier member it moves to the host into the
    /// place of the expert it moves into the tier (<see cref="Move"/>): the places, and so the pinned bytes, never change.
    /// </summary>
    public sealed class HostExpertCopies
    {
        private readonly IntPtr[] _stacks, _deviceStacks;
        private readonly long[] _expertBytes;
        private readonly int[] _place, _owner;

        /// <param name="stacks">The places of each matrix (gate, up, down) in host memory.</param>
        /// <param name="deviceStacks">The same places' addresses on the device.</param>
        /// <param name="expertBytes">An expert's bytes in each matrix.</param>
        /// <param name="place">Each expert's place, or -1 (it has none: a tier member).</param>
        public HostExpertCopies(IntPtr[] stacks, IntPtr[] deviceStacks, long[] expertBytes, int[] place)
        {
            if (stacks?.Length != 3 || deviceStacks?.Length != 3 || expertBytes?.Length != 3 || place == null)
                throw new ArgumentException("host expert copies take three matrices and a place per expert");
            _stacks = (IntPtr[])stacks.Clone();
            _deviceStacks = (IntPtr[])deviceStacks.Clone();
            _expertBytes = (long[])expertBytes.Clone();
            _place = (int[])place.Clone();
            int places = 0;
            foreach (int p in _place) places = Math.Max(places, p + 1);
            _owner = new int[places];
            Array.Fill(_owner, -1);
            for (int e = 0; e < _place.Length; e++)
                if (_place[e] >= 0) _owner[_place[e]] = e;
        }

        /// <summary>Places per matrix: the experts the stacks hold (what the host kernel takes as the stack's expert count).</summary>
        public int Places => _owner.Length;

        /// <summary>Matrix <paramref name="matrix"/>'s places in host memory, and on the device.</summary>
        public IntPtr Stack(int matrix) => _stacks[matrix];

        /// <inheritdoc cref="Stack"/>
        public IntPtr DeviceStack(int matrix) => _deviceStacks[matrix];

        /// <summary>An expert's bytes in matrix <paramref name="matrix"/>.</summary>
        public long ExpertBytes(int matrix) => _expertBytes[matrix];

        /// <summary>
        /// The place that holds <paramref name="expert"/>'s bytes, or -1. Once a trade gave the incoming expert's place to its
        /// outgoing partner, a request for the incoming expert (the host serving a step queued before the trade) finds none and
        /// reads the expert where its stack's mapping holds it - the same bytes.
        /// </summary>
        public int PlaceOf(int expert)
        {
            if ((uint)expert >= (uint)_place.Length) return -1;
            int p = System.Threading.Volatile.Read(ref _place[expert]);
            return p >= 0 && System.Threading.Volatile.Read(ref _owner[p]) == expert ? p : -1;
        }

        /// <summary>Expert <paramref name="expert"/>'s bytes of matrix <paramref name="matrix"/> in host memory (zero: it has no
        /// place), and on the device.</summary>
        public IntPtr Bytes(int matrix, int expert)
        {
            int p = PlaceOf(expert);
            return p < 0 ? IntPtr.Zero : _stacks[matrix] + (nint)(p * _expertBytes[matrix]);
        }

        /// <inheritdoc cref="Bytes"/>
        public IntPtr DeviceBytes(int matrix, int expert)
        {
            int p = PlaceOf(expert);
            return p < 0 ? IntPtr.Zero : _deviceStacks[matrix] + (nint)(p * _expertBytes[matrix]);
        }

        /// <summary>
        /// A trade's outgoing expert <paramref name="outOf"/> now holds the place of incoming expert <paramref name="into"/>
        /// (its bytes copied there first). A place <paramref name="outOf"/> held before (copies made before the tier was named)
        /// is left free.
        /// </summary>
        public void Move(int into, int outOf)
        {
            int p = PlaceOf(into);
            if (p < 0) throw new InvalidOperationException($"expert {into} has no place to give expert {outOf}");
            int old = PlaceOf(outOf);
            if (old >= 0) System.Threading.Volatile.Write(ref _owner[old], -1);
            System.Threading.Volatile.Write(ref _place[outOf], p);
            System.Threading.Volatile.Write(ref _owner[p], outOf);
        }
    }

    /// <summary>
    /// An allocator whose device reads page-locked host memory in place (a discrete GPU's mapped host memory): the host
    /// experts' packed copies (<see cref="HostExpertCopies"/>) are allocated here, once, at load.
    /// </summary>
    public interface IMappedHostAllocator
    {
        /// <summary><paramref name="bytes"/> of page-locked host memory the device addresses at
        /// <paramref name="devicePointer"/>, or zero where the host refuses them.</summary>
        IntPtr AllocateMappedHost(long bytes, out IntPtr devicePointer);

        /// <summary>Memory from <see cref="AllocateMappedHost"/> freed (nothing may read it any more).</summary>
        void FreeMappedHost(IntPtr host);
    }
}
