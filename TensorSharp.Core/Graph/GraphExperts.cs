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
        /// routed more than twice as often lately as the tier member it would replace. The executor that can move them
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

        /// <summary>
        /// The routed experts of <paramref name="rows"/> rows on the host: <paramref name="x"/> [rows, hidden],
        /// <paramref name="ids"/> and <paramref name="weights"/> [rows, K]; a route whose <paramref name="skip"/> entry
        /// is nonzero is the device's and adds nothing here. Writes the weighted sum [rows, hidden] to
        /// <paramref name="output"/>. False when the host kernels do not cover the experts' type or shape.
        /// </summary>
        bool Compute(int layer, ReadOnlySpan<float> x, ReadOnlySpan<int> ids, ReadOnlySpan<float> weights,
            ReadOnlySpan<byte> skip, int rows, Span<float> output);
    }
}
