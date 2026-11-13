// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;

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
    public interface IGraphExpertHost
    {
        /// <summary>Whether the executor computes <paramref name="expert"/> of <paramref name="layer"/> on its device:
        /// a member of the layer's tier, kept resident there.</summary>
        bool IsResident(int layer, int expert);

        /// <summary>Whether a step of <paramref name="rows"/> rows runs every route on the device, the experts its
        /// device does not hold streamed in for the step (a prompt chunk: reading each selected expert once beats
        /// computing many rows of it on the host).</summary>
        bool StreamsStep(int layer, int rows);

        /// <summary>Count a step's routes ([rows, K] expert ids) for the routing profile and the tier policy.</summary>
        void Record(int layer, ReadOnlySpan<int> ids);

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
