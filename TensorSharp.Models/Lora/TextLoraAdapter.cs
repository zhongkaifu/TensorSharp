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

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// One low-rank update term for a single base-model projection, in a form that
    /// makes the forward path two small GEMMs with no per-call transpose.
    ///
    /// <para>
    /// A PEFT adapter stores <c>lora_A</c> as <c>[rank, in]</c> and <c>lora_B</c> as
    /// <c>[out, rank]</c>. The delta for input <c>x</c> (<c>[seq, in]</c>) is
    /// <c>(alpha / rank) * x @ A^T @ B^T</c>. This type stores <see cref="AT"/>
    /// (<c>[in, rank]</c>) and <see cref="BT"/> (<c>[rank, out]</c>) with the effective
    /// scale already folded into <c>BT</c>, so the forward computes
    /// <c>x @ AT @ BT</c> and adds it to the base projection.
    /// </para>
    /// </summary>
    public sealed class LoraDelta
    {
        /// <summary>Input feature count of the base projection.</summary>
        public long In { get; init; }

        /// <summary>Output feature count of the base projection.</summary>
        public long Out { get; init; }

        /// <summary>Low rank.</summary>
        public int Rank { get; init; }

        /// <summary><c>[In, Rank]</c> row-major — the transpose of <c>lora_A</c>.</summary>
        public float[] AT { get; init; } = Array.Empty<float>();

        /// <summary><c>[Rank, Out]</c> row-major — the transpose of <c>lora_B</c>, scale folded in.</summary>
        public float[] BT { get; init; } = Array.Empty<float>();
    }

    /// <summary>
    /// A loaded LoRA adapter: a name, the low rank and alpha, and one
    /// <see cref="LoraDelta"/> per target projection keyed by base-model weight name
    /// (e.g. <c>"blk.0.attn_qkv.weight"</c>). Immutable after load, so it is safe to
    /// share between the registry, the model and concurrent inference readers.
    /// </summary>
    public sealed class TextLoraAdapter
    {
        public string Name { get; }

        /// <summary>Low rank inferred from the <c>lora_A</c> tensors (or the PEFT config).</summary>
        public int Rank { get; }

        /// <summary>The PEFT <c>lora_alpha</c>, or <see cref="Rank"/> when unspecified.</summary>
        public float Alpha { get; }

        /// <summary>Per-target deltas keyed by base-model weight name.</summary>
        public IReadOnlyDictionary<string, LoraDelta> Deltas { get; }

        /// <summary>One human-readable line summarising the source files.</summary>
        public string Summary { get; }

        public TextLoraAdapter(
            string name,
            int rank,
            float alpha,
            IReadOnlyDictionary<string, LoraDelta> deltas,
            string summary)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));
            Name = name;
            Rank = rank;
            Alpha = alpha;
            Deltas = deltas ?? throw new ArgumentNullException(nameof(deltas));
            Summary = summary ?? string.Empty;
        }
    }

    /// <summary>Public metadata for <c>GET /v1/models</c>, sized to stay cheap to list.</summary>
    public sealed record LoraAdapterInfo(string Name, int Rank, float Alpha, int Modules);
}