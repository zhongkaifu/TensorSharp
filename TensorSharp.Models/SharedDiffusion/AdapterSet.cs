// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// The sampling a step-distillation (speed) adapter was trained for, which it fixes: its steps, sigmas, head policy,
    /// guidance and generation floor. A request may leave steps and guidance unset or repeat them; any other value is
    /// refused, never overridden.
    /// </summary>
    public sealed record SamplingContract
    {
        /// <summary>The adapter that declares it (refusals name it).</summary>
        public required string Name { get; init; }

        /// <summary>The sigmas, steps + 1 of them, ending in 0.</summary>
        public required float[] Sigmas { get; init; }

        public HeadPolicy Heads { get; init; } = HeadPolicy.None;
        public float Cfg { get; init; } = 1f;

        /// <summary>The area the adapter generates at, below the family's floor (0: the family's).</summary>
        public long MinGenerationArea { get; init; }

        public int Steps => Sigmas.Length - 1;

        /// <summary>Refuse request values that contradict the contract.</summary>
        public void Validate(int steps, float cfg)
        {
            if (steps != 0 && steps != Steps)
                throw new ArgumentException(
                    $"{Name} samples exactly {Steps} steps on its own schedule; {steps} steps contradicts it. " +
                    $"Leave the step count unset or set it to {Steps}.");
            if (cfg != 0f && cfg != Cfg)
                throw new ArgumentException(
                    $"{Name} samples at guidance {Cfg.ToString("0.###", CultureInfo.InvariantCulture)}; " +
                    $"guidance {cfg.ToString("0.###", CultureInfo.InvariantCulture)} contradicts it. Leave it unset or set it to " +
                    $"{Cfg.ToString("0.###", CultureInfo.InvariantCulture)}.");
        }

        /// <summary>The recipe a run under this contract samples with: <paramref name="family"/>'s, with the contract's
        /// explicit sigmas, heads and guidance (the family's grid stays for head selection).</summary>
        public SamplingRecipe Apply(SamplingRecipe family)
        {
            ArgumentNullException.ThrowIfNull(family);
            var recipe = family with
            {
                Schedule = DiffusionSchedule.Explicit,
                Steps = Steps,
                Sigmas = (float[])Sigmas.Clone(),
                Heads = Heads,
                Cfg = Cfg,
            };
            recipe.Validate();
            return recipe;
        }
    }

    /// <summary>One LoRA plug-in bound to a family's components: its name, strength, the family's loaded payload, and
    /// a speed adapter's <see cref="SamplingContract"/> (null for a style adapter).</summary>
    public sealed record DiffusionAdapter
    {
        public required string Name { get; init; }
        public float Scale { get; init; } = 1f;
        public SamplingContract Contract { get; init; }

        /// <summary>What the family loaded for it (its factors), opaque to the runtime.</summary>
        public object Payload { get; init; }
    }

    /// <summary>
    /// The adapters a request runs with: any number of style adapters, each with its own strength, and at most one
    /// speed adapter, whose contract then fixes the sampling. Two speed adapters are refused (two schedules).
    /// </summary>
    public sealed class AdapterSet
    {
        public static readonly AdapterSet Empty = new(Array.Empty<DiffusionAdapter>());

        public AdapterSet(IEnumerable<DiffusionAdapter> adapters)
        {
            ArgumentNullException.ThrowIfNull(adapters);
            Adapters = adapters.ToArray();
            foreach (DiffusionAdapter adapter in Adapters)
            {
                ArgumentNullException.ThrowIfNull(adapter, nameof(adapters));
                if (!float.IsFinite(adapter.Scale) || adapter.Scale < 0)
                    throw new ArgumentException($"LoRA {adapter.Name}'s strength must be finite and nonnegative.");
            }
            DiffusionAdapter[] speed = Adapters.Where(a => a.Contract != null).ToArray();
            if (speed.Length > 1)
                throw new ArgumentException($"Two speed adapters ({speed[0].Name}, {speed[1].Name}) each fix the sampling schedule; use one.");
            Contract = speed.Length == 1 ? speed[0].Contract : null;
        }

        public IReadOnlyList<DiffusionAdapter> Adapters { get; }

        /// <summary>The speed adapter's contract, or null.</summary>
        public SamplingContract Contract { get; }

        public bool IsEmpty => Adapters.Count == 0;
    }
}
