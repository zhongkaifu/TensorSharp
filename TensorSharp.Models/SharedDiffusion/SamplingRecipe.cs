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
    /// <summary>Where a recipe's sigmas come from.</summary>
    public enum DiffusionSchedule
    {
        /// <summary>The sigma list itself (<see cref="SamplingRecipe.Sigmas"/>): steps + 1 values ending in 0. A
        /// step-distillation adapter's contract fixes one.</summary>
        Explicit,

        /// <summary>ComfyUI's "simple" scheduler over a model sampling that is piecewise linear in a grid's index
        /// over 1000 discrete timesteps (<see cref="SamplingRecipe.Grid"/>): MiniMax-H3-Image's template.</summary>
        ComfySimpleOverGrid,

        /// <summary>An endpoint-inclusive linspace u_i = 1 - i/N through the flow shift
        /// sigma = s*u / (1 + (s-1)*u), then a terminal 0 (diffusers' flow-matching set_timesteps), per stream.</summary>
        LinspaceShift,

        /// <summary>
        /// Exponential dynamic shifting by the image's token count, then terminal stretching (diffusers'
        /// FlowMatchEulerDiscreteScheduler with use_dynamic_shifting and shift_terminal, as Qwen/Qwen-Image-2.1's
        /// scheduler_config.json sets it): mu is linear in the tokens through (256, 0.5) and (8192, 0.9), each linspace point
        /// t of 1 .. 1/N becomes e^mu / (e^mu + 1/t - 1), the shifted list is stretched so that its last point is 0.02, and a
        /// terminal 0 follows. One step has no interval to stretch: [1, 0].
        /// </summary>
        DynamicShift,

        /// <summary>
        /// Sigma nodes per step count (<see cref="SamplingRecipe.NodeSets"/>: a distilled checkpoint's or a step-distillation
        /// adapter's trained schedules), each optionally through the exponential dynamic shift of
        /// <see cref="DynamicShift"/> (<see cref="SamplingRecipe.ShiftNodes"/>, no terminal stretch), then a terminal 0. Only the
        /// listed step counts run.
        /// </summary>
        NodeSets,
    }

    /// <summary>How a model with per-interval output heads (MiniMax-H3-Image's PDD heads) picks a step's heads.</summary>
    public enum HeadPolicy
    {
        /// <summary>The model has one head.</summary>
        None,

        /// <summary>The grid intervals nearest the step's ends, at least one (ComfyUI's final layer).</summary>
        GridInterval,

        /// <summary>Head i at step i, over the interval [i, i + 1) (diffusers).</summary>
        ByStep,
    }

    /// <summary>The update a recipe's sampler applies.</summary>
    public enum DiffusionSamplerKind
    {
        /// <summary>First-order Euler on the flow ODE (<see cref="EulerFlowSampler"/>).</summary>
        EulerFlow,
    }

    /// <summary>
    /// How a family (or a step-distillation adapter's contract) samples: the schedule, the default step count, the
    /// head policy, the sampler and the default guidance. Immutable; <see cref="SigmasFor"/> gives the sigmas a run
    /// of a step count denoises on, in float32 exactly as the reference implementation computes them.
    /// </summary>
    public sealed record SamplingRecipe
    {
        public required DiffusionSchedule Schedule { get; init; }

        /// <summary>Steps a request that names none takes.</summary>
        public required int Steps { get; init; }

        /// <summary><see cref="DiffusionSchedule.Explicit"/>: the sigmas (steps + 1, decreasing, ending in 0).</summary>
        public float[] Sigmas { get; init; }

        /// <summary>The model's decreasing sigma grid, first 1 or below and last 0: what
        /// <see cref="DiffusionSchedule.ComfySimpleOverGrid"/> samples over and what <see cref="HeadPolicy.GridInterval"/>
        /// picks heads on.</summary>
        public float[] Grid { get; init; }

        /// <summary><see cref="DiffusionSchedule.LinspaceShift"/>: the primary stream's flow shift.</summary>
        public float Shift { get; init; } = 1f;

        /// <summary><see cref="DiffusionSchedule.LinspaceShift"/>: the shift of each further named stream (MiniMax-H3
        /// video's audio), or null.</summary>
        public IReadOnlyDictionary<string, float> StreamShifts { get; init; }

        public HeadPolicy Heads { get; init; } = HeadPolicy.None;

        /// <summary><see cref="DiffusionSchedule.NodeSets"/>: the sigma nodes (decreasing, in (0, 1], no terminal 0) of each
        /// step count the recipe runs.</summary>
        public IReadOnlyDictionary<int, float[]> NodeSets { get; init; }

        /// <summary><see cref="DiffusionSchedule.NodeSets"/>: the nodes go through the token count's exponential dynamic
        /// shift.</summary>
        public bool ShiftNodes { get; init; }

        /// <summary>The denoiser reads its timestep as a bf16 pipeline computes it (sigma * 1000 rounded to bf16, then / 1000
        /// in bf16): an adapter trained in bf16 whose distilled sigmas sit between bf16 values.</summary>
        public bool TimestepBf16 { get; init; }

        /// <summary>What a refusal of a step count names as the recipe's owner ("Qwen-Image-2.1-Turbo, which was distilled
        /// for one published 8-step schedule,"); null names the recipe itself.</summary>
        public string Owner { get; init; }

        public DiffusionSamplerKind Sampler { get; init; } = DiffusionSamplerKind.EulerFlow;

        /// <summary>Guidance a request that names none takes (1: a single branch).</summary>
        public float Cfg { get; init; } = 1f;

        /// <summary>Whether the recipe fixes its step count (an explicit sigma list has exactly one, node sets their own).</summary>
        public bool FixedSteps => Schedule is DiffusionSchedule.Explicit or DiffusionSchedule.NodeSets;

        /// <summary>Whether the sigmas depend on the image's token count (<see cref="SigmasFor(int, long)"/>).</summary>
        public bool DependsOnTokens => Schedule == DiffusionSchedule.DynamicShift || (Schedule == DiffusionSchedule.NodeSets && ShiftNodes);

        /// <summary>The step count a request asking for <paramref name="requested"/> (0: the default) runs; an explicit
        /// list refuses any other count than its own, node sets any count they do not list.</summary>
        public int ResolveSteps(int requested)
        {
            if (requested < 0) throw new ArgumentOutOfRangeException(nameof(requested), "Steps must be nonnegative (zero selects the default).");
            if (requested == 0) return Steps;
            if (Schedule == DiffusionSchedule.NodeSets)
            {
                if (NodeSets == null || !NodeSets.ContainsKey(requested))
                    throw new ArgumentException(
                        $"{Owner ?? "This sampling recipe"} defines schedules for {string.Join(", ", (NodeSets?.Keys ?? Array.Empty<int>()).OrderBy(k => k))} " +
                        $"step(s), not {requested}. Use one of those step counts, or omit --diffusion-steps / \"steps\" to use its default ({Steps}).");
                return requested;
            }
            if (FixedSteps && requested != Steps)
                throw new ArgumentException($"This sampling recipe denoises exactly {Steps} steps on its own sigmas; {requested} steps contradicts it.");
            return requested;
        }

        /// <summary>The sigmas a run of <paramref name="steps"/> steps denoises on: steps + 1 values ending in 0. A schedule
        /// that depends on the image's token count (<see cref="DependsOnTokens"/>) needs <see cref="SigmasFor(int, long)"/>.</summary>
        public float[] SigmasFor(int steps)
        {
            if (DependsOnTokens)
                throw new InvalidOperationException($"The {Schedule} schedule depends on the image's token count; pass it.");
            return SigmasFor(steps, 0);
        }

        /// <summary>The sigmas a run of <paramref name="steps"/> steps over an image of <paramref name="imageTokens"/> denoiser
        /// tokens denoises on (the token count is read only by a schedule that <see cref="DependsOnTokens"/>).</summary>
        public float[] SigmasFor(int steps, long imageTokens)
        {
            if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
            if (DependsOnTokens && imageTokens <= 0) throw new ArgumentOutOfRangeException(nameof(imageTokens));
            switch (Schedule)
            {
                case DiffusionSchedule.DynamicShift:
                    return DiffusionSchedules.DynamicShift(steps, imageTokens);
                case DiffusionSchedule.NodeSets:
                    if (NodeSets == null || !NodeSets.TryGetValue(steps, out float[] nodes))
                        throw new ArgumentException($"{Owner ?? "This sampling recipe"} defines no schedule for {steps} step(s).");
                    return DiffusionSchedules.Nodes(nodes, ShiftNodes ? imageTokens : 0);
                case DiffusionSchedule.Explicit:
                    if (Sigmas == null || steps != Sigmas.Length - 1)
                        throw new ArgumentException($"This sampling recipe's explicit sigmas take {(Sigmas?.Length ?? 1) - 1} steps, not {steps}.");
                    return (float[])Sigmas.Clone();
                case DiffusionSchedule.ComfySimpleOverGrid:
                    return DiffusionSchedules.ComfySimpleOverGrid(Grid, steps);
                case DiffusionSchedule.LinspaceShift:
                    return DiffusionSchedules.LinspaceShift(steps, Shift);
                default:
                    throw new NotSupportedException($"Schedule {Schedule} is not implemented.");
            }
        }

        /// <summary>The sigmas of the named further stream (<see cref="StreamShifts"/>) at <paramref name="steps"/> steps.</summary>
        public float[] StreamSigmas(string stream, int steps)
        {
            if (Schedule != DiffusionSchedule.LinspaceShift)
                throw new InvalidOperationException("Only a linspace-and-shift recipe has per-stream sigmas.");
            if (StreamShifts == null || !StreamShifts.TryGetValue(stream ?? string.Empty, out float shift))
                throw new ArgumentException($"The recipe has no stream '{stream}'.", nameof(stream));
            return DiffusionSchedules.LinspaceShift(steps, shift);
        }

        /// <summary>The grid intervals [Start, Stop) step <paramref name="step"/> (from <paramref name="sigma"/> to
        /// <paramref name="next"/>) predicts with; (0, 0) for a model with one head.</summary>
        public (int Start, int Stop) HeadsFor(int step, float sigma, float next) => Heads switch
        {
            HeadPolicy.None => (0, 0),
            HeadPolicy.GridInterval => DiffusionSchedules.GridInterval(Grid, sigma, next),
            HeadPolicy.ByStep => ByStep(step),
            _ => throw new NotSupportedException($"Head policy {Heads} is not implemented."),
        };

        private (int, int) ByStep(int step)
        {
            int heads = Grid != null ? Grid.Length - 1 : int.MaxValue;
            if (step < 0 || step >= heads)
                throw new ArgumentOutOfRangeException(nameof(step), $"Step {step + 1} has no head of its own: the model has {heads}.");
            return (step, step + 1);
        }

        /// <summary>A new sampler for one run of this recipe.</summary>
        public IDiffusionSampler CreateSampler() => Sampler switch
        {
            DiffusionSamplerKind.EulerFlow => new EulerFlowSampler(),
            _ => throw new NotSupportedException($"Sampler {Sampler} is not implemented."),
        };

        /// <summary>Structural checks (a family's default recipe is checked when the pipeline is made).</summary>
        public void Validate()
        {
            if (Steps < 1) throw new ArgumentException("A sampling recipe takes at least one step.");
            if (!float.IsFinite(Cfg) || Cfg < 0) throw new ArgumentException("A sampling recipe's guidance must be finite and nonnegative.");
            switch (Schedule)
            {
                case DiffusionSchedule.Explicit:
                    if (Sigmas == null || Sigmas.Length != Steps + 1)
                        throw new ArgumentException($"An explicit recipe of {Steps} steps lists {Steps + 1} sigmas.");
                    DiffusionSchedules.CheckDecreasingToZero(Sigmas, "explicit sigmas");
                    break;
                case DiffusionSchedule.ComfySimpleOverGrid:
                    if (Grid == null || Grid.Length < 2) throw new ArgumentException("A grid schedule needs a grid of at least two points.");
                    DiffusionSchedules.CheckDecreasingToZero(Grid, "grid");
                    break;
                case DiffusionSchedule.LinspaceShift:
                    if (!float.IsFinite(Shift) || Shift <= 0) throw new ArgumentException("A flow shift must be finite and positive.");
                    if (StreamShifts != null && StreamShifts.Values.Any(s => !float.IsFinite(s) || s <= 0))
                        throw new ArgumentException("A stream's flow shift must be finite and positive.");
                    break;
                case DiffusionSchedule.NodeSets:
                    if (NodeSets == null || NodeSets.Count == 0) throw new ArgumentException("A node-set recipe lists at least one schedule.");
                    if (!NodeSets.ContainsKey(Steps)) throw new ArgumentException($"A node-set recipe's default of {Steps} steps has no schedule.");
                    foreach (var (count, nodes) in NodeSets)
                        DiffusionSchedules.CheckNodes(nodes, count);
                    break;
            }
            if (Heads != HeadPolicy.None && (Grid == null || Grid.Length < 2))
                throw new ArgumentException($"Head policy {Heads} picks heads on the model's grid, which the recipe does not give.");
        }

        /// <summary>One phrase for logs: "6 steps, comfy-simple over a 7-point grid, grid-interval heads, cfg 1".</summary>
        public string Describe(int steps)
        {
            string schedule = Schedule switch
            {
                DiffusionSchedule.Explicit => "explicit sigmas",
                DiffusionSchedule.ComfySimpleOverGrid => $"comfy-simple over a {Grid?.Length ?? 0}-point grid",
                DiffusionSchedule.LinspaceShift => $"linspace, shift {Shift.ToString("0.###", CultureInfo.InvariantCulture)}",
                DiffusionSchedule.DynamicShift => "dynamic exponential shift",
                DiffusionSchedule.NodeSets => ShiftNodes ? "its own sigmas, dynamically shifted" : "its own fixed sigmas",
                _ => Schedule.ToString(),
            };
            string heads = Heads switch { HeadPolicy.GridInterval => ", grid-interval heads", HeadPolicy.ByStep => ", by-step heads", _ => "" };
            return $"{steps} steps, {schedule}{heads}, cfg {Cfg.ToString("0.###", CultureInfo.InvariantCulture)}" +
                   (TimestepBf16 ? ", bf16 timesteps" : "");
        }
    }

    /// <summary>The schedules' arithmetic, in float32 as the reference implementations compute it.</summary>
    public static class DiffusionSchedules
    {
        /// <summary>
        /// ComfyUI's KSampler "simple" scheduler at <paramref name="steps"/> steps over a model sampling whose sigma is
        /// piecewise linear in the grid index over 1000 discrete timesteps (comfy MiniMaxH3ImageModelSampling). At six
        /// steps over MiniMax-H3-Image's grid it lands within 0.001 of the grid (1, 0.96765, 0.88771, 0.75, 0.53364,
        /// 0.23261, 0), reproduced in float32 the way PyTorch computes it so both engines denoise on the same sigmas.
        /// Ends with 0.
        /// </summary>
        public static float[] ComfySimpleOverGrid(float[] grid, int steps)
        {
            ArgumentNullException.ThrowIfNull(grid);
            if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
            if (grid.Length < 2) throw new ArgumentException("The grid needs at least two points.", nameof(grid));
            int n = grid.Length - 1;
            var ascending = new float[grid.Length];
            for (int i = 0; i < grid.Length; i++) ascending[i] = grid[grid.Length - 1 - i];

            const int timesteps = 1000;
            const float multiplier = 1000f;
            var table = new float[timesteps];
            for (int k = 1; k <= timesteps; k++)
            {
                float timestep = (float)k / timesteps * multiplier;
                float pos = Math.Clamp(timestep / multiplier, 0f, 1f) * n;
                int i = Math.Min((int)MathF.Floor(pos), n - 1);
                table[k - 1] = Lerp(ascending[i], ascending[i + 1], pos - i);
            }

            var sigmas = new float[steps + 1];
            double stride = (double)timesteps / steps;
            for (int x = 0; x < steps; x++)
                sigmas[x] = table[timesteps - 1 - (int)(x * stride)];
            sigmas[steps] = 0f;
            return sigmas;
        }

        /// <summary>torch.lerp for float32, both of its branches.</summary>
        private static float Lerp(float start, float end, float weight) =>
            Math.Abs(weight) < 0.5f ? start + weight * (end - start) : end - (end - start) * (1f - weight);

        /// <summary>The endpoint-inclusive linspace grid through the flow shift: u_i = 1 - i/N for i = 0..N-1,
        /// sigma_i = s*u_i / (1 + (s-1)*u_i) in double, then a terminal 0 (diffusers' flow-matching set_timesteps; the
        /// grid MiniMax-H3's Turbo adapters were distilled on). Returns steps + 1 values from 1 down to exactly 0.</summary>
        public static float[] LinspaceShift(int steps, float shift)
        {
            if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
            if (!float.IsFinite(shift) || shift <= 0) throw new ArgumentOutOfRangeException(nameof(shift));
            var sigmas = new float[steps + 1];
            for (int i = 0; i < steps; i++)
                sigmas[i] = (float)TimeSnrShift(shift, 1.0 - (double)i / steps);
            sigmas[steps] = 0f;
            return sigmas;
        }

        /// <summary><see cref="DiffusionSchedule.DynamicShift"/>'s sigmas, in double as the reference computes them and rounded
        /// once to float32: steps + 1 values ending in 0.</summary>
        public static float[] DynamicShift(int steps, long imageTokens)
        {
            if (steps <= 0 || imageTokens <= 0) throw new ArgumentOutOfRangeException(steps <= 0 ? nameof(steps) : nameof(imageTokens));
            // The anchors are interpolated and extrapolated (16,384 tokens at 2048 x 2048 lies past 8192).
            if (steps == 1) return new[] { 1f, 0f };
            double exp = Math.Exp(DynamicShiftMu(imageTokens));
            double lastShifted = exp / (exp + steps - 1);
            double terminalScale = (1 - lastShifted) / (1 - 0.02);
            var result = new float[steps + 1];
            for (int i = 0; i < steps; i++)
            {
                // linspace(1, 1 / steps, steps).
                double t = (double)(steps - i) / steps;
                double shifted = exp / (exp + (1 / t - 1));
                result[i] = (float)(1 - (1 - shifted) / terminalScale);
            }
            result[steps - 1] = 0.02f;
            return result;
        }

        /// <summary>The dynamic shift's mu at <paramref name="imageTokens"/>: linear through (256, 0.5) and (8192, 0.9).</summary>
        public static double DynamicShiftMu(long imageTokens) => 0.5 + (imageTokens - 256) * (0.9 - 0.5) / (8192 - 256);

        /// <summary><see cref="DiffusionSchedule.NodeSets"/>' sigmas: the nodes, through the exponential dynamic shift at
        /// <paramref name="shiftTokens"/> tokens when it is positive, then a terminal 0.</summary>
        public static float[] Nodes(float[] nodes, long shiftTokens)
        {
            ArgumentNullException.ThrowIfNull(nodes);
            var result = new float[nodes.Length + 1];
            double exp = shiftTokens > 0 ? Math.Exp(DynamicShiftMu(shiftTokens)) : 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                double t = nodes[i];
                result[i] = shiftTokens > 0 ? (float)(exp / (exp + (1 / t - 1))) : (float)t;
            }
            result[nodes.Length] = 0f;
            return result;
        }

        internal static void CheckNodes(float[] nodes, int steps)
        {
            if (nodes == null || nodes.Length != steps)
                throw new ArgumentException($"A node set of {steps} step(s) lists {steps} sigma nodes.");
            for (int i = 0; i < nodes.Length; i++)
            {
                if (!float.IsFinite(nodes[i]) || nodes[i] <= 0f || nodes[i] > 1f)
                    throw new ArgumentException("Sigma nodes lie in (0, 1].");
                if (i > 0 && !(nodes[i] < nodes[i - 1])) throw new ArgumentException("Sigma nodes must strictly decrease.");
            }
        }

        /// <summary>The flow-matching SNR shift: sigma' = a*t / (1 + (a-1)*t).</summary>
        public static double TimeSnrShift(double shift, double t) => shift * t / (1.0 + (shift - 1.0) * t);

        /// <summary>The grid intervals the step from <paramref name="sigma"/> to <paramref name="next"/> spans,
        /// [start, stop), as ComfyUI's final layer picks them: the grid points nearest each end, at least one interval.</summary>
        public static (int Start, int Stop) GridInterval(float[] grid, float sigma, float next)
        {
            ArgumentNullException.ThrowIfNull(grid);
            int n = grid.Length - 1;
            int start = Nearest(grid, sigma), stop = Nearest(grid, next);
            start = Math.Min(start, n - 1);
            stop = Math.Max(stop, start + 1);
            return (start, stop);
        }

        private static int Nearest(float[] grid, float value)
        {
            int best = 0;
            for (int k = 1; k < grid.Length; k++)
                if (Math.Abs(grid[k] - value) < Math.Abs(grid[best] - value)) best = k;
            return best;
        }

        internal static void CheckDecreasingToZero(float[] sigmas, string what)
        {
            if (sigmas[^1] != 0f) throw new ArgumentException($"The {what} must end in 0.");
            if (!(sigmas[0] <= 1f)) throw new ArgumentException($"The {what} must start at 1 or below.");
            for (int i = 1; i < sigmas.Length; i++)
                if (!(sigmas[i] < sigmas[i - 1])) throw new ArgumentException($"The {what} must decrease.");
        }
    }
}
