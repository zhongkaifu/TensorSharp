// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image sampling: the PDD (progressive distillation) step grid, the
// schedule ComfyUI's template samples on it and the head each step uses. The
// initial noise is the diffusion runtime's TorchCpuRandn (SharedDiffusion).
//
// The checkpoint is distilled to six Euler steps with guidance 1 on its own sigma
// grid (the "pdd_sigmas" of its config), and its final layer carries one output
// head per grid interval: block 0 is a full head, block k an offset from it, so the
// step from grid[k] to grid[k+1] predicts with head 0 + head k.
using System;
using System.Collections.Generic;
using System.Text.Json;

using TensorSharp.Models.MiniMaxH3;

namespace TensorSharp.Models.MiniMaxH3Image
{
    internal static class MiniMaxH3ImageSampling
    {
        /// <summary>The step count the checkpoint is distilled to.</summary>
        public const int Steps = 6;

        /// <summary>The release's grid (Comfy-Org/MiniMax-H3-Image config, "pdd_sigmas"),
        /// for a checkpoint whose file does not carry it (a GGUF converted without metadata).</summary>
        public static readonly float[] ReleaseSigmas =
        {
            1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
            0.5327739119529724f, 0.23200887441635132f, 0.0f,
        };

        /// <summary>The grid a checkpoint declares in its safetensors metadata
        /// (<c>{"transformer": {"pdd_sigmas": [...]}}</c>), or the release grid.</summary>
        public static float[] GridFrom(IReadOnlyDictionary<string, string> metadata, int pddSteps)
        {
            float[] grid = ReleaseSigmas;
            if (metadata != null && metadata.TryGetValue("config", out string json) && !string.IsNullOrWhiteSpace(json))
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("transformer", out var t)
                    && t.TryGetProperty("pdd_sigmas", out var s) && s.ValueKind == JsonValueKind.Array)
                {
                    var values = new List<float>();
                    foreach (var v in s.EnumerateArray()) values.Add(v.GetSingle());
                    grid = values.ToArray();
                }
            }
            if (grid.Length != pddSteps + 1)
                throw new InvalidOperationException(
                    $"MiniMax-H3-Image has {pddSteps} output heads but a {grid.Length}-point sigma grid.");
            for (int i = 1; i < grid.Length; i++)
                if (!(grid[i] < grid[i - 1]))
                    throw new InvalidOperationException("MiniMax-H3-Image's sigma grid must decrease.");
            return grid;
        }

        /// <summary>
        /// The schedule ComfyUI's template samples: KSampler's "simple" scheduler at
        /// <paramref name="steps"/> steps over MiniMaxH3ImageModelSampling, whose sigma is
        /// piecewise linear in the grid index over 1000 discrete timesteps. At six steps it
        /// lands within 0.001 of the grid (1, 0.96765, 0.88766, 0.75, 0.53364, 0.23261, 0),
        /// reproduced here in float32 the way PyTorch computes it so both engines denoise
        /// on the same sigmas. Ends with 0.
        /// </summary>
        public static float[] Schedule(float[] grid, int steps)
        {
            if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
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

        /// <summary>The grid intervals the step from <paramref name="sigma"/> to
        /// <paramref name="next"/> spans, [start, stop), as ComfyUI's final layer picks them:
        /// the grid points nearest each end, at least one interval.</summary>
        public static (int Start, int Stop) Interval(float[] grid, float sigma, float next)
        {
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

        /// <summary>The head for grid intervals [start, stop): block 0 plus the dt-weighted
        /// mean of the offset blocks the step spans (block 0 is a full head, so it is never
        /// counted twice). <paramref name="bank"/> is [steps * rows][cols] row-major and
        /// <paramref name="bias"/> [steps * rows].</summary>
        public static (float[] Weight, float[] Bias) CombineHead(
            float[] bank, float[] bias, int rows, int cols, float[] grid, int start, int stop)
        {
            int steps = grid.Length - 1;
            if (bank.Length != (long)steps * rows * cols || bias.Length != (long)steps * rows)
                throw new ArgumentException("the PDD head bank does not match the grid.");
            var weight = new float[(long)rows * cols];
            var b = new float[rows];
            Array.Copy(bank, weight, weight.Length);
            Array.Copy(bias, b, rows);

            float total = 0f;
            for (int k = start; k < stop; k++) total += grid[k] - grid[k + 1];
            for (int k = Math.Max(start, 1); k < stop; k++)
            {
                float w = (grid[k] - grid[k + 1]) / total;
                long rowBase = (long)k * rows * cols;
                for (long i = 0; i < weight.Length; i++) weight[i] += w * bank[rowBase + i];
                for (int r = 0; r < rows; r++) b[r] += w * bias[(long)k * rows + r];
            }
            return (weight, b);
        }
    }
}
