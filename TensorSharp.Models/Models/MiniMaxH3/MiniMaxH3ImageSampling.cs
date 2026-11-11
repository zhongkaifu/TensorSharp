// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image sampling: the PDD (progressive distillation) step grid, the
// schedule ComfyUI's template samples on it, the head each step uses, and the
// initial noise.
//
// The checkpoint is distilled to six Euler steps with guidance 1 on its own sigma
// grid (the "pdd_sigmas" of its config), and its final layer carries one output
// head per grid interval: block 0 is a full head, block k an offset from it, so the
// step from grid[k] to grid[k+1] predicts with head 0 + head k.
using System;
using System.Collections.Generic;
using System.Text.Json;

namespace TensorSharp.Models.MiniMaxH3
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

    /// <summary>
    /// <c>torch.randn</c> on the CPU generator for a float32 tensor: a 32-bit Mersenne
    /// Twister seeded with <c>torch.manual_seed</c>, 24-bit uniforms, and the Box-Muller
    /// transform over blocks of 16 (pairs j and j+8), the last block redrawn when the
    /// count is not a multiple of 16. This is the noise ComfyUI's KSampler starts from
    /// (comfy.sample.prepare_noise), so a TensorSharp image at the same seed starts from
    /// the same latent. PyTorch's AVX2 build evaluates log/sin/cos with SLEEF's 1-ULP
    /// functions, so a value can differ from this in the last bit.
    /// </summary>
    internal static class TorchCpuRandn
    {
        public static float[] Normal(long count, ulong seed)
        {
            if (count < 16)
                throw new ArgumentOutOfRangeException(nameof(count), "only the vectorized (16 or more values) path is reproduced.");
            var mt = new Mt19937((uint)seed);
            var data = new float[count];
            for (long i = 0; i < count; i++) data[i] = Uniform(mt);
            for (long i = 0; i + 16 <= count; i += 16) Fill16(data, i);
            if (count % 16 != 0)
            {
                long last = count - 16;
                for (int j = 0; j < 16; j++) data[last + j] = Uniform(mt);
                Fill16(data, last);
            }
            return data;
        }

        private static float Uniform(Mt19937 mt) => (mt.Next() & 0xFFFFFFu) * (1.0f / 16777216f);

        private static void Fill16(float[] d, long o)
        {
            for (int j = 0; j < 8; j++)
            {
                float u1 = 1f - d[o + j];
                float u2 = d[o + j + 8];
                float radius = MathF.Sqrt(-2f * MathF.Log(u1));
                float theta = (float)(2.0 * Math.PI * u2);
                d[o + j] = radius * MathF.Cos(theta);
                d[o + j + 8] = radius * MathF.Sin(theta);
            }
        }

        private sealed class Mt19937
        {
            private readonly uint[] _state = new uint[624];
            private int _index = 624;

            public Mt19937(uint seed)
            {
                _state[0] = seed;
                for (int i = 1; i < 624; i++)
                    _state[i] = unchecked(1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) + (uint)i);
            }

            public uint Next()
            {
                if (_index >= 624) Twist();
                uint y = _state[_index++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9D2C5680u;
                y ^= (y << 15) & 0xEFC60000u;
                y ^= y >> 18;
                return y;
            }

            private void Twist()
            {
                for (int i = 0; i < 624; i++)
                {
                    uint y = (_state[i] & 0x80000000u) | (_state[(i + 1) % 624] & 0x7FFFFFFFu);
                    uint v = _state[(i + 397) % 624] ^ (y >> 1);
                    if ((y & 1u) != 0) v ^= 0x9908B0DFu;
                    _state[i] = v;
                }
                _index = 0;
            }
        }
    }
}
