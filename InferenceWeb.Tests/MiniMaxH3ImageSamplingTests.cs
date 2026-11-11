// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image sampling, layout and tiling: pure arithmetic, no weights. The expected
// values are ComfyUI 0.45.0's and PyTorch's (torch 2.14.1 CPU), captured with
// comfy.ldm.minimax.sampling.MiniMaxH3ImageModelSampling + comfy.samplers.simple_scheduler
// and torch.randn(..., generator=torch.manual_seed(seed)), not this implementation's.
using System;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3ImageSamplingTests
    {
        private static readonly float[] Grid = MiniMaxH3ImageSampling.ReleaseSigmas;

        [Fact]
        public void Schedule_MatchesComfySimpleSchedulerAtSixAndEightSteps()
        {
            float[] six = { 1.0f, 0.9676535725593567f, 0.8877084255218506f, 0.75f, 0.5336428284645081f, 0.23261038959026337f, 0f };
            float[] eight = { 1.0f, 0.9756428003311157f, 0.927536129951477f, 0.8531613945960999f, 0.75f,
                              0.5870804190635681f, 0.38239139318466187f, 0.1740066558122635f, 0f };
            Assert.Equal(six, MiniMaxH3ImageSampling.Schedule(Grid, 6), new FloatTolerance(2e-7f));
            Assert.Equal(eight, MiniMaxH3ImageSampling.Schedule(Grid, 8), new FloatTolerance(2e-7f));
        }

        [Fact]
        public void SixSteps_UseOneHeadPerGridInterval()
        {
            float[] s = MiniMaxH3ImageSampling.Schedule(Grid, 6);
            for (int i = 0; i < 6; i++)
                Assert.Equal((i, i + 1), MiniMaxH3ImageSampling.Interval(Grid, s[i], s[i + 1]));
        }

        [Fact]
        public void CombineHead_IsHeadZeroPlusTheStepsOffset()
        {
            int rows = 2, cols = 3, steps = Grid.Length - 1;
            var bank = new float[steps * rows * cols];
            var bias = new float[steps * rows];
            for (int i = 0; i < bank.Length; i++) bank[i] = i;
            for (int i = 0; i < bias.Length; i++) bias[i] = 100 + i;

            // The first step is head 0 alone: block 0 is a full head, not an offset.
            var (w0, b0) = MiniMaxH3ImageSampling.CombineHead(bank, bias, rows, cols, Grid, 0, 1);
            Assert.Equal(bank.Take(rows * cols), w0);
            Assert.Equal(bias.Take(rows), b0);

            // Step k (k >= 1) adds block k with weight 1.
            var (w3, b3) = MiniMaxH3ImageSampling.CombineHead(bank, bias, rows, cols, Grid, 3, 4);
            for (int i = 0; i < rows * cols; i++) Assert.Equal(bank[i] + bank[3 * rows * cols + i], w3[i]);
            for (int r = 0; r < rows; r++) Assert.Equal(bias[r] + bias[3 * rows + r], b3[r]);

            // A step over two intervals takes their dt-weighted mean (offsets only).
            var (w12, _) = MiniMaxH3ImageSampling.CombineHead(bank, bias, rows, cols, Grid, 1, 3);
            float d1 = Grid[1] - Grid[2], d2 = Grid[2] - Grid[3];
            float expected = bank[0] + (d1 * bank[1 * rows * cols] + d2 * bank[2 * rows * cols]) / (d1 + d2);
            Assert.Equal(expected, w12[0], 4);
        }

        [Fact]
        public void TorchRandn_MatchesTorchCpuGenerator()
        {
            float[] x = TorchCpuRandn.Normal(24L * 128 * 128, 7001);
            float[] first =
            {
                0.9944289922714233f, -1.680071473121643f, 1.1282382011413574f, 0.4605950117111206f, -1.019861102104187f,
                -0.11738254874944687f, -0.9547377228736877f, -0.17150536179542542f, -0.40882498025894165f, -0.27517831325531006f,
                -0.5369579792022705f, 1.4541001319885254f, 1.490768551826477f, -0.09017905592918396f, 0.30817699432373047f,
                -0.2556890547275543f, -1.218509554862976f, 0.06188349798321724f, -0.7430685758590698f, -0.2073061317205429f,
            };
            float[] last =
            {
                -1.1085244417190552f, -0.5795342326164246f, 1.1282215118408203f, -0.7842656373977661f, -0.7858785390853882f,
                -1.0466594696044922f, -1.064545750617981f, -0.7785916924476624f, 0.7777312397956848f, 0.06218326464295387f,
                0.5452315807342529f, -0.6495028138160706f, -0.34323927760124207f, 1.1148576736450195f, -0.4509247839450836f,
                0.09528206288814545f, 1.4073721170425415f, -1.4519617557525635f, -0.388069212436676f, 1.4653820991516113f,
            };
            // SLEEF's 1-ULP log/sin/cos in PyTorch's AVX2 build against .NET's: a few ULP.
            Assert.Equal(first, x.Take(20), new FloatTolerance(2e-6f));
            Assert.Equal(last, x.Skip(x.Length - 20), new FloatTolerance(2e-6f));
            // Over all 393,216 values the last-bit differences add up to under 0.01.
            Assert.InRange(x.Sum(v => (double)v), 1078.1769687583535 - 0.05, 1078.1769687583535 + 0.05);
            Assert.InRange(x.Sum(v => Math.Abs((double)v)), 314231.2349930563 - 0.5, 314231.2349930563 + 0.5);
        }

        [Fact]
        public void TorchRandn_RedrawsTheLastBlockWhenNotAMultipleOf16()
        {
            float[] y = TorchCpuRandn.Normal(360, 42);
            float[] last =
            {
                0.5666652917861938f, -0.7097623348236084f, -0.4875054359436035f, 0.050095997750759125f, 0.8478653430938721f,
                -0.6953384280204773f, 0.3056167960166931f, 0.29090648889541626f, 0.40854430198669434f, -1.2609347105026245f,
                0.9165152907371521f, -0.028006095439195633f, 0.17867784202098846f, -0.038506485521793365f, -0.08688340336084366f,
                -1.180280089378357f, 1.5460413694381714f, 0.5447550415992737f, 0.9934602379798889f, 0.5066711902618408f,
            };
            Assert.Equal(last, y.Skip(y.Length - 20), new FloatTolerance(2e-6f));
        }

        [Theory]
        // ComfyUI's split_tiles at 256-pixel tiles, 64-pixel minimum overlap.
        [InlineData(256, 1)]
        [InlineData(2048, 11)]
        [InlineData(4096, 21)]
        public void SplitTiles_CoversTheCanvasLikeComfy(int length, int tiles)
        {
            var (start, len, overlap) = MiniMaxH3ImageVae.SplitTiles(length);
            Assert.Equal(tiles, start.Length);
            Assert.Equal(length, start[^1] + len[^1]);
            Assert.All(overlap, o => Assert.True(o >= 64 && o % 16 == 0));
            for (int i = 1; i < start.Length; i++) Assert.Equal(start[i - 1] + 256 - overlap[i - 1], start[i]);
        }

        [Fact]
        public void Layout_PacksTextThenTargetAtOneTimestep()
        {
            var layout = MiniMaxH3ImageLayout.Build(textLength: 5, latentHeight: 8, latentWidth: 4, sigma: 0.75f);
            Assert.Equal(5 + 4 * 2, layout.TokenCount);
            Assert.Equal(new[] { 0.25f }, layout.Timesteps);
            Assert.Equal(5, layout.VideoSpan.Start);
            Assert.Equal(13, layout.VideoSpan.End);
            Assert.Equal(0, layout.AudioSpan.Length);
            // text modality 1 then visual 0, both on timestep row 0
            Assert.Equal(new[] { 1, 0 }, layout.ModulationSpans.Select(s => s.Row));
            // the target sits one step past the text on the time axis
            Assert.Equal(5f, layout.Positions[5 * 3]);
            Assert.Equal(new[] { 0f, 0f }, new[] { layout.Positions[4 * 3 + 1], layout.Positions[4 * 3 + 2] });
        }

        [Fact]
        public void Layout_PinsReferencesNearClean()
        {
            var layout = MiniMaxH3ImageLayout.Build(3, 4, 4, 0.5f, references: new[] { (4, 6) });
            Assert.Equal(new[] { 0.5f, 0.999f }, layout.Timesteps);
            Assert.Equal(3 + 2 * 3 + 2 * 2, layout.TokenCount);
            // the reference occupies one unit of the timeline; the target follows it
            Assert.Equal(3f, layout.Positions[3 * 3]);
            Assert.Equal(4f, layout.Positions[layout.VideoSpan.Start * 3]);
            Assert.Equal(1 * 3 + 0, layout.ModulationSpans[1].Row);
        }

        private sealed class FloatTolerance(float tolerance) : System.Collections.Generic.IEqualityComparer<float>
        {
            public bool Equals(float a, float b) => Math.Abs(a - b) <= tolerance;
            public int GetHashCode(float v) => 0;
        }
    }
}
