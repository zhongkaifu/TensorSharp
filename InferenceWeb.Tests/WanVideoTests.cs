// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan text-to-video unit tests: the pure-managed pieces (scheduler sigmas,
// T5 relative-attention buckets, patchify/unpatchify round trip, parameter
// snapping, RoPE axis split) plus — when the local model files are present —
// the UMT5 tokenizer against ids produced by the HuggingFace reference
// tokenizer (google/umt5-xxl via Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    public class WanVideoTests
    {
        // ---- scheduler --------------------------------------------------------

        [Fact]
        public void SchedulerSigmasMatchFlowShiftFormula()
        {
            // shift*s/(1+(shift-1)*s) over s = linspace(1, 1/N, N), N=4, shift=3
            var sch = new WanScheduler(4, 3.0f);
            Assert.Equal(5, sch.Sigmas.Length);
            Assert.Equal(1.0f, sch.Sigmas[0], 5);        // s=1 -> 1
            Assert.Equal(0.9f, sch.Sigmas[1], 5);        // s=0.75 -> 2.25/2.5
            Assert.Equal(0.75f, sch.Sigmas[2], 5);       // s=0.5 -> 1.5/2
            Assert.Equal(0.5f, sch.Sigmas[3], 5);        // s=0.25 -> 0.75/1.5
            Assert.Equal(0f, sch.Sigmas[4], 5);
            Assert.Equal(1000f, sch.Timesteps[0], 3);
        }

        [Fact]
        public void SchedulerEulerStepMovesTowardVelocity()
        {
            var sch = new WanScheduler(2, 3.0f);
            var x = new float[] { 1f, -1f };
            var v = new float[] { 2f, 2f };
            float dt = sch.Sigmas[1] - sch.Sigmas[0];   // negative
            sch.Step(x, v, 0);
            Assert.Equal(1f + dt * 2f, x[0], 5);
            Assert.Equal(-1f + dt * 2f, x[1], 5);
        }

        // ---- T5 relative position buckets ------------------------------------

        [Fact]
        public void RelativeBucketsMatchT5Reference()
        {
            // Reference values from transformers T5Attention._relative_position_bucket
            // (bidirectional=True, num_buckets=32, max_distance=128).
            int n = 40;
            int[] b = WanTextEncoder.ComputeRelativeBuckets(n);
            Assert.Equal(0, b[0]);                 // rel 0
            Assert.Equal(16 + 1, b[0 * n + 1]);    // rel +1 -> future bucket 17
            Assert.Equal(1, b[1 * n + 0]);         // rel -1 -> bucket 1
            Assert.Equal(16 + 7, b[0 * n + 7]);    // rel +7 (exact range)
            Assert.Equal(7, b[7 * n + 0]);         // rel -7
            // rel +8 enters the log range: 8 + int(log(8/8)/log(128/8)*8) = 8
            Assert.Equal(16 + 8, b[0 * n + 8]);
            // rel +16: 8 + int(log(2)/log(16)*8) = 8 + 2 = 10
            Assert.Equal(16 + 10, b[0 * n + 16]);
            // rel +32: 8 + int(log(4)/log(16)*8) = 8 + 4 = 12
            Assert.Equal(16 + 12, b[0 * n + 32]);
            // rel -39: 8 + int(log(39/8)/log(16)*8) = 8 + 4 = 12
            Assert.Equal(12, b[39 * n + 0]);
        }

        // ---- patchify ---------------------------------------------------------

        [Fact]
        public void PatchifyUnpatchifyRoundTrip()
        {
            int lt = 3, lh = 6, lw = 4;
            var rng = new Random(1);
            var latent = new float[16 * lt * lh * lw];
            for (int i = 0; i < latent.Length; i++) latent[i] = (float)rng.NextDouble();
            var tokens = new float[latent.Length];
            WanVideoPipeline.Patchify(latent, tokens, lt, lh, lw, 16, 64, 0);
            var back = new float[latent.Length];
            WanVideoPipeline.Unpatchify(tokens, back, lt, lh, lw, 16, 64);
            Assert.Equal(latent, back);
        }

        [Fact]
        public void PatchifyUnpatchifyRoundTrip48Channels()
        {
            // Wan 2.2 TI2V: 48-channel latent, 192-wide tokens.
            int lt = 2, lh = 4, lw = 4;
            var rng = new Random(7);
            var latent = new float[48 * lt * lh * lw];
            for (int i = 0; i < latent.Length; i++) latent[i] = (float)rng.NextDouble();
            var tokens = new float[latent.Length];
            WanVideoPipeline.Patchify(latent, tokens, lt, lh, lw, 48, 192, 0);
            var back = new float[latent.Length];
            WanVideoPipeline.Unpatchify(tokens, back, lt, lh, lw, 48, 192);
            Assert.Equal(latent, back);
        }

        [Fact]
        public void PatchifyFeatureLayoutIsConvKernelOrder()
        {
            // feature index must be kw + 2*kh + 4*c (the flattened GGUF conv kernel
            // layout [kw, kh, kd, ic, oc]).
            int lt = 1, lh = 2, lw = 2;
            var latent = new float[16 * lt * lh * lw];
            // channel c=3, position (h=1, w=0) => kh=1, kw=0 of token 0
            latent[3 * (lh * lw) + 1 * lw + 0] = 42f;
            var tokens = new float[latent.Length];
            WanVideoPipeline.Patchify(latent, tokens, lt, lh, lw, 16, 64, 0);
            Assert.Equal(42f, tokens[0 + 2 * 1 + 4 * 3]);
        }

        [Fact]
        public void PatchifyChannelOffsetWritesBehindNoiseFeatures()
        {
            // A14B I2V: the 20 conditioning channels (4 mask + 16 image latent) land at
            // feature offset 64 of the 144-wide token (global channel index 16 + c).
            int lt = 1, lh = 2, lw = 2;
            var cond = new float[20 * lt * lh * lw];
            cond[0] = 7f;                       // mask channel 0 at (h=0, w=0)
            var tokens = new float[144];
            WanVideoPipeline.Patchify(cond, tokens, lt, lh, lw, 20, 80, 0);
            var full = new float[144];
            Array.Copy(tokens, 0, full, 64, 80);
            Assert.Equal(7f, full[64 + 0]);     // feature kw=0,kh=0 of global channel 16
        }

        // ---- parameter snapping ----------------------------------------------

        [Theory]
        [InlineData(832, 832)]
        [InlineData(830, 832)]
        [InlineData(1, 16)]
        [InlineData(479, 480)]
        public void SnapDimRoundsToSixteen(int input, int expected)
            => Assert.Equal(expected, WanVideoPipeline.SnapDim(input));

        [Theory]
        [InlineData(1, 1)]
        [InlineData(2, 5)]
        [InlineData(33, 33)]
        [InlineData(30, 33)]
        [InlineData(16, 17)]
        public void SnapFramesRoundsToVaeTemporalGrid(int input, int expected)
            => Assert.Equal(expected, WanVideoPipeline.SnapFrames(input));

        // ---- degenerate-decode guard ------------------------------------------

        private static TensorSharp.Models.QwenImage.RgbImage MakeFrame(int w, int h, Func<int, float> pixel)
        {
            var chw = new float[3 * w * h];
            for (int i = 0; i < chw.Length; i++) chw[i] = pixel(i);
            return TensorSharp.Models.QwenImage.RgbImage.FromPlanarChw(w, h, chw);
        }

        [Fact]
        public void FlatDecodeIsRejected()
        {
            // Every frame a single colour = the Metal tensor-API VAE corruption.
            var frames = new[] { MakeFrame(8, 8, _ => 0f), MakeFrame(8, 8, _ => 0f), MakeFrame(8, 8, _ => 0f) };
            var ex = Assert.Throws<InvalidOperationException>(
                () => WanVideoPipeline.AssertFramesAreNotDegenerate(frames));
            Assert.Contains("TS_WAN_METAL_TENSOR_API=0", ex.Message);
        }

        [Fact]
        public void NaNDecodeIsRejected()
        {
            var frames = new[] { MakeFrame(8, 8, _ => float.NaN), MakeFrame(8, 8, _ => float.NaN) };
            Assert.Throws<InvalidOperationException>(
                () => WanVideoPipeline.AssertFramesAreNotDegenerate(frames));
        }

        [Fact]
        public void RealDecodeIsAccepted()
        {
            var frames = new[] { MakeFrame(8, 8, i => (i % 97) / 97f), MakeFrame(8, 8, i => (i % 53) / 53f) };
            WanVideoPipeline.AssertFramesAreNotDegenerate(frames);   // must not throw
        }

        [Fact]
        public void ASingleFlatFrameIsNotEnoughToReject()
        {
            // A legitimately flat frame (a fade to black at the end) must not fail
            // the whole video — the guard only fires when every sampled frame is flat.
            var frames = new[]
            {
                MakeFrame(8, 8, i => (i % 97) / 97f), MakeFrame(8, 8, i => (i % 53) / 53f),
                MakeFrame(8, 8, i => (i % 31) / 31f), MakeFrame(8, 8, _ => 0f),
            };
            WanVideoPipeline.AssertFramesAreNotDegenerate(frames);
        }

        // ---- DiT GGUF probe ----------------------------------------------------

        /// <summary>
        /// The loader tells the Wan family apart by the DiT's 5-D patch embedding
        /// [kw, kh, kd, ic, oc], so a GGUF reader held to ggml's four dimensions refused every
        /// Wan DiT at probe time. A synthetic T2V DiT (ic 16) resolves to the wan architecture,
        /// passes the model's own 5-D check and stops only at its missing companion files.
        /// </summary>
        [Fact]
        public void DiTWithAFiveDimensionalPatchEmbedding_GetsPastTheProbeToItsCompanions()
        {
            using var env = new EnvScope();
            foreach (string name in new[] { "TS_VIDEO_VAE", "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_DIT2" })
                env.Set(name, null);
            // Nested, so the companion search (the DiT's directory and its parent) stays in it.
            string root = Directory.CreateTempSubdirectory("ts-wan-probe-").FullName;
            string path = Path.Combine(Directory.CreateDirectory(Path.Combine(root, "dit")).FullName, "wan2.1-t2v.gguf");
            try
            {
                SyntheticGguf.Write(path,
                    new List<SyntheticGguf.Kv> { new SyntheticGguf.Str { Key = "general.architecture", V = "wan" } },
                    new List<SyntheticGguf.Tensor> { SyntheticGguf.Gen("patch_embedding.weight", 0.1f, 2, 2, 1, 16, 8) });

                using (var probe = new GgufFile(path))
                {
                    Assert.Equal("wan", ModelArchitectureRegistry.Resolve(probe.GetString("general.architecture"), probe).Id);
                    Assert.Equal(new ulong[] { 2, 2, 1, 16, 8 }, probe.Tensors["patch_embedding.weight"].Shape);
                }
                var ex = Assert.Throws<FileNotFoundException>(() => ModelBase.Create(path, BackendType.Cpu));
                Assert.Contains("needs companion models", ex.Message);
                Assert.Contains("wan_2.1_vae.safetensors", ex.Message);
            }
            finally
            {
                // The constructor that threw never disposed the DiT it opened; let its stream close.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                try { Directory.Delete(root, recursive: true); } catch { /* best effort */ }
            }
        }

        // ---- step-distilled checkpoint detection -------------------------------

        [Theory]
        // Published distilled Wan checkpoints / LoRAs.
        [InlineData("Wan2_2-TI2V-5B-Turbo-Q8_0.gguf", 4)]
        [InlineData("Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors", 4)]
        [InlineData("Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors", 4)]
        [InlineData("Wan2.2-I2V-A14B-HighNoise-Lightning-4steps-Q4_K_M.gguf", 4)]
        [InlineData("wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step.safetensors", 4)]
        [InlineData("Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v-8steps.gguf", 8)]
        // Ordinary checkpoints must NOT be mistaken for distilled ones.
        [InlineData("Wan2.2-TI2V-5B-Q8_0.gguf", 0)]
        [InlineData("wan2.1-t2v-14b-Q4_K_M.gguf", 0)]
        [InlineData("Wan2.1-T2V-1.3B-F16.gguf", 0)]
        [InlineData("", 0)]
        public void DistilledStepsParsedFromCheckpointName(string fileName, int expected)
            => Assert.Equal(expected, WanVideoModel.ParseDistilledSteps(fileName));

        [Fact]
        public void DistilledStepCountIsBoundedToASaneRange()
        {
            // A "1.3B" or a Q4_K_M shard index must not be read as a step count, and
            // an absurd count falls back to the marker default rather than trusting it.
            Assert.Equal(0, WanVideoModel.ParseDistilledSteps("Wan2.1-T2V-1.3B-F16.gguf"));
            Assert.Equal(4, WanVideoModel.ParseDistilledSteps("wan-turbo-9999steps.gguf"));
        }

        // ---- VAE decode band layout -------------------------------------------

        [Theory]
        // (latent rows, plane width px, spatial scale) — TI2V-5B (16x) and Wan 2.1 (8x)
        [InlineData(52, 1088, 16)]   // 1088x832, the 121-frame 720p-class I2V shape
        [InlineData(44, 1280, 16)]   // 1280x704, the official TI2V 720p recipe
        [InlineData(80, 1920, 16)]
        [InlineData(120, 1664, 8)]
        public void VaeBandsCoverThePlaneWithinTheMemoryBudget(int lh, int w, int scale)
        {
            const long threshold = 640_000;
            var starts = WanVaeBase.PlanBands(lh, w, scale, threshold, out int bandLat);
            Assert.NotNull(starts);
            Assert.True(starts.Count >= 2);

            // every band fits the per-band pixel budget
            Assert.True((long)w * bandLat * scale <= threshold,
                $"band {w}x{bandLat * scale} exceeds the {threshold} px budget");
            // the plane is fully covered, first band at the top, last flush with lh
            Assert.Equal(0, starts[0]);
            Assert.Equal(lh - bandLat, starts[^1]);
            for (int i = 1; i < starts.Count; i++)
            {
                Assert.True(starts[i] > starts[i - 1], "band starts must advance");
                int overlap = starts[i - 1] + bandLat - starts[i];
                Assert.True(overlap >= WanVaeBase.OverlapLat,
                    $"seam {i} overlaps {overlap} rows, want >= {WanVaeBase.OverlapLat}");
            }
        }

        [Fact]
        public void VaeBandsDecodeFewerRowsThanTheFixedHeightWalk()
        {
            // 1088x832 (lh 52): the old fixed-height walk used a 24-row band at
            // starts 0/16/28 — 72 rows of work for 52 rows of output.
            const long threshold = 640_000;
            var starts = WanVaeBase.PlanBands(52, 1088, 16, threshold, out int bandLat);
            Assert.Equal(2, starts.Count);
            Assert.Equal(30, bandLat);
            Assert.Equal(60, starts.Count * bandLat);   // was 72
        }

        [Fact]
        public void VaeSkipsTilingWhenThePlaneFitsWhole()
        {
            // 832x480 on the TI2V VAE is 30 latent rows — under the budget, one graph.
            Assert.Null(WanVaeBase.PlanBands(30, 832, 16, 640_000, out int bandLat));
            Assert.Equal(30, bandLat);
        }

        // ---- guidance cache ---------------------------------------------------

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void GuidanceCacheOffRunsBothPassesEveryStep(int stride)
        {
            for (int i = 0; i < 50; i++)
                Assert.True(WanVideoPipeline.UsesUncondPass(i, 50, stride));
        }

        [Theory]
        [InlineData(2, 50, 27)]   // 3 warm-up + every 2nd of steps 3..49 (which includes the last)
        [InlineData(3, 50, 20)]   // 3 warm-up + 16 strided + the last
        [InlineData(4, 50, 16)]
        public void GuidanceCacheStrideSkipsTheExpectedUncondPasses(int stride, int steps, int expected)
        {
            int uncond = 0;
            for (int i = 0; i < steps; i++)
                if (WanVideoPipeline.UsesUncondPass(i, steps, stride)) uncond++;
            Assert.Equal(expected, uncond);
        }

        [Fact]
        public void GuidanceCacheAlwaysRecomputesWarmupAndFinalStep()
        {
            const int steps = 50, stride = 4;
            // The steps that decide structure (the first few) and the one that
            // produces the final latent must never run on a stale guidance delta.
            for (int i = 0; i < WanVideoPipeline.CfgCacheWarmup; i++)
                Assert.True(WanVideoPipeline.UsesUncondPass(i, steps, stride));
            Assert.True(WanVideoPipeline.UsesUncondPass(steps - 1, steps, stride));
            // ... and something in the middle must actually be skipped, or the
            // cache would be a no-op.
            Assert.False(WanVideoPipeline.UsesUncondPass(WanVideoPipeline.CfgCacheWarmup + 1, steps, stride));
        }

        // ---- progress / ETA formatting ----------------------------------------

        [Theory]
        [InlineData(-1, "ETA unknown")]
        [InlineData(43, "~43s")]
        [InlineData(432, "~7m 12s")]
        [InlineData(3840, "~1h 04m")]
        public void FormatEtaIsHumanReadable(double seconds, string expected)
            => Assert.Equal(expected, WanVideoPipeline.FormatEta(seconds));

        // ---- RoPE -------------------------------------------------------------

        [Fact]
        public void RopeAxisSplitIs44_42_42()
        {
            // head_dim 128 splits t/h/w = 44/42/42 (diffusers WanRotaryPosEmbed).
            var rope = WanRope.Build(2, 3, 4);
            Assert.Equal(2 * 3 * 4 * 128, rope.Cos.Length);
            // pair-duplicated layout: cos[2i] == cos[2i+1]
            for (int i = 0; i < 64; i++)
                Assert.Equal(rope.Cos[2 * i], rope.Cos[2 * i + 1]);
            // token (t=0,h=0,w=0) has angle 0 everywhere -> cos 1, sin 0
            for (int i = 0; i < 128; i++)
            {
                Assert.Equal(1f, rope.Cos[i], 5);
                Assert.Equal(0f, rope.Sin[i], 5);
            }
            // token (t=1,h=0,w=0): first t-axis pair rotates by angle 1 (pos 1, freq 1)
            long tok12 = 12 * 128;
            Assert.Equal(MathF.Cos(1f), rope.Cos[tok12 + 0], 5);
            // h/w-axis pairs of that token stay at angle 0 (h=w=0)
            Assert.Equal(1f, rope.Cos[tok12 + 2 * 22], 5);
            Assert.Equal(1f, rope.Cos[tok12 + 2 * 43], 5);
        }

        // ---- UniPC scheduler ---------------------------------------------------

        [Fact]
        public void UniPCMatchesDiffusersOracle()
        {
            // Reference trajectory from diffusers UniPCMultistepScheduler
            // (prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=8,
            // solver_order=2), 10 steps over x0 = cos(i+1) with the deterministic
            // fake model v = sin(0.7*x + 0.3*k) + 0.1*x.
            var sch = new WanUniPCScheduler(10, 8.0f);
            Assert.Equal(0.999999f, sch.Sigmas[0], 5);
            Assert.Equal(0.9863164f, sch.Sigmas[1], 5);
            Assert.Equal(0.4730704f, sch.Sigmas[9], 5);
            Assert.Equal(0f, sch.Sigmas[10], 5);
            Assert.Equal(999f, sch.Timesteps[0]);
            Assert.Equal(986f, sch.Timesteps[1]);
            Assert.Equal(473f, sch.Timesteps[9]);

            int n = 8;
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = MathF.Cos(i + 1);
            for (int k = 0; k < sch.Steps; k++)
            {
                var v = new float[n];
                for (int i = 0; i < n; i++) v[i] = MathF.Sin(x[i] * 0.7f + k * 0.3f) + 0.1f * x[i];
                x = sch.Step(x, v);
            }
            float[] expected = { 0.038064f, -1.197444f, -1.761833f, -1.445976f, -0.329927f, 0.685251f, 0.361508f, -0.886314f };
            for (int i = 0; i < n; i++)
                Assert.True(Math.Abs(x[i] - expected[i]) < 2e-4,
                    $"lane {i}: {x[i]} vs diffusers {expected[i]}");
        }

        // ---- timestep sinusoid -------------------------------------------------

        [Fact]
        public void SinusoidIsCosThenSin()
        {
            var dst = new float[256];
            WanDiT.FillSinusoid(dst, 0f);
            for (int i = 0; i < 128; i++) Assert.Equal(1f, dst[i], 5);       // cos(0)
            for (int i = 128; i < 256; i++) Assert.Equal(0f, dst[i], 5);     // sin(0)
            WanDiT.FillSinusoid(dst, 1000f);
            Assert.Equal(MathF.Cos(1000f), dst[0], 3);
            Assert.Equal(MathF.Sin(1000f), dst[128], 3);
            // freq j: 10000^(-j/128)
            Assert.Equal(MathF.Cos(1000f * MathF.Pow(10000f, -1f / 128f)), dst[1], 3);
        }

        // ---- tokenizer vs HuggingFace reference (needs the local UMT5 GGUF) ----

        private const string TePath = @"C:\Works\models\wan\umt5-xxl-encoder-Q8_0.gguf";

        [Fact]
        [Trait("Requires", "Models")]
        public void TokenizerMatchesHuggingFaceReference()
        {
            if (!File.Exists(TePath)) return;   // model not present on this machine
            using var te = new WanTextEncoder(TePath);

            // Reference ids from transformers AutoTokenizer
            // (Wan-AI/Wan2.1-T2V-1.3B-Diffusers, subfolder=tokenizer), add_special_tokens=True.
            AssertTokens(te, "a lovely cat", new[] { 289, 51486, 6283, 1 });
            AssertTokens(te,
                "a close-up photo of a fluffy orange cat sitting on a wooden table, warm lighting",
                new[] { 289, 13304, 280, 1753, 7821, 329, 289, 8778, 139412, 36062, 6283, 83267,
                        369, 289, 129805, 8338, 275, 13809, 76320, 1 });
            // CJK coverage: a prefix of the official Wan negative prompt.
            AssertTokens(te,
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
                new[] { 273, 1838, 8483, 181911, 30993, 356, 4915, 85615, 356, 15665, 13607, 356,
                        17121, 8024, 7921, 162805, 1013, 3815, 356, 5549, 16096, 356, 6865, 3967,
                        356, 9103, 356, 7084, 2354, 356, 16927, 356, 15665, 10671, 356, 44776,
                        2390, 46846, 356, 2740, 5731, 20499, 356, 3846, 20499, 1 });
        }

        private static void AssertTokens(WanTextEncoder te, string text, int[] expected)
        {
            int[] padded = te.PrepareTokens(text, out int realLen);
            Assert.Equal(expected.Length, realLen);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], padded[i]);
            for (int i = realLen; i < padded.Length; i++)
                Assert.Equal(0, padded[i]);
        }
    }
}
