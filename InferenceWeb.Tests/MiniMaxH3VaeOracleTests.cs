// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 VAE numeric oracle tests: compare the native whole-graph decoders
// against reference outputs produced by the UPSTREAM PyTorch bundle
// (MiniMaxAI/MiniMax-H3's own video_vae/ and audio_vae/ sources) on deterministic
// inputs.
//
// Fixtures are raw little-endian F32 blobs written by
// InferenceWeb.Tests/tools/minimax_h3_oracle.py; each has a .shape sidecar. Every
// test no-ops when the fixtures or the weights are absent.
//
//   export TS_MINIMAX_H3_DIR=~/work/models/minimax-h3   # weights + fixtures/
//   export TS_TEST_GGML_BACKEND=metal
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3VaeOracleTests : IDisposable
    {
        private const string DirEnv = "TS_MINIMAX_H3_DIR";
        private readonly ITestOutputHelper _output;
        private readonly MiniMaxH3OracleBackend _backend;

        public MiniMaxH3VaeOracleTests(ITestOutputHelper output)
        { _output=output; _backend=new MiniMaxH3OracleBackend(); _output.WriteLine($"[h3-oracle] backend={_backend.Backend}"); }
        public void Dispose() => _backend.Dispose();

        private static string Root => Environment.GetEnvironmentVariable(DirEnv);
        private static string FixtureDir =>
            Root == null ? null : Path.Combine(Root, "fixtures");

        private static bool Have(params string[] names)
        {
            if (string.IsNullOrEmpty(Root)) return false;
            foreach (var n in names)
                if (!File.Exists(Path.Combine(FixtureDir, n))) return false;
            return true;
        }

        private static float[] ReadF32(string name)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(FixtureDir, name));
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return data;
        }

        private static int[] ReadShape(string name) =>
            File.ReadAllText(Path.Combine(FixtureDir, name + ".shape"))
                .Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();

        private static double Cosine(float[] a, float[] b)
        {
            Assert.Equal(a.Length, b.Length);
            double dot = 0, na = 0, nb = 0;
            for (long i = 0; i < a.LongLength; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
        }

        private static double MaxAbsDiff(float[] a, float[] b)
        {
            double m = 0;
            for (long i = 0; i < a.LongLength; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
            return m;
        }

        private static double Psnr(float[] reference, float[] actual)
        {
            double se = 0;
            for (long i = 0; i < reference.LongLength; i++)
            {
                double d = reference[i] - actual[i];
                se += d * d;
            }
            double mse = se / reference.LongLength;
            return mse <= 0 ? double.PositiveInfinity : 10.0 * Math.Log10(1.0 / mse);
        }

        private static string VaePath =>
            Path.Combine(Root ?? ".", "minimax_h3_video_vae_fp16.safetensors");

        [ModelFact(DirEnv)]
        public void VideoVaeDecodeMatchesTheUpstreamReference()
        {
            if (!Have("h3_video_latent_in.bin", "h3_video_pixels_out.bin") || !File.Exists(VaePath))
            {
                _output.WriteLine("[h3-vae] fixtures or weights absent; skipping");
                return;
            }

            // Reference latent is [B, C, T, H, W]; the decoder wants one token per
            // voxel with the channel axis contiguous, i.e. [T, H, W, C].
            int[] zs = ReadShape("h3_video_latent_in.bin");
            Assert.Equal(5, zs.Length);
            int c = zs[1], t = zs[2], h = zs[3], w = zs[4];
            float[] zchw = ReadF32("h3_video_latent_in.bin");
            var latent = new float[zchw.Length];
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            latent[(((long)ti * h + hi) * w + wi) * c + ci] =
                                zchw[(((long)ci * t + ti) * h + hi) * w + wi];

            using var vae = _backend.Video(VaePath);
            Assert.Equal(36, vae.NumBlocks);
            Assert.Equal(3072, vae.PatchDim);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] patches = vae.DecodeTokens(latent, t, h, w);
            sw.Stop();
            float[] actual = MiniMaxH3VideoVae.UnpackPatches(patches, t, h, w);

            float[] expected = ReadF32("h3_video_pixels_out.bin");
            Assert.Equal(expected.Length, actual.Length);

            double cos = Cosine(expected, actual);
            double maxDiff = MaxAbsDiff(expected, actual);
            _output.WriteLine($"[h3-vae] decode {t}x{h}x{w} -> {expected.Length} px in " +
                              $"{sw.Elapsed.TotalSeconds:F2}s  cos={cos:F6} maxAbsDiff={maxDiff:F5}");

            // The reference runs F32 throughout; we bind the shipped fp16 weights and
            // accumulate on the GPU, so exact equality is not the bar — direction is.
            Assert.True(cos > 0.999, $"cosine similarity {cos:F6} is too low");
        }

        [ModelFact(DirEnv)]
        public void VideoVaeRgbConversionMatchesTheReference()
        {
            if (!Have("h3_video_pixels_out.bin", "h3_video_rgb_out.bin")) return;

            int[] s = ReadShape("h3_video_pixels_out.bin");
            int frames = s[2], height = s[3], width = s[4];
            float[] pixels = ReadF32("h3_video_pixels_out.bin");
            MiniMaxH3VideoVae.ToRgb(pixels, frames, height, width);

            float[] expected = ReadF32("h3_video_rgb_out.bin");
            double psnr = Psnr(expected, pixels);
            _output.WriteLine($"[h3-vae] ImageNet denormalize PSNR={psnr:F1} dB");
            Assert.True(psnr > 80, $"denormalization PSNR {psnr:F1} dB is too low");
        }

        [ModelFact(DirEnv)]
        public void VideoVaeLatentStatisticsMatchTheCheckpoint()
        {
            if (!Have("h3_video_latents_mean.bin", "h3_video_latents_std.bin")
                || !File.Exists(VaePath)) return;

            using var vae = _backend.Video(VaePath);
            float[] mean = ReadF32("h3_video_latents_mean.bin");
            float[] std = ReadF32("h3_video_latents_std.bin");

            Assert.Equal(24, vae.LatentsMean.Length);
            Assert.Equal(24, vae.LatentsStd.Length);
            for (int i = 0; i < 24; i++)
            {
                Assert.Equal(mean[i], vae.LatentsMean[i], 3);
                Assert.Equal(std[i], vae.LatentsStd[i], 3);
            }
        }

        // ---- text encoder ---------------------------------------------------

        private static string TePath =>
            Path.Combine(Root ?? ".", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf");

        [ModelFact(DirEnv)]
        public void TextEncoderMatchesTheUpstreamReferenceForTheFirstLayers()
        {
            if (!Have("h3_te_embeddings.bin", "h3_te_layer1_out.bin") || !File.Exists(TePath))
            {
                _output.WriteLine("[h3-te] fixtures or weights absent; skipping");
                return;
            }
            string idsPath = Path.Combine(FixtureDir, "h3_te_token_ids.txt");
            if (!File.Exists(idsPath)) return;
            int[] ids = File.ReadAllText(idsPath).Split(',')
                .Select(x => int.Parse(x.Trim(), CultureInfo.InvariantCulture)).ToArray();

            using var te = _backend.Text(TePath,Root);
            _output.WriteLine("[h3-te] " + te.Config);

            // The published H3 encoder: 50 layers, no final norm, GQA 64/8.
            Assert.Equal(50, te.Config.NumLayers);
            Assert.Equal(5120, te.Config.Hidden);
            Assert.Equal(64, te.Config.Heads);
            Assert.Equal(8, te.Config.KvHeads);
            Assert.Equal(128, te.Config.HeadDim);
            Assert.Equal(25600, te.Config.Intermediate);
            Assert.False(te.Config.HasFinalNorm,
                "H3 truncates Qwen3-VL and drops the final norm; the DiT wants the raw state");

            // The embedding gather must match before the trunk can be blamed.
            float[] expectedEmb = ReadF32("h3_te_embeddings.bin");
            float[] actualEmb = te.Embed(ids);
            Assert.Equal(expectedEmb.Length, actualEmb.Length);
            double embCos = Cosine(expectedEmb, actualEmb);
            _output.WriteLine($"[h3-te] embedding gather cos={embCos:F6} " +
                              $"maxAbsDiff={MaxAbsDiff(expectedEmb, actualEmb):F6}");
            Assert.True(embCos > 0.99999, $"embedding cosine {embCos:F6} is too low");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] actual = te.Encode(ids, layerLimit: 2);
            sw.Stop();

            float[] expected = ReadF32("h3_te_layer1_out.bin");
            Assert.Equal(expected.Length, actual.Length);
            double cos = Cosine(expected, actual);
            _output.WriteLine($"[h3-te] 2 layers x {ids.Length} tokens in {sw.Elapsed.TotalSeconds:F2}s " +
                              $"cos={cos:F6} maxAbsDiff={MaxAbsDiff(expected, actual):F4}");
            Assert.True(cos > 0.999, $"cosine similarity {cos:F6} is too low");
        }

        [ModelFact(DirEnv)]
        public void TextEncoderTokenizesWithoutAChatTemplate()
        {
            if (!File.Exists(TePath) || !File.Exists(Path.Combine(Root, "vocab.json"))) return;
            using var te = _backend.Text(TePath,Root);

            // H3 adds no BOS and no EOS, and wraps the prompt in nothing at all.
            // Round-tripping the prompt is the cheapest proof of both.
            const string prompt = "a red fox trotting through falling snow, cinematic";
            var ids = te.Tokenize(prompt);
            Assert.NotEmpty(ids);
            string decoded = te.Tokenizer.Decode(ids);
            Assert.Equal(prompt, decoded);
            _output.WriteLine($"[h3-te] {ids.Count} tokens for {prompt.Length} chars, round-trip exact");

            // The vision placeholders the reference presentation relies on must resolve.
            Assert.True(te.Tokenizer.LookupToken("<|vision_start|>") > 0);
            Assert.True(te.Tokenizer.LookupToken("<|image_pad|>") > 0);
            Assert.True(te.Tokenizer.LookupToken("<|vision_end|>") > 0);
        }

        // ---- diffusion transformer ------------------------------------------

        private static string DitPath =>
            Path.Combine(Root ?? ".", "minimax_h3_fl2va_pruned-Q4_K.gguf");

        [ModelFact(DirEnv)]
        public void DitForwardMatchesTheUpstreamReferenceForTheFirstBlocks()
        {
            if (!Have("h3_dit_video_latent.bin", "h3_dit_video_out.bin", "h3_dit_audio_out.bin")
                || !File.Exists(DitPath))
            {
                _output.WriteLine("[h3-dit] fixtures or weights absent; skipping");
                return;
            }

            // The oracle's tiny request: 5 text tokens, a 2x4x4 latent, 3 audio latents.
            const int textLen = 5, latT = 2, latH = 4, latW = 4, audioT = 3;
            const float sigma = 0.5f;

            float[] videoLatent = ReadF32("h3_dit_video_latent.bin");   // [T][H][W][24]
            float[] audioLatent = ReadF32("h3_dit_audio_latent.bin");   // [2][T][32]
            float[] textHidden = ReadF32("h3_dit_text_hidden.bin");     // [5][5120]

            using var dit = _backend.Dit(DitPath);
            _output.WriteLine("[h3-dit] " + dit.Config);

            var shape = new MiniMaxH3Shape
            {
                Width = latW * 16, Height = latH * 16,
                Frames = MiniMaxH3Geometry.LatentFramesToVideoFrames(latT),
                Fps = 24,
                LatentWidth = latW, LatentHeight = latH, LatentFrames = latT,
                AudioLatentFrames = audioT,
            };
            var layout = MiniMaxH3Layout.Build(textLen, shape, sigma);
            Assert.Equal(19, layout.TokenCount);   // 5 text + 6 audio + 8 video

            float[] patches = MiniMaxH3Latents.PatchifyVideo(videoLatent, latT, latH, latW, 24);
            int videoCount = latT * (latH / 2) * (latW / 2);
            Assert.Equal((long)videoCount * dit.Config.VideoPatchDim, patches.LongLength);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (video, audio) = dit.Forward(
                patches, videoCount, audioLatent, audioT * 2,
                textHidden, textLen, layout, sigma, blockLimit: 2);
            sw.Stop();

            float[] expectedVideo = ReadF32("h3_dit_video_out.bin");
            float[] expectedAudio = ReadF32("h3_dit_audio_out.bin");
            double vcos = Cosine(expectedVideo, video);
            double acos = Cosine(expectedAudio, audio);
            _output.WriteLine($"[h3-dit] 2 blocks x {layout.TokenCount} tokens in " +
                              $"{sw.Elapsed.TotalSeconds:F2}s  video cos={vcos:F6} " +
                              $"maxAbsDiff={MaxAbsDiff(expectedVideo, video):F3}  " +
                              $"audio cos={acos:F6} maxAbsDiff={MaxAbsDiff(expectedAudio, audio):F3}");

            Assert.True(vcos > 0.999, $"video velocity cosine {vcos:F6} is too low");
            Assert.True(acos > 0.999, $"audio velocity cosine {acos:F6} is too low");
        }

        [ModelFact(DirEnv)]
        public void DenoiseLoopStatisticsAreInDistribution()
        {
            // A diagnostic, not a tight assertion: it prints the magnitude of every
            // stage so an out-of-distribution conditioning or an over-large velocity
            // is visible at a glance. Flow-matching velocities must be O(1); the
            // schedule's last step is ~0.65 in sigma, so a velocity of O(100) would
            // destroy the latent no matter how correct the rest is.
            if (!File.Exists(DitPath) || !File.Exists(TePath)) return;

            const string prompt = "a red fox trotting through falling snow, cinematic";
            float[] textHidden;
            int textLen;
            using (var te = _backend.Text(TePath,Root))
            {
                var ids = te.Tokenize(prompt);
                textLen = ids.Count;
                textHidden = te.Encode(ids);
                _output.WriteLine($"[probe] {textLen} tokens; text hidden rms={Rms(textHidden):F3} " +
                                  $"absmax={AbsMax(textHidden):F3}");
            }

            using var dit = _backend.Dit(DitPath);
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22);
            int videoCount = shape.VideoTokenCount, audioCount = shape.AudioTokenCount;
            var (video, audio) = MiniMaxH3Noise.Initial(shape, 1234);

            float[] sigmas = MiniMaxH3Scheduler.BuildSigmas(8);
            for (int step = 0; step < 8; step++)
            {
                float sigma = sigmas[step];
                var layout = MiniMaxH3Layout.Build(textLen, shape, sigma);
                float[] audioInput = MiniMaxH3Scheduler.AudioInputFromCarrier(audio, sigma);
                var (v, a) = dit.Forward(video, videoCount, audioInput, audioCount,
                                         textHidden, textLen, layout, sigma);
                MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(audioInput, a, sigma);
                _output.WriteLine(
                    $"[probe] step {step} sigma={sigma:F4} dt={sigmas[step + 1] - sigma:+0.0000} " +
                    $"| latent rms={Rms(video):F3} | v rms={Rms(v):F3} absmax={AbsMax(v):F2} " +
                    $"| audio v rms={Rms(a):F3}");
                MiniMaxH3Scheduler.EulerStep(video, v, sigma, sigmas[step + 1]);
                MiniMaxH3Scheduler.EulerStep(audio, a, sigma, sigmas[step + 1]);
            }
            _output.WriteLine($"[probe] final latent rms={Rms(video):F3} absmax={AbsMax(video):F2}");
            _output.WriteLine($"[probe] final AUDIO latent rms={Rms(audio):F4} absmax={AbsMax(audio):F3}");
            if (File.Exists(AudioVaePath))
            {
                using var av = _backend.Audio(AudioVaePath);
                _output.WriteLine($"[probe] audio latents_std mean=" +
                    $"{(av.LatentsStd is null ? double.NaN : av.LatentsStd.Average()):F3}");
                MiniMaxH3Scheduler.AudioOutputFromCarrierInPlace(audio);
                var track = av.DecodeStereo(audio, shape.AudioLatentFrames);
                _output.WriteLine($"[probe] decoded waveform rms={Rms(track.Channels[0]):F5} " +
                                  $"peak={AbsMax(track.Channels[0]):F5}");
            }
        }

        private static double Rms(float[] a)
        {
            double s = 0;
            for (long i = 0; i < a.LongLength; i++) s += (double)a[i] * a[i];
            return Math.Sqrt(s / a.LongLength);
        }

        private static double AbsMax(float[] a)
        {
            double m = 0;
            for (long i = 0; i < a.LongLength; i++) m = Math.Max(m, Math.Abs(a[i]));
            return m;
        }

        private static float Gauss(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        [ModelFact(DirEnv)]
        public void DecodesAReferenceLatentFromStableDiffusionCpp()
        {
            // The decisive DiT-vs-decode split: take a latent produced by the
            // reference implementation's own denoise loop and run it through THIS
            // VAE. A good picture means the decode path is right and any remaining
            // problem is upstream in the diffusion loop.
            string latentPath = Environment.GetEnvironmentVariable("TS_H3_REF_LATENT");
            if (string.IsNullOrEmpty(latentPath) || !File.Exists(latentPath) || !File.Exists(VaePath))
            {
                _output.WriteLine("[h3-ref] no reference latent (set TS_H3_REF_LATENT); skipping");
                return;
            }

            byte[] raw = File.ReadAllBytes(latentPath);
            int w = BitConverter.ToInt32(raw, 0), h = BitConverter.ToInt32(raw, 4);
            int t = BitConverter.ToInt32(raw, 8), c = BitConverter.ToInt32(raw, 12);
            var flat = new float[(raw.Length - 16) / 4];
            Buffer.BlockCopy(raw, 16, flat, 0, raw.Length - 16);
            _output.WriteLine($"[h3-ref] latent {w}x{h}x{t}x{c} ({flat.Length} floats)");
            Assert.Equal((long)w * h * t * c, flat.LongLength);

            // The reference stores [W,H,T,C] with W fastest; this decoder wants one
            // token per voxel in (t,h,w) order with the channel axis contiguous.
            var latent = new float[flat.Length];
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            latent[(((long)ti * h + hi) * w + wi) * c + ci] =
                                flat[(((long)ci * t + ti) * h + hi) * w + wi];

            using var vae = _backend.Video(VaePath);
            vae.DenormalizeLatent(latent);
            float[] pixels = vae.Decode(latent, t, h, w);
            int frames = t * 4, height = h * 16, width = w * 16;
            MiniMaxH3VideoVae.ToRgb(pixels, frames, height, width);

            // Write a few frames so the result can be looked at, not just measured.
            string outDir = Environment.GetEnvironmentVariable("TS_H3_REF_OUT");
            if (!string.IsNullOrEmpty(outDir))
            {
                Directory.CreateDirectory(outDir);
                long hw = (long)height * width;
                long plane = (long)frames * hw;
                foreach (int f in new[] { 3, 10, 19 })
                {
                    if (f >= frames) continue;
                    var chw = new float[3 * hw];
                    for (int ch = 0; ch < 3; ch++)
                        Array.Copy(pixels, ch * plane + f * hw, chw, ch * hw, hw);
                    TensorSharp.Models.QwenImage.ImageIO.SavePng(
                        Path.Combine(outDir, $"ref_latent_frame{f:D2}.png"),
                        TensorSharp.Models.QwenImage.RgbImage.FromPlanarChw(width, height, chw));
                }
                _output.WriteLine($"[h3-ref] wrote frames to {outDir}");
            }

            // Decode a 16x16-latent crop as its own call. The reference always decodes
            // 256 px tiles, and its RoPE coordinates are length-normalized over
            // whatever extent it is given -- so a crop tells us whether the decoder
            // needs that bounded extent or is happy with a whole frame at once.
            if (!string.IsNullOrEmpty(outDir) && h >= 16 && w >= 16)
            {
                const int cw = 16, chh = 16;
                var crop = new float[(long)t * chh * cw * c];
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < chh; hi++)
                        for (int wi = 0; wi < cw; wi++)
                            for (int ci = 0; ci < c; ci++)
                                crop[(((long)ti * chh + hi) * cw + wi) * c + ci] =
                                    latent[(((long)ti * h + hi) * w + wi) * c + ci];
                float[] cp = vae.DecodeTokens(crop, t, chh, cw);
                float[] cpx = MiniMaxH3VideoVae.UnpackPatches(cp, t, chh, cw);
                int cf = t * 4, chp = chh * 16, cwp = cw * 16;
                MiniMaxH3VideoVae.ToRgb(cpx, cf, chp, cwp);
                long chw2 = (long)chp * cwp;
                long cplane = (long)cf * chw2;
                var cchw = new float[3 * chw2];
                for (int ch2 = 0; ch2 < 3; ch2++)
                    Array.Copy(cpx, ch2 * cplane + 10 * chw2, cchw, ch2 * chw2, chw2);
                TensorSharp.Models.QwenImage.ImageIO.SavePng(
                    Path.Combine(outDir, "ref_latent_crop16.png"),
                    TensorSharp.Models.QwenImage.RgbImage.FromPlanarChw(cwp, chp, cchw));
                _output.WriteLine("[h3-ref] wrote 16x16-latent crop");
            }

            // A decoded natural image has structure: its per-frame variance should be
            // well below that of decoded noise, and the mean should sit mid-range.
            double mean = 0;
            for (long i = 0; i < pixels.LongLength; i++) mean += pixels[i];
            mean /= pixels.LongLength;
            _output.WriteLine($"[h3-ref] decoded {frames}x{height}x{width} mean={mean:F4}");
            Assert.InRange(mean, 0.05, 0.95);
        }

        [ModelFact(DirEnv)]
        public void TextConditioningMatchesStableDiffusionCpp()
        {
            // The reference's Qwen3-VL conditioning for the same prompt, dumped from
            // its own encoder. Tokenization, the 50-layer trunk and the "no final
            // norm" decision all have to agree for this to line up.
            string condPath = Environment.GetEnvironmentVariable("TS_H3_REF_COND");
            if (string.IsNullOrEmpty(condPath) || !File.Exists(condPath) || !File.Exists(TePath))
            {
                _output.WriteLine("[h3-cond] no reference conditioning; skipping");
                return;
            }
            byte[] raw = File.ReadAllBytes(condPath);
            int hidden = BitConverter.ToInt32(raw, 0), tokens = BitConverter.ToInt32(raw, 4);
            var expected = new float[(raw.Length - 8) / 4];
            Buffer.BlockCopy(raw, 8, expected, 0, raw.Length - 8);
            _output.WriteLine($"[h3-cond] reference {hidden}x{tokens} rms={Rms(expected):F3} " +
                              $"absmax={AbsMax(expected):F1}");

            const string prompt = "a red fox trotting through falling snow, cinematic";
            using var te = _backend.Text(TePath,Root);
            var ids = te.Tokenize(prompt);
            _output.WriteLine($"[h3-cond] tokenized to {ids.Count} tokens " +
                              $"(reference used {tokens})");
            Assert.Equal(tokens, ids.Count);

            float[] actual = te.Encode(ids);
            Assert.Equal(expected.Length, actual.Length);
            double cos = Cosine(expected, actual);
            _output.WriteLine($"[h3-cond] rms={Rms(actual):F3} absmax={AbsMax(actual):F1} " +
                              $"cos={cos:F6}");
            Assert.True(cos > 0.99, $"conditioning cosine {cos:F6} is too low");
        }

        [ModelFact(DirEnv)]
        public void DitStepMatchesStableDiffusionCpp()
        {
            // The decisive DiT comparison: identical packed latent, identical
            // conditioning, identical timestep -> the velocities must agree.
            string dumpDir = Environment.GetEnvironmentVariable("TS_H3_REF_DUMP");
            if (string.IsNullOrEmpty(dumpDir) ||
                !File.Exists(Path.Combine(dumpDir, "h3_step_x.bin")) || !File.Exists(DitPath))
            {
                _output.WriteLine("[h3-step] no reference step dump; skipping");
                return;
            }

            var (xShape, packedIn) = ReadShaped4(Path.Combine(dumpDir, "h3_step_x.bin"));
            var (oShape, packedOut) = ReadShaped4(Path.Combine(dumpDir, "h3_step_out.bin"));
            int w = xShape[0], h = xShape[1], t = xShape[2], ch = xShape[3];
            _output.WriteLine($"[h3-step] packed {w}x{h}x{t}x{ch}");
            Assert.Equal(xShape, oShape);

            int audioLen = 0;
            float timestep = 1000f;
            foreach (string line in File.ReadAllLines(Path.Combine(dumpDir, "h3_step_meta.txt")))
            {
                if (line.StartsWith("audio_length=")) audioLen = int.Parse(line[13..]);
                if (line.StartsWith("timestep="))
                    timestep = float.Parse(line[9..], CultureInfo.InvariantCulture);
            }
            float sigma = Math.Clamp(timestep / 1000f, 1e-6f, 1f);
            _output.WriteLine($"[h3-step] audioLength={audioLen} sigma={sigma:F4}");

            byte[] condRaw = File.ReadAllBytes(Path.Combine(dumpDir, "h3_cond.bin"));
            int textLen = BitConverter.ToInt32(condRaw, 4);
            var textHidden = new float[(condRaw.Length - 8) / 4];
            Buffer.BlockCopy(condRaw, 8, textHidden, 0, condRaw.Length - 8);

            using var dit = _backend.Dit(DitPath);
            var shape = new MiniMaxH3Shape
            {
                Width = w * 16, Height = h * 16,
                Frames = MiniMaxH3Geometry.LatentFramesToVideoFrames(t), Fps = 24,
                LatentWidth = w, LatentHeight = h, LatentFrames = t,
                AudioLatentFrames = audioLen,
            };
            var layout = MiniMaxH3Layout.Build(textLen, shape, sigma);

            float[] videoPatches = PackedToVideoPatches(packedIn, w, h, t);
            float[] audioLatent = PackedToAudioLatent(packedIn, w, h, t, audioLen);
            float[] audioInput = MiniMaxH3Scheduler.AudioInputFromCarrier(audioLatent, sigma);

            var (vVel, aVel) = dit.Forward(videoPatches, shape.VideoTokenCount,
                                           audioInput, shape.AudioTokenCount,
                                           textHidden, textLen, layout, sigma);
            MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(audioInput, aVel, sigma);

            float[] expectedVideo = PackedToVideoLatent(packedOut, w, h, t);
            float[] actualVideo = MiniMaxH3Latents.UnpatchifyVideo(vVel, t, h, w, 24);
            double vcos = Cosine(expectedVideo, actualVideo);

            float[] expectedAudio = PackedToAudioLatent(packedOut, w, h, t, audioLen);
            double acos = Cosine(expectedAudio, aVel);

            _output.WriteLine($"[h3-step] video cos={vcos:F6} rms ref={Rms(expectedVideo):F4} " +
                              $"mine={Rms(actualVideo):F4}");
            _output.WriteLine($"[h3-step] audio cos={acos:F6} rms ref={Rms(expectedAudio):F4} " +
                              $"mine={Rms(aVel):F4}");
            Assert.True(vcos > 0.99, $"video velocity cosine {vcos:F6} is too low");
            Assert.True(acos > 0.99, $"audio velocity cosine {acos:F6} is too low");
        }

        private static (int[] Shape, float[] Data) ReadShaped4(string path)
        {
            byte[] raw = File.ReadAllBytes(path);
            var shape = new int[4];
            for (int i = 0; i < 4; i++) shape[i] = BitConverter.ToInt32(raw, i * 4);
            var data = new float[(raw.Length - 16) / 4];
            Buffer.BlockCopy(raw, 16, data, 0, raw.Length - 16);
            return (shape, data);
        }

        // The reference packs [W,H,T,C] with W fastest, so channel c is the
        // contiguous run [c*W*H*T, (c+1)*W*H*T).
        private static float[] PackedToVideoLatent(float[] packed, int w, int h, int t)
        {
            const int vc = 24;
            long spatial = (long)w * h * t;
            var latent = new float[spatial * vc];
            for (int c = 0; c < vc; c++)
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            latent[(((long)ti * h + hi) * w + wi) * vc + c] =
                                packed[c * spatial + ((long)ti * h + hi) * w + wi];
            return latent;
        }

        private static float[] PackedToVideoPatches(float[] packed, int w, int h, int t) =>
            MiniMaxH3Latents.PatchifyVideo(PackedToVideoLatent(packed, w, h, t), t, h, w, 24);

        // The audio latent is flat-copied straight after the 24 video channels, and is
        // shaped {audioLength, 2, 32} with audioLength fastest — the transpose of the
        // channel-contiguous token layout the transformer wants.
        private static float[] PackedToAudioLatent(float[] packed, int w, int h, int t, int audioLen)
        {
            const int vc = 24, ac = 32, stereo = 2;
            long spatial = (long)w * h * t;
            long baseOff = vc * spatial;
            var outp = new float[(long)audioLen * stereo * ac];
            for (int c = 0; c < ac; c++)
                for (int s = 0; s < stereo; s++)
                    for (int i = 0; i < audioLen; i++)
                        outp[((long)s * audioLen + i) * ac + c] =
                            packed[baseOff + (long)c * (stereo * audioLen) + (long)s * audioLen + i];
            return outp;
        }

        [ModelFact(DirEnv)]
        public void VideoVaeEncodeMatchesTheUpstreamReference()
        {
            if (!Have("h3_video_encode_image.bin", "h3_video_encode_latent.bin")
                || !File.Exists(VaePath))
            {
                _output.WriteLine("[h3-enc] fixtures or weights absent; skipping");
                return;
            }
            int[] ishape = ReadShape("h3_video_encode_image.bin");   // [1,3,1,H,W]
            int h = ishape[3], w = ishape[4];
            float[] planar = ReadF32("h3_video_encode_image.bin");   // channel-planar [0,1]

            long hw = (long)h * w;
            var px = new float[3 * hw];
            for (long q = 0; q < hw; q++)
                for (int c = 0; c < 3; c++)
                    px[q * 3 + c] = planar[c * hw + q];
            var image = new TensorSharp.Models.QwenImage.RgbImage(w, h, px);

            using var vae = _backend.Video(VaePath);
            Assert.True(vae.HasEncoder, "this VAE checkpoint should carry encoder weights");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] tokens = vae.EncodeFrame(image);
            sw.Stop();

            // Reference is [1,24,1,lh,lw]; ours is [lh][lw][24] token order.
            int lh = h / 16, lw = w / 16;
            float[] expected = ReadF32("h3_video_encode_latent.bin");
            Assert.Equal(expected.Length, tokens.Length);
            var reordered = new float[expected.Length];
            for (int c = 0; c < 24; c++)
                for (int y = 0; y < lh; y++)
                    for (int x = 0; x < lw; x++)
                        reordered[((long)c * lh + y) * lw + x] =
                            tokens[((long)y * lw + x) * 24 + c];

            double cos = Cosine(expected, reordered);
            _output.WriteLine($"[h3-enc] encode {w}x{h} -> {lw}x{lh}x24 in {sw.Elapsed.TotalSeconds:F2}s " +
                              $"cos={cos:F6} maxAbsDiff={MaxAbsDiff(expected, reordered):F4}");
            Assert.True(cos > 0.999, $"encoder cosine {cos:F6} is too low");
        }

        private static string AudioVaePath =>
            Path.Combine(Root ?? ".", "minimax_h3_audio_vae_fp32.safetensors");

        [ModelFact(DirEnv)]
        public void AudioVaeDecodeMatchesTheUpstreamReference()
        {
            if (!Have("h3_audio_latent_in.bin", "h3_audio_wave_out.bin")
                || !File.Exists(AudioVaePath))
            {
                _output.WriteLine("[h3-audio] fixtures or weights absent; skipping");
                return;
            }
            // Reference latent is [2, 32, T] (stereo as a batch of two mono planes).
            int[] zs = ReadShape("h3_audio_latent_in.bin");
            int stereo = zs[0], ch = zs[1], t = zs[2];
            float[] z = ReadF32("h3_audio_latent_in.bin");

            using var vae = _backend.Audio(AudioVaePath);
            Assert.Equal(7, vae.NumStages);
            Assert.Equal(t * 800, MiniMaxH3AudioVae.SamplesFor(t));

            float[] expected = ReadF32("h3_audio_wave_out.bin");   // [2, 1, samples]
            int samples = MiniMaxH3AudioVae.SamplesFor(t);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int s = 0; s < stereo; s++)
            {
                var plane = new float[(long)ch * t];
                Array.Copy(z, (long)s * ch * t, plane, 0, plane.Length);
                float[] wave = vae.DecodeMono(plane, t);
                Assert.Equal(samples, wave.Length);

                var want = new float[samples];
                Array.Copy(expected, (long)s * samples, want, 0, samples);
                double cos = Cosine(want, wave);
                _output.WriteLine($"[h3-audio] plane {s}: {t} latents -> {samples} samples " +
                                  $"cos={cos:F6} maxAbsDiff={MaxAbsDiff(want, wave):F5}");
                Assert.True(cos > 0.999, $"audio plane {s} cosine {cos:F6} is too low");
            }
            sw.Stop();
            _output.WriteLine($"[h3-audio] stereo decode in {sw.Elapsed.TotalSeconds:F2}s");
        }

        [ModelTheory(DirEnv)]
        [InlineData(5, 2)]
        [InlineData(17, 5)]
        public void VideoVaeMultiFrameEncodeMatchesTheUpstreamReference(int frames, int latentFrames)
        {
            // Fixtures come from MiniMax-H3's OWN PyTorch encoder (video_vae/klvae.py,
            // AutoencoderKLLegacy) on the checkpoint's source_config. 17 pixel frames
            // must collapse to 5 latent frames and 5 to 2 — that mapping is the whole
            // reason a reference CLIP needs the real 3-D network rather than the
            // single-frame reduction the keyframe path uses.
            string inName = $"h3_video_enc3d_in_t{frames}.bin";
            string outName = $"h3_video_enc3d_mean_t{frames}.bin";
            if (!Have(inName, outName) || !File.Exists(VaePath))
            {
                _output.WriteLine("[h3-enc3d] fixtures or weights absent; skipping");
                return;
            }

            int[] inShape = ReadShape(inName);        // 1, 3, T, H, W
            float[] pixels = ReadF32(inName);
            int t = inShape[2], h = inShape[3], w = inShape[4];
            Assert.Equal(frames, t);

            // The fixture is planar CHW per clip; RgbImage wants interleaved per frame.
            var clip = new List<TensorSharp.Models.QwenImage.RgbImage>(t);
            long plane = (long)h * w;
            for (int fi = 0; fi < t; fi++)
            {
                var rgb = new float[plane * 3];
                for (int c = 0; c < 3; c++)
                    for (long i = 0; i < plane; i++)
                        rgb[i * 3 + c] = pixels[((long)c * t + fi) * plane + i];
                clip.Add(new TensorSharp.Models.QwenImage.RgbImage(w, h, rgb));
            }

            using var vae = _backend.Video(VaePath);
            Assert.Equal(latentFrames, MiniMaxH3VideoVae.ChunkLatentFrames(frames));

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] tokens = vae.EncodeChunk(clip);
            sw.Stop();

            // Reference is [1, 24, Tl, lh, lw]; ours is [Tl][lh][lw][24].
            int lh = h / MiniMaxH3VideoVae.PatchSpatial, lw = w / MiniMaxH3VideoVae.PatchSpatial;
            float[] expected = ReadF32(outName);
            var want = new float[tokens.Length];
            for (int c = 0; c < MiniMaxH3VideoVae.LatentChannels; c++)
                for (int ti = 0; ti < latentFrames; ti++)
                    for (int y = 0; y < lh; y++)
                        for (int x = 0; x < lw; x++)
                            want[(((long)ti * lh + y) * lw + x) * MiniMaxH3VideoVae.LatentChannels + c] =
                                expected[(((long)c * latentFrames + ti) * lh + y) * lw + x];

            double cos = Cosine(want, tokens);
            _output.WriteLine($"[h3-enc3d] {t}x{h}x{w} -> {latentFrames} latent frames in " +
                              $"{sw.Elapsed.TotalSeconds:F2}s  cos={cos:F6} " +
                              $"maxAbsDiff={MaxAbsDiff(want, tokens):F5}");
            Assert.True(cos > 0.999, $"3-D encode cosine {cos:F6} is too low");
        }

        [ModelFact(DirEnv)]
        public void SingleFrameEncodeAgreesBetweenThe2DReductionAndThe3DPath()
        {
            // A one-frame clip is causally padded with two ZERO frames, so only the
            // last temporal slice of each kernel contributes and the 3-D network must
            // reproduce the 2-D reduction EXACTLY. That makes this a sharp test of the
            // 3-D kernel layout alone, with the temporal arithmetic held at identity.
            if (!Have("h3_video_enc3d_in_t5.bin") || !File.Exists(VaePath)) return;

            int[] inShape = ReadShape("h3_video_enc3d_in_t5.bin");
            float[] pixels = ReadF32("h3_video_enc3d_in_t5.bin");
            int t = inShape[2], h = inShape[3], w = inShape[4];
            long plane = (long)h * w;
            var rgb = new float[plane * 3];
            for (int c = 0; c < 3; c++)
                for (long i = 0; i < plane; i++)
                    rgb[i * 3 + c] = pixels[((long)c * t + 0) * plane + i];
            var frame = new TensorSharp.Models.QwenImage.RgbImage(w, h, rgb);

            using var vae = _backend.Video(VaePath);
            float[] flat = vae.EncodeFrame(frame);
            float[] cubic = vae.EncodeChunk(new[] { frame });

            Assert.Equal(flat.Length, cubic.Length);
            double cos = Cosine(flat, cubic);
            _output.WriteLine($"[h3-enc3d] single frame 2-D vs 3-D cos={cos:F6} " +
                              $"maxAbsDiff={MaxAbsDiff(flat, cubic):F6}");
            Assert.True(cos > 0.9999, $"the 3-D path disagrees with its own 2-D reduction ({cos:F6})");
        }

        [ModelFact(DirEnv)]
        public void AudioVaeEncodeMatchesTheUpstreamReference()
        {
            // Fixtures come from MiniMax-H3's OWN PyTorch audio VAE (audio_vae/
            // dac_audio_vae.py) run on a deterministic waveform: 4000 samples of a
            // stereo pair, which is exactly 5 latent frames at the 800-sample hop.
            if (!Have("h3_audio_enc_wave_in.bin", "h3_audio_enc_mean.bin")
                || !File.Exists(AudioVaePath))
            {
                _output.WriteLine("[h3-audio-enc] fixtures or weights absent; skipping");
                return;
            }

            int[] waveShape = ReadShape("h3_audio_enc_wave_in.bin");     // planes, 1, samples
            float[] wave = ReadF32("h3_audio_enc_wave_in.bin");
            int planes = waveShape[0], samples = waveShape[2];

            int[] meanShape = ReadShape("h3_audio_enc_mean.bin");        // planes, 32, frames
            float[] expected = ReadF32("h3_audio_enc_mean.bin");
            int frames = meanShape[2];

            using var vae = _backend.Audio(AudioVaePath);
            Assert.True(vae.HasEncoder, "the audio VAE checkpoint should carry the DAC encoder");
            Assert.Equal(frames, MiniMaxH3AudioVae.FramesFor(samples));

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int p = 0; p < planes; p++)
            {
                var plane = new float[samples];
                Array.Copy(wave, (long)p * samples, plane, 0, samples);

                float[] latent = vae.EncodeMono(plane);
                var want = new float[(long)MiniMaxH3AudioVae.LatentChannels * frames];
                Array.Copy(expected, (long)p * want.Length, want, 0, want.Length);

                double cos = Cosine(want, latent);
                _output.WriteLine($"[h3-audio-enc] plane {p}: {samples} samples -> {frames} frames " +
                                  $"cos={cos:F6} maxAbsDiff={MaxAbsDiff(want, latent):F5}");
                Assert.True(cos > 0.999, $"encode plane {p} cosine {cos:F6} is too low");
            }
            sw.Stop();
            _output.WriteLine($"[h3-audio-enc] encoded {planes} planes in {sw.Elapsed.TotalSeconds:F2}s");
        }

        [ModelFact(DirEnv)]
        public void AudioEncodeDecodeRoundTripsThroughTheWhitenedLatentSpace()
        {
            // The encoder and decoder must agree about which space the denoiser works
            // in. Normalizing after encode and denormalizing before decode has to be a
            // no-op end to end; getting the direction wrong is inaudible as anything
            // but a level change, which is why it is asserted rather than eyeballed.
            if (!Have("h3_audio_enc_wave_in.bin") || !File.Exists(AudioVaePath)) return;

            int[] waveShape = ReadShape("h3_audio_enc_wave_in.bin");
            float[] wave = ReadF32("h3_audio_enc_wave_in.bin");
            int samples = waveShape[2];
            var plane = new float[samples];
            Array.Copy(wave, plane, samples);

            using var vae = _backend.Audio(AudioVaePath);
            if (!vae.HasEncoder) return;
            int frames = MiniMaxH3AudioVae.FramesFor(samples);

            float[] native = vae.EncodeMono(plane);
            var whitened = (float[])native.Clone();
            vae.NormalizePlane(whitened, frames);
            var restored = (float[])whitened.Clone();
            vae.DenormalizePlane(restored, frames);

            double maxDiff = MaxAbsDiff(native, restored);
            _output.WriteLine($"[h3-audio-enc] whiten/unwhiten round trip maxAbsDiff={maxDiff:E3}");
            Assert.True(maxDiff < 1e-3, $"normalize/denormalize is not an identity ({maxDiff:E3})");

            // And the whitened latent should actually be whitened: the encoder's native
            // output has the VAE's own spread, roughly latents_std.
            double nativeRms = Rms(native), whitenedRms = Rms(whitened);
            _output.WriteLine($"[h3-audio-enc] native rms={nativeRms:F4} whitened rms={whitenedRms:F4}");
            Assert.True(whitenedRms < nativeRms,
                "whitening should shrink the latent; latents_std averages about 1.9");
        }

        [ModelFact(DirEnv)]
        public void PipelineAudioLatentIsDenormalizedBeforeDecoding()
        {
            // The denoiser emits a WHITENED audio latent and the decoder wants the
            // VAE's own scale, so the pipeline entry point has to undo the whitening
            // exactly as the video path does. Skipping it is not audible as silence or
            // noise — it is a track that is spectrally plausible and simply wrong, so
            // this compares against the reference implementation's own decode of the
            // SAME latent rather than against a plausibility band.
            //
            // The fixture pair is one stable-diffusion.cpp run: SD_DUMP_AUDIO_LATENT
            // (the latent handed to its audio VAE) and SD_DUMP_AUDIO_WAVE (what came
            // back out of it).
            if (!Have("h3_audio_pipeline_latent.bin", "h3_audio_pipeline_wave.bin")
                || !File.Exists(AudioVaePath))
            {
                _output.WriteLine("[h3-audio] pipeline fixtures absent; skipping");
                return;
            }

            int[] latentShape = ReadShape("h3_audio_pipeline_latent.bin");   // T, 2, 32
            float[] flat = ReadF32("h3_audio_pipeline_latent.bin");
            int t = latentShape[0], stereo = latentShape[1], ch = latentShape[2];

            // Reference order is [32][2][T] with T fastest; ours is [channel][t][32].
            var tokens = new float[(long)stereo * t * ch];
            for (int c = 0; c < ch; c++)
                for (int s2 = 0; s2 < stereo; s2++)
                    for (int i = 0; i < t; i++)
                        tokens[((long)s2 * t + i) * ch + c] =
                            flat[(long)c * (stereo * t) + (long)s2 * t + i];

            int[] waveShape = ReadShape("h3_audio_pipeline_wave.bin");       // samples, channels
            float[] expected = ReadF32("h3_audio_pipeline_wave.bin");
            int samples = waveShape[0];

            using var vae = _backend.Audio(AudioVaePath);
            var track = vae.DecodeStereo(tokens, t);
            Assert.Equal(samples, track.SampleCount);

            // The dump is ggml-ordered, so samples are the FASTEST axis: the file is
            // one plane per channel, not interleaved frames.
            for (int c = 0; c < track.ChannelCount; c++)
            {
                var want = new float[samples];
                for (int i = 0; i < samples; i++) want[i] = expected[(long)c * samples + i];
                double cos = Cosine(want, track.Channels[c]);
                double gain = Rms(track.Channels[c]) / Math.Max(1e-12, Rms(want));
                _output.WriteLine($"[h3-audio] pipeline channel {c}: cos={cos:F5} " +
                                  $"rms={Rms(track.Channels[c]):F5} (reference {Rms(want):F5}, " +
                                  $"gain {gain:F4}x)");
                // Level is the whole point: dropping the denormalization leaves the
                // waveform at about 0.81x, which cosine alone would not catch.
                Assert.InRange(gain, 0.95, 1.05);
                Assert.True(cos > 0.95, $"channel {c} cosine {cos:F5} is too low");
            }
        }

        [Fact]
        public void VideoPatchifyRoundTrips()
        {
            const int t = 3, h = 6, w = 8, c = 24;
            var latent = new float[t * h * w * c];
            for (int i = 0; i < latent.Length; i++) latent[i] = i;
            float[] tokens = MiniMaxH3Latents.PatchifyVideo(latent, t, h, w, c);
            Assert.Equal((long)t * (h / 2) * (w / 2) * (c * 4), tokens.LongLength);
            float[] back = MiniMaxH3Latents.UnpatchifyVideo(tokens, t, h, w, c);
            Assert.Equal(latent.Length, back.Length);
            for (int i = 0; i < latent.Length; i++) Assert.Equal(latent[i], back[i]);
        }

        [Fact]
        public void TextEncoderRopeSharesAnglesAcrossHalvesAndStartsAtIdentity()
        {
            var (cos, sin) = MiniMaxH3TextEncoder.BuildRope(seq: 4, headDim: 128, theta: 5_000_000f);
            int hd = 128, half = hd / 2;
            Assert.Equal(hd * 4, cos.Length);

            for (int pos = 0; pos < 4; pos++)
                for (int j = 0; j < half; j++)
                {
                    long b = (long)pos * hd;
                    Assert.Equal(cos[b + j], cos[b + j + half], 6);
                    Assert.Equal(sin[b + j], sin[b + j + half], 6);
                }

            // Position 0 must be the identity rotation.
            for (int j = 0; j < hd; j++)
            {
                Assert.Equal(1f, cos[j], 6);
                Assert.Equal(0f, sin[j], 6);
            }
        }

        [Fact]
        public void RopeTablesDuplicateAnglesAcrossTheRotatingHalves()
        {
            // The 48 rotated dims are 3 axes x 8 frequencies, tiled twice, so dim j
            // and dim j+24 must share an angle — that is what makes rotate-half with
            // pairs (j, j+24) a rotation rather than a shear.
            var (cos, sin) = MiniMaxH3VideoVae.BuildRope(2, 3, 4, suffixTokens: 5);
            int rot = MiniMaxH3VideoVae.RotDim, half = rot / 2;
            int seq = 2 * 3 * 4 + 5;
            Assert.Equal(rot * seq, cos.Length);

            for (int token = 0; token < seq; token++)
                for (int j = 0; j < half; j++)
                {
                    long b = (long)token * rot;
                    Assert.Equal(cos[b + j], cos[b + j + half], 6);
                    Assert.Equal(sin[b + j], sin[b + j + half], 6);
                }

            // Suffix tokens sit at position 0, i.e. the identity rotation.
            for (int token = 2 * 3 * 4; token < seq; token++)
                for (int j = 0; j < rot; j++)
                {
                    long b = (long)token * rot;
                    Assert.Equal(1f, cos[b + j], 6);
                    Assert.Equal(0f, sin[b + j], 6);
                }
        }

        [Fact]
        public void UnpackPatchesIsTheInverseOfTheDepthToSpaceLayout()
        {
            // Feed each patch element its own destination index, then check every
            // pixel landed where the reference's permute would have put it.
            const int t = 2, h = 2, w = 3;
            int patchDim = 3 * 4 * 16 * 16;
            int frames = t * 4, height = h * 16, width = w * 16;
            var patches = new float[(long)t * h * w * patchDim];
            for (int ti = 0; ti < t; ti++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++)
                        for (int p = 0; p < patchDim; p++)
                        {
                            int c = p / (4 * 16 * 16);
                            int rem = p % (4 * 16 * 16);
                            int kt = rem / (16 * 16), kh = (rem / 16) % 16, kw = rem % 16;
                            long dst = (((long)c * frames + (ti * 4 + kt)) * height + (hi * 16 + kh))
                                       * width + (wi * 16 + kw);
                            patches[(((long)ti * h + hi) * w + wi) * patchDim + p] = dst;
                        }

            float[] pixels = MiniMaxH3VideoVae.UnpackPatches(patches, t, h, w);
            Assert.Equal((long)3 * frames * height * width, pixels.LongLength);
            for (long i = 0; i < pixels.LongLength; i++)
                Assert.Equal(i, (long)pixels[i]);
        }
    }
}
