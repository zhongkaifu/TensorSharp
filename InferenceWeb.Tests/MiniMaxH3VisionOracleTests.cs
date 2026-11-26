// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Parity for the Qwen3-VL vision tower against stable-diffusion.cpp, which is the
// reference implementation the MiniMax-H3 GGUF checkpoints were produced for.
//
// Fixtures come from an sd.cpp run patched to dump SD_DUMP_H3_VISION (the four
// tower outputs) and SD_DUMP_H3_VISION_PIXELS (the exact resized, [-1,1] pixels
// it fed them). Feeding the same pixels is what makes this a test of the tower
// rather than of two independently-written resize routines.
//
//   export TS_MINIMAX_H3_DIR=~/work/models/minimax-h3
//   export TS_TEST_GGML_BACKEND=metal
using System;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3VisionOracleTests : IDisposable
    {
        private const string DirEnv = "TS_MINIMAX_H3_DIR";
        private readonly ITestOutputHelper _output;
        private readonly MiniMaxH3OracleBackend _backend;

        public MiniMaxH3VisionOracleTests(ITestOutputHelper output)
        { _output=output; _backend=new MiniMaxH3OracleBackend(); _output.WriteLine($"[h3-oracle] backend={_backend.Backend}"); }
        public void Dispose() => _backend.Dispose();

        private static string Root => Environment.GetEnvironmentVariable(DirEnv);
        private static string FixtureDir => Root == null ? null : Path.Combine(Root, "fixtures");
        private static string TePath =>
            Path.Combine(Root ?? ".", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf");

        private static bool Have(params string[] names)
        {
            if (string.IsNullOrEmpty(Root)) return false;
            return names.All(n => File.Exists(Path.Combine(FixtureDir, n)));
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
                .Split(',').Select(s => int.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();

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

        /// <summary>The whole Ref2VA conditioning path — reference resize, prompt
        /// presentation, vision tower, image-embedding substitution, DeepStack
        /// injection and interleaved M-RoPE — against the same run of the reference
        /// implementation. Anything wrong in any of those shows up here as a token
        /// mismatch or a collapsed cosine.</summary>
        [ModelFact(DirEnv)]
        public void ReferenceConditioningMatchesTheUpstreamReference()
        {
            if (!Have("h3_vision_pixels.bin", "h3_ref_cond.bin", "h3_ref_prompt_ids.txt")
                || !File.Exists(TePath))
            {
                _output.WriteLine("[h3-ref] fixtures or weights absent; skipping");
                return;
            }

            RgbImage image = LoadFixtureImage();
            // The fixture pixels are what the reference fed its vision tower, which is
            // the output of its reference resize — so the resize rule must reproduce
            // those exact dimensions from the source photo at the same canvas.
            _output.WriteLine($"[h3-ref] reference image {image.Width}x{image.Height}");

            int[] expectedIds = File.ReadAllText(Path.Combine(FixtureDir, "h3_ref_prompt_ids.txt"))
                .Split(',').Select(x => int.Parse(x.Trim(), CultureInfo.InvariantCulture)).ToArray();

            using var te = _backend.Text(TePath,Root);
            var (prompt, images) = MiniMaxH3Requests.BuildReferencePrompt(
                te,
                new[]
                {
                    new MiniMaxH3Requests.MiniMaxH3Reference
                    {
                        Kind = MiniMaxH3ReferenceKind.Image,
                        Frames = new List<RgbImage> { image },
                        Presented = new List<RgbImage> { image },
                        Timestamps = new List<float> { 0f },
                    },
                },
                "two people talking");
            _output.WriteLine($"[h3-ref] prompt = {prompt.Length} chars, " +
                              $"image span at {images[0].TokenIndex} x {images[0].Vision.TokenCount}");

            var ids = te.Tokenize(prompt);
            Assert.Equal(expectedIds.Length, ids.Count);
            Assert.Equal(expectedIds, ids.ToArray());

            float[] expected = ReadF32("h3_ref_cond.bin");
            float[] actual = te.Encode(ids, images);
            Assert.Equal(expected.Length, actual.Length);
            double cos = Cosine(expected, actual);
            _output.WriteLine($"[h3-ref] conditioning cos={cos:F6}");
            // Looser than the single-network checks on purpose: this is the ONLY
            // assertion that runs the full 50-layer trunk, so it compounds whatever
            // per-layer difference the backend's Q4_K matmul has against the
            // reference's. Metal lands at 0.9997 and CUDA at 0.9910 from a per-2-layer
            // 0.99985 — and the two backends' generated frames are visually
            // indistinguishable at the same seed. The tight bounds live on the
            // vision tower and the 2-layer trunk, where drift cannot hide.
            Assert.True(cos > 0.98, $"conditioning cosine {cos:F6} is too low");

            // Without the DeepStack taps the trunk still runs and still produces
            // plausible-looking numbers, so prove they actually changed the answer
            // rather than trusting that wiring them up did something.
            float[] withoutDeepStack = te.Encode(ids);
            double plainCos = Cosine(expected, withoutDeepStack);
            _output.WriteLine($"[h3-ref] same prompt with no image conditioning cos={plainCos:F6}");
            Assert.True(plainCos < 0.99,
                "dropping the image embeddings and DeepStack taps should change the result");
        }

        /// <summary>The Ref2VA resize rule: references are scaled DOWN to the generated
        /// clip's area, keep their aspect ratio, and land on the VAE's 32-pixel grid.</summary>
        [ModelFact(DirEnv)]
        public void ReferenceResizeReproducesTheUpstreamDimensions()
        {
            if (!Have("h3_vision_pixels.bin")) return;
            string source = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "Downloads", "card.jpeg");
            if (!File.Exists(source))
            {
                _output.WriteLine("[h3-ref] source photo absent; skipping");
                return;
            }
            RgbImage expected = LoadFixtureImage();
            var p = new TensorSharp.Models.Video.VideoGenerationParams
            {
                ReferenceImagePaths = new List<string> { source },
            };
            // The fixture was produced by the reference at a 256x256 canvas.
            var resized = MiniMaxH3Requests.LoadReferenceImages(p, 256, 256);
            Assert.Single(resized);
            Assert.Equal(expected.Width, resized[0].Width);
            Assert.Equal(expected.Height, resized[0].Height);
        }

        // The vision fixture holds planar CHW pixels already mapped to [-1,1] and
        // already clamped, so undoing the map reproduces exactly what the reference
        // implementation fed its own tower.
        private static RgbImage LoadFixtureImage()
        {
            int[] shape = ReadShape("h3_vision_pixels.bin");   // C, H, W
            float[] chw = ReadF32("h3_vision_pixels.bin");
            int h = shape[1], w = shape[2];
            var pixels = new float[(long)h * w * 3];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        pixels[((long)y * w + x) * 3 + c] = (chw[((long)c * h + y) * w + x] + 1f) * 0.5f;
            return new RgbImage(w, h, pixels);
        }

        [ModelFact(DirEnv)]
        public void VisionTowerMatchesTheUpstreamReference()
        {
            if (!Have("h3_vision_pixels.bin", "h3_vision_merged.bin", "h3_vision_ds0.bin",
                      "h3_vision_ds1.bin", "h3_vision_ds2.bin") || !File.Exists(TePath))
            {
                _output.WriteLine("[h3-vision] fixtures or weights absent; skipping");
                return;
            }

            RgbImage image = LoadFixtureImage();
            int h = image.Height, w = image.Width;

            using var te = _backend.Text(TePath,Root);
            Assert.True(te.HasVision, "the Qwen3-VL checkpoint should carry the vision tower");
            var vision = te.Vision;
            _output.WriteLine($"[h3-vision] blocks={vision.NumBlocks} dim={vision.Dim} " +
                              $"heads={vision.Heads} out={vision.OutDim}");
            Assert.Equal(1152, vision.Dim);
            Assert.Equal(5120, vision.OutDim);
            Assert.Equal(27, vision.NumBlocks);

            // The image is already on the 32-pixel grid, so nothing should be resampled
            // and the parity below stays a pure test of the tower.
            RgbImage fitted = MiniMaxH3VisionEncoder.FitForVision(image);
            Assert.Equal(w, fitted.Width);
            Assert.Equal(h, fitted.Height);

            var actual = vision.Encode(image);
            int[] mergedShape = ReadShape("h3_vision_merged.bin");   // tokens, dim
            Assert.Equal(mergedShape[0], actual.TokenCount);
            Assert.Equal(h / MiniMaxH3VisionEncoder.PatchSize, actual.GridHeight);
            Assert.Equal(w / MiniMaxH3VisionEncoder.PatchSize, actual.GridWidth);

            var expected = new[]
            {
                ("merged", ReadF32("h3_vision_merged.bin"), actual.Merged),
                ("deepstack[0]", ReadF32("h3_vision_ds0.bin"), actual.DeepStack[0]),
                ("deepstack[1]", ReadF32("h3_vision_ds1.bin"), actual.DeepStack[1]),
                ("deepstack[2]", ReadF32("h3_vision_ds2.bin"), actual.DeepStack[2]),
            };
            foreach (var (label, want, got) in expected)
            {
                double cos = Cosine(want, got);
                _output.WriteLine($"[h3-vision] {label} cos={cos:F6}");
                // The reference runs the tower in Q4_K on the same backend, so agreement
                // is limited by quantization noise rather than by the graph.
                Assert.True(cos > 0.99, $"{label} cosine {cos:F6} is too low");
            }
        }
    }
}
