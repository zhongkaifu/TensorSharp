// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan 2.2 numeric oracle tests: compare the native whole-graph kernels against
// reference outputs produced by the REAL diffusers models (AutoencoderKLWan,
// WanTransformer3DModel) on deterministic inputs. The fixtures are raw
// little-endian F32 blobs written by InferenceWeb.Tests/tools/wan22_oracle.py
// (needs torch + diffusers >= 0.38 and the local model files); every test
// no-ops when the fixtures (or the local model files) are absent.
using System;
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    [Trait("Requires", "Models")]
    public class WanVideoOracleTests
    {
        private readonly ITestOutputHelper _output;
        public WanVideoOracleTests(ITestOutputHelper output) { _output = output; }

        // TS_WAN_MODEL_DIR points the whole oracle suite at a local Wan checkout
        // (the fixtures live in <dir>/fixtures); without it the historical Windows
        // paths apply, so an existing checkout keeps running unchanged and the
        // tests still no-op wherever the files are absent.
        private static readonly string ModelDir =
            Environment.GetEnvironmentVariable("TS_WAN_MODEL_DIR") ?? @"C:\Works\models\wan";
        private static readonly string FixtureDir = Path.Combine(ModelDir, "fixtures");
        private static readonly string Vae22Path = Path.Combine(ModelDir, "VAE", "Wan2.2_VAE.safetensors");
        private static readonly string Vae21Path = Path.Combine(ModelDir, "wan_2.1_vae.safetensors");
        private static readonly string Ti2vGguf = Path.Combine(ModelDir, "Wan2.2-TI2V-5B-Q8_0.gguf");

        private static float[] ReadF32(string name)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(FixtureDir, name));
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return data;
        }

        private static bool Have(params string[] fixtures)
        {
            foreach (var f in fixtures)
                if (!File.Exists(Path.Combine(FixtureDir, f))) return false;
            return true;
        }

        // The whole-suite run may already have initialized a different ggml backend in
        // this process (backend init is process-global); fall back to whatever works
        // and skip when neither can be brought up.
        private static bool UseBackend()
        {
            bool cpu = Environment.GetEnvironmentVariable("TS_TEST_WAN_CPU") == "1";
            foreach (var b in cpu
                ? new[] { GgmlBackendType.Cpu }
                : new[] { GgmlBackendType.Cuda, GgmlBackendType.Cpu })
            {
                try
                {
                    GgmlBasicOps.EnsureBackendAvailable(b);
                    return true;
                }
                catch (InvalidOperationException) { }
            }
            return false;
        }

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
            return dot / Math.Sqrt(Math.Max(na * nb, 1e-30));
        }

        private static double Psnr01(float[] a, float[] b)
        {
            double mse = 0;
            for (long i = 0; i < a.LongLength; i++)
            {
                double d = a[i] - b[i];
                mse += d * d;
            }
            mse /= a.LongLength;
            return 10.0 * Math.Log10(1.0 / Math.Max(mse, 1e-12));
        }

        private static string Stat(float[] a)
        {
            double mn = double.MaxValue, mx = double.MinValue, sum = 0; long nan = 0;
            foreach (float v in a)
            {
                if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
                if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
            }
            return $"min={mn:E2} max={mx:E2} mean={sum / Math.Max(1, a.LongLength - nan):E3} nan={nan}/{a.LongLength}";
        }

        // Normalize a raw-latent fixture into the DiT space ((z - mean) / std).
        private static float[] Normalize(float[] z, float[] mean, float[] std, int c, long perChannel)
        {
            var o = new float[z.Length];
            for (int ch = 0; ch < c; ch++)
                for (long i = 0; i < perChannel; i++)
                    o[ch * perChannel + i] = (z[ch * perChannel + i] - mean[ch]) / std[ch];
            return o;
        }

        [Fact]
        public void Vae22DecodeMatchesDiffusers()
        {
            if (!File.Exists(Vae22Path) || !Have("vae22_dec_z.bin", "vae22_dec_out.bin")) return;
            if (!UseBackend()) return;

            // fixture z: [1, 48, 3, 10, 10] raw latent; Decode() takes the normalized space
            int t = 3, lh = 10, lw = 10;
            float[] z = ReadF32("vae22_dec_z.bin");
            float[] zn = Normalize(z, WanVae.LatentsMean22, WanVae.LatentsStd22, 48, (long)t * lh * lw);

            using var vae = new WanVae(Vae22Path);
            Assert.Equal(2, vae.Version);
            Assert.Equal(48, vae.ZDim);
            Assert.Equal(16, vae.SpatialScale);
            var frames = vae.Decode(zn, t, lh, lw);

            // oracle: [1, 3, 9, 160, 160] in [-1, 1]
            float[] expect = ReadF32("vae22_dec_out.bin");
            int outT = 9, H = 160, W = 160;
            Assert.Equal(outT, frames.Length);
            Assert.Equal(W, frames[0].Width);

            var mine = new float[(long)outT * 3 * H * W];
            var reference = new float[mine.Length];
            long hw = (long)H * W;
            for (int f = 0; f < outT; f++)
            {
                var px = frames[f].Pixels;   // interleaved HWC [0,1]
                for (long i = 0; i < hw; i++)
                    for (int c = 0; c < 3; c++)
                        mine[((long)f * 3 + c) * hw + i] = px[i * 3 + c];
                for (int c = 0; c < 3; c++)
                    for (long i = 0; i < hw; i++)
                    {
                        float v = (expect[((long)c * outT + f) * hw + i] + 1f) * 0.5f;
                        reference[((long)f * 3 + c) * hw + i] = Math.Clamp(v, 0f, 1f);
                    }
            }
            double psnr = Psnr01(mine, reference);
            var perFrame = new System.Text.StringBuilder();
            for (int f = 0; f < outT; f++)
            {
                var mf = new float[3 * hw]; var rf = new float[3 * hw];
                Array.Copy(mine, (long)f * 3 * hw, mf, 0, 3 * hw);
                Array.Copy(reference, (long)f * 3 * hw, rf, 0, 3 * hw);
                perFrame.Append($" f{f}={Psnr01(mf, rf):F1}");
            }
            // Printed on success too (as the DiT test prints its cosine): the decode's
            // banded im2col lowering is tuned for speed, so the margin over the 35 dB
            // bar is the number that says whether a tuning change cost accuracy.
            _output.WriteLine($"[wan-oracle] Wan2.2 VAE decode PSNR vs diffusers: {psnr:F1} dB; per-frame:{perFrame}");
            Assert.True(psnr > 35.0, $"Wan2.2 VAE decode PSNR vs diffusers = {psnr:F1} dB (want > 35); per-frame:{perFrame}");
        }

        [Fact]
        public void Vae22EncodeMatchesDiffusers()
        {
            if (!File.Exists(Vae22Path) || !Have("vae22_enc_in.bin", "vae22_enc_mu.bin")) return;
            if (!UseBackend()) return;

            int t = 5, H = 160, W = 192, lt = 2, lh = 10, lw = 12;
            float[] clip = ReadF32("vae22_enc_in.bin");         // [3, 5, 160, 192] planar
            using var vae = new WanVae(Vae22Path);
            Assert.True(vae.HasEncoder);
            float[] mine = vae.Encode(clip, t, H, W);           // normalized [48, 2, 10, 12]

            float[] mu = ReadF32("vae22_enc_mu.bin");
            float[] expect = Normalize(mu, WanVae.LatentsMean22, WanVae.LatentsStd22, 48, (long)lt * lh * lw);
            double cos = Cosine(mine, expect);
            Assert.True(cos > 0.999, $"Wan2.2 VAE encode cosine vs diffusers = {cos:F6} (want > 0.999)");
        }

        [Fact]
        public void Vae21EncodeMatchesDiffusers()
        {
            if (!File.Exists(Vae21Path) || !Have("vae21_enc_in.bin", "vae21_enc_mu.bin")) return;
            if (!UseBackend()) return;

            int t = 5, H = 128, W = 160, lt = 2, lh = 16, lw = 20;
            float[] clip = ReadF32("vae21_enc_in.bin");
            using var vae = new WanVae(Vae21Path);
            Assert.Equal(1, vae.Version);
            Assert.True(vae.HasEncoder);
            float[] mine = vae.Encode(clip, t, H, W);

            float[] mu = ReadF32("vae21_enc_mu.bin");
            float[] expect = Normalize(mu, WanVae.LatentsMean21, WanVae.LatentsStd21, 16, (long)lt * lh * lw);
            double cos = Cosine(mine, expect);
            Assert.True(cos > 0.999, $"Wan2.1 VAE encode cosine vs diffusers = {cos:F6} (want > 0.999)");
        }

        [Fact]
        public void Dit5bMatchesDiffusersUniformAndMasked()
        {
            if (!File.Exists(Ti2vGguf) || !Have("dit_x.bin", "dit_ctx.bin", "dit_v_uniform.bin", "dit_v_masked.bin"))
                return;
            if (!UseBackend()) return;

            int T = 2, Hh = 8, Ww = 8;
            int hLen = Hh / 2, wLen = Ww / 2, seq = T * hLen * wLen, s0 = hLen * wLen;
            float[] xLat = ReadF32("dit_x.bin");                // [48, 2, 8, 8] planar
            float[] ctx = ReadF32("dit_ctx.bin");               // [512, 4096] token-major

            using var gguf = new GgufFile(Ti2vGguf);
            using var dit = new WanDiT(gguf);
            Assert.Equal(48, dit.InDim);
            var x = new float[(long)dit.InTok * seq];
            WanVideoPipeline.Patchify(xLat, x, T, Hh, Ww, 48, dit.InTok, 0);
            var rope = WanRope.Build(T, hLen, wLen);

            // (a) uniform timestep 500
            float[] v = dit.Predict(x, ctx, 500f, rope, seq);
            var vLat = new float[xLat.Length];
            WanVideoPipeline.Unpatchify(v, vLat, T, Hh, Ww, 48, dit.OutTok);
            double cosU = Cosine(vLat, ReadF32("dit_v_uniform.bin"));
            Assert.True(cosU > 0.995,
                $"TI2V-5B DiT (uniform t) cosine vs diffusers = {cosU:F6} (want > 0.995); " +
                $"x[{Stat(x)}] v[{Stat(v)}]");

            // (b) per-token: frame-0 tokens at t=0, the rest at t=500 (i2v conditioning)
            float[] vm = dit.Predict(x, ctx, 500f, rope, seq, s0, 0f);
            var vmLat = new float[xLat.Length];
            WanVideoPipeline.Unpatchify(vm, vmLat, T, Hh, Ww, 48, dit.OutTok);
            double cosM = Cosine(vmLat, ReadF32("dit_v_masked.bin"));
            _output.WriteLine($"[wan-oracle] TI2V-5B DiT cosine vs diffusers: uniform={cosU:F6} masked={cosM:F6}");
            Assert.True(cosM > 0.995, $"TI2V-5B DiT (masked t) cosine vs diffusers = {cosM:F6} (want > 0.995)");
        }
    }
}
