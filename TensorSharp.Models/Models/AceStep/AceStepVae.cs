// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ACE-Step 1.5's audio VAE decoder: Stable Audio's Oobleck decoder (ComfyUI comfy/ldm/audio/
// autoencoder.py) with channels 128, multipliers [1, 2, 4, 8, 16] and strides [2, 4, 4, 6, 10]:
// 25 Hz latents [64, T] -> 48 kHz stereo, T x 1920 samples.
//
//   conv k7 (64 -> 2048)
//   5 x DecoderBlock: SnakeBeta, ConvTranspose (k = 2 stride), 3 residual units (SnakeBeta, conv k7
//                     dilated 1/3/9, SnakeBeta, conv k1)
//   SnakeBeta, conv k7 (128 -> 2), no tanh
//
// Weight norm is folded by the converter. Runs on the CPU (the strided packed SGEMM of KokoroOps).
using System;
using System.Threading.Tasks;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;
using O = TensorSharp.Models.Kokoro.KokoroOps;

namespace TensorSharp.Models.AceStep
{
    internal sealed class AceStepVae
    {
        private static readonly int[] Strides = { 2, 4, 4, 6, 10 };
        private readonly GgufFile _gguf;
        private const string P = "vae.decoder.layers.";

        public AceStepVae(GgufFile gguf)
        {
            _gguf = gguf;
            if (!gguf.Tensors.ContainsKey(P + "0.weight"))
                throw new InvalidOperationException("the ACE-Step GGUF has no VAE decoder (vae.decoder.layers.0.weight).");
        }

        private float[] W(string name) => DirectOps.DequantTensor(_gguf, name);
        private int[] Shape(string name)
        {
            var s = _gguf.Tensors[name].Shape;
            var r = new int[s.Length];
            for (int i = 0; i < s.Length; i++) r[i] = (int)s[s.Length - 1 - i];
            return r;
        }

        /// <summary>
        /// <see cref="Decode"/> in tiles of <paramref name="tile"/> latent frames, each decoded with
        /// <paramref name="overlap"/> frames of context on both sides and cropped to its own span, so the
        /// 48 kHz stage never holds more than a tile (the decoder's receptive field is a few frames).
        /// </summary>
        public float[][] DecodeTiled(float[] latentRows, int frames, int tile = 192, int overlap = 24)
        {
            if (frames <= tile + 2 * overlap) return Decode(latentRows, frames);
            float[][] output = null;
            for (int start = 0; start < frames; start += tile)
            {
                int end = Math.Min(frames, start + tile);
                int a = Math.Max(0, start - overlap), b = Math.Min(frames, end + overlap);
                float[] rows = latentRows.AsSpan(a * 64, (b - a) * 64).ToArray();
                float[][] part = Decode(rows, b - a);
                output ??= new float[part.Length][];
                for (int ch = 0; ch < part.Length; ch++)
                {
                    output[ch] ??= new float[(long)frames * 1920];
                    Array.Copy(part[ch], (long)(start - a) * 1920, output[ch], (long)start * 1920, (long)(end - start) * 1920);
                }
            }
            return output;
        }

        /// <summary>Latent rows [frames, 64] -> planar stereo [2][frames * 1920].</summary>
        public float[][] Decode(float[] latentRows, int frames)
        {
            float[] x = O.Transpose(latentRows, frames, 64);   // [64, frames]
            int c = 64, t = frames;
            x = Conv(P + "0", x, ref c, t, 7, 1, 3);
            for (int b = 0; b < Strides.Length; b++)
            {
                string bp = $"{P}{b + 1}.layers.";
                int stride = Strides[Strides.Length - 1 - b];
                Snake(bp + "0", x, c, t);
                int[] s = Shape(bp + "1.weight");                // [in, out, k]
                int cout = s[1], k = s[2];
                int pad = (stride + 1) / 2;
                x = O.ConvTranspose1d(x, c, t, W(bp + "1.weight"), cout, k, W(bp + "1.bias"), stride, pad, stride % 2);
                t = (t - 1) * stride - 2 * pad + k + stride % 2;
                c = cout;
                int[] dil = { 1, 3, 9 };
                for (int r = 0; r < 3; r++)
                {
                    string rp = $"{bp}{r + 2}.layers.";
                    float[] y = (float[])x.Clone();
                    Snake(rp + "0", y, c, t);
                    int cc = c;
                    y = Conv(rp + "1", y, ref cc, t, 7, dil[r], 3 * dil[r]);
                    Snake(rp + "2", y, c, t);
                    y = Conv(rp + "3", y, ref cc, t, 1, 1, 0);
                    O.AddInPlace(x, y);
                }
            }
            Snake(P + "6", x, c, t);
            int outCh = c;
            x = Conv(P + "7", x, ref outCh, t, 7, 1, 3);
            var stereo = new float[outCh][];
            for (int ch = 0; ch < outCh; ch++) stereo[ch] = x.AsSpan(ch * t, t).ToArray();
            return stereo;
        }

        private float[] Conv(string prefix, float[] x, ref int c, int t, int k, int dilation, int padding)
        {
            int[] s = Shape(prefix + ".weight");                 // [out, in, k]
            float[] bias = _gguf.Tensors.ContainsKey(prefix + ".bias") ? W(prefix + ".bias") : null;
            float[] y = O.Conv1d(x, c, t, W(prefix + ".weight"), s[0], s[2], bias, 1, padding, dilation);
            c = s[0];
            return y;
        }

        /// <summary>SnakeBeta with log-scale parameters: x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) x)^2.</summary>
        private void Snake(string prefix, float[] x, int c, int t)
        {
            float[] alpha = W(prefix + ".alpha"), beta = W(prefix + ".beta");
            Parallel.For(0, c, ch =>
            {
                float a = MathF.Exp(alpha[ch]);
                float inv = 1f / (MathF.Exp(beta[ch]) + 1e-9f);
                long o = (long)ch * t;
                for (int i = 0; i < t; i++)
                {
                    float s = MathF.Sin(x[o + i] * a);
                    x[o + i] += inv * s * s;
                }
            });
        }
    }
}
