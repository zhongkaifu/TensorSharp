// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The CPU primitives of Kokoro's StyleTTS 2 / iSTFTNet network, on channel-major float arrays
// ([channels, time], time contiguous). Convolutions run as one strided packed SGEMM per kernel
// tap (no im2col copy), transposed convolutions as one GEMM followed by an overlap-add, and the
// bidirectional LSTMs as one input-projection GEMM plus a recurrent mat-vec per step.
using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp.Models.Kokoro
{
    internal static unsafe class KokoroOps
    {
        /// <summary>C[m,n] = alpha * A * B + beta * C over raw arrays with element strides.</summary>
        public static void Gemm(int m, int n, int k, float[] a, long aOff, long aRow, long aCol,
            float[] b, long bOff, long bRow, long bCol, float beta, float[] c, long cOff, long ldc)
        {
            if (m <= 0 || n <= 0) return;
            fixed (float* ap = a) fixed (float* bp = b) fixed (float* cp = c)
                CpuSgemm.Gemm(m, n, k, 1f, ap + aOff, aRow, aCol, bp + bOff, bRow, bCol, beta, cp + cOff, ldc);
        }

        /// <summary>torch.nn.Conv1d: x [cin, t], w [cout, cin, k] -> [cout, tOut].</summary>
        public static float[] Conv1d(float[] x, int cin, int t, float[] w, int cout, int k, float[] bias,
            int stride = 1, int padding = 0, int dilation = 1)
        {
            int tOut = (t + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
            var y = new float[(long)cout * tOut];
            if (bias != null)
                for (int o = 0; o < cout; o++) new Span<float>(y, o * tOut, tOut).Fill(bias[o]);
            for (int tap = 0; tap < k; tap++)
            {
                int shift = tap * dilation - padding;
                // Output columns whose input index j*stride + shift lies inside [0, t).
                int j0 = shift >= 0 ? 0 : (-shift + stride - 1) / stride;
                int j1 = Math.Min(tOut, (t - 1 - shift) / stride + 1);
                if (t - 1 - shift < 0) j1 = 0;
                if (j1 <= j0) continue;
                Gemm(cout, j1 - j0, cin, w, tap, (long)cin * k, k,
                     x, (long)j0 * stride + shift, t, stride, 1f, y, j0, tOut);
            }
            return y;
        }

        /// <summary>torch.nn.ConvTranspose1d (groups 1): x [cin, t], w [cin, cout, k] -> [cout, tOut].</summary>
        public static float[] ConvTranspose1d(float[] x, int cin, int t, float[] w, int cout, int k, float[] bias,
            int stride, int padding, int outputPadding = 0)
        {
            int tOut = (t - 1) * stride - 2 * padding + k + outputPadding;
            // z[(o, tap), j] = sum_c w[c, o, tap] x[c, j]: one GEMM with A(row = o*k + tap, col = c).
            int rows = cout * k;
            var z = new float[(long)rows * t];
            Gemm(rows, t, cin, w, 0, 1, (long)cout * k, x, 0, t, 1, 0f, z, 0, t);
            var y = new float[(long)cout * tOut];
            Parallel.For(0, cout, o =>
            {
                long yo = (long)o * tOut;
                float bv = bias != null ? bias[o] : 0f;
                for (int i = 0; i < tOut; i++) y[yo + i] = bv;
                for (int tap = 0; tap < k; tap++)
                {
                    long zr = ((long)o * k + tap) * t;
                    for (int j = 0; j < t; j++)
                    {
                        int pos = j * stride - padding + tap;
                        if ((uint)pos < (uint)tOut) y[yo + pos] += z[zr + j];
                    }
                }
            });
            return y;
        }

        /// <summary>Depthwise ConvTranspose1d (groups = channels): w [c, 1, k].</summary>
        public static float[] DepthwiseConvTranspose1d(float[] x, int c, int t, float[] w, int k, float[] bias,
            int stride, int padding, int outputPadding)
        {
            int tOut = (t - 1) * stride - 2 * padding + k + outputPadding;
            var y = new float[(long)c * tOut];
            Parallel.For(0, c, ch =>
            {
                long yo = (long)ch * tOut, xo = (long)ch * t;
                float bv = bias != null ? bias[ch] : 0f;
                for (int i = 0; i < tOut; i++) y[yo + i] = bv;
                for (int j = 0; j < t; j++)
                {
                    float v = x[xo + j];
                    for (int tap = 0; tap < k; tap++)
                    {
                        int pos = j * stride - padding + tap;
                        if ((uint)pos < (uint)tOut) y[yo + pos] += v * w[ch * k + tap];
                    }
                }
            });
            return y;
        }

        /// <summary>torch.nn.Linear over rows: x [n, inDim] row-major, w [outDim, inDim] -> [n, outDim].</summary>
        public static float[] Linear(float[] x, int n, int inDim, float[] w, int outDim, float[] bias)
        {
            var y = new float[(long)n * outDim];
            if (bias != null)
                for (int r = 0; r < n; r++) Array.Copy(bias, 0, y, (long)r * outDim, outDim);
            Gemm(n, outDim, inDim, x, 0, inDim, 1, w, 0, 1, inDim, bias != null ? 1f : 0f, y, 0, outDim);
            return y;
        }

        public static float[] Transpose(float[] x, int rows, int cols)
        {
            var y = new float[x.Length];
            Parallel.For(0, rows, r =>
            {
                for (int c = 0; c < cols; c++) y[(long)c * rows + r] = x[(long)r * cols + c];
            });
            return y;
        }

        /// <summary>
        /// A one-layer bidirectional torch.nn.LSTM (batch_first) over x [t, inDim]: [t, 2h], the
        /// forward direction's hidden state then the reverse's. Gate order i, f, g, o.
        /// </summary>
        public static float[] BiLstm(float[] x, int t, int inDim, KokoroLstm p)
        {
            int h = p.Hidden;
            var y = new float[(long)t * 2 * h];
            Parallel.For(0, 2, dir =>
            {
                float[] wih = dir == 0 ? p.WihF : p.WihR, whh = dir == 0 ? p.WhhF : p.WhhR;
                float[] bih = dir == 0 ? p.BihF : p.BihR, bhh = dir == 0 ? p.BhhF : p.BhhR;
                float[] proj = Linear(x, t, inDim, wih, 4 * h, bih);   // [t, 4h]
                var hs = new float[h];
                var cs = new float[h];
                var gates = new float[4 * h];
                for (int step = 0; step < t; step++)
                {
                    int ti = dir == 0 ? step : t - 1 - step;
                    for (int g = 0; g < 4 * h; g++)
                        gates[g] = proj[(long)ti * 4 * h + g] + bhh[g] +
                                   TensorPrimitives.Dot(new ReadOnlySpan<float>(whh, g * h, h), hs);
                    for (int j = 0; j < h; j++)
                    {
                        float ig = Sigmoid(gates[j]);
                        float fg = Sigmoid(gates[h + j]);
                        float gg = MathF.Tanh(gates[2 * h + j]);
                        float og = Sigmoid(gates[3 * h + j]);
                        cs[j] = fg * cs[j] + ig * gg;
                        hs[j] = og * MathF.Tanh(cs[j]);
                        y[(long)ti * 2 * h + dir * h + j] = hs[j];
                    }
                }
            });
            return y;
        }

        public static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

        /// <summary>LayerNorm over channels at each time step of x [c, t] (StyleTTS2's LayerNorm module).</summary>
        public static void LayerNormChannels(float[] x, int c, int t, float[] gamma, float[] beta, float eps = 1e-5f)
        {
            Parallel.For(0, t, j =>
            {
                double mean = 0;
                for (int i = 0; i < c; i++) mean += x[(long)i * t + j];
                mean /= c;
                double var = 0;
                for (int i = 0; i < c; i++) { double d = x[(long)i * t + j] - mean; var += d * d; }
                float inv = (float)(1.0 / Math.Sqrt(var / c + eps));
                for (int i = 0; i < c; i++)
                {
                    long idx = (long)i * t + j;
                    float v = (float)((x[idx] - mean) * inv);
                    x[idx] = gamma != null ? v * gamma[i] + beta[i] : v;
                }
            });
        }

        /// <summary>LayerNorm over the last axis of rows x [n, d].</summary>
        public static void LayerNormRows(float[] x, int n, int d, float[] gamma, float[] beta, float eps)
        {
            Parallel.For(0, n, r =>
            {
                var row = new Span<float>(x, r * d, d);
                double mean = 0;
                foreach (float v in row) mean += v;
                mean /= d;
                double var = 0;
                foreach (float v in row) { double dv = v - mean; var += dv * dv; }
                float inv = (float)(1.0 / Math.Sqrt(var / d + eps));
                for (int i = 0; i < d; i++)
                {
                    float v = (float)((row[i] - mean) * inv);
                    row[i] = gamma != null ? v * gamma[i] + beta[i] : v;
                }
            });
        }

        /// <summary>AdaIN1d: (1 + gamma) * InstanceNorm(x) + beta, gamma and beta = fc(style) (InstanceNorm1d,
        /// eps 1e-5, biased variance per channel over time; its affine weights are absent from the checkpoint).</summary>
        public static float[] AdaIn(float[] x, int c, int t, float[] fcW, float[] fcB, float[] style)
        {
            float[] h = Linear(style, 1, style.Length, fcW, 2 * c, fcB);
            var y = new float[x.Length];
            Parallel.For(0, c, ch =>
            {
                var row = new ReadOnlySpan<float>(x, ch * t, t);
                double mean = 0;
                foreach (float v in row) mean += v;
                mean /= t;
                double var = 0;
                foreach (float v in row) { double d = v - mean; var += d * d; }
                float inv = (float)(1.0 / Math.Sqrt(var / t + 1e-5));
                float gain = 1f + h[ch], shift = h[c + ch];
                for (int i = 0; i < t; i++)
                    y[(long)ch * t + i] = (float)((row[i] - mean) * inv) * gain + shift;
            });
            return y;
        }

        public static void LeakyRelu(float[] x, float slope)
        {
            Parallel.For(0, (x.Length + 65535) / 65536, b =>
            {
                int end = Math.Min(x.Length, (b + 1) * 65536);
                for (int i = b * 65536; i < end; i++) if (x[i] < 0) x[i] *= slope;
            });
        }

        /// <summary>Snake1D: x + (1 / alpha) * sin(alpha * x)^2 with alpha per channel.</summary>
        public static void Snake(float[] x, int c, int t, float[] alpha)
        {
            Parallel.For(0, c, ch =>
            {
                float a = alpha[ch], inv = 1f / a;
                long o = (long)ch * t;
                for (int i = 0; i < t; i++)
                {
                    float s = MathF.Sin(a * x[o + i]);
                    x[o + i] += inv * s * s;
                }
            });
        }

        public static void AddInPlace(float[] y, float[] x)
        {
            TensorPrimitives.Add(y, x, y);
        }

        public static void Scale(float[] y, float s) => TensorPrimitives.Multiply(y, s, y);

        /// <summary>Stack channel blocks [c_i, t] into [sum c_i, t].</summary>
        public static float[] ConcatChannels(int t, params (float[] Data, int Channels)[] parts)
        {
            int total = 0;
            foreach (var p in parts) total += p.Channels;
            var y = new float[(long)total * t];
            long off = 0;
            foreach (var p in parts)
            {
                Array.Copy(p.Data, 0, y, off, (long)p.Channels * t);
                off += (long)p.Channels * t;
            }
            return y;
        }

        /// <summary>Nearest-neighbour upsampling along time by an integer factor.</summary>
        public static float[] UpsampleNearest(float[] x, int c, int t, int factor)
        {
            var y = new float[(long)c * t * factor];
            Parallel.For(0, c, ch =>
            {
                for (int i = 0; i < t * factor; i++) y[(long)ch * t * factor + i] = x[(long)ch * t + i / factor];
            });
            return y;
        }

        /// <summary>F.interpolate(mode="linear", align_corners=False) with an explicit scale_factor,
        /// along time of x [c, t]: the source index is (dst + 0.5) / factor - 0.5, clamped at 0.</summary>
        public static float[] InterpolateLinear(float[] x, int c, int t, double factor, int tOut)
        {
            var y = new float[(long)c * tOut];
            double scale = 1.0 / factor;
            Parallel.For(0, c, ch =>
            {
                for (int i = 0; i < tOut; i++)
                {
                    double src = scale * (i + 0.5) - 0.5;
                    if (src < 0) src = 0;
                    int i0 = (int)src;
                    if (i0 > t - 1) i0 = t - 1;
                    int i1 = i0 < t - 1 ? i0 + 1 : i0;
                    double l1 = src - i0, l0 = 1 - l1;
                    y[(long)ch * tOut + i] = (float)(l0 * x[(long)ch * t + i0] + l1 * x[(long)ch * t + i1]);
                }
            });
            return y;
        }

        /// <summary>torch.hann_window(n, periodic=True).</summary>
        public static float[] Hann(int n)
        {
            var w = new float[n];
            for (int i = 0; i < n; i++) w[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / n));
            return w;
        }

        /// <summary>torch.stft(center=True, reflect padding, onesided) of a mono signal:
        /// magnitude and angle [nFft/2+1, frames], frames = len/hop + 1.</summary>
        public static (float[] Mag, float[] Phase, int Frames) Stft(float[] x, int nFft, int hop)
        {
            int pad = nFft / 2, n = x.Length, bins = nFft / 2 + 1;
            int frames = n / hop + 1;
            float[] window = Hann(nFft);
            var mag = new float[(long)bins * frames];
            var phase = new float[(long)bins * frames];
            Parallel.For(0, frames, f =>
            {
                Span<double> frame = stackalloc double[nFft];
                for (int i = 0; i < nFft; i++)
                {
                    int idx = f * hop - pad + i;
                    if (n == 1) idx = 0;
                    else
                        while (idx < 0 || idx >= n) idx = idx < 0 ? -idx : 2 * (n - 1) - idx;
                    frame[i] = x[idx] * window[i];
                }
                for (int k = 0; k < bins; k++)
                {
                    double re = 0, im = 0;
                    for (int i = 0; i < nFft; i++)
                    {
                        var (c, s) = Twiddle((int)((long)k * i % nFft), nFft);
                        re += frame[i] * c;
                        im -= frame[i] * s;
                    }
                    // A real FFT's DC and Nyquist bins are exactly real; a stray 1e-16 would flip their angle
                    // between +pi and -pi.
                    if (k == 0 || (nFft % 2 == 0 && k == nFft / 2)) im = 0;
                    mag[(long)k * frames + f] = (float)Math.Sqrt(re * re + im * im);
                    phase[(long)k * frames + f] = (float)Math.Atan2(im, re);
                }
            });
            return (mag, phase, frames);
        }

        /// <summary>cos and sin of 2 pi j / n, exact at the multiples of a quarter turn.</summary>
        private static (double Cos, double Sin) Twiddle(int j, int n)
        {
            if (j == 0) return (1, 0);
            if (n % 2 == 0 && j == n / 2) return (-1, 0);
            if (n % 4 == 0 && j == n / 4) return (0, 1);
            if (n % 4 == 0 && j == 3 * n / 4) return (0, -1);
            double a = 2 * Math.PI * j / n;
            return (Math.Cos(a), Math.Sin(a));
        }

        /// <summary>torch.istft(center=True) of magnitude * exp(i * phase) [nFft/2+1, frames]:
        /// windowed inverse real DFTs overlap-added and divided by the summed squared window,
        /// with nFft/2 trimmed from each end; hop * (frames - 1) samples.</summary>
        public static float[] Istft(float[] mag, float[] phase, int frames, int nFft, int hop)
        {
            int bins = nFft / 2 + 1, pad = nFft / 2;
            int full = nFft + hop * (frames - 1);
            float[] window = Hann(nFft);
            var frameOut = new float[(long)frames * nFft];
            Parallel.For(0, frames, f =>
            {
                for (int i = 0; i < nFft; i++)
                {
                    // irfft: real part of the Hermitian-extended inverse DFT.
                    double s = 0;
                    for (int k = 0; k < bins; k++)
                    {
                        double m = mag[(long)k * frames + f], p = phase[(long)k * frames + f];
                        double re = m * Math.Cos(p), im = m * Math.Sin(p);
                        if (k == 0 || (nFft % 2 == 0 && k == nFft / 2)) im = 0; // the DC and Nyquist bins are real
                        var (cs, sn) = Twiddle((int)((long)k * i % nFft), nFft);
                        double term = re * cs - im * sn;
                        s += (k == 0 || (nFft % 2 == 0 && k == nFft / 2)) ? term : 2 * term;
                    }
                    frameOut[(long)f * nFft + i] = (float)(s / nFft) * window[i];
                }
            });
            var y = new double[full];
            var env = new double[full];
            for (int f = 0; f < frames; f++)
                for (int i = 0; i < nFft; i++)
                {
                    y[f * hop + i] += frameOut[(long)f * nFft + i];
                    env[f * hop + i] += (double)window[i] * window[i];
                }
            int length = hop * (frames - 1);
            var outp = new float[length];
            for (int i = 0; i < length; i++)
            {
                double e = env[i + pad];
                outp[i] = (float)(e > 1e-11 ? y[i + pad] / e : y[i + pad]);
            }
            return outp;
        }
    }

    /// <summary>The weights of one bidirectional LSTM layer.</summary>
    internal sealed class KokoroLstm
    {
        public int Hidden;
        public float[] WihF, WhhF, BihF, BhhF, WihR, WhhR, BihR, BhhR;
    }
}
