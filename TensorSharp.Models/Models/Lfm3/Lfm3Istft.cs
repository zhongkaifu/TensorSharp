// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// The LiquidCodec's streaming inverse STFT on the host (llama.cpp mtmd_audio_streaming_istft + mtmd_gen_wav_lfm3a; HF
    /// LiquidCodec ISTFT): each spectrum frame (log-magnitude and phase of n_fft / 2 + 1 bins) is turned into the frame's
    /// samples by an inverse real FFT, a periodic Hann window and overlap-add with hop <c>hop</c>, normalized by the
    /// overlapping windows' squared sum; the first (n_fft - hop) / 2 samples are padding and dropped, as are the last as many
    /// at <see cref="Flush"/>. A frame emits hop samples (after the padding), the flush the n_fft - hop the overlap holds
    /// back minus the trailing padding, so N frames give N * hop samples in all.
    /// </summary>
    public sealed class Lfm3StreamingIstft
    {
        private readonly int _n, _hop, _bins, _trim;
        private readonly float[] _window;
        private readonly float[] _overlap, _windowSum;
        private readonly double[] _cos, _sin;
        private readonly double[] _re, _im, _scratchRe, _scratchIm;
        private readonly float[] _frame;
        private int _paddingToRemove;

        public Lfm3StreamingIstft(int fftSize = 1280, int hopLength = 320)
        {
            if (fftSize < 2 || fftSize % 2 != 0 || hopLength < 1 || hopLength > fftSize)
                throw new ArgumentOutOfRangeException(nameof(hopLength), $"an ISTFT of {fftSize} points with hop {hopLength}");
            _n = fftSize;
            _hop = hopLength;
            _bins = fftSize / 2 + 1;
            _trim = (fftSize - hopLength) / 2;
            _window = new float[_n];
            for (int i = 0; i < _n; i++) _window[i] = (float)(0.5 * (1.0 - MathF.Cos((float)(2.0 * Math.PI * i / _n))));   // periodic
            _overlap = new float[_n];
            _windowSum = new float[_n];
            _cos = new double[_n];
            _sin = new double[_n];
            for (int i = 0; i < _n; i++)
            {
                _cos[i] = Math.Cos(2.0 * Math.PI * i / _n);
                _sin[i] = Math.Sin(2.0 * Math.PI * i / _n);
            }
            _re = new double[_n];
            _im = new double[_n];
            _scratchRe = new double[2 * _n];
            _scratchIm = new double[2 * _n];
            _frame = new float[_n];
            _paddingToRemove = _trim;
        }

        public int FftSize => _n;
        public int HopLength => _hop;
        public int Bins => _bins;

        /// <summary>Back to a stream's start (no overlap held, the leading padding pending).</summary>
        public void Reset()
        {
            Array.Clear(_overlap);
            Array.Clear(_windowSum);
            _paddingToRemove = _trim;
        }

        /// <summary>
        /// One spectrum frame: <paramref name="logMagnitude"/> and <paramref name="phase"/> (<see cref="Bins"/> each; the
        /// magnitude is exp(log magnitude), unclamped). Appends this frame's samples to <paramref name="output"/> (hop
        /// samples, fewer while the leading padding is dropped) and returns how many.
        /// </summary>
        public int ProcessFrame(ReadOnlySpan<float> logMagnitude, ReadOnlySpan<float> phase, List<float> output)
        {
            if (logMagnitude.Length < _bins || phase.Length < _bins) throw new ArgumentException($"a spectrum frame has {_bins} bins");
            // The Hermitian spectrum (the imaginary parts of DC and Nyquist reach only the discarded imaginary output).
            for (int j = 0; j < _bins; j++)
            {
                float mag = MathF.Exp(logMagnitude[j]);
                _re[j] = mag * MathF.Cos(phase[j]);
                _im[j] = mag * MathF.Sin(phase[j]);
            }
            for (int j = 1; j < _bins - 1; j++)
            {
                _re[_n - j] = _re[j];
                _im[_n - j] = -_im[j];
            }
            InverseFft(_re, _im, 0, 1, _n, _scratchRe, _scratchIm, 0);
            for (int j = 0; j < _n; j++)
            {
                _windowSum[j] += _window[j] * _window[j];
                _overlap[j] += (float)(_scratchRe[j] / _n) * _window[j];
            }
            return Emit(_hop, output, trimLeading: true);
        }

        /// <summary>The end of the stream: appends the samples the overlap still holds, without the trailing padding, and
        /// resets.</summary>
        public int Flush(List<float> output)
        {
            int before = output.Count;
            int remaining = _n - _hop;
            var tail = new List<float>(remaining);
            while (remaining > 0)
            {
                int chunk = Math.Min(remaining, _hop);
                Emit(chunk, tail, trimLeading: false);
                remaining -= chunk;
            }
            int keep = Math.Max(0, tail.Count - _trim);
            for (int i = 0; i < keep; i++) output.Add(tail[i]);
            Reset();
            return output.Count - before;
        }

        private int Emit(int count, List<float> output, bool trimLeading)
        {
            for (int i = 0; i < count; i++) _frame[i] = _windowSum[i] > 1e-8f ? _overlap[i] / _windowSum[i] : _overlap[i];
            Array.Copy(_overlap, count, _overlap, 0, _n - count);
            Array.Clear(_overlap, _n - count, count);
            Array.Copy(_windowSum, count, _windowSum, 0, _n - count);
            Array.Clear(_windowSum, _n - count, count);
            int skip = 0;
            if (trimLeading)
            {
                skip = Math.Min(_paddingToRemove, count);
                _paddingToRemove -= skip;
            }
            for (int i = skip; i < count; i++) output.Add(_frame[i]);
            return count - skip;
        }

        /// <summary>
        /// Unscaled inverse DFT (sign +) of the n points of (re, im) at <paramref name="offset"/> + k * <paramref name="stride"/>
        /// into out[outAt .. outAt + n): radix-2 decimation in time down to odd lengths (5 for 1280 points), which take a direct
        /// DFT; the twiddles index the full-size table, in double precision.
        /// </summary>
        private void InverseFft(double[] re, double[] im, int offset, int stride, int n, double[] outRe, double[] outIm, int outAt)
        {
            if (n % 2 == 1)
            {
                int step = _n / n;
                for (int k = 0; k < n; k++)
                {
                    double sr = 0, si = 0;
                    for (int t = 0; t < n; t++)
                    {
                        int idx = (int)((long)k * t * step % _n);
                        double c = _cos[idx], s = _sin[idx];
                        double xr = re[offset + t * stride], xi = im[offset + t * stride];
                        sr += xr * c - xi * s;
                        si += xr * s + xi * c;
                    }
                    outRe[outAt + k] = sr;
                    outIm[outAt + k] = si;
                }
                return;
            }
            int half = n / 2;
            // Even samples into out[outAt, + half), odd ones into out[outAt + half, + half).
            InverseFft(re, im, offset, stride * 2, half, outRe, outIm, outAt);
            InverseFft(re, im, offset + stride, stride * 2, half, outRe, outIm, outAt + half);
            int tw = _n / n;
            for (int k = 0; k < half; k++)
            {
                double c = _cos[k * tw], s = _sin[k * tw];
                double er = outRe[outAt + k], ei = outIm[outAt + k];
                double orr = outRe[outAt + half + k], oi = outIm[outAt + half + k];
                double tr = c * orr - s * oi, ti = c * oi + s * orr;
                outRe[outAt + k] = er + tr;
                outIm[outAt + k] = ei + ti;
                outRe[outAt + half + k] = er - tr;
                outIm[outAt + half + k] = ei - ti;
            }
        }
    }
}
