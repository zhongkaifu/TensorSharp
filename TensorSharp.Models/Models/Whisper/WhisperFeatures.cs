// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading.Tasks;

namespace TensorSharp.Models.Whisper
{
    /// <summary>
    /// openai/whisper's log_mel_spectrogram: a centred, reflect-padded STFT with a periodic Hann
    /// window (n_fft 400, hop 160 at 16 kHz), power spectrum, the Slaney mel filterbank the GGUF
    /// carries, log10 clamped at 1e-10, floored at (max - 8), and scaled as (x + 4) / 4. The last
    /// STFT frame is dropped, so N samples give N / hop frames.
    /// </summary>
    public sealed class WhisperFeatures
    {
        private readonly int _nFft, _hop, _nFreq, _nMels;
        private readonly float[] _filters;   // [nMels, nFreq]
        private readonly float[] _window;    // [nFft]
        private readonly float[] _cos, _sin; // [nFreq, nFft]

        public int MelBins => _nMels;

        public WhisperFeatures(float[] melFilters, int nMels, int nFft, int hop)
        {
            _nFft = nFft;
            _hop = hop;
            _nFreq = nFft / 2 + 1;
            _nMels = nMels;
            if (melFilters.Length != (long)nMels * _nFreq)
                throw new ArgumentException($"mel filterbank has {melFilters.Length} values, expected {nMels} x {_nFreq}.");
            _filters = melFilters;
            _window = new float[nFft];
            for (int n = 0; n < nFft; n++)
                _window[n] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * n / nFft)); // torch.hann_window(periodic=True)
            _cos = new float[_nFreq * nFft];
            _sin = new float[_nFreq * nFft];
            for (int k = 0; k < _nFreq; k++)
                for (int n = 0; n < nFft; n++)
                {
                    // Reduce k*n mod nFft before the trig call so large products keep their precision.
                    double angle = 2 * Math.PI * ((long)k * n % nFft) / nFft;
                    _cos[k * nFft + n] = (float)Math.Cos(angle);
                    _sin[k * nFft + n] = (float)Math.Sin(angle);
                }
        }

        /// <summary>Log-mel features [nMels, frames] (frames = samples / hop), normalised over the whole clip.</summary>
        public float[] Compute(float[] samples, out int frames)
        {
            int n = samples.Length;
            frames = n / _hop;
            int pad = _nFft / 2;
            int nf = frames;
            var power = new float[(long)nf * _nFreq];
            int nFft = _nFft, nFreq = _nFreq, hop = _hop;
            float[] window = _window, cos = _cos, sin = _sin;

            Parallel.For(0, nf, () => new float[nFft], (f, _, buf) =>
            {
                int start = f * hop - pad;
                for (int i = 0; i < nFft; i++)
                    buf[i] = Reflect(samples, start + i) * window[i];
                var spanBuf = new ReadOnlySpan<float>(buf);
                for (int k = 0; k < nFreq; k++)
                {
                    float re = System.Numerics.Tensors.TensorPrimitives.Dot(spanBuf, new ReadOnlySpan<float>(cos, k * nFft, nFft));
                    float im = System.Numerics.Tensors.TensorPrimitives.Dot(spanBuf, new ReadOnlySpan<float>(sin, k * nFft, nFft));
                    power[(long)f * nFreq + k] = re * re + im * im;
                }
                return buf;
            }, _ => { });

            var mel = new float[(long)_nMels * nf];
            int nMels = _nMels;
            float[] filters = _filters;
            float globalMax = float.NegativeInfinity;
            object gate = new();
            Parallel.For(0, nf, () => float.NegativeInfinity, (f, _, localMax) =>
            {
                var p = new ReadOnlySpan<float>(power, f * nFreq, nFreq);
                for (int m = 0; m < nMels; m++)
                {
                    float v = System.Numerics.Tensors.TensorPrimitives.Dot(p, new ReadOnlySpan<float>(filters, m * nFreq, nFreq));
                    float l = MathF.Log10(MathF.Max(v, 1e-10f));
                    mel[(long)m * nf + f] = l;
                    if (l > localMax) localMax = l;
                }
                return localMax;
            }, localMax => { lock (gate) if (localMax > globalMax) globalMax = localMax; });

            float floor = globalMax - 8f;
            for (long i = 0; i < mel.LongLength; i++)
                mel[i] = (MathF.Max(mel[i], floor) + 4f) / 4f;
            return mel;
        }

        /// <summary>torch reflect padding: index -1 reads x[1], index n reads x[n - 2].</summary>
        private static float Reflect(float[] x, int i)
        {
            int n = x.Length;
            if (n == 1) return x[0];
            while (i < 0 || i >= n)
            {
                if (i < 0) i = -i;
                if (i >= n) i = 2 * (n - 1) - i;
            }
            return x[i];
        }
    }
}
