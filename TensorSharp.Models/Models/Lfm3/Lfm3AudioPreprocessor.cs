// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using TensorSharp.Models.Media;
using TensorSharp.Models.Video;

namespace TensorSharp.Models
{
    /// <summary>
    /// LFM3-Omni's audio front end (llama.cpp <c>mtmd_audio_preprocessor_lfm3a</c>, HF <c>Lfm3OmniFeatureExtractor</c>):
    /// 16 kHz mono PCM to normalized log-mel frames. Frame k is the periodic-Hann-windowed samples
    /// [k * 160, k * 160 + 400) (no centering, no pre-emphasis, no dither), its 201-bin power spectrum, 128 Slaney-scale,
    /// Slaney-normalized triangular mels over 0-8 kHz, <c>log10(max(mel, 1e-12))</c>, then <c>(v + 4) / 4</c>. A clip is
    /// zero-padded on the right to whole hops (<see cref="PaddedLength"/>), so it has <see cref="FrameCount"/> frames.
    /// <para>
    /// The constants are not in the mmproj (llama.cpp hard-codes them, its converter refuses a checkpoint whose feature
    /// extractor differs); <see cref="Lfm3AudioEncoder"/> checks the mmproj's mel count against <see cref="MelBins"/>.
    /// </para>
    /// <para>
    /// Features are frame-major: frame t's mel m is <c>features[t * 128 + m]</c> (each frame contiguous, so a stream appends
    /// frames). The spectrum is a double-precision real FFT (a 200-point complex Stockham FFT, radices 5, 5, 4, 2), the
    /// filterbank sparse; frames run in parallel.
    /// </para>
    /// </summary>
    public static class Lfm3AudioPreprocessor
    {
        public const int SampleRate = 16000;
        public const int FftSize = 400;
        public const int WindowLength = 400;
        public const int HopLength = 160;
        public const int MelBins = 128;
        public const int FrequencyBins = FftSize / 2 + 1;   // 201
        public const double LogFloor = 1e-12;
        public const float NormalizationOffset = 4f;
        public const float NormalizationDivisor = 4f;

        /// <summary>Mel frames a row of the encoder's output covers: three stride-2 convolutions.</summary>
        public const int FramesPerRow = 8;

        /// <summary>Samples a row covers (80 ms; 12.5 rows a second).</summary>
        public const int SamplesPerRow = FramesPerRow * HopLength;   // 1280

        private const int HalfSize = FftSize / 2;   // the complex FFT's length
        private static readonly int[] s_radices = { 5, 5, 4, 2 };
        private static readonly double[] s_window = BuildWindow();
        private static readonly Stage[] s_stages = BuildStages();
        private static readonly double[] s_postCos = new double[HalfSize + 1], s_postSin = new double[HalfSize + 1];
        private static readonly MelFilter[] s_filters = BuildFilters();

        private readonly record struct MelFilter(int FirstBin, float[] Weights);

        private sealed class Stage
        {
            public int Radix, Length, Stride;
            public double[] TwiddleRe, TwiddleIm;   // [q * radix + k] = W_length^(q k)
        }

        static Lfm3AudioPreprocessor()
        {
            for (int k = 0; k <= HalfSize; k++)
            {
                double a = -2 * Math.PI * k / FftSize;
                s_postCos[k] = Math.Cos(a);
                s_postSin[k] = Math.Sin(a);
            }
        }

        // ================================================================ sizes

        /// <summary>A clip's length after the right zero padding: one window at least, then whole hops past it (llama.cpp
        /// mtmd-audio.cpp lfm3a <c>process</c>; HF <c>pad_length</c>).</summary>
        public static long PaddedLength(long samples)
        {
            if (samples < 0) throw new ArgumentOutOfRangeException(nameof(samples));
            return samples <= WindowLength ? WindowLength : samples + (HopLength - (samples - WindowLength) % HopLength) % HopLength;
        }

        /// <summary>Mel frames of a clip of <paramref name="samples"/> samples: (padded - 400) / 160 + 1.</summary>
        public static int FrameCount(long samples) => checked((int)((PaddedLength(samples) - WindowLength) / HopLength + 1));

        /// <summary>Encoder rows (LM positions) for <paramref name="frames"/> mel frames: ceil(frames / 8).</summary>
        public static int RowCount(int frames)
        {
            if (frames < 0) throw new ArgumentOutOfRangeException(nameof(frames));
            return (frames + FramesPerRow - 1) / FramesPerRow;
        }

        /// <summary>Encoder rows for a clip of <paramref name="samples"/> 16 kHz samples (0 samples: no rows).</summary>
        public static int RowCountForSamples(long samples) => samples == 0 ? 0 : RowCount(FrameCount(samples));

        /// <summary>Frames complete in the first <paramref name="samples"/> samples of a stream (frame k needs samples up
        /// to k * 160 + 400).</summary>
        public static int CompleteFrames(long samples) => samples < WindowLength ? 0 : checked((int)((samples - WindowLength) / HopLength + 1));

        // ================================================================ features

        /// <summary>The normalized log-mel features of a 16 kHz mono clip, frame-major [frames][128]; empty for an empty
        /// clip.</summary>
        public static float[] ComputeFeatures(ReadOnlySpan<float> samples, out int frames)
        {
            if (samples.Length == 0)
            {
                frames = 0;
                return Array.Empty<float>();
            }
            frames = FrameCount(samples.Length);
            var features = new float[(long)frames * MelBins];
            ComputeFrames(samples.ToArray(), 0, samples.Length, 0, frames, features, 0);
            return features;
        }

        /// <summary>Features of a clip in llama.cpp's encoder input layout, mel-major [128][frames] (its <c>inp_raw</c>):
        /// for comparisons with mtmd.</summary>
        public static float[] ToMelMajor(float[] features, int frames)
        {
            ArgumentNullException.ThrowIfNull(features);
            if (features.LongLength != (long)frames * MelBins) throw new ArgumentException($"{features.Length} values for {frames} frames");
            var melMajor = new float[features.Length];
            for (int t = 0; t < frames; t++)
                for (int m = 0; m < MelBins; m++)
                    melMajor[(long)m * frames + t] = features[(long)t * MelBins + m];
            return melMajor;
        }

        /// <summary>
        /// Frames [<paramref name="firstFrame"/>, + <paramref name="count"/>) of <paramref name="samples"/> (whose first
        /// element is the stream's sample <paramref name="sampleOffset"/>; samples at or past <paramref name="validSamples"/>
        /// are the right zero padding) into <paramref name="dest"/> at <paramref name="destOffset"/>, frame-major.
        /// </summary>
        internal static void ComputeFrames(float[] samples, long sampleOffset, long validSamples, int firstFrame, int count,
            float[] dest, long destOffset)
        {
            if (count <= 0) return;
            const int Block = 256;   // ~0.5 ms of work a block (about 2 us a frame): the thread pool only for clips past 2.5 s
            int blocks = (count + Block - 1) / Block;
            void RunBlock(int block)
            {
                var re = new double[HalfSize];
                var im = new double[HalfSize];
                var re2 = new double[HalfSize];
                var im2 = new double[HalfSize];
                var power = new double[FrequencyBins];
                int end = Math.Min(count, (block + 1) * Block);
                for (int i = block * Block; i < end; i++)
                {
                    long start = (long)(firstFrame + i) * HopLength - sampleOffset;
                    long valid = validSamples - sampleOffset;
                    if (start + FftSize <= valid)
                    {
                        for (int j = 0; j < HalfSize; j++)
                        {
                            re[j] = samples[start + 2 * j] * s_window[2 * j];
                            im[j] = samples[start + 2 * j + 1] * s_window[2 * j + 1];
                        }
                    }
                    else
                    {
                        for (int j = 0; j < HalfSize; j++)
                        {
                            long a = start + 2 * j, b = a + 1;
                            re[j] = (a < valid ? samples[a] : 0f) * s_window[2 * j];
                            im[j] = (b < valid ? samples[b] : 0f) * s_window[2 * j + 1];
                        }
                    }
                    PowerSpectrum(re, im, re2, im2, power);
                    long o = destOffset + (long)i * MelBins;
                    for (int m = 0; m < MelBins; m++)
                    {
                        MelFilter f = s_filters[m];
                        double sum = 0;
                        for (int k = 0; k < f.Weights.Length; k++) sum += power[f.FirstBin + k] * f.Weights[k];
                        double v = Math.Log10(Math.Max(sum, LogFloor));
                        dest[o + m] = ((float)v + NormalizationOffset) / NormalizationDivisor;
                    }
                }
            }
            if (blocks == 1) RunBlock(0);
            else Parallel.For(0, blocks, RunBlock);
        }

        /// <summary>|X[k]|^2, k = 0..200, of the 400 real samples packed as z[j] = x[2j] + i x[2j+1] in
        /// (<paramref name="re"/>, <paramref name="im"/>) (both clobbered).</summary>
        private static void PowerSpectrum(double[] re, double[] im, double[] re2, double[] im2, double[] power)
        {
            // Z = FFT200(z), in natural order in (zr, zi).
            Fft(ref re, ref im, ref re2, ref im2);
            for (int k = 0; k <= HalfSize; k++)
            {
                int a = k % HalfSize, b = (HalfSize - k) % HalfSize;
                double zr = re[a], zi = im[a], cr = re[b], ci = -im[b];
                // E = (Z[k] + conj Z[N/2 - k]) / 2 (the even samples' DFT), O = (Z[k] - conj Z[N/2 - k]) / 2i (the odd ones').
                double er = 0.5 * (zr + cr), ei = 0.5 * (zi + ci);
                double odr = 0.5 * (zi - ci), odi = -0.5 * (zr - cr);
                double wr = s_postCos[k], wi = s_postSin[k];
                double xr = er + (odr * wr - odi * wi), xi = ei + (odr * wi + odi * wr);
                power[k] = xr * xr + xi * xi;
            }
        }

        /// <summary>The 200-point complex FFT (Stockham autosort, decimation in frequency: stage (p, n, s) maps
        /// a_r = x[t + s (q + r m)], r &lt; p, to y[t + s (p q + k)] = (sum_r a_r w_p^(r k)) W_n^(q k), m = n / p); the
        /// result ends in (re, im), whose buffers may be swapped with the scratch pair.</summary>
        private static void Fft(ref double[] re, ref double[] im, ref double[] re2, ref double[] im2)
        {
            foreach (Stage st in s_stages)
            {
                switch (st.Radix)
                {
                    case 5: Radix5(st, re, im, re2, im2); break;
                    case 4: Radix4(st, re, im, re2, im2); break;
                    default: Radix2(st, re, im, re2, im2); break;
                }
                (re, re2) = (re2, re);
                (im, im2) = (im2, im);
            }
        }

        private static readonly double s_c1 = Math.Cos(2 * Math.PI / 5), s_c2 = Math.Cos(4 * Math.PI / 5);
        private static readonly double s_s1 = Math.Sin(2 * Math.PI / 5), s_s2 = Math.Sin(4 * Math.PI / 5);

        private static void Radix5(Stage st, double[] xr, double[] xi, double[] yr, double[] yi)
        {
            int s = st.Stride, m = st.Length / 5;
            double c1 = s_c1, c2 = s_c2, s1 = s_s1, s2 = s_s2;
            double[] tr = st.TwiddleRe, ti = st.TwiddleIm;
            for (int q = 0; q < m; q++)
            {
                double w1r = tr[q * 5 + 1], w1i = ti[q * 5 + 1], w2r = tr[q * 5 + 2], w2i = ti[q * 5 + 2];
                double w3r = tr[q * 5 + 3], w3i = ti[q * 5 + 3], w4r = tr[q * 5 + 4], w4i = ti[q * 5 + 4];
                for (int t = 0; t < s; t++)
                {
                    int i0 = t + s * q, sm = s * m;
                    double a0r = xr[i0], a0i = xi[i0];
                    double a1r = xr[i0 + sm], a1i = xi[i0 + sm], a2r = xr[i0 + 2 * sm], a2i = xi[i0 + 2 * sm];
                    double a3r = xr[i0 + 3 * sm], a3i = xi[i0 + 3 * sm], a4r = xr[i0 + 4 * sm], a4i = xi[i0 + 4 * sm];
                    double t1r = a1r + a4r, t1i = a1i + a4i, t2r = a2r + a3r, t2i = a2i + a3i;
                    double t3r = a1r - a4r, t3i = a1i - a4i, t4r = a2r - a3r, t4i = a2i - a3i;
                    double m1r = a0r + c1 * t1r + c2 * t2r, m1i = a0i + c1 * t1i + c2 * t2i;
                    double m2r = a0r + c2 * t1r + c1 * t2r, m2i = a0i + c2 * t1i + c1 * t2i;
                    double n1r = s1 * t3r + s2 * t4r, n1i = s1 * t3i + s2 * t4i;
                    double n2r = s2 * t3r - s1 * t4r, n2i = s2 * t3i - s1 * t4i;
                    // b1 = m1 - i n1, b4 = m1 + i n1, b2 = m2 - i n2, b3 = m2 + i n2.
                    double b1r = m1r + n1i, b1i = m1i - n1r, b4r = m1r - n1i, b4i = m1i + n1r;
                    double b2r = m2r + n2i, b2i = m2i - n2r, b3r = m2r - n2i, b3i = m2i + n2r;
                    int o = t + s * 5 * q;
                    yr[o] = a0r + t1r + t2r;
                    yi[o] = a0i + t1i + t2i;
                    yr[o + s] = b1r * w1r - b1i * w1i;
                    yi[o + s] = b1r * w1i + b1i * w1r;
                    yr[o + 2 * s] = b2r * w2r - b2i * w2i;
                    yi[o + 2 * s] = b2r * w2i + b2i * w2r;
                    yr[o + 3 * s] = b3r * w3r - b3i * w3i;
                    yi[o + 3 * s] = b3r * w3i + b3i * w3r;
                    yr[o + 4 * s] = b4r * w4r - b4i * w4i;
                    yi[o + 4 * s] = b4r * w4i + b4i * w4r;
                }
            }
        }

        private static void Radix4(Stage st, double[] xr, double[] xi, double[] yr, double[] yi)
        {
            int s = st.Stride, m = st.Length / 4;
            double[] tr = st.TwiddleRe, ti = st.TwiddleIm;
            for (int q = 0; q < m; q++)
            {
                double w1r = tr[q * 4 + 1], w1i = ti[q * 4 + 1], w2r = tr[q * 4 + 2], w2i = ti[q * 4 + 2], w3r = tr[q * 4 + 3], w3i = ti[q * 4 + 3];
                for (int t = 0; t < s; t++)
                {
                    int i0 = t + s * q, sm = s * m;
                    double a0r = xr[i0], a0i = xi[i0], a1r = xr[i0 + sm], a1i = xi[i0 + sm];
                    double a2r = xr[i0 + 2 * sm], a2i = xi[i0 + 2 * sm], a3r = xr[i0 + 3 * sm], a3i = xi[i0 + 3 * sm];
                    // w4 = -i: b0 = a0 + a1 + a2 + a3, b1 = a0 - i a1 - a2 + i a3, b2 = a0 - a1 + a2 - a3, b3 = a0 + i a1 - a2 - i a3.
                    double s02r = a0r + a2r, s02i = a0i + a2i, d02r = a0r - a2r, d02i = a0i - a2i;
                    double s13r = a1r + a3r, s13i = a1i + a3i, d13r = a1r - a3r, d13i = a1i - a3i;
                    double b1r = d02r + d13i, b1i = d02i - d13r, b2r = s02r - s13r, b2i = s02i - s13i;
                    double b3r = d02r - d13i, b3i = d02i + d13r;
                    int o = t + s * 4 * q;
                    yr[o] = s02r + s13r;
                    yi[o] = s02i + s13i;
                    yr[o + s] = b1r * w1r - b1i * w1i;
                    yi[o + s] = b1r * w1i + b1i * w1r;
                    yr[o + 2 * s] = b2r * w2r - b2i * w2i;
                    yi[o + 2 * s] = b2r * w2i + b2i * w2r;
                    yr[o + 3 * s] = b3r * w3r - b3i * w3i;
                    yi[o + 3 * s] = b3r * w3i + b3i * w3r;
                }
            }
        }

        private static void Radix2(Stage st, double[] xr, double[] xi, double[] yr, double[] yi)
        {
            int s = st.Stride, m = st.Length / 2;
            double[] tr = st.TwiddleRe, ti = st.TwiddleIm;
            for (int q = 0; q < m; q++)
            {
                double wr = tr[q * 2 + 1], wi = ti[q * 2 + 1];
                for (int t = 0; t < s; t++)
                {
                    int i0 = t + s * q, i1 = i0 + s * m;
                    double dr = xr[i0] - xr[i1], di = xi[i0] - xi[i1];
                    int o = t + s * 2 * q;
                    yr[o] = xr[i0] + xr[i1];
                    yi[o] = xi[i0] + xi[i1];
                    yr[o + s] = dr * wr - di * wi;
                    yi[o + s] = dr * wi + di * wr;
                }
            }
        }

        private static Stage[] BuildStages()
        {
            var stages = new List<Stage>();
            int n = HalfSize, s = 1;
            foreach (int p in s_radices)
            {
                int m = n / p;
                var st = new Stage { Radix = p, Length = n, Stride = s, TwiddleRe = new double[m * p], TwiddleIm = new double[m * p] };
                for (int q = 0; q < m; q++)
                    for (int k = 0; k < p; k++)
                    {
                        double a = -2 * Math.PI * ((long)q * k % n) / n;
                        st.TwiddleRe[q * p + k] = Math.Cos(a);
                        st.TwiddleIm[q * p + k] = Math.Sin(a);
                    }
                stages.Add(st);
                n = m;
                s *= p;
            }
            if (n != 1) throw new InvalidOperationException("the FFT radices do not multiply to 200");
            return stages.ToArray();
        }

        /// <summary>The periodic Hann window, 0.5 (1 - cos(2 pi i / 400)).</summary>
        private static double[] BuildWindow()
        {
            var w = new double[WindowLength];
            for (int i = 0; i < w.Length; i++) w[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / WindowLength));
            return w;
        }

        /// <summary>The Slaney-scale, Slaney-area-normalized triangular filterbank over 0..8 kHz, as llama.cpp
        /// <c>fill_mel_filterbank_matrix</c> builds it (double arithmetic, float weights) and HF <c>mel_filter_bank(norm =
        /// "slaney", mel_scale = "slaney")</c>.</summary>
        private static MelFilter[] BuildFilters()
        {
            const double MinLogHz = 1000.0, LinSlope = 3 / 200.0, MinLogMel = MinLogHz * LinSlope;
            double logStep = Math.Log(6.4) / 27.0;
            double HzToMel(double f) => f < MinLogHz ? f * LinSlope : MinLogMel + Math.Log(f / MinLogHz) / logStep;
            double MelToHz(double m) => m < MinLogMel ? m / LinSlope : MinLogHz * Math.Exp((m - MinLogMel) * logStep);
            double lo = HzToMel(0), hi = HzToMel(SampleRate / 2.0);
            var hz = new double[MelBins + 2];
            for (int i = 0; i < hz.Length; i++) hz[i] = MelToHz(lo + (hi - lo) * (i / (double)(MelBins + 1)));
            double binHz = SampleRate / (double)FftSize;
            var filters = new MelFilter[MelBins];
            for (int m = 0; m < MelBins; m++)
            {
                double left = hz[m], center = hz[m + 1], right = hz[m + 2];
                double denomL = Math.Max(1e-30, center - left), denomR = Math.Max(1e-30, right - center);
                double enorm = 2.0 / Math.Max(1e-30, right - left);
                var weights = new float[FrequencyBins];
                int first = -1, last = -1;
                for (int k = 0; k < FrequencyBins; k++)
                {
                    double f = k * binHz, w = 0;
                    if (f >= left && f <= center) w = (f - left) / denomL;
                    else if (f > center && f <= right) w = (right - f) / denomR;
                    weights[k] = (float)(w * enorm);
                    if (weights[k] != 0)
                    {
                        if (first < 0) first = k;
                        last = k;
                    }
                }
                filters[m] = first < 0 ? new MelFilter(0, Array.Empty<float>()) : new MelFilter(first, weights[first..(last + 1)]);
            }
            return filters;
        }

        /// <summary>Filterbank weight of mel <paramref name="mel"/> at frequency bin <paramref name="bin"/> (tests).</summary>
        internal static float FilterWeight(int mel, int bin)
        {
            MelFilter f = s_filters[mel];
            int i = bin - f.FirstBin;
            return i >= 0 && i < f.Weights.Length ? f.Weights[i] : 0f;
        }

        /// <summary>|X[k]|^2 of 400 real samples through the FFT path (tests compare it with a direct DFT).</summary>
        internal static double[] PowerSpectrumOf(ReadOnlySpan<double> frame)
        {
            if (frame.Length != FftSize) throw new ArgumentException("a 400-sample frame");
            var re = new double[HalfSize];
            var im = new double[HalfSize];
            for (int j = 0; j < HalfSize; j++)
            {
                re[j] = frame[2 * j];
                im[j] = frame[2 * j + 1];
            }
            var power = new double[FrequencyBins];
            PowerSpectrum(re, im, new double[HalfSize], new double[HalfSize], power);
            return power;
        }

        // ================================================================ audio files

        /// <summary>
        /// A file's audio as 16 kHz mono: WAV, MP3 and Ogg Vorbis through the managed decoders, anything else through the
        /// platform decoder on <see cref="MediaCodecs.Audio"/> (ffmpeg on desktop hosts, AVFoundation on Apple apps: .flac,
        /// .m4a, .aac, the soundtrack of an .mp4 / .mov / .webm). Channels are averaged; other rates are resampled with the
        /// band-limited windowed-sinc resampler the Gemma 4 front end uses (<see cref="ToMono16k"/>).
        /// </summary>
        public static float[] LoadAudio(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"audio file not found: {path}", path);
            DecodedAudio raw = AudioIO.TryDecodeManaged(path) ?? MediaCodecs.Audio.Decode(path);
            return ToMono16k(raw);
        }

        /// <summary>Decoded PCM folded to mono (the channels' mean) and resampled to 16 kHz (band-limited windowed sinc,
        /// <see cref="Gemma4AudioResampler"/>; a 16 kHz source passes through unchanged).</summary>
        public static float[] ToMono16k(DecodedAudio audio)
        {
            if (audio?.Channels is not { Length: > 0 } || audio.SampleRate <= 0)
                throw new InvalidDataException("the audio decoder returned no samples");
            int n = audio.SampleCount;
            foreach (float[] channel in audio.Channels)
                if (channel == null || channel.Length != n) throw new InvalidDataException("the audio decoder returned channels of different lengths");
            float[] mono;
            if (audio.ChannelCount == 1)
            {
                mono = audio.Channels[0];
            }
            else
            {
                mono = new float[n];
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int c = 0; c < audio.ChannelCount; c++) sum += audio.Channels[c][i];
                    mono[i] = (float)(sum / audio.ChannelCount);
                }
            }
            foreach (float v in mono)
                if (!float.IsFinite(v)) throw new InvalidDataException("the audio holds a non-finite sample");
            return audio.SampleRate == SampleRate ? mono : Gemma4AudioResampler.Resample(mono, audio.SampleRate, SampleRate);
        }
    }

    /// <summary>
    /// The front end over a live stream of 16 kHz samples: frame k is produced once samples [k * 160, k * 160 + 400) have
    /// arrived; <see cref="Finish"/> pads the tail exactly as the whole clip is padded, so a stream's frames equal
    /// <see cref="Lfm3AudioPreprocessor.ComputeFeatures"/> of everything pushed. Keeps only the samples later frames read.
    /// </summary>
    public sealed class Lfm3AudioFeatureStream
    {
        private float[] _pending = new float[Lfm3AudioPreprocessor.WindowLength * 4];
        private int _pendingCount;
        private long _pendingStart;   // stream index of _pending[0]
        private long _received;
        private int _emitted;
        private bool _finished;

        /// <summary>Samples pushed so far.</summary>
        public long SamplesReceived => _received;

        /// <summary>Frames returned so far.</summary>
        public int FramesEmitted => _emitted;

        /// <summary>Whether <see cref="Finish"/> ran.</summary>
        public bool Finished => _finished;

        /// <summary>Append samples.</summary>
        public void Push(ReadOnlySpan<float> samples)
        {
            if (_finished) throw new InvalidOperationException("the audio stream has finished");
            if (_pendingCount + samples.Length > _pending.Length)
                Array.Resize(ref _pending, Math.Max(_pending.Length * 2, _pendingCount + samples.Length));
            samples.CopyTo(_pending.AsSpan(_pendingCount));
            _pendingCount += samples.Length;
            _received += samples.Length;
        }

        /// <summary>The frames complete since the last call (frame-major [frames][128]; possibly none).</summary>
        public float[] Read(out int frames) => Take(Lfm3AudioPreprocessor.CompleteFrames(_received), out frames);

        /// <summary>End the stream: the remaining frames, the tail padded as the whole clip would be.</summary>
        public float[] Finish(out int frames)
        {
            if (_finished) throw new InvalidOperationException("the audio stream has finished");
            _finished = true;
            return Take(_received == 0 ? 0 : Lfm3AudioPreprocessor.FrameCount(_received), out frames);
        }

        private float[] Take(int total, out int frames)
        {
            frames = Math.Max(0, total - _emitted);
            var features = new float[(long)frames * Lfm3AudioPreprocessor.MelBins];
            if (frames > 0)
            {
                Lfm3AudioPreprocessor.ComputeFrames(_pending, _pendingStart, _received, _emitted, frames, features, 0);
                _emitted += frames;
            }
            // Drop the samples no later frame reads (frame _emitted starts at _emitted * 160).
            long keepFrom = Math.Min(_received, (long)_emitted * Lfm3AudioPreprocessor.HopLength);
            int drop = (int)(keepFrom - _pendingStart);
            if (drop > 0)
            {
                Array.Copy(_pending, drop, _pending, 0, _pendingCount - drop);
                _pendingCount -= drop;
                _pendingStart = keepFrom;
            }
            return features;
        }
    }
}
