// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using NLayer;
using NVorbis;

namespace TensorSharp.Models
{
    /// <summary>
    /// Parakeet-style audio preprocessor used by the NVIDIA Nemotron 3 Nano Omni
    /// audio encoder. Mirrors the ollama <c>process_audio.go</c> reference exactly:
    ///   * mono resample to 16 kHz
    ///   * 0.97 pre-emphasis
    ///   * STFT(n_fft=512, hop=160, win=400, center=True, pad_mode="constant")
    ///   * Slaney-style mel filter bank (128 bins, 0..8 kHz)
    ///   * log(power + 2^-24) followed by per-mel mean/var normalization across
    ///     the valid (non-padded) frames.
    /// </summary>
    public static class NemotronAudioPreprocessor
    {
        public const int SampleRate = 16000;
        public const int MelBins = 128;
        public const int NFFT = 512;
        public const int WinLength = 400;
        public const int HopLength = 160;
        public const float PreEmphasis = 0.97f;
        public const float LogZeroGuard = 1.0f / (1 << 24);
        public const float NormalizeEps = 1e-5f;

        private static readonly float[] Window = BuildHannWindow();
        private static readonly float[] MelFilters = BuildSlaneyMelFilterBank(NFFT / 2 + 1, MelBins, SampleRate);

        public static float[] DecodeAudioFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext switch
            {
                ".wav" => DecodeWAV(File.ReadAllBytes(path)),
                ".mp3" => DecodeMp3(path),
                ".ogg" => DecodeOgg(path),
                _ => throw new NotSupportedException(
                    $"Audio format '{ext}' is not supported by Nemotron Omni. Supported: .wav, .mp3, .ogg")
            };
        }

        public static float[] DecodeMp3(string path)
        {
            using var stream = File.OpenRead(path);
            var reader = new MpegFile(stream);
            int srcSampleRate = reader.SampleRate;
            int channels = reader.Channels;

            int chunk = 4096;
            float[] readBuf = new float[chunk];
            using var allSamples = new ResizableFloatBuffer(srcSampleRate * 16);
            int n;
            while ((n = reader.ReadSamples(readBuf, 0, chunk)) > 0)
                allSamples.Append(readBuf.AsSpan(0, n));

            return ToMono16k(allSamples.AsSpan(), srcSampleRate, channels);
        }

        public static float[] DecodeOgg(string path)
        {
            using var vorbis = new VorbisReader(path);
            int srcSampleRate = vorbis.SampleRate;
            int channels = vorbis.Channels;

            int chunk = 4096;
            float[] readBuf = new float[chunk];
            using var allSamples = new ResizableFloatBuffer(srcSampleRate * 16);
            int n;
            while ((n = vorbis.ReadSamples(readBuf, 0, chunk)) > 0)
                allSamples.Append(readBuf.AsSpan(0, n));

            return ToMono16k(allSamples.AsSpan(), srcSampleRate, channels);
        }

        public static float[] DecodeWAV(byte[] data)
        {
            if (data.Length < 12) throw new InvalidDataException("WAV file too short");
            if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F' ||
                data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E')
                throw new InvalidDataException("Not a WAV file");

            int offset = 12;
            ushort audioFormat = 0;
            int numChannels = 0, sampleRate = 0, bitsPerSample = 0;
            byte[] audioData = null;
            bool foundFmt = false;

            while (offset + 8 <= data.Length)
            {
                string chunkId = System.Text.Encoding.ASCII.GetString(data, offset, 4);
                int chunkSize = BitConverter.ToInt32(data, offset + 4);
                int chunkEnd = Math.Min(offset + 8 + chunkSize, data.Length);
                int dataStart = offset + 8;
                int dataLen = chunkEnd - dataStart;

                if (chunkId == "fmt ")
                {
                    if (dataLen < 16) throw new InvalidDataException("fmt chunk too short");
                    audioFormat = BitConverter.ToUInt16(data, dataStart);
                    numChannels = BitConverter.ToUInt16(data, dataStart + 2);
                    sampleRate = BitConverter.ToInt32(data, dataStart + 4);
                    bitsPerSample = BitConverter.ToUInt16(data, dataStart + 14);
                    if (audioFormat == 0xFFFE && dataLen >= 26)
                        audioFormat = BitConverter.ToUInt16(data, dataStart + 24);
                    foundFmt = true;
                }
                else if (chunkId == "data")
                {
                    audioData = new byte[dataLen];
                    Buffer.BlockCopy(data, dataStart, audioData, 0, dataLen);
                }

                offset += 8 + chunkSize;
                if ((chunkSize & 1) != 0) offset++;
            }

            if (!foundFmt) throw new InvalidDataException("WAV missing fmt chunk");
            if (audioData == null) throw new InvalidDataException("WAV missing data chunk");
            if (numChannels <= 0) throw new InvalidDataException("WAV invalid channel count");
            if (audioFormat != 1 && audioFormat != 3)
                throw new InvalidDataException($"Unsupported WAV format {audioFormat} (need PCM=1 or float=3)");

            float[] mono = DecodeWAVSamples(audioData, audioFormat, bitsPerSample, numChannels);
            if (sampleRate != SampleRate)
                mono = ResampleLinear(mono, sampleRate, SampleRate);
            return mono;
        }

        private static float[] DecodeWAVSamples(byte[] data, ushort format, int bits, int channels)
        {
            int bytesPerSample = bits / 8;
            if (bytesPerSample <= 0 || channels <= 0) return Array.Empty<float>();

            int totalFrames = data.Length / (bytesPerSample * channels);
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                double sum = 0;
                for (int ch = 0; ch < channels; ch++)
                {
                    int off = (i * channels + ch) * bytesPerSample;
                    if (off + bytesPerSample > data.Length) break;
                    if (format == 1 && bits == 16)
                    {
                        short v = BitConverter.ToInt16(data, off);
                        sum += v / 32768.0;
                    }
                    else if (format == 1 && bits == 32)
                    {
                        int v = BitConverter.ToInt32(data, off);
                        sum += v / 2147483648.0;
                    }
                    else if (format == 1 && bits == 24)
                    {
                        int v = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16);
                        if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                        sum += v / 8388608.0;
                    }
                    else if (format == 3 && bits == 32)
                    {
                        sum += BitConverter.ToSingle(data, off);
                    }
                    else if (format == 1 && bits == 8)
                    {
                        sum += (data[off] - 128.0) / 128.0;
                    }
                }
                mono[i] = (float)(sum / channels);
            }
            return mono;
        }

        private static float[] ToMono16k(ReadOnlySpan<float> interleaved, int srcSampleRate, int channels)
        {
            int totalFrames = interleaved.Length / channels;
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                float sum = 0;
                int baseIdx = i * channels;
                for (int ch = 0; ch < channels; ch++)
                    sum += interleaved[baseIdx + ch];
                mono[i] = sum / channels;
            }
            return srcSampleRate == SampleRate
                ? mono
                : ResampleLinear(mono, srcSampleRate, SampleRate);
        }

        private static float[] ResampleLinear(float[] samples, int fromRate, int toRate)
        {
            if (fromRate <= 0 || toRate <= 0 || samples.Length == 0) return samples;
            int n = (int)((double)samples.Length / fromRate * toRate);
            if (n <= 1) return new float[] { samples.Length > 0 ? samples[0] : 0 };
            float[] outArr = new float[n];
            double srcSpan = samples.Length - 1;
            for (int i = 0; i < n; i++)
            {
                double pos = i * srcSpan / (n - 1);
                int idx = (int)pos;
                float frac = (float)(pos - idx);
                outArr[i] = idx + 1 < samples.Length
                    ? samples[idx] * (1 - frac) + samples[idx + 1] * frac
                    : samples[idx];
            }
            return outArr;
        }

        /// <summary>
        /// Compute the Parakeet mel spectrogram. Returns (mel data laid out as
        /// <c>frames * MelBins</c> in time-major order, totalFrames, validFrames).
        /// </summary>
        public static (float[] mel, int frames, int validFrames) ComputeParakeetMelSpectrogram(float[] samples)
        {
            if (samples == null || samples.Length == 0)
                throw new ArgumentException("Audio is empty", nameof(samples));

            int freqBins = NFFT / 2 + 1;

            // Pre-emphasis.
            float[] emphasized = new float[samples.Length];
            emphasized[0] = samples[0];
            for (int i = 1; i < samples.Length; i++)
                emphasized[i] = samples[i] - PreEmphasis * samples[i - 1];

            int frames = samples.Length / HopLength + 1;
            int validFrames = Math.Max(1, samples.Length / HopLength);
            if (validFrames > frames) validFrames = frames;

            int winOffset = (NFFT - WinLength) / 2;
            int centerPad = NFFT / 2;
            float[] result = new float[frames * MelBins];

            // Threadable: each frame runs an independent FFT.
            Parallel.For(0, frames, frame =>
            {
                Span<Complex> fftBuf = stackalloc Complex[NFFT];
                for (int i = 0; i < WinLength; i++)
                {
                    int src = frame * HopLength + i + winOffset - centerPad;
                    if (src >= 0 && src < emphasized.Length)
                        fftBuf[i + winOffset] = new Complex(emphasized[src] * Window[i], 0);
                }

                FFTInPlace(fftBuf);

                int outOff = frame * MelBins;
                for (int mel = 0; mel < MelBins; mel++)
                {
                    double v = 0;
                    int filterOff = mel * freqBins;
                    for (int f = 0; f < freqBins; f++)
                    {
                        double mag = fftBuf[f].Magnitude;
                        v += MelFilters[filterOff + f] * mag * mag;
                    }
                    result[outOff + mel] = (float)Math.Log(v + LogZeroGuard);
                }
            });

            // Per-mel mean/var normalization over the valid frames.
            for (int mel = 0; mel < MelBins; mel++)
            {
                double sum = 0;
                for (int frame = 0; frame < validFrames; frame++)
                    sum += result[frame * MelBins + mel];
                double mean = sum / validFrames;

                double variance = 0;
                for (int frame = 0; frame < validFrames; frame++)
                {
                    double d = result[frame * MelBins + mel] - mean;
                    variance += d * d;
                }
                int denom = Math.Max(1, validFrames - 1);
                double std = Math.Sqrt(variance / denom);

                for (int frame = 0; frame < frames; frame++)
                {
                    int idx = frame * MelBins + mel;
                    if (frame >= validFrames)
                        result[idx] = 0;
                    else
                        result[idx] = (float)((result[idx] - mean) / (std + NormalizeEps));
                }
            }

            return (result, frames, validFrames);
        }

        private static float[] BuildHannWindow()
        {
            float[] win = new float[WinLength];
            for (int i = 0; i < WinLength; i++)
                win[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (WinLength - 1)));
            return win;
        }

        private static float[] BuildSlaneyMelFilterBank(int numFreqBins, int numMels, int sampleRate)
        {
            static double HzToMel(double f) => f < 1000 ? 3 * f / 200 : 15 + Math.Log(f / 1000) * 27 / Math.Log(6.4);
            static double MelToHz(double m) => m < 15 ? 200 * m / 3 : 1000 * Math.Exp(Math.Log(6.4) * (m - 15) / 27);

            double minMel = HzToMel(0);
            double maxMel = HzToMel(sampleRate / 2.0);
            double[] mels = new double[numMels + 2];
            double[] freqs = new double[numMels + 2];
            for (int i = 0; i < mels.Length; i++)
            {
                mels[i] = minMel + (maxMel - minMel) * i / (numMels + 1);
                freqs[i] = MelToHz(mels[i]);
            }

            double[] fftFreqs = new double[numFreqBins];
            for (int i = 0; i < numFreqBins; i++)
                fftFreqs[i] = (double)i * sampleRate / NFFT;

            float[] filters = new float[numMels * numFreqBins];
            for (int mel = 0; mel < numMels; mel++)
            {
                double left = freqs[mel], center = freqs[mel + 1], right = freqs[mel + 2];
                double enorm = 2.0 / (right - left);
                for (int f = 0; f < numFreqBins; f++)
                {
                    double fftFreq = fftFreqs[f];
                    double lower = 0, upper = 0;
                    if (center > left) lower = (fftFreq - left) / (center - left);
                    if (right > center) upper = (right - fftFreq) / (right - center);
                    double v = Math.Max(0, Math.Min(lower, upper));
                    filters[mel * numFreqBins + f] = (float)(v * enorm);
                }
            }
            return filters;
        }

        /// <summary>
        /// In-place radix-2 Cooley-Tukey FFT for power-of-two lengths.
        /// </summary>
        private static void FFTInPlace(Span<Complex> x)
        {
            int n = x.Length;
            int j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                while ((j & bit) != 0)
                {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                if (i < j)
                {
                    var tmp = x[i];
                    x[i] = x[j];
                    x[j] = tmp;
                }
            }

            for (int size = 2; size <= n; size <<= 1)
            {
                int halfSize = size / 2;
                var w = new Complex(Math.Cos(2 * Math.PI / size), -Math.Sin(2 * Math.PI / size));
                for (int start = 0; start < n; start += size)
                {
                    var wn = new Complex(1, 0);
                    for (int k = 0; k < halfSize; k++)
                    {
                        var t = wn * x[start + k + halfSize];
                        x[start + k + halfSize] = x[start + k] - t;
                        x[start + k] = x[start + k] + t;
                        wn *= w;
                    }
                }
            }
        }

        /// <summary>
        /// Tiny growable float buffer that lets the audio decoder loop avoid the
        /// O(N) cost of List&lt;float&gt;.Add for multi-MB inputs.
        /// </summary>
        private sealed class ResizableFloatBuffer : IDisposable
        {
            private float[] _arr;
            private int _len;

            public ResizableFloatBuffer(int initialCapacity)
            {
                _arr = ArrayPool<float>.Shared.Rent(Math.Max(1024, initialCapacity));
                _len = 0;
            }

            public void Append(ReadOnlySpan<float> data)
            {
                if (_len + data.Length > _arr.Length)
                {
                    int newCap = Math.Max(_arr.Length * 2, _len + data.Length);
                    var nu = ArrayPool<float>.Shared.Rent(newCap);
                    Buffer.BlockCopy(_arr, 0, nu, 0, _len * sizeof(float));
                    ArrayPool<float>.Shared.Return(_arr);
                    _arr = nu;
                }
                data.CopyTo(_arr.AsSpan(_len));
                _len += data.Length;
            }

            public ReadOnlySpan<float> AsSpan() => _arr.AsSpan(0, _len);

            public void Dispose()
            {
                if (_arr != null)
                    ArrayPool<float>.Shared.Return(_arr);
                _arr = null;
            }
        }
    }
}
