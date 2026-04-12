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
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using NLayer;
using NVorbis;

namespace TensorSharp.Models
{
    public class Gemma4AudioPreprocessor
    {
        private const int SampleRate = 16000;
        private const int MelBins = 128;
        private const double FrameLengthMs = 20.0;
        private const double HopLengthMs = 10.0;
        private const double MinFrequency = 0.0;
        private const double MaxFrequency = 8000.0;
        private const double MelFloor = 1e-3;

        private static readonly int FrameLength = (int)Math.Round(SampleRate * FrameLengthMs / 1000.0); // 320
        private static readonly int HopLength = (int)Math.Round(SampleRate * HopLengthMs / 1000.0); // 160
        private static readonly int FftLength = ComputeFftLength();
        private static readonly int NumFreqBins = FftLength / 2 + 1;
        private static readonly double[] HannWindow = BuildWindow();
        private static readonly float[] MelFilters = BuildMelFilterBank(NumFreqBins, MelBins, MinFrequency, MaxFrequency, SampleRate);

        public static float[] DecodeAudioFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext switch
            {
                ".wav" => DecodeWAV(File.ReadAllBytes(path)),
                ".mp3" => DecodeMp3(path),
                ".ogg" => DecodeOgg(path),
                _ => throw new NotSupportedException(
                    $"Audio format '{ext}' is not supported. Supported formats: .wav, .mp3, .ogg")
            };
        }

        private static float[] DecodeMp3(string path)
        {
            using var stream = File.OpenRead(path);
            var reader = new MpegFile(stream);
            int mpegSampleRate = reader.SampleRate;
            int channels = reader.Channels;

            var allSamples = new System.Collections.Generic.List<float>();
            float[] readBuf = new float[4096];
            int samplesRead;
            while ((samplesRead = reader.ReadSamples(readBuf, 0, readBuf.Length)) > 0)
            {
                for (int i = 0; i < samplesRead; i++)
                    allSamples.Add(readBuf[i]);
            }

            float[] interleaved = allSamples.ToArray();
            int totalFrames = interleaved.Length / channels;
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += interleaved[i * channels + ch];
                mono[i] = sum / channels;
            }

            if (mpegSampleRate != SampleRate)
                mono = ResampleLinear(mono, mpegSampleRate, SampleRate);
            return mono;
        }

        private static float[] DecodeOgg(string path)
        {
            using var vorbis = new VorbisReader(path);
            int oggSampleRate = vorbis.SampleRate;
            int channels = vorbis.Channels;

            var allSamples = new System.Collections.Generic.List<float>();
            float[] readBuf = new float[4096];
            int samplesRead;
            while ((samplesRead = vorbis.ReadSamples(readBuf, 0, readBuf.Length)) > 0)
            {
                for (int i = 0; i < samplesRead; i++)
                    allSamples.Add(readBuf[i]);
            }

            float[] interleaved = allSamples.ToArray();
            int totalFrames = interleaved.Length / channels;
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += interleaved[i * channels + ch];
                mono[i] = sum / channels;
            }

            if (oggSampleRate != SampleRate)
                mono = ResampleLinear(mono, oggSampleRate, SampleRate);
            return mono;
        }

        public static float[] DecodeWAV(byte[] data)
        {
            if (data.Length < 12 || System.Text.Encoding.ASCII.GetString(data, 0, 4) != "RIFF" ||
                System.Text.Encoding.ASCII.GetString(data, 8, 4) != "WAVE")
                throw new Exception("Not a WAV file");

            ushort audioFormat = 0;
            int numChannels = 0, sampleRate = 0, bitsPerSample = 0;
            byte[] audioData = null;
            bool foundFmt = false;

            int offset = 12;
            while (offset + 8 <= data.Length)
            {
                string chunkId = System.Text.Encoding.ASCII.GetString(data, offset, 4);
                int chunkSize = BitConverter.ToInt32(data, offset + 4);
                int chunkEnd = Math.Min(offset + 8 + chunkSize, data.Length);

                if (chunkId == "fmt " && chunkEnd - offset - 8 >= 16)
                {
                    audioFormat = BitConverter.ToUInt16(data, offset + 8);
                    numChannels = BitConverter.ToUInt16(data, offset + 10);
                    sampleRate = BitConverter.ToInt32(data, offset + 12);
                    bitsPerSample = BitConverter.ToUInt16(data, offset + 22);
                    if (audioFormat == 0xFFFE && chunkEnd - offset - 8 >= 26)
                        audioFormat = BitConverter.ToUInt16(data, offset + 32);
                    foundFmt = true;
                }
                else if (chunkId == "data")
                {
                    audioData = new byte[chunkEnd - offset - 8];
                    Array.Copy(data, offset + 8, audioData, 0, audioData.Length);
                }

                offset += 8 + chunkSize;
                if (chunkSize % 2 != 0) offset++;
            }

            if (!foundFmt) throw new Exception("No fmt chunk in WAV");
            if (audioFormat != 1 && audioFormat != 3)
                throw new Exception($"Unsupported WAV format: {audioFormat}");
            if (audioData == null) throw new Exception("No data chunk in WAV");

            float[] samples = DecodeSamples(audioData, audioFormat, bitsPerSample, numChannels);
            if (sampleRate != SampleRate)
                samples = ResampleLinear(samples, sampleRate, SampleRate);

            return samples;
        }

        private static float[] DecodeSamples(byte[] data, ushort format, int bits, int channels)
        {
            int bytesPerSample = bits / 8;
            int totalSamples = data.Length / (bytesPerSample * channels);
            float[] mono = new float[totalSamples];

            for (int i = 0; i < totalSamples; i++)
            {
                double sum = 0;
                for (int ch = 0; ch < channels; ch++)
                {
                    int off = (i * channels + ch) * bytesPerSample;
                    if (off + bytesPerSample > data.Length) break;

                    if (format == 1 && bits == 16)
                        sum += BitConverter.ToInt16(data, off) / 32768.0;
                    else if (format == 1 && bits == 32)
                        sum += BitConverter.ToInt32(data, off) / 2147483648.0;
                    else if (format == 1 && bits == 24)
                    {
                        int v = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16);
                        if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                        sum += v / 8388608.0;
                    }
                    else if (format == 3 && bits == 32)
                        sum += BitConverter.ToSingle(data, off);
                    else if (format == 1 && bits == 8)
                        sum += (data[off] - 128.0) / 128.0;
                }
                mono[i] = (float)(sum / channels);
            }
            return mono;
        }

        private static float[] ResampleLinear(float[] samples, int fromRate, int toRate)
        {
            int n = (int)((double)samples.Length / fromRate * toRate);
            float[] output = new float[n];
            for (int i = 0; i < n; i++)
            {
                double pos = (double)i * (samples.Length - 1) / (n - 1);
                int idx = (int)pos;
                float frac = (float)(pos - idx);
                output[i] = idx + 1 < samples.Length
                    ? samples[idx] * (1 - frac) + samples[idx + 1] * frac
                    : samples[idx];
            }
            return output;
        }

        public static (float[] melData, int numFrames) ComputeMelSpectrogram(float[] samples)
        {
            int frameSizeForUnfold = FrameLength + 1;
            int numFrames = (samples.Length - frameSizeForUnfold) / HopLength;
            if (numFrames <= 0) return (null, 0);

            float[] result = new float[numFrames * MelBins];
            if (numFrames < 8)
            {
                var fftInput = new Complex[FftLength];
                for (int f = 0; f < numFrames; f++)
                    ComputeMelFrame(samples, f, fftInput, result);
            }
            else
            {
                Parallel.For(0, numFrames,
                    () => new Complex[FftLength],
                    (f, _, fftInput) =>
                    {
                        ComputeMelFrame(samples, f, fftInput, result);
                        return fftInput;
                    },
                    _ => { });
            }

            return (result, numFrames);
        }

        private static float[] BuildMelFilterBank(int numFreqBins, int numMels, double fMin, double fMax, int sr)
        {
            double HzToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
            double MelToHz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);

            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);

            double[] melPts = new double[numMels + 2];
            for (int i = 0; i < melPts.Length; i++)
                melPts[i] = melMin + (double)i * (melMax - melMin) / (numMels + 1);
            double[] filterFreqs = new double[numMels + 2];
            for (int i = 0; i < melPts.Length; i++)
                filterFreqs[i] = MelToHz(melPts[i]);

            double[] fftFreqs = new double[numFreqBins];
            for (int i = 0; i < numFreqBins; i++)
                fftFreqs[i] = (double)i * sr / (2.0 * (numFreqBins - 1));

            float[] filters = new float[numFreqBins * numMels];
            for (int m = 0; m < numMels; m++)
            {
                double fLeft = filterFreqs[m];
                double fCenter = filterFreqs[m + 1];
                double fRight = filterFreqs[m + 2];
                for (int k = 0; k < numFreqBins; k++)
                {
                    double f = fftFreqs[k];
                    double v = 0;
                    if (f >= fLeft && f <= fCenter && fCenter > fLeft)
                        v = (f - fLeft) / (fCenter - fLeft);
                    else if (f > fCenter && f <= fRight && fRight > fCenter)
                        v = (fRight - f) / (fRight - fCenter);
                    if (v > 0)
                        filters[k * numMels + m] = (float)v;
                }
            }
            return filters;
        }

        private static void FFT(Complex[] x)
        {
            int n = x.Length;
            if (n <= 1) return;

            int j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
                j ^= bit;
                if (i < j) { var tmp = x[i]; x[i] = x[j]; x[j] = tmp; }
            }

            for (int size = 2; size <= n; size <<= 1)
            {
                int halfSize = size / 2;
                Complex w = new Complex(Math.Cos(2 * Math.PI / size), -Math.Sin(2 * Math.PI / size));
                for (int start = 0; start < n; start += size)
                {
                    Complex wn = Complex.One;
                    for (int k = 0; k < halfSize; k++)
                    {
                        Complex t = wn * x[start + k + halfSize];
                        x[start + k + halfSize] = x[start + k] - t;
                        x[start + k] = x[start + k] + t;
                        wn *= w;
                    }
                }
            }
        }

        public static int ComputeAudioTokenCount(float[] samples)
        {
            if (samples.Length % 128 != 0)
            {
                int padded = samples.Length + (128 - samples.Length % 128);
                samples = new float[padded];
            }
            int frameSizeForUnfold = FrameLength + 1;
            int numFrames = (samples.Length - frameSizeForUnfold) / HopLength;
            if (numFrames <= 0) return 0;

            int tConv0 = (numFrames + 2 - 3) / 2 + 1;
            int tConv1 = (tConv0 + 2 - 3) / 2 + 1;
            return tConv1;
        }

        private static void ComputeMelFrame(float[] samples, int frameIndex, Complex[] fftInput, float[] result)
        {
            int start = frameIndex * HopLength;
            for (int i = 0; i < FrameLength; i++)
                fftInput[i] = new Complex(samples[start + i] * HannWindow[i], 0);
            for (int i = FrameLength; i < FftLength; i++)
                fftInput[i] = Complex.Zero;

            FFT(fftInput);

            int dstOffset = frameIndex * MelBins;
            for (int m = 0; m < MelBins; m++)
            {
                double melVal = 0;
                for (int k = 0; k < NumFreqBins; k++)
                    melVal += fftInput[k].Magnitude * MelFilters[k * MelBins + m];
                if (melVal < MelFloor) melVal = MelFloor;
                result[dstOffset + m] = (float)Math.Log(melVal);
            }
        }

        private static int ComputeFftLength()
        {
            int fftLen = 1;
            while (fftLen < FrameLength) fftLen <<= 1;
            return fftLen * 2;
        }

        private static double[] BuildWindow()
        {
            double[] window = new double[FrameLength];
            double arg = Math.PI * 2.0 / FrameLength;
            for (int i = 0; i < FrameLength; i++)
                window[i] = 0.5 - 0.5 * Math.Cos(arg * (i + 0.5));
            return window;
        }
    }
}

