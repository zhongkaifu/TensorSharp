// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Buffers.Binary;

using TensorSharp.Models.Video;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// Speech output encodings, byte-compatible with llama.cpp's speech endpoints (mtmd-helper-gen.cpp write_wav16 /
    /// append_wav16_pcm): mono 16-bit little-endian PCM (each sample clamped to [-1, 1], times 32767, rounded half away from
    /// zero), a RIFF/WAVE file of it, and a streaming WAV header whose sizes are 0xFFFFFFFF ("unknown length", as ffmpeg
    /// writes on a pipe) followed by PCM16 chunks as frames are decoded.
    /// </summary>
    public static class Lfm3SpeechAudio
    {
        /// <summary>The PCM16 bytes of <paramref name="samples"/>.</summary>
        public static byte[] Pcm16(ReadOnlySpan<float> samples)
        {
            var bytes = new byte[samples.Length * 2];
            WritePcm16(samples, bytes);
            return bytes;
        }

        /// <summary>Write <paramref name="samples"/> as PCM16 into <paramref name="destination"/> (2 bytes a sample).</summary>
        public static void WritePcm16(ReadOnlySpan<float> samples, Span<byte> destination)
        {
            if (destination.Length < samples.Length * 2) throw new ArgumentException("2 bytes a sample", nameof(destination));
            for (int i = 0; i < samples.Length; i++)
            {
                short s = (short)MathF.Round(Math.Clamp(samples[i], -1f, 1f) * 32767f, MidpointRounding.AwayFromZero);
                BinaryPrimitives.WriteInt16LittleEndian(destination.Slice(2 * i), s);
            }
        }

        /// <summary>A whole RIFF/WAVE file (mono PCM16) of <paramref name="samples"/>.</summary>
        public static byte[] Wav(float[] samples, int sampleRate)
            => WavWriter.Encode(new[] { samples ?? throw new ArgumentNullException(nameof(samples)) }, sampleRate);

        /// <summary>The 44-byte header of a WAV stream of unknown length (RIFF and data sizes 0xFFFFFFFF), mono PCM16; the
        /// stream's PCM16 chunks (<see cref="Pcm16"/>) follow it.</summary>
        public static byte[] StreamingWavHeader(int sampleRate) => WavHeader(uint.MaxValue, sampleRate);

        /// <summary>The 44-byte mono PCM16 WAV header for <paramref name="dataBytes"/> bytes of samples (uint.MaxValue: a
        /// stream of unknown length).</summary>
        public static byte[] WavHeader(uint dataBytes, int sampleRate)
        {
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            var h = new byte[44];
            "RIFF"u8.CopyTo(h);
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(4), dataBytes == uint.MaxValue ? uint.MaxValue : 36 + dataBytes);
            "WAVE"u8.CopyTo(h.AsSpan(8));
            "fmt "u8.CopyTo(h.AsSpan(12));
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(16), 16);
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(20), 1);                       // PCM
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(22), 1);                       // mono
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(24), (uint)sampleRate);
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(28), (uint)sampleRate * 2);    // byte rate
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(32), 2);                       // block align
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(34), 16);                      // bits
            "data"u8.CopyTo(h.AsSpan(36));
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(40), dataBytes);
            return h;
        }
    }
}
