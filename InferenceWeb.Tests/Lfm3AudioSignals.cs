// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;

namespace InferenceWeb.Tests;

/// <summary>
/// The 16 kHz mono clips the LFM3 audio-input validation runs (tests, and eng/lfm3/audio-in/Lfm3AudioProbe, which links
/// this file and writes them for the llama.cpp reference): speech from the repo's ticket.wav (macOS "say", 5.85 s; also
/// repeated to 10 s and 30 s), a chirp, synthetic music, silence, odd-length noise and a clip shorter than one window.
/// Every value comes from IEEE additions, multiplications and divisions only (a polynomial sine, multiplicative decays, an
/// integer noise generator), so the samples are bit-identical on every platform and the llama.cpp fixtures stay valid.
/// </summary>
public static class Lfm3AudioSignals
{
    public const int Rate = 16000;

    /// <summary>The clips, in fixture order.</summary>
    public static readonly string[] Names =
    {
        "chirp_1s", "speech_6s", "speech_10s", "speech_30s", "silence_3s", "music_10s", "noise_odd", "tiny",
    };

    /// <summary>Clip <paramref name="name"/>; <paramref name="repoRoot"/> locates InferenceWeb.Tests/Fixtures.</summary>
    public static float[] Make(string name, string repoRoot) => name switch
    {
        "chirp_1s" => Chirp(Rate, 100, 7000, 0.3),
        "speech_6s" => Speech(repoRoot),
        "speech_10s" => Loop(Speech(repoRoot), 10 * Rate),
        "speech_30s" => Loop(Speech(repoRoot), 30 * Rate),
        "silence_3s" => new float[3 * Rate],
        "music_10s" => Music(10 * Rate),
        "noise_odd" => Noise(12345, 0.1, 99),
        "tiny" => Tone(300, 1000, 0.5),
        _ => throw new ArgumentException($"no LFM3 test clip '{name}'"),
    };

    /// <summary>ticket.wav: PCM16 at 16 kHz mono, each sample / 32768 (what llama.cpp's miniaudio and the managed WAV reader
    /// produce for it).</summary>
    public static float[] Speech(string repoRoot)
    {
        byte[] wav = File.ReadAllBytes(Path.Combine(repoRoot, "InferenceWeb.Tests", "Fixtures", "JevAttachments", "ticket.wav"));
        int at = 12;
        while (at + 8 <= wav.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(wav, at, 4);
            int size = BitConverter.ToInt32(wav, at + 4);
            if (id == "fmt ")
            {
                if (BitConverter.ToInt16(wav, at + 8) != 1 || BitConverter.ToInt16(wav, at + 10) != 1 ||
                    BitConverter.ToInt32(wav, at + 12) != Rate || BitConverter.ToInt16(wav, at + 22) != 16)
                    throw new InvalidDataException("ticket.wav is expected to be 16 kHz mono PCM16");
            }
            else if (id == "data")
            {
                var samples = new float[size / 2];
                for (int i = 0; i < samples.Length; i++) samples[i] = BitConverter.ToInt16(wav, at + 8 + 2 * i) / 32768f;
                return samples;
            }
            at += 8 + size + (size & 1);
        }
        throw new InvalidDataException("ticket.wav has no data chunk");
    }

    /// <summary><paramref name="clip"/>, a quarter second of silence, the clip again, ... cut at <paramref name="length"/>.</summary>
    public static float[] Loop(float[] clip, int length)
    {
        var y = new float[length];
        int at = 0;
        while (at < length)
        {
            int n = Math.Min(clip.Length, length - at);
            Array.Copy(clip, 0, y, at, n);
            at += n + Rate / 4;
        }
        return y;
    }

    /// <summary>A linear chirp from <paramref name="f0"/> to <paramref name="f1"/> Hz over the clip, plus faint noise.</summary>
    public static float[] Chirp(int length, double f0, double f1, double amplitude)
    {
        var y = new float[length];
        double phase = 0;
        uint state = 7;
        for (int i = 0; i < length; i++)
        {
            double f = f0 + (f1 - f0) * i / length;
            y[i] = (float)(amplitude * Sin(phase) + 0.01 * Uniform(ref state));
            phase += 2 * Math.PI * f / Rate;
            if (phase > Math.PI) phase -= 2 * Math.PI;
        }
        return y;
    }

    public static float[] Tone(int length, double f, double amplitude)
    {
        var y = new float[length];
        double phase = 0;
        for (int i = 0; i < length; i++)
        {
            y[i] = (float)(amplitude * Sin(phase));
            phase += 2 * Math.PI * f / Rate;
            if (phase > Math.PI) phase -= 2 * Math.PI;
        }
        return y;
    }

    public static float[] Noise(int length, double amplitude, uint seed)
    {
        var y = new float[length];
        uint state = seed;
        for (int i = 0; i < length; i++) y[i] = (float)(amplitude * Uniform(ref state));
        return y;
    }

    /// <summary>A melody of decaying harmonic notes (an up-and-down C major scale, a quarter second each) over a bass note
    /// per bar and a noise "hi-hat" every half second.</summary>
    public static float[] Music(int length)
    {
        double[] scale = { 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66 };
        double[] bass = { 65.41, 87.31, 98.00, 65.41 };
        var y = new double[length];
        int note = Rate / 4;
        var phases = new double[6];
        double env = 0;
        for (int i = 0; i < length; i++)
        {
            if (i % note == 0)
            {
                env = 1;
                Array.Clear(phases);
            }
            double f = scale[i / note % scale.Length], v = 0;
            for (int h = 0; h < phases.Length; h++)
            {
                v += Sin(phases[h]) / (h + 1);
                phases[h] += 2 * Math.PI * f * (h + 1) / Rate;
                if (phases[h] > Math.PI) phases[h] -= 2 * Math.PI;
            }
            y[i] = 0.25 * env * v;
            env *= 0.99985;
        }
        double bassPhase = 0;
        for (int i = 0; i < length; i++)
        {
            double f = bass[i / Rate % bass.Length];
            y[i] += 0.2 * Sin(bassPhase);
            bassPhase += 2 * Math.PI * f / Rate;
            if (bassPhase > Math.PI) bassPhase -= 2 * Math.PI;
        }
        uint state = 12345;
        double hat = 0;
        for (int i = 0; i < length; i++)
        {
            if (i % (Rate / 2) == 0) hat = 1;
            y[i] += 0.15 * hat * Uniform(ref state);
            hat *= 0.997;
        }
        var result = new float[length];
        for (int i = 0; i < length; i++) result[i] = (float)(0.8 * y[i]);
        return result;
    }

    /// <summary>A uniform value in [-1, 1) from a 32-bit xorshift.</summary>
    private static double Uniform(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state / 2147483648.0 - 1.0;
    }

    /// <summary>sin(x) for x in [-pi, pi] by IEEE arithmetic alone (a Taylor series to x^23, below 1e-15 there), so every
    /// platform computes the same bits.</summary>
    private static double Sin(double x)
    {
        if (x > Math.PI) x -= 2 * Math.PI;
        if (x < -Math.PI) x += 2 * Math.PI;
        // Fold to [-pi/2, pi/2]: sin(pi - x) = sin(x).
        if (x > Math.PI / 2) x = Math.PI - x;
        else if (x < -Math.PI / 2) x = -Math.PI - x;
        double x2 = x * x, term = x, sum = x;
        for (int k = 1; k <= 11; k++)
        {
            term *= -x2 / ((2 * k) * (2 * k + 1));
            sum += term;
        }
        return sum;
    }

    /// <summary>SHA-256 of a clip's F32 little-endian bytes (fixtures record it so a changed generator is caught).</summary>
    public static string Hash(float[] samples)
    {
        var bytes = new byte[samples.Length * 4];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant();
    }

    /// <summary>The F32 little-endian bytes of a clip (llama.cpp reference input).</summary>
    public static void WriteF32(string path, float[] samples)
    {
        var bytes = new byte[samples.Length * 4];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (string n in Names) yield return new object[] { n };
    }
}
