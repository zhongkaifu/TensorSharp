// Regression coverage for the Gemma 4 "unified" (encoder-free) audio mmproj.
//
// Bug: loading the gemma-4-12b mmproj (projector_type "gemma4ua") threw a
// System.DivideByZeroException in the Gemma4AudioEncoder constructor:
//
//     _headDim = _hiddenSize / _numHeads;   // _numHeads == 0
//
// Encoder-free "gemma4ua" mmproj files have no conformer, so the GGUF writes
// clip.audio.attention.head_count / block_count / feed_forward_length as 0.
// These models go through EncodeRawWaveform and never touch the conformer
// attention path, so _headDim is unused — the constructor must not divide by it.
//
// The tests load only the 175 MB mmproj (not the 12B language model) with a CPU
// allocator, so they are fast. They skip when the mmproj is not present.

using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4AudioEncoderTests
{
    private readonly ITestOutputHelper _output;
    public Gemma4AudioEncoderTests(ITestOutputHelper output) { _output = output; }

    private static string? FindMmproj()
    {
        string fileName = "gemma-4-12b-mmproj-BF16.gguf";
        string? modelEnv = Environment.GetEnvironmentVariable("TS_GEMMA4_12B");
        string[] candidates =
        {
            Environment.GetEnvironmentVariable("TS_GEMMA4_12B_MMPROJ") ?? "",
            string.IsNullOrEmpty(modelEnv) ? "" : Path.Combine(Path.GetDirectoryName(modelEnv)!, fileName),
            Path.Combine("C:", "Works", "models", fileName),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "work", "model", fileName),
        };
        return candidates.FirstOrDefault(p => !string.IsNullOrEmpty(p) && File.Exists(p));
    }

    [Fact]
    public void EncoderFreeMmproj_LoadsWithoutDivideByZero()
    {
        string? mmproj = FindMmproj();
        if (mmproj == null) { _output.WriteLine("gemma-4-12b mmproj not found; skipping"); return; }

        var allocator = new CpuAllocator(BlasEnum.DotNet);

        // The reported crash happened right here, inside the constructor.
        using var encoder = new Gemma4AudioEncoder(mmproj, allocator);

        Assert.True(encoder.IsEncoderFree, "gemma-4-12b mmproj should be the encoder-free (gemma4ua) projector");
        Assert.True(encoder.ProjectionDim > 0);
        _output.WriteLine($"Loaded encoder-free audio projector: projectionDim={encoder.ProjectionDim}");
    }

    [Fact]
    public void EncoderFreeMmproj_EncodeRawWaveform_ProducesFiniteEmbeddings()
    {
        string? mmproj = FindMmproj();
        if (mmproj == null) { _output.WriteLine("gemma-4-12b mmproj not found; skipping"); return; }

        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var encoder = new Gemma4AudioEncoder(mmproj, allocator);
        Assert.True(encoder.IsEncoderFree);

        // 0.25 s of 16 kHz mono audio (a quiet sine), enough to span several
        // 640-sample frames and exercise the trailing zero-padded frame.
        int sampleCount = 4000;
        var samples = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
            samples[i] = 0.1f * MathF.Sin(2f * MathF.PI * 220f * i / 16000f);

        const int frameSize = 640; // 40 ms/token at 16 kHz
        int expectedTokens = (sampleCount + frameSize - 1) / frameSize;

        using Tensor emb = encoder.EncodeRawWaveform(samples);

        Assert.Equal(2, emb.DimensionCount);
        Assert.Equal(expectedTokens, (int)emb.Sizes[0]);
        Assert.Equal(encoder.ProjectionDim, (int)emb.Sizes[1]);

        int count = (int)emb.Sizes[0] * (int)emb.Sizes[1];
        float[] values = emb.GetElementsAsFloat(count);
        Assert.All(values, v => Assert.True(float.IsFinite(v), "embedding contains NaN/Inf"));
        // RMS-normalized + projected embeddings should not be all-zero.
        Assert.Contains(values, v => v != 0f);

        _output.WriteLine($"EncodeRawWaveform -> [{emb.Sizes[0]}, {emb.Sizes[1]}], all finite, non-zero");
    }
}
