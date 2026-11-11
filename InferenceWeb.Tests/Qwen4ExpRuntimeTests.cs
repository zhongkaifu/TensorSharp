// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next (qwen4exp) on the shared decoder runtime against Qwen4ExpModel on the synthetic checkpoint:
// hyper-connections, the sigmoid-gated recurrence, gated attention, the PLE layer (its n-gram and conv
// histories across steps) and the shared-expert MoE.

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpRuntimeTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("q4e-runtime").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // An EOS (255) inside the prompt cuts the n-gram context after it.
    private static readonly int[] Prompt = { 1, 17, 33, 2, 99, 5, 255, 45, 12, 200, 7, 63, 88, 31 };
    private static readonly int[] Decode = { 41, 9, 130, 77 };

    // A top-k of 64 blocks keeps the QSA layers dense (exact) for these lengths. Qwen4ExpModel's per-layer path
    // (the managed CPU) has no QSA layers at all.
    private string Fixture(bool qsa = true) => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"q4e-{qsa}.gguf"), indexerTopK: 64, qsa: qsa);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnTheManagedCpu(bool chunkedPrompt)
    {
        string path = Fixture(qsa: false);
        float[][] expected = Run(() => new Qwen4ExpModel(path, BackendType.Cpu), chunkedPrompt);
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), chunkedPrompt);
        // One prompt runs the same products in the same order: the same bits within rounding. A chunked prompt's
        // managed products pick their kernels by row count, so either model moves by ~5e-3 against its own
        // unchunked run (Qwen4ExpModel: 4.7e-3 measured), and the two by as much against each other.
        AssertClose(expected, actual, chunkedPrompt ? 1e-2 : 1e-5, "managed cpu");
    }

    /// <summary>MLX (F16 attention over F16 caches, MLX's own kernels) against Qwen4ExpModel's managed run.</summary>
    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeOnMlxMatchesQwen4ExpModel(bool chunkedPrompt)
    {
        string path = Fixture(qsa: false);
        float[][] expected = Run(() => new Qwen4ExpModel(path, BackendType.Cpu), chunkedPrompt);
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Mlx), chunkedPrompt);
        AssertClose(expected, actual, 2e-2, "mlx");
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlCpu(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlCpu, chunkedPrompt);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlMetal(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlMetal, chunkedPrompt);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlCuda(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlCuda, chunkedPrompt);

    /// <summary>Qwen4ExpModel's token-span graphs (QSA layers dense below the budget) against the runtime.</summary>
    private void MatchesOnGgml(BackendType backend, bool chunkedPrompt)
    {
        string path = Fixture();
        float[][] expected = Run(() => new Qwen4ExpModel(path, backend), chunkedPrompt);
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, backend), chunkedPrompt);
        AssertClose(expected, actual, 2e-2, backend.ToString());
    }

    [Fact]
    public void RuntimeRefusesSpeculation()
    {
        using var model = new Qwen4ExpRuntimeModel(Fixture(), BackendType.Cpu);
        Assert.Contains("does not speculate", model.SpeculationRefusal);
    }

    private static float[][] Run(Func<ModelBase> create, bool chunkedPrompt)
    {
        using ModelBase model = create();
        model.ResetKVCache();
        var rows = new List<float[]>();
        if (chunkedPrompt)
        {
            model.ForwardRefill(Prompt[..6]);
            rows.Add((float[])model.Forward(Prompt[6..]).Clone());
        }
        else
        {
            rows.Add((float[])model.ForwardRefill(Prompt).Clone());
        }
        foreach (int token in Decode)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows.ToArray();
    }

    private static void AssertClose(float[][] expected, float[][] actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        var errors = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++)
        {
            double num = 0, den = 0;
            for (int j = 0; j < expected[i].Length; j++)
            {
                double d = expected[i][j] - actual[i][j];
                num += d * d;
                den += (double)expected[i][j] * expected[i][j];
            }
            errors[i] = Math.Sqrt(num / Math.Max(den, 1e-30));
        }
        Assert.True(errors.All(e => e < tolerance), $"{label} relL2 per row: {string.Join(", ", errors.Select(e => e.ToString("E2")))}");
    }
}
