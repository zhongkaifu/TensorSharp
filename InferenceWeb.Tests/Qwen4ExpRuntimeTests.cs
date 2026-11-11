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
        // On arm64 one prompt runs the same products in the same order: the same bits within 1.5e-7. A chunked
        // prompt's managed products pick their kernels by row count, so either model moves by ~5e-3 against its
        // own unchunked run (Qwen4ExpModel: 4.7e-3 measured), and the two by as much against each other; on x64
        // the two models' products differ by as much in one prompt too (6.4e-3 measured on Linux).
        AssertClose(expected, actual, chunkedPrompt || !System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.Equals(System.Runtime.InteropServices.Architecture.Arm64) ? 1e-2 : 1e-5, "managed cpu");
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

    // ---- speculation: the PLE histories follow a verify's accept count; the MTP head chains the stream

    private (string Target, string Mtp) SpecFixtures(bool qsa)
        => (Fixture(qsa), Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, $"q4e-mtp-{qsa}.gguf"), indexerTopK: 64));

    /// <summary>n-gram drafts (accepted and rejected): the verify keeps each row's recurrent and PLE state, and the
    /// accept count commits the kept prefix's n-gram and conv histories.</summary>
    [Fact]
    public void SpeculationMatchesPlainGreedyOnTheManagedCpu()
    {
        var (target, mtp) = SpecFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu), "qwen4exp cpu");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu, mtp), "qwen4exp cpu mtp");
    }

    [MlxFact]
    public void SpeculationMatchesPlainGreedyOnMlx()
    {
        var (target, mtp) = SpecFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Mlx), "qwen4exp mlx");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Mlx, mtp), "qwen4exp mlx mtp");
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(1e-2)]
    public void SpeculationMatchesPlainGreedyOnGgmlCpu(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlCpu, tieTolerance);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(0)]
    public void SpeculationMatchesPlainGreedyOnGgmlMetal(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlMetal, tieTolerance);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(1e-2)]
    public void SpeculationMatchesPlainGreedyOnGgmlCuda(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlCuda, tieTolerance);

    /// <summary>ggml_cpu and ggml_cuda round a verify window's activations unlike one row's: the streams may part
    /// only at near-ties, to the runner-up.</summary>
    private void SpeculationOnGgml(BackendType backend, double tieTolerance)
    {
        var (target, mtp) = SpecFixtures(qsa: true);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, backend), $"qwen4exp {backend}", tieTolerance: tieTolerance);
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, backend, mtp), $"qwen4exp {backend} mtp",
            tieTolerance: tieTolerance);
    }

    /// <summary>The runtime's draft layer computes what Qwen4ExpModel's MTP head does (which runs on GGML only):
    /// prompt catch-up and two chained draft steps.</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void MtpDraftsMatchQwen4ExpModelOnMetal(BackendType backend) => MtpDraftsMatch(backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void MtpDraftsMatchQwen4ExpModelOnGgmlCpu(BackendType backend) => MtpDraftsMatch(backend);

    private void MtpDraftsMatch(BackendType backend)
    {
        var (target, mtp) = SpecFixtures(qsa: true);
        List<float[]> legacy, actual;
        using (var family = new Qwen4ExpModel(target, backend))
        {
            family.LoadMtpDraftWeights(mtp);
            legacy = DraftTrace(family);
        }
        using (var runtime = new Qwen4ExpRuntimeModel(target, backend, mtp))
            actual = DraftTrace(runtime);
        double[] rel = Errors(legacy.ToArray(), actual.ToArray());
        Assert.True(rel.All(r => r <= 3e-2), $"{backend}: draft rows relL2 from Qwen4ExpModel: {string.Join(", ", rel.Select(r => r.ToString("E2")))}");
    }

    /// <summary>Logits of the trunk and of the head's prompt catch-up and two chained drafts.</summary>
    private static List<float[]> DraftTrace(ModelBase model)
    {
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
        Assert.Equal(TensorSharp.Runtime.Speculative.DraftHeadKind.PerToken, head.DraftHeadKind);
        int width = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int n = prompt.Length;
        var rows = new List<float[]>();
        var hAll = new float[n * width];
        var logits = new float[vocab];
        model.ResetKVCache();
        spec.SpecForward(prompt, hAll, logits, false);
        rows.Add((float[])logits.Clone());
        // As the speculative executor commits a prompt: from position 0, row k the hidden state of the token before
        // tokens[k] (zeros before the first).
        head.DraftCatchUp(prompt, new float[width].Concat(hAll[..((n - 1) * width)]).ToArray(), 0);
        var draft = new float[vocab];
        var h1 = new float[width];
        head.DraftStep(ArgMax(logits), hAll[((n - 1) * width)..], n, draft, h1);
        rows.Add((float[])draft.Clone());
        var h2 = new float[width];
        head.DraftStep(ArgMax(draft), h1, n + 1, draft, h2);
        rows.Add((float[])draft.Clone());
        return rows;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double[] Errors(float[][] expected, float[][] actual)
    {
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
        return errors;
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
