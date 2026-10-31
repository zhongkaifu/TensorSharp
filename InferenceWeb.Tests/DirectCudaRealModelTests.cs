// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Weights-gated direct-CUDA (BackendType.Cuda) checks on real checkpoints. The synthetic files of
// DirectCudaPrefillModelTests keep every routing-relevant shape small; Stage 0 of the direct-CUDA
// prefill plan (and the "toy dimensions hid real bugs" lesson behind PLAN.md's G4) asks for at
// least one run on real weights per family, where the hidden sizes, vocabularies and K tails are
// the production ones. Each test needs TS_TEST_MODEL_DIR (a directory or a file) holding a
// matching GGUF and skips visibly otherwise. The reference is ggml_cpu, the backend every test lane
// pins; the parity GATE for the prefill work is still the harness's G3a/G3b against ggml_cuda on
// the A40s (eng/validation/README-direct-cuda-prefill.md): these tests catch gross faults (a stale
// state, a wrong route) on a developer's GPU, not 1.10x drifts.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.DirectCudaModelChecks;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class DirectCudaRealModelTests : IDisposable
{
    private const string ModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public DirectCudaRealModelTests(ITestOutputHelper output)
    {
        _output = output;
        _environment.Set("MAX_CONTEXT", "1024");
        _environment.Set("TS_SCHED_DISABLE_BATCHED", null);
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
    }

    // Real English prose (written for this test), long enough for every prompt length used here
    // with any of the tokenizers; cut to an exact token count after encoding.
    private const string Passage =
        "The night ferry left the harbour twenty minutes late, which surprised nobody who had taken it before. " +
        "A mechanic had been called to look at the port engine, and he walked the length of the deck twice, " +
        "listening, before he waved the captain on. The passengers were mostly students going home for the " +
        "holidays, a few lorry drivers who slept in their cabs below, and an elderly couple who sat in the " +
        "forward lounge with a thermos of tea and a map they never unfolded. Outside, the lights of the town " +
        "slid backwards and thinned out until only the lighthouse remained, turning slowly, as if it were " +
        "keeping count of something. The sea was calm at first. Then, past the headland, the swell came in " +
        "long low ridges from the west, and the ferry began to roll, gently at first and then with a deeper, " +
        "slower motion that sent cups sliding along the counter of the cafeteria. A child laughed every time " +
        "the floor tilted. Her father did not. In the engine room the mechanic stood with one hand on the " +
        "rail and the other on the casing of the port engine, feeling for the vibration he had heard on deck. " +
        "It was there, faint and irregular, a stutter in the rhythm that came and went with the roll of the " +
        "ship. He wrote the time in his notebook, then the engine speed, then the oil pressure, and underlined " +
        "the last figure twice. When the chief engineer came down he showed him the page without a word. " +
        "They agreed to reduce speed by a tenth and to watch the pressure every quarter of an hour until they " +
        "reached the other side. The crossing would take longer, but nobody on board would notice, and the " +
        "report could be written in the morning, when the engine had cooled and could be opened properly.";

    private static string Find(string pattern)
    {
        string value = Environment.GetEnvironmentVariable(ModelDir);
        return TestGates.FindGguf(value, pattern) ?? throw new InvalidOperationException($"no '*{pattern}*' GGUF in {value}");
    }

    private static int[] Tokens(ModelBase model, int count)
    {
        var ids = model.Tokenizer.Encode(Passage, addSpecial: false);
        if (ids.Count < count)
            throw new InvalidOperationException($"the passage encodes to {ids.Count} tokens, fewer than {count}");
        return ids.Take(count).ToArray();
    }

    private static int[] Quarters(int length)
    {
        int q = Math.Max(1, length / 4);
        return new[] { q, q, q };
    }

    /// <summary>Row metrics of <paramref name="x"/> against <paramref name="reference"/>.</summary>
    private sealed record Rows(double[] RelL2, double[] Kl, int Top1Mismatches)
    {
        public double Step0Rel => RelL2[0];
        public double Step0Kl => Kl[0];
        public double DecodeMedianRel => Quantile(RelL2.Skip(1), 0.5);
        public double DecodeMedianKl => Quantile(Kl.Skip(1), 0.5);
        public override string ToString() =>
            $"step0 relL2 {Step0Rel:E2} KL {Step0Kl:E2}; decode median relL2 {DecodeMedianRel:E2} KL {DecodeMedianKl:E2}; top-1 mismatches {Top1Mismatches}/{RelL2.Length}";
    }

    private static Rows Compare(List<float[]> x, List<float[]> reference)
        => new(x.Select((r, i) => RelL2(r, reference[i])).ToArray(),
               x.Select((r, i) => Kl(reference[i], r)).ToArray(),
               x.Select((r, i) => ArgMax(r) != ArgMax(reference[i])).Count(m => m));

    /// <summary>KL(p_reference || p_x) of the softmaxes, in double.</summary>
    private static double Kl(float[] reference, float[] x)
    {
        double mr = reference.Max(), mx = x.Max(), zr = 0, zx = 0;
        for (int i = 0; i < reference.Length; i++) { zr += Math.Exp(reference[i] - mr); zx += Math.Exp(x[i] - mx); }
        double lr = mr + Math.Log(zr), lx = mx + Math.Log(zx), kl = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double lp = reference[i] - lr;
            kl += Math.Exp(lp) * (lp - (x[i] - lx));
        }
        return Math.Max(0, kl);
    }

    /// <summary>
    /// The teacher (ggml_cpu, whole-prompt prefill, greedy) fixes the forced tokens. The floor is
    /// Stage 0's ggml-vs-ggml distance: the same teacher-forced rows on ggml_cpu with the prompt in
    /// four pieces (L/4 chunks), a configuration that should agree. Direct CUDA must stay within
    /// max(<paramref name="ratio"/> x its calibrated distance, <paramref name="ratio"/> x the floor)
    /// on every metric, and within the floor's top-1 mismatches + 1.
    /// </summary>
    private void TracksGgmlCpu(string pattern, int promptLength, int steps, Calibration calibrated, double ratio = 1.5)
    {
        string path = Find(pattern);
        _output.WriteLine($"model {Path.GetFileName(path)}, prompt {promptLength} tokens, {steps} teacher-forced steps");
        int[] prompt;
        List<float[]> teacher, chunked;
        int[] forced;
        using (var ggml = ModelBase.Create(path, BackendType.GgmlCpu))
        {
            prompt = Tokens(ggml, promptLength);
            ggml.ResetKVCache();
            var rows = new List<float[]> { (float[])ggml.ForwardPrompt(prompt).Clone() };
            forced = new int[steps];
            for (int s = 0; s < steps; s++)
            {
                forced[s] = ArgMax(rows[^1]);
                rows.Add((float[])ggml.Forward(new[] { forced[s] }).Clone());
            }
            teacher = rows;
            chunked = Trajectory(ggml, prompt, forced, Quarters(promptLength));
        }
        List<float[]> cuda;
        using (var model = ModelBase.Create(path, BackendType.Cuda))
            cuda = Trajectory(model, prompt, forced);
        Assert.True(cuda.All(r => r.All(float.IsFinite)), "direct CUDA produced a non-finite logit");
        Rows floor = Compare(chunked, teacher), actual = Compare(cuda, teacher);
        _output.WriteLine($"floor (ggml_cpu L/4 chunks vs whole): {floor}");
        _output.WriteLine($"direct CUDA vs ggml_cpu:               {actual}");
        _output.WriteLine($"calibrated (laptop RTX 3080):          {calibrated}");
        void Check(string what, double value, double cal, double fl)
        {
            double limit = Math.Max(ratio * cal, ratio * fl);
            Assert.True(value <= limit, $"{what}: {value:E3} > max({ratio} x calibrated {cal:E2}, {ratio} x floor {fl:E2}) = {limit:E3}");
        }
        Check("step-0 relative L2", actual.Step0Rel, calibrated.Step0Rel, floor.Step0Rel);
        Check("step-0 KL", actual.Step0Kl, calibrated.Step0Kl, floor.Step0Kl);
        Check("decode median relative L2", actual.DecodeMedianRel, calibrated.DecodeMedianRel, floor.DecodeMedianRel);
        Check("decode median KL", actual.DecodeMedianKl, calibrated.DecodeMedianKl, floor.DecodeMedianKl);
        Assert.True(actual.Top1Mismatches <= Math.Max(floor.Top1Mismatches, calibrated.Top1) + 1,
            $"{actual.Top1Mismatches} top-1 mismatches (floor {floor.Top1Mismatches}, calibrated {calibrated.Top1})");
    }

    private sealed record Calibration(double Step0Rel, double Step0Kl, double DecodeMedianRel, double DecodeMedianKl, int Top1)
    {
        public override string ToString() =>
            $"step0 relL2 {Step0Rel:E2} KL {Step0Kl:E2}; decode median relL2 {DecodeMedianRel:E2} KL {DecodeMedianKl:E2}; top-1 mismatches {Top1}";
    }

    /// <summary>
    /// The same prompt and forced steps three times on one loaded model (the first run plans the
    /// Qwen3.5 prefill graph, the second captures it, the third replays it). On real weights the
    /// FIRST run's decode rows legitimately differ a little from later runs: its first decode step
    /// runs per-op and later ones replay the captured decode graph, and the decode graph is keyed
    /// by the KV capacity the process has grown to (Qwen3.5-0.8B: 1-3e-2 relative L2, measured;
    /// Stage 0 VM2's "first-decode effect" outside D5/D3). So: run 3 bit-identical to run 2, and
    /// run 2 within <see cref="FirstRunBand"/> of run 1 (the prefill-graph ring-phase defect D5,
    /// fixed, moved rows 0.13-0.75).
    /// </summary>
    private const double FirstRunBand = 0.06;

    private void RepeatedPrefill(string pattern, int promptLength)
    {
        string path = Find(pattern);
        using var model = ModelBase.Create(path, BackendType.Cuda);
        int[] prompt = Tokens(model, promptLength), forced = Forced(6, 7).Select(t => t % model.Config.VocabSize).ToArray();
        var runs = Enumerable.Range(0, 3).Select(_ => Trajectory(model, prompt, forced)).ToList();
        for (int r = 1; r < runs.Count; r++)
        {
            var rel = runs[r].Select((row, i) => RelL2(row, runs[0][i])).ToList();
            int same = runs[r].Select((row, i) => BitIdentical(row, runs[0][i])).Count(x => x);
            _output.WriteLine($"{Path.GetFileName(path)} {promptLength} tokens: run {r + 1} vs run 1: bit-identical {same}/{rel.Count}, relative L2 {Fmt(rel)}");
        }
        for (int i = 0; i < runs[2].Count; i++)
            Assert.True(BitIdentical(runs[2][i], runs[1][i]),
                $"run 3 row {i} differs from run 2 (relative L2 {RelL2(runs[2][i], runs[1][i]):E3})");
        for (int i = 0; i < runs[1].Count; i++)
            Assert.True(RelL2(runs[1][i], runs[0][i]) <= FirstRunBand,
                $"run 2 row {i} is {RelL2(runs[1][i], runs[0][i]):E3} from run 1 (allowed {FirstRunBand}: the first-decode effect)");
        Assert.True(BitIdentical(runs[1][0], runs[0][0]), "the prefill rows of runs 1 and 2 differ");
    }

    /// <summary>
    /// A 48-token prefix snapshot (K/V + recurrent state) extracted, the cache reset, the block
    /// injected: the 16-token suffix and 4 forced steps after it must be bit-identical to the
    /// straight run. A warm-up trajectory first, so both runs replay the same decode graph (the
    /// first-decode effect). D1 before its fix, on these files: Qwen3.5-0.8B 0.10-0.16 relative L2,
    /// gemma-4-E4B 0.07-0.22 and gpt-oss-20b 0.14-0.28 with argmax flips (the injected rows never
    /// reached the device, so attention ran over a zero prefix).
    /// </summary>
    private void SnapshotRestoreIsExact(string pattern)
    {
        string path = Find(pattern);
        using var model = ModelBase.Create(path, BackendType.Cuda);
        int[] ids = Tokens(model, 68);
        int[] prefix = ids[..48], suffix = ids[48..64], forced = ids[64..68];
        Trajectory(model, ids[..64], forced);
        SnapshotResult result = SnapshotRestore(model, prefix, suffix, forced);
        Assert.True(result != null, $"{Path.GetFileName(path)} refused to snapshot a 48-token prefix");
        var rel = result.Restored.Select((row, i) => RelL2(row, result.Straight[i])).ToList();
        int argmaxSame = result.Restored.Select((row, i) => ArgMax(row) == ArgMax(result.Straight[i])).Count(x => x);
        _output.WriteLine($"{Path.GetFileName(path)} ({result.BlockBytes} B block): restored vs straight relative L2 {Fmt(rel)}; same argmax {argmaxSame}/{rel.Count}");
        for (int i = 0; i < result.Restored.Length; i++)
            Assert.True(BitIdentical(result.Restored[i], result.Straight[i]),
                $"row {i} after the restore is {RelL2(result.Restored[i], result.Straight[i]):E3} from the straight run");
    }

    [CudaFact(ModelDir, "gpt-oss-20b")]
    public void GptOss_20B_SnapshotRestore_IsExact() => SnapshotRestoreIsExact("gpt-oss-20b");

    // ---- Qwen3.5-0.8B (qwen35 dense, the qwen08 gate model's architecture) ----

    [CudaFact(ModelDir, "qwen3.5-0.8b", GgmlBackend = BackendType.GgmlCpu)]
    public void Qwen35_08B_PrefillAndDecodeRows_TrackGgmlCpu()
        => TracksGgmlCpu("qwen3.5-0.8b", 130, 8, Calibrated["qwen3.5-0.8b"]);

    [CudaTheory(ModelDir, "qwen3.5-0.8b")]
    [InlineData(96)]
    [InlineData(97)]
    public void Qwen35_08B_RepeatedPrefill_Reproduces(int promptLength) => RepeatedPrefill("qwen3.5-0.8b", promptLength);

    [CudaFact(ModelDir, "qwen3.5-0.8b")]
    public void Qwen35_08B_SnapshotRestore_IsExact() => SnapshotRestoreIsExact("qwen3.5-0.8b");

    // ---- Qwen3.5-9B (qwen35 dense at production width) ----

    [CudaFact(ModelDir, "qwen3.5-9b", GgmlBackend = BackendType.GgmlCpu)]
    public void Qwen35_9B_PrefillAndDecodeRows_TrackGgmlCpu()
        => TracksGgmlCpu("qwen3.5-9b", 130, 8, Calibrated["qwen3.5-9b"]);

    [CudaTheory(ModelDir, "qwen3.5-9b")]
    [InlineData(96)]
    [InlineData(97)]
    public void Qwen35_9B_RepeatedPrefill_Reproduces(int promptLength) => RepeatedPrefill("qwen3.5-9b", promptLength);

    // ---- gemma-4-E4B (gemma4 dense: sliding-window + global layers, per-layer embeddings) ----

    [CudaFact(ModelDir, "gemma-4-e4b", GgmlBackend = BackendType.GgmlCpu)]
    public void Gemma4_E4B_PrefillAndDecodeRows_TrackGgmlCpu()
        => TracksGgmlCpu("gemma-4-e4b", 130, 8, Calibrated["gemma-4-e4b"]);

    [CudaTheory(ModelDir, "gemma-4-e4b")]
    [InlineData(96)]
    [InlineData(97)]
    public void Gemma4_E4B_RepeatedPrefill_Reproduces(int promptLength) => RepeatedPrefill("gemma-4-e4b", promptLength);

    [CudaFact(ModelDir, "gemma-4-e4b")]
    public void Gemma4_E4B_SnapshotRestore_IsExact() => SnapshotRestoreIsExact("gemma-4-e4b");

    /// <summary>Direct CUDA vs ggml_cpu on the laptop RTX 3080 (sm_86, Windows), base revision,
    /// prompt 130 tokens of <see cref="Passage"/>, 8 teacher-forced steps. The tests allow 1.5x
    /// this or 1.5x the live ggml floor, whichever is larger.</summary>
    private static readonly Dictionary<string, Calibration> Calibrated = new()
    {
        // floor (ggml_cpu L/4 vs whole) on the same rows: 1.52e-2 / 9.65e-4 / 1.29e-2 / 8.49e-4 / 0
        ["qwen3.5-0.8b"] = new(1.27e-2, 1.14e-3, 1.30e-2, 1.01e-3, 0),
        // floor: 1.27e-2 / 6.70e-4 / 1.53e-2 / 3.96e-4 / 0
        ["qwen3.5-9b"] = new(1.47e-2, 1.42e-3, 1.59e-2, 3.62e-4, 0),
        // floor: 2.90e-2 / 7.78e-4 / 8.95e-2 / 2.66e-2 / 1
        ["gemma-4-e4b"] = new(3.27e-2, 5.43e-4, 8.62e-2, 3.27e-2, 1),
    };
}
