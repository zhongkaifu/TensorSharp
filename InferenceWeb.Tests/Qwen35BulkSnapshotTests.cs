// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Paged;

namespace InferenceWeb.Tests;

public sealed class Qwen35BulkSnapshotTests
{
    private const string Pattern = "Qwen3.5-0.8B-Q8_0";

    [CudaFact("TS_TEST_MODEL_DIR", Pattern, GgmlBackend = BackendType.GgmlCuda)]
    public void BulkRestoreMatchesEverySnapshotByteAndEveryDecodeLogitOfSerialRestore()
        => CheckRestore(Pattern);

    [CudaFact("TS_TEST_MODEL_DIR", "Qwen3.8-27B-UD-IQ4_XS", GgmlBackend = BackendType.GgmlCuda)]
    public void OmittedDraft27BMatchesEverySnapshotByteAndEveryDecodeLogit()
        => CheckRestore("Qwen3.8-27B-UD-IQ4_XS");

    private static void CheckRestore(string pattern)
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), pattern);
        using var model = (Qwen35Model)ModelBase.Create(path, BackendType.GgmlCuda, 1, null, null, 1, null,
            new ModelMemoryPolicy(128, 16) { OmitEmbeddedDraftWeights = true });
        Assert.True(model.SupportsCrossSequenceKvReuse);
        int[] tokens = model.Tokenizer.Encode(string.Concat(Enumerable.Repeat(
            "The archive lists rivers, forests and mountains in alphabetical order. ", 8)), addSpecial: false).Take(52).ToArray();
        Assert.Equal(52, tokens.Length);
        const int block = 16, prefix = 48;
        using var storage = new PagedKvStorage(3, model.ComputeKVBlockByteSize(block));
        for (int i = 0; i < 3; i++)
        {
            model.Forward(tokens[(i * block)..((i + 1) * block)]);
            Assert.True(model.TryExtractKVBlock(i * block, block, storage.GetSpan(i)));
        }
        byte[] Snapshot()
        {
            var bytes = new byte[model.ComputeKVBlockByteSize(prefix)];
            Assert.True(model.TryExtractKVBlock(0, prefix, bytes));
            return bytes;
        }
        model.ResetKVCache();
        for (int i = 0; i < 3; i++)
            Assert.True(model.TryInjectKVBlock(i * block, block, storage.GetReadOnlySpan(i)));
        byte[] expected = Snapshot();
        var logits = new List<float[]>();
        foreach (int token in tokens[prefix..]) logits.Add((float[])model.Forward([token]).Clone());

        model.ResetKVCache();
        Assert.Equal(prefix, model.RestoreKvSnapshots(block, prefix, b => storage.Acquire(b)));
        Assert.Equal(expected, Snapshot());
        for (int i = 0; i < logits.Count; i++)
        {
            float[] actual = model.Forward([tokens[prefix + i]]);
            Assert.All(actual, v => Assert.True(float.IsFinite(v)));
            Assert.Equal(logits[i], actual);
        }
    }
}
