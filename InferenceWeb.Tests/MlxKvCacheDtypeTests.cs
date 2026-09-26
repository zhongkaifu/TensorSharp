// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

[CollectionDefinition("MLX KV cache dtype", DisableParallelization = true)]
public sealed class MlxKvCacheDtypeCollection { }

/// <summary>
/// A block-quantized K/V cache request on the MLX backend. config/agent-qwen3.8-27b.json
/// asks for q8_0 (right for ggml_metal); run with --backend mlx it allocated Q8_0 cache
/// tensors that MLX storage cannot hold, and the first chat request died at the first
/// cache write with "MLX dtype mapping does not support Q8_0".
/// </summary>
[Collection("MLX KV cache dtype")]
public sealed class MlxKvCacheDtypeTests : IDisposable
{
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"mlx-kv-probe-{Guid.NewGuid():N}.gguf");

    public MlxKvCacheDtypeTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [MlxTheory]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.Q4_0)]
    public void BlockQuantizedRequestBecomesF16AndSaysSo(KvCacheDtype requested)
    {
        KvCacheDtypeConfig.Set(requested);
        var stderr = new StringWriter();
        TextWriter original = Console.Error;
        Console.SetError(stderr);
        ProbeModel model;
        try { model = new ProbeModel(_path, BackendType.Mlx); }
        finally { Console.SetError(original); }

        using (model)
        {
            Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype);
            Assert.Contains($"the MLX backend has no {requested.ToShortString()} K/V cache; using f16", stderr.ToString());
        }
    }

    [MlxFact]
    public void FloatRequestsAreLeftAlone()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        using var model = new ProbeModel(_path, BackendType.Mlx);
        Assert.Equal(KvCacheDtype.F32, model.KvCacheDtype);
    }

    [Fact]
    public void OtherBackendsKeepTheBlockQuantizedCache()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
        using var model = new ProbeModel(_path, BackendType.Cpu);
        Assert.Equal(KvCacheDtype.Q8_0, model.KvCacheDtype);
    }

    private sealed class ProbeModel : ModelBase
    {
        public ProbeModel(string path, BackendType backend) : base(path, backend)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = 4 };
        }

        protected override float[] ForwardCore(int[] tokens) => new float[4];

        protected override void ResetKVCacheCore()
        {
        }
    }
}
