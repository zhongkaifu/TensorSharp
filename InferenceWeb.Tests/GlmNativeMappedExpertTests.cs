// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Mapped CPU experts, including loader prefault, must preserve model logits.</summary>
public sealed class GlmNativeMappedExpertTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm-mapped-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [GlmNativeCudaFact(1)]
    public void MappedExperts_MatchGpuExperts_SingleGpu() => Check(deviceCount: 1, tp: 1);

    [GlmNativeCudaFact(1)]
    public void UnicodeCheckpointPath_MappedExpertsMatchGpuExperts() => Check(deviceCount: 1, tp: 1, unicodePath: true);

    [GlmNativeCudaFact(2)]
    public void MappedExperts_MatchGpuExperts_LayerSplit() => Check(deviceCount: 2, tp: 1);

    [GlmNativeCudaFact(2)]
    public void MappedExperts_MatchGpuExperts_TensorParallel() => Check(deviceCount: 2, tp: 2);

    private void Check(int deviceCount, int tp, bool unicodePath = false)
    {
        Assert.True(TensorSharp.Cuda.CudaDevice.GetDeviceCount() >= deviceCount);
        using var environment = new NativeEnvScope();
        environment.Set("TS_GLM_MOE_MMAP", "1");
        string directory = unicodePath ? Path.Combine(_dir, "权重测试-🦙") : _dir;
        Directory.CreateDirectory(directory);
        string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(directory, unicodePath ? "微型模型-🌍.gguf" : "tiny-glm.gguf"));
        int[] prompt = Enumerable.Range(0, 37).Select(i => 3 + i * 11 % 90).ToArray();
        // Fixed continuations compare the same distributions even if close
        // candidates exchange order across the CPU and CUDA reductions.
        int[] continuation = [17, 31, 43, 59, 71, 83];
        List<float[]> Run(int cpuLayers)
        {
            IntPtr model = GgmlGlmNative.LoadModel(path, deviceCount, 256, 16, 2,
                nCpuMoe: cpuLayers, backendName: "CUDA", tp: tp, ctxIsHardLimit: true);
            Assert.NotEqual(IntPtr.Zero, model);
            try
            {
                var rows = new List<float[]>();
                foreach (int[] tokens in new[] { prompt }.Concat(continuation.Select(t => new[] { t })))
                {
                    var logits = new float[GgmlGlmNative.VocabSize(model)];
                    Assert.True(GgmlGlmNative.Forward(model, tokens, logits));
                    rows.Add(logits);
                }
                return rows;
            }
            finally { GgmlGlmNative.Free(model); }
        }

        List<float[]> gpu = Run(0);
        List<float[]> mapped = Run(3);
        double worst = 0;
        for (int row = 0; row < gpu.Count; row++)
        {
            Assert.All(mapped[row], value => Assert.True(float.IsFinite(value)));
            double maxAbs = gpu[row].Zip(mapped[row], (a, b) => Math.Abs((double)a - b)).Max();
            worst = Math.Max(worst, maxAbs);
            Assert.True(maxAbs < 1e-3, $"devices={deviceCount}, tp={tp}, row={row}: max absolute logit error {maxAbs:G9}");
        }
        output.WriteLine($"devices={deviceCount}, tp={tp}, 37 prefill + 6 decode tokens: max absolute GPU/mapped logit error {worst:G9}");
    }
}
