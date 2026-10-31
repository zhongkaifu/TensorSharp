// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Gemma-4 tensor-parallel per-op forward (Gemma4TransformerBlockTP / Gemma4AttentionTP) against
// the single-device path on the same tokens. --backend cuda --tp N always runs it; the GGML backends
// run it whenever their fused TP graphs are not used (the 26B MoE with TS_GEMMA4_TP_FUSED_MOE=0, a
// fused call that declines, or shards the fused graphs cannot take, as with this file's F32 global
// Q/K/V). Its global layers rotated Q and K with Ops.RoPEEx, which has no rope_freqs input, so every
// pair rotated at the plain base frequency where the dense Gemma-4 files' proportional RoPE leaves
// three quarters of the pairs unrotated (the batched forward had the same omission, see
// Gemma4DenseBatchedForwardParityTests). Both rows run on one GPU: direct CUDA through a test group
// of two ranks on device 0 (the per-op path does the same work wherever the ranks live; the
// AllReduce is staged through the host, as Cuda.TensorParallelGroup does under
// TENSORSHARP_TP_HOST_ALLREDUCE=1), ggml-cuda through GGML_CUDA_DEVICES=2.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Gemma4DenseTensorParallelParityTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "g4dense-tp-" + Guid.NewGuid().ToString("N"));

    public Gemma4DenseTensorParallelParityTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    [CudaFact]
    public void DirectCuda_TwoRanks_TrackTheSingleDevicePath() =>
        Check(BackendType.Cuda, path => new Gemma4Model(path, BackendType.Cuda, tpGroup: new OneGpuCudaGroup(2)));

    /// <summary>GGML_CUDA_DEVICES=2 emulates the two devices on one GPU.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData(2)]
    public void GgmlCuda_TrackTheSingleDevicePath(int ranks) =>
        Check(BackendType.GgmlCuda, path => Assert.IsType<Gemma4Model>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: ranks)));

    // Three prompt lengths inside the 208-token window, then six teacher-forced decode steps (the
    // first prompt's steps cross the window). Both paths run one sequence at a time.
    private static readonly int[][] Prompts = { Tokens(205, seed: 3), Tokens(120, seed: 5), Tokens(41, seed: 7) };
    private static readonly int[] Forced = { 40, 41, 42, 43, 44, 45 };

    // Measured (laptop RTX 3080): with the fix the direct-CUDA rows are within 1.1e-2 and the
    // ggml-cuda rows within 2.4e-2 of their single-device path; without it 4.3e-2..0.42 and
    // 4.2e-2..0.42. ggml-cuda's own band (as in Gemma4DenseBatchedForwardParityTests) sits too
    // close to 0.025 for that tolerance.
    private static double Tolerance(BackendType backend) => backend == BackendType.Cuda ? 0.025 : 0.04;

    private void Check(BackendType backend, Func<string, Gemma4Model> createTp)
    {
        string path = new Gemma4DenseSyntheticModelBuilder().Write(Path.Combine(_dir, "gemma4-dense.gguf"));

        // On GGML the TP degree is process-global native state that the last model loaded sets, so
        // the single-device reference runs to completion before the TP model loads. The reference
        // runs per op (no fused prefill, no captured decode graph).
        List<float[]>[] reference;
        using (var single = new Gemma4Model(path, backend))
        {
            single.ForceUnfused = true;
            reference = Rows(single);
        }

        List<float[]>[] tp;
        using (var model = createTp(path))
        {
            var group = (ITensorParallelGroup)typeof(ModelBase).GetField("_tpGroup", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);
            Assert.True(group is { IsActive: true, Degree: 2 }, "the model must run with two tensor-parallel ranks");
            Assert.False((bool)typeof(Gemma4Model).GetField("_tpFusedDecodeReady", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!,
                "the fused TP graphs must stay out: this test is about the per-op path");
            tp = Rows(model);
        }

        double tolerance = Tolerance(backend);
        var failures = new List<string>();
        for (int s = 0; s < Prompts.Length; s++)
        {
            for (int r = 0; r < tp[s].Count; r++)
            {
                double rel = RelL2(tp[s][r], reference[s][r]);
                _output.WriteLine($"{backend} tp=2 sequence {s} ({Prompts[s].Length} tokens) row {r}: relL2 {rel:E2}");
                if (!(rel <= tolerance))
                    failures.Add($"sequence {s} row {r}: relL2 {rel:E3} > {tolerance}");
            }
        }
        Assert.True(failures.Count == 0, $"{backend}: the tensor-parallel forward left the single-device path:\n" + string.Join("\n", failures));
    }

    /// <summary>Each prompt's logits after the prefill, then after each forced decode token.</summary>
    private static List<float[]>[] Rows(Gemma4Model model)
    {
        var rows = new List<float[]>[Prompts.Length];
        for (int s = 0; s < Prompts.Length; s++)
        {
            model.ResetKVCache();
            rows[s] = new List<float[]> { (float[])model.ForwardPrompt(Prompts[s]).Clone() };
            foreach (int t in Forced)
                rows[s].Add((float[])model.Forward(new[] { t }).Clone());
        }
        return rows;
    }

    /// <summary>Two tensor-parallel ranks on CUDA device 0, each with its own allocator and stream
    /// (both on the device's primary context); the AllReduce sums through the host.</summary>
    private sealed class OneGpuCudaGroup : ITensorParallelGroup
    {
        private readonly CudaAllocator[] _allocators;

        public OneGpuCudaGroup(int degree) =>
            _allocators = Enumerable.Range(0, degree).Select(_ => new CudaAllocator(0)).ToArray();

        public int Degree => _allocators.Length;
        public bool IsActive => Degree > 1;
        public int GlobalDegree => Degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => _allocators[rank];

        public void AllReduce(Tensor[] tensors)
        {
            Synchronize();
            int n = (int)tensors[0].Storage.ElementCount;
            float[] sum = tensors[0].GetElementsAsFloat(n);
            for (int r = 1; r < tensors.Length; r++)
            {
                float[] part = tensors[r].GetElementsAsFloat(n);
                for (int i = 0; i < n; i++)
                    sum[i] += part[i];
            }
            foreach (var tensor in tensors)
            {
                tensor.SetElementsAsFloat(sum);
                tensor.EnsureDeviceCurrent();
            }
            Synchronize();
        }

        public void Synchronize()
        {
            foreach (var allocator in _allocators)
                allocator.Synchronize();
        }

        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();

        public void Dispose()
        {
            foreach (var allocator in _allocators)
                allocator.Dispose();
        }
    }

    private static int[] Tokens(int count, int seed)
    {
        var tokens = new int[count];
        tokens[0] = 1;   // BOS
        uint state = (uint)seed * 2654435761u;
        for (int i = 1; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            tokens[i] = 3 + (int)(state % 250);
        }
        return tokens;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-300));
    }
}
