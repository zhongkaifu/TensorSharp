// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A prompt chunk's ggml_gated_delta_net in chunks of 16 rows on the tensor cores (ts_ssm_gdn_chunk_f32, the WY form
/// with BF16-split operands) against the graph executor's exact per-row kernel: the outputs and the state after the
/// last row agree within the split's ~16 mantissa bits, for several sequences, partial final chunks, key heads shared
/// by value heads, and the state written over the one it read.
/// </summary>
public sealed class CudaGatedDeltaNetChunkTests
{
    private const int S = 128;
    private readonly ITestOutputHelper _output;

    public CudaGatedDeltaNetChunkTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(1, 2, 4, 37, false)]
    [InlineData(2, 2, 4, 16, false)]
    [InlineData(1, 4, 4, 300, true)]
    [InlineData(3, 1, 3, 5, false)]
    public void ChunksMatchThePerRowRecurrence(int sequences, int keyHeads, int heads, int tokens, bool inPlace)
    {
        using var allocator = new CudaAllocator();
        CudaSsmPrefillKernels ssm = allocator.SsmPrefillKernels;
        if (ssm is not { RunsGatedDeltaNetChunks: true })
        {
            _output.WriteLine("no chunked Gated DeltaNet kernel for this GPU");
            return;
        }
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels rows, out string reason), reason);
        using (rows)
        {
            var rng = new Random(11);
            float[] Normed(int count)
            {
                var a = new float[count];
                for (int i = 0; i < count; i += S)
                {
                    double ss = 0;
                    for (int j = 0; j < S; j++) { a[i + j] = (float)(rng.NextDouble() * 2 - 1); ss += a[i + j] * a[i + j]; }
                    float inv = (float)(1 / Math.Sqrt(ss + 1e-6));
                    for (int j = 0; j < S; j++) a[i + j] *= inv;
                }
                return a;
            }
            float[] Uniform(int count, double lo, double hi)
            {
                var a = new float[count];
                for (int i = 0; i < count; i++) a[i] = (float)(lo + rng.NextDouble() * (hi - lo));
                return a;
            }
            int rowsAll = sequences * tokens;
            long rowFloats = (long)rowsAll * heads * S, stateFloats = (long)sequences * heads * S * S;
            using var q = Tensor.FromArray(allocator, Normed(rowsAll * keyHeads * S));
            using var k = Tensor.FromArray(allocator, Normed(rowsAll * keyHeads * S));
            using var v = Tensor.FromArray(allocator, Uniform(rowsAll * heads * S, -1, 1));
            using var g = Tensor.FromArray(allocator, Uniform(rowsAll * heads, -0.6, -0.01));
            using var beta = Tensor.FromArray(allocator, Uniform(rowsAll * heads, 0.05, 0.95));
            float[] state0 = Uniform((int)stateFloats, -0.2, 0.2);
            using var state = Tensor.FromArray(allocator, state0);
            using var expected = new Tensor(allocator, DType.Float32, rowFloats + stateFloats);
            using var got = new Tensor(allocator, DType.Float32, rowFloats + stateFloats);
            // The kernels read device memory: the inputs' host writes go there first, and the results are marked written.
            foreach (Tensor t in new[] { q, k, v, g, beta, state, expected, got })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            IntPtr stream = allocator.Stream.Handle;
            rows.LaunchGatedDeltaNet(P(q), P(k), P(v), P(g), P(beta), P(state), P(expected), S, heads, keyHeads, tokens, sequences, 1,
                false, stream);
            // In place: the chunked kernel reads the state from where it writes the result (the tail of its output).
            IntPtr stateIn = P(state);
            if (inPlace)
            {
                using Tensor tail = got.Narrow(0, rowFloats, stateFloats);
                Ops.Copy(tail, state);
                stateIn = ((CudaStorage)got.Storage).DevicePtrAtElement(rowFloats);
            }
            ssm.LaunchGatedDeltaNetChunks(P(q), P(k), P(v), P(g), P(beta), stateIn, ((CudaStorage)got.Storage).DevicePtrAtElement(rowFloats),
                P(got), keyHeads, heads, tokens, sequences, stream);
            allocator.Stream.Synchronize();
            ((CudaStorage)expected.Storage).MarkDeviceModified();
            ((CudaStorage)got.Storage).MarkDeviceModified();
            Assert.Contains(expected.GetElementsAsFloat(16), x => x != 0);
            float[] want = expected.GetElementsAsFloat((int)(rowFloats + stateFloats));
            float[] have = got.GetElementsAsFloat((int)(rowFloats + stateFloats));
            double Rel(long from, long count)
            {
                double num = 0, den = 0;
                for (long i = from; i < from + count; i++)
                {
                    double d = have[i] - want[i];
                    num += d * d;
                    den += (double)want[i] * want[i];
                }
                return Math.Sqrt(num / Math.Max(den, 1e-30));
            }
            double outputs = Rel(0, rowFloats), states = Rel(rowFloats, stateFloats);
            _output.WriteLine($"{sequences} x {tokens} rows, {keyHeads}/{heads} heads: outputs {outputs:E2}, state {states:E2} relative L2");
            Assert.True(outputs <= 2e-4 && states <= 2e-4, $"outputs {outputs:E2}, state {states:E2}");
        }
    }
}
