// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed unsafe class Qwen35ArgmaxTests
{
    private readonly ITestOutputHelper _output;
    public Qwen35ArgmaxTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CpuOracleIncludesFirstNaNAndFirstMaximumSemantics()
    {
        var sampler = new TokenSampler(SamplingConfig.Greedy);
        Assert.Equal(0, sampler.Sample(new[] { float.NaN, float.PositiveInfinity }));
        Assert.Equal(2, sampler.Sample(new[] { -1f, float.NaN, 2f, 2f }));
        Assert.Equal(0, sampler.Sample(new[] { BitConverter.Int32BitsToSingle(int.MinValue), 0f }));
        Assert.Equal(0, sampler.Sample(new[] { float.NegativeInfinity, float.NegativeInfinity }));
        Assert.Equal(0, sampler.Sample(new[] { float.NaN, float.NaN }));
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx")]
    public void DeviceArgmaxMatchesActualTokenSamplerIncludingNonFiniteAndTies()
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var module = CudaModule.LoadFromFile(CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx"), identify: true);
        IntPtr fn = module.GetFunction("ts_q35_argmax_f32");
        _output.WriteLine($"module_sha256={module.ImageSha256}");
        const int rows = 12, guard = 16, canary = 0x31267912;
        var sampler = new TokenSampler(SamplingConfig.Greedy);
        void** args = stackalloc void*[3];
        foreach (int n in new[] { 1, 2, 31, 32, 33, 1023, 1024, 1025, 16387, 248320, 248323 })
        {
            var random = new Random(9107 + n);
            var input = new float[rows * n];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 20 - 10);
            Span<float> Row(int row) => input.AsSpan(row * n, n);
            Row(1).Fill(3);                                      // plateau
            Row(2).Fill(float.NegativeInfinity);                 // all -Inf
            Row(3).Fill(float.NaN);                              // all NaN
            Row(4).Fill(0); Row(4)[0] = BitConverter.Int32BitsToSingle(int.MinValue); // signed-zero tie
            Row(5)[0] = float.NaN; Row(5)[n - 1] = float.PositiveInfinity;
            Row(6).Fill(float.NaN); Row(6)[0] = -2; Row(6)[n - 1] = 3;
            Row(7).Fill(float.NegativeInfinity); Row(7)[0] = float.NaN;
            Row(8).Fill(-3); Row(8)[n / 2] = float.PositiveInfinity; Row(8)[n - 1] = float.PositiveInfinity;
            Row(9).Fill(-3); Row(9)[n / 3] = 8; Row(9)[n - 1] = 8; // distant tie
            Row(10)[0] = BitConverter.Int32BitsToSingle(unchecked((int)0x7f800001)); // signaling NaN
            Row(11).Fill(float.PositiveInfinity);
            var expected = Enumerable.Range(0, rows).Select(r => sampler.Sample(Row(r).ToArray())).ToArray();
            IntPtr deviceInput = IntPtr.Zero, deviceOutput = IntPtr.Zero;
            try
            {
                CudaDriverApi.cuMemAlloc(out deviceInput, new UIntPtr((ulong)input.Length * 4)).ThrowOnError();
                CudaDriverApi.cuMemAlloc(out deviceOutput, new UIntPtr((rows + 2 * guard) * 4)).ThrowOnError();
                var output = Enumerable.Repeat(canary, rows + 2 * guard).ToArray();
                fixed (float* p = input)
                    CudaDriverApi.cuMemcpyHtoD(deviceInput, (IntPtr)p, new UIntPtr((ulong)input.Length * 4)).ThrowOnError();
                fixed (int* p = output)
                    CudaDriverApi.cuMemcpyHtoD(deviceOutput, (IntPtr)p, new UIntPtr((ulong)output.Length * 4)).ThrowOnError();
                int width = n;
                IntPtr destination = deviceOutput + guard * 4;
                args[0] = &deviceInput;
                args[1] = &width;
                args[2] = &destination;
                CudaDriverApi.cuLaunchKernel(fn, rows, 1, 1, 1024, 1, 1, 0, allocator.Stream.Handle, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(allocator.Stream.Handle).ThrowOnError();
                fixed (int* p = output)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)p, deviceOutput, new UIntPtr((ulong)output.Length * 4)).ThrowOnError();
                Assert.Equal(expected, output[guard..(guard + rows)]);
                Assert.All(output.Take(guard).Concat(output.Skip(guard + rows)), value => Assert.Equal(canary, value));
            }
            finally
            {
                if (deviceOutput != IntPtr.Zero) CudaDriverApi.cuMemFree(deviceOutput);
                if (deviceInput != IntPtr.Zero) CudaDriverApi.cuMemFree(deviceInput);
            }
        }
        _output.WriteLine("11 vocabulary widths × 12 distributions: actual TokenSampler ids exactly equal device ids, including leading/signaling/all NaNs, infinities, ties and signed zeros; output guards intact.");
    }
}
