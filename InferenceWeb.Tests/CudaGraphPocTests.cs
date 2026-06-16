// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Go/no-go PoC for the CUDA-graph rearchitecture of the pure-C# CUDA backend.
// MTP-verify profiling showed the verify issues ~1050 kernel launches/step (42
// layers x ~25 ops); the open question is how much of that is per-op CPU/WDDM
// LAUNCH overhead (which a captured CUDA graph replays in ONE launch) vs real GPU
// compute. This measures the per-launch overhead on THIS machine by timing N per-op
// launches against one graph replay of the same N launches, then projects the
// headroom onto a ~1050-launch verify. Opt-in: TS_CUDA_GRAPH_POC=1 (needs CUDA).
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaGraphPocTests
{
    private readonly ITestOutputHelper _output;
    public CudaGraphPocTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void CudaGraph_LaunchOverhead_GoNoGo()
    {
        if (Environment.GetEnvironmentVariable("TS_CUDA_GRAPH_POC") != "1")
        { _output.WriteLine("[graph-poc] set TS_CUDA_GRAPH_POC=1 to run; skipping"); return; }

        const int iters = 5000;
        const int verifyLaunches = 1050;   // measured ~25 ops x 42 layers per verify

        var allocator = new CudaAllocator(0);
        // Small tensor so the per-kernel GPU time is tiny and the per-op CPU/WDDM
        // launch overhead (what a graph eliminates) dominates the per-op timing.
        using var t = new Tensor(allocator, DType.Float32, 1, 256);
        Ops.Fill(t, 1.0f);

        // One capturable in-place launch (scalar mul by 1.0 -> no host sync / alloc),
        // representative of the C# op-dispatch + cuLaunchKernel cost the verify pays.
        Action issueOneLaunch = () => Ops.Mul(t, t, 1.0f);

        var (peropMs, graphMs, captured) = CudaFusedOps.MeasureGraphReplay(t, iters, issueOneLaunch);

        _output.WriteLine($"[graph-poc] iters={iters} captured={captured}");
        _output.WriteLine($"[graph-poc] per-op : {peropMs:F2} ms  ({peropMs / iters * 1000.0:F2} us/launch)");
        _output.WriteLine($"[graph-poc] graph  : {graphMs:F2} ms  ({graphMs / iters * 1000.0:F2} us/launch)");

        Assert.True(captured, "stream capture failed — Ops.Mul issued a non-capturable op (host sync/alloc)");

        double perLaunchSavedUs = (peropMs - graphMs) / iters * 1000.0;
        double verifyHeadroomMs = perLaunchSavedUs * verifyLaunches / 1000.0;
        double speedup = graphMs > 0 ? peropMs / graphMs : 0;
        _output.WriteLine($"[graph-poc] launch overhead eliminated by graph: {perLaunchSavedUs:F2} us/launch");
        _output.WriteLine($"[graph-poc] per-op/graph speedup on this sequence: {speedup:F1}x");
        _output.WriteLine($"[graph-poc] => projected launch headroom in a ~{verifyLaunches}-launch verify: {verifyHeadroomMs:F1} ms");
        _output.WriteLine($"[graph-poc] (verify is ~51 ms; ~43 ms is the ggml-parity target.)");
        _output.WriteLine(verifyHeadroomMs >= 8.0
            ? "[graph-poc] VERDICT: GO — launch overhead is large; CUDA graphs are worth the rearchitecture."
            : "[graph-poc] VERDICT: NO-GO — launch overhead is small; graphs won't close the gap, pivot to fused mega-kernels.");
    }
}
