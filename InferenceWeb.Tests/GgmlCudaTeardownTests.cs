// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A ggml CUDA buffer cannot be freed from GgmlOps' thread_local destructors or atexit
// handlers: on Windows they run inside the DLL's thread or process detach, where cudaFree
// fails ("CUDA-capable device(s) is/are busy or unavailable") and ggml aborts the process.
// Three native caches were released only there: the Mamba2 prefill graphs (an atexit
// handler), and the fused prefill-attention and paged-attention sessions (thread_local, of
// every thread except the one that called TSGgml_Shutdown). That is what aborted the
// ggml_cuda test lane after a Nemotron forward, once every test had passed. A model unload
// (TSGgml_ClearHostBufferCache) and TSGgml_Shutdown now release all three, and a thread's
// exit leaves its sessions to be freed on another thread.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class GgmlCudaTeardownTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-cuda-teardown-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    /// <summary>
    /// The CLI shuts GGML down before it exits, but the shutdown never reached the Mamba2
    /// prefill graphs, so a Nemotron-H run that reached the native Mamba2 prefill (8 tokens
    /// or more; the CLI's own kernel warmup is 64) left one for the atexit handler: the
    /// answer was printed, then the process aborted. A child process, because the abort
    /// happens at exit.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void Cli_ExitsCleanlyAfterANativeMamba2Prefill()
    {
        string model = new NemotronSyntheticModelBuilder { Layout = "SAMS" }.Write(Path.Combine(_dir, "nemotron.gguf"));
        string input = Path.Combine(_dir, "prompt.txt");
        File.WriteAllText(input, "Tell me how the tide follows the moon.");

        var startInfo = new ProcessStartInfo
        {
            FileName = DotnetHost(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _dir,
        };
        foreach (string arg in new[] { HostAssembly("TensorSharp.Cli", "TensorSharp.Cli.dll"), "--model", model,
                     "--backend", "ggml_cuda", "--input", input, "--max-tokens", "4", "--no-skills",
                     "--log-dir", Path.Combine(_dir, "logs") })
            startInfo.ArgumentList.Add(arg);
        startInfo.Environment["TENSORSHARP_LOG_DIR"] = Path.Combine(_dir, "logs");
        // Nothing inherited from the test run may change what the host loads.
        foreach (string name in new[] { "MAX_CONTEXT", "KV_CACHE_DTYPE", "TENSORSHARP_TP_DEGREE",
                     "TS_NEMOTRON_MAMBA2_NATIVE_PREFILL_MIN_TOKENS", "TS_MAMBA2_PREFILL_CACHE_MB" })
            startInfo.Environment.Remove(name);

        using var process = new Process { StartInfo = startInfo };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) => { if (e.Data != null) lock (stdout) stdout.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (stderr) stderr.AppendLine(e.Data); };
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        if (!process.WaitForExit(120_000))
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail("the CLI did not exit within 2 minutes");
        }
        process.WaitForExit();

        string output = $"stdout:\n{stdout}\nstderr:\n{stderr}";
        Assert.Contains("cli.inference.start", output);
        Assert.DoesNotContain("CUDA error", output);
        Assert.True(process.ExitCode == 0, $"exit code {process.ExitCode}\n{output}");
    }

    /// <summary>
    /// Fused prefill-attention sessions are cached per thread. TSGgml_Shutdown freed only
    /// the calling thread's and a model unload none, so another thread's session waited for
    /// that thread's thread_local destructor, which aborted the process when the thread
    /// exited. A model unload now frees every live thread's sessions; a thread that exits
    /// with one hands it over, and the next unload frees it.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void ModelUnload_ReleasesThePrefillAttentionSessionsOfEveryThread()
    {
        // 32 heads x 1024 queries over 1024 keys: the cached session holds two 128 MiB
        // score tensors, about 300 MiB in all, so its release is plain in the free memory.
        // Measured against what the session took, since a GGML_CUDA_DEVICES device reports
        // its share of the GPU.
        const int heads = 32, kvHeads = 8, headDim = 64, rows = 1024;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0);
        float[] first = null, second = null;
        Exception failure = null;
        using var computed = new SemaphoreSlim(0);
        using var proceed = new SemaphoreSlim(0);
        var worker = new Thread(() =>
        {
            try
            {
                first = Attend(allocator, heads, kvHeads, headDim, rows);
                computed.Release();
                proceed.Wait();
                // Builds the session again, and exits holding it.
                second = Attend(allocator, heads, kvHeads, headDim, rows);
            }
            catch (Exception ex) { failure = ex; computed.Release(); }
        });
        try
        {
            // The same shape once on this thread first, as the paged test does: the attention's first run in a process
            // grows what the backend keeps for good (its pool, the device code of the kernels it loads), so in a process
            // of its own the worker's first run would count that too (RTX PRO 4500: 400 MiB held, 296 released, the
            // session's alone). The unload frees this thread's session; whatever an earlier test left resident goes with it.
            float[] reference = Attend(allocator, heads, kvHeads, headDim, rows);
            GgmlBasicOps.ClearHostBufferCache();
            long idle = FreeMiB();

            worker.Start();
            computed.Wait();
            Assert.Null(failure);
            long held = idle - FreeMiB();
            Assert.True(held >= 64, $"the session took {held} MiB of device memory");
            long released = ReleasedMiBByUnload();
            Assert.True(released >= held * 3 / 4,
                $"a model unload released {released} of {held} MiB: it left the session of a thread that is still running");

            proceed.Release();
            worker.Join();
            Assert.Null(failure);
            released = ReleasedMiBByUnload();
            Assert.True(released >= held * 3 / 4,
                $"a model unload released {released} of {held} MiB: it left the session of a thread that has exited");
            Assert.Equal(reference, first);
            Assert.Equal(first, second);
        }
        finally
        {
            proceed.Release();
            if (worker.IsAlive)
                worker.Join();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>
    /// Paged-attention sessions were cached per thread the same way, and freed by nothing but
    /// that thread's thread_local destructor (TSGgml_Shutdown reached only its own thread's,
    /// a model unload none). Continuous batching reaches TSGgml_PagedAttentionForward from
    /// server and ThreadPool threads for Nemotron, Qwen3.5, Gemma 4 and GPT-OSS, so a batching
    /// thread that exited holding a session aborted the process. The sessions are now owned
    /// like the prefill ones: a model unload frees every live thread's, and a thread that
    /// exits hands its session over for the next unload to free.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void ModelUnload_ReleasesThePagedAttentionSessionsOfEveryThread()
    {
        // 512 queries of 32 heads over 8192 keys of 8 x 128: the session's K/V inputs and
        // their permuted copies are 128 MiB, about 170 MiB in all.
        const int heads = 32, kvHeads = 8, headDim = 128, rows = 512, keys = 8192;
        _ = new GgmlContext([0], GgmlBackendType.Cuda);
        var batch = new PagedBatch(heads, kvHeads, headDim, rows, keys);
        float[] first = null, second = null;
        Exception failure = null;
        using var computed = new SemaphoreSlim(0);
        using var proceed = new SemaphoreSlim(0);
        var worker = new Thread(() =>
        {
            try
            {
                first = batch.Attend();
                computed.Release();
                proceed.Wait();
                // Builds the session again, and exits holding it.
                second = batch.Attend();
            }
            catch (Exception ex) { failure = ex; computed.Release(); }
        });
        try
        {
            // The same shape once on this thread first: flash attention's F16 copies of K
            // and V come from the backend's own pool, which keeps what it grew, so the memory
            // measured below is the worker's session alone. The unload frees this thread's
            // session; whatever an earlier test left resident goes with it.
            float[] reference = batch.Attend();
            GgmlBasicOps.ClearHostBufferCache();
            long idle = FreeMiB();

            worker.Start();
            computed.Wait();
            Assert.Null(failure);
            long held = idle - FreeMiB();
            Assert.True(held >= 64, $"the session took {held} MiB of device memory");
            long released = ReleasedMiBByUnload();
            Assert.True(released >= held * 3 / 4,
                $"a model unload released {released} of {held} MiB: it left the session of a thread that is still running");

            proceed.Release();
            worker.Join();
            Assert.Null(failure);
            released = ReleasedMiBByUnload();
            Assert.True(released >= held * 3 / 4,
                $"a model unload released {released} of {held} MiB: it left the session of a thread that has exited");
            Assert.Equal(reference, first);
            Assert.Equal(first, second);
        }
        finally
        {
            proceed.Release();
            if (worker.IsAlive)
                worker.Join();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    private static long FreeMiB()
    {
        Assert.True(GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out _));
        return free >> 20;
    }

    private static long ReleasedMiBByUnload()
    {
        long before = FreeMiB();
        GgmlBasicOps.ClearHostBufferCache();
        return FreeMiB() - before;
    }

    private static float[] Attend(IAllocator allocator, int heads, int kvHeads, int headDim, int rows)
    {
        using var q = Filled(allocator, heads * rows * headDim, seed: 1);
        using var k = Filled(allocator, kvHeads * rows * headDim, seed: 2);
        using var v = Filled(allocator, kvHeads * rows * headDim, seed: 3);
        using var output = new Tensor(allocator, DType.Float32, rows, heads * headDim);
        GgmlBasicOps.FusedPrefillAttention(q, k, v, output, heads, kvHeads, headDim,
            rows, rows, maskStartPos: 0, slidingWindow: 0, scale: 1f / MathF.Sqrt(headDim));
        return output.GetElementsAsFloat(rows * heads * headDim);
    }

    /// <summary>One sequence whose last <c>rows</c> tokens query all <c>keys</c>, its K/V
    /// paged in 64-token blocks stored in reverse order.</summary>
    private sealed class PagedBatch(int heads, int kvHeads, int headDim, int rows, int keys)
    {
        private const int BlockSize = 64;
        private readonly float[] _q = Values(rows * heads * headDim, seed: 1);
        private readonly float[] _k = Values(keys * kvHeads * headDim, seed: 2);
        private readonly float[] _v = Values(keys * kvHeads * headDim, seed: 3);

        public float[] Attend()
        {
            int blocks = keys / BlockSize;
            var output = new float[_q.Length];
            GgmlBasicOps.PagedAttentionForward(_q, _k, _v, output,
                queryStartLoc: new[] { 0, rows }, seqLens: new[] { keys },
                positions: Enumerable.Range(keys - rows, rows).ToArray(),
                blockTableFlat: Enumerable.Range(0, blocks).Reverse().ToArray(), blockTableOffsets: new[] { 0 },
                numSeqs: 1, numTokens: rows, numHeads: heads, numKvHeads: kvHeads, headDim: headDim,
                blockSize: BlockSize, scale: 1f / MathF.Sqrt(headDim));
            return output;
        }
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static Tensor Filled(IAllocator allocator, int count, int seed)
    {
        var tensor = new Tensor(allocator, DType.Float32, count);
        tensor.SetElementsAsFloat(Values(count, seed));
        return tensor;
    }

    private static string DotnetHost()
    {
        string fromSdk = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(fromSdk) && File.Exists(fromSdk))
            return fromSdk;
        string root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (!string.IsNullOrWhiteSpace(root))
        {
            string candidate = Path.Combine(root, OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet");
            if (File.Exists(candidate))
                return candidate;
        }
        return "dotnet";
    }

    private static string HostAssembly(string project, string fileName)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        string path = Path.Combine(dir.FullName, project, "bin", fileName);
        Assert.True(File.Exists(path), $"Required host build output not found: {path}");
        return path;
    }
}
