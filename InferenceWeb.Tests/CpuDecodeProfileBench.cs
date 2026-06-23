// Opt-in CPU-backend decode profiler. Loads a GGUF on BackendType.Cpu, prefills a
// prompt, decodes N tokens, and reports tok/s, average cores used
// (process CPU-time / wall), and the matmul-vs-rest split (TS_CPU_MATMUL_PROFILE=1).
//
//   TS_CPU_BENCH=1 TS_CPU_MATMUL_PROFILE=1 \
//   TS_CPU_MODEL="C:\Works\models\gemma_mtp\qat\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf" \
//   dotnet test --filter CpuDecodeProfile_Bench -c Release
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

public class CpuDecodeProfileBench
{
    private readonly ITestOutputHelper _out;
    public CpuDecodeProfileBench(ITestOutputHelper o) => _out = o;

    [Fact]
    public void CpuDecodeProfile_Bench()
    {
        if (Environment.GetEnvironmentVariable("TS_CPU_BENCH") != "1")
        { _out.WriteLine("set TS_CPU_BENCH=1 to run; skipping"); return; }

        string path = Environment.GetEnvironmentVariable("TS_CPU_MODEL")
            ?? @"C:\Works\models\gemma_mtp\qat\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
        if (!File.Exists(path)) { _out.WriteLine($"missing model {path}; skipping"); return; }

        int maxNew = int.TryParse(Environment.GetEnvironmentVariable("TS_CPU_NEW"), out int n) ? n : 32;
        int ctx = int.TryParse(Environment.GetEnvironmentVariable("TS_CPU_CTX"), out int c) ? c : 0;
        string prompt = Environment.GetEnvironmentVariable("TS_CPU_PROMPT") ?? "请详细介绍最终幻想7";

        _out.WriteLine($"cores={Environment.ProcessorCount} model={Path.GetFileName(path)}");
        using var model = ModelBase.Create(path, BackendType.Cpu);
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: true).ToArray();
        if (ctx > tokens.Length)
        {
            // Pad the prompt with a repeated synthetic context to simulate a long
            // running generation (large KV cache) so we can observe how decode
            // utilisation changes with context length.
            var padded = new List<int>(tokens);
            int filler = tokens.Length > 1 ? tokens[1] : tokens[0];
            while (padded.Count < ctx) padded.Add(filler);
            tokens = padded.ToArray();
        }
        _out.WriteLine($"prompt tokens={tokens.Length} maxNew={maxNew}");

        // Prefill (not counted in decode timing).
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(tokens);
        int t = Argmax(logits);
        var outTok = new List<int> { t };

        // Warm one decode step.
        logits = model.Forward(new[] { t }); t = Argmax(logits); outTok.Add(t);

        ManagedQuantizedOps.ResetMatmulProfile();
        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        var cpu0 = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew();
        for (int i = 2; i < maxNew; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            outTok.Add(t);
        }
        sw.Stop();
        proc.Refresh();
        double wall = sw.Elapsed.TotalSeconds;
        double cpu = (proc.TotalProcessorTime - cpu0).TotalSeconds;
        int decoded = maxNew - 2;
        double avgCores = cpu / wall;

        _out.WriteLine($"decoded {decoded} tok in {wall:F2}s = {decoded / wall:F2} tok/s");
        _out.WriteLine($"avg cores = {avgCores:F2} / {Environment.ProcessorCount}  (util {100.0 * avgCores / Environment.ProcessorCount:F0}%)");

        if (ManagedQuantizedOps.MatmulProfileEnabled)
        {
            var (mm, gib, calls) = ManagedQuantizedOps.ReadMatmulProfile();
            _out.WriteLine($"matmul: {mm:F0} ms total ({100.0 * mm / (wall * 1000):F0}% of wall), {calls} calls, {gib:F1} GiB weights read");
            _out.WriteLine($"matmul bandwidth = {gib / (mm / 1000.0):F1} GiB/s; rest (serial+attn+overhead) = {wall * 1000 - mm:F0} ms ({100.0 * (wall * 1000 - mm) / (wall * 1000):F0}%)");
        }
        _out.WriteLine($"sample out: {Trim(model.Tokenizer.Decode(outTok))}");
        double checksum = 0; for (int i = 0; i < logits.Length; i++) checksum += logits[i] * (double)((i & 1023) + 1);
        _out.WriteLine($"tokens=[{string.Join(",", outTok.GetRange(0, Math.Min(12, outTok.Count)))}] logitsChecksum={checksum:R}");
    }

    private static int Argmax(float[] v)
    {
        int bi = 0; float bv = v[0];
        for (int i = 1; i < v.Length; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
        return bi;
    }
    private static string Trim(string s) => s.Length > 160 ? s.Substring(0, 160) + "…" : s;
}
