// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Actual-model file-prefill qualification. Complete-logit parity and
/// quiet ABBA use the same registered sources, allocator budgets and chunk shapes.
/// HTTP quality, request admission and hard physical limits are separate checks.</summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpFilePipelineIntegrationTests(ITestOutputHelper output)
{
    [ModelFact("TS_TEST_QWEN4EXP_MODEL", GgmlBackend = BackendType.GgmlCuda)]
    public void ScopedFilePrefill_HasIdenticalCompleteLogitsAndTeacherContinuation() => Run(benchmark: false);

    [ModelFact("TS_TEST_QWEN4EXP_MODEL", GgmlBackend = BackendType.GgmlCuda)]
    public void ScopedFilePrefill_QuietAbba() => Run(benchmark: true);

    private void Run(bool benchmark)
    {
        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "1024"); env.Set("TS_KV_INITIAL_TOKENS", "1024");
        env.Set("TS_Q4E_DISABLE_ARENA_DECODE", null); env.ClearSpeculationVars();
        if (benchmark) Assert.NotEqual("1", Environment.GetEnvironmentVariable("TS_HOST_MOE_TIMING"));
        MoeCpuOffloadConfig.Reset(); MoeCpuOffloadConfig.SetAllLayers();
        string modelPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MODEL")!;
        var comparisons = new List<object>(); var timings = new List<object>();
        bool completed = false; string? failure = null;
        object? hardwareReport = null;
        try
        {
            var hardware = InferenceHardwareMemory.CaptureCuda();
            var host = hardware.HostAllocationCapacity();
            long hostCeiling = Math.Max(0, host.Available - (1L << 30));
            long deviceCeiling = Math.Max(0, hardware.DeviceAvailable - (1L << 30));
            hardwareReport = new { hardware, host_ceiling = hostCeiling, device_ceiling = deviceCeiling };
            Assert.True(hostCeiling > (4L << 30) && deviceCeiling > (8L << 30),
                "File-prefill qualification requires available RAM/commit and CUDA capacity after 1GiB margins.");
            var budget = new MemoryBudget([new("pipeline/ram", hostCeiling), new("pipeline/cuda0", deviceCeiling)]);
            using var native = new GgmlCacheBudgetScope(budget, [["pipeline/cuda0"]], true, ["pipeline/ram"]);
            using var anonymousHost = new HostAllocationBudgetScope(budget, ["pipeline/ram"]);
            using var experts = new GgmlHostMoeExpertCacheScope(4096L << 20, 32, 48, 3);
            using var model = Assert.IsType<Qwen4ExpModel>(ModelBase.Create(modelPath, BackendType.GgmlCuda));
            string[] texts = ["请详细介绍最终幻想7", "请详细介绍时间简史", "请详细介绍最终幻想8"];
            int[] pad = model.Tokenizer.Encode("Read this context carefully. ", addSpecial: false).ToArray();
            Assert.NotEmpty(pad);
            var cases = texts.SelectMany(text => new[] { 128, 512 }.Select(chunk =>
            {
                int[] suffix = model.Tokenizer.Encode("<|im_start|>user\n" + text
                    + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
                int[] tokens = Enumerable.Range(0, Math.Max(0, 513 - suffix.Length))
                    .Select(i => pad[i % pad.Length]).Concat(suffix).ToArray();
                return (Text: text, Chunk: chunk, Tokens: tokens);
            })).ToArray();
            if (!benchmark)
            {
                foreach (var item in cases)
                {
                    var serial = Execute(item.Text, item.Chunk, item.Tokens, false, null, capture: true, decode: true);
                    var pipelined = Execute(item.Text, item.Chunk, item.Tokens, true, serial.Teacher, capture: true, decode: true);
                    Assert.Equal(serial.Logits.Length, pipelined.Logits.Length);
                    for (int phase = 0; phase < serial.Logits.Length; ++phase)
                    {
                        float[] left = serial.Logits[phase], right = pipelined.Logits[phase];
                        Assert.Equal(model.Config.VocabSize, left.Length); Assert.Equal(left.Length, right.Length);
                        double maximum = 0;
                        for (int token = 0; token < left.Length; ++token)
                        {
                            Assert.True(float.IsFinite(left[token]) && float.IsFinite(right[token]));
                            maximum = Math.Max(maximum, Math.Abs((double)left[token] - right[token]));
                        }
                        comparisons.Add(new { prompt = item.Text, chunk = item.Chunk, phase,
                            vocabulary = left.Length, max_absolute_logit_difference = maximum });
                        Assert.Equal(0, maximum);
                    }
                    output.WriteLine($"File pipeline parity: chunk={item.Chunk}, rows={pipelined.Logits.Length}, "
                        + $"successful pipeline executions={pipelined.PipelineCalls}, source bytes={pipelined.PipelineBytes}");
                }
            }
            else
            {
                // Prime both paths before timing. Every arm resets only sequence
                // state and trims the same idle file workspace; expert weights
                // share one planned owner. Shape order and threads stay fixed.
                int cycles = int.TryParse(Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_PIPELINE_ABBA_CYCLES"), out int n) ? n : 2;
                Assert.InRange(cycles, 2, 8);
                foreach (var item in cases.Where(item => item.Text == texts[0]))
                {
                    Execute(item.Text, item.Chunk, item.Tokens, false, null, capture: false, decode: false);
                    Execute(item.Text, item.Chunk, item.Tokens, true, null, capture: false, decode: false);
                    for (int cycle = 0; cycle < cycles; ++cycle)
                        foreach (bool enabled in new[] { false, true, true, false })
                        {
                            var phase = Execute(item.Text, item.Chunk, item.Tokens, enabled, null, capture: false, decode: false);
                            var hostUsage = anonymousHost.Usage;
                            timings.Add(new { prompt = item.Text, chunk = item.Chunk, cycle, enabled,
                                prefill_seconds = phase.Seconds, prefill_tokens_per_second = item.Tokens.Length / phase.Seconds,
                                pipeline_calls = phase.PipelineCalls, pipeline_source_bytes = phase.PipelineBytes,
                                pipeline_retained_ring_bytes_after_prefill = phase.RetainedRingBytes,
                                native_usage = native.AllocationUsage,
                                anonymous_host = new { hostUsage.Bytes, hostUsage.PeakBytes, hostUsage.Allocations },
                                budget = budget.Snapshot() });
                        }
                }
            }
            Assert.Null(native.CallbackError);
            completed = true;

            (float[][] Logits, int[] Teacher, double Seconds, long PipelineCalls, long PipelineBytes, long RetainedRingBytes) Execute(
                string text, int chunk, int[] tokens, bool enabled, int[]? teacher, bool capture, bool decode)
            {
                model.ResetKVCache();
                // Identical quiescent release in every arm, retaining selected
                // experts while removing previous full-layer file workspaces.
                experts.Trim(experts.CachedBytes);
                using var pipeline = new GgmlHostMoeFilePipelineScope(enabled, 64, 32L << 20, 16L << 20);
                var before = GgmlBasicOps.GetHostMoeFilePipelineUsage();
                var logits = new List<float[]>(); var forced = decode ? new int[8] : [];
                float[] current = [];
                var clock = Stopwatch.StartNew();
                for (int offset = 0; offset < tokens.Length; offset += chunk)
                {
                    current = model.Forward(tokens.AsSpan(offset, Math.Min(chunk, tokens.Length - offset)).ToArray());
                    if (capture) logits.Add((float[])current.Clone());
                }
                double seconds = clock.Elapsed.TotalSeconds;
                var afterPrefill = GgmlBasicOps.GetHostMoeFilePipelineUsage();
                long calls = afterPrefill.Calls - before.Calls, bytes = afterPrefill.SourceBytes - before.SourceBytes;
                Assert.True(enabled ? calls > 0 && bytes >= 32L << 20 : calls == 0,
                    "Requested file pipeline did not select its actual eligible path, or disabled scope executed it.");
                Assert.True(afterPrefill.RetainedRingBytes <= 16L << 20);
                for (int step = 0; step < forced.Length; ++step)
                {
                    int next = teacher?[step] ?? Array.IndexOf(current, current.Max()); forced[step] = next;
                    current = model.Forward([next]);
                    if (capture) logits.Add((float[])current.Clone());
                }
                Assert.Equal(afterPrefill.Calls, GgmlBasicOps.GetHostMoeFilePipelineUsage().Calls);
                return (logits.ToArray(), forced, seconds, calls, bytes, afterPrefill.RetainedRingBytes);
            }
        }
        catch (Exception error) { failure = error.ToString(); throw; }
        finally
        {
            MoeCpuOffloadConfig.Reset();
            string? directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_REPORT_DIR");
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
                File.WriteAllText(Path.Combine(directory, benchmark ? "file-pipeline-abba.json" : "file-pipeline-parity.json"),
                    JsonSerializer.Serialize(new { completed, failure, model = modelPath, hardware = hardwareReport,
                        comparisons, timings, scope = "Same loaded actual weights, registered file sources, shared native/host quotas, "
                        + "cache prefill ceiling32 and planned concurrency3. Complete full-vocabulary prefill/decode parity or "
                        + "quiet within-process ABBA at identical128/512-token chunking,513-token prompts. "
                        + "Parity covers all three prompt stems; ABBA covers the FF7 stem at both shapes. "
                        + "No HTTP, vision, request-admission, factual-quality, OS hard-limit or cross-device claim." },
                        new JsonSerializerOptions { WriteIndented = true }));
            }
        }
    }
}
