// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Opt-in numerical parity and decode timing on one weight instance.
/// HTTP language quality and projector/vision validation are separate checks.</summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpRealBatchedDecodeTests(ITestOutputHelper output)
{
    [ModelFact("TS_TEST_QWEN4EXP_MODEL")]
    public void SerialAndBatchedTeacherForcedLogitsAndSoloContinuation()
        => RunParity([2, 3, 4], 2, 0);

    [ModelFact("TS_TEST_QWEN4EXP_MODEL")]
    public void TwoRequestLongContextTeacherForcedLogitsAndSoloContinuation()
        => RunParity([2], 1, 1479);

    [ModelFact("TS_TEST_QWEN4EXP_MODEL", GgmlBackend = BackendType.GgmlCuda)]
    public void ThreeRequestsCrossQsaBoundary_KeepCompleteLogitsAfterDepartures()
        => RunParity([3], 1, 2039, qsaBoundary: true);

    private void RunParity(int[] widths, int repetitions, int promptTokens, bool qsaBoundary = false)
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MODEL")!;
        int steps = int.TryParse(Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_BATCH_STEPS"), out int requested)
            ? requested : 16;
        Assert.True(steps >= 8, "Use at least eight steps per repetition (two repetitions per width).");
        Assert.True(!qsaBoundary || steps >= 16, "The QSA boundary case requires at least sixteen decode steps.");
        int? serialRebuildStep = null;
        string rebuildText = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_BATCH_REBUILD_STEP");
        if (!string.IsNullOrEmpty(rebuildText))
        {
            Assert.True(int.TryParse(rebuildText, out int rebuildStep) && rebuildStep >= 0 && rebuildStep < steps,
                "The diagnostic serial rebuild step must be within the requested decode trajectory.");
            serialRebuildStep = rebuildStep;
        }
        using var env = new EnvScope();
        int initialCapacity = 128;
        while (initialCapacity < checked(steps + 2 + Math.Max(64, promptTokens + (qsaBoundary ? 4 : 0))))
            initialCapacity = checked(initialCapacity * 2);
        env.Set("MAX_CONTEXT", Math.Max(256, initialCapacity).ToString());
        env.Set("TS_KV_INITIAL_TOKENS", initialCapacity.ToString());
        env.Set("TS_Q4E_DISABLE_ARENA_DECODE", null); env.ClearSpeculationVars();
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var model = Assert.IsType<Qwen4ExpModel>(ModelBase.Create(path, backend));
        using GgmlHostMoeExpertCacheScope? cachePlan = qsaBoundary
            ? new(4096L << 20, 32, model.Config.NumLayers, 3) : null;
        string[] texts =
        [
            "请详细介绍最终幻想7，包括故事背景、主要人物和游戏机制。",
            "请详细介绍《时间简史》，包括宇宙、黑洞和时间的主要观点。",
            "Explain why the sky appears blue and why sunsets are red.",
            "Describe the main differences between a compiler and an interpreter.",
        ];
        if (qsaBoundary) texts = ["请详细介绍最终幻想7", "请详细介绍时间简史", "请详细介绍最终幻想8"];
        int[][] prompts = texts.Select(text => model.Tokenizer.Encode(
            "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray()).ToArray();
        if (promptTokens > 0)
        {
            // A fixed, identical teacher-forcing context exercises production
            // QSA/capacity geometry. It is not a rendered WebUI prompt or a
            // language-quality benchmark; HTTP acceptance supplies those.
            int[] pad = model.Tokenizer.Encode("Read the supplied context carefully. ", addSpecial: false).ToArray();
            Assert.NotEmpty(pad);
            prompts = prompts.Select((p, i) => Enumerable.Range(0, Math.Max(0, promptTokens + (qsaBoundary ? i * 2 : 0) - p.Length))
                .Select(padIndex => pad[padIndex % pad.Length]).Concat(p).ToArray()).ToArray();
        }
        if (qsaBoundary)
            Assert.True(prompts.All(p => p.Length < 2048 && p.Length + steps > 2048),
                "Every independent request must cross the 2048-token selector boundary.");
        bool WithinAbsoluteBound(double value) => qsaBoundary ? value == 0 : value < .1;
        var reports = new List<object>();
        var trace = new List<object>();
        string failure = null;
        bool completed = false;
        // The long-context diagnostic must not construct a 1479-row graph on
        // a constrained device. Use the same bounded prefill geometry for
        // serial, independent, round-robin and batched-history controls.
        int prefillChunkTokens = promptTokens > 0 ? 128 : 0;
        float[] Prefill(int[] prompt)
        {
            if (prefillChunkTokens == 0) return model.Forward(prompt);
            float[] logits = null;
            for (int offset = 0; offset < prompt.Length; offset += prefillChunkTokens)
                logits = model.Forward(prompt.AsSpan(offset, Math.Min(prefillChunkTokens, prompt.Length - offset)).ToArray());
            return logits ?? throw new InvalidOperationException("The diagnostic prompt must not be empty.");
        }
        try
        {
        Assert.True(prompts.All(prompt => prompt.Length + steps + 1 <= initialCapacity),
            "The direct arena probe must reserve its complete decode span; cache growth is tested separately through the engine.");
        foreach (int width in widths)
        {
            for (int repetition = 0; repetition < repetitions; ++repetition)
            {
                var serialIds = Enumerable.Range(0, width).Select(i => $"serial-{width}-{repetition}-{i}").ToArray();
                var batchIds = Enumerable.Range(0, width).Select(i => $"batch-{width}-{repetition}-{i}").ToArray();
                var expected = new float[width][][];
                var initialExpected = new float[width][];
                var continuationExpected = new float[width][];
                var secondContinuationExpected = new float[width][];
                var forced = new int[width][];
                long serialTicks = 0;
                for (int i = 0; i < width; ++i)
                {
                    Assert.True(model.BindSequenceCache(serialIds[i]));
                    float[] logits = Prefill(prompts[i]);
                    float[] initial = (float[])logits.Clone();
                    initialExpected[i] = initial;
                    expected[i] = new float[steps][]; forced[i] = new int[steps];
                    for (int t = 0; t < steps; ++t)
                    {
                        forced[i][t] = ArgMax(logits);
                        if (serialRebuildStep == t)
                        {
                            // Diagnostic only: rebuild this holder's solo graph
                            // at the same history position as a differing arena
                            // step. The release keeps authoritative hybrid state,
                            // but checks the holder in, so bind it again before
                            // consuming the unchanged teacher token.
                            model.BindSequenceCache(serialIds[i]); // false means an existing holder was bound
                            model.ReleaseUnifiedPrefillGraphs(serialIds[i]);
                            model.BindSequenceCache(serialIds[i]);
                            Assert.True(model.CanBatchDecode(serialIds[i], prompts[i].Length + t),
                                "Rebuilding the solo graph must preserve the live holder's history position.");
                            output.WriteLine($"diagnostic serial rebuild: id={serialIds[i]}, row={i}, step={t}, position={prompts[i].Length + t}, token={forced[i][t]}");
                        }
                        long started = Stopwatch.GetTimestamp();
                        logits = model.Forward([forced[i][t]]);
                        serialTicks += Stopwatch.GetTimestamp() - started;
                        expected[i][t] = (float[])logits.Clone();
                    }
                    // Save this independent trajectory before reusing its
                    // holder capacity for the interleaved control arm.
                    continuationExpected[i] = (float[])model.Forward([ArgMax(expected[i][^1])]).Clone();
                    if (qsaBoundary) secondContinuationExpected[i] = (float[])model.Forward([ArgMax(continuationExpected[i])]).Clone();
                    Assert.True(model.BindSequenceCache(batchIds[i]));
                    float[] independent = Prefill(prompts[i]);
                    Assert.Equal(initial.Length, independent.Length);
                    double initialError = initial.Zip(independent, (x, y) => Math.Abs((double)x - y)).Max();
                    double initialKl = Kl(initial, independent);
                    trace.Add(new { phase = "independent-prefill", width, repetition, row = i, max_logit_error = initialError, kl = initialKl });
                    output.WriteLine($"initial width={width}, repetition={repetition}, row={i}: max |dlogit|={initialError:G8}, KL={initialKl:G8}");
                    Assert.True(WithinAbsoluteBound(initialError) && initialKl < 1e-4, $"Independent prefill row {i}: max |dlogit|={initialError:G8}, KL={initialKl:G8}");
                    if (i == 0)
                    {
                        string controlId = $"control-{width}-{repetition}";
                        Assert.True(model.BindSequenceCache(controlId));
                        float[] control = Prefill(prompts[i]);
                        Assert.Equal(initial.Length, control.Length);
                        double controlError = initial.Zip(control, (x, y) => Math.Abs((double)x - y)).Max();
                        double controlKl = Kl(initial, control);
                        trace.Add(new { phase = "serial-prefill-control", width, repetition, max_logit_error = controlError, kl = controlKl });
                        output.WriteLine($"serial control width={width}, repetition={repetition}: max |dlogit|={controlError:G8}, KL={controlKl:G8}");
                        Assert.True(WithinAbsoluteBound(controlError) && controlKl < 1e-4, $"Serial control: max |dlogit|={controlError:G8}, KL={controlKl:G8}");
                        model.OnSequenceReleased(controlId);
                    }
                }
                foreach (string id in serialIds) model.OnSequenceReleased(id);
                model.RestorePrimaryCache();
                var roundRobinIds = Enumerable.Range(0, width).Select(i => $"round-robin-{width}-{repetition}-{i}").ToArray();
                long roundRobinTicks = 0;
                double roundRobinWorst = 0, roundRobinWorstKl = 0;
                int roundRobinGreedyDifferences = 0;
                for (int i = 0; i < width; ++i)
                {
                    Assert.True(model.BindSequenceCache(roundRobinIds[i]));
                    float[] initial = Prefill(prompts[i]);
                    Assert.Equal(initialExpected[i].Length, initial.Length);
                    double difference = initialExpected[i].Zip(initial, (x, y) => Math.Abs((double)x - y)).Max();
                    double kl = Kl(initialExpected[i], initial);
                    trace.Add(new { phase = "round-robin-prefill", width, repetition, row = i, max_logit_error = difference, kl });
                    Assert.True(WithinAbsoluteBound(difference) && kl < 1e-4,
                        $"Round-robin prefill width={width}, rep={repetition}, row={i}: max |dlogit|={difference:G8}, KL={kl:G8}");
                }
                // Match the scheduler's fallback ordering: switch to each live
                // sequence for one token, then advance the next decode step.
                // Cache binding is deliberately part of this arm's timing.
                for (int t = 0; t < steps; ++t)
                    for (int i = 0; i < width; ++i)
                    {
                        long started = Stopwatch.GetTimestamp();
                        model.BindSequenceCache(roundRobinIds[i]);
                        float[] actual = model.Forward([forced[i][t]]);
                        roundRobinTicks += Stopwatch.GetTimestamp() - started;
                        float[] reference = expected[i][t];
                        Assert.Equal(reference.Length, actual.Length);
                        Assert.All(actual, x => Assert.True(float.IsFinite(x)));
                        double difference = reference.Zip(actual, (x, y) => Math.Abs((double)x - y)).Max();
                        double kl = Kl(reference, actual);
                        roundRobinWorst = Math.Max(roundRobinWorst, difference);
                        roundRobinWorstKl = Math.Max(roundRobinWorstKl, kl);
                        int best = ArgMax(reference), actualBest = ArgMax(actual);
                        double margin = reference[best] - reference.Where((_, index) => index != best).Max();
                        if (best != actualBest) ++roundRobinGreedyDifferences;
                        trace.Add(new { phase = "round-robin-decode", width, repetition, step = t, row = i,
                            max_logit_error = difference, kl, serial_argmax = best, round_robin_argmax = actualBest, greedy_margin = margin });
                        Assert.True(WithinAbsoluteBound(difference) && kl < 1e-4,
                            $"Round-robin width={width}, rep={repetition}, step={t}, row={i}: max |dlogit|={difference:G8}, KL={kl:G8}");
                        if (margin > 2 * difference) Assert.Equal(best, actualBest);
                    }
                foreach (string id in roundRobinIds) model.OnSequenceReleased(id);
                model.RestorePrimaryCache();
                long batchTicks = 0, before = model.ArenaBatchedDecodeSteps;
                double worst = 0, worstKl = 0, smallestMargin = double.PositiveInfinity;
                int greedyDifferences = 0;
                for (int t = 0; t < steps; ++t)
                {
                    var order = Enumerable.Range(0, width).OrderBy(i => (i + t) % width).ToArray();
                    var rows = new float[width][];
                    long started = Stopwatch.GetTimestamp();
                    Assert.True(model.TryForwardBatchedFusedDecode(order.Select(i => batchIds[i]).ToArray(),
                        order.Select(i => forced[i][t]).ToArray(), order.Select(i => prompts[i].Length + t).ToArray(), rows),
                        model.BatchedFusedDecodeDeclineReason);
                    batchTicks += Stopwatch.GetTimestamp() - started;
                    for (int j = 0; j < width; ++j)
                    {
                        var a = expected[order[j]][t]; var b = rows[j];
                        Assert.Equal(a.Length, b.Length);
                        Assert.All(b, x => Assert.True(float.IsFinite(x)));
                        double difference = a.Zip(b, (x, y) => Math.Abs((double)x - y)).Max();
                        double kl = Kl(a, b); worst = Math.Max(worst, difference); worstKl = Math.Max(worstKl, kl);
                        int best = ArgMax(a); double margin = a[best] - a.Where((_, index) => index != best).Max();
                        smallestMargin = Math.Min(smallestMargin, margin);
                        if (best != ArgMax(b)) ++greedyDifferences;
                        trace.Add(new { phase = "batched-decode", width, repetition, step = t, row = order[j],
                            max_logit_error = difference, kl, serial_argmax = best, batched_argmax = ArgMax(b), greedy_margin = margin });
                        Assert.True(WithinAbsoluteBound(difference), $"width={width}, rep={repetition}, step={t}, row={j}: max |dlogit|={difference:G8}");
                        Assert.True(kl < 1e-4, $"width={width}, step={t}, row={j}: KL={kl:G8}");
                        if (margin > 2 * difference) Assert.Equal(best, ArgMax(b));
                    }
                }
                Assert.Equal(steps, model.ArenaBatchedDecodeSteps - before);
                double continuationError = 0;
                if (qsaBoundary)
                {
                    model.OnSequenceReleased(batchIds[0]);
                    int[] survivors = [1, 2];
                    var rows = new float[2][];
                    Assert.True(model.TryForwardBatchedFusedDecode(survivors.Select(i => batchIds[i]).ToArray(),
                        survivors.Select(i => ArgMax(expected[i][^1])).ToArray(),
                        survivors.Select(i => prompts[i].Length + steps).ToArray(), rows),
                        model.BatchedFusedDecodeDeclineReason);
                    for (int row = 0; row < survivors.Length; ++row)
                    {
                        int original = survivors[row];
                        double difference = continuationExpected[original].Zip(rows[row], (a, b) => Math.Abs((double)a - b)).Max();
                        continuationError = Math.Max(continuationError, difference);
                        trace.Add(new { phase = "departure-n3-to-n2", width, repetition, row = original, max_logit_error = difference });
                        Assert.Equal(0, difference);
                    }
                    model.OnSequenceReleased(batchIds[1]);
                    model.BindSequenceCache(batchIds[2]);
                    float[] solo = model.Forward([ArgMax(continuationExpected[2])]);
                    double soloDifference = secondContinuationExpected[2].Zip(solo, (a, b) => Math.Abs((double)a - b)).Max();
                    continuationError = Math.Max(continuationError, soloDifference);
                    trace.Add(new { phase = "departure-n2-to-n1", width, repetition, row = 2, max_logit_error = soloDifference });
                    Assert.Equal(0, soloDifference);
                }
                else for (int i = 0; i < width; ++i)
                {
                    int next = ArgMax(expected[i][^1]);
                    float[] serial = continuationExpected[i];
                    model.BindSequenceCache(batchIds[i]); float[] batched = model.Forward([next]);
                    Assert.Equal(serial.Length, batched.Length);
                    double difference = serial.Zip(batched, (x, y) => Math.Abs((double)x - y)).Max();
                    continuationError = Math.Max(continuationError, difference);
                    trace.Add(new { phase = "solo-continuation", width, repetition, row = i, max_logit_error = difference });
                    Assert.True(WithinAbsoluteBound(difference), $"width={width}, continuation row={i}: max |dlogit|={difference:G8}");
                }
                var report = new
                {
                    width, repetition, steps, tokens = width * steps, max_logit_error = worst, max_kl = worstKl,
                    greedy_differences = greedyDifferences, minimum_greedy_margin = smallestMargin,
                    solo_continuation_error = continuationError,
                    serial_decode_seconds = serialTicks / (double)Stopwatch.Frequency,
                    round_robin_decode_seconds = roundRobinTicks / (double)Stopwatch.Frequency,
                    batched_decode_seconds = batchTicks / (double)Stopwatch.Frequency,
                    serial_tokens_per_second = width * steps * (double)Stopwatch.Frequency / serialTicks,
                    round_robin_tokens_per_second = width * steps * (double)Stopwatch.Frequency / roundRobinTicks,
                    batched_tokens_per_second = width * steps * (double)Stopwatch.Frequency / batchTicks,
                    round_robin_max_logit_error = roundRobinWorst, round_robin_max_kl = roundRobinWorstKl,
                    round_robin_greedy_differences = roundRobinGreedyDifferences,
                };
                reports.Add(report); output.WriteLine(JsonSerializer.Serialize(report));
                foreach (string id in batchIds) model.OnSequenceReleased(id);
                model.RestorePrimaryCache();
            }
        }
        completed = true;
        }
        catch (Exception error)
        {
            failure = error.ToString();
            throw;
        }
        finally
        {
        string directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_REPORT_DIR");
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "managed-batch-probe.json"), JsonSerializer.Serialize(new
            {
                model = path, backend = backend.ToString(), steps_per_repetition = steps, repetitions, widths,
                prompt_token_counts = prompts.Select(p => p.Length).ToArray(), completed, failure,
                prefill_chunk_tokens = prefillChunkTokens > 0 ? (int?)prefillChunkTokens : null,
                prefill_policy = prefillChunkTokens > 0 ? "identical bounded chunks in every arm" : "one complete prompt forward in every arm",
                diagnostic_serial_rebuild_step = serialRebuildStep,
                qsa_boundary_requested = qsaBoundary,
                departures = qsaBoundary ? "N3 -> N2 after releasing row0; N2 -> N1 after releasing row1" : "independent solo continuations",
                absolute_logit_bound = qsaBoundary ? 0 : .1,
                native_library_path = Qwen4ExpExpertCacheScenario.MappedNativePath(),
                round_robin_iteration_order = "step-then-row", round_robin_cache_bind_in_timing = true,
                scope = "Decode-only teacher forcing; one shared weight instance. Independent serial timing completes each sequence; round-robin timing interleaves one token per sequence and includes cache binding. No prefill, HTTP, image, language-quality, or long-context performance claim.",
                reports, trace,
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        }
    }

    private static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());

    private static double Kl(float[] p, float[] q)
    {
        double maxP = p.Max(), maxQ = q.Max();
        double zP = p.Sum(x => Math.Exp(x - maxP)), zQ = q.Sum(x => Math.Exp(x - maxQ));
        double logZP = maxP + Math.Log(zP), logZQ = maxQ + Math.Log(zQ), sum = 0;
        for (int i = 0; i < p.Length; ++i)
            sum += Math.Exp(p[i] - logZP) * (p[i] - logZP - q[i] + logZQ);
        return Math.Max(0, sum);
    }
}
