// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;

namespace TensorSharp.Server.Jev;

/// <summary>One-step structured reads. Only exact requested label logits contribute to answers.</summary>
internal static class JevInference
{
    internal delegate float[][] Read(int[] prompt, int[] canvas, int[] positions, int[][] tokenIds, CancellationToken cancellation);

    internal static object Run(JevRequest request, string modelName, Func<string, int[]> encode,
        Func<string, string, int[]> renderPrompt, Read read, int maxWidth, int maxContext,
        int eos, int pad, int vocabSize, CancellationToken ct,
        int? imageCount = null, object? attachments = null, double preprocessingMs = 0)
    {
        ct.ThrowIfCancellationRequested();
        var timer = Stopwatch.StartNew();
        var templates = JevCompiler.Compile(request, encode, maxWidth);
        // Validate every prompt before executing any model work. Context is never truncated:
        // truncation could silently drop a question or change the meaning of the state.
        var prompts = templates.Select(t => renderPrompt(JevCompiler.SystemText(request,
            request.SharedPrompt ? request.Questions : t.Questions, request.SharedPrompt && templates.Count > 1, imageCount), request.State)).ToArray();
        for (int i = 0; i < prompts.Length; ++i)
            if ((long)prompts[i].Length + templates[i].CanvasWidth > maxContext)
                throw new JevValidationException($"state and question prompt need {prompts[i].Length + templates[i].CanvasWidth} tokens; model context holds {maxContext}");
        var answers = new Dictionary<string, object>(StringComparer.Ordinal);
        var diagnostics = new Dictionary<string, object>(StringComparer.Ordinal);
        var groups = new List<object>();
        int totalReads = 0, totalInput = 0, totalOutput = 0;
        for (int group = 0; group < templates.Count; ++group)
        {
            var template = templates[group];
            var draws = new List<double[][]>();
            int count = request.Samples == 0 ? 1 : request.Samples;
            bool extended = false;
            for (int sample = 0; sample < count; ++sample)
            {
                ct.ThrowIfCancellationRequested();
                int seed = unchecked(request.Seed + group * 104729 + sample * 7919);
                var canvas = JevCompiler.Canvas(template, eos, pad, vocabSize, seed);
                float[][] probabilities = read(prompts[group], canvas, template.Positions, template.LabelTokenIds, ct);
                var normalized = ValidateProbabilities(probabilities, template);
                draws.Add(normalized);
                totalReads++;
                if (sample == 0 && request.Samples == 0 && request.AutoMax > 1 &&
                    normalized.Any(p => Entropy(p) > request.AutoThreshold))
                {
                    count = request.AutoMax;
                    extended = true;
                }
            }
            totalInput += prompts[group].Length;
            totalOutput += template.Tokens.Length + 1;
            for (int qi = 0; qi < template.Questions.Length; ++qi)
            {
                var q = template.Questions[qi];
                double[] mean = Enumerable.Range(0, q.Labels.Length).Select(i => draws.Average(d => d[qi][i])).ToArray();
                int top = ArgMax(mean);
                answers.Add(q.Id, Answer(q, mean));
                double stderr = draws.Count < 2 ? 0 : Math.Sqrt(draws.Sum(d => Math.Pow(d[qi][top] - mean[top], 2)) / (draws.Count - 1) / draws.Count);
                diagnostics.Add(q.Id, new
                {
                    pos = template.Positions[qi],
                    conditional_entropy = draws.Select(d => Entropy(d[qi])).ToArray(),
                    stderr = draws.Count > 1 ? (double?)stderr : null,
                    agreement = draws.Count > 1 ? (double?)draws.Count(d => ArgMax(d[qi]) == top) / draws.Count : null,
                    samples = draws.Count,
                });
            }
            groups.Add(new { questions = template.Questions.Select(q => q.Id).ToArray(),
                canvas_width = template.CanvasWidth, samples = draws.Count, extended,
                first_read_conditional_entropy = draws[0].Select(Entropy).ToArray() });
        }
        ct.ThrowIfCancellationRequested();
        double inferenceMs = timer.Elapsed.TotalMilliseconds;
        return new
        {
            model = modelName,
            answers,
            usage = new { input_tokens = totalInput, output_tokens = totalOutput },
            diagnostics = new
            {
                engine = "tensorsharp", steps = 1,
                images = imageCount ?? request.Images.Length,
                attachments = attachments ?? Array.Empty<object>(),
                probability_semantics = "conditional_label_softmax_temperature_1",
                entropy_semantics = "conditional_label_entropy_nats",
                seed = request.Seed,
                samples = new { policy = request.Samples == 0 ? "auto" : "fixed", n = request.Samples == 0 ? (int?)null : request.Samples,
                    auto_max = request.AutoMax, auto_threshold = request.AutoThreshold },
                timing = new { total_ms = inferenceMs + preprocessingMs,
                    preprocessing_ms = preprocessingMs, inference_ms = inferenceMs, reads = totalReads },
                chunks = groups,
                questions = diagnostics,
            },
        };
    }

    private static double[][] ValidateProbabilities(float[][] result, JevTemplate template)
    {
        if (result == null || result.Length != template.Questions.Length)
            throw new InvalidOperationException("Structured read returned the wrong number of question distributions.");
        var normalized = new double[result.Length][];
        for (int q = 0; q < result.Length; ++q)
        {
            var p = result[q];
            if (p == null || p.Length != template.LabelTokenIds[q].Length || p.Any(v => !float.IsFinite(v) || v < 0 || v > 1))
                throw new InvalidOperationException("Structured read returned an invalid probability distribution.");
            double sum = p.Sum(v => (double)v);
            if (sum <= 0 || Math.Abs(sum - 1) > 0.001)
                throw new InvalidOperationException("Structured read probabilities are not normalized.");
            normalized[q] = p.Select(v => v / sum).ToArray();
        }
        return normalized;
    }

    internal static double Entropy(double[] p) => -p.Where(v => v > 0).Sum(v => v * Math.Log(v));
    internal static int ArgMax(double[] p)
    {
        int top = 0;
        for (int i = 1; i < p.Length; ++i) if (p[i] > p[top]) top = i;
        return top;
    }
    internal static object Answer(JevQuestion q, double[] p)
    {
        int top = ArgMax(p);
        if (q.Type == "noul") return new { type = "noul", noul = p[0] };
        if (q.Type == "choice") return new { type = "choice", choice = q.Names[top],
            probabilities = q.Names.Select((name, i) => (name, p: p[i])).ToDictionary(x => x.name, x => x.p), confidence = p[top] };
        return new { type = "score", score = p.Select((v, i) => i * v).Sum(),
            legend = q.Names.Select((name, i) => (value: q.Levels is null ? (object)name : q.Levels[i], key: i.ToString(CultureInfo.InvariantCulture))).ToDictionary(x => x.key, x => x.value),
            probabilities = p.Select((v, i) => (v, key: i.ToString(CultureInfo.InvariantCulture))).ToDictionary(x => x.key, x => x.v), confidence = p[top] };
    }
}
