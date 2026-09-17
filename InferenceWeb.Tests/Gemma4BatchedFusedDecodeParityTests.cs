// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Token-batched fused decode parity for the Gemma 4 E2B/E4B family.
//
// The engine serves N>=2 concurrent Gemma 4 requests through
// Gemma4Model.TryForwardBatchedFusedDecode (ONE fused graph for all N decode
// tokens) and falls back to round-robin single-token decodes when the model
// declines. The E2B/E4B checkpoints used to decline ALWAYS: per-layer embeddings
// (PLE dim 256), 18 shared-KV (donor) layers and a 512-slot SWA ring that most
// chats outgrow were all outside the v1 kernel's scope, so "4 concurrent
// requests" ran at ~1x aggregate throughput.
//
// This test drives the same sequences through both paths and requires the
// greedy continuations to match token for token, for 2, 3 and 4 concurrent
// sequences, with one sequence prefilled PAST the SWA ring so the batched
// kernel's wrap path (write pos % ring, read the ring flat) is exercised too.
// The horizon is short (12 steps) on purpose: batching changes GEMM shapes and
// therefore rounding, so over hundreds of greedy tokens a low-margin token can
// flip on ANY batched decode (measured on the v1 kernel with gemma-4-12B as
// well; see docs/models/gemma4.md). A divergence inside 12 steps on a
// confident prompt is a bug; one at step 200 is not.
//
// Opt-in: TS_TEST_MODEL_DIR=<dir containing gemma-4-E4B*.gguf>, and
// TS_TEST_GGML_BACKEND=cuda|metal|cpu (default cpu; one GGML backend per
// process). When the native build predates the extended kernel the batched
// path declines by design; the test then reports that and passes without
// having proven anything, so read the output.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4BatchedFusedDecodeParityTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;

    public Gemma4BatchedFusedDecodeParityTests(ITestOutputHelper output) { _output = output; }

    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4-e4b")]
    public async Task BatchedFusedDecode_MatchesRoundRobin_TokenForToken()
    {
        string modelPath = FindGemma4();
        if (modelPath == null) { _output.WriteLine("no gemma-4-e4b model; skipping"); return; }

        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").ToLowerInvariant() switch
        {
            "cuda" => BackendType.GgmlCuda,
            "metal" => BackendType.GgmlMetal,
            "vulkan" => BackendType.GgmlVulkan,
            _ => BackendType.GgmlCpu,
        };
        ModelBase model;
        try
        {
            model = ModelBase.Create(modelPath, backend);
        }
        catch (Exception ex)
        {
            _output.WriteLine($"cannot load on {backend} ({ex.GetType().Name}: {ex.Message}); skipping");
            return;
        }
        using var _model = model;
        await Task.Yield();

        if (model is not Gemma4Model g4 || model is not IBatchedPagedModel seq || !seq.SupportsPerSequenceFusedForward)
        {
            _output.WriteLine("model has no per-sequence fused path on this backend; skipping");
            return;
        }

        var caps = GgmlBasicOps.Gemma4BatchedDecodeCapabilities();
        _output.WriteLine($"[parity] backend={backend} native batched-decode caps={caps}");

        // Four prompts; the second is repeated past the 512-token SWA ring so its
        // local layers wrap during the batched steps.
        var prompts = new List<int[]>
        {
            model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray(),
            LongPrompt(model, minTokens: 560),
            model.Tokenizer.Encode("Write three sentences about the ocean.", addSpecial: true).ToArray(),
            model.Tokenizer.Encode("List the planets of the solar system in order:", addSpecial: true).ToArray(),
        };
        _output.WriteLine($"[parity] prompt lengths = {string.Join(",", prompts.Select(p => p.Length))}");

        const int steps = 12;
        bool anyFused = false;
        foreach (int n in new[] { 2, 3, 4 })
        {
            var ids = Enumerable.Range(0, n).Select(i => $"parity-{n}-{i}").ToArray();

            // --- serial: each sequence decoded alone on its own holder (the
            // round-robin fallback the engine would run).
            var serial = new List<List<int>>();
            for (int i = 0; i < n; i++)
            {
                seq.BindSequenceCache(ids[i]);
                var outs = new List<int>();
                int tok = ArgMax(model.Forward(prompts[i]));
                outs.Add(tok);
                for (int s = 1; s < steps; s++)
                {
                    tok = ArgMax(model.Forward(new[] { tok }));
                    outs.Add(tok);
                }
                serial.Add(outs);
                seq.OnSequenceReleased(ids[i]);
            }

            // --- batched: same prefills, then one fused step per token for all n.
            var lastTok = new int[n];
            var pos = new int[n];
            var batched = new List<List<int>>();
            for (int i = 0; i < n; i++)
            {
                seq.BindSequenceCache(ids[i]);
                lastTok[i] = ArgMax(model.Forward(prompts[i]));
                pos[i] = prompts[i].Length;
                batched.Add(new List<int> { lastTok[i] });
            }
            var outLogits = new float[n][];
            int fusedSteps = 0, fallbackSteps = 0;
            for (int s = 1; s < steps; s++)
            {
                if (seq.TryForwardBatchedFusedDecode(ids, lastTok, pos, outLogits))
                {
                    fusedSteps++;
                    for (int i = 0; i < n; i++)
                    {
                        lastTok[i] = ArgMax(outLogits[i]);
                        pos[i]++;
                        batched[i].Add(lastTok[i]);
                    }
                }
                else
                {
                    fallbackSteps++;
                    for (int i = 0; i < n; i++)
                    {
                        seq.BindSequenceCache(ids[i]);
                        lastTok[i] = ArgMax(model.Forward(new[] { lastTok[i] }));
                        pos[i]++;
                        batched[i].Add(lastTok[i]);
                    }
                }
            }
            for (int i = 0; i < n; i++) seq.OnSequenceReleased(ids[i]);

            _output.WriteLine($"[parity] n={n}: fused steps={fusedSteps} fallback steps={fallbackSteps}");
            for (int i = 0; i < n; i++)
            {
                bool same = serial[i].SequenceEqual(batched[i]);
                _output.WriteLine($"[parity] n={n} seq{i} (prompt {prompts[i].Length} tok): {(same ? "MATCH" : "DIFF")}");
                if (!same)
                {
                    _output.WriteLine($"          serial : {string.Join(' ', serial[i])}");
                    _output.WriteLine($"          batched: {string.Join(' ', batched[i])}");
                }
                Assert.True(same, $"n={n} seq{i}: batched fused decode diverged from the round-robin decode");
            }

            const GgmlBasicOps.Gemma4BatchedDecodeCaps needed =
                GgmlBasicOps.Gemma4BatchedDecodeCaps.Ple
                | GgmlBasicOps.Gemma4BatchedDecodeCaps.KvDonor
                | GgmlBasicOps.Gemma4BatchedDecodeCaps.SwaWrap;
            if ((caps & needed) == needed)
            {
                // The extended kernel must actually serve every step: a silent
                // decline would make this a round-robin-vs-round-robin comparison.
                Assert.Equal(steps - 1, fusedSteps);
                anyFused = true;
            }
        }

        if (!anyFused)
            _output.WriteLine("[parity] the native build lacks the extended batched kernel; the batched path declined (nothing proven).");
        _ = g4;
    }

    private static int[] LongPrompt(ModelBase model, int minTokens)
    {
        var sb = new System.Text.StringBuilder("Here is some background before the question. ");
        int i = 0;
        while (true)
        {
            sb.Append($"Fact {i}: the river valley town kept a written record of every harvest, flood and market day. ");
            var toks = model.Tokenizer.Encode(sb.ToString() + "Question: what did the town record? Answer:", addSpecial: true);
            if (toks.Count >= minTokens) return toks.ToArray();
            i++;
        }
    }

    private static string FindGemma4()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").FirstOrDefault(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("gemma-4-e4b") && !n.Contains("mmproj") && !n.Contains("assistant");
        });
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }
}
