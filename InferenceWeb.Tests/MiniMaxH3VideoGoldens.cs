// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;
using Xunit;
using Frozen = InferenceWeb.Tests.MiniMaxH3ImageGoldens.Frozen;

namespace InferenceWeb.Tests;

/// <summary>
/// The retired MiniMax-H3 audio-video DiT's steps, frozen before it was deleted: MiniMaxH3DiT.Forward (the native whole-graph
/// TSGgml_MiniMaxH3DitForward on the GGML backends, MiniMaxH3DirectDiT on the direct cpu and cuda backends) on the synthetic
/// fixtures (<see cref="MiniMaxH3VideoFixture"/>), one file per lane (backend, runtime identifier and device:
/// Fixtures/MiniMaxH3/LegacyGoldens/*.json), in <see cref="MiniMaxH3ImageGoldens"/>' format: each output's SHA-256, size,
/// largest magnitude, L2 norm and a strided sample, and whether the runtime's denoiser gave the legacy bytes on that lane when
/// they were frozen. The cases cover every segment kind the trunk modulates: text with vision spans, keyframes at both ends,
/// Ref2VA picture and sound references (the conditioning chunk list), audio and video, two sigmas, and the unmerged LoRA.
/// </summary>
internal static class MiniMaxH3VideoGoldens
{
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.MiniMaxH3.LegacyGoldens.";

    /// <summary>One frozen step: its fixture's projection format, the request's kind, and the step's sigma.</summary>
    internal sealed record StepCase(string Name, int Projections, string Kind, float Sigma, bool Lora = false)
    {
        internal const int TextTokens = 9;

        internal MiniMaxH3Shape Shape => new()
        {
            Width = 96, Height = 64, Frames = 6, Fps = 24,
            LatentWidth = 6, LatentHeight = 4, LatentFrames = 2, AudioLatentFrames = 3,
        };

        /// <summary>Text tags: two vision spans of the prompt modulated as visual (a keyframe's and a reference's tokens).</summary>
        internal int[] TextModalities => Kind == "t2v" ? null : new[] { 1, 0, 0, 0, 1, 1, 0, 0, 1 };

        private static float[] Wave(long count, float step, float phase, Func<float, float> f)
        {
            var v = new float[count];
            for (long i = 0; i < count; i++) v[i] = f(i * step + phase);
            return v;
        }

        internal float[] Text => Wave((long)TextTokens * MiniMaxH3VideoFixture.TextDim, .31f, 0f, MathF.Sin);
        internal float[] Video => Wave((long)Shape.VideoTokenCount * MiniMaxH3VideoFixture.PatchDim, .17f, 0f, MathF.Sin);
        internal float[] Audio => Wave((long)Shape.AudioTokenCount * MiniMaxH3VideoFixture.AudioDim, .23f, 0f, MathF.Cos);

        /// <summary>Keyframes: two (one at each end), each a frame's 2x3 tokens. Ref2VA: a 4x4-latent picture (4 tokens) and a
        /// 2-latent-frame soundtrack (4 audio tokens), in reference order sound then picture.</summary>
        internal int ConditionCount => Kind switch { "keyframes" => 12, "ref2va" => 4, _ => 0 };
        internal int ConditionAudioCount => Kind == "ref2va" ? 4 : 0;
        internal float[] Condition => ConditionCount == 0 ? null : Wave((long)ConditionCount * MiniMaxH3VideoFixture.PatchDim, .29f, 1f, MathF.Sin);
        internal float[] ConditionAudio => ConditionAudioCount == 0 ? null : Wave((long)ConditionAudioCount * MiniMaxH3VideoFixture.AudioDim, .37f, .5f, MathF.Cos);

        /// <summary>The conditioning run's chunks in sequence order (kind 0 video patches, 1 audio latents), null: none.</summary>
        internal (int Kind, int Count)[] Chunks => Kind == "ref2va" ? new[] { (1, 4), (0, 4) } : null;

        internal MiniMaxH3PackedLayout Layout() => Kind switch
        {
            "keyframes" => MiniMaxH3Layout.Build(TextTokens, Shape, Sigma, conditionFrames: new[] { 1, 1 },
                keyframeAtEnd: new[] { false, true }, textTokenModalities: TextModalities),
            "ref2va" => MiniMaxH3Layout.Build(TextTokens, Shape, Sigma, textTokenModalities: TextModalities, references: new[]
            {
                new MiniMaxH3ReferenceBlock { Kind = MiniMaxH3ReferenceKind.Audio, AudioIndex = 0, AudioLatentFrames = 2 },
                new MiniMaxH3ReferenceBlock { Kind = MiniMaxH3ReferenceKind.Image, VideoIndex = 0, LatentFrames = 1, LatentHeight = 4, LatentWidth = 4 },
            }),
            _ => MiniMaxH3Layout.Build(TextTokens, Shape, Sigma),
        };
    }

    /// <summary>The frozen cases: each kind on the BF16 fixture at two sigmas, on the quantized fixtures once, and with a LoRA.</summary>
    internal static readonly StepCase[] Cases =
    {
        new("t2v-bf16-s1", MiniMaxH3ImageFixture.BF16, "t2v", 1f),
        new("t2v-bf16-s060", MiniMaxH3ImageFixture.BF16, "t2v", .6f),
        new("keyframes-bf16-s075", MiniMaxH3ImageFixture.BF16, "keyframes", .75f),
        new("ref2va-bf16-s075", MiniMaxH3ImageFixture.BF16, "ref2va", .75f),
        new("t2v-q8_0-s075", MiniMaxH3ImageFixture.Q8_0, "t2v", .75f),
        new("keyframes-q4_k-s075", MiniMaxH3ImageFixture.Q4_K, "keyframes", .75f),
        new("lora-keyframes-bf16-s075", MiniMaxH3ImageFixture.BF16, "keyframes", .75f, Lora: true),
        new("lora-t2v-q4_k-s1", MiniMaxH3ImageFixture.Q4_K, "t2v", 1f, Lora: true),
    };

    // ---------------------------------------------------------------- lanes and fixtures

    /// <summary>The lane a backend runs on here: the GGML backends' as MiniMax-H3-Image's; the tensor executors' (cpu, cuda)
    /// with the CPU's model or the CUDA device's name (asked of the CUDA driver: a process pinned to ggml-cpu cannot ask
    /// ggml-cuda).</summary>
    internal static string LaneKey(BackendType backend) => backend switch
    {
        BackendType.Cpu => "cpu|" + MiniMaxH3ImageGoldens.HostLaneKey().Split('|')[1] + "|" +
            MiniMaxH3ImageGoldens.LaneKey(BackendType.GgmlCpu).Split('|')[2],
        BackendType.Cuda => "cuda|" + MiniMaxH3ImageGoldens.HostLaneKey().Split('|')[1] + "|" + CudaDeviceName(),
        _ => MiniMaxH3ImageGoldens.LaneKey(backend),
    };

    private static string CudaDeviceName()
    {
        try { return TensorSharp.Cuda.CudaDevice.GetDevice(0).Name; }
        catch (Exception) { return "?"; }
    }

    /// <summary>What a case ran on: the denoiser file's SHA-256, and the LoRA weights' after it when the case has one.</summary>
    internal static string FixtureKey(string ditPath, string loraWeights = null) =>
        MiniMaxH3ImageGoldens.FileSha(ditPath) + (loraWeights != null ? "+" + MiniMaxH3ImageGoldens.FileSha(loraWeights) : "");

    /// <summary>A lane's file of frozen cases (the store's format).</summary>
    internal static string ToJson(string lane, IReadOnlyDictionary<string, Case> cases)
    {
        var root = new System.Text.Json.Nodes.JsonObject { ["lane"] = lane };
        var all = new System.Text.Json.Nodes.JsonObject();
        foreach (var (name, c) in cases)
            all[name] = new System.Text.Json.Nodes.JsonObject
            {
                ["fixtureSha256"] = c.FixtureSha256,
                ["runtimeBitExact"] = c.RuntimeBitExact,
                ["outputs"] = new System.Text.Json.Nodes.JsonArray(c.Outputs.Select(o => (System.Text.Json.Nodes.JsonNode)o.ToJson()).ToArray()),
            };
        root["cases"] = all;
        return root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    // ---------------------------------------------------------------- the store

    /// <summary>A frozen case on one lane: the fixture it ran on, its outputs (video, then audio), and whether the runtime gave
    /// their bytes there.</summary>
    internal sealed record Case(string Lane, string FixtureSha256, bool RuntimeBitExact, Frozen[] Outputs);

    private static readonly Lazy<Dictionary<string, Dictionary<string, Case>>> s_lanes = new(Load);

    private static Dictionary<string, Dictionary<string, Case>> Load()
    {
        var lanes = new Dictionary<string, Dictionary<string, Case>>(StringComparer.Ordinal);
        var asm = typeof(MiniMaxH3VideoGoldens).Assembly;
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal)))
        {
            using Stream s = asm.GetManifestResourceStream(name)!;
            using JsonDocument doc = JsonDocument.Parse(s);
            string lane = doc.RootElement.GetProperty("lane").GetString();
            var cases = new Dictionary<string, Case>(StringComparer.Ordinal);
            foreach (JsonProperty c in doc.RootElement.GetProperty("cases").EnumerateObject())
                cases[c.Name] = new Case(lane, c.Value.GetProperty("fixtureSha256").GetString(), c.Value.GetProperty("runtimeBitExact").GetBoolean(),
                    c.Value.GetProperty("outputs").EnumerateArray().Select(Frozen.FromJson).ToArray());
            lanes[lane] = cases;
        }
        return lanes;
    }

    internal static IReadOnlyCollection<string> Lanes => s_lanes.Value.Keys;

    /// <summary>Every lane's frozen <paramref name="caseName"/> from the fixture bytes <paramref name="fixtureSha256"/>.</summary>
    internal static IEnumerable<Case> Everywhere(string caseName, string fixtureSha256) =>
        s_lanes.Value.OrderBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => kv.Value.TryGetValue(caseName, out Case c) && c.FixtureSha256 == fixtureSha256 ? c : null)
            .Where(c => c != null);

    /// <summary>The largest sampled difference of <paramref name="got"/> from <paramref name="frozen"/>, over its largest value.</summary>
    internal static double SampledRelative(Frozen frozen, float[] got)
    {
        double worst = 0;
        for (int k = 0; k < frozen.Sample.Length; k++)
            worst = Math.Max(worst, Math.Abs(got[(long)k * frozen.Stride] - frozen.Sample[k]));
        return frozen.MaxAbs > 0 ? worst / frozen.MaxAbs : worst;
    }

    /// <summary>The frozen case for this lane; when this lane was never frozen, one frozen from the same fixture bytes: the
    /// same backend's on this runtime identifier first, else the GGML CPU lane's, else any lane's. Null when none exists.</summary>
    internal static Case Find(string caseName, string lane, string fixtureSha256)
    {
        var all = s_lanes.Value;
        if (all.TryGetValue(lane, out var mine) && mine.TryGetValue(caseName, out Case exact) && exact.FixtureSha256 == fixtureSha256) return exact;
        string[] parts = lane.Split('|');
        int Rank(string key)
        {
            string[] k = key.Split('|');
            if (k[0] == parts[0]) return k.ElementAtOrDefault(1) == parts.ElementAtOrDefault(1) ? 0 : 1;
            return k[0] == "ggml_cpu" ? 2 : 3;
        }
        foreach (var (key, cases) in all.OrderBy(kv => Rank(kv.Key)).ThenBy(kv => kv.Key, StringComparer.Ordinal))
            if (cases.TryGetValue(caseName, out Case c) && c.FixtureSha256 == fixtureSha256) return c;
        return null;
    }
}
