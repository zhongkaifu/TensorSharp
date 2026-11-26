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
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Gemma 4 model's logits on the synthetic fixtures, frozen on the managed backend (F32 caches) before the legacy
/// code was deleted (Fixtures/Gemma4/LegacySynthetic/managed.json): for each case a prompt in one chunk, then single-token
/// steps, every step's logits. <see cref="Gemma4RuntimeTests"/> holds the shared runtime's model to them. The cases cover
/// the E2B/E4B layout (per-layer embeddings, KV sharing, proportional global RoPE; a prompt past the 208-token window) and
/// the 26B-A4B layout (routed experts with fused gate|up, V = K global layers, a 32-token window crossed by the prompt; moe-kv2
/// with two global KV heads and 128-wide experts, which split over two ranks).
/// </summary>
internal static class Gemma4LegacySynthetic
{
    internal const string ResourceName = "InferenceWeb.Tests.Fixtures.Gemma4.LegacySynthetic.managed.json";

    internal sealed record Case(string Name, Func<string, string> Write, int[] Prompt, int[] Decode);

    internal static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 3 + (i * 7 + salt) % 120).ToArray();

    internal static readonly Case[] Cases =
    {
        new("dense", p => new Gemma4DenseSyntheticModelBuilder().Write(p), Tokens(37, 0), new[] { 11, 12, 13, 14, 15, 16 }),
        new("dense-long", p => new Gemma4DenseSyntheticModelBuilder().Write(p), Tokens(230, 5), new[] { 21, 22, 23, 24 }),
        new("moe", p => new Gemma4MoeSyntheticModelBuilder().Write(p), Tokens(45, 3), new[] { 31, 32, 33, 34, 35, 36 }),
        new("moe-kv2", p => new Gemma4MoeSyntheticModelBuilder { GlobalKvHeadCount = 2, ExpertFf = 128 }.Write(p), Tokens(37, 9), new[] { 41, 42, 43, 44 }),
    };

    internal static Case Get(string name) => Cases.Single(c => c.Name == name);

    internal sealed record Frozen(string Sha256, int[] Prompt, int[] Decode, int Vocab, float[] Logits);

    private static readonly Lazy<Dictionary<string, Frozen>> s_store = new(() =>
    {
        using Stream s = typeof(Gemma4LegacySynthetic).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new FileNotFoundException("no frozen legacy Gemma 4 logits embedded", ResourceName);
        using JsonDocument doc = JsonDocument.Parse(s);
        var store = new Dictionary<string, Frozen>(StringComparer.Ordinal);
        foreach (JsonProperty c in doc.RootElement.GetProperty("cases").EnumerateObject())
        {
            JsonElement v = c.Value;
            byte[] bytes = Convert.FromBase64String(v.GetProperty("logits").GetString());
            var logits = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, logits, 0, bytes.Length);
            store[c.Name] = new Frozen(v.GetProperty("sha256").GetString(),
                v.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                v.GetProperty("decode").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                v.GetProperty("vocab").GetInt32(), logits);
        }
        return store;
    });

    /// <summary>The frozen rows of <paramref name="name"/> (the prompt's last row, then each step's), after checking that
    /// the fixture the generator writes today is the one they were frozen from.</summary>
    internal static List<float[]> Rows(string name, string fixturePath)
    {
        Frozen f = s_store.Value[name];
        Assert.Equal(f.Sha256, FileSha(fixturePath));
        Case c = Get(name);
        Assert.Equal(f.Prompt, c.Prompt);
        Assert.Equal(f.Decode, c.Decode);
        var rows = new List<float[]>();
        for (int r = 0; r * f.Vocab < f.Logits.Length; r++) rows.Add(f.Logits.AsSpan(r * f.Vocab, f.Vocab).ToArray());
        return rows;
    }

    /// <summary>A case's rows from <paramref name="model"/>: the prompt in one chunk, then each decode token alone.</summary>
    internal static List<float[]> Run(ModelBase model, Case c)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardRefill(c.Prompt).Clone() };
        foreach (int t in c.Decode) rows.Add((float[])model.Forward(new[] { t }).Clone());
        return rows;
    }

    internal static string FileSha(string path)
    {
        using FileStream s = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant();
    }

    /// <summary>Freeze the legacy model's rows of every case into <paramref name="outPath"/> (run once, before the legacy
    /// code went: TS_GEMMA4_LEGACY_GOLDENS_OUT=&lt;file&gt; dotnet test --filter Gemma4LegacySyntheticWriter).</summary>
    internal static void Write(string outPath, string dir, Func<string, ModelBase> legacy)
    {
        var cases = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (Case c in Cases)
        {
            string path = c.Write(Path.Combine(dir, c.Name + ".gguf"));
            List<float[]> rows;
            int vocab;
            using (ModelBase model = legacy(path))
            {
                rows = Run(model, c);
                vocab = model.Config.VocabSize;
            }
            float[] all = rows.SelectMany(r => r).ToArray();
            var bytes = new byte[all.Length * 4];
            Buffer.BlockCopy(all, 0, bytes, 0, bytes.Length);
            cases[c.Name] = new { sha256 = FileSha(path), prompt = c.Prompt, decode = c.Decode, vocab, logits = Convert.ToBase64String(bytes) };
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, JsonSerializer.Serialize(new { backend = "cpu", kv = "f32", cases }, new JsonSerializerOptions { WriteIndented = true }));
    }
}
