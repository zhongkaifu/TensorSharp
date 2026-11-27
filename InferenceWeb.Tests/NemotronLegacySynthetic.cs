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
/// The legacy Nemotron-H model's logits on the synthetic fixtures, frozen on the managed backend (F32 caches) before the
/// legacy code was deleted (Fixtures/Nemotron/LegacySynthetic/managed.json, written by its writer test with the legacy model
/// on the managed backend; the writer went with the model): for each case a prompt in one chunk, then single-token steps,
/// every step's logits. <see cref="NemotronRuntimeTests"/> holds the shared runtime's model to them. The
/// cases cover the dense nemotron_h hybrid (Mamba-2, attention, ReLU^2 FFN) and the nemotron_h_moe layouts: sigmoid-routed
/// ReLU^2 experts with a shared expert; a latent bottleneck with NVFP4-style sidecar scales (per tensor, and per expert on
/// the up and down stacks); and per-layer expert counts and top-k (Bolt).
/// </summary>
internal static class NemotronLegacySynthetic
{
    internal const string ResourceName = "InferenceWeb.Tests.Fixtures.Nemotron.LegacySynthetic.managed.json";

    internal sealed record Case(string Name, Func<string, string> Write, int[] Prompt, int[] Decode);

    internal static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 3 + (i * 7 + salt) % 120).ToArray();

    internal static readonly Case[] Cases =
    {
        new("dense", p => new NemotronSyntheticModelBuilder { Layout = "SFASF" }.Write(p), Tokens(37, 0), new[] { 11, 12, 13, 14, 15, 16 }),
        new("moe", p => new NemotronSyntheticModelBuilder { Layout = "SMASM" }.Write(p), Tokens(41, 3), new[] { 21, 22, 23, 24, 25, 26 }),
        new("moe-latent-scaled", p => new NemotronSyntheticModelBuilder { Layout = "SMAM", Latent = true, Scaled = true }.Write(p),
            Tokens(29, 5), new[] { 31, 32, 33, 34 }),
        new("moe-per-layer", p => new NemotronSyntheticModelBuilder
        {
            Layout = "SMASM", LayerExperts = new[] { 8, 6 }, UsedPerLayer = new uint[] { 0, 3, 0, 0, 2 },
        }.Write(p), Tokens(33, 7), new[] { 41, 42, 43, 44 }),
    };

    internal static Case Get(string name) => Cases.Single(c => c.Name == name);

    internal sealed record Frozen(string Sha256, int[] Prompt, int[] Decode, int Vocab, float[] Logits);

    private static readonly Lazy<Dictionary<string, Frozen>> s_store = new(() =>
    {
        using Stream s = typeof(NemotronLegacySynthetic).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new FileNotFoundException("no frozen legacy Nemotron-H logits embedded", ResourceName);
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
}
