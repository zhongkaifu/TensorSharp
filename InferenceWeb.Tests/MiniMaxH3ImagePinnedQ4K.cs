// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The synthetic denoiser's Q4_K projections as the reference lane's machine (ggml_cpu on the M5 Pro that froze the legacy
/// outputs) quantized them, so any machine can rebuild the exact fixture those outputs came from.
/// </summary>
/// <remarks>
/// ggml's Q4_K quantizer rounds its floats differently with the CPU and the compiler flags it was built with: the frozen
/// fixtures are one per machine that froze them (the M5 Pro and the RTX PRO 4500 box), and the CI runners' x64 and ARM64 builds
/// quantize the same seed to other bytes again. On such a machine the frozen-legacy tests rebuild the fixture with these
/// bytes (<see cref="MiniMaxH3ImageFixture"/>'s <c>pinnedQ4K</c>) and hold the runtime to the reference lane's outputs within
/// the bound. Every entry carries the SHA-256 of the seed's values it was quantized from, so a change to the fixture's
/// generator fails here instead of pairing new values with old bytes.
/// </remarks>
internal static class MiniMaxH3ImagePinnedQ4K
{
    internal const string Resource = "InferenceWeb.Tests.Fixtures.MiniMaxH3Image.PinnedQ4K.denoiser-seed2051.bin.gz";
    private const string Magic = "TSPQ4K1";

    private static readonly Lazy<Dictionary<string, (byte[] SourceSha256, byte[] Bytes)>> s_entries = new(Load);

    /// <summary>The fixture's encoding of a tensor, with a Q4_K tensor's bytes taken from the store.</summary>
    internal static byte[] Encode(MiniMaxH3ImageFixture.T t)
    {
        if (t.Type != MiniMaxH3ImageFixture.Q4_K) return MiniMaxH3ImageFixture.Encode(t);
        if (!s_entries.Value.TryGetValue(t.Name, out var entry))
            throw new InvalidOperationException($"the pinned Q4_K store has no {t.Name}");
        if (!SourceSha256(t.Data).AsSpan().SequenceEqual(entry.SourceSha256))
            throw new InvalidOperationException(
                $"{t.Name}: the fixture's values are not the ones the pinned Q4_K bytes were quantized from; a fixture generator change needs new reference outputs");
        return entry.Bytes;
    }

    internal static byte[] SourceSha256(float[] data) => SHA256.HashData(MemoryMarshal.AsBytes(data.AsSpan()));

    /// <summary>Writes the Q4_K tensors of <paramref name="tensors"/>, quantized by this machine's ggml, as the store.</summary>
    internal static void Write(string path, IReadOnlyList<MiniMaxH3ImageFixture.T> tensors)
    {
        var q4k = tensors.Where(t => t.Type == MiniMaxH3ImageFixture.Q4_K).ToArray();
        using var file = File.Create(path);
        using var gzip = new GZipStream(file, CompressionLevel.SmallestSize);
        using var writer = new BinaryWriter(gzip, Encoding.UTF8);
        writer.Write(Magic);
        writer.Write(q4k.Length);
        foreach (var t in q4k)
        {
            byte[] bytes = MiniMaxH3ImageFixture.Encode(t);
            writer.Write(t.Name);
            writer.Write(SourceSha256(t.Data));
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }
    }

    private static Dictionary<string, (byte[], byte[])> Load()
    {
        using Stream resource = typeof(MiniMaxH3ImagePinnedQ4K).Assembly.GetManifestResourceStream(Resource)
            ?? throw new InvalidOperationException($"missing embedded resource {Resource}");
        using var gzip = new GZipStream(resource, CompressionMode.Decompress);
        using var reader = new BinaryReader(gzip, Encoding.UTF8);
        if (reader.ReadString() != Magic) throw new InvalidDataException($"{Resource} is not a pinned Q4_K store");
        int count = reader.ReadInt32();
        var entries = new Dictionary<string, (byte[], byte[])>(StringComparer.Ordinal);
        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            byte[] sourceSha = reader.ReadBytes(32);
            byte[] bytes = reader.ReadBytes(reader.ReadInt32());
            entries[name] = (sourceSha, bytes);
        }
        return entries;
    }
}

public sealed class MiniMaxH3ImagePinnedQ4KTests(ITestOutputHelper output)
{
    /// <summary>Names the file <see cref="WritesThePinnedBytes"/> writes; the writer skips without it.</summary>
    internal const string WriterEnv = "TS_TEST_PIN_H3IMAGE_Q4K";

    private static string ReferenceLane =>
        MiniMaxH3ImageGoldens.Lanes.Single(l => l.StartsWith("ggml_cpu|osx-arm64|", StringComparison.Ordinal));

    private static IEnumerable<string> Q4KCases =>
        MiniMaxH3ImageFrozenLegacyTests.DenoiserCases().Where(c => (int)c[1] == MiniMaxH3ImageFixture.Q4_K)
            .Select(c => MiniMaxH3ImageGoldens.DenoiserCase((string)c[0], (int)c[1]));

    [Fact]
    public void ThePinnedBytesRebuildTheFixtureTheReferenceLaneFroze()
    {
        using var fx = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.Q4_K, pinnedQ4K: true);
        string sha = MiniMaxH3ImageGoldens.FileSha(fx.DitPath);
        foreach (string name in Q4KCases)
        {
            MiniMaxH3ImageGoldens.Case frozen = MiniMaxH3ImageGoldens.Find(name, ReferenceLane);
            Assert.True(frozen != null && frozen.Lane == ReferenceLane, $"{ReferenceLane} has no {name}");
            Assert.Equal(frozen.FixtureSha256, sha);
        }
    }

    [Fact]
    public void ThePinnedStoreRefusesValuesItWasNotQuantizedFrom()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.Q4_K, pinnedQ4K: true,
            edit: (name, values) => name == "blocks.0.mlp.fc1.weight" ? values.Select(v => v * 2f).ToArray() : null));
        Assert.Contains("blocks.0.mlp.fc1.weight", ex.Message);
    }

    /// <summary>Writes the store from this machine's quantizer, which must give the reference lane's fixture bytes (run it on
    /// that machine: <c>TS_TEST_PIN_H3IMAGE_Q4K=&lt;file&gt;</c>).</summary>
    [PinWriterFact]
    public void WritesThePinnedBytes()
    {
        string path = Environment.GetEnvironmentVariable(WriterEnv);
        using var live = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.Q4_K);
        string sha = MiniMaxH3ImageGoldens.FileSha(live.DitPath);
        foreach (string name in Q4KCases)
            Assert.Equal(MiniMaxH3ImageGoldens.Find(name, ReferenceLane).FixtureSha256, sha);
        MiniMaxH3ImagePinnedQ4K.Write(path, MiniMaxH3ImageFixture.Tensors(MiniMaxH3ImageFixture.Q4_K, 2051));
        output.WriteLine($"wrote {path} ({new FileInfo(path).Length} bytes) from the fixture {sha}");
    }

    private sealed class PinWriterFactAttribute : FactAttribute
    {
        public PinWriterFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(WriterEnv)))
                Skip = $"writes the pinned Q4_K store when {WriterEnv} names the file";
        }
    }
}
