// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests.PrefixCache.Fakes;

/// <summary>
/// The oracle fakes' state function (DESIGN §12.3). A cache's state after token <c>i</c> is a
/// 64-bit hash folded over every (token, position + rope delta) pair before it, and the logits
/// peak at a token derived from that hash. Any wrong reuse — another conversation's tail, a
/// forbidden truncation, a wrong position, a stale holder, a zero slab, a partial inject —
/// therefore changes the greedy token stream, and a test that compares against a cold run sees it.
/// </summary>
internal static class OracleHash
{
    internal const ulong Seed = 0x6A09E667F3BCC909UL;
    /// <summary>XORed into a state an operation could not restore faithfully (a wrapped-ring rewind,
    /// a recurrent page whose end is not a forward boundary).</summary>
    internal const ulong Poison = 0xDEADBEEFCAFEF00DUL;

    internal static ulong Mix(ulong previous, int token, int position)
    {
        ulong x = previous
            ^ ((ulong)(uint)token * 0x9E3779B97F4A7C15UL)
            ^ (((ulong)(uint)position << 32 | (uint)position) * 0xBF58476D1CE4E5B9UL);
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    internal static int PeakToken(ulong state, int vocab) => (int)(state % (ulong)vocab);

    internal static float[] Logits(ulong state, int vocab)
    {
        var logits = new float[vocab];
        logits[PeakToken(state, vocab)] = 10.0f;
        // A second, weaker peak so a sampler that is not argmax still has structure to read.
        logits[(int)((state >> 29) % (ulong)vocab)] += 1.0f;
        return logits;
    }
}

/// <summary>One cache of an oracle fake: the primary, a private holder, a retained payload or a native slot.</summary>
internal sealed class OracleCache
{
    /// <summary><c>Chain[i]</c> is the state after <c>i</c> tokens; <c>Chain[0]</c> is the seed.</summary>
    internal readonly List<ulong> Chain = new() { OracleHash.Seed };
    /// <summary>Positions at which a forward ended (recurrent state is restorable only there).</summary>
    internal readonly HashSet<int> ForwardBoundaries = new() { 0 };

    internal int Length => Chain.Count - 1;
    /// <summary>Length whose state the host copy holds; below <see cref="Length"/> the device is authoritative.</summary>
    internal int HostLength;
    internal int RopeDelta;
    /// <summary>The cache was bound as an active cache at least once (it has device mirrors).</summary>
    internal bool WasBound;
    internal bool Retired;
    internal long Serial;

    internal bool DeviceDirty => HostLength < Length;

    internal ulong State => Chain[^1];

    internal void Append(int token)
    {
        int position = Length;
        Chain.Add(OracleHash.Mix(Chain[^1], token, position + RopeDelta));
    }

    internal void Flush() => HostLength = Length;

    /// <summary>Keeps tokens <c>[0, n)</c>; <paramref name="corrupt"/> models a rewind that cannot restore the state.</summary>
    internal void TruncateTo(int n, bool corrupt)
    {
        if (n < 0 || n > Length) throw new ArgumentOutOfRangeException(nameof(n));
        Chain.RemoveRange(n + 1, Chain.Count - n - 1);
        ForwardBoundaries.RemoveWhere(b => b > n);
        ForwardBoundaries.Add(n);
        if (corrupt) Chain[n] ^= OracleHash.Poison;
        HostLength = Length;
    }

    /// <summary>An independent copy of the host-authoritative state. A dirty source copies stale
    /// host bytes, exactly what a real family's copy of an unsettled holder would read.</summary>
    internal OracleCache CopyHost()
    {
        var copy = new OracleCache { RopeDelta = RopeDelta };
        copy.Chain.Clear();
        for (int i = 0; i <= Length; i++)
            copy.Chain.Add(i <= HostLength ? Chain[i] : Chain[i] ^ OracleHash.Poison);
        copy.ForwardBoundaries.Clear();
        foreach (int b in ForwardBoundaries) copy.ForwardBoundaries.Add(b);
        copy.HostLength = copy.Length;
        return copy;
    }

    internal int CapacityTokens => Math.Max(256, (Length + 255) / 256 * 256);
}

/// <summary>A tokenizer whose vocabulary is the token ids themselves.</summary>
internal sealed class OracleTokenizer : ITokenizer
{
    internal OracleTokenizer(int vocabSize, int eosToken = -1)
    {
        Vocab = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++) Vocab[i] = "t" + i;
        EosTokenIds = eosToken >= 0 ? new[] { eosToken } : Array.Empty<int>();
    }

    public string[] Vocab { get; }
    public int BosTokenId => -1;
    public int[] EosTokenIds { get; }
    public int VocabSize => Vocab.Length;
    public List<int> Encode(string text, bool addSpecial = true) => new();
    public string Decode(List<int> ids) => string.Join(",", ids);
    public void AppendTokenBytes(int tokenId, List<byte> buffer)
    {
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes("t" + tokenId)) buffer.Add(b);
    }
    public bool IsEos(int tokenId) => Array.IndexOf(EosTokenIds, tokenId) >= 0;
    public int LookupToken(string tokenStr) => -1;
}
