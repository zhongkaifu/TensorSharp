// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>
/// Radix key elements (DESIGN §4.1, DEC-10). A text token is its id; a media position is
/// <c>bit63 | low 63 bits of the content digest</c>, so one sign test separates the two.
/// Model input ids never change: only the radix key carries media elements.
/// </summary>
public static class KeyElem
{
    public const long MediaBit = long.MinValue;                 // bit 63
    private const ulong Low63 = 0x7FFF_FFFF_FFFF_FFFFUL;

    public static long Text(int tokenId) =>
        tokenId >= 0 ? tokenId : throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, "Token ids must lie in [0, 2^31).");

    public static long Media(in MediaId256 id) => MediaBit | (long)(id.D & Low63);

    public static bool IsMedia(long e) => e < 0;
}

/// <summary>Full content identity of one media span (a SHA-256 digest as four big-endian words).</summary>
public readonly struct MediaId256 : IEquatable<MediaId256>
{
    public readonly ulong A, B, C, D;                           // big-endian SHA-256 words

    public MediaId256(ulong a, ulong b, ulong c, ulong d)
    {
        A = a; B = b; C = c; D = d;
    }

    /// <summary>
    /// 64 lowercase hex chars → the digest. Anything else (Phase 0's path fallback) →
    /// SHA-256(UTF-8("path:" + contentId)). Never throws; null or empty → SHA-256("path:").
    /// </summary>
    public static MediaId256 FromContentId(string? contentId)
    {
        if (contentId is { Length: 64 } && IsLowerHex(contentId))
        {
            return new MediaId256(
                ulong.Parse(contentId.AsSpan(0, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture),
                ulong.Parse(contentId.AsSpan(16, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture),
                ulong.Parse(contentId.AsSpan(32, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture),
                ulong.Parse(contentId.AsSpan(48, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture));
        }
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes("path:" + (contentId ?? string.Empty)));
        return FromDigest(digest);
    }

    /// <summary>Builds an id from a 32-byte digest (big-endian words).</summary>
    public static MediaId256 FromDigest(ReadOnlySpan<byte> digest)
    {
        if (digest.Length != 32) throw new ArgumentException("A SHA-256 digest is 32 bytes.", nameof(digest));
        return new MediaId256(
            System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(digest.Slice(0, 8)),
            System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(digest.Slice(8, 8)),
            System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(digest.Slice(16, 8)),
            System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(digest.Slice(24, 8)));
    }

    private static bool IsLowerHex(string s)
    {
        foreach (char ch in s)
        {
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
                return false;
        }
        return true;
    }

    public bool Equals(MediaId256 o) => A == o.A && B == o.B && C == o.C && D == o.D;

    public override bool Equals(object? obj) => obj is MediaId256 o && Equals(o);

    public override int GetHashCode() => HashCode.Combine(A, B, C, D);

    public static bool operator ==(MediaId256 left, MediaId256 right) => left.Equals(right);

    public static bool operator !=(MediaId256 left, MediaId256 right) => !left.Equals(right);

    public override string ToString() => $"{A:x16}{B:x16}{C:x16}{D:x16}";
}

/// <summary>One media span of a prompt or of a node edge, in absolute token positions.</summary>
public readonly record struct MediaSpanRecord(int Start, int End, MediaId256 Id)
{
    /// <summary>Convenience: a record from Phase 0's content id string.</summary>
    public static MediaSpanRecord FromContentId(int start, int end, string? contentId)
        => new(start, end, MediaId256.FromContentId(contentId));
}

/// <summary>8,192 × 8 B = 65,536 B, below the 85 KB large-object-heap threshold (DEC-29).</summary>
internal sealed class KeyChunk
{
    internal const int Size = 8192, Shift = 13, Mask = Size - 1;
    internal readonly long[] Data = new long[Size];
    internal int RefCount;                                      // ropes referencing this chunk; worker-thread only
}

/// <summary>Per-engine bounded free list of key chunks (256 chunks).</summary>
internal sealed class KeyChunkPool
{
    internal const int DefaultMaxFree = 256;
    private readonly KeyChunk[] _free;
    private int _freeCount;

    internal KeyChunkPool(int maxFree = DefaultMaxFree)
    {
        _free = new KeyChunk[Math.Max(0, maxFree)];
    }

    internal long Rented { get; private set; }
    internal long Allocated { get; private set; }
    internal int FreeCount => _freeCount;

    internal KeyChunk Rent()
    {
        Rented++;
        if (_freeCount > 0)
        {
            KeyChunk c = _free[--_freeCount];
            _free[_freeCount] = null!;
            c.RefCount = 0;
            return c;
        }
        Allocated++;
        return new KeyChunk();
    }

    internal void Return(KeyChunk c)
    {
        if (c == null) return;
        if (_freeCount < _free.Length)
            _free[_freeCount++] = c;
    }
}

/// <summary>
/// An append-only key sequence owned by one request, shared by the node edges sliced from it
/// (DESIGN §4.1). Elements below <see cref="Length"/> are immutable once written.
/// </summary>
internal sealed class KeyRope
{
    private static readonly KeyChunk[] s_noChunks = Array.Empty<KeyChunk>();

    internal KeyChunk[] Chunks = s_noChunks;
    internal int ChunkCount;
    internal int Length;
    internal int SliceRefs;                                     // node edges referencing this rope
    internal int LiveSliceTokens;                               // Σ edge lengths referencing it (compaction trigger, DEC-29)
    internal bool OwnerReleased;                                // the request that built it is gone
    internal bool Disposed;                                     // chunks returned to the pool
    internal RadixNode? FirstSliceNode;                         // intrusive list of nodes whose Edge uses this rope

    internal long this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Chunks[i >> KeyChunk.Shift].Data[i & KeyChunk.Mask];
    }

    /// <summary>Contiguous elements starting at <paramref name="start"/>, up to the end of that chunk.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ReadOnlySpan<long> Segment(int start, int maxLen)
    {
        int off = start & KeyChunk.Mask;
        int n = Math.Min(Math.Min(maxLen, KeyChunk.Size - off), Length - start);
        if (n <= 0) return ReadOnlySpan<long>.Empty;
        return new ReadOnlySpan<long>(Chunks[start >> KeyChunk.Shift].Data, off, n);
    }

    /// <summary>Appends <paramref name="keys"/> at <see cref="Length"/>.</summary>
    internal void Append(ReadOnlySpan<long> keys, KeyChunkPool pool)
    {
        if (Disposed) throw new ObjectDisposedException(nameof(KeyRope));
        int written = 0;
        while (written < keys.Length)
        {
            int ci = Length >> KeyChunk.Shift;
            if (ci >= ChunkCount)
                AddChunk(pool);
            int off = Length & KeyChunk.Mask;
            int n = Math.Min(KeyChunk.Size - off, keys.Length - written);
            keys.Slice(written, n).CopyTo(new Span<long>(Chunks[ci].Data, off, n));
            written += n;
            Length += n;
        }
    }

    /// <summary>Appends one element at <see cref="Length"/>.</summary>
    internal void Append(long key, KeyChunkPool pool)
    {
        if (Disposed) throw new ObjectDisposedException(nameof(KeyRope));
        int ci = Length >> KeyChunk.Shift;
        if (ci >= ChunkCount)
            AddChunk(pool);
        Chunks[ci].Data[Length & KeyChunk.Mask] = key;
        Length++;
    }

    private void AddChunk(KeyChunkPool pool)
    {
        if (ChunkCount == Chunks.Length)
            Array.Resize(ref Chunks, Math.Max(4, Chunks.Length * 2));
        KeyChunk c = pool.Rent();
        c.RefCount++;
        Chunks[ChunkCount++] = c;
    }

    /// <summary>Returns every chunk to the pool. Only legal when no slice references the rope.</summary>
    internal void ReturnChunks(KeyChunkPool pool)
    {
        if (Disposed) return;
        for (int i = 0; i < ChunkCount; i++)
        {
            KeyChunk c = Chunks[i];
            if (--c.RefCount == 0)
                pool.Return(c);
            Chunks[i] = null!;
        }
        ChunkCount = 0;
        Length = 0;
        Disposed = true;
    }

    internal static KeyRope FromKeys(ReadOnlySpan<long> keys, KeyChunkPool pool)
    {
        var rope = new KeyRope();
        rope.Append(keys, pool);
        return rope;
    }
}

/// <summary>A node edge: <c>[Start, Start + Length)</c> of a rope.</summary>
internal readonly struct KeySlice
{
    internal readonly KeyRope Rope;
    internal readonly int Start;
    internal readonly int Length;

    internal KeySlice(KeyRope rope, int start, int length)
    {
        Rope = rope; Start = start; Length = length;
    }

    internal long this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Rope[Start + i];
    }

    internal bool IsEmpty => Rope is null || Length == 0;
}

/// <summary>Builds radix keys from prompt tokens and media spans (DESIGN §4.1, §5.1).</summary>
internal static class RadixKeyBuilder
{
    /// <summary>
    /// Builds the prompt rope: token ids, then every position of every span overwritten with
    /// <see cref="KeyElem.Media"/> of the span id. Throws <see cref="ArgumentException"/>
    /// (surfaced as a submit failure, never a silent miss) if a token id is negative, or spans
    /// are empty, unsorted, overlapping or outside the prompt.
    /// </summary>
    /// <remarks>
    /// M1 deviation: Phase 0's <c>PromptMediaSpan</c> is not on the base commit, so spans are
    /// passed as (Start, End, ContentId) triples. M3 adds the <c>PromptMediaSpan</c> overload.
    /// </remarks>
    internal static KeyRope BuildPrompt(IReadOnlyList<int> promptTokens,
                                        IReadOnlyList<(int Start, int End, string ContentId)>? spans,
                                        KeyChunkPool pool, out MediaSpanRecord[] records)
    {
        if (promptTokens is null) throw new ArgumentNullException(nameof(promptTokens));
        int spanCount = spans?.Count ?? 0;
        records = spanCount == 0 ? Array.Empty<MediaSpanRecord>() : new MediaSpanRecord[spanCount];
        for (int i = 0; i < spanCount; i++)
        {
            (int start, int end, string contentId) = spans![i];
            records[i] = new MediaSpanRecord(start, end, MediaId256.FromContentId(contentId));
        }
        ValidateSpans(records, promptTokens.Count);
        return BuildPrompt(promptTokens, records, pool);
    }

    /// <summary>Builds the prompt rope from already-parsed span records.</summary>
    internal static KeyRope BuildPrompt(IReadOnlyList<int> promptTokens, MediaSpanRecord[] records, KeyChunkPool pool)
    {
        if (promptTokens is null) throw new ArgumentNullException(nameof(promptTokens));
        records ??= Array.Empty<MediaSpanRecord>();
        ValidateSpans(records, promptTokens.Count);
        var rope = new KeyRope();
        Span<long> buffer = stackalloc long[512];
        int n = promptTokens.Count;
        int spanIx = 0;
        int pos = 0;
        while (pos < n)
        {
            int count = Math.Min(buffer.Length, n - pos);
            for (int i = 0; i < count; i++)
            {
                int p = pos + i;
                while (spanIx < records.Length && records[spanIx].End <= p) spanIx++;
                if (spanIx < records.Length && records[spanIx].Start <= p)
                {
                    buffer[i] = KeyElem.Media(records[spanIx].Id);
                    continue;
                }
                int token = promptTokens[p];
                if (token < 0)
                    throw new ArgumentException($"Prompt token at position {p} has a negative id ({token}); ids must lie in [0, 2^31).", nameof(promptTokens));
                buffer[i] = token;
            }
            rope.Append(buffer.Slice(0, count), pool);
            pos += count;
        }
        return rope;
    }

    /// <summary>Throws when spans are empty, unsorted, overlapping or outside [0, promptLength].</summary>
    internal static void ValidateSpans(ReadOnlySpan<MediaSpanRecord> records, int promptLength)
    {
        int prevEnd = 0;
        for (int i = 0; i < records.Length; i++)
        {
            MediaSpanRecord s = records[i];
            if (s.Start < 0 || s.End > promptLength)
                throw new ArgumentException($"Media span {i} [{s.Start}, {s.End}) lies outside the prompt of {promptLength} tokens.");
            if (s.End <= s.Start)
                throw new ArgumentException($"Media span {i} [{s.Start}, {s.End}) is empty.");
            if (s.Start < prevEnd)
                throw new ArgumentException($"Media span {i} [{s.Start}, {s.End}) is unsorted or overlaps the previous span (ends at {prevEnd}).");
            prevEnd = s.End;
        }
    }

    /// <summary>
    /// Appends output tokens at absolute positions <c>[from, from + tokens.Length)</c> (never media),
    /// for insertion at finish or preemption. Positions below <c>rope.Length</c> are already present
    /// (and immutable), so only the part past it is written.
    /// </summary>
    /// <remarks>M1 deviation: takes the token span rather than a <c>SequenceState</c> (no product references).</remarks>
    internal static void AppendOutput(KeyRope rope, int from, ReadOnlySpan<int> tokens, KeyChunkPool pool)
    {
        if (rope is null) throw new ArgumentNullException(nameof(rope));
        if (from < 0 || from > rope.Length)
            throw new ArgumentOutOfRangeException(nameof(from), from, $"Output must continue the key (length {rope.Length}).");
        int skip = rope.Length - from;
        for (int i = skip; i < tokens.Length; i++)
        {
            int token = tokens[i];
            if (token < 0)
                throw new ArgumentException($"Output token at position {from + i} has a negative id ({token}).", nameof(tokens));
            rope.Append(token, pool);
        }
    }
}

/// <summary>Vectorized common-prefix comparisons across chunk segments.</summary>
internal static class KeyCompare
{
    /// <summary>
    /// Length of the common prefix of <c>edge[edgeOffset..]</c> and <c>key[keyStart..]</c>, at most
    /// <paramref name="maxLen"/> (and at most the remaining edge and key lengths).
    /// Uses <see cref="MemoryExtensions.CommonPrefixLength{T}(ReadOnlySpan{T}, ReadOnlySpan{T})"/>.
    /// </summary>
    internal static int CommonPrefixLength(in KeySlice edge, int edgeOffset, KeyRope key, int keyStart, int maxLen)
    {
        int limit = Math.Min(maxLen, Math.Min(edge.Length - edgeOffset, key.Length - keyStart));
        if (limit <= 0) return 0;
        KeyRope a = edge.Rope;
        int ai = edge.Start + edgeOffset;
        int bi = keyStart;
        int matched = 0;
        while (matched < limit)
        {
            ReadOnlySpan<long> sa = a.Segment(ai + matched, limit - matched);
            ReadOnlySpan<long> sb = key.Segment(bi + matched, sa.Length);
            if (sb.Length < sa.Length) sa = sa.Slice(0, sb.Length);
            int c = sa.CommonPrefixLength(sb);
            matched += c;
            if (c < sa.Length) break;
        }
        return matched;
    }

    /// <summary>Common prefix of two ropes over <c>[0, maxLen)</c>.</summary>
    internal static int CommonPrefixLength(KeyRope a, KeyRope b, int maxLen)
    {
        int limit = Math.Min(maxLen, Math.Min(a.Length, b.Length));
        return limit <= 0 ? 0 : CommonPrefixLength(new KeySlice(a, 0, a.Length), 0, b, 0, limit);
    }
}
