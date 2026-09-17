// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Security.Cryptography;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Content identity of a media file (image, video frame, audio clip): the lowercase
    /// hex SHA-256 of its bytes.
    ///
    /// <para>
    /// Media used to be identified by its path, and every base64 image an API client
    /// sent was written under a fresh random name, so a client resending the same
    /// picture with each turn looked like a new picture every turn: no prompt reuse
    /// past it and a vision re-encode per request. Hashing the bytes makes identity
    /// independent of where and how often the file was stored. Results are memoized by
    /// (full path, length, last write time), so a file is read once until it changes.
    /// </para>
    /// </summary>
    public static class MediaContentId
    {
        private readonly record struct Version(long Length, long LastWriteUtcTicks, string Hash);

        private const int MaxMemoized = 4096;
        private static readonly ConcurrentDictionary<string, Version> s_memo = new(StringComparer.Ordinal);

        /// <summary>The SHA-256 of <paramref name="bytes"/> as lowercase hex.</summary>
        public static string OfBytes(ReadOnlySpan<byte> bytes)
            => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

        /// <summary>
        /// The content hash of the file at <paramref name="path"/>, or null when it cannot
        /// be read (missing, permissions): the caller then falls back to the path, which
        /// is what identity was before.
        /// </summary>
        public static string OfFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return null;
            string fullPath;
            try
            {
                fullPath = Path.GetFullPath(path);
                var info = new FileInfo(fullPath);
                if (!info.Exists)
                    return null;
                long length = info.Length;
                long ticks = info.LastWriteTimeUtc.Ticks;
                if (s_memo.TryGetValue(fullPath, out Version known)
                    && known.Length == length && known.LastWriteUtcTicks == ticks)
                    return known.Hash;

                string hash;
                using (var stream = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
                    hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();

                if (s_memo.Count >= MaxMemoized)
                    s_memo.Clear();   // a bounded memo, not a cache worth an LRU
                s_memo[fullPath] = new Version(length, ticks, hash);
                return hash;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                       or ArgumentException or NotSupportedException)
            {
                return null;
            }
        }
    }
}
