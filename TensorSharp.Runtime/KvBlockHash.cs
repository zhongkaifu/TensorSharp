// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// 16-byte (128-bit) opaque block identifier. We truncate SHA-256 to 128 bits
    /// to keep the key small while leaving collision probability astronomically low
    /// for any realistic cache size (birthday bound ≈ 2^64 blocks).
    /// </summary>
    public readonly struct KvBlockHash : IEquatable<KvBlockHash>
    {
        private readonly ulong _lo;
        private readonly ulong _hi;

        public KvBlockHash(ulong lo, ulong hi)
        {
            _lo = lo;
            _hi = hi;
        }

        public bool IsEmpty => _lo == 0 && _hi == 0;

        public static KvBlockHash FromBytes(ReadOnlySpan<byte> bytes)
        {
            if (bytes.Length < 16)
                throw new ArgumentException("Hash bytes must be at least 16 bytes long.", nameof(bytes));
            ulong lo = BinaryPrimitives.ReadUInt64LittleEndian(bytes);
            ulong hi = BinaryPrimitives.ReadUInt64LittleEndian(bytes[8..]);
            return new KvBlockHash(lo, hi);
        }

        public void WriteTo(Span<byte> destination)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(destination, _lo);
            BinaryPrimitives.WriteUInt64LittleEndian(destination[8..], _hi);
        }

        public string ToHexString()
        {
            Span<byte> buf = stackalloc byte[16];
            WriteTo(buf);
            return Convert.ToHexString(buf);
        }

        public bool Equals(KvBlockHash other) => _lo == other._lo && _hi == other._hi;
        public override bool Equals(object obj) => obj is KvBlockHash h && Equals(h);
        public override int GetHashCode() => HashCode.Combine(_lo, _hi);
        public override string ToString() => ToHexString();

        public static bool operator ==(KvBlockHash a, KvBlockHash b) => a.Equals(b);
        public static bool operator !=(KvBlockHash a, KvBlockHash b) => !a.Equals(b);
    }

    /// <summary>
    /// Builds per-block chain hashes. The chain links parent block content into the
    /// child's hash so equal tokens *only* match when their entire prefix matches,
    /// preventing accidental sharing of K/V state between sequences whose prompts
    /// happen to share suffixes but not prefixes.
    /// </summary>
    public static class KvBlockHasher
    {
        /// <summary>
        /// Compute hashes for the largest number of full blocks contained in
        /// <paramref name="tokens"/>. A trailing partial block is not hashed - paging
        /// only operates on whole blocks. The fingerprint pins the hashes to a
        /// specific (model, dtype, layer count) tuple so cache entries cannot leak
        /// across incompatible models loaded into the same store.
        /// </summary>
        public static List<KvBlockHash> ComputeBlockHashes(
            IReadOnlyList<int> tokens,
            int blockSize,
            string fingerprint)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            if (blockSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(blockSize));
            if (fingerprint == null)
                fingerprint = string.Empty;

            int fullBlocks = tokens.Count / blockSize;
            var hashes = new List<KvBlockHash>(fullBlocks);
            if (fullBlocks == 0)
                return hashes;

            byte[] fingerprintBytes = Encoding.UTF8.GetBytes(fingerprint);
            Span<byte> blockTokenBytes = stackalloc byte[Math.Min(blockSize, 4096) * sizeof(int)];
            byte[] rented = null;
            try
            {
                int needed = blockSize * sizeof(int);
                if (needed > blockTokenBytes.Length)
                {
                    rented = new byte[needed];
                    blockTokenBytes = rented;
                }
                blockTokenBytes = blockTokenBytes[..needed];

                Span<byte> parentDigest = stackalloc byte[32];
                Span<byte> childDigest = stackalloc byte[32];
                Span<byte> sentinel = stackalloc byte[32];

                for (int b = 0; b < fullBlocks; b++)
                {
                    int start = b * blockSize;
                    SerializeTokens(tokens, start, blockSize, blockTokenBytes);

                    using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                    sha.AppendData(fingerprintBytes);
                    // Chain the parent digest. For the first block the parent is the
                    // all-zero sentinel so the same prompt prefix always hashes the
                    // same way regardless of who hashed it.
                    sha.AppendData(b == 0 ? sentinel : parentDigest);
                    sha.AppendData(blockTokenBytes);

                    if (!sha.TryGetHashAndReset(childDigest, out int written) || written != 32)
                        throw new InvalidOperationException("SHA-256 digest produced unexpected length.");

                    hashes.Add(KvBlockHash.FromBytes(childDigest));
                    childDigest.CopyTo(parentDigest);
                }

                return hashes;
            }
            finally
            {
                if (rented != null)
                    Array.Clear(rented, 0, rented.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SerializeTokens(IReadOnlyList<int> tokens, int start, int count, Span<byte> destination)
        {
            for (int i = 0; i < count; i++)
                BinaryPrimitives.WriteInt32LittleEndian(destination[(i * sizeof(int))..], tokens[start + i]);
        }
    }
}
