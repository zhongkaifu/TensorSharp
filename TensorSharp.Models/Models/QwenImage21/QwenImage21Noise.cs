// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Security.Cryptography;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>
    /// Qwen-Image-2.1's noise: Philox4x32-10 and Box-Muller as stable-diffusion.cpp's <c>--rng cuda</c> draws it, keyed by the
    /// seed, on a stream that an edit takes from its reference pictures (<see cref="ReferenceStream"/>). A stream continues
    /// across draws: value i of the stream is drawn from Philox counter (0, stream low, i, stream high), so the initial latent
    /// (the first draw, in channel-planar order) is sd.cpp's sequence bit for bit on stream 0.
    /// </summary>
    public sealed class QwenImage21Noise : INoiseSource
    {
        /// <summary><c>references</c> (the default): an edit's noise also depends on its reference pictures. <c>seed</c>: an edit
        /// starts from the seed's text-to-image noise, as stable-diffusion.cpp and diffusers do, for matched-noise A/B runs.</summary>
        public const string EditNoiseVariable = "TS_QWEN21_EDIT_NOISE";

        private readonly long _seed;
        private readonly ulong _stream;
        private long _next;

        public QwenImage21Noise(long seed, ulong stream)
        {
            _seed = seed;
            _stream = stream;
        }

        public ulong Stream => _stream;

        public float[] Next(long count)
        {
            if (count < 0 || _next + count > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(count));
            float[] values = PhiloxNoise.Draw(checked((int)count), _seed, _stream, 0, (uint)_next);
            _next += count;
            return values;
        }

        /// <summary>Whether an edit's noise follows its references; anything other than <c>references</c> (or unset) and
        /// <c>seed</c> is refused.</summary>
        public static bool EditNoiseFollowsReferences(string value) =>
            (value ?? "").Trim().ToLowerInvariant() switch
            {
                "" or "references" => true,
                "seed" => false,
                _ => throw new ArgumentException($"{EditNoiseVariable}='{value}' is not one of references, seed."),
            };

        /// <summary>The Philox stream of a request's noise: 0 for text to image (and for an edit when
        /// <paramref name="followReferences"/> is false), otherwise <see cref="ReferenceStream"/> of the pictures as the caller
        /// passed them.</summary>
        public static ulong StreamFor(IReadOnlyList<RgbImage> callerInputs, bool followReferences)
        {
            ArgumentNullException.ThrowIfNull(callerInputs);
            return callerInputs.Count == 0 || !followReferences ? 0 : ReferenceStream(callerInputs);
        }

        /// <summary>
        /// A nonzero stream id for an edit, keyed to its reference pictures. Without it, the noise depends on the seed and size
        /// alone, so an edit of a picture at the seed and size it was drawn at starts from the very latent that became that
        /// picture, and Qwen-Image 2.1 can retrace the picture instead of following the instruction.
        /// <para>SHA-256 over a versioned encoding: the domain string, the reference count, then per reference its width, height,
        /// an alpha flag, its RGB quantized to 8 bits as a PNG stores it (<see cref="Quantize8"/>), and the quantized alpha when
        /// any pixel is not opaque. All-opaque alpha hashes as no alpha. The first eight digest bytes, little-endian, are the
        /// id; 0 (text to image) maps to 1.</para>
        /// </summary>
        public static ulong ReferenceStream(IReadOnlyList<RgbImage> references)
        {
            ArgumentNullException.ThrowIfNull(references);
            if (references.Count == 0) return 0;
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            hash.AppendData("TensorSharp/qwen-image-2.1/edit-noise/v1"u8);
            Span<byte> header = stackalloc byte[9];
            BinaryPrimitives.WriteInt32LittleEndian(header, references.Count);
            hash.AppendData(header[..4]);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
            try
            {
                foreach (var image in references)
                {
                    ArgumentNullException.ThrowIfNull(image);
                    bool translucent = false;
                    ReadOnlySpan<float> alphas = image.Alpha;
                    for (int i = 0; i < alphas.Length && !translucent; i++) translucent = Quantize8(alphas[i]) != 255;
                    BinaryPrimitives.WriteInt32LittleEndian(header, image.Width);
                    BinaryPrimitives.WriteInt32LittleEndian(header[4..], image.Height);
                    header[8] = translucent ? (byte)1 : (byte)0;
                    hash.AppendData(header);
                    AppendQuantized(hash, image.Pixels, buffer);
                    if (translucent) AppendQuantized(hash, image.Alpha, buffer);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
            Span<byte> digest = stackalloc byte[32];
            hash.GetHashAndReset(digest);
            ulong stream = BinaryPrimitives.ReadUInt64LittleEndian(digest);
            return stream == 0 ? 1 : stream;
        }

        // One plain loop per chunk: the Mono hosts run vector helpers slower than this.
        private static void AppendQuantized(IncrementalHash hash, float[] values, byte[] buffer)
        {
            for (int start = 0; start < values.Length; start += buffer.Length)
            {
                ReadOnlySpan<float> source = values.AsSpan(start, Math.Min(buffer.Length, values.Length - start));
                Span<byte> bytes = buffer.AsSpan(0, source.Length);
                for (int i = 0; i < source.Length; i++) bytes[i] = Quantize8(source[i]);
                hash.AppendData(bytes);
            }
        }

        /// <summary>ImageIO's float-to-byte rule (round half up, clamped), with NaN as 0 on every runtime.</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static byte Quantize8(float value)
        {
            float scaled = value * 255f + 0.5f;
            return scaled >= 0f ? scaled <= 255f ? (byte)scaled : (byte)255 : (byte)0;
        }
    }
}
