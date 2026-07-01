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
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Group-wise int2 / int4 / int8 quantization codec for the paged KV tier,
    /// inspired by the TurboQuant-KV scheme used in upstream oMLX. Splits the
    /// captured payload into fixed-size groups (32 elements) and stores each
    /// element as a low-bit integer with a per-group scale.
    ///
    /// Two quantization schemes are used, selected by bit width:
    ///   * <b>int4 / int8 — symmetric.</b> A single fp16 scale per group;
    ///     reconstruction is <c>x ≈ q * scale</c> with
    ///     <c>q ∈ [-2^(N-1), 2^(N-1)-1]</c>. At 4/8 bits the symmetric range is
    ///     accurate, and keeping this layout byte-stable means blocks already
    ///     written to the SSD tier stay decodable.
    ///   * <b>int2 — affine (min + scale).</b> With only four codes
    ///     <c>q ∈ {0,1,2,3}</c> a symmetric mapping wastes one level and leaves
    ///     ~50% worst-case error, so the 2-bit path stores a per-group fp16
    ///     <c>min</c> alongside the fp16 <c>scale = (max - min) / 3</c> and
    ///     reconstructs <c>x ≈ min + q * scale</c>. Spreading the four codes
    ///     across the group's actual range roughly halves the error versus a
    ///     symmetric 2-bit mapping — the same trick llama.cpp's Q2_K / Q4_1
    ///     block formats use (block min + scale) for their lowest-bit tiers.
    ///
    /// Why aggressive quantization is safe for paged-tier blocks: the live
    /// (in-model) KV cache stays in its native dtype (F32 / F16 / Q8_0). The
    /// codec only kicks in when a block has already been evicted from the model
    /// and copied into the paged store; the small per-group quantization error
    /// therefore only affects the reuse path - a fresh prefill produces
    /// bit-identical output - and prefix blocks attended at large distances see
    /// attention weights orders of magnitude smaller than the leading tokens, so
    /// even int2 noise is overwhelmed by the softmax on the far prefix that the
    /// paged tier is built to recycle.
    ///
    /// Q8_0 payloads are passed through untouched: re-quantizing already-quantized
    /// bytes would compound error without shrinking the block much, and the
    /// underlying ggml layout's per-32-block fp16 scale already lives in the
    /// stream.
    /// </summary>
    public sealed class TurboQuantKvCodec : IKvBlockCodec
    {
        // "TQKV" little-endian. Increment the version byte if the on-disk
        // layout below changes so old SSD blocks are rejected instead of
        // silently misdecoded.
        private const uint Magic = 0x564B5154u;
        private const int HeaderBytes = 16;
        private const int GroupSize = 32;
        private const byte FormatVersion = 1;

        private readonly KvCodecElementType _elementType;
        private readonly int _bits;
        private readonly int _qMin;
        private readonly int _qMax;
        // 2-bit uses an affine (min + scale) layout; 4/8-bit use the symmetric
        // (scale only) layout. Cached here so the hot path branches on a bool
        // instead of re-deriving from the bit count.
        private readonly bool _affine;

        /// <summary>
        /// Construct a codec that interprets raw payload bytes as elements of
        /// <paramref name="elementType"/> and quantizes them to
        /// <paramref name="bits"/> bits per element (must be 2, 4, or 8). 2-bit
        /// uses an affine min+scale layout; 4/8-bit use a symmetric scale-only
        /// layout. For <see cref="KvCodecElementType.Q8_0"/> the bit count is
        /// ignored and every encode is a passthrough.
        /// </summary>
        public TurboQuantKvCodec(KvCodecElementType elementType, int bits)
        {
            if (bits != 2 && bits != 4 && bits != 8)
                throw new ArgumentOutOfRangeException(nameof(bits), "TurboQuant supports 2, 4, or 8 bits per element.");
            _elementType = elementType;
            _bits = bits;
            _affine = bits == 2;
            // Symmetric range (used by the 4/8-bit path). The affine 2-bit path
            // maps to the unsigned code range [0, 3] instead and ignores these.
            _qMax = (1 << (bits - 1)) - 1;
            _qMin = -(1 << (bits - 1));
        }

        public string Name => $"turboquant-int{_bits}";
        public int BitsPerElement => _bits;
        public KvCodecElementType ElementType => _elementType;

        public byte[] Encode(ReadOnlySpan<byte> rawBlock)
        {
            if (rawBlock.IsEmpty)
                return Array.Empty<byte>();

            // Q8_0 already lives in the same per-32-block layout TurboQuant
            // would re-derive; re-quantizing would just compound rounding
            // error. Keep the original bytes and tag them with the codec
            // header so the decoder knows to pass them through.
            if (_elementType == KvCodecElementType.Q8_0)
                return PackPassthrough(rawBlock);

            int elementBytes = ElementBytes(_elementType);
            if (rawBlock.Length % elementBytes != 0)
                return PackPassthrough(rawBlock);

            long elementCount = rawBlock.Length / elementBytes;
            if (elementCount % GroupSize != 0)
            {
                // The paged tier always serializes the same fixed-shape
                // per-block payload, which for every model currently in
                // the repo divides cleanly into 32-element groups. Bail
                // gracefully if a future model lands with an odd shape.
                return PackPassthrough(rawBlock);
            }

            long groupCount = elementCount / GroupSize;
            int perGroupBytes = PerGroupBytes(_bits);
            long encodedBytes = HeaderBytes + groupCount * perGroupBytes;
            if (encodedBytes > int.MaxValue)
                return PackPassthrough(rawBlock);

            byte[] result = new byte[encodedBytes];
            WriteHeader(result, _bits, elementCount, _elementType);

            int offset = HeaderBytes;
            Span<float> scratch = stackalloc float[GroupSize];

            for (long g = 0; g < groupCount; g++)
            {
                long elemStart = g * GroupSize;
                LoadGroup(rawBlock, elemStart, scratch);

                if (_affine)
                {
                    // 2-bit affine: spread codes {0,1,2,3} across [min, max].
                    float min = scratch[0], max = scratch[0];
                    for (int i = 1; i < GroupSize; i++)
                    {
                        float v = scratch[i];
                        if (v < min) min = v;
                        else if (v > max) max = v;
                    }

                    float aScale = max > min ? (max - min) / 3f : 0f;
                    float aInvScale = aScale > 0f ? 1f / aScale : 0f;

                    // Per-group scale then min, both fp16.
                    BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(offset, 2),
                        System.BitConverter.HalfToUInt16Bits((System.Half)aScale));
                    offset += 2;
                    BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(offset, 2),
                        System.BitConverter.HalfToUInt16Bits((System.Half)min));
                    offset += 2;

                    // Re-snap min through fp16 so the encoder quantizes against
                    // the exact value the decoder will reconstruct from.
                    float minQ = (float)System.BitConverter.UInt16BitsToHalf(
                        System.BitConverter.HalfToUInt16Bits((System.Half)min));

                    // Pack four 2-bit codes per byte, element 0 in the low bits.
                    for (int i = 0; i < GroupSize; i += 4)
                    {
                        int q0 = QuantizeAffine(scratch[i], minQ, aInvScale);
                        int q1 = QuantizeAffine(scratch[i + 1], minQ, aInvScale);
                        int q2 = QuantizeAffine(scratch[i + 2], minQ, aInvScale);
                        int q3 = QuantizeAffine(scratch[i + 3], minQ, aInvScale);
                        result[offset++] = (byte)(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
                    }

                    continue;
                }

                float maxAbs = 0f;
                for (int i = 0; i < GroupSize; i++)
                {
                    float v = scratch[i];
                    float a = v < 0 ? -v : v;
                    if (a > maxAbs) maxAbs = a;
                }

                float scale = maxAbs > 0f ? maxAbs / _qMax : 0f;
                float invScale = scale > 0f ? 1f / scale : 0f;

                // Per-group scale stored as fp16.
                ushort scaleBits = System.BitConverter.HalfToUInt16Bits((System.Half)scale);
                BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(offset, 2), scaleBits);
                offset += 2;

                if (_bits == 4)
                {
                    // Pack two 4-bit signed values per byte. Low nibble = even
                    // index, high nibble = odd. Stored unsigned-biased so the
                    // top bit of the nibble encodes sign; we offset by +8 on
                    // write and subtract on read.
                    for (int i = 0; i < GroupSize; i += 2)
                    {
                        int q0 = QuantizeOne(scratch[i] * invScale);
                        int q1 = QuantizeOne(scratch[i + 1] * invScale);
                        byte b = (byte)(((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4));
                        result[offset++] = b;
                    }
                }
                else // _bits == 8
                {
                    for (int i = 0; i < GroupSize; i++)
                    {
                        int q = QuantizeOne(scratch[i] * invScale);
                        result[offset++] = (byte)(sbyte)q;
                    }
                }
            }

            return result;
        }

        public bool TryDecode(ReadOnlySpan<byte> encoded, Span<byte> rawDestination)
        {
            if (encoded.Length < HeaderBytes || rawDestination.IsEmpty)
                return false;

            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(encoded);
            if (magic != Magic)
                return false;
            byte version = encoded[4];
            if (version != FormatVersion)
                return false;
            byte bits = encoded[5];
            byte dtypeByte = encoded[6];
            // encoded[7] reserved.
            ulong elementCount = BinaryPrimitives.ReadUInt64LittleEndian(encoded[8..]);

            if (dtypeByte > (byte)KvCodecElementType.Q8_0)
                return false;
            var dtype = (KvCodecElementType)dtypeByte;
            if (dtype != _elementType)
                return false;

            // Passthrough block: header followed by raw bytes.
            if (bits == 0)
            {
                int payloadLength = encoded.Length - HeaderBytes;
                if (payloadLength != rawDestination.Length)
                    return false;
                encoded.Slice(HeaderBytes, payloadLength).CopyTo(rawDestination);
                return true;
            }

            if (bits != 2 && bits != 4 && bits != 8)
                return false;
            if (bits != _bits)
                return false;
            if (dtype == KvCodecElementType.Q8_0)
                return false; // Q8_0 should never carry a non-zero bit count

            int elementBytes = ElementBytes(dtype);
            if (rawDestination.Length != (long)elementCount * elementBytes)
                return false;
            if (elementCount % GroupSize != 0)
                return false;

            long groupCount = (long)elementCount / GroupSize;
            int perGroupBytes = PerGroupBytes(bits);
            long expectedEncoded = HeaderBytes + groupCount * perGroupBytes;
            if (encoded.Length != expectedEncoded)
                return false;

            int offset = HeaderBytes;
            Span<float> scratch = stackalloc float[GroupSize];

            for (long g = 0; g < groupCount; g++)
            {
                if (_affine)
                {
                    // 2-bit affine: [fp16 scale][fp16 min][8 packed bytes].
                    float aScale = (float)System.BitConverter.UInt16BitsToHalf(
                        BinaryPrimitives.ReadUInt16LittleEndian(encoded[offset..]));
                    offset += 2;
                    float aMin = (float)System.BitConverter.UInt16BitsToHalf(
                        BinaryPrimitives.ReadUInt16LittleEndian(encoded[offset..]));
                    offset += 2;

                    for (int i = 0; i < GroupSize; i += 4)
                    {
                        byte packed = encoded[offset++];
                        scratch[i] = aMin + (packed & 0x03) * aScale;
                        scratch[i + 1] = aMin + ((packed >> 2) & 0x03) * aScale;
                        scratch[i + 2] = aMin + ((packed >> 4) & 0x03) * aScale;
                        scratch[i + 3] = aMin + ((packed >> 6) & 0x03) * aScale;
                    }

                    StoreGroup(scratch, g * GroupSize, rawDestination);
                    continue;
                }

                ushort scaleBits = BinaryPrimitives.ReadUInt16LittleEndian(encoded[offset..]);
                offset += 2;
                float scale = (float)System.BitConverter.UInt16BitsToHalf(scaleBits);

                if (bits == 4)
                {
                    for (int i = 0; i < GroupSize; i += 2)
                    {
                        byte packed = encoded[offset++];
                        int q0 = (packed & 0x0F) - 8;
                        int q1 = ((packed >> 4) & 0x0F) - 8;
                        scratch[i] = q0 * scale;
                        scratch[i + 1] = q1 * scale;
                    }
                }
                else
                {
                    for (int i = 0; i < GroupSize; i++)
                    {
                        sbyte q = unchecked((sbyte)encoded[offset++]);
                        scratch[i] = q * scale;
                    }
                }

                StoreGroup(scratch, g * GroupSize, rawDestination);
            }

            return true;
        }

        private byte[] PackPassthrough(ReadOnlySpan<byte> rawBlock)
        {
            int length = rawBlock.Length;
            byte[] result = new byte[HeaderBytes + length];
            // bits=0 signals "no quantization, raw payload follows".
            long elementBytes = ElementBytes(_elementType);
            long elementCount = elementBytes > 0 ? length / elementBytes : length;
            WriteHeader(result, bits: 0, elementCount, _elementType);
            rawBlock.CopyTo(result.AsSpan(HeaderBytes));
            return result;
        }

        private static void WriteHeader(Span<byte> destination, int bits, long elementCount, KvCodecElementType dtype)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(destination, Magic);
            destination[4] = FormatVersion;
            destination[5] = (byte)bits;
            destination[6] = (byte)dtype;
            destination[7] = 0;
            BinaryPrimitives.WriteUInt64LittleEndian(destination[8..], (ulong)elementCount);
        }

        private void LoadGroup(ReadOnlySpan<byte> rawBlock, long elementStart, Span<float> destination)
        {
            switch (_elementType)
            {
                case KvCodecElementType.Float32:
                {
                    int byteStart = (int)(elementStart * 4);
                    for (int i = 0; i < GroupSize; i++)
                        destination[i] = BinaryPrimitives.ReadSingleLittleEndian(rawBlock.Slice(byteStart + i * 4, 4));
                    break;
                }
                case KvCodecElementType.Float16:
                {
                    int byteStart = (int)(elementStart * 2);
                    for (int i = 0; i < GroupSize; i++)
                    {
                        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(rawBlock.Slice(byteStart + i * 2, 2));
                        destination[i] = (float)System.BitConverter.UInt16BitsToHalf(bits);
                    }
                    break;
                }
                default:
                    throw new InvalidOperationException($"LoadGroup invoked for unsupported dtype {_elementType}");
            }
        }

        private void StoreGroup(ReadOnlySpan<float> source, long elementStart, Span<byte> rawDestination)
        {
            switch (_elementType)
            {
                case KvCodecElementType.Float32:
                {
                    int byteStart = (int)(elementStart * 4);
                    for (int i = 0; i < GroupSize; i++)
                        BinaryPrimitives.WriteSingleLittleEndian(rawDestination.Slice(byteStart + i * 4, 4), source[i]);
                    break;
                }
                case KvCodecElementType.Float16:
                {
                    int byteStart = (int)(elementStart * 2);
                    for (int i = 0; i < GroupSize; i++)
                    {
                        ushort bits = System.BitConverter.HalfToUInt16Bits((System.Half)source[i]);
                        BinaryPrimitives.WriteUInt16LittleEndian(rawDestination.Slice(byteStart + i * 2, 2), bits);
                    }
                    break;
                }
                default:
                    throw new InvalidOperationException($"StoreGroup invoked for unsupported dtype {_elementType}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int QuantizeOne(float scaled)
        {
            int q = (int)MathF.Round(scaled);
            if (q < _qMin) q = _qMin;
            else if (q > _qMax) q = _qMax;
            return q;
        }

        // Affine 2-bit quantize: map x into the unsigned code range [0, 3]
        // anchored at the group min. invScale is 1 / ((max - min) / 3); a
        // degenerate (all-equal) group has invScale == 0 and collapses to code 0.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int QuantizeAffine(float x, float min, float invScale)
        {
            int q = (int)MathF.Round((x - min) * invScale);
            if (q < 0) q = 0;
            else if (q > 3) q = 3;
            return q;
        }

        // Encoded bytes per 32-element group. 2-bit carries an extra fp16 min
        // (affine layout): 2 (scale) + 2 (min) + 8 (codes). 4/8-bit carry only
        // the fp16 scale: 2 + GroupSize * bits / 8.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PerGroupBytes(int bits)
            => bits == 2 ? 4 + (GroupSize * 2 / 8) : 2 + (GroupSize * bits / 8);

        private static int ElementBytes(KvCodecElementType dtype) => dtype switch
        {
            KvCodecElementType.Float32 => 4,
            KvCodecElementType.Float16 => 2,
            // Q8_0 always passthrough - element bytes here are nominal; the
            // codec only branches on Q8_0 to short-circuit encoding.
            KvCodecElementType.Q8_0 => 1,
            _ => 0,
        };

        /// <summary>
        /// Read TS_KV_PAGED_QUANT_BITS and return a codec for the given KV dtype
        /// (or null when the env var is unset / requests passthrough). Designed
        /// so the paged manager can call this once at construction and stash
        /// the codec; nothing in the hot path looks at the env var.
        /// </summary>
        public static IKvBlockCodec FromEnvironment(KvCodecElementType dtype)
        {
            string raw = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
            if (string.IsNullOrWhiteSpace(raw) || !int.TryParse(raw.Trim(), out int bits))
                return null;
            if (bits == 0)
                return null;
            if (bits != 2 && bits != 4 && bits != 8)
                return null;
            // Q8_0 lives in 32-element ggml blocks with their own fp16 scale;
            // wrapping it in TurboQuant would just add a header without
            // compressing anything. Skip the codec wiring so the paged tier
            // stores the bytes unchanged.
            if (dtype == KvCodecElementType.Q8_0)
                return null;
            return new TurboQuantKvCodec(dtype, bits);
        }

        /// <summary>
        /// Model-aware codec selection. Returns the env-configured codec when
        /// it is safe to apply to <paramref name="model"/>, or null in two
        /// cases: (a) the env var is unset / disabled, or (b) the model's
        /// captured KV bytes include recurrent-state snapshots that quantization
        /// would corrupt.
        ///
        /// Recurrent-state guard: any architecture that reports
        /// <see cref="IModelArchitecture.RequiresPerBlockCapture"/> = true
        /// records a running SSM accumulator at every block boundary, not an
        /// order-invariant attention KV slice. Quantization noise on that
        /// accumulator propagates multiplicatively through subsequent
        /// positions — observed empirically on Qwen3.5/3.6 GatedDeltaNet as
        /// all-zero greedy samples after restore. The gate force-disables the
        /// codec for those architectures regardless of the env var, so pure-
        /// attention models can opt in via TS_KV_PAGED_QUANT_BITS without
        /// silently corrupting hybrid-architecture inference.
        /// </summary>
        public static IKvBlockCodec FromEnvironment(IModelArchitecture model)
        {
            if (model == null) return null;
            if (model.RequiresPerBlockCapture) return null;
            return FromEnvironment(model.KVStateElementType);
        }
    }

}
