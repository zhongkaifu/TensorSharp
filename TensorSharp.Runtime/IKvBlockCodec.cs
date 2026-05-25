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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Element type the codec interprets the raw payload bytes as. The paged KV
    /// tier serializes a model's per-layer K/V tensors as opaque bytes, so the
    /// codec needs to know whether a 4-byte run is one F32 element or two F16
    /// elements before it can group and re-quantize anything.
    /// </summary>
    public enum KvCodecElementType
    {
        Float32 = 0,
        Float16 = 1,
        /// <summary>
        /// Storage already uses GGML Q8_0 blocks. The TurboQuant codec leaves
        /// these bytes untouched (re-quantizing quantized data would compound
        /// precision loss for no clear win) and the encoder simply round-trips
        /// the payload unchanged.
        /// </summary>
        Q8_0 = 2,
    }

    /// <summary>
    /// Lossy / lossless byte-block codec used by the paged KV cache tier to
    /// re-encode captured K/V payloads before storing them. Implementations are
    /// allowed to return a smaller buffer than the input - the tier accounts
    /// for residency in encoded bytes so a tighter codec lets more blocks fit
    /// inside the same RAM / SSD budget.
    ///
    /// The codec is constructed with the model's KV element type; the payload
    /// always represents a flat sequence of elements of that type laid out by
    /// <see cref="TensorSharp.Models.KvBlockTransfer"/> (head-major, then
    /// position, then head-dim within each head). Implementations MUST be
    /// thread-safe for concurrent encode / decode calls because the paged
    /// store routes captures and restorations from different inference
    /// threads through a single codec instance.
    /// </summary>
    public interface IKvBlockCodec
    {
        /// <summary>Short identifier used in logs and the codec fingerprint
        /// (e.g. <c>"turboquant-int4"</c>). Round-tripped through the SSD
        /// header so blocks encoded with a different codec are not silently
        /// reinterpreted.</summary>
        string Name { get; }

        /// <summary>Number of bits the codec compresses each element down to.
        /// Used for logging and for the budget estimate; 16/32 means the
        /// codec is lossless / passthrough.</summary>
        int BitsPerElement { get; }

        /// <summary>
        /// Encode <paramref name="rawBlock"/> into a freshly-allocated byte
        /// array. Implementations may return the input unchanged for
        /// element types they don't compress (returning the same array is
        /// fine; callers treat the return value as read-only).
        /// </summary>
        byte[] Encode(ReadOnlySpan<byte> rawBlock);

        /// <summary>
        /// Decode <paramref name="encoded"/> into <paramref name="rawDestination"/>.
        /// The destination buffer is sized by the caller from
        /// <see cref="TensorSharp.Runtime.IModelArchitecture.ComputeKVBlockByteSize"/>
        /// so it always equals the original raw payload length the encoder saw.
        /// Returns false (without partially populating the destination) on
        /// header mismatch, version skew, or any condition that makes the
        /// encoded blob unsafe to reinject into the model.
        /// </summary>
        bool TryDecode(ReadOnlySpan<byte> encoded, Span<byte> rawDestination);
    }
}
