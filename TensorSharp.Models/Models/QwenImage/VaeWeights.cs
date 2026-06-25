// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Lazily fetches Qwen-Image VAE tensors (stored F32 in the converted GGUF) by name
    /// into managed <c>float[]</c> buffers, in their original PyTorch row-major order
    /// (conv weight index <c>((((oc*IC+ic)*KD+kd)*KH+kh)*KW+kw)</c>). The 5D conv weights
    /// were collapsed to 4D <c>(OC*IC,KD,KH,KW)</c> at conversion time but the byte order
    /// is unchanged, so callers index with the logical 5D shape they already know from
    /// the architecture.
    /// </summary>
    internal sealed class VaeWeights
    {
        private readonly GgufFile _gguf;
        private readonly Dictionary<string, float[]> _cache = new();

        private VaeWeights(GgufFile gguf) { _gguf = gguf; }

        public static VaeWeights Load(GgufFile gguf) => new VaeWeights(gguf);

        public bool Has(string name) => _gguf.Tensors.ContainsKey(name);

        /// <summary>Fetch a tensor by name as a flat F32 array (cached). Throws if absent.</summary>
        public float[] Get(string name)
        {
            if (_cache.TryGetValue(name, out var cached)) return cached;
            if (!_gguf.Tensors.TryGetValue(name, out var info))
                throw new KeyNotFoundException($"VAE tensor not found: {name}");
            long n = info.NumElements;
            var dst = new float[n];
            _gguf.ReadTensorDataToFloat32(info, dst, n);
            _cache[name] = dst;
            return dst;
        }

        public float[] TryGet(string name) => Has(name) ? Get(name) : null;

        /// <summary>ggml shape (ne0-fastest) of a named tensor, e.g. conv weight = [KW,KH,KD,OC*IC].</summary>
        public ulong[] Shape(string name) => _gguf.Tensors[name].Shape;
    }
}
