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
    /// Lazily fetches Qwen-Image VAE tensors by name into managed <c>float[]</c> buffers, in their
    /// original PyTorch row-major order (conv weight index <c>((((oc*IC+ic)*KD+kd)*KH+kh)*KW+kw)</c>).
    /// The weights come from an <see cref="IFloatTensorStore"/>, which is either the original
    /// <c>.safetensors</c> file (BF16 upcast to F32 on read) or the converted VAE GGUF (stored F32):
    /// both yield bit-identical floats, so the VAE runs unchanged regardless of source. The 5D conv
    /// weights are returned as a flat array (byte order unchanged); callers index with the logical 5D
    /// shape they already know from the architecture.
    /// </summary>
    internal sealed class VaeWeights
    {
        private readonly IFloatTensorStore _src;
        private readonly Dictionary<string, float[]> _cache = new();

        private VaeWeights(IFloatTensorStore src) { _src = src; }

        public static VaeWeights Load(IFloatTensorStore src) => new VaeWeights(src);
        public static VaeWeights Load(GgufFile gguf) => new VaeWeights(new GgufFloatTensorStore(gguf));

        public bool Has(string name) => _src.HasTensor(name);

        /// <summary>Fetch a tensor by name as a flat F32 array (cached). Throws if absent.</summary>
        public float[] Get(string name)
        {
            if (_cache.TryGetValue(name, out var cached)) return cached;
            var dst = _src.ReadFloat32(name);
            _cache[name] = dst;
            return dst;
        }

        public float[] TryGet(string name) => Has(name) ? Get(name) : null;

        /// <summary>Logical row-major shape (outermost dim first) of a named tensor.</summary>
        public long[] Shape(string name) => _src.TensorShape(name);
    }
}
