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
    /// Lazily fetches Qwen-Image-2.1 VAE tensors by name into managed <c>float[]</c> buffers, in their
    /// original PyTorch row-major order (conv weight index <c>((((oc*IC+ic)*KD+kd)*KH+kh)*KW+kw)</c>).
    /// The weights come from an <see cref="IFloatTensorStore"/>, which is either the original
    /// <c>.safetensors</c> file (BF16 upcast to F32 on read) or a converted VAE GGUF (stored F32):
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

        public bool Has(string name) => _src.HasTensor(name);

        /// <summary>Fetch a tensor by name as a flat F32 array (cached). Throws if absent.</summary>
        public float[] Get(string name)
        {
            if (_cache.TryGetValue(name, out var cached)) return cached;
            var dst = _src.ReadFloat32(name);
            _cache[name] = dst;
            return dst;
        }

        /// <summary>Logical row-major shape (outermost dim first) of a named tensor.</summary>
        public long[] Shape(string name) => _src.TensorShape(name);

        private readonly Dictionary<string, float[]> _slices = new();
        private readonly Dictionary<string, PackedPanels> _packed = new();

        /// <summary>The T=1 2-D kernel (OC, IC*KH*KW) of a causal conv weight: its last temporal
        /// slice, extracted once per layer (the weight itself when KD == 1).</summary>
        public float[] KernelSlice(string name, int oc, int ic, int kd, int kh, int kw)
        {
            if (kd == 1) return Get(name);
            if (!_slices.TryGetValue(name, out var slice))
                _slices[name] = slice = VaeReferenceMath.LastTemporalSlice(Get(name), oc, ic, kd, kh, kw);
            return slice;
        }

        /// <summary>
        /// The T=1 kernel packed for the managed GEMM convolution, built once per layer. The pack
        /// is the only copy that path reads, so the raw F32 weight is dropped from the cache (a
        /// device path that needs it later re-reads it): the managed decoder holds one copy of
        /// its ~1.1 GB of weights, not two.
        /// </summary>
        internal unsafe PackedPanels PackedKernel(string name, int oc, int ic, int kd, int khw)
        {
            CpuGemmIsa isa = CpuPackedGemm.Isa;
            if (_packed.TryGetValue(name, out var packed) && packed.Isa == isa) return packed;
            float[] raw = _cache.TryGetValue(name, out var cached) ? cached : _src.ReadFloat32(name);
            int k = ic * khw;
            float[] slice = kd == 1 ? raw : VaeReferenceMath.LastTemporalSlice(raw, oc, ic, kd, khw, 1);
            if (slice.LongLength != (long)oc * k)
                throw new ArgumentException($"VAE weight {name} has {slice.LongLength} values, expected {oc} x {k}.");
            fixed (float* w = slice) packed = CpuPackedGemm.PackA(w, oc, k, k, 1, isa, VaeReferenceMath.CpuPool);
            _packed[name] = packed;
            _cache.Remove(name);
            _slices.Remove(name);
            return packed;
        }

        /// <summary>Fused whole-VAE device graph, built lazily on first encode/decode
        /// (null after a failed build → the per-conv path is used). Holds the weights in
        /// stable unmanaged buffers for resident binding; lives as long as the weights.</summary>
        internal QwenImageVaeGraph FusedGraph;
        internal bool FusedGraphBuildFailed;
    }
}
