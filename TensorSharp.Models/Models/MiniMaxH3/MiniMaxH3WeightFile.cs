// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The file a MiniMax-H3 denoiser is read from: a GGUF, as the video denoisers ship
// (unsloth/MiniMax-H3-GGUF), or a safetensors file in the original tensor names, as
// MiniMax-H3-Image ships (Comfy-Org/MiniMax-H3-Image, minimax_h3_image_pruned_bf16).
// Both expose one tensor table in GGUF terms (ne-order shapes, ggml types) and a
// pointer into the mapping, so the DiT binds either the same way.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    internal sealed class MiniMaxH3WeightFile : IDisposable
    {
        private readonly SafetensorsFile _safetensors;
        private readonly Dictionary<string, GgufTensorInfo> _tensors;
        private readonly Dictionary<string, SafetensorTensorInfo> _safetensorsInfo;

        /// <summary>The GGUF, or null when the weights are a safetensors file.</summary>
        public GgufFile Gguf { get; }
        public string Path { get; }

        /// <summary>Every tensor in ggml terms: the shape in ne order (contraction dim first)
        /// and the ggml type of the stored data.</summary>
        public IReadOnlyDictionary<string, GgufTensorInfo> Tensors => Gguf != null ? Gguf.Tensors : _tensors;

        /// <summary>The safetensors <c>__metadata__</c> map, empty for a GGUF.</summary>
        public IReadOnlyDictionary<string, string> SafetensorsMetadata =>
            _safetensors?.Metadata ?? (IReadOnlyDictionary<string, string>)new Dictionary<string, string>();

        public static bool IsSafetensors(string path) =>
            path != null && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);

        public MiniMaxH3WeightFile(string path)
        {
            Path = path ?? throw new ArgumentNullException(nameof(path));
            if (!IsSafetensors(path))
            {
                Gguf = new GgufFile(path);
                return;
            }

            _safetensors = new SafetensorsFile(path);
            try
            {
                _tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
                _safetensorsInfo = new Dictionary<string, SafetensorTensorInfo>(StringComparer.Ordinal);
                foreach (var (name, info) in _safetensors.Tensors)
                {
                    GgmlTensorType type;
                    switch (info.Dtype)
                    {
                        case SafetensorDtype.F32: type = GgmlTensorType.F32; break;
                        case SafetensorDtype.F16: type = GgmlTensorType.F16; break;
                        case SafetensorDtype.BF16: type = GgmlTensorType.BF16; break;
                        default:
                            // ComfyUI's own quantized repacks (int8 + convrot scales, fp8) store
                            // the projections in formats ggml has no type for. Their U8 side
                            // tensors (an attention config per block) are not weights and are skipped.
                            if (info.Dtype != SafetensorDtype.U8 && name.EndsWith(".weight", StringComparison.Ordinal))
                                throw new NotSupportedException(
                                    $"'{System.IO.Path.GetFileName(path)}' stores '{name}' as {info.Dtype}, a ComfyUI-specific " +
                                    "quantization TensorSharp does not read. Use the bf16 safetensors release " +
                                    "(e.g. minimax_h3_image_pruned_bf16.safetensors) or a GGUF.");
                            continue;
                    }
                    var shape = new ulong[info.Shape.Length];
                    for (int i = 0; i < shape.Length; i++)
                        shape[i] = (ulong)info.Shape[info.Shape.Length - 1 - i];
                    _tensors[name] = new GgufTensorInfo
                    {
                        Name = name,
                        Shape = shape,
                        Type = type,
                        Offset = (ulong)info.Begin,
                    };
                    _safetensorsInfo[name] = info;
                }
            }
            catch
            {
                _safetensors.Dispose();
                throw;
            }
        }

        public bool TryGetTensorDataPointer(GgufTensorInfo info, out IntPtr pointer)
        {
            if (Gguf != null) return Gguf.TryGetTensorDataPointer(info, out pointer);
            pointer = IntPtr.Zero;
            return info != null && _safetensorsInfo.TryGetValue(info.Name, out var st)
                   && _safetensors.TryGetTensorDataPointer(st, out pointer);
        }

        /// <summary>The mappings behind the tensor pointers, for the native direct reads.</summary>
        public IReadOnlyList<(IntPtr Base, long Length, string Path)> MappedRegions()
        {
            if (Gguf != null) return Gguf.MappedRegions();
            var (address, length) = _safetensors.MappedRegion();
            return new[] { (address, length, Path) };
        }

        public void Dispose()
        {
            Gguf?.Dispose();
            _safetensors?.Dispose();
        }
    }
}
