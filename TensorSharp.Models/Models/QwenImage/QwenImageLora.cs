// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Runtime LoRA for the Qwen-Image DiT (Lightning distillation support).
//
// The LoRA is applied as a runtime side-path next to each targeted projection:
//     y = W_quant · x + b + (alpha / rank) * multiplier * lora_up · (lora_down · x)
// with the A/B factors held in F32 and the quantized base weights UNTOUCHED.
//
// Why not merge into the weights (the stable-diffusion.cpp lora.hpp approach of
// dequantize -> add -> requantize)? That is only sound when the storage type can
// represent the delta. The Lightning step-distillation deltas are ~1e-4 RMS while
// Q2_K quantization steps are ~100x larger — measured on this model, a merge
// changed the velocity by 24% relL2 but the change was requantization NOISE, not
// the distillation direction (the 8-step output stayed un-distilled). The runtime
// side-path keeps the delta exact at ~4% extra FLOPs and ~2.3 GB resident F32
// factors, and is capture-safe (A/B are resident leaves like the weights).
//
// Primary use: lightx2v Qwen-Image-Edit-Lightning LoRAs (4/8-step editing at
// cfg 1.0, fixed timestep shift 3). Those checkpoints cover exactly the 12
// per-block matmuls (attn to_q/k/v/out, add_q/k/v, to_add_out, img/txt GEGLU
// MLPs) as {base}.lora_down.weight [rank,in] / {base}.lora_up.weight [out,rank] /
// {base}.alpha, with base keys that match our GGUF tensor names 1:1. The factors
// are loaded into stable unmanaged buffers whose layout feeds the native
// QImgAttnW.LoraA/LoraB fields directly (row-major, no transposes).
// ============================================================================
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// LoRA factors resolved per GGUF weight name, in stable unmanaged F32 buffers
    /// ready for the native runtime side-path. Owned by the DiT; dispose frees them.
    /// </summary>
    internal sealed class QwenImageLoraTable : IDisposable
    {
        internal sealed class Entry
        {
            public IntPtr A;        // [rank, in] row-major F32 (lora_down)
            public IntPtr B;        // [out, rank] row-major F32 (lora_up)
            public long Rank, In, Out;
            public float Scale;     // alpha/rank * multiplier
        }

        private readonly Dictionary<string, Entry> _byWeightName;   // key = "<base>.weight"
        private bool _disposed;

        public int Count => _byWeightName.Count;

        private QwenImageLoraTable(Dictionary<string, Entry> entries) => _byWeightName = entries;

        /// <summary>Lookup by full GGUF weight tensor name (e.g. "transformer_blocks.0.attn.to_q.weight").</summary>
        public bool TryGet(string weightName, out Entry e) => _byWeightName.TryGetValue(weightName, out e);

        /// <summary>
        /// Steps encoded in a Lightning LoRA filename ("...-4steps-..." / "...8step...");
        /// 0 when the filename carries no step hint (generic LoRA — leave sampling defaults alone).
        /// </summary>
        public static int ParseLightningSteps(string path)
        {
            var m = Regex.Match(Path.GetFileNameWithoutExtension(path), @"(\d+)\s*step", RegexOptions.IgnoreCase);
            return m.Success && int.TryParse(m.Groups[1].Value, out int s) && s > 0 && s <= 16 ? s : 0;
        }

        /// <summary>
        /// Load the LoRA safetensors into unmanaged F32 factor buffers. Accepts both the
        /// lightx2v/kohya naming (lora_down/lora_up/alpha) and the diffusers-PEFT naming
        /// (lora_A=down, lora_B=up), with or without a "transformer."/"diffusion_model." prefix.
        /// </summary>
        public static QwenImageLoraTable Load(string loraPath, float multiplier)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var lora = new SafetensorsFile(loraPath);

            var pairs = new Dictionary<string, (string down, string up, string alpha)>(StringComparer.Ordinal);
            foreach (var name in lora.Tensors.Keys)
            {
                string baseKey; int kind; // 0=down 1=up 2=alpha
                if (name.EndsWith(".lora_down.weight", StringComparison.Ordinal)) { baseKey = name[..^17]; kind = 0; }
                else if (name.EndsWith(".lora_up.weight", StringComparison.Ordinal)) { baseKey = name[..^15]; kind = 1; }
                else if (name.EndsWith(".lora_A.weight", StringComparison.Ordinal)) { baseKey = name[..^14]; kind = 0; }
                else if (name.EndsWith(".lora_B.weight", StringComparison.Ordinal)) { baseKey = name[..^14]; kind = 1; }
                else if (name.EndsWith(".alpha", StringComparison.Ordinal)) { baseKey = name[..^6]; kind = 2; }
                else continue;
                if (baseKey.StartsWith("transformer.", StringComparison.Ordinal)) baseKey = baseKey["transformer.".Length..];
                else if (baseKey.StartsWith("diffusion_model.", StringComparison.Ordinal)) baseKey = baseKey["diffusion_model.".Length..];
                var e = pairs.TryGetValue(baseKey, out var cur) ? cur : default;
                if (kind == 0) e.down = name; else if (kind == 1) e.up = name; else e.alpha = name;
                pairs[baseKey] = e;
            }

            var entries = new Dictionary<string, Entry>(pairs.Count, StringComparer.Ordinal);
            int skipped = 0;
            long bytes = 0;
            foreach (var (baseKey, p) in pairs)
            {
                if (p.down == null || p.up == null) { skipped++; continue; }
                var downInfo = lora.GetInfo(p.down);   // [rank, in]
                var upInfo = lora.GetInfo(p.up);       // [out, rank]
                if (downInfo.Shape.Length != 2 || upInfo.Shape.Length != 2 ||
                    upInfo.Shape[1] != downInfo.Shape[0]) { skipped++; continue; }
                long rank = downInfo.Shape[0], nIn = downInfo.Shape[1], nOut = upInfo.Shape[0];

                float scale = multiplier;
                if (p.alpha != null) scale *= lora.ReadFloat32(p.alpha)[0] / rank;

                entries[baseKey + ".weight"] = new Entry
                {
                    A = ToUnmanaged(lora.ReadFloat32(p.down)),
                    B = ToUnmanaged(lora.ReadFloat32(p.up)),
                    Rank = rank,
                    In = nIn,
                    Out = nOut,
                    Scale = scale,
                };
                bytes += (rank * nIn + nOut * rank) * 4;
            }
            sw.Stop();
            Console.WriteLine($"  [lora] loaded {entries.Count} runtime LoRA pairs from {Path.GetFileName(loraPath)}" +
                              $" ({bytes / (1024.0 * 1024 * 1024):F2} GB F32, x{multiplier:F2}) in {sw.Elapsed.TotalSeconds:F1}s" +
                              (skipped > 0 ? $"; {skipped} unusable keys skipped" : ""));
            if (entries.Count == 0)
                throw new InvalidOperationException($"LoRA '{loraPath}' contained no usable lora_down/lora_up pairs.");
            return new QwenImageLoraTable(entries);
        }

        private static unsafe IntPtr ToUnmanaged(float[] data)
        {
            IntPtr p = Marshal.AllocHGlobal((IntPtr)((long)data.Length * sizeof(float)));
            fixed (float* src = data)
            {
                Buffer.MemoryCopy(src, (void*)p, (long)data.Length * sizeof(float), (long)data.Length * sizeof(float));
            }
            return p;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var e in _byWeightName.Values)
            {
                if (e.A != IntPtr.Zero) Marshal.FreeHGlobal(e.A);
                if (e.B != IntPtr.Zero) Marshal.FreeHGlobal(e.B);
                e.A = e.B = IntPtr.Zero;
            }
            _byWeightName.Clear();
        }
    }
}
