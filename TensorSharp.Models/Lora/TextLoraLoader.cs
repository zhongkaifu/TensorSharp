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
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// Loads text-LoRA adapters from a <c>.safetensors</c> file whose tensor names mirror
    /// TensorSharp weight names with a <c>.lora_a</c> / <c>.lora_b</c> suffix. This is the
    /// same "down/up factors" convention PEFT uses, only the target name is the TensorSharp
    /// weight name rather than a <c>q_proj</c>-style module id:
    ///
    /// <code>
    /// blk.0.attn_qkv.weight.lora_a   [rank, in]
    /// blk.0.attn_qkv.weight.lora_b   [out, rank]
    /// blk.0.ffn_gate_up.weight.lora_a
    /// blk.0.ffn_gate_up.weight.lora_b
    /// blk.0.ffn_down.weight.lora_a
    /// blk.0.ffn_down.weight.lora_b
    /// output.weight.lora_a
    /// output.weight.lora_b
    /// </code>
    ///
    /// The optional <c>adapter_config.json</c> (PEFT style) supplies <c>r</c> and
    /// <c>lora_alpha</c>; without it the rank is inferred from the <c>lora_a</c> shape and
    /// <c>alpha</c> defaults to <c>rank</c> (scale 1.0). <c>r</c> and <c>lora_alpha</c> may
    /// also come from the safetensors <c>__metadata__</c> <c>rank</c> / <c>alpha</c> keys.
    /// </summary>
    public static class TextLoraLoader
    {
        public static TextLoraAdapter Load(
            string name,
            string safetensorsPath,
            string? configPath = null,
            float? scaleOverride = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));
            if (!File.Exists(safetensorsPath))
                throw new FileNotFoundException($"LoRA weights file not found: {safetensorsPath}", safetensorsPath);

            configPath = ResolveConfigPath(safetensorsPath, configPath);
            int configuredRank = 0;
            float configuredAlpha = 0f;
            if (configPath != null)
                ReadConfig(configPath, ref configuredRank, ref configuredAlpha);

            using var st = new SafetensorsFile(safetensorsPath);

            // A lora_a tensor pairs with a lora_b tensor on the same target weight.
            var pairs = new Dictionary<string, (SafetensorTensorInfo? A, SafetensorTensorInfo? B)>(StringComparer.Ordinal);
            foreach (var kv in st.Tensors)
            {
                string tensorName = kv.Key;
                if (tensorName.EndsWith(".lora_a", StringComparison.Ordinal))
                    Upsert(ref pairs, tensorName[..^".lora_a".Length], a: kv.Value);
                else if (tensorName.EndsWith(".lora_b", StringComparison.Ordinal))
                    Upsert(ref pairs, tensorName[..^".lora_b".Length], b: kv.Value);
            }

            // Metadata rank/alpha are the least specific source, so only use them when
            // the config did not already supply a value.
            if (configuredRank <= 0 || configuredAlpha <= 0f)
            {
                st.Metadata.TryGetValue("rank", out string? metaRank);
                st.Metadata.TryGetValue("alpha", out string? metaAlpha);
                if (configuredRank <= 0 && metaRank != null && int.TryParse(metaRank, NumberStyles.Integer, CultureInfo.InvariantCulture, out int mr))
                    configuredRank = mr;
                if (configuredAlpha <= 0f && metaAlpha != null && float.TryParse(metaAlpha, NumberStyles.Float, CultureInfo.InvariantCulture, out float ma))
                    configuredAlpha = ma;
            }

            if (pairs.Count == 0)
                throw new InvalidDataException(
                    $"LoRA file '{safetensorsPath}' contains no '<weight>.lora_a' / '<weight>.lora_b' tensor pairs.");

            int rank = configuredRank;
            var deltas = new Dictionary<string, LoraDelta>(StringComparer.Ordinal);
            foreach (var (target, (aInfo, bInfo)) in pairs)
            {
                if (aInfo == null)
                    throw new InvalidDataException($"LoRA file '{safetensorsPath}' has no '{target}.lora_a' tensor for '{target}.lora_b'.");
                if (bInfo == null)
                    throw new InvalidDataException($"LoRA file '{safetensorsPath}' has no '{target}.lora_b' tensor for '{target}.lora_a'.");

                if (aInfo.Shape.Length != 2 || bInfo.Shape.Length != 2)
                    throw new InvalidDataException(
                        $"LoRA tensor '{target}.lora_a/lora_b' must be rank-2, got {string.Join("x", aInfo.Shape)} / {string.Join("x", bInfo.Shape)}.");

                long rankA = aInfo.Shape[0];
                long inDim = aInfo.Shape[1];
                long outDim = bInfo.Shape[0];
                long rankB = bInfo.Shape[1];

                if (rankA != rankB)
                    throw new InvalidDataException(
                        $"LoRA rank mismatch for '{target}': lora_a rank {rankA} != lora_b rank {rankB}.");

                if (rank == 0)
                    rank = checked((int)rankA);
                else if (rank != (int)rankA)
                    throw new InvalidDataException(
                        $"LoRA rank mismatch for '{target}': file has rank {rankA}, config declares {rank}.");

                int r = (int)rankA;

                // Transpose A ([rank, in] -> [in, rank]) and B ([out, rank] -> [rank, out]),
                // keeping row-major layout so Ops.Addmm(x, AT) and Addmm(h, BT) are plain GEMMs.
                float[] at = ReadTransposed(st, aInfo, rankA, inDim);
                float[] bt = ReadTransposed(st, bInfo, outDim, rankB);

                float alpha = configuredAlpha > 0f ? configuredAlpha : r;
                float scale = alpha / r;
                if (scaleOverride is { } userScale)
                    scale *= userScale;
                for (int i = 0; i < bt.Length; i++)
                    bt[i] *= scale;

                deltas[target] = new LoraDelta
                {
                    In = inDim,
                    Out = outDim,
                    Rank = r,
                    AT = at,
                    BT = bt,
                };
            }

            float resolvedAlpha = configuredAlpha > 0f ? configuredAlpha : rank;
            string summary =
                $"lora {name}: {Path.GetFileName(safetensorsPath)} (r={rank}, alpha={resolvedAlpha}, modules={deltas.Count})" +
                (configPath != null ? $" config={Path.GetFileName(configPath)}" : string.Empty);
            return new TextLoraAdapter(name, rank, resolvedAlpha, deltas, summary);
        }

        private static void Upsert(
            ref Dictionary<string, (SafetensorTensorInfo? A, SafetensorTensorInfo? B)> pairs,
            string target,
            SafetensorTensorInfo? a = null,
            SafetensorTensorInfo? b = null)
        {
            pairs.TryGetValue(target, out var existing);
            pairs[target] = (a ?? existing.A, b ?? existing.B);
        }

        private static void ReadConfig(string configPath, ref int rank, ref float alpha)
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"LoRA config file not found: {configPath}", configPath);

            using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"LoRA config '{configPath}' must be a JSON object.");

            if (root.TryGetProperty("r", out var rEl) && rEl.ValueKind == JsonValueKind.Number && rEl.TryGetInt32(out int r))
                rank = r;
            if (root.TryGetProperty("lora_alpha", out var aEl) && aEl.ValueKind == JsonValueKind.Number && aEl.TryGetSingle(out float a))
                alpha = a;
        }

        private static string? ResolveConfigPath(string weightsPath, string? configPath)
        {
            if (configPath != null)
                return configPath;

            string dir = Path.GetDirectoryName(Path.GetFullPath(weightsPath)) ?? string.Empty;
            string sibling = Path.Combine(dir, "adapter_config.json");
            return File.Exists(sibling) ? sibling : null;
        }

        private static float[] ReadTransposed(SafetensorsFile st, SafetensorTensorInfo info, long rows, long cols)
        {
            // info is [rows, cols] row-major; the output is [cols, rows] row-major.
            if (info.NumElements != rows * cols)
                throw new InvalidDataException(
                    $"LoRA tensor '{info.Name}' has {info.NumElements} elements, expected {rows}x{cols}.");

            float[] src = st.ReadFloat32(info.Name);
            var dst = new float[src.Length];
            for (long r = 0; r < rows; r++)
                for (long c = 0; c < cols; c++)
                    dst[c * rows + r] = src[r * cols + c];
            return dst;
        }
    }
}