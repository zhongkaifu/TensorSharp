// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Moe
{
    /// <summary>
    /// Loads standalone MoE expert modules from a <c>.safetensors</c> file. The
    /// layout is expert-major F32 and self-contained per layer:
    ///
    /// <code>
    /// experts.{layer}.ffn_gate.weight   [num_experts, n_ff, hidden]
    /// experts.{layer}.ffn_up.weight     [num_experts, n_ff, hidden]
    /// experts.{layer}.ffn_down.weight   [num_experts, hidden, n_ff]
    /// router.{layer}.weight             [num_experts, hidden]        (optional)
    /// </code>
    ///
    /// Safetensors <c>__metadata__</c> may carry <c>architecture</c>, <c>num_experts</c>,
    /// <c>target_layers</c> (comma-separated ints or a JSON array) and an optional
    /// <c>expert_type</c>. Target layers are also inferred from the tensor names and
    /// cross-checked against any declared <c>target_layers</c>.
    /// </summary>
    public static class MoeModuleLoader
    {
        public static MoeModule Load(string id, string safetensorsPath)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("A MoE module id is required.", nameof(id));
            if (!File.Exists(safetensorsPath))
                throw new FileNotFoundException($"MoE module file not found: {safetensorsPath}", safetensorsPath);

            using var st = new SafetensorsFile(safetensorsPath);

            string architecture = st.Metadata.TryGetValue("architecture", out var arch) ? arch : string.Empty;
            string? expertType = st.Metadata.TryGetValue("expert_type", out var eType) ? eType : null;
            int declaredExperts = st.Metadata.TryGetValue("num_experts", out var ne) ? ParseInt(ne, "num_experts") : 0;
            var declaredLayers = st.Metadata.TryGetValue("target_layers", out var tl) ? ParseLayers(tl) : null;

            // Detect per-layer expert sets from the tensor names.
            var gateLayers = new HashSet<int>();
            var upLayers = new HashSet<int>();
            var downLayers = new HashSet<int>();
            var routerLayers = new HashSet<int>();

            foreach (var name in st.Tensors.Keys)
            {
                if (TryParseLayerTensor(name, out int layer, out string kind))
                {
                    switch (kind)
                    {
                        case "ffn_gate": gateLayers.Add(layer); break;
                        case "ffn_up": upLayers.Add(layer); break;
                        case "ffn_down": downLayers.Add(layer); break;
                        case "router": routerLayers.Add(layer); break;
                    }
                }
            }

            if (gateLayers.Count == 0)
                throw new InvalidDataException($"MoE module '{safetensorsPath}' has no 'experts.<layer>.ffn_gate.weight' tensors.");

            // Every gated layer must have gate, up and down.
            var missingUp = gateLayers.Except(upLayers).ToList();
            var missingDown = gateLayers.Except(downLayers).ToList();
            if (missingUp.Count > 0 || missingDown.Count > 0)
                throw new InvalidDataException(
                    $"MoE module '{safetensorsPath}' is missing expert tensors: " +
                    $"layers {string.Join(",", missingUp)} lack ffn_up, layers {string.Join(",", missingDown)} lack ffn_down.");

            if (declaredLayers != null)
            {
                var inferred = gateLayers.ToHashSet();
                var declared = declaredLayers.ToHashSet();
                if (!inferred.SetEquals(declared))
                    throw new InvalidDataException(
                        $"MoE module '{safetensorsPath}' target_layers metadata {JsonSerializer.Serialize(declaredLayers)} " +
                        $"does not match inferred layers {JsonSerializer.Serialize(inferred.OrderBy(x => x))}.");
            }

            int numExperts = declaredExperts;
            var experts = new SortedDictionary<int, ExpertSet>();
            foreach (int layer in gateLayers.OrderBy(x => x))
            {
                var set = ReadExpertSet(st, layer, declaredExperts, ref numExperts);
                if (routerLayers.Contains(layer))
                    set.RouterWeights = ReadRouter(st, layer, set.NumExperts);
                experts[layer] = set;
            }

            string summary =
                $"moe {id}: {Path.GetFileName(safetensorsPath)} (arch={architecture}, experts={numExperts}, " +
                $"layers={experts.Keys.First()}-{experts.Keys.Last()}, router={(routerLayers.Count > 0 ? "yes" : "no")})";
            return new MoeModule(id, architecture, expertType, experts, summary);
        }

        private static ExpertSet ReadExpertSet(SafetensorsFile st, int layer, int declaredExperts, ref int numExperts)
        {
            var gate = st.Tensors[$"experts.{layer}.ffn_gate.weight"];
            var up = st.Tensors[$"experts.{layer}.ffn_up.weight"];
            var down = st.Tensors[$"experts.{layer}.ffn_down.weight"];

            if (gate.Shape.Length != 3 || up.Shape.Length != 3 || down.Shape.Length != 3)
                throw new InvalidDataException($"MoE expert tensors for layer {layer} must be rank-3.");

            int nExp = (int)gate.Shape[0];
            long nFf = gate.Shape[1];
            long hidden = gate.Shape[2];

            if (up.Shape[0] != nExp || up.Shape[1] != nFf || up.Shape[2] != hidden)
                throw new InvalidDataException($"Layer {layer}: ffn_up shape {Shape(up)} does not match ffn_gate {Shape(gate)}.");
            if (down.Shape[0] != nExp || down.Shape[1] != hidden || down.Shape[2] != nFf)
                throw new InvalidDataException($"Layer {layer}: ffn_down shape {Shape(down)} does not match [experts, hidden, n_ff] {Shape(gate)}.");

            if (numExperts == 0)
                numExperts = nExp;
            else if (numExperts != nExp)
                throw new InvalidDataException($"Layer {layer}: has {nExp} experts, but the module declares {numExperts}.");

            return new ExpertSet
            {
                NumExperts = nExp,
                Nff = nFf,
                Hidden = hidden,
                GateWeights = ReadFloat32(st, gate),
                UpWeights = ReadFloat32(st, up),
                DownWeights = ReadFloat32(st, down),
            };
        }

        private static float[] ReadRouter(SafetensorsFile st, int layer, int numExperts)
        {
            var router = st.Tensors[$"router.{layer}.weight"];
            if (router.Shape.Length != 2 || (int)router.Shape[0] != numExperts)
                throw new InvalidDataException($"Layer {layer}: router shape {Shape(router)} must be [num_experts, hidden].");
            return ReadFloat32(st, router);
        }

        private static float[] ReadFloat32(SafetensorsFile st, SafetensorTensorInfo info)
        {
            // ReadFloat32 handles F32/F16/BF16 and returns a flat row-major F32 array
            // whose length matches the declared shape.
            float[] data = st.ReadFloat32(info.Name);
            long expected = 1;
            foreach (var d in info.Shape) expected *= d;
            if (data.LongLength != expected)
                throw new InvalidDataException($"Tensor '{info.Name}' element count {data.LongLength} != {expected}.");
            return data;
        }

        private static bool TryParseLayerTensor(string name, out int layer, out string kind)
        {
            layer = 0;
            kind = string.Empty;
            var parts = name.Split('.');
            if (parts.Length == 4 && parts[0] == "experts" && int.TryParse(parts[1], out layer))
            {
                // parts[2] is ffn_gate / ffn_up / ffn_down, parts[3] == "weight"
                if (parts[3] == "weight" && (parts[2] == "ffn_gate" || parts[2] == "ffn_up" || parts[2] == "ffn_down"))
                {
                    kind = parts[2];
                    return true;
                }
            }
            else if (parts.Length == 3 && parts[0] == "router" && int.TryParse(parts[1], out layer) && parts[2] == "weight")
            {
                kind = "router";
                return true;
            }
            return false;
        }

        private static string Shape(SafetensorTensorInfo info) => $"[{string.Join(", ", info.Shape)}]";

        private static int ParseInt(string value, string field)
        {
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result))
                return result;
            throw new InvalidDataException($"MoE module metadata '{field}' must be an integer, got '{value}'.");
        }

        private static IReadOnlyList<int> ParseLayers(string value)
        {
            value = value.Trim();
            if (value.StartsWith("[", StringComparison.Ordinal))
            {
                using var doc = JsonDocument.Parse(value);
                var arr = doc.RootElement;
                if (arr.ValueKind != JsonValueKind.Array)
                    throw new InvalidDataException($"MoE module metadata 'target_layers' must be an array, got '{value}'.");
                var list = new List<int>();
                foreach (var el in arr.EnumerateArray())
                    list.Add(el.GetInt32());
                return list;
            }

            return value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => ParseInt(s.Trim(), "target_layers"))
                .ToList();
        }
    }
}