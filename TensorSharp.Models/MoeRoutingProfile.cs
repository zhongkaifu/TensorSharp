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
using System.Linq;
using System.Text.Json;

namespace TensorSharp.Models
{
    /// <summary>
    /// How often each routed expert of a Mixture-of-Experts model was selected,
    /// per layer, accumulated across runs. It ranks the experts a host-offloaded
    /// layer keeps in its accelerator tier (<see cref="SelectTiers"/>): a route
    /// whose expert is in the tier runs on the accelerator, every other route on
    /// the host. The profile never changes routing.
    ///
    /// <para>Counts come from the host seams, which already read every routed id
    /// (<c>TSGgml_MoeRoutingCounts</c>). They are saved when a model is disposed
    /// and merged into the file for that model, so the tier follows the user's
    /// own workload. <c>TENSORSHARP_MOE_PROFILE_DIR</c> moves the files;
    /// <c>TENSORSHARP_MOE_PROFILE</c> names one file explicitly (read and
    /// written).</para>
    /// </summary>
    public sealed class MoeRoutingProfile
    {
        public const int FormatVersion = 1;

        /// <summary>Selections a layer needs before its ranking is trusted: about
        /// 40 per expert of a 512-expert layer at top-10.</summary>
        public const long MinimumRoutesPerLayer = 20_000;

        public string Identity { get; }
        public int Layers { get; }
        public int Experts { get; }

        private readonly ulong[][] _counts;

        public MoeRoutingProfile(string identity, int layers, int experts)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(identity);
            if (layers <= 0 || experts <= 0) throw new ArgumentOutOfRangeException(nameof(layers));
            Identity = identity;
            Layers = layers;
            Experts = experts;
            _counts = new ulong[layers][];
            for (int l = 0; l < layers; l++) _counts[l] = new ulong[experts];
        }

        public ReadOnlySpan<ulong> Counts(int layer) => _counts[layer];

        public ulong Routes(int layer)
        {
            ulong total = 0;
            foreach (ulong c in _counts[layer]) total += c;
            return total;
        }

        /// <summary>True when every MoE layer in <paramref name="layers"/> has enough
        /// selections to rank its experts.</summary>
        public bool CoversLayers(IEnumerable<int> layers)
            => layers.All(l => l >= 0 && l < Layers && Routes(l) >= MinimumRoutesPerLayer);

        public void Add(int layer, ReadOnlySpan<ulong> counts)
        {
            if (layer < 0 || layer >= Layers) throw new ArgumentOutOfRangeException(nameof(layer));
            if (counts.Length != Experts) throw new ArgumentException("Expert count differs from the profile.", nameof(counts));
            for (int e = 0; e < Experts; e++) _counts[layer][e] = checked(_counts[layer][e] + counts[e]);
        }

        /// <summary>Experts of <paramref name="layer"/>, most selected first; ties keep
        /// the lower id first so a ranking is deterministic.</summary>
        public int[] Rank(int layer)
        {
            var order = Enumerable.Range(0, Experts).ToArray();
            ulong[] c = _counts[layer];
            Array.Sort(order, (a, b) => c[a] != c[b] ? c[b].CompareTo(c[a]) : a.CompareTo(b));
            return order;
        }

        /// <summary>
        /// Choose each layer's tier within <paramref name="budgetBytes"/>: experts are
        /// taken in order of their share of their own layer's selections per byte,
        /// across all layers, so the bytes go wherever they cover the most routes.
        /// Layers absent from <paramref name="expertBytes"/> (value 0) get no tier.
        /// Returns, per layer, the chosen experts in rank order (possibly empty).
        /// </summary>
        public int[][] SelectTiers(IReadOnlyList<long> expertBytes, long budgetBytes)
        {
            ArgumentNullException.ThrowIfNull(expertBytes);
            if (expertBytes.Count != Layers) throw new ArgumentException("One size per layer is required.", nameof(expertBytes));
            var ranks = new int[Layers][];
            var candidates = new List<(double Value, int Layer, int Index)>(Layers * Experts);
            for (int l = 0; l < Layers; l++)
            {
                ranks[l] = Rank(l);
                double routes = Routes(l);
                if (expertBytes[l] <= 0 || routes <= 0) continue;
                for (int i = 0; i < Experts; i++)
                    candidates.Add((_counts[l][ranks[l][i]] / routes / expertBytes[l], l, i));
            }
            // Within a layer the values fall with rank, so taking the global order
            // always takes each layer's prefix.
            candidates.Sort((a, b) => a.Value != b.Value ? b.Value.CompareTo(a.Value)
                : a.Layer != b.Layer ? a.Layer.CompareTo(b.Layer) : a.Index.CompareTo(b.Index));
            var take = new int[Layers];
            long used = 0;
            foreach (var (value, layer, index) in candidates)
            {
                if (value <= 0) break;
                if (index != take[layer] || used + expertBytes[layer] > budgetBytes) continue;
                used += expertBytes[layer];
                take[layer]++;
            }
            var tiers = new int[Layers][];
            for (int l = 0; l < Layers; l++) tiers[l] = ranks[l].AsSpan(0, take[l]).ToArray();
            return tiers;
        }

        // ---- persistence ------------------------------------------------------

        /// <summary>Stable name for a model file set: the first file's name, the total
        /// byte size of all files, and the routed shape.</summary>
        public static string IdentityFor(IReadOnlyList<string> modelFiles, string architecture, int layers, int experts)
        {
            if (modelFiles == null || modelFiles.Count == 0) return null;
            long size = 0;
            foreach (string f in modelFiles)
            {
                try { size = checked(size + new FileInfo(f).Length); }
                catch (IOException) { return null; }
                catch (UnauthorizedAccessException) { return null; }
            }
            return string.Create(CultureInfo.InvariantCulture,
                $"{Path.GetFileName(modelFiles[0])}|{size}|{architecture}|{layers}x{experts}");
        }

        public static string DefaultDirectory()
        {
            string dir = Environment.GetEnvironmentVariable("TENSORSHARP_MOE_PROFILE_DIR");
            if (!string.IsNullOrWhiteSpace(dir)) return dir.Trim();
            string cache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            if (string.IsNullOrWhiteSpace(cache))
            {
                string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                if (string.IsNullOrEmpty(home)) return null;
                cache = Path.Combine(home, ".cache");
            }
            return Path.Combine(cache, "tensorsharp", "moe-routing");
        }

        /// <summary>The file a profile with <paramref name="identity"/> lives in.</summary>
        public static string PathFor(string identity)
        {
            string explicitPath = Environment.GetEnvironmentVariable("TENSORSHARP_MOE_PROFILE");
            if (!string.IsNullOrWhiteSpace(explicitPath)) return explicitPath.Trim();
            string dir = DefaultDirectory();
            if (dir == null || identity == null) return null;
            var name = new char[identity.Length];
            for (int i = 0; i < identity.Length; i++)
                name[i] = char.IsLetterOrDigit(identity[i]) || identity[i] is '.' or '-' or '_' ? identity[i] : '_';
            return Path.Combine(dir, new string(name) + ".json");
        }

        /// <summary>Read the profile at <paramref name="path"/>; null when it is missing,
        /// unreadable, or describes another model or shape.</summary>
        public static MoeRoutingProfile TryLoad(string path, string identity, int layers, int experts)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
                var root = doc.RootElement;
                if (root.GetProperty("version").GetInt32() != FormatVersion
                    || root.GetProperty("layers").GetInt32() != layers
                    || root.GetProperty("experts").GetInt32() != experts)
                    return null;
                string fileIdentity = root.GetProperty("identity").GetString();
                // An explicitly named profile may come from another quantization of the
                // same model (same routed shape); a cached one must match exactly.
                bool explicitFile = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_MOE_PROFILE"));
                if (!explicitFile && !string.Equals(fileIdentity, identity, StringComparison.Ordinal))
                    return null;
                var profile = new MoeRoutingProfile(identity, layers, experts);
                int l = 0;
                foreach (var row in root.GetProperty("counts").EnumerateArray())
                {
                    if (l >= layers || row.GetArrayLength() != experts) return null;
                    int e = 0;
                    foreach (var v in row.EnumerateArray()) profile._counts[l][e++] = v.GetUInt64();
                    l++;
                }
                return l == layers ? profile : null;
            }
            catch (Exception ex) when (ex is IOException or JsonException or InvalidOperationException
                or KeyNotFoundException or FormatException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        /// <summary>Write atomically (temp file + rename). Returns false on I/O failure.</summary>
        public bool TrySave(string path)
        {
            if (string.IsNullOrEmpty(path)) return false;
            string temp = path + "." + Environment.ProcessId.ToString(CultureInfo.InvariantCulture) + ".tmp";
            try
            {
                string dir = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                using (var stream = File.Create(temp))
                using (var w = new Utf8JsonWriter(stream))
                {
                    w.WriteStartObject();
                    w.WriteNumber("version", FormatVersion);
                    w.WriteString("identity", Identity);
                    w.WriteNumber("layers", Layers);
                    w.WriteNumber("experts", Experts);
                    w.WriteStartArray("counts");
                    foreach (ulong[] row in _counts)
                    {
                        w.WriteStartArray();
                        foreach (ulong v in row) w.WriteNumberValue(v);
                        w.WriteEndArray();
                    }
                    w.WriteEndArray();
                    w.WriteEndObject();
                }
                File.Move(temp, path, overwrite: true);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                try { File.Delete(temp); } catch (IOException) { } catch (UnauthorizedAccessException) { }
                return false;
            }
        }
    }
}
