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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// One LoRA plug-in: a weights file (<c>.safetensors</c>), an optional strength and an
    /// optional companion config (a TensorSharp LoRA config with a sampling recipe, a PEFT
    /// <c>adapter_config.json</c> or a VideoX-Fun <c>pdd_config.json</c>).
    /// </summary>
    public sealed record LoraSpec(string Path, float? Scale = null, string? ConfigPath = null);

    /// <summary>
    /// The <c>--lora</c>, <c>--lora-scale</c> and <c>--lora-config</c> options shared by
    /// <c>TensorSharp.Cli</c> and <c>TensorSharp.Server</c>, and the environment channel
    /// (<see cref="EnvironmentVariable"/>) through which both hosts hand the parsed list to
    /// the model that applies them (Qwen-Image-2.1 or MiniMax-H3).
    /// </summary>
    /// <remarks>
    /// <c>--lora</c> may repeat; each one starts a new plug-in. <c>--lora-scale</c> and
    /// <c>--lora-config</c> bind to the closest preceding <c>--lora</c> and a later value
    /// replaces an earlier one, so a command-line <c>--lora-scale</c> after a config file's
    /// plug-in overrides that plug-in's strength. Either one without a preceding
    /// <c>--lora</c> is an error, never ignored.
    /// </remarks>
    public static class LoraCliFlags
    {
        public const string LoraFlag = "--lora";
        public const string ScaleFlag = "--lora-scale";
        public const string ConfigFlag = "--lora-config";

        /// <summary>JSON array of <c>{ "path", "scale", "config" }</c> objects read by the model that applies them.</summary>
        public const string EnvironmentVariable = "TS_LORAS";

        /// <summary>Every flag this class owns, for the usage pages and drift tests.</summary>
        public static readonly IReadOnlyList<string> Flags = new[] { LoraFlag, ScaleFlag, ConfigFlag };

        /// <summary>True when <paramref name="arg"/> is one of the LoRA flags (spaced or joined spelling).</summary>
        public static bool IsLoraFlag(string? arg)
        {
            if (arg == null) return false;
            int equals = arg.IndexOf('=');
            string name = equals >= 0 ? arg.Substring(0, equals) : arg;
            foreach (var flag in Flags)
                if (name.Equals(flag, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        /// <summary>
        /// Collect the LoRA plug-ins named in <paramref name="args"/>, in order. With
        /// <paramref name="remaining"/> non-null the other arguments are copied there so a
        /// caller can drop the LoRA flags before its own parsing.
        /// </summary>
        /// <exception cref="ArgumentException">A flag has no value, a scale is not a finite
        /// number, or a scale/config appears before any <c>--lora</c>.</exception>
        public static List<LoraSpec> Parse(IReadOnlyList<string> args, List<string>? remaining = null)
        {
            var result = new List<LoraSpec>();
            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i];
                string? value = null;
                string? flag = null;
                // Case-insensitive, like the server's option parser and the other shared flag
                // tables: a spelling the server accepts must never be silently left unapplied.
                foreach (var f in Flags)
                {
                    if (string.Equals(arg, f, StringComparison.OrdinalIgnoreCase))
                    {
                        if (i + 1 >= args.Count) throw new ArgumentException($"{f} requires a value.");
                        flag = f;
                        value = args[++i];
                        break;
                    }
                    if (arg.StartsWith(f + "=", StringComparison.OrdinalIgnoreCase))
                    {
                        flag = f;
                        value = arg.Substring(f.Length + 1);
                        break;
                    }
                }
                if (flag == null)
                {
                    remaining?.Add(arg);
                    continue;
                }
                if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException($"{flag} requires a value.");
                if (flag == LoraFlag)
                {
                    result.Add(new LoraSpec(value));
                    continue;
                }
                if (result.Count == 0)
                    throw new ArgumentException($"{flag} applies to the LoRA named by the preceding {LoraFlag}; none precedes it.");
                var last = result[^1];
                if (flag == ScaleFlag)
                {
                    if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float scale) || !float.IsFinite(scale))
                        throw new ArgumentException($"{ScaleFlag} expects a finite number, not '{value}'.");
                    result[^1] = last with { Scale = scale };
                }
                else
                {
                    result[^1] = last with { ConfigPath = value };
                }
            }
            return result;
        }

        /// <summary>The <c>"type"</c> of a Qwen-Image-2.1 LoRA plug-in config.</summary>
        public const string QwenImagePluginType = "qwen-image-2.1-lora";

        /// <summary>The <c>"type"</c> of a MiniMax-H3 LoRA plug-in config, which declares the
        /// checkpoint partition and the sampling contract of a step-distillation adapter.</summary>
        public const string MiniMaxH3PluginType = "minimax-h3-lora";

        /// <summary>Every TensorSharp plug-in config type, one per model family that takes LoRA plug-ins.</summary>
        public static readonly IReadOnlyList<string> PluginTypes = new[] { QwenImagePluginType, MiniMaxH3PluginType };

        /// <summary>
        /// The TensorSharp plug-in config that <paramref name="spec"/> is selected by:
        /// <see cref="LoraSpec.Path"/> when it is an existing <c>.json</c>, else
        /// <see cref="LoraSpec.ConfigPath"/> when it is an existing plug-in <c>.json</c>
        /// (one whose <c>"type"</c> is in <see cref="PluginTypes"/>), else null. Reads the
        /// manifest only; never downloads.
        /// </summary>
        /// <exception cref="ArgumentException">The manifest is not valid JSON.</exception>
        public static string? ManifestPath(LoraSpec spec)
        {
            ArgumentNullException.ThrowIfNull(spec);
            if (spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && File.Exists(spec.Path))
                return System.IO.Path.GetFullPath(spec.Path);
            if (spec.ConfigPath != null && spec.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase) &&
                File.Exists(spec.ConfigPath))
            {
                string config = System.IO.Path.GetFullPath(spec.ConfigPath);
                using var doc = ReadPlugin(config, out bool isPlugin);
                return isPlugin ? config : null;
            }
            return null;
        }

        /// <summary>
        /// The plug-in type <paramref name="spec"/> declares: the <c>"type"</c> of the
        /// <c>.json</c> named by <c>--lora</c>, or else by <c>--lora-config</c>, when it is one of
        /// <see cref="PluginTypes"/>; null for bare weights, a PEFT or PDD config, or any other
        /// JSON. Hosts and models read it to route a plug-in to the family that can apply it
        /// before anything is downloaded or loaded. Reads the manifest only; never downloads.
        /// </summary>
        /// <exception cref="FileNotFoundException">The <c>--lora</c> or <c>--lora-config</c> file is
        /// missing, with the text <see cref="Resolve"/> uses, so a typo'd manifest is reported as
        /// missing rather than as bare weights.</exception>
        /// <exception cref="ArgumentException">The manifest is not valid JSON.</exception>
        public static string? DeclaredPluginType(LoraSpec spec)
        {
            ArgumentNullException.ThrowIfNull(spec);
            if (!File.Exists(spec.Path))
                throw new FileNotFoundException($"{LoraFlag} file not found: {spec.Path}", spec.Path);
            if (spec.ConfigPath != null && !File.Exists(spec.ConfigPath))
                throw new FileNotFoundException($"{ConfigFlag} file not found: {spec.ConfigPath}", spec.ConfigPath);
            string? manifest = spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? spec.Path
                : spec.ConfigPath != null && spec.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? spec.ConfigPath
                : null;
            if (manifest == null) return null;
            using var doc = ReadPlugin(System.IO.Path.GetFullPath(manifest), out _);
            return PluginTypeOf(doc!.RootElement);
        }

        /// <summary>
        /// Check that every named file exists and return absolute paths. A <c>--lora</c> that
        /// names a TensorSharp plug-in config (a <c>.json</c> whose <c>"type"</c> is one of
        /// <see cref="PluginTypes"/>) is expanded here: its <c>"weights"</c> entry (downloaded and
        /// hash-checked when missing) becomes the weights, and the plug-in itself becomes the
        /// config, or its <c>"config"</c> entry when a Qwen-Image-2.1 plug-in forwards a
        /// third-party one (a PDD <c>pdd_config.json</c>).
        /// </summary>
        /// <exception cref="FileNotFoundException">A weights or config file is missing.</exception>
        /// <exception cref="ArgumentException">A plug-in config is malformed.</exception>
        public static List<LoraSpec> Resolve(IEnumerable<LoraSpec> specs)
        {
            var result = new List<LoraSpec>();
            foreach (var original in specs)
            {
                var spec = original;
                if (spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && File.Exists(spec.Path))
                    spec = ExpandPlugin(spec);
                else if (spec.ConfigPath != null && spec.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase) &&
                         File.Exists(spec.ConfigPath))
                    spec = ForwardPluginConfig(spec);
                if (!File.Exists(spec.Path))
                    throw new FileNotFoundException($"{LoraFlag} file not found: {spec.Path}", spec.Path);
                if (spec.ConfigPath != null && !File.Exists(spec.ConfigPath))
                    throw new FileNotFoundException($"{ConfigFlag} file not found: {spec.ConfigPath}", spec.ConfigPath);
                result.Add(spec with
                {
                    Path = System.IO.Path.GetFullPath(spec.Path),
                    ConfigPath = spec.ConfigPath == null ? null : System.IO.Path.GetFullPath(spec.ConfigPath),
                });
            }
            return result;
        }

        private static readonly JsonDocumentOptions PluginOptions =
            new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

        /// <summary>The plug-in's JSON root, or null when <paramref name="path"/> is some other JSON
        /// (a PEFT or PDD config). Malformed JSON is a configuration error naming the file.</summary>
        private static JsonDocument? ReadPlugin(string path, out bool isPlugin)
        {
            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(File.ReadAllText(path), PluginOptions);
            }
            catch (JsonException ex)
            {
                throw new ArgumentException($"LoRA config {path} is not valid JSON: {ex.Message}", ex);
            }
            isPlugin = PluginTypeOf(doc.RootElement) != null;
            return doc;
        }

        /// <summary>The root's <c>"type"</c> when it is one of <see cref="PluginTypes"/>, else null.</summary>
        private static string? PluginTypeOf(JsonElement root)
        {
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("type", out var type) ||
                type.ValueKind != JsonValueKind.String)
                return null;
            string? value = type.GetString();
            foreach (string known in PluginTypes)
                if (value == known) return known;
            return null;
        }

        /// <summary>A MiniMax-H3 plug-in is its own config; it never forwards a third-party one.</summary>
        private static void RefuseForwardingH3(JsonElement root, string manifest)
        {
            if (PluginTypeOf(root) == MiniMaxH3PluginType && root.TryGetProperty("config", out _))
                throw new ArgumentException(
                    $"{manifest}: a {MiniMaxH3PluginType} plug-in cannot forward a \"config\"; its own file is the config.");
        }

        /// <summary>A plug-in's own strength, when it states one.</summary>
        private static float? PluginScale(JsonElement root, string path)
        {
            if (!root.TryGetProperty("scale", out var scale)) return null;
            if (scale.ValueKind != JsonValueKind.Number || !scale.TryGetSingle(out float value) || !float.IsFinite(value))
                throw new ArgumentException($"{path}: \"scale\" must be a finite number.");
            return value;
        }

        /// <summary>
        /// A plug-in used as <c>--lora-config</c> that forwards a third-party config (its
        /// <c>"config"</c> entry, e.g. a pdd_config.json) hands that config on, with its strength;
        /// its own <c>"weights"</c> are not used because the <c>--lora</c> named them.
        /// </summary>
        private static LoraSpec ForwardPluginConfig(LoraSpec spec)
        {
            string manifest = System.IO.Path.GetFullPath(spec.ConfigPath!);
            bool forwards;
            float? scale = null;
            using (var doc = ReadPlugin(manifest, out bool isPlugin))
            {
                if (!isPlugin) return spec;
                RefuseForwardingH3(doc!.RootElement, manifest);
                forwards = doc.RootElement.TryGetProperty("config", out _);
                if (forwards) scale = PluginScale(doc.RootElement, manifest);
            }
            if (!forwards) return spec;
            string forwarded = ConfigFileArgs.ResolveFileEntry(manifest, "config")!;
            return spec with { ConfigPath = forwarded, Scale = spec.Scale ?? scale };
        }

        private static LoraSpec ExpandPlugin(LoraSpec spec)
        {
            string manifest = System.IO.Path.GetFullPath(spec.Path);
            using (var doc = ReadPlugin(manifest, out bool isPlugin))
            {
                var root = doc!.RootElement;
                if (!isPlugin)
                    throw new ArgumentException(
                        $"{LoraFlag} {manifest}: a .json LoRA must be a TensorSharp plug-in config " +
                        $"(\"type\": \"{QwenImagePluginType}\" or \"{MiniMaxH3PluginType}\"). " +
                        $"Pass the weights (.safetensors) with {LoraFlag} and a third-party config with {ConfigFlag}.");
                if (spec.ConfigPath != null)
                    throw new ArgumentException($"{LoraFlag} {manifest} is a plug-in config already; drop the {ConfigFlag} after it.");
                RefuseForwardingH3(root, manifest);
                if (root.TryGetProperty("config", out _))
                {
                    // The forwarded config (a pdd_config.json, say) becomes THE config, so
                    // settings it would shadow must live there; only the strength carries over.
                    foreach (var key in new[] { "sampling", "alpha", "use_rslora" })
                        if (root.TryGetProperty(key, out _))
                            throw new ArgumentException(
                                $"{manifest}: \"{key}\" cannot sit beside a forwarded \"config\"; set it in that config instead.");
                    if (spec.Scale == null)
                        spec = spec with { Scale = PluginScale(root, manifest) };
                }
            }
            string weights = ConfigFileArgs.ResolveFileEntry(manifest, "weights")
                ?? throw new ArgumentException($"{manifest}: a LoRA plug-in config names its weights with a \"weights\" entry.");
            string config = ConfigFileArgs.ResolveFileEntry(manifest, "config") ?? manifest;
            return spec with { Path = weights, ConfigPath = config };
        }

        /// <summary>Serialize for <see cref="EnvironmentVariable"/>.</summary>
        public static string ToJson(IEnumerable<LoraSpec> specs)
        {
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartArray();
                foreach (var spec in specs)
                {
                    writer.WriteStartObject();
                    writer.WriteString("path", spec.Path);
                    if (spec.Scale is { } scale) writer.WriteNumber("scale", scale);
                    if (spec.ConfigPath != null) writer.WriteString("config", spec.ConfigPath);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
            }
            return System.Text.Encoding.UTF8.GetString(stream.ToArray());
        }

        /// <summary>Parse the <see cref="EnvironmentVariable"/> format; empty or unset is no LoRA.</summary>
        /// <exception cref="ArgumentException">The value is not the expected JSON array.</exception>
        public static List<LoraSpec> FromJson(string? json)
        {
            var result = new List<LoraSpec>();
            if (string.IsNullOrWhiteSpace(json)) return result;
            try
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.ValueKind != JsonValueKind.Array) throw new FormatException("expected a JSON array");
                foreach (var item in doc.RootElement.EnumerateArray())
                {
                    string path = item.GetProperty("path").GetString() ?? throw new FormatException("path is null");
                    float? scale = item.TryGetProperty("scale", out var s) ? s.GetSingle() : null;
                    string? config = item.TryGetProperty("config", out var c) ? c.GetString() : null;
                    result.Add(new LoraSpec(path, scale, config));
                }
            }
            catch (Exception e) when (e is JsonException or FormatException or InvalidOperationException or KeyNotFoundException)
            {
                throw new ArgumentException($"{EnvironmentVariable} must be a JSON array of {{\"path\", \"scale\", \"config\"}} objects: {e.Message}");
            }
            return result;
        }

        /// <summary>One line per plug-in for startup logs.</summary>
        public static string Describe(IReadOnlyList<LoraSpec> specs)
        {
            var parts = new List<string>();
            foreach (var s in specs)
                parts.Add(System.IO.Path.GetFileName(s.Path) +
                    (s.Scale is { } scale ? $" x{scale.ToString("0.###", CultureInfo.InvariantCulture)}" : "") +
                    (s.ConfigPath != null ? $" (config {System.IO.Path.GetFileName(s.ConfigPath)})" : ""));
            return string.Join(", ", parts);
        }
    }
}
