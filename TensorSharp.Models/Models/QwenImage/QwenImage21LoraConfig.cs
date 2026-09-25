// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TensorSharp.Models.QwenImage;

/// <summary>How a LoRA recipe's sigma nodes become the sampler's schedule.</summary>
internal enum QwenImage21SigmaShift
{
    /// <summary>The nodes are the schedule (a terminal 0 is appended when missing).</summary>
    None,
    /// <summary>The nodes are pre-shift positions t: sigma = e^mu / (e^mu + 1/t - 1) with the
    /// checkpoint's resolution-dependent mu and no terminal stretch, then a terminal 0.
    /// This is diffusers' QwenImage21Pipeline(sigmas=nodes) with shift_terminal=None.</summary>
    Dynamic,
}

/// <summary>
/// The sampling contract that comes with a LoRA, typically a step-distillation adapter
/// trained for a fixed schedule. Explicit host settings (steps, CFG) still win; a step
/// count the recipe has no schedule for is refused rather than silently resampled.
/// </summary>
internal sealed class QwenImage21LoraRecipe
{
    internal string Source { get; init; } = "";
    internal int DefaultSteps { get; init; }
    /// <summary>Sigma nodes per supported step count (without the terminal 0).</summary>
    internal IReadOnlyDictionary<int, float[]> Nodes { get; init; } = new Dictionary<int, float[]>();
    internal QwenImage21SigmaShift Shift { get; init; } = QwenImage21SigmaShift.None;
    internal float? Cfg { get; init; }
    /// <summary>Round the DiT timestep as a bf16 pipeline does (sigma*1000 to bf16, /1000 in bf16).</summary>
    internal bool TimestepBf16 { get; init; }

    /// <summary>True when the recipe fixes the sigma schedule; otherwise it only sets defaults
    /// (steps, CFG) and the checkpoint's own schedule is used.</summary>
    internal bool HasSchedule => Nodes.Count > 0;

    /// <summary>The steps+1 sigmas for <paramref name="steps"/>, ending in 0.</summary>
    internal float[] Sigmas(int steps, int imageTokens)
    {
        if (!Nodes.TryGetValue(steps, out var nodes))
            throw new ArgumentException(
                $"The LoRA sampling recipe from {Source} defines schedules for {string.Join(", ", Nodes.Keys.OrderBy(k => k))} " +
                $"step(s), not {steps}. Use one of those step counts, or omit --diffusion-steps / \"steps\" to use its default ({DefaultSteps}).");
        var result = new float[steps + 1];
        double mu = 0.5 + (imageTokens - 256) * (0.9 - 0.5) / (8192 - 256);
        double exp = Math.Exp(mu);
        for (int i = 0; i < steps; i++)
        {
            double t = nodes[i];
            result[i] = Shift == QwenImage21SigmaShift.Dynamic ? (float)(exp / (exp + (1 / t - 1))) : (float)t;
        }
        result[steps] = 0f;
        return result;
    }

    internal string Describe(int steps, int imageTokens) =>
        $"{steps} steps on {(Shift == QwenImage21SigmaShift.Dynamic ? "shifted" : "fixed")} sigmas " +
        $"[{string.Join(", ", Sigmas(steps, imageTokens).Select(s => s.ToString("0.#####", CultureInfo.InvariantCulture)))}]" +
        (TimestepBf16 ? ", bf16 timesteps" : "");

    internal static float[] ValidateNodes(IReadOnlyList<float> nodes, string where)
    {
        var list = nodes.ToList();
        if (list.Count > 1 && list[^1] == 0f) list.RemoveAt(list.Count - 1);
        if (list.Count == 0 || list.Any(v => !float.IsFinite(v) || v <= 0f || v > 1f))
            throw new InvalidDataException($"{where}: sigma nodes must lie in (0, 1], optionally followed by a terminal 0.");
        for (int i = 1; i < list.Count; i++)
            if (!(list[i] < list[i - 1]))
                throw new InvalidDataException($"{where}: sigma nodes must strictly decrease.");
        return list.ToArray();
    }
}

/// <summary>
/// A LoRA's companion configuration. Three formats are recognized by content:
/// TensorSharp's plug-in config (<c>"type": "qwen-image-2.1-lora"</c>: default strength,
/// alpha and a sampling recipe), a PEFT <c>adapter_config.json</c> (<c>lora_alpha</c>, <c>r</c>,
/// <c>use_rslora</c>, <c>alpha_pattern</c>) and a VideoX-Fun PDD <c>pdd_config.json</c>
/// (alpha, the fixed sigma grid and the replaced full parameters).
/// </summary>
internal sealed class QwenImage21LoraConfig
{
    internal const string TensorSharpType = "qwen-image-2.1-lora";

    internal string Path { get; init; } = "";
    internal string Format { get; init; } = "";
    internal float? Scale { get; init; }
    internal float? Alpha { get; init; }
    /// <summary>rsLoRA (alpha / sqrt(rank)); null when the config does not say, so a file's own metadata decides.</summary>
    internal bool? UseRsLora { get; init; }
    internal IReadOnlyDictionary<string, float> AlphaPattern { get; init; } = new Dictionary<string, float>();
    internal QwenImage21LoraRecipe Recipe { get; init; }
    // VideoX-Fun parallel decoding distillation.
    internal bool IsPdd { get; init; }
    internal int PddSteps { get; init; }
    internal HashSet<string> PddFullParameters { get; init; } = new(StringComparer.Ordinal);

    private static readonly JsonDocumentOptions Options = new()
    {
        CommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    internal static QwenImage21LoraConfig Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA config '{path}' does not exist.", path);
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path), Options);
            return Parse(doc.RootElement, System.IO.Path.GetFullPath(path));
        }
        catch (Exception e) when (e is JsonException or FormatException or InvalidOperationException or KeyNotFoundException)
        {
            // A value of the wrong kind ("steps": "six") names the file, not a JSON API.
            throw new InvalidDataException($"LoRA config '{path}' is malformed: {e.Message}", e);
        }
    }

    internal static QwenImage21LoraConfig Parse(JsonElement root, string path)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"LoRA config '{path}' must be a JSON object.");
        if (root.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String)
        {
            if (type.GetString() != TensorSharpType)
                throw new InvalidDataException($"LoRA config '{path}' has type '{type.GetString()}'; expected '{TensorSharpType}'.");
            return ParseTensorSharp(root, path);
        }
        if (root.TryGetProperty("pdd_num_steps", out _))
            return ParsePdd(root, path);
        if (root.TryGetProperty("lora_alpha", out _) || root.TryGetProperty("peft_type", out _))
            return ParsePeft(root, path, "PEFT adapter_config.json");
        throw new InvalidDataException(
            $"LoRA config '{path}' is not a recognized format: expected a TensorSharp LoRA config (\"type\": \"{TensorSharpType}\"), " +
            "a PEFT adapter_config.json (lora_alpha, r) or a VideoX-Fun pdd_config.json (pdd_num_steps).");
    }

    /// <summary>The PEFT LoraConfig a diffusers saver embeds as safetensors metadata
    /// (<c>lora_adapter_metadata</c>), whose keys carry a component prefix such as
    /// <c>transformer.</c>; the transformer's entries win over other components'.</summary>
    internal static QwenImage21LoraConfig FromAdapterMetadata(string json, string path)
    {
        JsonDocument doc;
        try { doc = JsonDocument.Parse(json, Options); }
        catch (JsonException e)
        {
            throw new InvalidDataException($"LoRA '{path}': its lora_adapter_metadata is not valid JSON ({e.Message}).", e);
        }
        using var _ = doc;
        if (doc.RootElement.ValueKind != JsonValueKind.Object) return null;
        // Group the keys by component: "transformer.lora_alpha" -> ("transformer", "lora_alpha").
        var groups = new Dictionary<string, Dictionary<string, JsonElement>>(StringComparer.Ordinal);
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            int dot = property.Name.LastIndexOf('.');
            string component = dot < 0 ? "" : property.Name.Substring(0, dot);
            string key = dot < 0 ? property.Name : property.Name.Substring(dot + 1);
            if (!groups.TryGetValue(component, out var group)) groups[component] = group = new Dictionary<string, JsonElement>();
            group[key] = property.Value.Clone();
        }
        var chosen = groups.TryGetValue("transformer", out var t) ? t : groups.TryGetValue("", out var bare) ? bare : groups.Values.FirstOrDefault();
        if (chosen == null || !chosen.ContainsKey("lora_alpha")) return null;
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var (key, value) in chosen) { writer.WritePropertyName(key); value.WriteTo(writer); }
            writer.WriteEndObject();
        }
        using var flat = JsonDocument.Parse(buffer.WrittenMemory);
        return ParsePeft(flat.RootElement, path, "PEFT metadata embedded in the safetensors file");
    }

    private static QwenImage21LoraConfig ParsePeft(JsonElement root, string path, string format)
    {
        var pattern = new Dictionary<string, float>(StringComparer.Ordinal);
        if (root.TryGetProperty("alpha_pattern", out var ap) && ap.ValueKind == JsonValueKind.Object)
            foreach (var p in ap.EnumerateObject()) pattern[p.Name] = p.Value.GetSingle();
        if (root.TryGetProperty("use_dora", out var dora) && dora.ValueKind == JsonValueKind.True)
            Console.WriteLine($"  [lora] {path}: the config says use_dora; the magnitudes are read from the file's dora_scale tensors.");
        return new QwenImage21LoraConfig
        {
            Path = path,
            Format = format,
            Alpha = root.TryGetProperty("lora_alpha", out var alpha) && alpha.ValueKind == JsonValueKind.Number ? alpha.GetSingle() : null,
            UseRsLora = ReadBool(root, "use_rslora"),
            AlphaPattern = pattern,
        };
    }

    private static QwenImage21LoraConfig ParsePdd(JsonElement root, string path)
    {
        int steps = root.GetProperty("pdd_num_steps").GetInt32();
        int block = root.TryGetProperty("pdd_block_size", out var b) ? b.GetInt32() : 1;
        if (block != 1)
            throw new NotSupportedException(
                $"'{path}': pdd_block_size {block} mixes several step heads per prediction; TensorSharp supports block size 1 (one head per step).");
        if (!root.TryGetProperty("pdd_sigmas", out var sigmas) || sigmas.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"'{path}' has no pdd_sigmas grid; this PDD checkpoint needs its trained sigmas.");
        var grid = sigmas.EnumerateArray().Select(s => s.GetSingle()).ToList();
        if (grid.Count != steps + 1 || grid[0] != 1f || grid[^1] != 0f)
            throw new InvalidDataException($"'{path}': pdd_sigmas must have pdd_num_steps + 1 entries from 1 to 0.");
        var full = new HashSet<string>(StringComparer.Ordinal);
        if (root.TryGetProperty("pdd_full_parameters", out var fp))
        {
            // VideoX-Fun writes either a JSON list or a Python list rendered as a string
            // ("['a.weight', 'b.weight']").
            IEnumerable<string> names = fp.ValueKind == JsonValueKind.Array
                ? fp.EnumerateArray().Select(e => e.GetString())
                : (fp.GetString() ?? "").Trim().TrimStart('[').TrimEnd(']')
                    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(n => n.Trim('\'', '"', ' '));
            foreach (var n in names) if (!string.IsNullOrEmpty(n)) full.Add(n);
        }
        float? alpha = root.TryGetProperty("lora_alpha", out var a) ? a.GetSingle()
            : root.TryGetProperty("network_alpha", out var na) ? na.GetSingle() : null;
        return new QwenImage21LoraConfig
        {
            Path = path,
            Format = "VideoX-Fun PDD pdd_config.json",
            Alpha = alpha,
            IsPdd = true,
            PddSteps = steps,
            PddFullParameters = full,
            Recipe = new QwenImage21LoraRecipe
            {
                Source = path,
                DefaultSteps = steps,
                Nodes = new Dictionary<int, float[]> { [steps] = QwenImage21LoraRecipe.ValidateNodes(grid, path) },
                // "Already shifted/stretched at conversion time; never shift twice."
                Shift = QwenImage21SigmaShift.None,
                Cfg = 1f,
                // The reference hook feeds ((sigma * 1000).to(bf16) / 1000) to the transformer.
                TimestepBf16 = true,
            },
        };
    }

    private static QwenImage21LoraConfig ParseTensorSharp(JsonElement root, string path)
    {
        QwenImage21LoraRecipe recipe = null;
        if (root.TryGetProperty("sampling", out var s) && s.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"{path}: \"sampling\" must be an object (steps, sigmas, shift, cfg, timestep).");
        if (root.TryGetProperty("sampling", out s))
        {
            var nodes = new Dictionary<int, float[]>();
            if (s.TryGetProperty("sigmas", out var sigmas))
            {
                if (sigmas.ValueKind == JsonValueKind.Array)
                {
                    var list = QwenImage21LoraRecipe.ValidateNodes(sigmas.EnumerateArray().Select(v => v.GetSingle()).ToList(), path);
                    nodes[list.Length] = list;
                }
                else if (sigmas.ValueKind == JsonValueKind.Object)
                {
                    foreach (var entry in sigmas.EnumerateObject())
                    {
                        if (!int.TryParse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int count) || count <= 0)
                            throw new InvalidDataException($"{path}: sampling.sigmas keys must be step counts, not '{entry.Name}'.");
                        var list = QwenImage21LoraRecipe.ValidateNodes(entry.Value.EnumerateArray().Select(v => v.GetSingle()).ToList(), path);
                        if (list.Length != count)
                            throw new InvalidDataException($"{path}: the {count}-step schedule has {list.Length} nodes.");
                        nodes[count] = list;
                    }
                }
                else throw new InvalidDataException($"{path}: sampling.sigmas must be an array or an object keyed by step count.");
            }
            int steps = s.TryGetProperty("steps", out var st) ? st.GetInt32() : nodes.Count == 1 ? nodes.Keys.First() : 0;
            if (steps <= 0)
                throw new InvalidDataException($"{path}: sampling.steps must name the default step count.");
            if (nodes.Count > 0 && !nodes.ContainsKey(steps))
                throw new InvalidDataException($"{path}: sampling.steps {steps} has no entry in sampling.sigmas.");
            var shift = (s.TryGetProperty("shift", out var sh) ? sh.GetString() : "none")?.Trim().ToLowerInvariant() switch
            {
                "none" or null => QwenImage21SigmaShift.None,
                "dynamic" => QwenImage21SigmaShift.Dynamic,
                var other => throw new InvalidDataException($"{path}: sampling.shift '{other}' is not one of none, dynamic."),
            };
            bool bf16 = (s.TryGetProperty("timestep", out var ts) ? ts.GetString() : "fp32")?.Trim().ToLowerInvariant() switch
            {
                "fp32" or "f32" or null => false,
                "bf16" => true,
                var other => throw new InvalidDataException($"{path}: sampling.timestep '{other}' is not one of fp32, bf16."),
            };
            float? cfg = s.TryGetProperty("cfg", out var c) ? c.GetSingle() : null;
            if (cfg is { } g && (!float.IsFinite(g) || g < 1f))
                throw new InvalidDataException($"{path}: sampling.cfg must be >= 1.");
            recipe = new QwenImage21LoraRecipe
            {
                Source = path,
                DefaultSteps = steps,
                Nodes = nodes,
                Shift = shift,
                Cfg = cfg,
                TimestepBf16 = bf16,
            };
        }
        return new QwenImage21LoraConfig
        {
            Path = path,
            Format = "TensorSharp LoRA config",
            Scale = ReadScale(root, path),
            Alpha = ReadAlpha(root),
            UseRsLora = ReadBool(root, "use_rslora"),
            Recipe = recipe,
        };
    }

    private static float? ReadScale(JsonElement root, string path)
    {
        if (!root.TryGetProperty("scale", out var s)) return null;
        float v = s.GetSingle();
        if (!float.IsFinite(v)) throw new InvalidDataException($"{path}: scale must be finite.");
        return v;
    }

    private static bool? ReadBool(JsonElement root, string key) =>
        root.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;

    private static float? ReadAlpha(JsonElement root) =>
        root.TryGetProperty("alpha", out var a) && a.ValueKind == JsonValueKind.Number ? a.GetSingle() : null;

    /// <summary>Alpha for <paramref name="module"/>: the PEFT alpha_pattern entry whose key is a
    /// suffix of the module path, else the config-wide alpha.</summary>
    internal float? AlphaFor(string module)
    {
        foreach (var (pattern, value) in AlphaPattern)
            if (module == pattern || module.EndsWith("." + pattern, StringComparison.Ordinal)) return value;
        return Alpha;
    }
}
