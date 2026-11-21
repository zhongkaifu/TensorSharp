// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image LoRA plug-in configs ("type": "minimax-h3-image-lora", config/lora/minimax-h3-image-*.json) and the
// sampling contract a step-distillation (speed) adapter fixes (docs/design/diffusion-runtime.md section 5.3).
//
// A style adapter's config names its weights, and optionally the key grammar, alpha and strength. A speed adapter's config
// adds "sampling": the steps, the sigmas (steps + 1, ending in 0), the head policy (grid-interval: ComfyUI's, the base
// checkpoint's own; by-step: diffusers', head i at step i), guidance 1 and the generation floor in pixels (diffusers'
// min_generation_area: 1024 generates at the asked size). The safetensors files record none of that, so the config declares
// it and the model enforces it: a request may leave steps and guidance unset or repeat them; anything else is refused,
// never overridden (SamplingContract.Validate).
//
// This file reads configs only. The adapter weights are loaded by MiniMaxH3ImageLoraSet.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>The key grammar a plug-in config declares for its weights (<c>"key_format"</c>).</summary>
    public enum MiniMaxH3ImageLoraKeyFormat
    {
        /// <summary>Detected from the tensor names; a file that mixes grammars is refused.</summary>
        Auto,
        Diffusers,
        Comfy,
        Kohya,
    }

    /// <summary>A parsed <c>minimax-h3-image-lora</c> plug-in config.</summary>
    public sealed class MiniMaxH3ImageLoraConfig
    {
        /// <summary>The plug-in config's <c>"type"</c>.</summary>
        public const string Type = LoraCliFlags.MiniMaxH3ImagePluginType;

        /// <summary>The smallest generation floor diffusers accepts (min_generation_area): the asked size.</summary>
        public const long MinGenerationAreaLow = 1024;

        /// <summary>The largest generation floor diffusers accepts: 4096 x 4096.</summary>
        public const long MinGenerationAreaHigh = 4096L * 4096;

        internal const string GridIntervalHeads = "grid-interval";
        internal const string ByStepHeads = "by-step";
        internal const int MaxSteps = 100;

        private static readonly string[] TopLevelKeys = { "type", "name", "variables", "weights", "key_format", "alpha", "scale", "sampling" };
        private static readonly string[] SamplingKeys = { "steps", "sigmas", "heads", "cfg", "min_generation_area" };

        private static readonly JsonDocumentOptions Options = new()
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };

        private MiniMaxH3ImageLoraConfig(string path, string name, MiniMaxH3ImageLoraKeyFormat keyFormat, float? alpha, float? scale,
            string weightsFileName, SamplingContract contract)
        {
            Path = path;
            Name = name;
            KeyFormat = keyFormat;
            Alpha = alpha;
            Scale = scale;
            WeightsFileName = weightsFileName;
            Contract = contract;
        }

        /// <summary>The config's full path.</summary>
        public string Path { get; }

        /// <summary>The plug-in's display name, e.g. "pDMD 2-step (linoyts)".</summary>
        public string Name { get; }

        /// <summary>The key grammar the weights are in (<see cref="MiniMaxH3ImageLoraKeyFormat.Auto"/>: detected).</summary>
        public MiniMaxH3ImageLoraKeyFormat KeyFormat { get; }

        /// <summary>The LoRA alpha the config declares, or null to take it from the file (metadata, or per-module tensors), else
        /// the rank.</summary>
        public float? Alpha { get; }

        /// <summary>The plug-in's default strength, or null for 1.</summary>
        public float? Scale { get; }

        /// <summary>The file name of the plug-in's <c>"weights"</c> entry: the adapter this config was written for.</summary>
        public string WeightsFileName { get; }

        /// <summary>A speed adapter's sampling contract, or null for a style adapter.</summary>
        public SamplingContract Contract { get; }

        /// <summary>Read and validate a plug-in config. Reads the config only: the weights entry is resolved to its path, never
        /// downloaded.</summary>
        /// <exception cref="FileNotFoundException">The config does not exist.</exception>
        /// <exception cref="InvalidDataException">The config is malformed; the message names the file and the key.</exception>
        public static MiniMaxH3ImageLoraConfig Load(string path)
        {
            ArgumentNullException.ThrowIfNull(path);
            if (!File.Exists(path))
                throw new FileNotFoundException($"LoRA config '{path}' does not exist.", path);
            string full = System.IO.Path.GetFullPath(path);
            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(File.ReadAllText(full), Options);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException($"LoRA config '{full}' is not valid JSON: {e.Message}", e);
            }
            using (doc)
                return Parse(doc.RootElement, full);
        }

        private static MiniMaxH3ImageLoraConfig Parse(JsonElement root, string path)
        {
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"{path}: a {Type} config must be a JSON object, not {root.ValueKind}.");
            MiniMaxH3LoraConfig.CheckKeys(root, path, "", TopLevelKeys, Type);

            string type = MiniMaxH3LoraConfig.RequiredString(root, path, "type", "type");
            if (type != Type)
                throw MiniMaxH3LoraConfig.Invalid(path, "type", $"must be \"{Type}\", not \"{type}\"");
            string name = MiniMaxH3LoraConfig.RequiredString(root, path, "name", "name");

            if (root.TryGetProperty("variables", out var variables))
            {
                if (variables.ValueKind != JsonValueKind.Object)
                    throw MiniMaxH3LoraConfig.Invalid(path, "variables", "must be an object of names");
                foreach (var v in variables.EnumerateObject())
                    if (v.Value.ValueKind is not (JsonValueKind.String or JsonValueKind.Number))
                        throw MiniMaxH3LoraConfig.Invalid(path, "variables." + v.Name, "must be a string or a number");
            }

            MiniMaxH3LoraConfig.CheckWeightsEntry(MiniMaxH3LoraConfig.Required(root, path, "weights", "weights"), path);

            var keyFormat = MiniMaxH3ImageLoraKeyFormat.Auto;
            if (root.TryGetProperty("key_format", out _))
            {
                string format = MiniMaxH3LoraConfig.RequiredString(root, path, "key_format", "key_format");
                keyFormat = format switch
                {
                    "auto" => MiniMaxH3ImageLoraKeyFormat.Auto,
                    "diffusers" => MiniMaxH3ImageLoraKeyFormat.Diffusers,
                    "comfy" => MiniMaxH3ImageLoraKeyFormat.Comfy,
                    "kohya" => MiniMaxH3ImageLoraKeyFormat.Kohya,
                    _ => throw MiniMaxH3LoraConfig.Invalid(path, "key_format",
                        $"must be \"auto\", \"diffusers\", \"comfy\" or \"kohya\", not \"{format}\""),
                };
            }

            float? alpha = null;
            if (root.TryGetProperty("alpha", out var alphaElement))
            {
                float a = MiniMaxH3LoraConfig.FiniteNumber(alphaElement, path, "alpha");
                if (!(a > 0f))
                    throw MiniMaxH3LoraConfig.Invalid(path, "alpha", $"must be positive, not {alphaElement.GetRawText()}");
                alpha = a;
            }

            float? scale = null;
            if (root.TryGetProperty("scale", out var scaleElement))
            {
                float s = MiniMaxH3LoraConfig.FiniteNumber(scaleElement, path, "scale");
                if (s < 0f)
                    throw MiniMaxH3LoraConfig.Invalid(path, "scale", $"must be a strength >= 0, not {scaleElement.GetRawText()}");
                scale = s;
            }

            SamplingContract contract = null;
            if (root.TryGetProperty("sampling", out var sampling))
                contract = ParseSampling(sampling, path, name);

            string weightsPath;
            try
            {
                weightsPath = ConfigFileArgs.ResolveFileEntryPath(path, "weights");
            }
            catch (ArgumentException e)
            {
                throw new InvalidDataException($"{path}: \"weights\" cannot be resolved: {e.Message}", e);
            }
            return new MiniMaxH3ImageLoraConfig(path, name, keyFormat, alpha, scale, System.IO.Path.GetFileName(weightsPath), contract);
        }

        private static SamplingContract ParseSampling(JsonElement sampling, string path, string name)
        {
            if (sampling.ValueKind != JsonValueKind.Object)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling", "must be an object (steps, sigmas, heads, cfg, min_generation_area)");
            MiniMaxH3LoraConfig.CheckKeys(sampling, path, "sampling.", SamplingKeys, Type);

            var stepsElement = MiniMaxH3LoraConfig.Required(sampling, path, "steps", "sampling.steps");
            if (stepsElement.ValueKind != JsonValueKind.Number || !stepsElement.TryGetInt32(out int steps) || steps < 1 || steps > MaxSteps)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.steps", $"must be an integer from 1 to {MaxSteps}, not {stepsElement.GetRawText()}");

            var sigmasElement = MiniMaxH3LoraConfig.Required(sampling, path, "sigmas", "sampling.sigmas");
            if (sigmasElement.ValueKind != JsonValueKind.Array)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.sigmas", "must be an array of numbers");
            var sigmas = new List<float>();
            foreach (var v in sigmasElement.EnumerateArray())
                sigmas.Add(MiniMaxH3LoraConfig.FiniteNumber(v, path, "sampling.sigmas"));
            if (sigmas.Count != steps + 1)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.sigmas",
                    $"must list steps + 1 = {steps + 1} values (the last 0), not {sigmas.Count}");
            if (sigmas[^1] != 0f)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.sigmas", "must end in 0");
            if (!(sigmas[0] <= 1f))
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.sigmas", "must start at 1 or below");
            for (int i = 1; i < sigmas.Count; i++)
                if (!(sigmas[i] < sigmas[i - 1]))
                    throw MiniMaxH3LoraConfig.Invalid(path, "sampling.sigmas", "must decrease");

            string heads = MiniMaxH3LoraConfig.RequiredString(sampling, path, "heads", "sampling.heads");
            HeadPolicy policy = heads switch
            {
                GridIntervalHeads => HeadPolicy.GridInterval,
                ByStepHeads => HeadPolicy.ByStep,
                _ => throw MiniMaxH3LoraConfig.Invalid(path, "sampling.heads",
                    $"must be \"{GridIntervalHeads}\" (the base checkpoint's, ComfyUI's) or \"{ByStepHeads}\" (diffusers': head i at step i), not \"{heads}\""),
            };

            var cfgElement = MiniMaxH3LoraConfig.Required(sampling, path, "cfg", "sampling.cfg");
            float cfg = MiniMaxH3LoraConfig.FiniteNumber(cfgElement, path, "sampling.cfg");
            if (cfg != 1f)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.cfg", $"must be 1 (MiniMax-H3-Image is guidance-distilled), not {cfgElement.GetRawText()}");

            var floorElement = MiniMaxH3LoraConfig.Required(sampling, path, "min_generation_area", "sampling.min_generation_area");
            if (floorElement.ValueKind != JsonValueKind.Number || !floorElement.TryGetInt64(out long floor) ||
                floor < MinGenerationAreaLow || floor > MinGenerationAreaHigh)
                throw MiniMaxH3LoraConfig.Invalid(path, "sampling.min_generation_area",
                    $"must be an integer number of pixels from {MinGenerationAreaLow} (generate at the asked size) to {MinGenerationAreaHigh}, " +
                    $"not {floorElement.GetRawText()}");

            return new SamplingContract
            {
                Name = name,
                Sigmas = sigmas.ToArray(),
                Heads = policy,
                Cfg = cfg,
                MinGenerationArea = floor,
            };
        }

        // ---- the adapter tie and the strength ---------------------------------------------

        /// <summary>
        /// Refuse weights this config was not written for: a config used as <c>--lora-config</c> beside <c>--lora weights</c>
        /// must name those weights in its <c>"weights"</c> entry, or a speed adapter's contract would land on other weights.
        /// </summary>
        /// <exception cref="ArgumentException">The file names differ.</exception>
        public void CheckWeights(string weightsPath)
        {
            ArgumentNullException.ThrowIfNull(weightsPath);
            string actual = System.IO.Path.GetFileName(weightsPath);
            if (!string.Equals(actual, WeightsFileName, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException(
                    $"{LoraCliFlags.ConfigFlag} {System.IO.Path.GetFileName(Path)} is the {Name} plug-in for {WeightsFileName}; " +
                    $"{LoraCliFlags.LoraFlag} names {actual}. Pass the plug-in alone with {LoraCliFlags.LoraFlag} " +
                    "(it names its own weights), or the weights it was written for.");
        }

        /// <summary>Refuse a negative or non-finite strength. 0 is legal: the terms stay with zero up factors (the identity).</summary>
        /// <exception cref="ArgumentException">The strength is negative or not finite.</exception>
        internal static void CheckStrength(string name, string file, float strength, bool speed)
        {
            if (!float.IsFinite(strength))
                throw new ArgumentException($"{name} ({file}): strength {MiniMaxH3LoraConfig.Num(strength)} is not a finite number.");
            if (strength < 0f)
                throw new ArgumentException(
                    $"{name} ({file}): strength {MiniMaxH3LoraConfig.Num(strength)} is negative" +
                    (speed ? "; its schedule was distilled at strength 1" : "") + " (0 keeps the adapter's terms with zero up factors).");
        }

        // ---- the hosts' checks, from the configs alone ------------------------------------------

        /// <summary>
        /// The configs <paramref name="specs"/> declare (null where a spec is bare weights), read without the weights and
        /// without a download, after the checks every host runs before it fetches anything: each config is this type and,
        /// beside <c>--lora weights</c>, names those weights; each strength is legal; at most one is a speed adapter.
        /// </summary>
        /// <exception cref="ArgumentException">A refusal, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named config is missing.</exception>
        public static IReadOnlyList<MiniMaxH3ImageLoraConfig> ReadConfigs(IReadOnlyList<LoraSpec> specs)
        {
            var result = new List<MiniMaxH3ImageLoraConfig>();
            if (specs == null) return result;
            try
            {
                MiniMaxH3ImageLoraConfig speed = null;
                foreach (LoraSpec spec in specs)
                {
                    ArgumentNullException.ThrowIfNull(spec, nameof(specs));
                    string manifest = ManifestOf(spec);
                    string file = System.IO.Path.GetFileName(spec.Path);
                    if (manifest == null)
                    {
                        result.Add(null);
                        CheckStrength(file, file, spec.Scale ?? 1f, speed: false);
                        continue;
                    }
                    string declared = LoraCliFlags.DeclaredPluginType(new LoraSpec(manifest));
                    if (declared != Type)
                        throw new ArgumentException(ForeignConfigRefusal(file, declared));
                    var config = Load(manifest);
                    bool weightsForm = !spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase);
                    if (weightsForm) config.CheckWeights(spec.Path);
                    CheckStrength(config.Name, weightsForm ? file : config.WeightsFileName, spec.Scale ?? config.Scale ?? 1f, config.Contract != null);
                    if (config.Contract != null)
                    {
                        if (speed != null) throw new ArgumentException(TwoSpeedAdaptersRefusal(speed.Name, config.Name));
                        speed = config;
                    }
                    result.Add(config);
                }
            }
            catch (Exception e) when (e is InvalidDataException or JsonException)
            {
                throw new ArgumentException(e.Message, e);
            }
            return result;
        }

        /// <summary>
        /// The hosts' startup check of their sampling flags against the speed adapter among <paramref name="specs"/>, if any
        /// (<see cref="ReadConfigs"/> first): <paramref name="steps"/> and <paramref name="cfg"/> (0: the flag was not given)
        /// may only repeat the contract's. Returns the contract, or null.
        /// </summary>
        /// <exception cref="ArgumentException">A refusal, with the canonical text.</exception>
        public static SamplingContract CheckHostSettings(IReadOnlyList<LoraSpec> specs, int steps, string stepsFlag, float cfg, string cfgFlag)
        {
            SamplingContract contract = ReadConfigs(specs).FirstOrDefault(c => c?.Contract != null)?.Contract;
            if (contract == null) return null;
            if (steps != 0 && steps != contract.Steps)
                throw new ArgumentException(
                    $"{stepsFlag} {steps} contradicts {contract.Name}, which samples exactly {contract.Steps} steps on its own schedule. " +
                    $"Drop {stepsFlag} or pass {contract.Steps}.");
            if (cfg != 0f && cfg != contract.Cfg)
                throw new ArgumentException(
                    $"{cfgFlag} {MiniMaxH3LoraConfig.Num(cfg)} contradicts {contract.Name}, which samples at guidance " +
                    $"{MiniMaxH3LoraConfig.Num(contract.Cfg)}. Drop {cfgFlag} or pass {MiniMaxH3LoraConfig.Num(contract.Cfg)}.");
            return contract;
        }

        /// <summary>The plug-in config a spec is selected by (its <c>--lora</c> .json, else its <c>--lora-config</c> .json), or
        /// null for bare weights. A <c>--lora-config</c> that is some other JSON (a PEFT adapter_config.json) is refused here: this
        /// family reads its own configs only.</summary>
        internal static string ManifestOf(LoraSpec spec)
        {
            if (spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                if (!File.Exists(spec.Path))
                    throw new FileNotFoundException($"{LoraCliFlags.LoraFlag} file not found: {spec.Path}", spec.Path);
                return spec.Path;
            }
            if (spec.ConfigPath == null) return null;
            if (!File.Exists(spec.ConfigPath))
                throw new FileNotFoundException($"{LoraCliFlags.ConfigFlag} file not found: {spec.ConfigPath}", spec.ConfigPath);
            if (!spec.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ||
                LoraCliFlags.DeclaredPluginType(new LoraSpec(spec.ConfigPath)) != Type)
                throw new ArgumentException(
                    $"{LoraCliFlags.ConfigFlag} {System.IO.Path.GetFileName(spec.ConfigPath)}: MiniMax-H3-Image reads its own {Type} " +
                    $"configs only (a PEFT or third-party config is not one). Drop {LoraCliFlags.ConfigFlag} to apply the weights " +
                    "as a style adapter, or pass a " + Type + " config.");
            return spec.ConfigPath;
        }

        // ---- canonical refusal texts --------------------------------------------------------------

        internal static string ForeignConfigRefusal(string file, string declared) =>
            $"{LoraCliFlags.LoraFlag} {file} is a {declared ?? "non-plug-in"} config; MiniMax-H3-Image ({MiniMaxH3ImageFamily.ArchitectureId}) " +
            $"takes {Type} plug-ins, or bare weights as a style adapter.";

        internal static string TwoSpeedAdaptersRefusal(string a, string b) =>
            $"Two speed adapters ({a}, {b}) each fix the sampling schedule; use one.";

        /// <summary>The spelling of a head policy in a config.</summary>
        internal static string Spell(HeadPolicy heads) => heads switch
        {
            HeadPolicy.ByStep => ByStepHeads,
            HeadPolicy.GridInterval => GridIntervalHeads,
            _ => "none",
        };

        /// <summary>One phrase for logs: "2 steps, sigmas [1, 0.92308, 0], by-step heads, cfg 1, floor 1024 px".</summary>
        internal static string Describe(SamplingContract c) =>
            $"{c.Steps} steps, sigmas [{string.Join(", ", c.Sigmas.Select(s => s.ToString("0.#####", CultureInfo.InvariantCulture)))}], " +
            $"{Spell(c.Heads)} heads, cfg {MiniMaxH3LoraConfig.Num(c.Cfg)}, generation floor {c.MinGenerationArea} px";
    }
}
