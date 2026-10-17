// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 LoRA plug-in configs ("type": "minimax-h3-lora", config/lora/minimax-h3-*.json)
// and the sampling contract they lock.
//
// A step-distillation adapter (the LightX2V Turbo LoRAs for the FL2VA checkpoint) is only
// correct on the schedule it was distilled on: ModelTC/Minimax-H3-Turbo samples N Euler
// steps on the endpoint-inclusive linspace grid q_i = (N - i) / N, shifted by 6 for video
// and 3 for audio, at CFG 1 (diffusers scheduling_minimax_h3.py set_timesteps gives the
// same grid). The safetensors files record none of that, so the manifest declares it and
// every host and the model enforce it: 0 / unset means "use the recipe", the recipe's own
// value is accepted, and any other value is refused, never overridden.
//
// This file reads manifests only. The adapter weights are loaded by MiniMaxH3LoraSet.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>A parsed <c>minimax-h3-lora</c> plug-in config.</summary>
    public sealed class MiniMaxH3LoraConfig
    {
        /// <summary>The plug-in config's <c>"type"</c>.</summary>
        public const string Type = LoraCliFlags.MiniMaxH3PluginType;

        /// <summary>The only adapter key layout accepted: diffusers' MiniMax-H3 transformer names
        /// (<c>transformer_blocks.N.attn.to_q.lora_A.weight</c>, ...).</summary>
        public const string KeyFormat = "minimax-h3-diffusers";

        internal const string LinspaceSchedule = "linspace";
        internal const string EulerSampler = "euler";
        internal const string Fl2vaPartition = "fl2va";
        internal const string Ref2vaPartition = "ref2va";
        internal const int MaxSteps = 100;

        private static readonly string[] TopLevelKeys =
            { "type", "name", "variables", "weights", "partition", "key_format", "alpha", "scale", "sampling" };
        private static readonly string[] WeightsKeys = { "path", "urls", "sha256" };
        private static readonly string[] SamplingKeys = { "steps", "schedule", "video_shift", "audio_shift", "sampler", "cfg" };

        private static readonly JsonDocumentOptions Options = new()
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };

        private MiniMaxH3LoraConfig(string path, string name, MiniMaxH3Partition partition, float? alpha, float? scale,
            string weightsFileName, MiniMaxH3SamplingRecipe recipe)
        {
            Path = path;
            Name = name;
            Partition = partition;
            Alpha = alpha;
            Scale = scale;
            WeightsFileName = weightsFileName;
            Recipe = recipe;
        }

        /// <summary>The config's full path.</summary>
        public string Path { get; }

        /// <summary>The plug-in's display name, e.g. "LightX2V Turbo 8-step (FL2V v1.0, 768p)".</summary>
        public string Name { get; }

        /// <summary>The checkpoint the adapter was trained for. Always
        /// <see cref="MiniMaxH3Partition.FirstLastFrame"/>: a Ref2VA plug-in is refused.</summary>
        public MiniMaxH3Partition Partition { get; }

        /// <summary>The LoRA alpha the config declares, or null to take it from the weights' metadata.</summary>
        public float? Alpha { get; }

        /// <summary>The plug-in's default strength, or null for 1.</summary>
        public float? Scale { get; }

        /// <summary>The file name of the plug-in's <c>"weights"</c> entry: the adapter this
        /// config was written for.</summary>
        public string WeightsFileName { get; }

        /// <summary>The sampling contract the plug-in locks.</summary>
        public MiniMaxH3SamplingRecipe Recipe { get; }

        /// <summary>Read and validate a plug-in config. Reads the config only: the weights entry
        /// is resolved to its path, never downloaded.</summary>
        /// <exception cref="FileNotFoundException">The config does not exist.</exception>
        /// <exception cref="InvalidDataException">The config is malformed; the message names the
        /// file and the key.</exception>
        public static MiniMaxH3LoraConfig Load(string path)
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

        private static MiniMaxH3LoraConfig Parse(JsonElement root, string path)
        {
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"{path}: a {Type} config must be a JSON object, not {root.ValueKind}.");
            CheckKeys(root, path, "", TopLevelKeys);

            string type = RequiredString(root, path, "type", "type");
            if (type != Type)
                throw Invalid(path, "type", $"must be \"{Type}\", not \"{type}\"");
            string name = RequiredString(root, path, "name", "name");

            if (root.TryGetProperty("variables", out var variables))
            {
                if (variables.ValueKind != JsonValueKind.Object)
                    throw Invalid(path, "variables", "must be an object of names");
                foreach (var v in variables.EnumerateObject())
                    if (v.Value.ValueKind is not (JsonValueKind.String or JsonValueKind.Number))
                        throw Invalid(path, "variables." + v.Name, "must be a string or a number");
            }

            CheckWeightsEntry(Required(root, path, "weights", "weights"), path);

            string partition = RequiredString(root, path, "partition", "partition");
            if (partition == Ref2vaPartition)
                throw Invalid(path, "partition",
                    "is ref2va, and Ref2VA Turbo is not supported; only the MiniMax-H3 FL2VA checkpoint takes these plug-ins");
            if (partition != Fl2vaPartition)
                throw Invalid(path, "partition", $"must be \"{Fl2vaPartition}\", not \"{partition}\"");

            string keyFormat = RequiredString(root, path, "key_format", "key_format");
            if (keyFormat != KeyFormat)
                throw Invalid(path, "key_format",
                    $"must be \"{KeyFormat}\" (the Diffusers-format *_bf16.safetensors files), not \"{keyFormat}\"");

            float? alpha = null;
            if (root.TryGetProperty("alpha", out var alphaElement))
            {
                float a = FiniteNumber(alphaElement, path, "alpha");
                if (!(a > 0f))
                    throw Invalid(path, "alpha", $"must be positive, not {alphaElement.GetRawText()}");
                alpha = a;
            }

            float? scale = null;
            if (root.TryGetProperty("scale", out var scaleElement))
            {
                float s = FiniteNumber(scaleElement, path, "scale");
                if (s < 0f)
                    throw Invalid(path, "scale",
                        $"must be a strength >= 0, not {scaleElement.GetRawText()}; the Turbo schedule is distilled at strength 1");
                scale = s;
            }

            var sampling = Required(root, path, "sampling", "sampling");
            if (sampling.ValueKind != JsonValueKind.Object)
                throw Invalid(path, "sampling", "must be an object (steps, schedule, video_shift, audio_shift, sampler, cfg)");
            CheckKeys(sampling, path, "sampling.", SamplingKeys);

            var stepsElement = Required(sampling, path, "steps", "sampling.steps");
            if (stepsElement.ValueKind != JsonValueKind.Number || !stepsElement.TryGetInt32(out int steps) ||
                steps < 1 || steps > MaxSteps)
                throw Invalid(path, "sampling.steps", $"must be an integer from 1 to {MaxSteps}, not {stepsElement.GetRawText()}");

            string schedule = RequiredString(sampling, path, "schedule", "sampling.schedule");
            if (schedule != LinspaceSchedule)
                throw Invalid(path, "sampling.schedule",
                    $"must be \"{LinspaceSchedule}\" (the endpoint-inclusive grid the adapters were distilled on), not \"{schedule}\"");

            var videoShiftElement = Required(sampling, path, "video_shift", "sampling.video_shift");
            float videoShift = FiniteNumber(videoShiftElement, path, "sampling.video_shift");
            if (!(videoShift > 0f))
                throw Invalid(path, "sampling.video_shift", $"must be positive, not {videoShiftElement.GetRawText()}");

            var audioShiftElement = Required(sampling, path, "audio_shift", "sampling.audio_shift");
            float audioShift = FiniteNumber(audioShiftElement, path, "sampling.audio_shift");
            if (audioShift != MiniMaxH3Scheduler.AudioShift)
                throw Invalid(path, "sampling.audio_shift",
                    $"must be {Num(MiniMaxH3Scheduler.AudioShift)} (MiniMax-H3's audio stream is fixed at that shift), not {audioShiftElement.GetRawText()}");

            string sampler = RequiredString(sampling, path, "sampler", "sampling.sampler");
            if (sampler != EulerSampler)
                throw Invalid(path, "sampling.sampler", $"must be \"{EulerSampler}\", not \"{sampler}\"");

            var cfgElement = Required(sampling, path, "cfg", "sampling.cfg");
            float cfg = FiniteNumber(cfgElement, path, "sampling.cfg");
            if (cfg != 1f)
                throw Invalid(path, "sampling.cfg", $"must be 1 (MiniMax-H3 is CFG-distilled), not {cfgElement.GetRawText()}");

            // The file name is spelled literally (CheckWeightsEntry), so only the directory
            // depends on variables and the environment.
            string weightsPath;
            try
            {
                weightsPath = ConfigFileArgs.ResolveFileEntryPath(path, "weights");
            }
            catch (ArgumentException e)
            {
                throw new InvalidDataException($"{path}: \"weights\" cannot be resolved: {e.Message}", e);
            }
            string weightsFileName = System.IO.Path.GetFileName(weightsPath);

            var recipe = new MiniMaxH3SamplingRecipe(path, name, weightsFileName, steps, videoShift, audioShift, cfg);
            return new MiniMaxH3LoraConfig(path, name, MiniMaxH3Partition.FirstLastFrame, alpha, scale, weightsFileName, recipe);
        }

        /// <summary>The <c>"weights"</c> entry: a path, or a download object whose
        /// <c>path</c> names a <c>.safetensors</c> file literally.</summary>
        private static void CheckWeightsEntry(JsonElement weights, string path)
        {
            string key;
            string raw;
            if (weights.ValueKind == JsonValueKind.String)
            {
                key = "weights";
                raw = weights.GetString();
                if (string.IsNullOrWhiteSpace(raw))
                    throw Invalid(path, key, "must not be empty");
            }
            else if (weights.ValueKind == JsonValueKind.Object)
            {
                CheckKeys(weights, path, "weights.", WeightsKeys);
                key = "weights.path";
                raw = RequiredString(weights, path, "path", key);
                if (weights.TryGetProperty("urls", out var urls))
                {
                    if (urls.ValueKind != JsonValueKind.Array || urls.GetArrayLength() == 0)
                        throw Invalid(path, "weights.urls", "must be a non-empty array of URLs");
                    foreach (var url in urls.EnumerateArray())
                    {
                        if (url.ValueKind != JsonValueKind.String ||
                            !Uri.TryCreate(url.GetString(), UriKind.Absolute, out var uri) ||
                            (uri.Scheme != Uri.UriSchemeHttps && uri.Scheme != Uri.UriSchemeHttp))
                            throw Invalid(path, "weights.urls", $"must hold http(s) URLs, not {url.GetRawText()}");
                    }
                }
                if (weights.TryGetProperty("sha256", out var sha))
                {
                    if (sha.ValueKind != JsonValueKind.String || sha.GetString() is not { Length: 64 } hex ||
                        !hex.All(Uri.IsHexDigit))
                        throw Invalid(path, "weights.sha256", $"must be 64 hexadecimal digits, not {sha.GetRawText()}");
                }
            }
            else
            {
                throw Invalid(path, "weights", "must be a path or a download object { \"path\", \"urls\", \"sha256\" }");
            }

            // The file name ties a manifest to its adapter (CheckWeights), so it must not
            // change with the environment.
            string fileName = raw.Substring(raw.LastIndexOfAny(new[] { '/', '\\' }) + 1);
            if (fileName.Contains("${", StringComparison.Ordinal))
                throw Invalid(path, key, $"must spell the weights file name literally; '{fileName}' contains a ${{...}} reference");
            if (!fileName.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                throw Invalid(path, key, $"must name a .safetensors file, not '{fileName}'");
        }

        private static void CheckKeys(JsonElement obj, string path, string prefix, string[] allowed)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in obj.EnumerateObject())
            {
                string key = prefix + property.Name;
                if (!seen.Add(property.Name))
                    throw new InvalidDataException($"{path}: \"{key}\" appears twice.");
                if (prefix.Length == 0 && property.Name == "config")
                    throw new InvalidDataException(
                        $"{path}: \"config\" is not a {Type} key; a MiniMax-H3 plug-in is its own config and forwards none.");
                if (Array.IndexOf(allowed, property.Name) < 0)
                    throw new InvalidDataException(
                        $"{path}: unknown key \"{key}\"; {(prefix.Length == 0 ? $"a {Type} config" : $"\"{prefix.TrimEnd('.')}\"")} " +
                        $"takes {string.Join(", ", allowed)}.");
            }
        }

        private static JsonElement Required(JsonElement obj, string path, string name, string key) =>
            obj.TryGetProperty(name, out var value) ? value : throw new InvalidDataException($"{path}: \"{key}\" is required.");

        private static string RequiredString(JsonElement obj, string path, string name, string key)
        {
            var value = Required(obj, path, name, key);
            if (value.ValueKind != JsonValueKind.String)
                throw Invalid(path, key, $"must be a string, not {value.GetRawText()}");
            string text = value.GetString();
            if (string.IsNullOrWhiteSpace(text))
                throw Invalid(path, key, "must not be empty");
            return text;
        }

        private static float FiniteNumber(JsonElement value, string path, string key)
        {
            if (value.ValueKind != JsonValueKind.Number)
                throw Invalid(path, key, $"must be a number, not {value.GetRawText()}");
            if (!value.TryGetDouble(out double d) || !double.IsFinite(d) || !float.IsFinite((float)d))
                throw Invalid(path, key, $"must be a finite number, not {value.GetRawText()}");
            return (float)d;
        }

        private static InvalidDataException Invalid(string path, string key, string problem) =>
            new($"{path}: \"{key}\" {problem}.");

        internal static string Num(float value) => value.ToString(CultureInfo.InvariantCulture);

        // ---- the adapter tie and the strength ---------------------------------------------

        /// <summary>
        /// Refuse weights this config was not written for: a config used as
        /// <c>--lora-config</c> beside <c>--lora weights</c> must name those weights in its
        /// <c>"weights"</c> entry, or the 4-step adapter would run on the 8-step schedule.
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

        /// <summary>
        /// Refuse a negative strength (the reference's "--lora-scale must be a finite
        /// non-negative number"). 0 is legal: it keeps every term with zero up factors, the
        /// identity probe.
        /// </summary>
        /// <exception cref="ArgumentException">The strength is negative or not finite.</exception>
        public static void CheckStrength(MiniMaxH3LoraConfig c, string weightsFile, float strength)
        {
            ArgumentNullException.ThrowIfNull(c);
            string file = System.IO.Path.GetFileName(weightsFile ?? c.WeightsFileName);
            if (!float.IsFinite(strength))
                throw new ArgumentException($"{c.Name} ({file}): strength {Num(strength)} is not a finite number.");
            if (strength < 0f)
                throw new ArgumentException(
                    $"{c.Name} ({file}): strength {Num(strength)} is negative; the Turbo schedule is distilled at strength 1 " +
                    "(0 keeps the adapter's terms with zero up factors).");
        }

        // ---- the recipe a list of specs selects, from the configs alone -----------------------

        /// <summary>
        /// The recipe <paramref name="specs"/> select for a model of <paramref name="partition"/>,
        /// or null for none. Reads plug-in configs only (never weights, never a download), so a
        /// host can check a command line before it fetches anything. Each spec must be a
        /// <c>minimax-h3-lora</c> config (<c>--lora config.json</c>, or <c>--lora weights
        /// --lora-config config.json</c> naming those weights); at most one is allowed; its
        /// partition must match; its strength (<c>--lora-scale</c>, else the config's
        /// <c>"scale"</c>, else 1) must not be negative.
        /// </summary>
        /// <exception cref="ArgumentException">The specs are refused, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named config is missing.</exception>
        public static MiniMaxH3SamplingRecipe RecipeFor(IReadOnlyList<LoraSpec> specs, MiniMaxH3Partition partition) =>
            RecipeFor(specs, partition, modelName: null);

        internal static MiniMaxH3SamplingRecipe RecipeFor(IReadOnlyList<LoraSpec> specs, MiniMaxH3Partition partition, string modelName)
        {
            if (specs == null || specs.Count == 0) return null;
            try
            {
                var plugins = new List<(LoraSpec Spec, string Manifest)>();
                foreach (var spec in specs)
                {
                    ArgumentNullException.ThrowIfNull(spec, nameof(specs));
                    string declared = DeclaredType(spec, out string manifest);
                    string file = System.IO.Path.GetFileName(spec.Path);
                    if (declared == null)
                        throw new ArgumentException(BareWeightsRefusal(file));
                    if (declared != Type)
                        throw new ArgumentException(ForeignTypeRefusal(file, declared, modelName ?? "MiniMax-H3",
                            MiniMaxH3Model.ArchitectureId, Type));
                    plugins.Add((spec, manifest));
                }
                if (plugins.Count > 1)
                    throw new ArgumentException(TwoPluginsRefusal(
                        System.IO.Path.GetFileName(plugins[0].Spec.Path), System.IO.Path.GetFileName(plugins[1].Spec.Path)));

                var (only, manifestPath) = plugins[0];
                var config = Load(manifestPath);
                if (partition != config.Partition)
                    throw new ArgumentException(PartitionRefusal(System.IO.Path.GetFileName(only.Path), modelName));
                bool weightsForm = !only.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase);
                if (weightsForm)
                    config.CheckWeights(only.Path);
                CheckStrength(config, weightsForm ? only.Path : config.WeightsFileName, only.Scale ?? config.Scale ?? 1f);
                return config.Recipe;
            }
            catch (Exception e) when (e is InvalidDataException or JsonException)
            {
                throw new ArgumentException(e.Message, e);
            }
        }

        /// <summary>
        /// The host's startup check: <see cref="RecipeFor(IReadOnlyList{LoraSpec}, MiniMaxH3Partition)"/>
        /// for the model at <paramref name="modelPath"/> (its partition from the file name; FL2VA
        /// when unknown), then the host's sampling flags against the recipe. Returns the recipe,
        /// or null when <paramref name="specs"/> is empty.
        /// </summary>
        /// <exception cref="ArgumentException">A refusal, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named config is missing.</exception>
        public static MiniMaxH3SamplingRecipe CheckHostSettings(IReadOnlyList<LoraSpec> specs, string? modelPath, MiniMaxH3HostSampling s)
        {
            ArgumentNullException.ThrowIfNull(s);
            var partition = modelPath != null ? MiniMaxH3Config.PartitionFromFileName(modelPath) : MiniMaxH3Partition.FirstLastFrame;
            var recipe = RecipeFor(specs, partition, modelPath != null ? System.IO.Path.GetFileName(modelPath) : null);
            recipe?.CheckHost(s);
            return recipe;
        }

        /// <summary>The plug-in type <paramref name="spec"/> declares and the config that declares
        /// it, reading that config alone: unlike <see cref="LoraCliFlags.DeclaredPluginType"/>,
        /// the weights of a <c>--lora weights --lora-config config.json</c> spec need not exist yet.</summary>
        private static string DeclaredType(LoraSpec spec, out string manifest)
        {
            manifest = null;
            if (spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                if (!File.Exists(spec.Path))
                    throw new FileNotFoundException($"{LoraCliFlags.LoraFlag} file not found: {spec.Path}", spec.Path);
                manifest = spec.Path;
            }
            else if (spec.ConfigPath != null)
            {
                if (!File.Exists(spec.ConfigPath))
                    throw new FileNotFoundException($"{LoraCliFlags.ConfigFlag} file not found: {spec.ConfigPath}", spec.ConfigPath);
                if (spec.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                    manifest = spec.ConfigPath;
            }
            return manifest == null ? null : LoraCliFlags.DeclaredPluginType(new LoraSpec(manifest));
        }

        // ---- canonical refusal texts shared with the hosts' preflight ------------------------

        /// <summary>Bare weights (or weights with a PEFT config) offered to MiniMax-H3.</summary>
        internal static string BareWeightsRefusal(string file) =>
            $"{LoraCliFlags.LoraFlag} {file}: MiniMax-H3 takes LoRA plug-ins only as a {Type} config " +
            "(for example config/lora/minimax-h3-fl2v-turbo-8step.json), which declares the checkpoint partition " +
            "and the sampling contract. Pass that config instead of the bare weights.";

        /// <summary>A plug-in of another family's type.</summary>
        internal static string ForeignTypeRefusal(string file, string declared, string model, string arch, string expected) =>
            $"{LoraCliFlags.LoraFlag} {file} is a {declared} plug-in; {model} ({arch}) takes {expected} plug-ins only.";

        internal static string TwoPluginsRefusal(string a, string b) =>
            $"Two MiniMax-H3 plug-ins ({a}, {b}) each fix the sampling schedule; use one.";

        internal static string PartitionRefusal(string file, string model) =>
            $"LoRA plug-in {file} is for the MiniMax-H3 FL2VA checkpoint; {model ?? "the loaded model"} is the Ref2VA checkpoint, " +
            "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.";
    }

    /// <summary>
    /// The sampling flags a host was started with, as checked against a plug-in's recipe
    /// before anything is downloaded. Zero (or a null sampler) means the flag was not given.
    /// The flag names are the host's own spellings, for the refusal text.
    /// </summary>
    public sealed record MiniMaxH3HostSampling(int Steps, string StepsFlag, float FlowShift, string ShiftFlag,
        float Cfg, string CfgFlag, string? Sampler, string SamplerFlag);

    /// <summary>
    /// A MiniMax-H3 plug-in's locked sampling contract: exactly <see cref="Steps"/> Euler steps
    /// on the endpoint-inclusive linspace grid (<see cref="MiniMaxH3Scheduler.BuildUniformSigmas"/>),
    /// shifted by <see cref="VideoShift"/> for video and <see cref="AudioShift"/> for audio, at CFG 1.
    /// </summary>
    public sealed class MiniMaxH3SamplingRecipe
    {
        /// <summary>Every recipe id starts with this, followed by the config's file stem.</summary>
        public const string IdPrefix = "minimax-h3-lora:";

        internal MiniMaxH3SamplingRecipe(string source, string name, string weightsFileName, int steps,
            float videoShift, float audioShift, float cfg)
        {
            Source = source;
            Id = IdPrefix + System.IO.Path.GetFileNameWithoutExtension(source);
            Name = name;
            WeightsFileName = weightsFileName;
            Steps = steps;
            VideoShift = videoShift;
            AudioShift = audioShift;
            Cfg = cfg;
        }

        /// <summary>Stable identity, e.g. <c>minimax-h3-lora:minimax-h3-fl2v-turbo-8step</c>.</summary>
        public string Id { get; }

        /// <summary>The plug-in's display name.</summary>
        public string Name { get; }

        /// <summary>The plug-in config's full path.</summary>
        public string Source { get; }

        /// <summary>The adapter file the recipe belongs to.</summary>
        public string WeightsFileName { get; }

        public int Steps { get; }
        public float VideoShift { get; }
        public float AudioShift { get; }
        public float Cfg { get; }
        public string Schedule => MiniMaxH3LoraConfig.LinspaceSchedule;
        public string Sampler => MiniMaxH3LoraConfig.EulerSampler;

        /// <summary>The video stream's <see cref="Steps"/>+1 sigmas, ending in 0.</summary>
        public float[] VideoSigmas() => MiniMaxH3Scheduler.BuildUniformSigmas(Steps, VideoShift);

        /// <summary>The audio stream's sigmas: the video grid through
        /// <see cref="MiniMaxH3Scheduler.MapSigma"/>, which is the same linspace grid at
        /// <see cref="AudioShift"/>.</summary>
        public float[] AudioSigmas()
        {
            float[] sigmas = VideoSigmas();
            for (int i = 0; i < Steps; i++)
                sigmas[i] = MiniMaxH3Scheduler.MapSigma(sigmas[i], VideoShift, AudioShift);
            sigmas[Steps] = 0f;
            return sigmas;
        }

        /// <summary>
        /// Refuse a request that contradicts the recipe. Steps, flow shift and sampler may be
        /// unset (0 / null / empty) or equal to the recipe's; guidance must be 1 or unset (the
        /// base MiniMax-H3 refusal). Never overrides a value.
        /// </summary>
        /// <exception cref="ArgumentException">A value contradicts the recipe.</exception>
        public void Validate(VideoGenerationParams p)
        {
            if (p == null) return;
            MiniMaxH3Pipeline.ValidateSamplingParameters(p);
            if (p.Steps != 0 && p.Steps != Steps)
                throw new ArgumentException(
                    $"{Name} ({WeightsFileName}) samples exactly {Steps} steps on its own schedule; {p.Steps} steps contradicts it. " +
                    "Leave the step count unset (--diffusion-steps on the CLI, --video-steps on the server, \"steps\" in a request) " +
                    $"or set it to {Steps}.");
            CheckShift(p.FlowShift);
            CheckSampler(p.Sampler);
        }

        /// <summary>Refuse host flags that contradict the recipe, before anything is downloaded.</summary>
        /// <exception cref="ArgumentException">A flag contradicts the recipe.</exception>
        public void CheckHost(MiniMaxH3HostSampling s)
        {
            ArgumentNullException.ThrowIfNull(s);
            if (s.Steps != 0 && s.Steps != Steps)
                throw new ArgumentException(
                    $"{s.StepsFlag} {s.Steps} contradicts {WeightsFileName}: {Name} samples exactly {Steps} steps on its own schedule. " +
                    $"Drop {s.StepsFlag} or pass {Steps}.");
            CheckShift(s.FlowShift);
            CheckSampler(s.Sampler);
            if (s.Cfg != 0f && s.Cfg != Cfg)
                throw new ArgumentException($"MiniMax-H3 is CFG-distilled and requires guidance 1.0. Use {s.CfgFlag} 1.0 (the default).");
        }

        private void CheckShift(float shift)
        {
            if (shift != 0f && shift != VideoShift)
                throw new ArgumentException(
                    $"{Name} ({WeightsFileName}) samples with video flow shift {MiniMaxH3LoraConfig.Num(VideoShift)} " +
                    $"(audio {MiniMaxH3LoraConfig.Num(AudioShift)}); flow shift {MiniMaxH3LoraConfig.Num(shift)} contradicts it. " +
                    $"Leave it unset (--flow-shift, \"flowShift\") or set it to {MiniMaxH3LoraConfig.Num(VideoShift)}.");
        }

        private void CheckSampler(string sampler)
        {
            if (!string.IsNullOrWhiteSpace(sampler) &&
                !string.Equals(sampler.Trim(), MiniMaxH3LoraConfig.EulerSampler, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException(
                    $"{Name} ({WeightsFileName}) samples with Euler; sampler '{sampler}' contradicts it. " +
                    $"Leave it unset (--sampler, \"sampler\") or pass {MiniMaxH3LoraConfig.EulerSampler}.");
        }

        /// <summary>The contract a host advertises and binds jobs to. Its source is the config's
        /// file name, so no local path reaches a client.</summary>
        public VideoSamplingContract ToContract() =>
            new(Id, Name, System.IO.Path.GetFileName(Source), Steps, Cfg, Schedule, Sampler, VideoShift, AudioShift);

        /// <summary>One phrase for provenance: "{name}: N steps, endpoint-inclusive linspace,
        /// video shift 6 / audio 3, euler, cfg 1".</summary>
        public string Describe() =>
            $"{Name}: {Steps} steps, endpoint-inclusive {Schedule}, video shift {MiniMaxH3LoraConfig.Num(VideoShift)} / " +
            $"audio {MiniMaxH3LoraConfig.Num(AudioShift)}, {Sampler}, cfg {MiniMaxH3LoraConfig.Num(Cfg)}";

        /// <summary>The line a run logs before denoising, with both sigma grids.</summary>
        public string LogLine() =>
            $"  [lora] sampling recipe ({WeightsFileName}) {Describe()}; video sigmas [{Format(VideoSigmas())}], " +
            $"audio sigmas [{Format(AudioSigmas())}]";

        private static string Format(float[] sigmas) =>
            string.Join(", ", sigmas.Select(s => s.ToString("0.#####", CultureInfo.InvariantCulture)));
    }
}
