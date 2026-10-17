// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The LoRA plug-in check both hosts run at startup, before any plug-in or model file is
// resolved or downloaded. TensorSharp.Cli and TensorSharp.Server share it so the two cannot
// drift: the command line's --lora specs (or, when it names none, the TS_LORAS the process
// inherited) are matched against the model's family, and a MiniMax-H3 plug-in's locked
// sampling contract against the host's own sampling flags. With --config it runs twice: on
// the no-download preview of the expansion (CheckPreview), and again on the final command
// line just before LoraCliFlags.Resolve fetches the plug-in weights.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.MiniMaxH3;

namespace TensorSharp.Models
{
    /// <summary>The host a <see cref="LoraHostPreflight"/> check runs for.</summary>
    public enum LoraHost
    {
        /// <summary>TensorSharp.Cli: <c>--diffusion-steps</c>, <c>--flow-shift</c>, <c>--cfg</c> and <c>--sampler</c>.</summary>
        Cli,

        /// <summary>TensorSharp.Server.Host: <c>--video-steps</c>; a request carries the other sampling values.</summary>
        Server,
    }

    /// <summary>
    /// What a host checks before it resolves its LoRA plug-ins.
    /// </summary>
    /// <param name="Host">The host, for its flag spellings.</param>
    /// <param name="FlagSpecs">The command line's <c>--lora</c> specs, in order.</param>
    /// <param name="EnvSpecs">The specs of the inherited <c>TS_LORAS</c>. They count only when
    /// <paramref name="FlagSpecs"/> is empty: the hosts publish the command line's plug-ins
    /// over the variable.</param>
    /// <param name="ModelPath">The model the host will load, or null when none is known yet.</param>
    /// <param name="Sampling">The host's sampling flags (<see cref="LoraHostPreflight.CliSampling"/>,
    /// <see cref="LoraHostPreflight.ServerSampling"/>).</param>
    /// <param name="IsPendingDownload">True for a path a <c>--config</c> expansion has yet to
    /// download, which the check must skip rather than call missing.</param>
    public sealed record LoraHostRequest(LoraHost Host, IReadOnlyList<LoraSpec> FlagSpecs, IReadOnlyList<LoraSpec> EnvSpecs,
        string? ModelPath, MiniMaxH3HostSampling Sampling, Func<string, bool>? IsPendingDownload = null);

    /// <summary>
    /// The outcome of a check that passed: the model's family as probed (null when the file is
    /// missing, still to be downloaded or unreadable; its load reports that), and one warning per
    /// group of inherited plug-ins the model will skip.
    /// </summary>
    public sealed record LoraHostVerdict(ModelArchitectureDescriptor? Architecture, IReadOnlyList<string> EnvWarnings);

    /// <summary>
    /// The startup check of a host's LoRA plug-ins. It reads plug-in configs and the model's
    /// GGUF header only: never weights, never a download.
    /// </summary>
    public static class LoraHostPreflight
    {
        /// <summary>The CLI's step count flag.</summary>
        public const string CliStepsFlag = "--diffusion-steps";

        /// <summary>The CLI's video flow shift flag.</summary>
        public const string CliShiftFlag = "--flow-shift";

        /// <summary>The CLI's guidance flag.</summary>
        public const string CliCfgFlag = "--cfg";

        /// <summary>The CLI's sampler flag.</summary>
        public const string CliSamplerFlag = "--sampler";

        /// <summary>The server's default step count for video requests.</summary>
        public const string ServerStepsFlag = "--video-steps";

        private const string ModelFlag = "--model";

        /// <summary>A plug-in whose config declares no TensorSharp type, in the warning texts.</summary>
        private const string NoDeclaredType = "none (bare weights)";

        /// <summary>The CLI's sampling flags; 0 (or a null sampler) means the flag was not given.</summary>
        public static MiniMaxH3HostSampling CliSampling(int steps, float shift, float cfg, string? sampler) =>
            new(steps, CliStepsFlag, shift, CliShiftFlag, cfg, CliCfgFlag, sampler, CliSamplerFlag);

        /// <summary>
        /// The server's sampling flags: <c>--video-steps</c> only (0 when not given). A request
        /// names its own flow shift, guidance and sampler, which the model checks per request,
        /// so those are spelled as the request fields and never given here.
        /// </summary>
        public static MiniMaxH3HostSampling ServerSampling(int videoSteps) =>
            new(videoSteps, ServerStepsFlag, 0f, "\"flowShift\"", 0f, "\"cfg\"", null, "\"sampler\"");

        /// <summary>
        /// Check a host's plug-ins against the model it will load. In order:
        /// <list type="number">
        /// <item>Every named plug-in file must exist (<see cref="LoraCliFlags.DeclaredPluginType"/>,
        /// so a typo'd config is reported as missing, not as bare weights). A path still to be
        /// downloaded is skipped: weights count as bare weights, a config's type stays unknown.
        /// Inherited plug-ins are not read for a model that takes none.</item>
        /// <item>The model is probed (its GGUF header) when the file exists.</item>
        /// <item>Command-line plug-ins: a model that takes none refuses them all; otherwise each
        /// must be of the model's type (Qwen-Image-2.1 also takes bare weights; MiniMax-H3 takes
        /// only its own configs).</item>
        /// <item>Inherited plug-ins the model will skip produce warnings.</item>
        /// <item>The MiniMax-H3 plug-ins the model will take (on a MiniMax-H3 model, or one not
        /// probed) go through <see cref="MiniMaxH3LoraConfig.CheckHostSettings"/>: the checkpoint
        /// partition, the locked sampling flags, one plug-in at most, the strength and the
        /// weights tie.</item>
        /// </list>
        /// </summary>
        /// <exception cref="ArgumentException">A refusal, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named plug-in file is missing.</exception>
        public static LoraHostVerdict Check(LoraHostRequest r)
        {
            ArgumentNullException.ThrowIfNull(r);
            ArgumentNullException.ThrowIfNull(r.Sampling, nameof(r.Sampling));
            IReadOnlyList<LoraSpec> flagSpecs = r.FlagSpecs ?? Array.Empty<LoraSpec>();
            // The hosts replace TS_LORAS with the command line's plug-ins, so inherited ones
            // are read only when the command line names none.
            IReadOnlyList<LoraSpec> envSpecs = flagSpecs.Count == 0 ? r.EnvSpecs ?? Array.Empty<LoraSpec>() : Array.Empty<LoraSpec>();
            Func<string, bool> pending = r.IsPendingDownload ?? (_ => false);

            // The probe reads a header and never throws, so it runs first: inherited plug-ins
            // offered to a model that takes none are only named in a warning, never read.
            ModelArchitectureDescriptor? arch = Probe(r.ModelPath);
            string model = r.ModelPath != null ? Path.GetFileName(r.ModelPath) : null;

            // 1. Files.
            var flagTypes = flagSpecs.Select(s => Declare(s, pending)).ToList();
            bool envRead = arch == null || arch.LoraPluginType != null;
            var envTypes = envRead ? envSpecs.Select(s => Declare(s, pending)).ToList() : null;

            // 3. Command-line plug-ins against the model's family.
            if (arch != null && flagSpecs.Count > 0)
            {
                if (arch.LoraPluginType == null)
                    throw new ArgumentException(NoPluginsRefusal(model, arch.Id));
                for (int i = 0; i < flagSpecs.Count; i++)
                {
                    if (!flagTypes[i].Known || Takes(arch, flagTypes[i].Type)) continue;
                    string file = Path.GetFileName(flagSpecs[i].Path);
                    if (flagTypes[i].Type == null)
                        throw new ArgumentException(MiniMaxH3LoraConfig.BareWeightsRefusal(file));
                    throw new ArgumentException(MiniMaxH3LoraConfig.ForeignTypeRefusal(
                        file, flagTypes[i].Type, model, arch.Id, arch.LoraPluginType));
                }
            }

            // 4. Inherited plug-ins the model skips.
            var warnings = new List<string>();
            if (arch != null && envSpecs.Count > 0)
            {
                if (arch.LoraPluginType == null)
                {
                    warnings.Add(EnvNoPluginsWarning(Names(envSpecs), model, arch.Id));
                }
                else
                {
                    var skipped = envSpecs
                        .Select((spec, i) => (Spec: spec, Declared: envTypes[i]))
                        .Where(e => e.Declared.Known && !Takes(arch, e.Declared.Type))
                        .GroupBy(e => e.Declared.Type ?? NoDeclaredType, StringComparer.Ordinal);
                    foreach (var group in skipped)
                        warnings.Add(EnvForeignTypeWarning(Names(group.Select(e => e.Spec)), group.Key, model, arch.Id,
                            arch.LoraPluginType));
                }
            }

            // 5. The MiniMax-H3 plug-ins the model will take, against the host's sampling flags.
            if (arch == null || arch.LoraPluginType == LoraCliFlags.MiniMaxH3PluginType)
            {
                var (specs, types) = flagSpecs.Count > 0 ? (flagSpecs, flagTypes) : (envSpecs, envTypes);
                var h3Specs = specs
                    .Where((_, i) => types[i].Known && types[i].Type == LoraCliFlags.MiniMaxH3PluginType)
                    .ToList();
                if (h3Specs.Count > 0)
                    MiniMaxH3LoraConfig.CheckHostSettings(h3Specs, r.ModelPath, r.Sampling);
            }

            return new LoraHostVerdict(arch, warnings);
        }

        /// <summary>
        /// The same check on a <c>--config</c> expansion's preview, before any of its files is
        /// downloaded (<see cref="ConfigFileArgs.Expand(string[], Action{ConfigExpansionPreview})"/>):
        /// the merged tokens' <c>--lora</c> specs (or the inherited <c>TS_LORAS</c> when they name
        /// none), the last <c>--model</c>, and the host's sampling flags. A value the host's own
        /// parser will refuse is skipped here; the parser reports it. The warnings are dropped:
        /// the host's check on the final command line says them once.
        /// </summary>
        /// <exception cref="ArgumentException">A refusal, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named plug-in file is missing.</exception>
        public static void CheckPreview(ConfigExpansionPreview preview, LoraHost host) =>
            CheckPreview(preview, host, Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable));

        internal static void CheckPreview(ConfigExpansionPreview preview, LoraHost host, string? inheritedLoras)
        {
            ArgumentNullException.ThrowIfNull(preview);
            IReadOnlyList<string> tokens = preview.Tokens;
            List<LoraSpec> flagSpecs = LoraCliFlags.Parse(tokens);
            List<LoraSpec> envSpecs = flagSpecs.Count == 0 ? LoraCliFlags.FromJson(inheritedLoras) : new List<LoraSpec>();
            if (flagSpecs.Count == 0 && envSpecs.Count == 0)
                return;

            MiniMaxH3HostSampling sampling = host == LoraHost.Cli
                ? CliSampling(
                    ParseInt(LastValue(tokens, CliStepsFlag)),
                    ParseFloat(LastValue(tokens, CliShiftFlag)),
                    ParseFloat(LastValue(tokens, CliCfgFlag)),
                    LastValue(tokens, CliSamplerFlag))
                : ServerSampling(Math.Max(0, ParseInt(LastValue(tokens, ServerStepsFlag))));
            Check(new LoraHostRequest(host, flagSpecs, envSpecs, LastValue(tokens, ModelFlag), sampling, preview.IsPending));
        }

        // ---- canonical texts ------------------------------------------------------------------

        /// <summary>The warning for inherited plug-ins of a type the model does not take: it
        /// skips them.</summary>
        /// <param name="names">The plug-ins' file names, comma-separated.</param>
        /// <param name="declared">Their declared type, or null for bare weights.</param>
        /// <param name="model">The model's file name.</param>
        /// <param name="arch">The model's architecture id.</param>
        /// <param name="type">The plug-in type the model takes.</param>
        public static string EnvForeignTypeWarning(string names, string? declared, string model, string arch, string type) =>
            $"{LoraCliFlags.EnvironmentVariable} names LoRA plug-ins {names} of type {declared ?? NoDeclaredType}; " +
            $"{model} ({arch}) takes {type} plug-ins and runs without them.";

        /// <summary>The warning for inherited plug-ins offered to a model that takes none.</summary>
        public static string EnvNoPluginsWarning(string names, string model, string arch) =>
            $"{LoraCliFlags.EnvironmentVariable} names LoRA plug-ins ({names}), but {model} ({arch}) takes none; it runs without them.";

        internal static string NoPluginsRefusal(string model, string arch) =>
            $"{LoraCliFlags.LoraFlag} applies to Qwen-Image-2.1 and MiniMax-H3 (FL2VA) models only; " +
            $"'{model}' ({arch}) does not take LoRA plug-ins.";

        // ---- helpers ----------------------------------------------------------------------------

        /// <summary>A spec's declared plug-in type, or unknown (<see cref="Known"/> false) when
        /// the config that declares it is still to be downloaded.</summary>
        private readonly record struct Declared(bool Known, string? Type);

        private static Declared Declare(LoraSpec spec, Func<string, bool> pending)
        {
            ArgumentNullException.ThrowIfNull(spec);
            bool pathPending = pending(spec.Path);
            bool configPending = spec.ConfigPath != null && pending(spec.ConfigPath);
            if (!pathPending && !configPending)
                return new(true, LoraCliFlags.DeclaredPluginType(spec));

            if (IsJson(spec.Path))
            {
                // A config named by --lora declares the type itself (a --lora-config beside it
                // is refused when the plug-in is resolved).
                return pathPending ? new(false, null) : new(true, LoraCliFlags.DeclaredPluginType(spec with { ConfigPath = null }));
            }

            // Weights: here (a missing file is the not-found refusal) or still to come.
            if (!pathPending)
                LoraCliFlags.DeclaredPluginType(new LoraSpec(spec.Path));
            if (spec.ConfigPath == null)
                return new(true, null);
            if (configPending)
                return IsJson(spec.ConfigPath) ? new(false, null) : new(true, null);
            if (!File.Exists(spec.ConfigPath))
                throw new FileNotFoundException($"{LoraCliFlags.ConfigFlag} file not found: {spec.ConfigPath}", spec.ConfigPath);
            return new(true, IsJson(spec.ConfigPath) ? LoraCliFlags.DeclaredPluginType(new LoraSpec(spec.ConfigPath)) : null);
        }

        /// <summary>True when <paramref name="arch"/>'s model applies a plug-in that declares
        /// <paramref name="declared"/>: its own type, and for Qwen-Image-2.1 bare weights too.</summary>
        private static bool Takes(ModelArchitectureDescriptor arch, string? declared) =>
            declared == arch.LoraPluginType ||
            (declared == null && arch.LoraPluginType == LoraCliFlags.QwenImagePluginType);

        private static ModelArchitectureDescriptor? Probe(string? modelPath)
        {
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
                return null;
            try
            {
                using var probe = new GgufFile(modelPath);
                return ModelArchitectureRegistry.Resolve(probe.GetString("general.architecture"), probe);
            }
            catch (Exception)
            {
                // An unreadable or unknown model file is reported by the load itself.
                return null;
            }
        }

        private static bool IsJson(string path) => path.EndsWith(".json", StringComparison.OrdinalIgnoreCase);

        private static string Names(IEnumerable<LoraSpec> specs) =>
            string.Join(", ", specs.Select(s => Path.GetFileName(s.Path)));

        /// <summary>The value of the last <paramref name="flag"/> (spaced or joined, any case,
        /// as both hosts accept it), or null.</summary>
        private static string? LastValue(IReadOnlyList<string> tokens, string flag)
        {
            string? value = null;
            for (int i = 0; i < tokens.Count; i++)
            {
                string token = tokens[i];
                if (string.Equals(token, flag, StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < tokens.Count)
                        value = tokens[++i];
                }
                else if (token.StartsWith(flag + "=", StringComparison.OrdinalIgnoreCase))
                {
                    value = token.Substring(flag.Length + 1);
                }
            }
            return value;
        }

        private static int ParseInt(string? value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : 0;

        private static float ParseFloat(string? value) =>
            float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ? parsed : 0f;
    }
}
