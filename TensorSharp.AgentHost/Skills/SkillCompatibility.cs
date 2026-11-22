// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.CodeExec;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// What a host can do for a skill's scripts, in the terms a skill states its needs in:
    /// programs, script languages and packages.
    /// </summary>
    public sealed class SkillHostProfile
    {
        /// <summary>How results name this host: "iOS", or "this host".</summary>
        public string Name { get; init; } = "this host";

        /// <summary>Whether skill scripts run here at all (the host's Run code switch).</summary>
        public bool CanRunScripts { get; init; }

        /// <summary>
        /// Whether a script may start other programs. False for an in-process runtime
        /// (TensorAgent on iOS): it has an embedded Python and a JavaScript engine, and
        /// nothing else can be launched.
        /// </summary>
        public bool CanLaunchPrograms { get; init; }

        /// <summary>Whether a program is available to a script, by command name.</summary>
        public Func<string, bool> HasProgram { get; init; } = static _ => false;

        /// <summary>
        /// Why a package cannot be used here (language, package) -> reason, or null when it
        /// can be or the host cannot tell before trying.
        /// </summary>
        public Func<string, string, string?> PackageUnavailable { get; init; } = static (_, _) => null;

        /// <summary>Whether this host installs a script's missing packages itself.</summary>
        public bool InstallsPackages { get; init; }

        /// <summary>The profile of the host behind one conversation's tools.</summary>
        public static SkillHostProfile From(SkillToolContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            IShellBackend? backend = (context.ScriptRunner as SkillScriptRunner)?.Backend ?? context.CodeRunner?.Backend;
            bool launches = backend?.UsesHostProcesses ?? !HostOS.IsAppleMobile;
            ICodeRunner? installer = context.CodeRunner;
            return new SkillHostProfile
            {
                Name = launches ? "this host" : (HostOS.IsAppleMobile ? "iOS" : "this in-process host"),
                CanRunScripts = context.ScriptRunner != null,
                CanLaunchPrograms = launches,
                HasProgram = launches ? static name => CodeEnvironment.Which(name) != null : static _ => false,
                PackageUnavailable = (language, package) => installer?.ExplainUnavailablePackage(language, package),
                InstallsPackages = installer?.CanInstallPackagesFor("python") == true,
            };
        }
    }

    /// <summary>One skill's needs, checked against a <see cref="SkillHostProfile"/>.</summary>
    /// <param name="Blockers">Needs this host can never meet; empty when nothing was found.</param>
    /// <param name="Notes">Facts worth knowing before the first run (packages that will be installed, programs to provide).</param>
    /// <param name="Packages">Python packages the scripts declare (inline script metadata and requirements.txt).</param>
    /// <param name="HasScripts">Whether the skill ships anything runnable.</param>
    public sealed record SkillCompatibilityReport(
        IReadOnlyList<string> Blockers, IReadOnlyList<string> Notes, IReadOnlyList<string> Packages, bool HasScripts)
    {
        /// <summary>True when nothing found stands in the way of running the skill here.</summary>
        public bool CanRun => Blockers.Count == 0;

        /// <summary>One paragraph for a tool result.</summary>
        public string Describe(string hostName)
        {
            var sb = new StringBuilder();
            if (!HasScripts)
                sb.Append("Runs on ").Append(hostName).Append(": instructions only, no bundled scripts.");
            else if (CanRun)
                sb.Append("Runs on ").Append(hostName).Append(": yes, as far as its files show.");
            else
                sb.Append("Cannot run on ").Append(hostName).Append(": ").Append(string.Join(" ", Blockers));
            foreach (string note in Notes)
                sb.Append(' ').Append(note);
            return sb.ToString();
        }
    }

    /// <summary>
    /// Reads what a skill needs out of its files — the programs its frontmatter requires,
    /// the languages of its scripts, the packages they declare — and says whether a host
    /// can provide them. Static analysis only: nothing in the skill is executed.
    /// </summary>
    public static class SkillCompatibility
    {
        /// <summary>Script extensions a skill may ship, by how a host runs them.</summary>
        private static readonly HashSet<string> InProcessLanguages = new(StringComparer.OrdinalIgnoreCase)
        {
            ".py", ".js", ".mjs", ".sh", ".bash",
        };

        private static readonly HashSet<string> ScriptExtensions = new(StringComparer.OrdinalIgnoreCase)
        {
            ".py", ".js", ".mjs", ".cjs", ".ts", ".sh", ".bash", ".zsh", ".rb", ".pl", ".php", ".ps1", ".go", ".lua",
        };

        /// <summary>
        /// Programs a host stands in for: <c>uv run</c> is carried out with the host's own
        /// Python and package installer (<see cref="SkillTools"/>), and Python itself is the
        /// interpreter every host here provides.
        /// </summary>
        private static readonly HashSet<string> ProvidedPrograms = new(StringComparer.OrdinalIgnoreCase)
        {
            "uv", "python", "python3", "pip", "pip3",
        };

        /// <summary>Most packages checked one by one; a longer list is summarized.</summary>
        private const int MaxCheckedPackages = 12;

        /// <summary>Assess an installed skill.</summary>
        public static SkillCompatibilityReport Assess(Skill skill, SkillHostProfile host)
        {
            ArgumentNullException.ThrowIfNull(skill);
            return Assess(skill.RootDirectory, skill.Manifest, host);
        }

        /// <summary>Assess a skill directory, installed or not (an extracted download).</summary>
        public static SkillCompatibilityReport Assess(string skillRoot, SkillManifest manifest, SkillHostProfile host)
        {
            ArgumentNullException.ThrowIfNull(manifest);
            ArgumentNullException.ThrowIfNull(host);
            var blockers = new List<string>();
            var notes = new List<string>();

            List<string> scripts = ListScripts(skillRoot);
            bool hasScripts = scripts.Count > 0;

            // Programs the frontmatter says it needs (metadata.*.requires.bins, the
            // convention several hub publishers use).
            foreach (string program in RequiredPrograms(skillRoot))
            {
                if (ProvidedPrograms.Contains(program))
                    continue;
                if (!host.CanLaunchPrograms)
                    blockers.Add($"It needs the `{program}` program, and {host.Name} cannot launch programs.");
                else if (!host.HasProgram(program))
                    notes.Add($"It needs the `{program}` program, which is not on this host's PATH.");
            }

            if (!host.CanLaunchPrograms)
            {
                foreach (string script in scripts)
                {
                    string extension = Path.GetExtension(script);
                    if (!InProcessLanguages.Contains(extension))
                        blockers.Add($"{script} is a {extension} script, and {host.Name} runs only Python, JavaScript and shell scripts in-process.");
                }
            }

            List<string> packages = DeclaredPythonPackages(skillRoot, scripts);
            if (packages.Count > 0)
            {
                int checkedCount = 0;
                foreach (string package in packages)
                {
                    if (checkedCount++ >= MaxCheckedPackages)
                        break;
                    string? reason = host.PackageUnavailable("python", package);
                    if (reason != null)
                        blockers.Add($"Its scripts need the Python package {package}, which cannot be used on {host.Name}: {reason}.");
                }
                if (blockers.Count == 0 || host.CanLaunchPrograms)
                {
                    notes.Add(host.InstallsPackages
                        ? "Its scripts declare Python packages (" + string.Join(", ", packages)
                          + "); skills_run installs them before the first run."
                        : "Its scripts declare Python packages (" + string.Join(", ", packages)
                          + "), and installing packages is turned off on this host.");
                }
            }

            if (hasScripts && !host.CanRunScripts)
                notes.Add("Running skill scripts is turned off on this host; only its instructions can be used.");

            return new SkillCompatibilityReport(
                blockers.Distinct().ToList(), notes.Distinct().ToList(), packages, hasScripts);
        }

        private static List<string> ListScripts(string skillRoot)
        {
            var scripts = new List<string>();
            if (string.IsNullOrEmpty(skillRoot) || !Directory.Exists(skillRoot))
                return scripts;
            foreach (string path in Directory.EnumerateFiles(skillRoot, "*", new EnumerationOptions
            {
                RecurseSubdirectories = true,
                AttributesToSkip = FileAttributes.ReparsePoint | FileAttributes.System,
                IgnoreInaccessible = true,
                MaxRecursionDepth = 6,
            }))
            {
                if (scripts.Count >= 256)
                    break;
                string relative = Path.GetRelativePath(skillRoot, path).Replace('\\', '/');
                if (relative.Split('/').Any(segment => segment is "node_modules" or "__pycache__" || segment.StartsWith('.')))
                    continue;
                if (ScriptExtensions.Contains(Path.GetExtension(relative)))
                    scripts.Add(relative);
            }
            scripts.Sort(StringComparer.Ordinal);
            return scripts;
        }

        /// <summary>Programs named under any <c>requires.bins</c> / <c>requires.anyBins</c> in the frontmatter's metadata.</summary>
        internal static IReadOnlyList<string> RequiredPrograms(string skillRoot)
        {
            string manifestPath = Path.Combine(skillRoot ?? string.Empty, SkillManifestParser.SkillFileName);
            if (!File.Exists(manifestPath))
                return Array.Empty<string>();
            YamlValue? frontmatter;
            try
            {
                frontmatter = YamlFrontmatter.ParseDocument(File.ReadAllText(manifestPath), out _);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or YamlFrontmatterException)
            {
                return Array.Empty<string>();
            }
            if (frontmatter == null || !frontmatter.Mapping.TryGetValue("metadata", out YamlValue? metadata))
                return Array.Empty<string>();

            var found = new List<string>();
            if (metadata.Kind == YamlValueKind.Scalar && metadata.Scalar is { Length: > 1 } text && text.TrimStart().StartsWith('{'))
            {
                try
                {
                    using JsonDocument json = JsonDocument.Parse(text, new JsonDocumentOptions { MaxDepth = 16 });
                    CollectBins(json.RootElement, found, inRequires: false, depth: 0);
                }
                catch (JsonException)
                {
                }
            }
            else
            {
                CollectBins(metadata, found, inRequires: false, depth: 0);
            }
            return found.Where(IsCommandName).Distinct(StringComparer.OrdinalIgnoreCase).Take(16).ToList();
        }

        private static void CollectBins(YamlValue node, List<string> found, bool inRequires, int depth)
        {
            if (depth > 8) return;
            foreach ((string key, YamlValue value) in node.Mapping)
            {
                if (inRequires && key is "bins" or "anyBins")
                {
                    if (value.Kind == YamlValueKind.Sequence)
                        found.AddRange(value.Sequence.Select(v => v.Scalar).OfType<string>());
                    else if (value.Scalar is { } scalar)
                        found.AddRange(scalar.Trim('[', ']').Split(',').Select(s => s.Trim().Trim('"', '\'')));
                }
                else if (value.Kind == YamlValueKind.Mapping)
                {
                    CollectBins(value, found, inRequires || key == "requires", depth + 1);
                }
            }
        }

        private static void CollectBins(JsonElement node, List<string> found, bool inRequires, int depth)
        {
            if (depth > 8 || node.ValueKind != JsonValueKind.Object) return;
            foreach (JsonProperty property in node.EnumerateObject())
            {
                if (inRequires && property.Name is "bins" or "anyBins" && property.Value.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement item in property.Value.EnumerateArray())
                        if (item.ValueKind == JsonValueKind.String && item.GetString() is { } name)
                            found.Add(name);
                }
                else
                {
                    CollectBins(property.Value, found, inRequires || property.Name == "requires", depth + 1);
                }
            }
        }

        private static bool IsCommandName(string? name) =>
            !string.IsNullOrWhiteSpace(name) && name.Length <= 64
            && name.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.');

        /// <summary>Python packages declared by the scripts' inline metadata and any requirements.txt, in order.</summary>
        internal static List<string> DeclaredPythonPackages(string skillRoot, IReadOnlyList<string> scripts)
        {
            var packages = new List<string>();
            foreach (string script in scripts.Where(s => s.EndsWith(".py", StringComparison.OrdinalIgnoreCase)))
            {
                string path = Path.Combine(skillRoot, script);
                try
                {
                    var info = new FileInfo(path);
                    if (info.Length > 1024 * 1024) continue;
                    packages.AddRange(InlineScriptMetadata.Dependencies(File.ReadAllText(path)));
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                }
            }
            foreach (string requirements in new[] { "requirements.txt", "scripts/requirements.txt" })
            {
                string path = Path.Combine(skillRoot, requirements);
                if (!File.Exists(path)) continue;
                try
                {
                    foreach (string line in File.ReadLines(path).Take(200))
                        if (InlineScriptMetadata.ToInstallSpec(line.Split('#')[0]) is { } spec)
                            packages.Add(spec);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                }
            }
            return packages
                .GroupBy(p => InlineScriptMetadata.PackageName(p), StringComparer.OrdinalIgnoreCase)
                .Select(g => g.First())
                .ToList();
        }
    }

    /// <summary>
    /// PEP 723 inline script metadata: the <c># /// script</c> block a script carries so
    /// that <c>uv run</c> (or pipx) can install its dependencies first. TensorSharp reads the
    /// same block and installs the dependencies through its own package installer, so a
    /// script written for <c>uv run</c> runs under <c>skills_run</c> unchanged.
    /// </summary>
    public static class InlineScriptMetadata
    {
        private static readonly Regex Block = new(
            @"(?m)^# /// script[ \t]*\r?\n(?<content>(?:^#(?:| .*)\r?\n)+?)^# ///[ \t]*$",
            RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));

        private static readonly Regex QuotedString = new(
            "\"(?<v>(?:[^\"\\\\]|\\\\.)*)\"|'(?<v>[^']*)'", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));

        private static readonly Regex NameAndSpec = new(
            @"^\s*(?<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?<spec>[^;]*)(?<marker>;.*)?$",
            RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));

        /// <summary>
        /// The dependencies a script's inline metadata declares, as specs a host installer
        /// takes (a bare name, or name==version for an exact pin). Requirements with an
        /// environment marker are left out: the host cannot evaluate one, and installing a
        /// dependency meant for another platform would fail the whole install.
        /// </summary>
        public static IReadOnlyList<string> Dependencies(string? scriptText)
        {
            if (string.IsNullOrEmpty(scriptText))
                return Array.Empty<string>();
            Match block;
            try
            {
                block = Block.Match(scriptText);
            }
            catch (RegexMatchTimeoutException)
            {
                return Array.Empty<string>();
            }
            if (!block.Success)
                return Array.Empty<string>();

            var toml = new StringBuilder();
            foreach (string raw in block.Groups["content"].Value.Split('\n'))
            {
                string line = raw.TrimEnd('\r');
                toml.Append(line.Length >= 2 ? line.Substring(2) : string.Empty).Append('\n');
            }
            string content = toml.ToString();
            Match key = Regex.Match(content, @"(?m)^\s*dependencies\s*=\s*\[", RegexOptions.CultureInvariant);
            if (!key.Success)
                return Array.Empty<string>();
            int start = key.Index + key.Length;
            int end = -1;
            char quote = '\0';
            for (int i = start; i < content.Length && end < 0; i++)
            {
                char c = content[i];
                if (quote != '\0')
                {
                    if (c == '\\' && quote == '"') i++;
                    else if (c == quote) quote = '\0';
                }
                else if (c is '"' or '\'') quote = c;
                else if (c == '#') { int eol = content.IndexOf('\n', i); i = eol < 0 ? content.Length : eol; }
                else if (c == ']') end = i;
            }
            if (end < 0)
                return Array.Empty<string>();
            var specs = new List<string>();
            foreach (Match value in QuotedString.Matches(content.Substring(start, end - start)))
            {
                if (ToInstallSpec(value.Groups["v"].Value) is { } spec)
                    specs.Add(spec);
                if (specs.Count >= 64) break;
            }
            return specs;
        }

        /// <summary>A PEP 508 requirement as an installer spec, or null when it is not one this host can take.</summary>
        public static string? ToInstallSpec(string? requirement)
        {
            if (string.IsNullOrWhiteSpace(requirement) || requirement.Contains("://", StringComparison.Ordinal)
                || requirement.TrimStart().StartsWith('-'))
                return null;
            Match match = NameAndSpec.Match(requirement);
            if (!match.Success || match.Groups["marker"].Success)
                return null;
            string name = match.Groups["name"].Value;
            string spec = match.Groups["spec"].Value.Trim();
            // An exact pin is kept; a range is left to the installer's newest compatible pick.
            if (spec.StartsWith("==", StringComparison.Ordinal)
                && Regex.IsMatch(spec.Substring(2).Trim(), @"^[A-Za-z0-9.+!_-]+$", RegexOptions.CultureInvariant))
                return name + "==" + spec.Substring(2).Trim();
            return name;
        }

        /// <summary>The distribution name of a spec.</summary>
        public static string PackageName(string spec)
        {
            int cut = spec.IndexOfAny(new[] { '=', '<', '>', '!', '~', '[', ';', ' ' });
            return (cut < 0 ? spec : spec.Substring(0, cut)).Trim();
        }
    }
}
