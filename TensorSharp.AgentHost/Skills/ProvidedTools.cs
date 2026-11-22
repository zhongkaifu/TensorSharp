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
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// A tool a reachable skill provides, offered as a real tool: the skill declares
    /// <c>provides-tools: WebSearch, WebFetch</c> and the call that does their job
    /// (<c>provides-call</c>), and the host declares <c>web_search(query)</c> and
    /// <c>web_fetch(url)</c>, which run that call through <c>skills_run</c>, in its sandbox.
    ///
    /// <para>
    /// Why: a small model had to fill <c>skills_run(skill, path, args=[JSON array])</c> to get
    /// any current fact, and could not do it reliably — a missing comma, a placeholder copied
    /// as it was, the script typed into a shell. One string parameter has no such shape to get
    /// wrong, and a model trained on a web search tool uses one.
    /// </para>
    /// </summary>
    /// <param name="Name">The tool's name, the provided tool's in snake case (<c>WebSearch</c> -> <c>web_search</c>).</param>
    /// <param name="Provides">The provided tool as the skill names it.</param>
    /// <param name="Skill">The providing skill.</param>
    /// <param name="Path">Its script.</param>
    /// <param name="Arguments">The declared arguments, the placeholder among them.</param>
    /// <param name="Placeholder">The argument the tool's parameter replaces, such as <c>&lt;question&gt;</c>.</param>
    /// <param name="Parameter">The tool's one parameter.</param>
    /// <param name="ReadAfter">The file the declared call says to read afterwards, or null.</param>
    public sealed record ProvidedTool(
        string Name, string Provides, Skill Skill, string Path, IReadOnlyList<string> Arguments,
        string Placeholder, string Parameter, string? ReadAfter)
    {
        /// <summary>The longest output file a result carries; the rest is a read_file away.</summary>
        internal const int MaxInlineBytes = 24 * 1024;

        private static readonly Regex DeclaredCall = new(
            @"^\s*skills_run\(\s*skill=""(?<skill>[^""]+)"",\s*path=""(?<path>[^""]+)"",\s*args=(?<args>\[[^\]]*\])\s*\)"
            + @"(?:\s*,\s*then read (?<read>[A-Za-z0-9_./-]+))?\s*$",
            RegexOptions.CultureInvariant);

        /// <summary>
        /// The tools the reachable skills provide on this host: only from a skill that can run
        /// here, with a call this parser reads exactly, under a name no other tool has.
        /// </summary>
        internal static IReadOnlyList<ProvidedTool> For(SkillToolContext context, IEnumerable<string> takenNames)
        {
            if (context.ScriptRunner == null)
                return Array.Empty<ProvidedTool>();
            var taken = new HashSet<string>(takenNames, StringComparer.OrdinalIgnoreCase);
            var tools = new List<ProvidedTool>();
            SkillHostProfile? host = null;
            foreach (Skill skill in context.Reachable)
            {
                if (!skill.Manifest.Metadata.TryGetValue("provides-tools", out string? provides)
                    || !skill.Manifest.Metadata.TryGetValue("provides-call", out string? declared))
                    continue;
                Match call = DeclaredCall.Match(declared);
                if (!call.Success || !string.Equals(call.Groups["skill"].Value, skill.Id, StringComparison.Ordinal))
                    continue;
                string path = call.Groups["path"].Value;
                if (!skill.Files.Any(f => f.Kind == SkillFileKind.Script && string.Equals(f.Path, path, SkillPathGuard.PathComparison)))
                    continue;
                string[]? arguments;
                try
                {
                    arguments = JsonSerializer.Deserialize<string[]>(call.Groups["args"].Value);
                }
                catch (JsonException)
                {
                    continue;
                }
                string? placeholder = arguments?.FirstOrDefault(a => SkillTools.PlaceholdersOf("\"" + a + "\"").Count == 1);
                if (arguments == null || placeholder == null)
                    continue;
                host ??= SkillHostProfile.From(context);
                if (!SkillCompatibility.Assess(skill, host).CanRun)
                    continue;
                string? readAfter = call.Groups["read"].Success ? call.Groups["read"].Value : null;
                foreach (string name in provides.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    string tool = SnakeCase(name);
                    if (tool.Length == 0 || !taken.Add(tool) || SkillTools.IsBuiltInTool(tool))
                        continue;
                    tools.Add(new ProvidedTool(tool, name, skill, path, arguments, placeholder, ParameterFor(name), readAfter));
                }
            }
            return tools;
        }

        /// <summary>
        /// The parameter of a well-known provided tool, as the agents that define it name it
        /// (WebSearch's <c>query</c>, WebFetch's <c>url</c>); <c>input</c> for any other.
        /// </summary>
        private static string ParameterFor(string provides) => provides switch
        {
            "WebSearch" => "query",
            "WebFetch" => "url",
            _ => "input",
        };

        internal static string SnakeCase(string name)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < name.Length; i++)
            {
                char c = name[i];
                if (!char.IsAsciiLetterOrDigit(c))
                {
                    if (sb.Length > 0 && sb[^1] != '_')
                        sb.Append('_');
                    continue;
                }
                if (char.IsUpper(c) && i > 0 && sb.Length > 0 && sb[^1] != '_' && !char.IsUpper(name[i - 1]))
                    sb.Append('_');
                sb.Append(char.ToLowerInvariant(c));
            }
            return sb.ToString().Trim('_');
        }

        /// <summary>The declaration: the providing skill's own description, and one string parameter.</summary>
        internal ToolFunction Declare() => new()
        {
            Name = Name,
            Description = Skill.Description + " (" + Provides + ", provided by the '" + Skill.Id + "' skill"
                + (ReadAfter == null ? ".)" : "; returns " + ReadAfter + ".)"),
            Parameters = new()
            {
                [Parameter] = new()
                {
                    Type = "string",
                    Description = Parameter switch
                    {
                        "query" => "What to search for.",
                        "url" => "The page's URL.",
                        _ => "The input.",
                    },
                },
            },
            Required = new() { Parameter },
        };

        /// <summary>
        /// Run the declared call with the parameter in the placeholder's place, through
        /// <c>skills_run</c>, and return what the call says to read afterwards with it.
        /// </summary>
        internal SkillToolResult Execute(ToolCall call, SkillToolContext context, Action<string>? onOutput)
        {
            string? value = SkillTools.ReadString(call, Parameter)
                ?? call.Arguments?.Values.OfType<object>().Select(v => v is JsonElement { ValueKind: JsonValueKind.String } e ? e.GetString() : v as string)
                    .FirstOrDefault(v => !string.IsNullOrWhiteSpace(v));
            if (string.IsNullOrWhiteSpace(value))
                return SkillToolResult.Failure($"{Name} needs a '{Parameter}' argument.");
            var run = new ToolCall
            {
                Name = SkillTools.RunToolName,
                Arguments = new Dictionary<string, object>
                {
                    ["skill"] = Skill.Id,
                    ["path"] = Path,
                    ["args"] = Arguments.Select(a => a == Placeholder ? value.Trim() : a).ToArray(),
                },
            };
            SkillToolResult ran = SkillTools.Execute(run, context, onOutput);
            if (!ran.Ok || ReadAfter == null || context.Workspace is not { } workspace)
                return ran;
            string file = System.IO.Path.Combine(workspace.WorkDirectory, ReadAfter);
            if (!SkillPathGuard.TryResolveExistingFile(workspace.WorkDirectory, ReadAfter, out _, out _) || !File.Exists(file))
                return ran;
            byte[] bytes = File.ReadAllBytes(file);
            string text = Encoding.UTF8.GetString(bytes, 0, Math.Min(bytes.Length, MaxInlineBytes));
            var sb = new StringBuilder(ran.Content).Append("\n\n").Append(ReadAfter).Append(":\n").Append(text);
            if (bytes.Length > MaxInlineBytes)
                sb.Append("\n[").Append(ReadAfter).Append(" continues; read the rest with read_file(path=\"").Append(ReadAfter).Append("\").]");
            return ran with { Content = sb.ToString() };
        }
    }
}
