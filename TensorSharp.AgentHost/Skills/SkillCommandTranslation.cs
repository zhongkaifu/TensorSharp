// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Recognizes a shell command that runs one of a skill's bundled scripts the way the
    /// skill's own instructions spell it for another agent — <c>uv run {baseDir}/scripts/x.py
    /// AAPL</c>, <c>python3 $SKILL_DIR/scripts/x.py</c> — so the host can run it as
    /// <c>skills_run</c> instead: with the skill resolved where it really is, the script's
    /// declared dependencies installed by the host's installer, and the skill-script
    /// sandbox. A shell cannot do any of that here: the placeholder names no directory, and
    /// a command never gets a network socket, so <c>uv</c> could not fetch anything.
    ///
    /// <para>
    /// Deliberately narrow. One simple command and nothing else (no pipes, lists,
    /// redirections or substitutions); <c>uv run</c> with only options whose meaning is
    /// kept, or <c>python</c>/<c>python3</c>; and a script path that resolves to exactly
    /// one reachable skill's file. A plain relative path is taken only when the
    /// conversation's own workspace has no such file, so a model's own program is never
    /// mistaken for a skill's.
    /// </para>
    /// </summary>
    internal static class SkillCommandTranslation
    {
        internal readonly record struct Translated(
            Skill Skill, string Path, IReadOnlyList<string> Arguments, IReadOnlyList<string> Packages);

        private static readonly string[] Placeholders =
        {
            "{baseDir}", "{basedir}", "{skillDir}", "${CLAUDE_SKILL_DIR}", "$CLAUDE_SKILL_DIR",
            "${SKILL_DIR}", "$SKILL_DIR", "${SKILL_ROOT}", "$SKILL_ROOT", "{SKILL_DIR}",
        };

        internal static bool TryTranslate(string? command, SkillToolContext context, out Translated translated)
        {
            translated = default;
            if (string.IsNullOrWhiteSpace(command) || command.Length > 4096
                || command.IndexOfAny(new[] { '\n', '\r', '|', ';', '>', '<', '`', '&' }) >= 0
                || command.Contains("$(", StringComparison.Ordinal))
                return false;

            List<string> words = SkillScriptRunner.SplitArguments(command);
            if (words.Count < 2)
                return false;

            var packages = new List<string>();
            int at;
            bool viaUv = string.Equals(words[0], "uv", StringComparison.Ordinal);
            if (viaUv)
            {
                if (!string.Equals(words[1], "run", StringComparison.Ordinal))
                    return false;
                at = 2;
                while (at < words.Count && words[at].StartsWith('-'))
                {
                    string option = words[at];
                    if (option is "-q" or "--quiet" or "--no-project" or "--script" or "--isolated")
                    {
                        at++;
                    }
                    else if (option == "--with" && at + 1 < words.Count)
                    {
                        packages.AddRange(Split(words[at + 1]));
                        at += 2;
                    }
                    else if (option.StartsWith("--with=", StringComparison.Ordinal))
                    {
                        packages.AddRange(Split(option.Substring("--with=".Length)));
                        at++;
                    }
                    else if (option == "--python" && at + 1 < words.Count)
                    {
                        at += 2;   // the host's own interpreter runs it
                    }
                    else
                    {
                        return false;
                    }
                }
                // `uv run python script.py` is the same request.
                if (at < words.Count && words[at] is "python" or "python3")
                    at++;
            }
            else if (words[0] is "python" or "python3")
            {
                at = 1;
            }
            else
            {
                return false;
            }
            if (at >= words.Count)
                return false;

            string written = words[at].Replace('\\', '/');
            bool named = false;
            foreach (string placeholder in Placeholders)
            {
                if (written.StartsWith(placeholder + "/", StringComparison.Ordinal))
                {
                    written = written.Substring(placeholder.Length + 1);
                    named = true;
                    break;
                }
            }

            var matches = new List<(Skill Skill, string Path)>();
            foreach (Skill skill in context.Reachable)
            {
                string relative = written;
                if (Path.IsPathRooted(written))
                {
                    string root = skill.RootDirectory.Replace('\\', '/').TrimEnd('/') + "/";
                    if (!written.StartsWith(root, SkillPathGuard.PathComparison))
                        continue;
                    relative = written.Substring(root.Length);
                }
                if (skill.Files.Any(f => f.Kind == SkillFileKind.Script
                        && string.Equals(f.Path, relative, SkillPathGuard.PathComparison)))
                    matches.Add((skill, relative));
            }
            if (matches.Count != 1)
                return false;

            // A bare relative path that the workspace also has is the model's own file.
            if (!named && !Path.IsPathRooted(written) && context.Workspace is { } workspace
                && File.Exists(Path.Combine(workspace.WorkDirectory, written)))
                return false;
            // `python3 scripts/x.py` without a placeholder is only a skill's script when uv
            // or the placeholder says so; plain python on a relative path stays the model's.
            if (!viaUv && !named && !Path.IsPathRooted(written))
                return false;

            translated = new Translated(matches[0].Skill, matches[0].Path, words.Skip(at + 1).ToList(), packages);
            return true;
        }

        private static IEnumerable<string> Split(string value) =>
            value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => InlineScriptMetadata.ToInstallSpec(p.Trim()))
                .OfType<string>();
    }
}
