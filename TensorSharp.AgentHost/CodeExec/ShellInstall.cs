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
using System.Linq;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>What one install command in a line is asking for.</summary>
    /// <param name="Language">Which installer answers it.</param>
    /// <param name="Packages">The names, validated. Empty means "whatever the manifest says".</param>
    /// <param name="Segment">Where the command sits in the line, so it can be substituted out.</param>
    public readonly record struct ShellInstallRequest(
        CodeLanguage Language, IReadOnlyList<string> Packages, ShellSegment Segment);

    /// <summary>
    /// Reading a typed <c>pip install</c> well enough that the HOST can perform it.
    ///
    /// <para>
    /// This is what makes the shell tool's network policy hold on every platform instead
    /// of only on macOS. The obvious design — classify the command, then hand it a socket
    /// and let it run — has two holes that cannot both be closed. The command line is
    /// written by the model, so <c>--index-url</c> points the installer wherever it likes;
    /// and the socket belongs to the whole line, so anything sharing it with the install
    /// shares its reach. Screening arguments patches the first hole and not the second,
    /// and on a host whose sandbox cannot pin egress to a proxy (bubblewrap is
    /// all-or-nothing about the network) the second hole is the whole internet.
    /// </para>
    /// <para>
    /// So the host does not run the model's install at all. It READS it — tool, packages,
    /// nothing else — validates the names the way it always did, and performs the install
    /// itself through <see cref="PackageInstaller"/> with an argument vector it built:
    /// wheels only, no install scripts, egress pinned to the registry allow-list where the
    /// sandbox can pin it. The rewritten command then runs under the operator's independent
    /// command-network policy: offline by default, or unrestricted only when that broader
    /// permission was explicitly enabled. The ergonomics are unchanged — the model types
    /// <c>pip install pandas</c> and it works — while the guarantee is the one the
    /// two-phase program runner had, restored and now platform-independent.
    /// </para>
    /// <para>
    /// The cost is that this understands a SUBSET of what pip and npm accept, and says so
    /// rather than guessing. An argument it does not recognise is refused by name, because
    /// the alternative — ignoring it — would install something other than what the model
    /// asked for and report success.
    /// </para>
    /// </summary>
    public static class ShellInstall
    {
        /// <summary>
        /// Read every install in <paramref name="command"/>.
        /// </summary>
        /// <param name="workspace">Where a <c>-r requirements.txt</c> is read from, or null.</param>
        /// <param name="requests">One per install command found, in the order they appear.</param>
        /// <param name="error">Why an install cannot be answered here, phrased for the model.</param>
        public static bool TryRead(
            string? command,
            Func<string, string?>? readFile,
            out IReadOnlyList<ShellInstallRequest> requests,
            out string? error,
            bool includePackageRunners = true)
        {
            var found = new List<ShellInstallRequest>();
            requests = found;
            error = null;

            foreach (ShellSegment segment in ShellCommand.SplitSegments(command))
            {
                if (!ShellCommand.IsInstallCommand(segment.Text, includePackageRunners))
                    continue;
                if (!TryReadOne(segment, readFile, out ShellInstallRequest request, out error))
                    return false;
                found.Add(request);
            }

            return true;
        }

        private static bool TryReadOne(
            ShellSegment segment,
            Func<string, string?>? readFile,
            out ShellInstallRequest request,
            out string? error)
        {
            request = default;
            error = null;

            IReadOnlyList<string> words = ShellCommand.WordsOf(segment.Text);
            if (words.Count == 0)
            {
                error = "an empty install command";
                return false;
            }

            // `python3 -m pip install x` is the same request as `pip install x`; unwrap it
            // so one parser handles both, exactly as the classifier does.
            int m = words.ToList().IndexOf("-m");
            string tool = System.IO.Path.GetFileNameWithoutExtension(words[0]);
            if (m >= 0 && m + 1 < words.Count && IsPython(tool))
            {
                words = words.Skip(m + 1).ToList();
                tool = System.IO.Path.GetFileNameWithoutExtension(words[0]);
            }

            CodeLanguage language = LanguageOf(tool);
            if (language == CodeLanguage.Unknown)
            {
                // Ends in the ONE thing to do next, per tool, rather than only in the rule.
                // A model told "this host cannot install that" and nothing else abandons
                // the task; told "run the binary from node_modules/.bin" it finishes it.
                string instead = tool switch
                {
                    "npx" or "npm" =>
                        " If you were trying to RUN something rather than install it, run the binary "
                        + "directly from node_modules/.bin/ — `npx` fetches the package itself, which would "
                        + "go around the host's installer.",
                    "uv" or "uvx" or "poetry" or "pipenv" =>
                        " To install a Python package here, write it as a plain `pip install <name>` and "
                        + "the host will perform it. To RUN a script, call the interpreter directly.",
                    "go" or "cargo" or "gem" or "dotnet" =>
                        $" There is no {tool} package installer on this host. Do the step in Python or "
                        + "JavaScript, or say in your answer that it needs a toolchain the host does not "
                        + "provide.",
                    "apt" or "apt-get" or "brew" or "yum" or "dnf" or "apk" or "pacman" =>
                        " System packages and programs cannot be installed here at all — only Python and "
                        + "JavaScript libraries. If the step needs a PROGRAM, say so in your answer.",
                    _ =>
                        " Write a plain `pip install <name>` or `npm install <name>` and the host will "
                        + "perform it.",
                };
                error = $"'{tool}' is not an installer this host can run on your behalf. "
                      + "It installs Python packages (pip) and JavaScript packages (npm), and nothing "
                      + "else — a program that is not a library cannot be installed here at all."
                      + instead;
                return false;
            }

            var packages = new List<string>();
            bool subcommandRead = false;
            for (int i = 1; i < words.Count; i++)
            {
                string word = words[i];

                if (word.Length == 0)
                    continue;

                // The subcommand is the first word that is not an option, which is the rule
                // the classifier applies (ShellCommand.IsInstallCommand). This used to look
                // at the second word only, and to know fewer subcommands than the classifier
                // does, so the two disagreed and every disagreement became a PACKAGE NAME:
                // `pip -q install x` asked the host for a package called "install", and
                // `npm update`, `pnpm update`, `yarn up` and `npm exec cowsay hi` asked it
                // for "update", "up" and "exec" (plus "cowsay" and "hi"). Those names are
                // valid registry specs, so the host went to the registry for them, reported
                // the outcome as the install, and substituted the model's command out of the
                // line — `npm exec` never ran what it named, and `npm update` updated nothing.
                if (!subcommandRead && !word.StartsWith('-'))
                {
                    subcommandRead = true;
                    if (IsSubcommand(word))
                        continue;
                    error = UnperformableSubcommand(tool, word);
                    return false;
                }

                if (word.StartsWith('-'))
                {
                    if (Harmless.Contains(word, StringComparer.Ordinal))
                        continue;

                    // A requirements file is a list of names, so it can be read and each
                    // line validated — that is still the host choosing what to install.
                    if ((word == "-r" || word == "--requirement") && i + 1 < words.Count)
                    {
                        if (!TryReadRequirements(words[++i], readFile, packages, out error))
                            return false;
                        continue;
                    }

                    error = $"'{word}' is not an option this host can honour. Installs here are "
                          + "performed by the host from the package names you give, so that they "
                          + "reach only this host's package registries — options that change where "
                          + "a package comes from, or run code while installing, cannot be applied. "
                          + "Ask for the packages by name.";
                    return false;
                }

                packages.Add(word);
            }

            // `npm install` on its own means "whatever package.json says", which the host
            // can still perform: it builds the same argument vector with no names and lets
            // npm read the manifest out of the working directory. `pip install` on its own
            // is not the same thing — it is a usage error — so it is refused with the two
            // shapes that do work rather than being reported as an install that happened.
            if (packages.Count == 0 && language == CodeLanguage.Python)
            {
                error = "this asks pip to install nothing. Name the packages — "
                      + "`pip install pandas reportlab` — or point it at a requirements file "
                      + "with `-r requirements.txt`.";
                return false;
            }

            request = new ShellInstallRequest(language, packages, segment);
            return true;
        }

        private static bool TryReadRequirements(
            string path, Func<string, string?>? readFile, List<string> packages, out string? error)
        {
            error = null;
            string? text = readFile?.Invoke(path);
            if (text == null)
            {
                error = $"'{path}' could not be read from this conversation's working directory, "
                      + "so its requirements cannot be installed.";
                return false;
            }

            foreach (string raw in text.Split('\n'))
            {
                string line = raw.Trim();
                int comment = line.IndexOf('#');
                if (comment >= 0)
                    line = line.Substring(0, comment).Trim();
                if (line.Length == 0)
                    continue;
                if (line.StartsWith('-'))
                {
                    error = $"'{path}' contains the option '{line.Split(' ')[0]}', which this host "
                          + "cannot honour — a requirements file it installs from may name packages "
                          + "and versions only.";
                    return false;
                }
                packages.Add(line);
            }
            return true;
        }

        private static bool IsPython(string tool) =>
            tool.Equals("python", StringComparison.OrdinalIgnoreCase)
            || tool.Equals("python3", StringComparison.OrdinalIgnoreCase)
            || tool.Equals("py", StringComparison.OrdinalIgnoreCase)
            || (tool.StartsWith("python3.", StringComparison.OrdinalIgnoreCase)
                && tool["python3.".Length..].All(char.IsAsciiDigit));

        private static CodeLanguage LanguageOf(string tool)
        {
            if (tool.Equals("pip", StringComparison.OrdinalIgnoreCase)
                || tool.Equals("pip3", StringComparison.OrdinalIgnoreCase)
                || tool.StartsWith("pip3.", StringComparison.OrdinalIgnoreCase))
            {
                return CodeLanguage.Python;
            }
            if (tool.Equals("npm", StringComparison.OrdinalIgnoreCase)
                || tool.Equals("pnpm", StringComparison.OrdinalIgnoreCase)
                || tool.Equals("yarn", StringComparison.OrdinalIgnoreCase))
            {
                return CodeLanguage.JavaScript;
            }
            return CodeLanguage.Unknown;
        }

        // Only the subcommands whose result IS an install. `pip download` and `pip wheel`
        // reach the registry too, so the classifier routes them here, but what they produce
        // is archives in a directory: performing an install instead and reporting success
        // left the next command looking for files that were never written.
        private static bool IsSubcommand(string word) =>
            word is "install" or "i" or "add" or "ci";

        /// <summary>
        /// The refusal for a subcommand the classifier counts as reaching the registry but
        /// the host does not perform. Each ends in the form that DOES work here, for the
        /// same reason the unknown-tool refusal does: a model told only "not supported"
        /// abandons the step.
        /// </summary>
        private static string UnperformableSubcommand(string tool, string subcommand)
        {
            string typed = $"{tool} {subcommand}";
            // pnpm and yarn spell it `add`; pip and npm spell it `install`.
            string install = tool.Equals("pnpm", StringComparison.OrdinalIgnoreCase)
                || tool.Equals("yarn", StringComparison.OrdinalIgnoreCase)
                ? $"{tool} add"
                : $"{tool} install";
            return subcommand switch
            {
                // Reached only while package runners count as installs (no general network, or
                // a package allow-list), so the reason is the installer, not the network.
                "exec" =>
                    $"`{typed}` fetches a package in order to run it, which would go around the host's "
                    + "installer — installs here are performed on your behalf, and only by name. Install the "
                    + $"tool by name first — `{install} <name>` — then run its binary directly from "
                    + "node_modules/.bin/.",
                "download" or "wheel" =>
                    $"`{typed}` writes package archives to a directory, and this host does not fetch archives: "
                    + "it installs the packages you name into this session's environment, where your code can "
                    + $"import them. Write `{install} <name>` to use a package; its archive cannot be obtained here.",
                "update" or "up" =>
                    $"`{typed}` is not an install this host can perform on your behalf: it installs the "
                    + "packages you name, and nothing else. To get a newer version, name the package "
                    + $"and the version — `{install} <name>@latest` — and the host will install that.",
                _ =>
                    $"`{typed}` is not an install this host can perform on your behalf: it installs the "
                    + $"packages you name, and nothing else. Write `{install} <name>` and the host will "
                    + "perform it.",
            };
        }

        /// <summary>
        /// Options that change nothing about WHAT is installed or WHERE it comes from, so
        /// they can simply be dropped — the host's own argument vector already implies
        /// most of them.
        /// </summary>
        private static readonly string[] Harmless =
        {
            "-q", "--quiet", "-qq", "-v", "--verbose", "--no-input", "--disable-pip-version-check",
            "--no-warn-script-location", "--no-cache-dir", "--upgrade", "-U", "--no-audit",
            "--no-fund", "--no-progress", "--silent", "--save", "--save-dev", "-D", "--no-package-lock",
            "--no-color", "--yes", "-y",
        };
    }
}
