// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Turns a code failure into a small, actionable edit loop. Tracebacks are useful to
    /// a human, but a model also needs an explicit choice between changing the faulty
    /// region and sending the entire file again. This hint is emitted only by callers
    /// that offered the persistent file tools for the current request.
    /// </summary>
    public static class CodeRepairHint
    {
        private const int MaxSourceBytes = 1024 * 1024;

        private static readonly Regex PythonFrame = new(
            "^\\s*File \\\"(?<path>[^\\\"]+)\\\", line (?<line>[0-9]+)(?:, in .*)?\\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex JavaScriptLocation = new(
            @"^(?<path>.+\.(?:[cm]?js|jsx|tsx?)):(?<line>[0-9]+)(?::[0-9]+)?\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

        private static readonly Regex CompilerLocation = new(
            @"(?m)^(?<path>[^:\r\n]+\.[A-Za-z0-9_+-]+):(?<line>[0-9]+)(?::[0-9]+)?:\s*(?:fatal\s+)?error\b",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

        private static readonly Regex ParenthesizedCompilerLocation = new(
            @"(?m)^(?<path>[^\r\n()]+\.[A-Za-z0-9_+-]+)\((?<line>[0-9]+)(?:,[0-9]+)?\):\s*(?:fatal\s+)?error\b",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

        /// <summary>
        /// Advice plus a bounded source excerpt, or null when the failure is not known to
        /// come from generated code. No file outside the workspace is ever opened.
        /// </summary>
        public static string? Create(
            string? diagnosis,
            SessionWorkspace workspace,
            string? currentDirectory,
            bool networkConfined)
        {
            ArgumentNullException.ThrowIfNull(workspace);

            if (!TryFindSource(diagnosis, workspace, currentDirectory,
                    out string? displayPath, out string? fullPath, out int line))
            {
                // Inline code and installed-library frames have no workspace file the
                // declared patch tool can change. Naming apply_patch there would prescribe
                // an impossible next action; the original diagnostic remains intact.
                return null;
            }

            CodeDiagnostics.FailureCause cause = CodeDiagnostics.ClassifyFailure(
                diagnosis, CodeLanguage.Unknown, networkConfined);
            bool compilerDiagnostic = CompilerLocation.IsMatch(diagnosis ?? string.Empty)
                || ParenthesizedCompilerLocation.IsMatch(diagnosis ?? string.Empty);
            if (cause.Source == CodeDiagnostics.FailureSource.Environment
                || (cause.Source != CodeDiagnostics.FailureSource.Code && !compilerDiagnostic))
            {
                return null;
            }

            var sb = new StringBuilder();
            sb.Append("\nThis failure came from the program code. ");

            string? excerpt = null;
            try
            {
                if (ShellSession.TryReadBoundedRegularTextUnderRoot(
                        workspace.WorkDirectory, fullPath!, MaxSourceBytes, out string content))
                {
                    IReadOnlyList<string> lines = NumberedListing.SplitLines(content);
                    int total = NumberedListing.RealLineCount(lines);
                    if (!NumberedListing.LooksBinary(lines) && line <= total)
                    {
                        int first = Math.Max(1, line - 3);
                        int last = Math.Min(total, line + 3);
                        var excerptBuilder = new StringBuilder();
                        NumberedListing.Append(
                            excerptBuilder,
                            lines,
                            first - 1,
                            last - 1,
                            NumberedListing.MaxExcerptChars,
                            NumberedListing.MaxExcerptLineChars,
                            out int lastShownIndex);
                        excerpt = excerptBuilder.ToString();

                        // Credit only complete, unclipped source lines. A long line is
                        // deliberately shortened in the prompt, and the character budget
                        // can stop before the requested end; either must remain unread in
                        // the ledger or replace_all could later be authorized over bytes
                        // the model never actually received.
                        bool exact = lastShownIndex >= first - 1;
                        for (int index = first - 1; exact && index <= lastShownIndex; index++)
                            exact = lines[index].Length <= NumberedListing.MaxExcerptLineChars;
                        if (exact)
                        {
                            workspace.Reads.Record(
                                fullPath!,
                                content.Replace("\r\n", "\n", StringComparison.Ordinal),
                                first,
                                lastShownIndex + 1,
                                complete: first == 1 && lastShownIndex + 1 == total);
                        }
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException
                                          or PathTooLongException)
            {
                // The location is still useful. A concurrent delete or unreadable
                // file must not replace the real program failure with host noise.
            }

            if (!string.IsNullOrEmpty(excerpt))
            {
                sb.Append("The failing region in '").Append(displayPath)
                  .Append("' around line ").Append(line.ToString(CultureInfo.InvariantCulture))
                  .Append(" is:\n").Append(excerpt);
            }
            else
            {
                sb.Append("The traceback points to '").Append(displayPath)
                  .Append("' at line ").Append(line.ToString(CultureInfo.InvariantCulture))
                  .Append(". ");
            }

            sb.Append("Fix the smallest incorrect region with `")
              .Append(ShellTools.PatchToolName)
              .Append("` for a change in one file or across multiple files, then run the same check again. Do not use `")
              .Append(ShellTools.WriteToolName)
              .Append("` or re-type the whole file for a local bug. If the patch context no longer "
                    + "matches, read that region and retry against its current text.\n");

            return sb.ToString();
        }

        private static bool TryFindSource(
            string? diagnosis,
            SessionWorkspace workspace,
            string? currentDirectory,
            out string? displayPath,
            out string? fullPath,
            out int line)
        {
            displayPath = null;
            fullPath = null;
            line = 0;
            if (string.IsNullOrEmpty(diagnosis))
                return false;

            string from = string.IsNullOrWhiteSpace(currentDirectory)
                ? workspace.WorkDirectory
                : currentDirectory!;

            // Runtime stacks are parsed as blocks rather than searched as arbitrary
            // substrings. Otherwise an exception message or echoed workspace input can
            // forge `File "..."` / `at ...` text and point the repair tool at an
            // unrelated file. Within a genuine block the deepest resolvable frame wins;
            // installed-library paths are rejected by the workspace guard, so an earlier
            // workspace caller may still be selected.
            foreach (IReadOnlyList<SourceLocation> locations in new[]
                     { PythonLocations(diagnosis!), JavaScriptLocations(diagnosis!) })
            {
                for (int index = locations.Count - 1; index >= 0; index--)
                {
                    SourceLocation location = locations[index];
                    string candidate = NormalizeCandidate(location.Path);
                    if (candidate.Length == 0 || string.Equals(candidate, "command", StringComparison.Ordinal))
                        continue;
                    if (location.Line < 1)
                        continue;
                    // Existence is only a selection heuristic: it lets an earlier real
                    // workspace frame win over a deepest stale/missing one. The actual
                    // read is still an exact-handle, root-anchored operation below.
                    if (!workspace.TryResolveFrom(from, candidate, out string resolved, out _)
                        || !File.Exists(resolved))
                        continue;

                    string relative = Path.GetRelativePath(from, resolved).Replace('\\', '/');
                    displayPath = relative.Length == 0 ? Path.GetFileName(resolved) : relative;
                    fullPath = resolved;
                    line = location.Line;
                    return true;
                }
            }

            foreach (Regex pattern in new[] { CompilerLocation, ParenthesizedCompilerLocation })
            {
                MatchCollection matches = pattern.Matches(diagnosis!);
                for (int index = matches.Count - 1; index >= 0; index--)
                {
                    Match match = matches[index];
                    string candidate = NormalizeCandidate(match.Groups["path"].Value);
                    if (!int.TryParse(match.Groups["line"].Value, NumberStyles.None,
                            CultureInfo.InvariantCulture, out int candidateLine) || candidateLine < 1
                        || !workspace.TryResolveFrom(from, candidate, out string resolved, out _)
                        || !File.Exists(resolved))
                    {
                        continue;
                    }

                    string relative = Path.GetRelativePath(from, resolved).Replace('\\', '/');
                    displayPath = relative.Length == 0 ? Path.GetFileName(resolved) : relative;
                    fullPath = resolved;
                    line = candidateLine;
                    return true;
                }
            }

            return false;
        }

        private readonly record struct SourceLocation(string Path, int Line);

        private static IReadOnlyList<SourceLocation> PythonLocations(string diagnosis)
        {
            string[] lines = diagnosis.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
            int header = -1;
            for (int index = lines.Length - 1; index >= 0; index--)
            {
                if (string.Equals(
                    lines[index].Trim(),
                    "Traceback (most recent call last):",
                    StringComparison.Ordinal))
                {
                    header = index;
                    break;
                }
            }

            var found = new List<SourceLocation>();
            if (header >= 0)
            {
                bool complete = false;
                for (int index = header + 1; index < lines.Length; index++)
                {
                    if (IsPythonTerminalLine(lines[index].Trim()))
                    {
                        complete = true;
                        break;
                    }
                    AddPythonLocation(lines[index], found);
                }
                return complete ? found : Array.Empty<SourceLocation>();
            }

            // Compile-time SyntaxError diagnostics have no Traceback header. Bind only
            // to the closest preceding File line in that diagnostic, not every such
            // substring elsewhere in output.
            for (int terminal = lines.Length - 1; terminal >= 0; terminal--)
            {
                string value = lines[terminal].TrimStart();
                if (!value.StartsWith("SyntaxError", StringComparison.Ordinal)
                    && !value.StartsWith("IndentationError", StringComparison.Ordinal)
                    && !value.StartsWith("TabError", StringComparison.Ordinal))
                {
                    continue;
                }

                for (int index = terminal - 1; index >= 0; index--)
                {
                    string trimmed = lines[index].Trim();
                    if (IsPythonTerminalLine(trimmed)
                        || string.Equals(trimmed, "Traceback (most recent call last):", StringComparison.Ordinal))
                    {
                        break;
                    }
                    if (AddPythonLocation(lines[index], found))
                        return found;
                }
                break;
            }
            return found;
        }

        private static bool AddPythonLocation(string line, List<SourceLocation> found)
        {
            Match match = PythonFrame.Match(line);
            if (!match.Success
                || !int.TryParse(match.Groups["line"].Value, NumberStyles.None,
                    CultureInfo.InvariantCulture, out int number)
                || number < 1)
            {
                return false;
            }
            found.Add(new SourceLocation(match.Groups["path"].Value, number));
            return true;
        }

        private static bool IsPythonTerminalLine(string value)
        {
            int colon = value.IndexOf(':');
            string name = colon >= 0 ? value.Substring(0, colon) : value;
            return name.EndsWith("Error", StringComparison.Ordinal)
                || name.EndsWith("Exception", StringComparison.Ordinal);
        }

        private static IReadOnlyList<SourceLocation> JavaScriptLocations(string diagnosis)
        {
            string[] lines = diagnosis.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
            int error = -1;
            for (int index = lines.Length - 1; index >= 0; index--)
            {
                if (IsJavaScriptErrorHeader(lines[index].Trim()))
                {
                    error = index;
                    break;
                }
            }
            if (error < 0)
                return Array.Empty<SourceLocation>();

            var found = new List<SourceLocation>();
            bool stackStarted = false;
            for (int index = error + 1; index < lines.Length; index++)
            {
                string trimmed = lines[index].TrimStart();
                if (!trimmed.StartsWith("at ", StringComparison.Ordinal))
                {
                    if (stackStarted && trimmed.Length > 0)
                        break;
                    continue;
                }

                stackStarted = true;
                if (TryParseJavaScriptStackLocation(trimmed, out SourceLocation location))
                    found.Add(location);
            }

            if (found.Count == 0 && error > 0)
            {
                // Node prints a syntax-error file location above the source excerpt and
                // error header. Search only that adjacent diagnostic preamble.
                for (int index = error - 1; index >= 0 && index >= error - 6; index--)
                {
                    if (TryParseJavaScriptDirectLocation(lines[index].Trim(), out SourceLocation location))
                    {
                        found.Add(location);
                        break;
                    }
                }
            }
            return found;
        }

        private static bool IsJavaScriptErrorHeader(string line)
        {
            int colon = line.IndexOf(':');
            string name = colon >= 0 ? line.Substring(0, colon) : line;
            int code = name.IndexOf(" [", StringComparison.Ordinal);
            if (code > 0)
                name = name.Substring(0, code);
            return string.Equals(name, "Error", StringComparison.Ordinal)
                || name.EndsWith("Error", StringComparison.Ordinal);
        }

        private static bool TryParseJavaScriptStackLocation(
            string stackLine,
            out SourceLocation location)
        {
            location = default;
            string body = stackLine.Substring(3).Trim(); // anchored `at ` above
            bool wrapped = body.EndsWith(")", StringComparison.Ordinal);
            if (wrapped)
                body = body.Substring(0, body.Length - 1);

            Match match = JavaScriptLocation.Match(body);
            if (!match.Success
                || !int.TryParse(match.Groups["line"].Value, NumberStyles.None,
                    CultureInfo.InvariantCulture, out int number)
                || number < 1)
            {
                return false;
            }

            string path = match.Groups["path"].Value;
            if (wrapped)
            {
                int wrapper = path.IndexOf('(');
                if (wrapper >= 0)
                    path = path.Substring(wrapper + 1);
            }
            else
            {
                int uri = path.IndexOf("file://", StringComparison.OrdinalIgnoreCase);
                int unix = path.IndexOf('/');
                Match drive = Regex.Match(path, @"[A-Za-z]:[\\/]", RegexOptions.CultureInvariant);
                int start = uri >= 0 ? uri
                    : unix >= 0 ? unix
                    : drive.Success ? drive.Index
                    : 0;
                path = path.Substring(start);
            }

            location = new SourceLocation(path.Trim(), number);
            return true;
        }

        private static bool TryParseJavaScriptDirectLocation(
            string line,
            out SourceLocation location)
        {
            location = default;
            Match match = JavaScriptLocation.Match(line);
            if (!match.Success
                || !int.TryParse(match.Groups["line"].Value, NumberStyles.None,
                    CultureInfo.InvariantCulture, out int number)
                || number < 1)
            {
                return false;
            }
            location = new SourceLocation(match.Groups["path"].Value, number);
            return true;
        }

        private static string NormalizeCandidate(string path)
        {
            string candidate = (path ?? string.Empty).Trim();

            // Node prints bare frames as "at file:///…/main.mjs:line:column" and
            // sometimes inserts "async" or a function name before the path. The regex
            // deliberately accepts that whole prefix so it also handles paths in
            // parentheses; peel it here before asking the workspace guard to resolve it.
            if (candidate.StartsWith("at ", StringComparison.Ordinal))
            {
                int separator = candidate.LastIndexOf(' ');
                if (separator >= 0)
                    candidate = candidate.Substring(separator + 1);
            }

            if (candidate.StartsWith("-->", StringComparison.Ordinal))
                candidate = candidate.Substring(3).TrimStart();

            if (candidate.StartsWith("file://", StringComparison.OrdinalIgnoreCase)
                && Uri.TryCreate(candidate, UriKind.Absolute, out Uri? uri)
                && uri.IsFile)
            {
                candidate = uri.LocalPath;
            }

            return candidate;
        }
    }
}
