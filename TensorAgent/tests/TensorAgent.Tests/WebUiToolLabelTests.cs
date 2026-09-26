// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The page's <c>TOOL_LABEL</c> table against the tools the host actually runs.
///
/// <para>
/// A tool with no entry still works, but its progress and its kept trace line fall
/// back to the tool's raw name ("Running spawn agent"), which is how the five
/// sub-agent tools read on the phone after the host started offering them. An entry
/// for a tool the host no longer runs is dead weight that reads as if it did. So
/// the table is held equal to the host's own name lists, in both directions.
/// </para>
/// </summary>
public sealed class WebUiToolLabelTests
{
    private static readonly string Repo = FindRepoRoot();

    private static string PageScript =>
        File.ReadAllText(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Core", "WebUi", "tensoragent.js"));

    [Fact]
    public void ToolLabelsCoverExactlyTheToolsTheHostReportsProgressFor()
    {
        string script = PageScript;
        int start = script.IndexOf("var TOOL_LABEL = {", StringComparison.Ordinal);
        Assert.True(start >= 0, "tensoragent.js no longer declares TOOL_LABEL; update this test with it.");
        int end = script.IndexOf("};", start, StringComparison.Ordinal);
        Assert.True(end > start, "TOOL_LABEL's closing brace was not found.");

        string[] labelled = Regex.Matches(
                script[start..end],
                @"^\s*([a-z_]+):\s*\['[^']+',\s*'[^']+'\],?\s*$",
                RegexOptions.Multiline | RegexOptions.CultureInvariant)
            .Select(match => match.Groups[1].Value)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        // Code tools include edit_file: it is no longer declared, but the adapter still
        // dispatches it, so a model that calls it by habit still produces its frames.
        string[] hostTools = SkillToolNames.CodeTools
            .Concat(new[] { SkillTools.ListToolName, SkillTools.ReadToolName, SkillTools.RunToolName })
            .Concat(MultiAgentTools.Create().Select(tool => tool.Name))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(hostTools, labelled);
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException($"no TensorAgent above {AppContext.BaseDirectory}");
    }
}
