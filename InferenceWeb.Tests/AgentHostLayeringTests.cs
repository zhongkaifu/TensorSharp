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
using System.Linq;
using System.Reflection;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The layering the split exists to create: TensorSharp.Runtime carries the
/// fundamentals a plain OpenAI/Ollama server needs, and TensorSharp.AgentHost
/// adds the agentic layer — skills and code execution — ON TOP of it.
///
/// <para>
/// A compile-time reference would enforce this on its own, except that adding
/// the reverse reference is a one-line mistake nobody would notice until the
/// runtime package quietly started shipping a sandbox, an egress proxy and a
/// script runner to every consumer that wanted a tokenizer. These tests fail
/// the moment that happens.
/// </para>
/// </summary>
public class AgentHostLayeringTests
{
    private static Assembly Runtime => typeof(TensorSharp.Runtime.ChatMessage).Assembly;
    private static Assembly AgentHost => typeof(TensorSharp.AgentHost.Skills.Skill).Assembly;

    [Fact]
    public void TheyAreSeparateAssemblies()
    {
        Assert.NotEqual(Runtime.GetName().Name, AgentHost.GetName().Name);
        Assert.Equal("TensorSharp.Runtime", Runtime.GetName().Name);
        Assert.Equal("TensorSharp.AgentHost", AgentHost.GetName().Name);
    }

    [Fact]
    public void AgentHost_BuildsOnTheRuntime()
    {
        Assert.Contains(
            AgentHost.GetReferencedAssemblies(),
            a => a.Name == "TensorSharp.Runtime");
    }

    [Fact]
    public void TheRuntime_DoesNotReferenceTheAgentHost()
    {
        // The direction that must never invert: a host serving plain chat
        // completions takes TensorSharp.Runtime and carries no skill registry,
        // no sandbox, no code runner.
        Assert.DoesNotContain(
            Runtime.GetReferencedAssemblies(),
            a => a.Name == "TensorSharp.AgentHost");
    }

    [Fact]
    public void TheRuntime_ShipsNoSkillOrCodeExecutionTypes()
    {
        // Namespace-level guard, so moving a type back by hand fails here even
        // if it compiles.
        string[] strays = Runtime.GetExportedTypes()
            .Select(t => t.FullName ?? string.Empty)
            .Where(n => n.Contains(".Skills.", StringComparison.Ordinal)
                        || n.Contains(".CodeExec.", StringComparison.Ordinal)
                        || n.Contains(".Agents.", StringComparison.Ordinal))
            .ToArray();

        Assert.True(strays.Length == 0,
            "These agentic types are back in TensorSharp.Runtime:\n  " + string.Join("\n  ", strays));
    }

    [Fact]
    public void TheAgenticSurface_LivesInTheAgentHost()
    {
        // The other half of the same guarantee: the split actually moved the
        // feature rather than leaving a shim behind.
        foreach (Type t in new[]
        {
            typeof(TensorSharp.AgentHost.Skills.SkillRegistry),
            typeof(TensorSharp.AgentHost.Skills.SkillSandboxMode),
            typeof(TensorSharp.AgentHost.Skills.SessionWorkspace),
            typeof(TensorSharp.AgentHost.CodeExec.ShellRunner),
            typeof(TensorSharp.AgentHost.CodeExec.ConfinedProcess),
            typeof(TensorSharp.AgentHost.CodeExec.EgressProxy),
            typeof(TensorSharp.AgentHost.Agents.SubAgentRuntime),
            typeof(TensorSharp.AgentHost.Agents.SubAgentScope),
            typeof(TensorSharp.AgentHost.Agents.SubAgentOptions),
        })
        {
            Assert.Equal("TensorSharp.AgentHost", t.Assembly.GetName().Name);
        }
    }
}
