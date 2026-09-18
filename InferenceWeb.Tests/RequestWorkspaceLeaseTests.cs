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
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class RequestWorkspaceLeaseTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "ts-request-workspace-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Lease_IsPrivateToOneRequest_AndDeletesItsWorkspaceOnDispose()
    {
        var manager = new SessionWorkspaceManager(_root);
        var runner = new AvailableRunner();

        string firstRoot;
        string secondRoot;
        using (RequestWorkspaceLease first = Assert.IsType<RequestWorkspaceLease>(
                   RequestWorkspaceLease.Acquire(manager, runner, "qwen2")))
        using (RequestWorkspaceLease second = Assert.IsType<RequestWorkspaceLease>(
                   RequestWorkspaceLease.Acquire(manager, runner, "qwen2")))
        {
            firstRoot = first.Workspace.Root;
            secondRoot = second.Workspace.Root;
            Assert.NotEqual(first.Workspace.Id, second.Workspace.Id);
            Assert.NotEqual(firstRoot, secondRoot);
            Assert.True(Directory.Exists(firstRoot));
            Assert.True(Directory.Exists(secondRoot));
        }

        Assert.False(Directory.Exists(firstRoot));
        Assert.False(Directory.Exists(secondRoot));
    }

    [Fact]
    public void Lease_IsNotAllocated_WhenCodeToolsCannotBeOffered()
    {
        var manager = new SessionWorkspaceManager(_root);
        var runner = new AvailableRunner();

        Assert.Null(RequestWorkspaceLease.Acquire(manager, runner, "qwen2", allowTools: false));
        Assert.Null(RequestWorkspaceLease.Acquire(manager, runner, "mistral3"));
        Assert.False(Directory.Exists(_root));
    }

    [Fact]
    public void Lease_MakesTheRequestPlanOfferThePatchFirstFileSurface()
    {
        var manager = new SessionWorkspaceManager(_root);
        var runner = new AvailableRunner();
        using RequestWorkspaceLease lease = Assert.IsType<RequestWorkspaceLease>(
            RequestWorkspaceLease.Acquire(manager, runner, "qwen2"));
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--no-skills" }, _root);

        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, Array.Empty<string>(), discovery: false, clientTools: null,
            architecture: "qwen2", contextTokens: 32768, options,
            out IReadOnlyList<string> unknown,
            codeRunner: runner,
            workspace: lease.Workspace);

        Assert.Empty(unknown);
        Assert.NotNull(plan);
        Assert.True(runner.LastPersists);
        Assert.Equal(
            new[]
            {
                SkillToolNames.ReadFile,
                SkillToolNames.ApplyPatch,
                SkillToolNames.WriteFile,
                SkillToolNames.Shell,
            },
            plan.Tools.Select(tool => tool.Name));
        Assert.Contains(CodePrompt.Heading, plan.Prompt.Instructions, StringComparison.Ordinal);
        Assert.Contains("Never rewrite a whole file", plan.Prompt.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void SeparateToolRounds_CreateThenEditTheSameRequestFile()
    {
        var manager = new SessionWorkspaceManager(_root);
        using RequestWorkspaceLease lease = Assert.IsType<RequestWorkspaceLease>(
            RequestWorkspaceLease.Acquire(manager, new AvailableRunner(), "qwen2"));
        using var fileTools = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_root, "scratch"),
        });

        CodeExecResult created = fileTools.WriteFile(
            new ShellTools.WriteRequest("main.py", "answer = 40 + 1\nprint(answer)\n"),
            lease.Workspace);
        CodeExecResult edited = fileTools.ApplyPatch(
            "*** Begin Patch\n*** Update File: main.py\n@@\n-answer = 40 + 1\n+answer = 40 + 2\n*** End Patch",
            lease.Workspace);

        Assert.True(created.Ok, created.Content);
        Assert.True(edited.Ok, edited.Content);
        Assert.Equal(
            "answer = 40 + 2\nprint(answer)\n",
            File.ReadAllText(Path.Combine(lease.Workspace.WorkDirectory, "main.py")));
    }

    private sealed class AvailableRunner : ICodeRunner
    {
        public bool LastPersists { get; private set; }
        public bool CanRun => true;
        public string UnavailableReason => null;

        public ToolFunction Declare() =>
            new() { Name = SkillToolNames.Shell, Description = "runs commands" };

        public IReadOnlyList<ToolFunction> DeclareTools(bool persists)
        {
            LastPersists = persists;
            string[] names = persists
                ? new[]
                {
                    SkillToolNames.ReadFile,
                    SkillToolNames.ApplyPatch,
                    SkillToolNames.WriteFile,
                    SkillToolNames.Shell,
                }
                : new[] { SkillToolNames.Shell };
            return names.Select(name => new ToolFunction { Name = name, Description = name }).ToArray();
        }

        public SkillToolResult Execute(
            ToolCall call, IReadOnlyList<CodeInputFile> inputFiles = null,
            Action<string> onOutput = null, SessionWorkspace workspace = null,
            IReadOnlyList<string> skillDirectories = null) =>
            SkillToolResult.Failure("not used in these tests");
    }
}
