// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.CodeExec;

/// <summary>A capability bound to one child workspace; call arguments cannot widen it.</summary>
internal sealed class WorkspaceCodeRunner : ICodeRunner
{
    private readonly ICodeRunner _parent;
    private readonly SessionWorkspace _workspace;
    private readonly bool _allowWrite;
    private readonly IReadOnlyList<ToolFunction> _tools;

    internal WorkspaceCodeRunner(ICodeRunner parent, SessionWorkspace workspace, bool allowWrite)
    {
        _parent = parent;
        _workspace = workspace;
        _allowWrite = allowWrite;
        _tools = parent.DeclareWorkspaceTools(allowWrite);
    }

    public bool CanRun => _parent.CanRun;
    public string? UnavailableReason => _parent.UnavailableReason;
    public IShellBackend? Backend => Allows(SkillToolNames.Shell) ? _parent.Backend : null;
    public ToolFunction Declare() => _tools.First();
    public IReadOnlyList<ToolFunction> DeclareTools() => _tools;
    public IReadOnlyList<ToolFunction> DeclareTools(bool persists) => _tools;
    public IReadOnlyList<ToolFunction> DeclareWorkspaceTools(bool allowWrite) =>
        _tools.Where(t => (allowWrite && _allowWrite) || t.Name == SkillToolNames.ReadFile).ToArray();
    public ICodeRunner ForkForWorkspace(SessionWorkspace workspace, bool allowWrite) =>
        new WorkspaceCodeRunner(_parent, workspace, allowWrite && _allowWrite);

    private bool Allows(string name) => _tools.Any(t => t.Name == SkillToolNames.CanonicalName(name))
        || (_allowWrite && SkillToolNames.ResolveFileTool(name) == SkillToolNames.EditFile
            && _tools.Any(t => t.Name == SkillToolNames.ApplyPatch));

    public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
        Action<string>? onOutput = null, SessionWorkspace? workspace = null,
        IReadOnlyList<string>? skillDirectories = null)
    {
        if (!Allows(call.Name))
            return SkillToolResult.Failure("This tool is not permitted in the subagent workspace.");
        if (workspace != null && !ReferenceEquals(workspace, _workspace))
            return SkillToolResult.Failure("A subagent cannot use another agent's workspace.");
        // Files were explicitly staged at spawn. Never follow caller-supplied host paths.
        return _parent.Execute(call, Array.Empty<CodeInputFile>(), onOutput, _workspace, skillDirectories);
    }

    public bool CanInstallPackages => Allows(SkillToolNames.Shell) && _parent.CanInstallPackages;
    public bool CanInstallPackagesFor(string language) => CanInstallPackages && _parent.CanInstallPackagesFor(language);
    public string? InstallPackages(string language, IReadOnlyList<string> packages,
        SessionWorkspace workspace, Action<string>? onOutput = null) =>
        ReferenceEquals(workspace, _workspace) && CanInstallPackages
            ? _parent.InstallPackages(language, packages, _workspace, onOutput)
            : "Package installation is not permitted outside this subagent's workspace.";
}
