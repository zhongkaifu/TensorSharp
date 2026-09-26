// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.Agents;

/// <summary>Owns a child's files and capability-scoped tools for the request lifetime.
/// Inputs and outputs cross workspace boundaries only through bounded copies.</summary>
internal sealed class AgentWorkspace : IDisposable
{
    internal const int MaxTransferFiles = 128;
    internal const int MaxTransferBytes = 64 * 1024 * 1024;
    private static readonly Lazy<string> StorageRoot = new(CreateStorageRoot);
    private readonly SessionWorkspaceManager _manager;
    private readonly SkillToolContext _parent;
    private readonly Dictionary<string, byte[]> _inputs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, byte[]> _handedOff = new(StringComparer.Ordinal);
    private readonly string _agentName;
    private bool _disposed;

    private AgentWorkspace(SkillToolContext parent, string agentId)
    {
        _parent = parent;
        _agentName = SafeName(agentId);
        _manager = new SessionWorkspaceManager(StorageRoot.Value);
        Workspace = _manager.GetOrCreate(agentId);
        Context = null!;
    }

    internal SessionWorkspace Workspace { get; }
    internal SkillToolContext Context { get; private set; }

    private static string CreateStorageRoot()
    {
        // System temp is deliberately accessible to sandboxed runtime programs on
        // macOS. Agent files belong under the denied home boundary, never there.
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrWhiteSpace(home) || !Path.IsPathRooted(home))
            throw new IOException("A private user home is required for isolated subagent workspaces.");
        home = Path.GetFullPath(home);
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string withinHome = string.IsNullOrWhiteSpace(local) ? ".." : Path.GetRelativePath(home, local);
        if (Path.IsPathRooted(withinHome) || withinHome == ".."
            || withinHome.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            local = Path.Combine(home, ".tensorsharp");
        string root = Path.Combine(local, "TensorSharp", "agent-workspaces");
        EnsurePrivateRootHasNoLinks(home, root);
        if (OperatingSystem.IsWindows()) Directory.CreateDirectory(root);
        else Directory.CreateDirectory(root, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        EnsurePrivateRootHasNoLinks(home, root);
        // CreateDirectory's mode applies only to new directories. An existing
        // storage root must also be private before any agent files are created.
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(root, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        new SessionWorkspaceManager(root).SweepOrphans();
        return root;
    }

    private static void EnsurePrivateRootHasNoLinks(string home, string root)
    {
        string current = home;
        foreach (string segment in Path.GetRelativePath(home, root).Split(Path.DirectorySeparatorChar))
        {
            current = Path.Combine(current, segment);
            var directory = new DirectoryInfo(current);
            if (directory.LinkTarget != null
                || (directory.Exists && (directory.Attributes & FileAttributes.ReparsePoint) != 0))
                throw new IOException("Private agent storage cannot traverse a symbolic link.");
        }
    }

    internal static AgentWorkspace Create(SkillToolContext parent, string agentId,
        IReadOnlyList<string>? inputFiles = null, bool mutable = false)
    {
        ArgumentNullException.ThrowIfNull(parent);
        var owner = new AgentWorkspace(parent, agentId);
        try
        {
            owner.StageInputs(inputFiles ?? Array.Empty<string>());
            ICodeRunner? runner = parent.CodeRunner?.ForkForWorkspace(owner.Workspace, mutable);
            owner.Context = new SkillToolContext(parent.Reachable, parent.MaxReadBytes)
            {
                Workspace = owner.Workspace,
                CodeRunner = runner,
                ScriptRunner = mutable
                    ? parent.ScriptRunner?.ForkForWorkspace(owner.Workspace, runner) : null,
                // Input files now live in this workspace. Retaining the parent's host
                // paths here would let later calls silently restage unrelated files.
                CodeInputFiles = Array.Empty<CodeInputFile>(),
            };
            return owner;
        }
        catch
        {
            owner.Dispose();
            throw;
        }
    }

    internal static bool AllowsTool(SkillToolContext parent, bool mutable, string toolName)
    {
        if (toolName is SkillTools.ReadToolName or SkillTools.ListToolName) return true;
        if (toolName == SkillTools.RunToolName) return mutable && parent.ScriptRunner?.CanForkForWorkspace == true;
        string? canonical = SkillToolNames.CanonicalName(toolName);
        return parent.CodeRunner?.DeclareWorkspaceTools(mutable).Any(t => t.Name == canonical) == true;
    }

    internal bool AllowsTool(string toolName)
    {
        if (toolName is SkillTools.ReadToolName or SkillTools.ListToolName) return true;
        if (toolName == SkillTools.RunToolName) return Context.ScriptRunner != null;
        string? canonical = SkillToolNames.CanonicalName(toolName);
        return Context.CodeRunner?.DeclareTools().Any(t => t.Name == canonical) == true;
    }

    private void StageInputs(IReadOnlyList<string> paths)
    {
        if (paths.Count > MaxTransferFiles) throw new ArgumentException($"At most {MaxTransferFiles} input files may be staged.");
        int remaining = MaxTransferBytes;
        using IDisposable? operation = _parent.Workspace?.EnterExecution();
        foreach (string supplied in paths)
        {
            string relative = ValidateRelativePath(supplied);
            if (_inputs.ContainsKey(relative)) continue;
            byte[] bytes;
            if (_parent.Workspace != null && File.Exists(Path.Combine(_parent.Workspace.WorkDirectory, relative)))
            {
                bytes = Read(_parent.Workspace, relative, remaining);
            }
            else
            {
                CodeInputFile? attachment = _parent.CodeInputFiles
                    .Where(f => string.Equals(f.Name, relative, StringComparison.Ordinal))
                    .Select(f => (CodeInputFile?)f).FirstOrDefault();
                if (attachment == null)
                    throw new IOException($"Input file '{relative}' is not available in the parent workspace or attachments.");
                string source = Path.GetFullPath(attachment.Value.SourcePath);
                // Attachments are host-approved inputs, but leaf symlinks are still
                // refused by the anchored read; no arbitrary host path comes from the model.
                bytes = ShellSession.ReadBoundedRegularBytesUnderRoot(Path.GetDirectoryName(source)!, source, remaining);
            }
            CreateFile(Workspace, relative, bytes);
            _inputs[relative] = SHA256.HashData(bytes);
            remaining -= bytes.Length;
        }
    }

    /// <summary>Import prerequisite output without granting live access to its workspace.</summary>
    internal IReadOnlyList<string> ImportDependencyFiles(string dependencyName, AgentWorkspace source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = new List<string>();
        using IDisposable operation = source.Workspace.EnterExecution();
        // Preflight every source before publishing anything in the destination.
        var outputs = source.Outputs().ToArray();
        foreach ((string relative, byte[] bytes) in outputs)
        {
            string destination = "dependencies/" + SafeName(dependencyName) + "/" + relative;
            CreateFile(Workspace, destination, bytes);
            _inputs[destination] = SHA256.HashData(bytes);
            result.Add(destination);
        }
        return result;
    }

    /// <summary>Publish changed/new files to a fresh parent directory for explicit review.
    /// Parent source files are never overwritten or automatically merged.</summary>
    internal IReadOnlyList<SkillProducedFile> Handoff()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        using IDisposable operation = Workspace.EnterExecution();
        var outputs = Outputs().ToArray();
        if (_parent.Workspace == null)
        {
            if (outputs.Length != 0)
                throw new IOException("Subagent produced files, but the parent has no workspace to receive them. Files cannot be handed off.");
            return Array.Empty<SkillProducedFile>();
        }
        using IDisposable parentOperation = _parent.Workspace.EnterExecution();
        string prefix = "agent-results-" + _agentName + "-" + Guid.NewGuid().ToString("N");
        var files = new List<SkillProducedFile>();
        var published = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        try
        {
            foreach ((string relative, byte[] bytes) in outputs)
            {
                byte[] digest = SHA256.HashData(bytes);
                if (_handedOff.TryGetValue(relative, out byte[]? previous) && digest.AsSpan().SequenceEqual(previous)) continue;
                string destination = prefix + "/" + relative;
                CreateFile(_parent.Workspace, destination, bytes);
                files.Add(new SkillProducedFile(destination, bytes.Length,
                    Path.Combine(_parent.Workspace.WorkDirectory, destination)));
                published[relative] = digest;
            }
        }
        catch
        {
            string directory = Path.Combine(_parent.Workspace.WorkDirectory, prefix);
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            throw;
        }
        foreach (var entry in published) _handedOff[entry.Key] = entry.Value;
        return files;
    }

    private IEnumerable<(string Relative, byte[] Bytes)> Outputs()
    {
        int scanned = 0, count = 0, remaining = MaxTransferBytes;
        foreach (string path in WorkspaceScan.Files(Workspace.WorkDirectory, WorkspaceScan.SnapshotOptions))
        {
            if (++scanned > 4096) throw new IOException("Subagent workspace exceeds the 4096-file inspection limit.");
            string relative = ValidateRelativePath(Path.GetRelativePath(Workspace.WorkDirectory, path)
                .Replace(Path.DirectorySeparatorChar, '/'));
            byte[] bytes = Read(Workspace, relative, MaxTransferBytes);
            if (!_inputs.TryGetValue(relative, out byte[]? original)
                || !SHA256.HashData(bytes).AsSpan().SequenceEqual(original))
            {
                if (++count > MaxTransferFiles || bytes.Length > remaining)
                    throw new IOException($"Subagent output exceeds {MaxTransferFiles} files or {MaxTransferBytes} bytes.");
                remaining -= bytes.Length;
                yield return (relative, bytes);
            }
        }
    }

    private static byte[] Read(SessionWorkspace workspace, string relative, int maxBytes)
    {
        RejectLinks(workspace, relative);
        return ShellSession.ReadBoundedRegularBytesUnderRoot(workspace.WorkDirectory,
            Path.Combine(workspace.WorkDirectory, relative), maxBytes);
    }

    private static void CreateFile(SessionWorkspace workspace, string relative, byte[] bytes)
    {
        RejectLinks(workspace, relative);
        bool created = workspace.TryCreateFile(relative, bytes, out bool alreadyExists, out string? error);
        if (!created || alreadyExists)
            throw new IOException(error ?? $"Agent transfer could not create '{relative}' without replacing an existing file.");
    }

    private static void RejectLinks(SessionWorkspace workspace, string relative)
    {
        string current = workspace.WorkDirectory;
        foreach (string segment in relative.Split('/'))
        {
            current = Path.Combine(current, segment);
            var info = new FileInfo(current);
            if (info.LinkTarget != null || (info.Exists && (info.Attributes & FileAttributes.ReparsePoint) != 0))
                throw new IOException("Agent file transfers cannot follow symbolic links.");
        }
    }

    private static string ValidateRelativePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || path.Length > 2048 || Path.IsPathRooted(path)
            || path.Contains(':') || path.Contains('\\') || path.Contains('\0')
            || path.Split('/').Any(p => p is "" or "." or ".."))
            throw new ArgumentException("Agent inputs must be explicit relative file paths without traversal.");
        return path;
    }

    private static string SafeName(string name) =>
        new(name.Select(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' ? c : '_').Take(96).ToArray());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _manager.Release(Workspace.Id);
    }
}
