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

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// A file of the USER's that a <c>shell</c> command may read — an upload from the
    /// conversation, staged into the run's working directory under <paramref name="Name"/>.
    /// </summary>
    /// <param name="Name">The file name the code opens, e.g. <c>report.md</c>. No directories.</param>
    /// <param name="SourcePath">Where the host holds the file. Never shown to the model.</param>
    public readonly record struct CodeInputFile(string Name, string SourcePath);

    /// <summary>
    /// A file a <c>shell</c> command produced and the host kept for the user.
    ///
    /// <para>
    /// This is the STRUCTURED copy of what the tool-result text already tells the model.
    /// It exists for the host's own UI: a small model relays a download link erratically,
    /// so a user interface that wants the link to always appear must get it from here,
    /// not by parsing the model's prose.
    /// </para>
    /// </summary>
    /// <param name="Name">Path relative to the run's working directory.</param>
    /// <param name="Bytes">Its size.</param>
    /// <param name="Url">Where the user fetches it: a URL on a server, a file path on the CLI.</param>
    public readonly record struct SkillProducedFile(string Name, long Bytes, string Url);

    /// <summary>
    /// Copies a run's output files somewhere durable and hands back where a user
    /// fetches them. Supplied by the host (a server points it at its artifact store);
    /// takes the directory the run wrote in and a predicate naming what NOT to keep —
    /// the files that were already there before the run.
    /// </summary>
    public delegate IReadOnlyList<SkillProducedFile> WorkspaceFileCapture(
        string workDirectory, Func<string, bool>? exclude);

    /// <summary>
    /// Something that can answer the code-execution tools.
    ///
    /// <para>
    /// Deliberately declared here rather than beside the implementation, so the dependency
    /// runs one way: code execution knows about skills (it borrows their sandbox and path
    /// guard), and skills know only this interface. Without it the two namespaces would
    /// reference each other, and the tool dispatch would drag the whole code-execution
    /// stack into every host that only ever wanted skills.
    /// </para>
    /// </summary>
    public interface ICodeRunner
    {
        /// <summary>Whether this host will actually run code right now.</summary>
        bool CanRun { get; }

        /// <summary>
        /// What actually executes a launch here, or null on a host that has not said.
        ///
        /// <para>
        /// It is on the interface because a SKILL's bundled script has to run the same
        /// way the model's own program does, and until this existed it did not: the
        /// request planner built its script runner with no backend, which falls back to
        /// launching a child process. On a desktop that quietly ran a DIFFERENT
        /// interpreter — the system python3, without the packages the app staged — and
        /// on iOS, where no process can be started at all, it could not run anything.
        /// One host, one way of running code.
        /// </para>
        /// </summary>
        CodeExec.IShellBackend? Backend => null;

        /// <summary>Why <see cref="CanRun"/> is false, or null.</summary>
        string? UnavailableReason { get; }

        /// <summary>
        /// The sampling this host wants for a turn in which code can be run, given what
        /// the caller asked for. The default is to change nothing.
        ///
        /// <para>
        /// It hangs off the RUNNER rather than being decided by the server because the
        /// runner is what knows this is a coding turn and what the operator configured.
        /// The caller applies the result to the whole turn: the tool rounds and the final
        /// answer alike, because a turn cannot be told in advance which of its generations
        /// will be the last one, and neither reference implementation varies sampling
        /// within a turn either.
        /// </para>
        /// </summary>
        [return: System.Diagnostics.CodeAnalysis.NotNullIfNotNull(nameof(requested))]
        SamplingConfig? ForCodingTurn(SamplingConfig? requested) => requested;

        /// <summary>
        /// The tool declaration to offer the model.
        ///
        /// <para>
        /// The runner declares its own tool because only it knows which languages this
        /// host enabled and whether installing is permitted — a declaration that offers a
        /// language the host will refuse costs the model a round to discover.
        /// </para>
        /// </summary>
        ToolFunction Declare();

        /// <summary>
        /// Every tool this runner answers: the program runner and, where the host keeps a
        /// conversation's working directory, the delta editor that fixes the last program
        /// without resending it. Defaults to just <see cref="Declare"/> so a runner that
        /// only runs code needs no change.
        /// </summary>
        IReadOnlyList<ToolFunction> DeclareTools() => new[] { Declare() };

        /// <summary>
        /// Declare the tools, telling the runner whether anything the model does survives
        /// between calls on this request.
        ///
        /// <para>
        /// False for callers that supply no request/session workspace and therefore get a
        /// fresh empty directory per call.
        /// It is passed IN rather than patched out of the finished text afterwards: editing
        /// the prose swapped eight words out of a paragraph whose other sixty went on
        /// asserting that files persist, ending with "Do not re-create what you already
        /// made" — advice that guarantees the model will not re-create a file that is in
        /// fact gone.
        /// </para>
        /// </summary>
        IReadOnlyList<ToolFunction> DeclareTools(bool persists) => DeclareTools();

        /// <summary>Tools that can be safely bound to a separate child workspace.
        /// Custom runners must opt in; declarations and execution must enforce the same scope.</summary>
        IReadOnlyList<ToolFunction> DeclareWorkspaceTools(bool allowWrite) => Array.Empty<ToolFunction>();

        /// <summary>Create a runner permanently bound to this workspace, preserving or
        /// narrowing the host's permissions. Null withholds execution from children.</summary>
        ICodeRunner? ForkForWorkspace(SessionWorkspace workspace, bool allowWrite) => null;

        /// <summary>Run what the call asks for and return the tool result.</summary>
        /// <param name="inputFiles">
        /// Files from the conversation to place in the program's working directory before
        /// it runs, so <c>open("report.md")</c> works on the file the user attached. Null
        /// or empty when the request carried none.
        /// </param>
        /// <param name="onOutput">
        /// Live tap: called per line the tool's install and run phases print, while they
        /// run, so a host can stream progress to its user. May be called from other
        /// threads; null taps nothing. The result still carries the full output.
        /// </param>
        /// <param name="workspace">
        /// The session's persistent workspace. When present, the code runs in its
        /// shared working directory with its accumulated package environment, and what
        /// it writes SURVIVES the call for later steps of the same conversation. Null
        /// keeps the original behavior: a fresh scratch deleted when the call returns.
        /// </param>
        SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null);

        /// <summary>
        /// Whether this runner can install packages into a session workspace at all —
        /// the host's install switch, not a per-call judgement.
        /// </summary>
        bool CanInstallPackages => false;

        /// <summary>
        /// Whether packages for <paramref name="language"/> can be installed. Hosts that
        /// support every language may rely on the aggregate default.
        /// </summary>
        bool CanInstallPackagesFor(string language) => CanInstallPackages;

        /// <summary>
        /// Install <paramref name="packages"/> into <paramref name="workspace"/>'s
        /// environment, under the same rules every host-built install obeys (wheels
        /// only, confined, allow-list honored). Exists so a skill SCRIPT's missing
        /// dependencies can be set up without a round-trip through the model: the script
        /// runner detects the missing import, installs, and re-runs.
        /// </summary>
        /// <param name="language">The language name: "python" or "javascript".</param>
        /// <returns>Null on success (or nothing to do); otherwise why it failed, phrased for the model.</returns>
        string? InstallPackages(
            string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
            => "package installation is not available on this host";
    }
}
