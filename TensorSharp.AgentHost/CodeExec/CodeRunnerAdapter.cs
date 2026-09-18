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
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Binds <see cref="ShellRunner"/> to the tool interface every host talks to.
    ///
    /// <para>
    /// The seam exists so that the skills namespace — and through it the servers and the
    /// CLI — knows only <see cref="ICodeRunner"/>, and code execution knows about skills
    /// rather than the other way round. Without it the two would reference each other and
    /// a host that only wanted skills would drag the whole execution stack in with them.
    /// </para>
    /// </summary>
    public sealed class CodeRunnerAdapter : ICodeRunner
    {
        private readonly ShellRunner _runner;
        private readonly CodeExecOptions _options;
        private readonly Action<CodeExecResult>? _onCompleted;
        private readonly string? _packageInstallInstructions;
        private readonly string? _networkExecutionInstructions;
        private readonly string? _providedPackagesInstructions;
        private readonly string? _executionInstructions;
        private readonly Func<bool>? _networkInstructionsAvailable;
        private readonly Func<IReadOnlyList<string>>? _networkHosts;

        /// <param name="runner">The engine.</param>
        /// <param name="options">The host's terms, for the declaration.</param>
        /// <param name="onCompleted">
        /// Observer for each finished call, so a host can record what a command produced
        /// without parsing the model's prose about it.
        /// </param>
        /// <param name="packageInstallInstructions">
        /// Stable host-specific package capabilities for the shell declaration, or null
        /// for the desktop pip/npm description.
        /// </param>
        /// <param name="networkExecutionInstructions">
        /// Stable host-specific guidance for efficient use of this host's network,
        /// separate from package-manager capabilities. It is exposed only while the
        /// network switch is on.
        /// </param>
        /// <param name="networkInstructionsAvailable">
        /// Optional live check that the host-specific source is admitted by the current
        /// network allow-list. Null means the guidance has no narrower host requirement.
        /// </param>
        /// <param name="networkHosts">
        /// Optional live view of the host's outbound allow-list, for an accurate
        /// declaration. Null or empty means unrestricted when networking is enabled.
        /// </param>
        public CodeRunnerAdapter(
            ShellRunner runner,
            CodeExecOptions? options = null,
            Action<CodeExecResult>? onCompleted = null,
            string? packageInstallInstructions = null,
            string? networkExecutionInstructions = null,
            Func<bool>? networkInstructionsAvailable = null,
            Func<IReadOnlyList<string>>? networkHosts = null,
            string? providedPackagesInstructions = null,
            string? executionInstructions = null)
        {
            _runner = runner ?? throw new ArgumentNullException(nameof(runner));
            _options = options ?? runner.Options;
            _onCompleted = onCompleted;
            _packageInstallInstructions = packageInstallInstructions;
            _networkExecutionInstructions = networkExecutionInstructions;
            _providedPackagesInstructions = providedPackagesInstructions;
            _executionInstructions = executionInstructions;
            _networkInstructionsAvailable = networkInstructionsAvailable;
            _networkHosts = networkHosts;
        }

        /// <inheritdoc/>
        public bool CanRun => _runner.CanRun;

        /// <inheritdoc/>
        public IShellBackend? Backend => _runner.Backend;

        /// <inheritdoc/>
        public string? UnavailableReason => _runner.UnavailableReason;

        /// <inheritdoc/>
        [return: System.Diagnostics.CodeAnalysis.NotNullIfNotNull(nameof(requested))]
        public SamplingConfig? ForCodingTurn(SamplingConfig? requested) =>
            // Two different operations, and only one of them is opt-in.
            //
            // REMOVING the repetition penalty is on by default: it is an Ollama
            // chat-compatibility default that neither reference has any analogue of —
            // Codex sends no penalty at all — and on code it penalises the indentation,
            // the `return` and the closing delimiters against each other. Taking it off
            // for coding turns moves toward both references, not away.
            //
            // SETTING a temperature is opt-in, because that would ADD something neither
            // reference sets: Codex leaves it None and omits it from the wire, and Claude
            // Code exposes no sampling setting at all.
            requested == null || !CanRun ? requested : requested.ForCodingTurn(_options.Temperature);

        /// <inheritdoc/>
        public ToolFunction Declare()
        {
            // BY NAME, never by index. This returned declarations[0] until the file tools
            // were added in front of the shell, at which point "the tool" silently became
            // read_file — the kind of change that compiles, passes a set-equality test,
            // and is only visible in what the model was told. The same hazard is fixed in
            // SkillRequestPlan, which patched declarations[0]'s description to mention the
            // conversation's attachments.
            IReadOnlyList<ToolFunction> declarations = DeclareTools();
            foreach (ToolFunction declaration in declarations)
            {
                if (string.Equals(declaration.Name, ShellTools.ShellToolName, StringComparison.Ordinal))
                    return declaration;
            }
            return declarations.Count > 0 ? declarations[0] : new ToolFunction();
        }

        /// <inheritdoc/>
        public IReadOnlyList<ToolFunction> DeclareTools() => DeclareTools(persists: true);

        /// <inheritdoc/>
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists)
        {
            if (_runner.Shell is not { } shell)
            {
                // Declaring a tool this host cannot answer is strictly worse than staying
                // quiet: the model emits the call, nothing services it, and the raw tool
                // markup reaches the user as the answer.
                return Array.Empty<ToolFunction>();
            }

            // The file tools need a workspace: they read and write files that have to
            // outlive the call, and the read ledger that authorises an edit lives on the
            // request/session. A caller that supplies no workspace gets a fresh empty
            // directory per call, so
            // there is nothing there to read and nothing that would survive being
            // written — declaring them would offer capability the host cannot honour.
            if (!persists)
            {
                return new[]
                {
                    ShellTools.DeclareShell(
                        _options, shell, _runner.KeepsArtifacts, persists: false, fileTools: false,
                        networkConfinementGuaranteed: _runner.NetworkConfinementGuaranteed,
                        packageInstallInstructions: PackageInstallInstructions(),
                        networkExecutionInstructions: NetworkExecutionInstructions(),
                        networkHosts: _networkHosts?.Invoke(),
                        providedPackagesInstructions: _providedPackagesInstructions,
                        executionInstructions: _executionInstructions),
                };
            }

            // Keep reading and patching before the shell: apply_patch handles every
            // modification, from a single line in one file to an atomic multi-file change.
            // Legacy edit calls remain dispatchable but are not advertised to the model.
            return new[]
            {
                ShellTools.DeclareRead(),
                ShellTools.DeclarePatch(),
                ShellTools.DeclareWrite(),
                ShellTools.DeclareShell(
                    _options, shell, _runner.KeepsArtifacts, persists, fileTools: true,
                    networkConfinementGuaranteed: _runner.NetworkConfinementGuaranteed,
                    packageInstallInstructions: PackageInstallInstructions(),
                    networkExecutionInstructions: NetworkExecutionInstructions(),
                    networkHosts: _networkHosts?.Invoke(),
                        providedPackagesInstructions: _providedPackagesInstructions,
                        executionInstructions: _executionInstructions),
            };
        }

        /// <inheritdoc/>
        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            if (!_runner.CanRun)
                return SkillToolResult.Failure(_runner.UnavailableReason ?? "code execution is unavailable");

            // One session is one shell and one package tree. A replacement turn may
            // arrive before the cancelled turn's synchronous tool has returned, so all
            // access is serialized through the workspace rather than allowing a run to
            // observe a half-extracted install or half-written source file.
            using IDisposable? execution = workspace?.EnterExecution();

            // Every built-in file/code tool sees the same staged attachments. This used
            // to happen only immediately before a shell command, so a model following
            // the cheaper path advertised in the prompt -- read_file("responses.csv")
            // as its first action -- was told the file existed and then got "not found".
            // Stage before dispatch so read_file, edits, patches and shell all start from
            // the identical workspace.
            if (workspace != null)
                CodeInputFileStager.Stage(inputFiles, workspace);

            // Patching is a workspace operation, not a run: it returns without entering
            // the execution pipeline at all, and nothing is launched.
            if (ShellTools.IsPatchTool(call?.Name))
            {
                if (!ShellTools.TryReadPatch(call!, out string patch, out string? patchError))
                    return SkillToolResult.Failure(patchError!);
                return Finish(_runner.ApplyPatch(patch, workspace));
            }

            // The file tools, likewise: nothing is launched, no sandbox is entered, and
            // the host places the bytes. Dispatched through the shared resolver so an
            // invented spelling costs nothing — a round spent on `str_replace` instead of
            // `edit_file` teaches the model nothing and fixes nothing.
            switch (ShellTools.ResolveFileTool(call?.Name))
            {
                case SkillToolNames.ReadFile:
                    if (!ShellTools.TryReadRead(call!, out ShellTools.ReadRequest read, out string? readError))
                        return SkillToolResult.Failure(readError!);
                    return Finish(_runner.ReadFile(read, workspace));

                case SkillToolNames.EditFile:
                    if (!ShellTools.TryReadEdit(call!, out ShellTools.EditRequest edit, out string? editError))
                        return SkillToolResult.Failure(editError!);
                    return Finish(_runner.EditFile(edit, workspace));

                case SkillToolNames.WriteFile:
                    if (!ShellTools.TryReadWrite(call!, out ShellTools.WriteRequest write, out string? writeError))
                        return SkillToolResult.Failure(writeError!);
                    return Finish(_runner.WriteFile(write, workspace));
            }

            if (!ShellTools.TryReadShell(call!, out ShellRequest request, out string? error))
                return SkillToolResult.Failure(error!);

            CodeExecResult result = _runner.Run(
                request with { ReadablePaths = skillDirectories ?? Array.Empty<string>() },
                workspace,
                onOutput);

            return Finish(result);
        }

        /// <summary>
        /// Turn an engine result into a tool result.
        ///
        /// <para>
        /// A failure carries the FULL output with no "Error:" prefix, because the output
        /// is the useful part — the model is about to fix its own command with it. The
        /// produced files ride along structurally as well as in the text, so a host's UI
        /// can offer the downloads itself whatever the model's answer ends up saying
        /// about them.
        /// </para>
        /// </summary>
        private SkillToolResult Finish(CodeExecResult result)
        {
            _onCompleted?.Invoke(result);

            var files = new List<SkillProducedFile>(result.Artifacts.Count);
            foreach (CodeArtifact artifact in result.Artifacts)
                files.Add(new SkillProducedFile(artifact.Path, artifact.Bytes, artifact.Pointer));

            return new SkillToolResult(result.Ok, result.Content, null, null) { Files = files };
        }

        /// <inheritdoc/>
        public bool CanInstallPackages => _options.AllowInstall && _runner.Installer.CanInstall;

        private string? PackageInstallInstructions()
        {
            if (_packageInstallInstructions == null || _runner.Installer.CanInstall)
                return _packageInstallInstructions;

            string reason = _runner.Installer.UnavailableReason?.Trim().TrimEnd('.')
                ?? "the configured installer is unavailable";
            return $"Python package installation is currently unavailable: {reason}. "
                 + "Use the standard library and packages already present; do not retry the install.";
        }

        private string? NetworkExecutionInstructions() =>
            _networkInstructionsAvailable?.Invoke() == false
                ? null
                : _networkExecutionInstructions;

        /// <inheritdoc/>
        public bool CanInstallPackagesFor(string language)
        {
            CodeLanguage parsed = CodeExecOptions.ParseLanguage(language);
            return parsed != CodeLanguage.Unknown
                && _options.AllowInstall
                && _runner.Installer.CanInstallLanguage(parsed);
        }

        /// <inheritdoc/>
        public string? InstallPackages(
            string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
        {
            using IDisposable execution = workspace.EnterExecution();
            return _runner.Installer.Install(
                workspace, CodeExecOptions.ParseLanguage(language), packages, onOutput);
        }
    }
}
