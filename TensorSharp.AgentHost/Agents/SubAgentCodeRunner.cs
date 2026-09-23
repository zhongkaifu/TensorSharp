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
using System.Threading;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents
{
    /// <summary>
    /// A sub-agent's view of the turn's code runner: the same runner, except that a
    /// command whose agent was stopped while it waited for the workspace never starts.
    ///
    /// <para>
    /// Every agent's code tools take the workspace's execution lock one at a time, and
    /// that lock cannot be cancelled. The loop checks the agent's token before each call,
    /// but a call queued behind another agent's <c>pip install</c> passes that check, then
    /// waits — and when the turn is stopped meanwhile, it would acquire the lock and run
    /// after the turn had ended. So the lock is taken HERE first (it is re-entrant, and the
    /// runner takes it again inside) and the token is checked with it held.
    /// </para>
    /// </summary>
    internal sealed class SubAgentCodeRunner : ICodeRunner
    {
        internal const string StoppedMessage =
            "this agent was stopped before the command could start, so it did not run.";

        private readonly ICodeRunner _inner;
        private readonly CancellationToken _stop;

        public SubAgentCodeRunner(ICodeRunner inner, CancellationToken stop)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _stop = stop;
        }

        public bool CanRun => _inner.CanRun;
        public CodeExec.IShellBackend? Backend => _inner.Backend;
        public string? UnavailableReason => _inner.UnavailableReason;
        public SamplingConfig? ForCodingTurn(SamplingConfig? requested) => _inner.ForCodingTurn(requested);
        public ToolFunction Declare() => _inner.Declare();
        public IReadOnlyList<ToolFunction> DeclareTools() => _inner.DeclareTools();
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists) => _inner.DeclareTools(persists);
        public bool CanInstallPackages => _inner.CanInstallPackages;
        public bool CanInstallPackagesFor(string language) => _inner.CanInstallPackagesFor(language);

        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            using IDisposable? execution = workspace?.EnterExecution();
            if (_stop.IsCancellationRequested)
                return SkillToolResult.Failure(StoppedMessage);
            return _inner.Execute(call, inputFiles, onOutput, workspace, skillDirectories);
        }

        public string? InstallPackages(
            string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
        {
            using IDisposable execution = workspace.EnterExecution();
            if (_stop.IsCancellationRequested)
                return StoppedMessage;
            return _inner.InstallPackages(language, packages, workspace, onOutput);
        }
    }
}
