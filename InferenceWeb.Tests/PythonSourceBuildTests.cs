// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Where a pip install may build a source distribution: only where the install's sandbox
/// confines writes and holds the network to the package index (Seatbelt with the egress
/// proxy). Everywhere else installs stay prebuilt wheels only.
/// </summary>
public sealed class PythonSourceBuildTests
{
    [Fact]
    public void SourceBuildsAreAllowedOnlyWhereTheInstallSandboxHoldsTheNetworkToTheIndex()
    {
        Assert.Contains("--only-binary=:all:", CodeEnvironment.PythonInstall("python3", "/env", new[] { "x" }).Arguments);
        CodeInstallPlan sourcePlan = CodeEnvironment.PythonInstall("python3", "/env", new[] { "x" }, allowSourceBuilds: true);
        Assert.Contains("--prefer-binary", sourcePlan.Arguments);
        Assert.DoesNotContain("--only-binary=:all:", sourcePlan.Arguments);

        var options = new CodeExecOptions { Enabled = true, AllowInstall = true, AllowNetwork = false };
        var honest = new SkillSandboxCapabilities(true, true, true, true);
        Assert.True(new PackageInstaller(new FakeShellBackend { Sandbox = new NamedSandbox("sandbox-exec", honest) }, options).AllowsSourceBuilds);
        Assert.False(new PackageInstaller(new FakeShellBackend { Sandbox = new NamedSandbox("bubblewrap", honest) }, options).AllowsSourceBuilds);
        Assert.False(new PackageInstaller(new FakeShellBackend { Sandbox = new NamedSandbox("sandbox-exec",
            new SkillSandboxCapabilities(false, true, true, true)) }, options).AllowsSourceBuilds);
        Assert.False(new PackageInstaller(new FakeShellBackend(), options).AllowsSourceBuilds);
    }

    private sealed class NamedSandbox(string name, SkillSandboxCapabilities capabilities) : ISkillSandbox
    {
        public string Name => name;
        public bool IsAvailable => true;
        public SkillSandboxCapabilities Capabilities => capabilities;
        public string Describe() => name;
        public bool TryWrap(SkillSandboxRequest request, out string fileName, out IReadOnlyList<string> arguments,
            out IDisposable cleanup, out string error) => throw new NotSupportedException();
    }
}
