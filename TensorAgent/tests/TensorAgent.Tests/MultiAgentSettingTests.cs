// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime;
using TensorSharp.Server.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The Sub-agents switch.
///
/// <para>
/// Delegation used to be on in the app with no way to turn it off: the host passed
/// no multi-agent options, so the server's default applied, and nothing the phone
/// page sends could change it. Each sub-agent is another conversation on the loaded
/// model, so on a phone that is memory the user had no say over. The switch has to
/// default to what the app already did, survive a save, reach the options a host is
/// built with, and move the running host so the NEXT message is planned without
/// the coordination tools -- the same bar the skills switch is held to.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class MultiAgentSettingTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-agents-" + Guid.NewGuid().ToString("N"));

    // ApplySettings also applies the speculation setting, which is written to the
    // process environment; put back whatever was there.
    private readonly string? _savedEnabled = Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable);
    private readonly string? _savedType = Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable);
    private readonly string? _savedDraft = Environment.GetEnvironmentVariable(SpeculationPolicy.DraftModelVariable);

    private AgentPaths Paths => new(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
    {
        ExecutionMode = AgentExecutionMode.InProcess,
    };

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(SpeculationPolicy.EnabledVariable, _savedEnabled);
        Environment.SetEnvironmentVariable(SpeculationPolicy.TypeVariable, _savedType);
        Environment.SetEnvironmentVariable(SpeculationPolicy.DraftModelVariable, _savedDraft);
        CodeEnvironment.Reset();
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void OnByDefault_RoundTrips_AndAFileFromBeforeTheSwitchKeepsDelegationOn()
    {
        var store = new SettingsStore(Path.Combine(_root, "settings.json"));
        AppSettings defaults = store.Load();
        Assert.True(defaults.MultiAgentEnabled);

        defaults.MultiAgentEnabled = false;
        store.Save(defaults);
        Assert.False(store.Load().MultiAgentEnabled);
        using (JsonDocument saved = JsonDocument.Parse(File.ReadAllText(store.Path)))
            Assert.False(saved.RootElement.GetProperty("multiAgentEnabled").GetBoolean());

        // Written by a build that had no switch: the key is absent, and the answer has
        // to be what that build did rather than a silent change of behaviour.
        File.WriteAllText(store.Path, "{ \"allowCodeExecution\": true, \"skillsEnabled\": true }");
        Assert.True(store.Load().MultiAgentEnabled);
    }

    [Fact]
    public void AHostStartedWithSubAgentsOffPlansNoCoordinationTools()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var store = new SettingsStore(paths.SettingsFile);
        AppSettings off = store.Load();
        off.MultiAgentEnabled = false;
        store.Save(off);

        using var host = new AgentAppHost(paths);

        Assert.False(host.Options.MultiAgent.Enabled);
        Assert.False(DeclaresDelegation(PlanNextMessage(host)));
    }

    [Fact]
    public void TurningSubAgentsOffReachesTheNextMessageWithoutARestart()
    {
        using var host = new AgentAppHost(Paths);
        MultiAgentOptions limits = host.Options.MultiAgent;
        Assert.True(limits.Enabled);
        Assert.True(DeclaresDelegation(PlanNextMessage(host)));

        AppSettings off = host.Settings.Load();
        off.MultiAgentEnabled = false;
        host.Settings.Save(off);
        host.ApplySettings(off);

        Assert.False(host.Options.MultiAgent.Enabled);
        Assert.False(DeclaresDelegation(PlanNextMessage(host)));
        // Only the switch moved: the bounds are still the ones the host was built with.
        Assert.Equal(limits.MaxConcurrentAgents, host.Options.MultiAgent.MaxConcurrentAgents);
        Assert.Equal(limits.MaxAgents, host.Options.MultiAgent.MaxAgents);
        Assert.Equal(limits.AllowWorkerTools, host.Options.MultiAgent.AllowWorkerTools);

        // And back: a switch that only turns off is half a switch.
        AppSettings on = host.Settings.Load();
        on.MultiAgentEnabled = true;
        host.Settings.Save(on);
        host.ApplySettings(on);

        Assert.True(host.Options.MultiAgent.Enabled);
        Assert.True(DeclaresDelegation(PlanNextMessage(host)));
    }

    /// <summary>What the next chat message would be planned with: the host's own
    /// registry, options and runner, on a family that renders tool declarations.</summary>
    private static SkillRequestPlan? PlanNextMessage(AgentAppHost host)
    {
        SkillRequestPlan? plan = SkillRequestPlan.Create(
            host.Skills,
            Array.Empty<string>(),
            discovery: false,
            clientTools: new List<ToolFunction>(),
            architecture: "qwen35",
            contextTokens: 32768,
            options: host.Options,
            out IReadOnlyList<string> unknown,
            codeRunner: host.CodeRunner);
        Assert.Empty(unknown);
        return plan;
    }

    private static bool DeclaresDelegation(SkillRequestPlan? plan) =>
        plan is not null && plan.Tools.Any(tool => MultiAgentTools.IsTool(tool.Name));
}
