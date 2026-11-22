// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// Every skill this app bundles has to appear in the catalog the model is given.
///
/// <para>
/// It did not. The catalog is filled in ordinal id order under a 1,024-token budget,
/// and the 13 bundled descriptions come to roughly 1,450 tokens — so six were dropped,
/// and which six was decided by the alphabet. The casualties included <c>documents</c>
/// and <c>research</c>, the two that TensorAgentSkillRouter itself routes to, while two
/// ~990-character entries at the head of the alphabet took half the budget between them
/// (both were upstream Claude-product skills and have since been unbundled). The
/// shortening below is what makes the outcome independent of the alphabet.
/// </para>
/// <para>
/// It is not a theoretical loss. Asked to look something up, a model on this build
/// reasoned that it had "skills for learning, art, design, and documentation" — the
/// seven that survived — found nothing that fetches a page, and refused the request.
/// The router's own comment records the other half of it: "Qwen overlooked the
/// documents catalog entry". There was no documents catalog entry.
/// </para>
/// </summary>
public sealed class SkillCatalogVisibilityTests
{
    /// <summary>The budget a phone actually gets: Clamp(context * 2%, 1024, 10000).</summary>
    private const int PhoneContextTokens = 32_768;

    private static string SkillsDirectory()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null
            && !Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
        {
            directory = directory.Parent;
        }
        Assert.NotNull(directory);
        return Path.Combine(directory!.FullName, "TensorAgent", "skills");
    }

    private static SkillRegistry BundledSkills()
    {
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { SkillsDirectory() } });
    }

    [Fact]
    public void EveryBundledSkillReachesTheModelsCatalog()
    {
        SkillRegistry registry = BundledSkills();

        // Derived from disk, not a magic number: a count pinned here breaks every time a
        // skill is added or removed, which is a maintenance tax rather than a regression.
        int onDisk = Directory
            .EnumerateDirectories(SkillsDirectory())
            .Count(directory => File.Exists(Path.Combine(directory, "SKILL.md")));
        Assert.True(onDisk >= 5, $"only {onDisk} skill folders on disk; the bundle is not being read");
        Assert.Equal(onDisk, registry.Skills.Count);

        SkillPlan plan = SkillPrompt.Plan(
            Array.Empty<Skill>(),
            registry.Skills,
            new SkillPromptOptions { ContextTokens = PhoneContextTokens, ToolsAvailable = true });

        Assert.Equal(registry.Skills.Count, plan.Catalog.Count);
        Assert.Equal(0, plan.OmittedFromCatalog);

        // The two the app's own router depends on, named rather than merely counted:
        // they are the ones the alphabet dropped, and a count could pass while they are
        // the pair still missing.
        foreach (string required in new[] { "documents", "find-skills", "research" })
        {
            Assert.Contains(plan.Catalog, skill =>
                string.Equals(skill.Id, required, StringComparison.OrdinalIgnoreCase));
            Assert.Contains("- " + required + ":", plan.Instructions, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Each line must still say enough to choose on, whether it fits unchanged or
    /// needs shortening to keep the complete catalog visible.
    /// </summary>
    [Fact]
    public void TheCatalogFitsItsBudgetAndEveryLineStillDescribesItsSkill()
    {
        SkillRegistry registry = BundledSkills();
        SkillPlan plan = SkillPrompt.Plan(
            Array.Empty<Skill>(),
            registry.Skills,
            new SkillPromptOptions { ContextTokens = PhoneContextTokens, ToolsAvailable = true });

        // The complete catalog must remain visible. Descriptions may fit unchanged
        // after authors make them more concise; requiring shortening would penalize
        // that improvement. The planner may trim metadata but must not expand it.
        Assert.Equal(0, plan.OmittedFromCatalog);
        int rendered = plan.Instructions
            .Split('\n')
            .Where(l => registry.Skills.Any(s => l.StartsWith("- " + s.Id + ":", StringComparison.Ordinal)))
            .Sum(l => l.Length);
        int raw = registry.Skills.Sum(s => s.Description.Length + s.Id.Length + 4);
        Assert.True(rendered <= raw,
            $"the rendered descriptions grew from {raw} to {rendered} characters");

        foreach (Skill skill in registry.Skills)
        {
            string line = plan.Instructions
                .Split('\n')
                .Single(l => l.StartsWith("- " + skill.Id + ":", StringComparison.Ordinal));
            string described = line[(skill.Id.Length + 4)..].Trim();
            Assert.True(described.Length >= 60,
                $"'{skill.Id}' is described in {described.Length} characters, too few to choose on: {described}");
        }
    }

    /// <summary>
    /// A catalog that already fits is left exactly as it was — the shortening exists to
    /// prevent eviction, not to trim prompts that were never in trouble.
    /// </summary>
    [Fact]
    public void ACatalogThatAlreadyFitsIsNotShortened()
    {
        SkillRegistry registry = BundledSkills();
        Skill[] few = registry.Skills.Take(3).ToArray();

        SkillPlan plan = SkillPrompt.Plan(
            Array.Empty<Skill>(), few,
            new SkillPromptOptions { ContextTokens = PhoneContextTokens, ToolsAvailable = true });

        Assert.Equal(3, plan.Catalog.Count);
        foreach (Skill skill in few)
        {
            Assert.Contains(
                "- " + skill.Id + ": " + skill.Description.Replace("\n", " "),
                plan.Instructions.Replace("\n- ", "- ").Replace("\n", " ").Replace("- ", "\n- "),
                StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// The old market-data description put its applicability condition after 308
    /// characters. Both phone budgets shortened it to "Use...", retaining examples
    /// but losing the condition that makes them relevant. Check what the request
    /// actually gives the model, not just the unabridged manifest on disk.
    /// </summary>
    [Theory]
    [InlineData(8_192)]
    [InlineData(32_768)]
    public void MarketDataScopeSurvivesThePhoneCatalogBudget(int contextTokens)
    {
        SkillRegistry registry = BundledSkills();
        SkillRequestPlan plan = DiscoveryPlan(registry, contextTokens);
        Skill market = Assert.Single(registry.Skills, skill => skill.Id == "market-data");
        string line = Assert.Single(plan.Prompt.Instructions.Split('\n'),
            entry => entry.StartsWith("- market-data:", StringComparison.Ordinal));

        Assert.Equal("- market-data: " + market.Description, line);
        Assert.Contains("Use only for current stock/share prices", line, StringComparison.Ordinal);
        Assert.Contains("ticker quotes", line, StringComparison.Ordinal);
        Assert.Contains("financial market movers", line, StringComparison.Ordinal);
        // Keep unrelated domains out of routing metadata: even a negative example
        // can make a small model associate that topic with the skill.
        Assert.DoesNotContain("weather", line, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("forecast", line, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Network switch", line, StringComparison.Ordinal);

        Assert.Empty(plan.Selected);
        Assert.Empty(plan.Prompt.Inlined);
        Assert.DoesNotContain(market.Manifest.Body, plan.Prompt.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("market_movers.py", plan.Prompt.Instructions, StringComparison.Ordinal);
        Assert.Contains(plan.ToolContext.Reachable, skill => skill.Id == market.Id);
    }

    /// <summary>
    /// Current information lookups need a discoverable general research capability.
    /// Its old description spent the budget naming indexes and lost the clause
    /// explaining when to use it, leaving market-data as the apparent lookup option.
    /// </summary>
    [Theory]
    [InlineData(8_192)]
    [InlineData(32_768)]
    public void ResearchLookupScopeSurvivesThePhoneCatalogBudget(int contextTokens)
    {
        SkillRegistry registry = BundledSkills();
        SkillRequestPlan plan = DiscoveryPlan(registry, contextTokens);
        Skill research = Assert.Single(registry.Skills, skill => skill.Id == "research");
        string line = Assert.Single(plan.Prompt.Instructions.Split('\n'),
            entry => entry.StartsWith("- research:", StringComparison.Ordinal));

        Assert.Equal("- research: " + research.Description, line);
        Assert.Contains("Use for web searches and current information lookups", line, StringComparison.Ordinal);
        Assert.Contains("finding sources", line, StringComparison.Ordinal);
        Assert.Contains("fact-checking", line, StringComparison.Ordinal);
        Assert.Contains("without being given any URLs", line, StringComparison.Ordinal);
        Assert.Contains("Network switch", line, StringComparison.Ordinal);
        Assert.Empty(plan.Prompt.Inlined);
        Assert.DoesNotContain(research.Manifest.Body, plan.Prompt.Instructions, StringComparison.Ordinal);
        Assert.Contains(plan.ToolContext.Reachable, skill => skill.Id == research.Id);
    }

    /// <summary>
    /// Ordinary turns remain discovery requests, with no forced first skill read.
    /// This checks the host's routing and disclosure contract; actual LLM selection
    /// requires separate live-model validation and is not simulated here.
    /// </summary>
    [Theory]
    [InlineData("明天天气怎么样？")]
    [InlineData("北京明天天气怎么样？")]
    [InlineData("What will the weather be like tomorrow?")]
    [InlineData("你好")]
    [InlineData("Explain how a rainbow forms.")]
    public void UnrelatedRequestsDoNotActivateAMarketWorkflow(string prompt)
    {
        SkillRegistry registry = BundledSkills();
        var messages = new List<ChatMessage> { new() { Role = "user", Content = prompt } };

        Assert.Null(TensorAgentSkillRouter.Route(messages, requestedSkills: null, registry));
        SkillRequestPlan plan = DiscoveryPlan(registry, PhoneContextTokens);
        List<ChatMessage> injected = plan.Apply(messages);

        Assert.Empty(plan.Selected);
        Assert.Empty(plan.Prompt.Inlined);
        Assert.Null(plan.CompletionRequirement);
        Assert.Equal(prompt, injected[^1].Content);
        Assert.DoesNotContain(injected, message => message.Role == "tool"
            || message.ToolCalls is { Count: > 0 });
    }

    [Fact]
    public void TheMarketSkillRemainsReadableAfterDiscovery()
    {
        SkillRequestPlan plan = DiscoveryPlan(BundledSkills(), PhoneContextTokens);

        // Exercise the real read handler to catch a "fix" that hides or disables the
        // skill instead of correcting the metadata the model uses to choose it.
        Assert.Contains(plan.Tools, tool => tool.Name == SkillTools.ReadToolName);
        SkillToolResult result = SkillTools.Execute(new ToolCall
        {
            Name = SkillTools.ReadToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "market-data",
                ["path"] = "SKILL.md",
            },
        }, plan.ToolContext);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("scripts/market_movers.py --quote", result.Content, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(8_192)]
    [InlineData(32_768)]
    public void FindSkillsRemainsDiscoverableWithoutInliningItsInstructions(int contextTokens)
    {
        SkillRegistry registry = BundledSkills();
        SkillRequestPlan plan = DiscoveryPlan(registry, contextTokens);
        Skill skill = Assert.Single(registry.Skills, skill => skill.Id == "find-skills");
        string line = Assert.Single(plan.Prompt.Instructions.Split('\n'),
            entry => entry.StartsWith("- find-skills:", StringComparison.Ordinal));

        Assert.Equal("- find-skills: " + skill.Description, line);
        Assert.Contains("Find, install and use a hub skill", line, StringComparison.Ordinal);
        Assert.Contains("Network switch", line, StringComparison.Ordinal);
        Assert.Empty(plan.Prompt.Inlined);
        Assert.DoesNotContain(skill.Manifest.Body, plan.Prompt.Instructions, StringComparison.Ordinal);
        Assert.Contains(plan.ToolContext.Reachable, item => item.Id == skill.Id);
    }

    private static SkillRequestPlan DiscoveryPlan(SkillRegistry registry, int contextTokens)
    {
        var options = new ServerHostingOptions(
            startupModelPath: string.Empty,
            startupMmProjPath: string.Empty,
            defaultBackend: "ggml_cpu",
            supportedBackends: Array.Empty<BackendOption>(),
            defaultMaxTokens: 512,
            maxTokensPinned: false,
            defaultVideoFrames: 0,
            defaultVideoFps: 0,
            defaultVideoWidth: 0,
            defaultVideoHeight: 0,
            defaultVideoSteps: 0,
            defaultVideoMode: string.Empty,
            uploadDirectory: string.Empty,
            logDirectory: string.Empty,
            fileLoggingEnabled: false,
            samplingDefaults: new SamplingDefaults(new SamplingConfig()),
            skillsEnabled: true,
            skillsDiscovery: true);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry,
            requestedSkills: null!,
            discovery: null,
            clientTools: new List<ToolFunction>(),
            architecture: "qwen35",
            contextTokens: contextTokens,
            options: options,
            out IReadOnlyList<string> unknown);

        Assert.Empty(unknown);
        Assert.NotNull(plan);
        return plan;
    }
}
