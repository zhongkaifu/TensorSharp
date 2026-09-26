using System.Text;
using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// The deterministic front door for research followed by a PowerPoint deliverable.
/// These are deliberately model-free: whether a prompt is routed must not depend on
/// sampling. They are also deliberately subject-free: the route once fired only for
/// one reported comparison (Apple M6 against M5) and searched a hard-coded English
/// query, so the same request about anything else fell back to discovery. That
/// comparison is now one example among several, and every expected query below is
/// made only of words the user wrote.
/// </summary>
public sealed class CompoundSkillIntentRouterTests : IDisposable
{
    private const string NeutralPrompt = "搜索量子计算的最新进展，然后生成幻灯片";
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-compound-router-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    [Fact]
    public void ACompoundRequestSelectsBothSkillsAndTheSubjectFreeEvidenceContract()
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(
            new[] { TensorAgentSkillRouter.ResearchSkill, TensorAgentSkillRouter.DocumentsSkill },
            route.Skills);
        WebUiArtifactRequirement artifact = Assert.IsType<WebUiArtifactRequirement>(route.ArtifactRequirement);
        Assert.Equal(".pptx", artifact.Extension);
        Assert.Equal(4, artifact.MinimumSlides);
        // The deck is checked for structure and for citing this turn's research, never
        // for naming particular terms: those would only fit one subject.
        Assert.True(artifact.RequiredVisibleTerms is null or { Count: 0 },
            "the route demands subject-specific terms: " + string.Join(", ", artifact.RequiredVisibleTerms ?? Array.Empty<string>()));
        Assert.True(artifact.RequireVisibleHttpUrl);
        Assert.Equal("notes.md", artifact.CitationEvidencePath);
        Assert.True(route.RequiresNetwork);
        Assert.Collection(
            artifact.RequiredRuns,
            run =>
            {
                Assert.Equal("research", run.SkillId);
                Assert.Equal("scripts/research.py", run.ResourcePath);
                Assert.False(run.ProducesArtifact);
                Assert.Equal(
                    new[] { "搜索量子计算的最新进展", "--pages", "3", "--out", "notes.md" },
                    run.DefaultArguments);
                Assert.True(run.EnforceArguments);
                Assert.Null(run.RequiredInputPath);
            },
            run =>
            {
                Assert.Equal("documents", run.SkillId);
                Assert.Equal("scripts/make_pptx.py", run.ResourcePath);
                Assert.True(run.ProducesArtifact);
                Assert.Equal(
                    new[] { "--spec", "pptx_spec.json", "--out", "report.pptx" },
                    run.DefaultArguments);
                Assert.True(run.EnforceArguments);
                Assert.Equal("pptx_spec.json", run.RequiredInputPath);
            });
        Assert.Contains("skills_read(skill=\"research\"", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("skills_read(skill=\"documents\"", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("source URLs and dates", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("exactly three slide entries", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("below 6,000 characters", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("without reading, copying, or rewriting that script", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("real .pptx", route.Instructions, StringComparison.Ordinal);
        Assert.Contains("shared workspace", route.Instructions, StringComparison.Ordinal);
    }

    /// <summary>
    /// Any subject, in English or Chinese, routes, and the research query is the
    /// user's research clause: the trailing connective ("then", "并", "然后") and the
    /// deck request are cut, nothing is added. A word the user wrote after the format
    /// name ("report", "报告") stays, because the route cannot tell it from a subject.
    /// </summary>
    [Theory]
    [InlineData(
        "搜索apple M6的信息，并对比M5芯片，然后生成pptx报告",
        "搜索apple M6的信息，并对比M5芯片 报告")]
    [InlineData(
        "Search for current Apple M6 information, compare it with M5, and create a PowerPoint report.",
        "Search for current Apple M6 information, compare it with M5 report")]
    [InlineData(
        "Search current Rust releases and create a PowerPoint report.",
        "Search current Rust releases report")]
    [InlineData(
        "Research the history of the Roman aqueducts and make a slide deck.",
        "Research the history of the Roman aqueducts")]
    [InlineData(
        "调研一下2026年电动汽车电池技术，并制作演示文稿",
        "调研一下2026年电动汽车电池技术")]
    [InlineData(
        "搜索量子计算的最新进展。生成pptx",
        "搜索量子计算的最新进展")]
    public void ACompoundRequestOnAnySubjectSearchesTheUsersOwnWords(string prompt, string expectedQuery)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(
            new[] { TensorAgentSkillRouter.ResearchSkill, TensorAgentSkillRouter.DocumentsSkill },
            route.Skills);
        Assert.Equal(expectedQuery, ResearchQueryOf(route));
    }

    /// <summary>
    /// The route enforces its research arguments, so a model cannot repair a query
    /// that lost the subject. The subject survives wherever the request puts it:
    /// after the deck request, inside it (the usual Chinese order), or everywhere
    /// but the deck request when that comes first.
    /// </summary>
    [Theory]
    [InlineData(
        "Search the web and create a PowerPoint about the Roman aqueducts",
        "Search the web about the Roman aqueducts")]
    [InlineData(
        "搜索最新资料，然后生成一份关于量子计算的演示文稿",
        "搜索最新资料 关于量子计算的")]
    [InlineData(
        "Search the web and make me a Rust vs Go presentation",
        "Search the web Rust vs Go")]
    [InlineData(
        "Search the web and create a US election presentation",
        "Search the web US election")]
    [InlineData(
        "Create a PowerPoint deck on the history of Rome after you search the web for sources.",
        "deck on the history of Rome after you search the web for sources")]
    public void TheSubjectSurvivesWhereverTheRequestNamesIt(string prompt, string expectedQuery)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(expectedQuery, ResearchQueryOf(route));
    }

    /// <summary>
    /// Punctuation marks a new request whether or not a space follows it, as does a
    /// line break. Before, only "X,create" counted, so the ordinary English "X, create
    /// a presentation" never routed.
    /// </summary>
    [Theory]
    [InlineData(
        "Look up recent research on sleep and memory, create a presentation.",
        "Look up recent research on sleep and memory")]
    [InlineData(
        "Search the Rust release notes\n  create a pptx",
        "Search the Rust release notes")]
    public void APunctuatedRequestCountsEvenWhenASpaceFollowsThePunctuation(string prompt, string expectedQuery)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(expectedQuery, ResearchQueryOf(route));
    }

    /// <summary>
    /// Connectives and creation verbs are whole words: "Poland" does not end in the
    /// connective "and", and "remake" or "recreate" are not requests to make a deck.
    /// </summary>
    [Fact]
    public void ConnectivesAndCreationVerbsMustBeWholeWords()
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn("Search the history of Poland, create a pptx"), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal("Search the history of Poland", ResearchQueryOf(route));
        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn("Look up why teams remake their slides every quarter"), requestedSkills: null, registry));
        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn("Research how people recreate PowerPoint slides"), requestedSkills: null, registry));
    }

    /// <summary>
    /// research.py parses its question with argparse, which reads a one-word argument
    /// that starts with '-' as an unknown option and fails the run.
    /// </summary>
    [Theory]
    [InlineData("- Search the Rust release notes and create a pptx", "Search the Rust release notes")]
    [InlineData("-搜索量子计算，然后生成pptx", "搜索量子计算")]
    public void TheQueryNeverStartsLikeACommandLineOption(string prompt, string expectedQuery)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(expectedQuery, ResearchQueryOf(route));
    }

    [Fact]
    public void ALongCompoundRequestProducesABoundedValidResearchArgument()
    {
        string prompt = "Search the history of the Roman aqueducts " + new string('x', 5_000)
            + " and create a PowerPoint report.";
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        string query = ResearchQueryOf(route);
        Assert.InRange(query.Length, 1, TensorAgentSkillRouter.MaxResearchQueryLength);
        // The artifact contract refuses a routed argument above 4,096 characters.
        Assert.InRange(query.Length, 1, 4096);
        Assert.StartsWith("Search the history of the Roman aqueducts xxx", query, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBoundNeverSplitsASurrogatePair()
    {
        // The emoji's high surrogate lands on the last character the bound keeps.
        string lead = "Search " + new string('x', TensorAgentSkillRouter.MaxResearchQueryLength - 8);
        string prompt = lead + "\U0001F600 and more detail, then create a PowerPoint report.";
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());
        Assert.True(char.IsHighSurrogate(prompt[TensorAgentSkillRouter.MaxResearchQueryLength - 1]));

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        string query = ResearchQueryOf(route);
        Assert.Equal(lead, query);
        Assert.Equal(query, Encoding.UTF8.GetString(Encoding.UTF8.GetBytes(query)));
    }

    [Theory]
    [InlineData("搜索 Apple M6 的信息并与 M5 对比")]
    [InlineData("Search the latest Rust release notes and summarize them")]
    [InlineData("根据我已经附上的资料生成 pptx 报告")]
    [InlineData("Create a PowerPoint deck about the Roman aqueducts")]
    [InlineData("Research how to repair a timber slide deck")]
    public void ARequestWithOnlyOneSideOfTheWorkflowIsNotBroadened(string prompt)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry));
    }

    /// <summary>
    /// Without a topic gate the cues are what keeps an ordinary request out of a forced,
    /// network-only workflow (which answers 503 while network access is off). A research
    /// word must be ASKED for, not merely present; a format name must be a word of its own;
    /// and it must be what the verb makes, not the subject of something else it makes.
    /// </summary>
    [Theory]
    [InlineData("Create a presentation about our research results")]
    [InlineData("Search the news and write a summary of recent landslides in Nepal")]
    [InlineData("Research quantum computing and create a representation of the qubit states")]
    [InlineData("Search the literature and write a summary of the clinical presentation of Lyme disease")]
    [InlineData("制作一份关于搜索引擎优化的演示文稿")]
    [InlineData("Make slides that explain how a web search engine ranks pages")]
    public void AResearchWordOrAFormatNameThatIsOnlyMentionedDoesNotRoute(string prompt)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        Assert.Null(TensorAgentSkillRouter.Route(UserTurn(prompt), requestedSkills: null, registry));
    }

    /// <summary>
    /// Only the connective run directly before the deck request is cut, and never across
    /// clause punctuation: a CJK connective needs no word boundary, so a looser cut took
    /// the 请 of 申请 and the 并 of 合并, and an English one took "next" out of the subject.
    /// </summary>
    [Theory]
    [InlineData("搜索美国签证申请，生成pptx", "搜索美国签证申请")]
    [InlineData("搜索2025年的企业合并，然后生成幻灯片", "搜索2025年的企业合并")]
    [InlineData("搜索2025年的企业合并然后生成幻灯片", "搜索2025年的企业合并")]
    [InlineData("Research what Apple will release next, and create a presentation", "Research what Apple will release next")]
    [InlineData("Research what Apple will release next and create a presentation", "Research what Apple will release next")]
    [InlineData("Search the Rust release notes and then create a pptx", "Search the Rust release notes")]
    public void TheSubjectKeepsWordsThatOnlyLookLikeConnectives(string prompt, string expectedQuery)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        Assert.Equal(expectedQuery, ResearchQueryOf(route));
    }

    /// <summary>
    /// research.py reads an argument that starts with http(s):// as a page to read, so a
    /// URL the user wrote travels as its own argument and never leads the question.
    /// </summary>
    [Theory]
    [InlineData(
        "https://example.com/post - search it and make slides",
        new[] { "search it", "https://example.com/post" })]
    [InlineData(
        "Research https://example.com/post and the pages it cites, then create a presentation.",
        new[] { "Research and the pages it cites", "https://example.com/post" })]
    public void AUrlTheUserWroteIsReadAsAPageNotSearchedAsText(string prompt, string[] expectedPositional)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry);

        Assert.NotNull(route);
        WebUiArtifactRequirement artifact = Assert.IsType<WebUiArtifactRequirement>(route.ArtifactRequirement);
        WebUiSkillRunRequirement research = Assert.Single(
            artifact.RequiredRuns, run => run.SkillId == TensorAgentSkillRouter.ResearchSkill);
        Assert.Equal(
            expectedPositional.Concat(new[] { "--pages", "3", "--out", "notes.md" }),
            research.DefaultArguments);
    }

    [Fact]
    public void CuesInDifferentTurnsDoNotBecomeOneCompoundRequest()
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Search the web for the latest Rust release notes." },
            new() { Role = "assistant", Content = "Here is what I found." },
            new() { Role = "user", Content = "Summarize the attached notes as a pptx." },
        };

        Assert.Null(TensorAgentSkillRouter.Route(messages, requestedSkills: null, registry));
    }

    [Theory]
    [InlineData("attachment")]
    [InlineData("text")]
    [InlineData("image")]
    [InlineData("audio")]
    public void AttachedContentCannotActivateTheAutomaticNetworkRoute(string kind)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());
        var message = new ChatMessage
        {
            Role = "user",
            Content = "summarize this attachment\n\n" + NeutralPrompt,
        };
        var paths = new List<string> { "uploaded." + kind };
        switch (kind)
        {
            case "attachment": message.AttachmentPaths = paths; break;
            case "text": message.TextFilePaths = paths; break;
            case "image": message.ImagePaths = paths; break;
            default: message.AudioPaths = paths; break;
        }

        Assert.Null(TensorAgentSkillRouter.Route(
            new[] { message }, requestedSkills: null, registry));
    }

    [Theory]
    [InlineData("Research why tools generate PowerPoint presentations")]
    [InlineData("Look up articles that explain how software creates PPTX files")]
    public void AnExplanatoryMentionOfDeckGenerationDoesNotForceADeliverable(string prompt)
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn(prompt), requestedSkills: null, registry));
    }

    [Fact]
    public void AnExplicitSkillSelectionIsNeverBroadened()
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), new[] { TensorAgentSkillRouter.DocumentsSkill }, registry);

        Assert.Null(route);
    }

    [Fact]
    public void AnExplicitEmptySkillSelectionIsAlsoAnOptOut()
    {
        SkillRegistry registry = Registry(ResearchManifest(), DocumentsManifest());

        WebUiSkillRoute? route = TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), Array.Empty<string>(), registry);

        Assert.Null(route);
    }

    [Theory]
    [InlineData(TensorAgentSkillRouter.ResearchSkill)]
    [InlineData(TensorAgentSkillRouter.DocumentsSkill)]
    public void RoutingDoesNotClaimARequiredSkillThatIsNotInstalled(string missingSkill)
    {
        SkillRegistry registry = missingSkill == TensorAgentSkillRouter.ResearchSkill
            ? Registry(DocumentsManifest())
            : Registry(ResearchManifest());

        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), requestedSkills: null, registry));
    }

    [Theory]
    [InlineData(TensorAgentSkillRouter.ResearchSkill, "research-v2.py")]
    [InlineData(TensorAgentSkillRouter.DocumentsSkill, "make_pptx-v2.py")]
    public void InstalledShadowWithoutItsExactRequiredScriptCannotActivateTheRoute(
        string shadowedSkill,
        string wrongScriptName)
    {
        string bundled = Path.Combine(_root, "bundled");
        string installed = Path.Combine(_root, "installed");
        WriteSkill(bundled, TensorAgentSkillRouter.ResearchSkill, ResearchManifest(), includeScript: true);
        WriteSkill(bundled, TensorAgentSkillRouter.DocumentsSkill, DocumentsManifest(), includeScript: true);
        string shadowManifest = shadowedSkill == TensorAgentSkillRouter.ResearchSkill
            ? ResearchManifest()
            : DocumentsManifest();
        WriteSkill(installed, shadowedSkill, shadowManifest, includeScript: false);
        string wrongScripts = Path.Combine(installed, shadowedSkill, "scripts");
        Directory.CreateDirectory(wrongScripts);
        File.WriteAllText(Path.Combine(wrongScripts, wrongScriptName), "print('wrong resource')");

        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            Roots = new[] { bundled },
            InstallDirectory = installed,
        });

        Assert.True(registry.TryGet(shadowedSkill, out Skill winner));
        Assert.Equal(SkillOrigin.Installed, winner.Origin);
        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), requestedSkills: null, registry));
    }

    [Theory]
    [InlineData(TensorAgentSkillRouter.ResearchSkill)]
    [InlineData(TensorAgentSkillRouter.DocumentsSkill)]
    public void InstalledShadowWithTheSameScriptPathStillCannotBeAutoExecuted(
        string shadowedSkill)
    {
        string bundled = Path.Combine(_root, "bundled-exact");
        string installed = Path.Combine(_root, "installed-exact");
        WriteSkill(bundled, TensorAgentSkillRouter.ResearchSkill, ResearchManifest(), includeScript: true);
        WriteSkill(bundled, TensorAgentSkillRouter.DocumentsSkill, DocumentsManifest(), includeScript: true);
        WriteSkill(
            installed,
            shadowedSkill,
            shadowedSkill == TensorAgentSkillRouter.ResearchSkill
                ? ResearchManifest("USER_CONTROLLED_SENTINEL")
                : DocumentsManifest("USER_CONTROLLED_SENTINEL"),
            includeScript: true);

        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            Roots = new[] { bundled },
            InstallDirectory = installed,
        });

        Assert.True(registry.TryGet(shadowedSkill, out Skill winner));
        Assert.Equal(SkillOrigin.Installed, winner.Origin);
        Assert.Null(TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), requestedSkills: null, registry));
    }

    [Fact]
    public void TheEightKPromptSelectsBothSkillsWithoutInliningTheirBodies()
    {
        const string researchSentinel = "RESEARCH_FULL_BODY_MUST_NOT_BE_INLINED";
        const string documentsSentinel = "DOCUMENTS_FULL_BODY_MUST_NOT_BE_INLINED";
        string research = ResearchManifest(researchSentinel + new string('R', 13_000));
        string documents = DocumentsManifest(documentsSentinel + new string('D', 13_000));
        SkillRegistry registry = Registry(research, documents);
        WebUiSkillRoute route = Assert.IsType<WebUiSkillRoute>(TensorAgentSkillRouter.Route(
            UserTurn(NeutralPrompt), requestedSkills: null, registry));

        IReadOnlyList<Skill> selected = registry.Resolve(route.Skills, out IReadOnlyList<string> unknown);
        SkillPlan plan = SkillPrompt.Plan(selected, registry.Skills, new SkillPromptOptions
        {
            ContextTokens = 8_192,
            ToolsAvailable = true,
        });
        List<ChatMessage> messages = SkillPrompt.Apply(UserTurn(NeutralPrompt).ToList(), plan);
        messages = SkillPrompt.Apply(messages, route.Instructions);

        Assert.Empty(unknown);
        Assert.Equal(2, plan.Selected.Count);
        Assert.Empty(plan.Inlined);
        Assert.Equal(2, plan.Deferred.Count);
        Assert.DoesNotContain(researchSentinel, messages[0].Content, StringComparison.Ordinal);
        Assert.DoesNotContain(documentsSentinel, messages[0].Content, StringComparison.Ordinal);
        Assert.Contains("- documents:", messages[0].Content, StringComparison.Ordinal);
        Assert.Contains("- research:", messages[0].Content, StringComparison.Ordinal);
        Assert.Contains(TensorAgentSkillRouter.ActivationInstructions, messages[0].Content, StringComparison.Ordinal);

        int promptBytes = Encoding.UTF8.GetByteCount(messages[0].Content);
        Assert.True(promptBytes < 6_000,
            $"The compact route used {promptBytes} prompt bytes; it should not approach the 26 KB skill bodies.");
    }

    private static IReadOnlyList<ChatMessage> UserTurn(string content) =>
        new[] { new ChatMessage { Role = "user", Content = content } };

    private static string ResearchQueryOf(WebUiSkillRoute route)
    {
        WebUiArtifactRequirement artifact = Assert.IsType<WebUiArtifactRequirement>(route.ArtifactRequirement);
        WebUiSkillRunRequirement research = Assert.Single(
            artifact.RequiredRuns, run => run.SkillId == TensorAgentSkillRouter.ResearchSkill);
        return research.DefaultArguments[0];
    }

    private SkillRegistry Registry(params string[] manifests)
    {
        foreach (string manifest in manifests)
        {
            string name = manifest.Contains("name: research", StringComparison.Ordinal)
                ? TensorAgentSkillRouter.ResearchSkill
                : TensorAgentSkillRouter.DocumentsSkill;
            WriteSkill(_root, name, manifest, includeScript: true);
        }

        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _root } });
    }

    private static void WriteSkill(string root, string name, string manifest, bool includeScript)
    {
        string directory = Path.Combine(root, name);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, SkillManifestParser.SkillFileName), manifest);
        if (!includeScript)
            return;

        string scripts = Path.Combine(directory, "scripts");
        Directory.CreateDirectory(scripts);
        string scriptName = name == TensorAgentSkillRouter.ResearchSkill
            ? "research.py"
            : "make_pptx.py";
        File.WriteAllText(Path.Combine(scripts, scriptName), "print('fixture')");
    }

    private static string ResearchManifest(string? body = null) => $$"""
        ---
        name: research
        description: Research a question on the open web, find sources, compare claims, and cite their URLs and dates.
        ---
        # Research
        Use the bundled research workflow and write notes.md plus notes.json.
        {{body}}
        """;

    private static string DocumentsManifest(string? body = null) => $$"""
        ---
        name: documents
        description: Create real PDF, XLSX, DOCX and PPTX documents with bundled writers in the shared workspace.
        ---
        # Documents
        Use the bundled make_pptx.py writer for a PowerPoint deck.
        {{body}}
        """;
}
