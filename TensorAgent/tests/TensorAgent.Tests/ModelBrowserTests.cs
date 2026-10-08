using TensorAgent.Core.Catalog;
using TensorAgent.Core.Localization;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Tests;

[Collection(UiLanguageCollection.Name)]
public sealed class ModelBrowserTests
{
    private static readonly IReadOnlySet<string> NoneInstalled = new HashSet<string>(StringComparer.Ordinal);

    [Fact]
    public void EveryCatalogEntryAppearsExactlyOnceUnderItsFamilyWithoutCopyingIt()
    {
        var groups = Browse();
        Assert.Equal(14, groups.Count);
        Assert.Equal(groups.Count, groups.Select(group => group.Id).Distinct().Count());
        CatalogModel[] results = groups.SelectMany(group => group.Models).ToArray();
        Assert.Equal(ModelCatalog.BuiltIn.Count, results.Length);
        foreach (CatalogModel model in ModelCatalog.BuiltIn)
        {
            var group = Assert.Single(groups, group => group.Id == ModelBrowser.FamilyId(model));
            Assert.Same(model, Assert.Single(group.Models, result => result.Id == model.Id));
            Assert.Equal(ModelBrowser.FamilyName(model), group.DisplayName);
        }
    }

    [Theory]
    [InlineData(CatalogFamily.Gemma4, "gemma", "Gemma")]
    [InlineData(CatalogFamily.Qwen35, "qwen", "Qwen")]
    [InlineData(CatalogFamily.Qwen36, "qwen", "Qwen")]
    [InlineData(CatalogFamily.Qwen38, "qwen", "Qwen")]
    [InlineData(CatalogFamily.Qwen38FlashNext, "qwen", "Qwen")]
    [InlineData(CatalogFamily.QwenImage, "qwen-image", "Qwen Image")]
    [InlineData(CatalogFamily.GptOss, "gpt-oss", "GPT-OSS")]
    [InlineData(CatalogFamily.Bonsai, "bonsai", "Bonsai")]
    [InlineData(CatalogFamily.MuseGlimmer, "muse-glimmer", "Muse-Glimmer")]
    [InlineData(CatalogFamily.MiniMaxH3, "minimax-h3", "MiniMax H3")]
    [InlineData(CatalogFamily.DeepSeek4, "deepseek", "DeepSeek")]
    [InlineData(CatalogFamily.DeepSeek41, "deepseek", "DeepSeek")]
    [InlineData(CatalogFamily.Glm5, "glm", "GLM")]
    [InlineData(CatalogFamily.Nemotron, "nemotron", "Nemotron")]
    [InlineData(CatalogFamily.Mistral3, "mistral", "Mistral")]
    [InlineData(CatalogFamily.HunyuanDense, "hunyuan", "Hunyuan")]
    [InlineData(CatalogFamily.DiffusionGemma, "diffusiongemma", "DiffusionGemma")]
    [InlineData(CatalogFamily.Wan, "wan", "Wan")]
    public void FamilyNamesAreHumanReadableAndVersionIndependent(CatalogFamily family, string id, string name)
    {
        CatalogModel model = ModelCatalog.BuiltIn.First(model => model.Family == family);
        Assert.Equal(id, ModelBrowser.FamilyId(model));
        Assert.Equal(name, ModelBrowser.FamilyName(model));
    }

    [Theory]
    [InlineData("qwen3.8")]
    [InlineData("QWEN 3.8")]
    [InlineData("  qwen-3.8  ")]
    public void VersionsIgnoreCaseWhitespaceAndPunctuation(string query)
    {
        string[] ids = Results(query).Select(model => model.Id).Order().ToArray();
        Assert.Equal(new[] { "qwen3.8-27b-q4kxl", "qwen3.8-flash-next-iq1m", "qwen3.8-flash-next-q2kxl" }, ids);
    }

    [Theory]
    [InlineData("q4 k m")]
    [InlineData("Q4_K_M")]
    [InlineData("q4-k-m")]
    public void QuantizationSeparatorsAreOptional(string query)
    {
        var results = Results("qwen3.6 " + query);
        Assert.Equal(2, results.Count);
        Assert.All(results, model => Assert.Contains("Q4_K_M", model.Quantization));
    }

    [Fact]
    public void SearchRequiresEveryTermButAllowsTermsFromDifferentFields()
    {
        CatalogModel result = Assert.Single(Results("qwen3.8 27b vision dflash"));
        Assert.Equal("qwen3.8-27b-q4kxl", result.Id);
        Assert.Empty(Results("qwen3.8 27b no-such-feature"));
    }

    [Fact]
    public void VersionSearchDoesNotMistakeParameterCountsForVersionNumbers()
    {
        var results = Results("qwen 3.5");
        Assert.Equal(2, results.Count);
        Assert.All(results, model => Assert.Equal(CatalogFamily.Qwen35, model.Family));
    }

    [Fact]
    public void SpacedQuantizationDoesNotMatchItsSuffixAgainstUnrelatedFields()
    {
        var results = Results("q4 k m");
        Assert.NotEmpty(results);
        Assert.All(results, model => Assert.Contains("Q4_K_M", model.Quantization));
        Assert.DoesNotContain(results, model => model.Id == "qwen3.8-27b-q4kxl");
    }

    [Fact]
    public void FamilySearchFindsNamesThatDoNotContainTheirFamily()
    {
        var group = Assert.Single(Browse("hunyuan"));
        Assert.Equal("hunyuan", group.Id);
        Assert.Equal("hy-mt2-1.8b-q4km", Assert.Single(group.Models).Id);
    }

    [Fact]
    public void SearchFindsCompanionsWithoutClaimingUnsupportedDrafts()
    {
        Assert.Contains(Results("mmproj"), model => model.Id == "qwen3.8-flash-next-iq1m");
        Assert.Contains(Results("speculative draft"), model => model.Id == "gemma-4-e4b-iq4xs");
        Assert.Contains(Results("dspark"), model => model.Id == "deepseek-v4-flash-0731-q2kxl");
        Assert.Empty(Results("eagle3"));
        Assert.DoesNotContain(Results("draft"), model => model.Family == CatalogFamily.GptOss);
    }

    [Fact]
    public void CompatibleFilterNeverIncludesAnIncompatibleSelectedOrDownloadedModel()
    {
        CatalogModel large = ModelCatalog.BuiltIn.Single(model => model.Id == "qwen3.8-flash-next-q2kxl");
        var results = Browse(filter: ModelBrowseFilter.Compatible, memory: 12,
            installed: new HashSet<string> { large.Id }, selected: large.Id).SelectMany(group => group.Models).ToArray();
        Assert.NotEmpty(results);
        Assert.All(results, model => Assert.True(model.MinDeviceMemoryGB <= 12));
        Assert.DoesNotContain(results, model => model.Id == large.Id);
        Assert.Equal(48, large.MinDeviceMemoryGB);
    }

    [Fact]
    public void DownloadedFilterIncludesOnlyInstalledWeightsEvenOnASmallerDevice()
    {
        string[] installed = ["gemma-4-e4b-iq4xs", "qwen3.8-flash-next-q2kxl"];
        var results = Browse(filter: ModelBrowseFilter.Downloaded, memory: 8,
            installed: installed.ToHashSet(StringComparer.Ordinal)).SelectMany(group => group.Models).ToArray();
        Assert.Equal(installed.Order(), results.Select(model => model.Id).Order());
        Assert.All(results, model => Assert.Same(ModelCatalog.Find(model.Id), model));
    }

    [Fact]
    public void SelectedThenInstalledThenCompatibleModelsLeadBrowsingWithinTheirFamily()
    {
        string selected = "qwen3.8-flash-next-q2kxl";
        string installed = "qwen3.8-27b-q4kxl";
        var groups = Browse(memory: 16, installed: new HashSet<string> { installed }, selected: selected);
        var qwen = groups.Single(group => group.Id == "qwen").Models;
        Assert.Equal(selected, qwen[0].Id);
        Assert.Equal(installed, qwen[1].Id);
        Assert.True(qwen[2].MinDeviceMemoryGB <= 16);
    }

    [Fact]
    public void FamilyOrderRemainsStableAcrossSelectionChanges()
    {
        Assert.Equal(Browse().Select(group => group.Id),
            Browse(selected: "qwen3.8-27b-q4kxl").Select(group => group.Id));
    }

    [Fact]
    public void ExactNameMatchRanksBeforeSelectedDescriptiveMatch()
    {
        CatalogModel original = ModelCatalog.BuiltIn[0];
        CatalogModel exact = original with { Id = "exact", DisplayName = "Vision" };
        CatalogModel descriptive = original with { Id = "selected", DisplayName = "Alpha", Modalities = CatalogModalities.Image };
        var group = Assert.Single(ModelBrowser.Browse([descriptive, exact], "vision", ModelBrowseFilter.All,
            128, new HashSet<string> { descriptive.Id }, descriptive.Id));
        Assert.Same(exact, group.Models[0]);
    }

    [Fact]
    public void FlatSearchRanksNamesAcrossFamiliesBeforeCompanionMatches()
    {
        var results = ModelBrowser.Search(ModelCatalog.BuiltIn, "qwen", ModelBrowseFilter.All,
            128, NoneInstalled, "minimax-h3-fl2va-q4k");
        int firstCompanion = results.ToList().FindIndex(model => model.Family == CatalogFamily.MiniMaxH3);
        int lastQwen = results.ToList().FindLastIndex(model => ModelBrowser.FamilyId(model) is "qwen" or "qwen-image");
        Assert.True(firstCompanion > lastQwen);
    }

    [Fact]
    public void NaturalOrderingIsStableWhenTheInputOrderChanges()
    {
        CatalogModel original = ModelCatalog.BuiltIn[0];
        CatalogModel[] models = [original with { Id = "12", DisplayName = "Model 12B" },
            original with { Id = "2", DisplayName = "Model 2B" },
            original with { Id = "9", DisplayName = "Model 9B" }];
        string[] Order(IEnumerable<CatalogModel> input) => ModelBrowser.Browse(input, null,
            ModelBrowseFilter.All, 128, NoneInstalled, null).SelectMany(group => group.Models).Select(model => model.Id).ToArray();
        Assert.Equal(new[] { "2", "9", "12" }, Order(models));
        Assert.Equal(Order(models), Order(models.Reverse()));
    }

    [Fact]
    public void SearchIgnoresDiacritics()
    {
        CatalogModel model = ModelCatalog.BuiltIn[0] with { DisplayName = "Café modèle" };
        var group = Assert.Single(ModelBrowser.Browse([model], "CAFE modele", ModelBrowseFilter.All,
            128, NoneInstalled, null));
        Assert.Same(model, Assert.Single(group.Models));
    }

    [Fact]
    public void CapabilitySearchUsesTheCurrentInterfaceLanguage()
    {
        try
        {
            Loc.Use(UiLanguages.Resolve("zh-Hans", []));
            string query = Loc.T("models.row.input.audio");
            Assert.Equal("音频", query);
            var results = Results(query);
            Assert.Contains(results, model => model.Id == "gemma-4-e2b-q8");
            Assert.DoesNotContain(results, model => model.Id == "hy-mt2-1.8b-q4km");
        }
        finally { Loc.Use(UiLanguages.English); }
    }

    [Fact]
    public void EmptyQueriesBrowseAndNoMatchesProduceNoEmptyGroups()
    {
        Assert.Equal(Browse().Select(group => group.Id), Browse("  \t -- ").Select(group => group.Id));
        Assert.Empty(Browse("no-such-model"));
        Assert.Empty(Browse(filter: ModelBrowseFilter.Downloaded));
    }

    private static IReadOnlyList<ModelFamilyGroup> Browse(string? query = null,
        ModelBrowseFilter filter = ModelBrowseFilter.All, int memory = 128,
        IReadOnlySet<string>? installed = null, string? selected = null) =>
        ModelBrowser.Browse(ModelCatalog.BuiltIn, query, filter, memory, installed ?? NoneInstalled, selected);

    private static IReadOnlyList<CatalogModel> Results(string query) =>
        Browse(query).SelectMany(group => group.Models).ToArray();
}
