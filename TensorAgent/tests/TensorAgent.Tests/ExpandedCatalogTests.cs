using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

[Collection(ProcessEnvironmentCollection.Name)]
public sealed class ExpandedCatalogTests
{
    [Fact]
    public void EveryGenerationFamilyHasADownloadableEntry()
    {
        foreach (CatalogFamily family in Enum.GetValues<CatalogFamily>())
        {
            var entries = ModelCatalog.BuiltIn.Where(m => m.Family == family).ToArray();
            Assert.NotEmpty(entries);
            Assert.All(entries, m => Assert.All(m.Files, f => Assert.NotEmpty(f.Url)));
        }
    }

    [Fact]
    public void EverySavedSelectionResolvesItsOwnWeightsAndProjector()
    {
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-catalog-paths");
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"));
        foreach (CatalogModel model in ModelCatalog.BuiltIn)
        {
            var settings = new AppSettings { SelectedModelId = model.Id };
            Assert.Equal(Path.Combine(paths.ModelsDirectory, model.Id, model.Weights.FileName), paths.SelectedModelPath(settings));
            Assert.Equal(model.Projector is { } projector
                ? Path.Combine(paths.ModelsDirectory, model.Id, projector.FileName) : null,
                paths.SelectedProjectorPath(settings));
        }
    }

    [Fact]
    public void WanCompanionsAreRequiredAndSwitchingFamiliesClearsStaleNetworks()
    {
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-wan-catalog-" + Guid.NewGuid().ToString("N"));
        string[] variables = { "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE", "TS_VIDEO_DIT2", "TS_VIDEO_AUDIO_VAE", "TS_VIDEO_TOKENIZER" };
        var saved = variables.ToDictionary(v => v, Environment.GetEnvironmentVariable);
        var store = new ModelStore(root);
        try
        {
            foreach (CatalogModel model in ModelCatalog.BuiltIn.Where(m => m.Family == CatalogFamily.Wan))
            {
                Assert.True(model.IsVideoGenerator);
                Assert.False(model.Modalities.HasFlag(CatalogModalities.AudioOutput));
                Assert.Single(model.Files, f => f.Role == CatalogFileRole.TextEncoder && !f.Optional);
                Assert.Single(model.Files, f => f.Role == CatalogFileRole.Vae && !f.Optional);
                Directory.CreateDirectory(store.DirectoryFor(model));
                // Path publication is tested without allocating or loading multi-GB networks.
                foreach (CatalogFile file in model.Files)
                    File.WriteAllBytes(store.PathFor(model, file), Array.Empty<byte>());
                Environment.SetEnvironmentVariable("TS_VIDEO_AUDIO_VAE", "stale-audio");
                Environment.SetEnvironmentVariable("TS_VIDEO_TOKENIZER", "stale-tokenizer");
                Environment.SetEnvironmentVariable("TS_VIDEO_DIT2", "stale-expert");

                var published = DiffusionCompanions.Publish(model, store);
                foreach (var pair in new[]
                {
                    (CatalogFileRole.TextEncoder, "TS_VIDEO_TEXT_ENCODER"),
                    (CatalogFileRole.Vae, "TS_VIDEO_VAE"),
                    (CatalogFileRole.SecondaryWeights, "TS_VIDEO_DIT2"),
                })
                {
                    CatalogFile? file = model.Files.SingleOrDefault(f => f.Role == pair.Item1);
                    string? expected = file is null ? null : store.PathFor(model, file);
                    Assert.Equal(expected, Environment.GetEnvironmentVariable(pair.Item2));
                    Assert.Equal(file is not null, published.ContainsKey(pair.Item2));
                }
                Assert.Null(Environment.GetEnvironmentVariable("TS_VIDEO_AUDIO_VAE"));
                Assert.Null(Environment.GetEnvironmentVariable("TS_VIDEO_TOKENIZER"));
            }
            DiffusionCompanions.Publish(ModelCatalog.Find("qwen3.5-9b-iq4xs"), store);
            Assert.All(variables, v => Assert.Null(Environment.GetEnvironmentVariable(v)));
        }
        finally
        {
            foreach (var pair in saved) Environment.SetEnvironmentVariable(pair.Key, pair.Value);
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    [Theory]
    [InlineData(95_500_000_000L, 96)]
    [InlineData(191_500_000_000L, 192)]
    [InlineData(511_500_000_000L, 512)]
    [InlineData(1_024_000_000_000L, 1024)]
    public void WorkstationMemoryIsNotClampedTo128Gb(long bytes, int expected) =>
        Assert.Equal(expected, ModelCatalog.DeviceMemoryTier(bytes));
}
