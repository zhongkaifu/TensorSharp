using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

public sealed class Bonsai2CatalogTests : IDisposable
{
    private const string Repo = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/b072e1d3b35a0a630cece372c2127528e0994386/";

    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-bonsai2-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort test cleanup */ }
    }

    [Fact]
    public void CatalogPinsTheBonsai2PublisherArtifacts()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn,
            m => m.Family == CatalogFamily.Bonsai);

        Assert.Equal("bonsai-2-27b-ptq1-0", model.Id);
        Assert.Equal("Bonsai 2 27B", model.DisplayName);
        Assert.Equal("PTQ1_0", model.Quantization);
        Assert.False(model.SideloadOnly);

        Assert.Equal("Ternary-Bonsai-2-27B-PTQ1_0.gguf", model.Weights.FileName);
        Assert.Equal(Repo + "Ternary-Bonsai-2-27B-PTQ1_0.gguf", model.Weights.Url);
        Assert.Equal(5_946_648_928, model.Weights.Bytes);
        Assert.Equal("53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
            model.Weights.Sha256);

        // Vision is an optional companion: text-only use never downloads it.
        CatalogFile projector = Assert.IsType<CatalogFile>(model.Projector);
        Assert.True(projector.Optional);
        Assert.Equal("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf", projector.FileName);
        Assert.Equal(Repo + "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf", projector.Url);
        Assert.Equal(629_246_976, projector.Bytes);
        Assert.Equal("6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903",
            projector.Sha256);
        Assert.Equal(model.Weights.Bytes, model.TotalBytes);
        Assert.Equal(CatalogModalities.Image, model.Modalities);

        Assert.Equal(new CatalogSampling(1.0f, 20, 0.95f, 0.05f), model.Sampling);
        Assert.True(model.SupportsThinking);
        Assert.Equal("q8_0", model.KvCacheDtype);
        Assert.Equal(32768, model.ContextLength);
        Assert.True(model.Experimental);
        Assert.Equal("Apache-2.0", model.License);
    }

    [Fact]
    public void TheRepackedWeightsKeepBonsai2OffTwelveGigabytePhones()
    {
        // The loader repacks PTQ1_0 losslessly to GGML Q2_0 in anonymous memory (~7.7 GB
        // for this file), which does not fit the ~8.5 GB a 12 GB iPhone grants the app.
        CatalogModel model = ModelCatalog.Find("bonsai-2-27b-ptq1-0")!;
        Assert.Equal(16, model.MinDeviceMemoryGB);
        Assert.DoesNotContain(model, ModelCatalog.ForDevice(12));
        Assert.Contains(model, ModelCatalog.ForDevice(16));
    }

    [Fact]
    public void SavedSelectionResolvesToTheCatalogOwnedWeightsPath()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        var settings = new AppSettings { SelectedModelId = "bonsai-2-27b-ptq1-0" };

        Assert.Equal(
            Path.Combine(paths.ModelsDirectory, "bonsai-2-27b-ptq1-0", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
            paths.SelectedModelPath(settings));
    }
}
