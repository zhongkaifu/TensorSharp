using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// The IQ1_M download has its own identity and 32 GB tier. Its Windows/CUDA validation
/// does not change the Q2_K_XL entry's files, 48 GB gate, or measured Mac footprint.
/// </summary>
public sealed class Qwen38FlashNextIq1MCatalogTests : IDisposable
{
    private const string Id = "qwen3.8-flash-next-iq1m";
    private const string Repo = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-flash-iq1m-" + Guid.NewGuid().ToString("N"));

    private static CatalogModel Model => ModelCatalog.Find(Id)!;

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort test cleanup */ }
    }

    [Fact]
    public void TheIq1MVariantPinsAllThreeRequiredShardsAndTheOptionalVisionProjector()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn, m => m.Id == Id);
        Assert.Equal("UD-IQ1_M", model.Quantization);
        Assert.Equal(CatalogFamily.Qwen38FlashNext, model.Family);
        Assert.Equal(CatalogArchitectureKind.MixtureOfExperts, model.Kind);
        Assert.Equal(ModelCatalog.Find("qwen3.8-flash-next-q2kxl")!.Parameters, model.Parameters);

        (string Name, long Bytes, string Sha)[] expected =
        {
            ("Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf", 10_946_624,
                "6584f289c808a486ac33aef20906774bcf6616785535c03b842072951e5112d2"),
            ("Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf", 49_988_981_792,
                "a7cbafba2cbc1ccde484ed05e1bbeaef4d67e0924ef51cc8b8de18e62edb5586"),
            ("Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf", 24_538_827_360,
                "ae757ff9347651adacc7746f137d8c29b51f25cf6bb27cd49086ef45bd8cc203"),
        };
        CatalogFile[] weights = model.WeightFiles.ToArray();
        Assert.Equal(3, weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(i == 0 ? CatalogFileRole.Weights : CatalogFileRole.WeightsShard, weights[i].Role);
            Assert.Equal(expected[i].Name, weights[i].FileName);
            Assert.Equal(Repo + "UD-IQ1_M/" + expected[i].Name, weights[i].Url);
            Assert.Equal(expected[i].Bytes, weights[i].Bytes);
            Assert.Equal(expected[i].Sha, weights[i].Sha256);
            Assert.False(weights[i].Optional);
        }

        CatalogFile projector = Assert.IsType<CatalogFile>(model.Projector);
        Assert.Equal(CatalogFileRole.Projector, projector.Role);
        Assert.Equal("mmproj-BF16.gguf", projector.FileName);
        Assert.Equal(Repo + projector.FileName, projector.Url);
        Assert.Equal(907_542_944, projector.Bytes);
        Assert.Equal("2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355", projector.Sha256);
        Assert.True(projector.Optional);
        Assert.Equal(CatalogModalities.Image | CatalogModalities.Video, model.Modalities);
        Assert.Equal(5, model.Files.Count);
        Assert.Equal(74_538_755_776, model.WeightsBytes);
        Assert.Equal(model.WeightsBytes, model.TotalBytes);
        CatalogFile draft = Assert.Single(model.Files, f => f.Role == CatalogFileRole.Draft);
        Assert.True(draft.Optional);
        Assert.Equal(ModelCatalog.Find("qwen3.8-flash-next-q2kxl")!.Files.Single(f => f.Role == CatalogFileRole.Draft), draft);
        Assert.Equal(model.WeightsBytes + projector.Bytes + draft.Bytes, model.TotalBytesWithOptional);
    }

    [Fact]
    public void TheIq1MEntryIsOfferedAt32GbWhileQ2RemainsAt48Gb()
    {
        Assert.Equal(32, Model.MinDeviceMemoryGB);
        Assert.DoesNotContain(ModelCatalog.ForDevice(24), m => m.Id == Id);
        Assert.Contains(ModelCatalog.ForDevice(32), m => m.Id == Id);
        Assert.Contains(ModelCatalog.ForDevice(48), m => m.Id == Id);

        CatalogModel q2 = ModelCatalog.Find("qwen3.8-flash-next-q2kxl")!;
        Assert.Equal(48, q2.MinDeviceMemoryGB);
        Assert.DoesNotContain(ModelCatalog.ForDevice(32), m => m.Id == q2.Id);
        Assert.Contains(ModelCatalog.ForDevice(48), m => m.Id == q2.Id);
        Assert.Equal(63_468_682_240, Model.WeightsPagedFromDiskBytes);
        Assert.Equal(11_070_073_536, Model.ResidentWeightsBytes);
        Assert.True(Model.WeightsBytes > Model.MinDeviceMemoryGB * 1_000_000_000L);
        Assert.True(Model.ResidentWeightsBytes < Model.MinDeviceMemoryGB * 1_000_000_000L);
    }

    [Fact]
    public void TheIq1MEntryRequestsLeanCachesAndTheSupportedSpanPrecision()
    {
        Assert.Equal(32768, Model.ContextLength);
        Assert.Equal("f16", Model.KvCacheDtype);
        Assert.True(Model.LeanCaches);
        Assert.True(Model.SupportsThinking);
        Assert.True(Model.Experimental);
        Assert.False(Model.SideloadOnly);
        Assert.Equal(new CatalogSampling(1.0f, 20, 0.95f, 0.0f), Model.Sampling);
        Assert.Equal("Qwen Community License 1.0", Model.License);
    }

    [Fact]
    public void TheSavedChoiceResolvesTheFirstIq1MShardAndItsProjector()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        var settings = new AppSettings { SelectedModelId = Id };
        Assert.Equal(Path.Combine(paths.ModelsDirectory, Id, Model.Weights.FileName), paths.SelectedModelPath(settings));
        Assert.Equal(Path.Combine(paths.ModelsDirectory, Id, "mmproj-BF16.gguf"), paths.SelectedProjectorPath(settings));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void EveryIq1MShardMustBeCompleteBeforeTheEntryIsInstalled(int incompleteIndex)
    {
        CatalogModel model = SmallModel();
        var store = new ModelStore(Path.Combine(_root, "models"));
        CatalogFile[] weights = model.WeightFiles.ToArray();
        Directory.CreateDirectory(store.DirectoryFor(model));
        for (int i = 0; i < weights.Length; i++)
            if (i != incompleteIndex)
                WriteFile(store, model, weights[i], weights[i].Bytes);
        Assert.Equal(InstallState.Partial, store.StateOf(model));
        Assert.Null(store.WeightsPath(model));

        WriteFile(store, model, weights[incompleteIndex], weights[incompleteIndex].Bytes - 1);
        Assert.Equal(InstallState.Partial, store.StateOf(model));
        Assert.Null(store.WeightsPath(model));

        WriteFile(store, model, weights[incompleteIndex], weights[incompleteIndex].Bytes);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.Equal(store.PathFor(model, weights[0]), store.WeightsPath(model));
        Assert.Null(store.CompanionPath(model, CatalogFileRole.Projector));
    }

    [Fact]
    public void VisionIsAvailableOnlyWithTheCompleteOptionalProjector()
    {
        CatalogModel model = SmallModel();
        var store = new ModelStore(Path.Combine(_root, "models"));
        Directory.CreateDirectory(store.DirectoryFor(model));
        foreach (CatalogFile file in model.WeightFiles)
            WriteFile(store, model, file, file.Bytes);
        CatalogFile projector = model.Projector!;

        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.False(store.IsFileInstalled(model, projector));
        Assert.Null(store.CompanionPath(model, CatalogFileRole.Projector));
        Assert.Equal(0, store.RemainingBytes(model));
        Assert.Equal(projector.Bytes, store.RemainingBytes(model, new[] { CatalogFileRole.Projector }));
        Assert.Equal(model.Files.Where(f => f.Optional).Sum(f => f.Bytes), store.RemainingBytes(model, includeOptional: true));

        WriteFile(store, model, projector, projector.Bytes - 1);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.False(store.IsFileInstalled(model, projector));
        Assert.Null(store.CompanionPath(model, CatalogFileRole.Projector));

        WriteFile(store, model, projector, projector.Bytes);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.True(store.IsFileInstalled(model, projector));
        Assert.Equal(store.PathFor(model, projector), store.CompanionPath(model, CatalogFileRole.Projector));
        Assert.Equal(0, store.RemainingBytes(model, new[] { CatalogFileRole.Projector }));
        Assert.Equal(model.Files.Single(f => f.Role == CatalogFileRole.Draft).Bytes,
            store.RemainingBytes(model, includeOptional: true));
    }

    // Verify the production sizes and hashes above, then use small fixtures to exercise
    // installation transitions without creating 75 GB of files on ordinary file systems.
    private static CatalogModel SmallModel() => Model with
    {
        Files = Model.Files.Select((file, index) => file with { Bytes = 64 + index * 16 }).ToArray(),
    };

    private static void WriteFile(ModelStore store, CatalogModel model, CatalogFile file, long bytes)
    {
        using FileStream stream = File.Create(store.PathFor(model, file));
        stream.SetLength(bytes);
    }
}
