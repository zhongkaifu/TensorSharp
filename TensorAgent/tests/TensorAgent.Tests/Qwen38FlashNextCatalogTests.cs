// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// Qwen3.8 Flash Next: the catalog's first split GGUF (three shards) and its first entry whose
/// file is larger than the memory of the tier it is offered from, because the engine reads its
/// n-gram table and most of its experts from the SSD as tokens need them.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class Qwen38FlashNextCatalogTests : IDisposable
{
    private const string Id = "qwen3.8-flash-next-q2kxl";
    private const string Repo = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q2_K_XL/";

    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-q4e-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort test cleanup */ }
    }

    private static CatalogModel Model => ModelCatalog.Find(Id)!;

    [Fact]
    public void CatalogPinsTheThreeShardsOfThePublishedFile()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn, m => m.Id == Id);
        Assert.Equal(CatalogFamily.Qwen38FlashNext, model.Family);
        Assert.Equal(Id, model.Id);
        Assert.Equal("Qwen3.8 Flash Next", model.DisplayName);
        Assert.Equal(CatalogArchitectureKind.MixtureOfExperts, model.Kind);
        Assert.Equal("UD-Q2_K_XL", model.Quantization);

        (string Name, long Bytes, string Sha)[] expected =
        {
            ("Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", 10_946_624,
                "a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86"),
            ("Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf", 49_979_779_296,
                "2e3bf1ee7d2a04e261e9f342a2d968f696cce5941d082b0e434deb9b1edc12c6"),
            ("Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf", 28_878_402_944,
                "ec8c106759fdf4f463039c34c0707718d7d8908d53d892bd4f002e71620803f9"),
        };
        CatalogFile[] weights = model.WeightFiles.ToArray();
        Assert.Equal(expected.Length, weights.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(i == 0 ? CatalogFileRole.Weights : CatalogFileRole.WeightsShard, weights[i].Role);
            Assert.Equal(expected[i].Name, weights[i].FileName);
            Assert.Equal(Repo + expected[i].Name, weights[i].Url);
            Assert.Equal(expected[i].Bytes, weights[i].Bytes);
            Assert.Equal(expected[i].Sha, weights[i].Sha256);
            Assert.False(weights[i].Optional);
        }
        Assert.Equal(expected.Length + 2, model.Files.Count);
        Assert.True(model.Projector is { Optional: true });
        CatalogFile draft = Assert.Single(model.Files, f => f.Role == CatalogFileRole.Draft);
        Assert.True(draft.Optional);
        Assert.Equal("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", draft.FileName);
        Assert.Equal(2_786_568_256, draft.Bytes);
        Assert.Equal("5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6", draft.Sha256);
        Assert.Equal(78_869_128_864, model.TotalBytes);
        Assert.Equal(model.TotalBytes, model.WeightsBytes);
        Assert.Equal(CatalogModalities.Image | CatalogModalities.Video, model.Modalities);
    }

    [Fact]
    public void ItIsOfferedOnlyToA48GbMacAndPagesMostOfItFromTheSsd()
    {
        CatalogModel model = Model;
        Assert.Equal(48, model.MinDeviceMemoryGB);
        Assert.DoesNotContain(ModelCatalog.ForDevice(32), m => m.Id == Id);
        Assert.Contains(ModelCatalog.ForDevice(48), m => m.Id == Id);
        // The n-gram table plus the 33 layers' experts the engine offloads on a 48 GiB Mac.
        Assert.Equal(28_800_138_240 + 31_732_531_200, model.WeightsPagedFromDiskBytes);
        Assert.Equal(18_336_459_424, model.ResidentWeightsBytes);
    }

    [Fact]
    public void ItAsksForAnF16CacheTheSpanCanReadAndThePhonesBudget()
    {
        CatalogModel model = Model;
        Assert.Equal("f16", model.KvCacheDtype);
        Assert.Equal(32768, model.ContextLength);
        Assert.True(model.LeanCaches);
        Assert.True(model.SupportsThinking);
        Assert.True(model.Experimental);
        Assert.Equal(new CatalogSampling(1.0f, 20, 0.95f, 0.0f), model.Sampling);
        Assert.Equal("Qwen Community License 1.0", model.License);
    }

    [Fact]
    public void SavedSelectionResolvesToTheFirstShard()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        var settings = new AppSettings { SelectedModelId = Id };
        Assert.Equal(
            Path.Combine(paths.ModelsDirectory, Id, "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"),
            paths.SelectedModelPath(settings));
    }

    /// <summary>The store counts every shard: the first file alone is not an installed model, and a
    /// short later shard is not either.</summary>
    [Fact]
    public void TheStoreCallsItInstalledOnlyWithEveryShardComplete()
    {
        // CatalogPinsTheThreeShardsOfThePublishedFile checks the actual byte counts.
        // Exercise installation states with small files; SetLength allocates the full
        // published 79 GB on file systems that do not create sparse files automatically.
        CatalogModel model = Model with
        {
            Files = Model.Files.Select((file, index) => file with { Bytes = 64 + index * 16 }).ToArray()
        };
        var store = new ModelStore(Path.Combine(_root, "models"));
        Directory.CreateDirectory(store.DirectoryFor(model));
        Assert.NotEqual(InstallState.Installed, store.StateOf(model));
        foreach (CatalogFile f in model.WeightFiles)
        {
            using FileStream s = File.Create(store.PathFor(model, f));
            s.SetLength(f == model.WeightFiles.Last() ? f.Bytes - 1 : f.Bytes);
        }
        Assert.NotEqual(InstallState.Installed, store.StateOf(model));
        using (FileStream s = File.OpenWrite(store.PathFor(model, model.WeightFiles.Last())))
            s.SetLength(model.WeightFiles.Last().Bytes);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
    }

    /// <summary>
    /// With the first shard present and the others missing, the app refuses the load itself
    /// instead of letting the engine fail on a file name the user never saw.
    /// </summary>
    [Fact]
    public void AnIncompleteSplitIsNotLoaded()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache")) { DeviceMemoryGB = 48 };
        paths.EnsureCreated();
        using var host = new AgentAppHost(paths);
        var store = new ModelStore(paths.ModelsDirectory);
        CatalogModel model = Model;
        Directory.CreateDirectory(store.DirectoryFor(model));
        using (FileStream s = File.Create(store.PathFor(model, model.Weights)))
            s.SetLength(model.Weights.Bytes);

        var ex = Assert.Throws<FileNotFoundException>(() => host.UseModel(model, warmAfterwards: false));
        Assert.Equal("Qwen3.8 Flash Next is not completely downloaded yet.", ex.Message);
        Assert.Equal(AgentAppHost.ModelLoadState.Failed, host.ModelLoad);
    }
}
