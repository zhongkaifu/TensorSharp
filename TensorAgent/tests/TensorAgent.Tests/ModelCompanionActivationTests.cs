using System.Collections;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server;

namespace TensorAgent.Tests;

/// <summary>Downloading a companion after a model is selected must reach the real
/// model-load path. Tiny fixture files and a managed fake factory exercise the host
/// and lifecycle without loading production weights or calling native kernels.</summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class ModelCompanionActivationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-companions-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, string?> _environment = Environment.GetEnvironmentVariables()
        .Cast<DictionaryEntry>().ToDictionary(e => (string)e.Key, e => (string?)e.Value, StringComparer.Ordinal);

    public void Dispose()
    {
        foreach (string key in Environment.GetEnvironmentVariables().Keys.Cast<string>().Except(_environment.Keys))
            Environment.SetEnvironmentVariable(key, null);
        foreach ((string key, string? value) in _environment)
            Environment.SetEnvironmentVariable(key, value);
        KvCacheDtypeConfig.ConfigureFromEnvironment();
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [InlineData(CatalogFileRole.Projector)]
    [InlineData(CatalogFileRole.Draft)]
    public void UsingTheSelectedModelAgainLoadsANewCompanion(CatalogFileRole role)
    {
        SpeculationPolicy.UseLaunchEnvironment(null, null);
        CatalogModel original = ModelCatalog.Find("qwen3.5-9b-iq4xs")!;
        CatalogFile weights = original.Weights with { Bytes = 32 };
        CatalogFile companion = new(role, "optional.gguf", "", 32, "", Optional: true);
        CatalogModel model = original with { Files = new[] { weights, companion }, KvCacheDtype = "f16" };
        var loads = new List<(FakeModel Model, string? Draft)>();
        var service = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, draft) =>
        {
            var result = new FakeModel(path, draft is not null);
            loads.Add((result, draft));
            return result;
        });
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        using var host = new AgentAppHost(paths, modelService: service,
            backends: new[] { new BackendOption("cpu", "CPU") });
        Directory.CreateDirectory(host.Models.DirectoryFor(model));
        WriteGguf(host.Models.PathFor(model, weights));

        Assert.Equal("cpu", host.UseModel(model, warmAfterwards: false));
        Assert.Single(loads);
        Assert.Null(service.LoadedMmProjPath);
        Assert.False(host.CatalogDraftHeadAttached);
        // Re-selecting an unchanged installation still avoids an expensive reload.
        host.UseModel(model, warmAfterwards: false);
        Assert.Single(loads);

        string companionPath = host.Models.PathFor(model, companion);
        WriteGguf(companionPath);
        Assert.Equal("cpu", host.UseModel(model, warmAfterwards: false));
        Assert.Equal(2, loads.Count);
        Assert.True(loads[0].Model.Disposed);
        if (role == CatalogFileRole.Projector)
        {
            Assert.Equal(companionPath, service.LoadedMmProjPath);
            Assert.Equal(companionPath, loads[1].Model.ProjectorPath);
            Assert.True(service.Model.HasVisionEncoder());
        }
        else
        {
            Assert.Equal(companionPath, loads[1].Draft);
            Assert.True(host.CatalogDraftHeadAttached);
            Assert.Equal(SpeculatorRegistry.Auto, Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable));
        }
        host.UseModel(model, warmAfterwards: false);
        Assert.Equal(2, loads.Count);
    }

    private static void WriteGguf(string path)
    {
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(0UL);
        writer.Write(0UL);
        writer.Write(new byte[8]);
    }

    [Fact]
    public void AFailedExternalDraftDoesNotActivateAnEmbeddedHeadAndCanBeRetried()
    {
        SpeculationPolicy.UseLaunchEnvironment(null, null);
        CatalogModel original = ModelCatalog.Find("qwen3.5-9b-iq4xs")!;
        CatalogFile weights = original.Weights with { Bytes = 32 };
        CatalogFile draft = new(CatalogFileRole.Draft, "draft.gguf", "", 32, "", Optional: true);
        CatalogModel model = original with { Files = new[] { weights, draft }, KvCacheDtype = "f16" };
        int loads = 0;
        var service = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) =>
        {
            loads++;
            // This target has an embedded head, but cannot attach the requested external
            // draft. The loader must report that failure even though HasDraftHead is true.
            return new FakeModel(path, draft: false, embedded: true);
        });
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        using var host = new AgentAppHost(paths, modelService: service,
            backends: new[] { new BackendOption("cpu", "CPU") });
        Directory.CreateDirectory(host.Models.DirectoryFor(model));
        WriteGguf(host.Models.PathFor(model, weights));
        host.UseModel(model, warmAfterwards: false);
        Assert.True(host.DraftHeadAttached);
        Assert.False(host.CatalogDraftHeadAttached);

        WriteGguf(host.Models.PathFor(model, draft));
        host.UseModel(model, warmAfterwards: false);
        Assert.Equal(2, loads);
        Assert.NotNull(service.DraftHeadActivationError);
        Assert.False(host.CatalogDraftHeadAttached);
        Assert.Equal(SpeculatorRegistry.NGram, Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable));
        host.ApplySpeculationSetting(host.Settings.Load());
        Assert.Equal(SpeculatorRegistry.NGram, Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable));

        host.UseModel(model, warmAfterwards: false);
        Assert.Equal(3, loads);
        Assert.False(host.CatalogDraftHeadAttached);
    }

    private sealed class FakeModel : ModelBase, IDraftHead, IVisionCapableModel
    {
        public FakeModel(string path, bool draft, bool embedded = false) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "llama" };
            DraftHeadKind = draft ? DraftHeadKind.Block : embedded ? DraftHeadKind.PerToken : DraftHeadKind.None;
        }

        public DraftHeadKind DraftHeadKind { get; }
        public string? ProjectorPath { get; private set; }
        public bool IsVisionEncoderLoaded => ProjectorPath is not null;
        public bool Disposed { get; private set; }
        public void DraftCatchUp(int[] tokens, float[]? hiddenStates, int startPos) => throw new NotSupportedException();
        public void LoadVisionEncoder(string path) => ProjectorPath = path;
        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => throw new NotSupportedException();
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
        public override void Dispose() { Disposed = true; base.Dispose(); }
    }
}
