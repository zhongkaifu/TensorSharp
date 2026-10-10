using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace TensorAgent.Tests;

/// <summary>
/// The record of a model load in progress, and turns sent while a load runs. A model
/// whose load or first prefill takes the process down used to do it again at every
/// launch (the choice is saved before the load, and every launch loads it); and a
/// message sent during a switch could reach a half-loaded model. A managed fake factory
/// stands in for the weights, so these run without native kernels.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class ModelLoadRecordTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-loadrecord-" + Guid.NewGuid().ToString("N"));

    private AgentPaths Paths => new(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
    {
        ExecutionMode = AgentExecutionMode.InProcess,
        DeviceMemoryGB = 48,
        DeviceClass = DeviceClass.Desktop,
    };

    private string Marker => Path.Combine(Paths.DataRoot, "model-load-in-progress.json");

    public void Dispose()
    {
        SpeculationPolicy.UseLaunchEnvironment(null, null);
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    /// <summary>The record names the model while its native load runs, and is gone once the load is through.</summary>
    [Fact]
    public void ALoadIsRecordedWhileItRunsAndForgottenOnceItIsThrough()
    {
        string? seenDuringLoad = null;
        CatalogModel model = SmallModel();
        using AgentAppHost host = HostWith((path, _, _, _) =>
        {
            seenDuringLoad = File.Exists(Marker) ? File.ReadAllText(Marker) : null;
            return new FakeModel(path);
        });
        Install(host, model);

        host.UseModel(model, warmAfterwards: false);

        Assert.NotNull(seenDuringLoad);
        Assert.Contains(model.Id, seenDuringLoad);
        Assert.False(File.Exists(Marker));
        Assert.Equal(AgentAppHost.ModelLoadState.Loaded, host.ModelLoad);
    }

    /// <summary>A load that fails without taking the process down is not a crash to remember.</summary>
    [Fact]
    public void ALoadThatFailsCleanlyIsForgottenAndEndsFailed()
    {
        CatalogModel model = SmallModel();
        using AgentAppHost host = HostWith((_, _, _, _) => throw new InvalidDataException("not a model this build can run"));
        Install(host, model);

        Assert.ThrowsAny<Exception>(() => host.UseModel(model, warmAfterwards: false));

        Assert.False(File.Exists(Marker));
        Assert.Equal(AgentAppHost.ModelLoadState.Failed, host.ModelLoad);
    }

    /// <summary>
    /// The warm-up's first prefill is part of the load: the record stays while it runs,
    /// and goes when it ends.
    /// </summary>
    [Fact]
    public async Task TheWarmUpKeepsTheRecordUntilItEnds()
    {
        CatalogModel model = SmallModel();
        using AgentAppHost host = HostWith((path, _, _, _) => new FakeModel(path));
        Install(host, model);
        var warming = new TaskCompletionSource();
        var release = new TaskCompletionSource();
        host.WarmUpFrames = (_, ct) => BlockedFrames(warming, release, ct);

        host.UseModel(model, warmAfterwards: true);
        await warming.Task.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.True(File.Exists(Marker));
        Assert.Contains(model.Id, File.ReadAllText(Marker));

        release.SetResult();
        await host.StopWarmingThePrefixCacheAndWaitAsync();
        Assert.False(File.Exists(Marker));
    }

    /// <summary>
    /// The system may end an app that is away, and that is no evidence a load crashed it:
    /// leaving the foreground clears this process's record, so the next launch still
    /// loads the model.
    /// </summary>
    [Fact]
    public async Task LeavingTheForegroundForgetsTheWarmUpInProgress()
    {
        CatalogModel model = SmallModel();
        using AgentAppHost host = HostWith((path, _, _, _) => new FakeModel(path));
        Install(host, model);
        var warming = new TaskCompletionSource();
        var release = new TaskCompletionSource();
        host.WarmUpFrames = (_, ct) => BlockedFrames(warming, release, ct);
        host.UseModel(model, warmAfterwards: true);
        await warming.Task.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.True(File.Exists(Marker));

        host.Compute.Close();
        Assert.False(File.Exists(Marker));

        host.Compute.Open();
        release.SetResult();
        await host.StopWarmingThePrefixCacheAndWaitAsync();
    }

    /// <summary>
    /// A record that kept a model from loading at launch keeps doing so at every launch,
    /// until the user chooses a model: quitting does not clear another process's record,
    /// or a model that crashes on load would crash every other launch.
    /// </summary>
    [Fact]
    public void ARecordThatStoppedTheAutoLoadStaysUntilTheUserChoosesAModel()
    {
        CatalogModel model = ModelCatalog.Find("bonsai-2-27b-ptq1-0")!;
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = model.Id;
        settings.Save(chosen);
        var store = new ModelStore(paths.ModelsDirectory);
        Directory.CreateDirectory(store.DirectoryFor(model));
        SparseFileFixture.Create(store.PathFor(model, model.Weights), model.Weights.Bytes);
        File.WriteAllText(Marker, JsonSerializer.Serialize(new { ModelId = model.Id }));

        for (int launch = 0; launch < 3; launch++)
        {
            using var host = new AgentAppHost(paths);
            host.Start();
            Assert.Equal(AgentAppHost.ModelLoadState.Failed, host.ModelLoad);
            Assert.True(File.Exists(Marker), $"launch {launch + 2} lost the record");
        }
    }

    /// <summary>
    /// A message sent while a model loads waits for the load, and is answered by the
    /// model that was loading, instead of finding no model (or a half-loaded one).
    /// </summary>
    [Fact]
    public async Task AMessageSentDuringALoadWaitsForIt()
    {
        CatalogModel model = SmallModel();
        using var loadMayFinish = new ManualResetEventSlim();
        var loading = new TaskCompletionSource();
        using AgentAppHost host = HostWith((path, _, _, _) =>
        {
            loading.TrySetResult();
            loadMayFinish.Wait(TimeSpan.FromSeconds(30));
            return new FakeModel(path);
        });
        Install(host, model);
        host.Start();
        using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");

        Task load = Task.Run(() => host.UseModel(model, warmAfterwards: false));
        await loading.Task.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal(AgentAppHost.ModelLoadState.Loading, host.ModelLoad);

        string session = (await (await client.PostAsync("/api/sessions?conversation=new", null))
            .Content.ReadFromJsonAsync<JsonElement>()).GetProperty("sessionId").GetString()!;
        using var body = JsonContent.Create(new
        {
            sessionId = session,
            messages = new[] { new { role = "user", content = "hi" } },
            maxTokens = 4,
        });
        Task<HttpResponseMessage> chat = client.PostAsync("/api/chat", body);
        await Task.Delay(1500);
        loadMayFinish.Set();
        await load.WaitAsync(TimeSpan.FromSeconds(30));
        string stream = await (await chat.WaitAsync(TimeSpan.FromSeconds(60))).Content.ReadAsStringAsync();

        Assert.Equal(AgentAppHost.ModelLoadState.Loaded, host.ModelLoad);
        Assert.DoesNotContain("No model loaded", stream);
    }

    private static async IAsyncEnumerable<object> BlockedFrames(
        TaskCompletionSource warming, TaskCompletionSource release, [EnumeratorCancellation] CancellationToken ct)
    {
        warming.TrySetResult();
        await release.Task.WaitAsync(ct);
        yield break;
    }

    private AgentAppHost HostWith(Func<string, BackendType, ITensorParallelGroup?, string?, ModelBase> factory)
    {
        SpeculationPolicy.UseLaunchEnvironment(null, null);
        var service = new ModelService(NullLogger<ModelService>.Instance, (path, backend, group, draft) => factory(path, backend, group, draft));
        return new AgentAppHost(Paths, modelService: service, backends: new[] { new BackendOption("cpu", "CPU") });
    }

    private static CatalogModel SmallModel()
    {
        CatalogModel original = ModelCatalog.Find("qwen3.5-9b-iq4xs")!;
        return original with { Files = new[] { original.Weights with { Bytes = 32 } }, KvCacheDtype = "f16" };
    }

    private static void Install(AgentAppHost host, CatalogModel model)
    {
        Directory.CreateDirectory(host.Models.DirectoryFor(model));
        using var writer = new BinaryWriter(File.Create(host.Models.PathFor(model, model.Weights)));
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(0UL);
        writer.Write(0UL);
        writer.Write(new byte[8]);
    }

    private sealed class FakeModel : ModelBase
    {
        public FakeModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "llama" };
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
