// Runs the real TensorAgent host on desktop for reproducible HTTP/WebUI validation.
// Evidence and mutable app data belong in an ignored artifacts/ or docs/validation/ root.
using System.Net.Http.Json;
using System.Text.Json;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.Server;

if (args.Contains("--help"))
{
    Console.WriteLine("TensorAgentHost --root <evidence-dir> --skills <skills-dir> [--catalog-model <installed-id> | --weights <gguf>] [--mmproj <gguf>] [--backend ggml_metal] [--port 0] [--network] [--unconfined] [--context 32768] [--max-tokens 4096] [--web-root <dir>] [--stop-file <file>]");
    Console.WriteLine("  iOS path on a desktop: --in-process --python-root <CPython prefix> [--phone] [--settings <a TensorAgent settings.json>]");
    Console.WriteLine("    --in-process  the phone's execution: embedded Python and JavaScript in this process, no program launches");
    Console.WriteLine("    --phone       the phone's device class (its engine memory budget and settings defaults), 12 GB tier");
    Console.WriteLine("    --settings    take every setting from that file (a copy of a device's settings.json) instead of the overrides above");
    Console.WriteLine("  --trace-log     also print the host's Information log (chat rounds, skill tool calls) for eng/validation/skill-install-trace.py;");
    Console.WriteLine("                  it holds prompts and outputs, so its file belongs in ignored artifacts/ only");
    return 0;
}

var values = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (args[i] is "--network" or "--unconfined" or "--in-process" or "--phone" or "--trace-log")
        values[args[i]] = "true";
    else if (i + 1 < args.Length && args[i].StartsWith("--", StringComparison.Ordinal))
        values[args[i]] = args[++i];
    else
        throw new ArgumentException($"Expected an option and value, got '{args[i]}'.");
}
string Required(string option) => values.TryGetValue(option, out string? value)
    ? Path.GetFullPath(value) : throw new ArgumentException($"Missing {option}. Run --help for usage.");
int Number(string option, int fallback) => values.TryGetValue(option, out string? value) ? int.Parse(value) : fallback;

string root = Required("--root");
CatalogModel? catalogModel = values.TryGetValue("--catalog-model", out string? catalogId)
    ? ModelCatalog.Find(catalogId) ?? throw new ArgumentException($"Unknown catalog model '{catalogId}'.")
    : null;
if (catalogModel is not null && (values.ContainsKey("--weights") || values.ContainsKey("--mmproj")))
    throw new ArgumentException("--catalog-model uses the app's model store; it cannot be combined with --weights or --mmproj.");
bool phone = values.ContainsKey("--phone");
bool inProcess = values.ContainsKey("--in-process");
if (values.ContainsKey("--python-root") && !inProcess)
    throw new ArgumentException("--python-root is the embedded runtime of --in-process.");
var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"))
{
    BundledSkillsDirectory = Required("--skills"),
    DeviceMemoryGB = phone ? 12 : 32,
    DeviceClass = phone ? DeviceClass.Phone : DeviceClass.Desktop,
    // The iOS execution path (InProcessShellBackend + EmbeddedPython) on this machine:
    // the same SkillHostProfile as a phone, so a skill's verdicts and the shell's
    // abilities are the phone's. The CPython prefix stands in for the app bundle's.
    ExecutionMode = inProcess ? AgentExecutionMode.InProcess : AgentExecutionMode.Auto,
    PythonRuntimeDirectory = inProcess ? Required("--python-root") : string.Empty,
};
paths.EnsureCreated();
var settingsStore = new SettingsStore(paths.SettingsFile, phone ? null : AppSettings.DesktopDefaults);
AppSettings settings;
if (values.ContainsKey("--settings"))
{
    // A device's own settings, as they are: the reproduction is of that configuration.
    settings = new SettingsStore(Required("--settings"), phone ? null : AppSettings.DesktopDefaults).Load();
    settings.SelectedModelId = null;
}
else
{
    settings = settingsStore.Load();
    settings.SelectedModelId = null;
    settings.AllowCodeExecution = true;
    settings.AllowUnconfinedExecution = values.ContainsKey("--unconfined");
    settings.AllowNetwork = values.ContainsKey("--network");
    settings.NetworkHosts.Clear();
    settings.ContextLength = Number("--context", 32768);
    settings.MaxTokens = Number("--max-tokens", 4096);
    settings.ToolTimeoutSeconds = 600;
    settings.SpeculativeDecoding = false;
}
settingsStore.Save(settings);
// This launcher loads an arbitrary local GGUF through the HTTP route, without a
// catalog entry whose UseModel path would normally apply the context budget.
if (catalogModel is null)
{
    Environment.SetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable, settings.ContextLength.ToString());
    Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable, settings.KvCacheDtype);
    TensorSharp.Models.KvCacheDtypeConfig.ConfigureFromEnvironment();
}
// A local GGUF bypasses UseModel's catalog policy, so apply the same desktop
// cache policy explicitly rather than inheriting a phone budget from the shell.
foreach (string variable in new[]
{
    EngineMemoryPolicy.KvInitialTokensVariable,
    EngineMemoryPolicy.KvGenerationReserveMaxVariable,
    EngineMemoryPolicy.KvHolderPoolMaxVariable,
    EngineMemoryPolicy.RetainedFusedCacheMaxVariable,
})
    Environment.SetEnvironmentVariable(variable, null);
SpeculationPolicy.PrepareLoad(settings, null);

string backend = values.GetValueOrDefault("--backend", OperatingSystem.IsMacOS() ? "ggml_metal" : "ggml_cpu");
string? webRoot = values.GetValueOrDefault("--web-root");
if (webRoot is null)
{
    string candidate = Path.Combine(Directory.GetCurrentDirectory(), "TensorAgent", "src", "TensorAgent.Maui", "wwwroot");
    if (Directory.Exists(candidate))
        webRoot = candidate;
}
bool traceLog = values.ContainsKey("--trace-log");
using var rankingLogging = LoggerFactory.Create(builder => builder.AddProvider(new RankingLogProvider(traceLog)));
using var host = new AgentAppHost(paths,
    loggerFactory: rankingLogging,
    webRoot: webRoot,
    port: Number("--port", 0),
    backends: new[] { new BackendOption(backend, backend) });
if (values.TryGetValue("--weights", out string? weights))
{
    weights = Path.GetFullPath(weights);
    if (!File.Exists(weights))
        throw new FileNotFoundException("Model does not exist.", weights);
    string projector = values.TryGetValue("--mmproj", out string? mmproj) ? Path.GetFullPath(mmproj) : string.Empty;
    if (projector.Length > 0 && !File.Exists(projector))
        throw new FileNotFoundException("Projector does not exist.", projector);
    host.Options.RepointHostedModel(weights, projector);
}
host.Server.Start();
string cookie = $"{LoopbackServer.TokenCookie}={host.Server.Token}";
var loadClock = Stopwatch.StartNew();
if (catalogModel is not null)
{
    if (!host.Catalog.Any(model => model.Id == catalogModel.Id))
        throw new InvalidOperationException($"{catalogModel.Id} is not offered on a {paths.DeviceMemoryGB} GB device.");
    host.UseModel(catalogModel, warmAfterwards: false);
}
else if (weights is not null)
{
    using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl), Timeout = TimeSpan.FromMinutes(10) };
    client.DefaultRequestHeaders.Add("Cookie", cookie);
    using HttpResponseMessage response = await client.PostAsJsonAsync("/api/models/load", new
    {
        model = Path.GetFileName(weights),
        backend,
    });
    string payload = await response.Content.ReadAsStringAsync();
    if (!response.IsSuccessStatusCode)
        throw new InvalidOperationException($"Model load failed ({response.StatusCode}): {payload}");
    Console.WriteLine(payload);
}
loadClock.Stop();
string connection = JsonSerializer.Serialize(new
{
    baseUrl = host.Server.BaseUrl,
    dataRoot = paths.DataRoot,
    cacheRoot = paths.CacheRoot,
    entryUrl = host.Server.EntryUrl,
    cookie,
    executionBackend = host.Backend.Name,
    engine = host.DescribeEngine(),
    model = catalogModel is null ? weights : host.Models.PathFor(catalogModel, catalogModel.Weights),
    catalogModel = catalogModel?.Id,
    eligibleCatalog = host.Catalog.Select(model => model.Id),
    backend,
    contextLength = settings.ContextLength,
    deviceClass = paths.DeviceClass.ToString(),
    kvCacheDtype = settings.KvCacheDtype,
    speculativeDecoding = settings.SpeculativeDecoding,
    loadSeconds = loadClock.Elapsed.TotalSeconds,
    projector = catalogModel is null ? values.GetValueOrDefault("--mmproj")
        : host.Models.CompanionPath(catalogModel, CatalogFileRole.Projector),
    cacheBudget = new
    {
        initialTokens = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable),
        generationReserve = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable),
        retainedCaches = Environment.GetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable),
        holderPool = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable),
    },
});
await File.WriteAllTextAsync(Path.Combine(root, "connection.json"), connection);
Console.WriteLine(connection);
using var stop = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
try
{
    if (values.TryGetValue("--stop-file", out string? stopFile))
    {
        while (!File.Exists(stopFile))
            await Task.Delay(250, stop.Token);
    }
    else
        await Task.Delay(Timeout.Infinite, stop.Token);
}
catch (OperationCanceledException) { }
return 0;

// By default only bounded, non-sensitive ranking diagnostics: source text and
// authentication material are never included. --trace-log adds the host's Information
// entries in the console logger's layout ("info: Category[id]" then indented lines), which
// is what skill-install-trace.py reads; those entries hold prompts and model output.
sealed class RankingLogProvider(bool traceLog) : ILoggerProvider
{
    private static readonly object Gate = new();
    public ILogger CreateLogger(string categoryName) => new RankingLogger(categoryName, traceLog);
    public void Dispose() { }
    sealed class RankingLogger(string category, bool traceLog) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            string message = formatter(state, exception);
            if (message.StartsWith("Knowledge rerank ", StringComparison.Ordinal))
            {
                Console.WriteLine(message);
                return;
            }
            if (!traceLog || logLevel < LogLevel.Information)
                return;
            string level = logLevel switch
            {
                LogLevel.Information => "info",
                LogLevel.Warning => "warn",
                LogLevel.Error => "fail",
                _ => "crit",
            };
            string body = string.Join("\n      ", message.Replace("\r", string.Empty).Split('\n'));
            lock (Gate)
                Console.WriteLine($"{level}: {category}[{eventId.Id}]\n      {body}");
        }
    }
}
