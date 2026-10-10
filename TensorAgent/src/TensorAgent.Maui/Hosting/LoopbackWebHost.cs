// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#if IOS || MACCATALYST
using Foundation;
#endif
using Microsoft.Extensions.Logging;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Python;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace TensorAgent.Maui.Hosting;

/// <summary>
/// The platform half of the app's host: where the files live on this device, and
/// nothing else.
///
/// <para>
/// Everything the server actually does — the routes, the chat pipeline, the code
/// runner, the model catalog — is <see cref="AgentAppHost"/> in the platform-neutral
/// project, so it can be tested on a development machine. What is left here is the
/// part that genuinely cannot be: which directory the system gives an app for data it
/// must back up, which for files it can re-fetch, how much memory the device has, and
/// where the bundle put the Web UI.
/// </para>
/// <para>
/// Three heads build this file. The phone keeps the phone's lifecycle helpers (the GPU
/// is withdrawn from a background app, a suspended app loses its listening socket,
/// shares arrive through an App Group) and runs with the engine budget measured on a
/// 12 GB iPhone. The Mac and Windows apps have none of those constraints: nothing is
/// suspended when the window loses focus, code runs as real processes the way the
/// desktop hosts run it, and the engine keeps its desktop defaults.
/// </para>
/// </summary>
public sealed class LoopbackWebHost : IDisposable
{
    private readonly AgentAppHost _host;
#if IOS
    private readonly Platforms.iOS.BackgroundDownloads _backgroundDownloads;
    private readonly Platforms.iOS.BackgroundGeneration _backgroundGeneration;
    private readonly Platforms.iOS.ShareInbox _shareInbox;
    private readonly Platforms.iOS.LoopbackLifecycle _loopbackLifecycle;
#else
    private readonly Services.DesktopActivity _desktopActivity;
#endif

    /// <param name="webRoot">The app's own Web UI, TensorAgent.Maui/wwwroot, as bundled under webui/.</param>
    /// <param name="loggerFactory">Where the engine logs; console output is what <c>simctl launch --console</c> shows.</param>
    /// <param name="python">The embedded interpreter, when this build has one.</param>
    /// <param name="javaScript">The embedded JavaScript engine, when this build has one.</param>
    public LoopbackWebHost(
        string webRoot,
        ILoggerFactory? loggerFactory = null,
        IPythonRuntime? python = null,
        IJavaScriptRuntime? javaScript = null)
    {
        WebRoot = Path.GetFullPath(webRoot);
        string index = Path.Combine(WebRoot, "index.html");
        if (!File.Exists(index))
        {
            // A bundle with no page is a build problem, and a blank WebView is the
            // worst possible way to learn about it.
            throw new FileNotFoundException(
                $"The bundled Web UI is missing: expected {index}. TensorAgent.Maui.csproj links "
                + "TensorAgent/src/TensorAgent.Maui/wwwroot/** into the bundle as webui/.", index);
        }

        // The app, and only the app, installs the process-exit net: ggml-metal's
        // device is a C++ static whose destructor asserts every residency set was
        // handed back, and a user who closes the app without unloading first — which
        // is every user — would otherwise abort instead of exiting.
        AgentAppHost.ReleaseTheEngineWhenTheProcessExits();

        // What the page offers is what this build can actually run. The simulator's
        // slice of the engine has no Metal, and a page whose default backend does not
        // exist puts the user one tap from a load that fails.
        _host = new AgentAppHost(
            DevicePaths(), WebRoot, loggerFactory, python, javaScript,
            backends: BackendsFor(Compute.Selection));

        // Which language the interface came up in, and from what: the line a person reads
        // when the app shows a language they did not expect.
        Console.WriteLine($"TensorAgent: interface language {Core.Localization.Loc.Language.Tag} " +
            $"(saved choice '{_host.Settings.Load().UiLanguage}', system {string.Join(",", Core.Localization.Loc.SystemLanguages())})");
#if IOS
        // The one part of a download that needs iOS: staying alive for a while after
        // the user leaves the app, and picking itself up when they come back.
        _backgroundDownloads = new Platforms.iOS.BackgroundDownloads(_host.Downloads);
        // And the same for a generation, which needs it more: a turn takes a minute and
        // the display sleeps in less than that.
        _backgroundGeneration = new Platforms.iOS.BackgroundGeneration(_host);
        _shareInbox = new Platforms.iOS.ShareInbox(_host);
        _loopbackLifecycle = new Platforms.iOS.LoopbackLifecycle(_host);
        // The share extension speaks the language the user chose here; Settings writes it
        // again whenever it changes. Written at every launch so an install that predates the
        // file, or an App Group that was reset, catches up.
        Platforms.iOS.SharedContainer.WriteLanguageChoice(_host.Settings.Load().UiLanguage);
#else
        // A desktop keeps working behind other windows; it only has to say that it is,
        // or App Nap and idle sleep slow the model down or stop it. See DesktopActivity.
        _desktopActivity = new Services.DesktopActivity(_host);
#endif
    }

    /// <summary>
    /// The backend list the page shows, best first: the GPU backend
    /// <see cref="Compute"/> chose, then the CPU. A GPU backend that cannot
    /// initialise here is not offered at all, because offering a backend that cannot
    /// initialise is worse than offering one fewer.
    /// </summary>
    private static IReadOnlyList<BackendOption> BackendsFor(ComputeSelection selection)
        => selection.Backend switch
        {
            BackendType.GgmlMetal => new[] { new BackendOption("ggml_metal", "GPU (Metal)"), new BackendOption("ggml_cpu", "CPU") },
            BackendType.GgmlCuda => new[] { new BackendOption("ggml_cuda", "GPU (CUDA)"), new BackendOption("ggml_cpu", "CPU") },
            BackendType.GgmlVulkan => new[] { new BackendOption("ggml_vulkan", "GPU (Vulkan)"), new BackendOption("ggml_cpu", "CPU") },
            _ => new[] { new BackendOption("ggml_cpu", "CPU") },
        };

    /// <summary>Where the Web UI is served from, for the startup log.</summary>
    public string WebRoot { get; }

    public int Port => _host.Server.Port;
    public string BaseUrl => _host.Server.BaseUrl;
    public string EntryUrl => _host.EntryUrl;
    public string Token => _host.Server.Token;

    /// <summary>The assembled application, for the native pages that drive it directly.</summary>
    public AgentAppHost App => _host;

    public void Start()
    {
        // Local validation may use a different quantization from the download
        // catalog. Keep its real file identity and avoid copying multi-GB weights
        // or pretending it passed the catalog's size/hash verification.
        if (ValidationRoot() is not null
            && Environment.GetEnvironmentVariable("TENSORAGENT_VALIDATION_WEIGHTS") is { Length: > 0 } weights)
        {
            weights = Path.GetFullPath(weights);
            if (!File.Exists(weights))
                throw new FileNotFoundException("Validation model does not exist.", weights);
            string projector = Environment.GetEnvironmentVariable("TENSORAGENT_VALIDATION_MMPROJ") ?? string.Empty;
            if (projector.Length > 0)
            {
                projector = Path.GetFullPath(projector);
                if (!File.Exists(projector))
                    throw new FileNotFoundException("Validation projector does not exist.", projector);
            }
            _host.Options.RepointHostedModel(weights, projector);
        }
        // A validation run can tap a model's action on the Models page from a script: the
        // GUI cannot be driven while the screen is locked, and a switch made any other way
        // skips the page's own path (Select, Refresh, the way back to the chat). Mapped
        // before the server starts because its route table is not safe to grow while it
        // serves.
        if (ValidationRoot() is not null)
            MapValidationRoutes();
        _running = this;
        _host.Start();
        // Windows GUI launches have no attached console. An explicitly isolated
        // validation run still needs its loopback address to exercise the same host
        // the WebView uses, without touching the user's ordinary application data.
        if (ValidationRoot() is { } validationRoot)
        {
            Directory.CreateDirectory(validationRoot);
            File.WriteAllText(Path.Combine(validationRoot, "connection.json"),
                System.Text.Json.JsonSerializer.Serialize(new
                {
                    baseUrl = BaseUrl,
                    entryUrl = EntryUrl,
                    cookie = $"{LoopbackServer.TokenCookie}={Token}",
                    processId = Environment.ProcessId,
                    executionBackend = _host.Backend.Name,
                    dataRoot = _host.Paths.DataRoot,
                    cacheRoot = _host.Paths.CacheRoot,
                }));
        }
#if IOS
        _shareInbox.Start();
#endif
    }

    /// <summary>
    /// What a validation script's "tap" lands on: the Models page sets it to its own action
    /// handler. Null until that page exists.
    /// </summary>
    internal static Func<string, Task<string>>? ValidationTapModel { get; set; }

    private void MapValidationRoutes()
    {
        // POST /api/agent/validation/tap-model {"id": "<catalog id>"}: press that model's
        // action button, on the main thread, exactly as a click does. Answers once the tap
        // is delivered; the load it starts is followed through /api/agent/engine.
        _host.Server.MapPost("/api/agent/validation/tap-model", async (request, ct) =>
        {
            System.Text.Json.JsonElement body = await request.ReadJsonAsync(ct);
            string id = body.TryGetProperty("id", out var value) ? value.GetString() ?? string.Empty : string.Empty;
            Func<string, Task<string>>? tap = ValidationTapModel;
            if (tap is null)
                return LoopbackResponse.Json(new { ok = false, error = "the Models page does not exist yet" }, 409);
            string outcome = await MainThread.InvokeOnMainThreadAsync(() => tap(id));
            return LoopbackResponse.Json(new { ok = outcome == "tapped", outcome, id });
        });
    }

    /// <summary>The host the app is running, while it is running; see <see cref="ShutDownForTermination"/>.</summary>
    private static LoopbackWebHost? _running;

    /// <summary>
    /// Shut the running host down in its own safe order and then hand the engine back, for
    /// an app that is being quit.
    ///
    /// <para>The host is a DI singleton that nothing disposes, and the process-exit net
    /// (<see cref="AgentAppHost.ReleaseTheEngineWhenTheProcessExits"/>) never fires on a Mac:
    /// quitting ends in AppKit's <c>exit()</c>, and Mono raises ProcessExit only for a managed
    /// shutdown. So every quit after the GPU had done any work - a reply, a warm-up, a clip -
    /// aborted on <c>GGML_ASSERT([rsets-&gt;data count] == 0)</c> in ggml-metal's static
    /// destructor. A turn still running is stopped and waited for first, as on any
    /// shutdown, because releasing the model under a GPU graph is a crash too.</para>
    /// </summary>
    public static void ShutDownForTermination()
    {
        LoopbackWebHost? running = Interlocked.Exchange(ref _running, null);
        // First, because the quit can end in _exit below, which skips Dispose: a model
        // still loading or warming up when the user quits did not crash the app, and the
        // next launch must load it as usual (see AgentAppHost.LoadSelectedModelInBackground).
        running?._host.ForgetModelLoadInProgress();
        // Mac Catalyst gives applicationWillTerminate about five seconds: past that,
        // UIKitMacHelper's lifecycle watchdog calls exit() itself. A clip mid-step takes
        // longer than that to stop (MEASURED: a quit three steps into a 22-frame clip was
        // ended by the watchdog at 5.7 s and aborted in ggml-metal's destructor), and
        // neither way out of that is safe: releasing the model under the running graph is a
        // crash, and exit() runs the destructor on buffers still registered. So the work
        // gets most of the budget to stop, and if it has not, the process leaves without
        // the C++ static destructors - the kernel reclaims its memory and its GPU work.
        if (running is not null && !running._host.StopWorkWithin(TimeSpan.FromSeconds(3)))
        {
            Console.WriteLine("TensorAgent: still generating when the app was quit; leaving without releasing the model");
            Console.Out.Flush();
            LeaveNow(0);
        }
        running?.Dispose();
        AgentAppHost.ReleaseTheEngine();
    }

    /// <summary>End the process without running exit()'s handlers (POSIX <c>_exit</c>).</summary>
    [System.Runtime.InteropServices.DllImport("libSystem.dylib", EntryPoint = "_exit")]
    private static extern void LeaveNow(int status);

    public void Dispose()
    {
        Interlocked.CompareExchange(ref _running, null, this);
#if IOS
        _loopbackLifecycle.Dispose();
        _shareInbox.Dispose();
        _backgroundGeneration.Dispose();
        _backgroundDownloads.Dispose();
#else
        _desktopActivity.Dispose();
#endif
        _host.Dispose();
    }

    /// <summary>
    /// The two directories iOS gives an app, used for what each is actually for.
    ///
    /// <para>
    /// Library/Application Support is backed up to iCloud and restored onto a new
    /// device, which is right for conversations, settings and installed skills and
    /// badly wrong for model weights: a single entry in this catalog is five to ten
    /// gigabytes, it is a byte-identical copy of a public file, and pushing it into
    /// the user's iCloud quota would be indefensible. Weights therefore go to Caches,
    /// which is excluded from backup — and which the system may purge under storage
    /// pressure, so the app has to treat a missing model as "download it again"
    /// rather than as an error. <see cref="Core.Catalog.ModelStore"/> already does.
    /// </para>
    /// </summary>
    /// <summary>
    /// Where a durable log belongs on this device. Exposed because logging is
    /// configured before the host exists, and the two must agree on the directory.
    /// </summary>
    public static string DeviceLogsDirectory() => DevicePaths().LogsDirectory;

    private static string? ValidationRoot()
    {
        // Explicit opt-in, including Release so measured app performance uses
        // the shipped configuration. No validation variables affect normal runs.
        return Environment.GetEnvironmentVariable("TENSORAGENT_VALIDATION_ROOT") is { Length: > 0 } root
            ? Path.GetFullPath(root) : null;
    }

    private static AgentPaths DevicePaths()
    {
#if IOS || MACCATALYST
        // On a Mac these are ~/Library/Application Support and ~/Library/Caches (the app
        // is not sandboxed; see Platforms/MacCatalyst/Info.plist), and the split does the
        // same job: Time Machine skips Caches, so gigabytes of re-downloadable weights
        // stay out of the user's backups.
        string data = NSSearchPath.GetDirectories(NSSearchPathDirectory.ApplicationSupportDirectory, NSSearchPathDomain.User, true)[0];
        string cache = NSSearchPath.GetDirectories(NSSearchPathDirectory.CachesDirectory, NSSearchPathDomain.User, true)[0];
        string dataRoot = Path.Combine(data, "TensorAgent");
        string cacheRoot = Path.Combine(cache, "TensorAgent");
#else
        // %LOCALAPPDATA%, not the roaming profile: a roaming profile is copied between
        // machines at sign-in, which is no place for model weights or scratch space.
        string local = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TensorAgent");
        string dataRoot = Path.Combine(local, "Data");
        string cacheRoot = Path.Combine(local, "Cache");
#endif

        if (ValidationRoot() is { } validationRoot)
        {
            dataRoot = Path.Combine(validationRoot, "Data");
            cacheRoot = Path.Combine(validationRoot, "Cache");
        }

        return new AgentPaths(dataRoot, cacheRoot)
        {
            DeviceMemoryGB = DeviceMemoryGigabytes(),
            BundledSkillsDirectory = Path.Combine(AppBundle.ResourceDirectory, "skills"),
#if IOS
            // The interpreter's standard library is staged into the bundle beside the
            // Python framework, which is where PyConfig's module search paths point.
            PythonRuntimeDirectory = NSBundle.MainBundle.BundlePath,
            SharedInboxDirectory = Platforms.iOS.SharedContainer.InboxDirectory(),
#else
            // A desktop has memory to spare: the engine's own defaults size the caches,
            // not the budget measured against a phone's jetsam limit, and a first launch
            // starts from the desktop settings (AppSettings.DesktopDefaults).
            DeviceClass = DeviceClass.Desktop,
#endif
        };
    }

    /// <summary>
    /// Physical memory in whole gigabytes, which decides which catalog entries the
    /// app will even offer. It is deliberately the DEVICE's memory and not the app's
    /// jetsam budget: the budget is roughly two thirds of it, and the catalog's
    /// per-entry minimum is already written against the device figure.
    /// </summary>
    private static int DeviceMemoryGigabytes()
    {
        long bytes = Services.DeviceState.PhysicalMemoryBytes();

        // ModelCatalog.DeviceMemoryTier is the rule, and this used to be a second,
        // quieter copy of it that disagreed. iOS reports a little UNDER the marketing
        // number and this divided by 1024^3 on top of that, so a 12 GB iPhone 17 Pro Max
        // reporting ~11.6e9 bytes came out as 10.8 -> 11. Every catalog entry starts at
        // 12, so ForDevice filtered out ALL OF THEM and the app offered no models at
        // all -- on the exact device it is built for. Measured on hardware: the catalog
        // came back empty. The helper does the same job in decimal with the tolerance
        // that under-reporting needs, and is what the tests are written against.
        int tier = ModelCatalog.DeviceMemoryTier(bytes);
        // The tier decides what is OFFERED; the per-process headroom is what decides
        // whether a load survives, and they are different numbers. Both are logged
        // because a jetsam kill leaves no message of its own -- see
        // DeviceState.DescribeMemory and EngineMemoryPolicy.
        Console.WriteLine(
            $"TensorAgent: physical memory {bytes / 1_000_000_000.0:0.00} GB -> catalog tier {tier} GB " +
            $"({Services.DeviceState.DescribeMemory()})");
        return tier;
    }
}
