// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Python;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Sharing;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Runtime.Speculative;
using TensorAgent.Sharing;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Where everything the app runs is assembled and wired together.
///
/// <para>
/// The desktop server does this in <c>Program.cs</c> with a dependency-injection
/// container and a hundred command-line flags. A phone has neither, so the same
/// object graph is built here from a directory and a settings file: the model
/// service and its session manager, the skills registry, the code runner over the
/// in-process shell, and the loopback server that puts the Web UI in front of them.
/// </para>
/// <para>
/// It deliberately lives in the platform-neutral project rather than in the iOS
/// head. Everything here is ordinary .NET, so it can be started, driven over real
/// HTTP and torn down by a test on a development machine — which is the only way the
/// wiring gets checked at all, since an iOS app cannot be unit-tested from a
/// terminal.
/// </para>
/// </summary>
public sealed class AgentAppHost : IDisposable
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly List<IDisposable> _owned = new();
    private readonly object _shareClaimsLock = new();
    private readonly Dictionary<string, string> _shareClaims = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ClaimedShare> _shareReleaseRetries = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _shareDrain = new(1, 1);
    private readonly object _shareOwnershipLock = new();
    private readonly HashSet<string> _sharesBeingSent = new(StringComparer.Ordinal);

    /// <param name="paths">Where this installation keeps its files.</param>
    /// <param name="webRoot">The bundled copy of the Web UI, or null to serve no static files.</param>
    /// <param name="loggerFactory">Where the engine and the code runner log.</param>
    /// <param name="python">An interpreter to use instead of the default; null discovers one.</param>
    /// <param name="javaScript">An engine to use instead of the default; null discovers one.</param>
    /// <param name="port">A fixed loopback port, or 0 to take a free one.</param>
    /// <param name="backends">What this build can run, best first; null offers Metal then CPU.</param>
    /// <param name="modelService">
    /// The engine service to own, or null to build one. The app never passes one; a
    /// test does, because the shutdown order this class exists to enforce — stop
    /// serving, wait for the engine, then free the weights — is otherwise invisible
    /// from outside, and getting it wrong costs a segmentation fault rather than a
    /// failed assertion.
    /// </param>
    /// <param name="installer">
    /// A package hook to use instead of the PyPI wheel installer; null builds the app's
    /// normal installer. Tests inject one to verify the complete model-command route
    /// without depending on the public package index.
    /// </param>
    public AgentAppHost(
        AgentPaths paths,
        string? webRoot = null,
        ILoggerFactory? loggerFactory = null,
        IPythonRuntime? python = null,
        IJavaScriptRuntime? javaScript = null,
        int port = 0,
        IReadOnlyList<BackendOption>? backends = null,
        ModelService? modelService = null,
        IInstallHook? installer = null)
    {
        Paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
        paths.EnsureCreated();

        Settings = new SettingsStore(paths.SettingsFile);
        AppSettings settings = Settings.Load();

        Models = new ModelStore(paths.ModelsDirectory);
        // Weights whose catalog entry is gone -- the previous quantization of an entry
        // that now points at a different file. Nothing else can reach them: the Models
        // list is built from the catalog, so a directory no entry claims has no row and
        // no delete button, and it is gigabytes. Swept once per launch, before anything
        // reads the store.
        Models.SweepOrphanedModels();
        // And the checkpoints of models the catalog no longer has: a directory no
        // entry claims has no delete button either.
        PrefixCheckpointFileStore.SweepOrphans(
            Path.Combine(Paths.CacheRoot, "prefix-cache"),
            id => ModelCatalog.Find(id) is not null,
            HostLog);
        // The downloads belong to the APP, not to the model list: a five-gigabyte
        // transfer must not end because the user went back to the chat. See
        // ModelDownloadManager.
        Downloads = new ModelDownloadManager(Models, _loggerFactory.CreateLogger("TensorAgent.Downloads"));
        Conversations = new ConversationStore(paths.ConversationsDirectory);
        Catalog = ModelCatalog.ForDevice(paths.DeviceMemoryGB);

        // The two switches the user actually sees. They are read here, at startup,
        // and again on every launch the runner builds, so flipping one in the app
        // takes effect on the next command rather than on the next restart.
        CodeExec = new CodeExecOptions
        {
            Enabled = settings.AllowCodeExecution,
            AllowNetwork = settings.AllowNetwork,
            AllowInstall = settings.AllowNetwork,
            ScratchDirectory = paths.ScratchDirectory,
            Timeout = TimeSpan.FromSeconds(Math.Clamp(settings.ToolTimeoutSeconds, 5, 600)),
            InstallTimeout = TimeSpan.FromSeconds(Math.Clamp(settings.ToolTimeoutSeconds, 5, 600)),
            // Without this the runner hands back the artifact's ABSOLUTE PATH ON DISK
            // instead of a URL, and everything downstream faithfully carries it: the
            // model is told "tell the user where they are on disk", the page renders
            // /var/mobile/Containers/Data/.../report.pdf as an ordinary link, and tapping
            // it asks the loopback server for a path it serves nothing at --
            // {"error":"not found"}. Mapping the artifact route (below) fixed the half of
            // this that was a missing endpoint; this is the half that was a link pointing
            // somewhere else entirely. TensorSharp.Server sets the same prefix in its
            // Program.cs, which is why the desktop never had either symptom.
            ArtifactUriPrefix = WebUiChatService.DefaultArtifactUriPrefix,
        };

        Workspaces = new SessionWorkspaceManager(paths.ScratchDirectory, _loggerFactory.CreateLogger("TensorAgent.Workspaces"));
        Workspaces.SweepOrphans();

        // Both runtimes are discovered rather than required. Each reports its own
        // availability with a reason, the shell repeats that reason when a model tries
        // to use one, and the engine line says so before anything is attempted — so a
        // build without an interpreter is a build that says it has no interpreter,
        // never one that fails halfway through a script.
        Python = python ?? Discover(() => new EmbeddedPython(paths.PythonRuntimeDirectory));
        JavaScript = javaScript ?? Discover(() => new JavaScriptCoreEngine());
        ConfigureCodeEnvironment(Python, JavaScript);
        // The installer's own policy answers "can this app install anything at all",
        // which is the user's network switch; each individual install is re-checked
        // against the policy of the launch that asked for it.
        Installer = installer ?? new WheelInstaller(new ExecutionPolicy(
            AllowScripts: settings.AllowCodeExecution,
            AllowNetwork: settings.AllowNetwork,
            WorkRoot: paths.ScratchDirectory,
            ReadableRoots: Array.Empty<string>(),
            TempRoot: paths.ScratchDirectory)
        {
            NetworkHosts = settings.NetworkHosts,
        }, pythonVersion: Python?.Version);
        // Do not expose the raw hook inside the model's shell. The ShellRunner bridge
        // below is the sole install path and applies validation, the package allow-list,
        // the session ledger, target directory and timeout. A nested `sh -c 'pip ...'`
        // must not step around those terms.
        Backend = new InProcessShellBackend(
            Python, JavaScript, installer: null, networkHosts: settings.NetworkHosts);
        Artifacts = new CodeArtifactStore(paths.ArtifactsDirectory);
        // ShellRunner intercepts every recognised `pip install` before the remaining
        // command reaches Backend. Supplying the bridge is therefore essential: its
        // desktop default tries to launch a real `python -m pip`, while this host has no
        // processes and deliberately does not stage pip into embedded CPython.
        // The bundle is asked before the index: lxml, numpy and Pillow are compiled into
        // the app, and `pip install lxml` from a model must be told so rather than
        // refused as "compiled code" by a lookup that never needed to happen.
        var packageInstaller = new InstallHookPackageInstaller(
            Installer, CodeExec, () => Backend.NetworkHosts,
            () => Python?.BundledDistributions ?? Array.Empty<BundledDistribution>());
        Backend.HostPerformsInstalls = packageInstaller.CanInstall;
        ShellRunner runner = new(
            CodeExec,
            _loggerFactory.CreateLogger("TensorAgent.CodeExec"),
            Artifacts,
            backend: Backend,
            installer: packageInstaller);
        // Always built, never conditional on the switch. ShellRunner.CanRun reads
        // CodeExec.Enabled every time it is asked, and the request planner offers the
        // code tools only for a runner that says it can run -- so a runner that exists
        // and answers "no" is the same refusal as no runner at all, and it is one the
        // user can lift from the Settings screen without relaunching the app. Building
        // it conditionally is what made both sandbox switches take effect "next time
        // TensorAgent starts", which on a phone reads as a switch that does nothing.
        CodeRunner = new CodeRunnerAdapter(
            runner,
            CodeExec,
            packageInstallInstructions: InstallHookPackageInstaller.ModelInstallInstructions,
            networkExecutionInstructions: InstallHookPackageInstaller.ModelExecutionInstructions,
            // No host gate. The guidance used to be shown only when one particular
            // finance host was allow-listed, because it WAS about that host; now that
            // every sentence of it is true of any request, a narrowed allow-list must
            // not be what decides whether the model is told how to write a heredoc.
            networkHosts: () => Backend.NetworkHosts,
            providedPackagesInstructions: InstallHookPackageInstaller.ModelProvidedPackagesInstructions,
            executionInstructions: InstallHookPackageInstaller.ModelShellInstructions);

        Skills = new SkillRegistry(new SkillRegistryOptions
        {
            Roots = new[] { paths.BundledSkillsDirectory, paths.InstalledSkillsDirectory },
            InstallDirectory = paths.InstalledSkillsDirectory,
            // Most of this app's skills ship INSIDE the bundle, which is read-only and
            // is rewritten by every install of the app. Deleting one therefore cannot
            // be done by deleting files: without this record the delete either failed
            // outright or lasted until the next launch. It lives in the data root, so
            // it survives app updates the way the user's conversations do.
            RemovedRecordFile = Path.Combine(paths.DataRoot, "removed-skills.txt"),
        });

        ModelService = modelService ?? new ModelService(_loggerFactory.CreateLogger<ModelService>());
        EngineHasWorkInFlight = EngineIsProcessing;
        // Whether the model may run at all right now. Closed by the iOS head while the
        // app is not frontmost, because iOS refuses GPU work from the background and
        // ggml-metal answers a refused command buffer by latching into a permanent
        // error state. Handed to the engine host so the ENGINE's own step loop parks on
        // it -- the engine decodes on its own thread into an unbounded channel, so a
        // wrapper that merely stops pulling would stop nothing -- and consulted again
        // by the stream wrapper below before anything new is submitted. See ComputeGate.
        Compute = new ComputeGate();
        ModelService.EngineHost.ComputeGate = Compute;
        Sessions = new SessionManager();
        Uploads = new UploadStoragePolicy(paths.UploadsDirectory);

        Options = BuildOptions(paths, settings, backends);
        // The startup selection's card values, so the first message of a launch is
        // sampled the same way the tenth is (UseModel repoints these on every switch).
        if (settings.SelectedModelId is { Length: > 0 } startupId && ModelCatalog.Find(startupId) is { } startupModel)
            Options.RepointSamplingDefaults(SamplingDefaultsFor(startupModel));

        // A diffusion entry is five files, not one, and only three of them are found by
        // the scan the pipeline does next to the weights. Publishing all of them here —
        // before anything can ask for a load — is what makes the catalog's file list the
        // whole story rather than most of it. See DiffusionCompanions.
        CatalogModel? selected = settings.SelectedModelId is { Length: > 0 } selectedId
            ? ModelCatalog.Find(selectedId)
            : null;
        IReadOnlyDictionary<string, string> companions = DiffusionCompanions.Publish(
            selected?.Kind == CatalogArchitectureKind.Diffusion ? selected : null, Models, paths.DeviceMemoryGB);
        if (companions.Count > 0)
        {
            _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation(
                "image-generation companions: {Companions}",
                string.Join(", ", companions.Select(c => $"{c.Key}={Path.GetFileName(c.Value)}")));
        }

        Chat = new WebUiChatService(
            ModelService, Sessions, Options, Uploads, Skills,
            CodeRunner!, Workspaces, Artifacts, _loggerFactory,
            WebUiChatService.DefaultArtifactUriPrefix,
            skillRouter: TensorAgentSkillRouter.Route);

        // The share contract stays platform-neutral. On iOS the path points into the
        // App Group container; tests use an ordinary temporary directory.
        Shares = new ShareIntake();
        ShareImports = new ShareImporter(Chat, Shares, _loggerFactory);
        ShareInbox = paths.SharedInboxDirectory is { Length: > 0 } inbox
            ? new ShareEnvelopeStore(inbox)
            : null;
        if (ShareInbox is not null)
            ShareInbox.RecoverClaims();
        Shares.Acknowledging += OnShareAcknowledged;
        Chat.AcquireChatRequestLease = AcquireShareSendLease;

        // A turn the user just took has to survive the app being closed, and the Web
        // UI page keeps its history only in memory. This is the hook the chat service
        // exposes for exactly that: the transcript is written on the host side, keyed
        // by the conversation the request named.
        Recorder = new ConversationRecorder(Conversations);
        Chat.OnChatRequest = (sessionId, requestBody) =>
        {
            // A turn being finished after a GPU fault re-sends the conversation with
            // the half-written answer and an instruction to carry on. That is how the
            // answer is being produced, not part of it: recording it would put words
            // into the transcript that the user never typed and the model never chose.
            if (requestBody.TryGetProperty(ResumedTurnMarker, out JsonElement resumed)
                && resumed.ValueKind == JsonValueKind.True)
                return;
            Recorder.Record(sessionId, requestBody);

            // Only an accepted request, recorded durably above, consumes a shared
            // draft. Merely showing it in WebKit is not terminal: the content process
            // or app can be killed at any time and the composer itself is not stored.
            bool multipleSharedDrafts = HasMultipleShareIdEntries(requestBody);
            string[] sharedIds = ShareIdsOf(requestBody);
            if (!multipleSharedDrafts && sharedIds.Length == 1)
            {
                // The request-start lease holds these ids against an explicit discard;
                // this lock also preserves the invariant for tests/hosts that invoke the
                // accepted-request hook directly.
                lock (_shareOwnershipLock)
                {
                    string id = sharedIds[0];
                    if (!Shares.Acknowledge(id))
                    {
                        _loggerFactory.CreateLogger("TensorAgent.Share").LogWarning(
                            "Accepted chat request could not acknowledge shared draft {Id}; retaining it", id);
                    }
                }
            }
            else if (multipleSharedDrafts || sharedIds.Length > 1)
            {
                // AcquireShareSendLease rejects this in the real request path. Keep the
                // accepted-request hook defensive as well: tests and alternate hosts
                // can invoke it directly, and one turn must never consume two shares.
                _loggerFactory.CreateLogger("TensorAgent.Share").LogWarning(
                    "Accepted chat request named {Count} shared drafts; retaining all of them", sharedIds.Length);
            }
        };

        // A generation belongs to the app, not to the HTTP request that asked for it.
        // See ChatTurnManager: on a phone the reader goes away constantly -- another
        // screen, another app, a display that dimmed -- and every one of those used to
        // be an answer thrown away halfway through.
        Turns = new ChatTurnManager(Recorder);
        // One line per finished turn saying what the process is charged and what the
        // device has left. A jetsam kill leaves no message of its own; these lines,
        // and the one the memory warning writes, are the evidence it leaves behind.
        Turns.BusyChanged += busy =>
        {
            if (busy)
                StartMemoryTrace();
            else
            {
                StopMemoryTrace();
                LogMemory("after the turn");
            }
        };

        SkillsService = new SkillsService(Skills, Options, Uploads, _loggerFactory);

        Server = new LoopbackServer(_loggerFactory.CreateLogger("TensorAgent.Loopback"), port)
        {
            StaticRoot = webRoot,
        };
        // Through the gate below rather than the chat service directly: a turn must
        // never run beside the prefix-cache warm-up, must not submit GPU work while the
        // app is away, and has to recognise and repair a poisoned engine. See
        // GatedChatFrames.
        Server.MapWebUi(Chat, Options.UploadDirectory, SkillsService, Recorder, chatFrames: GatedChatFrames, turns: Turns);
        // Without this the model's "here is your PDF" link 404s: the runner emits
        // /api/code/artifacts/... and nothing served it. See MapCodeArtifacts.
        Server.MapCodeArtifacts(Artifacts);
        Server.MapAgent(
            Catalog, Models, Conversations, Settings, DescribeEngine, RaisePageEvent, Downloads,
            onSettingsChanged: ApplySettings, describeModel: DescribeModelState, shares: Shares,
            hasShareContainer: () => ShareInbox is not null, discardShare: DiscardPendingShare);
    }

    public AgentPaths Paths { get; }
    public SettingsStore Settings { get; }
    public ModelStore Models { get; }
    /// <summary>Every model download this launch started, independent of any screen.</summary>
    public ModelDownloadManager Downloads { get; }
    public ConversationStore Conversations { get; }
    public IReadOnlyList<CatalogModel> Catalog { get; }
    public CodeExecOptions CodeExec { get; }
    public SessionWorkspaceManager Workspaces { get; }
    public InProcessShellBackend Backend { get; }
    public IPythonRuntime? Python { get; }
    public IJavaScriptRuntime? JavaScript { get; }
    public IInstallHook? Installer { get; }
    public CodeArtifactStore Artifacts { get; }
    public ICodeRunner? CodeRunner { get; }
    public SkillRegistry Skills { get; }
    public ModelService ModelService { get; }
    public SessionManager Sessions { get; }
    public UploadStoragePolicy Uploads { get; }
    public ServerHostingOptions Options { get; }
    public WebUiChatService Chat { get; }
    public ConversationRecorder Recorder { get; }
    public ShareIntake Shares { get; }
    public ShareImporter ShareImports { get; }
    public ShareEnvelopeStore? ShareInbox { get; }

    /// <summary>
    /// Whether the model may run right now. See <see cref="ComputeGate"/>; the iOS head
    /// closes it while the app is not in front of the user, and both the engine's step
    /// loop and <see cref="GatedChatFrames"/> wait on it.
    /// </summary>
    public ComputeGate Compute { get; }
    /// <summary>Every generation this launch started, independent of any page or request.</summary>
    public ChatTurnManager Turns { get; }
    public SkillsService SkillsService { get; }
    public LoopbackServer Server { get; }

    /// <summary>
    /// Raised for each message the page posts about itself: <c>ready</c>,
    /// <c>conversation</c>, and whatever the injected script adds later. The native
    /// chrome subscribes so the title bar and the sessions list follow what the page
    /// is actually showing rather than what the app last asked it to show.
    /// </summary>
    public event Action<string, System.Text.Json.JsonElement>? PageEvent;

    private void RaisePageEvent(string kind, System.Text.Json.JsonElement message) => PageEvent?.Invoke(kind, message);

    /// <summary>The URL the WebView opens: the page plus the launch token that sets its cookie.</summary>
    public string EntryUrl => Server.EntryUrl;

    public void Start()
    {
        Server.Start();
        LoadSelectedModelInBackground();
    }

    // ---- coming back to the foreground -----------------------------------------
    //
    // iOS reclaims the sockets of a suspended app, listening socket included, and the
    // managed HttpListener cannot tell (see LoopbackServer.EnsureListeningAsync). The
    // page then gets "Load failed" on every request while the host is perfectly healthy
    // -- the turn resumes, the gate opens, nothing is logged -- and the only thing the
    // user could do about it was force-quit. This is the host's half of the answer:
    // probe the listener as the app comes back, rebuild it if it is dead, write down
    // what was found, and tell whoever owns the WebView so the page can be nudged (or,
    // if the port had to change, reloaded).

    /// <summary>What <see cref="OnForegroundAsync"/> found and did.</summary>
    /// <param name="Listener">The listener's state after the check.</param>
    /// <param name="Detail">Why, in the words the trace uses.</param>
    /// <param name="Port">The loopback port in force afterwards.</param>
    /// <param name="EntryUrlChanged">True when the page's origin moved and the WebView must be navigated to <see cref="EntryUrl"/> again.</param>
    public sealed record ForegroundReport(
        LoopbackServer.ListenerHealth Listener, string Detail, int Port, TimeSpan Elapsed, bool EntryUrlChanged);

    /// <summary>
    /// Raised after every <see cref="OnForegroundAsync"/>, on the thread that ran it.
    /// The WebView's owner uses it to nudge the page once the transport is known good.
    /// </summary>
    public event Action<ForegroundReport>? ForegroundChecked;

    /// <summary>
    /// Raised when the loopback server had to move to another port, with the new
    /// <see cref="EntryUrl"/>. The page's origin, and so its token cookie, are gone with
    /// the old port: the WebView has to be navigated to the new URL.
    /// </summary>
    public event Action<string>? EntryUrlChanged;

    /// <summary>How many times the listener was found dead and rebuilt this launch.</summary>
    public int ListenerRestarts => Server.Restarts;

    /// <summary>
    /// The app is coming back in front of the user: make sure the page can reach the
    /// host. Cheap on a healthy listener (one loopback round trip), and the only path
    /// that repairs a reclaimed one. Never throws.
    /// </summary>
    public async Task<ForegroundReport> OnForegroundAsync(CancellationToken cancellationToken = default)
    {
        string entryBefore = EntryUrl;
        LoopbackServer.ListenerReport listener;
        try
        {
            listener = await Server.EnsureListeningAsync(ListenerProbeTimeout, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            listener = new LoopbackServer.ListenerReport(
                LoopbackServer.ListenerHealth.Failed, ex.GetType().Name + ": " + ex.Message, Server.Port, TimeSpan.Zero);
        }
        bool moved = !string.Equals(entryBefore, EntryUrl, StringComparison.Ordinal);

        TraceBackground(listener.Health switch
        {
            LoopbackServer.ListenerHealth.Alive => $"foreground: loopback listener alive ({listener.Detail})",
            LoopbackServer.ListenerHealth.Restarted => $"foreground: loopback listener DEAD; {listener.Detail}",
            LoopbackServer.ListenerHealth.Relisted => $"foreground: loopback listener DEAD; {listener.Detail}; the page will be reloaded at {EntryUrl}",
            _ => $"foreground: loopback listener FAILED; {listener.Detail}",
        });
        if (listener.Health != LoopbackServer.ListenerHealth.Alive)
        {
            _loggerFactory.CreateLogger("TensorAgent.Loopback").LogWarning(
                "the loopback listener was {Health} on returning to the foreground: {Detail}", listener.Health, listener.Detail);
        }

        var report = new ForegroundReport(listener.Health, listener.Detail, Server.Port, listener.Elapsed, moved);
        if (moved)
        {
            try { EntryUrlChanged?.Invoke(EntryUrl); }
            catch (Exception ex) { TraceBackground("foreground: the entry-url listener threw: " + ex.Message); }
        }
        try { ForegroundChecked?.Invoke(report); }
        catch (Exception ex) { TraceBackground("foreground: a listener threw: " + ex.Message); }
        return report;
    }

    /// <summary>
    /// How long the foreground probe waits for the listener to answer. Generous for a
    /// loopback round trip because it runs the instant the app wakes, when every thread
    /// pool thread is also waking; a dead listener refuses immediately, so the budget is
    /// only ever spent on a slow live one.
    /// </summary>
    public TimeSpan ListenerProbeTimeout { get; set; } = TimeSpan.FromSeconds(3);

    /// <summary>
    /// Import durable envelopes left by the share extension. A claimed directory is
    /// retained until an accepted, recorded chat request or explicit discard consumes
    /// the draft; on any transient failure it is atomically returned to the ready queue.
    /// </summary>
    public async Task<int> DrainSharedInboxAsync(CancellationToken cancellationToken = default)
    {
        if (ShareInbox is null)
            return 0;

        await _shareDrain.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {

            ILogger log = _loggerFactory.CreateLogger("TensorAgent.Share");
            await RetryClaimReleasesAsync(log).ConfigureAwait(false);
            try { ShareInbox.Purge(DateTimeOffset.UtcNow); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                log.LogWarning(ex, "The share inbox could not be purged");
            }

            int imported = 0;
            foreach (string id in ShareInbox.ListReady())
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Backpressure keeps the durable envelope on disk. Dropping an already
                // imported share to make room would silently lose user data.
                if (!Shares.CanAccept)
                    break;

                ClaimedShare? claim = ShareInbox.TryClaim(id, out string? why);
                if (claim is null)
                {
                    if (why is { Length: > 0 })
                        log.LogInformation("Share {Id} not claimed: {Reason}", id, why);
                    continue;
                }

                lock (_shareClaimsLock)
                    _shareClaims[id] = claim.Directory;
                try
                {
                    ShareImportResult result = await ShareImports
                        .ImportWithOutcomeAsync(claim.Payload, claim.Directory, cancellationToken)
                        .ConfigureAwait(false);
                    if (result.Status == ShareImportStatus.Offered)
                    {
                        imported++;
                        continue;
                    }

                    lock (_shareClaimsLock)
                        _shareClaims.Remove(id);
                    if (result.Status == ShareImportStatus.RetryLater)
                        await ReleaseClaimAsync(claim, log).ConfigureAwait(false);
                    else
                        ShareInbox.Discard(claim.Directory);
                }
                catch
                {
                    lock (_shareClaimsLock)
                        _shareClaims.Remove(id);
                    await ReleaseClaimAsync(claim, log).ConfigureAwait(false);
                    throw;
                }
            }

            if (imported > 0)
                log.LogInformation("Imported {Count} share(s) from the durable inbox", imported);
            return imported;
        }
        finally
        {
            _shareDrain.Release();
        }
    }

    private async Task<bool> ReleaseClaimAsync(ClaimedShare claim, ILogger log)
    {
        if (ShareInbox is null)
            return false;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            if (ShareInbox.Release(claim))
            {
                lock (_shareClaimsLock)
                    _shareReleaseRetries.Remove(claim.Payload.Id);
                return true;
            }
            if (attempt < 2)
                await Task.Delay(attempt == 0 ? 20 : 100).ConfigureAwait(false);
        }

        lock (_shareClaimsLock)
            _shareReleaseRetries[claim.Payload.Id] = claim;
        log.LogWarning("Share {Id} could not be returned to the ready inbox; it remains claimed and will be retried",
            claim.Payload.Id);
        return false;
    }

    private async Task RetryClaimReleasesAsync(ILogger log)
    {
        ClaimedShare[] retries;
        lock (_shareClaimsLock)
            retries = _shareReleaseRetries.Values.ToArray();
        foreach (ClaimedShare claim in retries)
            await ReleaseClaimAsync(claim, log).ConfigureAwait(false);
    }

    private bool OnShareAcknowledged(string id)
    {
        if (ShareInbox is null)
            return true;
        string? directory;
        lock (_shareClaimsLock)
        {
            if (!_shareClaims.TryGetValue(id, out directory))
                return false;
        }
        if (!ShareInbox.Acknowledge(id, directory!))
            return false;
        lock (_shareClaimsLock)
            _shareClaims.Remove(id);
        // An acknowledgement frees a bounded in-memory slot. Pull the next durable
        // envelope immediately instead of making it wait for another foreground event.
        ScheduleShareDrain();
        return true;
    }

    private void ScheduleShareDrain() => _ = Task.Run(async () =>
    {
        try { await DrainSharedInboxAsync().ConfigureAwait(false); }
        catch (Exception ex) { HostLog.LogWarning(ex, "Could not continue draining shares after acknowledgement"); }
    });

    private IDisposable AcquireShareSendLease(JsonElement requestBody)
    {
        // Engine recovery continues the already-accepted turn with a body derived from
        // the original one. Its durable share was acknowledged by that original request;
        // leasing it again would incorrectly turn recovery into a 409.
        if (requestBody.TryGetProperty(ResumedTurnMarker, out JsonElement resumed)
            && resumed.ValueKind == JsonValueKind.True)
        {
            return EmptyLease.Instance;
        }

        if (HasMultipleShareIdEntries(requestBody))
        {
            throw new WebUiRequestRejectedException(409, new
            {
                code = "multiple_shared_drafts",
                error = "Each shared item starts its own chat. Send or remove the current shared item before opening the next one.",
            });
        }

        string[] ids = ShareIdsOf(requestBody);
        if (ids.Length == 0)
            return EmptyLease.Instance;
        if (ids.Length != 1)
        {
            throw new WebUiRequestRejectedException(409, new
            {
                code = "multiple_shared_drafts",
                error = "Each shared item starts its own chat. Send or remove the current shared item before opening the next one.",
            });
        }

        lock (_shareOwnershipLock)
        {
            PendingShare? head = Shares.Peek();
            if (head is null
                || !string.Equals(head.Id, ids[0], StringComparison.Ordinal)
                || ids.Any(id => _sharesBeingSent.Contains(id)))
            {
                throw new WebUiRequestRejectedException(409, new
                {
                    code = "shared_draft_unavailable",
                    error = "The shared draft changed before Send was accepted. Review the composer and try again.",
                });
            }

            foreach (string id in ids)
                _sharesBeingSent.Add(id);
        }
        return new ShareSendLease(this, ids);
    }

    private string[] ShareIdsOf(JsonElement requestBody)
    {
        if (requestBody.ValueKind != JsonValueKind.Object
            || !requestBody.TryGetProperty("shareIds", out JsonElement shared)
            || shared.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<string>();
        }

        return shared.EnumerateArray()
            .Take(Shares.MaxPending)
            .Select(value => value.ValueKind == JsonValueKind.String ? value.GetString() : null)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.Ordinal)
            .Cast<string>()
            .ToArray();
    }

    private static bool HasMultipleShareIdEntries(JsonElement requestBody) =>
        requestBody.ValueKind == JsonValueKind.Object
        && requestBody.TryGetProperty("shareIds", out JsonElement shared)
        && shared.ValueKind == JsonValueKind.Array
        && shared.GetArrayLength() > 1;

    private void ReleaseShareSendLease(IEnumerable<string> ids)
    {
        lock (_shareOwnershipLock)
        {
            foreach (string id in ids)
                _sharesBeingSent.Remove(id);
        }
    }

    private sealed class ShareSendLease : IDisposable
    {
        private AgentAppHost? _owner;
        private readonly string[] _ids;

        public ShareSendLease(AgentAppHost owner, string[] ids)
        {
            _owner = owner;
            _ids = ids;
        }

        public void Dispose()
        {
            AgentAppHost? owner = Interlocked.Exchange(ref _owner, null);
            owner?.ReleaseShareSendLease(_ids);
        }
    }

    private sealed class EmptyLease : IDisposable
    {
        public static readonly EmptyLease Instance = new();
        public void Dispose() { }
    }

    /// <summary>
    /// Explicitly reject the durable head share and reclaim its app-side staged files.
    /// This is separate from removing an ordinary attachment: it is the user's escape
    /// hatch for a share they no longer want resurfacing after every WebView restart.
    /// </summary>
    public bool DiscardPendingShare(string id)
    {
        PendingShare pending;
        lock (_shareOwnershipLock)
        {
            // Send owns the draft from request-start through every preflight. Whichever
            // operation acquired this lock first wins: discard-first makes the later
            // request lease reject, send-first leaves the durable draft and bytes intact.
            if (_sharesBeingSent.Contains(id))
                return false;
            PendingShare? head = Shares.Peek();
            if (head is null || !string.Equals(head.Id, id, StringComparison.Ordinal))
                return false;
            if (!Shares.Acknowledge(id))
                return false;
            pending = head;
        }

        foreach (object attachment in pending.Attachments)
        {
            if (!Chat.DiscardUpload(attachment))
            {
                _loggerFactory.CreateLogger("TensorAgent.Share").LogWarning(
                    "Discarded share {Id}, but one staged attachment will remain until ordinary upload cleanup", id);
            }
        }
        // Acknowledgement starts a drain immediately, while the just-discarded upload
        // may still count against quota. Run once more after cleanup so a next envelope
        // transiently released by that first drain does not wait for another foreground.
        ScheduleShareDrain();
        return true;
    }

    // ---- making the FIRST message as fast as the second ----------------------------

    private CancellationTokenSource? _warmup;
    private Task? _warmupTask;
    private readonly object _warmupLock = new();
    private string? _warmUpSession;

    /// <summary>
    /// Where the warm-up's frames come from, or null for the chat service's own stream.
    /// A seam for tests, which have no model to warm: the app never sets it.
    /// </summary>
    internal Func<JsonElement, CancellationToken, IAsyncEnumerable<object>>? WarmUpFrames { get; set; }

    /// <summary>
    /// Whether the shared-prompt warm-up request completed successfully. This does not
    /// guarantee a retained payload: model capability and cache budgets still apply.
    /// False until completion, and false again when another warm-up starts.
    /// </summary>
    public bool PrefixCacheIsWarm { get; private set; }

    /// <summary>
    /// How many warm-ups have actually been STARTED. Counted because "did it start" and
    /// "did it finish" are different questions and the dangerous one is the first: a
    /// warm-up that starts beside a turn has already submitted a graph by the time it
    /// notices, which is the whole of the bug this counts for.
    /// </summary>
    public long PrefixCacheWarmupsStarted { get; private set; }

    /// <summary>
    /// Forward the prompt every conversation begins with, before the user asks for
    /// anything, so their first message does not have to.
    ///
    /// <para>
    /// Measured on an iPhone 17 Pro Max, and the numbers are the whole argument. A
    /// conversation's first turn forwards several thousand tokens of system prompt,
    /// tool schemas and skill descriptions before the model writes a character: 0% KV
    /// reuse, and twenty to forty seconds before the first token. Every LATER turn
    /// reuses 99% of that and answers in under a second — and so does the first turn
    /// of a brand new conversation, because the expensive part is shared by all of
    /// them. So the prompt was never slow. It was paid for once, and the user was the
    /// one paying.
    /// </para>
    /// <para>
    /// This pays it instead, on a throwaway one-token request issued as soon as the
    /// weights are in — while the user is still reading the screen — and again after
    /// every load, which throws the cache away with the previous model. It is cancelled
    /// the moment a real turn wants the engine, and it binds no conversation, so
    /// nothing is recorded and no chat appears in the user's list.
    /// </para>
    /// <para>
    /// The one-token request uses the same system prompt and tools as a real chat.
    /// The pipeline declares that shared prefix public, so a new conversation can
    /// reuse it without sharing the warm-up's user message or generated token.
    /// </para>
    /// </summary>
    public void WarmThePrefixCache()
    {
        if (!(ModelService.EngineHost.SchedulerConfigOverride ?? SchedulerConfig.FromEnvironment()).EnablePrefixCaching)
        {
            HostLog.LogInformation("not warming the prefix cache: runtime prefix reuse is disabled");
            return;
        }
        // Never beside a turn. The warm-up is opportunistic by definition -- it exists to
        // save the NEXT message a wait -- so contending with a message already being
        // answered is all cost and no benefit, and on a model that cannot take two
        // requests at once it is a corrupted answer. The turn repopulates the cache
        // itself on its way through.
        if (Turns.IsBusy)
        {
            HostLog.LogInformation("not warming the prefix cache: a turn is already using the engine");
            return;
        }

        CancellationTokenSource source;
        lock (_warmupLock)
        {
            StopWarmingThePrefixCacheWhileHoldingTheLock();
            _warmup = source = new CancellationTokenSource();
            // A fresh model means a cache with nothing in it -- whatever was true a
            // moment ago is not true now.
            PrefixCacheIsWarm = false;
            PrefixCacheWarmupsStarted++;
        }

        // The TOKEN, captured now: the source is disposed the moment a turn stops
        // this warm-up, and a token copied before that stays readable afterwards
        // where `source.Token` would throw ObjectDisposedException.
        CancellationToken token = source.Token;
        bool rebuildAfterwards = false;
        Task running = Task.Run(async () =>
        {
            try
            {
                // Not while the app is away: the warm-up is GPU work like any other, and
                // one started at the moment of backgrounding is refused exactly as a
                // token would be. Waited for on both sides of the settling delay, because
                // either can take a while and the app can leave during either.
                await Compute.WaitAsync(token).ConfigureAwait(false);
                // A moment for the engine to finish settling after a load. Nobody is
                // waiting on this and the cost of being early is a crash.
                await Task.Delay(TimeSpan.FromSeconds(2), token).ConfigureAwait(false);
                await Compute.WaitAsync(token).ConfigureAwait(false);

                // Checked again on the other side of the waits, because a turn can start
                // during them.
                if (Turns.IsBusy)
                {
                    HostLog.LogInformation("dropping the prefix-cache warm-up: a turn started while it was waiting");
                    return;
                }
                token.ThrowIfCancellationRequested();

                // WITH A SESSION, and that is the difference between warming the right
                // prompt and warming a different one. A request that carries no session
                // is served by the DEFAULT session; WorkspaceFor returns null for that
                // one, and a chat with no workspace is declared only `shell` with no
                // file tools, where a real turn is declared read_file, apply_patch,
                // write_file and a persisting shell, plus the "Working with
                // files" instructions. KV reuse is a longest-common-PREFIX match, so a
                // tool block that differs makes everything after it unshareable -- the
                // warm-up would run for twenty seconds and save a real turn only the
                // part before the tools.
                //
                // And with the user's thinking default, because Gemma 4's template puts
                // its thinking marker at the very top of the system turn: a prompt
                // warmed with thinking off shares nothing with one the user sends with
                // it on.
                bool think = Settings.Load().ThinkByDefault;
                Func<JsonElement, CancellationToken, IAsyncEnumerable<object>> frames = WarmUpFrames ?? Chat.ChatStreamAsync;
                PrefixCacheWarmup.Result warmed = await PrefixCacheWarmup.RunAsync(
                    frames, WarmUpSessionId(), think, HostLog, token).ConfigureAwait(false);
                if (token.IsCancellationRequested)
                    return;
                if (!warmed.Warmed)
                {
                    string failure = warmed.Detail;
                    HostLog.LogWarning("warming the prefix cache failed: {Error}", failure);
                    Console.WriteLine("TensorAgent: warm-up failed: " + failure);
                    // The warm-up is the likeliest thing to meet a dead GPU: it runs
                    // seconds after every load, launch included, and a GPU reset caused by
                    // the previous process being killed mid-compute lands on it twice a day
                    // in testing. Marking is what makes the next message rebuild before it
                    // is attempted -- but a rebuild at the next message still costs that
                    // message the whole prompt (40 s on the phone, measured), because the
                    // checkpoint went with the backend. Nothing is running right now, so
                    // the rebuild AND the re-warm happen here, while the user is still
                    // reading, and the message pays nothing. See RebuildAfterAPoisonedWarmUp.
                    if (ReadsLikeAPoisonedEngine(failure) is { Length: > 0 } poison)
                    {
                        NoteEngineMayBePoisoned(poison);
                        rebuildAfterwards = true;
                    }
                    return;
                }
                PrefixCacheIsWarm = true;
                string line = $"shared-prompt warm-up completed ({warmed.Elapsed.TotalSeconds:0.#}s)";
                HostLog.LogInformation("{Line}", line);
                Console.WriteLine("TensorAgent: warm-up: " + line);
                LogMemory("after the warm-up");
            }
            catch (OperationCanceledException)
            {
                // A real turn arrived, or the app is shutting down. Either way this had
                // one job and something more important is doing it.
            }
            catch (Exception ex)
            {
                // Never worth surfacing: the app is exactly as usable as it was before,
                // just as slow on its first message.
                HostLog.LogWarning(ex, "warming the prefix cache did not finish");
                Console.WriteLine("TensorAgent: warm-up did not finish: " + ex.Message);
            }
            finally
            {
                // From a task of its own, after this one is out of the way: the rebuild
                // stops and waits for "the warm-up", which is this very task.
                if (rebuildAfterwards)
                    RebuildAfterAPoisonedWarmUp();
            }
        }, CancellationToken.None);

        lock (_warmupLock)
        {
            // Only if nothing has replaced it in the meantime.
            if (ReferenceEquals(_warmup, source))
                _warmupTask = running;
        }
    }

    /// <summary>
    /// Stop a warm-up AND wait for it to actually be gone.
    ///
    /// <para>
    /// Cancelling is not enough, and a device reproduction is why. Cancellation is
    /// cooperative: the warm-up is inside a generation, and it stops at the next place
    /// the engine looks at the token, not instantly. A caller that cancelled and carried
    /// straight on started its own generation while the warm-up was still running one --
    /// two at once, on a model whose scheduler says in as many words that "a second
    /// request would corrupt attention". What that produced on an iPhone was
    /// "Native GGML get_rows_quant failed" and, for the user who reported it,
    /// "Object reference not set to an instance of an object".
    /// </para>
    /// </summary>
    /// <remarks>
    /// Wait for cancellation to drain instead of forcing the warm-up to finish. A
    /// later conversation can reuse any public prefix already captured; an unfinished
    /// warm-up may leave no reusable shared checkpoint.
    /// </remarks>
    public async Task StopWarmingThePrefixCacheAndWaitAsync()
    {
        Task? running;
        lock (_warmupLock)
        {
            running = _warmupTask;
            StopWarmingThePrefixCacheWhileHoldingTheLock();
        }
        if (running is null)
            return;
        try
        {
            // Bounded: a warm-up that will not stop must not hold a turn for ever. The
            // turn is the thing the user is waiting for.
            await running.WaitAsync(TimeSpan.FromSeconds(30)).ConfigureAwait(false);
        }
        catch (Exception)
        {
            // Cancelled, faulted, or too slow. Nothing here is worth failing a turn over,
            // and the wait is the point rather than the result.
        }
    }

    /// <summary>Stop a warm-up in flight. Safe to call when there is none.</summary>
    public void StopWarmingThePrefixCache()
    {
        lock (_warmupLock)
            StopWarmingThePrefixCacheWhileHoldingTheLock();
    }

    private void StopWarmingThePrefixCacheWhileHoldingTheLock()
    {
        _warmupTask = null;
        if (_warmup is null)
            return;
        try { _warmup.Cancel(); }
        catch (ObjectDisposedException) { /* already gone */ }
        _warmup.Dispose();
        _warmup = null;
    }

    /// <summary>
    /// A session for the warm-up to speak through, made once and kept.
    ///
    /// <para>
    /// It is never bound to a conversation — <c>ConversationRecorder.Bind</c> is called
    /// by the <c>/api/sessions</c> route, not by this — so nothing is recorded and no
    /// chat appears in the user's list. All it supplies is the one thing the default
    /// session cannot: a workspace, and therefore the same tool declarations a real turn
    /// gets.
    /// </para>
    /// </summary>
    private string WarmUpSessionId()
    {
        if (_warmUpSession is { Length: > 0 })
            return _warmUpSession;
        object created = Chat.CreateSession();
        _warmUpSession = created.GetType().GetProperty("sessionId")?.GetValue(created) as string ?? string.Empty;
        return _warmUpSession;
    }

    private ILogger HostLog => _loggerFactory.CreateLogger("TensorAgent.Host");

    // ---- running the model only when the app is allowed to -------------------------

    /// <summary>
    /// The chat stream, with three things in front of it that <c>/api/chat</c> must
    /// never do without: the warm-up stopped and waited for, the compute gate
    /// consulted before every pull, and a poisoned engine recognised and rebuilt.
    ///
    /// <para>
    /// The gate is the fix for "I switched away during an answer and it failed". iOS
    /// refuses GPU work from an app that is not frontmost, and ggml-metal treats a
    /// refused command buffer as terminal: it sets a sticky flag and every later
    /// <c>graph_compute</c> returns <c>GGML_STATUS_FAILED</c> "until the backend is
    /// recreated" (ggml-metal-context.m). So one badly-timed submission does not cost a
    /// token, it costs the model — the turn dies AND every message after it. The
    /// engine's own step loop parks on the same gate (see
    /// <see cref="InferenceEngine.ComputeGate"/>), which is what actually stops the GPU
    /// work; waiting here as well means no NEW request is submitted while the app is
    /// away either, and that a turn started from the background — a share, a
    /// notification — waits rather than fails.
    /// </para>
    /// <para>
    /// A COUNT of closures is taken around the turn rather than a flag, because what
    /// matters afterwards is not "is the app in front now" — by the time a failure is
    /// handled it always is — but "was it ever away while this was running". That is
    /// what tells a genuine engine fault apart from the one this exists for.
    /// </para>
    /// </summary>
    private async IAsyncEnumerable<object> GatedChatFrames(
        JsonElement body,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // Bounded, because a turn that silently regenerates forever is worse than one
        // that reports what went wrong -- but bounded at TWO retries rather than one,
        // and the device is why. A GPU that has just refused a command buffer does not
        // come back the instant the app does: the run that produced this number
        // rebuilt the backend, started the answer again, and lost that attempt to a
        // second fault ("Discarded (victim of GPU error/recovery)") while the GPU was
        // still settling. The message AFTER it worked first time. One more go inside
        // the turn is the difference between the user reading an error and the user
        // reading their answer.

        // What the user has already watched appear. Kept across a retry so the answer
        // can be carried on rather than written again from the top -- see below.
        var written = new System.Text.StringBuilder();
        JsonElement attemptBody = body;

        for (int attempt = 0; ; attempt++)
        {
            // BEFORE the rebuild, not after. A rebuild unloads the model and frees the
            // backend; doing that while a warm-up is still generating tears the weights
            // out from under a live step. Ordering this after the rebuild protected the
            // turn and left the rebuild itself exposed.
            await StopWarmingThePrefixCacheAndWaitAsync().ConfigureAwait(false);

            // A poisoned engine cannot answer anything, so it is rebuilt before the
            // attempt rather than after the failure.
            //
            // BEHIND THE GATE, though, and a device run is the reason that is not an
            // afterthought. Loading a model is GPU work like any other -- ggml_metal_init
            // builds its pipelines and the load runs a warmup graph -- so a rebuild
            // started while the app is away is refused exactly as the token that provoked
            // it was. The first attempt at this repaired the engine at the moment of
            // backgrounding and produced a backend that was poisoned before its first
            // token, which reads in the trace as a repair that worked and an answer that
            // died anyway.
            if (EngineNeedsReload)
            {
                await WaitForTheAppToBeInFrontAsync(cancellationToken).ConfigureAwait(false);
                TraceBackground("rebuilding the engine before answering");
                // Off the calling thread: this reads a multi-gigabyte file and the caller
                // is the response writer. And NOT followed by a warm-up: this turn is
                // about to forward the very prompt a warm-up would forward.
                await Task.Run(() => RecoverEngineIfNeeded(warmAfterwards: false), cancellationToken)
                    .ConfigureAwait(false);
                // Unless the rebuild had already been done by another path -- the return
                // to the foreground, or a warm-up that met the dead backend -- while this
                // turn waited for the recovery lock: THAT rebuild warms afterwards, and
                // its warm-up must not run beside this turn either.
                await StopWarmingThePrefixCacheAndWaitAsync().ConfigureAwait(false);
            }

            long closuresAtStart = Compute.Closures;
            bool poisoned = false;

            await using (IAsyncEnumerator<object> frames =
                Chat.ChatStreamAsync(attemptBody, cancellationToken).GetAsyncEnumerator(cancellationToken))
            {
                while (true)
                {
                    await WaitForTheAppToBeInFrontAsync(cancellationToken).ConfigureAwait(false);

                    bool more;
                    try
                    {
                        more = await frames.MoveNextAsync().ConfigureAwait(false);
                    }
                    // A refusal is a DECISION, not a fault: "no model is loaded", a body
                    // the parser will not take, a policy that says no. Those must not be
                    // read as engine damage however long the app was away, or the first
                    // message after a background switch would throw away a loaded model
                    // to fix nothing.
                    catch (Exception ex) when (
                        ex is not OperationCanceledException and not WebUiRequestRejectedException)
                    {
                        // What is left is a fault, and a fault in a turn that ran while
                        // the app was not in front is the signature of the sticky Metal
                        // error: the backend cannot recover from that on its own.
                        if (Compute.Closures != closuresAtStart || ReadsLikeAPoisonedEngine(ex.Message) is not null)
                            NoteEngineMayBePoisoned(ex.Message);
                        throw;
                    }

                    if (!more)
                        yield break;

                    // The failure this whole class exists for does NOT arrive as an
                    // exception. The chat service catches it and ends the stream with a
                    // `done` frame carrying the message, so a wrapper that only watches
                    // for throws watches the wrong thing -- which is exactly what the
                    // first device run showed: the turn died, the engine stayed marked
                    // as healthy, and every message afterwards died too.
                    if (ReadsLikeAPoisonedEngine(ErrorIn(frames.Current)) is { Length: > 0 } fault)
                    {
                        // Marked every time, retried while there are goes left: the
                        // mark is what makes the NEXT message rebuild before it starts,
                        // and a turn that gave up must still leave that behind. Without
                        // it the last failure passes through as an ordinary error and
                        // the app stays broken until it is force-quit.
                        NoteEngineMayBePoisoned(fault);
                        if (attempt < RetriesAfterAPoisonedEngine)
                        {
                            poisoned = true;
                            break; // and the `done` frame is swallowed with it
                        }
                    }

                    if (TokenIn(frames.Current) is { Length: > 0 } piece)
                        written.Append(piece);
                    else if (ReplaceIn(frames.Current) is { } whole)
                        written.Clear().Append(whole);
                    yield return frames.Current;
                }
            }

            if (!poisoned)
                yield break;

            // Carry on rather than start again, wherever that is possible. The answer
            // on screen is what the user has been reading; wiping it and rewriting it
            // is the visible part of the failure, and doing that twice is worse than
            // the failure. So the half-written answer is handed back to the model as
            // its own and it is asked to continue -- the KV cache went with the
            // backend, so the prompt is re-read either way, but the READER loses
            // nothing.
            //
            // Not always, though. A fragment too short to be worth keeping, or one
            // that stops in the middle of a tool call, would make the continuation
            // harder to produce than the answer; those start cleanly instead.
            string soFar = written.ToString();
            if (CanBeCarriedOn(soFar))
            {
                attemptBody = WithTheAnswerSoFar(body, soFar);
                yield return new
                {
                    restart = "The GPU was interrupted while the app was in the background. "
                              + "Picking this answer up where it stopped.",
                };
            }
            else
            {
                attemptBody = body;
                written.Clear();
                yield return new
                {
                    replace = string.Empty,
                    restart = "The GPU was taken away while the app was in the background. "
                              + "Starting this answer again.",
                };
            }
        }
    }

    /// <summary>
    /// How many times a turn will rebuild the engine and start its answer again before
    /// giving up and reporting the failure. See <see cref="GatedChatFrames"/>.
    /// </summary>
    private const int RetriesAfterAPoisonedEngine = 2;

    /// <summary>
    /// Hold here until the app is frontmost and the GPU is ours to use again.
    ///
    /// <para>
    /// In the ordinary case this is a completed task and costs nothing: it is on the
    /// path of every token. It says so in the trace only when it actually waits, which
    /// is the only time anybody reads that file.
    /// </para>
    /// </summary>
    private async Task WaitForTheAppToBeInFrontAsync(CancellationToken cancellationToken)
    {
        if (Compute.IsOpen)
            return;
        TraceBackground("turn paused: the app is not in front, so the GPU is not ours to use");
        await Compute.WaitAsync(cancellationToken).ConfigureAwait(false);
        TraceBackground($"turn resumed after {Compute.TotalClosed.TotalSeconds:0.#}s of not being in front");
    }

    /// <summary>
    /// Whether a half-written answer is worth handing back to the model to continue.
    ///
    /// <para>
    /// Two things make it not worth it. A fragment of a few characters carries no
    /// context a continuation could use, and asking for one costs a round of prompt
    /// tokens to save nothing. And an answer that stops inside a tool call is not text
    /// the model can carry on writing -- the call has to be made whole or made again,
    /// and making it again from the top is the simpler of the two.
    /// </para>
    /// </summary>
    internal static bool CanBeCarriedOn(string? soFar)
    {
        if (soFar is not { Length: >= 40 })
            return false;
        foreach ((string open, string close) in StructureAModelCanBeHalfwayThrough)
        {
            if (soFar.LastIndexOf(open, StringComparison.Ordinal) > soFar.LastIndexOf(close, StringComparison.Ordinal))
                return false;
        }
        return true;
    }

    /// <summary>
    /// The openers whose closers a half-written answer might be missing, one pair per
    /// output syntax the engine can parse.
    ///
    /// <para>
    /// Not one pair, which is what this started as. <c>&lt;tool_call&gt;</c> is Qwen's
    /// and the ChatML parsers'; Harmony (gpt-oss) writes channels, GLM writes its own
    /// bracketed form, and Gemma writes <c>&lt;function=</c>. Handing a model back half
    /// of any of those and asking it to carry on is asking it to finish a structure it
    /// cannot see the beginning of. The list is pinned by a test against
    /// <c>TensorSharp.Runtime/OutputParser.cs</c> so a new syntax cannot be added there
    /// without this being considered.
    /// </para>
    /// </summary>
    internal static readonly (string Open, string Close)[] StructureAModelCanBeHalfwayThrough =
    [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call>", "</tool_call>"),
        ("<tool_call|>", "</tool_call>"),
        ("<function=", "</function>"),
        ("<parameter=", "</parameter>"),
        ("<arg_key>", "</arg_key>"),
        ("<arg_value>", "</arg_value>"),
        ("<think>", "</think>"),
        ("<|channel>", "<|message|>"),
        ("<|channel|>", "<|message|>"),
        ("<channel|>", "<|message|>"),
        ("<|start|>", "<|end|>"),
    ];

    /// <summary>
    /// The same request with the answer so far attached, so the model continues it.
    ///
    /// <para>
    /// The two added messages are NOT part of the conversation: they are how this turn
    /// is being produced, not something the user said or the model decided. They are
    /// marked so <see cref="ConversationRecorder"/> leaves the transcript alone -- a
    /// user who scrolled back and found an instruction they never typed would be right
    /// to call that a bug.
    /// </para>
    /// </summary>
    internal static JsonElement WithTheAnswerSoFar(JsonElement body, string soFar)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (JsonProperty property in body.EnumerateObject())
            {
                if (string.Equals(property.Name, "messages", StringComparison.Ordinal)
                    || string.Equals(property.Name, ResumedTurnMarker, StringComparison.Ordinal))
                    continue;
                property.WriteTo(writer);
            }

            writer.WriteBoolean(ResumedTurnMarker, true);
            writer.WritePropertyName("messages");
            writer.WriteStartArray();
            if (body.TryGetProperty("messages", out JsonElement messages)
                && messages.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement message in messages.EnumerateArray())
                    message.WriteTo(writer);
            }

            writer.WriteStartObject();
            writer.WriteString("role", "assistant");
            writer.WriteString("content", soFar);
            writer.WriteEndObject();

            writer.WriteStartObject();
            writer.WriteString("role", "user");
            writer.WriteString("content",
                "Your previous answer was cut off by a hardware interruption, not by you. "
                + "Continue it from exactly where it stops. Do not repeat any of it, do not "
                + "start again, and do not mention the interruption.");
            writer.WriteEndObject();
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return JsonDocument.Parse(buffer.ToArray()).RootElement.Clone();
    }

    /// <summary>
    /// The marker on a request that is one turn being finished rather than a new one
    /// being asked for. See <see cref="WithTheAnswerSoFar"/>.
    /// </summary>
    internal const string ResumedTurnMarker = "resumedTurn";

    /// <summary>The token text on a stream frame, or null if it carries none.</summary>
    private static string? TokenIn(object? frame) => PropertyIn(frame, "token", TokenProperties);

    /// <summary>The whole-answer text a <c>replace</c> frame sets, or null if the frame is not one.</summary>
    private static string? ReplaceIn(object? frame) => PropertyIn(frame, "replace", ReplaceProperties);

    /// <summary>
    /// The error text on a stream frame, or null if it carries none.
    ///
    /// <para>
    /// Frames are anonymous types, so this asks each one whether it has an
    /// <c>error</c> property and remembers the answer per type — there are a handful of
    /// shapes in a turn and thousands of frames, and this sits on the path of every
    /// token. Reflection is safe on them here for the same reason the SSE writer works:
    /// these objects are already serialized reflectively one line later.
    /// </para>
    /// </summary>
    private static string? ErrorIn(object? frame) => PropertyIn(frame, "error", ErrorProperties);

    /// <summary>A named string property of a chat frame (an anonymous object), or null.</summary>
    internal static string? StringIn(object? frame, string name) => ValueIn(frame, name) as string;

    /// <summary>A named property of a chat frame, boxed, or null when absent.</summary>
    internal static object? ValueIn(object? frame, string name)
        => frame?.GetType().GetProperty(name)?.GetValue(frame);

    internal static int IntIn(object? frame, string name) => ValueIn(frame, name) switch
    {
        int i => i,
        long l => (int)Math.Clamp(l, int.MinValue, int.MaxValue),
        double d => (int)d,
        _ => 0,
    };

    internal static double DoubleIn(object? frame, string name) => ValueIn(frame, name) switch
    {
        double d => d,
        float f => f,
        int i => i,
        long l => l,
        decimal m => (double)m,
        _ => 0,
    };

    private static readonly Dictionary<Type, System.Reflection.PropertyInfo?> TokenProperties = new();
    private static readonly Dictionary<Type, System.Reflection.PropertyInfo?> ReplaceProperties = new();
    private static readonly Dictionary<Type, System.Reflection.PropertyInfo?> ErrorProperties = new();

    private static string? PropertyIn(
        object? frame, string name, Dictionary<Type, System.Reflection.PropertyInfo?> known)
    {
        if (frame is null)
            return null;
        Type type = frame.GetType();
        System.Reflection.PropertyInfo? property;
        lock (known)
        {
            if (!known.TryGetValue(type, out property))
                known[type] = property = type.GetProperty(name);
        }
        return property?.GetValue(frame) as string;
    }

    /// <summary>
    /// Whether an error message is the one that means the GPU backend is finished for
    /// the rest of the process, rather than an ordinary failure of one turn.
    ///
    /// <para>
    /// Matched on the wording ggml and the OS themselves use: a command buffer that came
    /// back with a status, Metal's own name for the background refusal, and the sticky
    /// flag whose comment says the backend has to be recreated. Anything else — a
    /// refusal, a bad request, running out of context — is a turn's problem, and
    /// rebuilding the engine for it would cost the user a reload to fix nothing.
    /// </para>
    /// </summary>
    internal static string? ReadsLikeAPoisonedEngine(string? error)
    {
        if (error is not { Length: > 0 })
            return null;
        bool terminal =
            error.Contains("recreate the backend", StringComparison.OrdinalIgnoreCase)
            || error.Contains("cannot recover in this process", StringComparison.OrdinalIgnoreCase)
            || error.Contains("backend is in error state", StringComparison.OrdinalIgnoreCase)
            || error.Contains("BackgroundExecutionNotPermitted", StringComparison.OrdinalIgnoreCase)
            || error.Contains("to submit GPU work from background", StringComparison.OrdinalIgnoreCase)
            || error.Contains("victim of GPU error", StringComparison.OrdinalIgnoreCase)
            || (error.Contains("command buffer", StringComparison.OrdinalIgnoreCase)
                && error.Contains("failed with status", StringComparison.OrdinalIgnoreCase));
        return terminal ? error : null;
    }

    /// <summary>
    /// One line, appended to a file, about something that happened while nobody could
    /// see the screen.
    ///
    /// <para>
    /// It exists because of how this class's worst failure is observed. Everything else
    /// the app says goes to stdout, and on a phone stdout is only readable while a
    /// console is attached — which detaches the moment the app is backgrounded, which is
    /// precisely the moment worth reading about. "It broke while I was in another app"
    /// left no record at all. This is that record: small, always on, and pullable off a
    /// device afterwards with <c>devicectl device copy from</c>.
    /// </para>
    /// </summary>
    public void TraceBackground(string line)
    {
        string stamped = $"{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff} {line}";
        Console.WriteLine("TensorAgent: bgtrace " + line);
        try
        {
            string path = Path.Combine(Paths.LogsDirectory, "background.log");
            // Rotated once rather than deleted: the previous megabyte moves aside and
            // the current run keeps its earlier lines, which are exactly the ones a
            // check reads back. Growing without limit on a device the user cannot see
            // it on would be the only worse outcome.
            if (new FileInfo(path) is { Exists: true, Length: > 1024 * 1024 })
                File.Move(path, Path.Combine(Paths.LogsDirectory, "background.1.log"), overwrite: true);
            File.AppendAllText(path, stamped + Environment.NewLine);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth failing anything over.
        }
    }

    /// <summary>
    /// Whether the engine is believed to be in the unrecoverable state a refused GPU
    /// submission leaves behind, and must be rebuilt before it is used again.
    /// </summary>
    public bool EngineNeedsReload { get; private set; }

    /// <summary>
    /// How many times the engine has been thrown away and loaded again after a refused
    /// GPU submission. Zero on a healthy run; a check reports it because
    /// <see cref="EngineNeedsReload"/> is cleared by the repair and so says nothing
    /// afterwards about whether one was needed.
    /// </summary>
    public long EngineRebuilds { get; private set; }

    /// <summary>Raised when <see cref="EngineNeedsReload"/> becomes true.</summary>
    public event Action<string>? EnginePoisoned;

    private int _rebuildsAfterPoisonedWarmUps;

    /// <summary>
    /// How many times a warm-up that met a dead backend has had the engine rebuilt on
    /// the spot rather than leaving it to the next message. See
    /// <see cref="RebuildAfterAPoisonedWarmUp"/>.
    /// </summary>
    public int RebuildsAfterPoisonedWarmUps => Volatile.Read(ref _rebuildsAfterPoisonedWarmUps);

    /// <summary>
    /// The engine is dead and nothing is running: rebuild it now and warm it again,
    /// so the user's next message finds a working, warm engine instead of paying for
    /// the rebuild and the whole prompt itself.
    ///
    /// <para>
    /// Three attempts per launch, and not back to back. On the phone the fault that
    /// reaches a warm-up is a GPU reset — every command buffer for the next minute or so
    /// comes back "victim of GPU error/recovery", whatever backend submits it — so a
    /// rebuild that follows the fault immediately meets the same storm (three warm-ups
    /// faulted in 53 s on three fresh backends, observed). The second and third attempts
    /// therefore wait, and a GPU that is still poisoning warm-ups after that is not one a
    /// loop of reloads will cure: the next message still rebuilds on its own. Not while
    /// the app is away (loading is GPU work; the resume path rebuilds then) and not
    /// beside a turn (the turn rebuilds for itself, and a rebuild under a live
    /// generation is a crash).
    /// </para>
    /// </summary>
    private void RebuildAfterAPoisonedWarmUp()
    {
        int attempt = Interlocked.Increment(ref _rebuildsAfterPoisonedWarmUps);
        if (attempt > RebuildBackoff.Length)
        {
            TraceBackground("the warm-up met a dead backend again; leaving the rebuild to the next message");
            return;
        }
        TimeSpan wait = RebuildBackoff[attempt - 1];
        _ = Task.Run(async () =>
        {
            try
            {
                if (wait > TimeSpan.Zero)
                {
                    TraceBackground($"the warm-up met a dead backend; waiting {wait.TotalSeconds:0}s for the GPU to settle before rebuilding again");
                    await Task.Delay(wait).ConfigureAwait(false);
                }
                if (!Compute.IsOpen || Turns.IsBusy || !EngineNeedsReload)
                {
                    TraceBackground("the warm-up met a dead backend; the app is away, a turn is running, or it has been rebuilt already, so this leaves it");
                    return;
                }
                TraceBackground("the warm-up met a dead backend: rebuilding the engine now, while nothing is running");
                RecoverEngineIfNeeded();
            }
            catch (Exception ex)
            {
                TraceBackground("rebuilding after a poisoned warm-up threw: " + ex.Message);
            }
        });
    }

    /// <summary>How long each proactive rebuild waits before it starts; the length is the attempt cap.</summary>
    private static readonly TimeSpan[] RebuildBackoff =
    [
        TimeSpan.Zero,
        TimeSpan.FromSeconds(15),
        TimeSpan.FromSeconds(60),
    ];

    private void NoteEngineMayBePoisoned(string cause)
    {
        if (EngineNeedsReload)
            return;
        EngineNeedsReload = true;
        TraceBackground("a turn FAILED in a way only a rebuilt engine recovers from: " + cause);
        HostLog.LogWarning("the GPU backend is poisoned and will be rebuilt: {Cause}", cause);
        try { EnginePoisoned?.Invoke(cause); }
        catch (Exception) { /* a listener must not break the turn */ }
    }

    /// <summary>
    /// Rebuild the engine after a refused GPU submission poisoned it.
    ///
    /// <para>
    /// There is nothing gentler available: ggml-metal's error flag is cleared only by
    /// <c>ggml_metal_init</c>, so the backend is torn down and built again and the
    /// model loaded onto the new one. It costs the seconds a load costs, which is why
    /// it happens once — on the way back to the foreground when nothing is running,
    /// or at the start of the next turn otherwise.
    /// </para>
    /// </summary>
    /// <returns>Whether a reload was needed and done.</returns>
    public bool RecoverEngineIfNeeded() => RecoverEngineIfNeeded(warmAfterwards: true);

    /// <param name="warmAfterwards">
    /// Whether to repopulate the prefix cache after the reload. False when a TURN is
    /// driving the recovery: that turn is about to forward the very prompt the warm-up
    /// would forward, so warming would not save it a second and would run a second
    /// generation beside it.
    /// </param>
    public bool RecoverEngineIfNeeded(bool warmAfterwards)
    {
        // Serialized, because two callers race for it by design: the turn that hit the
        // failure rebuilds before answering again, and coming back to the foreground
        // rebuilds too. Whichever arrives second must wait and then find nothing left to
        // do, rather than load the weights a second time on top of the first.
        lock (_recoveryLock)
            return RecoverEngineWhileHoldingTheLock(warmAfterwards);
    }

    private readonly object _recoveryLock = new();

    private bool RecoverEngineWhileHoldingTheLock(bool warmAfterwards)
    {
        if (!EngineNeedsReload)
            return false;

        AppSettings settings = Settings.Load();
        if (settings.SelectedModelId is not { Length: > 0 } id || ModelCatalog.Find(id) is not { } model)
        {
            // Nothing to reload onto. Clear the flag rather than trying forever: with no
            // model selected there is no poisoned engine either.
            EngineNeedsReload = false;
            return false;
        }

        try
        {
            // The order here is the whole repair, and each step is load-bearing.
            //
            // Reloading the weights ALONE does nothing, which took a device to find
            // out: the ggml backend is a process global that a model load never
            // touches, so the rebuilt engine was the same poisoned Metal context with
            // fresh weights in it, and the answer failed again with the same sentence.
            // Only recreating the backend clears ggml-metal's latched has_error.
            //
            // And the model has to go FIRST. Its tensors live in buffers the recreate
            // frees; releasing them afterwards is a crash rather than an error. The
            // warm-up is stopped and waited for before either, because it is a live
            // generation on the weights about to be unmapped.
            StopWarmingThePrefixCacheAndWaitAsync().GetAwaiter().GetResult();
            TraceBackground($"releasing {model.Id} and rebuilding the GPU backend");

            // Under the same lock every other load takes, so a user tapping "Use" on
            // the Models list during the rebuild waits for it rather than loading
            // weights onto a backend that is being freed underneath them. (Monitor is
            // re-entrant, so UseModel below taking it again is fine.)
            lock (_modelGate)
            {
                SetModelLoad(ModelLoadState.Loading, null);
                // Nothing may be inside a graph compute when the backend goes away.
                // The turn that asked for this has stopped reading, but another
                // conversation's turn could still be stepping; the gate is open here, so
                // whatever is in flight finishes or is aborted, and this waits for that.
                WaitForTheEngineToStop();
                if (!ModelService.UnloadModelAndRecreateBackend())
                {
                    TraceBackground("the GPU backend could not be rebuilt; the model is not reloaded");
                    SetModelLoad(ModelLoadState.Failed, "The GPU backend could not be rebuilt.");
                    return false;
                }

                UseModel(model, warmAfterwards);
            }

            // Loading is GPU work too. If the app left during it, the new backend is
            // poisoned before its first token; the flag stays set so the next turn
            // rebuilds again rather than failing on a backend already marked dead.
            if (GgmlBasicOps.HasBackendFailure())
            {
                TraceBackground("the rebuilt backend failed during the reload (the app left again?); it will be rebuilt once more");
                return false;
            }

            EngineNeedsReload = false;
            EngineRebuilds++;
            TraceBackground($"the engine is back: {model.Id} reloaded on a new backend");
            return true;
        }
        catch (Exception ex)
        {
            // Left set on purpose: a reload that failed has not fixed anything, and the
            // next attempt should still try.
            TraceBackground("the recovery reload FAILED: " + ex.Message);
            HostLog.LogWarning(ex, "rebuilding the engine after a GPU fault failed");
            return false;
        }
    }

    // ---- the model the user last used --------------------------------------------

    /// <summary>Where the automatic startup load has got to.</summary>
    public enum ModelLoadState
    {
        /// <summary>No model has ever been chosen, or its files are not on the device.</summary>
        None,
        /// <summary>The weights are being read right now.</summary>
        Loading,
        /// <summary>A model is loaded and the chat can be used.</summary>
        Loaded,
        /// <summary>The load was tried and refused; <see cref="ModelLoadError"/> says why.</summary>
        Failed,
    }

    private int _autoLoadStarted;

    /// <summary>Where the model this app is meant to be using has got to.</summary>
    public ModelLoadState ModelLoad { get; private set; } = ModelLoadState.None;

    /// <summary>Why <see cref="ModelLoad"/> is <see cref="ModelLoadState.Failed"/>, or null.</summary>
    public string? ModelLoadError { get; private set; }

    /// <summary>Raised whenever <see cref="ModelLoad"/> moves, for native chrome that shows it.</summary>
    public event Action<ModelLoadState>? ModelLoadChanged;

    /// <summary>
    /// Load the model the user last used, without being asked.
    ///
    /// <para>
    /// The app remembers the choice — it has always written <c>selectedModelId</c> — and
    /// then did nothing with it until the user went back to the Models list and tapped
    /// "Use" again. Every launch therefore started at "No model yet" with a send button
    /// that refuses, which is the app apparently forgetting a setting it can see. There
    /// is exactly one model the app should be holding, and this is it.
    /// </para>
    /// <para>
    /// On a background thread, because reading four gigabytes of weights off flash and
    /// handing them to Metal takes seconds and the page has to be able to paint while it
    /// happens. It runs at most once per launch, and it does nothing at all when the
    /// selected model's files are not on the device — a purged cache is "download it
    /// again", never an error at startup.
    /// </para>
    /// </summary>
    public void LoadSelectedModelInBackground()
    {
        if (Interlocked.Exchange(ref _autoLoadStarted, 1) != 0)
            return;

        AppSettings settings = Settings.Load();
        if (settings.SelectedModelId is not { Length: > 0 } id)
            return;

        // An id no entry claims any more. Catalogs are re-pointed at better files and
        // the id carries the quantization, so every such swap leaves whoever had the
        // old one selected holding a name that resolves to nothing. Cleared for the
        // same reason the gated-off case below is: a selection nothing can act on is
        // worse than none, because the picker goes on presenting it as the choice.
        if (ModelCatalog.Find(id) is not { } model)
        {
            _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation(
                "the last used model {Model} is no longer in the catalog; clearing the selection", id);
            settings.SelectedModelId = null;
            Settings.Save(settings);
            return;
        }

        // A tier the picker would not offer this device is not one to load behind the
        // user's back either. Find() searches the WHOLE catalog while the Models list is
        // built from ForDevice(), so an entry that was offered when it was chosen -- or
        // that came from a backup of a larger device -- would otherwise be loaded on a
        // phone that cannot run it, with no row in the list to explain or undo it. The
        // choice is cleared rather than merely skipped, so the picker starts clean.
        if (model.MinDeviceMemoryGB > Paths.DeviceMemoryGB)
        {
            _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation(
                "{Model} needs a {Needs} GB device and this one reports {Has} GB; clearing the selection",
                model.Id, model.MinDeviceMemoryGB, Paths.DeviceMemoryGB);
            settings.SelectedModelId = null;
            Settings.Save(settings);
            return;
        }

        if (!File.Exists(Paths.SelectedModelPath(settings)))
        {
            _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation(
                "the last used model {Model} is not on this device; nothing to load", id);
            return;
        }

        SetModelLoad(ModelLoadState.Loading, null);
        _ = Task.Run(() =>
        {
            try
            {
                Console.WriteLine($"TensorAgent: loaded the last used model {model.Id} on {UseModel(model)}");
            }
            catch (Exception ex)
            {
                _loggerFactory.CreateLogger("TensorAgent.Host")
                    .LogWarning(ex, "the last used model {Model} could not be loaded", model.Id);
            }
        });
    }

    private void SetModelLoad(ModelLoadState state, string? error)
    {
        ModelLoad = state;
        ModelLoadError = error;
        try { ModelLoadChanged?.Invoke(state); }
        catch (Exception) { /* a listener must not break the load */ }
    }

    /// <summary>
    /// What the page shows while the weights are still being read: which model, and how
    /// far it has got. Without it the chat says "No model yet" for the whole of a load
    /// that is going perfectly well, and the send button refuses for reasons the user
    /// cannot see.
    /// </summary>
    private object DescribeModelState()
    {
        AppSettings settings = Settings.Load();
        CatalogModel? model = settings.SelectedModelId is { Length: > 0 } id ? ModelCatalog.Find(id) : null;
        return new
        {
            id = model?.Id,
            name = model?.DisplayName,
            state = ModelLoad.ToString(),
            loading = ModelLoad == ModelLoadState.Loading,
            error = ModelLoadError,
            // Whether the first message will be answered as fast as the second. See
            // WarmThePrefixCache; a probe waits on this rather than measuring the cold
            // path it exists to remove.
            prefixCacheWarm = PrefixCacheIsWarm,
            // The numbers a kill is decided on, so a probe (and a bench) can read them
            // without a console attached. See ProcessMemory.
            memory = ProcessMemory.Describe(),
        };
    }

    private Timer? _memoryTrace;

    /// <summary>
    /// While a turn runs, one memory line every half minute. A turn on a phone is
    /// minutes of tool rounds, and a kill in the middle of one leaves no report of its
    /// own often enough; the trace is what says how much the process was charged and
    /// how much the device had left at the last half-minute before it stopped.
    /// </summary>
    private void StartMemoryTrace()
    {
        Timer? previous = Interlocked.Exchange(ref _memoryTrace, new Timer(
            _ => LogMemory("during the turn"), null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30)));
        previous?.Dispose();
    }

    private void StopMemoryTrace() => Interlocked.Exchange(ref _memoryTrace, null)?.Dispose();

    /// <summary>
    /// Write one memory line to the console and the host log: what this process is
    /// charged, what the device has wired and free. The label says which moment.
    /// </summary>
    private void LogMemory(string moment)
    {
        try
        {
            string line = $"memory {moment} -- {ProcessMemory.Describe()}";
            Console.WriteLine("TensorAgent: " + line);
            HostLog.LogInformation("{Line}", line);
        }
        catch (Exception)
        {
            // A memory line must never be what breaks the moment it describes.
        }
    }

    /// <summary>
    /// The system has asked for memory back; give what can be given without touching
    /// the turn in progress.
    ///
    /// <para>
    /// Until this existed the warning was only logged, on the grounds that the weights
    /// are a mapping the engine reads and the K/V cache belongs to a generation that
    /// may be mid-token. Both remain true, and both are untouched here. What IS free
    /// to go is everything the engine keeps only so that the NEXT request is faster:
    /// finished conversations' caches beyond the newest (each a whole window of K/V,
    /// paid twice on Metal), holders parked for reuse, the host memory pool's spare
    /// blocks. The engine releases them on its own thread between steps, so a forward
    /// in flight is never underneath the free; this only queues the request. Then the
    /// managed heap is collected, which returns what the transcript, the page's frames
    /// and the tool outputs of a long agentic turn left behind.
    /// </para>
    /// <para>
    /// It cannot promise survival -- a page shortage can kill the process before any
    /// of this runs -- which is why the budget the engine is given up front
    /// (EngineMemoryPolicy) is the real defence and this is the second line.
    /// </para>
    /// </summary>
    public void RelieveMemoryPressure()
    {
        LogMemory("at the memory warning");
        // Off the caller's thread: iOS delivers the warning on the UI thread, and a
        // blocking collection there -- or a wait on an engine that is mid-swap -- is
        // how an app earns a watchdog kill on top of a memory one. The request itself
        // is queued to the engine and returns at once; the collection is what takes time.
        _ = Task.Run(() =>
        {
            bool asked = false;
            try
            {
                asked = ModelService.TrimIdleMemory();
            }
            catch (Exception ex)
            {
                HostLog.LogWarning(ex, "asking the engine to trim idle memory failed");
            }
            try
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            catch (Exception)
            {
                // Nothing to do about a collector that will not run; the log below says so.
            }
            HostLog.LogInformation(
                asked ? "memory warning: the engine was asked to release idle caches; managed heap collected"
                      : "memory warning: the engine was busy loading or not built, so nothing was asked of it; managed heap collected");
            LogMemory("after the memory warning was acted on");
        });
    }

    /// <summary>
    /// Take a settings change now rather than at the next launch.
    ///
    /// <para>
    /// "Allow network access" is the one that made this necessary. It is a switch on a
    /// settings screen, and the page under it said the change would apply the next time
    /// the app started — but an iPhone app is not restarted by leaving it, so the honest
    /// instruction was "force-quit TensorAgent from the app switcher", which nobody
    /// does. The reported symptom is exactly what that produces: network turned on,
    /// <c>curl</c> still answering "network access is disabled by the user".
    /// </para>
    /// <para>
    /// Everything a run's permissions are read from is moved here, in one place, so the
    /// four holders cannot drift apart: the code runner's options, the installer's
    /// standing policy, the shell's host allow-list, and the terms a skill's scripts are
    /// planned against. A command already running keeps the terms it started with.
    /// </para>
    /// </summary>
    public void ApplySettings(AppSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        CodeExec.Enabled = settings.AllowCodeExecution;
        CodeExec.AllowNetwork = settings.AllowNetwork;
        CodeExec.AllowInstall = settings.AllowNetwork;
        CodeExec.Timeout = TimeSpan.FromSeconds(Math.Clamp(settings.ToolTimeoutSeconds, 5, 600));
        CodeExec.InstallTimeout = CodeExec.Timeout;
        // The reply length limit is read by /api/models, which the page re-reads every
        // time it comes back to the front, so moving it here is all it takes for the
        // stepper to mean something before the next launch.
        Options.RepointGenerationDefaults(settings.MaxTokens);

        Backend.NetworkHosts = settings.NetworkHosts;
        if (Installer is WheelInstaller wheels)
        {
            wheels.Policy = wheels.Policy with
            {
                AllowScripts = settings.AllowCodeExecution,
                AllowNetwork = settings.AllowNetwork,
                NetworkHosts = settings.NetworkHosts,
            };
        }
        Backend.HostPerformsInstalls = CodeExec.AllowInstall && Installer is { CanInstall: true };

        Options.RepointSandboxPermissions(settings.AllowCodeExecution, settings.AllowNetwork);
        Options.RepointSkills(settings.SkillsEnabled);
        ApplySpeculationSetting(settings);
        _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation("settings applied: {Engine}", DescribeEngine());
    }

    /// <summary>
    /// The speculative-decoding switch, applied to the engine that is standing: the
    /// policy goes into the environment (where the next engine reads it) AND to the
    /// current engine, which drops its armed drafters and follows the new policy on
    /// the next turn. Without the second half the switch would only mean something
    /// at the next model load, which is exactly the kind of setting nobody can tell
    /// is inert. The draft head is the selected model's, when it is downloaded.
    /// </summary>
    /// <summary>True when the loaded model carries a usable draft head.</summary>
    public bool DraftHeadAttached => ModelService.Model is IDraftHead { HasDraftHead: true };

    /// <summary>
    /// Switch speculation on or off for the running engine without touching the saved
    /// settings - what the on-device benchmark does between its passes. Returns a
    /// one-line account for the log.
    /// </summary>
    internal string SetSpeculationEnabled(bool enabled)
    {
        AppSettings settings = Settings.Load();
        settings.SpeculativeDecoding = enabled;
        return ApplySpeculationSetting(settings);
    }

    internal string ApplySpeculationSetting(AppSettings settings)
    {
        CatalogModel? model = settings.SelectedModelId is { Length: > 0 } id ? ModelCatalog.Find(id) : null;
        string? draftHead = model is null ? null : Models.CompanionPath(model, CatalogFileRole.Draft);
        string note = SpeculationPolicy.PrepareLoad(settings, draftHead);
        bool draftAttached = ModelService.Model is IDraftHead { HasDraftHead: true };
        string algorithm = SpeculationPolicy.ChooseAlgorithm(draftAttached);
        bool live = ModelService.EngineHost.UpdateSpeculation(SpeculationOptions.FromEnvironment());
        string account = $"{note}; algorithm {algorithm}; {(live ? "applied to the running engine" : "no engine standing, applies at the next load")}";
        HostLog.LogInformation("{Speculation}", account);
        return account;
    }

    private int _speculationBenchStarted;

    /// <summary>Start the on-device plain-vs-speculative benchmark once, when the
    /// launch environment asks for it (see <see cref="SpeculationBench"/>).</summary>
    private void StartSpeculationBenchIfRequested()
    {
        if (!SpeculationBench.Requested || Interlocked.Exchange(ref _speculationBenchStarted, 1) != 0)
            return;
        _ = Task.Run(() => new SpeculationBench(this).RunAsync(CancellationToken.None));
    }

    /// <summary>
    /// One line naming what will actually run a command and what will run a script,
    /// for the status bar and for <c>/api/agent/engine</c>. A user who is told
    /// "python is not available" before writing a script is better served than one
    /// who finds out from a failed run.
    /// </summary>
    public string DescribeEngine()
    {
        var parts = new List<string> { Backend.Describe() };
        // Asked of the runner rather than of the field, because the runner is always
        // there now and it is its answer -- read live from CodeExec.Enabled -- that
        // decides whether the model is offered the tools at all.
        parts.Add(CodeRunner is { CanRun: true } ? "code execution on" : "code execution off");
        parts.Add(CodeExec.AllowNetwork ? "network on" : "network off");
        parts.Add($"{Skills.Skills.Count} skills");
        return string.Join(" · ", parts);
    }

    /// <summary>
    /// Run a handful of representative commands through the real backend and report
    /// what each did.
    ///
    /// <para>
    /// It exists because the interesting failures on this platform are not compile
    /// errors. An interpreter that links but cannot find its standard library, a
    /// sandbox that refuses a path it should allow, a shell builtin that behaves
    /// differently under ahead-of-time compilation — all of those produce an app that
    /// starts perfectly and then fails the first time a model tries to do anything.
    /// This turns that into a line in the launch log.
    /// </para>
    /// <para>
    /// It runs the checks under the app's real policy, in a throwaway directory, and
    /// it never runs on its own: the caller decides. Nothing here writes outside that
    /// directory, and the network check is expected to be refused.
    /// </para>
    /// </summary>
    public IReadOnlyList<SelfTestResult> SelfTest()
    {
        string root = Path.Combine(Paths.ScratchDirectory, "selftest-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            return new[]
            {
                Check("shell", new[] { "sh", "-c", "echo hello | tr a-z A-Z" }, root, "HELLO"),
                Check("shell:files", new[] { "sh", "-c", "printf 'b\na\n' > f.txt && sort f.txt | tr -d '\n'" }, root, "ab"),
                Check("shell:awk", new[] { "sh", "-c", "echo 'x 2' | awk '{print $2*3}'" }, root, "6"),
                Check("python", new[] { "python3", "-c", "import sys, json; print(json.dumps({'v': sys.version_info[:2]}))" }, root, "[3, 13]"),
                Check("python:stdlib", new[] { "python3", "-c", "import re, zipfile, sqlite3; print('stdlib ok')" }, root, "stdlib ok"),
                // The packages the bundled skills import. A staged wheel whose compiled
                // extension did not make it into the bundle imports fine on a laptop
                // and fails here, which is exactly the failure this catches.
                Check("python:numpy", new[] { "python3", "-c", "import numpy; print(numpy.arange(3).sum())" }, root, "3"),
                Check("python:pillow", new[] { "python3", "-c", "from PIL import Image; print(Image.new('RGB', (2, 2)).size)" }, root, "(2, 2)"),
                // lxml is the one this repository compiles itself (scripts/build-lxml-ios.sh):
                // seven frameworks that link libxml2 and libxslt statically. Parsing,
                // XPath and an XSLT transform touch all of etree's linkage at once.
                Check("python:lxml", new[] { "python3", "-c",
                    "from lxml import etree; d = etree.XML('<r><a n=\"1\"/><a n=\"2\"/></r>'); "
                    + "x = etree.XSLT(etree.XML('<xsl:stylesheet xmlns:xsl=\"http://www.w3.org/1999/XSL/Transform\" version=\"1.0\"><xsl:template match=\"/\"><o><xsl:value-of select=\"count(//a)\"/></o></xsl:template></xsl:stylesheet>')); "
                    + "print(d.xpath('sum(//a/@n)'), etree.tostring(x(d)).decode())" }, root, "3.0 <o>2</o>"),
                // And the two document libraries that exist only because lxml does. Each
                // writes a file and reads it back, which is the whole of what a model
                // asks of them.
                Check("python:pptx", new[] { "python3", "-c",
                    "from pptx import Presentation; p = Presentation(); s = p.slides.add_slide(p.slide_layouts[5]); "
                    + "s.shapes.title.text = 'ok'; p.save('deck.pptx'); print(len(Presentation('deck.pptx').slides))" }, root, "1"),
                Check("python:docx", new[] { "python3", "-c",
                    "import docx; d = docx.Document(); d.add_paragraph('ok'); d.save('note.docx'); "
                    + "print(len(docx.Document('note.docx').paragraphs))" }, root, "1"),
                Check("node", new[] { "node", "-e", "console.log([1,2,3].map(n => n * 2).join(','))" }, root, "2,4,6"),
                Check("node:print", new[] { "node", "-p", "1 + 1" }, root, "2"),
                Check("sandbox:write", new[] { "sh", "-c", "echo x > /tmp/tensoragent-selftest-escape" }, root, expectFailure: true),
                Check("sandbox:network", new[] { "sh", "-c", "curl https://example.com" }, root, expectFailure: true),
            };
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
        }
    }

    private SelfTestResult Check(string name, string[] argv, string root, string? expected = null, bool expectFailure = false)
    {
        try
        {
            ConfinedResult result = ((IShellBackend)Backend).Run(new ShellLaunch
            {
                Argv = argv,
                WorkingDirectory = root,
                WriteDirectory = root,
                ReadOnlyDirectory = root,
                AllowNetwork = false,
                Timeout = TimeSpan.FromSeconds(30),
            });

            string output = (result.Stdout + result.Stderr).Trim();
            bool ok = expectFailure
                ? !result.Ok
                : result.Ok && (expected is null || output.Contains(expected, StringComparison.Ordinal));

            // A Python traceback puts the useful sentence last and the useless frames
            // first, so a failure is reported from the end. A one-line log entry that
            // says "Traceback (most recent call last):" and nothing else is worthless.
            string detail = ok || !output.Contains('\n')
                ? output
                : output[(output.LastIndexOf('\n') + 1)..];
            return new SelfTestResult(name, ok, detail.Length > 240 ? detail[..240] + "…" : detail);
        }
        catch (Exception ex)
        {
            return new SelfTestResult(name, false, ex.Message);
        }
    }

    /// <summary>
    /// Build a runtime, and treat "it could not be built" as an absence rather than
    /// as a crash. A missing framework, a bundle staged without its interpreter or a
    /// platform that has neither must cost the app its scripting, not its startup.
    /// </summary>
    private T? Discover<T>(Func<T> build) where T : class
    {
        try
        {
            return build();
        }
        catch (Exception ex)
        {
            _loggerFactory.CreateLogger("TensorAgent.Runtimes")
                .LogWarning(ex, "{Runtime} is not available on this host", typeof(T).Name);
            return null;
        }
    }

    /// <summary>
    /// Tell the shared agent layer about runtimes that have no executable on PATH.
    /// Syntax checks, API probes and skill-script planning all resolve interpreters
    /// through <see cref="CodeEnvironment"/>; without this call they silently probe the
    /// phone's empty process environment and skip work the embedded runtimes can do.
    /// </summary>
    private static void ConfigureCodeEnvironment(
        IPythonRuntime? python, IJavaScriptRuntime? javaScript)
    {
        bool hasPython = python is { IsAvailable: true };
        bool hasJavaScript = javaScript is { IsAvailable: true };
        string pythonVersionText = hasPython ? python!.Version : string.Empty;

        var available = new List<string>();
        if (hasPython)
            available.Add($"python3 {pythonVersionText}");
        if (hasJavaScript)
            available.Add("node (JavaScriptCore)");

        string numericVersion = new(
            pythonVersionText.TakeWhile(character => char.IsAsciiDigit(character) || character == '.').ToArray());
        numericVersion = numericVersion.TrimEnd('.');
        Version? pythonVersion = Version.TryParse(numericVersion, out Version? parsed) ? parsed : null;

        CodeEnvironment.Configure(
            available,
            language => language switch
            {
                CodeLanguage.Python when hasPython => "python3",
                CodeLanguage.JavaScript when hasJavaScript => "node",
                _ => null,
            },
            interpreter => Path.GetFileName(interpreter).StartsWith("python", StringComparison.OrdinalIgnoreCase)
                ? pythonVersion
                : null);
    }

    /// <summary>
    /// The backends this build can actually offer, best first.
    ///
    /// <para>
    /// The default names Metal first because that is the point of running on a phone.
    /// It is a parameter rather than a constant because it is not always true: the
    /// simulator's slice of the engine has no Metal at all, and a page whose default
    /// backend does not exist puts the user one tap from a load that fails. The iOS
    /// head passes what its probe found.
    /// </para>
    /// </summary>
    public static IReadOnlyList<BackendOption> DefaultBackends { get; } = new[]
    {
        new BackendOption("ggml_metal", "GPU (Metal)"),
        new BackendOption("ggml_cpu", "CPU"),
    };

    /// <summary>The catalog entry's card values as the defaults a request falls back to.</summary>
    private static SamplingDefaults SamplingDefaultsFor(CatalogModel model) =>
        new(new TensorSharp.Runtime.SamplingConfig
        {
            Temperature = model.Sampling.Temperature,
            TopK = model.Sampling.TopK,
            TopP = model.Sampling.TopP,
            MinP = model.Sampling.MinP,
        });

    private static ServerHostingOptions BuildOptions(
        AgentPaths paths, AppSettings settings, IReadOnlyList<BackendOption>? backends) => new(
        startupModelPath: paths.SelectedModelPath(settings),
        startupMmProjPath: paths.SelectedProjectorPath(settings),
        defaultBackend: (backends ?? DefaultBackends)[0].Value,
        supportedBackends: backends ?? DefaultBackends,
        defaultMaxTokens: settings.MaxTokens,
        maxTokensPinned: false,
        defaultVideoFrames: 0,
        defaultVideoFps: 0,
        defaultVideoWidth: 0,
        defaultVideoHeight: 0,
        defaultVideoSteps: 0,
        defaultVideoMode: null,
        uploadDirectory: paths.UploadsDirectory,
        logDirectory: paths.LogsDirectory,
        fileLoggingEnabled: true,
        samplingDefaults: null,
        // A skill that cannot run its own scripts is a document, not a skill: the
        // bundled ones are chosen precisely because they do work end to end. What
        // gates them is the user's own switch, read here, not a build-time default.
        skillsEnabled: settings.SkillsEnabled,
        skillsDiscovery: true,
        skillsAllowScripts: settings.AllowCodeExecution,
        // Zero means "the operator did not choose a cap". The agent plan then uses
        // SkillHostOptions.CodeExecutionRounds (24) when code is offered. Omitting this
        // argument selects the constructor's literal default of 8 and marks it as an
        // explicit cap, which cut Qwen off while it was still retrieving the answer.
        skillsMaxRounds: 0,
        skillsAllowNetwork: settings.AllowNetwork);

    /// <summary>
    /// Wait until the engine has nothing in flight.
    ///
    /// <para>
    /// Stopping the server ends the HTTP requests, but a generation those requests
    /// started can still be inside a graph compute on the engine's own threads —
    /// cancellation is delivered between tokens, and a token can take a while.
    /// Releasing the model at that moment unmaps weights those threads are reading,
    /// and the process dies with a segmentation fault in whichever kernel happened to
    /// be running. The engine reports what it is processing; this waits for that to
    /// reach zero, with a cap so a wedged request cannot stop the app from closing.
    /// </para>
    /// </summary>
    /// <summary>
    /// Make <paramref name="model"/> the model this app is using, now.
    ///
    /// <para>
    /// The engine enforces one hosted model per process because the desktop server is
    /// launched against one <c>--model</c>. An app is not: its user picks from a list,
    /// and until this existed the pick only took effect on the next launch -- so the
    /// Models page said "selected" while the chat said "No model is configured", which
    /// is indistinguishable from a broken button. This saves the choice, moves the
    /// guard, and loads the weights, in that order.
    /// </para>
    /// <para>
    /// Backends are tried best-first and a refusal is reported rather than swallowed:
    /// a phone that silently fell back to the CPU would look like the model loading
    /// very slowly rather than like Metal being unavailable.
    /// </para>
    /// </summary>
    /// <param name="model">The catalog entry to use. Its files must already be installed.</param>
    /// <returns>The backend that answered.</returns>
    public string UseModel(CatalogModel model) => UseModel(model, warmAfterwards: true);

    /// <summary>
    /// Remove a model's files and everything kept about it: the weights (and a partial
    /// download), and the shared-prefix checkpoints saved for it, which are its state
    /// and are useless without it.
    /// </summary>
    public void DeleteModel(CatalogModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        Models.Delete(model);
        new PrefixCheckpointFileStore(Paths.PrefixCheckpointDirectoryFor(model), log: HostLog).Clear();
    }

    /// <param name="warmAfterwards">
    /// Whether to forward the shared prompt afterwards so the next message reuses it
    /// (see <see cref="WarmThePrefixCache"/>). False for a caller that is itself about
    /// to generate.
    /// </param>
    public string UseModel(CatalogModel model, bool warmAfterwards)
    {
        ArgumentNullException.ThrowIfNull(model);

        // Whatever was being warmed belongs to weights that are about to be unmapped.
        // First, because the warm-up submits graphs and this is about to free the
        // buffers they run on.
        StopWarmingThePrefixCacheAndWaitAsync().GetAwaiter().GetResult();
        string? loaded = null;

        // One load at a time, whoever asked. There are three callers now — the startup
        // load, the Models list, and the device hook — and the startup one takes twenty
        // seconds on real weights, which is exactly the window in which a user who sees
        // "No model yet" goes to the list and taps Use. Two threads inside the engine's
        // load at once free and remap the same buffers; the failure is a segmentation
        // fault in a kernel with nothing to do with either of them.
        lock (_modelGate)
        {
            AppSettings settings = Settings.Load();
            settings.SelectedModelId = model.Id;
            Settings.Save(settings);
            string weights = Paths.SelectedModelPath(settings);

            // The model that is asked for is the one already standing: nothing to load.
            // Two callers reach here for the same model at launch -- the startup load of
            // the remembered choice and the device hook -- and the second used to reload
            // it: twenty seconds of the user's time, and a swap of the weights under the
            // turn the first load had already let through. Only a load that FINISHED is
            // trusted; a failed one is retried by loading again.
            if (ModelLoad == ModelLoadState.Loaded
                && string.Equals(ModelService.LoadedModelPath, weights, StringComparison.Ordinal)
                && ModelService.LoadedBackend is { Length: > 0 } standing)
            {
                HostLog.LogInformation("{Model} is already loaded on {Backend}; not loading it again", model.Id, standing);
                loaded = standing;
            }
            else
            {
                SetModelLoad(ModelLoadState.Loading, null);

                // Loading is GPU work too -- ggml_metal_init and the load's own warm-up
                // graph -- and it is refused from the background exactly as a token is.
                // The startup load begins a few seconds BEFORE UIKit calls the process
                // active, which on the phone produced a backend poisoned before its first
                // token: the warm-up failed and the user's first message opened on "backend
                // is in error state". So a load holds here until the app is in front.
                // Bounded, because every caller is off the UI thread but the wait must never
                // become a way for a stuck gate to make "Use" hang for ever.
                if (!Compute.IsOpen)
                {
                    TraceBackground("holding the model load until the app is in front");
                    try { Compute.Wait(new CancellationTokenSource(TimeSpan.FromMinutes(2)).Token); }
                    catch (OperationCanceledException)
                    {
                        TraceBackground("the app did not come to the front within two minutes; loading anyway");
                    }
                }

                if (!File.Exists(weights))
                {
                    var missing = new FileNotFoundException($"{model.DisplayName} is not downloaded yet.", weights);
                    SetModelLoad(ModelLoadState.Failed, missing.Message);
                    throw missing;
                }

                // A filename existing is not enough: an interrupted optional download can
                // leave a truncated destination behind. Only a catalog-size-complete
                // projector is safe to hand to the engine. Optional projectors may be
                // absent for text-only use; required ones make the install incomplete.
                string? projector = Models.CompanionPath(model, CatalogFileRole.Projector);
                if (model.Projector is { Optional: false } requiredProjector && projector is null)
                {
                    string path = Models.PathFor(model, requiredProjector);
                    var missing = new FileNotFoundException(
                        $"{model.DisplayName}'s image projector is not downloaded yet.", path);
                    SetModelLoad(ModelLoadState.Failed, missing.Message);
                    throw missing;
                }

                // Before the load, not after: the engine reads its context length and KV
                // dtype when the model is constructed. This is the only funnel for a load
                // (startup, the Models list, the device hook all arrive here), which is why
                // it is the right place for the budget. See EngineMemoryPolicy for the
                // measurements -- this is what stops a pasted document from growing the KV
                // cache until jetsam kills the app.
                // A load releases the model that is standing, and a turn may be running on
                // it: the engine's threads read weights that LoadModel is about to unmap, and
                // the request pipeline holds tensors that the model's disposal frees. The
                // phone showed the result -- a segmentation fault in managed code the moment
                // the second load landed under the first turn. So the turns are stopped
                // (recorded as stopped, like the Stop button) and the engine is drained
                // before the swap, the same order the host's own shutdown uses.
                if (Turns.IsBusy)
                {
                    HostLog.LogWarning("loading {Model} stops the turn in progress", model.Id);
                    Turns.StopAll();
                }
                WaitForTheEngineToStop();

                EngineMemoryPolicy.Apply(model, settings);

                // Speculative decoding, and the draft head that makes it best: the
                // catalog's optional companion, handed to the loader the way the CLI's
                // --draft-model is. Before the load for the same reason as the budget.
                string? draftHead = Models.CompanionPath(model, CatalogFileRole.Draft);
                HostLog.LogInformation("{Model}: {Speculation}", model.Id, SpeculationPolicy.PrepareLoad(settings, draftHead));

                // The entry's own card values, for the same reason and in the same place.
                // CatalogModel.Sampling was written for every entry and read by nothing, so
                // every model was sampled at the built-in Ollama-compatible default
                // (temperature 0.8, top-k 40, top-p 0.9) whatever its card said -- Gemma 4
                // asks for 1.0 / 64 / 0.95 and Qwen for 0.7 / 20 / 0.8. A wrong sampler does
                // not fail, it just answers worse, which is the hardest kind of setting to
                // notice is inert.
                Options.RepointSamplingDefaults(SamplingDefaultsFor(model));

                Options.RepointHostedModel(weights, projector);

                // Where this model's shared-prefix checkpoint outlives the process: set BEFORE
                // the load, so the engine built for it is born with it, and named by these
                // weights, so a file made from other weights of the same shape is never
                // restored. The warm-up that follows reads it back instead of prefilling it,
                // and a first message sent before the warm-up finds it too.
                ModelService.EngineHost.PrefixCheckpointStore = new PrefixCheckpointFileStore(
                    Paths.PrefixCheckpointDirectoryFor(model),
                    PrefixCheckpointFileStore.WeightsIdentityOf(weights, projector),
                    HostLog);

                var refusals = new List<string>();
                foreach (BackendOption backend in Options.SupportedBackends)
                {
                    try
                    {
                        ModelService.LoadModel(weights, projector, backend.Value);
                        // The engine is built after this, so the algorithm it reads is
                        // decided here, from whether the draft head really attached.
                        bool draftAttached = ModelService.Model is IDraftHead { HasDraftHead: true };
                        string algorithm = SpeculationPolicy.ChooseAlgorithm(draftAttached);
                        if (draftHead is not null && !draftAttached)
                            HostLog.LogWarning("{Model}: the draft head {File} did not attach ({Reason}); speculating with {Algorithm} instead",
                                model.Id, Path.GetFileName(draftHead), ModelService.DraftHeadActivationError ?? "no reason given", algorithm);
                        _loggerFactory.CreateLogger("TensorAgent.Host").LogInformation(
                            "using {Model} on {Backend} (speculation: {Algorithm})", model.Id, backend.Value, algorithm);
                        LogMemory($"after loading {model.Id}");
                        SetModelLoad(ModelLoadState.Loaded, null);
                        loaded = backend.Value;
                        break;
                    }
                    catch (Exception ex)
                    {
                        refusals.Add($"{backend.Value}: {ex.Message}");
                    }
                }

                if (loaded is null)
                {
                    var refused = new InvalidOperationException(
                        $"{model.DisplayName} could not be loaded on any backend this build offers:"
                        + Environment.NewLine + "  " + string.Join(Environment.NewLine + "  ", refusals));
                    SetModelLoad(ModelLoadState.Failed, refused.Message);
                    throw refused;
                }
            }
        }

        // OUTSIDE the lock, and a device crash is the reason that is spelled out here.
        // Warming the cache submits graphs, and starting it from inside the load put
        // those graphs on the engine at the same moment the load was still finishing
        // with it: "ggml_metal_buffer_map: error: failed to allocate buffer" and then a
        // native fault, on every launch. The comment at the top of this lock already
        // said what happens when two threads are inside the engine's load at once; this
        // was the same mistake wearing a different hat.
        if (warmAfterwards)
            WarmThePrefixCache();
        StartSpeculationBenchIfRequested();
        return loaded;
    }

    private readonly object _modelGate = new();

    /// <summary>
    /// What <see cref="WaitForTheEngineToStop"/> polls, and the only thing that decides
    /// how long the shutdown waits. Replaceable so a test can hold the shutdown open
    /// deterministically instead of racing a real engine.
    /// </summary>
    internal Func<bool> EngineHasWorkInFlight { get; set; } = () => false;

    /// <summary>
    /// Whether the engine is still working, straight from its own counters. A component
    /// that cannot say what it is doing is not a reason to keep the app open; the wait
    /// is a precaution, not a contract, so an engine that will not answer reads as idle.
    /// </summary>
    private bool EngineIsProcessing()
    {
        try
        {
            if (!ModelService.EngineHost.TryGetLiveStats(out int processing, out int waiting, out _))
                return false;
            return processing > 0 || waiting > 0;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private void WaitForTheEngineToStop()
    {
        var deadline = Stopwatch.StartNew();
        while (deadline.Elapsed < EngineDrainTimeout)
        {
            if (!EngineHasWorkInFlight())
                return;
            Thread.Sleep(25);
        }
        _loggerFactory.CreateLogger("TensorAgent.Host")
            .LogWarning("the engine was still working after {Seconds}s; releasing the model anyway",
                EngineDrainTimeout.TotalSeconds);
    }

    /// <summary>How long <see cref="Dispose"/> waits for the engine before releasing the model regardless.</summary>
    public static TimeSpan EngineDrainTimeout { get; set; } = TimeSpan.FromSeconds(30);

    private static int _exitHookInstalled;

    /// <summary>
    /// Hand the GPU back before the C runtime tears itself down, which is the one
    /// piece of shutdown <see cref="Dispose"/> cannot do.
    ///
    /// <para>
    /// ggml-metal's device is a C++ static whose destructor asserts that every
    /// residency set has been given back, and that destructor runs from
    /// <c>__cxa_finalize</c> — after the last managed code. A user who closes the app
    /// without unloading first (which is every user) therefore leaves the loaded
    /// model's buffers registered, and the process aborts on
    /// <c>GGML_ASSERT([rsets-&gt;data count] == 0)</c> instead of exiting. Disposing
    /// this host releases them, but nothing guarantees anyone disposes it.
    /// </para>
    /// <para>
    /// So: the same wiring the desktop server and the CLI already have
    /// (TensorSharp.Server/Program.cs registers both ApplicationStopped and
    /// ProcessExit), for the same reason. The call is idempotent and does nothing
    /// when no GGML backend was ever initialised.
    /// </para>
    /// <para>
    /// The APP calls this, not the constructor. A net that catches an undisposed
    /// engine also hides one, and a test process that acquired it merely by building
    /// a host would stop aborting on exactly the leak this exists to survive — which
    /// is how the leak went unnoticed in the first place. Tests build hosts and get
    /// no net; the app installs it deliberately, once, at startup.
    /// </para>
    /// </summary>
    public static void ReleaseTheEngineWhenTheProcessExits()
    {
        if (Interlocked.Exchange(ref _exitHookInstalled, 1) != 0)
            return;

        AppDomain.CurrentDomain.ProcessExit += static (_, _) =>
        {
            try
            {
                GgmlBasicOps.Shutdown();
            }
            catch (DllNotFoundException)
            {
                // No GgmlOps in this build, so there is no device holding anything
                // and nothing to release. Not a fallback: the engine that would need
                // shutting down was never there.
            }
            catch (EntryPointNotFoundException)
            {
                // An older GgmlOps without the entry point. Same conclusion, and the
                // process is already on its way out; there is nowhere left to report.
            }
        };
    }

    /// <summary>
    /// Shut down in the only order that is safe: the server first, then the engine.
    ///
    /// <para>
    /// The server owns the requests, and a request in flight is very often inside the
    /// model. Freeing the model first hands the native compute threads memory that has
    /// been unmapped underneath them, and the process dies with a segmentation fault
    /// in whichever kernel happened to be reading. Stopping the server waits for those
    /// requests to finish, so by the time the model is released nothing is using it.
    /// </para>
    /// </summary>
    public void Dispose()
    {
        Shares.Acknowledging -= OnShareAcknowledged;
        StopMemoryTrace();

        // Before the turns are asked to stop, because a turn parked on a closed gate --
        // and an engine step loop parked on it -- is not running and cannot notice that
        // it should stop; shutting down while the app is not frontmost is the ordinary
        // case, not the odd one. Cancellation does release the waiters on its own, but
        // opening costs nothing and removes the question.
        Compute.Open();

        // A warm-up still waiting to start would otherwise begin a generation on an
        // engine that is being torn down. Cancelled AND waited for, because
        // cancellation is cooperative (see StopWarmingThePrefixCacheAndWaitAsync).
        StopWarmingThePrefixCacheAndWaitAsync().GetAwaiter().GetResult();

        // And this is now load-bearing: a turn no longer stops when its
        // reader goes away, so closing the server is not enough to end one. Asking the
        // turns to stop is what lets WaitForTheEngineToStop below ever return -- and
        // that wait is the only thing between a generation on the engine's threads and
        // the weights being unmapped underneath it.
        Turns.StopAll();
        Close(Server);
        // Before the engine wait, because a download holds a socket and a file handle
        // and has nothing to do with the model: making a five-second transfer teardown
        // wait behind a generation would be for no reason.
        Close(Downloads);
        WaitForTheEngineToStop();
        Close(Turns);
        Close(ModelService);
        foreach (IDisposable owned in _owned)
            Close(owned);
        _owned.Clear();

        static void Close(IDisposable owned)
        {
            try { owned.Dispose(); }
            catch (Exception) { /* teardown is best effort; the process is going away */ }
        }
    }
}

/// <summary>One self-test check: what was tried, whether it behaved, and what it said.</summary>
public sealed record SelfTestResult(string Name, bool Ok, string Detail)
{
    public override string ToString() => $"{(Ok ? "ok  " : "FAIL")} {Name}: {Detail}";
}

/// <summary>
/// Where this installation keeps its files.
///
/// <para>
/// The split matters on iOS more than it does on a desktop. Models go somewhere
/// excluded from iCloud backup, because a 6 GB weight file that can be downloaded
/// again must not be uploaded to the user's iCloud account; conversations and
/// settings go somewhere that IS backed up, because they cannot be recovered any
/// other way; and scratch space goes somewhere the system may reclaim.
/// </para>
/// </summary>
public sealed record AgentPaths(string DataRoot, string CacheRoot)
{
    /// <summary>Physical memory in whole gigabytes, which decides what the catalog offers.</summary>
    public int DeviceMemoryGB { get; init; } = 12;

    public string ModelsDirectory => Path.Combine(CacheRoot, "models");
    public string ConversationsDirectory => Path.Combine(DataRoot, "conversations");
    public string UploadsDirectory => Path.Combine(CacheRoot, "uploads");
    public string ScratchDirectory => Path.Combine(CacheRoot, "scratch");
    public string ArtifactsDirectory => Path.Combine(CacheRoot, "artifacts");
    public string LogsDirectory => Path.Combine(CacheRoot, "logs");

    /// <summary>Where a model's shared-prefix checkpoints are kept between launches
    /// (a cache: the engine rebuilds one it cannot read). Beside the models, not
    /// inside a model's own directory, which the store's completeness check walks.</summary>
    public string PrefixCheckpointDirectoryFor(CatalogModel model)
        => Path.Combine(CacheRoot, "prefix-cache", model.Id);
    public string InstalledSkillsDirectory => Path.Combine(DataRoot, "skills");
    public string BundledSkillsDirectory { get; init; } = string.Empty;

    /// <summary>
    /// Where the bundled CPython lives: the directory holding <c>python/</c> and the
    /// extension frameworks. Empty when this build ships no interpreter, which the
    /// runtime reports as unavailable rather than failing to construct.
    /// </summary>
    public string PythonRuntimeDirectory { get; init; } = string.Empty;
    /// <summary>The App Group inbox shared with the iOS extension, or empty.</summary>
    public string SharedInboxDirectory { get; init; } = string.Empty;
    public string SettingsFile => Path.Combine(DataRoot, "settings.json");

    public void EnsureCreated()
    {
        foreach (string directory in new[]
                 {
                     DataRoot, CacheRoot, ModelsDirectory, ConversationsDirectory, UploadsDirectory,
                     ScratchDirectory, ArtifactsDirectory, LogsDirectory, InstalledSkillsDirectory,
                 })
        {
            Directory.CreateDirectory(directory);
        }
    }

    /// <summary>
    /// The weights file for whichever model the user picked, or a path inside the
    /// model directory that does not exist yet.
    ///
    /// <para>
    /// A non-existent path rather than null is deliberate: the chat service reads the
    /// startup path to decide what <c>/api/models</c> offers and refuses a chat
    /// request until something is loaded, and a null there would read as "this build
    /// has no model support" rather than "no model has been downloaded yet".
    /// </para>
    /// </summary>
    public string SelectedModelPath(AppSettings settings)
    {
        CatalogModel? model = settings.SelectedModelId is { Length: > 0 } id
            ? ModelCatalog.Find(id)
            : null;
        return model is null
            ? Path.Combine(ModelsDirectory, "no-model-selected.gguf")
            : Path.Combine(ModelsDirectory, model.Id, model.Weights.FileName);
    }

    /// <summary>The multimodal projector beside the selected model, or null when it has none.</summary>
    public string? SelectedProjectorPath(AppSettings settings)
    {
        CatalogModel? model = settings.SelectedModelId is { Length: > 0 } id
            ? ModelCatalog.Find(id)
            : null;
        return model?.Projector is { } projector
            ? Path.Combine(ModelsDirectory, model.Id, projector.FileName)
            : null;
    }
}
