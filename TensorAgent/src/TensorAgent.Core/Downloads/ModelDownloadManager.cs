// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Downloads;

/// <summary>Where one model's download stands.</summary>
public enum DownloadState
{
    /// <summary>Bytes are moving, or about to.</summary>
    Running,
    /// <summary>Every file is on disk and verified.</summary>
    Completed,
    /// <summary>It stopped on an error; the part file is kept and it can be resumed.</summary>
    Failed,
    /// <summary>The user stopped it; the part file is kept and it can be resumed.</summary>
    Cancelled,
}

/// <summary>One snapshot of a download, safe to hand to a UI thread or write to a stream.</summary>
/// <param name="ModelId">The catalog id being downloaded.</param>
/// <param name="State">Where it stands.</param>
/// <param name="Progress">The last progress report, or a zeroed one before the first.</param>
/// <param name="Error">Why it failed, when it did.</param>
public readonly record struct ModelDownloadStatus(
    string ModelId, DownloadState State, ModelDownloadProgress Progress, string? Error)
{
    /// <summary>True while this download is still moving.</summary>
    public bool IsRunning => State == DownloadState.Running;

    /// <summary>
    /// The optional companions this job was asked to fetch when it started. Null means
    /// required files only. This is a snapshot, so changing a setting while a transfer
    /// is suspended cannot silently change what resumes.
    /// </summary>
    public IReadOnlyCollection<CatalogFileRole>? RequestedOptionalRoles { get; init; }

    /// <summary>Whether this job was explicitly started for one optional companion.</summary>
    public bool RequestsOnly(CatalogFileRole role) =>
        RequestedOptionalRoles is { Count: 1 } roles && roles.Contains(role);
}

/// <summary>
/// Owns every model download the app has started, for as long as the app is running.
///
/// <para>
/// It exists because a download used to belong to the Models PAGE: the task was started
/// by a button, its cancellation source lived in the page's dictionary, and its progress
/// was written straight into a row. That is a five-gigabyte transfer whose lifetime is
/// tied to a screen — leaving for the chat, or for any other page, took the download's
/// only owner with it. Here the job outlives every screen: a page attaches to it when it
/// opens, detaches when it closes, and the bytes keep arriving in between.
/// </para>
/// <para>
/// Two consumers, one job. The native Models page subscribes to <see cref="Changed"/>;
/// the Web UI's route watches one download as a server-sent-event stream. Neither owns
/// the transfer, so a page that closes and a reader that disconnects both leave it
/// running — which is the whole point — and only <see cref="Cancel"/> stops it.
/// </para>
/// <para>
/// Nothing here is lost when it does stop. Every file is written through the
/// <c>.part</c> beside its destination, so a job that is cancelled, fails, or is
/// suspended by the operating system resumes from the byte it reached rather than from
/// zero. <see cref="ResumeInterrupted"/> is what the app calls when it comes back to the
/// foreground.
/// </para>
/// </summary>
public sealed class ModelDownloadManager : IDisposable
{
    private readonly ModelStore _store;
    private readonly ILogger _log;
    private readonly ConcurrentDictionary<string, Job> _jobs = new(StringComparer.Ordinal);
    private readonly object _jobsGate = new();
    private int _running;
    private bool _disposed;

    public ModelDownloadManager(ModelStore store, ILogger? log = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _log = log ?? NullLogger.Instance;
    }

    /// <summary>
    /// The optional files a download fetches when the user has asked for them: the
    /// multimodal projector, the speculative draft head and the diffusion companions.
    ///
    /// <para>
    /// One definition, because there were three — the model list's, the download
    /// route's and the foreground resume's — and they had already drifted: two of them
    /// fetched the vision projector and one did not, so which companions a model ended
    /// up with depended on whether the download had been started by a tap or by the
    /// page. A model missing its projector downloads perfectly and then cannot see.
    /// </para>
    /// </summary>
    public static IReadOnlyCollection<CatalogFileRole>? OptionalRolesFor(bool include) =>
        include ? Optional : null;

    private static readonly CatalogFileRole[] Optional =
    {
        CatalogFileRole.Projector,
        // The speculative draft head (Gemma 4 E4B / 12B): listed in the catalog as
        // optional since the start, fetched by nothing until it was wired to the
        // engine (see SpeculationPolicy).
        CatalogFileRole.Draft,
        CatalogFileRole.TextEncoder,
        CatalogFileRole.Vae,
        CatalogFileRole.VisionProjector,
        // LTX 2.5's gated Ingredients adapter enables Video Studio identity references.
        CatalogFileRole.IngredientsLora,
    };

    /// <summary>Raised for every progress report and every state change, on the reporting thread.</summary>
    public event Action<ModelDownloadStatus>? Changed;

    /// <summary>
    /// Raised when the app goes from having no download running to having one, and back.
    ///
    /// <para>
    /// The iOS head uses it to hold a background-task assertion for exactly as long as
    /// something is transferring, which is what lets a download keep going for a while
    /// after the user leaves the app instead of stopping the moment the screen does.
    /// </para>
    /// </summary>
    public event Action<bool>? BusyChanged;

    /// <summary>True while at least one download is running.</summary>
    public bool IsBusy => Volatile.Read(ref _running) > 0;

    /// <summary>Every download this launch knows about, running or finished.</summary>
    public IReadOnlyList<ModelDownloadStatus> All =>
        _jobs.Values.Select(j => j.Status).ToArray();

    /// <summary>Where <paramref name="modelId"/> stands, or null when it was never started here.</summary>
    public ModelDownloadStatus? StatusOf(string modelId) =>
        _jobs.TryGetValue(modelId ?? string.Empty, out Job? job) ? job.Status : null;

    /// <summary>
    /// Change the model directory between transfers and forget jobs belonging to the
    /// previous library. Starting a job and switching directories share this lock so a
    /// queued transfer cannot begin against an unexpected root.
    /// </summary>
    internal void ChangeModelDirectory(string root, Action? beforeChange = null)
    {
        lock (_jobsGate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (string.Equals(_store.Root, root,
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            {
                _store.ChangeRoot(root, beforeChange);
                return;
            }
            if (IsBusy)
                throw new InvalidOperationException(Loc.T("settings.storage.modelCache.busy"));

            _store.ChangeRoot(root, beforeChange);
            foreach (KeyValuePair<string, Job> entry in _jobs.ToArray())
            {
                if (_jobs.TryRemove(entry))
                    entry.Value.Dispose();
            }
        }
    }

    /// <summary>
    /// Start downloading <paramref name="model"/>, or return the running job's status if
    /// it is already being downloaded.
    ///
    /// <para>
    /// Idempotent on purpose: the button that calls this is on a list a user can leave
    /// and come back to, and two transfers of the same file into the same
    /// <c>.part</c> would corrupt it. A finished, failed or cancelled job is replaced —
    /// that is what "resume" means here, and it resumes from the part file.
    /// </para>
    /// </summary>
    /// <exception cref="LicenseAcceptanceRequiredException">The entry declares license terms that have not
    /// been accepted (<see cref="ModelStore.NeedsLicenseAcceptance"/>): ask, record the answer, then start.</exception>
    public ModelDownloadStatus Start(CatalogModel model, IReadOnlyCollection<CatalogFileRole>? optionalRoles = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!model.IsAvailable) throw new NotSupportedException(model.UnavailableReason);
        // Refused here, not only by the store: a caller that did not ask for the terms gets the answer now,
        // instead of a job that fails a moment later.
        _store.EnsureLicenseAccepted(model);
        return StartJob(model.Id, optionalRoles,
            (progress, ct, roles) => _store.DownloadAsync(model, progress, ct, roles));
    }

    /// <summary>
    /// Start a download that is not a catalog model's (a LoRA plug-in's files, see
    /// <see cref="LoraStore"/>) under <paramref name="key"/>, or return the running job's
    /// status: the same lifetime, cancellation and progress as a model's, so it outlives the
    /// page that started it too. <see cref="ResumeInterrupted"/> leaves it alone (its key is
    /// no catalog id); starting it again resumes it from its part files.
    /// </summary>
    public ModelDownloadStatus Start(string key, Func<IProgress<ModelDownloadProgress>, CancellationToken, Task> download)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        ArgumentNullException.ThrowIfNull(download);
        return StartJob(key, null, (progress, ct, _) => download(progress, ct));
    }

    private ModelDownloadStatus StartJob(
        string key, IReadOnlyCollection<CatalogFileRole>? optionalRoles,
        Func<IProgress<ModelDownloadProgress>, CancellationToken, IReadOnlyCollection<CatalogFileRole>?, Task> download)
    {
        lock (_jobsGate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_jobs.TryGetValue(key, out Job? existing))
            {
                if (existing.Status.IsRunning)
                    return existing.Status;
                _jobs.TryRemove(new KeyValuePair<string, Job>(key, existing));
                existing.Dispose();
            }

            var job = new Job(key, optionalRoles);
            _jobs[key] = job;

            // The increment's own result, not a second read of the field: the pair
            // (rise on 0->1, fall on 1->0) is what the iOS side turns into a
            // background-task assertion, and an assertion begun twice or ended twice
            // is a terminated app rather than a warning.
            if (Interlocked.Increment(ref _running) == 1)
                BusyChanged?.Invoke(true);

            job.Run(RunAsync(key, (progress, ct) => download(progress, ct, job.RequestedOptionalRoles), job));
            return job.Status;
        }
    }

    /// <summary>Stop <paramref name="modelId"/>'s download, keeping what it has fetched. False when nothing was running.</summary>
    public bool Cancel(string modelId)
    {
        if (!_jobs.TryGetValue(modelId ?? string.Empty, out Job? job) || !job.Status.IsRunning)
            return false;
        job.Cancel();
        return true;
    }

    /// <summary>
    /// Restart every download that stopped on an error rather than by the user's
    /// choice, from the bytes it already has.
    ///
    /// <para>
    /// Called when the app returns to the foreground. A phone suspends the process
    /// while the user is in another app, and a suspended socket is a dead socket: the
    /// transfer ends with an I/O error through no fault of the user's. Making them
    /// notice that and tap Resume is making them do the app's job — and a download that
    /// silently stopped hours ago is the thing they were waiting for.
    /// </para>
    /// </summary>
    /// <param name="find">Resolves a catalog id back to its entry.</param>
    /// <returns>The ids that were restarted.</returns>
    public IReadOnlyList<string> ResumeInterrupted(Func<string, CatalogModel?> find)
    {
        ArgumentNullException.ThrowIfNull(find);
        lock (_jobsGate)
        {
            if (_disposed)
                return Array.Empty<string>();

            var resumed = new List<string>();
            foreach (Job job in _jobs.Values.ToArray())
            {
                if (job.Status.State != DownloadState.Failed)
                    continue;
                if (find(job.Status.ModelId) is not { } model)
                    continue;
                if (!model.IsAvailable)
                    continue;
                // Accepted when it started; withdrawn since (a decline), it waits for the user to ask again.
                if (_store.NeedsLicenseAcceptance(model))
                    continue;
                _log.LogInformation("resuming the interrupted download of {Model}", model.Id);
                // Preserve the transfer the user actually requested. In particular, an
                // explicit projector-only download must not turn into a required-files-only
                // no-op merely because the global optional-download setting is off when
                // iOS brings the app back to the foreground.
                Start(model, job.RequestedOptionalRoles);
                resumed.Add(model.Id);
            }
            return resumed;
        }
    }

    /// <summary>
    /// Follow one download to its end: the current status first, then every change.
    ///
    /// <para>
    /// The reader leaving does NOT stop the download — cancelling
    /// <paramref name="ct"/> ends this enumeration and nothing else. That asymmetry is
    /// the feature: the route that streams progress to the Web UI is a window onto the
    /// job, not its owner, so navigating away from the model list closes the window and
    /// leaves the transfer running.
    /// </para>
    /// </summary>
    public async IAsyncEnumerable<ModelDownloadStatus> WatchAsync(
        string modelId, [EnumeratorCancellation] CancellationToken ct)
    {
        if (!_jobs.TryGetValue(modelId ?? string.Empty, out Job? job))
            yield break;

        Channel<ModelDownloadStatus> updates = Channel.CreateUnbounded<ModelDownloadStatus>(
            new UnboundedChannelOptions { SingleReader = true });

        // Subscribed BEFORE the current status is yielded, so a job that finishes
        // between the two is reported by the subscription rather than lost between them.
        job.Subscribe(updates.Writer);
        try
        {
            ModelDownloadStatus current = job.Status;
            yield return current;
            if (!current.IsRunning)
                yield break;

            await foreach (ModelDownloadStatus update in updates.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                yield return update;
                if (!update.IsRunning)
                    yield break;
            }
        }
        finally
        {
            job.Unsubscribe(updates.Writer);
        }
    }

    private async Task RunAsync(string key, Func<IProgress<ModelDownloadProgress>, CancellationToken, Task> download, Job job)
    {
        // Yield first so Start() returns before any of this runs: the caller is a UI
        // thread, and DownloadAsync's first act is a synchronous directory scan.
        await Task.Yield();
        var progress = new Progress<ModelDownloadProgress>(p => job.Report(p, Changed));
        try
        {
            await download(progress, job.Token).ConfigureAwait(false);
            job.Finish(DownloadState.Completed, null, Changed);
            _log.LogInformation("download of {Model} finished", key);
        }
        catch (OperationCanceledException)
        {
            job.Finish(DownloadState.Cancelled, null, Changed);
            _log.LogInformation("download of {Model} was cancelled; the part file is kept", key);
        }
        catch (Exception ex)
        {
            job.Finish(DownloadState.Failed, ex.Message, Changed);
            _log.LogWarning(ex, "download of {Model} failed", key);
        }
        finally
        {
            if (Interlocked.Decrement(ref _running) == 0)
                BusyChanged?.Invoke(false);
        }
    }

    public void Dispose()
    {
        Job[] jobs;
        lock (_jobsGate)
        {
            if (_disposed)
                return;
            _disposed = true;
            jobs = _jobs.Values.ToArray();
        }
        foreach (Job job in jobs)
            job.Cancel();
        // Waited for, not abandoned: the store's downloads hold a semaphore and open
        // file handles on a .part, and letting the process tear those down under a
        // running write is how a resumable download stops being resumable.
        Task.WaitAll(jobs.Select(j => j.Completion).ToArray(), TimeSpan.FromSeconds(10));
        foreach (Job job in jobs)
            job.Dispose();
    }

    /// <summary>One model's transfer: its cancellation, its task, its latest status, its watchers.</summary>
    private sealed class Job : IDisposable
    {
        private readonly CancellationTokenSource _cancel = new();
        private readonly List<ChannelWriter<ModelDownloadStatus>> _watchers = new();
        private readonly object _gate = new();
        private readonly IReadOnlyCollection<CatalogFileRole>? _requestedOptionalRoles;
        private ModelDownloadStatus _status;
        private Task _task = Task.CompletedTask;

        public Job(string modelId, IReadOnlyCollection<CatalogFileRole>? requestedOptionalRoles)
        {
            CatalogFileRole[]? snapshot = requestedOptionalRoles?.Distinct().ToArray();
            _requestedOptionalRoles = snapshot is null ? null : Array.AsReadOnly(snapshot);
            _status = new ModelDownloadStatus(
                modelId, DownloadState.Running,
                new ModelDownloadProgress(string.Empty, 0, 0, 0, 0, 0, "downloading"), null)
            {
                RequestedOptionalRoles = _requestedOptionalRoles,
            };
        }

        public CancellationToken Token => _cancel.Token;

        public IReadOnlyCollection<CatalogFileRole>? RequestedOptionalRoles =>
            _requestedOptionalRoles;

        public Task Completion => _task;

        public ModelDownloadStatus Status
        {
            get { lock (_gate) return _status; }
        }

        public void Run(Task task) => _task = task;

        public void Cancel()
        {
            try { _cancel.Cancel(); } catch (ObjectDisposedException) { }
        }

        public void Subscribe(ChannelWriter<ModelDownloadStatus> writer)
        {
            lock (_gate) _watchers.Add(writer);
        }

        public void Unsubscribe(ChannelWriter<ModelDownloadStatus> writer)
        {
            lock (_gate) _watchers.Remove(writer);
        }

        public void Report(ModelDownloadProgress progress, Action<ModelDownloadStatus>? changed)
        {
            ModelDownloadStatus next;
            lock (_gate)
            {
                if (_status.State != DownloadState.Running)
                    return;                    // a late report after the end says nothing
                _status = _status with { Progress = progress };
                next = _status;
            }
            Publish(next, changed);
        }

        public void Finish(DownloadState state, string? error, Action<ModelDownloadStatus>? changed)
        {
            ModelDownloadStatus next;
            lock (_gate)
            {
                _status = _status with { State = state, Error = error };
                next = _status;
            }
            Publish(next, changed);
            lock (_gate)
            {
                foreach (ChannelWriter<ModelDownloadStatus> writer in _watchers)
                    writer.TryComplete();
                _watchers.Clear();
            }
        }

        private void Publish(ModelDownloadStatus status, Action<ModelDownloadStatus>? changed)
        {
            ChannelWriter<ModelDownloadStatus>[] writers;
            lock (_gate) writers = _watchers.ToArray();
            foreach (ChannelWriter<ModelDownloadStatus> writer in writers)
                writer.TryWrite(status);
            changed?.Invoke(status);
        }

        public void Dispose()
        {
            try { _cancel.Dispose(); } catch (Exception) { /* already gone */ }
        }
    }
}
