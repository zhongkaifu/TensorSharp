// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.Skills;

namespace TensorSharp.Chat
{
    /// <summary>
    /// What one video generation produced: the download URLs the Web UI shows, the
    /// file on disk (so a transport that wants to inline the bytes can read it) and
    /// the geometry the reply reports.
    /// </summary>
    public sealed record VideoGenerationResult(
        string Url,
        string AudioUrl,
        string OutputPath,
        int Width,
        int Height,
        int Frames,
        int Fps,
        long Seed,
        string Codec,
        double ElapsedSeconds);

    /// <summary>
    /// The Web UI's request handlers with the transport taken out: queue status,
    /// session lifecycle, model state and reload, file upload, Qwen-Image-Edit, Wan
    /// video generation, and the chat stream — every reply is the JSON payload object
    /// the browser expects (built with <see cref="WebUiSseEvents"/> where it is a
    /// stream frame), every refusal is a <see cref="WebUiRequestRejectedException"/>
    /// carrying the status code, and every stream is an <see cref="IAsyncEnumerable{T}"/>
    /// of frames.
    ///
    /// <para>
    /// This exists so that TensorSharp.Server (ASP.NET Core: <c>HttpContext</c> in,
    /// SSE out), an in-process loopback server inside an app, and a native view model
    /// drive ONE implementation of the contract instead of three drifting copies. The
    /// ASP.NET adapter that used to hold this code is now a few lines per route: read
    /// the body, call here, write status + JSON or <c>data:</c> frames. The wire
    /// contract — property names, frame order, status codes — is unchanged.
    /// </para>
    /// <para>
    /// The service owns NO state of its own beyond the two static locks that
    /// serialise image edits and video generations against models that are not
    /// thread-safe; everything else is injected, so one instance serves every request
    /// and is trivially faked in tests. A host that offers no code execution passes
    /// null for the runner, the workspace manager and the artifact store, and the
    /// skills plan simply never offers the shell tool.
    /// </para>
    /// </summary>
    public sealed class WebUiChatService
    {
        /// <summary>
        /// Route prefix baked into the download URLs of files a shell command or a skill
        /// script produced (<c>skill_step.files[].url</c>). The Server's artifact route
        /// uses the same string; a host that serves artifacts elsewhere passes its own.
        /// </summary>
        public const string DefaultArtifactUriPrefix = "/api/code/artifacts";

        // A share is prepared on a phone before the model sees it. Reading hundreds of
        // PDF pages (or materialising a very large scanned PDF for page-image recovery)
        // is neither useful for one prompt nor safe under iOS memory pressure.
        private const int SharedPdfMaxPages = 32;
        private const long SharedScannedPdfMaxBytes = 32L * 1024 * 1024;

        private readonly ModelService _svc;
        private readonly SessionManager _sessions;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly SkillRegistry _skills;
        private readonly ICodeRunner _codeRunner;
        private readonly SessionWorkspaceManager _workspaces;
        private readonly CodeArtifactStore _codeArtifacts;
        private readonly ILoggerFactory _loggerFactory;
        private readonly string _artifactUriPrefix;
        private readonly Func<IReadOnlyList<ChatMessage>, IReadOnlyList<string>, SkillRegistry, WebUiSkillRoute> _skillRouter;
        // UploadAsync's public response intentionally stays the Web UI's existing
        // anonymous JSON shape. This side table adds transactional ownership without
        // changing that wire contract, so a multi-file share can roll back files it
        // already uploaded when a later one hits a transient quota/I/O failure.
        private readonly ConditionalWeakTable<object, StoredUploadState> _storedUploads = new();

        private sealed class StoredUploadState
        {
            public StoredUploadState(IEnumerable<StoredUploadFile> files) => Files = files.ToList();
            public List<StoredUploadFile> Files { get; }
        }

        private sealed class StoredUploadFile
        {
            public StoredUploadFile(string path, long accountedBytes)
            {
                Path = path;
                AccountedBytes = accountedBytes;
            }
            public string Path { get; }
            public long AccountedBytes { get; }
            public bool Released { get; set; }
        }

        /// <summary>
        /// Construct the generic Web UI service without host-specific intent routing.
        /// Keep this exact signature for already-compiled TensorSharp.Chat consumers;
        /// optional parameters provide source compatibility, not binary compatibility.
        /// </summary>
        public WebUiChatService(
            ModelService svc,
            SessionManager sessions,
            ServerHostingOptions options,
            UploadStoragePolicy uploads,
            SkillRegistry skills,
            ICodeRunner codeRunner,
            SessionWorkspaceManager workspaces,
            CodeArtifactStore codeArtifacts,
            ILoggerFactory loggerFactory,
            string artifactUriPrefix = DefaultArtifactUriPrefix)
            : this(
                svc, sessions, options, uploads, skills, codeRunner, workspaces,
                codeArtifacts, loggerFactory, artifactUriPrefix, skillRouter: null)
        {
        }

        /// <summary>Construct a host service with an explicit narrow intent router.</summary>
        public WebUiChatService(
            ModelService svc,
            SessionManager sessions,
            ServerHostingOptions options,
            UploadStoragePolicy uploads,
            SkillRegistry skills,
            ICodeRunner codeRunner,
            SessionWorkspaceManager workspaces,
            CodeArtifactStore codeArtifacts,
            ILoggerFactory loggerFactory,
            string artifactUriPrefix,
            Func<IReadOnlyList<ChatMessage>, IReadOnlyList<string>, SkillRegistry, WebUiSkillRoute> skillRouter)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _uploads = uploads ?? throw new ArgumentNullException(nameof(uploads));
            _skills = skills ?? throw new ArgumentNullException(nameof(skills));
            _codeRunner = codeRunner;
            _workspaces = workspaces;
            _codeArtifacts = codeArtifacts;
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
            _artifactUriPrefix = string.IsNullOrWhiteSpace(artifactUriPrefix) ? DefaultArtifactUriPrefix : artifactUriPrefix;
            _skillRouter = skillRouter;
        }

        /// <summary>
        /// Raised once per accepted <see cref="ChatStreamAsync"/> request — after every
        /// preflight rejection has passed and before the first frame — with the session
        /// the turn runs in and the request body as received. A host that persists
        /// conversations (the iOS app keeps its transcripts server-side, because the
        /// Web UI page itself holds history only in memory) records the turn here.
        /// Never raised for a rejected request. An exception thrown by the handler
        /// fails the request; the host owns that decision.
        /// </summary>
        public Action<string, JsonElement> OnChatRequest { get; set; }

        /// <summary>
        /// Optional host-owned lease acquired near the start of a chat request, before
        /// attachment paths or bytes are read, and disposed when that request ends or is
        /// refused. TensorAgent uses it to make sending a durable shared draft atomic
        /// with explicitly discarding that draft. A callback may throw a
        /// <see cref="WebUiRequestRejectedException"/> to reject a stale/racing request.
        /// </summary>
        public Func<JsonElement, IDisposable> AcquireChatRequestLease { get; set; }

        /// <summary>
        /// The session's persistent execution workspace, or null when the feature is
        /// off or the session is the shared default (stateless API clients all pass
        /// through that one, and they must not see each other's files).
        /// </summary>
        private SessionWorkspace WorkspaceFor(ChatSession session) =>
            _workspaces != null && session != null
            && !string.Equals(session.Id, SessionManager.DefaultSessionId, StringComparison.Ordinal)
                ? _workspaces.GetOrCreate(session.Id)
                : null;

        /// <summary>
        /// How a skill script's output files become downloadable: captured into the
        /// same artifact store the shell tool uses, under a fresh run id.
        /// </summary>
        private WorkspaceFileCapture ScriptFileCapture()
        {
            if (_codeArtifacts == null)
                return null;

            string prefix = _artifactUriPrefix.TrimEnd('/');
            return (workDirectory, exclude) =>
            {
                string runId = Guid.NewGuid().ToString("N");
                IReadOnlyList<CodeArtifact> kept = _codeArtifacts.Capture(
                    runId, workDirectory,
                    (id, relative, _) => CodeArtifactStore.UrlFor(prefix, id, relative),
                    out _, exclude);
                return kept.Select(a => new SkillProducedFile(a.Path, a.Bytes, a.Pointer)).ToList();
            };
        }

        // ---- Queue ------------------------------------------------------------

        /// <summary>
        /// <c>GET /api/queue/status</c>. Real concurrency lives in the per-model
        /// inference engine; its live counters are peeked without side effects
        /// (nothing here may construct the engine), so before a model is loaded or a
        /// request has run everything reads idle. <paramref name="legacyTotalProcessed"/>
        /// is what <c>total_processed</c> reports until the engine has completed
        /// anything: the Server passes its deprecated queue's ticket count so the field
        /// keeps the value it always had; a host without that shim passes nothing.
        /// </summary>
        public object GetQueueStatus(long legacyTotalProcessed = 0)
        {
            _svc.EngineHost.TryGetLiveStats(out int processing, out int waiting, out long totalCompleted);

            // total_processed kept for API compatibility; sourced from the engine's
            // completed count (per loaded model) rather than the legacy queue.
            long totalProcessed = totalCompleted != 0 ? totalCompleted : legacyTotalProcessed;

            return new
            {
                busy = processing > 0,
                // Number of requests currently being generated concurrently.
                processing,
                // Requests admitted to the engine but still waiting for a batch slot.
                pending_requests = waiting,
                total_processed = totalProcessed,
            };
        }

        // ---- Sessions ---------------------------------------------------------

        /// <summary><c>POST /api/sessions</c> — <c>{ sessionId, createdAt }</c>.</summary>
        public object CreateSession()
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            var session = _sessions.CreateSession();
            sessionsLogger.LogInformation(LogEventIds.SessionCreated,
                "Created session via /api/sessions: {SessionId}", session.Id);
            return new
            {
                sessionId = session.Id,
                createdAt = session.CreatedAt.ToString("o"),
            };
        }

        /// <summary>
        /// <c>DELETE /api/sessions/{id}</c> — <c>{ ok = true, sessionId }</c>; 400 for
        /// the default session, 404 for an unknown one. Releases the session's
        /// execution workspace — its files, installed packages, everything its runs
        /// accumulated — which lives exactly as long as the session.
        /// </summary>
        public Task<object> DisposeSessionAsync(string id, CancellationToken cancellationToken)
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            if (string.Equals(id, SessionManager.DefaultSessionId, StringComparison.Ordinal))
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Refused to dispose default session via API: {SessionId}", id);
                throw new WebUiRequestRejectedException(400, new { ok = false, error = "Cannot dispose the default session." });
            }

            var removed = _sessions.TryRemove(id);
            if (removed == null)
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Session not found for disposal: {SessionId}", id);
                throw new WebUiRequestRejectedException(404, new { ok = false, error = $"Session '{id}' not found." });
            }

            _workspaces?.Release(id);

            // In-flight KV state is owned by the engine; disposing the session only
            // clears tracked chat history, so there is nothing to wait for.
            _svc.DisposeSession(removed);
            sessionsLogger.LogInformation(LogEventIds.SessionDisposed,
                "Disposed session via /api/sessions: {SessionId}", id);
            return Task.FromResult<object>(new { ok = true, sessionId = id });
        }

        // ---- Models ----------------------------------------------------------

        /// <summary><c>GET /api/models</c> — hosted files by NAME only (never host paths), what is loaded, and the capability blocks the UI keys its controls on.</summary>
        public object GetModels()
        {
            var files = string.IsNullOrWhiteSpace(_options.StartupModelPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupModelPath) };
            var mmProjFiles = string.IsNullOrWhiteSpace(_options.StartupMmProjPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupMmProjPath) };
            // What the loaded model does with attached pictures, when it is a video model.
            // The Web UI cannot work this out for itself and the answer changes what a
            // request should even contain: the same three images are three REFERENCES on
            // Ref2VA and an illegal request on FL2VA, which takes a first frame and a last
            // frame and nothing else. Null for every non-video model, so the UI can test
            // one field instead of pattern-matching an architecture string.
            object video = null;
            if (_svc.Model is TensorSharp.Models.Video.IVideoGenerationModel videoModel)
            {
                video = new
                {
                    family = videoModel.VideoModelFamily,
                    supportsAudio = videoModel.SupportsAudio,
                    supportsImageConditioning = videoModel.SupportsImageConditioning,
                    supportsEndImageConditioning = videoModel.SupportsEndImageConditioning,
                    supportsReferenceConditioning = videoModel.SupportsReferenceConditioning,
                    maxReferenceImages = videoModel.MaxReferenceImages,
                };
            }

            return new
            {
                models = files,
                mmProjModels = mmProjFiles,
                loaded = _svc.LoadedModelName,
                loadedMmProj = _svc.LoadedMmProjName,
                // A projector path is only configuration; this is the runtime
                // capability after the model has actually loaded and accepted it.
                // Phone clients use it to keep an image in the composer instead of
                // sending image-pad tokens to a text-only load of a vision model.
                visionReady = _svc.Model?.HasVisionEncoder() ?? false,
                // Separate architecture capability from current readiness. A false
                // visionReady on Qwen/Gemma means "install/enable the projector"; on
                // an inherently text-only model it means images may only be passed as
                // staged files to host-owned tools.
                acceptsVisionProjector =
                    _svc.Model is TensorSharp.Models.Architecture.IVisionCapableModel,
                loadedBackend = _svc.LoadedBackend,
                defaultBackend = _options.DefaultBackend,
                supportedBackends = _options.SupportedBackends,
                architecture = _svc.Architecture,
                // The effective input + output window of the loaded engine. This is
                // intentionally separate from the model artifact's own window and
                // from defaultMaxTokens, which is only the requested number of NEW
                // reply tokens and cannot enlarge either one.
                contextTokens = _svc.ContextTokens,
                modelContextTokens = _svc.ModelContextTokens,
                defaultMaxTokens = _options.DefaultMaxTokens,
                video,
                // Null when this build serves no skills at all, so the UI can hide the
                // whole control by testing one field rather than discovering it after a
                // failed /api/skills fetch. An older UI ignores the extra member.
                skills = _options.SkillsEnabled
                    ? new
                    {
                        enabled = true,
                        installable = _skills.CanInstall,
                        allowScripts = _options.SkillsAllowScripts,
                        count = _skills.Skills.Count,
                    }
                    : null,
            };
        }

        /// <summary>
        /// <c>POST /api/models/load</c> — <c>{ model, backend?, mmproj? }</c>. The model
        /// must be the hosted one (the guard resolves the file name against the startup
        /// path, so a client never needs a host path); 400 <c>{ ok = false, error }</c>
        /// for a refused request, 500 with the same shape when the load itself fails.
        /// </summary>
        public Task<object> LoadModelAsync(JsonElement body, CancellationToken cancellationToken)
        {
            var modelLoadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.ModelLoad");
            string modelName = body.GetProperty("model").GetString();
            string requestedBackend = body.TryGetProperty("backend", out var b) ? b.GetString() : null;
            string mmproj = body.TryGetProperty("mmproj", out var m) ? m.GetString() : null;

            modelLoadLogger.LogInformation(LogEventIds.ModelLoadStarted,
                "Web UI model load request: model={Model} backend={Backend} mmproj={MmProj}",
                modelName, requestedBackend ?? "(default)", mmproj ?? "(none)");

            if (!BackendSelector.TryResolveSupportedBackend(_options, requestedBackend, out string backend, out string backendError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", backendError);
                throw new WebUiRequestRejectedException(400, new { ok = false, error = backendError });
            }

            if (!HostedModelGuard.TryResolveHostedModelRequest(modelName, _options.StartupModelPath, out string modelPath, out string modelError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", modelError);
                throw new WebUiRequestRejectedException(400, new { ok = false, error = modelError });
            }

            if (!HostedModelGuard.TryValidateHostedMmProjRequest(mmproj, _options.StartupMmProjPath, out string mmProjError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI mmproj validation failed: {Reason}", mmProjError);
                throw new WebUiRequestRejectedException(400, new { ok = false, error = mmProjError });
            }

            try
            {
                _svc.LoadModel(modelPath, _options.StartupMmProjPath, backend);
                return Task.FromResult<object>(new
                {
                    ok = true,
                    model = _svc.LoadedModelName,
                    loadedMmProj = _svc.LoadedMmProjName,
                    architecture = _svc.Architecture,
                });
            }
            catch (Exception ex)
            {
                modelLoadLogger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Web UI model load failed: model={Model} backend={Backend}", modelName, backend);
                throw new WebUiRequestRejectedException(500, new { ok = false, error = ex.Message });
            }
        }

        // ---- Upload ----------------------------------------------------------

        /// <summary>
        /// <c>POST /api/upload</c>, once the transport has found the file part: the
        /// bytes, the client's file name (only its extension and its echo in the reply
        /// are used) and the declared length, which is reserved against the upload
        /// budget BEFORE a byte is written so a full disk fails in milliseconds rather
        /// than after a long upload.
        ///
        /// <para>
        /// The reply shape depends on what the file is: an image (HEIC/HEIF gain a PNG
        /// <c>previewUrl</c> because no browser renders them), a video (frames are
        /// extracted next to it, named after its GUID, so the Web UI can reference them
        /// by bare name), a text file (full content, never truncated, except CSV tables,
        /// which stay file-backed), or a PDF (its text layer; a scanned PDF falls back
        /// to page images for a vision model, or says exactly why it cannot be read).
        /// Rejections are 413/507 (budget), 400
        /// (unsupported extension, unreadable video/PDF).
        /// </para>
        /// </summary>
        public Task<object> UploadAsync(
            Stream content,
            string originalFileName,
            long length,
            CancellationToken cancellationToken) =>
            UploadAsync(content, originalFileName, length, cancellationToken, null, null);

        /// <summary>
        /// Share-import variant: a deterministic storage family bounds crash retries,
        /// and text can be kept to a phone-safe inline excerpt while the full file
        /// remains staged for file tools.
        /// </summary>
        public async Task<object> UploadAsync(
            Stream content,
            string originalFileName,
            long length,
            CancellationToken cancellationToken,
            string stableStorageKey,
            int? maxInlineTextChars)
        {
            if (content == null) throw new ArgumentNullException(nameof(content));
            var uploadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Upload");

            string ext = Path.GetExtension(originalFileName ?? string.Empty).ToLowerInvariant();
            // Classify before anything touches disk: an upload with an extension
            // outside the allow-list is rejected without ever being written, so
            // /uploads can only ever hold files the serve-side policy covers.
            string mediaType = UploadContentPolicy.Classify(ext);
            if (mediaType == "unknown")
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: unsupported extension {Extension} (name={FileName})",
                    ext.Length == 0 ? "(none)" : ext, originalFileName);
                throw new WebUiRequestRejectedException(400, new
                {
                    error = ext.Length == 0
                        ? "Files without an extension are not supported. Upload an image, video, audio, PDF, or plain-text/code file."
                        : $"Unsupported file type '{ext}'. Upload an image, video, audio, PDF, or plain-text/code file.",
                });
            }

            if (stableStorageKey != null && !IsStorageKey(stableStorageKey))
                throw new ArgumentException("A stable upload storage key must be 32 lowercase hexadecimal characters.", nameof(stableStorageKey));
            if (maxInlineTextChars is < 1)
                throw new ArgumentOutOfRangeException(nameof(maxInlineTextChars));

            string storageKey = stableStorageKey ?? Guid.NewGuid().ToString("N");
            string safeFileName = storageKey + ext;
            // A share envelope can be recovered after a process crash. Give each of its
            // files a deterministic 32-hex family so retry replaces the old primary,
            // previews and frames instead of consuming another copy's worth of storage.
            if (stableStorageKey != null && !DeleteAccountedUploadFamily(storageKey))
                throw new IOException("An earlier staged copy of this shared file is still in use.");

            if (!_uploads.TryReserveClientWrite(length, out string limitError, out int limitStatus))
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: {Reason} (name={FileName} bytes={Length})", limitError, originalFileName, length);
                throw new WebUiRequestRejectedException(limitStatus, new { error = limitError });
            }

            string savePath = Path.Combine(_options.UploadDirectory, safeFileName);
            string uploadUrl = BuildUploadUrl(safeFileName);

            try
            {
                using (var stream = File.Create(savePath))
                    await content.CopyToAsync(stream, cancellationToken);
            }
            catch
            {
                _uploads.Release(length);
                try { File.Delete(savePath); } catch { /* best effort */ }
                throw;
            }

            var storedFiles = new List<StoredUploadFile> { new(savePath, length) };
            try
            {

            // Include the full saved path and the classified media type so this entry
            // is self-sufficient for tracing back from the per-turn chat log
            // (which records each attachment by its saved path).
            uploadLogger.LogInformation(LogEventIds.UploadReceived,
                "Upload received: name={FileName} ext={Extension} mediaType={MediaType} bytes={Length} savedAs={SavedFile} savedPath={SavedPath}",
                originalFileName, ext, mediaType, length, safeFileName, savePath);

            if (mediaType == "video")
            {
                // Frames go into the upload directory, named after this upload's GUID, for
                // the same reason the saved video does: the Web UI refers to an attachment
                // by its bare file name, and that name is resolved against the upload root
                // (ChatMessageParser.ResolveAttachmentPaths) and served as /uploads/<name>.
                // Frames written anywhere else resolve to nothing at chat time and 404 as
                // thumbnails, and the GUID keeps two clips from both claiming frame_0001.png.
                List<string> frames;
                try
                {
                    frames = await Task.Run(() => MediaHelper.ExtractVideoFrames(
                        savePath, _options.UploadDirectory,
                        Path.GetFileNameWithoutExtension(safeFileName)));
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadRejected,
                        "Video frame extraction failed: name={FileName} savedPath={SavedPath} error={Error}",
                        originalFileName, savePath, ex.Message);
                    throw new WebUiRequestRejectedException(400, new { ok = false, error = "Could not read the video: " + ex.Message });
                }

                _uploads.RecordFiles(frames);
                TrackDerivedFiles(storedFiles, frames);
                return TrackUpload(new
                {
                    ok = true,
                    file = safeFileName,
                    url = uploadUrl,
                    mediaType,
                    fileName = originalFileName,
                    frames = frames.Select(f => Path.GetFileName(f)).ToList(),
                    frameUrls = frames.Select(f => BuildUploadUrl(Path.GetFileName(f))).ToList(),
                }, storedFiles);
            }

            if (mediaType == "text")
            {
                // A CSV is structured data and can tokenize far more densely than its
                // byte size suggests. Sending its full body back through the browser
                // makes every later chat request and saved transcript carry the table
                // again, before the server can replace it with a tool-backed reference.
                // Keep the complete upload on disk and make the wire contract explicit;
                // ordinary prose/code files retain the established inline contract.
                if (string.Equals(ext, ".csv", StringComparison.OrdinalIgnoreCase))
                {
                    return TrackUpload(new
                    {
                        ok = true,
                        file = safeFileName,
                        url = uploadUrl,
                        mediaType,
                        fileName = originalFileName,
                        fileBacked = true,
                        truncated = false,
                        truncateLimit = (int?)null,
                        truncateUnit = (string)null,
                        modelContextLimit = _svc.Model?.MaxContextLength,
                        originalTokenCount = (int?)null,
                        returnedTokenCount = (int?)null,
                    }, storedFiles);
                }

                string textContent;
                bool truncated;
                if (maxInlineTextChars is int textLimit)
                {
                    (textContent, truncated) = await ReadBoundedTextAsync(
                        savePath, textLimit, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    textContent = TextUploadHelper.PreserveFullText(
                        await File.ReadAllTextAsync(savePath, cancellationToken));
                    truncated = false;
                }

                return TrackUpload(new
                {
                    ok = true,
                    file = safeFileName,
                    url = uploadUrl,
                    mediaType,
                    fileName = originalFileName,
                    textContent,
                    truncated,
                    truncateLimit = truncated ? maxInlineTextChars : null,
                    truncateUnit = truncated ? "characters" : null,
                    modelContextLimit = _svc.Model?.MaxContextLength,
                    originalTokenCount = (int?)null,
                    returnedTokenCount = (int?)null,
                }, storedFiles);
            }

            if (mediaType == "pdf")
            {
                // PDFs are a document modality. First try the cheap text path: extract the
                // text layer and hand it back with the same contract as a plain-text upload,
                // so the Web UI inlines it into the message and the normal prefill path runs
                // it. A scanned / image-only PDF has no text layer, so we fall back to
                // recovering its page images and letting a vision model read them (mirroring
                // the video -> frames path) — or, if no vision model is loaded, we tell the
                // user exactly why the document can't be read instead of silently dropping it.
                int configuredPdfPages = ResolvePdfMaxPages();
                int pdfPageLimit = maxInlineTextChars.HasValue
                    ? configuredPdfPages > 0
                        ? Math.Min(configuredPdfPages, SharedPdfMaxPages)
                        : SharedPdfMaxPages
                    : configuredPdfPages;
                PdfTextResult pdf;
                try
                {
                    pdf = await Task.Run(() => PdfTextExtractor.ExtractFromFile(
                        savePath,
                        pdfPageLimit,
                        password: null,
                        maxTextCharacters: maxInlineTextChars ?? 0));
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadRejected,
                        "PDF text extraction failed: name={FileName} savedPath={SavedPath} error={Error}",
                        originalFileName, savePath, ex.Message);
                    throw new WebUiRequestRejectedException(400, new { ok = false, error = "Could not read the PDF: " + ex.Message });
                }

                if (!pdf.LooksTextless)
                {
                    string textContent = TextUploadHelper.PreserveFullText(pdf.Text);
                    bool allPagesExtracted = pdf.ExtractedPageCount == pdf.PageCount;
                    bool textTruncated = pdf.TextTruncated;
                    if (maxInlineTextChars is int pdfTextLimit)
                    {
                        (textContent, bool bounded) = BoundInlineText(
                            textContent, pdfTextLimit, alreadyTruncated: pdf.TextTruncated);
                        textTruncated |= bounded;
                    }

                    if (allPagesExtracted)
                    {
                        uploadLogger.LogInformation(LogEventIds.UploadReceived,
                            "PDF text extracted without upload truncation: name={FileName} pages={Pages} extractedPages={ExtractedPages} chars={Chars}",
                            originalFileName, pdf.PageCount, pdf.ExtractedPageCount, textContent.Length);
                    }
                    else
                    {
                        uploadLogger.LogWarning(LogEventIds.UploadReceived,
                            "PDF text extraction did not read every page: name={FileName} pages={Pages} extractedPages={ExtractedPages} chars={Chars}",
                            originalFileName, pdf.PageCount, pdf.ExtractedPageCount, textContent.Length);
                    }

                    return TrackUpload(new
                    {
                        ok = true,
                        file = safeFileName,
                        url = uploadUrl,
                        mediaType,
                        fileName = originalFileName,
                        renderedAsImages = false,
                        pageCount = pdf.PageCount,
                        extractedPageCount = pdf.ExtractedPageCount,
                        textContent,
                        truncated = textTruncated,
                        complete = allPagesExtracted && !textTruncated,
                        warning = textTruncated
                            ? allPagesExtracted
                                ? "Only a bounded excerpt of the PDF text is inline; the complete PDF remains attached."
                                : $"Only {pdf.ExtractedPageCount} of {pdf.PageCount} PDF pages and a bounded text excerpt could be included; the complete PDF remains attached."
                            : allPagesExtracted
                                ? null
                                : $"Only {pdf.ExtractedPageCount} of {pdf.PageCount} PDF pages could be read. The extracted pages were not token-truncated.",
                        truncateLimit = textTruncated ? maxInlineTextChars : null,
                        truncateUnit = textTruncated ? "characters" : null,
                        modelContextLimit = _svc.Model?.MaxContextLength,
                        originalTokenCount = (int?)null,
                        returnedTokenCount = (int?)null,
                    }, storedFiles);
                }

                // Scanned / image-only PDF (no selectable text layer).
                // PdfPageImageExtractor's legacy API opens a byte array. Keep that path
                // available for ordinary desktop uploads, but refuse an oversized shared
                // scan before it can duplicate the whole file in a phone process.
                if (maxInlineTextChars.HasValue && length > SharedScannedPdfMaxBytes)
                {
                    throw new WebUiRequestRejectedException(413, new
                    {
                        error = $"This scanned PDF is too large to render safely on this phone " +
                            $"({length / (1024.0 * 1024):0.#} MB; limit {SharedScannedPdfMaxBytes / (1024 * 1024)} MB). " +
                            "Share a smaller PDF or split it into parts.",
                    });
                }

                // A configured/projector filename is not proof that this model accepted
                // it (a mismatched GGUF can otherwise make a scanned PDF look usable).
                bool visionLoaded = _svc.Model?.HasVisionEncoder() ?? false;
                if (!visionLoaded)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF has no text layer and no vision model is loaded: name={FileName} pages={Pages}",
                        originalFileName, pdf.PageCount);
                    return TrackUpload(new
                    {
                        ok = true,
                        file = safeFileName,
                        url = uploadUrl,
                        mediaType,
                        fileName = originalFileName,
                        renderedAsImages = false,
                        needsVision = true,
                        pageCount = pdf.PageCount,
                        textContent = "",
                        warning = $"\"{originalFileName}\" has no selectable text — it looks scanned or image-only. " +
                                  "To analyze it, run the server with a vision-capable model and its projector (--mmproj <projector.gguf>).",
                    }, storedFiles);
                }

                PdfImageResult pdfImages;
                try
                {
                    pdfImages = await Task.Run(() => PdfPageImageExtractor.ExtractPageImages(
                        savePath, _options.UploadDirectory, pdfPageLimit,
                        Path.GetFileNameWithoutExtension(safeFileName)));
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadRejected,
                        "PDF page-image extraction failed: name={FileName} savedPath={SavedPath} error={Error}",
                        originalFileName, savePath, ex.Message);
                    throw new WebUiRequestRejectedException(400, new { ok = false, error = "Could not read the PDF: " + ex.Message });
                }

                _uploads.RecordFiles(pdfImages.ImagePaths);
                TrackDerivedFiles(storedFiles, pdfImages.ImagePaths);

                if (pdfImages.ImagePaths.Count == 0)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF yielded neither text nor images: name={FileName} pages={Pages}", originalFileName, pdf.PageCount);
                    return TrackUpload(new
                    {
                        ok = true,
                        file = safeFileName,
                        url = uploadUrl,
                        mediaType,
                        fileName = originalFileName,
                        renderedAsImages = false,
                        pageCount = pdf.PageCount,
                        textContent = "",
                        warning = $"Could not extract any text or images from \"{originalFileName}\".",
                    }, storedFiles);
                }

                var framePaths = pdfImages.ImagePaths.ToList();
                var frameNames = framePaths.Select(Path.GetFileName).ToList();
                var frameUrls = frameNames.Select(BuildUploadUrl).ToList();
                bool allPagesRendered = pdfImages.ExtractedPageCount == pdfImages.PageCount;
                string incompleteWarning = BuildIncompletePdfImageWarning(
                    pdfImages.ExtractedPageCount, pdfImages.PageCount);

                if (allPagesRendered)
                {
                    uploadLogger.LogInformation(LogEventIds.UploadReceived,
                        "PDF rendered as page images: name={FileName} pages={Pages} images={Images} complete=true",
                        originalFileName, pdf.PageCount, framePaths.Count);
                }
                else
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF page-image extraction was incomplete: name={FileName} pages={Pages} images={Images}",
                        originalFileName, pdf.PageCount, framePaths.Count);
                }

                return TrackUpload(new
                {
                    ok = true,
                    file = safeFileName,
                    url = uploadUrl,
                    mediaType,
                    fileName = originalFileName,
                    renderedAsImages = true,
                    pageCount = pdf.PageCount,
                    extractedPageCount = pdfImages.ExtractedPageCount,
                    complete = allPagesRendered,
                    warning = incompleteWarning,
                    frames = frameNames,
                    frameUrls,
                    note = $"This PDF has no selectable text; {framePaths.Count} page image(s) were attached for the vision model to read.",
                }, storedFiles);
            }

            // HEIC/HEIF images (e.g. iPhone photos): the server-side pipelines decode them
            // fine (Magick.NET), but no mainstream browser renders them in <img> — and the
            // default static-file content-type provider doesn't even serve the extension —
            // so the chat bubble showed a blank/broken preview. Convert a lightweight PNG
            // preview at upload time; the Web UI displays previewUrl while path (the
            // original file, full fidelity) is what the edit/vision pipelines consume.
            if (mediaType == "image" && ext is ".heic" or ".heif")
            {
                try
                {
                    string previewName = Path.GetFileNameWithoutExtension(safeFileName) + "-preview.png";
                    string previewPath = Path.Combine(_options.UploadDirectory, previewName);
                    await Task.Run(() =>
                    {
                        var img = TensorSharp.Models.QwenImage.ImageIO.Load(savePath);
                        const long previewArea = 768L * 768;   // plenty for the ~300 px bubble preview
                        if ((long)img.Width * img.Height > previewArea)
                            img = TensorSharp.Models.QwenImage.ImageIO.ResizeToArea(img, previewArea, multiple: 1);
                        TensorSharp.Models.QwenImage.ImageIO.SavePng(previewPath, img);
                    });
                    _uploads.RecordFile(previewPath);
                    TrackDerivedFiles(storedFiles, new[] { previewPath });
                    return TrackUpload(new
                    {
                        ok = true,
                        file = safeFileName,
                        url = uploadUrl,
                        previewUrl = BuildUploadUrl(previewName),
                        mediaType,
                        fileName = originalFileName,
                    }, storedFiles);
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "HEIC preview conversion failed for {FileName}: {Error} (chat preview will be blank; the edit itself is unaffected)",
                        originalFileName, ex.Message);
                }
            }

            return TrackUpload(
                new { ok = true, file = safeFileName, url = uploadUrl, mediaType, fileName = originalFileName },
                storedFiles);
            }
            catch
            {
                // UploadAsync is transactional even when decoding/extraction fails
                // after the primary file was copied. Delete accounted files and any
                // partially-created family members before propagating the refusal.
                DiscardStoredFiles(new StoredUploadState(storedFiles));
                DeleteUnaccountedUploadFamily(safeFileName);
                throw;
            }
        }

        /// <summary>
        /// Roll back a successful <see cref="UploadAsync"/> response. Used by the share
        /// importer when a later file fails transiently, so retrying the durable
        /// envelope neither duplicates files nor consumes the upload quota twice.
        /// </summary>
        public bool DiscardUpload(object uploadResponse)
        {
            if (uploadResponse == null || !_storedUploads.TryGetValue(uploadResponse, out StoredUploadState state))
                return false;
            bool discarded = DiscardStoredFiles(state);
            if (discarded)
                _storedUploads.Remove(uploadResponse);
            return discarded;
        }

        private object TrackUpload(object response, IEnumerable<StoredUploadFile> files)
        {
            _storedUploads.Add(response, new StoredUploadState(files));
            return response;
        }

        private static bool IsStorageKey(string value) =>
            value.Length == 32 && value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');

        private bool DeleteAccountedUploadFamily(string stem)
        {
            bool complete = true;
            try
            {
                foreach (string path in Directory.EnumerateFiles(_options.UploadDirectory, stem + "*"))
                {
                    try
                    {
                        var info = new FileInfo(path);
                        long bytes = info.Exists ? info.Length : 0;
                        info.Delete();
                        if (bytes > 0)
                            _uploads.Release(bytes);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        complete = false;
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                complete = false;
            }
            return complete;
        }

        private static async Task<(string Text, bool Truncated)> ReadBoundedTextAsync(
            string path, int maxChars, CancellationToken cancellationToken)
        {
            // One extra character proves truncation without ever materialising the rest
            // of a potentially 128 MB shared document in managed memory.
            using var reader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);
            var chars = new char[maxChars + 1];
            int read = 0;
            while (read < chars.Length)
            {
                int count = await reader.ReadAsync(chars.AsMemory(read, chars.Length - read), cancellationToken)
                    .ConfigureAwait(false);
                if (count == 0)
                    break;
                read += count;
            }
            bool truncated = read > maxChars;
            string text = new(chars, 0, Math.Min(read, maxChars));
            return BoundInlineText(text, maxChars, alreadyTruncated: truncated);
        }

        private static (string Text, bool Truncated) BoundInlineText(
            string text, int maxChars, bool alreadyTruncated = false)
        {
            text ??= string.Empty;
            if (!alreadyTruncated && text.Length <= maxChars)
                return (TextUploadHelper.PreserveFullText(text), false);
            const string marker = "\n\n[Shared file excerpt shortened here; the complete file remains attached.]\n\n";
            if (maxChars <= marker.Length)
            {
                int cut = Math.Min(maxChars, text.Length);
                if (cut > 0 && cut < text.Length && char.IsHighSurrogate(text[cut - 1]))
                    cut--;
                return (TextUploadHelper.PreserveFullText(text[..cut]), true);
            }
            int budget = Math.Max(0, maxChars - marker.Length);

            // The extractor stopped as soon as it filled its aggregate budget, so its
            // string is a head excerpt rather than the full source. Do not present the
            // end of that head as though it were the document's real tail.
            if (alreadyTruncated)
            {
                int cut = Math.Min(budget, text.Length);
                if (cut > 0 && cut < text.Length && char.IsHighSurrogate(text[cut - 1]))
                    cut--;
                return (TextUploadHelper.PreserveFullText(text[..cut] + marker), true);
            }

            int head = budget * 2 / 3;
            int tail = budget - head;
            if (head > 0 && head < text.Length && char.IsHighSurrogate(text[head - 1]))
                head--;
            int tailStart = text.Length - tail;
            if (tailStart > 0 && tailStart < text.Length && char.IsLowSurrogate(text[tailStart]))
                tailStart++;
            string bounded = text[..head] + marker + text[tailStart..];
            return (TextUploadHelper.PreserveFullText(bounded), true);
        }

        private static void TrackDerivedFiles(List<StoredUploadFile> destination, IEnumerable<string> paths)
        {
            foreach (string path in paths)
            {
                try
                {
                    var info = new FileInfo(path);
                    if (info.Exists)
                        destination.Add(new StoredUploadFile(path, info.Length));
                }
                catch (IOException)
                {
                    // RecordFiles uses the same FileInfo rule. A file that vanished is
                    // neither in its quota tally nor something rollback can delete.
                }
            }
        }

        private bool DiscardStoredFiles(StoredUploadState state)
        {
            bool complete = true;
            lock (state)
            {
                foreach (StoredUploadFile file in state.Files)
                {
                    if (file.Released)
                        continue;
                    try
                    {
                        if (File.Exists(file.Path))
                        {
                            File.Delete(file.Path);
                            _uploads.Release(file.AccountedBytes);
                        }
                        // Missing means a normal TTL cleanup already removed and
                        // released it; never subtract the same bytes twice.
                        file.Released = true;
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        complete = false;
                    }
                }
            }
            return complete;
        }

        private void DeleteUnaccountedUploadFamily(string storedFileName)
        {
            string stem = Path.GetFileNameWithoutExtension(storedFileName);
            if (stem.Length != 32 || stem.Any(c => !Uri.IsHexDigit(c)))
                return;
            try
            {
                foreach (string path in Directory.EnumerateFiles(_options.UploadDirectory, stem + "*"))
                {
                    try { File.Delete(path); }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
            }
        }

        // ---- Image editing (Qwen-Image-Edit) ---------------------------------

        // The model is not thread-safe; edit requests are serialised process-wide.
        private static readonly object _imageEditLock = new();

        /// <summary>
        /// The two refusals every image-edit route starts with, in order: 400 when the
        /// loaded model is not Qwen-Image-Edit, 507 when the upload directory has no
        /// room for the result PNG (checked BEFORE the slow diffusion runs, not after).
        /// Exposed so a transport can refuse before it reads a multipart body; the edit
        /// methods repeat the checks, so calling it first is optional.
        /// </summary>
        public void EnsureImageEditAvailable()
        {
            RequireImageEditModel();
            EnsureImageEditHeadroom(_loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit"), "Image edit rejected: {Reason}");
        }

        private TensorSharp.Models.QwenImage.QwenImageModel RequireImageEditModel()
        {
            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
                throw new WebUiRequestRejectedException(400, new { error = "The loaded model is not a Qwen-Image-Edit model." });
            return editModel;
        }

        private void EnsureImageEditHeadroom(ILogger logger, string rejectionTemplate)
        {
            // Fail before the (slow) diffusion runs, not after: the result PNG
            // has nowhere to go once the upload quota is exhausted.
            if (!_uploads.HasQuotaHeadroom(out string editQuotaError))
            {
                logger.LogWarning(LogEventIds.UploadRejected, rejectionTemplate, editQuotaError);
                throw new WebUiRequestRejectedException(507, new { error = editQuotaError });
            }
        }

        /// <summary>
        /// <c>POST /api/image-edit</c> with a JSON body: <c>{ imagePaths[] | imagePath,
        /// prompt, steps?, cfg?, seed?, targetArea? }</c> where the paths are the bare
        /// server file names <c>/api/upload</c> returned. Runs the loaded Qwen-Image-Edit
        /// model and returns <c>{ ok, url, width, height, elapsedSeconds }</c>. With
        /// multiple images the first drives the output geometry and the prompt can
        /// reference them as "Picture 1", "Picture 2", ... in upload order.
        /// </summary>
        public async Task<object> ImageEditAsync(JsonElement body, CancellationToken cancellationToken)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            var editModel = RequireImageEditModel();
            EnsureImageEditHeadroom(logger, "Image edit rejected: {Reason}");

            string prompt = body.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
            int steps = body.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 0;   // 0 = auto
            float cfg = body.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 0f;  // 0 = auto
            long seed = body.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
            long targetArea = 0;
            if (body.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                targetArea = tav;
            var imageBytesList = new List<byte[]>();
            string error = await ReadUploadedImagesAsync(body, imageBytesList, CancellationToken.None);
            if (error != null)
                throw new WebUiRequestRejectedException(400, new { error });

            return await RunImageEditAsync(editModel, prompt, steps, cfg, seed, targetArea, imageBytesList, logger);
        }

        /// <summary>
        /// <see cref="ImageEditAsync(JsonElement, CancellationToken)"/> for a transport
        /// that already has the image bytes (the Server's multipart form): the same
        /// checks, the same worker, the same reply.
        /// </summary>
        public async Task<object> ImageEditAsync(
            string prompt, int steps, float cfg, long seed, long targetArea, IReadOnlyList<byte[]> images, CancellationToken cancellationToken)
        {
            if (images == null) throw new ArgumentNullException(nameof(images));
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            var editModel = RequireImageEditModel();
            EnsureImageEditHeadroom(logger, "Image edit rejected: {Reason}");
            return await RunImageEditAsync(editModel, prompt ?? "", steps, cfg, seed, targetArea, images.ToList(), logger);
        }

        private async Task<object> RunImageEditAsync(
            TensorSharp.Models.QwenImage.QwenImageModel editModel,
            string prompt, int steps, float cfg, long seed, long targetArea, List<byte[]> imageBytesList, ILogger logger)
        {
            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);

            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit: prompt='{Prompt}' steps={Steps} cfg={Cfg} images={Count} bytes={Bytes}",
                prompt, steps, cfg, imageBytesList.Count, imageBytesList.Sum(b => (long)b.Length));

            var sw = Stopwatch.StartNew();
            (int w, int h) = await Task.Run(() =>
            {
                lock (_imageEditLock)
                {
                    var inputs = imageBytesList.ConvertAll(TensorSharp.Models.QwenImage.ImageIO.Decode);
                    var p = new TensorSharp.Models.QwenImage.QwenImageParams { Steps = steps, CfgScale = cfg, Seed = seed };
                    if (targetArea > 0) p.TargetArea = targetArea;
                    var output = editModel.EditImage(prompt, inputs, p);
                    TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                    return (output.Width, output.Height);
                }
            });
            sw.Stop();
            _uploads.RecordFile(outPath);

            string url = BuildUploadUrl(outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit done: {W}x{H} -> {Url} ({Sec:F1}s)", w, h, url, sw.Elapsed.TotalSeconds);
            return new { ok = true, url, width = w, height = h, elapsedSeconds = sw.Elapsed.TotalSeconds };
        }

        /// <summary>
        /// Read the referenced upload(s) from a JSON edit request into <paramref name="images"/>:
        /// <c>imagePaths</c> (array, multi-image) or legacy <c>imagePath</c> (single). References
        /// are the bare server filenames returned by <c>/api/upload</c>; absolute paths from older
        /// clients are accepted when they resolve inside the upload directory. Returns an error
        /// message, or null on success.
        /// </summary>
        internal async Task<string> ReadUploadedImagesAsync(JsonElement root, List<byte[]> images, CancellationToken ct)
        {
            var paths = new List<string>();
            if (root.TryGetProperty("imagePaths", out var ips) && ips.ValueKind == JsonValueKind.Array)
                foreach (var el in ips.EnumerateArray())
                    if (el.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(el.GetString()))
                        paths.Add(el.GetString());
            if (paths.Count == 0 && root.TryGetProperty("imagePath", out var ip) && ip.ValueKind == JsonValueKind.String)
                paths.Add(ip.GetString());
            if (paths.Count == 0)
                return "imagePath (or imagePaths) must reference a previously uploaded file.";

            foreach (var path in paths)
            {
                if (!UploadFileReference.TryResolve(_options.UploadDirectory, path, out string full) || !File.Exists(full))
                    return "imagePath must reference a previously uploaded file.";
                images.Add(await File.ReadAllBytesAsync(full, ct));
            }
            return null;
        }

        // A live denoising frame surfaced from the edit worker to the stream: a progress tick
        // (Png == null) or a decoded preview image; the terminal frame carries the final result.
        private sealed class EditFrame
        {
            public int Step, Total, Width, Height;
            public byte[] Png;        // preview PNG bytes (null = progress-only tick)
            public bool Final;        // true on the terminal frame
            public string Url;        // final image URL (Final only)
            public double Seconds;    // total elapsed (Final only)
            public string Error;      // set if the edit threw
        }

        /// <summary>
        /// <c>POST /api/image-edit/stream</c> — same JSON body as
        /// <see cref="ImageEditAsync(JsonElement, CancellationToken)"/> but a stream of
        /// frames so the Web UI can show live denoising progress: an
        /// <c>{ imageEdit, step, total, image?, width, height }</c> frame per step (with a
        /// decoded snapshot on throttled steps) and a final
        /// <c>{ done, url, width, height, elapsedSeconds }</c>. Refusals are
        /// <c>{ done, error }</c> frames, never exceptions: the original route had
        /// already started its response by then, and the page treats both alike.
        /// </summary>
        public async IAsyncEnumerable<object> ImageEditStreamAsync(
            JsonElement body,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            var ct = cancellationToken;

            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
            {
                yield return new { done = true, error = "The loaded model is not a Qwen-Image-Edit model." };
                yield break;
            }

            if (!_uploads.HasQuotaHeadroom(out string editStreamQuotaError))
            {
                logger.LogWarning(LogEventIds.UploadRejected, "Image edit (stream) rejected: {Reason}", editStreamQuotaError);
                yield return new { done = true, error = editStreamQuotaError };
                yield break;
            }

            // Parse the Web UI JSON body (mirrors ImageEditAsync).
            string prompt; int steps; float cfg; long seed; long targetArea = 0;
            var imageBytesList = new List<byte[]>();
            string parseError = null;
            try
            {
                prompt = body.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                steps = body.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 0;   // 0 = auto
                cfg = body.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 0f;  // 0 = auto
                seed = body.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
                if (body.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                    targetArea = tav;
                string error = await ReadUploadedImagesAsync(body, imageBytesList, ct);
                if (error != null)
                    parseError = error;
            }
            catch (Exception ex)
            {
                parseError = "Bad request: " + ex.Message;
                prompt = null; steps = 0; cfg = 0; seed = 0;
            }
            if (parseError != null)
            {
                yield return new { done = true, error = parseError };
                yield break;
            }

            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit (stream): prompt='{Prompt}' steps={Steps} cfg={Cfg} images={Count} bytes={Bytes}",
                prompt, steps, cfg, imageBytesList.Count, imageBytesList.Sum(b => (long)b.Length));

            // The edit worker pushes frames into this channel; the stream drains it. The callback
            // never blocks on the consumer (unbounded TryWrite) so it can't stall the denoise.
            var channel = Channel.CreateUnbounded<EditFrame>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
            // steps == 0 means "auto": the pipeline resolves the real count only later (e.g. a
            // Lightning LoRA's trained step count), so request the full preview budget and let
            // the pipeline's interval math fit it to the resolved steps. Clamping against the
            // raw 0 here disabled previews entirely for auto-step requests (the Web UI default).
            int previewCount = steps > 0 ? Math.Clamp(steps - 1, 0, 8) : 8;

            var editTask = Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    // The model is not thread-safe; serialize edit requests (shared with ImageEditAsync).
                    lock (_imageEditLock)
                    {
                        var inputs = imageBytesList.ConvertAll(TensorSharp.Models.QwenImage.ImageIO.Decode);
                        var p = new TensorSharp.Models.QwenImage.QwenImageParams
                        {
                            Steps = steps,
                            CfgScale = cfg,
                            Seed = seed,
                            PreviewCount = previewCount,
                            OnStep = (step, total, preview) =>
                            {
                                if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                                // Preview encoding is best-effort: a failure here must degrade to a
                                // plain progress tick (like the pipeline's own preview-decode guard),
                                // not abort a nearly-finished edit.
                                byte[] png = null;
                                if (preview != null)
                                {
                                    try { png = TensorSharp.Models.QwenImage.ImageIO.EncodePng(preview); }
                                    catch (Exception ex) { logger.LogWarning(LogEventIds.ChatFailed, ex, "Preview PNG encode failed; sending progress tick only"); }
                                }
                                channel.Writer.TryWrite(new EditFrame
                                {
                                    Step = step, Total = total, Png = png,
                                    Width = png != null ? preview.Width : 0, Height = png != null ? preview.Height : 0,
                                });
                            },
                        };
                        if (targetArea > 0) p.TargetArea = targetArea;
                        var output = editModel.EditImage(prompt, inputs, p);
                        TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                        _uploads.RecordFile(outPath);
                        channel.Writer.TryWrite(new EditFrame
                        {
                            Final = true, Url = BuildUploadUrl(outName),
                            Width = output.Width, Height = output.Height, Seconds = sw.Elapsed.TotalSeconds,
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = "cancelled" });
                }
                catch (Exception ex)
                {
                    logger.LogError(LogEventIds.ChatFailed, ex, "Image edit (stream) failed");
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = ex.Message });
                }
                finally { channel.Writer.Complete(); }
            }, CancellationToken.None);

            // Frames are yielded outside the try below because an iterator may not yield
            // from inside a try that has a catch; only the wait on the channel is guarded.
            while (true)
            {
                EditFrame f;
                try
                {
                    if (!await channel.Reader.WaitToReadAsync(ct))
                        break;
                    if (!channel.Reader.TryRead(out f))
                        continue;
                }
                catch (OperationCanceledException)
                {
                    // Consumer disconnected; the worker observes ct via OnStep and unwinds.
                    break;
                }

                if (f.Final)
                {
                    if (f.Error == "cancelled") break;
                    if (f.Error != null)
                        yield return new { done = true, error = f.Error };
                    else
                        yield return new { done = true, url = f.Url, width = f.Width, height = f.Height, elapsedSeconds = f.Seconds };
                    logger.LogInformation(LogEventIds.UploadReceived,
                        "Image edit (stream) done: {W}x{H} -> {Url} ({Sec:F1}s)", f.Width, f.Height, f.Url, f.Seconds);
                }
                else
                {
                    string image = f.Png != null ? "data:image/png;base64," + Convert.ToBase64String(f.Png) : null;
                    yield return new { imageEdit = true, step = f.Step, total = f.Total, image, width = f.Width, height = f.Height };
                }
            }

            // Drain the worker so its lock/VRAM is released before the next request (it finishes
            // promptly once cancellation is seen). Swallow — any error was already streamed.
            try { await editTask; } catch { /* already reported */ }
        }

        // ---- Text-to-video (Wan) ---------------------------------------------

        // The model is not thread-safe; generations are serialised process-wide.
        private static readonly object _videoGenLock = new();

        private sealed class VideoFrame
        {
            public int Step, Total;
            public bool Final;
            public string Url;
            // Sidecar WAV for models that generate an audio track jointly with the video.
            public string AudioUrl;
            public int Width, Height, Frames, Fps;
            public long Seed;
            public string Codec;
            public double Seconds;
            public string Error;
            // Live progress detail (see VideoGenerationProgress): which phase is running, how long
            // it has been running and the projected time left. A 720p/121-frame pass
            // is minutes long, so without these the UI has nothing to show between steps.
            public string Phase, Detail;
            public double Elapsed, Eta = -1;
        }

        /// <summary>
        /// The 400 every video route starts with when the loaded model cannot generate
        /// video. Exposed so a transport can refuse before it reads a body; the video
        /// methods repeat the check, so calling it first is optional.
        /// </summary>
        public void EnsureVideoGenerationAvailable() => RequireVideoModel();

        private TensorSharp.Models.Video.IVideoGenerationModel RequireVideoModel()
        {
            if (_svc.Model is not TensorSharp.Models.Video.IVideoGenerationModel videoModel)
                throw new WebUiRequestRejectedException(400, new { error = "The loaded model is not a video-generation model." });
            return videoModel;
        }

        /// <summary>
        /// <c>POST /api/video-generate</c> — JSON <c>{ prompt, width?, height?, frames?, steps?,
        /// cfg?, seed?, fps?, flowShift?, negativePrompt?, videoMode?, imagePath?, ... }</c>. Runs the
        /// loaded Wan / MiniMax-H3 model and returns <c>{ ok, url, audioUrl, width, height,
        /// frames, fps, seed, codec, elapsedSeconds }</c>. A request the model can explain
        /// (wrong checkpoint for the mode, a mode without its inputs) is a 400 carrying
        /// that explanation, not a 500. The non-streaming generation runs to completion;
        /// cancellation is honoured by <see cref="VideoGenerateStreamAsync"/>.
        /// </summary>
        public async Task<object> VideoGenerateAsync(JsonElement body, CancellationToken cancellationToken)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
            var videoModel = RequireVideoModel();

            string prompt = body.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
            if (string.IsNullOrWhiteSpace(prompt))
                throw new WebUiRequestRejectedException(400, new { error = "prompt is required." });
            var p = VideoGenerationParamsParser.Parse(body, _options, out string imgError);
            if (imgError != null)
                throw new WebUiRequestRejectedException(400, new { error = imgError });

            if (!_uploads.HasQuotaHeadroom(out string videoQuotaError))
            {
                logger.LogWarning(LogEventIds.UploadRejected, "Video generate rejected: {Reason}", videoQuotaError);
                throw new WebUiRequestRejectedException(507, new { error = videoQuotaError });
            }

            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate: prompt='{Prompt}' {W}x{H}x{F} steps={Steps} i2v={I2V}",
                prompt, p.Width, p.Height, p.Frames, p.Steps, p.ImageBytes != null);

            VideoGenerationResult result;
            try
            {
                result = await GenerateVideoCoreAsync(videoModel, prompt, p);
            }
            catch (Exception ex) when (IsVideoRequestRejection(ex))
            {
                // These carry the model's own explanation of what the request asked for
                // and why this checkpoint cannot do it — which checkpoint to load, which
                // flag to drop. Swallowing them into a generic 500 would throw away the
                // only thing that tells the caller how to fix it.
                logger.LogWarning(LogEventIds.UploadReceived, ex, "Video generate rejected");
                throw new WebUiRequestRejectedException(400, new { error = ex.Message });
            }

            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate done: {F} frames -> {Url} ({Sec:F1}s)", result.Frames, result.Url, result.ElapsedSeconds);
            return new
            {
                ok = true, url = result.Url, audioUrl = result.AudioUrl,
                width = result.Width, height = result.Height,
                frames = result.Frames, fps = result.Fps,
                seed = result.Seed, codec = result.Codec,
                elapsedSeconds = result.ElapsedSeconds,
            };
        }

        /// <summary>
        /// The generation itself, for a transport with its own envelope (the Server's
        /// OpenAI-shaped <c>/v1/videos/generations</c>): the loaded model must be a
        /// video model (400 otherwise) and the upload directory must have headroom
        /// (507); the model's own rejections propagate unwrapped so the caller can shape them.
        /// </summary>
        public Task<VideoGenerationResult> GenerateVideoAsync(string prompt, TensorSharp.Models.Video.VideoGenerationParams parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            var videoModel = RequireVideoModel();
            if (!_uploads.HasQuotaHeadroom(out string quotaError))
                throw new WebUiRequestRejectedException(507, new { error = quotaError });
            return GenerateVideoCoreAsync(videoModel, prompt ?? "", parameters);
        }

        private async Task<VideoGenerationResult> GenerateVideoCoreAsync(
            TensorSharp.Models.Video.IVideoGenerationModel videoModel, string prompt, TensorSharp.Models.Video.VideoGenerationParams p)
        {
            string outName = $"video-{Guid.NewGuid():N}.mp4";
            string outPath = Path.Combine(_options.UploadDirectory, outName);

            var sw = Stopwatch.StartNew();
            var result = await Task.Run(() =>
            {
                lock (_videoGenLock)
                {
                    var video = videoModel.GenerateVideo(prompt, p);
                    string codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outPath, video.Frames, video.Fps);
                    return (video, codec);
                }
            });
            sw.Stop();
            _uploads.RecordFile(outPath);

            string url = BuildUploadUrl(outName);
            string audioUrl = SaveAudioSidecar(result.video.Audio, outName);
            return new VideoGenerationResult(
                url, audioUrl, outPath,
                result.video.Frames[0].Width, result.video.Frames[0].Height,
                result.video.Frames.Length, result.video.Fps, result.video.Seed, result.codec,
                sw.Elapsed.TotalSeconds);
        }

        // A generation request the loaded model can explain rather than an internal
        // failure: the wrong checkpoint for the requested mode, a mode without its
        // inputs, or a conditioning kind this build does not implement yet.
        private static bool IsVideoRequestRejection(Exception ex) =>
            ex is ArgumentException or InvalidOperationException or NotSupportedException;

        // Models that generate audio jointly with the video hand back a track alongside
        // the frames. It is written as a sidecar WAV rather than muxed into the MP4:
        // muxing needs an encoder we cannot assume is installed, whereas a WAV always
        // writes and the client can play or mux it as it likes.
        private string SaveAudioSidecar(TensorSharp.Models.Video.GeneratedVideoAudio audio, string videoName)
        {
            if (audio is not { ChannelCount: > 0, SampleCount: > 0 }) return null;
            string name = Path.ChangeExtension(videoName, ".wav");
            string path = Path.Combine(_options.UploadDirectory, name);
            TensorSharp.Models.Video.WavWriter.Write(path, audio.Channels, audio.SampleRate);
            _uploads.RecordFile(path);
            return BuildUploadUrl(name);
        }

        /// <summary>
        /// <c>POST /api/video-generate/stream</c> — same JSON body as
        /// <see cref="VideoGenerateAsync"/> but a stream of progress frames
        /// (<c>{ videoGen, step, total, phase, detail, elapsedSeconds, etaSeconds }</c> per
        /// denoising step and heartbeat, then <c>{ done, url, ... }</c>) so the Web UI
        /// can show live progress. Refusals are <c>{ done, error }</c> frames.
        /// </summary>
        public async IAsyncEnumerable<object> VideoGenerateStreamAsync(
            JsonElement body,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
            var ct = cancellationToken;

            if (_svc.Model is not TensorSharp.Models.Video.IVideoGenerationModel videoModel)
            {
                yield return new { done = true, error = "The loaded model is not a video-generation model." };
                yield break;
            }

            string prompt = null;
            TensorSharp.Models.Video.VideoGenerationParams p = null;
            string parseError = null;
            try
            {
                prompt = body.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                p = VideoGenerationParamsParser.Parse(body, _options, out string imgError);
                if (imgError != null)
                    parseError = imgError;
                else if (string.IsNullOrWhiteSpace(prompt))
                    parseError = "prompt is required.";
            }
            catch (Exception ex)
            {
                parseError = "Bad request: " + ex.Message;
            }
            if (parseError != null)
            {
                yield return new { done = true, error = parseError };
                yield break;
            }

            if (!_uploads.HasQuotaHeadroom(out string videoStreamQuotaError))
            {
                logger.LogWarning(LogEventIds.UploadRejected, "Video generate (stream) rejected: {Reason}", videoStreamQuotaError);
                yield return new { done = true, error = videoStreamQuotaError };
                yield break;
            }

            string outName = $"video-{Guid.NewGuid():N}.mp4";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate (stream): prompt='{Prompt}' {W}x{H}x{F}", prompt, p.Width, p.Height, p.Frames);

            // SingleWriter is false: the heartbeat below fires OnProgress from a timer
            // thread while the generation thread is blocked inside a DiT pass, so two
            // threads publish into this channel.
            var channel = Channel.CreateUnbounded<VideoFrame>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });

            var genTask = Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    lock (_videoGenLock)
                    {
                        p.OnStep = (step, total) =>
                        {
                            if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                            channel.Writer.TryWrite(new VideoFrame { Step = step, Total = total });
                        };
                        // Heartbeats and phase transitions. These arrive from a timer
                        // thread mid-pass, so cancellation is only observed here — the
                        // OnStep hook above still owns aborting between steps.
                        p.OnProgress = prog => channel.Writer.TryWrite(new VideoFrame
                        {
                            Step = prog.Step, Total = prog.TotalSteps, Phase = prog.Phase,
                            Detail = prog.Detail, Elapsed = prog.ElapsedSeconds, Eta = prog.EtaSeconds,
                        });
                        var video = videoModel.GenerateVideo(prompt, p);
                        string codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outPath, video.Frames, video.Fps);
                        _uploads.RecordFile(outPath);
                        channel.Writer.TryWrite(new VideoFrame
                        {
                            Final = true, Url = BuildUploadUrl(outName),
                            AudioUrl = SaveAudioSidecar(video.Audio, outName),
                            Width = video.Frames[0].Width, Height = video.Frames[0].Height,
                            Frames = video.Frames.Length, Fps = video.Fps, Seed = video.Seed,
                            Codec = codec, Seconds = sw.Elapsed.TotalSeconds,
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new VideoFrame { Final = true, Error = "cancelled" });
                }
                catch (Exception ex)
                {
                    logger.LogError(LogEventIds.ChatFailed, ex, "Video generate (stream) failed");
                    channel.Writer.TryWrite(new VideoFrame { Final = true, Error = ex.Message });
                }
                finally { channel.Writer.Complete(); }
            }, CancellationToken.None);

            while (true)
            {
                VideoFrame f;
                try
                {
                    if (!await channel.Reader.WaitToReadAsync(ct))
                        break;
                    if (!channel.Reader.TryRead(out f))
                        continue;
                }
                catch (OperationCanceledException)
                {
                    // Consumer went away; the worker sees ct and stops.
                    break;
                }

                if (f.Final)
                {
                    if (f.Error == "cancelled") break;
                    if (f.Error != null)
                        yield return new { done = true, error = f.Error };
                    else
                        yield return new
                        {
                            done = true, url = f.Url, audioUrl = f.AudioUrl, width = f.Width, height = f.Height,
                            frames = f.Frames, fps = f.Fps, seed = f.Seed, codec = f.Codec,
                            elapsedSeconds = f.Seconds,
                        };
                    logger.LogInformation(LogEventIds.UploadReceived,
                        "Video generate (stream) done: {F} frames -> {Url} ({Sec:F1}s)", f.Frames, f.Url, f.Seconds);
                }
                else
                {
                    yield return new
                    {
                        videoGen = true, step = f.Step, total = f.Total,
                        phase = f.Phase, detail = f.Detail,
                        elapsedSeconds = f.Elapsed, etaSeconds = f.Eta,
                    };
                }
            }
            await genTask;
        }

        // ---- Helpers ---------------------------------------------------------

        private static string BuildUploadUrl(string fileName)
        {
            return "/uploads/" + Uri.EscapeDataString(fileName);
        }

        internal static string BuildIncompletePdfImageWarning(int extractedPages, int totalPages)
        {
            if (totalPages <= 0 || extractedPages >= totalPages)
                return null;

            return $"Only {extractedPages} of {totalPages} PDF pages could be extracted as images. " +
                "The missing pages will not be sent to the model. If TS_PDF_MAX_PAGES is set, " +
                "unset or increase it; otherwise repair or convert the PDF.";
        }

        /// <summary>
        /// Optional cap on the number of PDF pages read during upload, from the
        /// <c>TS_PDF_MAX_PAGES</c> environment variable. Returns <c>0</c> (all pages)
        /// when unset or invalid. Extracted text is otherwise preserved in full.
        /// </summary>
        private static int ResolvePdfMaxPages()
        {
            string raw = Environment.GetEnvironmentVariable("TS_PDF_MAX_PAGES");
            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int v) && v > 0)
                return v;
            return 0;
        }

        // ---- Chat -------------------------------------------------------------

        /// <summary>
        /// <c>POST /api/chat</c> as a stream of Web UI frames (see <see cref="WebUiSseEvents"/>).
        ///
        /// <para>
        /// The five preflight refusals — model/backend named in the body (400), an unknown
        /// or disposed session (404), no model loaded (400), an attachment path outside the
        /// upload directory (400), an unknown skill (400) — are thrown as
        /// <see cref="WebUiRequestRejectedException"/> from the first <c>MoveNextAsync</c>,
        /// before any frame, so the transport can still answer with a status code. After
        /// that, exactly one <c>done</c> frame ends every stream: generation failures
        /// surface as <c>done.error</c>, a consumer that goes away as <c>done.aborted</c>.
        /// </para>
        /// <para>
        /// Frame order per update: any <c>skill_step</c> lookups the disclosure loop has
        /// performed since the previous update, then that update's own
        /// <c>thinking</c> / <c>token</c> / <c>tool_calls</c> / <c>tool_progress</c>.
        /// A turn that reasoned itself out of an answer (truncated with no content, thinking
        /// on) is retried ONCE with thinking off; a turn that still produced nothing gets a
        /// visible placeholder answer rather than an empty bubble.
        /// </para>
        /// </summary>
        public async IAsyncEnumerable<object> ChatStreamAsync(
            JsonElement body,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var webUiLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.Chat");

            string requestedModel = body.TryGetProperty("model", out var modelEl) ? modelEl.GetString() : null;
            string requestedBackend = body.TryGetProperty("backend", out var beEl) ? beEl.GetString() : null;
            bool newChat = body.TryGetProperty("newChat", out var ncProp) && ncProp.GetBoolean();
            string requestedSessionId = body.TryGetProperty("sessionId", out var sidEl) ? sidEl.GetString() : null;

            if (!WebUiChatPolicy.TryValidateChatRequest(requestedModel, requestedBackend, out string selectionError))
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: {Reason} (requestedModel={Model}, requestedBackend={Backend})",
                    selectionError, requestedModel ?? "(none)", requestedBackend ?? "(none)");
                throw new WebUiRequestRejectedException(400, new { error = selectionError });
            }

            ChatSession chatSession;
            if (!string.IsNullOrWhiteSpace(requestedSessionId))
            {
                chatSession = _sessions.GetSession(requestedSessionId);
                if (chatSession == null || chatSession.IsDisposed)
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat rejected: session '{SessionId}' not found or disposed", requestedSessionId);
                    throw new WebUiRequestRejectedException(404,
                        new { error = $"Session '{requestedSessionId}' not found or has been disposed." });
                }
            }
            else
            {
                chatSession = _sessions.DefaultSession;
            }

            // Acquired before any attachment path is resolved or file is staged. A host
            // can therefore arbitrate a shared draft's Send-versus-Discard race without
            // either winner observing bytes the other has just deleted. `using` spans
            // every later preflight and the full async iteration, so refusals/cancellation
            // cannot strand host ownership.
            using IDisposable requestLease = AcquireChatRequestLease?.Invoke(body);

            if (newChat)
            {
                webUiLogger.LogInformation(LogEventIds.SessionReset,
                    "/api/chat newChat=true; resetting session {SessionId}", chatSession.Id);
                _svc.ResetSession(chatSession);
                // A new chat starts from a clean desk: the old conversation's files and
                // installs belong to the old conversation.
                if (!string.Equals(chatSession.Id, SessionManager.DefaultSessionId, StringComparison.Ordinal))
                    _workspaces?.Release(chatSession.Id);
            }

            if (!_svc.IsLoaded)
                throw new WebUiRequestRejectedException(400, new { error = "No model loaded" });

            var messagesEl = body.GetProperty("messages");
            int maxTokens = _options.ResolveMaxTokens(
                SamplingConfigParser.ReadRequestedMaxTokens(body, "maxTokens", "max_tokens"));

            var samplingConfig = SamplingConfigParser.ParseWebUi(body, _options.SamplingDefaults);
            bool uiThink = body.TryGetProperty("think", out var uiThinkProp) && uiThinkProp.GetBoolean();
            // Same rule as the HTTP APIs (an explicit think:false renders GPT-OSS at
            // low effort), so the startup warm-up - which goes through this surface with
            // both think values - prepares the prefixes real requests will ask for.
            if (!ReasoningEffortParser.TryParse(body, out string reasoningEffort, out string reasoningEffortError))
                throw new WebUiRequestRejectedException(400, new { error = reasoningEffortError });
            samplingConfig.ReasoningEffort = reasoningEffort;
            List<ToolFunction> uiTools = null;
            if (body.TryGetProperty("tools", out var uiToolsEl) && uiToolsEl.ValueKind == JsonValueKind.Array)
                uiTools = ToolFunctionParser.ParseOllama(body);

            var messages = ChatMessageParser.ParseWebUi(messagesEl);
            string audioInputError = ChatGenerationPipeline.UnsupportedAudioInputError(_svc.Architecture, messages,
                _svc.IsAudioEncoderLoaded);
            if (audioInputError != null)
                throw new WebUiRequestRejectedException(400, new { error = audioInputError });
            var requestedSkills = SkillSelectionParser.Parse(body);
            bool? requestedDiscovery = SkillSelectionParser.ParseDiscovery(body);
            WebUiSkillRoute inferredSkillRoute = null;

            // TensorSharp's generic Web UI has no opinion about prompt intent. A host
            // may opt into one narrow deterministic route for a workflow it owns. An
            // explicit discovery=false remains an opt-out, just like an explicit skill
            // selection remains scoped by the router itself.
            if (_skillRouter != null
                && MayInferSkillRoute(_options.SkillsEnabled, requestedDiscovery, uiTools))
            {
                inferredSkillRoute = _skillRouter(messages, requestedSkills, _skills);
                if (inferredSkillRoute?.Skills is { Count: > 0 })
                    requestedSkills = inferredSkillRoute.Skills.ToList();
            }

            string attachmentError = ChatMessageParser.ResolveAttachmentPaths(messages, _options.UploadDirectory);
            if (attachmentError != null)
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: attachment path outside upload directory");
                throw new WebUiRequestRejectedException(400, new { error = attachmentError });
            }

            bool hasImageInputs = messages.Any(
                message => message?.ImagePaths is { Count: > 0 });
            bool visionReady = _svc.Model?.HasVisionEncoder() ?? false;
            bool acceptsProjector =
                _svc.Model is TensorSharp.Models.Architecture.IVisionCapableModel;

            // A Qwen/Gemma model without its optional mmproj still renders image
            // placeholders, but it cannot replace them with embeddings. Letting that
            // reach generation produces a fluent hallucination about an image the
            // model never saw. Refuse before the first stream frame so every Web UI
            // gets an actionable HTTP error and OnChatRequest cannot persist a turn
            // that never had a valid input. This remains a hard refusal even when file
            // tools are enabled: a projector-capable checkpoint was explicitly chosen
            // for vision, and silently changing its input contract would hide a broken
            // model installation.
            //
            // An inherently text-only model is different. It may legitimately be asked
            // to transform an image FILE (for example, put a photo into a PDF) through
            // a host-owned tool. That case is decided after the tool plan and staging
            // below; the image channel is removed before generation, so the model is
            // never allowed to pretend that it saw pixels directly.
            if (hasImageInputs && !visionReady &&
                (acceptsProjector || _svc.IsDiffusionModel))
            {
                RejectImageInput(webUiLogger, acceptsProjector);
            }

            // DiffusionGemma streams a live "denoising preview" (whole-message replace per step)
            // rather than appended tokens, so it has its own loop.
            if (_svc.IsDiffusionModel)
            {
                try
                {
                    // Diffusion models expose no file/code tool loop. Reconstruct the
                    // former inline CSV shape so they either see the complete small
                    // table or reject it for context, never answer about unseen rows.
                    messages = ChatHistoryPreparer.RestoreUnstagedFileBackedCsvAttachments(messages);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat could not read a file-backed CSV: {Error}", ex.Message);
                    throw new WebUiRequestRejectedException(400, new
                    {
                        error = "The attached CSV could not be read from upload storage. Upload it again and retry.",
                    });
                }
                OnChatRequest?.Invoke(chatSession.Id, body);
                await foreach (object frame in ChatStreamDiffusionAsync(chatSession, messages, maxTokens, uiThink, webUiLogger, cancellationToken))
                    yield return frame;
                yield break;
            }

            // Resolved AFTER attachment paths so nothing about that check changes, and
            // before the parser gate because the built-in skill tools turn it on.
            IReadOnlyList<CodeInputFile> sourceCodeInputFiles = CollectCodeInputFiles(messages);
            IReadOnlyList<CodeInputFile> codeInputFiles = ReadableCodeInputFiles(sourceCodeInputFiles);
            SessionWorkspace workspace = WorkspaceFor(chatSession);
            var skillPlan = SkillRequestPlan.Create(
                _skills, requestedSkills, requestedDiscovery, uiTools,
                _svc.Architecture, _svc.ContextTokens, _options, out var unknownSkills, codeRunner: _codeRunner,
                codeInputFiles: codeInputFiles,
                workspace: workspace,
                captureProducedFiles: ScriptFileCapture(),
                logger: webUiLogger);

            if (unknownSkills.Count > 0)
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: unknown skills {Unknown}", string.Join(",", unknownSkills));
                throw new WebUiRequestRejectedException(400, new
                {
                    error = $"No skill called '{unknownSkills[0]}' is installed.",
                });
            }

            // This is the first point at which the loaded model's tool channel and the
            // host's script runner are both known. Reject before staging attachments or
            // starting generation so an impossible route is cheap and actionable.
            if (inferredSkillRoute != null)
            {
                string routedCapabilityError = RoutedWorkflowPreflightError(
                    inferredSkillRoute, skillPlan, _options);
                if (routedCapabilityError != null)
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat rejected: routed workflow capability unavailable: {Reason}",
                        routedCapabilityError);
                    throw new WebUiRequestRejectedException(503, new
                    {
                        code = inferredSkillRoute.RequiresNetwork && !_options.SkillsAllowNetwork
                            ? "network_disabled"
                            : "routed_workflow_unavailable",
                        error = routedCapabilityError,
                    });
                }
            }

            // CSV is structured data, not prose. Replace the browser's huge inline copy
            // only after the complete upload is demonstrably readable in this request's
            // persistent workspace. Building the plan first also lets us distinguish a
            // host-owned reader from a caller tool that merely shadows the same name.
            IReadOnlyDictionary<string, string> stagedFiles =
                new Dictionary<string, string>(StringComparer.Ordinal);
            if (skillPlan?.ToolContext?.Workspace is { } csvWorkspace && skillPlan.ToolsOffered)
            {
                bool hostCanReadStagedFile = HasHostOwnedFileReader(skillPlan);

                if (hostCanReadStagedFile)
                {
                    IReadOnlySet<string> stagedFileNames;
                    using (csvWorkspace.BeginOperation())
                        stagedFileNames = CodeInputFileStager.Stage(codeInputFiles, csvWorkspace);

                    var bySource = new Dictionary<string, string>(StringComparer.Ordinal);
                    for (int i = 0; i < codeInputFiles.Count; i++)
                    {
                        CodeInputFile input = codeInputFiles[i];
                        bool wasStaged = stagedFileNames.Contains(input.Name);
                        if (wasStaged &&
                            !string.IsNullOrEmpty(input.SourcePath) &&
                            !bySource.ContainsKey(input.SourcePath))
                        {
                            bySource.Add(input.SourcePath, input.Name);
                        }

                        // HEIC/HEIF inputs may have been converted to a readable PNG.
                        // The messages still point at the original upload, so retain an
                        // alias from that source path to the staged, converted name. It
                        // also lets the safety gate below prove that every visual input
                        // really is available to the host tool before removing it from
                        // the model's image channel.
                        if (wasStaged &&
                            i < sourceCodeInputFiles.Count &&
                            !string.IsNullOrEmpty(sourceCodeInputFiles[i].SourcePath) &&
                            !bySource.ContainsKey(sourceCodeInputFiles[i].SourcePath))
                        {
                            bySource.Add(sourceCodeInputFiles[i].SourcePath, input.Name);
                        }
                    }
                    stagedFiles = bySource;
                }
            }

            // Text-only checkpoints may carry a user-attached image solely as a file a
            // host tool can open. Require proof that every image path was staged (not
            // merely that a similarly named client tool was declared), then remove the
            // visual input before it reaches ChatGenerationPipeline's vision guard.
            // Missing attachments, derived video/PDF frames, shadowed tools and staging
            // failures all remain honest 400s rather than becoming hallucinated vision.
            if (hasImageInputs && !visionReady && !acceptsProjector)
            {
                if (!TryUseImagesAsStagedFiles(messages, stagedFiles))
                    RejectImageInput(webUiLogger, acceptsProjector: false);

                webUiLogger.LogInformation(LogEventIds.SkillSelected,
                    "/api/chat supplied image attachments to a text-only model as staged files " +
                    "(model={Model}, files={FileCount})",
                    _svc.LoadedModelName ?? "(none)", stagedFiles.Count);
            }

            messages = ChatHistoryPreparer.UseFileBackedCsvAttachments(messages, stagedFiles);

            // The metadata-only upload contract is an optimisation, never permission
            // to hide the table from the model. If this model/request cannot expose a
            // readable workspace, rebuild the former full-inline shape on the server.
            // Small CSVs therefore retain their old no-tools behaviour; large ones hit
            // the attached-document context check with an accurate explanation instead
            // of producing a plausible answer about data the model never received.
            try
            {
                messages = ChatHistoryPreparer.RestoreUnstagedFileBackedCsvAttachments(messages);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat could not read a file-backed CSV: {Error}", ex.Message);
                throw new WebUiRequestRejectedException(400, new
                {
                    error = "The attached CSV could not be read from upload storage. Upload it again and retry.",
                });
            }

            // A host-routed deliverable is complete only when the loop can prove the
            // file a tool reported is the same immutable artifact its URL downloads.
            // Snapshot after attachments have been staged: an attached deck already
            // exists at this point and must not masquerade as current-turn output after
            // a model merely touches it. Ordinary and explicitly selected skill
            // requests have no required artifact and retain their existing behavior.
            if (inferredSkillRoute?.ArtifactRequirement != null)
            {
                if (skillPlan is not { ToolsOffered: true }
                    || workspace == null
                    || _codeArtifacts == null
                    || uiTools is { Count: > 0 }
                    || !WorkspaceArtifactCompletionRequirement.TryCreate(
                        inferredSkillRoute.ArtifactRequirement,
                        workspace,
                        _codeArtifacts,
                        _artifactUriPrefix,
                        out WorkspaceArtifactCompletionRequirement completionRequirement))
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat rejected: routed artifact workflow is unavailable");
                    throw new WebUiRequestRejectedException(503, new
                    {
                        code = "routed_workflow_unavailable",
                        error = "This routed PowerPoint workflow requires host tools, a private session workspace, "
                            + "and durable artifact downloads. Start a named chat session on a host with code "
                            + "execution enabled, without caller-owned tools, and retry.",
                    });
                }

                skillPlan.CompletionRequirement = completionRequirement;
            }

            if (skillPlan != null)
            {
                messages = skillPlan.Apply(messages);
                if (!string.IsNullOrWhiteSpace(inferredSkillRoute?.Instructions))
                    messages = SkillPrompt.Apply(messages, inferredSkillRoute.Instructions);
                uiTools = skillPlan.Tools;
                webUiLogger.LogInformation(LogEventIds.SkillSelected,
                    "/api/chat skills: session={SessionId} selected={Selected} announced={Announced} inlined={Inlined} catalog={Catalog} tools={ToolsOffered}",
                    chatSession.Id, skillPlan.DescribeSelection(), skillPlan.Prompt.Deferred.Count,
                    skillPlan.Prompt.Inlined.Count,
                    skillPlan.Prompt.Catalog.Count, skillPlan.ToolsOffered);
            }

            // Every refusal is behind us: this request is going to stream.
            OnChatRequest?.Invoke(chatSession.Id, body);

            var sw = Stopwatch.StartNew();
            int tokenCount = 0;
            // How many of the plan's invocations have already been streamed to the
            // browser. The loop appends to that list as it runs, and every update we
            // receive is a chance to flush whatever is new — which is what turns a
            // multi-second skill lookup into visible progress rather than a hang.
            int reportedInvocations = 0;
            bool reportedVerifiedArtifact = false;
            bool alwaysNeedsParsing = OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            bool useUiParser = uiThink || (uiTools != null && uiTools.Count > 0) || alwaysNeedsParsing;

            IOutputParser uiParser = null;
            if (useUiParser)
            {
                uiParser = OutputParserFactory.Create(_svc.Architecture);
                uiParser.Init(uiThink, uiTools);
            }

            bool aborted = false;
            string inferenceError = null;
            // Captured from the metrics tuple's done item so the final frame can
            // report how much of this turn's prompt was served from the prior turn's
            // KV cache. Defaults to zero in case the stream is aborted before
            // generation finishes.
            int turnPromptTokens = 0;
            int turnKvReusedTokens = 0;
            // Whether the answer was cut off by the token budget. The UI renders this
            // as a "response was truncated" hint, so a user staring at a sentence that
            // stops mid-word knows to raise max tokens rather than blame the model.
            bool turnTruncated = false;
            string turnFinishReason = null;
            bool turnRepetitionExplained = false;
            // Whether the turn ever produced ANSWER text, as opposed to thinking or a
            // tool call. tokenCount cannot answer that: it counts every streamed piece,
            // so a turn that ran a skill and then stopped without writing anything has a
            // healthy tokenCount and nothing a user can read.
            bool sawContent = false;
            // Set when the skills loop hands over already-separated pieces. uiParser is
            // bypassed for those, so it must not be flushed at the end either — it holds
            // no state, and the loop's own parser already did its final flush.
            bool sawParsedUpdate = false;

            // The generation is pulled by hand rather than with `await foreach` because an
            // iterator may not yield from inside a try that has a catch: only the pull is
            // guarded, and the frames for each update are yielded outside it.
            IAsyncEnumerator<ChatStreamUpdate> stream = _svc
                .ChatStreamWithSkillsAsync(chatSession, messages, maxTokens, cancellationToken, samplingConfig,
                    uiTools, uiThink, skillPlan, webUiLogger)
                .GetAsyncEnumerator(cancellationToken);
            try
            {
                while (true)
                {
                    ChatStreamUpdate update;
                    try
                    {
                        if (!await stream.MoveNextAsync())
                            break;
                        update = stream.Current;
                    }
                    catch (OperationCanceledException)
                    {
                        aborted = true;
                        webUiLogger.LogWarning(LogEventIds.ChatAborted,
                            "Web UI chat aborted by client (sessionId={SessionId}, partialTokens={PartialTokens})",
                            chatSession.Id, tokenCount);
                        break;
                    }
                    catch (Exception ex)
                    {
                        webUiLogger.LogError(LogEventIds.ChatFailed, ex,
                            "Web UI chat failed (sessionId={SessionId}, partialTokens={PartialTokens})",
                            chatSession.Id, tokenCount);
                        inferenceError = ex.Message;
                        break;
                    }

                    // Keep the established skill_step -> tool_progress:finished pairing:
                    // the phone UI uses that adjacency to render one durable trace line.
                    // Guarded steps are streamed too, but DrainSkillTrace removes their
                    // provisional files. Once the loop proves one artifact, it travels in
                    // a separate one-shot frame and can safely be rendered/persisted.
                    var trace = DrainSkillTrace(skillPlan, reportedInvocations);
                    reportedInvocations = trace.Reported;
                    foreach (SkillToolInvocation invocation in trace.Pending)
                        yield return WebUiSseEvents.SkillStep(invocation);
                    if (TakeVerifiedArtifact(skillPlan, ref reportedVerifiedArtifact)
                        is { } verifiedArtifact)
                    {
                        yield return WebUiSseEvents.VerifiedArtifact(verifiedArtifact);
                    }

                    if (update.Done)
                    {
                        turnPromptTokens = update.PromptTokens;
                        turnKvReusedTokens = update.KvCacheReusedTokens;
                        turnTruncated = FinishReasonMapper.IsTruncated(update.FinishReason);
                        turnFinishReason = update.FinishReason;
                        turnRepetitionExplained = update.RepetitionExplained;
                        continue;
                    }

                    if (!update.Done && update.RawGenerationSuffix != null)
                    {
                        uiParser?.SetGenerationPromptSuffix(update.RawGenerationSuffix);
                        if (string.IsNullOrEmpty(update.Piece)) continue;
                    }

                    if (update.IsParsed)
                    {
                        sawParsedUpdate = true;
                        // The skills loop already parsed this round and is handing over
                        // the separated pieces (see SkillChatLoop). Running our own
                        // parser over them would be parsing parsed text.
                        if (!string.IsNullOrEmpty(update.ThinkingPiece))
                            yield return WebUiSseEvents.Thinking(update.ThinkingPiece);
                        if (HasParsedAnswerContent(update))
                        {
                            // SkillChatLoop has already separated content from thinking
                            // and tool progress. Count that content exactly as the
                            // non-loop parser branch below does; otherwise a successful
                            // tool-assisted answer is followed by the false "ended this
                            // turn without writing an answer" placeholder.
                            sawContent = true;
                            tokenCount++;
                            yield return WebUiSseEvents.Token(update.Piece);
                        }
                        if (update.ParsedToolCalls is { Count: > 0 })
                            yield return WebUiSseEvents.ToolCalls(update.ParsedToolCalls);
                        if (update.ToolProgressPhase != null)
                            yield return WebUiSseEvents.ToolProgress(
                                update.ToolProgressPhase, update.ToolProgressName,
                                update.ToolProgressPiece, update.ToolProgressSeconds,
                                update.ToolProgressDetail);
                        continue;
                    }

                    string piece = update.Piece;
                    if (string.IsNullOrEmpty(piece))
                        continue;

                    tokenCount++;
                    if (uiParser != null)
                    {
                        var parsed = uiParser.Add(piece, false);
                        if (!string.IsNullOrEmpty(parsed.Thinking))
                            yield return WebUiSseEvents.Thinking(parsed.Thinking);
                        if (!string.IsNullOrEmpty(parsed.Content))
                        {
                            sawContent = true;
                            yield return WebUiSseEvents.Token(parsed.Content);
                        }
                        if (parsed.ToolCalls != null)
                            yield return WebUiSseEvents.ToolCalls(parsed.ToolCalls);
                    }
                    else
                    {
                        // Unparsed text is answer text too. Without this every reply of
                        // a model that needs no parser ended with the "ended this turn
                        // without writing an answer" note, which the page then sent back
                        // as part of the assistant's message on the next turn.
                        sawContent = true;
                        yield return WebUiSseEvents.Token(piece);
                    }
                }
            }
            finally
            {
                await stream.DisposeAsync();
            }

            // The remedy for a turn that reasoned itself out of an answer: run it once
            // more with thinking OFF, so the model must write content from its first
            // token. Only when the first attempt produced literally nothing — a partial
            // answer is the model's to finish, and re-rolling it would discard work the
            // user can already see. One retry only; a second would double the cost of a
            // request that is simply too big for its budget.
            //
            // Gated on the thinking-budget stop specifically, NOT on truncation in
            // general. The retry is expensive: flipping thinking off re-renders the
            // conversation from the first system block (the shipped Qwen chat template
            // puts its reasoning-effort paragraph there), so the whole prompt diverges
            // at token 3 and re-prefills — 160 s on a 36k-token conversation, measured
            // 2026-09-10. Paying that is defensible when the model really did reason
            // past its budget. It is not defensible for a plain max_tokens stop, which
            // is what the startup prefix warm-up looks like: it asks for ONE token with
            // thinking on, gets `max_tokens` with no content, and used to trigger a
            // second engine request plus a warning blaming the model for a limit the
            // operator set. `repetition` is likewise not a reasoning overrun — the
            // engine's guard already ends and explains those turns.
            bool reasonedPastItsBudget = string.Equals(
                turnFinishReason, FinishReasonMapper.PipelineThinkingBudget, StringComparison.Ordinal);
            if (reasonedPastItsBudget && tokenCount == 0 && !aborted && inferenceError == null && uiThink)
            {
                webUiLogger.LogWarning(LogEventIds.ChatCompleted,
                    "chat.retry-without-thinking sessionId={SessionId}: the turn spent its whole "
                    + "token budget reasoning and produced no answer; retrying with thinking off.",
                    chatSession.Id);

                yield return WebUiSseEvents.Thinking(
                    "\n[the reasoning ran past this turn's budget - answering directly instead]\n");

                var retryParser = useUiParser ? OutputParserFactory.Create(_svc.Architecture) : null;
                retryParser?.Init(false, uiTools);
                bool retryCompleted = false;
                bool retrySawParsedUpdate = false;
                IAsyncEnumerator<ChatStreamUpdate> retry = _svc
                    .ChatStreamWithSkillsAsync(chatSession, messages, maxTokens, cancellationToken,
                        samplingConfig, uiTools, false, skillPlan, webUiLogger)
                    .GetAsyncEnumerator(cancellationToken);
                try
                {
                    while (true)
                    {
                        ChatStreamUpdate update;
                        try
                        {
                            if (!await retry.MoveNextAsync())
                            {
                                retryCompleted = true;
                                break;
                            }
                            update = retry.Current;
                        }
                        catch (OperationCanceledException)
                        {
                            aborted = true;
                            break;
                        }
                        catch (Exception ex)
                        {
                            // The retry is a rescue, not a contract: its failure must not replace
                            // the original turn's explanation with a new error.
                            webUiLogger.LogWarning(LogEventIds.ChatFailed, ex,
                                "chat.retry-without-thinking failed sessionId={SessionId}", chatSession.Id);
                            break;
                        }

                        if (update.Done)
                        {
                            turnPromptTokens = update.PromptTokens;
                            turnKvReusedTokens = update.KvCacheReusedTokens;
                            turnTruncated = FinishReasonMapper.IsTruncated(update.FinishReason);
                            turnFinishReason = update.FinishReason;
                            turnRepetitionExplained = update.RepetitionExplained;
                            continue;
                        }
                        retrySawParsedUpdate |= update.IsParsed;
                        update = ParseRetryUpdate(update, retryParser);
                        if (!string.IsNullOrEmpty(update.ThinkingPiece))
                            yield return WebUiSseEvents.Thinking(update.ThinkingPiece);
                        if (!string.IsNullOrEmpty(update.Piece))
                        {
                            // Only the separated answer counts as retry content; a
                            // prompt-opened thought channel can exist with thinking off.
                            sawContent = true;
                            tokenCount++;
                            yield return WebUiSseEvents.Token(update.Piece);
                        }
                        if (update.ParsedToolCalls is { Count: > 0 })
                            yield return WebUiSseEvents.ToolCalls(update.ParsedToolCalls);
                        if (update.ToolProgressPhase != null)
                            yield return WebUiSseEvents.ToolProgress(
                                update.ToolProgressPhase, update.ToolProgressName,
                                update.ToolProgressPiece, update.ToolProgressSeconds,
                                update.ToolProgressDetail);
                    }
                }
                finally
                {
                    await retry.DisposeAsync();
                }

                if (retryCompleted)
                {
                    uiParser = retryParser;
                    sawParsedUpdate = retrySawParsedUpdate;
                }
            }

            foreach (object frame in FinalFrames(sawParsedUpdate ? null : uiParser, aborted, inferenceError, chatSession, sw, tokenCount,
                turnPromptTokens, turnKvReusedTokens, turnTruncated, sawContent, turnFinishReason,
                turnRepetitionExplained))
            {
                yield return frame;
            }
        }

        // ---- Chat for DiffusionGemma: live denoising preview -------------------

        private async IAsyncEnumerable<object> ChatStreamDiffusionAsync(
            ChatSession chatSession, List<ChatMessage> messages, int maxTokens, bool think, ILogger webUiLogger,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sw = Stopwatch.StartNew();
            bool aborted = false;
            string inferenceError = null;
            int finalTokenCount = 0;
            int turnPromptTokens = 0;

            // The canvas text arrives with its channels already separated (the thought
            // block is dropped unless the request asked for it), so the replace frames
            // show the answer rather than Gemma's raw channel markup.
            IAsyncEnumerator<DiffusionStreamUpdate> stream = _svc
                .DiffusionChatStreamAsync(chatSession, messages, maxTokens, cancellationToken, think)
                .GetAsyncEnumerator(cancellationToken);
            try
            {
                while (true)
                {
                    DiffusionStreamUpdate u;
                    try
                    {
                        if (!await stream.MoveNextAsync())
                            break;
                        u = stream.Current;
                    }
                    catch (OperationCanceledException)
                    {
                        aborted = true;
                        webUiLogger.LogWarning(LogEventIds.ChatAborted,
                            "Web UI diffusion chat aborted by client (sessionId={SessionId})", chatSession.Id);
                        break;
                    }
                    catch (Exception ex)
                    {
                        webUiLogger.LogError(LogEventIds.ChatFailed, ex,
                            "Web UI diffusion chat failed (sessionId={SessionId})", chatSession.Id);
                        inferenceError = ex.Message;
                        break;
                    }

                    if (u.Done)
                    {
                        finalTokenCount = u.EvalTokens;
                        turnPromptTokens = u.PromptTokens;
                        continue;
                    }
                    // Both intermediate previews and the final answer use whole-message replace.
                    yield return WebUiSseEvents.Replace(u.Text, u.Step, u.TotalSteps, u.IsPreview);
                }
            }
            finally
            {
                await stream.DisposeAsync();
            }

            sw.Stop();
            double tokPerSec = finalTokenCount > 0 ? finalTokenCount / sw.Elapsed.TotalSeconds : 0;
            yield return WebUiSseEvents.Done(finalTokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError,
                chatSession.Id, turnPromptTokens, 0);
        }

        private void RejectImageInput(ILogger logger, bool acceptsProjector)
        {
            string error = acceptsProjector
                ? "This model is loaded without its image projector. Load the matching vision projector, then retry."
                : "The loaded model cannot directly analyze this image. Load a vision-capable model and its image projector. File-conversion workflows also require an uploaded attachment and a host file tool.";
            logger.LogWarning(LogEventIds.HttpRequestRejected,
                "/api/chat rejected: image input was supplied but vision is not ready " +
                "(model={Model}, architecture={Architecture}, mmproj={MmProj})",
                _svc.LoadedModelName ?? "(none)", _svc.Architecture ?? "(unknown)",
                _svc.LoadedMmProjName ?? "(none)");
            throw new WebUiRequestRejectedException(400, new
            {
                code = "vision_not_ready",
                error,
            });
        }

        /// <summary>
        /// True only when one of the declared file readers is implemented by this host.
        /// A caller tool with the same name deliberately does not count: TensorAgent
        /// cannot stage private files and then assume an external caller will read them.
        /// </summary>
        internal static bool HasHostOwnedFileReader(SkillRequestPlan plan)
        {
            if (plan?.ToolsOffered != true || plan.ToolContext?.Workspace == null)
                return false;

            return plan.Tools.Any(tool =>
                (string.Equals(tool?.Name, SkillToolNames.ReadFile, StringComparison.Ordinal) ||
                 string.Equals(tool?.Name, SkillToolNames.Shell, StringComparison.Ordinal) ||
                 (plan.ToolContext.ScriptRunner != null &&
                  string.Equals(tool?.Name, SkillTools.RunToolName, StringComparison.Ordinal))) &&
                !plan.ClientTools.Any(client => string.Equals(
                    client?.Name, tool?.Name, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Host inference must remain an optional convenience. An operator-level skills
        /// opt-out, a request-level discovery opt-out, or caller-owned tools leaves the
        /// request on its established path instead of turning a valid request into a
        /// routed-workflow 503 later in preflight.
        /// </summary>
        internal static bool MayInferSkillRoute(
            bool skillsEnabled,
            bool? requestedDiscovery,
            IReadOnlyCollection<ToolFunction> clientTools) =>
            skillsEnabled
            && requestedDiscovery != false
            && clientTools is not { Count: > 0 };

        /// <summary>
        /// Refuse a deterministic routed workflow before generation when its required
        /// execution channel is not usable. Network remains deny-by-default: a route
        /// can require the permission, but this check never grants or mutates it.
        /// </summary>
        internal static string RoutedWorkflowPreflightError(
            WebUiSkillRoute route,
            SkillRequestPlan plan,
            ServerHostingOptions options)
        {
            if (route == null || options == null)
                return null;

            var unavailable = new List<string>();
            bool requiresSkillRuns = route.ArtifactRequirement?.RequiredRuns is { Count: > 0 };

            if (requiresSkillRuns)
            {
                if (!options.SkillsAllowScripts)
                {
                    unavailable.Add(
                        "the host-owned skills_run tool is disabled; enable skill-script/code execution in host settings");
                }
                else
                {
                    bool hostRunDeclared = plan is { ToolsOffered: true }
                        && plan.ToolContext?.ScriptRunner != null
                        && plan.Tools.Any(tool => string.Equals(
                            tool?.Name, SkillTools.RunToolName, StringComparison.Ordinal))
                        && !plan.ClientTools.Any(tool => string.Equals(
                            tool?.Name, SkillTools.RunToolName, StringComparison.OrdinalIgnoreCase));
                    if (!hostRunDeclared)
                    {
                        unavailable.Add(
                            "the host-owned skills_run tool is unavailable; use a tool-capable chat model and a host that supports skill scripts");
                    }
                    else if (plan.ToolContext.ScriptRunner is SkillScriptRunner runner && !runner.CanRun)
                    {
                        unavailable.Add(
                            "skills_run is configured but its required safety sandbox is unavailable; "
                            + "configure a script backend that confines filesystem writes and network access");
                    }
                }
            }

            if (route.RequiresNetwork && !options.SkillsAllowNetwork)
            {
                unavailable.Add(
                    "network access is disabled by the user (network access for skill scripts is disabled); "
                    + "enable it in host settings for the web-research step");
            }

            return unavailable.Count == 0
                ? null
                : "This routed workflow cannot start: " + string.Join("; ", unavailable) + ". Then retry.";
        }

        /// <summary>
        /// Downgrade visual inputs to ordinary file attachments for a text-only model,
        /// but only after every referenced image is known to exist in the host tool's
        /// workspace. Validation is a separate pass so a failure never half-mutates
        /// historical messages.
        /// </summary>
        internal static bool TryUseImagesAsStagedFiles(
            List<ChatMessage> messages,
            IReadOnlyDictionary<string, string> stagedFiles)
        {
            if (messages == null || stagedFiles == null)
                return false;

            bool foundImage = false;
            foreach (ChatMessage message in messages)
            {
                if (message?.ImagePaths == null)
                    continue;

                foreach (string imagePath in message.ImagePaths)
                {
                    foundImage = true;
                    if (string.IsNullOrEmpty(imagePath) ||
                        !stagedFiles.ContainsKey(imagePath))
                    {
                        return false;
                    }
                }
            }

            if (!foundImage)
                return false;

            foreach (ChatMessage message in messages)
            {
                if (message?.ImagePaths is { Count: > 0 })
                    message.ImagePaths = null;
            }
            return true;
        }

        /// <summary>
        /// Everything the user attached to this conversation, as files a <c>shell</c>
        /// command may open by name.
        ///
        /// <para>
        /// A text upload's CONTENT is already inlined into the message, but content in
        /// the prompt is not a file on disk: asked to "convert this md file", a model with
        /// only the inline copy re-types it into its program, abridged. Staging the actual
        /// file under the name the user knows it by lets the code read all of it.
        /// </para>
        /// <para>
        /// It is every attachment and not only the text ones, which is the fix for the
        /// failure that reads as the model being stupid: "turn this photo into a PDF" put
        /// a picture in front of a vision model and NO file in front of its interpreter,
        /// so the turn was spent guessing at paths that were never going to exist. An
        /// image, a sound and a clip are files the same way a document is.
        /// </para>
        /// <para>
        /// The paths were resolved (and confined to the upload root) by
        /// <see cref="ChatMessageParser.ResolveAttachmentPaths"/> before this runs.
        /// </para>
        /// </summary>
        internal static IReadOnlyList<CodeInputFile> CollectCodeInputFiles(List<ChatMessage> messages)
        {
            List<CodeInputFile> files = null;
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (ChatMessage message in messages ?? new List<ChatMessage>())
            {
                List<string> paths = message?.AttachmentPaths ?? message?.TextFilePaths;
                if (paths == null)
                    continue;
                List<string> names = message.AttachmentPaths != null
                    ? message.AttachmentNames
                    : message.TextFileNames;

                for (int i = 0; i < paths.Count; i++)
                {
                    string path = paths[i];
                    if (string.IsNullOrEmpty(path))
                        continue;

                    // Same order as the paths; the stored name stands in when the client
                    // did not send display names. A repeated name keeps its first file —
                    // re-attaching the same document must not flip which copy the code
                    // reads mid-conversation.
                    string name = names != null && i < names.Count && !string.IsNullOrWhiteSpace(names[i])
                        ? Path.GetFileName(names[i])
                        : Path.GetFileName(path);

                    if (name.Length == 0 || !seen.Add(name))
                        continue;

                    (files ??= new List<CodeInputFile>()).Add(new CodeInputFile(name, path));
                }
            }

            return files ?? (IReadOnlyList<CodeInputFile>)Array.Empty<CodeInputFile>();
        }

        /// <summary>Image formats the bundled Pillow has no decoder for.</summary>
        private static readonly HashSet<string> UnreadableByPillow =
            new(StringComparer.OrdinalIgnoreCase) { ".heic", ".heif" };

        /// <summary>
        /// The same list, with every attachment the interpreter could not open replaced
        /// by one it can.
        ///
        /// <para>
        /// This is about one file type and one device: an iPhone photo is HEIC, the app
        /// decodes it everywhere (Apple's own image codec) and the bundled Pillow decodes
        /// it nowhere — there is no pure-Python HEIF decoder to stage. So "make a PDF of
        /// this photo" staged a file, named it to the model, and then failed inside
        /// Pillow on a format nothing in the message said anything about. The conversion
        /// is done once per upload, at full resolution (this is the copy a document is
        /// built from, not the thumbnail in the bubble), and cached beside the original.
        /// </para>
        /// <para>
        /// A conversion that fails leaves the original in place: the model then gets
        /// Pillow's own error, which is a better outcome than an attachment that silently
        /// disappears from the working directory.
        /// </para>
        /// </summary>
        private IReadOnlyList<CodeInputFile> ReadableCodeInputFiles(IReadOnlyList<CodeInputFile> files)
        {
            if (files.Count == 0)
                return files;

            List<CodeInputFile> converted = null;
            for (int i = 0; i < files.Count; i++)
            {
                CodeInputFile file = files[i];
                if (!UnreadableByPillow.Contains(Path.GetExtension(file.SourcePath ?? string.Empty)))
                {
                    converted?.Add(file);
                    continue;
                }

                converted ??= new List<CodeInputFile>(files.Take(i));
                if (TryRenderPng(file.SourcePath, out string png))
                    converted.Add(new CodeInputFile(
                        Path.GetFileNameWithoutExtension(file.Name) + ".png", png));
                else
                    converted.Add(file);
            }

            return converted ?? files;
        }

        private bool TryRenderPng(string source, out string png)
        {
            png = Path.Combine(
                _options.UploadDirectory,
                Path.GetFileNameWithoutExtension(source) + "-decoded.png");
            try
            {
                if (File.Exists(png))
                    return true;
                TensorSharp.Models.QwenImage.ImageIO.SavePng(
                    png, TensorSharp.Models.QwenImage.ImageIO.Load(source));
                _uploads.RecordFile(png);
                return true;
            }
            catch (Exception ex)
            {
                _loggerFactory.CreateLogger("TensorSharp.Server.Upload").LogWarning(
                    LogEventIds.UploadReceived,
                    "Could not decode {Source} for the interpreter: {Error}; the original is staged instead",
                    source, ex.Message);
                try { if (File.Exists(png)) File.Delete(png); } catch { /* best effort */ }
                return false;
            }
        }

        /// <summary>
        /// True when an update already parsed by <see cref="SkillChatLoop"/> contains
        /// user-visible answer text. Thinking and tool-progress-only updates do not
        /// satisfy the empty-answer guard.
        /// </summary>
        internal static bool HasParsedAnswerContent(ChatStreamUpdate update) =>
            update.IsParsed && !string.IsNullOrEmpty(update.Piece);

        /// <summary>Separates raw retry output while preserving already-parsed skill updates.</summary>
        internal static ChatStreamUpdate ParseRetryUpdate(ChatStreamUpdate update, IOutputParser parser)
        {
            if (update.Done || update.IsParsed)
                return update;

            if (update.RawGenerationSuffix != null)
                parser?.SetGenerationPromptSuffix(update.RawGenerationSuffix);
            if (parser == null)
                return update;

            var parsed = parser.Add(update.Piece ?? string.Empty, false);
            return ChatStreamUpdate.Parsed(parsed.Content, parsed.Thinking, parsed.ToolCalls);
        }

        /// <summary>
        /// The skill lookups the disclosure loop has performed since the last call, to be
        /// streamed as their own <c>skill_step</c> frames.
        ///
        /// <para>
        /// The loop deliberately does not forward an intermediate round's tokens - they
        /// carry the tool-call markup - so without this the user would watch a blank
        /// composer for as long as the model spends reading files. These frames are what
        /// they see instead: "read pdf / references/api.md".
        /// </para>
        /// </summary>
        /// <returns>The pending invocations and the new watermark, to pass back on the next call.</returns>
        internal static (SkillToolInvocation[] Pending, int Reported) DrainSkillTrace(
            SkillRequestPlan plan,
            int reported)
        {
            if (plan == null)
                return (Array.Empty<SkillToolInvocation>(), reported);

            lock (plan.Invocations)
            {
                if (plan.Invocations.Count <= reported)
                    return (Array.Empty<SkillToolInvocation>(), reported);
                SkillToolInvocation[] pending = plan.Invocations
                    .GetRange(reported, plan.Invocations.Count - reported)
                    .ToArray();

                if (plan.CompletionRequirement != null)
                {
                    for (int index = 0; index < pending.Length; index++)
                        pending[index] = pending[index] with
                        {
                            Files = Array.Empty<SkillProducedFile>(),
                        };
                }

                return (pending, plan.Invocations.Count);
            }
        }

        internal static SkillProducedFile? TakeVerifiedArtifact(
            SkillRequestPlan plan,
            ref bool reported)
        {
            if (reported || plan?.VerifiedArtifact is not { } artifact)
                return null;
            reported = true;
            return artifact;
        }

        private static IEnumerable<object> FinalFrames(
            IOutputParser uiParser, bool aborted, string inferenceError,
            ChatSession chatSession, Stopwatch sw, int tokenCount, int turnPromptTokens, int turnKvReusedTokens,
            bool truncated, bool sawContent = true, string finishReason = null,
            bool repetitionExplained = false)
        {
            if (uiParser != null && !aborted)
            {
                var finalParsed = uiParser.Add("", true);
                if (!string.IsNullOrEmpty(finalParsed.Thinking))
                    yield return WebUiSseEvents.Thinking(finalParsed.Thinking);
                if (!string.IsNullOrEmpty(finalParsed.Content))
                {
                    // The parser's final flush is answer text like any other, and the
                    // check below is three lines away: a turn whose whole answer arrived
                    // here would otherwise be told, immediately underneath it, that it
                    // never wrote one.
                    sawContent = true;
                    yield return WebUiSseEvents.Token(finalParsed.Content);
                }
                if (finalParsed.ToolCalls != null)
                    yield return WebUiSseEvents.ToolCalls(finalParsed.ToolCalls);
            }

            // The silence a user actually hits: the turn ran a skill
            // or a tool, streamed plenty of tokens doing it, and then ended without writing
            // an answer. tokenCount is healthy, truncated is false, and the page shows the
            // step that ran followed by nothing at all -- "I do not know what was
            // happening". Say that the turn ended, so the absence is legible.
            if (!aborted && inferenceError == null && !truncated && tokenCount > 0 && !sawContent)
            {
                yield return WebUiSseEvents.Token(
                    "_(The model ended this turn without writing an answer. Any tool or skill"
                    + " above did run -- its output is in the step detail -- but nothing was"
                    + " written after it. Ask it to summarise the result, or try again.)_");
            }

            // A turn the engine ended for repeating itself is the one truncation that IS
            // worth a sentence in the transcript: the user has just watched the same
            // phrase scroll by for a while and the alternative reading -- that the model
            // meant it -- is worse than a note. The skill loop says the same thing with
            // the repeated text quoted; this is the plain-chat path, which has no tools
            // to retry with.
            if (!aborted && inferenceError == null && !repetitionExplained
                && FinishReasonMapper.IsRepetition(finishReason))
            {
                yield return WebUiSseEvents.Token(
                    "\n\n_(The model's output started repeating itself and was stopped."
                    + " Ask it to try a different approach, or rephrase the request.)_");
            }

            // A turn truncated before it wrote anything used to explain itself here, in
            // the answer, as a paragraph about token budgets and thinking channels. It
            // read as the model's reply and it was not one -- the user asked a question
            // and got a lecture about settings. The `done` frame already carries both
            // `truncated` and the token count, so the page can say so in its own chrome
            // if it ever should; the transcript is not the place for it.

            sw.Stop();
            double tokPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
            // `truncated` is the page's "truncated (max tokens reached)" chip, and a
            // repetition stop is not that: the budget was nowhere near spent. The
            // protocols still call it a length stop (FinishReasonMapper.IsTruncated), which
            // is what stops a client dispatching a half-written tool call; the chip is a
            // sentence shown to a person and it would be a false one.
            yield return WebUiSseEvents.Done(tokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError, chatSession.Id,
                turnPromptTokens, turnKvReusedTokens,
                truncated && !FinishReasonMapper.IsRepetition(finishReason));
        }
    }
}
