// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Sharing;
using TensorAgent.Core.Settings;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Chat;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Every route the Web UI calls, bound to the same services the desktop server
/// binds them to.
///
/// <para>
/// This is the whole reason the chat pipeline was pulled out of the ASP.NET host:
/// the page in the WebView is <c>TensorSharp.Server/wwwroot/index.html</c> byte for
/// byte, so the API underneath it has to answer the same paths, with the same
/// payload shapes and the same server-sent-event frames, or the page silently
/// misbehaves in ways no compiler catches. What changes here is only the plumbing —
/// <see cref="LoopbackServer"/> instead of minimal APIs, because iOS has no ASP.NET
/// Core runtime pack — and never the contract.
/// </para>
/// <para>
/// The routes the app adds on top all live under <c>/api/agent</c>, so the shared
/// surface stays exactly the shared surface and there is no chance of a name
/// colliding with something the desktop adds later.
/// </para>
/// </summary>
public static class WebUiRoutes
{
    /// <summary>
    /// Bind the shared Web UI API. <paramref name="chat"/> and
    /// <paramref name="uploadDirectory"/> are required; the rest are optional and their
    /// routes answer 503 with a reason when absent, which is what lets the app start
    /// before a model has been chosen.
    /// </summary>
    /// <param name="uploadDirectory">
    /// The directory <c>/api/upload</c> writes into, which is also what <c>/uploads/</c>
    /// serves. It is a parameter rather than something read off the chat service because
    /// serving those files is not optional: every reply the service gives — an
    /// attachment, a video frame, an edited image, a generated clip — points at a
    /// <c>/uploads/</c> URL and nothing else.
    /// </param>
    /// <param name="chatFrames">
    /// Where <c>/api/chat</c>'s frames come from, or null for the chat service's own
    /// stream — which is what the app uses.
    ///
    /// <para>
    /// It is a parameter so that the wrapper this route puts around those frames can
    /// be tested. That wrapper, <see cref="Recording"/>, is what writes an answer down
    /// when the turn ends; the frames it reads are produced only by a loaded model, so
    /// without a seam here the wiring between the route and the recorder is checked by
    /// nothing, and losing it costs the user exactly the answer they are reading.
    /// </para>
    /// </param>
    /// <param name="turns">
    /// Who owns a generation once it starts, or null to keep the desktop's arrangement
    /// where the request owns it.
    ///
    /// <para>
    /// This is the difference between a browser tab and a phone. With a manager, a
    /// <c>POST /api/chat</c> STARTS a turn and then reads it like anyone else, so the
    /// user opening the model list — or the screen dimming — costs the answer nothing
    /// and the page can attach to it again when it comes back. Without one the stream is
    /// the turn, which is what the desktop server does and what the tests that predate
    /// this exercise.
    /// </para>
    /// </param>
    public static void MapWebUi(
        this LoopbackServer server,
        WebUiChatService chat,
        string uploadDirectory,
        SkillsService? skills = null,
        ConversationRecorder? recorder = null,
        Func<JsonElement, CancellationToken, IAsyncEnumerable<object>>? chatFrames = null,
        ChatTurnManager? turns = null)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(chat);
        ArgumentException.ThrowIfNullOrWhiteSpace(uploadDirectory);
        chatFrames ??= chat.ChatStreamAsync;

        // ---- chat ---------------------------------------------------------------
        server.MapGet("/api/queue/status", (_, _) => Ok(chat.GetQueueStatus()));
        server.MapGet("/api/models", (_, _) => Ok(chat.GetModels()));
        server.MapPost("/api/models/load", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            return Json(await Guarded(() => chat.LoadModelAsync(body, ct)));
        });
        server.MapPost("/api/chat", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            if (turns is null)
            {
                return LoopbackResponse.Sse(
                    Recording(Guarded(chatFrames(body, ct)), recorder), request.Cancellation);
            }

            // The request STARTS the turn and then reads it exactly as a later reader
            // would. Its own token is deliberately not handed to the generation: the
            // whole point is that this connection going away is not the turn ending.
            string id = turns.Start(
                TurnKeyFor(body, recorder), token => Guarded(chatFrames(body, token)));
            return LoopbackResponse.Sse(
                turns.WatchAsync(id, 0, CancellationToken.None),
                request.Cancellation,
                headers: new Dictionary<string, string>(StringComparer.Ordinal) { [TurnHeader] = id });
        });

        // ---- a turn the page has to find again ----------------------------------
        //
        // Registered here rather than in MapAgent because the manager is here, and it is
        // here because /api/sessions has to report the same turn: a page that reloaded
        // learns about the generation still running for its conversation from the very
        // request that binds it, without a second round trip.
        if (turns is not null)
        {
            server.MapGet("/api/agent/turns", (request, _) =>
                Ok(new { turn = Describe(turns.StatusOfKey(request.Query("conversation"))) }));

            server.MapGet("/api/agent/turns/{id}", (request, _) =>
            {
                string id = request.RouteValues["id"];
                if (turns.StatusOfId(id) is null)
                    return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { error = "no such turn" }, 404));
                int from = int.TryParse(request.Query("from"), out int parsed) ? parsed : 0;
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                    turns.WatchAsync(id, from, CancellationToken.None),
                    request.Cancellation,
                    headers: new Dictionary<string, string>(StringComparer.Ordinal) { [TurnHeader] = id }));
            });

            // Stopping is something a caller ASKS for now. It used to be what dropping
            // the connection meant, which is why leaving the page ended the answer.
            server.MapPost("/api/agent/turns/{id}/stop", (request, _) =>
                Ok(new { stopped = turns.StopId(request.RouteValues["id"]), id = request.RouteValues["id"] }));
        }

        // ---- sessions -----------------------------------------------------------
        //
        // Which chat the page should open as it comes up. The page asks because it
        // cannot answer for itself: its first load and its fourth look identical from
        // inside, and they want opposite things -- a launch wants a clean composer,
        // a reload wants the chat it was torn out of. Only the app knows which this is.
        if (recorder is not null)
        {
            server.MapGet("/api/agent/launch", (_, _) => Ok(new
            {
                cold = recorder.IsColdLaunch,
                // Which chat to come back to when it is NOT a launch. The page cannot
                // work this out from the conversation list: the empty chat a launch
                // just opened is not in it at all.
                conversation = recorder.CurrentConversationId,
            }));
        }

        // The desktop's route creates an engine session and says so. The app's page
        // also asks, in the query string, which saved conversation that session is
        // for, and needs the answer back so it can render the transcript it is
        // resuming. The extra member is additive: the desktop page ignores it.
        server.MapPost("/api/sessions", (request, _) =>
        {
            object created = chat.CreateSession();
            if (recorder is null)
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(created));

            string sessionId = SessionIdOf(created);
            Conversation conversation = recorder.Bind(sessionId, request.Query("conversation"));
            // Reopening a chat opens a new session; keep that chat's cached prompt state
            // reachable from it (one cache scope per conversation, not per session).
            chat.BindSessionConversation(sessionId, conversation.Id);
            return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new
            {
                sessionId,
                conversationId = conversation.Id,
                title = conversation.Title,
                messages = conversation.Messages,
                think = conversation.Think,
                skills = conversation.Skills,
                skillsExplicit = conversation.SkillsExplicit,
                modelId = conversation.ModelId,
                // The answer a previous page left running here, if there is one. Sent
                // with the binding rather than fetched afterwards, so a page that was
                // torn down mid-turn learns about it in the same breath as it learns
                // which conversation it is in.
                activeTurn = Describe(turns?.StatusOfKey(conversation.Id)),
            }));
        });
        server.MapDelete("/api/sessions/{id}", async (request, ct) =>
        {
            string id = request.RouteValues["id"];
            object disposed = await Guarded(() => chat.DisposeSessionAsync(id, ct));
            recorder?.Release(id);
            return Json(disposed);
        });

        // ---- uploads and generation --------------------------------------------
        server.MapPost("/api/upload", async (request, ct) =>
        {
            using MultipartForm form = await request.ReadFormAsync(ct);
            MultipartFile? file = form.Files.FirstOrDefault();
            if (file is null)
                return LoopbackResponse.Json(new { error = "no file was uploaded" }, 400);
            await using FileStream content = File.OpenRead(file.TempPath);
            // The name is resolved before the service sees it, because the service
            // classifies an upload by its extension alone and iOS's photo picker sends
            // a name that has none. See UploadNaming.
            return Json(await Guarded(() => chat.UploadAsync(
                content, UploadNaming.ResolveFileName(file), file.Length, ct)));
        });
        // The desktop server mounts the upload directory as static files; here it is a
        // route, and it has to exist for the same reason: the page renders an
        // attachment, a frame, an edited image and a generated clip from the URL the
        // API handed back, so a missing mount is not a missing feature but a chat full
        // of broken images with a 200 behind each one.
        server.MapGet("/uploads/{*path}", (request, _) => Task.FromResult(ServeUpload(uploadDirectory, request.RouteValues["path"])));

        server.MapPost("/api/image-edit", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            return Json(await Guarded(() => chat.ImageEditAsync(body, ct)));
        });
        server.MapPost("/api/image-edit/stream", async (request, ct) =>
            LoopbackResponse.Sse(Guarded(chat.ImageEditStreamAsync(await request.ReadJsonAsync(ct), ct)), request.Cancellation));
        server.MapPost("/api/video-generate", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            return Json(await Guarded(() => chat.VideoGenerateAsync(body, ct)));
        });
        server.MapPost("/api/video-generate/stream", async (request, ct) =>
            LoopbackResponse.Sse(Guarded(chat.VideoGenerateStreamAsync(await request.ReadJsonAsync(ct), ct)), request.Cancellation));

        // ---- skills -------------------------------------------------------------
        if (skills is null)
        {
            server.MapGet("/api/skills", (_, _) => Ok(new { skills = Array.Empty<object>(), canInstall = false }));
            return;
        }

        server.MapGet("/api/skills", (_, _) => OkGuarded(() => skills.ListForUi()));
        server.MapGet("/api/skills/{name}", (request, _) => OkGuarded(() => skills.GetForUi(request.RouteValues["name"])));
        server.MapGet("/api/skills/{name}/files/{*path}", (request, _) =>
            skills.TryGetFile(request.RouteValues["name"], request.RouteValues["path"], out string text, out object error)
                ? Task.FromResult<LoopbackResponse?>(LoopbackResponse.Text(text, contentType: "text/markdown; charset=utf-8"))
                : Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(error, 404)));
        server.MapPost("/api/skills/rescan", (_, _) => OkGuarded(() => skills.Rescan()));
        server.MapDelete("/api/skills/{name}", (request, _) => OkGuarded(() => skills.Remove(request.RouteValues["name"])));
        server.MapPost("/api/skills", async (request, ct) =>
        {
            EnsureGuarded(skills.EnsureInstallable);
            using MultipartForm form = await request.ReadFormAsync(ct);
            MultipartFile? file = form.Files.FirstOrDefault();
            if (file is null)
                return LoopbackResponse.Json(new { error = "no skill archive was uploaded" }, 400);
            await using FileStream zip = File.OpenRead(file.TempPath);
            bool overwrite = string.Equals(form["overwrite"], "true", StringComparison.OrdinalIgnoreCase);
            return Json(GuardedValue(() => skills.Install(zip, file.FileName, file.Length, overwrite)));
        });

        // Install from a URL, which is how a skill is actually shared: someone sends a
        // link, not a file a phone has nowhere to put. The body may name ONE archive or
        // a plain-text list of them, one per line, because a collection is the other
        // way skills travel.
        server.MapPost("/api/skills/from-url", async (request, ct) =>
        {
            EnsureGuarded(skills.EnsureInstallable);
            JsonElement body = await request.ReadJsonAsync(ct);
            string url = body.TryGetProperty("url", out JsonElement u) ? (u.GetString() ?? string.Empty).Trim() : string.Empty;
            if (url.Length == 0)
                return LoopbackResponse.Json(new { error = "a url is required" }, 400);
            bool overwrite = body.TryGetProperty("overwrite", out JsonElement o) && o.ValueKind == JsonValueKind.True;
            return Json(await GuardedValueAsync(() => skills.InstallFromUrlAsync(url, overwrite, ct)));
        });
    }

    /// <summary>
    /// The routes that exist only in the app: the model catalog and its downloads,
    /// the saved conversations, and the sandbox switches.
    ///
    /// <para>
    /// The desktop server has no equivalent because its models are files an operator
    /// put on disk and its chat history lives in the page. On a phone both have to be
    /// managed by the app, so these are additions rather than replacements, and they
    /// sit under their own prefix to keep that obvious.
    /// </para>
    /// </summary>
    /// <summary>
    /// Serve the files a model's own code produced.
    ///
    /// <para>
    /// The runner hands the page a link of the form
    /// <c>/api/code/artifacts/{runId}/{path}</c> (WebUiChatService.DefaultArtifactUriPrefix),
    /// and until this existed the app mapped no such route: the model correctly
    /// announced "here is your PDF", the link rendered, and tapping it produced
    /// "error: not found". TensorSharp.Server has had the endpoint all along
    /// (CodeArtifactEndpoints); only the app was missing it, so every generated
    /// document -- PDF, PPTX, XLSX, chart -- was unreachable on a phone.
    /// </para>
    ///
    /// <para>
    /// Everything here was written by a program a model wrote, so it is served the
    /// same defensive way the server serves it: confinement is re-checked by
    /// <see cref="CodeArtifactStore.TryResolve"/> rather than trusted from the route,
    /// the response is always an attachment, and the content type is never guessed
    /// into something a WebView would execute.
    /// </para>
    /// </summary>
    public static void MapCodeArtifacts(this LoopbackServer server, CodeArtifactStore artifacts)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(artifacts);

        const string prefix = WebUiChatService.DefaultArtifactUriPrefix;

        // What one run left behind, for a page that wants to list them.
        server.MapGet(prefix + "/{runId}", (request, _) =>
        {
            string runId = request.RouteValues["runId"] ?? string.Empty;
            IReadOnlyList<CodeArtifact> files = artifacts.List(
                runId, (id, rel, _) => CodeArtifactStore.UrlFor(prefix, id, rel));
            return Task.FromResult<LoopbackResponse?>(files.Count == 0
                ? LoopbackResponse.Json(new { error = "no files are held for that run" }, 404)
                : LoopbackResponse.Json(new
                {
                    runId,
                    files = files.Select(a => new { path = a.Path, bytes = a.Bytes, url = a.Pointer }).ToArray(),
                }));
        });

        // Catch-all so a nested path such as out/report.pdf binds whole.
        server.MapGet(prefix + "/{runId}/{*path}", (request, _) =>
        {
            string runId = request.RouteValues["runId"] ?? string.Empty;
            string path = request.RouteValues["path"] ?? string.Empty;
            if (!artifacts.TryResolve(runId, path, out string? full, out string? error))
                return Task.FromResult<LoopbackResponse?>(
                    LoopbackResponse.Json(new { error = error ?? "not found" }, 404));

            return Task.FromResult<LoopbackResponse?>(LoopbackResponse.File(
                full!, ContentTypes.For(full!), attachment: true, downloadName: Path.GetFileName(path)));
        });
    }

    public static void MapAgent(
        this LoopbackServer server,
        IReadOnlyList<CatalogModel> catalog,
        ModelStore models,
        ConversationStore conversations,
        SettingsStore settings,
        Func<string>? describeEngine = null,
        Action<string, JsonElement>? onPageEvent = null,
        ModelDownloadManager? downloads = null,
        Action<AppSettings>? onSettingsChanged = null,
        Func<object>? describeModel = null,
        ShareIntake? shares = null,
        Func<bool>? hasShareContainer = null,
        Func<string, bool>? discardShare = null)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(models);
        CatalogModel? Find(string id) => catalog.FirstOrDefault(m => string.Equals(m.Id, id, StringComparison.Ordinal));
        ArgumentNullException.ThrowIfNull(conversations);
        ArgumentNullException.ThrowIfNull(settings);

        server.MapGet("/api/agent/catalog", (_, _) => Ok(new
        {
            models = catalog.Select(m => Describe(m, models, downloads)).ToArray(),
        }));

        server.MapGet("/api/agent/catalog/{id}", (request, _) =>
        {
            CatalogModel? model = Find(request.RouteValues["id"]);
            return model is null
                ? Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { error = "no such model" }, 404))
                : Ok(Describe(model, models, downloads));
        });

        // What is transferring right now, whoever started it. A page that was closed
        // while a model was downloading needs this to find the job again when it
        // reopens; without it the only evidence a download existed was the stream the
        // page had already dropped.
        server.MapGet("/api/agent/downloads", (_, _) => Ok(new
        {
            downloads = (downloads?.All ?? Array.Empty<ModelDownloadStatus>()).Select(Describe).ToArray(),
        }));

        // Downloads are long and resumable, so this is a stream rather than a request
        // that blocks for gigabytes: the page shows progress from these frames and a
        // dropped connection leaves the .part file to be resumed, not restarted.
        //
        // The stream is a WINDOW on the transfer, never its owner. It used to be the
        // owner — the download ran on the request's cancellation token, so closing the
        // page, or opening any other one, killed a five-gigabyte transfer partway. The
        // job now lives in the download manager and outlives every reader; this route
        // starts it if it is not already running and reports what it is doing.
        server.MapPost("/api/agent/catalog/{id}/download", (request, ct) =>
        {
            CatalogModel? model = Find(request.RouteValues["id"]);
            if (model is null)
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { error = "no such model" }, 404));
            if (model.SideloadOnly)
            {
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new
                {
                    error = $"{model.DisplayName} has no verified publisher URL; import {model.Weights.FileName} from the native Models page.",
                }, 409));
            }

            IReadOnlyCollection<CatalogFileRole>? optional = OptionalRoles(settings);
            if (downloads is null)
            {
                // No manager (a host built for a test that does not need one): the old
                // request-scoped behaviour, so the route still answers.
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                    DownloadFrames(models, model, optional, ct), request.Cancellation));
            }

            downloads.Start(model, optional);
            return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                WatchFrames(downloads, model.Id, ct), request.Cancellation));
        });

        // Stopping is now something a caller has to ASK for, which is the other half of
        // a download that survives its reader.
        server.MapPost("/api/agent/catalog/{id}/download/cancel", (request, _) =>
        {
            string id = request.RouteValues["id"];
            return Ok(new { cancelled = downloads?.Cancel(id) ?? false, id });
        });

        server.MapDelete("/api/agent/catalog/{id}", (request, _) =>
        {
            CatalogModel? model = Find(request.RouteValues["id"]);
            if (model is null)
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { error = "no such model" }, 404));
            // Stopped first: deleting the directory under a running transfer leaves the
            // downloader writing into a path that no longer has a parent, and the error
            // it raises describes the symptom rather than the delete that caused it.
            downloads?.Cancel(model.Id);
            models.Delete(model);
            return Ok(Describe(model, models, downloads));
        });

        server.MapGet("/api/agent/conversations", (_, _) => Ok(new { conversations = conversations.List() }));
        server.MapPost("/api/agent/conversations", (_, _) => Ok(conversations.Create()));
        server.MapGet("/api/agent/conversations/{id}", (request, _) =>
        {
            Conversation? conversation = conversations.Load(request.RouteValues["id"]);
            return conversation is null
                ? Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { error = "no such conversation" }, 404))
                : Ok(conversation);
        });
        server.MapPost("/api/agent/conversations/{id}", async (request, ct) =>
        {
            Conversation? conversation = conversations.Load(request.RouteValues["id"]);
            if (conversation is null)
                return LoopbackResponse.Json(new { error = "no such conversation" }, 404);
            JsonElement body = await request.ReadJsonAsync(ct);
            if (body.TryGetProperty("title", out JsonElement title) && title.GetString() is { Length: > 0 } text)
                conversations.Rename(conversation.Id, text);
            return LoopbackResponse.Json(conversations.Load(conversation.Id)!);
        });
        server.MapDelete("/api/agent/conversations/{id}", (request, _) =>
            Ok(new { deleted = conversations.Delete(request.RouteValues["id"]) }));

        server.MapGet("/api/agent/settings", (_, _) => Ok(settings.Load()));
        server.MapPost("/api/agent/settings", async (request, ct) =>
        {
            AppSettings updated = JsonSerializer.Deserialize<AppSettings>(
                (await request.ReadJsonAsync(ct)).GetRawText(), SseFraming.JsonOptions) ?? new AppSettings();
            settings.Save(updated);
            AppSettings saved = settings.Load();
            // Applied, not merely stored. Saving alone is what made "Allow network
            // access" a switch that did nothing until the app was force-quit.
            onSettingsChanged?.Invoke(saved);
            return LoopbackResponse.Json(saved);
        });

        // The page tells the app what it just did — which conversation it bound, when
        // it finished loading, when a generation started or stopped — so the native
        // chrome around the WebView can follow along. A WebView message handler would
        // be the platform way; this is the same thing over the transport that already
        // exists, which keeps the injected script free of any iOS-specific API.
        server.MapPost("/api/agent/events", async (request, ct) =>
        {
            JsonElement message = await request.ReadJsonAsync(ct);
            string kind = message.TryGetProperty("type", out JsonElement type) ? type.GetString() ?? string.Empty : string.Empty;
            onPageEvent?.Invoke(kind, message);
            return LoopbackResponse.Json(new { ok = true });
        });

        // A share is leased, not consumed, by claim. It remains durable after the page
        // updates the draft and attachment chips; only an accepted send or the explicit
        // discard route consumes it, so reloads cannot eat user data.
        server.MapGet("/api/agent/share", (_, _) => Ok(new
        {
            pending = shares?.PendingCount ?? 0,
            container = hasShareContainer?.Invoke() ?? false,
        }));

        server.MapPost("/api/agent/share/claim", (_, _) =>
        {
            PendingShare? share = shares?.Peek();
            return Ok(new
            {
                share = share is null ? null : new
                {
                    id = share.Id,
                    text = share.Text,
                    attachments = share.Attachments,
                    notices = share.Notices,
                    title = share.Title,
                    newChat = share.NewChat,
                    autoSend = share.AutoSend,
                },
            });
        });

        server.MapPost("/api/agent/share/discard", async (request, ct) =>
        {
            JsonElement body = await request.ReadJsonAsync(ct);
            string id = body.ValueKind == JsonValueKind.Object
                && body.TryGetProperty("id", out JsonElement value)
                && value.ValueKind == JsonValueKind.String
                    ? value.GetString() ?? string.Empty
                    : string.Empty;
            bool discarded = id.Length > 0 && (discardShare?.Invoke(id) ?? false);
            return LoopbackResponse.Json(
                new { ok = discarded },
                discarded ? 200 : 409);
        });

        server.MapGet("/api/agent/engine", (_, _) => Ok(new
        {
            engine = describeEngine?.Invoke() ?? "unknown",
            // The page needs to RECOGNISE a network refusal to offer the switch that
            // fixes it, and the one thing it must not do is keep its own copy of the
            // wording: two spellings of the same message drift, and the day they do
            // the offer silently stops appearing. It is sent from the one definition.
            networkDisabledMessage = Sandbox.ExecutionPolicy.NetworkDisabledMessage,
            modelRoot = models.Root,
            conversationRoot = conversations.Root,
            // What the app is doing about the model the user last used. The page needs
            // it to tell "no model has ever been chosen" — which asks the user to go and
            // choose one — apart from "the weights are being read right now", which asks
            // them to wait a few seconds.
            model = describeModel?.Invoke(),
        }));
    }

    /// <summary>
    /// One file out of the upload directory, or a 404.
    ///
    /// <para>
    /// Two rules, both the Server's rather than this file's. The content type comes
    /// from <see cref="UploadContentPolicy"/>, which serves every text and code
    /// extension as <c>text/plain</c> — an uploaded HTML page must never run in the
    /// origin that holds the launch token — and an extension the policy does not list
    /// is a 404 rather than a download of unknown type. The path is resolved and then
    /// checked for containment, because <c>..</c> survives URL decoding and the
    /// directory above this one holds the conversations and the settings.
    /// </para>
    /// </summary>
    private static LoopbackResponse? ServeUpload(string uploadDirectory, string relative)
    {
        if (string.IsNullOrEmpty(relative) || relative.Contains("..", StringComparison.Ordinal))
            return LoopbackResponse.Json(new { error = "not found" }, 404);

        string root = Path.GetFullPath(uploadDirectory).TrimEnd(Path.DirectorySeparatorChar);
        string full = Path.GetFullPath(Path.Combine(root, relative));
        if (!full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal) || !File.Exists(full))
            return LoopbackResponse.Json(new { error = "not found" }, 404);

        return UploadContentPolicy.ServeContentTypes.TryGetValue(Path.GetExtension(full), out string? contentType)
            ? LoopbackResponse.File(full, contentType)
            : LoopbackResponse.Json(new { error = "not found" }, 404);
    }

    /// <summary>
    /// The optional files a download includes. Worth their bytes on a phone, but the
    /// user pays for them, so the setting decides rather than the catalog — and the
    /// LIST is the manager's, so this route and the model list cannot disagree about
    /// which companions a model ends up with.
    /// </summary>
    private static IReadOnlyCollection<CatalogFileRole>? OptionalRoles(SettingsStore settings) =>
        ModelDownloadManager.OptionalRolesFor(settings.Load().DownloadOptionalFiles);

    /// <summary>
    /// One running download, framed exactly as <see cref="DownloadFrames"/> framed it.
    ///
    /// <para>
    /// Byte-compatible on purpose: the shape is what the model list reads, and changing
    /// it at the same time as changing who owns the transfer would have made a UI that
    /// stopped updating impossible to attribute to either half.
    /// </para>
    /// </summary>
    private static async IAsyncEnumerable<object> WatchFrames(
        ModelDownloadManager downloads, string modelId,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (ModelDownloadStatus status in downloads.WatchAsync(modelId, ct).ConfigureAwait(false))
        {
            if (status.IsRunning)
            {
                yield return Frame(status.Progress);
                continue;
            }
            yield return status.State switch
            {
                DownloadState.Completed => new { done = true, id = modelId },
                DownloadState.Cancelled => (object)new { cancelled = true, id = modelId },
                _ => new { error = status.Error ?? "the download failed", id = modelId },
            };
        }
    }

    private static object Frame(ModelDownloadProgress value) => new
    {
        file = value.FileName,
        fileIndex = value.FileIndex,
        fileCount = value.FileCount,
        phase = value.Phase,
        received = value.BytesReceived,
        total = value.TotalBytes,
        fraction = value.Fraction,
        bytesPerSecond = value.BytesPerSecond,
        etaSeconds = value.Eta?.TotalSeconds,
    };

    private static object Describe(ModelDownloadStatus status) => new
    {
        id = status.ModelId,
        state = status.State.ToString(),
        running = status.IsRunning,
        error = status.Error,
        progress = Frame(status.Progress),
    };

    private static async IAsyncEnumerable<object> DownloadFrames(
        ModelStore store, CatalogModel model, IReadOnlyCollection<CatalogFileRole>? optionalRoles,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var frames = System.Threading.Channels.Channel.CreateUnbounded<object>();

        // The store reports progress on whatever thread the download is reading on,
        // and the response has to be written from this one, so the two are joined by
        // a channel rather than by a lock.
        var progress = new ChannelProgress(frames.Writer);
        Task download = Task.Run(async () =>
        {
            try
            {
                await store.DownloadAsync(model, progress, ct, optionalRoles).ConfigureAwait(false);
                frames.Writer.TryWrite(new { done = true, id = model.Id });
            }
            catch (OperationCanceledException)
            {
                frames.Writer.TryWrite(new { cancelled = true, id = model.Id });
            }
            catch (Exception ex)
            {
                frames.Writer.TryWrite(new { error = ex.Message, id = model.Id });
            }
            finally
            {
                frames.Writer.TryComplete();
            }
        }, ct);

        await foreach (object frame in frames.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return frame;
        await download.ConfigureAwait(false);
    }

    private sealed class ChannelProgress(System.Threading.Channels.ChannelWriter<object> writer)
        : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value) => writer.TryWrite(Frame(value));
    }

    private static object Describe(CatalogModel model, ModelStore store, ModelDownloadManager? downloads = null) => new
    {
        id = model.Id,
        name = model.DisplayName,
        family = model.Family.ToString(),
        parameters = model.Parameters,
        quantization = model.Quantization,
        notes = model.Notes,
        license = model.License,
        contextLength = model.ContextLength,
        modalities = model.Modalities.ToString(),
        kind = model.Kind.ToString(),
        experimental = model.Experimental,
        sideloadOnly = model.SideloadOnly,
        minDeviceMemoryGB = model.MinDeviceMemoryGB,
        totalBytes = model.TotalBytes,
        state = store.StateOf(model).ToString(),
        installedBytes = store.InstalledBytes(model),
        remainingBytes = store.RemainingBytes(model),
        path = store.WeightsPath(model),
        // Null unless this launch has a download for it. "Partly downloaded" and
        // "downloading right now" look identical on disk, and only one of them means
        // the user should wait rather than tap.
        download = downloads?.StatusOf(model.Id) is { } status ? Describe(status) : null,
    };

    /// <summary>
    /// Pass the turn's frames through, and write the answer down when it ends.
    ///
    /// <para>
    /// The frames are the only place the assistant's reply exists on this side: the
    /// service streams it and the page assembles it. Reassembling it here, from the
    /// same <c>token</c>, <c>replace</c> and <c>thinking</c> events the page reads,
    /// is what lets a chat survive the app being killed the moment after an answer
    /// appears. An aborted turn is still saved, because the partial answer is what
    /// the user is looking at.
    /// </para>
    /// </summary>
    private static async IAsyncEnumerable<object> Recording(IAsyncEnumerable<object> frames, ConversationRecorder? recorder)
    {
        if (recorder is null)
        {
            await foreach (object frame in frames.ConfigureAwait(false))
                yield return frame;
            yield break;
        }

        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var artifacts = new List<StoredArtifact>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        string? sessionId = null;

        await foreach (object frame in frames.ConfigureAwait(false))
        {
            yield return frame;

            // The frames are anonymous objects, so they are read the way the page
            // reads them: as JSON. Serialising each one costs a little, and it is the
            // only way to stay honest about what was actually sent.
            using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(frame, SseFraming.JsonOptions));
            JsonElement root = document.RootElement;
            if (root.TryGetProperty("token", out JsonElement token) && token.GetString() is { } piece)
                content.Append(piece);
            else if (root.TryGetProperty("replace", out JsonElement replace) && replace.GetString() is { } whole)
                content.Clear().Append(whole);
            else if (root.TryGetProperty("thinking", out JsonElement thought) && thought.GetString() is { } reasoning)
                thinking.Append(reasoning);
            if (root.TryGetProperty("sessionId", out JsonElement id) && id.GetString() is { Length: > 0 } value)
                sessionId = value;
            CollectArtifacts(root, artifacts, seen);
        }

        if (sessionId is not null)
            recorder.Complete(sessionId, content.ToString(), thinking.ToString(), artifacts);
    }

    /// <summary>
    /// The files a turn's tools produced, off the <c>skill_step</c> frames that carry
    /// them. Collected here rather than trusted to the page, for the same reason the
    /// answer is: the page may be gone by the time the turn ends.
    /// </summary>
    private static void CollectArtifacts(JsonElement frame, List<StoredArtifact> into, HashSet<string> seen)
    {
        if (!frame.TryGetProperty("files", out JsonElement files) || files.ValueKind != JsonValueKind.Array)
            return;
        foreach (JsonElement file in files.EnumerateArray())
        {
            if (file.ValueKind != JsonValueKind.Object)
                continue;
            string? url = file.TryGetProperty("url", out JsonElement u) ? u.GetString() : null;
            if (string.IsNullOrEmpty(url) || !seen.Add(url))
                continue;
            into.Add(new StoredArtifact
            {
                Name = (file.TryGetProperty("name", out JsonElement n) ? n.GetString() : null) ?? url,
                Bytes = file.TryGetProperty("bytes", out JsonElement b) && b.TryGetInt64(out long bytes) ? bytes : 0,
                Url = url,
            });
        }
    }

    /// <summary>
    /// The response header that names the turn a stream is a view of. A page that has
    /// this can stop the generation, and can tell whether the stream it is reading is
    /// still the one it started.
    /// </summary>
    public const string TurnHeader = "X-TensorAgent-Turn";

    /// <summary>
    /// Which turn a chat request belongs to: its conversation, so that the SAME
    /// conversation reached through a new engine session — which is what a reloaded page
    /// gets — finds the answer that is still being written for it. Only a request with
    /// no conversation at all falls back to its session.
    /// </summary>
    private static string TurnKeyFor(JsonElement body, ConversationRecorder? recorder)
    {
        string session = body.ValueKind == JsonValueKind.Object
            && body.TryGetProperty("sessionId", out JsonElement id)
                ? id.GetString() ?? string.Empty
                : string.Empty;
        if (session.Length > 0 && recorder?.ConversationFor(session) is { Length: > 0 } conversation)
            return conversation;
        return session.Length > 0 ? "session:" + session : "chat";
    }

    /// <summary>One turn, as the page reads it. Null stays null: "there is nothing running".</summary>
    private static object? Describe(ChatTurnStatus? turn) => turn is null ? null : new
    {
        id = turn.Id,
        state = turn.State.ToString(),
        running = turn.IsRunning,
        frames = turn.FrameCount,
    };

    /// <summary>
    /// Read the session id out of whatever shape the chat service returned. It is an
    /// anonymous type, so a round trip through JSON is the only way to it — cheap, and
    /// it fails loudly if the service ever renames the member.
    /// </summary>
    private static string SessionIdOf(object created)
    {
        using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(created, SseFraming.JsonOptions));
        return document.RootElement.GetProperty("sessionId").GetString()
            ?? throw new InvalidOperationException("the chat service created a session with no id");
    }

    /// <summary>
    /// The chat service refuses a request by throwing, carrying the status and the JSON
    /// body it wants sent. The loopback transport has its own exception for exactly
    /// that, so this is the one place the two vocabularies meet — and it is a
    /// translation, never a reinterpretation: the status and the payload are passed
    /// through untouched so the page sees what the desktop's page sees.
    /// </summary>
    private static async Task<object> Guarded(Func<Task<object>> call)
    {
        try { return await call().ConfigureAwait(false); }
        catch (WebUiRequestRejectedException ex) { throw new LoopbackHttpException(ex.StatusCode, ex.Payload); }
    }

    private static async IAsyncEnumerable<object> Guarded(IAsyncEnumerable<object> frames)
    {
        await using IAsyncEnumerator<object> enumerator = frames.GetAsyncEnumerator();
        while (true)
        {
            object current;
            try
            {
                if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                    yield break;
                current = enumerator.Current;
            }
            catch (WebUiRequestRejectedException ex)
            {
                throw new LoopbackHttpException(ex.StatusCode, ex.Payload);
            }
            yield return current;
        }
    }

    private static object GuardedValue(Func<object> call)
    {
        try { return call(); }
        catch (WebUiRequestRejectedException ex) { throw new LoopbackHttpException(ex.StatusCode, ex.Payload); }
    }

    private static async Task<object> GuardedValueAsync(Func<Task<object>> call)
    {
        try { return await call(); }
        catch (WebUiRequestRejectedException ex) { throw new LoopbackHttpException(ex.StatusCode, ex.Payload); }
    }

    private static void EnsureGuarded(Action call)
    {
        try { call(); }
        catch (WebUiRequestRejectedException ex) { throw new LoopbackHttpException(ex.StatusCode, ex.Payload); }
    }

    private static Task<LoopbackResponse?> OkGuarded(Func<object> call) =>
        Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(GuardedValue(call)));

    private static Task<LoopbackResponse?> Ok(object payload) =>
        Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(payload));

    private static LoopbackResponse? Json(object payload) => LoopbackResponse.Json(payload);
}
