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
using TensorSharp.AgentHost.Skills;
using System.Collections.Generic;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Immutable bag of values resolved at process start-up from CLI arguments
    /// and environment variables. Registered as a DI singleton so every
    /// endpoint, adapter, and helper can pull the same view of "what is hosted
    /// on this server" without each one re-parsing argv.
    /// </summary>
    public sealed class ServerHostingOptions
    {
        /// <summary>
        /// Address the server binds when the operator does not choose one.
        /// <c>0.0.0.0</c> (all interfaces) rather than <c>localhost</c> because
        /// the server is routinely reached from another machine or from outside
        /// a container.
        /// </summary>
        public const string DefaultListenUrls = "http://0.0.0.0:5000";

        /// <summary>Port used by <see cref="DefaultListenUrls"/>.</summary>
        public const int DefaultPort = 5000;

        public ServerHostingOptions(
            string startupModelPath,
            string startupMmProjPath,
            string defaultBackend,
            IReadOnlyList<BackendOption> supportedBackends,
            int defaultMaxTokens,
            bool maxTokensPinned,
            int defaultVideoFrames,
            int defaultVideoFps,
            int defaultVideoWidth,
            int defaultVideoHeight,
            int defaultVideoSteps,
            string defaultVideoMode,
            string uploadDirectory,
            string logDirectory,
            bool fileLoggingEnabled,
            SamplingDefaults samplingDefaults,
            string listenUrls = DefaultListenUrls,
            long uploadMaxFileBytes = UploadStoragePolicy.DefaultMaxFileBytes,
            long uploadQuotaBytes = 0,
            TimeSpan? uploadTtl = null,
            bool webUiEnabled = true,
            IReadOnlyList<string> skillDirectories = null,
            bool skillsEnabled = true,
            bool skillsDiscovery = true,
            bool skillsAllowScripts = false,
            int skillsMaxRounds = 8,
            IReadOnlyList<string> defaultSkills = null,
            SkillSandboxMode skillsSandbox = SkillSandboxMode.Required,
            bool skillsAllowNetwork = false,
            bool prefixCacheEnabled = true,
            string prefixCacheDirectory = null,
            bool embeddingsEnabled = false,
            int embeddingThreads = 0,
            int embeddingContextSize = 0)
        {
            EmbeddingsEnabled = embeddingsEnabled;
            EmbeddingThreads = embeddingThreads;
            EmbeddingContextSize = embeddingContextSize;
            PrefixCacheEnabled = prefixCacheEnabled;
            PrefixCacheDirectory = prefixCacheDirectory;
            WebUiEnabled = webUiEnabled;
            ListenUrls = string.IsNullOrWhiteSpace(listenUrls) ? DefaultListenUrls : listenUrls;
            StartupModelPath = startupModelPath;
            StartupMmProjPath = startupMmProjPath;
            DefaultBackend = defaultBackend;
            SupportedBackends = supportedBackends ?? Array.Empty<BackendOption>();
            SupportedBackendValues = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < SupportedBackends.Count; i++)
                SupportedBackendValues.Add(SupportedBackends[i].Value);
            DefaultMaxTokens = defaultMaxTokens;
            MaxTokensPinned = maxTokensPinned;
            DefaultVideoFrames = defaultVideoFrames;
            DefaultVideoFps = defaultVideoFps;
            DefaultVideoWidth = defaultVideoWidth;
            DefaultVideoHeight = defaultVideoHeight;
            DefaultVideoSteps = defaultVideoSteps;
            DefaultVideoMode = defaultVideoMode;
            UploadDirectory = uploadDirectory;
            LogDirectory = logDirectory;
            FileLoggingEnabled = fileLoggingEnabled;
            SamplingDefaults = samplingDefaults ?? new SamplingDefaults(new SamplingConfig());
            UploadMaxFileBytes = uploadMaxFileBytes;
            UploadQuotaBytes = uploadQuotaBytes;
            UploadTtl = uploadTtl;
            SkillDirectories = skillDirectories ?? Array.Empty<string>();
            SkillsEnabled = skillsEnabled;
            SkillsDiscovery = skillsDiscovery;
            SkillsAllowScripts = skillsAllowScripts;
            SkillsMaxRoundsSpecified = skillsMaxRounds > 0;
            SkillsMaxRounds = skillsMaxRounds > 0 ? skillsMaxRounds : 8;
            DefaultSkills = defaultSkills ?? Array.Empty<string>();
            SkillsSandbox = skillsSandbox;
            SkillsAllowNetwork = skillsAllowNetwork;
        }

        /// <summary>
        /// Semicolon-separated URL(s) Kestrel binds, resolved from
        /// <c>--port</c>/<c>--host</c>, <c>--urls</c>, the <c>PORT</c>/<c>HOST</c>
        /// or <c>ASPNETCORE_URLS</c> environment variables, or
        /// <see cref="DefaultListenUrls"/>. Never null or empty.
        /// </summary>
        public string ListenUrls { get; }

        /// <summary>Host an embedding encoder instead of a chat/generation model.</summary>
        public bool EmbeddingsEnabled { get; }

        /// <summary>True when the embedding encoder runs entirely in managed C# without native backend libraries.</summary>
        public bool UsesManagedEmbeddingBackend => EmbeddingsEnabled && BackendCatalog.Canonicalize(DefaultBackend) == "cpu";

        /// <summary>Embedding CPU threads; zero uses the backend default.</summary>
        public int EmbeddingThreads { get; }

        /// <summary>Embedding token limit; zero uses the model's context length.</summary>
        public int EmbeddingContextSize { get; }

        /// <summary>
        /// False when the operator passed <c>--no-webui</c> (or set
        /// <c>TS_NO_WEBUI</c> to anything but <c>0</c>): the bundled wwwroot UI
        /// is not served and <c>GET /</c> answers with the plain liveness text,
        /// as on a headless deployment that ships no wwwroot content. Every
        /// HTTP API endpoint (including <c>/uploads</c>, whose URLs the image
        /// and video APIs return) stays up.
        /// </summary>
        public bool WebUiEnabled { get; }

        /// <summary>Absolute path of the model the server was launched with, or null when no model is hosted.</summary>
        public string StartupModelPath { get; private set; }

        /// <summary>Absolute path of the projector the server was launched with, or null when none is hosted.</summary>
        public string StartupMmProjPath { get; private set; }

        /// <summary>
        /// Point the "one hosted model per process" invariant at a different pair.
        ///
        /// <para>
        /// The desktop server never calls this: it is launched against one
        /// <c>--model</c> and changing it means restarting, which is the right rule for
        /// a process an operator started with arguments. An app has no arguments and no
        /// operator. When someone picks a model in TensorAgent's own list, telling them
        /// it will apply "next time the app starts" is not a smaller version of the
        /// feature -- on a phone it reads as the button not working, and the chat keeps
        /// saying "No model is configured" while the list says the model is selected.
        /// </para>
        /// <para>
        /// This only moves the guard's target. The caller is still responsible for
        /// actually loading the new pair through <c>ModelService</c>, and for doing it
        /// in an order where no request can see a hosted path whose weights are not
        /// loaded yet.
        /// </para>
        /// </summary>
        /// <param name="modelPath">Absolute path of the model to host, or null for none.</param>
        /// <param name="mmProjPath">Absolute path of its projector, or null for none.</param>
        public void RepointHostedModel(string modelPath, string mmProjPath)
        {
            StartupModelPath = modelPath;
            StartupMmProjPath = mmProjPath;
        }

        /// <summary>
        /// Move the two sandbox permissions a skill's scripts are run under.
        ///
        /// <para>
        /// For the same reason as <see cref="RepointHostedModel"/>, and with the same
        /// division of labour: an operator sets these once on a command line, but an app
        /// user sets them from a switch on a settings screen, and "this will apply the
        /// next time the app starts" is indistinguishable from a switch that does
        /// nothing. The desktop server never calls this.
        /// </para>
        /// <para>
        /// It moves the DEFAULT the next request is planned against. A skill script
        /// already running keeps the terms it was launched with, which is the only
        /// answer that is safe in both directions.
        /// </para>
        /// </summary>
        public void RepointSandboxPermissions(bool allowScripts, bool allowNetwork)
        {
            SkillsAllowScripts = allowScripts;
            SkillsAllowNetwork = allowNetwork;
        }

        /// <summary>
        /// Turn the whole skills feature on or off, for the same reason as the two
        /// above: on a server this is a command-line decision made once, and in an app
        /// it is a switch a user expects to take effect on the next message rather than
        /// after a force-quit.
        ///
        /// <para>
        /// Off is the real thing and not a filter over the roster:
        /// <see cref="Skills.SkillRequestPlan.Create"/> returns no plan at all, so no
        /// skill is declared to the model, none is reachable, and a request that still
        /// names one gets a turn with no skills in it.
        /// </para>
        /// </summary>
        public void RepointSkills(bool enabled) => SkillsEnabled = enabled;

        /// <summary>
        /// Move the default generation budget, for the same reason and with the same
        /// caveat as the two above: an app user sets it on a settings screen, and a
        /// number that only applies after a force-quit is a control that does nothing.
        /// A request that names its own limit is unaffected, as always.
        /// </summary>
        public void RepointGenerationDefaults(int defaultMaxTokens)
        {
            if (defaultMaxTokens > 0)
                DefaultMaxTokens = defaultMaxTokens;
        }

        /// <summary>
        /// Move the sampling defaults a request falls back to, for the same reason as
        /// the four above and one specific to an app: an operator starts a server
        /// against ONE model and puts its card's numbers on the command line, while an
        /// app user switches models from a list and expects each to be sampled the way
        /// its own card says. Gemma 4 asks for temperature 1.0 / top-k 64 / top-p 0.95
        /// and Qwen for 0.7 / 20 / 0.8; serving both from one built-in default samples
        /// at least one of them wrongly, which shows up as quality rather than as an
        /// error. A request that names its own values is unaffected, as always.
        /// </summary>
        public void RepointSamplingDefaults(SamplingDefaults defaults)
        {
            if (defaults != null)
                SamplingDefaults = defaults;
        }

        /// <summary>Canonical name of the backend chosen at startup (e.g. <c>ggml_metal</c>).</summary>
        public string DefaultBackend { get; }

        /// <summary>Backends actually supported by this host (after probing the GGML runtime).</summary>
        public IReadOnlyList<BackendOption> SupportedBackends { get; }

        /// <summary>Fast lookup over <see cref="SupportedBackends"/>.</summary>
        internal HashSet<string> SupportedBackendValues { get; }

        /// <summary>
        /// Default generation budget applied by every endpoint (Web UI, Ollama,
        /// OpenAI chat + responses) when the request does not carry its own
        /// limit. Resolved from <c>--max-tokens</c> / <c>MAX_TOKENS</c>, falling
        /// back to 20000.
        /// </summary>
        public int DefaultMaxTokens { get; private set; }

        /// <summary>
        /// True when <see cref="DefaultMaxTokens"/> came from <c>--max-tokens</c>
        /// or <c>MAX_TOKENS</c> rather than the built-in fallback. A pinned value
        /// also caps requests that ask for more (see <see cref="ResolveMaxTokens"/>).
        /// </summary>
        public bool MaxTokensPinned { get; }

        /// <summary>
        /// Default Wan output frame count used when a video-generation request
        /// omits <c>frames</c>. Zero delegates to the loaded Wan model's native
        /// default (33, or 49 for Wan2.2-TI2V).
        /// </summary>
        public int DefaultVideoFrames { get; }

        /// <summary>
        /// Default Wan MP4 playback rate used when a video-generation request
        /// omits <c>fps</c>. Zero delegates to the loaded Wan model's native
        /// default (16, or 24 for Wan2.2-TI2V).
        /// </summary>
        public int DefaultVideoFps { get; }

        /// <summary>
        /// Default output width used when a video-generation request omits
        /// <c>width</c>, from <c>--video-width</c> (or <c>--width</c>). Zero
        /// delegates to the model's own default, which for MiniMax-H3 takes the
        /// aspect ratio from the conditioning image.
        /// </summary>
        public int DefaultVideoWidth { get; }

        /// <summary>Default output height; see <see cref="DefaultVideoWidth"/>.</summary>
        public int DefaultVideoHeight { get; }

        /// <summary>
        /// Default denoising steps used when a video-generation request omits
        /// <c>steps</c>, from <c>--video-steps</c>. Zero delegates to the model's own
        /// default. This is the main quality/time trade-off after resolution.
        /// </summary>
        public int DefaultVideoSteps { get; }

        /// <summary>
        /// Default conditioning mode used when a video-generation request omits
        /// <c>videoMode</c>, from <c>--video-mode</c>. Null lets the model infer it
        /// from what the request supplies, which is right for every model except a
        /// deliberately-pinned deployment.
        /// </summary>
        public string DefaultVideoMode { get; }

        /// <summary>Absolute path to the directory used for user uploads.</summary>
        public string UploadDirectory { get; }

        /// <summary>
        /// Per-file cap in bytes on client-originated upload-directory writes,
        /// from <c>--upload-max-mb</c> / <c>TS_UPLOAD_MAX_MB</c>. Defaults to
        /// the 500 MB request-body limit, i.e. no additional restriction.
        /// </summary>
        public long UploadMaxFileBytes { get; }

        /// <summary>
        /// Total upload-directory budget in bytes, from <c>--upload-quota-mb</c>
        /// / <c>TS_UPLOAD_QUOTA_MB</c>. 0 (the default) disables the quota.
        /// </summary>
        public long UploadQuotaBytes { get; }

        /// <summary>
        /// Age after which upload-directory files are deleted, from
        /// <c>--upload-ttl-hours</c> / <c>TS_UPLOAD_TTL_HOURS</c>. Null (the
        /// default) disables cleanup: chat sessions reference attachments by
        /// path and may legitimately reuse them much later.
        /// </summary>
        public TimeSpan? UploadTtl { get; }

        /// <summary>
        /// Directories scanned for Agent Skills, in precedence order, from
        /// <c>--skills-dir</c> / <c>TS_SKILLS_DIR</c>. Defaults to the single
        /// <c>skills/</c> directory next to the server binary, which is created
        /// on startup so an operator can drop a skill directory in and restart.
        /// </summary>
        public IReadOnlyList<string> SkillDirectories { get; }

        /// <summary>
        /// False when the operator passed <c>--no-skills</c> (or set
        /// <c>TS_NO_SKILLS</c> to anything but <c>0</c>): the skills API is not
        /// mapped, no directory is scanned, and a <c>skills</c> field on a chat
        /// request is rejected rather than silently ignored.
        /// </summary>
        public bool SkillsEnabled { get; private set; }

        /// <summary>
        /// Whether a chat request that selects no skill still sees the rest of
        /// the registry advertised, so the model can pick up one the caller did
        /// not name. From <c>--skills-no-discovery</c>; a request may override it
        /// per call with <c>"skills_discovery"</c>.
        /// </summary>
        public bool SkillsDiscovery { get; }

        /// <summary>
        /// True when <c>--skills-allow-exec</c> / <c>TS_SKILLS_ALLOW_EXEC</c> lets
        /// the model run a skill's bundled scripts.
        ///
        /// <para>
        /// Off by default, and it should stay off on any server that accepts skill
        /// uploads: a skill is content someone else supplied, so running its scripts
        /// is arbitrary code execution on this host, under this process's account,
        /// chosen by a model reading that same person's Markdown.
        /// </para>
        /// </summary>
        public bool SkillsAllowScripts { get; private set; }

        /// <summary>
        /// How many times a model may fetch skill content in one turn before it must
        /// answer, from <c>--skills-max-rounds</c> / <c>TS_SKILLS_MAX_ROUNDS</c>.
        /// Each round is a full generation, so this bounds what one malfunctioning
        /// request can cost.
        /// </summary>
        public int SkillsMaxRounds { get; }

        /// <summary>
        /// True when the operator chose <see cref="SkillsMaxRounds"/> rather than taking
        /// the default.
        ///
        /// <para>
        /// The distinction exists because the default has to mean different things for
        /// different work. Eight rounds is generous for fetching skill files and far too
        /// few once the same budget also gates writing a program, running it, reading the
        /// traceback and fixing it — a plan that offers code execution raises its own
        /// default. An operator's explicit number is never raised.
        /// </para>
        /// </summary>
        public bool SkillsMaxRoundsSpecified { get; }

        /// <summary>
        /// Skills made active for every request that does not name its own, from
        /// <c>--skill</c>. A request's <c>skills</c> array replaces this rather than
        /// adding to it, so a client can always narrow the selection - including to
        /// nothing, by sending an empty array.
        /// </summary>
        public IReadOnlyList<string> DefaultSkills { get; }

        /// <summary>
        /// How hard this server insists on OS isolation for a skill's scripts, from
        /// <c>--skills-sandbox</c> / <c>TS_SKILLS_SANDBOX</c>. Required by default, so a
        /// host with no sandbox refuses to run them rather than running them unconfined.
        /// </summary>
        public SkillSandboxMode SkillsSandbox { get; }

        /// <summary>
        /// Whether a sandboxed skill script may reach the network, from
        /// <c>--skills-allow-network</c> / <c>TS_SKILLS_ALLOW_NETWORK</c>. Off by
        /// default: denying it is what stops a script that read something it should not
        /// from sending it anywhere.
        /// </summary>
        public bool SkillsAllowNetwork { get; private set; }

        /// <summary>Resolved log directory (used by the file logger when it is enabled).</summary>
        public string LogDirectory { get; }

        /// <summary>
        /// Whether this server prepares the prompt every conversation shares before it
        /// starts serving, and keeps the result between launches.
        ///
        /// <para>
        /// The shared prefix is the system block the server itself builds — the skills
        /// catalog and the code tools — and on an agent configuration it is thousands of
        /// tokens: 6,459 of a first message's 6,475 on the shipped Qwen configuration.
        /// Whoever crosses that boundary first pays for it, and on a fresh process that is
        /// the user's first message: 21.8 s measured, against 0.65 s for every chat after
        /// it. Preparing it at load moves that cost off the user's first message, and
        /// keeping it on disk means the next launch restores it instead of computing it
        /// again.
        /// </para>
        /// <para>
        /// Turned off with <c>--no-prefix-cache</c>, which is the flag to reach for when
        /// the disk is read-only, when startup latency matters more than first-token
        /// latency, or when diagnosing whether a wrong answer came from a restored state.
        /// The host applies the same flag to the scheduler to disable runtime reuse.
        /// </para>
        /// </summary>
        public bool PrefixCacheEnabled { get; }

        /// <summary>
        /// Where kept checkpoints live, or null for the default beside the binary.
        /// Set with <c>TENSORSHARP_PREFIX_CACHE_DIR</c>, on the same precedent as
        /// <c>TENSORSHARP_LOG_DIR</c>: a path is an operator's deployment detail rather
        /// than a behaviour, so it stays out of the flag surface.
        /// </summary>
        public string PrefixCacheDirectory { get; }

        /// <summary>True when the file logger should be wired in.</summary>
        public bool FileLoggingEnabled { get; }

        /// <summary>
        /// Default sampling parameters resolved from CLI flags / environment,
        /// together with which of them the operator pinned and whether those
        /// pins outrank a client's request. Adapters seed per-request configs
        /// from this object so unspecified fields take the operator-configured
        /// defaults instead of the hard-coded library defaults. Never null.
        /// </summary>
        public SamplingDefaults SamplingDefaults { get; private set; }

        /// <summary>The resolved default sampling values (without the pinning metadata).</summary>
        public SamplingConfig DefaultSamplingConfig => SamplingDefaults.Values;

        /// <summary>
        /// Resolve the generation budget for one request. An absent (or
        /// non-positive, e.g. Ollama's <c>num_predict: -1</c>) request value
        /// takes the server default. A request that asks for more than a pinned
        /// <c>--max-tokens</c> is clamped to it — the flag names a *maximum*, and
        /// an operator who sized it against their KV cache should not have a
        /// client talk them out of it. A request asking for less is always
        /// honoured, so short completions stay short.
        /// </summary>
        public int ResolveMaxTokens(int? requestedMaxTokens)
        {
            if (!requestedMaxTokens.HasValue || requestedMaxTokens.Value <= 0)
                return DefaultMaxTokens;
            return MaxTokensPinned
                ? Math.Min(requestedMaxTokens.Value, DefaultMaxTokens)
                : requestedMaxTokens.Value;
        }
    }
}
