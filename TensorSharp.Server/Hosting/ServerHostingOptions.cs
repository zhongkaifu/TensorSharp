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
            bool skillsAllowNetwork = false)
        {
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
        public string StartupModelPath { get; }

        /// <summary>Absolute path of the projector the server was launched with, or null when none is hosted.</summary>
        public string StartupMmProjPath { get; }

        /// <summary>Canonical name of the backend chosen at startup (e.g. <c>ggml_metal</c>).</summary>
        public string DefaultBackend { get; }

        /// <summary>Backends actually supported by this host (after probing the GGML runtime).</summary>
        internal IReadOnlyList<BackendOption> SupportedBackends { get; }

        /// <summary>Fast lookup over <see cref="SupportedBackends"/>.</summary>
        internal HashSet<string> SupportedBackendValues { get; }

        /// <summary>
        /// Default generation budget applied by every endpoint (Web UI, Ollama,
        /// OpenAI chat + responses) when the request does not carry its own
        /// limit. Resolved from <c>--max-tokens</c> / <c>MAX_TOKENS</c>, falling
        /// back to 20000.
        /// </summary>
        public int DefaultMaxTokens { get; }

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
        public bool SkillsEnabled { get; }

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
        public bool SkillsAllowScripts { get; }

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
        public bool SkillsAllowNetwork { get; }

        /// <summary>Resolved log directory (used by the file logger when it is enabled).</summary>
        public string LogDirectory { get; }

        /// <summary>True when the file logger should be wired in.</summary>
        public bool FileLoggingEnabled { get; }

        /// <summary>
        /// Default sampling parameters resolved from CLI flags / environment,
        /// together with which of them the operator pinned and whether those
        /// pins outrank a client's request. Adapters seed per-request configs
        /// from this object so unspecified fields take the operator-configured
        /// defaults instead of the hard-coded library defaults. Never null.
        /// </summary>
        public SamplingDefaults SamplingDefaults { get; }

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
