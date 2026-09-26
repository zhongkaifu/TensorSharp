// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorAgent.Core.Settings;

/// <summary>User-controlled settings. Persisted as JSON; every field has a safe default so a
/// missing or older file still loads.</summary>
public sealed class AppSettings
{
    /// <summary>Catalog id of the model the app loads at start and uses for new chats.</summary>
    [JsonPropertyName("selectedModelId")] public string? SelectedModelId { get; set; }

    /// <summary>Whether the model may run programs and skill scripts (the shell tool,
    /// skills_run). Off means the tools are not even declared to the model.</summary>
    [JsonPropertyName("allowCodeExecution")] public bool AllowCodeExecution { get; set; } = true;

    /// <summary>Whether programs and scripts may reach the network (package installs, HTTP).
    /// Enforced in-process by the shell's builtins and Python's audit hook.</summary>
    /// <summary>
    /// Hosts code may reach when <see cref="AllowNetwork"/> is on. Empty means any
    /// host, which is the default: a list is a narrowing the user opts into, not a
    /// default that would quietly break every fetch the first time someone enables
    /// the network. Matched by exact name or as a parent domain.
    /// </summary>
    [JsonPropertyName("networkHosts")] public List<string> NetworkHosts { get; set; } = new();

    [JsonPropertyName("allowNetwork")] public bool AllowNetwork { get; set; } = false;


    /// <summary>Generation cap sent as maxTokens; the Web UI's server default is 20000,
    /// which is far past what a phone should decode in one turn.</summary>
    [JsonPropertyName("maxTokens")] public int MaxTokens { get; set; } = 2048;

    /// <summary>Default state of the Reasoning toggle for new chats.</summary>
    [JsonPropertyName("thinkByDefault")] public bool ThinkByDefault { get; set; } = false;

    /// <summary>Override of the catalog entry's context length (0 = catalog default).</summary>
    [JsonPropertyName("contextLength")] public int ContextLength { get; set; }

    /// <summary>
    /// How precisely the K/V cache is stored: <c>f16</c>, <c>q8_0</c> or <c>q4_0</c>.
    ///
    /// <para>
    /// The cache, not the weights. On a phone it is often the larger half of what a
    /// loaded model costs -- a 32k window is gigabytes, and on Metal every token is
    /// charged twice (see <see cref="Hosting.EngineMemoryPolicy"/>) -- so dropping it
    /// from 16 to 4 bits per value buys back more memory than any weight choice left
    /// on the table. q4_0 by default because that is the trade a phone should make.
    /// </para>
    /// <para>
    /// A REQUEST, not a guarantee. Architectures whose attention cannot read a
    /// block-quantized cache refuse it at load and use f16 instead -- Gemma 4, whose
    /// sliding-window layers keep a circular cache with float-only helpers, and
    /// GPT-OSS -- so on those this setting changes nothing and is not allowed to.
    /// Read at model construction, so a change applies to the next model that loads.
    /// </para>
    /// </summary>
    [JsonPropertyName("kvCacheDtype")] public string KvCacheDtype { get; set; } = "q4_0";

    /// <summary>Skills selected by default for new chats.</summary>
    [JsonPropertyName("defaultSkills")] public List<string> DefaultSkills { get; set; } = new();

    /// <summary>
    /// Whether the skills feature is available at all.
    ///
    /// <para>
    /// Off is not "no skill is ticked": no skill is declared to the model, so a turn
    /// costs nothing for the roster and nothing can decide to read one. That is the
    /// point — every bundled skill (ten today) announces itself in every prompt, which is thousands
    /// of tokens on a phone, and a user who wants a plain chat model should be able to
    /// have one.
    /// </para>
    /// </summary>
    [JsonPropertyName("skillsEnabled")] public bool SkillsEnabled { get; set; } = true;

    /// <summary>
    /// Whether the model may delegate parts of a turn to sub-agents: the
    /// <c>spawn_agent</c> family of tools and the coordination prompt that comes with
    /// them.
    ///
    /// <para>
    /// On by default, because that is what the app did before there was a switch. A
    /// sub-agent is a separate conversation on the loaded model, with its own session
    /// and generation state beside the chat's, and while this is on every prompt of a
    /// model that can call tools also declares the five coordination tools, whether or
    /// not it delegates. On a phone both cost memory and time, and a user who wants a
    /// single agent should be able to have one. Off means the tools are not declared at
    /// all, exactly as on a server started with <c>--no-multi-agent</c>. Applied to the
    /// running app at once; it takes effect on the next message.
    /// </para>
    /// </summary>
    [JsonPropertyName("multiAgentEnabled")] public bool MultiAgentEnabled { get; set; } = true;

    /// <summary>Keep the screen awake while generating.</summary>
    /// <summary>
    /// BCP-47 language the speech recogniser listens in, or empty to follow the
    /// device. iOS recognises ONE language per session -- it does not detect which
    /// one is being spoken -- so a bilingual user has to be able to say which, and
    /// the device language is only ever right for one of them.
    /// </summary>
    [JsonPropertyName("speechLanguage")] public string SpeechLanguage { get; set; } = string.Empty;

    [JsonPropertyName("keepAwakeWhileGenerating")] public bool KeepAwakeWhileGenerating { get; set; } = true;

    /// <summary>Allow model downloads over cellular.</summary>
    [JsonPropertyName("allowCellularDownloads")] public bool AllowCellularDownloads { get; set; } = false;

    /// <summary>Whether the optional projector/draft files are downloaded with a model.</summary>
    [JsonPropertyName("downloadOptionalFiles")] public bool DownloadOptionalFiles { get; set; } = true;

    /// <summary>
    /// Speculative decoding: draft a few tokens ahead (the model's draft head when it
    /// is downloaded, otherwise a lookup over the conversation's own tokens) and
    /// verify them in one forward. Same output, fewer forwards; the engine parks it
    /// by itself while it is not paying. Applied to the running engine at once.
    /// </summary>
    [JsonPropertyName("speculativeDecoding")] public bool SpeculativeDecoding { get; set; } = true;

    /// <summary>Per-turn wall-clock limit for a single tool call, seconds.</summary>
    [JsonPropertyName("toolTimeoutSeconds")] public int ToolTimeoutSeconds { get; set; } = 120;

    public AppSettings Clone() => (AppSettings)MemberwiseClone();
}

/// <summary>Loads and saves <see cref="AppSettings"/> atomically at a fixed path.</summary>
public sealed class SettingsStore
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private readonly object _lock = new();

    public string Path { get; }

    public SettingsStore(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = System.IO.Path.GetFullPath(path);
    }

    public AppSettings Load()
    {
        lock (_lock)
        {
            if (!File.Exists(Path))
                return new AppSettings();
            try
            {
                using FileStream stream = File.OpenRead(Path);
                return JsonSerializer.Deserialize<AppSettings>(stream, Json) ?? new AppSettings();
            }
            catch (JsonException)
            {
                return new AppSettings();
            }
        }
    }

    public void Save(AppSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        lock (_lock)
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
            string tmp = Path + ".tmp";
            using (FileStream stream = File.Create(tmp))
                JsonSerializer.Serialize(stream, settings, Json);
            File.Move(tmp, Path, overwrite: true);
        }
    }
}
