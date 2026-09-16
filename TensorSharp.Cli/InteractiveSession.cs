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
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Cli.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Cli
{
    /// <summary>
    /// Turn-by-turn REPL for chatting with a loaded model from the command line.
    ///
    /// The session shares the same KV cache reuse path used by
    /// <c>RunMultiTurnTest</c> so successive turns reuse the prefix from the
    /// previous turn, but adds:
    /// <list type="bullet">
    ///   <item>Live token-by-token printing of the model's reply.</item>
    ///   <item>Slash-prefixed commands (e.g. <c>/help</c>, <c>/reset</c>,
    ///         <c>/temp 0.7</c>) for managing the conversation and sampling
    ///         parameters mid-session.</item>
    ///   <item>Per-turn cancellation via Ctrl+C (the first press stops
    ///         generation and returns to the prompt; the second press at the
    ///         prompt exits).</item>
    /// </list>
    /// The CLI process intentionally uses a tiny self-contained design (no DI
    /// container) - an interactive loop with explicit dependencies is the
    /// simplest thing that keeps the command surface easy to script and test.
    /// </summary>
    internal sealed class InteractiveSession
    {
        internal delegate bool DraftHeadAttacher(ModelBase model, out string error);

        private readonly ILogger _log;
        private readonly IPromptRenderer _promptRenderer;

        private readonly List<ChatMessage> _history = new List<ChatMessage>();
        private readonly KVCache _kvCache = new KVCache();
        private readonly KVCachePromptRenderer _renderer;

        // Mutable so /model, /backend, /mmproj can swap the loaded model
        // without recreating the entire session object. Always paired with
        // _modelPath / _mmProjPath / _backend so /info can describe what is
        // currently loaded and so /backend can re-resolve the same .gguf
        // against a different compute backend.
        private ModelBase _model;
        // Pinned reference to the model that the caller passed in. Its
        // lifetime belongs to the caller (typically a `using var model = ...`
        // in Program.cs), so we never dispose it. Any model created by
        // /model or /backend, however, IS owned by the session and must be
        // disposed when Run() returns.
        private readonly ModelBase _originalModel;
        private string _modelPath;
        private string _mmProjPath;
        private BackendType _backend;

        private SamplingConfig _samplingConfig;
        // Not readonly: /skill rebuilds it, because turning a skill on adds the
        // built-in skills_* declarations to whatever --tools supplied.
        private List<ToolFunction> _tools;
        private readonly List<ToolFunction> _clientTools;
        private string _systemPrompt;

        // Agent Skills. _skillRegistry is the whole roster; _activeSkills is what this
        // conversation selected. _skillSystemBlock is the rendered instruction text,
        // held separately from _systemPrompt so /system and /skill do not overwrite
        // each other, and re-rendered whenever the selection changes.
        private readonly SkillRegistry _skillRegistry;
        private readonly SkillHostOptions _skillOptions;

        /// <summary>Answers the code-execution tools, or null when --code-exec is off.</summary>
        private readonly ICodeRunner _codeRunner;

        /// <summary>
        /// This session's persistent workspace: one working directory and one package
        /// environment shared by the shell tool and the skills' own scripts for the whole
        /// session. Null when code execution is off.
        /// </summary>
        private readonly SessionWorkspace _codeWorkspace;
        private readonly List<Skill> _activeSkills = new();
        private string _skillSystemBlock;
        private SkillToolContext _skillToolContext;

        /// <summary>
        /// The skill ids a call could reach, so a skill name used as a tool name is
        /// answered with how to reach it rather than "there is no such tool".
        /// </summary>
        private static IReadOnlyList<string> ReachableSkillIds(SkillToolContext? context)
        {
            var ids = new List<string>();
            foreach (Skill skill in context?.Reachable ?? (IReadOnlyList<Skill>)Array.Empty<Skill>())
                ids.Add(skill.Id);
            return ids;
        }
        private bool _enableThinking;
        private int _maxTokens;
        private bool _multilineInput;

        // Speculative decoding: a block drafter (DeepSeek V4 + DSpark) or a
        // per-token NextN/MTP head under --spec (GLM-5.2, Qwen 3.6). The
        // decoder is built on first use and kept for the session: it carries the
        // hidden state that pairs the trunk with the drafter, so a turn that
        // extends the cached prefix continues where the previous one stopped.
        // Rebuilt whenever /model or /backend swaps the loaded model.
        private SpeculativeDecoder _specDecoder;
        private ModelBase _specDecoderModel;
        private readonly SpeculativeDecodingOptions.Settings _specSettings;
        // One warning per session when speculation was asked for and refused.
        private bool _specDeclineLogged;

        // Pending attachments to inject into the next user turn. Keeping them as
        // mutable state lets the user run multiple slash commands (e.g. /image,
        // /audio, /video, /text) before submitting the actual question.
        private readonly List<string> _pendingImages = new List<string>();
        private readonly List<string> _pendingAudios = new List<string>();
        // Text attachments are inlined into the user message Content (the model
        // sees them as part of the prompt) AND surfaced via ChatMessage.TextFilePaths
        // for the audit log, mirroring the server's text-upload convention.
        private readonly List<(string Path, string Content)> _pendingTextFiles
            = new List<(string Path, string Content)>();
        private bool _pendingIsVideo;

        // Single-shot cancellation token for the in-flight generation. Replaced
        // each turn so a previous Ctrl+C doesn't leak into later turns.
        private CancellationTokenSource _generationCts;
        // Goes high when the user types /exit or /quit (or hits Ctrl+C twice at
        // the prompt). The outer loop checks this flag after each iteration.
        private bool _shouldExit;
        // Tracks whether we are currently streaming a response. Ctrl+C while
        // generating cancels generation; Ctrl+C at the prompt exits.
        private bool _isGenerating;

        // Maximum number of bytes we will inline from a single /text upload. The
        // model will still reject anything that overflows its context window, but
        // a soft cap keeps a misclick on a multi-GB log file from blowing up the
        // process.
        private const int MaxInlinedTextFileBytes = 256 * 1024;

        public InteractiveSession(
            ModelBase model,
            string modelPath,
            BackendType backend,
            string mmProjPath,
            IPromptRenderer promptRenderer,
            SamplingConfig samplingConfig,
            List<ToolFunction> tools,
            bool enableThinking,
            int maxTokens,
            ILogger log,
            int specDraftMax = 0,
            float specDraftConfMin = -1f,
            SkillRegistry skillRegistry = null,
            SkillHostOptions skillOptions = null,
            IReadOnlyList<Skill> initialSkills = null,
            ICodeRunner codeRunner = null,
            SessionWorkspace codeWorkspace = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _originalModel = _model;
            _modelPath = modelPath;
            _backend = backend;
            _mmProjPath = mmProjPath;
            _promptRenderer = promptRenderer ?? throw new ArgumentNullException(nameof(promptRenderer));
            _renderer = new KVCachePromptRenderer(_promptRenderer);
            _samplingConfig = samplingConfig ?? SamplingConfig.Default;
            _tools = tools;
            _clientTools = tools != null ? new List<ToolFunction>(tools) : null;
            _skillRegistry = skillRegistry;
            _skillOptions = skillOptions ?? new SkillHostOptions();
            _codeRunner = codeRunner;
            _codeWorkspace = codeWorkspace;
            if (initialSkills != null)
                _activeSkills.AddRange(initialSkills);
            _enableThinking = enableThinking;
            _maxTokens = maxTokens > 0 ? maxTokens : 512;
            _log = log;
            _specSettings = SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin);
            RebuildSkillContext();
        }

        /// <summary>
        /// Apply a starting system prompt before <see cref="Run"/>. Provided as
        /// a separate method (rather than a constructor arg) because the user
        /// can also change the prompt at any time via the <c>/system</c>
        /// slash command, and we want both code paths to share the same logic.
        /// </summary>
        public void SetInitialSystemPrompt(string prompt)
        {
            _systemPrompt = string.IsNullOrWhiteSpace(prompt) ? null : prompt;
        }

        public void Run()
        {
            // Make sure we own a clean KV state before we start so a previous
            // RunInference call (e.g. when the same Main invocation also did a
            // dump-prompt or test) doesn't poison the cache.
            _model.ResetKVCache();
            _kvCache.Reset();

            ConsoleCancelEventHandler cancelHandler = OnCancelKeyPress;
            Console.CancelKeyPress += cancelHandler;
            try
            {
                PrintBanner();
                WarmSystemPrefix();

                while (!_shouldExit)
                {
                    string input = ReadUserInput();
                    if (input == null)
                    {
                        Console.WriteLine();
                        break;
                    }

                    string trimmed = input.Trim();
                    if (trimmed.Length == 0)
                        continue;

                    if (trimmed.StartsWith("/"))
                    {
                        HandleCommand(trimmed);
                        continue;
                    }

                    RunTurn(input);
                }
            }
            finally
            {
                Console.CancelKeyPress -= cancelHandler;
                // If /model or /backend swapped in a fresh ModelBase, the
                // caller's `using var model` only knows about the original
                // and would leak the replacement. Dispose it here, but never
                // touch the original (caller owns its lifetime).
                if (_model != null && !ReferenceEquals(_model, _originalModel))
                {
                    try { _model.Dispose(); }
                    catch (Exception ex)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration, ex,
                            "Failed to dispose interactive model on exit: {Error}", ex.Message);
                    }
                }
            }
        }

        // ---- Prompt + I/O ----------------------------------------------------

        private string ReadUserInput()
        {
            string prompt = BuildInputPrompt();
            Console.Write(prompt);

            if (!_multilineInput)
                return Console.ReadLine();

            // Multi-line mode: keep accepting lines until the user enters a
            // line that is exactly ".". This mirrors the well-known shell
            // convention for here-doc terminators.
            var sb = new StringBuilder();
            while (true)
            {
                string line = Console.ReadLine();
                if (line == null)
                    return sb.Length == 0 ? null : sb.ToString();
                if (line == ".")
                    return sb.ToString();
                if (sb.Length > 0)
                    sb.Append('\n');
                sb.Append(line);
            }
        }

        private string BuildInputPrompt()
        {
            int turnNumber = _history.Count(m => m.Role == "user") + 1;
            string attachmentSuffix = "";
            int attachCount = _pendingImages.Count + _pendingAudios.Count + _pendingTextFiles.Count;
            if (attachCount > 0)
                attachmentSuffix = $" ({attachCount} attachment{(attachCount == 1 ? "" : "s")} pending)";
            return $"\n[turn {turnNumber}{attachmentSuffix}]> ";
        }

        private void PrintBanner()
        {
            Console.WriteLine();
            Console.WriteLine("=== TensorSharp interactive chat ===");
            Console.WriteLine($"Model: {(_modelPath != null ? Path.GetFileName(_modelPath) : "(unknown)")}");
            Console.WriteLine($"Backend: {_backend}");
            Console.WriteLine($"Architecture: {_model.Config.Architecture ?? "(unknown)"}");
            Console.WriteLine($"Context length: {_model.MaxContextLength} tokens");
            if (!string.IsNullOrEmpty(_mmProjPath))
                Console.WriteLine($"Multimodal projector: {Path.GetFileName(_mmProjPath)}");
            Console.WriteLine($"Max tokens per reply: {_maxTokens}");
            Console.WriteLine($"Thinking: {(_enableThinking ? "on" : "off")}");
            PrintSampling(prefix: "Sampling: ");
            Console.WriteLine("Type /help to see all available commands. Use /exit or Ctrl+D to leave.");
            Console.WriteLine("===============================");
        }

        private void PrintSampling(string prefix = "")
        {
            var c = _samplingConfig;
            string seed = c.Seed >= 0 ? c.Seed.ToString(CultureInfo.InvariantCulture) : "random";
            string stop = (c.StopSequences != null && c.StopSequences.Count > 0)
                ? "[" + string.Join(", ", c.StopSequences.Select(s => $"\"{s}\"")) + "]"
                : "(none)";
            Console.WriteLine($"{prefix}temp={c.Temperature.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"topK={c.TopK} topP={c.TopP.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"minP={c.MinP.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"repPen={c.RepetitionPenalty.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"presPen={c.PresencePenalty.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"freqPen={c.FrequencyPenalty.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"seed={seed} stop={stop}");
        }

        // ---- Slash commands --------------------------------------------------

        private void HandleCommand(string line)
        {
            string[] parts = SplitCommand(line);
            string cmd = parts[0].ToLowerInvariant();
            string arg = parts.Length > 1 ? parts[1].Trim() : "";

            switch (cmd)
            {
                case "/help":
                case "/?":
                    PrintHelp();
                    break;
                case "/exit":
                case "/quit":
                    _shouldExit = true;
                    break;
                case "/reset":
                case "/new":
                    ResetSession();
                    break;
                case "/sampling":
                case "/show":
                    PrintSampling();
                    break;
                case "/system":
                    SetSystemPrompt(arg);
                    break;
                case "/think":
                    SetThinking(arg);
                    break;
                case "/max":
                case "/maxtokens":
                    SetMaxTokens(arg);
                    break;
                case "/temp":
                case "/temperature":
                    UpdateSampling(arg, "temperature", v => _samplingConfig.Temperature = (float)v);
                    break;
                case "/topk":
                case "/top-k":
                case "/top_k":
                    UpdateSampling(arg, "top_k", v => _samplingConfig.TopK = (int)v, isInt: true);
                    break;
                case "/topp":
                case "/top-p":
                case "/top_p":
                    UpdateSampling(arg, "top_p", v => _samplingConfig.TopP = (float)v);
                    break;
                case "/minp":
                case "/min-p":
                case "/min_p":
                    UpdateSampling(arg, "min_p", v => _samplingConfig.MinP = (float)v);
                    break;
                case "/repeat":
                case "/repeat-penalty":
                case "/repetition-penalty":
                    UpdateSampling(arg, "repetition_penalty", v => _samplingConfig.RepetitionPenalty = (float)v);
                    break;
                case "/presence":
                case "/presence-penalty":
                    UpdateSampling(arg, "presence_penalty", v => _samplingConfig.PresencePenalty = (float)v);
                    break;
                case "/frequency":
                case "/frequency-penalty":
                    UpdateSampling(arg, "frequency_penalty", v => _samplingConfig.FrequencyPenalty = (float)v);
                    break;
                case "/seed":
                    SetSeed(arg);
                    break;
                case "/stop":
                    AddStopSequence(arg);
                    break;
                case "/clearstop":
                case "/stop-clear":
                    ClearStopSequences();
                    break;
                case "/image":
                case "/img":
                    AttachImage(arg);
                    break;
                case "/audio":
                    AttachAudio(arg);
                    break;
                case "/video":
                case "/vid":
                    AttachVideo(arg);
                    break;
                case "/text":
                case "/file":
                case "/txt":
                    AttachTextFile(arg);
                    break;
                case "/clearattach":
                case "/clear-attachments":
                    ClearAttachments();
                    break;
                case "/skills":
                    ListSkills();
                    break;
                case "/skill":
                    ToggleSkill(arg);
                    break;
                case "/multiline":
                    ToggleMultiline(arg);
                    break;
                case "/save":
                    SaveTranscript(arg);
                    break;
                case "/history":
                    PrintHistory();
                    break;
                case "/model":
                    LoadDifferentModel(arg);
                    break;
                case "/backend":
                    SwitchBackend(arg);
                    break;
                case "/mmproj":
                case "/projector":
                    LoadMmProj(arg);
                    break;
                case "/info":
                case "/status":
                    PrintInfo();
                    break;
                default:
                    Console.WriteLine($"Unknown command: {cmd}. Type /help for the list.");
                    break;
            }
        }

        private static string[] SplitCommand(string line)
        {
            int sp = line.IndexOf(' ');
            return sp < 0
                ? new[] { line }
                : new[] { line.Substring(0, sp), line.Substring(sp + 1) };
        }

        private void PrintHelp()
        {
            Console.WriteLine();
            Console.WriteLine("Conversation:");
            Console.WriteLine("  /help, /?              Show this message.");
            Console.WriteLine("  /exit, /quit           Leave the session.");
            Console.WriteLine("  /reset, /new           Clear conversation history and KV cache.");
            Console.WriteLine("  /history               Print the current conversation history.");
            Console.WriteLine("  /save <file>           Write the conversation transcript to a file.");
            Console.WriteLine("  /system <text>         Set (or clear when empty) the system prompt.");
            Console.WriteLine("  /think on|off          Toggle thinking/reasoning mode for supported models.");
            Console.WriteLine("  /multiline on|off      Toggle multi-line input (terminate with a single '.').");
            Console.WriteLine();
            Console.WriteLine("Model and runtime:");
            Console.WriteLine("  /info, /status         Show the loaded model, backend, and projector.");
            Console.WriteLine("  /model <path>          Load a different .gguf model (resets the session).");
            Console.WriteLine("  /backend <name>        Reload the current model on a different backend");
            Console.WriteLine("                         (cpu | cuda | ggml_cpu | ggml_metal | ggml_cuda).");
            Console.WriteLine("  /mmproj <path>         Load a multimodal projector for the current model");
            Console.WriteLine("                         (pass an empty value to clear).");
            Console.WriteLine();
            Console.WriteLine("Sampling:");
            Console.WriteLine("  /sampling, /show       Print the current sampling configuration.");
            Console.WriteLine("  /max <N>               Set maximum reply length in tokens.");
            Console.WriteLine("  /temp <float>          Set temperature.");
            Console.WriteLine("  /topk <int>            Set top_k (0 disables).");
            Console.WriteLine("  /topp <float>          Set top_p (1.0 disables).");
            Console.WriteLine("  /minp <float>          Set min_p (0 disables).");
            Console.WriteLine("  /repeat <float>        Set repetition_penalty (1 disables).");
            Console.WriteLine("  /presence <float>      Set presence_penalty (0 disables).");
            Console.WriteLine("  /frequency <float>     Set frequency_penalty (0 disables).");
            Console.WriteLine("  /seed <int>            Set sampling seed (-1 = non-deterministic).");
            Console.WriteLine("  /stop <text>           Add a stop sequence.");
            Console.WriteLine("  /clearstop             Remove all stop sequences.");
            Console.WriteLine();
            Console.WriteLine("Uploads (queued for the next user turn):");
            Console.WriteLine("  /image <path>          Attach an image (vision-capable models only).");
            Console.WriteLine("  /audio <path>          Attach an audio file (audio-capable models only).");
            Console.WriteLine("  /video <path>          Attach a video; frames are extracted automatically.");
            Console.WriteLine("  /text <path>           Inline a text/markdown/csv file into the next prompt.");
            Console.WriteLine("                         (alias /file)");
            Console.WriteLine("  /clearattach           Drop any pending image/audio/video/text attachments.");
            Console.WriteLine();
            Console.WriteLine("Agent skills:");
            Console.WriteLine("  /skills                List every installed skill and mark the active ones.");
            Console.WriteLine("  /skill <name>          Turn a skill on or off for this conversation.");
            Console.WriteLine("                         RESETS the conversation: the skill block is part of the");
            Console.WriteLine("                         leading system message, so the KV cache no longer matches.");
            Console.WriteLine();
            Console.WriteLine("Plain text without a leading slash is sent to the model as the next user turn.");
            Console.WriteLine("Press Ctrl+C while generating to interrupt; press Ctrl+C at the prompt to exit.");
        }

        private void ResetSession()
        {
            _history.Clear();
            ClearAttachments();
            // /reset IS starting a new chat, and a new chat is exactly the case the warm
            // prefix exists for. Clearing to nothing would make the next message pay for
            // the whole shared prompt again -- the cold turn 1 this feature removes,
            // reintroduced by the command a user reaches for most.
            RestoreWarmPrefixOrClear();
            Console.WriteLine(_warmPrefixTokens > 0
                ? $"Conversation history cleared. The shared prompt ({_warmPrefixTokens} tokens) stays in the cache."
                : "Conversation history and KV cache cleared.");
        }

        /// <summary>
        /// Put the cache back to the load-time warm prefix, so the next turn is a
        /// continuation rather than a cold prefill.
        /// </summary>
        /// <remarks>
        /// Two routes, because the models split on one capability. Where the K/V can be
        /// truncated this is exact and instant: cut both the model and the mirror back to
        /// the boundary. Where it cannot -- Qwen 3.5 and 3.8, whose recurrent state has no
        /// meaningful value at an earlier position -- there is nothing to cut to, so the
        /// prefix is forwarded again. That still wins, because it happens on the /reset
        /// command rather than while the user waits for their first token, but it is a
        /// re-computation and the message says so.
        /// </remarks>
        private void RestoreWarmPrefixOrClear()
        {
            if (_warmPrefixTokens <= 0)
            {
                _kvCache.Reset();
                _model.ResetKVCache();
                return;
            }

            // TryTruncate, not Truncate: a model whose rewind depth depends on where the
            // conversation got to can decline this one, and then re-forwarding the prefix
            // is the same fallback as for a model that cannot truncate at all.
            if (_model.SupportsKVCacheTruncation && _kvCache.Count >= _warmPrefixTokens
                && _model.TryTruncateKVCache(_warmPrefixTokens))
            {
                _kvCache.TruncateTo(_warmPrefixTokens);
                return;
            }

            _kvCache.Reset();
            _model.ResetKVCache();
            int previous = _warmPrefixTokens;
            _warmPrefixTokens = 0;
            _warmPrefixReported = false;
            Console.WriteLine($"[re-forwarding the shared prompt ({previous} tokens); "
                + "this model cannot rewind its cache that far]");
            WarmSystemPrefix();
        }

        private void SetSystemPrompt(string text)
        {
            _systemPrompt = string.IsNullOrWhiteSpace(text) ? null : text;
            // Switching the system prompt invalidates every cached prefix, so
            // reset both the model state and the tracked turns to keep
            // generation correct.
            _history.Clear();
            _kvCache.Reset();
            _model.ResetKVCache();
            // The warmed tokens were rendered FROM the old system prompt, so they cannot
            // be a prefix of anything rendered from the new one. Re-warm rather than
            // leaving the next turn cold.
            _warmPrefixTokens = 0;
            _warmPrefixReported = false;
            Console.WriteLine(_systemPrompt == null
                ? "System prompt cleared. Conversation reset."
                : $"System prompt set ({_systemPrompt.Length} chars). Conversation reset.");
            WarmSystemPrefix();
        }

        private void SetThinking(string arg)
        {
            if (string.IsNullOrEmpty(arg))
            {
                _enableThinking = !_enableThinking;
            }
            else if (TryParseBool(arg, out bool value))
            {
                _enableThinking = value;
            }
            else
            {
                Console.WriteLine($"Could not parse '{arg}' as boolean. Use 'on' or 'off'.");
                return;
            }
            Console.WriteLine($"Thinking is now {(_enableThinking ? "on" : "off")}.");
        }

        private void SetMaxTokens(string arg)
        {
            if (!int.TryParse(arg, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
            {
                Console.WriteLine($"Could not parse '{arg}' as a positive integer.");
                return;
            }
            _maxTokens = parsed;
            Console.WriteLine($"Max tokens per reply set to {_maxTokens}.");
        }

        private void UpdateSampling(string arg, string label, Action<double> setter, bool isInt = false)
        {
            if (string.IsNullOrEmpty(arg))
            {
                Console.WriteLine($"Usage: /{label.Replace('_', '-')} <{(isInt ? "int" : "float")}>");
                return;
            }
            if (isInt)
            {
                if (!int.TryParse(arg, NumberStyles.Integer, CultureInfo.InvariantCulture, out int intValue))
                {
                    Console.WriteLine($"Could not parse '{arg}' as int for {label}.");
                    return;
                }
                setter(intValue);
            }
            else
            {
                if (!double.TryParse(arg, NumberStyles.Float, CultureInfo.InvariantCulture, out double floatValue))
                {
                    Console.WriteLine($"Could not parse '{arg}' as float for {label}.");
                    return;
                }
                setter(floatValue);
            }
            Console.WriteLine($"{label} updated.");
            PrintSampling();
        }

        private void SetSeed(string arg)
        {
            if (!int.TryParse(arg, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
            {
                Console.WriteLine($"Could not parse '{arg}' as int for seed.");
                return;
            }
            _samplingConfig.Seed = parsed;
            Console.WriteLine($"Seed set to {parsed} ({(parsed >= 0 ? "deterministic" : "random")}).");
        }

        private void AddStopSequence(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                Console.WriteLine("Usage: /stop <text>. Use /clearstop to remove all.");
                return;
            }
            _samplingConfig.StopSequences ??= new List<string>();
            _samplingConfig.StopSequences.Add(text);
            Console.WriteLine($"Added stop sequence \"{text}\". Now {_samplingConfig.StopSequences.Count} configured.");
        }

        private void ClearStopSequences()
        {
            _samplingConfig.StopSequences = null;
            Console.WriteLine("Cleared all stop sequences.");
        }

        private void AttachImage(string path)
        {
            path = StripQuotes(path);
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Console.WriteLine($"Image file not found: {path}");
                return;
            }
            // Mixing still images with video frames in the same turn would
            // produce a confusing multimodal token stream (the chat templates
            // emit a single <|video> tag in front of all image placeholders),
            // so reject the combination outright.
            if (_pendingIsVideo)
            {
                Console.WriteLine("This turn already has a queued video. Use /clearattach before adding still images.");
                return;
            }
            _pendingImages.Add(path);
            Console.WriteLine($"Image attached: {path}. {_pendingImages.Count} image(s) queued for next turn.");
        }

        private void AttachAudio(string path)
        {
            path = StripQuotes(path);
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Console.WriteLine($"Audio file not found: {path}");
                return;
            }
            _pendingAudios.Add(path);
            Console.WriteLine($"Audio attached: {path}. {_pendingAudios.Count} audio file(s) queued for next turn.");
        }

        private void AttachVideo(string path)
        {
            path = StripQuotes(path);
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Console.WriteLine($"Video file not found: {path}");
                return;
            }
            // Each video occupies the entire image slot for this turn (the chat
            // template emits the <|video> marker once per user message); refuse
            // to add a second one rather than silently dropping frames.
            if (_pendingIsVideo)
            {
                Console.WriteLine("Another video is already queued for this turn. Use /clearattach to start over.");
                return;
            }
            if (_pendingImages.Count > 0)
            {
                Console.WriteLine("Cannot attach a video when still images are already queued. Use /clearattach first.");
                return;
            }

            List<string> frames;
            try
            {
                frames = MediaHelper.ExtractVideoFrames(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to extract frames from video: {ex.Message}");
                return;
            }

            if (frames == null || frames.Count == 0)
            {
                Console.WriteLine($"No frames could be extracted from video: {path}");
                return;
            }

            _pendingImages.AddRange(frames);
            _pendingIsVideo = true;
            Console.WriteLine($"Video attached: {path}. Extracted {frames.Count} frame(s) for next turn.");
        }

        private void AttachTextFile(string path)
        {
            path = StripQuotes(path);
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Console.WriteLine($"Text file not found: {path}");
                return;
            }
            try
            {
                string content;
                long size = new FileInfo(path).Length;
                if (size > MaxInlinedTextFileBytes)
                {
                    // Soft cap: read the prefix, but always tell the user we
                    // truncated so they're not surprised when the model only
                    // answers based on the head of the file.
                    using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
                    var buffer = new byte[MaxInlinedTextFileBytes];
                    int read = stream.Read(buffer, 0, buffer.Length);
                    content = Encoding.UTF8.GetString(buffer, 0, read);
                    Console.WriteLine($"Text file is {size} bytes; only the first {MaxInlinedTextFileBytes} bytes will be inlined.");
                }
                else
                {
                    content = File.ReadAllText(path);
                }
                _pendingTextFiles.Add((path, content));
                Console.WriteLine($"Text file attached: {path} ({content.Length} chars). " +
                    $"{_pendingTextFiles.Count} text file(s) queued for next turn.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to read text file: {ex.Message}");
            }
        }

        private void ClearAttachments()
        {
            int total = _pendingImages.Count + _pendingAudios.Count + _pendingTextFiles.Count;
            _pendingImages.Clear();
            _pendingAudios.Clear();
            _pendingTextFiles.Clear();
            _pendingIsVideo = false;
            Console.WriteLine(total > 0
                ? $"Dropped {total} pending attachment(s)."
                : "No pending attachments.");
        }

        private void PrintInfo()
        {
            Console.WriteLine();
            Console.WriteLine("--- Loaded model ---");
            Console.WriteLine($"  Path:         {_modelPath ?? "(unknown)"}");
            Console.WriteLine($"  Backend:      {_backend}");
            Console.WriteLine($"  Architecture: {_model.Config.Architecture ?? "(unknown)"}");
            Console.WriteLine($"  Context:      {_model.MaxContextLength} tokens (current KV: {_model.CacheSeqLen})");
            Console.WriteLine($"  Vocab size:   {_model.Config.VocabSize}");
            Console.WriteLine($"  Projector:    {_mmProjPath ?? "(none)"}");
            Console.WriteLine($"  Vision enc:   {(_model.HasVisionEncoder() ? "loaded" : "(none)")}");
            int turns = _history.Count(m => m.Role == "user");
            Console.WriteLine($"  Conversation: {turns} user turn(s), KV cache holds {_kvCache.Count} token(s).");
            int pendingImg = _pendingImages.Count;
            int pendingAud = _pendingAudios.Count;
            int pendingTxt = _pendingTextFiles.Count;
            if (pendingImg + pendingAud + pendingTxt > 0)
                Console.WriteLine($"  Pending:      {pendingImg} image(s){(_pendingIsVideo ? " (video frames)" : "")}, " +
                    $"{pendingAud} audio, {pendingTxt} text file(s).");
        }

        private void LoadDifferentModel(string arg)
        {
            string path = StripQuotes(arg);
            if (string.IsNullOrEmpty(path))
            {
                Console.WriteLine("Usage: /model <path-to.gguf>");
                return;
            }
            if (!File.Exists(path))
            {
                Console.WriteLine($"Model file not found: {path}");
                return;
            }
            // /model is the most invasive command in the session: it replaces
            // the underlying ModelBase, so the chat template, tokenizer, vocab,
            // and KV layout all change. Drop the projector + history + KV
            // cache so we don't try to splice old tokens through a brand new
            // tokenizer.
            ReloadModel(path, _backend, mmProjPath: null, label: "model");
        }

        private void SwitchBackend(string arg)
        {
            string requested = (arg ?? "").Trim().ToLowerInvariant();
            if (string.IsNullOrEmpty(requested))
            {
                Console.WriteLine($"Current backend: {_backend}. Usage: /backend cpu|cuda|ggml_cpu|ggml_metal|ggml_cuda");
                return;
            }
            if (!TryParseBackend(requested, out BackendType target))
            {
                Console.WriteLine($"Unknown backend '{requested}'. Use: cpu, cuda, ggml_cpu, ggml_metal, ggml_cuda");
                return;
            }
            if (target == _backend)
            {
                Console.WriteLine($"Already on backend {_backend}; nothing to do.");
                return;
            }
            if (string.IsNullOrEmpty(_modelPath) || !File.Exists(_modelPath))
            {
                Console.WriteLine($"Cannot switch backend: original model path is unknown or missing ({_modelPath ?? "(none)"}).");
                return;
            }
            // Backend swaps require a full reload - we keep the .gguf and
            // projector paths so the user doesn't have to repeat them.
            ReloadModel(_modelPath, target, _mmProjPath, label: "backend");
        }

        private void LoadMmProj(string arg)
        {
            string path = StripQuotes(arg);
            if (string.IsNullOrEmpty(path))
            {
                // Empty argument means "drop the projector". The current ModelBase
                // can't actually unload the encoders mid-session (the only way is
                // to reload the whole model), so warn the user instead of silently
                // pretending we cleared it.
                if (_mmProjPath != null)
                {
                    Console.WriteLine($"To unload the current projector ({Path.GetFileName(_mmProjPath)}), reload the model with /model {_modelPath}.");
                }
                else
                {
                    Console.WriteLine("Usage: /mmproj <path-to-mmproj.gguf>");
                }
                return;
            }
            if (!File.Exists(path))
            {
                Console.WriteLine($"Projector file not found: {path}");
                return;
            }
            try
            {
                _model.MultimodalInjector.LoadProjectors(path);
                _mmProjPath = path;
                Console.WriteLine($"Loaded multimodal projector: {Path.GetFileName(path)}");
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "interactive loaded multimodal projector {MmProj}", path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load projector: {ex.Message}");
                _log.LogError(LogEventIds.HostConfiguration, ex,
                    "Failed to load projector {MmProj}", path);
            }
        }

        private void ReloadModel(string modelPath, BackendType backend, string mmProjPath, string label)
        {
            string prevModel = _modelPath != null ? Path.GetFileName(_modelPath) : "(none)";
            ModelBase newModel = null;
            try
            {
                Console.WriteLine($"Loading {Path.GetFileName(modelPath)} on {backend}...");
                var sw = Stopwatch.StartNew();
                var loaded = CreateModelForReload(modelPath, backend);
                newModel = loaded.Model;
                string draftHeadError = loaded.DraftHeadError;
                if (draftHeadError != null)
                {
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "{Error} Speculative decoding will serve standard decoding instead.",
                        draftHeadError);
                }
                sw.Stop();

                string loadedMmProjPath = null;
                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    try
                    {
                        newModel.MultimodalInjector.LoadProjectors(mmProjPath);
                        loadedMmProjPath = mmProjPath;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Loaded model but failed to load projector: {ex.Message}");
                    }
                }

                // Finish initializing the replacement before committing the
                // handoff. If any of this fails, the catch below disposes the
                // replacement and the working model remains untouched.
                newModel.ResetKVCache();

                ModelBase previousModel = _model;
                _model = newModel;
                newModel = null; // ownership now belongs to the session
                _modelPath = modelPath;
                _backend = backend;
                _mmProjPath = loadedMmProjPath;

                // History / KV and speculative state from the previous tokenizer
                // are meaningless against the new one, so drop everything.
                _history.Clear();
                _kvCache.Reset();
                _specDecoder = null;
                _specDecoderModel = null;
                ClearAttachments();

                // Dispose only after the replacement has become the active model:
                // a backend's cleanup error must not strand the session between
                // models. The caller owns the original model's lifetime.
                if (previousModel != null && !ReferenceEquals(previousModel, _originalModel))
                {
                    try { previousModel.Dispose(); }
                    catch (Exception ex)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration, ex,
                            "Interactive model switch succeeded, but disposing the previous model failed: {Error}",
                            ex.Message);
                    }
                }

                Console.WriteLine($"{char.ToUpper(label[0])}{label.Substring(1)} switch complete: " +
                    $"{Path.GetFileName(modelPath)} ({_model.Config.Architecture ?? "?"}, " +
                    $"context={_model.MaxContextLength}) loaded in {sw.Elapsed.TotalMilliseconds:F0} ms.");
                Console.WriteLine($"Conversation history cleared (previous model: {prevModel}).");

                _log.LogInformation(LogEventIds.ModelLoadCompleted,
                    "interactive reloaded model={Model} backend={Backend} mmproj={MmProj} architecture={Architecture} elapsedMs={ElapsedMs:F1}",
                    Path.GetFileName(modelPath), backend, _mmProjPath ?? "(none)",
                    _model.Config.Architecture ?? "(unknown)", sw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                if (newModel != null)
                {
                    try { newModel.Dispose(); }
                    catch (Exception disposeEx)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration, disposeEx,
                            "Failed to dispose an incomplete interactive model reload: {Error}",
                            disposeEx.Message);
                    }
                }
                Console.WriteLine($"Failed to load model: {ex.Message}");
                _log.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Failed to reload model {Model} on backend {Backend}: {Error}",
                    Path.GetFileName(modelPath), backend, ex.Message);
            }
        }

        /// <summary>
        /// Construct a replacement model with the process-wide draft-model
        /// configuration and attach any draft weights that load after the trunk.
        /// The delegates are a unit-test seam; production uses the same factory and
        /// shared attachment loader as initial CLI startup.
        /// </summary>
        internal static (ModelBase Model, string DraftHeadError) CreateModelForReload(
            string modelPath,
            BackendType backend,
            Func<string, BackendType, string, ModelBase> createModel = null,
            DraftHeadAttacher attachDraftHead = null)
        {
            createModel ??= static (path, selectedBackend, draftPath) =>
                ModelBase.Create(path, selectedBackend, draftModelPath: draftPath);
            attachDraftHead ??= SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead;

            ModelBase model = null;
            try
            {
                string draftModelPath = Program.ResolveConfiguredDraftModelPath();
                model = createModel(modelPath, backend, draftModelPath);
                bool attached = attachDraftHead(model, out string error);
                return (model, attached ? null : error);
            }
            catch
            {
                model?.Dispose();
                throw;
            }
        }

        private static bool TryParseBackend(string raw, out BackendType backend)
        {
            switch ((raw ?? string.Empty).Trim().ToLowerInvariant())
            {
                case "cpu":
                    backend = BackendType.Cpu;
                    return true;
                case "cuda":
                case "direct_cuda":
                case "direct-cuda":
                    backend = BackendType.Cuda;
                    return true;
                case "ggml_cpu":
                case "ggml-cpu":
                    backend = BackendType.GgmlCpu;
                    return true;
                case "metal":
                case "ggml_metal":
                case "ggml-metal":
                    backend = BackendType.GgmlMetal;
                    return true;
                case "ggml_cuda":
                case "ggml-cuda":
                    backend = BackendType.GgmlCuda;
                    return true;
                case "ggml_vulkan":
                case "ggml-vulkan":
                    backend = BackendType.GgmlVulkan;
                    return true;
                default:
                    backend = BackendType.Cpu;
                    return false;
            }
        }

        // Allows users to paste shell-quoted paths (which is what most file
        // managers emit on drag-and-drop) without having to strip the quotes
        // themselves.
        private static string StripQuotes(string s)
        {
            if (string.IsNullOrEmpty(s))
                return s;
            s = s.Trim();
            if (s.Length >= 2 &&
                ((s[0] == '"' && s[s.Length - 1] == '"') ||
                 (s[0] == '\'' && s[s.Length - 1] == '\'')))
            {
                return s.Substring(1, s.Length - 2);
            }
            return s;
        }

        private void ToggleMultiline(string arg)
        {
            if (string.IsNullOrEmpty(arg))
            {
                _multilineInput = !_multilineInput;
            }
            else if (TryParseBool(arg, out bool value))
            {
                _multilineInput = value;
            }
            else
            {
                Console.WriteLine($"Could not parse '{arg}' as boolean. Use 'on' or 'off'.");
                return;
            }
            Console.WriteLine(_multilineInput
                ? "Multi-line input enabled. End each turn with a single '.' on its own line."
                : "Multi-line input disabled.");
        }

        private void SaveTranscript(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                Console.WriteLine("Usage: /save <file-path>");
                return;
            }
            try
            {
                using var writer = new StreamWriter(path, false, Encoding.UTF8);
                if (!string.IsNullOrEmpty(_systemPrompt))
                {
                    writer.WriteLine("=== System ===");
                    writer.WriteLine(_systemPrompt);
                }
                foreach (var msg in _history)
                {
                    writer.WriteLine($"=== {msg.Role} ===");
                    if (!string.IsNullOrEmpty(msg.Thinking))
                    {
                        writer.WriteLine("[thinking]");
                        writer.WriteLine(msg.Thinking);
                        writer.WriteLine("[/thinking]");
                    }
                    writer.WriteLine(msg.Content ?? string.Empty);
                }
                Console.WriteLine($"Transcript saved to {path} ({_history.Count} message(s)).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save transcript: {ex.Message}");
            }
        }

        private void PrintHistory()
        {
            if (_history.Count == 0 && string.IsNullOrEmpty(_systemPrompt))
            {
                Console.WriteLine("(no conversation yet)");
                return;
            }
            Console.WriteLine();
            if (!string.IsNullOrEmpty(_systemPrompt))
            {
                Console.WriteLine("--- system ---");
                Console.WriteLine(_systemPrompt);
            }
            foreach (var msg in _history)
            {
                Console.WriteLine($"--- {msg.Role} ---");
                if (!string.IsNullOrEmpty(msg.Thinking))
                {
                    Console.WriteLine("[thinking]");
                    Console.WriteLine(msg.Thinking);
                    Console.WriteLine("[/thinking]");
                }
                Console.WriteLine(msg.Content ?? string.Empty);
            }
        }

        // ---- Inference -------------------------------------------------------

        private void RunTurn(string userText)
        {
            var renderHistory = BuildRenderHistory(userText);

            try
            {
                _generationCts = new CancellationTokenSource();
                _isGenerating = true;

                // One pass normally; more when the model asks to read skill content
                // first. Each extra pass is a full generation, so the round budget is
                // what stops a model that keeps mis-naming a file from looping forever,
                // and the cancellation token is re-checked between rounds so Ctrl+C
                // stops the whole turn rather than just the round in flight.
                int maxRounds = _skillToolContext != null
                    ? Math.Max(1, _skillOptions.RoundsFor(_skillToolContext.CodeRunner is { CanRun: true }))
                    : 1;
                for (int round = 1; round <= maxRounds; round++)
                {
                    List<ToolCall> toolCalls = Stream(renderHistory, _generationCts.Token);

                    // Three ways. The caller's own tools are shown, as the --tools
                    // contract has always promised; a name NOBODY declared is answered
                    // in the conversation instead, so the model can correct itself
                    // rather than have its whole turn end on a guess.
                    SkillTools.Partition(
                        toolCalls, _clientTools,
                        out var skillCalls, out var clientCalls, out var unknownCalls);

                    if (_skillToolContext == null || (skillCalls.Count == 0 && unknownCalls.Count == 0))
                    {
                        foreach (var call in clientCalls.Concat(unknownCalls))
                            Console.WriteLine($"[tool call] {call}");
                        break;
                    }

                    foreach (var call in unknownCalls)
                    {
                        Console.WriteLine($"[tool call] {call.Name} (no such tool)");
                        _history.Add(BuildSkillResultMessage(
                            SkillTools.DescribeUnknownTool(
                                call.Name, _tools, ReachableSkillIds(_skillToolContext)),
                            call.Name));
                    }

                    _generationCts.Token.ThrowIfCancellationRequested();

                    foreach (var call in skillCalls)
                    {
                        var result = SkillTools.Execute(call, _skillToolContext);
                        Console.WriteLine(
                            $"[skill] {call.Name} {result.SkillId ?? "?"} {result.ResourcePath ?? string.Empty}"
                            + (result.Ok ? string.Empty : " (failed)"));
                        _log.LogInformation(LogEventIds.SkillToolInvoked,
                            "interactive.skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                            round, call.Name, result.SkillId ?? "-", result.ResourcePath ?? "-",
                            result.Ok, result.Content?.Length ?? 0);
                        _history.Add(BuildSkillResultMessage(result.Content, call.Name));
                    }

                    if (round == maxRounds)
                    {
                        _log.LogWarning(LogEventIds.SkillLoopCapped,
                            "interactive.skills.loop.capped rounds={Rounds}", maxRounds);

                        // Told IN THE CONVERSATION and then given one last generation, the
                        // way the one-shot path has always done it (Program.cs, the
                        // "limit on skill lookups" message). Printing a bracketed note to
                        // the terminal and breaking left the user with the tool trace and
                        // no answer at all — the model was mid-work, was never asked to
                        // wrap up, and never got a turn in which it could.
                        Console.WriteLine("[skill lookup limit reached for this turn — answering now]");
                        _history.Add(BuildSkillResultMessage(
                            "Error: the limit on tool calls for this turn has been reached. Answer now "
                            + "using what you have already read, and say which part you could not check.",
                            null));
                        Stream(BuildRenderHistoryForContinuation(), _generationCts.Token);
                        break;
                    }

                    renderHistory = BuildRenderHistoryForContinuation();
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine();
                Console.WriteLine("[generation cancelled]");
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine($"[error] {ex.Message}");
                _log.LogError(LogEventIds.ChatFailed, ex,
                    "Interactive turn failed: {Error}", ex.Message);
            }
            finally
            {
                _isGenerating = false;
                _generationCts?.Dispose();
                _generationCts = null;
            }
        }

        /// <summary>Print the roster and mark which skills this conversation is using.</summary>
        private void ListSkills()
        {
            if (_skillRegistry == null || !_skillOptions.Enabled)
            {
                Console.WriteLine("Agent skills are disabled for this session (--no-skills / TS_NO_SKILLS).");
                return;
            }
            if (_skillRegistry.Skills.Count == 0)
            {
                Console.WriteLine($"No skills found under: {string.Join(", ", _skillRegistry.Roots)}");
                Console.WriteLine("A skill is a directory containing SKILL.md. Add one with --skills-dir <path>.");
                return;
            }

            Console.WriteLine($"Skills ({_activeSkills.Count} active of {_skillRegistry.Skills.Count}):");
            foreach (var skill in _skillRegistry.Skills)
            {
                bool active = _activeSkills.Any(a => string.Equals(a.Id, skill.Id, StringComparison.OrdinalIgnoreCase));
                string mark = active ? "[on] " : "[  ] ";
                string summary = skill.Description.Length > 96
                    ? skill.Description.Substring(0, 93).TrimEnd() + "..."
                    : skill.Description;
                Console.WriteLine($"  {mark}{skill.Id,-24} {summary}");
            }
            Console.WriteLine();
            Console.WriteLine("Turn one on or off with /skill <name>.");
        }

        /// <summary>
        /// Toggle a skill for this conversation.
        ///
        /// <para>
        /// Changing the selection RESETS the conversation, for the same reason
        /// <c>/system</c> does: the skills block is rendered into the leading system
        /// message, so every cached KV block from the first token onward is invalidated.
        /// Continuing on a stale cache would answer from a prompt the model was never
        /// actually shown, so the reset is stated plainly rather than done quietly.
        /// </para>
        /// </summary>
        private void ToggleSkill(string arg)
        {
            if (_skillRegistry == null || !_skillOptions.Enabled)
            {
                Console.WriteLine("Agent skills are disabled for this session (--no-skills / TS_NO_SKILLS).");
                return;
            }
            if (string.IsNullOrWhiteSpace(arg))
            {
                Console.WriteLine("Usage: /skill <name>    (run /skills to see the names)");
                return;
            }

            string name = arg.Trim();
            int existing = _activeSkills.FindIndex(a => string.Equals(a.Id, name, StringComparison.OrdinalIgnoreCase));
            if (existing >= 0)
            {
                string removed = _activeSkills[existing].Id;
                _activeSkills.RemoveAt(existing);
                RebuildSkillContext();
                ResetConversationForSkillChange();
                Console.WriteLine($"[skill '{removed}' off — conversation reset]");
                return;
            }

            if (!_skillRegistry.TryGet(name, out Skill skill))
            {
                Console.WriteLine($"No skill called '{name}'. Run /skills to see what is available.");
                return;
            }

            _activeSkills.Add(skill);
            _activeSkills.Sort((a, b) => string.CompareOrdinal(a.Id, b.Id));
            RebuildSkillContext();
            ResetConversationForSkillChange();
            Console.WriteLine($"[skill '{skill.Id}' on — conversation reset]");
            foreach (string warning in skill.Manifest.Warnings)
                Console.WriteLine($"  warning: {warning}");
        }

        /// <summary>
        /// Re-render the skills block and re-derive the tool list and the sandbox from
        /// the current selection. Called from the constructor and after every
        /// <c>/skill</c>.
        /// </summary>
        private void RebuildSkillContext()
        {
            _skillSystemBlock = null;
            _skillToolContext = null;
            _tools = _clientTools != null ? new List<ToolFunction>(_clientTools) : null;

            BuildSkillPlanContext();

            // Code execution does not require skills: --code-exec with nothing selected
            // (or no skill registry at all) must still offer the shell tool, the way the
            // server's code-only plan does. Without this the flag looked accepted and
            // did nothing in a chat session.
            if (_codeRunner != null && _skillToolContext == null
                && SkillCapabilities.For(_model.Config.Architecture).ToolsRendered)
            {
                _tools = Program.AppendCodeTool(_tools, _codeRunner, _codeWorkspace);
                _skillToolContext = new SkillToolContext(new List<Skill>())
                {
                    CodeRunner = _codeRunner,
                    Workspace = _codeWorkspace,
                };
            }

            // The editing rules, on BOTH paths — with skills and without. They were
            // injected only by the server's plan, so a CLI chat was declared all five code
            // tools and told nothing at all about which to reach for, which is the exact
            // condition the measurement blamed for models re-typing whole files. Appended
            // after any skills block so a skill's own wording is read first.
            if (Program.CodeSystemBlock(_tools) is { Length: > 0 } editing)
            {
                _skillSystemBlock = string.IsNullOrEmpty(_skillSystemBlock)
                    ? editing
                    : _skillSystemBlock.TrimEnd() + "\n\n" + editing;
            }
        }

        /// <summary>The skills half of <see cref="RebuildSkillContext"/>: renders the
        /// instruction block and, where the family can carry tools, builds the tool
        /// context for the current selection. Leaves everything null when skills are
        /// off or the plan is empty.</summary>
        private void BuildSkillPlanContext()
        {
            if (_skillRegistry == null || !_skillOptions.Enabled)
                return;

            var catalog = _skillOptions.Discovery
                ? _skillRegistry.Skills
                : (IReadOnlyList<Skill>)Array.Empty<Skill>();
            if (_activeSkills.Count == 0 && catalog.Count == 0)
                return;

            var capabilities = SkillCapabilities.For(_model.Config.Architecture);
            var plan = SkillPrompt.Plan(_activeSkills, catalog, new SkillPromptOptions
            {
                ContextTokens = _model.MaxContextLength,
                ToolsAvailable = capabilities.ToolsRendered,
            });
            if (plan.IsEmpty)
                return;

            _skillSystemBlock = plan.Instructions;
            if (capabilities.ToolsRendered)
            {
                _tools = SkillTools.Merge(_clientTools, _skillOptions.AllowScripts, out _);
                if (_codeRunner != null)
                    _tools = Program.AppendCodeTool(_tools, _codeRunner, _codeWorkspace);

                _skillToolContext = new SkillToolContext(new List<Skill>(plan.Reachable))
                {
                    ScriptRunner = _skillOptions.AllowScripts
                        ? new SkillScriptRunner(
                            Program.ToScriptRunnerOptions(_skillOptions, _codeWorkspace, _codeRunner), _log)
                        : null,
                    CodeRunner = _codeRunner,
                    Workspace = _codeWorkspace,
                };
            }
        }

        /// <summary>
        /// Drop the conversation and the KV state after a skills change, exactly as
        /// <c>/system</c> does — the leading system block is different, so nothing
        /// cached from before it still describes this conversation.
        /// </summary>
        private void ResetConversationForSkillChange()
        {
            _history.Clear();
            _model.ResetKVCache();
            // The managed mirror has to go with it. ResetSession and SetSystemPrompt both
            // reset the pair; this one reset only the model, so after a /skill toggle the
            // mirror still claimed N cached tokens the model no longer held and the next
            // turn planned its reuse against state that was gone. The two are separate
            // truths kept in step only by convention -- see the invariant on KVCache.
            _kvCache.Reset();
            // Whatever the load-time warm prefix put there went with it, and the skill
            // block it was rendered from has just changed, so it could not have matched
            // the next turn anyway. Stop claiming it.
            _warmPrefixTokens = 0;
            _specDecoder = null;
            _specDecoderModel = null;
        }

        private List<ChatMessage> BuildRenderHistory(string userText)
        {
            // Materialise the current turn's user message (including any pending
            // attachments) and append it to the running tracked history. The
            // assistant turn is added once generation completes so a Ctrl+C
            // mid-turn doesn't leave a dangling user message in the history.
            string composedContent = ComposeUserContent(userText);
            var userMsg = new ChatMessage
            {
                Role = "user",
                Content = composedContent,
                ImagePaths = _pendingImages.Count > 0 ? new List<string>(_pendingImages) : null,
                AudioPaths = _pendingAudios.Count > 0 ? new List<string>(_pendingAudios) : null,
                TextFilePaths = _pendingTextFiles.Count > 0
                    ? _pendingTextFiles.Select(f => f.Path).ToList()
                    : null,
                IsVideo = _pendingIsVideo,
            };
            _history.Add(userMsg);

            var rendered = new List<ChatMessage>();
            // ONE leading system message carrying both, never two: several chat
            // templates recognise a system turn only at index 0, GPT-OSS's Harmony
            // format lifts messages[0] into its developer block and would emit a
            // duplicate system turn for a second one, and Mistral 3 drops a non-first
            // system message outright.
            string leadingSystem = ComposeSystemPrompt();
            if (!string.IsNullOrEmpty(leadingSystem))
                rendered.Add(new ChatMessage { Role = "system", Content = leadingSystem });
            rendered.AddRange(_history);
            return rendered;
        }

        /// <summary>
        /// Re-render the conversation for another round of the same turn: the same
        /// leading system block, plus everything the previous round appended. No new
        /// user message — <see cref="BuildRenderHistory"/> adds one, and calling it again
        /// mid-turn would duplicate the user's question.
        /// </summary>
        private List<ChatMessage> BuildRenderHistoryForContinuation()
        {
            var rendered = new List<ChatMessage>();
            string leadingSystem = ComposeSystemPrompt();
            if (!string.IsNullOrEmpty(leadingSystem))
                rendered.Add(new ChatMessage { Role = "system", Content = leadingSystem });
            rendered.AddRange(_history);
            return rendered;
        }

        /// <summary>
        /// Wrap a skill tool result in the message shape this family renders. Mistral 3
        /// drops <c>role: "tool"</c> messages outright, so there it is fed back as a user
        /// turn rather than vanishing from the prompt.
        /// </summary>
        private ChatMessage BuildSkillResultMessage(string content, string tool)
        {
            if (SkillCapabilities.For(_model.Config.Architecture).ToolResultsRendered)
                return new ChatMessage { Role = "tool", Content = content };

            return new ChatMessage
            {
                Role = "user",
                Content = $"Result of your {tool} call:\n\n{content}",
            };
        }

        /// <summary>The user's <c>--system</c> text and the Agent Skills block, in that order.</summary>
        private string ComposeSystemPrompt()
        {
            if (string.IsNullOrEmpty(_skillSystemBlock))
                return _systemPrompt;
            if (string.IsNullOrEmpty(_systemPrompt))
                return _skillSystemBlock;
            return _systemPrompt.TrimEnd() + "\n\n" + _skillSystemBlock;
        }

        // Inline the contents of every queued /text file into the user prompt
        // body. We use a simple delimited block so the model can reliably tell
        // attached file content apart from the user's actual question.
        private string ComposeUserContent(string userText)
        {
            if (_pendingTextFiles.Count == 0)
                return userText ?? string.Empty;

            var sb = new StringBuilder();
            foreach (var (path, content) in _pendingTextFiles)
            {
                sb.Append("[Attached file: ");
                sb.Append(Path.GetFileName(path));
                sb.Append("]\n");
                sb.Append(content);
                if (!content.EndsWith("\n"))
                    sb.Append('\n');
                sb.Append("[End of file]\n\n");
            }
            sb.Append(userText ?? string.Empty);
            return sb.ToString();
        }

        /// <summary>
        /// Generate and stream one assistant turn.
        /// </summary>
        /// <returns>
        /// The tool calls the model made, or an empty list.
        ///
        /// <para>
        /// This used to return void, and nothing anywhere in interactive chat ever read
        /// <c>ParsedOutput.ToolCalls</c>: a model that emitted a tool call produced no
        /// console output and no history entry at all, because the parser consumed the
        /// span and the session dropped it. Reporting them is what lets
        /// <see cref="RunTurn"/> answer an Agent Skills lookup and continue, and it fixes
        /// the silent drop for ordinary <c>--tools</c> calls at the same time.
        /// </para>
        /// </returns>
        private List<ToolCall> Stream(List<ChatMessage> renderHistory, CancellationToken cancellationToken)
        {
            string arch = _model.Config.Architecture;

            var inputTokens = _renderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                renderHistory,
                arch,
                addGenerationPrompt: true,
                out _,
                out string generationPromptTrailingWhitespace,
                tools: _tools,
                enableThinking: _enableThinking);

            // Expand image/audio placeholder tokens to their final width and
            // pre-compute the embeddings so that QueuePromptEmbeddings (called
            // from inside ApplyReusePlan) can hand them to the model right
            // before each Forward call. Without this, /image, /audio and /video
            // would render the placeholders into the prompt but the model would
            // never actually receive any vision/audio data.
            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens);

            _log.LogDebug(LogEventIds.ChatStarted,
                "interactive prompt tokens={PromptTokens} thinking={Thinking}",
                inputTokens.Count, _enableThinking);

            // A model with a draft head (a block drafter such as DeepSeek V4 +
            // DSpark, or a per-token NextN/MTP head under --spec) decodes
            // through the shared draft/verify core instead of one forward per
            // token. Its prefill has to go through the drafter-aware path too, so
            // the choice is made before the prompt is forwarded.
            SpeculativeDecoder specDecoder = ResolveSpeculativeDecoder(renderHistory);

            var prefillSw = Stopwatch.StartNew();
            ReusePlanKind planKind;
            float[] logits = specDecoder != null
                ? SpeculativePrefill(specDecoder, inputTokens, out planKind)
                : PlainPrefill(inputTokens, out planKind);
            prefillSw.Stop();
            ReportWarmPrefixOutcome(planKind, _kvCache.Count);
            double prefillMs = prefillSw.Elapsed.TotalMilliseconds;
            int promptTokenCount = inputTokens.Count;

            var sampler = new TokenSampler(_samplingConfig);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            // Streaming output parser so we strip <think> blocks from the live
            // console output (they're surfaced separately when --think is on).
            var parser = CliOutputParser.Create(arch, _enableThinking, _tools,
                _model.Tokenizer, inputTokens);
            bool useParser = _enableThinking || (_tools != null && _tools.Count > 0) || parser.AlwaysRequired;
            bool showThinking = _enableThinking || parser.AlwaysRequired;

            Console.WriteLine();
            Console.Write("Assistant: ");

            string finishReason = "max_tokens";
            var decodeSw = Stopwatch.StartNew();
            long firstTokenMs = 0;
            bool firstTokenSeen = false;
            bool inThinkingBlock = false;
            // Rendered counterpart of `renderHistory`: we prefer to
            // splice the assistant turn back in with raw token ids so the next
            // prefill can reuse the cache without re-tokenising.
            string assistantContentBuffer = string.Empty;
            string assistantThinkingBuffer = string.Empty;
            var turnToolCalls = new List<ToolCall>();
            // Per-turn speculative counters (null when the turn decoded plainly).
            SpeculationStats specStats = null;
            int specWindow = 0;

            // Streams one generated token: append its bytes, print the decoded
            // delta through the output parser, and report whether the turn should
            // keep going (a stop sequence or Ctrl+C ends it). Shared by the plain
            // and the speculative loops so both stream identically.
            bool EmitToken(int token)
            {
                generatedTokens.Add(token);
                _model.Tokenizer.AppendTokenBytes(token, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : string.Empty;
                prevCharLen = decoded.Length;

                if (!firstTokenSeen)
                {
                    firstTokenSeen = true;
                    firstTokenMs = (long)decodeSw.Elapsed.TotalMilliseconds;
                }

                if (piece.Length > 0)
                {
                    if (useParser)
                    {
                        var parsed = parser.Add(piece, false);
                        if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                            turnToolCalls.AddRange(parsed.ToolCalls);
                        if (showThinking && !string.IsNullOrEmpty(parsed.Thinking))
                        {
                            if (!inThinkingBlock)
                            {
                                Console.Write("\n[thinking] ");
                                inThinkingBlock = true;
                            }
                            Console.Write(parsed.Thinking);
                            assistantThinkingBuffer += parsed.Thinking;
                        }
                        if (!string.IsNullOrEmpty(parsed.Content))
                        {
                            if (inThinkingBlock)
                            {
                                Console.Write("\n[answer] ");
                                inThinkingBlock = false;
                            }
                            Console.Write(parsed.Content);
                            assistantContentBuffer += parsed.Content;
                        }
                    }
                    else
                    {
                        Console.Write(piece);
                        assistantContentBuffer += piece;
                    }
                }

                if (_samplingConfig.StopSequences != null && _samplingConfig.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        finishReason = "stop_sequence";
                        return false;
                    }
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    finishReason = "cancelled";
                    return false;
                }

                return true;
            }

            if (specDecoder != null)
            {
                // The drafter proposes a window per step and the trunk verifies it
                // in one batched forward. Every emitted token is still drawn from a
                // trunk row — with argmax under a greedy config, with this session's
                // sampler otherwise — so this is purely a speed path.
                int promptCached = _kvCache.Count;
                // Per turn, and the governor with them: a park decided on the last
                // turn describes that turn's context and prompt, not this one's.
                specDecoder.ResetStatsAndGovernor();
                bool specArgmax = IsArgmaxSampling(_samplingConfig);
                bool StopOnEos(int t)
                {
                    if (!_model.Tokenizer.IsEos(t))
                        return false;
                    finishReason = "eos";
                    return true;
                }
                List<int> specTokens = specArgmax
                    ? specDecoder.GenerateGreedyFrom(logits, promptCached, _maxTokens,
                        isStopToken: StopOnEos, onToken: EmitToken)
                    : specDecoder.GenerateSampledFrom(logits, promptCached, _maxTokens, sampler,
                        isStopToken: StopOnEos, onToken: EmitToken);

                // The trunk commits every accepted token plus the corrected one, but
                // never the token it will forward on the next step. Mirror exactly
                // what it holds so the next turn's prefix match stays sound. This
                // reads the decoder's own output, not the streamed tokens: a turn cut
                // short mid-block (Ctrl+C, a stop sequence) leaves the trunk holding
                // tokens the console never saw, and a cache that under-reports them
                // would make the next turn prefill at the wrong position.
                int trunkGenerated = ((ISpeculativeTarget)_model).CacheSeqLen - promptCached;
                int cachedGenerated = Math.Clamp(trunkGenerated, 0, specTokens.Count);
                if (cachedGenerated > 0)
                    _kvCache.RecordAppend(specTokens.GetRange(0, cachedGenerated), null);

                specStats = specDecoder.Stats;
                specWindow = specDecoder.MaxDraftTokens;
            }
            else
            {
                for (int step = 0; step < _maxTokens; step++)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        finishReason = "cancelled";
                        break;
                    }

                    int nextToken = sampler.Sample(logits, generatedTokens);
                    if (_model.Tokenizer.IsEos(nextToken))
                    {
                        finishReason = "eos";
                        break;
                    }

                    if (!EmitToken(nextToken))
                        break;

                    logits = _model.Forward(new[] { nextToken });
                    _kvCache.RecordAppend(nextToken, logits);
                }
            }
            decodeSw.Stop();

            if (useParser)
            {
                var finalParsed = parser.Add(string.Empty, true);
                if (finalParsed.ToolCalls != null && finalParsed.ToolCalls.Count > 0)
                    turnToolCalls.AddRange(finalParsed.ToolCalls);
                if (showThinking && !string.IsNullOrEmpty(finalParsed.Thinking))
                {
                    if (!inThinkingBlock) Console.Write("\n[thinking] ");
                    Console.Write(finalParsed.Thinking);
                    assistantThinkingBuffer += finalParsed.Thinking;
                }
                if (!string.IsNullOrEmpty(finalParsed.Content))
                {
                    if (inThinkingBlock) Console.Write("\n[answer] ");
                    Console.Write(finalParsed.Content);
                    assistantContentBuffer += finalParsed.Content;
                }
            }

            Console.WriteLine();

            double tokensPerSec = generatedTokens.Count > 0
                ? generatedTokens.Count / Math.Max(decodeSw.Elapsed.TotalSeconds, 1e-9)
                : 0;
            string specSummary = specStats == null
                ? string.Empty
                : $" spec=window{specWindow}/accepted{specStats.TokensAccepted}of{specStats.TokensDrafted}" +
                  $"({specStats.AcceptanceRate:P0})";
            Console.WriteLine($"[turn complete: tokens={generatedTokens.Count} prefillMs={prefillMs:F0} decodeMs={decodeSw.Elapsed.TotalMilliseconds:F0} tps={tokensPerSec:F1} ttftMs={firstTokenMs} reason={finishReason} kvPlan={planKind}{specSummary}]");

            _log.LogInformation(LogEventIds.ChatCompleted,
                "interactive.turn complete tokens={Tokens} promptTokens={PromptTokens} kvPlan={KvPlan} prefillMs={PrefillMs:F0} decodeMs={DecodeMs:F0} tps={TokensPerSec:F1} ttftMs={Ttft} reason={Reason}",
                generatedTokens.Count, promptTokenCount, planKind, prefillMs,
                decodeSw.Elapsed.TotalMilliseconds, tokensPerSec, firstTokenMs, finishReason);

            if (specStats != null)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "interactive.turn speculative: window={Window} confMin={ConfMin:F2} drafted={Drafted} accepted={Accepted} " +
                    "acceptanceRate={Rate:F3} verifySteps={Verify} plainSteps={Plain} rollbacks={Rollbacks} " +
                    "draftMs={DraftMs:F0} verifyMs={VerifyMs:F0} plainMs={PlainMs:F0} catchUpMs={CatchUpMs:F0}",
                    specWindow, specDecoder.MinDraftProb, specStats.TokensDrafted, specStats.TokensAccepted,
                    specStats.AcceptanceRate, specStats.VerifySteps, specStats.PlainSteps, specStats.RollbackSteps,
                    specStats.DraftMs, specStats.VerifyMs, specStats.PlainMs, specStats.CatchUpMs);
            }

            // Drop pending attachments on success - they belonged to the
            // user turn we just submitted.
            _pendingImages.Clear();
            _pendingAudios.Clear();
            _pendingTextFiles.Clear();
            _pendingIsVideo = false;

            // Append assistant entry to history with raw output tokens so the
            // next turn's renderer can splice them in.
            _history.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantContentBuffer,
                Thinking = assistantThinkingBuffer,
                // Kept on the assistant turn so the next render frames it as a tool call
                // rather than as prose, and so RawOutputTokens still splices: without the
                // tokens, every skill round-trip would re-tokenize the whole conversation
                // and re-prefill it.
                ToolCalls = turnToolCalls.Count > 0 ? new List<ToolCall>(turnToolCalls) : null,
                RawOutputTokens = new List<int>(generatedTokens),
                RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
            });

            return turnToolCalls;
        }

        /// <summary>
        /// The speculative decoder to serve this turn with, or null to decode one
        /// token per forward. Kept alive across turns so its draft cache can extend
        /// with the conversation instead of being rebuilt; rebuilt from scratch on a
        /// model swap (/model, /backend).
        /// </summary>
        private SpeculativeDecoder ResolveSpeculativeDecoder(List<ChatMessage> renderHistory)
        {
            // The session's decoder survives every turn but is not valid past a
            // /model or /backend swap: it holds the previous model's hidden state
            // and buffers sized to its vocabulary.
            SpeculativeDecoder reusable =
                ReferenceEquals(_specDecoderModel, _model) ? _specDecoder : null;
            var decoder = SpeculativeDecodingOptions.TryCreate(
                _model, _specSettings,
                HasMediaAttachments(renderHistory), out string declineReason, reusable);

            if (decoder == null)
            {
                // Say it once per session, not once per turn: a media turn in the
                // middle of a speculative session should not spam the console.
                if (_specSettings.Requested && declineReason != null && !_specDeclineLogged)
                {
                    _specDeclineLogged = true;
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "--spec was requested but speculative decoding is not available: {Reason} "
                        + "Serving standard decode.", declineReason);
                }
                return null;
            }

            if (!ReferenceEquals(_specDecoder, decoder))
            {
                _specDecoder = decoder;
                _specDecoderModel = _model;
                _log.LogInformation(LogEventIds.CliStarted,
                    "interactive speculative decoding armed: draft={DraftKind} window={Window} confMin={ConfMin:F2} verify={VerifyMode}",
                    SpeculativeDecodingOptions.DescribeDrafter(_specDecoder),
                    _specDecoder.MaxDraftTokens, _specDecoder.MinDraftProb,
                    SpeculativeDecodingOptions.DescribeVerification(_samplingConfig));
            }
            return _specDecoder;
        }

        /// <summary>
        /// True when <paramref name="cfg"/> selects the most probable token and
        /// nothing else. <see cref="SamplingConfig.IsGreedy"/> alone is not enough:
        /// the history penalties still rewrite the logits the drafts would be
        /// verified against.
        /// </summary>
        internal static bool IsArgmaxSampling(SamplingConfig cfg)
            => cfg != null
               && cfg.IsGreedy
               && Math.Abs(cfg.RepetitionPenalty - 1f) < 1e-6f
               && cfg.PresencePenalty == 0f
               && cfg.FrequencyPenalty == 0f
               && (cfg.FirstTokenAllowList == null || cfg.FirstTokenAllowList.Count == 0);

        private static bool HasMediaAttachments(List<ChatMessage> messages)
        {
            foreach (var m in messages)
            {
                if (m.IsVideo) return true;
                if (m.ImagePaths != null && m.ImagePaths.Count > 0) return true;
                if (m.AudioPaths != null && m.AudioPaths.Count > 0) return true;
            }
            return false;
        }

        /// <summary>Ordinary prompt prefill: plan the cache reuse and apply it.</summary>
        /// <summary>
        /// Whether to forward the shared part of the prompt before the first message.
        /// Off with <c>--no-prefix-cache</c>, which is the flag for a session that wants
        /// to start answering immediately rather than answer its first message quickly.
        /// </summary>
        public bool PrefixCacheEnabled { get; set; } = true;

        /// <summary>How many tokens the load-time warm prefix put in the cache, or 0.</summary>
        private int _warmPrefixTokens;

        /// <summary>Whether turn 1 has reported what the warm prefix was worth.</summary>
        private bool _warmPrefixReported;

        /// <summary>
        /// Below this a warm prefix is not worth a startup pause: the saving is a fraction
        /// of a second and the risk of getting the prefix wrong is the same either way.
        /// A bare CLI with no system prompt and no skills renders a handful of template
        /// tokens and lands here; an agent configuration renders thousands and does not.
        /// </summary>
        private const int MinimumWarmPrefixTokens = 64;

        /// <summary>
        /// Forward the part of the prompt that does not depend on what the user types,
        /// before they type it.
        ///
        /// <para>
        /// The CLI re-renders the whole conversation every turn and reuses the K/V only
        /// where the new token sequence still starts with the cached one. On turn 1 the
        /// cache is empty, so the system block, the tool declarations and the skill
        /// catalog are all forwarded while the user waits — thousands of tokens on an
        /// agent configuration. None of that depends on the message. This forwards it at
        /// startup so turn 1 begins as a continuation instead of a cold prefill.
        /// </para>
        /// <para>
        /// <b>The hard part is proving what we forward is a token PREFIX of what turn 1
        /// renders.</b> The system block is not one by inspection: the renderer emits a
        /// chat template, and the tokenizer sees the whole rendered string at once, so a
        /// BPE merge can span the boundary between the block and the user's first
        /// character and change a token BEFORE it. Rendering the system block alone and
        /// hoping is how this feature silently does nothing.
        /// </para>
        /// <para>
        /// So it is measured rather than assumed. Several complete first turns are
        /// rendered, with deliberately dissimilar first characters, and only the tokens
        /// on which ALL of them agree are kept — intersected with the system-only render,
        /// so no kept token can come from a probe's own text. Whatever survives that is a
        /// prefix of every first turn by construction, whatever the user types.
        /// </para>
        /// <para>
        /// It is also safe when it is wrong. A warm prefix that is not a prefix of the
        /// real prompt makes <see cref="KVCache.PlanReuse"/> return
        /// <see cref="ReusePlanKind.Reset"/> (or a shorter partial reuse), which throws the
        /// warmed state away and prefills normally: the cost is a wasted startup, never a
        /// wrong answer.
        /// </para>
        /// </summary>
        private void WarmSystemPrefix()
        {
            // Only ever from an idle session at the start of Run(). Warming on top of a
            // conversation would discard it, and every non-interactive entry point
            // (--dump-prompt, --test, --benchmark, --input-jsonl, --multi-turn-jsonl)
            // deliberately does not come through here.
            if (!PrefixCacheEnabled || _history.Count != 0 || _kvCache.Count != 0)
                return;

            try
            {
                if (!TryComputeWarmPrefix(out List<int> warm, out string skipped))
                {
                    if (!string.IsNullOrEmpty(skipped))
                        _log.LogDebug(LogEventIds.KvCacheReusePlan, "warm system prefix skipped: {Reason}", skipped);
                    return;
                }

                var sw = Stopwatch.StartNew();

                // Through whichever prefill path TURN 1 WILL USE, which is the whole point.
                // A speculative turn does not merely prefer its own path, it REFUSES a
                // cache the draft head did not build: SpeculativePrefill extends only when
                // decoder.CarryPosition == cached, and a decoder that has never run carries
                // -1. Warming through the plain path would therefore be discarded with
                // certainty on the first message of a speculative session — the startup
                // pause paid in full and nothing bought, which is strictly worse than not
                // warming at all. Resolving the decoder HERE also seeds it: the session
                // keeps the same instance, so it arrives at turn 1 carrying the warm
                // prefix's end position.
                //
                // Either path lays the tokens down on an empty cache through the same
                // reset-and-forward branch that already maintains the managed mirror and
                // the model's own sequence length together, so this does not become a
                // third place that can get that invariant wrong.
                SpeculativeDecoder warmDecoder = ResolveSpeculativeDecoder(WarmProbeHistory());
                if (warmDecoder != null)
                    SpeculativePrefill(warmDecoder, warm, out _);
                else
                    PlainPrefill(warm, out _);
                sw.Stop();

                // The one invariant the whole feature can violate. If it does not hold,
                // the cache is describing state the model does not have, and every later
                // turn would reuse from the wrong place.
                if (_kvCache.Count != warm.Count)
                {
                    _log.LogWarning(LogEventIds.KvCacheReusePlan,
                        "warm system prefix disarmed: mirror holds {Mirror} tokens for {Warm} forwarded",
                        _kvCache.Count, warm.Count);
                    _model.ResetKVCache();
                    _kvCache.Reset();
                    _warmPrefixTokens = 0;
                    return;
                }

                _warmPrefixTokens = warm.Count;
                Console.WriteLine($"[warm system prefix: {warm.Count} tokens in {sw.Elapsed.TotalSeconds:0.0}s]");
            }
            catch (OperationCanceledException)
            {
                // Ctrl+C during the warm forward. The model may hold part of a prefill
                // that the mirror does not describe, which is the one state that must
                // never reach a turn.
                _model.ResetKVCache();
                _kvCache.Reset();
                _warmPrefixTokens = 0;
                Console.WriteLine("[warm system prefix cancelled]");
            }
            catch (Exception ex)
            {
                _model.ResetKVCache();
                _kvCache.Reset();
                _warmPrefixTokens = 0;
                _log.LogWarning(LogEventIds.KvCacheReusePlan, ex, "warm system prefix failed; continuing cold");
            }
        }

        /// <summary>
        /// The tokens every possible first turn starts with, or false when there are too
        /// few to be worth forwarding.
        /// </summary>
        /// <remarks>
        /// The probes differ in their FIRST bytes on purpose. A merge that reaches back
        /// across the boundary depends on what follows it, so probes that all began the
        /// same way could agree on a token the real message would not produce. One starts
        /// with a letter, one with a space, one with a newline, one with punctuation that
        /// commonly merges leftwards, and one with the bracket the CLI itself prepends
        /// when a file is attached.
        /// </remarks>
        private bool TryComputeWarmPrefix(out List<int> warm, out string skipped)
        {
            warm = null;
            skipped = null;

            string leadingSystem = ComposeSystemPrompt();
            bool haveTools = _tools != null && _tools.Count > 0;
            if (string.IsNullOrEmpty(leadingSystem) && !haveTools)
            {
                skipped = "no system prompt and no tools";
                return false;
            }

            var probes = new[] { "Hello", " indented", "\nnewline", "```code", "[Attached file: a.txt]" };
            var renders = new List<IReadOnlyList<int>>();
            foreach (string probe in probes)
            {
                List<int> rendered = RenderFirstTurn(leadingSystem, probe);
                if (rendered == null || rendered.Count == 0)
                {
                    skipped = "the renderer produced nothing";
                    return false;
                }
                renders.Add(rendered);
            }

            // The structural bound: a token kept here must also appear, in the same place,
            // in a render that contains NO user text at all. Without it a long fixed
            // template could let the probes agree on something none of them should own.
            List<int> systemOnly = RenderSystemOnly(leadingSystem);
            if (systemOnly is { Count: > 0 })
                renders.Add(systemOnly);

            int keep = WarmPrefixLength(renders, MinimumWarmPrefixTokens);
            if (keep <= 0)
            {
                skipped = "too few tokens are common to every first turn";
                return false;
            }

            warm = ((List<int>)renders[0]).GetRange(0, keep);
            return true;
        }

        /// <summary>
        /// A stand-in first turn for decisions that must be made before the user has typed:
        /// text only, so nothing here can make the speculative path decline for media that
        /// a real turn may or may not carry.
        /// </summary>
        private List<ChatMessage> WarmProbeHistory()
        {
            var history = new List<ChatMessage>();
            string leadingSystem = ComposeSystemPrompt();
            if (!string.IsNullOrEmpty(leadingSystem))
                history.Add(new ChatMessage { Role = "system", Content = leadingSystem });
            history.Add(new ChatMessage { Role = "user", Content = "hi" });
            return history;
        }

        /// <summary>One complete first turn, rendered exactly as <see cref="Stream"/> renders it.</summary>
        private List<int> RenderFirstTurn(string leadingSystem, string userText)
        {
            var history = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(leadingSystem))
                history.Add(new ChatMessage { Role = "system", Content = leadingSystem });
            history.Add(new ChatMessage { Role = "user", Content = userText });
            return _renderer.RenderToTokens(
                _model.Tokenizer, _model.Config.ChatTemplate, history, _model.Config.Architecture,
                addGenerationPrompt: true, out _, out _,
                tools: _tools, enableThinking: _enableThinking);
        }

        /// <summary>The leading block with no user turn, as an upper bound on what may be warmed.</summary>
        private List<int> RenderSystemOnly(string leadingSystem)
        {
            if (string.IsNullOrEmpty(leadingSystem))
                return null;
            try
            {
                var history = new List<ChatMessage>
                {
                    new ChatMessage { Role = "system", Content = leadingSystem },
                };
                return _renderer.RenderToTokens(
                    _model.Tokenizer, _model.Config.ChatTemplate, history, _model.Config.Architecture,
                    addGenerationPrompt: false, out _, out _,
                    tools: _tools, enableThinking: _enableThinking);
            }
            catch (Exception)
            {
                // A template that refuses a system-only render costs the extra bound, not
                // the feature: the probe intersection alone is still sound.
                return null;
            }
        }

        private static List<int> CommonPrefix(List<int> a, List<int> b)
        {
            int n = Math.Min(a.Count, b.Count);
            int i = 0;
            while (i < n && a[i] == b[i])
                i++;
            return a.GetRange(0, i);
        }

        /// <summary>
        /// How many leading tokens every one of <paramref name="renders"/> agrees on, less
        /// one for margin, or 0 when that is below <paramref name="minimum"/>.
        /// </summary>
        /// <remarks>
        /// Separated from the rendering so the arithmetic that decides what is safe to
        /// forward can be tested without a model. This is the whole safety argument of the
        /// CLI warm prefix in one function: a token survives only if it is in the same
        /// position in every render it was given.
        /// </remarks>
        internal static int WarmPrefixLength(IReadOnlyList<IReadOnlyList<int>> renders, int minimum)
        {
            if (renders == null || renders.Count == 0)
                return 0;

            int common = int.MaxValue;
            foreach (IReadOnlyList<int> render in renders)
            {
                if (render == null || render.Count == 0)
                    return 0;
                common = Math.Min(common, render.Count);
            }

            for (int i = 0; i < common; i++)
            {
                int token = renders[0][i];
                for (int r = 1; r < renders.Count; r++)
                {
                    if (renders[r][i] != token)
                    {
                        common = i;
                        i = common;  // stop the outer scan
                        break;
                    }
                }
                if (i >= common)
                    break;
            }

            int keep = Math.Max(0, common - 1);
            return keep < minimum ? 0 : keep;
        }

        /// <summary>
        /// Say, once, what the warm prefix was actually worth on the turn that could use it.
        /// A warm prefill that is silently discarded — a template that renders the date, a
        /// tool set rebuilt between startup and the first message — looks exactly like a
        /// normal cold start without this line.
        /// </summary>
        private void ReportWarmPrefixOutcome(ReusePlanKind kind, int reused)
        {
            if (_warmPrefixReported || _warmPrefixTokens == 0)
                return;
            _warmPrefixReported = true;
            if (kind == ReusePlanKind.Reset)
            {
                Console.WriteLine($"[warm system prefix missed: {_warmPrefixTokens} tokens discarded, "
                    + "this turn prefilled in full]");
            }
        }

        private float[] PlainPrefill(List<int> inputTokens, out ReusePlanKind kind)
        {
            ReusePlan plan = _kvCache.PlanReuse(inputTokens, _model.SupportsKVCacheTruncation,
                _model.KVCacheTruncationGranularity);
            return ApplyReusePlan(plan, inputTokens, out kind);
        }

        /// <summary>
        /// Prompt prefill for the speculative path. Every forward runs through the
        /// drafter-aware path so the draft head's KV covers the prompt.
        ///
        /// The reuse policy is narrower than <see cref="PlainPrefill"/>'s by
        /// necessity: the cache is EXTENDED when the prompt continues it exactly,
        /// and otherwise rebuilt — it is never truncated to a common prefix the way
        /// the plain path can. A draft head chains from the hidden state of the
        /// token before the one it drafts, and that state is only held for the
        /// position the decoder last stopped at (<see cref="SpeculativeDecoder.CarryPosition"/>);
        /// resuming anywhere else would build the draft head's KV from the wrong
        /// hidden state. The emitted stream would stay correct — verification is
        /// trunk-driven — but acceptance would decay for the rest of the session
        /// with nothing in the log to say why.
        ///
        /// So a turn whose rendered prompt diverges from the cache re-prefills in
        /// full where a plain turn would truncate-and-refill. Chat templates that
        /// re-render a past turn's generation prompt differently make that the
        /// common case, which costs a prefill that grows with the conversation
        /// while the decode saving is per-token: on a long session, measure before
        /// assuming speculation still pays.
        /// </summary>
        private float[] SpeculativePrefill(SpeculativeDecoder decoder, List<int> inputTokens,
            out ReusePlanKind kind)
        {
            int cached = _kvCache.Count;
            int commonPrefix = cached > 0 ? _kvCache.CommonPrefixLength(inputTokens) : 0;
            bool extend = cached > 0
                && cached < inputTokens.Count
                && commonPrefix == cached
                && decoder.CarryPosition == cached;

            // Which of the two conditions refused a reuse is not guessable from the
            // outside, and they call for opposite fixes (a prompt that diverges vs
            // a draft head that cannot resume).
            _log.LogDebug(LogEventIds.KvCacheReusePlan,
                "speculative prefill plan: cached={Cached} promptTokens={PromptTokens} "
                + "commonPrefix={CommonPrefix} carryPosition={CarryPosition} extend={Extend}",
                cached, inputTokens.Count, commonPrefix, decoder.CarryPosition, extend);

            if (!extend)
            {
                _model.ResetKVCache();
                _kvCache.Reset();
                decoder.Reset();
                kind = ReusePlanKind.Reset;

                var all = inputTokens.ToArray();
                float[] fullLogits = decoder.Prefill(all);
                _kvCache.RecordAppend(all, fullLogits);
                return fullLogits;
            }

            kind = ReusePlanKind.PartialReuse;
            var suffix = new int[inputTokens.Count - cached];
            for (int i = 0; i < suffix.Length; i++)
                suffix[i] = inputTokens[cached + i];
            float[] logits = decoder.Prefill(suffix);
            _kvCache.RecordAppend(suffix, logits);
            return logits;
        }

        /// <summary>
        /// Carry out <paramref name="plan"/>. <paramref name="applied"/> reports what
        /// actually happened, which is not always what was planned: a model whose rewind
        /// depth depends on where the sequence is can decline the truncation a partial
        /// reuse needs, and then the only correct answer is the full re-prefill. The
        /// per-turn log line shows the applied kind, so a declined rewind is visible
        /// rather than being reported as a reuse that did not happen.
        /// </summary>
        private float[] ApplyReusePlan(ReusePlan plan, List<int> inputTokens, out ReusePlanKind applied)
        {
            applied = plan.Kind;

            if (plan.Kind == ReusePlanKind.ExactMatch)
                return plan.CachedLogits;

            if (plan.Kind == ReusePlanKind.PartialReuse)
            {
                int reused = plan.ReusedPrefixLength;
                // A reuse boundary inside an image or audio span would truncate half an
                // injection, and the re-forward would queue that span's embeddings at the
                // wrong offset; the injector pulls the boundary back to the span start.
                // Realigning after it matters for a model whose head can only stop on a
                // compression-block boundary - the clamp knows nothing about that.
                reused = _model.MultimodalInjector.ClampReusablePrefix(reused);
                int granularity = _model.KVCacheTruncationGranularity;
                if (granularity > 1) reused -= reused % granularity;

                if (reused > 0 && _model.TryTruncateKVCache(reused))
                {
                    int suffixLength = inputTokens.Count - reused;
                    _kvCache.TruncateTo(reused);

                    var suffix = new int[suffixLength];
                    for (int i = 0; i < suffixLength; i++)
                        suffix[i] = inputTokens[reused + i];
                    float[] suffixLogits = ForwardRefillChunked(suffix, promptStartToken: reused);
                    _kvCache.RecordAppend(suffix, suffixLogits);
                    return suffixLogits;
                }

                _log.LogDebug(LogEventIds.KvCacheReusePlan,
                    "partial reuse declined by the model at {Reused} of {Cached} cached token(s); "
                    + "re-prefilling {PromptTokens} token(s)",
                    reused, _kvCache.Count, inputTokens.Count);
                applied = ReusePlanKind.Reset;
            }

            _model.ResetKVCache();
            _kvCache.Reset();
            var allTokens = inputTokens.ToArray();
            float[] logits = ForwardRefillChunked(allTokens, promptStartToken: 0);
            _kvCache.RecordAppend(allTokens, logits);
            return logits;
        }

        /// <summary>
        /// Feed <paramref name="tokens"/> through the model in
        /// <see cref="PrefillChunking.ResolveChunkSize"/>-sized chunks so the
        /// attention score tensor stays bounded for long prompts. Each chunk
        /// queues its own multimodal-embedding slice so vision spans line up
        /// with the right forward call. Returns the next-token logits from the
        /// final chunk (sampler only ever consumes the trailing logits anyway).
        /// </summary>
        private float[] ForwardRefillChunked(int[] tokens, int promptStartToken)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Prompt token list cannot be null or empty.", nameof(tokens));

            int chunkSize = PrefillChunking.ResolveChunkSize(_backend, tokens.Length);
            if (chunkSize >= tokens.Length)
            {
                _model.MultimodalInjector.QueuePromptEmbeddingsForSlice(promptStartToken, tokens.Length);
                return _model.ForwardRefill(tokens);
            }

            float[] logits = null;
            for (int start = 0; start < tokens.Length; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Length - start);
                int[] chunk = new int[length];
                Array.Copy(tokens, start, chunk, 0, length);
                _model.MultimodalInjector.QueuePromptEmbeddingsForSlice(promptStartToken + start, length);
                logits = _model.ForwardRefill(chunk);
            }
            return logits;
        }

        // ---- Helpers ---------------------------------------------------------

        private void OnCancelKeyPress(object sender, ConsoleCancelEventArgs e)
        {
            // Always intercept so we control the process lifetime; the only
            // exit paths are /exit, /quit, EOF (Ctrl+D / closed stdin) and a
            // second Ctrl+C while the prompt is idle.
            e.Cancel = true;

            if (_isGenerating && _generationCts != null)
            {
                _generationCts.Cancel();
                return;
            }

            // Idle press at the prompt -> exit. We can't unblock the running
            // Console.ReadLine(), so we just flag the intent and the next loop
            // iteration will quit.
            _shouldExit = true;
            Console.WriteLine();
            Console.WriteLine("[exiting] press Enter to confirm, or run /exit to leave.");
        }

        private static int FindValidUtf8Length(List<byte> bytes)
        {
            int len = bytes.Count;
            if (len == 0) return 0;
            for (int i = 1; i <= Math.Min(4, len); i++)
            {
                byte b = bytes[len - i];
                if ((b & 0x80) == 0) return len;
                if ((b & 0xE0) == 0xC0) return (i >= 2) ? len : len - i;
                if ((b & 0xF0) == 0xE0) return (i >= 3) ? len : len - i;
                if ((b & 0xF8) == 0xF0) return (i >= 4) ? len : len - i;
                if ((b & 0xC0) == 0x80) continue;
                return len;
            }
            return len;
        }

        private static bool TryParseBool(string value, out bool result)
        {
            switch (value.Trim().ToLowerInvariant())
            {
                case "1":
                case "on":
                case "true":
                case "yes":
                    result = true;
                    return true;
                case "0":
                case "off":
                case "false":
                case "no":
                    result = false;
                    return true;
                default:
                    result = false;
                    return false;
            }
        }
    }
}
