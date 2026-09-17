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
    /// The shared inference engine owns radix prefix reuse and generation;
    /// the console session adds:
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
        private CliInferenceSession _inference;
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

        private readonly SpeculationOptions _specSettings;

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
            ResetInference();

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
                _inference?.Dispose();
                _inference = null;
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
            Console.WriteLine("  /reset, /new           Start a new chat; retain the shared system prefix.");
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
                ? "Conversation history cleared. Shared system prefixes remain eligible for reuse."
                : "Conversation history cleared. A fresh cache scope isolates the new chat.");
        }

        private CliInferenceSession Inference => _inference ??= new CliInferenceSession(
            _model, SchedulerConfig.FromEnvironment().WithSpeculation(_specSettings), _log);

        private void ResetInference()
        {
            _inference?.Dispose();
            _inference = null;
            _model.ResetKVCache();
            _sharedPrefix = null;
        }

        private void RestoreWarmPrefixOrClear()
        {
            // A new scope cannot reuse private turns from the previous conversation.
            // Keep the engine and its public checkpoint instead of prefilling it again.
            _inference?.StartNewConversation();
            _warmPrefixReported = false;
            WarmSystemPrefix();
        }

        private void SetSystemPrompt(string text)
        {
            _systemPrompt = string.IsNullOrWhiteSpace(text) ? null : text;
            // Switching the system prompt invalidates every cached prefix, so
            // reset both the model state and the tracked turns to keep
            // generation correct.
            _history.Clear();
            ResetInference();
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
            if (TensorSharp.Models.Architecture.AudioInputSupport.UnsupportedReasonFor(_model) is string audioError)
            {
                Console.WriteLine($"Audio not attached: {audioError}");
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
            Console.WriteLine($"  Conversation: {turns} user turn(s), last request computed {_inference?.CachedTokens ?? 0} token(s).");
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
                _inference?.Dispose();
                _inference = null;
                _model = newModel;
                newModel = null; // ownership now belongs to the session
                _modelPath = modelPath;
                _backend = backend;
                _mmProjPath = loadedMmProjPath;

                // History / KV and speculative state from the previous tokenizer
                // are meaningless against the new one, so drop everything.
                _history.Clear();
                _warmPrefixTokens = 0;
                _warmPrefixReported = false;
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
                WarmSystemPrefix();
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
            ResetInference();
            _warmPrefixTokens = 0;
            _warmPrefixReported = false;
            WarmSystemPrefix();
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

            string requestId = $"cli-{Guid.NewGuid():N}";
            try
            {
                inputTokens = _model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens, requestId);
            }
            catch
            {
                _model.MultimodalInjector.ClearPreparedPromptState(requestId);
                throw;
            }
            int promptTokenCount = inputTokens.Count;
            _log.LogDebug(LogEventIds.ChatStarted,
                "interactive prompt tokens={PromptTokens} thinking={Thinking}",
                promptTokenCount, _enableThinking);
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

            CliInferenceSession.Result result;
            try
            {
                result = Inference.Generate(inputTokens, _maxTokens, _samplingConfig,
                    EmitToken, cancellationToken, requestId,
                    _model.MultimodalInjector.GetPreparedMediaSpans(requestId),
                    enablePrefixCache: PrefixCacheEnabled,
                    sharedPrefixTokens: CliSharedPrefix.MatchingLength(_sharedPrefix, inputTokens));
            }
            finally
            {
                _model.MultimodalInjector.ClearPreparedPromptState(requestId);
            }
            decodeSw.Stop();
            double prefillMs = result.PrefillMs;
            int reusedTokens = result.Completion.PrefixCacheReusedTokens;
            string planKind = reusedTokens > 0 ? "RadixReuse" : "Prefill";
            ReportWarmPrefixOutcome(reusedTokens);
            if (finishReason == "max_tokens")
                finishReason = cancellationToken.IsCancellationRequested ? "cancelled" : result.Completion.FinishReason;
            var specStats = result.Sequence.SpecStats;

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
                ? generatedTokens.Count / Math.Max(result.DecodeMs / 1000.0, 1e-9)
                : 0;
            string specSummary = specStats == null
                ? string.Empty
                : $" spec=accepted{specStats.TokensAccepted}of{specStats.TokensDrafted}" +
                  $"({specStats.AcceptanceRate:P0})";
            Console.WriteLine($"[turn complete: tokens={generatedTokens.Count} prefillMs={prefillMs:F0} decodeMs={result.DecodeMs:F0} tps={tokensPerSec:F1} ttftMs={firstTokenMs} reason={finishReason} kvPlan={planKind}{specSummary}]");

            _log.LogInformation(LogEventIds.ChatCompleted,
                "interactive.turn complete tokens={Tokens} promptTokens={PromptTokens} kvPlan={KvPlan} prefillMs={PrefillMs:F0} decodeMs={DecodeMs:F0} tps={TokensPerSec:F1} ttftMs={Ttft} reason={Reason}",
                generatedTokens.Count, promptTokenCount, planKind, prefillMs,
                result.DecodeMs, tokensPerSec, firstTokenMs, finishReason);

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
        /// Whether to forward the shared part of the prompt before the first message.
        /// Off with <c>--no-prefix-cache</c>, which also disables radix prefix reuse.
        /// </summary>
        public bool PrefixCacheEnabled { get; set; } = true;

        /// <summary>How many tokens the load-time warm prefix put in the cache, or 0.</summary>
        private int _warmPrefixTokens;

        private List<int> _sharedPrefix;

        /// <summary>Whether turn 1 has reported what the warm prefix was worth.</summary>
        private bool _warmPrefixReported;

        /// <summary>
        /// Below this a warm prefix is not worth a startup pause: the saving is a fraction
        /// of a second and the risk of getting the prefix wrong is the same either way.
        /// A bare CLI with no system prompt and no skills renders a handful of template
        /// tokens and lands here; an agent configuration renders thousands and does not.
        /// </summary>
        private const int MinimumWarmPrefixTokens = 64;

        /// <summary>Warm the invariant system/tool prefix through the shared engine.</summary>
        private void WarmSystemPrefix()
        {
            if (!PrefixCacheEnabled || !SchedulerConfig.FromEnvironment().EnablePrefixCaching
                || _warmPrefixTokens > 0 || _history.Count != 0 || (_inference?.CachedTokens ?? 0) != 0)
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
                // A one-token request captures the prompt boundary. Its output is
                // discarded; normal turns resume the same engine-owned prefix.
                Inference.Generate(warm, 1, SamplingConfig.Greedy, sharedPrefixTokens: warm.Count);
                // The synthetic warmup tail belongs to no user conversation.
                Inference.StartNewConversation();
                _warmPrefixTokens = warm.Count;
                Console.WriteLine($"[warm system prefix: {warm.Count} tokens in {sw.Elapsed.TotalSeconds:0.0}s]");
            }
            catch (Exception ex)
            {
                ResetInference();
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
            _sharedPrefix = CliSharedPrefix.Compute(ComposeSystemPrompt(), _tools is { Count: > 0 },
                (messages, generationPrompt) => _renderer.RenderToTokens(
                    _model.Tokenizer, _model.Config.ChatTemplate, new List<ChatMessage>(messages),
                    _model.Config.Architecture, generationPrompt, out _, out _,
                    tools: _tools, enableThinking: _enableThinking));
            warm = _sharedPrefix;
            skipped = warm.Count < MinimumWarmPrefixTokens
                ? "too few tokens are common to every first turn" : null;
            return skipped == null;
        }

        internal static int WarmPrefixLength(IReadOnlyList<IReadOnlyList<int>> renders, int minimum)
            => CliSharedPrefix.Length(renders, minimum);

        /// <summary>
        /// Say, once, what the warm prefix was actually worth on the turn that could use it.
        /// A warm prefill that is silently discarded — a template that renders the date, a
        /// tool set rebuilt between startup and the first message — looks exactly like a
        /// normal cold start without this line.
        /// </summary>
        private void ReportWarmPrefixOutcome(int reused)
        {
            if (_warmPrefixReported || _warmPrefixTokens == 0)
                return;
            _warmPrefixReported = true;
            Console.WriteLine(reused == 0
                ? $"[warm system prefix missed: {_warmPrefixTokens} tokens were not reused]"
                : $"[warm system prefix reused: {Math.Min(reused, _warmPrefixTokens)}/{_warmPrefixTokens} tokens]");
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
