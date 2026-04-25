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
        private List<ToolFunction> _tools;
        private string _systemPrompt;
        private bool _enableThinking;
        private int _maxTokens;
        private bool _multilineInput;

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
            ILogger log)
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
            _enableThinking = enableThinking;
            _maxTokens = maxTokens > 0 ? maxTokens : 512;
            _log = log;
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
            Console.WriteLine("                         (cpu | ggml_cpu | ggml_metal | ggml_cuda).");
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
            Console.WriteLine("Plain text without a leading slash is sent to the model as the next user turn.");
            Console.WriteLine("Press Ctrl+C while generating to interrupt; press Ctrl+C at the prompt to exit.");
        }

        private void ResetSession()
        {
            _history.Clear();
            _kvCache.Reset();
            _model.ResetKVCache();
            ClearAttachments();
            Console.WriteLine("Conversation history and KV cache cleared.");
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
            Console.WriteLine(_systemPrompt == null
                ? "System prompt cleared. Conversation reset."
                : $"System prompt set ({_systemPrompt.Length} chars). Conversation reset.");
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
                Console.WriteLine($"Current backend: {_backend}. Usage: /backend cpu|ggml_cpu|ggml_metal|ggml_cuda");
                return;
            }
            if (!TryParseBackend(requested, out BackendType target))
            {
                Console.WriteLine($"Unknown backend '{requested}'. Use: cpu, ggml_cpu, ggml_metal, ggml_cuda");
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
            try
            {
                Console.WriteLine($"Loading {Path.GetFileName(modelPath)} on {backend}...");
                var sw = Stopwatch.StartNew();
                ModelBase newModel = ModelBase.Create(modelPath, backend);
                sw.Stop();

                // Only after the new model is constructed do we tear down the
                // old one - if Create() throws we must preserve the working
                // session for the user. Skip disposing the caller-owned
                // original; it gets cleaned up by the caller's own using.
                if (_model != null && !ReferenceEquals(_model, _originalModel))
                    _model.Dispose();
                _model = newModel;
                _modelPath = modelPath;
                _backend = backend;
                _mmProjPath = null;

                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    try
                    {
                        _model.MultimodalInjector.LoadProjectors(mmProjPath);
                        _mmProjPath = mmProjPath;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Loaded model but failed to load projector: {ex.Message}");
                    }
                }

                // History / KV state from the previous tokenizer is meaningless
                // against the new one, so drop everything.
                _history.Clear();
                _kvCache.Reset();
                _model.ResetKVCache();
                ClearAttachments();

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
                Console.WriteLine($"Failed to load model: {ex.Message}");
                _log.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Failed to reload model {Model} on backend {Backend}: {Error}",
                    Path.GetFileName(modelPath), backend, ex.Message);
            }
        }

        private static bool TryParseBackend(string raw, out BackendType backend)
        {
            switch ((raw ?? string.Empty).Trim().ToLowerInvariant())
            {
                case "cpu":
                    backend = BackendType.Cpu;
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
                case "cuda":
                case "ggml_cuda":
                case "ggml-cuda":
                    backend = BackendType.GgmlCuda;
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
                Stream(userText, renderHistory, _generationCts.Token);
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
            if (!string.IsNullOrEmpty(_systemPrompt))
                rendered.Add(new ChatMessage { Role = "system", Content = _systemPrompt });
            rendered.AddRange(_history);
            return rendered;
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

        private void Stream(string userText, List<ChatMessage> renderHistory, CancellationToken cancellationToken)
        {
            string arch = _model.Config.Architecture;

            var inputTokens = _renderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                renderHistory,
                arch,
                addGenerationPrompt: true,
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

            var prefillSw = Stopwatch.StartNew();
            ReusePlan plan = _kvCache.PlanReuse(inputTokens, _model.SupportsKVCacheTruncation);
            float[] logits = ApplyReusePlan(plan, inputTokens);
            prefillSw.Stop();
            double prefillMs = prefillSw.Elapsed.TotalMilliseconds;
            int promptTokenCount = inputTokens.Count;

            var sampler = new TokenSampler(_samplingConfig);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            // Streaming output parser so we strip <think> blocks from the live
            // console output (they're surfaced separately when --think is on).
            var parser = OutputParserFactory.Create(arch);
            parser.Init(_enableThinking, _tools);
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

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
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
                        break;
                    }
                }

                logits = _model.Forward(new[] { nextToken });
                _kvCache.RecordAppend(nextToken, logits);
            }
            decodeSw.Stop();

            if (useParser)
            {
                var finalParsed = parser.Add(string.Empty, true);
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
            Console.WriteLine($"[turn complete: tokens={generatedTokens.Count} prefillMs={prefillMs:F0} decodeMs={decodeSw.Elapsed.TotalMilliseconds:F0} tps={tokensPerSec:F1} ttftMs={firstTokenMs} reason={finishReason} kvPlan={plan.Kind}]");

            _log.LogInformation(LogEventIds.ChatCompleted,
                "interactive.turn complete tokens={Tokens} promptTokens={PromptTokens} kvPlan={KvPlan} prefillMs={PrefillMs:F0} decodeMs={DecodeMs:F0} tps={TokensPerSec:F1} ttftMs={Ttft} reason={Reason}",
                generatedTokens.Count, promptTokenCount, plan.Kind, prefillMs,
                decodeSw.Elapsed.TotalMilliseconds, tokensPerSec, firstTokenMs, finishReason);

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
                RawOutputTokens = new List<int>(generatedTokens),
            });
        }

        private float[] ApplyReusePlan(ReusePlan plan, List<int> inputTokens)
        {
            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                    // Even on a full reuse the model needs the embedding spans
                    // re-queued so the next Forward call's embedding lookup sees
                    // them at their original positions.
                    _model.MultimodalInjector.QueuePromptEmbeddings(inputTokens.Count);
                    return plan.CachedLogits;

                case ReusePlanKind.PartialReuse:
                {
                    int reused = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;
                    _model.TruncateKVCache(reused);
                    _kvCache.TruncateTo(reused);

                    _model.MultimodalInjector.QueuePromptEmbeddings(reused);
                    var suffix = new int[suffixLength];
                    for (int i = 0; i < suffixLength; i++)
                        suffix[i] = inputTokens[reused + i];
                    float[] logits = _model.ForwardRefill(suffix);
                    _kvCache.RecordAppend(suffix, logits);
                    return logits;
                }

                case ReusePlanKind.Reset:
                default:
                {
                    _model.ResetKVCache();
                    _kvCache.Reset();
                    _model.MultimodalInjector.QueuePromptEmbeddings(0);
                    var allTokens = inputTokens.ToArray();
                    float[] logits = _model.ForwardRefill(allTokens);
                    _kvCache.RecordAppend(allTokens, logits);
                    return logits;
                }
            }
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
