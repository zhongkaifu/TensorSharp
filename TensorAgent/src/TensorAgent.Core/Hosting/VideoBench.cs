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
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Benchmark: clips made through the app's own video turn — <see cref="VideoTurns"/> with the
/// LoRA plug-ins the user turned on (<see cref="AgentAppHost.PrepareVideoTurn"/>, a MiniMax-H3
/// Turbo plug-in among them) — timed, in whatever build is running.
///
/// <para>
/// Launched with <c>TENSORAGENT_VIDEO_BENCH=1</c> and a video model (via
/// <c>TENSORAGENT_USE_MODEL</c> or the remembered choice). It waits for the load, then makes
/// <c>TENSORAGENT_VIDEO_BENCH_RUNS</c> clips (2) of <c>TENSORAGENT_VIDEO_BENCH_PROMPT</c>, from the
/// photo at <c>TENSORAGENT_VIDEO_BENCH_IMAGE</c> when one is named (copied into the uploads, so the
/// clip starts from it as a chat photo does, on the app's compact canvas), one <c>videobench</c>
/// line each — the steps, the plug-ins, when the first frame came (the plug-ins are applied
/// before it), when the first denoising step ended, the seconds a step after it, the decode and
/// save, the whole clip, and the clip with its SHA-256 and the process's memory — on stdout and
/// in <c>Library/Caches/TensorAgent/logs/videobench.log</c>. The engine's own log line says how
/// long applying the plug-ins took. The first clip after a launch also pays for applying them;
/// the second is the steady state. Like <see cref="ImageBench"/>, it exists for the Release
/// build, whose page no harness can drive, and is honoured in every build.
/// </para>
/// </summary>
public sealed class VideoBench
{
    public const string EnableVariable = "TENSORAGENT_VIDEO_BENCH";
    public const string RunsVariable = "TENSORAGENT_VIDEO_BENCH_RUNS";
    public const string PromptVariable = "TENSORAGENT_VIDEO_BENCH_PROMPT";
    public const string ImageVariable = "TENSORAGENT_VIDEO_BENCH_IMAGE";

    public static bool Requested =>
        string.Equals(Environment.GetEnvironmentVariable(EnableVariable), "1", StringComparison.Ordinal);

    private readonly AgentAppHost _app;
    private readonly string _logPath;

    public VideoBench(AgentAppHost app)
    {
        _app = app ?? throw new ArgumentNullException(nameof(app));
        _logPath = Path.Combine(app.Paths.LogsDirectory, "videobench.log");
    }

    private void Say(string line)
    {
        Console.WriteLine("TensorAgent: " + line);
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!);
            File.AppendAllText(_logPath, $"{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff} {line}{Environment.NewLine}");
        }
        catch (Exception) { /* diagnostic only */ }
    }

    public async Task RunAsync(CancellationToken token)
    {
        try
        {
            int runs = int.TryParse(Environment.GetEnvironmentVariable(RunsVariable), NumberStyles.Integer,
                CultureInfo.InvariantCulture, out int r) && r > 0 ? r : 2;
            string prompt = Environment.GetEnvironmentVariable(PromptVariable) is { Length: > 0 } p
                ? p
                : "A lighthouse at dusk, waves breaking on the rocks below, the beam sweeping over the sea";
            for (int i = 0; i < 600 && _app.ModelLoad != AgentAppHost.ModelLoadState.Loaded; i++)
                await Task.Delay(1000, token).ConfigureAwait(false);
            if (_app.ModelLoad != AgentAppHost.ModelLoadState.Loaded || !_app.Chat.LoadedModelMakesVideo)
            {
                Say($"videobench FAIL no video model (state {_app.ModelLoad})");
                return;
            }
            string? photo = null;
            if (Environment.GetEnvironmentVariable(ImageVariable) is { Length: > 0 } image)
            {
                if (!File.Exists(image))
                {
                    Say($"videobench FAIL {ImageVariable}={image} does not exist");
                    return;
                }
                photo = "videobench-" + Guid.NewGuid().ToString("N")[..8] + Path.GetExtension(image);
                Directory.CreateDirectory(_app.Options.UploadDirectory);
                File.Copy(image, Path.Combine(_app.Options.UploadDirectory, photo));
            }
            Say($"videobench model {_app.ModelService.LoadedModelName} on {_app.ModelService.LoadedBackend}; "
                + $"{runs} clip(s) of \"{prompt}\"{(photo is null ? "" : " from " + Path.GetFileName(Environment.GetEnvironmentVariable(ImageVariable)))}; "
                + ProcessMemory.Describe());
            for (int run = 1; run <= runs; run++)
                await ClipAsync(run, prompt, photo, token).ConfigureAwait(false);
            Say("videobench done");
        }
        catch (OperationCanceledException)
        {
            Say("videobench cancelled");
        }
        catch (Exception ex)
        {
            Say("videobench FAIL " + ex.Message);
        }
    }

    private async Task ClipAsync(int run, string prompt, string? photo, CancellationToken token)
    {
        object created = _app.Chat.CreateSession();
        string session = created.GetType().GetProperty("sessionId")?.GetValue(created) as string ?? string.Empty;
        string[] photos = photo is null ? Array.Empty<string>() : new[] { photo };
        JsonElement body = JsonSerializer.SerializeToElement(new
        {
            sessionId = session,
            messages = new[] { new { role = "user", content = prompt, stillImagePaths = photos, imagePaths = photos } },
        });

        // The phone takes the GPU away from a backgrounded app; wait like a turn does.
        await _app.Compute.WaitAsync(token).ConfigureAwait(false);

        var clock = Stopwatch.StartNew();
        double firstFrame = 0, firstStep = 0, lastStep = 0;
        int steps = 0;
        string loras = "none";
        string? url = null, error = null;
        await foreach (object frame in VideoTurns.StreamAsync(_app.Chat, body, token, _ => _app.PrepareVideoTurn()).ConfigureAwait(false))
        {
            if (firstFrame == 0)
                firstFrame = clock.Elapsed.TotalSeconds;
            // A step is reported twice, by the hook and then with its timing; both end it.
            if (AgentAppHost.StringIn(frame, "video_phase") == "denoise" && AgentAppHost.IntIn(frame, "video_step") > 0)
            {
                if (firstStep == 0) firstStep = clock.Elapsed.TotalSeconds;
                lastStep = clock.Elapsed.TotalSeconds;
                steps = AgentAppHost.IntIn(frame, "video_steps");
            }
            if (AgentAppHost.ValueIn(frame, "video_loras") is IEnumerable<string> names)
                loras = string.Join(" + ", names);
            url ??= AgentAppHost.StringIn(frame, "videoUrl");
            error ??= AgentAppHost.StringIn(frame, "error");
        }
        double seconds = clock.Elapsed.TotalSeconds;
        if (error is not null || url is null)
        {
            Say($"videobench run {run} FAIL after {seconds:0.0}s: {error ?? "no clip"}");
            return;
        }
        string file = Path.Combine(_app.Options.UploadDirectory, url[(url.LastIndexOf('/') + 1)..]);
        string sha = File.Exists(file) ? Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file))) : "missing";
        Say($"videobench run {run}: {steps} steps with {loras}; first frame at {firstFrame:0.0}s, "
            + $"first step ended at {firstStep:0.0}s, then {(lastStep - firstStep) / Math.Max(1, steps - 1):0.00}s a step, "
            + $"decode and save {seconds - lastStep:0.0}s, clip in {seconds:0.0}s; "
            + $"{url} sha256 {sha}; {ProcessMemory.Describe()}");
    }
}
