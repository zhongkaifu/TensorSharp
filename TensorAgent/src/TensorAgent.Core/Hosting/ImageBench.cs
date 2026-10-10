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
/// Benchmark: pictures made through the app's own image turn — <see cref="ImageTurns"/> with
/// the LoRA plug-ins the user turned on (<see cref="AgentAppHost.PrepareImageTurn"/>) — timed,
/// in whatever build is running.
///
/// <para>
/// Launched with <c>TENSORAGENT_IMAGE_BENCH=1</c> and an image model (via
/// <c>TENSORAGENT_USE_MODEL</c> or the remembered choice). It waits for the load, then makes
/// <c>TENSORAGENT_IMAGE_BENCH_RUNS</c> pictures (2) of <c>TENSORAGENT_IMAGE_BENCH_PROMPT</c>, one
/// <c>imagebench</c> line each — the steps, the plug-ins, when the first step ended, the
/// seconds a step after it (preview decodes included), the final decode and save, the whole
/// picture, and the picture with its SHA-256 — on stdout and in
/// <c>Library/Caches/TensorAgent/logs/imagebench.log</c>. The first picture after a launch also
/// pays for applying the plug-ins; the second is the steady state. It exists for the Release
/// build, whose page no harness can drive: like <see cref="SpeculationBench"/>, it is honoured
/// in every build.
/// </para>
/// </summary>
public sealed class ImageBench
{
    public const string EnableVariable = "TENSORAGENT_IMAGE_BENCH";
    public const string RunsVariable = "TENSORAGENT_IMAGE_BENCH_RUNS";
    public const string PromptVariable = "TENSORAGENT_IMAGE_BENCH_PROMPT";

    public static bool Requested =>
        string.Equals(Environment.GetEnvironmentVariable(EnableVariable), "1", StringComparison.Ordinal);

    private readonly AgentAppHost _app;
    private readonly string _logPath;

    public ImageBench(AgentAppHost app)
    {
        _app = app ?? throw new ArgumentNullException(nameof(app));
        _logPath = Path.Combine(app.Paths.LogsDirectory, "imagebench.log");
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
                : "A red apple on a white table, soft daylight, studio photograph";
            for (int i = 0; i < 300 && _app.ModelLoad != AgentAppHost.ModelLoadState.Loaded; i++)
                await Task.Delay(1000, token).ConfigureAwait(false);
            if (_app.ModelLoad != AgentAppHost.ModelLoadState.Loaded || !_app.Chat.LoadedModelMakesImages)
            {
                Say($"imagebench FAIL no image model (state {_app.ModelLoad})");
                return;
            }
            Say($"imagebench model {_app.ModelService.LoadedModelName} on {_app.ModelService.LoadedBackend}; "
                + $"{runs} picture(s) of \"{prompt}\"; {ProcessMemory.Describe()}");
            for (int run = 1; run <= runs; run++)
                await PictureAsync(run, prompt, token).ConfigureAwait(false);
            Say("imagebench done");
        }
        catch (OperationCanceledException)
        {
            Say("imagebench cancelled");
        }
        catch (Exception ex)
        {
            Say("imagebench FAIL " + ex.Message);
        }
    }

    private async Task PictureAsync(int run, string prompt, CancellationToken token)
    {
        object created = _app.Chat.CreateSession();
        string session = created.GetType().GetProperty("sessionId")?.GetValue(created) as string ?? string.Empty;
        JsonElement body = JsonSerializer.SerializeToElement(new
        {
            sessionId = session,
            messages = new[] { new { role = "user", content = prompt } },
        });

        // The phone takes the GPU away from a backgrounded app; wait like a turn does.
        await _app.Compute.WaitAsync(token).ConfigureAwait(false);

        var clock = Stopwatch.StartNew();
        double firstStep = 0, lastStep = 0;
        int steps = 0;
        string loras = "none";
        string? url = null, error = null;
        await foreach (object frame in ImageTurns.StreamAsync(_app.Chat, body, _app.ImagePlanner, token, _app.PrepareImageTurn).ConfigureAwait(false))
        {
            int total = AgentAppHost.IntIn(frame, "image_steps");
            if (total > 0)
            {
                if (firstStep == 0) firstStep = clock.Elapsed.TotalSeconds;
                lastStep = clock.Elapsed.TotalSeconds;
                steps = total;
            }
            if (AgentAppHost.ValueIn(frame, "image_loras") is IEnumerable<string> names)
                loras = string.Join(" + ", names);
            url ??= AgentAppHost.StringIn(frame, "imageUrl");
            error ??= AgentAppHost.StringIn(frame, "error");
        }
        double seconds = clock.Elapsed.TotalSeconds;
        if (error is not null || url is null)
        {
            Say($"imagebench run {run} FAIL after {seconds:0.0}s: {error ?? "no picture"}");
            return;
        }
        string file = Path.Combine(_app.Options.UploadDirectory, url[(url.LastIndexOf('/') + 1)..]);
        string sha = File.Exists(file) ? Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file))) : "missing";
        // Between the first and the last step frame, not to the picture: the final decode and
        // save (about 5 s) spread over a few-step schedule's steps made a plug-in look 20%
        // dearer a step than the engine's own step timer measured.
        Say($"imagebench run {run}: {steps} steps with {loras}; first step at {firstStep:0.0}s, "
            + $"then {(lastStep - firstStep) / Math.Max(1, steps - 1):0.00}s a step (previews included), "
            + $"decode and save {seconds - lastStep:0.0}s, picture in {seconds:0.0}s; "
            + $"{url} sha256 {sha}; {ProcessMemory.Describe()}");
    }
}
