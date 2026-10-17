using System.Diagnostics;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioRenderTests : IAsyncLifetime
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-render-" + StudioJson.NewId());
    StudioService service = null!;
    public Task InitializeAsync() { Directory.CreateDirectory(root); service = new(root, options: new() { MinimumFreeDiskBytes = 0 }); return Task.CompletedTask; }
    public async Task DisposeAsync() { await service.DisposeAsync(); Directory.Delete(root, true); }
    static async Task Run(params string[] args)
    {
        using var p = new Process { StartInfo = new("ffmpeg") { UseShellExecute = false, RedirectStandardError = true, RedirectStandardOutput = true } };
        foreach (var a in args) p.StartInfo.ArgumentList.Add(a); p.Start();
        var stdout = p.StandardOutput.ReadToEndAsync(); var stderr = p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync(); await stdout; Assert.True(p.ExitCode == 0, await stderr);
    }
    async Task<StudioProject> Import(StudioProject project, string color, bool audio)
    {
        var path = Path.Combine(root, color + ".mp4");
        var args = new List<string> { "-v", "error", "-f", "lavfi", "-i", $"color=c={color}:s=160x96:r=24:d=2" };
        if (audio) args.AddRange(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2"]);
        args.AddRange(["-c:v", "mpeg4", "-c:a", "aac", "-t", "2", path]); await Run(args.ToArray());
        await using var stream = File.OpenRead(path); return await service.ImportAssetAsync(project.Id, stream, Path.GetFileName(path));
    }
    async Task<StudioJob> Complete(string projectId, string jobId)
    {
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        while (true)
        {
            var job = (await service.ListJobsAsync(projectId, stop.Token)).Single(j => j.Id == jobId);
            if (job.State is "completed" or "failed" or "cancelled") { Assert.True(job.State == "completed", job.Error); return job; }
            await Task.Delay(30, stop.Token);
        }
    }
    async Task<byte[]> Pixel(string path, double seconds, int x = 0, int y = 0)
    {
        var raw = Path.Combine(root, StudioJson.NewId() + ".rgb");
        await Run("-v", "error", "-ss", seconds.ToString(System.Globalization.CultureInfo.InvariantCulture), "-i", path, "-frames:v", "1", "-vf", $"crop=2:2:{x}:{y},scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", raw);
        return await File.ReadAllBytesAsync(raw);
    }
    async Task<float[]> AudioSamples(string output)
    {
        var raw = Path.Combine(root, StudioJson.NewId() + ".pcm");
        await Run("-v", "error", "-i", output, "-vn", "-ar", "48000", "-ac", "1", "-f", "f32le", raw);
        var bytes = await File.ReadAllBytesAsync(raw); var samples = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length); return samples;
    }
    static double Rms(float[] samples, double from, double to)
    {
        var slice = samples.Skip((int)(from * 48000)).Take((int)((to - from) * 48000)).ToArray();
        return Math.Sqrt(slice.Average(v => v * (double)v));
    }
    [StudioMediaFact] public async Task RealMultitrackExportHasCorrectTimingPixelsAudioAndCapturedRevision()
    {
        var p = await service.CreateProjectAsync("Rendered fixture", 160, 96); p = await Import(p, "red", true); p = await Import(p, "blue", false);
        var overlayId = StudioJson.NewId(); var commands = new List<StudioCommand>
        {
            new() { Type = "insertClip", TrackId = p.Tracks[0].Id, AssetId = p.Assets[0].Id, DurationFrames = 48 },
            new() { Type = "addTrack", TrackId = overlayId, Name = "Overlay" },
            new() { Type = "insertClip", TrackId = overlayId, AssetId = p.Assets[1].Id, StartFrame = 24, DurationFrames = 24 },
            new() { Type = "addCaption", StartFrame = 6, DurationFrames = 36, Text = "HELLO 123" }
        };
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), commands);
        var job = await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId());
        // A concurrent edit must not change the captured export composition or dimensions.
        await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "settings", Width = 320, Height = 180 }]);
        job = await Complete(p.Id, job.Id); Assert.Equal(p.Revision, job.ProjectRevision);
        var output = service.GetJobOutputPath(p.Id, job.Id); var probe = await service.Media.ProbeAsync(output, "sample", default);
        Assert.Equal(160, probe.Width); Assert.Equal(96, probe.Height); Assert.InRange(probe.DurationSeconds, 1.99, 2.06); Assert.True(probe.HasAudio);
        var first = await Pixel(output, .25); Assert.True(first[0] > 200 && first[2] < 40, string.Join(",", first));
        var second = await Pixel(output, 1.25); Assert.True(second[2] > 200 && second[0] < 40, string.Join(",", second));
        var captionBefore = await Pixel(output, .1, 52, 66); var captionDuring = await Pixel(output, .5, 52, 66);
        Assert.True(captionBefore[1] < 40, string.Join(",", captionBefore));
        Assert.True(captionDuring[1] > 70, "Caption glyph must affect the actual exported pixels: " + string.Join(",", captionDuring));
        var audio = await AudioSamples(output); Assert.InRange(Rms(audio, .25, .5), .01, .3);
        var other = await service.CreateProjectAsync(); Assert.Throws<StudioException>(() => service.GetJobOutputPath(other.Id, job.Id));
    }
    [StudioMediaFact] public async Task FadesAreCompositedAndTrimmedSourceCanReopenUndoWithoutDeletingMedia()
    {
        var p = await service.CreateProjectAsync("Fade", 160, 96); p = await Import(p, "white", true); var asset = p.Assets.Single(); var clipId = StudioJson.NewId();
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", ClipId = clipId, TrackId = p.Tracks[0].Id, AssetId = asset.Id, DurationFrames = 24, SourceStartFrame = 12, FadeInFrames = 12, FadeOutFrames = 12 }]);
        var job = await Complete(p.Id, (await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId())).Id); var output = service.GetJobOutputPath(p.Id, job.Id);
        var first = await Pixel(output, 0); var middle = await Pixel(output, .5); Assert.True(first[0] < 30); Assert.True(middle[0] > 200);
        var audio = await AudioSamples(output);
        Assert.True(Rms(audio, .01, .03) < Rms(audio, .45, .49) * .2, "Audio fade-in must match the video fade.");
        Assert.True(Rms(audio, .97, .99) < Rms(audio, .45, .49) * .2, "Audio fade-out must match the video fade.");
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.Empty(p.Clips); Assert.True(File.Exists(service.GetAssetPath(p.Id, asset.Id)));
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId(), redo: true); Assert.Single(p.Clips); Assert.Single(p.Assets);
    }
    [StudioMediaFact] public async Task MissingMediaIsReportedAndRelinkRequiresOriginalBytes()
    {
        var p = await service.CreateProjectAsync(); p = await Import(p, "green", false); var asset = p.Assets.Single(); var path = service.GetAssetPath(p.Id, asset.Id); var bytes = await File.ReadAllBytesAsync(path); File.Delete(path);
        Assert.True((await service.GetProjectAsync(p.Id)).Assets.Single().Missing);
        await Assert.ThrowsAsync<StudioException>(() => service.RelinkAssetAsync(p.Id, asset.Id, new MemoryStream([1, 2, 3])));
        var relinked = await service.RelinkAssetAsync(p.Id, asset.Id, new MemoryStream(bytes)); Assert.False(relinked.Assets.Single().Missing); Assert.Equal(bytes, await File.ReadAllBytesAsync(path));
    }
    sealed class FileGenerationDouble(string source) : IStudioGenerationAdapter
    {
        public int Calls;
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>([new() { ModelId = "file-fixture-double", Available = true, TextToVideo = true, DefaultFrameCount = 49 }]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) => Task.CompletedTask;
        public Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string dir, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested(); Calls++; var path = Path.Combine(dir, "fixture-generated.mp4"); File.Copy(source, path);
            return Task.FromResult(new StudioGenerationResult(path, "TEST DOUBLE; no model inference", new() { ["seed"] = "42", ["frames"] = "49" }));
        }
    }
    sealed class QueuedAssetGenerationDouble(string source) : IStudioGenerationAdapter
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public int Calls;
        public bool ReadRetainedConditioning;
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>(
            [new() { ModelId = "queued-asset-fixture", Available = true, TextToVideo = true, ImageConditioning = true, DefaultFrameCount = 49 }]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) => Task.CompletedTask;
        public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string dir, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            if (++Calls == 1) { Started.TrySetResult(); await Release.Task.WaitAsync(ct); }
            if (r.ConditioningAssetId != null) ReadRetainedConditioning = (await File.ReadAllBytesAsync(assets[r.ConditioningAssetId], ct)).Length > 0;
            var output = Path.Combine(dir, "fixture-generated.mp4"); File.Copy(source, output);
            return new(output, "TEST DOUBLE; no model inference", new());
        }
    }
    [StudioMediaFact] public async Task DeletingAssetsKeepsQueuedRenderAndGenerationSnapshotsValidAndPreservesNewOutputsOnUndo()
    {
        var video = Path.Combine(root, "queued-yellow.mp4"); var image = Path.Combine(root, "queued-guide.png");
        await Run("-v", "error", "-f", "lavfi", "-i", "color=c=yellow:s=64x64:r=24:d=3", "-frames:v", "49", "-c:v", "mpeg4", video);
        await Run("-v", "error", "-i", video, "-frames:v", "1", image);
        await service.DisposeAsync(); var adapter = new QueuedAssetGenerationDouble(video); service = new(root, adapter);
        var p = await service.CreateProjectAsync("Delete during queued work", 64, 64);
        await using (var input = File.OpenRead(video)) p = await service.ImportAssetAsync(p.Id, input, "source.mp4");
        await using (var input = File.OpenRead(image)) p = await service.ImportAssetAsync(p.Id, input, "guidance.png");
        var sourceAsset = p.Assets[0]; var guideAsset = p.Assets[1]; var sourcePath = service.GetAssetPath(p.Id, sourceAsset.Id); var guidePath = service.GetAssetPath(p.Id, guideAsset.Id);
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", AssetId = sourceAsset.Id, TrackId = p.Tracks[0].Id, DurationFrames = 49 }]);
        var first = await service.StartGenerationAsync(new() { ProjectId = p.Id, ModelId = "queued-asset-fixture", Prompt = "Queue blocker", Width = 64, Height = 64 });
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var conditioningRequest = new StudioGenerationRequest { ProjectId = p.Id, ModelId = "queued-asset-fixture", Prompt = "Retained guide", Width = 64, Height = 64, ConditioningAssetId = guideAsset.Id };
        var conditioned = await service.StartGenerationAsync(conditioningRequest);
        var render = await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId());
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [
            new() { Type = "deleteAsset", AssetId = sourceAsset.Id, RemoveTimelineClips = true },
            new() { Type = "deleteAsset", AssetId = guideAsset.Id }]);
        Assert.Empty(p.Assets); Assert.Empty(p.Clips); Assert.True(File.Exists(sourcePath)); Assert.True(File.Exists(guidePath));
        Assert.Equal("asset_not_found", Assert.Throws<StudioException>(() => service.GetAssetPath(p.Id, guideAsset.Id)).Code);
        Assert.Equal(conditioned.Id, (await service.StartGenerationAsync(conditioningRequest)).Id);
        var freshRequest = StudioJson.Clone(conditioningRequest); freshRequest.RequestId = StudioJson.NewId();
        Assert.Equal("asset_not_found", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(freshRequest))).Code);
        adapter.Release.TrySetResult(); await Complete(p.Id, first.Id); await Complete(p.Id, conditioned.Id); await Complete(p.Id, render.Id);
        Assert.True(adapter.ReadRetainedConditioning); Assert.Equal(2, adapter.Calls);
        var output = service.GetJobOutputPath(p.Id, render.Id); var metadata = await service.Media.ProbeAsync(output, "captured", default);
        Assert.Equal(64, metadata.Width); Assert.InRange(metadata.DurationSeconds, 49d / 24, 49d / 24 + .05);
        var pixel = await Pixel(output, .5); Assert.True(pixel[0] > 200 && pixel[1] > 200 && pixel[2] < 40);
        p = await service.GetProjectAsync(p.Id); var generatedIds = p.Assets.Select(a => a.Id).ToArray(); Assert.Equal(2, generatedIds.Length);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId());
        Assert.Equal(new[] { sourceAsset.Id, guideAsset.Id }.Concat(generatedIds), p.Assets.Select(a => a.Id)); Assert.Single(p.Clips);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId(), redo: true); Assert.Equal(generatedIds, p.Assets.Select(a => a.Id)); Assert.Empty(p.Clips);
    }
    [StudioMediaFact] public async Task GeneratedAssetProvenancePersistsAndUndoRedoNeverRerunsTheAdapter()
    {
        var fixture = Path.Combine(root, "generation-fixture.mp4");
        await Run("-v", "error", "-f", "lavfi", "-i", "color=c=yellow:s=64x64:r=24:d=3", "-frames:v", "49", "-c:v", "mpeg4", fixture);
        await service.DisposeAsync(); var adapter = new FileGenerationDouble(fixture); service = new(root, adapter);
        var p = await service.CreateProjectAsync("Generation adapter lifecycle"); var request = new StudioGenerationRequest { ProjectId = p.Id, ModelId = "file-fixture-double", Width = 64, Height = 64, Prompt = "Test fixture, not real inference", Seed = 42 };
        var job = await Complete(p.Id, (await service.StartGenerationAsync(request)).Id);
        p = await service.GetProjectAsync(p.Id); var asset = Assert.Single(p.Assets);
        Assert.Equal(job.OutputAssetId, asset.Id); Assert.Equal("42", asset.Generation!.EffectiveParameters["seed"]); Assert.Equal(49, asset.Generation.Request.FrameCount);
        Assert.Equal("TEST DOUBLE; no model inference", asset.Generation.ModelVersion);
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", AssetId = asset.Id, TrackId = p.Tracks[0].Id, DurationFrames = 49 }]);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.Empty(p.Clips); Assert.Single(p.Assets);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId(), redo: true); Assert.Single(p.Clips); Assert.Equal(1, adapter.Calls);
        Assert.Equal(job.Id, (await service.StartGenerationAsync(request)).Id); Assert.Equal(1, adapter.Calls);
        await service.DisposeAsync();
        // Simulate a process exit after asset commit but before terminal job bookkeeping.
        var recordPath = Path.Combine(root, "jobs", job.Id, "job.json"); var record = System.Text.Json.Nodes.JsonNode.Parse(await File.ReadAllTextAsync(recordPath))!;
        record["job"]!["state"] = "running"; record["job"]!["outputAssetId"] = null; await File.WriteAllTextAsync(recordPath, record.ToJsonString());
        service = new(root, adapter); var recovered = Assert.Single(await service.ListJobsAsync(p.Id)); Assert.Equal("completed", recovered.State); Assert.Equal(asset.Id, recovered.OutputAssetId);
        Assert.Equal(1, adapter.Calls); Assert.Single((await service.GetProjectAsync(p.Id)).Clips);
    }
    [StudioMediaFact] public async Task AudioTracksMixAtRequestedVolumesWithoutImplicitNormalization()
    {
        var p = await service.CreateProjectAsync("Audio mix", 160, 96); p = await Import(p, "black", true); var a = p.Assets.Single(); var secondTrack = StudioJson.NewId();
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [
            new() { Type = "insertClip", TrackId = p.Tracks[0].Id, AssetId = a.Id, DurationFrames = 48, Volume = 0 },
            new() { Type = "insertClip", TrackId = p.Tracks[1].Id, AssetId = a.Id, DurationFrames = 48, Volume = .25 },
            new() { Type = "addTrack", TrackId = secondTrack, Kind = "audio", Name = "Second audio" },
            new() { Type = "insertClip", TrackId = secondTrack, AssetId = a.Id, DurationFrames = 48, Volume = .25 }]);
        var job = await Complete(p.Id, (await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId())).Id);
        var original = await AudioSamples(service.GetAssetPath(p.Id, a.Id)); var mixed = await AudioSamples(service.GetJobOutputPath(p.Id, job.Id));
        var ratio = Rms(mixed, .2, 1.8) / Rms(original, .2, 1.8); Assert.InRange(ratio, .47, .56);
    }
    [StudioMediaFact] public async Task OverlappingDissolveProducesNoBlackDipAndComplementaryAudioGains()
    {
        var p = await service.CreateProjectAsync("Dissolve", 160, 96); p = await Import(p, "red", true); p = await Import(p, "blue", true);
        var outgoing = StudioJson.NewId(); var incoming = StudioJson.NewId();
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [
            new() { Type = "insertClip", ClipId = outgoing, TrackId = p.Tracks[0].Id, AssetId = p.Assets[0].Id, DurationFrames = 48 },
            new() { Type = "insertClip", ClipId = incoming, TrackId = p.Tracks[0].Id, AssetId = p.Assets[1].Id, StartFrame = 24, DurationFrames = 48 }]);
        var originalRevision = p.Revision;
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "dissolve", ClipId = incoming, OtherClipId = outgoing }]);
        Assert.Equal(72, p.DurationFrames); Assert.Equal(24, p.Clips.Single(c => c.Id == incoming).StartFrame);
        var job = await Complete(p.Id, (await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId())).Id); var output = service.GetJobOutputPath(p.Id, job.Id);
        var middle = await Pixel(output, 1.5); Assert.InRange(middle[0], (byte)105, (byte)145); Assert.InRange(middle[2], (byte)105, (byte)145);
        foreach (var time in new[] { 1.25, 1.5, 1.75 }) { var pixel = await Pixel(output, time); Assert.InRange(pixel[0] + pixel[2], 225, 270); }
        var original = await AudioSamples(service.GetAssetPath(p.Id, p.Assets[0].Id)); var mixed = await AudioSamples(output);
        Assert.InRange(Rms(mixed, 1.2, 1.8) / Rms(original, .2, .8), .9, 1.1);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.All(p.Clips, c => { Assert.Equal(0, c.FadeInFrames); Assert.Equal(0, c.AudioFadeOutFrames); });
        Assert.Equal(originalRevision + 2, p.Revision); Assert.Equal(72, p.DurationFrames);
    }
    [StudioMediaFact] public async Task FractionalOutputRateUsesTheRationalClockAndActualDecodedDuration()
    {
        var p = await service.CreateProjectAsync("Fractional rate", 160, 96, 30000, 1001); p = await Import(p, "orange", true);
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", TrackId = p.Tracks[0].Id, AssetId = p.Assets[0].Id, StartFrame = 15, DurationFrames = 45 }]);
        var job = await Complete(p.Id, (await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId())).Id);
        var output = service.GetJobOutputPath(p.Id, job.Id); var metadata = await service.Media.ProbeAsync(output, "fractional", default);
        Assert.Equal("30000/1001", metadata.SourceFrameRate); Assert.InRange(metadata.DurationSeconds, 2.001, 2.03);
        var gap = await Pixel(output, .2); Assert.All(gap, channel => Assert.InRange(channel, (byte)0, (byte)5));
        var visible = await Pixel(output, .8); Assert.True(visible[0] > 200 && visible[1] > 100);
        var audio = await AudioSamples(output); Assert.True(Rms(audio, .1, .4) < .001); Assert.True(Rms(audio, .7, 1.2) > .01);
    }
    static async Task<double[]> FrameTimestamps(string path)
    {
        using var process = new Process { StartInfo = new("ffprobe") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true } };
        foreach (var argument in new[] { "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path }) process.StartInfo.ArgumentList.Add(argument);
        process.Start(); var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync(); await process.WaitForExitAsync(); Assert.True(process.ExitCode == 0, await stderr);
        using var json = System.Text.Json.JsonDocument.Parse(await stdout);
        return json.RootElement.GetProperty("frames").EnumerateArray().Select(f => double.Parse(f.GetProperty("best_effort_timestamp_time").GetString()!, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
    }
    [StudioMediaFact] public async Task VariableFrameRateTrimUsesSourcePresentationTimeAndKeepsAudioSynchronized()
    {
        foreach (var color in new[] { "red", "blue", "green" })
            await Run("-v", "error", "-f", "lavfi", "-i", $"color=c={color}:s=160x96:r=1:d=1", "-frames:v", "1", Path.Combine(root, "vfr-" + color + ".png"));
        var list = Path.Combine(root, "variable.ffconcat");
        await File.WriteAllTextAsync(list, "ffconcat version 1.0\nfile 'vfr-red.png'\nduration 0.4\nfile 'vfr-blue.png'\nduration 0.6\nfile 'vfr-green.png'\nduration 1.0\nfile 'vfr-green.png'\n");
        var source = Path.Combine(root, "variable.mp4");
        await Run("-v", "error", "-f", "concat", "-safe", "1", "-i", list, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2.04", "-af", "volume=0:enable='lt(t,0.4)'", "-fps_mode", "vfr", "-c:v", "mpeg4", "-c:a", "aac", "-t", "2.04", source);
        var sourcePts = await FrameTimestamps(source);
        Assert.Equal(4, sourcePts.Length);
        Assert.InRange(sourcePts[1] - sourcePts[0], .399, .401);
        Assert.InRange(sourcePts[2] - sourcePts[1], .599, .601);
        Assert.InRange(sourcePts[3] - sourcePts[2], .999, 1.001);
        var p = await service.CreateProjectAsync("True VFR source trim", 160, 96);
        await using (var input = File.OpenRead(source)) p = await service.ImportAssetAsync(p.Id, input, "variable.mp4");
        // Trim starts in the middle of the first frame's 400 ms presentation interval.
        // Dropping frames before source t=.25 then resetting PTS would incorrectly start blue.
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", TrackId = p.Tracks[0].Id, AssetId = p.Assets[0].Id, SourceStartFrame = 6, StartFrame = 12, DurationFrames = 36 }]);
        var job = await Complete(p.Id, (await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId())).Id); var output = service.GetJobOutputPath(p.Id, job.Id);
        var outputPts = await FrameTimestamps(output); Assert.Equal(48, outputPts.Length);
        Assert.InRange(outputPts[^1], 47d / 24 - .0001, 47d / 24 + .0001);
        var red = await Pixel(output, .55); Assert.True(red[0] > 200 && red[2] < 40, "VFR trim must retain the source frame covering its in-point: " + string.Join(",", red));
        var blue = await Pixel(output, .8); Assert.True(blue[2] > 200 && blue[0] < 40);
        var green = await Pixel(output, 1.4); Assert.True(green[1] > 90 && green[0] < 40 && green[2] < 40);
        var audio = await AudioSamples(output); Assert.True(Rms(audio, .53, .60) < .001); Assert.True(Rms(audio, .73, .9) > .01);
        // The resampled visual boundary is frame 16 (2/3 s); the audio onset remains
        // source .4 - source trim .25 + placement .5 = .65 s within one output frame.
        var beforeBlue = await Pixel(output, 15d / 24); var firstBlue = await Pixel(output, 16d / 24);
        Assert.True(beforeBlue[0] > 200); Assert.True(firstBlue[2] > 200);
    }
    [StudioMediaFact] public async Task CorruptMediaFailsRenderDurablyWithoutChangingTheProject()
    {
        var p = await service.CreateProjectAsync("Failed render", 160, 96); p = await Import(p, "purple", false);
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "insertClip", AssetId = p.Assets[0].Id, TrackId = p.Tracks[0].Id, DurationFrames = 24 }]);
        await File.WriteAllBytesAsync(service.GetAssetPath(p.Id, p.Assets[0].Id), [1, 2, 3]);
        var job = await service.StartRenderAsync(p.Id, p.Revision, StudioJson.NewId());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        do { await Task.Delay(20, timeout.Token); job = (await service.ListJobsAsync(p.Id)).Single(j => j.Id == job.Id); } while (job.State is "queued" or "running");
        Assert.Equal("failed", job.State); Assert.NotEmpty(job.Error!); Assert.Equal(p.Revision, (await service.GetProjectAsync(p.Id)).Revision);
        Assert.Throws<StudioException>(() => service.GetJobOutputPath(p.Id, job.Id));
        await service.DisposeAsync(); service = new(root); Assert.Equal("failed", Assert.Single(await service.ListJobsAsync(p.Id)).State);
    }
    [Fact] public async Task UnsupportedPlatformGatesImportAndExportWithoutFakingSuccess()
    {
        await using var limited = new StudioService(Path.Combine(root, "limited"), options: new() { EnableExternalProcesses = false }); var p = await limited.CreateProjectAsync();
        using var stream = new MemoryStream([1, 2, 3]); Assert.Equal("unsupported_platform", (await Assert.ThrowsAsync<StudioException>(() => limited.ImportAssetAsync(p.Id, stream, "test.mp4"))).Code);
        Assert.Empty((await limited.GetProjectAsync(p.Id)).Assets);
        Assert.False((await limited.Media.GetCapabilitiesAsync()).Available);
    }
}
