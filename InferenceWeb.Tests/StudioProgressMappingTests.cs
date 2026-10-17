using TensorSharp.Chat.Studio;
using TensorSharp.Models.Video;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

/// <summary>How a video model's progress reaches a Studio job. No model runs: the reports are the shapes the Wan,
/// MiniMax-H3 and LTX pipelines send.</summary>
public sealed class StudioProgressMappingTests
{
    sealed class Recording(Exception? fail = null) : IProgress<StudioProgress>
    {
        public readonly List<StudioProgress> Reports = [];
        public void Report(StudioProgress value) { if (fail != null) throw fail; Reports.Add(value); }
    }
    static StudioGenerationRequest Request() => new() { ModelId = "wan", Prompt = "A paper boat drifts down a gutter.", Width = 64, Height = 64, FrameCount = 17, Fps = 16, Steps = 4, CfgScale = 5 };

    [Fact]
    public void JobStepsCountOnlyFinishedStepsWhateverTheReportNames()
    {
        var progress = new Recording();
        var p = NativeStudioGenerationAdapter.Parameters(Request(), new Dictionary<string, string>(), progress, default);
        p.OnProgress(new VideoGenerationProgress { Phase = "text-encode", Step = 0, TotalSteps = 4, Detail = "UMT5-XXL" });
        // Wan names the step it is running, both when it enters a pass and in the heartbeat ticks during it.
        p.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4, Detail = "cond pass" });
        p.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4, Detail = "cond pass", Heartbeat = true });
        p.OnStep(1, 4);
        p.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4, Detail = "step done" });
        p.OnStep(4, 4);
        p.OnProgress(new VideoGenerationProgress { Phase = "vae-decode", Step = 4, TotalSteps = 4, Detail = "band 2/3" });
        p.OnProgress(new VideoGenerationProgress { Phase = "done", Step = 4, TotalSteps = 4 });

        Assert.Equal(new[] { "text-encode", "denoise", "denoise", "denoise", "vae-decode", "encoding" }, progress.Reports.Select(r => r.Stage));
        Assert.Equal(new[] { 0, 0, 0, 1, 4, 4 }, progress.Reports.Select(r => r.Step));
        Assert.All(progress.Reports, r => Assert.Equal(4, r.TotalSteps));
        Assert.Equal(new[] { "UMT5-XXL", "cond pass", "cond pass", "step done", "band 2/3", null }, progress.Reports.Select(r => r.Detail));
        Assert.Equal(new[] { .1, .1, .1, .3, .9, .9 }, progress.Reports.Select(r => Math.Round(r.Fraction, 6)));
    }

    [Fact]
    public void AHeartbeatNeverThrowsIntoItsTimerThreadButAPhaseReportStillRaises()
    {
        var full = new StudioException("insufficient_disk", "Disk is full.");
        var p = NativeStudioGenerationAdapter.Parameters(Request(), new Dictionary<string, string>(), new Recording(full), default);
        p.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4, Heartbeat = true });
        Assert.Same(full, Assert.Throws<StudioException>(() => p.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4 })));
        // Without a progress sink (a capability or parity query) the callbacks do nothing.
        var quiet = NativeStudioGenerationAdapter.Parameters(Request(), new Dictionary<string, string>(), null, default);
        quiet.OnStep(1, 4); quiet.OnProgress(new VideoGenerationProgress { Phase = "denoise", Step = 1, TotalSteps = 4 });
    }
}
