// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3-Omni media-input probe (TensorSharp side of eng/lfm3/omni-in's llama.cpp comparisons; outputs belong in ignored
// artifacts/):
//   plan   VIDEO                         the clip as the model reads it: the decoder's probe, the 2 fps plan (frames,
//                                        size, pairs and labels), the soundtrack's length, its trimmed span and the rows
//                                        each pair's piece takes (llama.cpp's split)
//   split  MODEL MMPROJ BACKEND IMAGE    an image's encode as one program and split for yields (2 and 4 programs),
//                                        interleaved, seven times each: median ms and the rows' distance
//   stages MODEL MMPROJ BACKEND VIDEO     a video file's preparation stage by stage (decode + resize, soundtrack decode,
//                                        features, pair encodes, soundtrack encode), three times
//   layout MODEL MMPROJ BACKEND SYSTEM TEXT KIND=PATH...
//                                        one user turn (KIND image | audio | video) rendered by the GGUF template and
//                                        expanded by the model: prompt tokens before / after, every media span, and the
//                                        preparation (decode + encode) time, twice (the second hits the embedding cache);
//                                        then the prompt's prefill on the model with its media rows (one forward and in
//                                        1024-token chunks) against a text prompt of the same length, three times each
// BACKEND: ggml_metal | ggml_cpu | cpu | mlx | ggml_cuda | cuda
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

if (args.Length < 2) return Usage();
switch (args[0])
{
    case "plan":
        return Plan(args[1]);
    case "split" when args.Length >= 5:
        return Split(args[1], args[2], Backend(args[3]), args[4]);
    case "stages" when args.Length >= 5:
        return Stages(args[1], args[2], Backend(args[3]), args[4]);
    case "layout" when args.Length >= 6:
        return Layout(args[1], args[2], Backend(args[3]), args[4], args[5], args.Skip(6).ToArray());
    default:
        return Usage();
}

static BackendType Backend(string name) => name switch
{
    "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "mlx" => BackendType.Mlx,
    "ggml_cuda" => BackendType.GgmlCuda, "cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("unknown backend " + name),
};

static int Usage()
{
    Console.Error.WriteLine("usage: OmniInProbe plan VIDEO | layout MODEL MMPROJ BACKEND SYSTEM TEXT KIND=PATH...");
    return 2;
}

static int Plan(string video)
{
    VideoInfo info = MediaCodecs.Video.Probe(video);
    Console.WriteLine($"probe: {info.FrameCount} frames at {info.Fps:R} fps, {info.Width} x {info.Height} ({MediaCodecs.Video.GetType().Name})");
    var sw = Stopwatch.StartNew();
    (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(video, Lfm3Video.MaxFramesOmni);
    Console.WriteLine($"read: {frames.Length} frames in {sw.Elapsed.TotalMilliseconds:F0} ms; plan over {plan.SourceFrames} frames at {plan.SourceFps:R} fps: " +
        $"indices [{string.Join(", ", plan.FrameIndices)}], {plan.Width} x {plan.Height}, {plan.Groups.Count} pairs of {plan.RowsPerGroup} rows");
    Console.WriteLine("labels: " + string.Join(" ", plan.Groups.Select(g => g.Label)));
    float[] track;
    try { track = Lfm3AudioPreprocessor.LoadAudio(video); }
    catch (Exception ex) { Console.WriteLine($"soundtrack: none ({ex.Message})"); return 0; }
    var pieces = Lfm3AudioPrompt.SplitSoundtrack(plan.FrameIndices, plan.SourceFps, plan.SourceFrames, track.Length, out long trimmed);
    Console.WriteLine($"soundtrack: {track.Length} samples at 16 kHz ({track.Length / 16000.0:F3} s), trimmed to {trimmed} " +
        $"-> {Lfm3AudioPreprocessor.RowCountForSamples(trimmed)} rows; pieces (group: samples, rows): " +
        string.Join(" ", pieces.Select(p => $"{p.Group}:[{p.SampleStart},{p.SampleEnd})={p.Rows}")));
    return 0;
}

static int Layout(string modelPath, string mmproj, BackendType backend, string system, string text, string[] media)
{
    using ModelBase model = ModelBase.Create(modelPath, backend);
    model.MultimodalInjector.LoadProjectors(mmproj);
    var message = new ChatMessage { Role = "user", Content = text };
    foreach (string m in media)
    {
        int eq = m.IndexOf('=');
        string kind = m[..eq], path = Path.GetFullPath(m[(eq + 1)..]);
        switch (kind)
        {
            case "image": (message.ImagePaths ??= new()).Add(path); break;
            case "audio": (message.AudioPaths ??= new()).Add(path); break;
            case "video": (message.VideoFilePaths ??= new()).Add(path); message.IsVideo = true; break;
            default: throw new ArgumentException("media kind " + kind);
        }
    }
    var history = new List<ChatMessage>();
    if (!string.IsNullOrEmpty(system)) history.Add(new ChatMessage { Role = "system", Content = system });
    history.Add(message);
    string prompt = ChatTemplate.RenderFromGgufTemplate(model.Config.ChatTemplate, history, addGenerationPrompt: true, architecture: "lfm3");
    List<int> tokens = model.Tokenizer.Encode(prompt, addSpecial: true);
    for (int run = 0; run < 2; run++)
    {
        var sw = Stopwatch.StartNew();
        List<int> expanded;
        IReadOnlyList<PromptMediaSpan> spans;
        lock (model.GpuComputeLock)
        {
            expanded = model.MultimodalInjector.ProcessPromptTokens(history, new List<int>(tokens), "probe");
            spans = model.MultimodalInjector.GetPreparedMediaSpans("probe");
        }
        double ms = sw.Elapsed.TotalMilliseconds;
        model.MultimodalInjector.ClearPreparedPromptState("probe");
        Console.WriteLine($"run {run}: {tokens.Count} -> {expanded.Count} prompt tokens, {spans.Count} media spans, prepared in {ms:F1} ms");
        if (run == 0)
            foreach (PromptMediaSpan s in spans)
                Console.WriteLine($"  [{s.Start}, {s.End}) {s.End - s.Start} tokens  {s.ContentId[..Math.Min(48, s.ContentId.Length)]}");
        if (run == 1)
            Prefill(model, history, tokens, expanded);
    }
    return 0;
}

static int Split(string modelPath, string mmproj, BackendType backend, string imagePath)
{
    using ModelBase model = ModelBase.Create(modelPath, backend);
    model.MultimodalInjector.LoadProjectors(mmproj);
    Lfm3VisionEncoder vision = ((Lfm3RuntimeModel)model).VisionEncoder;
    Lfm3RgbImage image = Lfm3ImageProcessor.LoadImage(imagePath);
    (int w, int h) = vision.ImageSize(image.Width, image.Height);
    Lfm3RgbImage frame = Lfm3ImageProcessor.Resize(image, w, h);
    int patches = w / 16 * (h / 16);
    var times = new Dictionary<int, List<double>> { [1] = new(), [2] = new(), [4] = new() };
    var rows = new Dictionary<int, float[]>();
    foreach (int parts in times.Keys) { vision.YieldPatches = parts == 1 ? 0 : (patches + parts - 1) / parts; rows[parts] = vision.EncodeFrame(frame, () => { }).Rows; }
    for (int rep = 0; rep < 7; rep++)
        foreach (int parts in times.Keys)
        {
            vision.YieldPatches = parts == 1 ? 0 : (patches + parts - 1) / parts;
            var sw = Stopwatch.StartNew();
            vision.EncodeFrame(frame, () => { });
            times[parts].Add(sw.Elapsed.TotalMilliseconds);
        }
    foreach (int parts in times.Keys)
    {
        double num = 0, den = 0;
        for (int i = 0; i < rows[1].Length; i++) { double d = rows[parts][i] - rows[1][i]; num += d * d; den += (double)rows[1][i] * rows[1][i]; }
        var t = times[parts].OrderBy(x => x).ToList();
        Console.WriteLine($"{w}x{h} ({patches} patches) in {parts} program(s): median {t[t.Count / 2]:F1} ms (min {t[0]:F1}), rel RMS vs one {Math.Sqrt(num / den):E2}");
    }
    return 0;
}

static int Stages(string modelPath, string mmproj, BackendType backend, string video)
{
    using ModelBase model = ModelBase.Create(modelPath, backend);
    model.MultimodalInjector.LoadProjectors(mmproj);
    var lfm3 = (Lfm3RuntimeModel)model;
    for (int rep = 0; rep < 3; rep++)
    {
        var sw = Stopwatch.StartNew();
        string dir = Path.Combine(Path.GetTempPath(), "omni-probe-" + Guid.NewGuid().ToString("N"));
        var (pngs, _) = MediaHelper.ExtractVideoFramesWithTimestamps(video, dir, "f", 16, 1.0);
        Directory.Delete(dir, true);
        Console.WriteLine($"rep {rep}: video_url's host sampling (16 frames at 1 fps, PNG) {sw.Elapsed.TotalMilliseconds:F0} ms for {pngs.Count} frames");
        sw.Restart();
        string hash = MediaContentId.OfBytes(File.ReadAllBytes(video));
        double tHash = sw.Elapsed.TotalMilliseconds; sw.Restart();
        (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(video, lfm3.VisionEncoder.MaxVideoFrames);
        double tRead = sw.Elapsed.TotalMilliseconds; sw.Restart();
        float[] track = Lfm3AudioPreprocessor.LoadAudio(video);
        double tAudio = sw.Elapsed.TotalMilliseconds; sw.Restart();
        var pieces = Lfm3AudioPrompt.SplitSoundtrack(plan.FrameIndices, plan.SourceFps, plan.SourceFrames, track.Length, out long trimmed);
        float[] features = Lfm3AudioPreprocessor.ComputeFeatures(track.AsSpan(0, (int)trimmed), out int n);
        double tFeatures = sw.Elapsed.TotalMilliseconds; sw.Restart();
        Lfm3VideoEncoding enc = lfm3.VisionEncoder.EncodeVideo(plan, frames);
        double tPairs = sw.Elapsed.TotalMilliseconds; sw.Restart();
        float[] rows = lfm3.AudioEncoder.EncodeFeatures(features, n);
        double tSound = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"rep {rep}: hash {tHash:F0} ms, decode+resize {frames.Length} frames {tRead:F0} ms, soundtrack decode {tAudio:F0} ms, " +
            $"features {tFeatures:F0} ms, {enc.Groups.Count} pair encodes {tPairs:F0} ms, soundtrack encode {rows.Length / lfm3.AudioEncoder.OutputWidth} rows {tSound:F0} ms; " +
            $"vision programs built {lfm3.VisionEncoder.Graph.Builds}, replayed {lfm3.VisionEncoder.Graph.Replays}, runs {lfm3.VisionEncoder.Graph.Runs}, {plan.Width}x{plan.Height}");
    }
    return 0;
}

static void Prefill(ModelBase model, List<ChatMessage> history, List<int> tokens, List<int> expanded)
{
    int[] text = expanded.Select(t => t >= 124000 ? 1000 + (t % 7) : t).ToArray();
    for (int rep = 0; rep < 3; rep++)
    {
        foreach (int chunk in new[] { expanded.Count, 1024 })
        {
            lock (model.GpuComputeLock)
            {
                model.MultimodalInjector.ProcessPromptTokens(history, new List<int>(tokens), "prefill");
                model.ResetKVCache();
                var sw = Stopwatch.StartNew();
                for (int start = 0; start < expanded.Count; start += chunk)
                {
                    int n = Math.Min(chunk, expanded.Count - start);
                    model.MultimodalInjector.QueuePromptEmbeddingsForSlice(start, n, "prefill");
                    model.Forward(expanded.GetRange(start, n).ToArray());
                }
                double media = sw.Elapsed.TotalMilliseconds;
                model.MultimodalInjector.ClearPreparedPromptState("prefill");
                model.ResetKVCache();
                sw.Restart();
                for (int start = 0; start < text.Length; start += chunk)
                    model.Forward(text.AsSpan(start, Math.Min(chunk, text.Length - start)).ToArray());
                double plain = sw.Elapsed.TotalMilliseconds;
                Console.WriteLine($"prefill {expanded.Count} tokens in {(chunk >= expanded.Count ? "one forward" : chunk + "-token chunks")}: " +
                    $"with media rows {media:F0} ms ({expanded.Count / media * 1000:F0} tok/s), text {plain:F0} ms ({text.Length / plain * 1000:F0} tok/s)");
            }
        }
    }
}
