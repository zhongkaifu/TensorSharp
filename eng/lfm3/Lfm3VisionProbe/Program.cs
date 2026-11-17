// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp's side of the LFM3-Omni vision comparison with llama.cpp: runs Lfm3VisionEncoder over the cases of
// eng/lfm3/lfm3-vision-ref.cpp and writes the same files (<name>.json layout and times, <name>.pixels.f32 graph inputs,
// <name>.embd.f32 rows), which eng/lfm3/vision-compare.py diffs. A video case also writes <name>.frames/ (meta.txt and
// f<index>.rgb, the resized frames TensorSharp encoded) for the harness's frames: case, so llama.cpp can encode the very
// same pixels.
//
//   dotnet run --project eng/lfm3/Lfm3VisionProbe -c Release -- <mmproj> <out dir> [--backend ggml_metal|ggml_cpu|mlx|cpu]
//       [--reps N] case...
//   cases: synth:NAME:W:H:SEED | synthvideo:NAME:W:H:FRAMES:FPS | file:NAME:PATH | video:NAME:PATH
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Runtime;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length >= 3 && args[0] == "--frames")
        {
            // the decoder's own frames at their source size: Lfm3VisionProbe --frames <video> <out dir> <index>...
            Directory.CreateDirectory(args[2]);
            int[] wanted = args.Skip(3).Select(a => int.Parse(a, CultureInfo.InvariantCulture)).ToArray();
            int got = 0;
            DesktopMediaProvider.Instance.ReadFrames(args[1], wanted, (index, px, w, h, stride, layout) =>
            {
                File.WriteAllBytes(Path.Combine(args[2], $"src{index}.rgb"), Lfm3Video.ToRgb(px, w, h, stride, layout).Rgb);
                Console.WriteLine($"frame {index}: {w} x {h}");
                got++;
            });
            Console.WriteLine($"{got} of {wanted.Length} frames; probe {DesktopMediaProvider.Instance.Probe(args[1])}");
            return got == wanted.Length ? 0 : 1;
        }
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: Lfm3VisionProbe <mmproj> <out dir> [--backend ggml_metal|ggml_cpu|mlx|cpu] [--reps N] case...");
            return 2;
        }
        string mmproj = args[0], outDir = args[1], backendName = "ggml_metal";
        int reps = 5;
        var cases = new List<string>();
        for (int i = 2; i < args.Length; i++)
        {
            if (args[i] == "--backend") backendName = args[++i];
            else if (args[i] == "--reps") reps = Math.Max(1, int.Parse(args[++i], CultureInfo.InvariantCulture));
            else cases.Add(args[i]);
        }
        Directory.CreateDirectory(outDir);

        IAllocator allocator;
        BackendType backend;
        IDisposable owner = null;
        switch (backendName)
        {
            case "ggml_metal":
            case "ggml_cpu":
                bool metal = backendName == "ggml_metal";
                allocator = new GgmlAllocator(new GgmlContext(new[] { 0 }, metal ? GgmlBackendType.Metal : GgmlBackendType.Cpu), 0);
                backend = metal ? BackendType.GgmlMetal : BackendType.GgmlCpu;
                break;
            case "mlx":
                var mlx = new MlxAllocator();
                owner = mlx;
                allocator = mlx;
                backend = BackendType.Mlx;
                break;
            case "cpu":
                allocator = new CpuAllocator(BlasEnum.DotNet);
                backend = BackendType.Cpu;
                break;
            default:
                Console.Error.WriteLine($"unknown backend {backendName}");
                return 2;
        }

        using var encoder = new Lfm3VisionEncoder(mmproj, allocator, backend);
        Console.WriteLine($"lfm3 vision probe: {Path.GetFileName(mmproj)} on {backendName} ({encoder.Graph.Executor.GetType().Name})");
        int failures = 0;
        foreach (string spec in cases)
        {
            try
            {
                RunCase(encoder, spec, outDir, backendName, reps);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"{spec}: {ex}");
                failures++;
            }
        }
        owner?.Dispose();
        return failures == 0 ? 0 : 1;
    }

    private static void RunCase(Lfm3VisionEncoder encoder, string spec, string outDir, string backendName, int reps)
    {
        string[] p = spec.Split(':');
        string kind = p[0], name = p[1];
        string rest = spec.Substring(kind.Length + name.Length + 2);
        var pre = Stopwatch.StartNew();
        Lfm3VisionLayout layout;
        var blocks = new List<(Lfm3RgbImage First, Lfm3RgbImage Second)>();
        var extra = new StringBuilder();
        switch (kind)
        {
            case "synth":
            case "file":
            {
                Lfm3RgbImage src = kind == "synth"
                    ? Lfm3VisionSynthetic.Image(int.Parse(p[2]), int.Parse(p[3]), uint.Parse(p[4]))
                    : Lfm3ImageProcessor.LoadImage(rest);
                pre.Restart();
                (int w, int h) = encoder.ImageSize(src.Width, src.Height);
                Lfm3RgbImage frame = Lfm3ImageProcessor.Resize(src, w, h);
                layout = Lfm3VisionLayout.Image(Lfm3ImageProcessor.TokenCount(w, h));
                blocks.Add((frame, null));
                extra.Append($"\"source\":[{src.Width},{src.Height}],\"size\":[{w},{h}],");
                break;
            }
            case "synthvideo":
            case "video":
            {
                IVideoDecoder decoder = kind == "synthvideo"
                    ? new Lfm3SyntheticVideoDecoder(int.Parse(p[2]), int.Parse(p[3]), int.Parse(p[4]), double.Parse(p[5], CultureInfo.InvariantCulture))
                    : DesktopMediaProvider.Instance;
                string path = kind == "synthvideo" ? "synthetic.mp4" : rest;
                (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(path, encoder.MaxVideoFrames, decoder);
                layout = plan.Layout();
                foreach (Lfm3VideoGroup g in plan.Groups)
                    blocks.Add((frames[g.First], g.Second == g.First ? null : frames[g.Second]));
                extra.Append($"\"video\":{{\"source\":[{plan.SourceWidth},{plan.SourceHeight}],\"source_fps\":{plan.SourceFps.ToString("R", CultureInfo.InvariantCulture)}," +
                    $"\"source_frames\":{plan.SourceFrames},\"size\":[{plan.Width},{plan.Height}],\"frames\":[{string.Join(",", plan.FrameIndices)}]}},");
                if (kind == "video") WriteFrames(Path.Combine(outDir, name + ".frames"), plan, frames);
                break;
            }
            default:
                throw new ArgumentException($"case kind {kind}");
        }
        double preMs = pre.Elapsed.TotalMilliseconds;

        using var pixels = new BinaryWriter(File.Create(Path.Combine(outDir, name + ".pixels.f32")));
        using var embd = new BinaryWriter(File.Create(Path.Combine(outDir, name + ".embd.f32")));
        var chunkJson = new List<string>();
        var text = new StringBuilder();
        double total = 0, firstTotal = 0, hostTotal = 0;
        int nodes = 0;
        int block = 0;
        foreach (Lfm3VisionPiece piece in layout.Pieces)
        {
            if (piece.Kind == Lfm3VisionPieceKind.Token) { text.Append(piece.Token == Lfm3VisionEncoder.VisionStartToken ? "<|vision_start|>" : "<|vision_end|>"); continue; }
            if (piece.Kind == Lfm3VisionPieceKind.Text) { text.Append(piece.Text); continue; }
            chunkJson.Add(TextChunk(text.ToString()));
            text.Clear();
            (Lfm3RgbImage first, Lfm3RgbImage second) = blocks[block++];
            float[] input = Lfm3ImageProcessor.Normalize(first);
            if (second != null) input = input.Concat(Lfm3ImageProcessor.Normalize(second)).ToArray();
            foreach (float v in input) pixels.Write(v);
            var sw = Stopwatch.StartNew();
            Lfm3VisionEmbedding e = encoder.EncodePair(first, second);
            double firstMs = sw.Elapsed.TotalMilliseconds;
            var ms = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                sw.Restart();
                e = encoder.EncodePair(first, second);
                ms.Add(sw.Elapsed.TotalMilliseconds);
            }
            ms.Sort();
            double median = ms[ms.Count / 2];
            // the host share of a call: the frames' values and patches (EncodePair builds them every call)
            var host = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                sw.Restart();
                Lfm3ImageProcessor.Patchify(first);
                if (second != null) Lfm3ImageProcessor.Patchify(second);
                host.Add(sw.Elapsed.TotalMilliseconds);
            }
            host.Sort();
            hostTotal += host[host.Count / 2];
            nodes = Math.Max(nodes, encoder.Graph.LastNodeCount);
            total += median;
            firstTotal += firstMs;
            foreach (float v in e.Rows) embd.Write(v);
            chunkJson.Add($"{{\"type\":\"image\",\"rows\":{e.RowCount},\"pixels\":{input.Length},\"first_ms\":{F(firstMs)},\"encode_ms\":{F(median)}," +
                $"\"encode_ms_all\":[{string.Join(",", ms.Select(F))}]}}");
        }
        chunkJson.Add(TextChunk(text.ToString()));
        File.WriteAllText(Path.Combine(outDir, name + ".json"),
            $"{{\"name\":\"{name}\",\"case\":{JsonSerializer.Serialize(spec)},\"backend\":\"{backendName}\",{extra}\"preprocess_ms\":{F(preMs)}," +
            $"\"chunks\":[{string.Join(",", chunkJson)}],\"first_ms_total\":{F(firstTotal)},\"encode_ms_total\":{F(total)},\"host_ms_total\":{F(hostTotal)},\"nodes\":{nodes}}}\n");
        Console.WriteLine($"{name}: {layout.Blocks} blocks, {layout.RowCount} rows, preprocess {preMs:F1} ms, first {firstTotal:F1} ms, encode {total:F1} ms (median of {reps}; host values+patches {hostTotal:F1} ms; {nodes} nodes)");
    }

    private static string TextChunk(string text) => $"{{\"type\":\"text\",\"text\":{JsonSerializer.Serialize(text)}}}";

    private static string F(double v) => v.ToString("F3", CultureInfo.InvariantCulture);

    /// <summary>The resized frames, keyed by source frame index, for the harness's application decoder.</summary>
    private static void WriteFrames(string dir, Lfm3VideoPlan plan, Lfm3RgbImage[] frames)
    {
        Directory.CreateDirectory(dir);
        double duration = plan.SourceFrames / plan.SourceFps;
        File.WriteAllText(Path.Combine(dir, "meta.txt"), string.Join(" ", plan.SourceWidth, plan.SourceHeight,
            plan.SourceFps.ToString("R", CultureInfo.InvariantCulture), plan.SourceFrames, duration.ToString("R", CultureInfo.InvariantCulture),
            plan.Width, plan.Height) + "\n");
        for (int i = 0; i < frames.Length; i++)
            File.WriteAllBytes(Path.Combine(dir, $"f{plan.FrameIndices[i]}.rgb"), frames[i].Rgb);
    }
}
