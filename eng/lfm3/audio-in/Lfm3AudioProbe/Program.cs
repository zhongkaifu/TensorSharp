// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3-Omni audio input against llama.cpp (see eng/lfm3/audio-in/README.md):
//   signals OUT_DIR                                  the test clips as raw F32 PCM (the llama.cpp harness's input)
//   compare MMPROJ REF_DIR BACKEND REPS OUT_JSON     every value lfm3-audio-ref dumped (mel, downsampler, each layer's
//                                                    operator norm, final norm, rows) against TensorSharp's, and the time
//                                                    of both (BACKEND: ggml_metal | ggml_cpu | cpu | mlx)
//   rows MMPROJ CLIP BACKEND OUT.f32                 TensorSharp's rows of a clip (for lfm3-audio-ref asr ... ROWS.f32)
//   stream MMPROJ BACKEND STEP_FRAMES                live-stream rows (pushes of 0.1 s) against the whole clip's
//   fixture OUT_JSON MMPROJ=REF_DIR ...              the compact llama.cpp fixture the tests read
//   decode FILE LLAMA.f32                            LoadAudio of a resampled copy of ticket.wav against llama.cpp's decode
//                                                    of it (lfm3-audio-ref decode), both against the 16 kHz original
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class Program
{
    private static int Main(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        if (args.Length == 0) return Usage();
        switch (args[0])
        {
            case "signals" when args.Length == 2: return Signals(args[1]);
            case "compare" when args.Length == 6: return Compare(args[1], args[2], args[3], int.Parse(args[4]), args[5]);
            case "rows" when args.Length == 5: return Rows(args[1], args[2], args[3], args[4]);
            case "stream" when args.Length == 4: return Stream(args[1], args[2], int.Parse(args[3]));
            case "fixture" when args.Length >= 3: return Fixture(args[1], args.Skip(2).ToArray());
            case "frontend" when args.Length == 2: return FrontEnd(int.Parse(args[1]));
            case "shapes" when args.Length == 3: return Shapes(args[1], args[2]);
            case "decode" when args.Length == 3: return Decode(args[1], args[2]);
            default: return Usage();
        }
    }

    private static int Usage()
    {
        Console.Error.WriteLine("usage: Lfm3AudioProbe signals OUT_DIR | compare MMPROJ REF_DIR BACKEND REPS OUT_JSON | " +
            "rows MMPROJ CLIP BACKEND OUT.f32 | stream MMPROJ BACKEND STEP_FRAMES | fixture OUT_JSON MMPROJ=REF_DIR ...");
        return 2;
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("run from inside the repository (TensorSharp.slnx not found)");
    }

    private static int Signals(string outDir)
    {
        Directory.CreateDirectory(outDir);
        string root = RepoRoot();
        foreach (string name in Lfm3AudioSignals.Names)
        {
            float[] pcm = Lfm3AudioSignals.Make(name, root);
            Lfm3AudioSignals.WriteF32(Path.Combine(outDir, name + ".f32"), pcm);
            Console.WriteLine($"{name}: {pcm.Length} samples, sha256 {Lfm3AudioSignals.Hash(pcm)}");
        }
        return 0;
    }

    // ================================================================ backends

    private sealed class Backend : IDisposable
    {
        private readonly IDisposable _owner;

        public Backend(string name)
        {
            switch (name)
            {
                case "ggml_metal":
                    Allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
                    Type = BackendType.GgmlMetal;
                    break;
                case "ggml_cpu":
                    Allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
                    Type = BackendType.GgmlCpu;
                    break;
                case "cpu":
                    Allocator = new CpuAllocator(BlasEnum.DotNet);
                    Type = BackendType.Cpu;
                    break;
                case "mlx":
                    var mlx = new MlxAllocator();
                    Allocator = mlx;
                    _owner = mlx;
                    Type = BackendType.Mlx;
                    break;
                default: throw new ArgumentException($"backend '{name}' (ggml_metal, ggml_cpu, cpu, mlx)");
            }
        }

        public IAllocator Allocator { get; }
        public BackendType Type { get; }

        public void Dispose()
        {
            _owner?.Dispose();
            if (Allocator is GgmlAllocator)
            {
                // Release the programs' compute buffers and host-buffer bindings before exit (ggml-metal asserts on
                // residency sets left at process teardown).
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
        }
    }

    // ================================================================ compare

    private sealed record Metrics(double MaxAbs, double MaxRel, double MeanAbs, double RmsRel, double Cosine, double MinRowCosine, double RefMaxAbs)
    {
        public JsonObject Json() => new()
        {
            ["max_abs"] = MaxAbs, ["max_rel"] = MaxRel, ["mean_abs"] = MeanAbs, ["rms_rel"] = RmsRel, ["cosine"] = Cosine,
            ["min_row_cosine"] = MinRowCosine, ["ref_max_abs"] = RefMaxAbs,
        };

        public override string ToString() => $"max_abs {MaxAbs:G3} max_rel {MaxRel:G3} rms_rel {RmsRel:G3} cos {Cosine:F6} min_row_cos {MinRowCosine:F6}";
    }

    /// <summary>max |a - r|; max |a - r| / max(|r|, 1% of r's RMS); mean |a - r|; RMS(a - r) / RMS(r); cosine; the lowest
    /// cosine of a row of <paramref name="width"/>.</summary>
    private static Metrics Measure(float[] actual, float[] reference, int width)
    {
        if (actual.Length != reference.Length) throw new InvalidOperationException($"{actual.Length} values against {reference.Length}");
        double ss = 0, dd = 0, ab = 0, aa = 0, maxAbs = 0, sumAbs = 0, refMax = 0;
        foreach (float r in reference) ss += (double)r * r;
        double floor = 0.01 * Math.Sqrt(ss / Math.Max(1, reference.Length));
        double maxRel = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            dd += d * d;
            ab += (double)actual[i] * reference[i];
            aa += (double)actual[i] * actual[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            sumAbs += Math.Abs(d);
            refMax = Math.Max(refMax, Math.Abs(reference[i]));
            maxRel = Math.Max(maxRel, Math.Abs(d) / Math.Max(Math.Abs(reference[i]), floor == 0 ? 1e-30 : floor));
        }
        double minRow = 1;
        for (int row = 0; row + width <= reference.Length; row += width)
        {
            double rab = 0, raa = 0, rbb = 0;
            for (int i = row; i < row + width; i++)
            {
                rab += (double)actual[i] * reference[i];
                raa += (double)actual[i] * actual[i];
                rbb += (double)reference[i] * reference[i];
            }
            minRow = Math.Min(minRow, raa == 0 && rbb == 0 ? 1 : rab / Math.Sqrt(raa * rbb));
        }
        double cos = aa == 0 && ss == 0 ? 1 : ab / Math.Sqrt(aa * ss);
        return new Metrics(maxAbs, maxRel, sumAbs / reference.Length, ss == 0 ? Math.Sqrt(dd) : Math.Sqrt(dd / ss), cos, minRow, refMax);
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var v = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, v, 0, v.Length * 4);
        return v;
    }

    private static double Median(List<double> v)
    {
        v.Sort();
        return v[v.Count / 2];
    }

    private static int Compare(string mmproj, string refDir, string backendName, int reps, string outJson)
    {
        string root = RepoRoot();
        using var backend = new Backend(backendName);
        using var encoder = Lfm3AudioEncoder.Load(mmproj, backend.Allocator, backend.Type);
        var report = new JsonObject
        {
            ["mmproj"] = Path.GetFileName(mmproj),
            ["backend"] = backendName,
            ["executor"] = encoder.Executor.GetType().Name,
            ["weights"] = $"{encoder.Weights.Count} weights, {encoder.Weights.QuantizedBytes >> 20} MB blocks + {encoder.Weights.DenseBytes >> 20} MB F32",
        };
        var clips = new JsonObject();
        report["clips"] = clips;
        foreach (string name in Lfm3AudioSignals.Names)
        {
            string dir = Path.Combine(refDir, name);
            if (!Directory.Exists(dir)) continue;
            JsonNode meta = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
            float[] pcm = Lfm3AudioSignals.Make(name, root);
            if (pcm.Length != (int)meta["samples"]) throw new InvalidOperationException($"{name}: {pcm.Length} samples, the reference has {meta["samples"]}");
            var clip = new JsonObject { ["samples"] = pcm.Length };
            clips[name] = clip;

            // Front end.
            float[] features = Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
            clip["frames"] = frames;
            clip["mel"] = Measure(Lfm3AudioPreprocessor.ToMelMajor(features, frames), ReadF32(Path.Combine(dir, "mel.f32")), frames).Json();

            // Encoder, with the values llama.cpp names.
            var captured = new Dictionary<string, float[]>();
            float[] rows = encoder.EncodeFeatures(features, frames, captured);
            int nRows = Lfm3AudioPreprocessor.RowCount(frames);
            if (nRows != (int)meta["rows"]) throw new InvalidOperationException($"{name}: {nRows} rows, llama.cpp {meta["rows"]}");
            clip["rows"] = nRows;
            var layers = new JsonObject();
            clip["intermediates"] = layers;
            foreach ((string tensor, float[] values) in captured.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                string file = Path.Combine(dir, tensor + ".f32");
                if (!File.Exists(file)) continue;
                float[] expected = ReadF32(file);
                int width = tensor == "projected" ? encoder.OutputWidth : encoder.Width;
                if (tensor == "downsample.conv")
                {
                    // llama.cpp's [rows, mel, channel] (rows innermost) into this program's [channel, mel, rows].
                    int c = encoder.Channels, f = 16, n = values.Length / (c * f);
                    var laid = new float[expected.Length];
                    for (int t = 0; t < n; t++)
                        for (int mel = 0; mel < f; mel++)
                            for (int ch = 0; ch < c; ch++)
                                laid[ch + c * (mel + f * t)] = expected[t + n * (mel + f * ch)];
                    expected = laid;
                    width = c * f;
                }
                layers[tensor] = Measure(values, expected, width).Json();
            }
            Metrics m = Measure(rows, ReadF32(Path.Combine(dir, "rows.f32")), encoder.OutputWidth);
            clip["output"] = m.Json();

            // Time: the front end, then the encoder alone (a warm run first), medians.
            var fe = new List<double>();
            var enc = new List<double>();
            encoder.EncodeFeatures(features, frames);
            for (int r = 0; r < reps; r++)
            {
                var sw = Stopwatch.StartNew();
                Lfm3AudioPreprocessor.ComputeFeatures(pcm, out _);
                fe.Add(sw.Elapsed.TotalMilliseconds);
                sw.Restart();
                encoder.EncodeFeatures(features, frames);
                enc.Add(sw.Elapsed.TotalMilliseconds);
            }
            double feMs = Median(fe), encMs = Median(enc);
            clip["front_end_ms"] = feMs;
            clip["encode_ms"] = encMs;
            clip["llamacpp_encode_ms"] = (double)meta["encode_ms"];
            clip["llamacpp_mel_encode_ms"] = (double)meta["mel_encode_ms"];
            Console.WriteLine($"{name,-11} {frames,5} frames {nRows,4} rows | mel: {Measure(Lfm3AudioPreprocessor.ToMelMajor(features, frames), ReadF32(Path.Combine(dir, "mel.f32")), frames)}");
            Console.WriteLine($"{"",-11} rows: {m}");
            Console.WriteLine($"{"",-11} time: front end {feMs:F2} ms + encoder {encMs:F2} ms = {feMs + encMs:F2} ms; llama.cpp encoder {(double)meta["encode_ms"]:F2} ms, mel + encoder {(double)meta["mel_encode_ms"]:F2} ms");
        }
        report["graph"] = $"builds {encoder.Graph.Builds}, replays {encoder.Graph.Replays}, runs {encoder.Graph.Runs}";
        File.WriteAllText(outJson, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }

    private static int Rows(string mmproj, string clipName, string backendName, string outPath)
    {
        using var backend = new Backend(backendName);
        using var encoder = Lfm3AudioEncoder.Load(mmproj, backend.Allocator, backend.Type);
        float[] rows = encoder.Encode(Lfm3AudioSignals.Make(clipName, RepoRoot()));
        Lfm3AudioSignals.WriteF32(outPath, rows);
        Console.WriteLine($"{clipName}: {rows.Length / encoder.OutputWidth} rows -> {outPath}");
        return 0;
    }

    private static int Stream(string mmproj, string backendName, int stepFrames)
    {
        string root = RepoRoot();
        using var backend = new Backend(backendName);
        using var encoder = Lfm3AudioEncoder.Load(mmproj, backend.Allocator, backend.Type);
        foreach (string name in Lfm3AudioSignals.Names)
        {
            float[] pcm = Lfm3AudioSignals.Make(name, root);
            float[] whole = encoder.Encode(pcm);
            using var stream = encoder.OpenAudioStream(stepFrames);
            var streamed = new List<float>();
            var sw = Stopwatch.StartNew();
            for (int at = 0; at < pcm.Length; at += 1600)
                streamed.AddRange(stream.Push(pcm.AsSpan(at, Math.Min(1600, pcm.Length - at))));
            streamed.AddRange(stream.Finish());
            double ms = sw.Elapsed.TotalMilliseconds;
            Metrics m = Measure(streamed.ToArray(), whole, encoder.OutputWidth);
            Console.WriteLine($"{name,-11} step {stepFrames} frames: {stream.Rows} rows in {ms:F1} ms ({stream.Encoder.Graph.Runs} programs, " +
                $"{stream.Encoder.Graph.Builds} builds) vs whole clip: {m}");
        }
        return 0;
    }

    /// <summary>The front end's time per clip (median of REPS), against the old per-frame cost.</summary>
    private static int FrontEnd(int reps)
    {
        string root = RepoRoot();
        foreach (string name in Lfm3AudioSignals.Names)
        {
            float[] pcm = Lfm3AudioSignals.Make(name, root);
            Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
            var ms = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                var sw = Stopwatch.StartNew();
                Lfm3AudioPreprocessor.ComputeFeatures(pcm, out _);
                ms.Add(sw.Elapsed.TotalMilliseconds);
            }
            double med = Median(ms);
            // One thread: the stream computes each push's frames on the caller's thread (pushes of 256 frames).
            var seq = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                var stream = new Lfm3AudioFeatureStream();
                var sw = Stopwatch.StartNew();
                for (int at = 0; at < pcm.Length; at += 256 * 160)
                {
                    stream.Push(pcm.AsSpan(at, Math.Min(256 * 160, pcm.Length - at)));
                    stream.Read(out _);
                }
                stream.Finish(out _);
                seq.Add(sw.Elapsed.TotalMilliseconds);
            }
            double one = Median(seq);
            Console.WriteLine($"{name,-11} {frames,5} frames: whole clip {med:F2} ms median of {reps} (min {ms.Min():F2}); one thread " +
                $"{one:F2} ms (min {seq.Min():F2}, {1000 * seq.Min() / frames:F2} us/frame)");
        }
        return 0;
    }

    /// <summary>The cost of a clip length the encoder has not compiled a program for: twelve lengths, each encoded twice
    /// (the first run compiles), after a warm-up clip.</summary>
    private static int Shapes(string mmproj, string backendName)
    {
        using var backend = new Backend(backendName);
        using var encoder = Lfm3AudioEncoder.Load(mmproj, backend.Allocator, backend.Type);
        float[] speech = Lfm3AudioSignals.Loop(Lfm3AudioSignals.Speech(RepoRoot()), 30 * 16000);
        encoder.Encode(speech.AsSpan(0, 2 * 16000));
        foreach (int ms in new[] { 1000, 2370, 3100, 4730, 5500, 7210, 9000, 11330, 15000, 19870, 24000, 29990 })
        {
            float[] pcm = speech.AsSpan(0, ms * 16).ToArray();
            float[] features = Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
            int builds = encoder.Graph.Builds;
            var sw = Stopwatch.StartNew();
            encoder.EncodeFeatures(features, frames);
            double first = sw.Elapsed.TotalMilliseconds;
            var again = new List<double>();
            for (int r = 0; r < 3; r++)
            {
                sw.Restart();
                encoder.EncodeFeatures(features, frames);
                again.Add(sw.Elapsed.TotalMilliseconds);
            }
            Console.WriteLine($"{ms / 1000.0,6:F2} s ({frames,4} frames): first {first,7:F2} ms ({encoder.Graph.Builds - builds} build), replay median {Median(again),7:F2} ms");
        }
        return 0;
    }

    /// <summary>A copy of ticket.wav at another rate (and maybe stereo) decoded to 16 kHz by TensorSharp (LoadAudio: the
    /// band-limited resampler) and by llama.cpp (miniaudio's linear resampler), each against the original: the samples'
    /// relative RMS error and the log-mel features' differences.</summary>
    private static int Decode(string file, string llamaPcm)
    {
        float[] original = Lfm3AudioSignals.Speech(RepoRoot());
        float[] ours = Lfm3AudioPreprocessor.LoadAudio(file);
        Lfm3AudioSignals.WriteF32(llamaPcm + ".tensorsharp.f32", ours);   // for lfm3-audio-ref asr
        float[] theirs = ReadF32(llamaPcm);
        float[] reference = Lfm3AudioPreprocessor.ComputeFeatures(original, out int frames);
        foreach ((string who, float[] pcm) in new[] { ("TensorSharp", ours), ("llama.cpp", theirs) })
        {
            int n = Math.Min(pcm.Length, original.Length);
            double dd = 0, rr = 0;
            for (int i = 0; i < n; i++)
            {
                double d = pcm[i] - original[i];
                dd += d * d;
                rr += (double)original[i] * original[i];
            }
            float[] features = Lfm3AudioPreprocessor.ComputeFeatures(pcm.AsSpan(0, n), out int f);
            int common = Math.Min(f, frames);
            Metrics m = Measure(features.AsSpan(0, common * 128).ToArray(), reference.AsSpan(0, common * 128).ToArray(), 128);
            Console.WriteLine($"{Path.GetFileName(file)} {who,-11}: {pcm.Length} samples (original {original.Length}); samples rms rel err " +
                $"{Math.Sqrt(dd / rr):G3}; log-mel vs the original's: max abs {m.MaxAbs:G3}, mean abs {m.MeanAbs:G3}, cos {m.Cosine:F6}");
        }
        return 0;
    }

    // ================================================================ fixture

    /// <summary>The compact reference the tests read: per clip its hash, frame and row counts, three whole mel frames, each
    /// mel band's mean, and per mmproj three rows' first 128 values and every row's L2 norm.</summary>
    private static int Fixture(string outJson, string[] refs)
    {
        string root = RepoRoot();
        var clips = new JsonObject();
        var fixture = new JsonObject
        {
            ["source"] = "llama.cpp d3c3e242c libmtmd (eng/lfm3/audio-in/lfm3-audio-ref.cpp dump, ggml-metal), clips from InferenceWeb.Tests/Lfm3AudioSignals.cs",
            ["clips"] = clips,
        };
        static JsonArray Values(IEnumerable<float> v) => new(v.Select(x => (JsonNode)JsonValue.Create(float.Parse(x.ToString("G6")))).ToArray());
        bool first = true;
        foreach (string spec in refs)
        {
            string mmproj = spec[..spec.IndexOf('=')], refDir = spec[(spec.IndexOf('=') + 1)..];
            foreach (string name in Lfm3AudioSignals.Names)
            {
                string dir = Path.Combine(refDir, name);
                JsonNode meta = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
                int frames = (int)meta["frames"], rows = (int)meta["rows"], width = (int)meta["n_embd"];
                if (first || clips[name] == null)
                {
                    float[] pcm = Lfm3AudioSignals.Make(name, root);
                    float[] mel = ReadF32(Path.Combine(dir, "mel.f32"));   // mel-major
                    int[] idx = { 0, frames / 2, frames - 1 };
                    var band = new float[128];
                    for (int m = 0; m < 128; m++)
                    {
                        double s = 0;
                        for (int t = 0; t < frames; t++) s += mel[m * frames + t];
                        band[m] = (float)(s / frames);
                    }
                    clips[name] = new JsonObject
                    {
                        ["samples"] = pcm.Length,
                        ["sha256"] = Lfm3AudioSignals.Hash(pcm),
                        ["frames"] = frames,
                        ["rows"] = rows,
                        ["mel_frame_index"] = new JsonArray(idx.Distinct().Select(i => (JsonNode)i).ToArray()),
                        ["mel_frames"] = new JsonArray(idx.Distinct().Select(t => (JsonNode)Values(Enumerable.Range(0, 128).Select(m => mel[m * frames + t]))).ToArray()),
                        ["mel_band_mean"] = Values(band),
                        ["outputs"] = new JsonObject(),
                    };
                }
                float[] r = ReadF32(Path.Combine(dir, "rows.f32"));
                int[] ri = new[] { 0, rows / 2, rows - 1 }.Distinct().ToArray();
                var norms = Enumerable.Range(0, rows).Select(i =>
                {
                    double s = 0;
                    for (int j = 0; j < width; j++) s += (double)r[i * width + j] * r[i * width + j];
                    return (float)Math.Sqrt(s);
                });
                clips[name]["outputs"][Path.GetFileName(mmproj)] = new JsonObject
                {
                    ["row_index"] = new JsonArray(ri.Select(i => (JsonNode)i).ToArray()),
                    ["row_head"] = new JsonArray(ri.Select(i => (JsonNode)Values(r.Skip(i * width).Take(128))).ToArray()),
                    ["row_norm"] = Values(norms),
                };
            }
            first = false;
        }
        File.WriteAllText(outJson, fixture.ToJsonString(new JsonSerializerOptions { WriteIndented = false }) + "\n");
        Console.WriteLine($"wrote {outJson} ({new FileInfo(outJson).Length / 1024} KB)");
        return 0;
    }
}
