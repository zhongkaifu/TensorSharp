// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record InputCase(string Id, int[] PromptTokens, int[] ContinuationTokens, int ChunkSize);
    private sealed record Capture(string CaseId, int Pass, string Kind, int Position, string File, string Sha256, int Elements, int Argmax, int HeadArgmax, bool PaddedRowWins, int NonfinitePadding, double WallMilliseconds);
    private static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileHash(string file) { using var stream = File.OpenRead(file); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private static object AssemblyId(Assembly assembly) => new { path = assembly.Location, sha256 = FileHash(assembly.Location) };
    private static void Save(string path, object value) { using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None); JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10); }
    private static void SaveCompleted(string directory, object value)
    {
        string path = Path.Combine(directory, "completed-run.json");
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None))
        {
            JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10); stream.Flush(true);
        }
        using var sidecar = new FileStream(path + ".sha256", FileMode.CreateNew, FileAccess.Write, FileShare.None);
        sidecar.Write(System.Text.Encoding.ASCII.GetBytes(FileHash(path) + "  completed-run.json\n"));
        sidecar.Flush(true);
    }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static void FreshOutput(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output path required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, Math.Max(0, parts.Length - 1)).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Output must be under ignored artifacts/ or docs/validation/.");
        Need(!Path.Exists(path) && !IsLink(path), "Fresh output directory required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) Need(!IsLink(p) && !File.Exists(p), "Output parent aliases/non-directories refused.");
    }
    private static Dictionary<string, string> Args(string[] argv)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < argv.Length; i++)
        {
            string key = argv[i];
            Need(key is "--execute" or "--with-native-draft-taps" or "--prepare-inputs" or "--export-frontend" or "--lengths" or "--self-test" or "--help" or "--artifact" or "--expect-directory-sha" or "--inputs" or "--out" or "--context" or "--ubatch", "Unknown option: " + key);
            Need(!result.ContainsKey(key), "Duplicate option: " + key);
            if (key is "--execute" or "--with-native-draft-taps" or "--prepare-inputs" or "--export-frontend" or "--self-test" or "--help") result.Add(key, "1");
            else { Need(++i < argv.Length && !string.IsNullOrWhiteSpace(argv[i]), "Missing option value: " + key); result.Add(key, argv[i]); }
        }
        Need(new[] { "--execute", "--prepare-inputs", "--export-frontend", "--self-test" }.Count(result.ContainsKey) <= 1, "Execution modes are exclusive.");
        Need(!result.ContainsKey("--with-native-draft-taps") || !new[] { "--prepare-inputs", "--export-frontend", "--self-test" }.Any(result.ContainsKey), "Native draft taps modify only execution or its unexecuted plan.");
        return result;
    }
    private static void ValidateDrafterMode(bool nativeTaps, bool hasDrafter, bool hasNativeDrafter, bool hasExternalDrafter)
    {
        if (nativeTaps)
            Need(hasDrafter && hasNativeDrafter && !hasExternalDrafter, "The explicit native-taps mode requires an attached native NInfer drafter and no external GGUF drafter.");
        else
            Need(!hasDrafter && !hasNativeDrafter && !hasExternalDrafter, "The default target-only mode requires engine.HasDrafter=false; use --with-native-draft-taps for ordinary forwards with an attached native NInfer drafter.");
    }
    private static string Required(Dictionary<string, string> options, string name) => options.TryGetValue(name, out string? value) ? value : throw new ArgumentException("Missing " + name);
    private static int Positive(Dictionary<string, string> options, string name, int fallback)
    {
        if (!options.TryGetValue(name, out string? value)) return fallback;
        Need(int.TryParse(value, out int n) && n > 0, "Positive integer required: " + name); return n;
    }
    private static InputCase[] Inputs(byte[] bytes, int context, int validVocab)
    {
        using var doc = JsonDocument.Parse(bytes); var root = doc.RootElement;
        Need(root.ValueKind == JsonValueKind.Object && root.EnumerateObject().Select(x => x.Name).Order().SequenceEqual(new[] { "cases", "schema" }), "Exact input-manifest fields required.");
        Need(root.GetProperty("schema").GetString() == "ninfer-qwen38-body-inputs/1", "Input schema differs.");
        var cases = new List<InputCase>(); var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in root.GetProperty("cases").EnumerateArray())
        {
            Need(row.EnumerateObject().Select(x => x.Name).Order().SequenceEqual(new[] { "chunkSize", "continuationTokens", "id", "promptTokens" }), "Exact case fields required.");
            string id = row.GetProperty("id").GetString() ?? "";
            Need(id.Length is > 0 and <= 80 && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_') && ids.Add(id), "Unique portable case IDs required.");
            int[] prompt = row.GetProperty("promptTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int[] continuation = row.GetProperty("continuationTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int chunk = row.GetProperty("chunkSize").GetInt32();
            Need(prompt.Length > 0 && continuation.Length > 0 && chunk > 0 && chunk <= context && (long)prompt.Length + continuation.Length <= context, "Case token/chunk bounds invalid.");
            Need(prompt.Concat(continuation).All(x => x >= 0 && x < validVocab), "Input contains a padded/invalid token ID.");
            cases.Add(new(id, prompt, continuation, chunk));
        }
        Need(cases.Count is > 0 and <= 128, "Require1..128 cases."); return cases.ToArray();
    }
    private static object Snapshot(string path)
    {
        var file = new FileInfo(path); Need(file.Exists && !IsLink(path), "Original regular artifact required.");
        return new { file.FullName, file.Length, file.LastWriteTimeUtc, file.CreationTimeUtc };
    }
    private static void SelfTest()
    {
        const string valid = "{\"schema\":\"ninfer-qwen38-body-inputs/1\",\"cases\":[{\"id\":\"one\",\"promptTokens\":[1,2],\"continuationTokens\":[3],\"chunkSize\":2}]}";
        var bytes = System.Text.Encoding.UTF8.GetBytes(valid); Need(Inputs(bytes, 3, 4).Single().PromptTokens.Length == 2, "Valid input refused.");
        int count = 1;
        foreach (var bad in new[] { valid.Replace("\"one\"", "\"../one\""), valid.Replace("[1,2]", "[1,4]"), valid.Replace("[3]", "[]"), valid.Replace("\"chunkSize\":2", "\"chunkSize\":0"), valid.Replace("\"id\":", "\"extra\":1,\"id\":") })
        { bool rejected = false; try { Inputs(System.Text.Encoding.UTF8.GetBytes(bad), 3, 4); } catch (InvalidDataException) { rejected = true; } Need(rejected, "Invalid input accepted."); count++; }
        bool overflow = false; try { Inputs(bytes, 2, 4); } catch (InvalidDataException) { overflow = true; } Need(overflow, "Context overflow accepted."); count++;
        byte[] values = new byte[8]; BinaryPrimitives.WriteSingleLittleEndian(values, 1.25f); BinaryPrimitives.WriteSingleLittleEndian(values.AsSpan(4), -2.5f);
        Need(BinaryPrimitives.ReadSingleLittleEndian(values) == 1.25f && BinaryPrimitives.ReadSingleLittleEndian(values.AsSpan(4)) == -2.5f, "Little-endian fixture invalid."); count++;
        Need(!Args(Array.Empty<string>()).ContainsKey("--with-native-draft-taps"), "Default mode unexpectedly enables draft taps."); count++;
        Need(Args(new[] { "--execute", "--with-native-draft-taps" }).ContainsKey("--with-native-draft-taps"), "Explicit native-taps execution refused."); count++;
        Need(Args(new[] { "--with-native-draft-taps" }).ContainsKey("--with-native-draft-taps"), "Unexecuted native-taps plan refused."); count++;
        foreach (var bad in new[] { new[] { "--with-native-draft-taps", "--with-native-draft-taps" }, new[] { "--prepare-inputs", "--with-native-draft-taps" }, new[] { "--export-frontend", "--with-native-draft-taps" }, new[] { "--self-test", "--with-native-draft-taps" } })
        { bool rejected = false; try { Args(bad); } catch (InvalidDataException) { rejected = true; } Need(rejected, "Invalid native-taps option combination accepted."); count++; }
        foreach (bool nativeTaps in new[] { false, true })
            foreach (bool hasDrafter in new[] { false, true })
                foreach (bool hasNative in new[] { false, true })
                    foreach (bool hasExternal in new[] { false, true })
                    {
                        bool accepted = true; try { ValidateDrafterMode(nativeTaps, hasDrafter, hasNative, hasExternal); } catch (InvalidDataException) { accepted = false; }
                        bool expected = nativeTaps ? hasDrafter && hasNative && !hasExternal : !hasDrafter && !hasNative && !hasExternal;
                        Need(accepted == expected, "Drafter-mode admission differs from its explicit scope."); count++;
                    }
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", contracts = count, GPU = "NOT_RUN", inference = "NOT_RUN", quality = "NOT_RUN", performance = "NOT_RUN" }));
    }
    public static int Main(string[] argv)
    {
        string? output = null; bool owned = false;
        var report = new Dictionary<string, object?> { ["schema"] = "ninfer-qwen38-body-logits/1", ["complete"] = false, ["engine"] = "TensorSharp/native-cuda", ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_QUALIFIED", ["independentOracle"] = "NOT_RUN" };
        var captures = new List<Capture>(); report["captures"] = captures;
        try
        {
            var options = Args(argv);
            bool nativeTaps = options.ContainsKey("--with-native-draft-taps");
            report["mode"] = nativeTaps ? "ordinary-native-draft-taps" : "target-only";
            report["executionScope"] = "Fixed teacher-forced ordinary Forward/Decode calls only; full-head reset/replay and clone diagnostics. No Propose, Verify, Commit or speculative acceptance calls.";
            report["draftProposalInference"] = "NOT_RUN";
            report["speculativeAcceptance"] = "NOT_RUN";
            if (options.ContainsKey("--help")) { Console.WriteLine("--execute --artifact FILE --expect-directory-sha SHA --inputs TOKEN_MANIFEST --out FRESH_IGNORED_DIRECTORY [--context 8192] [--ubatch 512] [--with-native-draft-taps]; --prepare-inputs --artifact FILE --expect-directory-sha SHA --out FRESH_IGNORED_DIRECTORY [--lengths 1,7,8,9,33,128,129,4096]; --export-frontend --artifact FILE --expect-directory-sha SHA --out FRESH_IGNORED_DIRECTORY; --self-test. Default execution requires no engine drafter. --with-native-draft-taps requires an explicitly configured native NInfer drafter and runs ordinary forwards only. No GPU work without --execute."); return 0; }
            if (options.ContainsKey("--self-test")) { SelfTest(); return 0; }
            if (!options.ContainsKey("--execute") && !options.ContainsKey("--prepare-inputs") && !options.ContainsKey("--export-frontend")) { Console.WriteLine(JsonSerializer.Serialize(report, Json)); return 0; }
            string artifact = Path.GetFullPath(Required(options, "--artifact"));
            string directorySha = Required(options, "--expect-directory-sha").ToLowerInvariant();
            Need(directorySha.Length == 64 && directorySha.All(Uri.IsHexDigit), "Expected directory SHA invalid.");
            int context = Positive(options, "--context", 8192), ubatch = Positive(options, "--ubatch", 512);
            Need(ubatch <= context && context <= 262144, "Unsupported context/ubatch bounds.");
            output = Required(options, "--out"); FreshOutput(output);
            Need(!artifact.StartsWith(output + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Output cannot own an input ancestor.");
            object before = Snapshot(artifact);
            using var reader = new NinferArtifactReader(artifact);
            Need(reader.DirectorySha256 == directorySha, "Artifact directory SHA differs.");
            var frontend = NinferQwen38Frontend.FromReader(reader);
            if (options.ContainsKey("--export-frontend"))
            {
                Directory.CreateDirectory(output); owned = true;
                foreach (var resource in frontend.Resources.Values)
                {
                    // Names were validated against the six fixed resource names by
                    // FromReader; preserve their original bytes, including the template.
                    string file = Path.Combine(output, resource.Name);
                    Directory.CreateDirectory(Path.GetDirectoryName(file)!);
                    File.WriteAllBytes(file, resource.CopyBytes());
                    Need(FileHash(file) == resource.Sha256, "Exported frontend resource SHA differs.");
                }
                object exportAfter = Snapshot(artifact); Need(JsonSerializer.Serialize(before) == JsonSerializer.Serialize(exportAfter), "Artifact stat changed.");
                Save(Path.Combine(output, "frontend-export.json"), new { schema = "ninfer-qwen38-body-frontend-export/1", complete = true,
                    artifactDirectorySha256 = reader.DirectorySha256, artifactBefore = before, artifactAfter = exportAfter,
                    fullArtifactSha256Recomputed = false, resources = frontend.Resources.Values.Select(x => new { x.Name, x.Sha256, x.ByteLength }),
                    GPU = "NOT_RUN", inference = "NOT_RUN", quality = "NOT_RUN" });
                Console.WriteLine(JsonSerializer.Serialize(new { complete = true, output, resources = frontend.Resources.Count, GPU = "NOT_RUN", inference = "NOT_RUN" })); return 0;
            }
            if (options.ContainsKey("--prepare-inputs"))
            {
                int[] lengths = options.GetValueOrDefault("--lengths", "1,7,8,9,33,128,129,4096").Split(',').Select(int.Parse).ToArray();
                int[] unit = frontend.Tokenizer.Encode("A computer stores numbers in memory. Paris is the capital of France. Return a careful answer.\n", false).ToArray();
                int[] continuation = frontend.Tokenizer.Encode(" The result is Paris, and two plus two is four.\n", false).Take(8).ToArray();
                Need(lengths.Length is > 0 and <= 128 && lengths.Distinct().Count() == lengths.Length && lengths.All(n => n > 0 && (long)n + continuation.Length <= context), "Invalid preparation lengths/context.");
                var manifest = new { schema = "ninfer-qwen38-body-inputs/1", cases = lengths.Select(n => new { id = "tokens-" + n, promptTokens = Enumerable.Range(0, n).Select(i => unit[i % unit.Length]).ToArray(), continuationTokens = continuation, chunkSize = Math.Min(n, ubatch) }).ToArray() };
                byte[] prepared = JsonSerializer.SerializeToUtf8Bytes(manifest, Json); _ = Inputs(prepared, context, frontend.Tokenizer.VocabSize);
                Directory.CreateDirectory(output); owned = true; File.WriteAllBytes(Path.Combine(output, "inputs.json"), prepared);
                Save(Path.Combine(output, "preparation.json"), new { schema = "ninfer-qwen38-body-input-preparation/1", complete = true, directorySha256 = reader.DirectorySha256, inputsSha256 = Hash(prepared), lengths, validVocabSize = frontend.Tokenizer.VocabSize, rawTokenStream = true, chatTemplateApplied = false, GPU = "NOT_RUN", inference = "NOT_RUN", quality = "NOT_RUN" });
                Console.WriteLine(JsonSerializer.Serialize(new { complete = true, inputs = Path.Combine(output, "inputs.json"), GPU = "NOT_RUN", inference = "NOT_RUN" })); return 0;
            }
            string inputsPath = Path.GetFullPath(Required(options, "--inputs"));
            Need(!inputsPath.StartsWith(output + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Output cannot own an input ancestor.");
            byte[] inputBytes = File.ReadAllBytes(inputsPath);
            var cases = Inputs(inputBytes, context, frontend.Tokenizer.VocabSize);
            Directory.CreateDirectory(output); owned = true;
            File.WriteAllBytes(Path.Combine(output, "inputs.json"), inputBytes);
            report["inputsSha256"] = Hash(inputBytes); report["artifactBefore"] = before;
            report["artifactDirectorySha256"] = reader.DirectorySha256;
            report["artifactFullSha256Recomputed"] = false;
            report["frontendResourceHashes"] = frontend.Resources.OrderBy(x => x.Key, StringComparer.Ordinal).ToDictionary(x => x.Key, x => x.Value.Sha256);
            report["configuration"] = new { context, ubatch, draft = nativeTaps ? "ATTACHED_NATIVE_ORDINARY_TAPS_ONLY" : "NONE", sampling = "teacher-forced, no sampler", passes = 2, cloneProbe = "one token after full prefill", precision = "actual engine artifact policy; see numericalStateNamespace and engine startup log" };
            report["environment"] = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(x => ((string)x.Key).StartsWith("TS_Q35_", StringComparison.Ordinal)).OrderBy(x => x.Key).ToDictionary(x => (string)x.Key, x => (string?)x.Value);
            report["execution"] = new { tool = AssemblyId(Assembly.GetExecutingAssembly()), models = AssemblyId(typeof(Qwen35CudaExecutor).Assembly), runtime = AssemblyId(typeof(NinferArtifactReader).Assembly), backend = AssemblyId(typeof(TensorSharp.Cuda.Q35CudaEngine).Assembly) };
            Save(Path.Combine(output, "run.json"), report);
            var load = Stopwatch.StartNew(); using var executor = new Qwen35CudaExecutor(artifact, context, ubatch);
            report["loadWallMilliseconds"] = load.Elapsed.TotalMilliseconds;
            var engine = executor.Engine; var slot = engine.Active;
            report["GPU"] = "EXECUTED";
            report["loadRepack"] = new { engine.NinferLoadRepackedMatrices, engine.NinferLoadRepackedOriginalBytes, engine.NinferLoadRepackImageSha256 };
            report["draftState"] = new { engine.HasDrafter, engine.HasNinferDrafter, externalGgufDrafter = executor.Draft != null,
                blockSize = engine.HasDrafter ? engine.DraftBlockSize : (int?)null,
                ringRows = engine.HasDrafter ? engine.DraftRingRows : (int?)null,
                maximumLanes = engine.HasNinferDrafter ? engine.NinferDraftMaximumLanes : (int?)null,
                headPolicy = engine.HasNinferDrafter ? (engine.NinferDraftUsesShortlist ? "original-q4-g64-shortlist" : "original-full-target-w8g32") : "NONE",
                engine.NinferDraftPayloadSha256, engine.NinferDraftModuleSha256 };
            report["hostEmbedding"] = new { engine.NinferHostEmbeddingEnabled, engine.NinferHostEmbeddingPayloadBytes,
                engine.NinferHostEmbeddingPayloadSha256, engine.NinferHostEmbeddingPinnedBytes };
            report["numericalStateNamespace"] = engine.KVStateNamespaceSuffix;
            Save(Path.Combine(output, "run.json"), report);
            Need(engine.VocabSize == frontend.Tokenizer.HeadVocabSize, "Physical head size differs.");
            ValidateDrafterMode(nativeTaps, engine.HasDrafter, engine.HasNinferDrafter, executor.Draft != null);
            report["vocabSize"] = engine.VocabSize; report["validVocabSize"] = frontend.Tokenizer.VocabSize;
            bool allRepeat = true, allClone = true;
            Capture Write(InputCase item, int pass, string kind, int position, float[] logits, double elapsed)
            {
                Need(logits.Take(frontend.Tokenizer.VocabSize).All(float.IsFinite), "Nonfinite valid-vocabulary logits.");
                int argmax = 0; for (int i = 1; i < frontend.Tokenizer.VocabSize; i++) if (logits[i] > logits[argmax]) argmax = i;
                int headArgmax = 0; for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[headArgmax]) headArgmax = i;
                string file = $"{item.Id}.p{pass}.{kind}.{position:D6}.f32";
                // Explicit LE output remains portable if a future host is big-endian.
                byte[] bytes = new byte[checked(logits.Length * 4)]; for (int i = 0; i < logits.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), logits[i]);
                File.WriteAllBytes(Path.Combine(output, file), bytes);
                int nonfinitePadding = logits.Skip(frontend.Tokenizer.VocabSize).Count(x => !float.IsFinite(x));
                var capture = new Capture(item.Id, pass, kind, position, file, Hash(bytes), logits.Length, argmax, headArgmax,
                    headArgmax >= frontend.Tokenizer.VocabSize, nonfinitePadding, elapsed); captures.Add(capture); return capture;
            }
            foreach (var item in cases)
            {
                var first = new Dictionary<(string, int), string>();
                for (int pass = 0; pass < 2; pass++)
                {
                    engine.Active = slot; engine.ResetSlot(slot); var logits = new float[engine.VocabSize];
                    for (int start = 0; start < item.PromptTokens.Length; start += item.ChunkSize)
                    {
                        int length = Math.Min(item.ChunkSize, item.PromptTokens.Length - start);
                        var timer = Stopwatch.StartNew(); engine.Forward(item.PromptTokens.AsSpan(start, length).ToArray(), logits);
                        Need(slot.NPast == start + length, "Prefill position differs from supplied token count.");
                        var capture = Write(item, pass, "prefill", slot.NPast, logits, timer.Elapsed.TotalMilliseconds);
                        if (pass == 0) first.Add((capture.Kind, capture.Position), capture.Sha256); else allRepeat &= first[(capture.Kind, capture.Position)] == capture.Sha256;
                    }
                    var clone = engine.CreateSlot(0);
                    try
                    {
                        engine.CopySlot(slot, clone);
                        var timer = Stopwatch.StartNew(); engine.Forward(new[] { item.ContinuationTokens[0] }, logits);
                        Need(slot.NPast == item.PromptTokens.Length + 1, "Original continuation position differs.");
                        var original = Write(item, pass, "decode", slot.NPast, logits, timer.Elapsed.TotalMilliseconds);
                        if (pass == 0) first.Add((original.Kind, original.Position), original.Sha256); else allRepeat &= first[(original.Kind, original.Position)] == original.Sha256;
                        engine.Active = clone; timer.Restart(); engine.Forward(new[] { item.ContinuationTokens[0] }, logits);
                        Need(clone.NPast == slot.NPast, "Copied continuation position differs.");
                        var copied = Write(item, pass, "clone", clone.NPast, logits, timer.Elapsed.TotalMilliseconds);
                        allClone &= original.Sha256 == copied.Sha256;
                    }
                    finally { engine.Active = slot; engine.FreeSlot(clone); }
                    for (int i = 1; i < item.ContinuationTokens.Length; i++)
                    {
                        var timer = Stopwatch.StartNew(); engine.Forward(new[] { item.ContinuationTokens[i] }, logits);
                        Need(slot.NPast == item.PromptTokens.Length + i + 1, "Continuation position differs.");
                        var capture = Write(item, pass, "decode", slot.NPast, logits, timer.Elapsed.TotalMilliseconds);
                        if (pass == 0) first.Add((capture.Kind, capture.Position), capture.Sha256); else allRepeat &= first[(capture.Kind, capture.Position)] == capture.Sha256;
                    }
                    Save(Path.Combine(output, "run.json"), report);
                }
            }
            report["repeatFullBitsEqual"] = allRepeat; report["cloneFullBitsEqual"] = allClone;
            report["decodeGraphs"] = new { engine.CapturedNinferDecodeGraphCount, engine.NinferDecodeGraphCaptureCount,
                engine.NinferDecodeGraphReplayCount, engine.NinferDecodeGraphEvictionCount };
            object after = Snapshot(artifact); report["artifactAfter"] = after;
            Need(JsonSerializer.Serialize(before) == JsonSerializer.Serialize(after), "Artifact stat changed.");
            report["artifactIdentityScope"] = "Directory SHA + original path/stat; external whole-file SHA/inode guard required.";
            Need(allRepeat && allClone, "Repeat or cache-clone full-bit gate failed; captures retained.");
            report["complete"] = true; Save(Path.Combine(output, "run.json"), report);
            SaveCompleted(output, report);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, output, captures = captures.Count, repeatFullBitsEqual = allRepeat, cloneFullBitsEqual = allClone, modelQuality = "NOT_RUN", independentOracle = "NOT_RUN", performance = "DIAGNOSTIC_ONLY" })); return 0;
        }
        catch (Exception e)
        {
            report["error"] = e.ToString(); if (owned && output != null) Save(Path.Combine(output, "run.json"), report); Console.Error.WriteLine(e); return 1;
        }
    }
}
