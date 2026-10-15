// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GemmaScalarRingProbe;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

return RingProbe.Run(args);

internal static class RingProbe
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private const int ContinuationRows = 4;

    internal static int Run(string[] args)
    {
        if (args.SequenceEqual(new[] { "--self-test" })) return ExactComparison.SelfTest();
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || args[i] is not ("--model" or "--output" or "--expected-native-sha256" or "--expected-models-sha256"))
                throw new ArgumentException("Use --model GGUF --output FRESH_DIRECTORY --expected-native-sha256 SHA --expected-models-sha256 SHA; or --self-test.");
            options.Add(args[i], args[i + 1]);
        }
        string modelPath = Path.GetFullPath(options["--model"]), directory = Path.GetFullPath(options["--output"]);
        string expectedNative = Digest(options["--expected-native-sha256"]), expectedModels = Digest(options["--expected-models-sha256"]);
        if (Directory.Exists(directory) || File.Exists(directory)) throw new IOException("Evidence directory must be fresh.");
        Directory.CreateDirectory(directory);
        var managedBefore = ManagedIdentity();
        var checkpointBefore = CheckpointIdentity(modelPath);
        var cases = new List<CaseResult>();
        var errors = new List<string>();
        FileIdentity[] native = [];
        ModelBase? model = null;
        bool disposed = false;
        int window = 0;
        var timer = Stopwatch.StartNew();
        using var process = Process.GetCurrentProcess();
        Write(directory, "started.json", new { args, process_id = process.Id, process_started_utc = process.StartTime.ToUniversalTime(), managedBefore, checkpointBefore });
        try
        {
            if (Hash(typeof(ModelBase).Assembly.Location) != expectedModels)
                throw new InvalidDataException("Loaded Models assembly does not match the expected pin.");
            using (var gguf = new GgufFile(modelPath))
            {
                if (gguf.GetString("general.architecture") != "gemma4") throw new ArgumentException("This probe requires Gemma4.");
                if (gguf.GetUint32("gemma4.expert_count", 0) == 0) throw new ArgumentException("This probe requires the Gemma4 MoE target.");
                window = checked((int)gguf.GetUint32("gemma4.attention.sliding_window", 512));
                if (window < 2 || window > 8192) throw new ArgumentException("A sliding window of 2..8192 is required by this bounded probe.");
            }
            // An arbitrary stride (for example W+20=1044) can disable the
            // dim512 native flash kernel. Use a bounded power-of-two capacity;
            // W=1024 therefore matches the scheduler matrix's 2048 cache.
            int context = 256;
            while (context < checked(window + ContinuationRows + 16)) context = checked(context * 2);
            Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
            MoeCpuOffloadConfig.ConfigureFromEnvironment();
            KvCacheDtypeConfig.Set(KvCacheDtype.F16);
            model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
            native = NativeIdentity();
            if (native.Length != 1 || native[0].Sha256 != expectedNative)
                throw new InvalidDataException("Actual loaded GgmlOps does not match the required pin.");
            if (model.Config.Architecture != "gemma4" || model.Config.SlidingWindow != window || model.KvCacheDtype != KvCacheDtype.F16)
                throw new InvalidDataException("Actual architecture/window/KV does not match this diagnostic.");
            var target = (ISpeculativeTarget)model;
            if (!target.SpecVerifyPersistsAcceptedKv) throw new InvalidDataException("This probe requires the MoE target's persistent verified KV protocol.");
            Write(directory, "loaded.json", new { native, models_sha256 = Hash(typeof(ModelBase).Assembly.Location), window,
                kv = model.KvCacheDtype.ToString(), model.MaxContextLength, model.Config.HiddenSize, model.Config.VocabSize, model.Config.Architecture,
                target.SpecVerifyPersistsAcceptedKv, model_type = model.GetType().FullName });
            model.WarmUpKernels();
            foreach (int prefix in new[] { window - 1, window + 1 })
            foreach (bool accepted in new[] { false, true })
            {
                string stem = $"prefix{prefix}-{(accepted ? "accepted" : "rejected")}";
                var result = ExecuteCase(model, target, BuildPrompt(model, prefix), accepted, directory, stem);
                cases.Add(result);
                Write(directory, stem + ".json", result);
                Console.WriteLine(JsonSerializer.Serialize(new { stem, result.Passed, result.ExpectedFinalPosition, result.ActualFinalPosition }));
            }
        }
        catch (Exception error) { errors.Add(error.ToString()); Console.Error.WriteLine(error); }
        finally
        {
            try { model?.Dispose(); disposed = true; } catch (Exception error) { errors.Add("Dispose: " + error); }
        }
        FileIdentity[] managedAfter = [], nativeAfter = [];
        object? checkpointAfter = null;
        bool identityStable = false;
        try
        {
            managedAfter = ManagedIdentity(); nativeAfter = native.Select(x => Identity(x.Path)).ToArray(); checkpointAfter = CheckpointIdentity(modelPath);
            identityStable = managedBefore.SequenceEqual(managedAfter) && native.SequenceEqual(nativeAfter) &&
                JsonSerializer.Serialize(checkpointBefore) == JsonSerializer.Serialize(checkpointAfter);
        }
        catch (Exception error) { errors.Add("Final identities: " + error); }
        bool completed = errors.Count == 0 && disposed && identityStable && cases.Count == 4;
        bool passed = completed && cases.All(c => c.Passed);
        Write(directory, "report.json", new { completed, passed, errors, disposed, identity_stable = identityStable,
            elapsed_seconds = timer.Elapsed.TotalSeconds, window, cases, managedBefore, managedAfter, native, nativeAfter, checkpointBefore, checkpointAfter,
            scope = "Strict raw trunk/KV diagnostic: matched ForwardRefill prefixes; two scalar SpecForward(1) reference rows versus SpecForward(2), complete finite bit-exact logits and hidden rows; then accepted/rejected protocol and four teacher-forced ordinary Forward continuation rows. No MTP draft head, sampler, semantic/EOS or throughput qualification. Checkpoint continuity is size/mtime, not a new full weight hash." });
        return passed ? 0 : 3;
    }

    private static CaseResult ExecuteCase(ModelBase model, ISpeculativeTarget target, int[] prompt, bool accepted, string directory, string stem)
    {
        int vocab = model.Config.VocabSize, hidden = model.Config.HiddenSize, position = prompt.Length;
        var checks = new List<RowResult>();
        Write(directory, stem + "-prompt.json", prompt);
        float[] Prefill()
        {
            model.ResetKVCache();
            float[] row = (float[])model.ForwardRefill(prompt).Clone();
            if (target.CacheSeqLen != position) throw new InvalidDataException("Prefill cache position mismatch.");
            RequireRow(row, vocab);
            return row;
        }
        float[] Scalar(int token, out float[] h)
        {
            h = new float[hidden]; var logits = new float[vocab];
            target.SpecForward([token], h, logits, allLogitsRows: false);
            return logits;
        }
        void Compare(string phase, float[] reference, float[] actual)
        {
            var comparison = ExactComparison.Compare(reference, actual);
            Save(directory, stem + "-" + phase + "-reference.f32", reference);
            Save(directory, stem + "-" + phase + "-actual.f32", actual);
            checks.Add(new(phase, comparison));
        }
        float[] prefill = Prefill();
        int t0 = Argmax(prefill);
        float[] row0 = Scalar(t0, out float[] h0);
        int correct = Argmax(row0), t1 = accepted ? correct : (correct + 1) % vocab;
        float[] row1 = Scalar(t1, out float[] h1);
        if (target.CacheSeqLen != position + 2) throw new InvalidDataException("Scalar reference did not advance twice.");
        if (!accepted)
        {
            Compare("reference-reset-prefill", prefill, Prefill());
            Compare("reference-reset-row0", row0, Scalar(t0, out float[] resetHidden));
            Compare("reference-reset-hidden0", h0, resetHidden);
        }
        int keptRows = accepted ? 2 : 1;
        var continuationInputs = new int[ContinuationRows];
        var continuationRows = new List<float[]>();
        int token = Argmax(accepted ? row1 : row0);
        for (int row = 0; row < ContinuationRows; row++)
        {
            continuationInputs[row] = token;
            float[] logits = (float[])model.Forward([token]).Clone();
            RequireRow(logits, vocab); continuationRows.Add(logits); token = Argmax(logits);
        }
        Compare("candidate-prefill", prefill, Prefill());
        target.SpecEnsureCapacity(position + 2);
        target.SpecSnapshotRecurrentState();
        var verifyLogits = new float[checked(2 * vocab)]; var verifyHidden = new float[checked(2 * hidden)];
        target.SpecForward([t0, t1], verifyHidden, verifyLogits, allLogitsRows: true);
        int postVerify = target.CacheSeqLen;
        Compare("verify-row0", row0, verifyLogits.AsSpan(0, vocab).ToArray());
        Compare("verify-row1", row1, verifyLogits.AsSpan(vocab, vocab).ToArray());
        Compare("verify-hidden0", h0, verifyHidden.AsSpan(0, hidden).ToArray());
        Compare("verify-hidden1", h1, verifyHidden.AsSpan(hidden, hidden).ToArray());
        bool actualBranchValid = (Argmax(verifyLogits.AsSpan(0, vocab)) == t1) == accepted;
        target.SpecOnVerifyAccepted(accepted ? 1 : 0, 1);
        if (!accepted) target.SpecRewindCache(position + keptRows);
        int postSettlement = target.CacheSeqLen;
        for (int row = 0; row < ContinuationRows; row++)
        {
            float[] logits = (float[])model.Forward([continuationInputs[row]]).Clone();
            RequireRow(logits, vocab); Compare($"continuation{row}", continuationRows[row], logits);
        }
        int expectedFinal = position + keptRows + ContinuationRows;
        bool passed = checks.All(c => c.Comparison.Passed) && actualBranchValid && postVerify == position + 2 &&
            postSettlement == position + keptRows && target.CacheSeqLen == expectedFinal;
        return new(position, accepted, [t0, t1], continuationInputs, postVerify, postSettlement, expectedFinal, target.CacheSeqLen, actualBranchValid, checks, passed);
    }

    private static int[] BuildPrompt(ModelBase model, int length)
    {
        var text = new StringBuilder();
        for (int line = 1; line < 100000; line++)
        {
            text.Append($"{line} var item{line} = new Widget{line % 7}(); item{line}.Add({line * 3}); // total {line * line}\n");
            if (line % 40 != 0) continue;
            var tokens = model.Tokenizer.Encode(text.ToString(), addSpecial: true);
            if (tokens.Count >= length) return tokens.GetRange(0, length).ToArray();
        }
        throw new InvalidOperationException("Could not construct the bounded prompt.");
    }
    private static void RequireRow(float[] row, int vocab) { if (row.Length != vocab || row.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Incomplete/nonfinite model row."); }
    private static int Argmax(ReadOnlySpan<float> row)
    {
        if (row.IsEmpty) throw new InvalidDataException("Empty row.");
        int best = 0;
        for (int i = 0; i < row.Length; i++) { if (!float.IsFinite(row[i])) throw new InvalidDataException("Nonfinite model row."); if (row[i] > row[best]) best = i; }
        return best;
    }
    private static void Save(string directory, string name, float[] values) => File.WriteAllBytes(Path.Combine(directory, name), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    private static void Write(string directory, string name, object value) => File.WriteAllText(Path.Combine(directory, name), JsonSerializer.Serialize(value, Json));
    private static string Digest(string value) { if (value.Length != 64 || !value.All(Uri.IsHexDigit)) throw new ArgumentException("Expected SHA256."); return value.ToLowerInvariant(); }
    private static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private static FileIdentity Identity(string path) => new(Path.GetFullPath(path), Hash(path));
    private static FileIdentity[] ManagedIdentity() => Directory.GetFiles(AppContext.BaseDirectory).Where(p => p.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".deps.json") || p.EndsWith(".runtimeconfig.json")).Order().Select(Identity).ToArray();
    private static FileIdentity[] NativeIdentity()
    {
        using var process = Process.GetCurrentProcess();
        return process.Modules.Cast<ProcessModule>().Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase)).Select(m => Identity(m.FileName)).ToArray();
    }
    private static object CheckpointIdentity(string path) { var f = new FileInfo(path); return new { path = f.FullName, bytes = f.Length, modified_utc = f.LastWriteTimeUtc }; }
    private sealed record FileIdentity(string Path, string Sha256);
    private sealed record RowResult(string Phase, ExactComparison Comparison);
    private sealed record CaseResult(int Prefix, bool Accepted, int[] VerifyTokens, int[] ContinuationInputs,
        int PostVerifyPosition, int PostSettlementPosition, int ExpectedFinalPosition, int ActualFinalPosition,
        bool ActualBranchValid, List<RowResult> Rows, bool Passed);
}
