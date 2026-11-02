// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen3.5/3.8 direct-CUDA engine probe: compares the engine (TensorSharp.Backends.Cuda/Qwen35)
// with the ggml_cuda executor on one prompt and a greedy continuation.
//
//   --ggml <dir>                 run the ggml_cuda model: the prompt, then STEPS greedy tokens; write
//                                tokens.json and every step's logits (step_<i>.f32).
//   --native <ref dir> <dir>     run the engine on the reference's prompt and tokens (teacher-forced,
//                                so steps stay aligned when an argmax differs) and write the same.
//   --compare <ref dir> <dir>    per step: argmax agreement, top-5 overlap, logits correlation and
//                                the largest difference relative to the reference's range.
//   --host <ref dir>             a slot copied to pinned host memory and back decodes bit-identically.
//   --mixed <ref dir>            a prompt chunk and a verify in one forward match them run apart.
// Each run loads one model; run them as separate processes (one GPU holds one 27B model).
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

// Separate allocation diagnostic; the existing mixed-capacity mode remains unchanged.
if (args.Length > 0 && args[0] == "--shared-scratch-trace")
{
    if (args.Length < 2 || args.Length > 3 || (args.Length == 3 && args[2] != "--plan-only"))
        throw new ArgumentException("Use --shared-scratch-trace <fresh-output> [--plan-only].");
    return args.Length == 3
        ? SharedScratchTrace.Plan(args[1])
        : SharedScratchTrace.Run(Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
            ?? throw new ArgumentException("Set TS_TEST_QWEN35_MODEL."), args[1]);
}

// This diagnostic has a CPU-only plan mode and must be dispatched before the
// ordinary probe requires a model path. Existing modes keep their defaults.
if (args.Length >= 2 && args[0] == "--mixed-capacity-trace")
{
    if (args.Length > 3 || (args.Length == 3 && args[2] != "--plan-only"))
        throw new ArgumentException("Use --mixed-capacity-trace <fresh-output> [--plan-only].");
    return args.Length == 3
        ? MixedCapacityTrace.Plan(args[1])
        : MixedCapacityTrace.Run(Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
            ?? throw new ArgumentException("Set TS_TEST_QWEN35_MODEL."), args[1]);
}

const int Steps = 24;
string prompt = string.Concat(Enumerable.Repeat(
    "The river flows through a green valley. Trees grow beside the water and birds fly overhead. " +
    "A computer stores numbers in memory. A program performs arithmetic and produces a result. " +
    "Paris is the capital of France. Tokyo is the capital of Japan. Rome is the capital of Italy. ", 3))
    + "Question: what is the capital of Japan? Answer:";
// TS_PROBE_PROMPT overrides the prompt: one of at most eight tokens keeps every projection of both
// implementations on the q8_1 GEMV path, which isolates the layer semantics from the FP4 rounding.
prompt = Environment.GetEnvironmentVariable("TS_PROBE_PROMPT") ?? prompt;
// TS_PROBE_PROMPT_FILE reads it from a file (long prompts exercise the prefill attention).
if (Environment.GetEnvironmentVariable("TS_PROBE_PROMPT_FILE") is { Length: > 0 } promptFile)
    prompt = File.ReadAllText(promptFile);
string modelPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
    ?? throw new ArgumentException("Set TS_TEST_QWEN35_MODEL.");

if (args.Length >= 2 && args[0] == "--verify-graph-trace")
{
    string? lengths = args.Skip(2).FirstOrDefault(a => a != "--snapshots");
    return VerifyGraphTrace.Run(modelPath, args[1], lengths, args.Contains("--snapshots"));
}

if (args.Length >= 2 && args[0] == "--greedy-verify-trace")
    return GreedyVerifyTrace.Run(modelPath, args[1], args.Length > 2 ? int.Parse(args[2]) : 12);

if (args.Length >= 2 && args[0] == "--prefill-trace")
    return PrefillTrace.Run(modelPath, args[1], args.Length > 2 ? args[2] : null, args.Length > 3 ? int.Parse(args[3]) : 512);

if (args.Length >= 2 && args[0] == "--graph-trace")
{
    // Run twice, in separate processes, with TS_Q35_CUDA_GRAPHS=0 and =1. trace.json
    // must be byte-identical. The trace hashes every full-vocabulary draft/verify logit
    // and records confidence bits, including the first capture and repeated replays.
    // Lane order/width changes and slot replacement exercise stable pointer indirection.
    string draftPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT")
        ?? throw new ArgumentException("Set TS_TEST_QWEN35_DRAFT.");
    int rounds = args.Length > 2 ? int.Parse(args[2]) : 12;
    if (rounds < 6)
        throw new ArgumentException("Use at least 6 rounds to cover slot replacement.");
    Directory.CreateDirectory(args[1]);
    File.WriteAllText(Path.Combine(args[1], "run.json"), JsonSerializer.Serialize(new { complete = false }));
    using var metadata = new GgufFile(modelPath);
    var tokenizer = ModelBase.CreateTokenizerFromGguf(metadata);
    using var exec = new Qwen35CudaExecutor(modelPath, 8192, 512, draftPath);
    var eng = exec.Engine;
    var primary = eng.Active;
    string[] prompts = {
        "Write a Python function that returns the first n Fibonacci numbers. Return code only.\n",
        "Explain why the sky appears blue during a clear day.\n",
        "Translate to Chinese: The library is open every morning except Sunday.\n",
        "Return a JSON object with Paris as the capital of France and Tokyo as the capital of Japan.\n"
    };
    var slots = new TensorSharp.Cuda.Q35CudaEngine.Slot[4];
    var anchors = new int[4];
    var trace = new List<object>();
    var finalLogits = new float[exec.VocabSize];
    string Hash(float[] values)
    {
        if (values.Any(v => !float.IsFinite(v)))
            throw new InvalidOperationException("Graph trace encountered non-finite logits.");
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
    }
    void SaveFloats(string name, float[] values) => File.WriteAllBytes(Path.Combine(args[1], name),
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()));
    void Fill(int i, string text)
    {
        slots[i] = eng.CreateSlot(0);
        eng.Active = slots[i];
        eng.Forward(tokenizer.Encode(text, addSpecial: false).ToArray(), finalLogits);
        anchors[i] = ArgMax(finalLogits);
    }
    try
    {
        for (int i = 0; i < slots.Length; i++)
            Fill(i, prompts[i]);
        int[][] schedule = { new[] { 0 }, new[] { 0, 1 }, new[] { 0, 1, 2, 3 }, new[] { 3, 1 }, new[] { 2 }, new[] { 2, 0, 3, 1 } };
        int b = eng.DraftBlockSize, g = b - 1;
        for (int round = 0; round < rounds; round++)
        {
            if (round == rounds / 2)
            {
                eng.Active = primary;
                eng.FreeSlot(slots[1]);
                Fill(1, "Compute the sum of integers from 1 to 20 and explain the formula.\n");
            }
            int[] ids = schedule[round % schedule.Length];
            var current = ids.Select(i => slots[i]).ToArray();
            var currentAnchors = ids.Select(i => anchors[i]).ToArray();
            var proposed = ids.Select(_ => new int[g]).ToArray();
            var confidence = ids.Select(_ => new float[g]).ToArray();
            var draftLogits = new float[ids.Length * g * exec.VocabSize];
            eng.Draft(current, currentAnchors, proposed, confidence, draftLogits);
            SaveFloats($"round-{round:D3}.draft-a.f32", draftLogits);
            string draftHash = Hash(draftLogits);
            var firstLogits = (float[])draftLogits.Clone();
            var confidenceBits = confidence.Select(v => v.Select(BitConverter.SingleToInt32Bits).ToArray()).ToArray();
            var proposedCopy = proposed.Select(v => (int[])v.Clone()).ToArray();
            // Drafting must not mutate slots, so the first captured invocation and the replay
            // immediately after it must return identical logits, candidates and confidences.
            eng.Draft(current, currentAnchors, proposed, confidence, draftLogits);
            SaveFloats($"round-{round:D3}.draft-b.f32", draftLogits);
            if (Hash(draftLogits) != draftHash || !proposedCopy.Zip(proposed).All(p => p.First.SequenceEqual(p.Second))
                || !confidenceBits.Zip(confidence).All(p => p.First.SequenceEqual(p.Second.Select(BitConverter.SingleToInt32Bits))))
            {
                double maxAbs = 0, squaredError = 0, squaredReference = 0;
                int differing = 0;
                for (int i = 0; i < firstLogits.Length; i++)
                {
                    double delta = (double)draftLogits[i] - firstLogits[i];
                    maxAbs = Math.Max(maxAbs, Math.Abs(delta));
                    squaredError += delta * delta;
                    squaredReference += (double)firstLogits[i] * firstLogits[i];
                    differing += BitConverter.SingleToInt32Bits(draftLogits[i]) == BitConverter.SingleToInt32Bits(firstLogits[i]) ? 0 : 1;
                }
                File.WriteAllText(Path.Combine(args[1], "repeat-failure.json"), JsonSerializer.Serialize(new {
                    round, lanes = ids, max_abs = maxAbs, relative_l2 = Math.Sqrt(squaredError / Math.Max(squaredReference, 1e-300)),
                    differing_elements = differing, elements = firstLogits.Length,
                    first_proposed = proposedCopy, repeated_proposed = proposed,
                    first_confidence_bits = confidenceBits, repeated_confidence_bits = confidence.Select(v => v.Select(BitConverter.SingleToInt32Bits).ToArray()).ToArray(),
                    complete = false
                }, new JsonSerializerOptions { WriteIndented = true }));
                throw new InvalidOperationException($"DFlash repeat changed numerical results at round {round}");
            }
            var tokens = new int[ids.Length * b];
            for (int i = 0; i < ids.Length; i++)
            {
                tokens[i * b] = currentAnchors[i];
                Array.Copy(proposed[i], 0, tokens, i * b + 1, g);
            }
            var drawn = new int[tokens.Length];
            var verifyLogits = new float[tokens.Length * exec.VocabSize];
            eng.Verify(current, b, tokens, drawn, verifyLogits);
            SaveFloats($"round-{round:D3}.verify.f32", verifyLogits);
            int[] kept = new int[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                int accepted = 0;
                while (accepted < g && proposed[i][accepted] == drawn[i * b + accepted])
                    accepted++;
                kept[i] = accepted + 1;
            }
            trace.Add(new { round, lanes = ids, positions = current.Select(s => s.NPast).ToArray(),
                anchors = currentAnchors, proposed = proposedCopy, confidence_bits = confidenceBits,
                draft_logits_sha256 = draftHash, verify_logits_sha256 = Hash(verifyLogits), drawn, kept });
            eng.Commit(current, b, kept);
            for (int i = 0; i < ids.Length; i++)
                anchors[ids[i]] = drawn[i * b + kept[i] - 1];
        }
        // A cache reset and fresh prefill must still work after every captured lane shape.
        eng.Active = primary;
        eng.Reset();
        eng.Forward(tokenizer.Encode(prompts[0], addSpecial: false).ToArray(), finalLogits);
        trace.Add(new { reset_logits_sha256 = Hash(finalLogits), complete = true });
        if (Environment.GetEnvironmentVariable("TS_Q35_CUDA_GRAPHS") == "1" && eng.CapturedDraftGraphCount != 3)
            throw new InvalidOperationException($"Expected 3 captured lane shapes, got {eng.CapturedDraftGraphCount}");
        if (Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS") == "1" &&
            (eng.VerifyGraphCaptureCount < 1 || eng.VerifyGraphReplayCount <= eng.VerifyGraphCaptureCount))
            throw new InvalidOperationException("The requested Verify CUDA graphs did not exercise both capture and reuse.");
        File.WriteAllText(Path.Combine(args[1], "trace.json"), JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(args[1], "run.json"), JsonSerializer.Serialize(new {
            graph_setting = Environment.GetEnvironmentVariable("TS_Q35_CUDA_GRAPHS"), rounds,
            captured_lane_shapes = eng.CapturedDraftGraphCount,
            verify_graph_setting = Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS"),
            verify_graph_captures = eng.VerifyGraphCaptureCount, verify_graph_replays = eng.VerifyGraphReplayCount,
            draft_splitk_setting = Environment.GetEnvironmentVariable("TS_Q35_DRAFT_SPLITK"),
            target_splitk_setting = Environment.GetEnvironmentVariable("TS_Q35_GEMM_SPLITK"),
            partitioned_topk_setting = Environment.GetEnvironmentVariable("TS_Q35_TOPK_PARTITIONED"),
            complete = true
        }));
        Console.WriteLine($"[q35-probe] graph trace: {rounds} rounds, lanes 1/2/4, reordered/replaced slots, full draft/verify logits and confidence bits; complete");
    }
    finally
    {
        eng.Active = primary;
        foreach (var slot in slots)
            eng.FreeSlot(slot);
    }
    return 0;
}

if (args.Length == 2 && args[0] == "--ggml")
{
    Directory.CreateDirectory(args[1]);
    KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
    using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
    int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();
    var chosen = new List<int>();
    float[] logits = model.ForwardPrompt(tokens);
    for (int step = 0; step <= Steps; step++)
    {
        Save(args[1], step, logits);
        int next = ArgMax(logits);
        chosen.Add(next);
        if (step < Steps)
            logits = model.Forward(new[] { next });
    }
    File.WriteAllText(Path.Combine(args[1], "tokens.json"), JsonSerializer.Serialize(new Run(tokens, chosen.ToArray())));
    Console.WriteLine($"[q35-probe] ggml: {tokens.Length} prompt tokens -> {model.Tokenizer.Decode(chosen)}");
    return 0;
}
if (args.Length >= 3 && args[0] == "--native")
{
    Directory.CreateDirectory(args[2]);
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    int ubatch = args.Length > 3 ? int.Parse(args[3]) : 512;
    using var exec = new Qwen35CudaExecutor(modelPath, Math.Max(4096, reference.Prompt.Length + Steps + 64), ubatch);
    var logits = new float[exec.VocabSize];
    var chosen = new List<int>();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    exec.Forward(reference.Prompt, logits);
    double promptMs = sw.Elapsed.TotalMilliseconds;
    sw.Restart();
    for (int step = 0; step <= Steps; step++)
    {
        Save(args[2], step, logits);
        chosen.Add(ArgMax(logits));
        if (step < Steps)
            exec.Forward(new[] { reference.Chosen[step] }, logits);
    }
    File.WriteAllText(Path.Combine(args[2], "tokens.json"), JsonSerializer.Serialize(new Run(reference.Prompt, chosen.ToArray())));
    Console.WriteLine($"[q35-probe] native: prompt {promptMs:F0} ms ({reference.Prompt.Length} tokens, ubatch {ubatch}), " +
        $"{Steps} decode steps {sw.Elapsed.TotalMilliseconds / Steps:F1} ms/step");
    return 0;
}
if (args.Length >= 3 && args[0] == "--bench")
{
    // --bench <prompt tokens> <ubatch> [decode steps]: synthetic tokens, the engine only.
    int n = int.Parse(args[1]), ubatch = int.Parse(args[2]), decode = args.Length > 3 ? int.Parse(args[3]) : 32;
    using var exec = new Qwen35CudaExecutor(modelPath, n + decode + 16, ubatch);
    var rng = new Random(7);
    int[] tokens = Enumerable.Range(0, n).Select(_ => rng.Next(1000, 100000)).ToArray();
    var logits = new float[exec.VocabSize];
    exec.Forward(tokens.Take(Math.Min(ubatch, n)).ToArray(), logits);   // warm the kernels
    exec.Reset();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    exec.Forward(tokens, logits);
    double prefillMs = sw.Elapsed.TotalMilliseconds;
    sw.Restart();
    for (int i = 0; i < decode; i++)
        exec.Forward(new[] { ArgMax(logits) }, logits);
    Console.WriteLine($"[q35-probe] bench: prefill {n} tokens (ubatch {ubatch}) {prefillMs:F0} ms = {n / prefillMs * 1000:F0} tok/s; " +
        $"decode {sw.Elapsed.TotalMilliseconds / decode:F2} ms/token at {n} context");
    return 0;
}
if (args.Length == 2 && (args[0] == "--ggml-draft" || args[0] == "--native-draft"))
{
    // The DFlash drafter's block at every step of the reference run (TS_TEST_QWEN35_DRAFT names the
    // drafter): --ggml-draft drives the ggml model's draft head and writes drafts.json to the
    // reference dir; --native-draft replays the same anchors on the engine and compares.
    string draftPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT")
        ?? throw new ArgumentException("Set TS_TEST_QWEN35_DRAFT.");
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    int[] promptIds = reference.Prompt;
    int n = promptIds.Length;
    var blocks = new List<int[]>();
    var confs = new List<float[]>();
    if (args[0] == "--ggml-draft")
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
        using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda, draftModelPath: draftPath);
        var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
        var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
        int feat = spec.SpecFeatureSize, b = head.DraftBlockSize;
        var logits = new float[model.Config.VocabSize];
        var hAll = new float[(long)n * feat];
        spec.SpecForward(promptIds, hAll, logits, false);
        var hRows = new float[(long)n * feat];   // row k: the features of the token before prompt[k]
        Array.Copy(hAll, 0, hRows, feat, (long)(n - 1) * feat);
        head.DraftCatchUp(promptIds, hRows, 0);
        var carry = hAll[((n - 1) * feat)..];
        for (int t = 0; t < Steps; t++)
        {
            int anchor = reference.Chosen[t];
            var d = new int[b];
            var c = new float[b];
            head.DraftBlock(anchor, carry, n + t, d, c);
            blocks.Add(d);
            confs.Add(c);
            var h1 = new float[feat];
            spec.SpecForward(new[] { anchor }, h1, logits, false);
            head.DraftCatchUp(new[] { anchor }, carry, n + t);
            carry = h1;
        }
        File.WriteAllText(Path.Combine(args[1], "drafts.json"), JsonSerializer.Serialize(new Drafts(blocks.ToArray(), confs.ToArray())));
        Console.WriteLine($"[q35-probe] ggml drafts: {Steps} blocks of {b}");
        return 0;
    }
    var refDrafts = JsonSerializer.Deserialize<Drafts>(File.ReadAllText(Path.Combine(args[1], "drafts.json")))!;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, draftPath);
    var eng = exec.Engine;
    var lg = new float[exec.VocabSize];
    eng.Forward(promptIds, lg);
    int g = eng.DraftBlockSize - 1, same = 0, prefixSum = 0, refAccept = 0, natAccept = 0;
    double confDiff = 0;
    var blk = new[] { new int[g] };
    var cf = new[] { new float[g] };
    for (int t = 0; t < Steps; t++)
    {
        eng.Draft(new[] { eng.Active }, new[] { reference.Chosen[t] }, blk, cf);
        int[] r = refDrafts.Blocks[t];
        int prefix = 0;
        while (prefix < g && blk[0][prefix] == r[prefix])
            prefix++;
        prefixSum += prefix;
        for (int i = 0; i < g; i++)
        {
            same += blk[0][i] == r[i] ? 1 : 0;
            confDiff = Math.Max(confDiff, Math.Abs(cf[0][i] - refDrafts.Conf[t][i]));
        }
        // Drafts the greedy continuation would accept.
        int ra = 0, na = 0;
        while (ra < g && t + 1 + ra < reference.Chosen.Length && r[ra] == reference.Chosen[t + 1 + ra]) ra++;
        while (na < g && t + 1 + na < reference.Chosen.Length && blk[0][na] == reference.Chosen[t + 1 + na]) na++;
        refAccept += ra;
        natAccept += na;
        if (t < 4)
            Console.WriteLine($"  step {t}: ggml [{string.Join(",", r)}] native [{string.Join(",", blk[0])}]");
        eng.Forward(new[] { reference.Chosen[t] }, lg);
    }
    Console.WriteLine($"[q35-probe] drafts: {same}/{Steps * g} tokens equal, mean common prefix {(double)prefixSum / Steps:F2}, " +
        $"max conf diff {confDiff:F3}; accepted by the reference run: ggml {refAccept}, native {natAccept}");
    return 0;
}
if (args.Length == 2 && args[0] == "--clone")
{
    // A copied slot shares the source's attention blocks: decoding the source first (its writes
    // copy the shared tail block) must leave the copy decoding exactly as the source did.
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT"));
    var eng = exec.Engine;
    var lg = new float[exec.VocabSize];
    var a = eng.CreateSlot(0);
    eng.Active = a;
    eng.Forward(reference.Prompt, lg);
    var b = eng.CreateSlot(0);
    eng.CopySlot(a, b);
    var first = new List<float[]>();
    for (int t = 0; t < Steps; t++)
    {
        eng.Forward(new[] { reference.Chosen[t] }, lg);
        first.Add((float[])lg.Clone());
    }
    eng.Active = b;
    int same = 0;
    for (int t = 0; t < Steps; t++)
    {
        eng.Forward(new[] { reference.Chosen[t] }, lg);
        same += lg.AsSpan().SequenceEqual(first[t]) ? 1 : 0;
    }
    Console.WriteLine($"[q35-probe] clone: {same}/{Steps} decode steps bit-identical between a slot and its copy");
    return same == Steps ? 0 : 1;
}
if (args.Length == 2 && args[0] == "--mixed")
{
    // A prompt chunk and a verify in one forward (MixedVerify) against the same chunk forwarded and
    // the same windows verified on their own: the chunk's logits and the drawn tokens must match
    // bit for bit, and so must the decode that follows a commit of the same kept rows, for every
    // sequence. 2 x 8 verify rows take the FP4 GEMM on their own too, but split along K: run with
    // TS_Q35_GEMM_SPLITK=0 for one summation order (otherwise the verify rows differ in rounding).
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT"));
    var eng = exec.Engine;
    var lg = new float[exec.VocabSize];
    int[] full = reference.Prompt;
    int half = full.Length / 2;
    // Prompt slot: the first half forwarded; the chunk is the second half. Verify slots: the whole
    // prompt plus a few decoded tokens.
    TensorSharp.Cuda.Q35CudaEngine.Slot NewSlot(int[] tokens)
    {
        var s = eng.CreateSlot(0);
        eng.Active = s;
        eng.Forward(tokens, lg);
        return s;
    }
    var p1 = NewSlot(full[..half]);
    var v1 = new[] { NewSlot(full), NewSlot(full[..(full.Length - 3)]) };
    var p2 = eng.CreateSlot(0);
    eng.CopySlot(p1, p2);
    var v2 = new TensorSharp.Cuda.Q35CudaEngine.Slot[2];
    for (int i = 0; i < 2; i++)
    {
        v2[i] = eng.CreateSlot(0);
        eng.CopySlot(v1[i], v2[i]);
    }
    const int window = 8;
    var vt = new int[2 * window];
    for (int i = 0; i < vt.Length; i++)
        vt[i] = reference.Chosen[i % reference.Chosen.Length];
    int[] chunk = full[half..];
    // Separate: the chunk, then the verify.
    var logitsA = new float[exec.VocabSize];
    eng.Active = p1;
    eng.Forward(chunk, logitsA);
    var drawnA = new int[2 * window];
    eng.Verify(v1, window, vt, drawnA);
    // Fused.
    var logitsB = new float[exec.VocabSize];
    var drawnB = new int[2 * window];
    eng.MixedVerify(p2, chunk, logitsB, v2, window, vt, drawnB);
    bool logitsSame = logitsA.AsSpan().SequenceEqual(logitsB), drawnSame = drawnA.AsSpan().SequenceEqual(drawnB);
    // Commit the same rows on both sides, then decode every sequence on both sides.
    var kept = new[] { 3, 5 };
    eng.Commit(v1, window, kept);
    eng.Commit(v2, window, kept);
    int same = 0, total = 0, argmaxSame = 0, drawnDiff = 0;
    for (int i = 0; i < drawnA.Length; i++)
        drawnDiff += drawnA[i] != drawnB[i] ? 1 : 0;
    // The drafter's rings took the chunk's taps and the kept verify rows: its blocks must agree too.
    string drafts = "no drafter";
    bool draftsSame = true;
    if (exec.Draft != null)
    {
        int width = exec.Draft.BlockSize - 1;
        int[][] DraftOf(TensorSharp.Cuda.Q35CudaEngine.Slot[] slots)
        {
            var outTokens = new int[slots.Length][];
            var conf = new float[slots.Length][];
            for (int i = 0; i < slots.Length; i++)
            {
                outTokens[i] = new int[width];
                conf[i] = new float[width];
            }
            eng.Draft(slots, new[] { reference.Chosen[0], reference.Chosen[1], reference.Chosen[2] }, outTokens, conf);
            return outTokens;
        }
        var da = DraftOf(new[] { p1, v1[0], v1[1] });
        var db = DraftOf(new[] { p2, v2[0], v2[1] });
        int draftSame = 0;
        for (int i = 0; i < 3; i++)
            draftSame += da[i].AsSpan().SequenceEqual(db[i]) ? 1 : 0;
        drafts = $"drafts {draftSame}/3 identical";
        draftsSame = draftSame == 3;
    }
    foreach (var (a, b) in new[] { (p1, p2), (v1[0], v2[0]), (v1[1], v2[1]) })
    {
        var la = new float[exec.VocabSize];
        var lb = new float[exec.VocabSize];
        for (int t = 0; t < 6; t++)
        {
            eng.Active = a;
            eng.Forward(new[] { reference.Chosen[t] }, la);
            eng.Active = b;
            eng.Forward(new[] { reference.Chosen[t] }, lb);
            same += la.AsSpan().SequenceEqual(lb) ? 1 : 0;
            argmaxSame += ArgMax(la) == ArgMax(lb) ? 1 : 0;
            total++;
        }
    }
    Console.WriteLine($"[q35-probe] mixed: chunk logits {(logitsSame ? "bit-identical" : "DIFFER")}, drawn {(drawnSame ? "identical" : $"{drawnDiff}/{drawnA.Length} DIFFER")}, " +
        $"decode after commit {same}/{total} bit-identical, argmax {argmaxSame}/{total}, {drafts}");
    return logitsSame && drawnSame && same == total && draftsSame ? 0 : 1;
}
if (args.Length == 2 && args[0] == "--host")
{
    // A slot moved to host memory and back must decode exactly as before: once restored into a
    // fresh slot while its source still holds the blocks (shared by content key), once after the
    // source has written past them and been freed (blocks uploaded), and a checkpoint sharing the
    // same blocks restored beside it.
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT"));
    var eng = exec.Engine;
    var lg = new float[exec.VocabSize];
    // The checkpoint is taken 7 tokens short of the prompt (a copy of the slot then), and the
    // source then writes the rest of the prompt into the block they share.
    var a = eng.CreateSlot(0);
    eng.Active = a;
    eng.Forward(reference.Prompt[..^7], lg);
    var check = eng.CreateSlot(0);
    eng.CopySlot(a, check);
    var hostCheck = eng.CopyToHost(check);
    eng.FreeSlot(check);
    eng.Forward(reference.Prompt[^7..], lg);
    var hostA = eng.CopyToHost(a);
    var first = new List<float[]>();
    for (int t = 0; t < Steps; t++)
    {
        eng.Forward(new[] { reference.Chosen[t] }, lg);
        first.Add((float[])lg.Clone());
    }
    int Decode(TensorSharp.Cuda.Q35CudaEngine.Slot s)
    {
        eng.Active = s;
        int same = 0;
        for (int t = 0; t < Steps; t++)
        {
            eng.Forward(new[] { reference.Chosen[t] }, lg);
            same += lg.AsSpan().SequenceEqual(first[t]) ? 1 : 0;
        }
        return same;
    }
    var b = eng.CreateSlot(0);
    eng.CopyFromHost(hostA, b);
    int shared = Decode(b);
    eng.FreeSlot(b);
    eng.Active = eng.CreateSlot(0);
    eng.FreeSlot(a);
    var c = eng.CreateSlot(0);
    eng.CopyFromHost(hostA, c);
    var d = eng.CreateSlot(0);
    eng.CopyFromHost(hostCheck, d);
    int uploaded = Decode(c);
    // The checkpoint stopped 7 tokens short: feed them, then the same continuation.
    eng.Active = d;
    eng.Forward(reference.Prompt[^7..], lg);
    int check7 = Decode(d);
    long held = eng.HostBytesInUse;
    eng.FreeHostSlot(hostA);
    eng.FreeHostSlot(hostCheck);
    Console.WriteLine($"[q35-probe] host: {shared}/{Steps} bit-identical with blocks shared, {uploaded}/{Steps} with blocks uploaded, " +
        $"{check7}/{Steps} for a checkpoint restored beside it; {held >> 20} MiB pinned, {eng.HostBytesInUse} after release");
    return shared == Steps && uploaded == Steps && check7 == Steps && eng.HostBytesInUse == 0 ? 0 : 1;
}
if (args.Length == 2 && args[0] == "--batch-prefill")
{
    // Padded multi-sequence prefill against one sequence at a time: chunks of different lengths
    // (the shorter ones padded) must leave every sequence where its own prefill would.
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT"));
    var eng = exec.Engine;
    int[][] prompts = { reference.Prompt, reference.Prompt[..^37], reference.Prompt[..^90] };
    var solo = new List<float[]>();
    foreach (var p in prompts)
    {
        var s = eng.CreateSlot(0);
        eng.Active = s;
        var lg = new float[exec.VocabSize];
        eng.Forward(p, lg);
        for (int t = 0; t < 8; t++)
        {
            eng.Forward(new[] { reference.Chosen[t] }, lg);
            solo.Add((float[])lg.Clone());
        }
        eng.FreeSlot(s);
    }
    var slots = prompts.Select(_ => eng.CreateSlot(0)).ToArray();
    var outs = new float[prompts.Length][];
    eng.PrefillBatch(slots, prompts, outs);
    double worst = 1;
    int argAgree = 0, k = 0;
    for (int i = 0; i < prompts.Length; i++)
    {
        eng.Active = slots[i];
        var lg = new float[exec.VocabSize];
        for (int t = 0; t < 8; t++, k++)
        {
            eng.Forward(new[] { reference.Chosen[t] }, lg);
            worst = Math.Min(worst, Correlation(lg, solo[k]));
            argAgree += ArgMax(lg) == ArgMax(solo[k]) ? 1 : 0;
        }
    }
    Console.WriteLine($"[q35-probe] batch prefill: decode after it vs solo prefill: argmax {argAgree}/{k}, worst correlation {worst:F5}");
    return argAgree == k ? 0 : 1;
}
if (args.Length >= 2 && args[0] == "--vision")
{
    // --vision <mmproj> [encodes]: the Qwen-VL tower on the direct-CUDA tensor backend, a blank
    // 1344x768 screenshot encoded repeatedly (the first encode warms the backend).
    int encodes = args.Length > 2 ? int.Parse(args[2]) : 5;
    var alloc = new TensorSharp.Cuda.CudaAllocator(0);
    using var tower = new Qwen35VisionEncoder(args[1], alloc);
    const int h = 768, w = 1344;
    var rng = new Random(3);
    var pixels = new float[3 * h * w];
    for (int i = 0; i < pixels.Length; i++)
        pixels[i] = (float)(rng.NextDouble() * 2 - 1);
    var sw = System.Diagnostics.Stopwatch.StartNew();
    float[] last = null;
    for (int i = 0; i < encodes; i++)
    {
        sw.Restart();
        using var emb = tower.Encode(pixels, h, w);
        Console.WriteLine($"[q35-probe] vision encode {i}: {sw.Elapsed.TotalMilliseconds:F1} ms");
        last = emb.GetElementsAsFloat((int)emb.ElementCount());
    }
    // --vision <mmproj> <encodes> <file>: save the embeddings; a second file compares against it.
    if (args.Length > 3)
    {
        if (File.Exists(args[3]))
        {
            byte[] bytes = File.ReadAllBytes(args[3]);
            var other = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, other, 0, bytes.Length);
            double maxDiff = 0, mag = 0;
            for (int i = 0; i < last.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(last[i] - other[i]));
                mag = Math.Max(mag, Math.Abs(other[i]));
            }
            Console.WriteLine($"[q35-probe] vision embeddings vs {args[3]}: correlation {Correlation(last, other):F6}, max diff / max {maxDiff / mag:E2}");
        }
        else
        {
            var bytes = new byte[last.Length * 4];
            Buffer.BlockCopy(last, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(args[3], bytes);
        }
    }
    return 0;
}
if (args.Length >= 4 && args[0] == "--spec")
{
    // --spec <ref dir> <sequences> <window> [steps]: the speculative verify / commit against itself.
    // Path A decodes every sequence greedily through Verify with throwaway drafts, keeping only the
    // anchor; path B verifies A's own continuation, corrupted at a random draft, and keeps the
    // accepted prefix. Both verify the same rows per step, so every kept row must draw exactly A's
    // token. Sequence i runs the reference prompt minus its last i tokens (distinct positions).
    // With TS_TEST_QWEN35_DRAFT naming a DFlash2 drafter, path B verifies the drafter's blocks
    // instead (window = its block) and reports how many drafts the trunk accepts.
    var reference = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    int nSeq = int.Parse(args[2]), window = int.Parse(args[3]), steps = args.Length > 4 ? int.Parse(args[4]) : 32;
    using var exec = new Qwen35CudaExecutor(modelPath, 4096, 512, Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT"));
    var eng = exec.Engine;
    bool drafting = eng.HasDrafter;
    if (drafting)
        window = eng.DraftBlockSize;
    var draftBlocks = Enumerable.Range(0, nSeq).Select(_ => new int[Math.Max(1, window - 1)]).ToArray();
    var draftConf = Enumerable.Range(0, nSeq).Select(_ => new float[Math.Max(1, window - 1)]).ToArray();
    double draftMs = 0;
    var logits = new float[exec.VocabSize];
    TensorSharp.Cuda.Q35CudaEngine.Slot[] Prefill(int[] anchors)
    {
        var slots = new TensorSharp.Cuda.Q35CudaEngine.Slot[nSeq];
        for (int i = 0; i < nSeq; i++)
        {
            slots[i] = eng.CreateSlot(4096);
            eng.Active = slots[i];
            eng.Forward(reference.Prompt[..^i], logits);
            anchors[i] = ArgMax(logits);
        }
        return slots;
    }
    var tokens = new int[nSeq * window];
    var drawn = new int[nSeq * window];
    var kept = new int[nSeq];
    var outA = new List<int>[nSeq];
    var anchorsA = new int[nSeq];
    var slotsA = Prefill(anchorsA);
    for (int i = 0; i < nSeq; i++)
        outA[i] = new List<int> { anchorsA[i] };
    var swA = System.Diagnostics.Stopwatch.StartNew();
    for (int step = 0; step < steps + window; step++)
    {
        for (int i = 0; i < nSeq; i++)
        {
            tokens[i * window] = anchorsA[i];
            for (int t = 1; t < window; t++)
                tokens[i * window + t] = 1000 + 37 * t + i;
            kept[i] = 1;
        }
        eng.Verify(slotsA, window, tokens, drawn);
        eng.Commit(slotsA, window, kept);
        for (int i = 0; i < nSeq; i++)
        {
            anchorsA[i] = drawn[i * window];
            outA[i].Add(anchorsA[i]);
        }
    }
    double msA = swA.Elapsed.TotalMilliseconds / (steps + window);
    foreach (var s in slotsA)
        eng.FreeSlot(s);

    var rng = new Random(5);
    var anchorsB = new int[nSeq];
    var slotsB = Prefill(anchorsB);
    var produced = new int[nSeq];   // tokens of A's run that B has committed (the anchor at produced[i])
    int mismatches = 0, rounds = 0, accepted = 0;
    var swB = System.Diagnostics.Stopwatch.StartNew();
    while (produced.Min() < steps)
    {
        if (drafting)
        {
            var swD = System.Diagnostics.Stopwatch.StartNew();
            eng.Draft(slotsB, anchorsB, draftBlocks, draftConf);
            draftMs += swD.Elapsed.TotalMilliseconds;
        }
        // A sequence past `steps` rides along with throwaway drafts, unchecked.
        var live = new bool[nSeq];
        for (int i = 0; i < nSeq; i++)
        {
            live[i] = produced[i] < steps;
            if (live[i] && anchorsB[i] != outA[i][produced[i]])
                mismatches++;
            tokens[i * window] = anchorsB[i];
            int corrupt = rng.Next(1, window + 1);   // window: no corruption
            for (int t = 1; t < window; t++)
                tokens[i * window + t] = drafting ? draftBlocks[i][t - 1]
                    : !live[i] ? 1000 + t : t == corrupt ? outA[i][produced[i] + t] + 1 : outA[i][produced[i] + t];
        }
        eng.Verify(slotsB, window, tokens, drawn);
        for (int i = 0; i < nSeq; i++)
        {
            int m = 0;
            while (m < window - 1 && tokens[i * window + 1 + m] == drawn[i * window + m])
                m++;
            for (int j = 0; j <= m && live[i]; j++)
                if (drawn[i * window + j] != outA[i][produced[i] + j + 1])
                    mismatches++;
            kept[i] = m + 1;
            accepted += m;
        }
        eng.Commit(slotsB, window, kept);
        for (int i = 0; i < nSeq; i++)
        {
            anchorsB[i] = drawn[i * window + kept[i] - 1];
            produced[i] += kept[i];
        }
        rounds++;
    }
    double msB = swB.Elapsed.TotalMilliseconds / rounds;
    foreach (var s in slotsB)
        eng.FreeSlot(s);

    // Informational: A against plain single-row decoding (the q8_1 GEMV, not the verify's GEMM).
    int agree = 0, total = 0;
    var slot = eng.CreateSlot(4096);
    eng.Active = slot;
    eng.Forward(reference.Prompt, logits);
    for (int t = 0; t < Math.Min(steps, outA[0].Count); t++)
    {
        int next = ArgMax(logits);
        agree += next == outA[0][t] ? 1 : 0;
        total++;
        eng.Forward(new[] { outA[0][t] }, logits);
    }
    eng.FreeSlot(slot);
    Console.WriteLine($"[q35-probe] spec: {nSeq} x {window}: {rounds} rounds, {accepted} drafts accepted, {mismatches} mismatches; " +
        $"verify+commit {msA:F1} ms (path A step), {msB:F1} ms (path B round); A vs plain decode {agree}/{total}" +
        (drafting ? $"; drafter: {(double)accepted / (rounds * nSeq):F2} accepted per block, draft {draftMs / rounds:F2} ms" : ""));
    return mismatches == 0 ? 0 : 1;
}
if (args.Length == 3 && args[0] == "--compare")
{
    var a = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[1], "tokens.json")))!;
    var b = JsonSerializer.Deserialize<Run>(File.ReadAllText(Path.Combine(args[2], "tokens.json")))!;
    int agree = 0;
    double worstCorr = 1;
    for (int step = 0; step <= Steps; step++)
    {
        float[] x = Load(args[1], step), y = Load(args[2], step);
        double corr = Correlation(x, y);
        worstCorr = Math.Min(worstCorr, corr);
        double range = x.Max() - x.Min(), maxDiff = 0;
        for (int i = 0; i < x.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(x[i] - y[i]));
        var topA = Top(x, 5);
        var topB = Top(y, 5);
        bool same = a.Chosen[step] == b.Chosen[step];
        agree += same ? 1 : 0;
        Console.WriteLine($"step {step,2}: argmax {a.Chosen[step],6} {(same ? "==" : "!=")} {b.Chosen[step],6}  " +
            $"top5 overlap {topA.Intersect(topB).Count()}  corr {corr:F5}  maxdiff/range {maxDiff / range:F4}" +
            $"  margin {Margin(x):F3}");
    }
    Console.WriteLine($"[q35-probe] argmax agreement {agree}/{Steps + 1}, worst correlation {worstCorr:F5}");
    return agree >= Steps ? 0 : 1;
}
Console.Error.WriteLine("Usage: --ggml <dir> | --native <ref dir> <dir> [ubatch] | --compare <ref dir> <dir>");
return 2;

static int ArgMax(float[] v)
{
    int best = 0;
    for (int i = 1; i < v.Length; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static int[] Top(float[] v, int k) => v.Select((x, i) => (x, i)).OrderByDescending(p => p.x).Take(k).Select(p => p.i).ToArray();

static double Margin(float[] v)
{
    var t = v.OrderByDescending(x => x).Take(2).ToArray();
    return t[0] - t[1];
}

static double Correlation(float[] x, float[] y)
{
    double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < x.Length; i++)
    {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return sxy / Math.Sqrt(sxx * syy);
}

static void Save(string dir, int step, float[] logits)
{
    var bytes = new byte[logits.Length * 4];
    Buffer.BlockCopy(logits, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes(Path.Combine(dir, $"step_{step}.f32"), bytes);
}

static float[] Load(string dir, int step)
{
    byte[] bytes = File.ReadAllBytes(Path.Combine(dir, $"step_{step}.f32"));
    var v = new float[bytes.Length / 4];
    Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length);
    return v;
}

sealed record Run(int[] Prompt, int[] Chosen);

sealed record Drafts(int[][] Blocks, float[][] Conf);
