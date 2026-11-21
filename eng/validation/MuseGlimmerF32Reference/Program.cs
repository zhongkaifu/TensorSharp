// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

// Muse-Glimmer's F32 reference for the real-weight continuation gate (InferenceWeb.Tests/MuseGlimmerRuntimeRealTests.cs).
//
// Every distinct legacy continuation the gate holds a lane to (InferenceWeb.Tests/Fixtures/MuseGlimmer/LegacyReal/*.json),
// teacher-forced from its golden prompt (Fixtures/MuseGlimmer/LlamaCpp/ref_text.json and ref_text_long.json): the top 10
// log-probabilities of each of its 128 steps under exact arithmetic. The managed backend runs every block-format product as the
// weight's exactly dequantized values times F32 rows (StandardDecoderModel.FloatProductsForTests, an internal seam), with an F32
// KV cache and F32 activations; there is no reduced-precision accumulation on that path. Each sequence runs the prompt but its
// last token, then that token and the continuation as one scored chunk (ForwardPromptRowsForTests), so row i predicts
// continuation token i. The reference does not depend on a lane; the sequences do (lanes' legacy continuations part at a few
// steps), so each distinct one is written once with the lanes that use it.
//
//   dotnet run -c Release --project eng/validation/MuseGlimmerF32Reference -- \
//     --model <dir>/Muse-Glimmer-30B-UD-IQ2_XXS.gguf --fixtures InferenceWeb.Tests/Fixtures/MuseGlimmer \
//     --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LegacyReal/f32_reference.json --commit $(git rev-parse --short HEAD)
//
// A run holds the model's quantized weights on the host and dequantizes each product's weight per step (about 4 minutes a
// short sequence on two cores, about an hour for each 4,664-token one). --only <selectors> (comma-separated) limits a run to
// those sequences: a prompt's name selects all of its sequences, name@backend the one that backend's lane holds (a lane key's
// first field, as in history-of-computing@ggml_cuda). Runs over disjoint
// selections can go in parallel, and --out <json> --merge <parts...> joins their outputs into one fixture (the arithmetic is
// deterministic, so the parts are what one run writes).
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

string? modelPath = null, fixtures = null, outPath = null, commit = "unknown", only = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": modelPath = args[++i]; break;
        case "--fixtures": fixtures = args[++i]; break;
        case "--out": outPath = args[++i]; break;
        case "--commit": commit = args[++i]; break;
        case "--only": only = args[++i]; break;
        case "--merge": return Merge(args[(i + 1)..], outPath);
        default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
    }
}
if (modelPath == null || fixtures == null || outPath == null)
{
    Console.Error.WriteLine("usage: --model <gguf> --fixtures <Fixtures/MuseGlimmer dir> --out <json> [--commit <sha>] [--only <name|name@backend>,...]");
    return 2;
}

// The golden prompts, the gate's corpus: the short set, then the long prompt.
var prompts = new List<(string Name, int[] Tokens)>();
foreach (string file in new[] { "ref_text.json", "ref_text_long.json" })
    foreach (JsonNode? r in JsonNode.Parse(File.ReadAllText(Path.Combine(fixtures, "LlamaCpp", file)))!["records"]!.AsArray())
        prompts.Add((r!["name"]!.GetValue<string>(), r["prompt_tokens"]!.AsArray().Select(t => t!.GetValue<int>()).ToArray()));

// Every lane's frozen legacy continuations: the distinct sequences, with the lanes that hold each, and the weights they name.
var sequences = new List<(string Name, int[] Prompt, int[] Tokens, List<string> Lanes)>();
string? weights = null;
foreach (string file in Directory.GetFiles(Path.Combine(fixtures, "LegacyReal"), "*.json").Where(f => !f.EndsWith("f32_reference.json")).OrderBy(f => f, StringComparer.Ordinal))
{
    JsonNode lane = JsonNode.Parse(File.ReadAllText(file))!;
    string sha = lane["model"]!["sha256"]!.GetValue<string>();
    if (weights != null && weights != sha) throw new InvalidOperationException($"{file} names other weights ({sha}) than {weights}");
    weights = sha;
    foreach (JsonNode? c in lane["continuations"]!.AsArray())
    {
        string name = c!["name"]!.GetValue<string>();
        int[] tokens = c["tokens"]!.AsArray().Select(t => t!.GetValue<int>()).ToArray();
        var prompt = prompts.First(p => p.Name == name);
        if (Sha(prompt.Tokens) != c["promptSha256"]!.GetValue<string>()) throw new InvalidOperationException($"{file} {name}: another prompt");
        var seen = sequences.FirstOrDefault(s => s.Name == name && s.Tokens.SequenceEqual(tokens));
        if (seen.Lanes != null) seen.Lanes.Add(lane["lane"]!.GetValue<string>());
        else sequences.Add((name, prompt.Tokens, tokens, new List<string> { lane["lane"]!.GetValue<string>() }));
    }
}
Console.WriteLine($"{sequences.Count} distinct legacy continuations over {prompts.Count} prompts; weights {weights}");

KvCacheDtypeConfig.Set(KvCacheDtype.F32);
using var model = (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(modelPath, BackendType.Cpu);
model.FloatProductsForTests = true;
var written = new JsonArray();
foreach (var seq in sequences)
{
    if (only != null && !only.Split(',').Any(o => o == seq.Name || seq.Lanes.Any(l => o == $"{seq.Name}@{l.Split('|')[0]}"))) continue;
    var clock = Stopwatch.StartNew();
    model.ResetKVCache();
    if (seq.Prompt.Length > 1) model.Forward(seq.Prompt[..^1]);
    int[] scored = seq.Prompt[^1..].Concat(seq.Tokens).ToArray();
    float[] rows = model.ForwardPromptRowsForTests(scored);
    int vocab = rows.Length / scored.Length;
    var top = new JsonArray();
    for (int i = 0; i < seq.Tokens.Length; i++)
    {
        double max = double.MinValue, sum = 0;
        for (int v = 0; v < vocab; v++) max = Math.Max(max, rows[i * vocab + v]);
        for (int v = 0; v < vocab; v++) sum += Math.Exp(rows[i * vocab + v] - max);
        double lse = max + Math.Log(sum);
        var step = new JsonArray();
        foreach (int v in Enumerable.Range(0, vocab).OrderByDescending(v => rows[i * vocab + v]).Take(10))
            step.Add(new JsonArray(v, Math.Round(rows[i * vocab + v] - lse, 6)));
        top.Add(step);
    }
    if (model.GraphRuns == 0) throw new InvalidOperationException("the steps did not run as graphs: the F32 products seam did not apply");
    written.Add(new JsonObject
    {
        ["name"] = seq.Name,
        ["promptSha256"] = Sha(seq.Prompt),
        ["tokensSha256"] = Sha(seq.Tokens),
        ["lanes"] = new JsonArray(seq.Lanes.Select(l => (JsonNode)l).ToArray()),
        ["top"] = top,
    });
    Console.WriteLine($"{seq.Name} ({string.Join(", ", seq.Lanes)}): {seq.Tokens.Length} steps in {clock.Elapsed.TotalSeconds:F0} s");
}
var doc = new JsonObject
{
    ["about"] = "The F32 reference of the Muse-Glimmer real-weight continuation gate: each distinct legacy continuation of " +
        "LegacyReal/*.json teacher-forced from its golden prompt, the top 10 log-probabilities of every step.",
    ["method"] = "managed backend, StandardDecoderModel.FloatProductsForTests (every block-format product: exactly dequantized " +
        "weights times F32 rows), F32 KV cache, F32 activations; the prompt but its last token, then that token and the " +
        "continuation as one scored chunk (row i predicts continuation token i)",
    ["producer"] = "eng/validation/MuseGlimmerF32Reference",
    ["provenance"] = $"TensorSharp {commit}, {RuntimeInfo()}",
    ["generatedUtc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
    ["model"] = new JsonObject { ["file"] = Path.GetFileName(modelPath), ["sha256"] = weights },
    ["sequences"] = written,
};
File.WriteAllText(outPath, doc.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
Console.WriteLine($"wrote {outPath}: {written.Count} sequences");
return 0;

// --out <fixture> --merge <part>...: the parts' sequences in one fixture, the first part's provenance; each sequence once.
static int Merge(string[] parts, string? outPath)
{
    if (outPath == null || parts.Length == 0) { Console.Error.WriteLine("usage: --out <json> --merge <part.json>..."); return 2; }
    JsonNode doc = JsonNode.Parse(File.ReadAllText(parts[0]))!;
    var all = new JsonArray();
    var seen = new HashSet<string>();
    foreach (string part in parts)
    {
        JsonNode p = JsonNode.Parse(File.ReadAllText(part))!;
        if (p["model"]!["sha256"]!.GetValue<string>() != doc["model"]!["sha256"]!.GetValue<string>())
            throw new InvalidOperationException($"{part} names other weights");
        foreach (JsonNode? s in p["sequences"]!.AsArray())
            if (seen.Add(s!["name"]!.GetValue<string>() + "/" + s["tokensSha256"]!.GetValue<string>())) all.Add(s.DeepClone());
    }
    doc["sequences"] = all;
    File.WriteAllText(outPath, doc.ToJsonString());
    Console.WriteLine($"wrote {outPath}: {all.Count} sequences from {parts.Length} parts");
    return 0;
}

static string Sha(int[] tokens)
{
    var bytes = new byte[tokens.Length * sizeof(int)];
    Buffer.BlockCopy(tokens, 0, bytes, 0, bytes.Length);
    return Convert.ToHexStringLower(SHA256.HashData(bytes));
}

static string RuntimeInfo() => $"{System.Runtime.InteropServices.RuntimeInformation.OSDescription.Trim()}, " +
    $"{System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}, .NET {Environment.Version}";
