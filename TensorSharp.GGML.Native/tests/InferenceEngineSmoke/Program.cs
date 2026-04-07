using System;
using System.IO;
using InferenceEngine;

static BackendType ParseBackend(string backend) => backend.ToLowerInvariant() switch
{
    "cpu" => BackendType.Cpu,
    "ggml_cpu" => BackendType.GgmlCpu,
    "ggml_metal" => BackendType.GgmlMetal,
    "cuda" or "ggml_cuda" => BackendType.GgmlCuda,
    _ => throw new ArgumentException("Unknown backend. Use cpu, ggml_cpu, ggml_metal, or ggml_cuda.")
};

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: InferenceEngineSmoke <model.gguf> [backend]");
    return 1;
}

string modelPath = args[0];
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model file not found: {modelPath}");
    return 1;
}

BackendType backend = ParseBackend(args.Length >= 2 ? args[1] : "ggml_cpu");

using var model = ModelBase.Create(modelPath, backend);
var tokenIds = model.Tokenizer.Encode("Hello", addSpecial: true);
float[] logits = model.Forward(tokenIds.ToArray());

if (logits.Length != model.Config.VocabSize)
{
    throw new InvalidOperationException(
        $"Unexpected logits length {logits.Length}; expected vocab size {model.Config.VocabSize}.");
}

int topToken = model.SampleGreedy(logits);
Console.WriteLine(
    $"InferenceEngineSmoke passed ({backend}) - vocab={model.Config.VocabSize}, tokens={tokenIds.Count}, topToken={topToken}");
return 0;
