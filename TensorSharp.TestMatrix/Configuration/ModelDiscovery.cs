// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using TensorSharp.TestMatrix.Matrix;

namespace TensorSharp.TestMatrix.Configuration;

/// <summary>
/// Combines auto-discovery of GGUF files under a model directory with the
/// per-model overrides in <see cref="MatrixConfig.Models"/>. Auto-discovery
/// applies filename heuristics to detect family / mmproj / capabilities;
/// the explicit <see cref="ModelConfig"/> entries always win when present.
/// </summary>
public static class ModelDiscovery
{
    public static IReadOnlyList<ModelSpec> Discover(MatrixConfig cfg)
    {
        var byId = new Dictionary<string, ModelSpec>(StringComparer.OrdinalIgnoreCase);

        // Pass 1: explicit config models (authoritative)
        foreach (ModelConfig mc in cfg.Models)
        {
            if (!mc.Enabled)
            {
                continue;
            }
            if (string.IsNullOrWhiteSpace(mc.Gguf))
            {
                Console.Error.WriteLine($"[testmatrix] skipping model '{mc.Id}' (no gguf path)");
                continue;
            }
            string ggufPath = ResolveModelPath(cfg, mc.Gguf);
            if (!File.Exists(ggufPath))
            {
                Console.Error.WriteLine($"[testmatrix] skipping model '{mc.Id}' (file missing): {ggufPath}");
                continue;
            }
            string? mmprojPath = string.IsNullOrWhiteSpace(mc.Mmproj)
                ? null
                : ResolveModelPath(cfg, mc.Mmproj);
            if (mmprojPath is not null && !File.Exists(mmprojPath))
            {
                Console.Error.WriteLine($"[testmatrix] mmproj file missing for '{mc.Id}': {mmprojPath} (dropping mmproj)");
                mmprojPath = null;
            }
            byId[mc.Id] = new ModelSpec(
                Id: mc.Id,
                Family: mc.Family,
                DisplayName: string.IsNullOrWhiteSpace(mc.DisplayName) ? mc.Id : mc.DisplayName,
                GgufPath: ggufPath,
                MmprojPath: mmprojPath,
                SupportsImage: mc.SupportsImage,
                SupportsAudio: mc.SupportsAudio,
                SupportsVideo: mc.SupportsVideo,
                SupportsTools: mc.SupportsTools,
                SupportsThinking: mc.SupportsThinking);
        }

        // Pass 2: auto-discover GGUFs under model_dir
        string modelDir = Path.GetFullPath(cfg.ModelDir);
        if (Directory.Exists(modelDir))
        {
            foreach (string file in Directory.EnumerateFiles(modelDir, "*.gguf", SearchOption.TopDirectoryOnly))
            {
                string name = Path.GetFileName(file);
                if (LooksLikeMmproj(name))
                {
                    continue;
                }
                FamilyGuess guess = GuessFamily(name);
                string id = guess.Id;
                if (byId.ContainsKey(id))
                {
                    continue;
                }
                string? mmproj = FindMmproj(modelDir, guess.Family);
                byId[id] = new ModelSpec(
                    Id: id,
                    Family: guess.Family,
                    DisplayName: guess.DisplayName,
                    GgufPath: file,
                    MmprojPath: mmproj,
                    SupportsImage: guess.SupportsImage,
                    SupportsAudio: guess.SupportsAudio,
                    SupportsVideo: guess.SupportsVideo,
                    SupportsTools: guess.SupportsTools,
                    SupportsThinking: guess.SupportsThinking);
            }
        }
        else
        {
            Console.Error.WriteLine($"[testmatrix] model directory does not exist: {modelDir}");
        }

        return byId.Values.OrderBy(m => m.Id, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static string ResolveModelPath(MatrixConfig cfg, string path)
    {
        if (Path.IsPathRooted(path))
        {
            return path;
        }
        return Path.GetFullPath(Path.Combine(cfg.ModelDir, path));
    }

    private static bool LooksLikeMmproj(string fileName)
    {
        return fileName.Contains("mmproj", StringComparison.OrdinalIgnoreCase);
    }

    private static string? FindMmproj(string modelDir, string family)
    {
        // Prefer a same-family mmproj. Family substrings are tested in priority order.
        string[] needles = family.ToLowerInvariant() switch
        {
            "gemma4" => new[] { "gemma-4-mmproj", "gemma4-mmproj", "mmproj" },
            "gemma3" => new[] { "gemma-3-mmproj", "gemma3-mmproj", "mmproj" },
            "mistral3" => new[] { "ministral", "mistral", "mmproj" },
            "qwen35" or "qwen36" => new[] { "qwen", "mmproj" },
            "nemotron" => new[] { "nemotron", "mmproj" },
            _ => new[] { "mmproj" },
        };
        foreach (string needle in needles)
        {
            foreach (string file in Directory.EnumerateFiles(modelDir, "*mmproj*.gguf", SearchOption.TopDirectoryOnly))
            {
                string name = Path.GetFileName(file).ToLowerInvariant();
                if (name.Contains(needle, StringComparison.OrdinalIgnoreCase))
                {
                    return file;
                }
            }
        }
        return null;
    }

    private sealed record FamilyGuess(
        string Id,
        string Family,
        string DisplayName,
        bool SupportsImage,
        bool SupportsAudio,
        bool SupportsVideo,
        bool SupportsTools,
        bool SupportsThinking);

    private static FamilyGuess GuessFamily(string fileName)
    {
        string lower = fileName.ToLowerInvariant();
        string baseName = Path.GetFileNameWithoutExtension(fileName);

        // Gemma 4
        if (lower.Contains("gemma-4") || lower.Contains("gemma4"))
        {
            return new FamilyGuess(
                Id: baseName.ToLowerInvariant(),
                Family: "gemma4",
                DisplayName: baseName,
                SupportsImage: true,
                SupportsAudio: true,
                SupportsVideo: true,
                SupportsTools: true,
                SupportsThinking: true);
        }
        // Gemma 3
        if (lower.Contains("gemma-3") || lower.Contains("gemma3"))
        {
            return new FamilyGuess(baseName.ToLowerInvariant(), "gemma3", baseName,
                SupportsImage: true, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: false, SupportsThinking: false);
        }
        // GPT OSS
        if (lower.Contains("gpt-oss") || lower.Contains("gptoss"))
        {
            return new FamilyGuess(baseName.ToLowerInvariant(), "gptoss", baseName,
                SupportsImage: false, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: false, SupportsThinking: true);
        }
        // Nemotron-H (incl. Omni)
        if (lower.Contains("nemotron"))
        {
            bool omni = lower.Contains("omni");
            return new FamilyGuess(baseName.ToLowerInvariant(), "nemotron", baseName,
                SupportsImage: omni, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: true, SupportsThinking: true);
        }
        // Mistral 3 / Ministral
        if (lower.Contains("ministral") || lower.Contains("mistral-small-3") || lower.Contains("mistral3"))
        {
            return new FamilyGuess(baseName.ToLowerInvariant(), "mistral3", baseName,
                SupportsImage: true, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: false, SupportsThinking: false);
        }
        // Qwen 3.5 / 3.6 family
        if (lower.Contains("qwen3.5") || lower.Contains("qwen3.6")
            || lower.Contains("qwen3_5") || lower.Contains("qwen3_6")
            || lower.Contains("qwen35") || lower.Contains("qwen36"))
        {
            bool moe = lower.Contains("a3b") || lower.Contains("moe");
            string family = lower.Contains("qwen3.6") || lower.Contains("qwen36") ? "qwen36" : "qwen35";
            return new FamilyGuess(baseName.ToLowerInvariant(), family, baseName,
                SupportsImage: !moe, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: true, SupportsThinking: true);
        }
        // Qwen 3
        if (lower.Contains("qwen3"))
        {
            return new FamilyGuess(baseName.ToLowerInvariant(), "qwen3", baseName,
                SupportsImage: false, SupportsAudio: false, SupportsVideo: false,
                SupportsTools: true, SupportsThinking: true);
        }

        return new FamilyGuess(baseName.ToLowerInvariant(), "unknown", baseName,
            SupportsImage: false, SupportsAudio: false, SupportsVideo: false,
            SupportsTools: false, SupportsThinking: false);
    }
}
