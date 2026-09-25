// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies the JSON configuration-file layer shared by the CLI and the server:
/// <see cref="ConfigFileArgs.Expand"/> translates a <c>--config &lt;file&gt;</c>
/// into argv tokens, splices them ahead of the real command line so command-line
/// options win, and reports malformed input clearly. A few cases go end-to-end
/// through <see cref="ServerOptionsBuilder.Build"/> to confirm the merged args
/// parse and that override precedence holds through the real option parser.
/// </summary>
public class ConfigFileArgsTests : IDisposable
{
    private readonly string _dir;

    public ConfigFileArgsTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-configfile-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string WriteConfig(string json, string name = "config.json")
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, json);
        return path;
    }

    // ---- video companion keys, old and new ----------------------------------
    // Config keys ARE CLI flags (NormalizeFlag just prefixes '--'), so renaming the
    // companion flags renamed these keys too. Both spellings must expand, or every
    // config file written before the rename breaks.

    [Fact]
    public void Expand_LegacyWanCompanionKeys_StillExpandToFlags()
    {
        string cfg = WriteConfig("""
        {
          "model": "wan.gguf",
          "wan-vae": "vae.safetensors",
          "wan-te": "umt5.gguf",
          "wan-dit2": "low_noise.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Contains("--wan-vae", result);
        Assert.Contains("--wan-te", result);
        Assert.Contains("--wan-dit2", result);
    }

    [Fact]
    public void Expand_GenericVideoCompanionKeys_ExpandToFlags()
    {
        string cfg = WriteConfig("""
        {
          "model": "wan.gguf",
          "video-vae": "vae.safetensors",
          "video-text-encoder": "umt5.gguf",
          "video-dit2": "low_noise.gguf",
          "audio-vae": "audio_vae.safetensors"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Contains("--video-vae", result);
        Assert.Contains("--video-text-encoder", result);
        Assert.Contains("--video-dit2", result);
        Assert.Contains("--audio-vae", result);
    }

    [Fact]
    public void Expand_ShippedVideoConfigs_UseTheGenericKeys()
    {
        // Guard the repo's own presets: they were migrated to the generic keys, and a
        // stray legacy key here would be silently fine at runtime but inconsistent.
        string repoRoot = FindRepoRoot();
        if (repoRoot is null) return;   // running outside a source checkout

        foreach (string name in new[]
                 {
                     "wan-video-ti2v-5b.json",
                     "wan-video-ti2v-5b-turbo.json",
                     "wan-video-i2v-a14b.json",
                 })
        {
            string path = Path.Combine(repoRoot, "config", name);
            if (!File.Exists(path)) continue;
            string text = File.ReadAllText(path);
            Assert.DoesNotContain("\"wan-vae\"", text);
            Assert.DoesNotContain("\"wan-te\"", text);
            Assert.DoesNotContain("\"wan-dit2\"", text);
        }
    }

    // ---- every shipped config must be startable by BOTH hosts ----------------
    // config/README.md promises "Every file works with both hosts (only
    // host-recognized keys are used)". That promise has teeth on the server: an
    // unrecognized flag is not ignored there, it refuses to start. Config keys ARE
    // flags, so a CLI-only key in a shipped preset is a startup failure for every
    // server user of that file -- which is how minimax-h3-fl2va.json shipped with
    // "diffusion-steps" and "cfg" (the server spells steps --video-steps, and takes
    // no --cfg at all).
    //
    // Keep this set in step with the flags ServerOptionsBuilder actually reads. A
    // new server flag used by a new config has to be added here too; that is the
    // point -- the addition is where you notice whether the CLI spells it the same.
    private static readonly HashSet<string> ServerRecognizedConfigKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "mmproj", "backend", "max-tokens", "host", "port", "urls", "no-webui",
        "embeddings", "embedding-threads", "embedding-context-size",
        "no-prefix-cache",
        "temperature", "top-k", "top-p", "min-p", "seed", "stop", "sampling-precedence",
        "repeat-penalty", "repeat-last-n", "presence-penalty", "frequency-penalty",
        "continuous-batching", "no-continuous-batching", "paged-batching", "no-paged-batching",
        "prefill-chunk-size", "kv-cache-dtype", "gpu-device", "tp", "tp-node-id", "tp-peers",
        "paged-kv", "paged-kv-cache", "no-paged-kv", "no-paged-kv-cache", "paged-kv-block-size",
        "paged-kv-ram-mb", "paged-kv-ssd-dir", "paged-kv-ssd-mb", "paged-kv-quant-bits",
        "paged-kv-redis-url", "paged-kv-redis-ttl", "redis-url",
        "cpu-moe", "n-cpu-moe", "cpu-moe-threads",
        "spec", "no-spec", "spec-type", "spec-draft", "spec-pmin", "draft-model",
        "qwen-image-vae", "qwen-image-vl", "qwen-image-mmproj",
        "video-vae", "video-text-encoder", "video-te", "video-dit2", "audio-vae",
        "video-width", "video-height", "video-steps", "video-mode", "video-frames", "fps",
        "wan-vae", "wan-te", "wan-dit2", "width", "height",
        // Qwen-Image-2.1 LoRA plug-ins (LoraCliFlags.Flags): applied by the companion
        // pass and let through the unknown-option trap, spelled the same by the CLI.
        "lora", "lora-scale", "lora-config",
        "upload-max-mb", "upload-quota-mb", "upload-ttl-hours",
        "skills-dir", "skill", "list-skills", "no-skills", "skills-no-discovery",
        "skills-allow-exec", "skills-max-rounds", "skills-sandbox", "skills-allow-network",
        // Code execution. The server reads these because Program.cs strips them with
        // CodeExecOptions.Parse and ServerOptionsBuilder consumes CodeExecOptions'
        // SwitchFlags/ValueFlags before the unknown-argument trap, so a config may
        // carry them -- but nothing here knew that until config/agent-*.json needed
        // it. Both hosts share the one table in CodeExecOptions, so the CLI spells
        // every one of these the same way.
        "code-exec", "code-exec-allow-install", "code-exec-allow-network", "code-exec-unconfined",
        "code-exec-timeout", "code-exec-temperature", "code-exec-install-domains",
        "code-exec-install-index", "code-exec-shell", "code-exec-packages", "code-exec-max-output",
    };

    [Fact]
    public void ShippedConfigs_UseOnlyKeysTheServerAccepts()
    {
        string repoRoot = FindRepoRoot();
        if (repoRoot is null) return;   // running outside a source checkout
        string configDir = Path.Combine(repoRoot, "config");
        if (!Directory.Exists(configDir)) return;

        foreach (string path in Directory.GetFiles(configDir, "*.json"))
        {
            foreach (string key in TopLevelOptionKeys(path))
            {
                Assert.True(
                    ServerRecognizedConfigKeys.Contains(key),
                    $"config/{Path.GetFileName(path)}: option {Quote(key)} is not read by the server, "
                    + $"so --config on the server would fail with \"Unknown option '--{key}'\".");
            }
        }
    }

    // A file entry that names no URL cannot self-provision: on a machine without the
    // file the run stops at "path not found and no download URL was provided". The
    // shape matters as much as the presence -- an entry is understood only when its
    // source sits under "url"/"urls", and ConfigFileArgs silently ignores any other
    // field. minimax-h3-fl2va.json shipped naming its sources "repo"/"file": valid
    // JSON, obvious intent, and nothing downloaded.
    [Fact]
    public void ShippedConfigs_FileEntries_NameADownloadSource()
    {
        string repoRoot = FindRepoRoot();
        if (repoRoot is null) return;
        string configDir = Path.Combine(repoRoot, "config");
        if (!Directory.Exists(configDir)) return;

        var understood = new HashSet<string>(StringComparer.Ordinal) { "path", "url", "urls", "sha256" };

        foreach (string path in Directory.GetFiles(configDir, "*.json"))
        {
            using JsonDocument doc = ParseConfig(path);
            foreach (JsonProperty option in doc.RootElement.EnumerateObject())
            {
                if (option.Value.ValueKind != JsonValueKind.Object) continue;
                if (IsReserved(option.Name)) continue;

                string where = $"config/{Path.GetFileName(path)} option {Quote(option.Name)}";
                foreach (JsonProperty field in option.Value.EnumerateObject())
                {
                    Assert.True(
                        understood.Contains(field.Name),
                        $"{where} has field {Quote(field.Name)}, which ConfigFileArgs does not read "
                        + "-- a download entry is { path, urls[, sha256] }.");
                }

                Assert.True(
                    option.Value.TryGetProperty("urls", out _) || option.Value.TryGetProperty("url", out _),
                    $"{where} names no urls, so it cannot download on a machine that lacks the file.");
            }
        }
    }

    // The test above only sees entries that are already objects. A model file given as
    // a bare string path escapes it, and that is how fourteen configs shipped with no
    // source at all: fine on the machine that wrote them, "not found" everywhere else.
    // These are the options whose value is a model file to load.
    private static readonly HashSet<string> ModelFileConfigKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "mmproj", "draft-model",
        "qwen-image-vae", "qwen-image-vl", "qwen-image-mmproj",
        "video-vae", "video-text-encoder", "video-te", "video-dit2", "audio-vae",
        "wan-vae", "wan-te", "wan-dit2",
    };

    [Fact]
    public void ShippedConfigs_ModelFiles_AreDownloadEntries()
    {
        string repoRoot = FindRepoRoot();
        if (repoRoot is null) return;
        string configDir = Path.Combine(repoRoot, "config");
        if (!Directory.Exists(configDir)) return;

        foreach (string path in Directory.GetFiles(configDir, "*.json"))
        {
            using JsonDocument doc = ParseConfig(path);
            foreach (JsonProperty option in doc.RootElement.EnumerateObject())
            {
                if (!ModelFileConfigKeys.Contains(option.Name.TrimStart('-'))) continue;
                Assert.True(
                    option.Value.ValueKind == JsonValueKind.Object,
                    $"config/{Path.GetFileName(path)} option {Quote(option.Name)} is a bare path, so it cannot "
                    + "download on a machine that lacks the file -- write it as { \"path\", \"urls\", \"sha256\" }.");
            }
        }
    }

    private static string Quote(string value) => "\"" + value + "\"";

    private static bool IsReserved(string key) =>
        key.Equals("variables", StringComparison.OrdinalIgnoreCase)
        || key.Equals("vars", StringComparison.OrdinalIgnoreCase)
        || key.Equals("$schema", StringComparison.OrdinalIgnoreCase);

    /// <summary>The shipped configs are JSON with comments, read here exactly as
    /// <see cref="ConfigFileArgs"/> reads them.</summary>
    private static JsonDocument ParseConfig(string path) =>
        JsonDocument.Parse(
            File.ReadAllText(path),
            new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });

    private static IEnumerable<string> TopLevelOptionKeys(string path)
    {
        using JsonDocument doc = ParseConfig(path);
        return doc.RootElement.EnumerateObject()
            .Select(prop => prop.Name)
            .Where(name => !IsReserved(name))
            .ToList();
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    [Fact]
    public void Expand_NoConfigFlag_ReturnsArgsUnchanged()
    {
        var args = new[] { "--model", "a.gguf", "--backend", "ggml_cpu" };
        var result = ConfigFileArgs.Expand(args);
        Assert.Same(args, result);
    }

    [Fact]
    public void Expand_NullOrEmpty_ReturnsEmpty()
    {
        Assert.Empty(ConfigFileArgs.Expand(null));
        Assert.Empty(ConfigFileArgs.Expand(Array.Empty<string>()));
    }

    [Fact]
    public void Expand_StringAndNumberValues_BecomeFlagValuePairs()
    {
        string cfg = WriteConfig("""
        {
          "model": "a.gguf",
          "backend": "ggml_cuda",
          "max-tokens": 4096,
          "temperature": 0.7
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        // Order within the file is preserved; numbers keep their raw text.
        Assert.Equal(
            new[] { "--model", "a.gguf", "--backend", "ggml_cuda", "--max-tokens", "4096", "--temperature", "0.7" },
            result);
    }

    [Fact]
    public void Expand_BooleanTrue_BecomesBareSwitch_FalseIsSkipped()
    {
        string cfg = WriteConfig("""
        { "continuous-batching": true, "no-prefix-cache": false }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--continuous-batching" }, result);
    }

    [Fact]
    public void Expand_ArrayValue_BecomesRepeatedFlag()
    {
        string cfg = WriteConfig("""
        { "stop": ["</s>", "<|eot|>"] }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--stop", "</s>", "--stop", "<|eot|>" }, result);
    }

    [Fact]
    public void Expand_KeyWithLeadingDashes_IsAccepted()
    {
        string cfg = WriteConfig("""
        { "--backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_ConfigTokensComeBeforeCommandLine()
    {
        string cfg = WriteConfig("""
        { "model": "from-config.gguf", "backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg, "--backend", "ggml_cuda" });

        // File tokens first, then the pass-through command-line tokens.
        Assert.Equal(
            new[] { "--model", "from-config.gguf", "--backend", "ggml_cpu", "--backend", "ggml_cuda" },
            result);
    }

    [Fact]
    public void Expand_InlineConfigEqualsForm_IsSupported()
    {
        string cfg = WriteConfig("""
        { "backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config=" + cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_MultipleConfigFiles_LaterFileWins()
    {
        string a = WriteConfig("""{ "backend": "ggml_cpu", "max-tokens": 10 }""", "a.json");
        string b = WriteConfig("""{ "backend": "ggml_cuda" }""", "b.json");

        var result = ConfigFileArgs.Expand(new[] { "--config", a, "--config", b });

        // a's tokens, then b's tokens (b overrides a for backend under last-wins).
        Assert.Equal(
            new[] { "--backend", "ggml_cpu", "--max-tokens", "10", "--backend", "ggml_cuda" },
            result);
    }

    [Fact]
    public void Expand_MissingConfigFile_ThrowsFileNotFound()
    {
        var ex = Assert.Throws<FileNotFoundException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", Path.Combine(_dir, "nope.json") }));
        Assert.Contains("nope.json", ex.Message);
    }

    [Fact]
    public void Expand_InvalidJson_ThrowsArgumentException()
    {
        string cfg = WriteConfig("{ not valid json ");
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
        Assert.Contains(cfg, ex.Message);
    }

    [Fact]
    public void Expand_NonObjectRoot_ThrowsArgumentException()
    {
        string cfg = WriteConfig("[1, 2, 3]");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    [Fact]
    public void Expand_MissingValueForConfigFlag_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config" }));
        Assert.Contains("--config", ex.Message);
    }

    [Fact]
    public void Expand_JsonWithCommentsAndTrailingCommas_IsAccepted()
    {
        string cfg = WriteConfig("""
        {
          // the backend to host on
          "backend": "ggml_cpu",
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_ObjectWithoutPath_ThrowsArgumentException()
    {
        // An object value is interpreted as a downloadable-file spec; without a
        // "path" field it is not a valid spec.
        string cfg = WriteConfig("""
        { "model": { "foo": "bar" } }
        """);
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    // ----- Removed options -----
    // A config key IS a flag, so a stale config naming a removed option must fail with
    // the same message the command line gets. It is refused by NAME, whatever the value:
    // `false` expands to nothing and would otherwise slip past the hosts' own check.

    [Theory]
    [InlineData("qwen-image-lora", "\"Qwen-Image-Edit-Lightning-8steps.safetensors\"")]
    [InlineData("offload-cpu", "true")]
    [InlineData("offload-cpu", "false")]
    [InlineData("--offload-cpu", "true")]
    [InlineData("Qwen-Image-LoRA", "\"lora.safetensors\"")]
    public void Expand_RemovedOptionKey_FailsWithTheRemovalMessage(string key, string value)
    {
        string cfg = WriteConfig($$"""
        { "backend": "ggml_cpu", {{JsonQuote(key)}}: {{value}} }
        """);

        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));

        string flag = "--" + key.TrimStart('-').ToLowerInvariant();
        Assert.Contains($"{flag} was removed:", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Qwen-Image-2.1", ex.Message, StringComparison.Ordinal);
        Assert.Contains(Path.GetFileName(cfg), ex.Message, StringComparison.Ordinal);
        Assert.Equal(RemovedCliFlags.Describe(flag), ex.Message.Substring(ex.Message.IndexOf(flag, StringComparison.Ordinal)));
    }

    [Fact]
    public void Expand_RemovedOptionKey_IsRefusedBeforeAnythingIsDownloaded()
    {
        // The retired 2511 configs named the LoRA as a download spec. Refusing only
        // after expansion would fetch every file the stale config lists first, the
        // model included, and then refuse to use any of them.
        using var server = new TinyHttpServer();
        server.AddFile("/model.gguf", new byte[] { 1, 2, 3 });
        server.AddFile("/lora.safetensors", new byte[] { 4, 5, 6 });
        string model = Path.Combine(_dir, "model.gguf");
        string lora = Path.Combine(_dir, "lora.safetensors");
        string cfg = WriteConfig($$"""
        {
          "model": { "path": {{JsonQuote(model)}}, "urls": [ {{JsonQuote(server.UrlFor("/model.gguf"))}} ] },
          "qwen-image-lora": { "path": {{JsonQuote(lora)}}, "urls": [ {{JsonQuote(server.UrlFor("/lora.safetensors"))}} ] }
        }
        """);

        var ex = Assert.Throws<ArgumentException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));

        Assert.Contains("--qwen-image-lora was removed:", ex.Message, StringComparison.Ordinal);
        Assert.Equal(0, server.RequestCount("/model.gguf"));
        Assert.Equal(0, server.RequestCount("/lora.safetensors"));
        Assert.False(File.Exists(model));
        Assert.False(File.Exists(lora));
    }

    [Theory]
    [InlineData("--offload-cpu")]
    [InlineData("--qwen-image-lora=lora.safetensors")]
    public void Expand_RemovedFlagOnTheCommandLine_IsRefusedBeforeTheConfigDownloadsAnything(string removed)
    {
        // The hosts check their own command line only after expansion, so a removed
        // flag typed next to an otherwise valid --config must not first fetch the
        // multi-gigabyte files the configuration names.
        using var server = new TinyHttpServer();
        server.AddFile("/model.gguf", new byte[] { 1, 2, 3 });
        string model = Path.Combine(_dir, "model.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(model)}}, "urls": [ {{JsonQuote(server.UrlFor("/model.gguf"))}} ] } }
        """);

        var ex = Assert.Throws<ArgumentException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg, removed }, TextWriter.Null, interactiveProgress: false));

        Assert.Equal(RemovedCliFlags.Describe(removed), ex.Message);
        Assert.Equal(0, server.RequestCount("/model.gguf"));
        Assert.False(File.Exists(model));
    }

    // ----- End-to-end through the real server option parser -----

    [Fact]
    public void Build_WithConfigFile_AppliesSamplingDefaults()
    {
        string baseDir = Path.Combine(_dir, "base");
        Directory.CreateDirectory(baseDir);
        string cfg = WriteConfig("""
        { "temperature": 0.33, "top-k": 7, "stop": ["END"] }
        """);

        var options = ServerOptionsBuilder.Build(ConfigFileArgs.Expand(new[] { "--config", cfg }), baseDir);

        Assert.Equal(0.33f, options.DefaultSamplingConfig.Temperature);
        Assert.Equal(7, options.DefaultSamplingConfig.TopK);
        Assert.Equal(new[] { "END" }, options.DefaultSamplingConfig.StopSequences);
    }

    [Fact]
    public void Build_CommandLineOverridesConfigFile()
    {
        string baseDir = Path.Combine(_dir, "base2");
        Directory.CreateDirectory(baseDir);
        string cfg = WriteConfig("""
        { "temperature": 0.33 }
        """);

        // Config sets 0.33; the command line pins 0.9 and must win.
        var merged = ConfigFileArgs.Expand(new[] { "--config", cfg, "--temperature", "0.9" });
        var options = ServerOptionsBuilder.Build(merged, baseDir);

        Assert.Equal(0.9f, options.DefaultSamplingConfig.Temperature);
    }

    // ----- Variables -----

    [Fact]
    public void Expand_Variable_IsSubstitutedInStringValue()
    {
        string cfg = WriteConfig("""
        {
          "variables": { "root": "C:/models" },
          "backend": "ggml_cpu",
          "model": "${root}/a.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu", "--model", "C:/models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_Variable_CanReferenceAnotherVariable()
    {
        string cfg = WriteConfig("""
        {
          "variables": { "root": "C:/models", "gemma": "${root}/gemma" },
          "model": "${gemma}/model.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--model", "C:/models/gemma/model.gguf" }, result);
    }

    [Fact]
    public void Expand_Variable_FallsBackToEnvironmentVariable()
    {
        string name = "TS_CFG_TEST_ROOT_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "D:/envmodels");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + "}/a.gguf\" }");

            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "D:/envmodels/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_UndefinedVariable_ThrowsArgumentException()
    {
        string cfg = WriteConfig("""
        { "model": "${nope}/a.gguf" }
        """);
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
        Assert.Contains("nope", ex.Message);
    }

    // ${name:-fallback}. This is what lets the shipped configs name a model root
    // without hard-coding a drive letter: a Windows path like "C:/models/x.gguf"
    // is NOT rooted on Linux/macOS, so it would be treated as relative and glued
    // onto the config's own directory rather than failing loudly.

    [Fact]
    public void Expand_VariableDefault_UsedWhenNameIsUndefined()
    {
        string cfg = WriteConfig("""
        { "model": "${TS_CFG_NO_SUCH_VAR_12345:-../models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "../models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_VariableDefault_LosesToAnEnvironmentVariable()
    {
        string name = "TS_CFG_TEST_ROOT_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "D:/envmodels");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + ":-../models}/a.gguf\" }");
            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "D:/envmodels/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_VariableDefault_LosesToADeclaredVariable()
    {
        string cfg = WriteConfig("""
        { "variables": { "root": "/declared" }, "model": "${root:-../models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "/declared/a.gguf" }, result);
    }

    [Fact]
    public void Expand_VariableDefault_UsedWhenEnvironmentVariableIsEmpty()
    {
        // `set VAR=` on Windows leaves an empty value rather than removing it;
        // treating that as "set" would resolve the root to "" and silently build
        // a path off the filesystem root.
        string name = "TS_CFG_TEST_EMPTY_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + ":-../models}/a.gguf\" }");
            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "../models/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_VariableDefault_MayItselfReferenceAVariable()
    {
        string cfg = WriteConfig("""
        { "variables": { "base": "/base" }, "model": "${missing:-${base}/models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "/base/models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_CyclicVariable_ThrowsArgumentException()
    {
        string cfg = WriteConfig("""
        { "variables": { "a": "${b}", "b": "${a}" }, "model": "${a}" }
        """);
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    [Fact]
    public void Expand_Variable_SubstitutedInsideDownloadSpecPath()
    {
        // Pre-create the file so no download is attempted; only the path
        // substitution/resolution is under test.
        string modelDir = Path.Combine(_dir, "m");
        Directory.CreateDirectory(modelDir);
        string modelFile = Path.Combine(modelDir, "a.gguf");
        File.WriteAllText(modelFile, "stub");

        string cfg = WriteConfig($$"""
        {
          "variables": { "root": {{JsonQuote(modelDir)}} },
          "model": { "path": "${root}/a.gguf", "urls": ["http://127.0.0.1:9/never"] }
        }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal("--model", result[0]);
        Assert.Equal(Path.GetFullPath(modelFile), result[1]);
        Assert.Contains("using cached file", sink.ToString());
    }

    // ----- Auto-download -----

    [Fact]
    public void Expand_DownloadSpec_ExistingFile_UsesCachedWithoutDownloading()
    {
        string modelFile = Path.Combine(_dir, "cached.gguf");
        File.WriteAllText(modelFile, "already here");

        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(modelFile)}}, "urls": ["http://127.0.0.1:9/never"] } }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(new[] { "--model", Path.GetFullPath(modelFile) }, result);
        Assert.Contains("using cached", sink.ToString());
    }

    [Fact]
    public void Expand_DownloadSpec_MissingFileNoUrls_ThrowsFileNotFound()
    {
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(Path.Combine(_dir, "missing.gguf"))}} } }
        """);
        Assert.Throws<FileNotFoundException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
    }

    [Fact]
    public void Expand_DownloadSpec_DownloadsFromUrl_ThenReusesLocalCopy()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("hello-model-bytes");
        using var server = new TinyHttpServer();
        server.AddFile("/model.gguf", payload);

        string dest = Path.Combine(_dir, "downloaded", "model.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(dest)}}, "urls": [ {{JsonQuote(server.UrlFor("/model.gguf"))}} ] } }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(new[] { "--model", Path.GetFullPath(dest) }, result);
        Assert.True(File.Exists(dest));
        Assert.Equal(payload, File.ReadAllBytes(dest));
        string log = sink.ToString();
        Assert.Contains("downloading from", log);
        Assert.Contains("done", log);
        Assert.Equal(1, server.RequestCount("/model.gguf"));

        // Second run: the file exists, so no second request is made.
        var sink2 = new StringWriter();
        ConfigFileArgs.Expand(new[] { "--config", cfg }, sink2, interactiveProgress: false);
        Assert.Equal(1, server.RequestCount("/model.gguf"));
        Assert.Contains("using cached", sink2.ToString());
    }

    [Fact]
    public void Expand_DownloadSpec_FirstUrlFails_FallsBackToSecond()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("mirror-two-content");
        using var server = new TinyHttpServer();
        server.AddNotFound("/primary.gguf");
        server.AddFile("/backup.gguf", payload);

        string dest = Path.Combine(_dir, "fallback.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": {
            "path": {{JsonQuote(dest)}},
            "urls": [ {{JsonQuote(server.UrlFor("/primary.gguf"))}}, {{JsonQuote(server.UrlFor("/backup.gguf"))}} ]
          }
        }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(payload, File.ReadAllBytes(dest));
        string log = sink.ToString();
        Assert.Contains("source failed", log);
        Assert.Contains("trying next source", log);
    }

    [Fact]
    public void Expand_DownloadSpec_AllUrlsFail_ThrowsIOException()
    {
        using var server = new TinyHttpServer();
        server.AddNotFound("/a.gguf");

        string dest = Path.Combine(_dir, "never.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(dest)}}, "urls": [ {{JsonQuote(server.UrlFor("/a.gguf"))}} ] } }
        """);

        Assert.Throws<IOException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
        Assert.False(File.Exists(dest));
    }

    [Fact]
    public void Expand_DownloadSpec_Sha256Mismatch_IsTreatedAsFailure()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("some-bytes");
        using var server = new TinyHttpServer();
        server.AddFile("/m.gguf", payload);

        string dest = Path.Combine(_dir, "hashed.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": {
            "path": {{JsonQuote(dest)}},
            "urls": [ {{JsonQuote(server.UrlFor("/m.gguf"))}} ],
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
          }
        }
        """);

        Assert.Throws<IOException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
        // A failed transfer must not leave a file behind that a later run would trust.
        Assert.False(File.Exists(dest));
    }

    private static string JsonQuote(string s) =>
        System.Text.Json.JsonSerializer.Serialize(s);

    /// <summary>
    /// Minimal loopback HTTP server for exercising the downloader without a
    /// network dependency. Serves registered byte payloads with a Content-Length
    /// and can return 404 for URL-fallback tests.
    /// </summary>
    private sealed class TinyHttpServer : IDisposable
    {
        private readonly System.Net.HttpListener _listener = new();
        private readonly string _prefix;
        private readonly Dictionary<string, byte[]> _files = new();
        private readonly HashSet<string> _notFound = new();
        private readonly Dictionary<string, int> _hits = new();
        private readonly object _gate = new();

        public TinyHttpServer()
        {
            int port = GetFreePort();
            _prefix = $"http://127.0.0.1:{port}/";
            _listener.Prefixes.Add(_prefix);
            _listener.Start();
            System.Threading.Tasks.Task.Run(Loop);
        }

        public void AddFile(string path, byte[] content)
        {
            lock (_gate) { _files[path] = content; _hits[path] = 0; }
        }

        public void AddNotFound(string path)
        {
            lock (_gate) { _notFound.Add(path); _hits[path] = 0; }
        }

        public string UrlFor(string path) => _prefix.TrimEnd('/') + path;

        public int RequestCount(string path)
        {
            lock (_gate) { return _hits.TryGetValue(path, out int n) ? n : 0; }
        }

        private void Loop()
        {
            while (_listener.IsListening)
            {
                System.Net.HttpListenerContext ctx;
                try { ctx = _listener.GetContext(); }
                catch { return; }

                string path = ctx.Request.Url.AbsolutePath;
                byte[] body = null;
                bool notFound;
                lock (_gate)
                {
                    if (_hits.ContainsKey(path)) _hits[path]++;
                    notFound = _notFound.Contains(path);
                    if (!notFound) _files.TryGetValue(path, out body);
                }

                if (notFound || body == null)
                {
                    ctx.Response.StatusCode = 404;
                    ctx.Response.Close();
                    continue;
                }

                ctx.Response.StatusCode = 200;
                ctx.Response.ContentLength64 = body.Length;
                ctx.Response.OutputStream.Write(body, 0, body.Length);
                ctx.Response.OutputStream.Close();
            }
        }

        private static int GetFreePort()
        {
            var l = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
            l.Start();
            int port = ((System.Net.IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        public void Dispose()
        {
            try { _listener.Stop(); } catch { }
            try { _listener.Close(); } catch { }
        }
    }

    [Theory]
    [InlineData("lora-scale", "[0.5, 0.9]")]
    [InlineData("lora-config", "[\"a.json\", \"b.json\"]")]
    public void Expand_ALoraScaleOrConfigArray_IsRefused(string key, string value)
    {
        // Each binds to the one --lora before it; an array would bind every value to the last.
        string config = Path.Combine(_dir, "loras.json");
        File.WriteAllText(config, "{ \"lora\": [\"a.safetensors\", \"b.safetensors\"], \"" + key + "\": " + value + " }");

        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", config }, TextWriter.Null, false));
        Assert.Contains("cannot be an array", ex.Message, StringComparison.Ordinal);
    }
}
