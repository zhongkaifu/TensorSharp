// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Cli;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// The <c>--lora</c> / <c>--lora-scale</c> / <c>--lora-config</c> surface shared by both
/// hosts (<see cref="LoraCliFlags"/>): ordered binding, the joined spelling, errors that
/// never ignore a flag, the <c>TS_LORAS</c> environment channel, plug-in manifest
/// expansion, and the wiring into the usage pages and the server's option passes.
/// Only local temp files are used; nothing is downloaded.
/// </summary>
public sealed class LoraCliFlagsTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public LoraCliFlagsTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-lora-cli-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string Touch(string relative)
    {
        string path = Path.Combine(_dir, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        return path;
    }

    private string Write(string relative, string text)
    {
        string path = Path.Combine(_dir, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    private static string J(string s) => JsonSerializer.Serialize(s);

    // ---- Parse --------------------------------------------------------------------------

    [Fact]
    public void Parse_BindsScaleAndConfigToThePrecedingLora()
    {
        var specs = LoraCliFlags.Parse(new[]
        {
            "--lora", "a.safetensors", "--lora-scale", "0.5",
            "--lora", "b.safetensors", "--lora-config", "b.json",
            "--lora", "c.safetensors",
        });

        Assert.Equal(new[]
        {
            new LoraSpec("a.safetensors", 0.5f, null),
            new LoraSpec("b.safetensors", null, "b.json"),
            new LoraSpec("c.safetensors", null, null),
        }, specs);
    }

    [Fact]
    public void Parse_ALaterValueReplacesAnEarlierOneForTheSameLora()
    {
        var specs = LoraCliFlags.Parse(new[]
        {
            "--lora", "a.safetensors", "--lora-scale", "0.3", "--lora-config", "x.json",
            "--lora-scale", "0.9", "--lora-config", "y.json",
        });

        Assert.Equal(new LoraSpec("a.safetensors", 0.9f, "y.json"), Assert.Single(specs));
    }

    [Fact]
    public void Parse_AcceptsTheJoinedSpelling()
    {
        var specs = LoraCliFlags.Parse(new[]
        {
            "--lora=a.safetensors", "--lora-scale=0.25", "--lora-config=cfg=1.json",
            "--lora", "b.safetensors", "--lora-scale=-1.5",
        });

        Assert.Equal(new[]
        {
            // Only the first '=' separates the flag; the value keeps the rest.
            new LoraSpec("a.safetensors", 0.25f, "cfg=1.json"),
            // A negative strength subtracts the adapter; it is a valid request.
            new LoraSpec("b.safetensors", -1.5f, null),
        }, specs);
    }

    [Fact]
    public void Parse_CopiesEveryOtherArgumentToRemainingInOrder()
    {
        var remaining = new List<string>();
        var specs = LoraCliFlags.Parse(new[]
        {
            "--model", "m.gguf", "--lora", "a.safetensors", "--prompt", "a cat",
            "--lora-scale", "2", "--lora-foo", "--loras", "--lora-config=c.json", "--steps", "6",
        }, remaining);

        Assert.Equal(new[] { "--model", "m.gguf", "--prompt", "a cat", "--lora-foo", "--loras", "--steps", "6" }, remaining);
        Assert.Equal(new LoraSpec("a.safetensors", 2f, "c.json"), Assert.Single(specs));
    }

    [Fact]
    public void Parse_MatchesTheFlagsCaseInsensitively()
    {
        // As the server's parser and the other shared flag tables do: a spelling one pass
        // accepts must not be ignored by another.
        var specs = LoraCliFlags.Parse(new[] { "--LoRA", "a.safetensors", "--LORA-SCALE=0.5", "--Lora-Config", "c.json" });

        Assert.Equal(new LoraSpec("a.safetensors", 0.5f, "c.json"), Assert.Single(specs));
    }

    [Fact]
    public void Parse_NoLoraFlags_ReturnsAnEmptyListAndPassesEverythingThrough()
    {
        var args = new[] { "--model", "m.gguf", "--qwen-image-lora", "x.safetensors" };
        var remaining = new List<string>();

        Assert.Empty(LoraCliFlags.Parse(args, remaining));
        Assert.Equal(args, remaining);
        Assert.Empty(LoraCliFlags.Parse(Array.Empty<string>()));
    }

    [Theory]
    [InlineData(new[] { "--lora" }, "--lora requires a value")]
    [InlineData(new[] { "--lora=" }, "--lora requires a value")]
    [InlineData(new[] { "--lora", "   " }, "--lora requires a value")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale" }, "--lora-scale requires a value")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-config=" }, "--lora-config requires a value")]
    [InlineData(new[] { "--lora-scale", "0.5", "--lora", "a.safetensors" }, "none precedes it")]
    [InlineData(new[] { "--lora-config", "c.json" }, "none precedes it")]
    [InlineData(new[] { "--lora-scale=1" }, "none precedes it")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale", "strong" }, "finite number")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale", "NaN" }, "finite number")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale", "Infinity" }, "finite number")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale", "1e39" }, "finite number")]
    [InlineData(new[] { "--lora", "a.safetensors", "--lora-scale", "0,5" }, "finite number")]
    public void Parse_RefusesMalformedInputRatherThanIgnoringIt(string[] args, string fragment)
    {
        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Parse(args));
        Assert.Contains(fragment, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Parse_ScaleBeforeAnyLora_NamesBothFlags()
    {
        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Parse(new[] { "--lora-scale", "0.5" }));
        Assert.Contains("--lora-scale", ex.Message, StringComparison.Ordinal);
        Assert.Contains("preceding --lora", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--lora", true)]
    [InlineData("--lora=a.safetensors", true)]
    [InlineData("--lora-scale", true)]
    [InlineData("--lora-scale=0.5", true)]
    [InlineData("--lora-config", true)]
    [InlineData("--lora-config=c.json", true)]
    [InlineData("--loras", false)]
    [InlineData("--lora-foo", false)]
    [InlineData("--LORA", true)]
    [InlineData("--Lora-Scale=1", true)]
    [InlineData("--qwen-image-lora", false)]
    [InlineData("lora", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void IsLoraFlag_RecognizesTheThreeFlagsInAnyCase(string arg, bool expected)
    {
        Assert.Equal(expected, LoraCliFlags.IsLoraFlag(arg));
    }

    [Fact]
    public void Flags_AreTheThreeConstants()
    {
        Assert.Equal(new[] { "--lora", "--lora-scale", "--lora-config" }, LoraCliFlags.Flags);
        Assert.Equal("TS_LORAS", LoraCliFlags.EnvironmentVariable);
        Assert.Equal("qwen-image-2.1-lora", LoraCliFlags.PluginType);
    }

    // ---- the TS_LORAS environment channel ------------------------------------------------

    [Fact]
    public void Json_RoundTripsEveryField()
    {
        var specs = new List<LoraSpec>
        {
            new(Path.Combine(_dir, "a b", "Ünïcode \"quoted\".safetensors"), 0.75f, Path.Combine(_dir, "cfg.json")),
            new(Path.Combine(_dir, "plain.safetensors")),
            new(Path.Combine(_dir, "neg.safetensors"), -0.5f),
            new(Path.Combine(_dir, "cfg-only.safetensors"), null, Path.Combine(_dir, "adapter_config.json")),
        };

        string json = LoraCliFlags.ToJson(specs);

        Assert.Equal(specs, LoraCliFlags.FromJson(json));
    }

    [Fact]
    public void ToJson_OmitsAbsentScaleAndConfig()
    {
        string json = LoraCliFlags.ToJson(new[] { new LoraSpec("w.safetensors") });

        using var doc = JsonDocument.Parse(json);
        var item = Assert.Single(doc.RootElement.EnumerateArray().ToList());
        Assert.Equal("w.safetensors", item.GetProperty("path").GetString());
        Assert.False(item.TryGetProperty("scale", out _));
        Assert.False(item.TryGetProperty("config", out _));
        Assert.Equal("[]", LoraCliFlags.ToJson(Array.Empty<LoraSpec>()));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("[]")]
    public void FromJson_EmptyOrUnset_IsNoLora(string json)
    {
        Assert.Empty(LoraCliFlags.FromJson(json));
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{\"path\": \"a.safetensors\"}")]
    [InlineData("[{}]")]
    [InlineData("[1]")]
    [InlineData("[{\"path\": 5}]")]
    [InlineData("[{\"path\": null}]")]
    [InlineData("[{\"path\": \"a.safetensors\", \"scale\": \"strong\"}]")]
    [InlineData("[{\"path\": \"a.safetensors\", \"config\": 3}]")]
    public void FromJson_MalformedValue_NamesTheVariable(string json)
    {
        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.FromJson(json));
        Assert.Contains(LoraCliFlags.EnvironmentVariable, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Describe_ListsFileNamesScalesAndConfigs()
    {
        string text = LoraCliFlags.Describe(new[]
        {
            new LoraSpec("/models/loras/turbo.safetensors", 0.7f, "/configs/turbo.json"),
            new LoraSpec("/models/loras/style.safetensors"),
            new LoraSpec("/models/loras/third.safetensors", 1.23456f),
        });

        Assert.Equal("turbo.safetensors x0.7 (config turbo.json), style.safetensors, third.safetensors x1.235", text);
        Assert.Equal("", LoraCliFlags.Describe(Array.Empty<LoraSpec>()));
    }

    // ---- Resolve: files and plug-in manifests --------------------------------------------

    [Fact]
    public void Resolve_ReturnsNormalizedAbsolutePaths()
    {
        string weights = Touch("w/lora.safetensors");
        string config = Write("c/adapter_config.json", "{ \"lora_alpha\": 8 }");
        string indirect = Path.Combine(_dir, "w", "..", "w", "lora.safetensors");

        var resolved = LoraCliFlags.Resolve(new[] { new LoraSpec(indirect, 0.5f, config), new LoraSpec(weights) });

        Assert.Equal(new LoraSpec(Path.GetFullPath(weights), 0.5f, Path.GetFullPath(config)), resolved[0]);
        Assert.Equal(new LoraSpec(Path.GetFullPath(weights)), resolved[1]);
    }

    [Fact]
    public void Resolve_MissingWeights_ThrowsFileNotFound()
    {
        string missing = Path.Combine(_dir, "missing.safetensors");

        var ex = Assert.Throws<FileNotFoundException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(missing) }));

        Assert.Equal(missing, ex.FileName);
        Assert.Contains("--lora", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_MissingConfig_ThrowsFileNotFound()
    {
        string weights = Touch("lora.safetensors");
        string missing = Path.Combine(_dir, "missing.json");

        var ex = Assert.Throws<FileNotFoundException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(weights, null, missing) }));

        Assert.Equal(missing, ex.FileName);
        Assert.Contains("--lora-config", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_MissingJsonLora_IsAMissingFileNotAManifestError()
    {
        string missing = Path.Combine(_dir, "no-such-plugin.json");

        Assert.Throws<FileNotFoundException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(missing) }));
    }

    [Fact]
    public void Resolve_PluginManifest_BecomesItsWeightsWithItselfAsTheConfig()
    {
        string weights = Touch("models/loras/turbo.safetensors");
        string manifest = Write("config/lora/turbo.json", $$"""
        {
          // comments and trailing commas are allowed, as in every TensorSharp config
          "type": "qwen-image-2.1-lora",
          "variables": { "root": "../../models/loras" },
          "weights": { "path": "${root}/turbo.safetensors", "urls": ["https://example.invalid/never-fetched"], "sha256": "00" },
          "scale": 0.8,
          "sampling": { "steps": 2, "sigmas": [1.0, 0.5] },
        }
        """);

        var resolved = Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest, 0.5f) }));

        Assert.Equal(Path.GetFullPath(weights), resolved.Path);
        Assert.Equal(Path.GetFullPath(manifest), resolved.ConfigPath);
        // The command line's strength is kept; the manifest's "scale" is read later from the config.
        Assert.Equal(0.5f, resolved.Scale);
    }

    [Fact]
    public void Resolve_PluginManifest_AcceptsAPlainRelativeWeightsPath()
    {
        string weights = Touch("plugins/w.safetensors");
        string manifest = Write("plugins/p.JSON", """{ "type": "qwen-image-2.1-lora", "weights": "w.safetensors" }""");

        var resolved = Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));

        Assert.Equal(Path.GetFullPath(weights), resolved.Path);
        Assert.Equal(Path.GetFullPath(manifest), resolved.ConfigPath);
        Assert.Null(resolved.Scale);
    }

    [Fact]
    public void Resolve_PluginManifest_WithAbsoluteVariablePath()
    {
        string weights = Touch("elsewhere/abs.safetensors");
        string manifest = Write("m/abs.json", $$"""
        {
          "type": "qwen-image-2.1-lora",
          "variables": { "root": {{J(Path.GetDirectoryName(weights)!)}} },
          "weights": { "path": "${root}/abs.safetensors" }
        }
        """);

        var resolved = Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));

        Assert.Equal(Path.GetFullPath(weights), resolved.Path);
    }

    [Fact]
    public void Resolve_PluginManifest_ForwardsAThirdPartyConfig()
    {
        string weights = Touch("pdd/bundle.safetensors");
        string pdd = Write("pdd/pdd_config.json", """{ "pdd_num_steps": 1, "pdd_sigmas": [1.0, 0.0] }""");
        string manifest = Write("pdd-plugin.json", """
        {
          "type": "qwen-image-2.1-lora",
          "variables": { "root": "pdd" },
          "weights": { "path": "${root}/bundle.safetensors" },
          "config": { "path": "${root}/pdd_config.json" }
        }
        """);

        var resolved = Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));

        Assert.Equal(Path.GetFullPath(weights), resolved.Path);
        Assert.Equal(Path.GetFullPath(pdd), resolved.ConfigPath);
    }

    [Fact]
    public void Resolve_PluginManifest_ForwardingAConfig_CarriesItsStrength()
    {
        Touch("w.safetensors");
        Write("pdd_config.json", "{}");
        string manifest = Write("scaled.json", """
        { "type": "qwen-image-2.1-lora", "weights": "w.safetensors", "config": "pdd_config.json", "scale": 0.5 }
        """);

        Assert.Equal(0.5f, Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) })).Scale);
        // An explicit --lora-scale still wins over the plug-in's default.
        Assert.Equal(0.25f, Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest, 0.25f) })).Scale);
    }

    [Theory]
    [InlineData("\"alpha\": 64")]
    [InlineData("\"use_rslora\": true")]
    public void Resolve_PluginManifest_ForwardingAConfig_RefusesSettingsTheConfigWouldShadow(string setting)
    {
        Touch("w.safetensors");
        Write("pdd_config.json", "{}");
        string manifest = Write("shadow.json",
            "{ \"type\": \"qwen-image-2.1-lora\", \"weights\": \"w.safetensors\", \"config\": \"pdd_config.json\", " + setting + " }");

        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
        Assert.Contains("forwarded", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_APluginGivenAsLoraConfig_HandsOnItsForwardedConfigAndStrength()
    {
        string weights = Touch("pdd/bundle.safetensors");
        string pdd = Write("pdd/pdd_config.json", """{ "pdd_num_steps": 1, "pdd_sigmas": [1.0, 0.0] }""");
        string manifest = Write("forwarding.json", """
        { "type": "qwen-image-2.1-lora", "weights": "pdd/bundle.safetensors", "config": "pdd/pdd_config.json", "scale": 0.5 }
        """);

        var resolved = Assert.Single(LoraCliFlags.Resolve(new[] { new LoraSpec(weights, null, manifest) }));

        Assert.Equal(Path.GetFullPath(pdd), resolved.ConfigPath);
        Assert.Equal(0.5f, resolved.Scale);
    }

    [Theory]
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", ")]              // truncated JSON
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", \"weights\": \"w.safetensors\", \"config\": \"c.json\", \"scale\": \"high\" }")]
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", \"weights\": \"w.safetensors\", \"config\": \"c.json\", \"scale\": 1e39 }")]
    public void Resolve_AMalformedPlugin_IsAConfigurationError(string json)
    {
        Touch("w.safetensors");
        Write("c.json", "{}");
        string manifest = Write("bad.json", json);

        Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
    }

    [Fact]
    public void Resolve_PluginManifest_WithBothConfigAndSampling_Throws()
    {
        Touch("w.safetensors");
        Write("pdd_config.json", "{}");
        string manifest = Write("both.json", """
        {
          "type": "qwen-image-2.1-lora",
          "weights": "w.safetensors",
          "config": "pdd_config.json",
          "sampling": { "steps": 4 }
        }
        """);

        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
        Assert.Contains("\"sampling\" cannot sit beside a forwarded \"config\"", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_PluginManifestPlusLoraConfig_Throws()
    {
        Touch("w.safetensors");
        string extra = Write("adapter_config.json", """{ "lora_alpha": 4 }""");
        string manifest = Write("p.json", """{ "type": "qwen-image-2.1-lora", "weights": "w.safetensors" }""");

        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest, null, extra) }));
        Assert.Contains("--lora-config", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{ "lora_alpha": 16, "r": 8 }""")]
    [InlineData("""{ "type": "flux-lora", "weights": "w.safetensors" }""")]
    [InlineData("""{ "type": 7, "weights": "w.safetensors" }""")]
    [InlineData("""[ "w.safetensors" ]""")]
    public void Resolve_JsonThatIsNotAPlugin_Throws(string json)
    {
        Touch("w.safetensors");
        string manifest = Write("not-a-plugin.json", json);

        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
        Assert.Contains("qwen-image-2.1-lora", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_PluginManifestWithoutWeights_Throws()
    {
        string manifest = Write("no-weights.json", """{ "type": "qwen-image-2.1-lora", "scale": 1.0 }""");

        var ex = Assert.Throws<ArgumentException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
        Assert.Contains("\"weights\"", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_PluginManifestWhoseWeightsAreMissingAndHaveNoUrl_ThrowsFileNotFound()
    {
        string manifest = Write("dangling.json", """
        { "type": "qwen-image-2.1-lora", "weights": { "path": "nowhere/w.safetensors" } }
        """);

        Assert.Throws<FileNotFoundException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
    }

    [Fact]
    public void Resolve_PluginManifestWithAPlainMissingWeightsPath_ThrowsFileNotFound()
    {
        string manifest = Write("dangling-plain.json", """{ "type": "qwen-image-2.1-lora", "weights": "nowhere.safetensors" }""");

        var ex = Assert.Throws<FileNotFoundException>(() => LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }));
        Assert.Equal(Path.Combine(_dir, "nowhere.safetensors"), ex.FileName);
    }

    [Fact]
    public void ResolveFileEntry_ReportsAbsentKeysAndMalformedEntries()
    {
        Touch("w.safetensors");
        string doc = Write("entries.json", """{ "weights": "w.safetensors", "config": 5 }""");

        Assert.Equal(Path.Combine(_dir, "w.safetensors"), ConfigFileArgs.ResolveFileEntry(doc, "weights"));
        Assert.Null(ConfigFileArgs.ResolveFileEntry(doc, "absent"));
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.ResolveFileEntry(doc, "config"));

        string broken = Write("broken.json", "{ not json");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.ResolveFileEntry(broken, "weights"));

        string noPath = Write("no-path.json", """{ "weights": { "urls": ["https://example.invalid/w"] } }""");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.ResolveFileEntry(noPath, "weights"));
    }

    // ---- usage pages and the removed-flag advice -------------------------------------------

    [Fact]
    public void BothUsagePages_DocumentEveryLoraFlag()
    {
        var cli = new StringWriter();
        CliUsage.PrintUsage(cli);
        var server = new StringWriter();
        ServerUsage.PrintUsage(server);
        var cliDocumented = CliUsage.DocumentedFlags().ToHashSet(StringComparer.Ordinal);
        var serverDocumented = ServerUsage.DocumentedFlags().ToHashSet(StringComparer.Ordinal);

        foreach (string flag in LoraCliFlags.Flags)
        {
            Assert.Contains(flag, cliDocumented);
            Assert.Contains(flag, serverDocumented);
            Assert.Contains(flag + " <", cli.ToString(), StringComparison.Ordinal);
            Assert.Contains(flag + " <", server.ToString(), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RemovedQwenImageLoraFlag_PointsAtTheNewFlags()
    {
        string advice = RemovedCliFlags.Describe("--qwen-image-lora=x.safetensors")!;

        Assert.StartsWith("--qwen-image-lora was removed:", advice, StringComparison.Ordinal);
        foreach (string flag in LoraCliFlags.Flags)
            Assert.Contains(flag, advice, StringComparison.Ordinal);
        Assert.DoesNotContain("does not use LoRA", advice, StringComparison.Ordinal);
        // The replacement is a live flag, never itself reported as removed.
        foreach (string flag in LoraCliFlags.Flags)
            Assert.Null(RemovedCliFlags.Describe(flag));
    }

    // ---- the server's option passes -------------------------------------------------------

    private string NewBaseDir()
    {
        string dir = Path.Combine(_dir, "server-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static IEnumerable<object[]> ServerLoraLines() => new[]
    {
        new object[] { new[] { "--lora", "x.safetensors", "--lora-scale", "0.5", "--lora-config", "y.json" } },
        new object[] { new[] { "--lora=x.safetensors", "--lora-scale=0.5", "--lora-config=y.json" } },
        new object[] { new[] { "--lora", "a.safetensors", "--lora", "b.safetensors", "--lora-scale", "1.5" } },
    };

    [Theory]
    [MemberData(nameof(ServerLoraLines))]
    public void ServerBuild_AcceptsTheLoraFlags(string[] args)
    {
        // Build does not resolve the files (ApplyQwenImageCompanionCliFlags does, earlier);
        // it only has to let the flags and their values through its unknown-option trap.
        var options = ServerOptionsBuilder.Build(args, NewBaseDir());

        Assert.NotNull(options);
    }

    [Theory]
    [InlineData("--lora-scal", "--lora-scale")]
    [InlineData("--lora-confg", "--lora-config")]
    [InlineData("--lorra", "--lora")]
    public void ServerBuild_TypoOfALoraFlag_SuggestsIt(string typo, string flag)
    {
        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(new[] { typo, "1" }, NewBaseDir()));

        Assert.StartsWith("Unknown option", ex.Message, StringComparison.Ordinal);
        Assert.Contains($"Did you mean '{flag}'", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ServerCompanionPass_PublishesResolvedLorasAsTsLoras()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        string first = Touch("loras/first.safetensors");
        string second = Touch("loras/second.safetensors");
        string config = Write("loras/adapter_config.json", """{ "lora_alpha": 16 }""");
        string[] args =
        {
            "--model", "qwen-image-2.1.gguf",
            "--lora", first, "--lora-scale", "0.5", "--lora-config", config,
            "--lora=" + second,
        };

        Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));

        var published = LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable));
        Assert.Equal(new[]
        {
            new LoraSpec(Path.GetFullPath(first), 0.5f, Path.GetFullPath(config)),
            new LoraSpec(Path.GetFullPath(second)),
        }, published);
        // The later validation pass must accept the flags the companion pass left in argv.
        Assert.NotNull(ServerOptionsBuilder.Build(args.Skip(2).ToArray(), NewBaseDir()));
    }

    [Fact]
    public void ServerCompanionPass_PublishesEveryLoraTheValidationPassAccepts()
    {
        // The server's option parser matches flags case-insensitively (as do the shared
        // speculative and code-exec tables), so a spelling it lets through must also be
        // applied; otherwise the server starts without the adapter and says nothing.
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        string weights = Touch("case/w.safetensors");
        string[] args = { "--LoRA", weights, "--LORA-SCALE=0.5" };

        Assert.NotNull(ServerOptionsBuilder.Build(args, NewBaseDir()));
        Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));

        var published = Assert.Single(LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable)));
        Assert.Equal(new LoraSpec(Path.GetFullPath(weights), 0.5f), published);
    }

    [Fact]
    public void ServerCompanionPass_ExpandsAPluginManifest()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        string weights = Touch("plugin/w.safetensors");
        string manifest = Write("plugin/turbo.json", """
        { "type": "qwen-image-2.1-lora", "weights": { "path": "w.safetensors" }, "sampling": { "steps": 1, "sigmas": [1.0] } }
        """);

        Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { "--lora", manifest }));

        var published = Assert.Single(LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable)));
        Assert.Equal(new LoraSpec(Path.GetFullPath(weights), null, Path.GetFullPath(manifest)), published);
    }

    [Fact]
    public void ServerCompanionPass_MissingLoraFile_FailsAtStartup()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, null);

        Assert.Throws<FileNotFoundException>(() => ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(
            new[] { "--lora", Path.Combine(_dir, "typo.safetensors") }));
        Assert.Null(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable));
    }

    [Fact]
    public void ServerCompanionPass_ScaleWithoutLora_FailsAtStartup()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, null);

        Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(
            new[] { "--lora-scale", "0.5" }));
    }

    [Fact]
    public void ServerCompanionPass_WithoutLoraFlags_LeavesTsLorasAlone()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, null);

        Assert.False(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { "--model", "m.gguf" }));
        Assert.Null(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable));
    }
}
