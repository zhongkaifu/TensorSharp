// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3 LoRA plug-in configs (<see cref="MiniMaxH3LoraConfig"/>): the shipped Turbo
/// manifests, every malformed field, and the locked sampling contract as the hosts
/// (<see cref="MiniMaxH3LoraConfig.CheckHostSettings"/>), the selection
/// (<see cref="MiniMaxH3LoraConfig.RecipeFor(IReadOnlyList{LoraSpec}, MiniMaxH3Partition)"/>) and a
/// request (<see cref="MiniMaxH3SamplingRecipe.Validate"/>) see it. Manifests only: no weights are
/// read and nothing is downloaded.
/// </summary>
public sealed class MiniMaxH3LoraConfigTests : IDisposable
{
    private const string W8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors";
    private const string W4 = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors";
    private const string Revision = "a29e57d195f80ead07c89f89e55ecf2911e893c5";

    private readonly string _dir;

    public MiniMaxH3LoraConfigTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-h3-lora-config-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- helpers ------------------------------------------------------------------------

    private static JsonObject Valid(string name = "Test 8", string weights = W8, int steps = 8) =>
        JsonNode.Parse($$"""
        {
          "type": "minimax-h3-lora",
          "name": {{JsonSerializer.Serialize(name)}},
          "variables": { "root": "w" },
          "weights": {
            "path": {{JsonSerializer.Serialize("${root}/" + weights)}},
            "urls": [ {{JsonSerializer.Serialize("https://example.invalid/" + weights)}} ],
            "sha256": "{{new string('a', 64)}}"
          },
          "partition": "fl2va",
          "key_format": "minimax-h3-diffusers",
          "alpha": 8,
          "scale": 1.0,
          "sampling": { "steps": {{steps}}, "schedule": "linspace", "video_shift": 6.0, "audio_shift": 3.0, "sampler": "euler", "cfg": 1.0 }
        }
        """)!.AsObject();

    private string Write(string name, string text)
    {
        string path = Path.Combine(_dir, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    private string Write(JsonObject manifest, string name = "turbo-8step.json") => Write(name, manifest.ToJsonString());

    private string Manifest8() => Write(Valid(), "turbo-8step.json");

    private string Manifest4() => Write(Valid("Test 4", W4, 4), "turbo-4step.json");

    private string WeightsPath(string fileName) => Path.Combine(_dir, "w", fileName);

    private static void Mutate(JsonObject root, string dotted, string? valueJson)
    {
        string[] parts = dotted.Split('.');
        JsonObject parent = root;
        for (int i = 0; i < parts.Length - 1; i++)
            parent = parent[parts[i]]!.AsObject();
        if (valueJson == null) parent.Remove(parts[^1]);
        else parent[parts[^1]] = JsonNode.Parse(valueJson);
    }

    private static MiniMaxH3HostSampling Cli(int steps = 0, float shift = 0f, float cfg = 0f, string sampler = null) =>
        new(steps, "--diffusion-steps", shift, "--flow-shift", cfg, "--cfg", sampler, "--sampler");

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "The tests run from a source checkout; config/lora/ must be reachable.");
        return dir!.FullName;
    }

    private static string Shipped(string name) => Path.Combine(FindRepoRoot(), "config", "lora", name);

    private static void AssertSigmas(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6, $"sigma[{i}] = {actual[i]:R}, expected {expected[i]:R}");
    }

    private static readonly double[] Video8 = { 1, 42 / 43.0, 18 / 19.0, 10 / 11.0, 6 / 7.0, 18 / 23.0, 2 / 3.0, 6 / 13.0, 0 };
    private static readonly double[] Audio8 = { 1, 21 / 22.0, 0.9, 5 / 6.0, 0.75, 9 / 14.0, 0.5, 0.3, 0 };
    private static readonly double[] Video4 = { 1, 18 / 19.0, 6 / 7.0, 2 / 3.0, 0 };
    private static readonly double[] Audio4 = { 1, 0.9, 0.75, 0.5, 0 };

    // ---- the shipped manifests ------------------------------------------------------------

    [Theory]
    [InlineData("minimax-h3-fl2v-turbo-8step.json", "LightX2V Turbo 8-step (FL2V v1.0, 768p)", W8, 8,
        "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4")]
    [InlineData("minimax-h3-fl2v-turbo-4step.json", "LightX2V Turbo 4-step (FL2V v1.2, 768p)", W4, 4,
        "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689")]
    public void ShippedManifest_IsPinned(string file, string name, string weights, int steps, string sha256)
    {
        string path = Shipped(file);
        var config = MiniMaxH3LoraConfig.Load(path);

        Assert.Equal(Path.GetFullPath(path), config.Path);
        Assert.Equal(name, config.Name);
        Assert.Equal(MiniMaxH3Partition.FirstLastFrame, config.Partition);
        Assert.Equal(8f, config.Alpha);
        Assert.Equal(1f, config.Scale);
        Assert.Equal(weights, config.WeightsFileName);
        Assert.Equal(weights, Path.GetFileName(ConfigFileArgs.ResolveFileEntryPath(path, "weights")));

        var recipe = config.Recipe;
        Assert.Equal("minimax-h3-lora:" + Path.GetFileNameWithoutExtension(file), recipe.Id);
        Assert.Equal(name, recipe.Name);
        Assert.Equal(weights, recipe.WeightsFileName);
        Assert.Equal(steps, recipe.Steps);
        Assert.Equal(6f, recipe.VideoShift);
        Assert.Equal(3f, recipe.AudioShift);
        Assert.Equal(1f, recipe.Cfg);
        Assert.Equal("linspace", recipe.Schedule);
        Assert.Equal("euler", recipe.Sampler);
        AssertSigmas(steps == 8 ? Video8 : Video4, recipe.VideoSigmas());
        AssertSigmas(steps == 8 ? Audio8 : Audio4, recipe.AudioSigmas());

        // The download entry: the variable root, a revision-pinned URL and the file's hash.
        using var doc = JsonDocument.Parse(File.ReadAllText(path),
            new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        var root = doc.RootElement;
        Assert.Equal("${TENSORSHARP_MODELS:-../../models}/minimax-h3/loras", root.GetProperty("variables").GetProperty("root").GetString());
        var entry = root.GetProperty("weights");
        Assert.Equal("${root}/" + weights, entry.GetProperty("path").GetString());
        Assert.Equal($"https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/{Revision}/{weights}",
            Assert.Single(entry.GetProperty("urls").EnumerateArray().ToList()).GetString());
        Assert.Equal(sha256, entry.GetProperty("sha256").GetString());
    }

    [Fact]
    public void ShippedManifests_AreTheContractTheHostsAdvertise()
    {
        var contract = MiniMaxH3LoraConfig.Load(Shipped("minimax-h3-fl2v-turbo-8step.json")).Recipe.ToContract();

        Assert.Equal(new VideoSamplingContract("minimax-h3-lora:minimax-h3-fl2v-turbo-8step", "LightX2V Turbo 8-step (FL2V v1.0, 768p)",
            "minimax-h3-fl2v-turbo-8step.json", 8, 1f, "linspace", "euler", 6f, 3f), contract);
        Assert.NotEqual(VideoSamplingContract.NativeId, contract.Id);
    }

    [Fact]
    public void ShippedFourStepManifest_SaysItsRecipeIsInferred()
    {
        string text = File.ReadAllText(Shipped("minimax-h3-fl2v-turbo-4step.json"));

        Assert.Contains(
            "INFERRED: upstream publishes no recipe for v1.2 (ModelTC README lists 4-step v1.0 768p at 6/3; the v1.2 file has " +
            "no shift metadata). Linspace grid and shift 6/3 are carried over; see docs/models/minimax-h3.md for the validation status.",
            text, StringComparison.Ordinal);
        Assert.DoesNotContain("INFERRED", File.ReadAllText(Shipped("minimax-h3-fl2v-turbo-8step.json")), StringComparison.Ordinal);
    }

    [Fact]
    public void ShippedManifest_WithTheOtherVariantsWeights_IsRefusedByTheTie()
    {
        // The 4-step adapter on the 8-step schedule: the --lora-config form ties weights to manifest.
        string manifest = Shipped("minimax-h3-fl2v-turbo-8step.json");
        var spec = new LoraSpec(WeightsPath(W4), null, manifest);

        var ex = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.RecipeFor(new[] { spec }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal(
            "--lora-config minimax-h3-fl2v-turbo-8step.json is the LightX2V Turbo 8-step (FL2V v1.0, 768p) plug-in for " + W8 +
            "; --lora names " + W4 + ". Pass the plug-in alone with --lora (it names its own weights), or the weights it was written for.",
            ex.Message);
    }

    // ---- the schema ----------------------------------------------------------------------

    [Fact]
    public void Load_AValidManifest()
    {
        string path = Manifest8();

        var config = MiniMaxH3LoraConfig.Load(path);

        Assert.Equal("Test 8", config.Name);
        Assert.Equal(W8, config.WeightsFileName);
        Assert.Equal(8f, config.Alpha);
        Assert.Equal(1f, config.Scale);
        Assert.Equal("minimax-h3-lora:turbo-8step", config.Recipe.Id);
        Assert.Equal(Path.GetFullPath(path), config.Recipe.Source);
        // The weights are resolved to a path only; nothing is fetched from example.invalid.
        Assert.False(File.Exists(WeightsPath(W8)));
    }

    [Fact]
    public void Load_OptionalFieldsMayBeOmittedAndWeightsMayBeAPlainPath()
    {
        var json = Valid();
        Mutate(json, "alpha", null);
        Mutate(json, "scale", null);
        Mutate(json, "variables", null);
        Mutate(json, "weights", JsonSerializer.Serialize("loras/" + W8));

        var config = MiniMaxH3LoraConfig.Load(Write(json));

        Assert.Null(config.Alpha);
        Assert.Null(config.Scale);
        Assert.Equal(W8, config.WeightsFileName);
    }

    [Fact]
    public void Load_ScaleZeroIsAccepted()
    {
        var json = Valid();
        Mutate(json, "scale", "0");

        Assert.Equal(0f, MiniMaxH3LoraConfig.Load(Write(json)).Scale);
    }

    [Theory]
    [InlineData("type", null, "\"type\" is required")]
    [InlineData("type", "\"qwen-image-2.1-lora\"", "\"type\" must be \"minimax-h3-lora\"")]
    [InlineData("name", null, "\"name\" is required")]
    [InlineData("name", "\"  \"", "\"name\" must not be empty")]
    [InlineData("name", "5", "\"name\" must be a string")]
    [InlineData("extra", "1", "unknown key \"extra\"")]
    [InlineData("config", "\"pdd_config.json\"", "\"config\" is not a minimax-h3-lora key")]
    [InlineData("variables", "[]", "\"variables\" must be an object")]
    [InlineData("variables.root", "true", "\"variables.root\" must be a string or a number")]
    [InlineData("weights", null, "\"weights\" is required")]
    [InlineData("weights", "5", "\"weights\" must be a path or a download object")]
    [InlineData("weights.path", null, "\"weights.path\" is required")]
    [InlineData("weights.path", "\"${root}/${name}.safetensors\"", "\"weights.path\" must spell the weights file name literally")]
    [InlineData("weights.path", "\"w/turbo.ckpt\"", "\"weights.path\" must name a .safetensors file")]
    [InlineData("weights.mirror", "\"x\"", "unknown key \"weights.mirror\"")]
    [InlineData("weights.urls", "[]", "\"weights.urls\" must be a non-empty array")]
    [InlineData("weights.urls", "[\"ftp://example.invalid/w.safetensors\"]", "\"weights.urls\" must hold http(s) URLs")]
    [InlineData("weights.sha256", "\"abc\"", "\"weights.sha256\" must be 64 hexadecimal digits")]
    [InlineData("partition", null, "\"partition\" is required")]
    [InlineData("partition", "\"ref2va\"", "Ref2VA Turbo is not supported")]
    [InlineData("partition", "\"t2va\"", "\"partition\" must be \"fl2va\"")]
    [InlineData("key_format", null, "\"key_format\" is required")]
    [InlineData("key_format", "\"comfyui\"", "\"key_format\" must be \"minimax-h3-diffusers\"")]
    [InlineData("alpha", "0", "\"alpha\" must be positive")]
    [InlineData("alpha", "\"8\"", "\"alpha\" must be a number")]
    [InlineData("scale", "-0.5", "\"scale\" must be a strength >= 0")]
    [InlineData("scale", "1e39", "\"scale\" must be a finite number")]
    [InlineData("scale", "\"1\"", "\"scale\" must be a number")]
    [InlineData("sampling", null, "\"sampling\" is required")]
    [InlineData("sampling", "[]", "\"sampling\" must be an object")]
    [InlineData("sampling.sigmas", "[1, 0]", "unknown key \"sampling.sigmas\"")]
    [InlineData("sampling.steps", null, "\"sampling.steps\" is required")]
    [InlineData("sampling.steps", "0", "\"sampling.steps\" must be an integer from 1 to 100")]
    [InlineData("sampling.steps", "101", "\"sampling.steps\" must be an integer from 1 to 100")]
    [InlineData("sampling.steps", "8.5", "\"sampling.steps\" must be an integer from 1 to 100")]
    [InlineData("sampling.steps", "\"8\"", "\"sampling.steps\" must be an integer from 1 to 100")]
    [InlineData("sampling.schedule", null, "\"sampling.schedule\" is required")]
    [InlineData("sampling.schedule", "\"discrete\"", "\"sampling.schedule\" must be \"linspace\"")]
    [InlineData("sampling.video_shift", null, "\"sampling.video_shift\" is required")]
    [InlineData("sampling.video_shift", "0", "\"sampling.video_shift\" must be positive")]
    [InlineData("sampling.video_shift", "-6", "\"sampling.video_shift\" must be positive")]
    [InlineData("sampling.video_shift", "1e39", "\"sampling.video_shift\" must be a finite number")]
    [InlineData("sampling.audio_shift", null, "\"sampling.audio_shift\" is required")]
    [InlineData("sampling.audio_shift", "12", "\"sampling.audio_shift\" must be 3")]
    [InlineData("sampling.sampler", null, "\"sampling.sampler\" is required")]
    [InlineData("sampling.sampler", "\"unipc\"", "\"sampling.sampler\" must be \"euler\"")]
    [InlineData("sampling.cfg", null, "\"sampling.cfg\" is required")]
    [InlineData("sampling.cfg", "2", "\"sampling.cfg\" must be 1")]
    public void Load_AMalformedField_IsRefusedNamingTheFileAndTheKey(string key, string? valueJson, string fragment)
    {
        var json = Valid();
        Mutate(json, key, valueJson);
        string path = Write(json);

        var ex = Assert.Throws<InvalidDataException>(() => MiniMaxH3LoraConfig.Load(path));

        Assert.Contains(Path.GetFullPath(path), ex.Message, StringComparison.Ordinal);
        Assert.Contains(fragment, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("{ \"type\": \"minimax-h3-lora\", ")]
    [InlineData("[ \"minimax-h3-lora\" ]")]
    [InlineData("""{ "type": "minimax-h3-lora", "name": "a", "name": "b" }""")]
    public void Load_MalformedJson_IsInvalidData(string text)
    {
        string path = Write("broken.json", text);

        var ex = Assert.Throws<InvalidDataException>(() => MiniMaxH3LoraConfig.Load(path));
        Assert.Contains(Path.GetFullPath(path), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_MissingFile_IsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() => MiniMaxH3LoraConfig.Load(Path.Combine(_dir, "absent.json")));
    }

    // ---- the weights tie and the strength -----------------------------------------------------

    [Fact]
    public void CheckWeights_TiesTheConfigToItsFileName()
    {
        var config = MiniMaxH3LoraConfig.Load(Manifest8());

        config.CheckWeights(WeightsPath(W8));
        config.CheckWeights(Path.Combine("elsewhere", W8));
        var ex = Assert.Throws<ArgumentException>(() => config.CheckWeights(WeightsPath(W4)));

        Assert.Equal(
            $"--lora-config turbo-8step.json is the Test 8 plug-in for {W8}; --lora names {W4}. " +
            "Pass the plug-in alone with --lora (it names its own weights), or the weights it was written for.",
            ex.Message);
    }

    [Fact]
    public void CheckStrength_RefusesNegativeAndAcceptsZero()
    {
        var config = MiniMaxH3LoraConfig.Load(Manifest8());

        MiniMaxH3LoraConfig.CheckStrength(config, W8, 0f);
        MiniMaxH3LoraConfig.CheckStrength(config, W8, 1f);
        MiniMaxH3LoraConfig.CheckStrength(config, W8, 2.5f);
        var negative = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckStrength(config, WeightsPath(W8), -0.5f));
        Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckStrength(config, W8, float.NaN));

        Assert.Equal(
            $"Test 8 ({W8}): strength -0.5 is negative; the Turbo schedule is distilled at strength 1 " +
            "(0 keeps the adapter's terms with zero up factors).",
            negative.Message);
    }

    // ---- RecipeFor: which recipe a list of specs selects --------------------------------------

    [Fact]
    public void RecipeFor_NoSpecs_IsNoRecipe()
    {
        Assert.Null(MiniMaxH3LoraConfig.RecipeFor(null, MiniMaxH3Partition.FirstLastFrame));
        Assert.Null(MiniMaxH3LoraConfig.RecipeFor(Array.Empty<LoraSpec>(), MiniMaxH3Partition.Reference));
    }

    [Fact]
    public void RecipeFor_TheManifestOrItsWeightsWithTheManifestAsConfig_SelectsItsRecipe()
    {
        string manifest = Manifest8();

        var direct = MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(manifest) }, MiniMaxH3Partition.FirstLastFrame);
        // TensorAgent's form; the weights need not be on disk yet (manifest-only).
        var paired = MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(WeightsPath(W8), 1f, manifest) },
            MiniMaxH3Partition.FirstLastFrame);

        Assert.Equal(8, direct.Steps);
        Assert.Equal(direct.Id, paired.Id);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(1.5f)]
    public void RecipeFor_NonNegativeStrength_IsAccepted(float strength)
    {
        Assert.NotNull(MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(Manifest8(), strength) }, MiniMaxH3Partition.FirstLastFrame));
    }

    [Fact]
    public void RecipeFor_NegativeStrength_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(Manifest8(), -1f) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal(
            $"Test 8 ({W8}): strength -1 is negative; the Turbo schedule is distilled at strength 1 " +
            "(0 keeps the adapter's terms with zero up factors).",
            ex.Message);
    }

    [Fact]
    public void RecipeFor_ManifestScaleIsTheStrengthWhenTheSpecSetsNone()
    {
        // A config's own "scale" is checked by the parser; a spec's --lora-scale overrides it.
        var json = Valid();
        Mutate(json, "scale", "0");
        string manifest = Write(json);

        Assert.NotNull(MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(manifest) }, MiniMaxH3Partition.FirstLastFrame));
        Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(manifest, -0.25f) }, MiniMaxH3Partition.FirstLastFrame));
    }

    [Fact]
    public void RecipeFor_FourStepWeightsWithTheEightStepManifest_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.RecipeFor(
            new[] { new LoraSpec(WeightsPath(W4), null, Manifest8()) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.StartsWith($"--lora-config turbo-8step.json is the Test 8 plug-in for {W8}; --lora names {W4}.", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RecipeFor_BareWeightsOrAPeftConfig_IsRefused()
    {
        string peft = Write("adapter_config.json", """{ "lora_alpha": 8, "r": 128 }""");
        string expected =
            $"--lora {W8}: MiniMax-H3 takes LoRA plug-ins only as a minimax-h3-lora config " +
            "(for example config/lora/minimax-h3-fl2v-turbo-8step.json), which declares the checkpoint partition and the " +
            "sampling contract. Pass that config instead of the bare weights.";

        var bare = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(WeightsPath(W8)) }, MiniMaxH3Partition.FirstLastFrame));
        var withPeft = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(WeightsPath(W8), null, peft) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal(expected, bare.Message);
        Assert.Equal(expected, withPeft.Message);
    }

    [Fact]
    public void RecipeFor_AnotherFamilysPlugin_IsRefused()
    {
        string qwen = Write("qwen.json", """{ "type": "qwen-image-2.1-lora", "weights": "q.safetensors" }""");

        var ex = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(qwen) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal("--lora qwen.json is a qwen-image-2.1-lora plug-in; MiniMax-H3 (minimax-h3) takes minimax-h3-lora plug-ins only.",
            ex.Message);
    }

    [Fact]
    public void RecipeFor_TwoPlugins_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.RecipeFor(
            new[] { new LoraSpec(Manifest8()), new LoraSpec(Manifest4()) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal("Two MiniMax-H3 plug-ins (turbo-8step.json, turbo-4step.json) each fix the sampling schedule; use one.", ex.Message);
    }

    [Fact]
    public void RecipeFor_ARef2vaModel_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(Manifest8()) }, MiniMaxH3Partition.Reference));

        Assert.Equal(
            "LoRA plug-in turbo-8step.json is for the MiniMax-H3 FL2VA checkpoint; the loaded model is the Ref2VA checkpoint, " +
            "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.",
            ex.Message);
    }

    [Fact]
    public void RecipeFor_AMalformedManifest_IsAnArgumentException()
    {
        var json = Valid();
        Mutate(json, "sampling.steps", "0");
        string manifest = Write(json);

        var ex = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(manifest) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.IsType<InvalidDataException>(ex.InnerException);
        Assert.Contains("\"sampling.steps\"", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RecipeFor_MissingManifest_UsesTheResolveNotFoundTexts()
    {
        string missing = Path.Combine(_dir, "typo-8step.json");

        var lora = Assert.Throws<FileNotFoundException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(missing) }, MiniMaxH3Partition.FirstLastFrame));
        var config = Assert.Throws<FileNotFoundException>(() =>
            MiniMaxH3LoraConfig.RecipeFor(new[] { new LoraSpec(WeightsPath(W8), null, missing) }, MiniMaxH3Partition.FirstLastFrame));

        Assert.Equal($"--lora file not found: {missing}", lora.Message);
        Assert.Equal($"--lora-config file not found: {missing}", config.Message);
    }

    // ---- CheckHostSettings: the hosts' startup check -------------------------------------------

    [Fact]
    public void CheckHostSettings_UnsetOrMatchingFlags_ReturnTheRecipe()
    {
        var specs = new[] { new LoraSpec(Manifest8()) };

        Assert.Equal(8, MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli()).Steps);
        Assert.NotNull(MiniMaxH3LoraConfig.CheckHostSettings(specs, "/models/minimax_h3_fl2va_pruned-Q4_K.gguf",
            Cli(steps: 8, shift: 6f, cfg: 1f, sampler: "euler")));
        Assert.NotNull(MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli(sampler: "Euler")));
    }

    [Fact]
    public void CheckHostSettings_NoSpecs_ChecksNothing()
    {
        Assert.Null(MiniMaxH3LoraConfig.CheckHostSettings(Array.Empty<LoraSpec>(), null, Cli(steps: 20, shift: 12f, cfg: 2f, sampler: "unipc")));
    }

    [Fact]
    public void CheckHostSettings_ContradictingSteps_NamesTheHostsFlag()
    {
        var specs = new[] { new LoraSpec(Manifest8()) };

        var cli = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli(steps: 20)));
        var server = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(specs, null,
            new MiniMaxH3HostSampling(4, "--video-steps", 0f, "--flow-shift", 0f, "--cfg", null, "--sampler")));

        Assert.Equal($"--diffusion-steps 20 contradicts {W8}: Test 8 samples exactly 8 steps on its own schedule. " +
            "Drop --diffusion-steps or pass 8.", cli.Message);
        Assert.Equal($"--video-steps 4 contradicts {W8}: Test 8 samples exactly 8 steps on its own schedule. " +
            "Drop --video-steps or pass 8.", server.Message);
    }

    [Fact]
    public void CheckHostSettings_ContradictingShiftSamplerOrCfg_IsRefused()
    {
        var specs = new[] { new LoraSpec(Manifest8()) };

        var shift = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli(shift: 12f)));
        var sampler = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli(sampler: "unipc")));
        var cfg = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(specs, null, Cli(cfg: 2f)));

        Assert.Equal($"Test 8 ({W8}) samples with video flow shift 6 (audio 3); flow shift 12 contradicts it. " +
            "Leave it unset (--flow-shift, \"flowShift\") or set it to 6.", shift.Message);
        Assert.Equal($"Test 8 ({W8}) samples with Euler; sampler 'unipc' contradicts it. " +
            "Leave it unset (--sampler, \"sampler\") or pass euler.", sampler.Message);
        Assert.Equal("MiniMax-H3 is CFG-distilled and requires guidance 1.0. Use --cfg 1.0 (the default).", cfg.Message);
    }

    [Fact]
    public void CheckHostSettings_TakesThePartitionFromTheModelFileName()
    {
        var specs = new[] { new LoraSpec(Manifest8()) };

        var ex = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.CheckHostSettings(specs, "/models/minimax_h3_ref2va_pruned-Q4_K.gguf", Cli()));

        Assert.Equal(
            "LoRA plug-in turbo-8step.json is for the MiniMax-H3 FL2VA checkpoint; minimax_h3_ref2va_pruned-Q4_K.gguf is the " +
            "Ref2VA checkpoint, and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.",
            ex.Message);
    }

    [Fact]
    public void CheckHostSettings_ForeignSpec_NamesTheModel()
    {
        string qwen = Write("qwen.json", """{ "type": "qwen-image-2.1-lora", "weights": "q.safetensors" }""");

        var ex = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraConfig.CheckHostSettings(
            new[] { new LoraSpec(qwen) }, "/models/minimax_h3_fl2va_pruned-Q4_K.gguf", Cli()));

        Assert.Equal("--lora qwen.json is a qwen-image-2.1-lora plug-in; minimax_h3_fl2va_pruned-Q4_K.gguf (minimax-h3) " +
            "takes minimax-h3-lora plug-ins only.", ex.Message);
    }

    [Fact]
    public void CheckHostSettings_StrengthAndTie_AreChecked()
    {
        string manifest = Manifest8();

        Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.CheckHostSettings(new[] { new LoraSpec(manifest, -0.5f) }, null, Cli()));
        Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraConfig.CheckHostSettings(new[] { new LoraSpec(WeightsPath(W4), null, manifest) }, null, Cli()));
        Assert.NotNull(MiniMaxH3LoraConfig.CheckHostSettings(new[] { new LoraSpec(manifest, 0f) }, null, Cli()));
    }

    // ---- Validate: a request against the recipe --------------------------------------------

    public static IEnumerable<object[]> AcceptedRequests() => new[]
    {
        new object[] { new VideoGenerationParams() },
        new object[] { new VideoGenerationParams { Steps = 8 } },
        new object[] { new VideoGenerationParams { FlowShift = 6f } },
        new object[] { new VideoGenerationParams { CfgScale = 1f } },
        new object[] { new VideoGenerationParams { Sampler = "" } },
        new object[] { new VideoGenerationParams { Sampler = "euler" } },
        new object[] { new VideoGenerationParams { Steps = 8, FlowShift = 6f, CfgScale = 1f, Sampler = "Euler" } },
    };

    [Theory]
    [MemberData(nameof(AcceptedRequests))]
    public void Validate_UnsetOrMatchingValues_AreAccepted(VideoGenerationParams p)
    {
        MiniMaxH3LoraConfig.Load(Manifest8()).Recipe.Validate(p);
    }

    [Fact]
    public void Validate_NullParams_IsAccepted()
    {
        MiniMaxH3LoraConfig.Load(Manifest8()).Recipe.Validate(null);
    }

    [Fact]
    public void Validate_ContradictingValues_AreRefusedNeverOverridden()
    {
        var recipe = MiniMaxH3LoraConfig.Load(Manifest8()).Recipe;

        var steps = Assert.Throws<ArgumentException>(() => recipe.Validate(new VideoGenerationParams { Steps = 20 }));
        var shift = Assert.Throws<ArgumentException>(() => recipe.Validate(new VideoGenerationParams { FlowShift = 12f }));
        var sampler = Assert.Throws<ArgumentException>(() => recipe.Validate(new VideoGenerationParams { Sampler = "unipc" }));
        var cfg = Assert.Throws<ArgumentException>(() => recipe.Validate(new VideoGenerationParams { CfgScale = 2f }));
        var negative = Assert.Throws<ArgumentException>(() => recipe.Validate(new VideoGenerationParams { Steps = -1 }));

        Assert.Equal(
            $"Test 8 ({W8}) samples exactly 8 steps on its own schedule; 20 steps contradicts it. Leave the step count unset " +
            "(--diffusion-steps on the CLI, --video-steps on the server, \"steps\" in a request) or set it to 8.",
            steps.Message);
        Assert.Equal($"Test 8 ({W8}) samples with video flow shift 6 (audio 3); flow shift 12 contradicts it. " +
            "Leave it unset (--flow-shift, \"flowShift\") or set it to 6.", shift.Message);
        Assert.Equal($"Test 8 ({W8}) samples with Euler; sampler 'unipc' contradicts it. " +
            "Leave it unset (--sampler, \"sampler\") or pass euler.", sampler.Message);
        // The base MiniMax-H3 refusals still apply.
        Assert.StartsWith("MiniMax-H3 is CFG-distilled and requires guidance 1.0.", cfg.Message, StringComparison.Ordinal);
        Assert.StartsWith("MiniMax-H3 steps must be positive", negative.Message, StringComparison.Ordinal);
    }

    // ---- the recipe's own surface --------------------------------------------------------------

    [Fact]
    public void Recipe_ContractAndLogLine()
    {
        var recipe = MiniMaxH3LoraConfig.Load(Manifest4()).Recipe;

        Assert.Equal(new VideoSamplingContract("minimax-h3-lora:turbo-4step", "Test 4", "turbo-4step.json", 4, 1f,
            "linspace", "euler", 6f, 3f), recipe.ToContract());
        Assert.Equal("Test 4: 4 steps, endpoint-inclusive linspace, video shift 6 / audio 3, euler, cfg 1", recipe.Describe());
        string line = recipe.LogLine();
        Assert.Contains(recipe.Describe(), line, StringComparison.Ordinal);
        Assert.Contains("video sigmas [1, 0.94737, 0.85714, 0.66667, 0]", line, StringComparison.Ordinal);
        Assert.Contains("audio sigmas [1, 0.9, 0.75, 0.5, 0]", line, StringComparison.Ordinal);
    }

    [Fact]
    public void Recipe_SigmaArraysAreFreshCopies()
    {
        var recipe = MiniMaxH3LoraConfig.Load(Manifest8()).Recipe;

        float[] first = recipe.VideoSigmas();
        first[1] = 0.5f;

        AssertSigmas(Video8, recipe.VideoSigmas());
        AssertSigmas(Audio8, recipe.AudioSigmas());
    }
}
