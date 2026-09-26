// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// The LoRA companion configs (<see cref="QwenImage21LoraConfig"/>): TensorSharp's plug-in
/// config with its sampling recipe, PEFT <c>adapter_config.json</c> and embedded PEFT
/// metadata, VideoX-Fun <c>pdd_config.json</c>; the recipe's sigma schedules; and the
/// plug-in configs shipped in <c>config/lora/</c>.
/// </summary>
public sealed class QwenImage21LoraConfigTests : IDisposable
{
    private readonly string _dir;

    public QwenImage21LoraConfigTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-lora-config-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private QwenImage21LoraConfig Load(string json, string name = "config.json")
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, json);
        return QwenImage21LoraConfig.Load(path);
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"[{i}] expected {expected[i]}, got {actual[i]} (all: {string.Join(", ", actual)})");
    }

    private static readonly float[] ViggleSixNodes = { 1f, 0.9375f, 0.875f, 0.75f, 0.5f, 0.25f };
    private static readonly float[] ViggleSix1024 = { 1.0f, 0.96775f, 0.93336f, 0.85719f, 0.66676f, 0.4001f, 0.0f };
    private static readonly float[] ViggleSix2048 = { 1.0f, 0.98238f, 0.96299f, 0.9177f, 0.788f, 0.55337f, 0.0f };

    // ---- TensorSharp plug-in config ------------------------------------------------------

    [Theory]
    [InlineData("\"sampling\": [6]")]
    [InlineData("\"sampling\": \"fast\"")]
    public void TensorSharp_SamplingThatIsNotAnObject_IsRefused(string sampling)
    {
        var ex = Assert.Throws<InvalidDataException>(() => Load("{ \"type\": \"qwen-image-2.1-lora\", " + sampling + " }"));
        Assert.Contains("sampling", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", \"sampling\": { \"steps\": \"six\" } }")]
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", \"scale\": \"strong\" }")]
    [InlineData("{ \"type\": \"qwen-image-2.1-lora\", \"sampling\": { \"steps\": 4, \"cfg\": true } }")]
    public void TensorSharp_ValuesOfTheWrongKind_NameTheFile(string json)
    {
        var ex = Assert.Throws<InvalidDataException>(() => Load(json, "wrong-kind.json"));
        Assert.Contains("wrong-kind.json", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AdapterMetadata_ThatIsNotJson_IsRefusedAsInvalidData()
    {
        Assert.Throws<InvalidDataException>(() => QwenImage21LoraConfig.FromAdapterMetadata("{ not json", "file.safetensors"));
    }

    [Fact]
    public void TensorSharp_MinimalConfig_HasNoRecipe()
    {
        var config = Load("""{ "type": "qwen-image-2.1-lora" }""");

        Assert.Equal("TensorSharp LoRA config", config.Format);
        Assert.Equal(Path.GetFullPath(Path.Combine(_dir, "config.json")), config.Path);
        Assert.Null(config.Scale);
        Assert.Null(config.Alpha);
        Assert.Null(config.UseRsLora); // absent: the file's own metadata decides
        Assert.Null(config.Recipe);
        Assert.False(config.IsPdd);
        Assert.Null(config.AlphaFor("transformer_blocks.0.attn.to_q"));
    }

    [Fact]
    public void TensorSharp_ReadsScaleAlphaAndRsLora_AndIgnoresManifestFields()
    {
        var config = Load("""
        {
          // a plug-in manifest is also its own config
          "type": "qwen-image-2.1-lora",
          "variables": { "root": "../models" },
          "weights": { "path": "${root}/w.safetensors", "urls": ["https://example.invalid/w"], "sha256": "ab" },
          "scale": 0.7,
          "alpha": 16,
          "use_rslora": true,
        }
        """);

        Assert.Equal(0.7f, config.Scale);
        Assert.Equal(16f, config.Alpha);
        Assert.True(config.UseRsLora);
        Assert.Equal(16f, config.AlphaFor("transformer_blocks.3.img_mlp.out"));
        Assert.Null(config.Recipe);
    }

    [Fact]
    public void TensorSharp_SigmaArray_IsOneFixedSchedule()
    {
        var config = Load("""
        {
          "type": "qwen-image-2.1-lora",
          "sampling": { "steps": 5, "shift": "none", "cfg": 1.0, "sigmas": [1.0, 0.94, 0.85714287, 0.6666667, 0.4] }
        }
        """);

        var recipe = config.Recipe;
        Assert.NotNull(recipe);
        Assert.True(recipe.HasSchedule);
        Assert.Equal(5, recipe.DefaultSteps);
        Assert.Equal(new[] { 5 }, recipe.Nodes.Keys);
        Assert.Equal(QwenImage21SigmaShift.None, recipe.Shift);
        Assert.Equal(1f, recipe.Cfg);
        Assert.False(recipe.TimestepBf16);
        Assert.Equal(config.Path, recipe.Source);
        // Shift none: the nodes verbatim at every resolution, then a terminal 0.
        var expected = new[] { 1.0f, 0.94f, 0.85714287f, 0.6666667f, 0.4f, 0f };
        Assert.Equal(expected, recipe.Sigmas(5, 4096));
        Assert.Equal(expected, recipe.Sigmas(5, 16384));
    }

    [Fact]
    public void TensorSharp_SigmaArrayWithTerminalZero_InfersTheStepCount()
    {
        var recipe = Load("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [1.0, 0.75, 0.5, 0.0] } }""").Recipe;

        Assert.Equal(3, recipe.DefaultSteps);
        Assert.Equal(new[] { 1f, 0.75f, 0.5f }, recipe.Nodes[3]);
        Assert.Equal(new[] { 1f, 0.75f, 0.5f, 0f }, recipe.Sigmas(3, 4096));
    }

    [Fact]
    public void TensorSharp_SigmaObject_KeysSchedulesByStepCount()
    {
        var recipe = Load("""
        {
          "type": "qwen-image-2.1-lora",
          "sampling": {
            "steps": 6, "shift": " Dynamic ", "cfg": 1.0, "timestep": "BF16",
            "sigmas": {
              "4": [1.0, 0.75, 0.5, 0.25],
              "6": [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
            }
          }
        }
        """).Recipe;

        Assert.Equal(6, recipe.DefaultSteps);
        Assert.Equal(new[] { 4, 6 }, recipe.Nodes.Keys.OrderBy(k => k));
        Assert.Equal(ViggleSixNodes, recipe.Nodes[6]);
        Assert.Equal(QwenImage21SigmaShift.Dynamic, recipe.Shift);
        Assert.True(recipe.TimestepBf16);
        Assert.EndsWith(", bf16 timesteps", recipe.Describe(6, 4096), StringComparison.Ordinal);
    }

    [Fact]
    public void TensorSharp_SamplingWithoutSigmas_SetsOnlyDefaults()
    {
        var recipe = Load("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 12, "cfg": 2.5, "shift": null, "timestep": "f32" } }""").Recipe;

        Assert.NotNull(recipe);
        Assert.False(recipe.HasSchedule);
        Assert.Equal(12, recipe.DefaultSteps);
        Assert.Equal(2.5f, recipe.Cfg);
        Assert.Equal(QwenImage21SigmaShift.None, recipe.Shift);
        Assert.False(recipe.TimestepBf16);
    }

    [Theory]
    [InlineData("""{ "type": "flux-lora" }""", "expected 'qwen-image-2.1-lora'")]
    [InlineData("""[ 1, 2 ]""", "must be a JSON object")]
    [InlineData("""{ "rank": 16 }""", "not a recognized format")]
    [InlineData("""{ "type": 3 }""", "not a recognized format")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { } }""", "sampling.steps must name the default step count")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": { "4": [1, 0.75, 0.5, 0.25], "2": [1, 0.5] } } }""", "sampling.steps must name the default step count")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 7, "sigmas": [1, 0.5] } }""", "sampling.steps 7 has no entry")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": { "four": [1, 0.75, 0.5, 0.25] } } }""", "keys must be step counts")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": { "0": [1] } } }""", "keys must be step counts")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": { "-4": [1, 0.75, 0.5, 0.25] } } }""", "keys must be step counts")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": { "4": [1, 0.5] } } }""", "the 4-step schedule has 2 nodes")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": "1, 0.75" } }""", "must be an array or an object")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [1, 0.5, 0.5] } }""", "strictly decrease")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [0.5, 0.75] } }""", "strictly decrease")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [1.5, 0.5] } }""", "(0, 1]")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [1, -0.5] } }""", "(0, 1]")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "sigmas": [1, 0, 0.5] } }""", "(0, 1]")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 1, "sigmas": [0] } }""", "(0, 1]")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 1, "sigmas": [] } }""", "(0, 1]")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "shift": "exponential" } }""", "sampling.shift 'exponential' is not one of none, dynamic")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "timestep": "fp16" } }""", "sampling.timestep 'fp16' is not one of fp32, bf16")]
    [InlineData("""{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "cfg": 0.5 } }""", "sampling.cfg must be >= 1")]
    public void TensorSharp_MalformedConfig_IsRefusedWithTheReason(string json, string fragment)
    {
        var ex = Assert.Throws<InvalidDataException>(() => Load(json));
        Assert.Contains(fragment, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_MissingFile_ThrowsFileNotFound()
    {
        string missing = Path.Combine(_dir, "missing.json");
        var ex = Assert.Throws<FileNotFoundException>(() => QwenImage21LoraConfig.Load(missing));
        Assert.Equal(missing, ex.FileName);
    }

    // ---- recipe schedules ----------------------------------------------------------------

    private static QwenImage21LoraRecipe Viggle(QwenImage21SigmaShift shift) => new()
    {
        Source = "viggle.json",
        DefaultSteps = 6,
        Shift = shift,
        Nodes = new Dictionary<int, float[]>
        {
            [4] = new[] { 1f, 0.75f, 0.5f, 0.25f },
            [6] = ViggleSixNodes,
        },
    };

    [Fact]
    public void Recipe_DynamicShift_MatchesTheViggleScheduleAt1024()
    {
        // diffusers QwenImage21Pipeline(sigmas=nodes) with shift_terminal=None:
        // sigma = e^mu / (e^mu + 1/t - 1), mu from the 4096-token image.
        AssertClose(ViggleSix1024, Viggle(QwenImage21SigmaShift.Dynamic).Sigmas(6, 4096), 1e-4f);
    }

    [Fact]
    public void Recipe_DynamicShift_MatchesTheViggleScheduleAt2048()
    {
        AssertClose(ViggleSix2048, Viggle(QwenImage21SigmaShift.Dynamic).Sigmas(6, 16384), 1e-4f);
    }

    [Fact]
    public void Recipe_NoShift_ReturnsTheNodesVerbatimWithATerminalZero()
    {
        var recipe = Viggle(QwenImage21SigmaShift.None);

        Assert.Equal(ViggleSixNodes.Append(0f).ToArray(), recipe.Sigmas(6, 4096));
        Assert.Equal(new[] { 1f, 0.75f, 0.5f, 0.25f, 0f }, recipe.Sigmas(4, 16384));
    }

    [Fact]
    public void Recipe_UnsupportedStepCount_NamesTheSupportedCounts()
    {
        var ex = Assert.Throws<ArgumentException>(() => Viggle(QwenImage21SigmaShift.Dynamic).Sigmas(5, 4096));

        Assert.Contains("4, 6 step(s), not 5", ex.Message, StringComparison.Ordinal);
        Assert.Contains("default (6)", ex.Message, StringComparison.Ordinal);
        Assert.Contains("viggle.json", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Recipe_Describe_NamesTheScheduleKind()
    {
        string shifted = Viggle(QwenImage21SigmaShift.Dynamic).Describe(6, 4096);
        string fixedText = Viggle(QwenImage21SigmaShift.None).Describe(4, 4096);

        Assert.StartsWith("6 steps on shifted sigmas [1, 0.9677", shifted, StringComparison.Ordinal);
        Assert.EndsWith(", 0]", shifted, StringComparison.Ordinal);
        Assert.Equal("4 steps on fixed sigmas [1, 0.75, 0.5, 0.25, 0]", fixedText);
    }

    [Fact]
    public void Recipe_WithoutNodes_HasNoSchedule()
    {
        Assert.False(new QwenImage21LoraRecipe { DefaultSteps = 8 }.HasSchedule);
    }

    [Fact]
    public void ValidateNodes_DropsOnlyATrailingTerminalZero()
    {
        Assert.Equal(new[] { 1f, 0.5f }, QwenImage21LoraRecipe.ValidateNodes(new[] { 1f, 0.5f, 0f }, "x"));
        Assert.Equal(new[] { 1f, 0.5f }, QwenImage21LoraRecipe.ValidateNodes(new[] { 1f, 0.5f }, "x"));
        Assert.Equal(new[] { 0.25f }, QwenImage21LoraRecipe.ValidateNodes(new[] { 0.25f }, "x"));
        var ex = Assert.Throws<InvalidDataException>(() => QwenImage21LoraRecipe.ValidateNodes(new[] { 1f, float.NaN }, "where.json"));
        Assert.StartsWith("where.json:", ex.Message, StringComparison.Ordinal);
    }

    // ---- PEFT ----------------------------------------------------------------------------

    [Fact]
    public void Peft_AdapterConfig_ReadsAlphaRsLoraAndAlphaPattern()
    {
        var config = Load("""
        {
          "peft_type": "LORA", "r": 64, "lora_alpha": 128, "use_rslora": true, "use_dora": false,
          "target_modules": ["to_q", "to_k"],
          "alpha_pattern": { "attn.to_k": 32, "img_mlp.out": 8, "transformer_blocks.5.attn.to_q": 4 }
        }
        """, "adapter_config.json");

        Assert.Equal("PEFT adapter_config.json", config.Format);
        Assert.Equal(128f, config.Alpha);
        Assert.True(config.UseRsLora);
        Assert.Null(config.Recipe);
        Assert.Null(config.Scale);
        Assert.Equal(32f, config.AlphaFor("transformer_blocks.0.attn.to_k"));
        Assert.Equal(8f, config.AlphaFor("transformer_blocks.9.img_mlp.out"));
        Assert.Equal(4f, config.AlphaFor("transformer_blocks.5.attn.to_q"));
        Assert.Equal(32f, config.AlphaFor("attn.to_k"));
        // Suffix matching is on whole path components: 15.attn.to_q is not 5.attn.to_q.
        Assert.Equal(128f, config.AlphaFor("transformer_blocks.15.attn.to_q"));
        Assert.Equal(128f, config.AlphaFor("transformer_blocks.0.attn.to_q"));
        Assert.Equal(128f, config.AlphaFor("img_in"));
    }

    [Fact]
    public void Peft_PeftTypeWithoutAlpha_HasNoAlpha()
    {
        var config = Load("""{ "peft_type": "LORA", "r": 8, "use_dora": true }""", "adapter_config.json");

        Assert.Equal("PEFT adapter_config.json", config.Format);
        Assert.Null(config.Alpha);
        Assert.Null(config.UseRsLora); // absent: the file's own metadata decides
    }

    [Fact]
    public void Peft_EmbeddedMetadata_PrefersTheTransformerComponent()
    {
        var config = QwenImage21LoraConfig.FromAdapterMetadata("""
        {
          "text_encoder.lora_alpha": 4, "text_encoder.r": 4,
          "transformer.lora_alpha": 128, "transformer.r": 64, "transformer.use_rslora": false,
          "transformer.alpha_pattern": { "attn.to_v": 16 }
        }
        """, "file.safetensors");

        Assert.NotNull(config);
        Assert.Equal("PEFT metadata embedded in the safetensors file", config.Format);
        Assert.Equal("file.safetensors", config.Path);
        Assert.Equal(128f, config.Alpha);
        Assert.False(config.UseRsLora);
        Assert.Equal(16f, config.AlphaFor("transformer_blocks.0.attn.to_v"));
    }

    [Theory]
    [InlineData("""{ "lora_alpha": 16, "r": 8, "use_rslora": true }""", 16f, true)]
    [InlineData("""{ "unet.lora_alpha": 8, "unet.r": 4 }""", 8f, null)]
    public void Peft_EmbeddedMetadata_FallsBackToBareOrOtherComponentKeys(string json, float alpha, bool? rs)
    {
        var config = QwenImage21LoraConfig.FromAdapterMetadata(json, "f.safetensors");

        Assert.NotNull(config);
        Assert.Equal(alpha, config.Alpha);
        Assert.Equal(rs, config.UseRsLora);
    }

    [Theory]
    [InlineData("""{ "transformer.r": 64 }""")]
    [InlineData("""[ 1 ]""")]
    [InlineData("""{ }""")]
    public void Peft_EmbeddedMetadataWithoutAlpha_IsIgnored(string json)
    {
        Assert.Null(QwenImage21LoraConfig.FromAdapterMetadata(json, "f.safetensors"));
    }

    // ---- VideoX-Fun PDD ------------------------------------------------------------------

    [Fact]
    public void Pdd_Config_WithPythonListFullParameters()
    {
        var config = Load("""
        {
          "pdd_num_steps": 4,
          "pdd_block_size": 1,
          "pdd_sigmas": [1.0, 0.9, 0.7, 0.4, 0.0],
          "pdd_full_parameters": "['proj_out.weight', 'transformer_blocks.0.attn.norm_q.weight', \"txt_in.text_norm.weight\"]",
          "lora_alpha": 64
        }
        """, "pdd_config.json");

        Assert.True(config.IsPdd);
        Assert.Equal("VideoX-Fun PDD pdd_config.json", config.Format);
        Assert.Equal(4, config.PddSteps);
        Assert.Equal(64f, config.Alpha);
        Assert.Equal(new[] { "proj_out.weight", "transformer_blocks.0.attn.norm_q.weight", "txt_in.text_norm.weight" },
            config.PddFullParameters.OrderBy(n => n, StringComparer.Ordinal));
        var recipe = config.Recipe;
        Assert.NotNull(recipe);
        Assert.True(recipe.HasSchedule);
        Assert.Equal(4, recipe.DefaultSteps);
        Assert.Equal(QwenImage21SigmaShift.None, recipe.Shift);
        Assert.Equal(1f, recipe.Cfg);
        Assert.True(recipe.TimestepBf16);
        // The trained grid is used verbatim at every resolution (never shifted twice).
        Assert.Equal(new[] { 1f, 0.9f, 0.7f, 0.4f, 0f }, recipe.Sigmas(4, 4096));
        Assert.Equal(new[] { 1f, 0.9f, 0.7f, 0.4f, 0f }, recipe.Sigmas(4, 16384));
        Assert.Throws<ArgumentException>(() => recipe.Sigmas(8, 4096));
    }

    [Fact]
    public void Pdd_Config_WithJsonListFullParametersAndNetworkAlpha()
    {
        var config = Load("""
        {
          "pdd_num_steps": 2,
          "pdd_sigmas": [1.0, 0.5, 0.0],
          "pdd_full_parameters": ["proj_out.weight", "txt_in.text_norm.weight"],
          "network_alpha": 32
        }
        """, "pdd_config.json");

        Assert.True(config.IsPdd);
        Assert.Equal(32f, config.Alpha);
        Assert.Equal(2, config.PddFullParameters.Count);
        Assert.Contains("txt_in.text_norm.weight", config.PddFullParameters);
    }

    [Fact]
    public void Pdd_Config_WithAnEmptyPythonList_ReplacesNothing()
    {
        var config = Load("""{ "pdd_num_steps": 1, "pdd_sigmas": [1.0, 0.0], "pdd_full_parameters": "[]" }""", "pdd_config.json");

        Assert.Empty(config.PddFullParameters);
        Assert.Null(config.Alpha);
    }

    [Theory]
    [InlineData("""{ "pdd_num_steps": 4, "pdd_block_size": 2, "pdd_sigmas": [1, 0.9, 0.7, 0.4, 0] }""", typeof(NotSupportedException), "block size 1")]
    [InlineData("""{ "pdd_num_steps": 4 }""", typeof(InvalidDataException), "no pdd_sigmas")]
    [InlineData("""{ "pdd_num_steps": 4, "pdd_sigmas": [1, 0.5, 0] }""", typeof(InvalidDataException), "pdd_num_steps + 1 entries")]
    [InlineData("""{ "pdd_num_steps": 2, "pdd_sigmas": [0.9, 0.5, 0] }""", typeof(InvalidDataException), "from 1 to 0")]
    [InlineData("""{ "pdd_num_steps": 2, "pdd_sigmas": [1, 0.5, 0.1] }""", typeof(InvalidDataException), "from 1 to 0")]
    [InlineData("""{ "pdd_num_steps": 2, "pdd_sigmas": [1, 1, 0] }""", typeof(InvalidDataException), "strictly decrease")]
    public void Pdd_MalformedConfig_IsRefused(string json, Type exception, string fragment)
    {
        var ex = Assert.Throws(exception, () => Load(json, "pdd_config.json"));
        Assert.Contains(fragment, ex.Message, StringComparison.Ordinal);
    }

    // ---- shipped plug-in configs ---------------------------------------------------------

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    private static string[] ShippedPlugins()
    {
        string root = FindRepoRoot();
        Assert.True(root != null, "The tests run from a source checkout; config/lora/ must be reachable.");
        string dir = Path.Combine(root, "config", "lora");
        Assert.True(Directory.Exists(dir), $"{dir} is missing.");
        var files = Directory.GetFiles(dir, "*.json").OrderBy(f => f, StringComparer.Ordinal).ToArray();
        Assert.NotEmpty(files);
        return files;
    }

    private static readonly JsonDocumentOptions CommentedJson = new()
    {
        CommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private static void AssertDownloadEntry(string file, JsonElement entry, string key)
    {
        string where = $"{Path.GetFileName(file)} \"{key}\"";
        Assert.True(entry.ValueKind == JsonValueKind.Object, $"{where} must be a download entry.");
        foreach (var field in entry.EnumerateObject())
            Assert.True(field.Name is "path" or "urls" or "url" or "sha256", $"{where} has field \"{field.Name}\", which ConfigFileArgs does not read.");
        Assert.True(entry.TryGetProperty("path", out var path) && path.ValueKind == JsonValueKind.String, $"{where} has no path.");
        Assert.True(entry.TryGetProperty("urls", out var urls) && urls.ValueKind == JsonValueKind.Array && urls.GetArrayLength() > 0,
            $"{where} names no urls.");
        foreach (var url in urls.EnumerateArray())
            Assert.StartsWith("https://", url.GetString(), StringComparison.Ordinal);
        Assert.True(entry.TryGetProperty("sha256", out var sha) && sha.GetString() is { Length: 64 } hex &&
            hex.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f'), $"{where} needs a lowercase sha256.");
    }

    [Fact]
    public void ShippedPlugins_AreTensorSharpPluginsWithDownloadableWeights()
    {
        foreach (string file in ShippedPlugins())
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(file), CommentedJson);
            var root = doc.RootElement;
            Assert.Equal(LoraCliFlags.PluginType, root.GetProperty("type").GetString());
            AssertDownloadEntry(file, root.GetProperty("weights"), "weights");
            Assert.EndsWith(".safetensors", root.GetProperty("weights").GetProperty("path").GetString(), StringComparison.Ordinal);
            if (root.TryGetProperty("config", out var forwarded))
            {
                AssertDownloadEntry(file, forwarded, "config");
                Assert.False(root.TryGetProperty("sampling", out _), $"{Path.GetFileName(file)} has both config and sampling.");
            }
        }
    }

    [Fact]
    public void ShippedPlugins_ParseAsLoraConfigsWithUsableRecipes()
    {
        int recipes = 0;
        foreach (string file in ShippedPlugins())
        {
            var config = QwenImage21LoraConfig.Load(file);
            Assert.Equal("TensorSharp LoRA config", config.Format);
            Assert.False(config.IsPdd);
            if (config.Scale is { } scale) Assert.InRange(scale, 0.1f, 2f);
            if (config.Recipe == null) continue;
            recipes++;
            Assert.True(config.Recipe.HasSchedule, Path.GetFileName(file));
            Assert.Equal(1f, config.Recipe.Cfg);
            foreach (int steps in config.Recipe.Nodes.Keys)
                foreach (int tokens in new[] { 1024, 4096, 16384 })
                {
                    float[] sigmas = config.Recipe.Sigmas(steps, tokens);
                    Assert.Equal(steps + 1, sigmas.Length);
                    Assert.True(Math.Abs(sigmas[0] - 1f) < 1e-6f, $"{Path.GetFileName(file)}: first sigma {sigmas[0]}");
                    Assert.Equal(0f, sigmas[^1]);
                    for (int i = 1; i < sigmas.Length; i++) Assert.True(sigmas[i] < sigmas[i - 1], $"{Path.GetFileName(file)} {steps} steps");
                }
            _ = config.Recipe.Sigmas(config.Recipe.DefaultSteps, 4096);
        }
        Assert.True(recipes >= 3, $"Only {recipes} shipped plug-in(s) carry a sampling recipe.");
    }

    [Fact]
    public void ShippedViggleTurbo_ReproducesTheCardSchedule()
    {
        string file = ShippedPlugins().Single(f => Path.GetFileName(f) == "qwen-image-2.1-viggle-turbo.json");
        var recipe = QwenImage21LoraConfig.Load(file).Recipe;

        Assert.Equal(6, recipe.DefaultSteps);
        Assert.Equal(QwenImage21SigmaShift.Dynamic, recipe.Shift);
        Assert.Equal(new[] { 4, 5, 6, 7, 8 }, recipe.Nodes.Keys.OrderBy(k => k));
        Assert.Equal(ViggleSixNodes, recipe.Nodes[6]);
        AssertClose(ViggleSix1024, recipe.Sigmas(6, 4096), 1e-4f);
        AssertClose(ViggleSix2048, recipe.Sigmas(6, 16384), 1e-4f);
    }

    [Fact]
    public void ShippedPddPlugin_ForwardsItsPddConfigInsteadOfARecipe()
    {
        string file = ShippedPlugins().Single(f => Path.GetFileName(f) == "qwen-image-2.1-fun-acc-4step.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(file), CommentedJson);

        Assert.EndsWith("pdd_config.json", doc.RootElement.GetProperty("config").GetProperty("path").GetString(), StringComparison.Ordinal);
        Assert.Null(QwenImage21LoraConfig.Load(file).Recipe);
    }
}
