// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies that the server's CLI argument parser surfaces the new sampling
/// flags (and that env-var fallbacks layer correctly under the CLI overrides).
/// We isolate environment-variable mutation per test using a tiny RAII helper
/// so the tests are safe to run in parallel with the rest of the suite.
/// </summary>
public class ServerOptionsBuilderTests : IDisposable
{
    private readonly string _baseDir;
    private readonly EnvScope _env = new();

    public ServerOptionsBuilderTests()
    {
        // Build needs a writable base directory because it creates an
        // "uploads" folder under it. Use a temp dir per test instance to keep
        // the workspace clean.
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-server-opts-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Build_VideoMode_IsCapturedAndValidatedAtStartup()
    {
        var options = ServerOptionsBuilder.Build(new[] { "--video-mode", "ref" }, _baseDir);
        Assert.Equal("ref", options.DefaultVideoMode);

        // A typo should stop the server coming up rather than surfacing on the first
        // request an hour later.
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.Build(new[] { "--video-mode", "animate" }, _baseDir));
    }

    [Fact]
    public void Build_VideoMode_DefaultsToUnsetSoEachRequestInfersIt()
    {
        Assert.Null(ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir).DefaultVideoMode);
    }

    [Fact]
    public void Build_NoSamplingFlags_UsesSamplingConfigDefaults()
    {
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        var sampling = options.DefaultSamplingConfig;
        Assert.NotNull(sampling);
        // Match the SamplingConfig type's defaults (Ollama-compatible).
        var fallback = new SamplingConfig();
        Assert.Equal(fallback.Temperature, sampling.Temperature);
        Assert.Equal(fallback.TopK, sampling.TopK);
        Assert.Equal(fallback.TopP, sampling.TopP);
    }

    [Fact]
    public void Build_AllSamplingFlags_PopulatesDefaultSamplingConfig()
    {
        var args = new[]
        {
            "--temperature", "0.42",
            "--top-k", "12",
            "--top-p", "0.55",
            "--min-p", "0.07",
            "--repeat-penalty", "1.4",
            "--presence-penalty", "0.2",
            "--frequency-penalty", "0.3",
            "--seed", "1234",
            "--stop", "</s>",
            "--stop", "<|eot|>",
        };

        var options = ServerOptionsBuilder.Build(args, _baseDir);

        var sampling = options.DefaultSamplingConfig;
        Assert.Equal(0.42f, sampling.Temperature);
        Assert.Equal(12, sampling.TopK);
        Assert.Equal(0.55f, sampling.TopP);
        Assert.Equal(0.07f, sampling.MinP);
        Assert.Equal(1.4f, sampling.RepetitionPenalty);
        Assert.Equal(0.2f, sampling.PresencePenalty);
        Assert.Equal(0.3f, sampling.FrequencyPenalty);
        Assert.Equal(1234, sampling.Seed);
        Assert.Equal(new[] { "</s>", "<|eot|>" }, sampling.StopSequences);
    }

    [Fact]
    public void Build_EnvVarsLayerUnderCliOverrides()
    {
        // Env: temp=0.6 (will be overridden by CLI), top_k=15 (CLI absent so env wins).
        _env.Set("TENSORSHARP_TEMPERATURE", "0.6");
        _env.Set("TENSORSHARP_TOP_K", "15");

        var args = new[] { "--temperature", "0.9" };

        var options = ServerOptionsBuilder.Build(args, _baseDir);

        var sampling = options.DefaultSamplingConfig;
        // CLI wins over env for temperature.
        Assert.Equal(0.9f, sampling.Temperature);
        // No CLI for top-k -> env value applied.
        Assert.Equal(15, sampling.TopK);
        // No CLI, no env for top-p -> SamplingConfig default (0.9).
        Assert.Equal(new SamplingConfig().TopP, sampling.TopP);
    }

    [Fact]
    public void Build_InvalidTemperature_ThrowsArgumentException()
    {
        var args = new[] { "--temperature", "not-a-number" };

        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(args, _baseDir));
        Assert.Contains("--temperature", ex.Message);
    }

    [Fact]
    public void Build_InvalidTopK_ThrowsArgumentException()
    {
        var args = new[] { "--top-k", "abc" };

        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(args, _baseDir));
        Assert.Contains("--top-k", ex.Message);
    }

    [Fact]
    public void Build_DefaultSamplingConfigIsAlwaysNonNull()
    {
        // Even with zero overrides we expect a fresh, non-null config object so
        // adapters can call Clone() on it without a guard.
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        Assert.NotNull(options.DefaultSamplingConfig);
    }

    // ---- Wan video-generation defaults -------------------------------------

    [Fact]
    public void Build_NoWanVideoFlags_UsesModelSpecificDefaultsAtGenerationTime()
    {
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        // Zero is the Wan pipeline's sentinel for choosing the loaded model's
        // native defaults (33/16 generally, 49/24 for TI2V).
        Assert.Equal(0, options.DefaultVideoFrames);
        Assert.Equal(0, options.DefaultVideoFps);
    }

    [Fact]
    public void Build_WanVideoFlags_SetStartupDefaultsAndSupportEqualsForm()
    {
        var options = ServerOptionsBuilder.Build(
            new[] { "--video-frames", "81", "--fps=24", "--video-frames=121" },
            _baseDir);

        // Scalar options are last-one-wins, which also lets a real command line
        // override values expanded from --config ahead of it.
        Assert.Equal(121, options.DefaultVideoFrames);
        Assert.Equal(24, options.DefaultVideoFps);
    }

    [Theory]
    [InlineData("--video-frames", "0")]
    [InlineData("--video-frames", "-1")]
    [InlineData("--video-frames", "abc")]
    [InlineData("--fps", "0")]
    [InlineData("--fps", "-1")]
    [InlineData("--fps", "abc")]
    public void Build_InvalidWanVideoDefault_ThrowsArgumentException(string flag, string value)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.Build(new[] { flag, value }, _baseDir));

        Assert.Contains(flag, ex.Message);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_PagedKvFlag_SetsEnabledEnvVar()
    {
        _env.Set("TS_KV_PAGED_CACHE", null);
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--paged-kv" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_KV_PAGED_CACHE"));
        var cfg = PagedKvCacheConfig.FromEnvironment();
        Assert.True(cfg.Enabled);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_NoPagedKvFlag_DisablesEnabledEnvVar()
    {
        _env.Set("TS_KV_PAGED_CACHE", "1");
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--no-paged-kv" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_KV_PAGED_CACHE"));
        Assert.False(PagedKvCacheConfig.FromEnvironment().Enabled);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_AppliesBlockSizeAndCaps()
    {
        _env.Set("TS_KV_PAGED_CACHE", null);
        _env.Set("TS_KV_BLOCK_SIZE", null);
        _env.Set("TS_KV_CACHE_MAX_RAM_MB", null);
        _env.Set("TS_KV_CACHE_SSD_DIR", null);
        _env.Set("TS_KV_CACHE_MAX_SSD_MB", null);
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[]
        {
            "--paged-kv",
            "--paged-kv-block-size", "128",
            "--paged-kv-ram-mb", "2048",
            "--paged-kv-ssd-dir", "/tmp/ts-paged-ssd",
            "--paged-kv-ssd-mb", "32768",
        });
        Assert.True(applied);
        var cfg = PagedKvCacheConfig.FromEnvironment();
        Assert.True(cfg.Enabled);
        Assert.Equal(128, cfg.BlockSize);
        Assert.Equal(2048L * 1024 * 1024, cfg.MaxRamBytes);
        Assert.Equal("/tmp/ts-paged-ssd", cfg.SsdDirectory);
        Assert.Equal(32768L * 1024 * 1024, cfg.MaxSsdBytes);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_NoFlags_LeavesEnvUnchanged()
    {
        _env.Set("TS_KV_PAGED_CACHE", "1");
        _env.Set("TS_KV_BLOCK_SIZE", "256");
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--unrelated", "--value" });
        Assert.False(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_KV_PAGED_CACHE"));
        Assert.Equal("256", Environment.GetEnvironmentVariable("TS_KV_BLOCK_SIZE"));
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_RejectsBadInteger()
    {
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--paged-kv-block-size", "abc" }));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OnFlag_EnablesBothEnvVars()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        _env.Set("TS_QWEN35_BATCHED", null);
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--continuous-batching" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OffFlag_DisablesBatchedAtBothLayers()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        _env.Set("TS_QWEN35_BATCHED", "1");
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--no-continuous-batching" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_PagedBatchingAlias_BehavesSameAsCanonical()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        _env.Set("TS_QWEN35_BATCHED", null);
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--paged-batching" }));
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED"));
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--no-paged-batching" }));
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_NoFlag_LeavesEnvUnchanged()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        _env.Set("TS_QWEN35_BATCHED", "1");
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--unrelated", "value" });
        Assert.False(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OnFlag_ServerBuildDoesNotTripUnknownArgTrap()
    {
        // ParseArgs throws on unknown flags; this regression-tests that the
        // continuous-batching flag is recognised in the skip list inside
        // ParseArgs so the server boots cleanly when it's set.
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        _env.Set("TS_QWEN35_BATCHED", null);
        var options = ServerOptionsBuilder.Build(new[] { "--continuous-batching" }, _baseDir);
        Assert.NotNull(options);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_QuantBits4_SetsEnvVarAndCodecPicksItUp()
    {
        _env.Set("TS_KV_PAGED_QUANT_BITS", null);
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[]
        {
            "--paged-kv",
            "--paged-kv-quant-bits", "4",
        });
        Assert.True(applied);
        Assert.Equal("4", Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS"));

        // End-to-end: the codec factory must materialize an int4 codec from
        // the env var the flag just wrote.
        var codec = TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float16);
        Assert.NotNull(codec);
        Assert.Equal(4, codec.BitsPerElement);
        Assert.Equal("turboquant-int4", codec.Name);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_QuantBits8_SetsEnvVar()
    {
        _env.Set("TS_KV_PAGED_QUANT_BITS", null);
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[]
        {
            "--paged-kv-quant-bits", "8",
        });
        Assert.True(applied);
        Assert.Equal("8", Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS"));
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_QuantBits0_DisablesCodec()
    {
        _env.Set("TS_KV_PAGED_QUANT_BITS", "4");
        bool applied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[]
        {
            "--paged-kv-quant-bits", "0",
        });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS"));
        // 0 -> codec factory returns null (no quantization).
        Assert.Null(TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float16));
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_QuantBits_RejectsUnsupportedBitWidth()
    {
        // Anything other than 0 / 4 / 8 is rejected with a clear error so
        // operators don't silently get passthrough when they typed --quant-bits 6.
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--paged-kv-quant-bits", "6" }));
    }

    [Fact]
    public void Build_UnknownFlag_ThrowsWithTypoSuggestion()
    {
        // Repro for the user-reported bug: `--mproj` (single p) silently
        // dropped under the previous arg-parser, so the server launched with
        // no vision projector and produced text unrelated to the uploaded
        // image. Fail fast now and tell the operator what they probably meant.
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.Build(new[] { "--mproj", "/tmp/foo.gguf" }, _baseDir));
        Assert.Contains("--mproj", ex.Message);
        Assert.Contains("--mmproj", ex.Message);
    }

    [Fact]
    public void Build_PagedKvFlagsAlongsideMainFlags_DoNotTripUnknownArgCheck()
    {
        // The paged-kv flags are consumed by a separate pass before ParseArgs;
        // ParseArgs's unknown-arg guard must recognise them so the two passes
        // don't collide.
        var options = ServerOptionsBuilder.Build(
            new[]
            {
                "--paged-kv",
                "--paged-kv-block-size", "128",
                "--temperature", "0.42",
                "--no-paged-kv-cache",
            },
            _baseDir);
        Assert.Equal(0.42f, options.DefaultSamplingConfig.Temperature);
    }

    [Fact]
    public void ApplyPagedKvCacheCliFlags_QuantBits_RejectsNonInteger()
    {
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(new[] { "--paged-kv-quant-bits", "int4" }));
    }

    // ----- Vulkan GPU device selection -----

    [Fact]
    public void ApplyGpuDeviceCliFlag_SetsVulkanDeviceEnvVar()
    {
        _env.Set(TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar, null);
        bool applied = ServerOptionsBuilder.ApplyGpuDeviceCliFlag(new[] { "--gpu-device", "1" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable(TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar));
    }

    [Fact]
    public void ApplyGpuDeviceCliFlag_NoFlag_LeavesEnvUnchanged()
    {
        _env.Set(TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar, "1");
        bool applied = ServerOptionsBuilder.ApplyGpuDeviceCliFlag(new[] { "--unrelated", "value" });
        Assert.False(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable(TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar));
    }

    [Fact]
    public void ApplyGpuDeviceCliFlag_RejectsNegativeAndNonInteger()
    {
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyGpuDeviceCliFlag(new[] { "--gpu-device", "-1" }));
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyGpuDeviceCliFlag(new[] { "--gpu-device", "nvidia" }));
    }

    [Fact]
    public void Build_GpuDeviceFlag_DoesNotTripUnknownArgTrap()
    {
        // --gpu-device is consumed by ApplyGpuDeviceCliFlag before ParseArgs;
        // ParseArgs's unknown-arg guard must recognise and skip it.
        var options = ServerOptionsBuilder.Build(new[] { "--gpu-device", "1" }, _baseDir);
        Assert.NotNull(options);
    }

    // ----- Usage page / informational flags -----

    [Fact]
    public void ServerUsage_HelpRequested_RecognisesAliases()
    {
        Assert.True(ServerUsage.IsHelpRequested(new[] { "--help" }));
        Assert.True(ServerUsage.IsHelpRequested(new[] { "-h" }));
        Assert.True(ServerUsage.IsHelpRequested(new[] { "--model", "x.gguf", "--help" }));
        Assert.False(ServerUsage.IsHelpRequested(new[] { "--model", "x.gguf" }));
        Assert.False(ServerUsage.IsHelpRequested(Array.Empty<string>()));
    }

    [Fact]
    public void ServerUsage_ListGpusRequested_MatchesFlagAnywhere()
    {
        Assert.True(ServerUsage.IsListGpusRequested(new[] { "--list-gpus" }));
        Assert.True(ServerUsage.IsListGpusRequested(new[] { "--backend", "ggml_vulkan", "--list-gpus" }));
        Assert.False(ServerUsage.IsListGpusRequested(new[] { "--backend", "ggml_vulkan" }));
    }

    [Fact]
    public void ServerUsage_PrintUsage_DocumentsEveryKnownFlag()
    {
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        string usage = sw.ToString();

        // Every operator-facing flag the server accepts must appear on the
        // usage page, with defaults and an example per option.
        string[] flags =
        {
            "--model", "--mmproj", "--backend", "--gpu-device", "--list-gpus",
            "--tp", "--tp-node-id", "--tp-peers",
            "--max-tokens", "--temperature", "--top-k", "--top-p", "--min-p",
            "--video-frames", "--fps",
            "--repeat-penalty", "--presence-penalty", "--frequency-penalty",
            "--seed", "--stop", "--kv-cache-dtype",
            "--paged-kv", "--paged-kv-block-size", "--paged-kv-ram-mb",
            "--paged-kv-ssd-dir", "--paged-kv-ssd-mb", "--paged-kv-quant-bits",
            "--continuous-batching", "--prefill-chunk-size",
            "--spec", "--spec-type", "--spec-draft", "--spec-pmin", "--draft-model",
            "--qwen-image-vae", "--qwen-image-vl", "--qwen-image-mmproj", "--qwen-image-lora",
            "--wan-vae", "--wan-te", "--wan-dit2",
            "--offload-cpu",
            "--n-cpu-moe", "--cpu-moe", "--cpu-moe-threads",
            "--skill", "--list-skills",
            "--code-exec", "--code-exec-allow-install", "--code-exec-install-domains",
            "--code-exec-timeout", "--code-exec-shell", "--code-exec-max-output",
            "--config",
            "--help",
        };
        foreach (string flag in flags)
            Assert.Contains(flag, usage);

        Assert.Contains("Default:", usage);
        Assert.Contains("Example:", usage);
    }

    [Fact]
    public void ServerUsage_PrintUsage_DocumentsTheCurrentPerTokenSpeculationGate()
    {
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        string usage = sw.ToString();
        string flattened = System.Text.RegularExpressions.Regex.Replace(usage, @"\s+", " ");

        Assert.Contains("0.15 for a per-token draft head", flattened);
        Assert.DoesNotContain("0.75 for a per-token draft head", flattened);
    }

    /// <summary>
    /// The inverse of the hand-list above, for the flag families whose names live in
    /// shared constant tables: every flag such a table accepts must be mentioned
    /// SOMEWHERE on the usage page. This is the direction that actually drifted —
    /// all six --code-exec* flags were parsed and working while --help never named
    /// them, so an operator had no way to learn they existed. A prose mention
    /// counts: an alias documented inside another entry's description is documented.
    /// </summary>
    [Fact]
    public void ServerUsage_MentionsEveryFlagTheConstantTablesAccept()
    {
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        string usage = sw.ToString();

        var accepted = new List<string>
        {
            SkillHostOptions.RootsFlag, SkillHostOptions.SelectFlag, SkillHostOptions.ListFlag,
            SkillHostOptions.DisableFlag, SkillHostOptions.NoDiscoveryFlag,
            SkillHostOptions.AllowScriptsFlag, SkillHostOptions.MaxRoundsFlag,
            SkillHostOptions.SandboxFlag, SkillHostOptions.AllowNetworkFlag,
        };
        accepted.AddRange(SpeculativeCliFlags.SwitchFlags);
        accepted.AddRange(SpeculativeCliFlags.ValueFlags);
        // Driven off CodeExecOptions' own tables, in full. --code-exec-unconfined used to
        // be excluded here because the server refused it at startup; it no longer does.
        // Refusing it made --code-exec permanently inert on Windows, which has no
        // confining sandbox for a shell at all, so the server has to offer the same
        // explicit opt-in the CLI does — and therefore has to document it.
        accepted.AddRange(CodeExecOptions.SwitchFlags);
        accepted.AddRange(CodeExecOptions.ValueFlags);

        var missing = accepted.Where(f => !usage.Contains(f, StringComparison.Ordinal)).ToList();
        Assert.True(missing.Count == 0,
            "The server accepts these flags but --help never mentions them:\n  "
            + string.Join("\n  ", missing));
    }

    // ---- Wan video companion flags ----
    // These are applied by an earlier pass that READS but does not REMOVE them, so
    // the later validation pass has to recognise them too. It did not: --wan-vae and
    // --wan-te were both documented on the usage page and both made the server refuse
    // to start with "Unknown option '--wan-vae'". A --config file naming those keys
    // was therefore unusable, which is exactly what the Wan video configs need.

    [Theory]
    [InlineData("--wan-vae")]
    [InlineData("--wan-te")]
    [InlineData("--wan-dit2")]
    public void Build_WanCompanionFlags_AreAccepted(string flag)
    {
        string path = Path.Combine(_baseDir, "companion.bin");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        string env = flag switch
        {
            "--wan-vae" => "TS_WAN_VAE",
            "--wan-te" => "TS_WAN_TE",
            _ => "TS_WAN_DIT2",
        };
        string? saved = Environment.GetEnvironmentVariable(env);
        try
        {
            string[] args = { flag, path };

            // Pass 1 (what Program.cs runs first): the flag's whole job is to reach
            // the model loader as an env override.
            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));
            Assert.Equal(path, Environment.GetEnvironmentVariable(env));

            // Pass 2: that pass READS the flag but leaves it in argv, so the option
            // parser has to tolerate it. This is the half that was broken.
            Assert.NotNull(ServerOptionsBuilder.Build(args, _baseDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(env, saved);
        }
    }

    // ---- generic video companion flags, and the --wan-* aliases they replaced ----
    // The companion flags were renamed model-agnostic when a second video model arrived.
    // Both spellings must survive all THREE passes that know about them (env pass,
    // validation pass, typo-suggestion list), and both must land on the same env vars —
    // WanVideoModel still reads TS_WAN_*, so dropping that would silently break Wan.

    [Theory]
    [InlineData("--video-vae", "TS_VIDEO_VAE", "TS_WAN_VAE")]
    [InlineData("--video-text-encoder", "TS_VIDEO_TEXT_ENCODER", "TS_WAN_TE")]
    [InlineData("--video-te", "TS_VIDEO_TEXT_ENCODER", "TS_WAN_TE")]
    [InlineData("--video-dit2", "TS_VIDEO_DIT2", "TS_WAN_DIT2")]
    public void Build_VideoCompanionFlags_SetBothGenericAndLegacyEnv(
        string flag, string genericEnv, string legacyEnv)
    {
        string path = Path.Combine(_baseDir, "companion.bin");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        string? savedGeneric = Environment.GetEnvironmentVariable(genericEnv);
        string? savedLegacy = Environment.GetEnvironmentVariable(legacyEnv);
        try
        {
            string[] args = { flag, path };

            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));
            Assert.Equal(path, Environment.GetEnvironmentVariable(genericEnv));
            // The legacy name keeps being published so Wan keeps loading its companions.
            Assert.Equal(path, Environment.GetEnvironmentVariable(legacyEnv));

            // ...and the later validation pass must not reject the flag it left in argv.
            Assert.NotNull(ServerOptionsBuilder.Build(args, _baseDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(genericEnv, savedGeneric);
            Environment.SetEnvironmentVariable(legacyEnv, savedLegacy);
        }
    }

    [Fact]
    public void Build_AudioVaeFlag_IsAcceptedAndSetsItsEnv()
    {
        string path = Path.Combine(_baseDir, "audio-vae.safetensors");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        string? saved = Environment.GetEnvironmentVariable("TS_VIDEO_AUDIO_VAE");
        try
        {
            string[] args = { "--audio-vae", path };
            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));
            Assert.Equal(path, Environment.GetEnvironmentVariable("TS_VIDEO_AUDIO_VAE"));
            Assert.NotNull(ServerOptionsBuilder.Build(args, _baseDir));
        }
        finally { Environment.SetEnvironmentVariable("TS_VIDEO_AUDIO_VAE", saved); }
    }

    [Theory]
    // old spelling            new spelling                 env var they must agree on
    [InlineData("--wan-vae", "--video-vae", "TS_WAN_VAE")]
    [InlineData("--wan-te", "--video-text-encoder", "TS_WAN_TE")]
    [InlineData("--wan-dit2", "--video-dit2", "TS_WAN_DIT2")]
    public void Build_OldAndNewCompanionSpellings_AreEquivalent(
        string oldFlag, string newFlag, string env)
    {
        string path = Path.Combine(_baseDir, "companion.bin");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        string? saved = Environment.GetEnvironmentVariable(env);
        try
        {
            Environment.SetEnvironmentVariable(env, null);
            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { oldFlag, path }));
            string? viaOld = Environment.GetEnvironmentVariable(env);

            Environment.SetEnvironmentVariable(env, null);
            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { newFlag, path }));
            string? viaNew = Environment.GetEnvironmentVariable(env);

            Assert.Equal(path, viaOld);
            Assert.Equal(viaOld, viaNew);
        }
        finally { Environment.SetEnvironmentVariable(env, saved); }
    }

    // ---- MoE CPU offload (--n-cpu-moe / --cpu-moe) ----
    // These translate into the process-wide MoeCpuOffloadConfig BEFORE the
    // startup model loads, because weight residency is decided while preparing
    // the quantized weights. A parse bug here silently costs the operator the
    // VRAM the flag exists to save, so cover every accepted spelling.

    [Fact]
    public void ApplyMoeCpuOffloadCliFlags_ParsesLayerCount()
    {
        try
        {
            Assert.True(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(
                new[] { "--model", "m.gguf", "--n-cpu-moe", "32" }));
            Assert.Equal(32, TensorSharp.Models.MoeCpuOffloadConfig.CpuMoeLayers);
            Assert.False(TensorSharp.Models.MoeCpuOffloadConfig.AllLayers);
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Fact]
    public void ApplyMoeCpuOffloadCliFlags_ParsesShortAlias()
    {
        try
        {
            Assert.True(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(new[] { "-ncmoe", "8" }));
            Assert.Equal(8, TensorSharp.Models.MoeCpuOffloadConfig.CpuMoeLayers);
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Theory]
    [InlineData("--cpu-moe")]
    [InlineData("-cmoe")]
    public void ApplyMoeCpuOffloadCliFlags_ParsesAllLayersSwitch(string flag)
    {
        try
        {
            Assert.True(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(new[] { flag }));
            Assert.True(TensorSharp.Models.MoeCpuOffloadConfig.AllLayers);
            Assert.True(TensorSharp.Models.MoeCpuOffloadConfig.IsLayerOnCpu(99));
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Fact]
    public void ApplyMoeCpuOffloadCliFlags_ParsesAllKeyword()
    {
        try
        {
            Assert.True(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(new[] { "--n-cpu-moe", "all" }));
            Assert.True(TensorSharp.Models.MoeCpuOffloadConfig.AllLayers);
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Fact]
    public void ApplyMoeCpuOffloadCliFlags_ParsesThreadCount()
    {
        try
        {
            Assert.True(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(new[] { "--cpu-moe-threads", "12" }));
            Assert.Equal(12, TensorSharp.Models.MoeCpuOffloadConfig.CpuThreads);
        }
        finally
        {
            TensorSharp.Models.MoeCpuOffloadConfig.Reset();
            Environment.SetEnvironmentVariable("TS_CPU_MOE_THREADS", null);
        }
    }

    [Fact]
    public void ApplyMoeCpuOffloadCliFlags_AbsentLeavesConfigUntouched()
    {
        try
        {
            Assert.False(ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(
                new[] { "--model", "m.gguf", "--temperature", "0.7" }));
            Assert.False(TensorSharp.Models.MoeCpuOffloadConfig.IsEnabled);
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("half")]
    public void ApplyMoeCpuOffloadCliFlags_RejectsInvalidValue(string value)
    {
        try
        {
            Assert.Throws<ArgumentException>(() =>
                ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(new[] { "--n-cpu-moe", value }));
        }
        finally { TensorSharp.Models.MoeCpuOffloadConfig.Reset(); }
    }

    [Fact]
    public void Build_DoesNotTripTheUnknownArgTrapOnMoeOffloadFlags()
    {
        // The offload flags are consumed by a separate earlier pass, so Build
        // must skip them (and their values) rather than reject them.
        var options = ServerOptionsBuilder.Build(new[]
        {
            "--model", Path.Combine(_baseDir, "m.gguf"),
            "--n-cpu-moe", "32", "--cpu-moe-threads", "8", "--cpu-moe",
        }, _baseDir);
        Assert.NotNull(options);
    }

    [Fact]
    public void Build_InformationalFlags_DoNotTripUnknownArgTrap()
    {
        // Program.cs exits on --help/--list-gpus before Build runs, but Build
        // must still tolerate them (tests, future reordering of the passes).
        Assert.NotNull(ServerOptionsBuilder.Build(new[] { "--list-gpus" }, _baseDir));
        Assert.NotNull(ServerOptionsBuilder.Build(new[] { "--help" }, _baseDir));
    }

    [Fact]
    public void Build_PrefillChunkSize_DoesNotTripUnknownArgTrap()
    {
        // Regression: --prefill-chunk-size is consumed by
        // ApplyContinuousBatchingCliFlag's earlier pass but was missing from
        // ParseArgs's skip list, so passing it aborted server startup.
        _env.Set("TS_SCHED_PREFILL_CHUNK", null);
        var options = ServerOptionsBuilder.Build(new[] { "--prefill-chunk-size", "256" }, _baseDir);
        Assert.NotNull(options);
    }

    [Fact]
    public void ApplyQwenImageCompanionCliFlags_OffloadCpu_SetsEnvAndDoesNotTripUnknownArgTrap()
    {
        _env.Set("TS_QWEN_IMAGE_OFFLOAD_CPU", null);
        bool applied = ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { "--offload-cpu" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_OFFLOAD_CPU"));
        // The boolean flag has no value; the main parser must skip it, not abort.
        Assert.NotNull(ServerOptionsBuilder.Build(new[] { "--offload-cpu" }, _baseDir));
    }

    // ----- Tensor-parallelism CLI flags -----

    [Fact]
    public void ApplyTensorParallelCliFlags_TpFlag_SetsDegreeEnvVar()
    {
        _env.Set("TENSORSHARP_TP_DEGREE", null);
        bool applied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp", "2" });
        Assert.True(applied);
        Assert.Equal("2", Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_InlineEqualsForm_IsAccepted()
    {
        _env.Set("TENSORSHARP_TP_DEGREE", null);
        bool applied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp=4" });
        Assert.True(applied);
        Assert.Equal("4", Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_NoFlags_LeavesEnvUnchanged()
    {
        _env.Set("TENSORSHARP_TP_DEGREE", "2");
        bool applied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--unrelated", "value" });
        Assert.False(applied);
        Assert.Equal("2", Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_RejectsZeroNegativeAndNonInteger()
    {
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp", "0" }));
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp", "-2" }));
        Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp", "two" }));
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_DistributedPair_SetsBothEnvVars()
    {
        _env.Set("TENSORSHARP_TP_DEGREE", null);
        _env.Set("TENSORSHARP_TP_NODE_ID", null);
        _env.Set("TENSORSHARP_TP_PEERS", null);
        bool applied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[]
        {
            "--tp", "2",
            "--tp-node-id", "0",
            "--tp-peers", "192.168.1.10:9500,192.168.1.11:9500",
        });
        Assert.True(applied);
        Assert.Equal("2", Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
        Assert.Equal("0", Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID"));
        Assert.Equal("192.168.1.10:9500,192.168.1.11:9500", Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS"));
        // The model loader's config factory must see the distributed pair.
        var cfg = TensorSharp.Distributed.DistributedTpConfig.TryFromEnvironment(localDegree: 2);
        Assert.NotNull(cfg);
        Assert.Equal(0, cfg.NodeId);
        Assert.Equal(2, cfg.PeerEndpoints.Length);
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_NodeIdWithoutPeers_ThrowsFailFast()
    {
        _env.Set("TENSORSHARP_TP_NODE_ID", null);
        _env.Set("TENSORSHARP_TP_PEERS", null);
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp-node-id", "0" }));
        Assert.Contains("--tp-peers", ex.Message);
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_PeersWithoutNodeId_ThrowsFailFast()
    {
        _env.Set("TENSORSHARP_TP_NODE_ID", null);
        _env.Set("TENSORSHARP_TP_PEERS", null);
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp-peers", "10.0.0.1:9500,10.0.0.2:9500" }));
        Assert.Contains("--tp-node-id", ex.Message);
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_NodeIdFlagWithPeersFromEnv_IsAccepted()
    {
        // One half of the distributed pair may legitimately come from the
        // environment; only a half-configured RESULT should fail.
        _env.Set("TENSORSHARP_TP_NODE_ID", null);
        _env.Set("TENSORSHARP_TP_PEERS", "10.0.0.1:9500,10.0.0.2:9500");
        bool applied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[] { "--tp-node-id", "1" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID"));
    }

    [Fact]
    public void ApplyTensorParallelCliFlags_MalformedPeers_ThrowsWithFlagName()
    {
        _env.Set("TENSORSHARP_TP_NODE_ID", null);
        _env.Set("TENSORSHARP_TP_PEERS", null);
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyTensorParallelCliFlags(new[]
            {
                "--tp-node-id", "0",
                "--tp-peers", "not-an-endpoint",
            }));
        Assert.Contains("--tp-peers", ex.Message);
    }

    [Fact]
    public void Build_TensorParallelFlags_DoNotTripUnknownArgTrap()
    {
        // The TP flags are consumed by ApplyTensorParallelCliFlags before
        // ParseArgs; ParseArgs's unknown-arg guard must recognise and skip them.
        var options = ServerOptionsBuilder.Build(new[]
        {
            "--tp", "2",
            "--tp-node-id", "0",
            "--tp-peers", "10.0.0.1:9500,10.0.0.2:9500",
        }, _baseDir);
        Assert.NotNull(options);
    }

    // ----- speculative-decoding CLI flags -----

    [Fact]
    public void ApplySpeculativeCliFlags_SpecFlag_EnablesSchedulerSpeculation()
    {
        _env.ClearSpeculationVars();
        bool applied = ServerOptionsBuilder.ApplySpeculativeCliFlags(new[] { "--spec" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void ApplySpeculativeCliFlags_NoSpecFlag_DisablesSpeculation()
    {
        _env.ClearSpeculationVars();
        _env.Set("TS_MTP_SPEC", "1");
        bool applied = ServerOptionsBuilder.ApplySpeculativeCliFlags(new[] { "--no-spec" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void ApplySpeculativeCliFlags_MissingDraftModelFile_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplySpeculativeCliFlags(
                new[] { "--draft-model", Path.Combine(_baseDir, "does-not-exist.gguf") }));
        Assert.Contains("--draft-model", ex.Message);
    }

    [Fact]
    public void ApplySpeculativeCliFlags_OneDraftModel_ReachesEveryConsumer()
    {
        // There used to be two flags for one intent: --draft-model fed the model
        // FACTORY (a block drafter must be resident before the layer split) and
        // --spec-draft-model fed the attach-after-load path (a per-token head).
        // The operator cannot be expected to know which their file needs, and the
        // loaders already probe the GGUF's own declared architecture - so the ONE
        // surviving flag publishes to both channels, the loaders route by what
        // the file says it is, and TryAttachConfiguredDraftHead skips a drafter
        // the factory already attached. Naming the file also IS the request:
        // speculation turns on without a separate --spec.
        _env.Set("TS_DSV4_DSPARK", null);
        _env.Set("TS_QWEN35_DFLASH", null);
        _env.Set("TS_MUSE_GLIMMER_DFLASH", null);
        _env.ClearSpeculationVars();
        string draftFile = Path.Combine(_baseDir, "drafter.gguf");
        File.WriteAllText(draftFile, "stub");   // the parser validates File.Exists

        bool applied = ServerOptionsBuilder.ApplySpeculativeCliFlags(new[]
        {
            "--spec-draft", "5",
            "--draft-model", draftFile,
        });

        Assert.True(applied);
        // The window routed exactly, never by prefix.
        Assert.Equal("5", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
        // The factory channel, all three architectures.
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_DSV4_DSPARK"));
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_QWEN35_DFLASH"));
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_DFLASH"));
        // The attach-after-load channel.
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_MTP_DRAFT_MODEL"));
        // Naming the file is the request.
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void ApplySpeculativeCliFlags_MissingBlockDraftFile_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplySpeculativeCliFlags(
                new[] { "--draft-model", Path.Combine(_baseDir, "does-not-exist.gguf") }));
        Assert.Contains("--draft-model", ex.Message);
    }

    [Fact]
    public void SchedulerConfig_UnsetPmin_LeavesTheGateToTheDrafter()
    {
        // A per-token head and a block drafter threshold different quantities,
        // so an unset --mtp-pmin must stay unset rather than baking in either
        // one's default.
        _env.ClearSpeculationVars();
        Assert.Null(SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);

        _env.Set("TS_MTP_PMIN", "0.5");
        Assert.Equal(0.5f, SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Fact]
    public void SpeculationStartupValidation_NoActivationError_ReturnsNull()
    {
        Assert.Null(SpeculationStartupValidation.GetFatalActivationError(null));
        Assert.Null(SpeculationStartupValidation.GetFatalActivationError(string.Empty));
    }

    [Fact]
    public void MtpStartupValidation_ActivationError_ReturnsFatalMessageWithReasonAndHint()
    {
        // Repro for the user-reported bug: pairing the 12B target with the 26B-A4B
        // draft fails the backbone-dim check; that reason used to be a warning the
        // operator never saw, so the server ran with speculation silently off.
        // Startup must now fail fast, surfacing the reason plus a remediation hint.
        const string reason = "MTP draft backbone dim 2816 != target hidden size 3840.";
        string msg = SpeculationStartupValidation.GetFatalActivationError(reason);
        Assert.NotNull(msg);
        Assert.Contains(reason, msg);
        Assert.Contains("--draft-model", msg);
        Assert.Contains("embedding_length_out", msg);
    }

    // ---- Listen address (--port / --host / --urls) -------------------------
    // The ambient environment can carry PORT / HOST / ASPNETCORE_URLS (container
    // platforms inject them), so every test here clears all three first and then
    // sets only what it is exercising.

    private void ClearListenEnv()
    {
        _env.Set("PORT", null);
        _env.Set("HOST", null);
        _env.Set("ASPNETCORE_URLS", null);
    }

    private string BuildListenUrls(params string[] args)
    {
        return ServerOptionsBuilder.Build(args, _baseDir).ListenUrls;
    }

    [Fact]
    public void Build_NoListenFlags_UsesDefaultAddress()
    {
        ClearListenEnv();
        Assert.Equal("http://0.0.0.0:5000", BuildListenUrls());
    }

    [Fact]
    public void Build_PortFlag_OverridesDefaultPortAndKeepsDefaultHost()
    {
        ClearListenEnv();
        Assert.Equal("http://0.0.0.0:8080", BuildListenUrls("--port", "8080"));
        // The `--flag=value` form is supported by TryReadOption for every option.
        Assert.Equal("http://0.0.0.0:8080", BuildListenUrls("--port=8080"));
    }

    [Fact]
    public void Build_HostFlagAlone_KeepsDefaultPort()
    {
        ClearListenEnv();
        Assert.Equal("http://127.0.0.1:5000", BuildListenUrls("--host", "127.0.0.1"));
    }

    [Fact]
    public void Build_HostAndPortFlags_CombineIntoOneUrl()
    {
        ClearListenEnv();
        Assert.Equal("http://127.0.0.1:8080", BuildListenUrls("--host", "127.0.0.1", "--port", "8080"));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("65536")]
    [InlineData("-1")]
    [InlineData("abc")]
    [InlineData("")]
    public void Build_InvalidPort_Throws(string port)
    {
        ClearListenEnv();
        var ex = Assert.Throws<ArgumentException>(() => BuildListenUrls("--port", port));
        Assert.Contains("--port", ex.Message);
    }

    [Fact]
    public void Build_UrlsFlag_IsUsedVerbatim()
    {
        ClearListenEnv();
        Assert.Equal(
            "http://0.0.0.0:8080;https://0.0.0.0:8443",
            BuildListenUrls("--urls", "http://0.0.0.0:8080;https://0.0.0.0:8443"));
    }

    [Fact]
    public void Build_PortFlag_WinsOverUrlsFlag()
    {
        // --port is the more specific expression of intent, so it takes the
        // whole binding rather than being merged into the --urls list.
        ClearListenEnv();
        Assert.Equal("http://0.0.0.0:9999", BuildListenUrls("--urls", "http://0.0.0.0:8080", "--port", "9999"));
    }

    [Fact]
    public void Build_PortEnvVar_UsedWhenNoFlag()
    {
        ClearListenEnv();
        _env.Set("PORT", "7860");
        Assert.Equal("http://0.0.0.0:7860", BuildListenUrls());
    }

    [Fact]
    public void Build_HostEnvVar_UsedWhenNoFlag()
    {
        ClearListenEnv();
        _env.Set("HOST", "127.0.0.1");
        Assert.Equal("http://127.0.0.1:5000", BuildListenUrls());
    }

    [Fact]
    public void Build_PortFlag_WinsOverPortEnvVar()
    {
        ClearListenEnv();
        _env.Set("PORT", "7860");
        Assert.Equal("http://0.0.0.0:8080", BuildListenUrls("--port", "8080"));
    }

    [Fact]
    public void Build_InvalidPortEnvVar_Throws()
    {
        ClearListenEnv();
        _env.Set("PORT", "not-a-port");
        var ex = Assert.Throws<ArgumentException>(() => BuildListenUrls());
        Assert.Contains("PORT", ex.Message);
    }

    [Fact]
    public void Build_AspNetCoreUrlsEnvVar_HonouredInsteadOfSilentlyIgnored()
    {
        // app.Run(url) overrides whatever the host builder picked up, so this
        // variable only works because the resolver folds it in explicitly.
        ClearListenEnv();
        _env.Set("ASPNETCORE_URLS", "http://0.0.0.0:6001");
        Assert.Equal("http://0.0.0.0:6001", BuildListenUrls());
    }

    [Fact]
    public void Build_PortEnvVar_WinsOverAspNetCoreUrls()
    {
        ClearListenEnv();
        _env.Set("ASPNETCORE_URLS", "http://0.0.0.0:6001");
        _env.Set("PORT", "7860");
        Assert.Equal("http://0.0.0.0:7860", BuildListenUrls());
    }

    [Fact]
    public void Build_CliFlags_WinOverAspNetCoreUrls()
    {
        ClearListenEnv();
        _env.Set("ASPNETCORE_URLS", "http://0.0.0.0:6001");
        Assert.Equal("http://0.0.0.0:8080", BuildListenUrls("--port", "8080"));
    }

    [Fact]
    public void Build_IPv6Host_IsBracketedIntoAValidUrl()
    {
        ClearListenEnv();
        Assert.Equal("http://[::1]:8080", BuildListenUrls("--host", "::1", "--port", "8080"));
        // Already-bracketed input must not be double-bracketed.
        Assert.Equal("http://[::1]:8080", BuildListenUrls("--host", "[::1]", "--port", "8080"));
    }

    [Fact]
    public void Build_HostWithScheme_PreservesScheme()
    {
        ClearListenEnv();
        Assert.Equal("https://0.0.0.0:8443", BuildListenUrls("--host", "https://0.0.0.0", "--port", "8443"));
    }

    [Fact]
    public void Build_ResolvedListenUrls_IsAParseableUrl()
    {
        // Guards the string composition: whatever we hand to app.Run has to be
        // something Kestrel can actually parse as an endpoint.
        ClearListenEnv();
        foreach (string[] args in new[]
        {
            new[] { "--port", "8080" },
            new[] { "--host", "127.0.0.1", "--port", "8080" },
            new[] { "--host", "::1", "--port", "8080" },
            Array.Empty<string>(),
        })
        {
            string url = BuildListenUrls(args);
            Assert.True(Uri.TryCreate(url, UriKind.Absolute, out Uri parsed), $"not a valid URL: {url}");
            Assert.Equal("http", parsed.Scheme);
        }
    }

    [Fact]
    public void Build_UnknownPortLikeFlag_SuggestsPort()
    {
        ClearListenEnv();
        var ex = Assert.Throws<ArgumentException>(() => BuildListenUrls("--prot", "8080"));
        Assert.Contains("--port", ex.Message);
    }

    // ---- Upload storage limits -------------------------------------------

    [Fact]
    public void Build_NoUploadFlags_KeepsPermissiveDefaults()
    {
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        Assert.Equal(500L * 1024 * 1024, options.UploadMaxFileBytes);
        Assert.Equal(0, options.UploadQuotaBytes);
        Assert.Null(options.UploadTtl);
    }

    [Fact]
    public void Build_UploadFlags_ResolveToBytesAndTimeSpan()
    {
        var options = ServerOptionsBuilder.Build(
            new[] { "--upload-max-mb", "25", "--upload-quota-mb", "2048", "--upload-ttl-hours", "1.5" },
            _baseDir);

        Assert.Equal(25L * 1024 * 1024, options.UploadMaxFileBytes);
        Assert.Equal(2048L * 1024 * 1024, options.UploadQuotaBytes);
        Assert.Equal(TimeSpan.FromMinutes(90), options.UploadTtl);
    }

    [Fact]
    public void Build_UploadEnvVars_LayerUnderCliOverrides()
    {
        _env.Set("TS_UPLOAD_MAX_MB", "10");
        _env.Set("TS_UPLOAD_QUOTA_MB", "512");
        _env.Set("TS_UPLOAD_TTL_HOURS", "24");

        var options = ServerOptionsBuilder.Build(new[] { "--upload-max-mb", "50" }, _baseDir);

        // CLI wins over env for the per-file cap.
        Assert.Equal(50L * 1024 * 1024, options.UploadMaxFileBytes);
        // No CLI for the others -> env values applied.
        Assert.Equal(512L * 1024 * 1024, options.UploadQuotaBytes);
        Assert.Equal(TimeSpan.FromHours(24), options.UploadTtl);
    }

    [Theory]
    [InlineData("--upload-max-mb", "0")]
    [InlineData("--upload-max-mb", "abc")]
    [InlineData("--upload-quota-mb", "-5")]
    [InlineData("--upload-ttl-hours", "0")]
    [InlineData("--upload-ttl-hours", "soon")]
    public void Build_InvalidUploadValues_ThrowArgumentException(string flag, string value)
    {
        var ex = Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.Build(new[] { flag, value }, _baseDir));
        Assert.Contains(flag, ex.Message);
    }

    [Fact]
    public void Build_Default_WebUiEnabled()
    {
        _env.Set("TS_NO_WEBUI", null);
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        Assert.True(options.WebUiEnabled);
    }

    [Fact]
    public void Build_NoWebUiFlag_DisablesWebUi()
    {
        _env.Set("TS_NO_WEBUI", null);
        var options = ServerOptionsBuilder.Build(new[] { "--no-webui" }, _baseDir);
        Assert.False(options.WebUiEnabled);
    }

    [Fact]
    public void Build_NoWebUiEnvVar_DisablesWebUi()
    {
        _env.Set("TS_NO_WEBUI", "1");
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        Assert.False(options.WebUiEnabled);
    }

    [Fact]
    public void Build_NoWebUiEnvVarZero_KeepsWebUiEnabled()
    {
        _env.Set("TS_NO_WEBUI", "0");
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        Assert.True(options.WebUiEnabled);
    }

    [Fact]
    public void Build_NoWebUiFlag_OverridesEnvVarZero()
    {
        _env.Set("TS_NO_WEBUI", "0");
        var options = ServerOptionsBuilder.Build(new[] { "--no-webui" }, _baseDir);
        Assert.False(options.WebUiEnabled);
    }

    // ---- Usage page vs parser: the whole class of "documented but rejected" ----
    //
    // The server applies several flag families in passes that run BEFORE
    // ServerOptionsBuilder.Build and that READ argv without removing anything.
    // Build then walks the same argv and throws "Unknown option" for whatever it
    // does not explicitly recognise. Every such family therefore needs an entry in
    // Build's skip list, and twice one was missed: --wan-vae/--wan-te (fixed with a
    // one-off test below), then EVERY --spec* spelling, which made
    //   TensorSharp.Server --model m.gguf --draft-model d.gguf --mtp-spec --spec-draft 3
    // die with `Unknown option '--spec-draft'` even though --help documents it.
    //
    // These tests close the class instead of the instance: they enumerate the usage
    // page itself, so a flag can never again be documented-but-rejected, or
    // accepted-but-unsuggestible, without a red test.

    /// <summary>A plausible value for a flag, keyed on the placeholder its usage
    /// entry declares. Files must exist because several appliers stat them.</summary>
    private string[] SampleArgsFor(string flag, string usage)
    {
        string filePath = Path.Combine(_baseDir, "sample.gguf");
        if (!File.Exists(filePath)) File.WriteAllBytes(filePath, new byte[] { 1, 2, 3, 4 });

        // Value flags are the ones the usage page renders as "<flag> <placeholder>".
        int idx = usage.IndexOf(flag + " <", StringComparison.Ordinal);
        if (idx < 0)
            return new[] { flag };                       // bare switch

        int lt = idx + flag.Length + 1;
        int gt = usage.IndexOf('>', lt);
        string placeholder = gt > lt ? usage.Substring(lt + 1, gt - lt - 1) : string.Empty;

        string value = placeholder switch
        {
            "path" or "path|none" or "dir" => filePath,
            "url" => "localhost:6379",
            "t" => "f16",
            "name" => "ngram",
            "type" => "ggml_cpu",
            "mode" => "ref",
            "config|request" => "config",
            "list" => "10.0.0.1:9500",
            "text" => "</s>",
            "address" => "127.0.0.1",
            "urls" => "http://0.0.0.0:18099",
            "f" or "p" or "x" => "0.5",
            _ => "1",
        };
        return new[] { flag, value };
    }

    [Fact]
    public void Build_AcceptsEveryFlagOnTheUsagePage()
    {
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        string usage = sw.ToString();

        var rejected = new List<string>();
        int checkedFlags = 0;
        foreach (string flag in ServerUsage.DocumentedFlags())
        {
            // --config is consumed and REMOVED by ConfigFileArgs.Expand before
            // Build ever sees it, so Build legitimately does not know it.
            if (flag == "--config") continue;
            checkedFlags++;

            using var scope = new EnvScope();
            string[] args = SampleArgsFor(flag, usage);
            Exception ex = Record.Exception(() =>
            {
                ServerOptionsBuilder.ApplySpeculativeCliFlags(args);
                ServerOptionsBuilder.Build(args, _baseDir);
            });
            if (ex is ArgumentException ae &&
                ae.Message.StartsWith("Unknown option", StringComparison.Ordinal))
            {
                rejected.Add(flag + " -> " + ae.Message);
            }
        }

        // Guard against a vacuous pass: an accessor that yielded nothing would
        // otherwise make this test green while checking nothing at all.
        Assert.True(checkedFlags > 40, $"DocumentedFlags() yielded only {checkedFlags} flags.");
        Assert.True(rejected.Count == 0,
            "These flags are on the --help page but ServerOptionsBuilder.Build rejects them:\n  "
            + string.Join("\n  ", rejected));
    }

    [Fact]
    public void SuggestFlagCorrection_KnowsEverySpeculativeSpelling()
    {
        // A typo near a real flag must suggest THAT flag. Before the fix "--spe"
        // suggested "--seed" (Levenshtein 2) because no --spec* name was in the
        // known-flag table at all - an actively misleading hint.
        foreach (string flag in SpeculativeCliFlags.SwitchFlags)
        {
            string typo = flag.Substring(0, flag.Length - 1);
            var ex = Assert.Throws<ArgumentException>(
                () => ServerOptionsBuilder.Build(new[] { typo }, _baseDir));
            Assert.Contains("Did you mean '" + flag + "'", ex.Message);
        }
    }

    [Theory]
    [InlineData("--spec")]
    [InlineData("--no-spec")]
    [InlineData("--spec-draft")]
    [InlineData("--spec-type")]
    [InlineData("--spec-pmin")]
    [InlineData("--draft-model")]
    public void Build_SpeculativeFlags_SurviveBothPasses(string flag)
    {
        string draft = Path.Combine(_baseDir, "draft.gguf");
        File.WriteAllBytes(draft, new byte[] { 1, 2, 3, 4 });

        string[] args = flag switch
        {
            "--spec" or "--no-spec" => new[] { flag },
            "--spec-draft" => new[] { flag, "3" },
            "--spec-type" => new[] { flag, "ngram" },
            "--spec-pmin" => new[] { flag, "0.6" },
            _ => new[] { flag, draft },
        };

        // Pass 1: the applier consumes it.
        ServerOptionsBuilder.ApplySpeculativeCliFlags(args);

        // Pass 2: the unknown-arg trap must not trip on that very same argv.
        var ex = Record.Exception(() => ServerOptionsBuilder.Build(args, _baseDir));
        Assert.False(
            ex is ArgumentException ae && ae.Message.StartsWith("Unknown option", StringComparison.Ordinal),
            flag + " was consumed by ApplySpeculativeCliFlags but rejected by Build: " + ex?.Message);

        // And the "=" spelling, which takes TryReadOne's prefix branch.
        if (args.Length == 2)
        {
            string[] eqArgs = { flag + "=" + args[1] };
            ServerOptionsBuilder.ApplySpeculativeCliFlags(eqArgs);
            var ex2 = Record.Exception(() => ServerOptionsBuilder.Build(eqArgs, _baseDir));
            Assert.False(
                ex2 is ArgumentException ae2 && ae2.Message.StartsWith("Unknown option", StringComparison.Ordinal),
                flag + "=VALUE was rejected by Build: " + ex2?.Message);
        }
    }

    [Fact]
    public void Build_RemovedSpeculativeSpellings_ErrorWithAPointerToTheSurvivor()
    {
        // The removed duplicates must fail LOUDLY through the server's own entry
        // path, not survive as hidden aliases and not fall to a bare "Unknown
        // option" (Levenshtein('--mtp-spec','--spec') is above the suggestion
        // cutoff, so without RejectRemoved the operator would get no pointer at
        // all). Driven off the shared table so a spelling removed later cannot
        // dodge the guard.
        Assert.NotEmpty(SpeculativeCliFlags.RemovedFlags);
        foreach ((string flag, string survivor) in SpeculativeCliFlags.RemovedFlags)
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                ServerOptionsBuilder.ApplySpeculativeCliFlags(new[] { flag, "1" }));
            Assert.Contains(flag, ex.Message);
            Assert.Contains(survivor, ex.Message);
        }
    }

    [Fact]
    public void ConfigFile_RemovedSpecKeys_FailWithThePointerToo()
    {
        // Shipped configs used {"mtp-spec": true, "mtp-draft-model": "..."}; after
        // the rename a stale file must produce the same migration error as the
        // command line, since ConfigFileArgs.Expand turns keys into flags.
        string cfg = Path.Combine(_baseDir, "stale-spec.json");
        File.WriteAllText(cfg, "{ \"mtp-spec\": true }");

        string[] expanded = ConfigFileArgs.Expand(new[] { "--config", cfg });
        var ex = Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.ApplySpeculativeCliFlags(expanded));
        Assert.Contains("--mtp-spec", ex.Message);
        Assert.Contains("--spec", ex.Message);
    }

    [Fact]
    public void ConfigFile_SpecKeys_StartTheServer()
    {
        // A --config file naming the current spellings must work end to end:
        // ConfigFileArgs.Expand turns {"spec-draft": 3} into --spec-draft 3, which
        // then has to survive Build. This is the surface the shipped configs use.
        string cfg = Path.Combine(_baseDir, "spec.json");
        File.WriteAllText(cfg, "{ \"spec\": true, \"spec-draft\": 3, \"spec-pmin\": 0.6 }");

        string[] expanded = ConfigFileArgs.Expand(new[] { "--config", cfg });
        ServerOptionsBuilder.ApplySpeculativeCliFlags(expanded);
        var ex = Record.Exception(() => ServerOptionsBuilder.Build(expanded, _baseDir));
        Assert.False(
            ex is ArgumentException ae && ae.Message.StartsWith("Unknown option", StringComparison.Ordinal),
            "config-file spec keys were rejected: " + ex?.Message);
    }

}
