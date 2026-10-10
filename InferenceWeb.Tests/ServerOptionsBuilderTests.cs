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
using TensorSharp.Runtime.Scheduling.PrefixCache;
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
        _env.ClearSpeculationVars();
        foreach (string variable in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TENSORSHARP_TP_NODE_ID", "TENSORSHARP_TP_PEERS" })
            _env.Set(variable, null);
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
    public void NoPrefixCache_DisablesRuntimeReuseAndStartupPersistence()
    {
        _env.Set("TS_SCHED_PREFIX_CACHE", "1");
        string[] args = { "--no-prefix-cache" };

        Assert.True(ServerOptionsBuilder.ApplyPrefixCacheCliFlag(args));
        Assert.False(SchedulerConfig.FromEnvironment().EnablePrefixCaching);
        Assert.False(ServerOptionsBuilder.Build(args, _baseDir).PrefixCacheEnabled);
    }

    [Fact]
    public void NoPrefixCache_AbsentPreservesRuntimeEnvironment()
    {
        _env.Set("TS_SCHED_PREFIX_CACHE", "1");

        Assert.False(ServerOptionsBuilder.ApplyPrefixCacheCliFlag(Array.Empty<string>()));
        Assert.True(SchedulerConfig.FromEnvironment().EnablePrefixCaching);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SpecFlag_KeepsRadixCachingUnlessExplicitlyDisabled(bool disablePrefixCache)
    {
        _env.Set("TS_SCHED_PREFIX_CACHE", null);
        _env.Set("TS_PREFIX_CACHE_MODE", null);
        string[] args = disablePrefixCache
            ? new[] { "--spec", "--no-prefix-cache" }
            : new[] { "--spec" };

        ServerOptionsBuilder.ApplySpeculativeCliFlags(args);
        ServerOptionsBuilder.ApplyPrefixCacheCliFlag(args);

        var config = SchedulerConfig.FromEnvironment();
        Assert.True(config.Speculation.Enabled);
        Assert.Equal(!disablePrefixCache, config.EnablePrefixCaching);
        Assert.Equal(!disablePrefixCache, ServerOptionsBuilder.Build(args, _baseDir).PrefixCacheEnabled);
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OnFlag_EnablesTheBatchedPath()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--continuous-batching" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OffFlag_DisablesTheBatchedPath()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--no-continuous-batching" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_NoFlag_LeavesEnvUnchanged()
    {
        _env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        bool applied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(new[] { "--unrelated", "value" });
        Assert.False(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
    }

    [Fact]
    public void ApplyContinuousBatchingCliFlag_OnFlag_ServerBuildDoesNotTripUnknownArgTrap()
    {
        // ParseArgs throws on unknown flags; this regression-tests that the
        // continuous-batching flag is recognised in the skip list inside
        // ParseArgs so the server boots cleanly when it's set.
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        var options = ServerOptionsBuilder.Build(new[] { "--continuous-batching" }, _baseDir);
        Assert.NotNull(options);
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
            "--tp", "--layer-split", "--tp-node-id", "--tp-peers",
            "--max-tokens", "--temperature", "--top-k", "--top-p", "--min-p",
            "--video-frames", "--fps",
            "--repeat-penalty", "--presence-penalty", "--frequency-penalty",
            "--seed", "--stop", "--kv-cache-dtype",
            "--continuous-batching", "--prefill-chunk-size",
            "--spec", "--spec-type", "--spec-draft", "--spec-pmin", "--draft-model",
            "--qwen-image-vae", "--qwen-image-vl", "--qwen-image-mmproj", "--qwen-image-variant",
            "--video-vae", "--video-text-encoder", "--video-dit2", "--audio-vae",
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

    // ---- video companion flags ----
    // These are applied by an earlier pass that READS but does not REMOVE them, so the
    // later validation pass has to recognise them too, or a --config file naming them
    // makes the server refuse to start with "Unknown option".

    [Theory]
    [InlineData("--video-vae", "TS_VIDEO_VAE")]
    [InlineData("--video-text-encoder", "TS_VIDEO_TEXT_ENCODER")]
    [InlineData("--video-dit2", "TS_VIDEO_DIT2")]
    public void Build_VideoCompanionFlags_SetTheirEnv(string flag, string env)
    {
        string path = Path.Combine(_baseDir, "companion.bin");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
        string? saved = Environment.GetEnvironmentVariable(env);
        try
        {
            string[] args = { flag, path };

            Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));
            Assert.Equal(path, Environment.GetEnvironmentVariable(env));

            // ...and the later validation pass must not reject the flag it left in argv.
            Assert.NotNull(ServerOptionsBuilder.Build(args, _baseDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(env, saved);
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

    // ----- Options removed with the Qwen-Image-Edit-2511 pipeline -----

    /// <summary>Every spelling a removed flag can arrive in: spaced, joined and case-folded.</summary>
    public static IEnumerable<object[]> RemovedFlagSpellings()
    {
        foreach ((string flag, _) in RemovedCliFlags.RemovedFlags)
        {
            yield return new object[] { flag, new[] { flag, "lora.safetensors" } };
            yield return new object[] { flag, new[] { flag + "=lora.safetensors" } };
            yield return new object[] { flag, new[] { flag.ToUpperInvariant() } };
            yield return new object[] { flag, new[] { "--backend", "ggml_cpu", flag } };
        }
    }

    [Theory]
    [MemberData(nameof(RemovedFlagSpellings))]
    public void Build_RemovedFlags_FailWithWhyAndWhatToDoInstead(string flag, string[] args)
    {
        // Not a bare "Unknown option": the operator needs to know why the option went
        // and what to use instead - Qwen-Image-2.1's own options for the retired
        // pipeline's, the surviving spelling for the CLI-only penalty-window name.
        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(args, _baseDir));

        Assert.Equal(RemovedCliFlags.Describe(flag), ex.Message);
        Assert.StartsWith(flag + " was removed:", ex.Message, StringComparison.Ordinal);
        string survivor = flag switch
        {
            "--penalty-last-n" => "--repeat-last-n",
            "--paged-batching" => "Use --continuous-batching instead",
            "--no-paged-batching" => "Use --no-continuous-batching instead",
            "--wan-vae" => "Use --video-vae instead",
            "--wan-te" or "--video-te" => "Use --video-text-encoder instead",
            "--wan-dit2" => "Use --video-dit2 instead",
            "--offload-cpu" or "--qwen-image-lora" => "Qwen-Image-2.1",
            _ => "radix prefix cache",
        };
        Assert.Contains(survivor, ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("Unknown option", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RemovedQwenImageFlags_AreNoLongerAppliedOrPublished()
    {
        // Their env vars had no reader left, and a published TS_QWEN_IMAGE_LORA now makes
        // Qwen-Image-2.1 refuse to run, so the companion pass must not know them at all.
        _env.Set("TS_QWEN_IMAGE_LORA", null);
        _env.Set("TS_QWEN_IMAGE_OFFLOAD_CPU", null);

        bool applied = ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(
            new[] { "--qwen-image-lora", "lora.safetensors", "--offload-cpu" });

        Assert.False(applied);
        Assert.Null(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_LORA"));
        Assert.Null(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_OFFLOAD_CPU"));
    }

    [Theory]
    [InlineData("--offload-cpuu")]
    [InlineData("--offload-cp")]
    [InlineData("--qwen-image-lor")]
    [InlineData("--qwen-image-loraa")]
    public void Build_NearMissOfARemovedFlag_IsNeverSuggestedIt(string typo)
    {
        // The typo hint must not steer anyone back to an option that only errors.
        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(new[] { typo }, _baseDir));

        Assert.StartsWith("Unknown option", ex.Message, StringComparison.Ordinal);
        foreach ((string flag, _) in RemovedCliFlags.RemovedFlags)
            Assert.DoesNotContain($"'{flag}'", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ServerUsage_DoesNotListRemovedFlags()
    {
        var documented = new HashSet<string>(ServerUsage.DocumentedFlags(), StringComparer.OrdinalIgnoreCase);
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        string usage = sw.ToString();

        Assert.DoesNotContain("Removed options", usage, StringComparison.Ordinal);
        Assert.NotEmpty(RemovedCliFlags.RemovedFlags);
        foreach ((string flag, _) in RemovedCliFlags.RemovedFlags)
        {
            Assert.DoesNotContain(flag, documented);
            // Match whole flag names: --video-te is a prefix of --video-text-encoder.
            Assert.DoesNotMatch(System.Text.RegularExpressions.Regex.Escape(flag) + @"(?![\w-])", usage);
        }
    }

    [Theory]
    [InlineData("{ \"qwen-image-lora\": \"Qwen-Image-Edit-Lightning-8steps.safetensors\" }", "--qwen-image-lora")]
    [InlineData("{ \"offload-cpu\": true }", "--offload-cpu")]
    public void ConfigFile_RemovedQwenImageKeys_FailWithTheSameMessage(string json, string flag)
    {
        // The shipped qwen-image-edit*.json configs carried both keys, so a copy of one
        // kept from before must stop with the same advice the command line gets.
        string cfg = Path.Combine(_baseDir, "stale-qwen-image.json");
        File.WriteAllText(cfg, json);

        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));

        Assert.EndsWith(RemovedCliFlags.Describe(flag)!, ex.Message, StringComparison.Ordinal);
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

    [Theory]
    [InlineData("--tp=2")]
    [InlineData("--layer-split=2")]
    [InlineData("--tp-node-id=0")]
    public void PlacementLookingStopValuesStayLiteralInBothServerPasses(string literal)
    {
        var args = new[] { "--stop", literal };
        Assert.False(ServerOptionsBuilder.ApplyTensorParallelCliFlags(args));
        Assert.Null(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
        Assert.Null(Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE"));
        var options = ServerOptionsBuilder.Build(args, _baseDir);
        Assert.Contains(literal, options.DefaultSamplingConfig.StopSequences);

        // Literal placement-looking text must not hide a separate real option.
        var withPlacement = args.Concat(new[] { "--layer-split=2" }).ToArray();
        Assert.True(ServerOptionsBuilder.ApplyTensorParallelCliFlags(withPlacement));
        Assert.Equal("2", Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE"));
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

    [Theory]
    [InlineData("--layer-split", "2")]
    [InlineData("--layer-split=2", null)]
    public void LayerSplit_ReachesTheLoaderAndIsAcceptedByTheServer(string flag, string? value)
    {
        string[] args = value == null ? new[] { flag } : new[] { flag, value };
        Assert.True(ServerOptionsBuilder.ApplyTensorParallelCliFlags(args));
        Assert.Equal("2", Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE"));
        Assert.Null(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"));
        Assert.NotNull(ServerOptionsBuilder.Build(args, _baseDir));
        Assert.Contains("--layer-split", TensorSharp.Cli.CliUsage.DocumentedFlags());
        Assert.Contains("--layer-split", ServerUsage.DocumentedFlags());
    }

    // ----- speculative-decoding CLI flags -----

    [Fact]
    public void ApplySpeculativeCliFlags_SpecFlag_EnablesSchedulerSpeculation()
    {
        _env.ClearSpeculationVars();
        bool applied = ServerOptionsBuilder.ApplySpeculativeCliFlags(new[] { "--spec" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void ApplySpeculativeCliFlags_NoSpecFlag_DisablesSpeculation()
    {
        _env.ClearSpeculationVars();
        _env.Set("TS_SPEC", "1");
        bool applied = ServerOptionsBuilder.ApplySpeculativeCliFlags(new[] { "--no-spec" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SPEC"));
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
        // surviving flag publishes TS_SPEC_DRAFT_MODEL, which the host hands to the
        // model factory and to the attach-after-load path alike; the loaders route by
        // what the file says it is, and TryAttachConfiguredDraftHead skips a drafter
        // the factory already attached. Naming the file also IS the request:
        // speculation turns on without a separate --spec.
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
        Assert.Equal("5", Environment.GetEnvironmentVariable("TS_SPEC_DRAFT"));
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_SPEC_DRAFT_MODEL"));
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

        _env.Set("TS_SPEC_PMIN", "0.5");
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

    [Fact]
    public void SpeculationStartupValidation_ModelRefusal_SaysToDropTheFlagNotToSwapTheDraft()
    {
        // Nemotron-H refuses every drafter: suggesting "the draft GGUF that matches
        // this target" would send the operator after a file that cannot exist.
        string reason = "--draft-model 'x-DSpark.gguf' is not attached: " + NemotronModel.SpeculationRefusalReason;
        string msg = SpeculationStartupValidation.GetFatalActivationError(reason, refusedByModel: true);
        Assert.Contains(reason, msg);
        Assert.Contains("refuses it", msg);
        Assert.Contains("drop --draft-model", msg);
        Assert.DoesNotContain("embedding_length_out", msg);
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
    public void Build_UploadDirectoryCanLiveOutsideApplication()
    {
        string application = Path.Combine(_baseDir, "application");
        string uploads = Path.Combine(_baseDir, "runtime-media");
        Directory.CreateDirectory(application);
        _env.Set("TENSORSHARP_UPLOAD_DIR", uploads);

        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), application);

        Assert.Equal(uploads, options.UploadDirectory);
        Assert.True(Directory.Exists(uploads));
        Assert.False(Directory.Exists(Path.Combine(application, "uploads")));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public void Build_BlankUploadDirectoryKeepsApplicationDefault(string? configured)
    {
        _env.Set("TENSORSHARP_UPLOAD_DIR", configured);
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        Assert.Equal(Path.Combine(_baseDir, "uploads"), options.UploadDirectory);
        Assert.True(Directory.Exists(options.UploadDirectory));
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
    // Build's skip list, and twice one was missed: two video-companion spellings,
    // then EVERY --spec* spelling, which made
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
            scope.ClearSpeculationVars();
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
