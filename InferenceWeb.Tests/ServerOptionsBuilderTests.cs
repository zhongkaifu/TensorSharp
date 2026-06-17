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
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;

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

    // ----- MTP speculative-decoding CLI flags -----

    [Fact]
    public void ApplyMtpSpeculativeCliFlags_SpecFlag_EnablesSchedulerSpeculation()
    {
        _env.Set("TS_MTP_SPEC", null);
        bool applied = ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(new[] { "--mtp-spec" });
        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);
    }

    [Fact]
    public void ApplyMtpSpeculativeCliFlags_NoSpecFlag_DisablesSpeculation()
    {
        _env.Set("TS_MTP_SPEC", "1");
        bool applied = ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(new[] { "--no-mtp-spec" });
        Assert.True(applied);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.False(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);
    }

    [Fact]
    public void ApplyMtpSpeculativeCliFlags_DraftModel_DoesNotCollideWithDraftCount()
    {
        // --mtp-draft is a prefix of --mtp-draft-model; the parser must route each
        // to its own env var rather than mis-reading the longer flag as the shorter.
        _env.Set("TS_MTP_DRAFT", null);
        _env.Set("TS_MTP_DRAFT_MODEL", null);
        string draftFile = Path.Combine(_baseDir, "draft.gguf");
        File.WriteAllText(draftFile, "stub");   // the parser validates File.Exists

        bool applied = ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(new[]
        {
            "--mtp-draft", "5",
            "--mtp-draft-model", draftFile,
        });

        Assert.True(applied);
        Assert.Equal("5", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
        Assert.Equal(draftFile, Environment.GetEnvironmentVariable("TS_MTP_DRAFT_MODEL"));
        Assert.Equal(5, SchedulerConfig.FromEnvironment().MtpMaxDraftTokens);
    }

    [Fact]
    public void ApplyMtpSpeculativeCliFlags_MissingDraftModelFile_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(
                new[] { "--mtp-draft-model", Path.Combine(_baseDir, "does-not-exist.gguf") }));
        Assert.Contains("--mtp-draft-model", ex.Message);
    }

    [Fact]
    public void MtpStartupValidation_NoActivationError_ReturnsNull()
    {
        Assert.Null(MtpStartupValidation.GetFatalActivationError(null));
        Assert.Null(MtpStartupValidation.GetFatalActivationError(string.Empty));
    }

    [Fact]
    public void MtpStartupValidation_ActivationError_ReturnsFatalMessageWithReasonAndHint()
    {
        // Repro for the user-reported bug: pairing the 12B target with the 26B-A4B
        // draft fails the backbone-dim check; that reason used to be a warning the
        // operator never saw, so the server ran with speculation silently off.
        // Startup must now fail fast, surfacing the reason plus a remediation hint.
        const string reason = "MTP draft backbone dim 2816 != target hidden size 3840.";
        string msg = MtpStartupValidation.GetFatalActivationError(reason);
        Assert.NotNull(msg);
        Assert.Contains(reason, msg);
        Assert.Contains("--mtp-draft-model", msg);
        Assert.Contains("embedding_length_out", msg);
    }

    /// <summary>
    /// Disposable helper that snapshots and restores environment variables
    /// touched during a test. Without this, the env vars set by one test could
    /// leak into another test that runs in the same process.
    /// </summary>
    private sealed class EnvScope : IDisposable
    {
        private readonly Dictionary<string, string?> _originals = new();

        public void Set(string name, string value)
        {
            if (!_originals.ContainsKey(name))
                _originals[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
        }

        public void Dispose()
        {
            foreach (var kv in _originals)
                Environment.SetEnvironmentVariable(kv.Key, kv.Value);
            _originals.Clear();
        }
    }
}
