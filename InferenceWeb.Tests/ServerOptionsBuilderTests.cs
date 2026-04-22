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
