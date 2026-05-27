// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Text.Json.Serialization;

namespace TensorSharp.TestMatrix.Configuration;

/// <summary>
/// JSON-serializable shape of <c>matrix-config.json</c>. Loaded by
/// <see cref="MatrixConfigLoader"/> and merged with command-line overrides.
/// </summary>
public sealed class MatrixConfig
{
    [JsonPropertyName("model_dir")]
    public string ModelDir { get; set; } = "/Users/ZhongkaiFu/work/model";

    [JsonPropertyName("cli_executable")]
    public string CliExecutable { get; set; } = "";

    [JsonPropertyName("media_dir")]
    public string? MediaDir { get; set; }

    [JsonPropertyName("results_dir")]
    public string ResultsDir { get; set; } = "results";

    [JsonPropertyName("report_path")]
    public string ReportPath { get; set; } = "report.md";

    [JsonPropertyName("default_backends")]
    public List<string> DefaultBackends { get; set; } = new();

    [JsonPropertyName("default_features")]
    public List<string> DefaultFeatures { get; set; } = new();

    [JsonPropertyName("default_env_vars")]
    public List<string> DefaultEnvVars { get; set; } = new();

    [JsonPropertyName("models")]
    public List<ModelConfig> Models { get; set; } = new();

    [JsonPropertyName("timeout_seconds")]
    public int TimeoutSeconds { get; set; } = 900;

    [JsonPropertyName("warmup_runs")]
    public int WarmupRuns { get; set; } = 1;

    [JsonPropertyName("benchmark_runs")]
    public int BenchmarkRuns { get; set; } = 3;

    [JsonPropertyName("skip_existing")]
    public bool SkipExisting { get; set; }

    [JsonPropertyName("max_parallel")]
    public int MaxParallel { get; set; } = 1;

    [JsonPropertyName("baselines_dir")]
    public string BaselinesDir { get; set; } = "TensorSharp.TestMatrix/Baselines";

    [JsonPropertyName("regression_threshold_pct")]
    public double RegressionThresholdPct { get; set; } = 10.0;
}

/// <summary>
/// Per-model overrides. Auto-discovered models are merged with these; an entry
/// here can pin a non-default mmproj, override capability flags, or rename
/// the display id.
/// </summary>
public sealed class ModelConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("family")]
    public string Family { get; set; } = "";

    [JsonPropertyName("display_name")]
    public string DisplayName { get; set; } = "";

    [JsonPropertyName("gguf")]
    public string Gguf { get; set; } = "";

    [JsonPropertyName("mmproj")]
    public string? Mmproj { get; set; }

    [JsonPropertyName("supports_image")]
    public bool SupportsImage { get; set; }

    [JsonPropertyName("supports_audio")]
    public bool SupportsAudio { get; set; }

    [JsonPropertyName("supports_video")]
    public bool SupportsVideo { get; set; }

    [JsonPropertyName("supports_tools")]
    public bool SupportsTools { get; set; }

    [JsonPropertyName("supports_thinking")]
    public bool SupportsThinking { get; set; }

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; } = true;
}
