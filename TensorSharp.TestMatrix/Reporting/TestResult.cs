// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Text.Json.Serialization;
using TensorSharp.TestMatrix.Matrix;

namespace TensorSharp.TestMatrix.Reporting;

/// <summary>
/// The persisted outcome of one <see cref="TestCase"/>. One file per case is
/// dropped under <c>results/&lt;case-id&gt;.json</c>; the report generator
/// aggregates them.
/// </summary>
public sealed class TestResult
{
    [JsonPropertyName("case_id")]
    public string CaseId { get; set; } = "";

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("model_family")]
    public string ModelFamily { get; set; } = "";

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "";

    [JsonPropertyName("feature")]
    public string Feature { get; set; } = "";

    [JsonPropertyName("env_var")]
    public string? EnvVar { get; set; }

    [JsonPropertyName("env_value")]
    public string? EnvValue { get; set; }

    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("skipped")]
    public bool Skipped { get; set; }

    [JsonPropertyName("skip_reason")]
    public string? SkipReason { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    /// <summary>
    /// Outcome of the per-feature output-content check. <c>null</c> when no
    /// expected substrings were configured for the feature (synthetic
    /// benchmarks, audio, video by default). When non-null, <c>Ok</c> is
    /// gated on <see cref="CorrectnessOk"/>.
    /// </summary>
    [JsonPropertyName("correctness_ok")]
    public bool? CorrectnessOk { get; set; }

    [JsonPropertyName("correctness_error")]
    public string? CorrectnessError { get; set; }

    [JsonPropertyName("correctness_expected")]
    public string[]? CorrectnessExpected { get; set; }

    [JsonPropertyName("exit_code")]
    public int ExitCode { get; set; }

    [JsonPropertyName("command")]
    public string Command { get; set; } = "";

    [JsonPropertyName("model_load_ms")]
    public double ModelLoadMs { get; set; }

    [JsonPropertyName("prefill_tokens")]
    public int PrefillTokens { get; set; }

    [JsonPropertyName("prefill_ms")]
    public double PrefillMs { get; set; }

    [JsonPropertyName("prefill_tps")]
    public double PrefillTps { get; set; }

    [JsonPropertyName("decode_tokens")]
    public int DecodeTokens { get; set; }

    [JsonPropertyName("decode_ms")]
    public double DecodeMs { get; set; }

    [JsonPropertyName("decode_tps")]
    public double DecodeTps { get; set; }

    [JsonPropertyName("total_wall_ms")]
    public double TotalWallMs { get; set; }

    [JsonPropertyName("output_preview")]
    public string OutputPreview { get; set; } = "";

    [JsonPropertyName("stdout_tail")]
    public string StdoutTail { get; set; } = "";

    /// <summary>
    /// What was sent into the inference. Captured at run time from the
    /// prompt / tools / multi-turn files so the JSON is self-contained and
    /// readers don't have to chase file paths to understand the case.
    /// </summary>
    [JsonPropertyName("input")]
    public InputCapture Input { get; set; } = new();

    /// <summary>
    /// What the model produced. Separated from the truncated
    /// <see cref="OutputPreview"/> so users see the full text without
    /// re-running the case.
    /// </summary>
    [JsonPropertyName("output")]
    public OutputCapture Output { get; set; } = new();

    [JsonPropertyName("started_at")]
    public DateTimeOffset StartedAt { get; set; }

    [JsonPropertyName("duration_seconds")]
    public double DurationSeconds { get; set; }
}

public sealed class InputCapture
{
    /// <summary>Path of the prompt file (relative to the inputs root) if any.</summary>
    [JsonPropertyName("prompt_path")]
    public string? PromptPath { get; set; }

    /// <summary>Full text of the prompt file (truncated at 16 KiB).</summary>
    [JsonPropertyName("prompt_text")]
    public string? PromptText { get; set; }

    /// <summary>Path of the image / audio / video file if any.</summary>
    [JsonPropertyName("media_path")]
    public string? MediaPath { get; set; }

    /// <summary>JSON contents of the tools file if --tools was used.</summary>
    [JsonPropertyName("tools_text")]
    public string? ToolsText { get; set; }

    /// <summary>JSONL contents of the multi-turn file if --multi-turn-jsonl was used.</summary>
    [JsonPropertyName("multi_turn_text")]
    public string? MultiTurnText { get; set; }

    /// <summary>Whether --think was passed.</summary>
    [JsonPropertyName("thinking_enabled")]
    public bool ThinkingEnabled { get; set; }

    /// <summary>Max tokens the runner asked for (0 = synthetic benchmark mode).</summary>
    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; }

    /// <summary>Effective environment variables applied for this case.</summary>
    [JsonPropertyName("env")]
    public Dictionary<string, string> Env { get; set; } = new();
}

public sealed class OutputCapture
{
    /// <summary>
    /// Best-effort assistant-generated text, with log lines filtered out.
    /// Truncated at 8 KiB. Empty for synthetic benchmark modes (the CLI
    /// doesn't stream tokens in <c>--benchmark</c>).
    /// </summary>
    [JsonPropertyName("assistant_text")]
    public string AssistantText { get; set; } = "";

    /// <summary>True if the assistant text was truncated.</summary>
    [JsonPropertyName("assistant_text_truncated")]
    public bool AssistantTextTruncated { get; set; }

    /// <summary>Number of decoded tokens reported by the CLI (parsed from logs).</summary>
    [JsonPropertyName("decode_tokens")]
    public int DecodeTokens { get; set; }
}
