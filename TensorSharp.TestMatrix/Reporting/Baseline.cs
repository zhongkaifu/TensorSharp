// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Text.Json.Serialization;

namespace TensorSharp.TestMatrix.Reporting;

/// <summary>
/// Persisted "expected" outcome for one cell of the matrix on one host class
/// (e.g. macos-mlx, linux-cuda). The runner compares the current run against
/// this snapshot to flag throughput / correctness regressions and PR-gates
/// non-accepted ones.
/// </summary>
public sealed class BaselineEntry
{
    [JsonPropertyName("case_id")]
    public string CaseId { get; set; } = "";

    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("correctness_ok")]
    public bool? CorrectnessOk { get; set; }

    [JsonPropertyName("prefill_tps")]
    public double PrefillTps { get; set; }

    [JsonPropertyName("decode_tps")]
    public double DecodeTps { get; set; }

    [JsonPropertyName("model_load_ms")]
    public double ModelLoadMs { get; set; }
}

/// <summary>
/// Top-level baseline file format: a host label plus the case-by-case
/// snapshot. One file per host (e.g. <c>baseline-macos-mlx.json</c>) so
/// runners can pull only the snapshot that matches their hardware.
/// </summary>
public sealed class BaselineFile
{
    [JsonPropertyName("host_label")]
    public string HostLabel { get; set; } = "";

    [JsonPropertyName("captured_at")]
    public DateTimeOffset CapturedAt { get; set; }

    [JsonPropertyName("commit")]
    public string? Commit { get; set; }

    [JsonPropertyName("notes")]
    public string? Notes { get; set; }

    [JsonPropertyName("entries")]
    public List<BaselineEntry> Entries { get; set; } = new();
}
