// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One model under test. Built either from explicit JSON config or by
/// scanning a model directory for GGUFs.
/// </summary>
public sealed record ModelSpec(
    string Id,
    string Family,
    string DisplayName,
    string GgufPath,
    string? MmprojPath,
    bool SupportsImage,
    bool SupportsAudio,
    bool SupportsVideo,
    bool SupportsTools,
    bool SupportsThinking)
{
    public string FileName => Path.GetFileName(GgufPath);
}
