// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One row of the matrix: an inference run to execute. <see cref="EnvVar"/> is
/// the env var being toggled and <see cref="EnvValue"/> is the value it's
/// pinned to for this run. <see cref="EnvVar"/> is <c>null</c> when the row is
/// a baseline (no sweep) for the (model, backend, feature) cell.
/// </summary>
public sealed record TestCase(
    ModelSpec Model,
    BackendInfo Backend,
    FeatureSpec Feature,
    string? EnvVar,
    string? EnvValue,
    IReadOnlyDictionary<string, string> ExtraEnv)
{
    public string Id => string.Join("__", new[]
    {
        Model.Id,
        Backend.Id,
        Feature.Id,
        EnvVar is null ? "baseline" : $"{EnvVar}={EnvValue}",
    });

    public string ShortLabel => EnvVar is null
        ? "baseline"
        : $"{EnvVar}={EnvValue}";
}
