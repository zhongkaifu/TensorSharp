// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

public sealed record MatrixExpansion(
    IReadOnlyList<TestCase> Cases,
    IReadOnlyList<SkippedCombination> Skipped);

public sealed record SkippedCombination(
    ModelSpec Model,
    BackendInfo Backend,
    FeatureSpec Feature,
    string? EnvVar,
    string Reason);

/// <summary>
/// Expands the cartesian product (models × backends × features × env-var sweeps)
/// into <see cref="TestCase"/>s, applying availability and capability filters.
/// </summary>
public sealed class MatrixExpander
{
    private readonly IReadOnlyList<ModelSpec> _models;
    private readonly IReadOnlyList<BackendInfo> _backends;
    private readonly IReadOnlyList<FeatureSpec> _features;
    private readonly IReadOnlyList<EnvVarSpec> _envVars;

    public MatrixExpander(
        IReadOnlyList<ModelSpec> models,
        IReadOnlyList<BackendInfo> backends,
        IReadOnlyList<FeatureSpec> features,
        IReadOnlyList<EnvVarSpec> envVars)
    {
        _models = models;
        _backends = backends;
        _features = features;
        _envVars = envVars;
    }

    public MatrixExpansion Expand()
    {
        var cases = new List<TestCase>();
        var skipped = new List<SkippedCombination>();

        foreach (ModelSpec model in _models)
        {
            foreach (BackendInfo backend in _backends)
            {
                if (!backend.IsAvailableOnHost())
                {
                    skipped.Add(new SkippedCombination(model, backend, FeatureCatalog.SyntheticPrefill512, null,
                        $"backend '{backend.Id}' not available on this host"));
                    continue;
                }

                foreach (FeatureSpec feature in _features)
                {
                    string? reason = WhyFeatureNotApplicable(model, feature);
                    if (reason is not null)
                    {
                        skipped.Add(new SkippedCombination(model, backend, feature, null, reason));
                        continue;
                    }

                    // 1. Baseline case (no env var sweep).
                    cases.Add(new TestCase(
                        Model: model,
                        Backend: backend,
                        Feature: feature,
                        EnvVar: null,
                        EnvValue: null,
                        ExtraEnv: new Dictionary<string, string>()));

                    // 2. Per-env-var sweep cases.
                    foreach (EnvVarSpec env in _envVars)
                    {
                        if (!env.AppliesTo(model, backend, feature))
                        {
                            continue;
                        }
                        foreach (string value in env.Values)
                        {
                            cases.Add(new TestCase(
                                Model: model,
                                Backend: backend,
                                Feature: feature,
                                EnvVar: env.Name,
                                EnvValue: value,
                                ExtraEnv: new Dictionary<string, string> { { env.Name, value } }));
                        }
                    }
                }
            }
        }

        return new MatrixExpansion(cases, skipped);
    }

    private static string? WhyFeatureNotApplicable(ModelSpec model, FeatureSpec feature)
    {
        if (feature.RequiresImage && !model.SupportsImage)
        {
            return $"{model.Id} does not support image input";
        }
        if (feature.RequiresAudio && !model.SupportsAudio)
        {
            return $"{model.Id} does not support audio input";
        }
        if (feature.RequiresVideo && !model.SupportsVideo)
        {
            return $"{model.Id} does not support video input";
        }
        if (feature.RequiresTools && !model.SupportsTools)
        {
            return $"{model.Id} does not support tool calling";
        }
        if (feature.Kind == FeatureKind.Thinking && !model.SupportsThinking)
        {
            return $"{model.Id} does not support thinking mode";
        }
        if (feature.RequiresImage && string.IsNullOrEmpty(model.MmprojPath))
        {
            return $"{model.Id} has no mmproj — image disabled";
        }
        if (feature.RequiresAudio && string.IsNullOrEmpty(model.MmprojPath))
        {
            return $"{model.Id} has no mmproj — audio disabled";
        }
        if (feature.RequiresVideo && string.IsNullOrEmpty(model.MmprojPath))
        {
            return $"{model.Id} has no mmproj — video disabled";
        }
        return null;
    }
}
