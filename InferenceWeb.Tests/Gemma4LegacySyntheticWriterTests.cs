// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>Freezes the legacy Gemma 4 model's logits on the synthetic fixtures (<see cref="Gemma4LegacySynthetic"/>); runs
/// only with TS_GEMMA4_LEGACY_GOLDENS_OUT set to the file to write. It goes with the legacy model.</summary>
public sealed class Gemma4LegacySyntheticWriterTests : IDisposable
{
    private const string OutVariable = "TS_GEMMA4_LEGACY_GOLDENS_OUT";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-gemma4-goldens-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private sealed class WriterFactAttribute : FactAttribute
    {
        public WriterFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(OutVariable)))
                Skip = $"Writes the frozen legacy logits only with {OutVariable}=<file>.";
        }
    }

    [WriterFact]
    public void Gemma4LegacySyntheticWriter()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        Directory.CreateDirectory(_dir);
        Gemma4LegacySynthetic.Write(Environment.GetEnvironmentVariable(OutVariable), _dir, path => new Gemma4Model(path, BackendType.Cpu));
    }
}
