// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using TensorSharp.Memory;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

/// <summary>An architecture-specific owner for admitted model execution.</summary>
public interface IUnifiedMemoryModelSession : IDisposable
{
    ModelBase? Model { get; }
    MemoryBudget Budget { get; }
    int ContextTokens { get; }
    int PrefillChunkTokens { get; }
    string FeatureReport { get; }
    SchedulerConfig ConfigureScheduler(SchedulerConfig requested);
}
