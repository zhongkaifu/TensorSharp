// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Which owner holds an engine's reuse state (DEC-02). Never changes during an engine's life.</summary>
public enum PrefixCacheMode : byte { Legacy = 0, Shadow = 1, Tree = 2 }

public enum FamilyClass : byte { P, S, R, N }

public enum EndStateSupport : byte { None, DonateOnly, CopyAndDonate }

public enum PageSupport : byte { None, A1HostSlab, A2ModelPaged, Both }

public enum TruncationKind : byte { None, Any, WithinUnwrappedWindow, WithinRingSlack, ModelDecides }

/// <summary>
/// The declarative per-family prefix-cache contract (G-19, DESIGN §4.7). Models publish it;
/// <see cref="ResumabilityRules"/> evaluates it. Models never implement rules.
/// </summary>
public sealed record PrefixCacheCapabilities
{
    public required FamilyClass Class { get; init; }                 // diagnostics only
    public required PrefixCacheMode Readiness { get; init; }         // DEC-02; raised by per-family PRs
    public required string NamespaceFingerprint { get; init; }       // non-empty (K1a)

    // End states
    public EndStateSupport EndState { get; init; }
    public bool CanCaptureCopy { get; init; }                        // deep copy of the ACTIVE cache (P, breakpoints, branches)
    public bool AdoptPrimaryOnDisplacement { get; init; }            // PrimaryResident → EndState at zero copy
    public bool PrimaryResident { get; init; }                       // the primary may be registered at finish
    public int MinRetainTokens { get; init; } = 32;                  // DEC-18

    // Truncation (A3c)
    public TruncationKind Truncation { get; init; }
    public int TruncationParameter { get; init; }                    // W (WithinUnwrappedWindow) or slack (WithinRingSlack)
    public int TruncationGranularity { get; init; } = 1;             // IModelArchitecture.KVCacheTruncationGranularity
    public int RewindCapTokens { get; init; } = 16;                  // DEC-19; int.MaxValue once lifted

    // Pages
    public PageSupport Pages { get; init; }
    public bool PagesNeedStateAtEnd { get; init; }                   // Nemotron
    public int PageWindowTokens { get; init; }                       // 0 = unbounded; MuseGlimmer ringRows
    public bool SupportsCopyPagedToHolder { get; init; }             // class P, M7a

    // Media and positions
    public bool ReuseAcrossMediaSpan { get; init; }                  // = IModelArchitecture.SupportsReuseAcrossMediaSpan (Phase 0)
    public int MmReuseMinTokens { get; init; }                       // DEC-12

    // Persistence and threading
    public bool Persistable { get; init; }
    public bool OffThreadExport { get; init; }                       // M7c, after §12.6 proofs
    public bool OffThreadImportRead { get; init; }                   // M7c, after §12.6 proofs

    // Budgets
    public ResourceVector SubCapBytes { get; init; }                 // TS_Q4E_RETAINED_CACHE_MB, TS_DSV41_RETAINED_CACHE_MB, GLM slots
    public int MaxRetainedNativeSlots { get; init; }                 // GLM default 1 (DEC-39); 0 = byte cap only
}
