// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Runtime.Scheduling.PrefixCache;

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
    public required string NamespaceFingerprint { get; init; }       // non-empty (K1a)

    // End states
    public EndStateSupport EndState { get; init; }
    public bool CanCaptureCopy { get; init; }                        // deep copy of the ACTIVE cache (P, breakpoints, branches)
    public bool AdoptPrimaryOnDisplacement { get; init; }            // move primary K/V into an end state; may allocate a fresh primary
    public bool PrimaryResident { get; init; }                       // the primary may be registered at finish
    /// <summary>Keep a finished primary in place for an exact next turn. Convert it to an owned
    /// end state only when another request displaces it, so a continuation needs no replacement cache.</summary>
    public bool DeferPrimaryConversion { get; init; }
    public int MinRetainTokens { get; init; } = 32;                  // DEC-18
    /// <summary>End states of the batched route: a finished batched sequence's pool blocks stay with the
    /// prefix cache and the model keeps only its own per-sequence state (Nemotron-H's Mamba2 slot); donated
    /// at exact length to the same conversation's next turn, which continues on the batched route.</summary>
    public bool PagedEndStates { get; init; }

    // Truncation (A3c)
    public TruncationKind Truncation { get; init; }
    public int TruncationParameter { get; init; }                    // W (WithinUnwrappedWindow) or slack (WithinRingSlack)
    public int TruncationGranularity { get; init; } = 1;             // IModelArchitecture.KVCacheTruncationGranularity
    public int RewindCapTokens { get; init; } = 16;                  // DEC-19; int.MaxValue once lifted
    public int MinTailPrefillTokens { get; init; } = 1;              // prompt tokens a reusing request still forwards; DSV4.1 2 (checkpoint)

    // Pages
    public PageSupport Pages { get; init; }
    public bool PagesNeedStateAtEnd { get; init; }                   // Nemotron
    public int PageWindowTokens { get; init; }                       // 0 = unbounded; MuseGlimmer ringRows
    public bool SupportsCopyPagedToHolder { get; init; }             // class P, M7a

    // Media and positions
    public bool ReuseAcrossMediaSpan { get; init; }                  // = IModelArchitecture.SupportsReuseAcrossMediaSpan (Phase 0)
    // A prompt with media also gets the text-style automatic checkpoint at its tail, besides the one
    // before its newest media span: for a family whose checkpoint shares the media's K/V instead of
    // copying it, so the same image asked again (or a history that keeps it) resumes past it.
    public bool CheckpointPromptTailWithMedia { get; init; }

    // End states and checkpoints cost little and their footprints report their bytes: the count
    // caps (retained end states, public checkpoints) give way to the byte budgets unless the
    // operator sets them, since a count tuned for whole-cache copies evicts states that still fit.
    public bool CountCapsByBytes { get; init; }

    // A clone of an end state or checkpoint shares the payload's K/V in the model (copy-on-write),
    // so requests that clone the same payload share its blocks in the scheduler's pool too.
    public bool ClonesShareKv { get; init; }
    public int MmReuseMinTokens { get; init; }                       // DEC-12

    // Persistence and threading
    public bool Persistable { get; init; }
    public bool OffThreadExport { get; init; }                       // M7c, after §12.6 proofs
    public bool OffThreadImportRead { get; init; }                   // M7c, after §12.6 proofs

    // Budgets
    public ResourceVector SubCapBytes { get; init; }                 // TS_DSV41_RETAINED_CACHE_MB, GLM slots
    public int MaxRetainedNativeSlots { get; init; }                 // 0 = byte cap only (every family today)
}
