// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache.Fakes;

/// <summary>The seven oracle fakes of DESIGN §12.3, one per family behaviour the tree must respect.</summary>
internal static class OracleFakes
{
    /// <summary>Qwen3 / GptOss after M7a: holders with capture, clone and donate; any rewind;
    /// A1 pages on the primary and A2 pages on the batched route; A2 → holder copies.</summary>
    internal static OracleModel P(int blockSize = 16) => new(new OracleTraits
    {
        Name = "P",
        Class = FamilyClass.P,
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true,
        Truncation = TruncationKind.Any,
        Pages = PageSupport.Both,
        SupportsCopyPagedToHolder = true,
    }, blockSize);

    /// <summary>Mistral3: no holders; A1 and A2 pages. An A1 slab the engine never extracted is
    /// zeros, so injecting an A2-only page as A1 (D6) changes the output.</summary>
    internal static OracleModel P2(int blockSize = 16) => new(new OracleTraits
    {
        Name = "P2",
        Class = FamilyClass.P,
        EndState = EndStateSupport.None,
        Truncation = TruncationKind.Any,
        Pages = PageSupport.Both,
    }, blockSize);

    /// <summary>The window of <see cref="S"/>.</summary>
    internal const int GemmaWindow = 64;

    /// <summary>Gemma 4: holders with capture, clone and donate; rewind only within an unwrapped
    /// window of 64, and a rewind of a wrapped ring "succeeds" with a wrong state; forwards leave
    /// holders device-authoritative, so a clone of an unsettled donated holder is refused.</summary>
    internal static OracleModel S(int blockSize = 16) => new(new OracleTraits
    {
        Name = "S",
        Class = FamilyClass.S,
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true,
        Persistable = true,
        DeviceDirtyOnForward = true,
        Truncation = TruncationKind.WithinUnwrappedWindow,
        TruncationParameter = GemmaWindow,
        ForbiddenRewindCorrupts = true,
        Pages = PageSupport.None,
    }, blockSize);

    /// <summary>Ring rows and window of <see cref="S2"/>: the ring slack is rows − W − 1.</summary>
    internal const int MuseRingRows = 80, MuseWindow = 64;

    /// <summary>MuseGlimmer: no holders; A1 pages within the ring rows; the primary rewinds only
    /// within the ring slack (the M0c guard refuses beyond it); no reuse across media.</summary>
    internal static OracleModel S2(int blockSize = 16) => new(new OracleTraits
    {
        Name = "S2",
        Class = FamilyClass.S,
        EndState = EndStateSupport.None,
        Truncation = TruncationKind.WithinRingSlack,
        TruncationParameter = MuseRingRows - MuseWindow - 1,
        Pages = PageSupport.A1HostSlab,
        PageWindowTokens = MuseRingRows,
        ReuseAcrossMediaSpan = false,
    }, blockSize);

    /// <summary>Qwen 3.5 / Qwen4Exp: holders with capture, clone and donate; no truncation
    /// (exact-length materialization only); device-authoritative forwards; persistable.</summary>
    internal static OracleModel R(int blockSize = 16) => new(new OracleTraits
    {
        Name = "R",
        Class = FamilyClass.R,
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true,
        Persistable = true,
        DeviceDirtyOnForward = true,
        Truncation = TruncationKind.None,
        Pages = PageSupport.None,
        ReuseAcrossMediaSpan = false,
    }, blockSize);

    /// <summary>Nemotron-H: no holders; A1 pages whose recurrent state is restorable only at a
    /// forward boundary (an inject of any other page end yields a wrong state); no truncation.</summary>
    internal static OracleModel R2(int blockSize = 16) => new(new OracleTraits
    {
        Name = "R2",
        Class = FamilyClass.R,
        EndState = EndStateSupport.None,
        Truncation = TruncationKind.None,
        Pages = PageSupport.A1HostSlab,
        PagesNeedStateAtEnd = true,
        ReuseAcrossMediaSpan = false,
    }, blockSize);

    /// <summary>The ModelDecides span of <see cref="N"/>.</summary>
    internal const int NativeRewindSpan = 8;

    /// <summary>DeepSeek V4.1 / GLM: donate-only native slots, three in total (the primary's
    /// included); the model decides rewinds (within 8 tokens, aligned to 2); a bind that needs a
    /// slot reclaims the oldest retained one and reports it through the sink.</summary>
    internal static OracleModel N(int blockSize = 16) => new(new OracleTraits
    {
        Name = "N",
        Class = FamilyClass.N,
        EndState = EndStateSupport.DonateOnly,
        NativeSlotLimit = 3,
        AdoptPrimaryOnDisplacement = true,
        Truncation = TruncationKind.ModelDecides,
        TruncationParameter = NativeRewindSpan,
        TruncationGranularity = 2,
        RewindCapTokens = int.MaxValue,
        Pages = PageSupport.None,
        ReuseAcrossMediaSpan = false,
    }, blockSize);

    internal static IReadOnlyList<(string Name, Func<OracleModel> Create)> All { get; } = new (string, Func<OracleModel>)[]
    {
        ("P", () => P()), ("P2", () => P2()), ("S", () => S()), ("S2", () => S2()),
        ("R", () => R()), ("R2", () => R2()), ("N", () => N()),
    };
}

/// <summary>A sink that records what a model reported (DEC-23).</summary>
internal sealed class RecordingPayloadSink : IPrefixPayloadSink
{
    internal List<(string Key, InvalidationReason Reason)> Reports { get; } = new();

    public void OnPayloadInvalidated(string payloadKey, InvalidationReason reason) => Reports.Add((payloadKey, reason));
}
