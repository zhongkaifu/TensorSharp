// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>Pure restore-preflight contracts. Fake addresses are never dereferenced and no CUDA/device test is implied.</summary>
public sealed class NinferDraftHostTierTests
{
    private static readonly object Owner = new();
    private const string Namespace = "target-numerics|original-draft-module-weight-policy", Profile = "host-slot-v2:original-gdn48:bf16-ring2048", Draft = "original-draft-module-weight-policy";
    private static Q35CudaEngine.HostSlot Host(bool native = true, bool pending = false)
    {
        byte[] bytes = pending ? new byte[51200] : Array.Empty<byte>();
        if (pending) { BinaryPrimitives.WriteUInt16LittleEndian(bytes, 0x8000); BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2), 0x3f81); }
        return new() { Owner = Owner, StateNamespace = Namespace, StateProfile = Profile, NPast = 128, RopeDelta = 0,
            StateUnits = new[] { new IntPtr(4096), new IntPtr(8192) }, StatePartCount = 107,
            Blocks = new[] { new Q35CudaEngine.HostBlock { Data = new IntPtr(12288), Key = 1, Refs = 1 } },
            NinferDraft = native ? new Q35CudaEngine.NinferDraftSlotSnapshot(Draft, 128, pending ? 127 : 128, 0, bytes) : null,
            DraftMetadataBytes = native ? 64 + bytes.Length : 0 };
    }
    private static void Validate(Q35CudaEngine.HostSlot host, bool native = true, object? owner = null, string? identity = null,
        string? profile = null, int maximumContext = 8192, int units = 2, int parts = 107, string? draft = null)
        => Q35CudaEngine.ValidateHostSlotRestoreMetadata(host, owner ?? Owner, identity ?? Namespace, profile ?? Profile,
            maximumContext, units, parts, native, draft ?? Draft);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteNativeHostCapsulePreservesFullAndPendingCheckpointFrontiers(bool pending)
    {
        var host = Host(pending: pending); Validate(host);
        Assert.Equal(pending ? 51264 : 64, host.DraftMetadataBytes);
        Assert.Equal(pending ? 127 : 128, host.NinferDraft.Frontier);
        Assert.Equal(pending ? 51200 : 0, host.NinferDraft.Pending.Length);
    }

    [Fact]
    public void ForeignEngineOrNumericalNamespaceIsRejectedByPurePreflight()
    {
        var host = Host();
        Assert.Throws<InvalidOperationException>(() => Validate(host, owner: new object()));
        Assert.Throws<InvalidOperationException>(() => Validate(host, identity: "other-target-or-draft-policy"));
        Assert.Throws<InvalidOperationException>(() => Validate(host, profile: "wrong-bf16-ring-geometry"));
        Assert.Throws<InvalidOperationException>(() => Validate(host, draft: "other-draft-module"));
        Assert.Equal(128, host.NPast); Assert.Equal(1, host.Blocks[0].Refs); Assert.Equal(2, host.StateUnits.Length);
    }

    [Fact]
    public void NativeDraftStateCannotBeSilentlyDroppedOrInventedOnRestore()
    {
        var missing = Host(native: false);
        Assert.Throws<ArgumentNullException>(() => Validate(missing));
        Assert.Throws<InvalidOperationException>(() => Validate(Host(), native: false));
        Validate(missing, native: false); // The existing target/GGUF host path needs no native capsule.
    }

    [Fact]
    public void PositionMustMatchOwnedDraftCapsuleAndPhysicalAttentionBlockCount()
    {
        var negative = Host(); negative.NPast = -1;
        Assert.Throws<InvalidOperationException>(() => Validate(negative));
        var beyond = Host(); beyond.NPast = 8193;
        Assert.Throws<InvalidOperationException>(() => Validate(beyond));
        var wrongFrontier = Host(pending: true); wrongFrontier.NPast = 127;
        Assert.Throws<InvalidOperationException>(() => Validate(wrongFrontier));
        var wrongBlocks = Host(); wrongBlocks.NPast = 129;
        Assert.Throws<InvalidOperationException>(() => Validate(wrongBlocks));
        var delta = Host(); delta.RopeDelta = 1;
        Assert.Throws<InvalidOperationException>(() => Validate(delta));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public void WrongUnitCountCannotReachDestinationMutation(int units)
        => Assert.Throws<InvalidOperationException>(() => Validate(Host(), units: units));

    [Fact]
    public void ReleasedOrIncompleteHostStorageIsRejected()
    {
        var released = Host(); released.StateUnits = null;
        Assert.Throws<InvalidOperationException>(() => Validate(released));
        var missingUnit = Host(); missingUnit.StateUnits[1] = IntPtr.Zero;
        Assert.Throws<InvalidOperationException>(() => Validate(missingUnit));
        var wrongParts = Host(); wrongParts.StatePartCount--;
        Assert.Throws<InvalidOperationException>(() => Validate(wrongParts));
        var missingBlock = Host(); missingBlock.Blocks = Array.Empty<Q35CudaEngine.HostBlock>();
        Assert.Throws<InvalidOperationException>(() => Validate(missingBlock));
        var missingData = Host(); missingData.Blocks[0].Data = IntPtr.Zero;
        Assert.Throws<InvalidOperationException>(() => Validate(missingData));
        var unowned = Host(); unowned.Blocks[0].Refs = 0;
        Assert.Throws<InvalidOperationException>(() => Validate(unowned));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(51200)]
    [InlineData(51265)]
    public void PendingPayloadByteOwnershipMustMatchExactly(long charge)
    {
        var host = Host(pending: true); host.DraftMetadataBytes = charge;
        Assert.Throws<InvalidOperationException>(() => Validate(host));
    }

    [Fact]
    public void PreflightDoesNotRewritePendingBf16Bytes()
    {
        var host = Host(pending: true); byte[] before = host.NinferDraft.Pending.ToArray(); Validate(host);
        Assert.Equal(before, host.NinferDraft.Pending.ToArray());
        Assert.Equal(0x8000, BinaryPrimitives.ReadUInt16LittleEndian(host.NinferDraft.Pending.Span));
        Assert.Equal(0x3f81, BinaryPrimitives.ReadUInt16LittleEndian(host.NinferDraft.Pending.Span[2..]));
    }
}
