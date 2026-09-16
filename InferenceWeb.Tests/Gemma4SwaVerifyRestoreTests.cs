// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Which sliding-window slots a Gemma 4 verify puts back once the accept count is
// known. A verify writes every row's K/V at its true position; past the window
// those writes evict positions p+i-W, so SaveSwaSlotsForVerify keeps a copy and
// SpecOnVerifyAccepted restores the rows that must not stay.
//
// The bug these pin: on a trunk whose verify KV is not kept on a partial acceptance
// (dense Gemma 4 without per-layer embeddings - 12B, 31B - on the ggml and CPU
// backends), the restore ran on EVERY verify, including a fully accepted one that
// the executor never rolls back or re-forwards. The committed rows p+1..p+K then
// sat in the ring as the evicted positions p+1-W..p+K-W, and every later token
// attended stale keys: gemma-4-12B's second verify past a 1,024-token window was
// 30-47 logits away from plain decoding, and AgentTurnBench's spec, json and
// newchat streams diverged from plain greedy (token 10 of 192 on the spec prompt).
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class Gemma4SwaVerifyRestoreTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullyAcceptedWindow_RestoresNothing(bool verifyKvKept)
    {
        // 8-row verify past the window (every row but row 0 evicted a real position),
        // all seven drafts accepted: every row is committed and nothing is re-forwarded.
        var (_, count) = Gemma4Model.SwaSlotsToRestore(
            backupFirstRow: 1, backupRows: 8, acceptedRows: 7, verifyRows: 7, verifyKvKept);
        Assert.Equal(0, count);
    }

    [Fact]
    public void PartialAcceptance_WithoutKeptKv_RestoresEverySavedRow()
    {
        // The executor rolls back to the window start and re-forwards the kept prefix,
        // which attends the positions the kept rows' slots evicted as well.
        var (first, count) = Gemma4Model.SwaSlotsToRestore(
            backupFirstRow: 1, backupRows: 8, acceptedRows: 3, verifyRows: 7, verifyKvKept: false);
        Assert.Equal(1, first);
        Assert.Equal(7, count);
    }

    [Fact]
    public void PartialAcceptance_WithKeptKv_RestoresOnlyRejectedRows()
    {
        // Kept rows 0..3 stay as the verify wrote them; rows 4..7 go back.
        var (first, count) = Gemma4Model.SwaSlotsToRestore(
            backupFirstRow: 1, backupRows: 8, acceptedRows: 3, verifyRows: 7, verifyKvKept: true);
        Assert.Equal(4, first);
        Assert.Equal(4, count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NothingAccepted_RestoresFromTheFirstEvictingRow(bool verifyKvKept)
    {
        // The window crosses the ring size inside the batch: rows before FirstRow
        // landed on never-used slots and were not saved.
        var (first, count) = Gemma4Model.SwaSlotsToRestore(
            backupFirstRow: 5, backupRows: 8, acceptedRows: 0, verifyRows: 7, verifyKvKept);
        Assert.Equal(5, first);
        Assert.Equal(3, count);
    }

    [Fact]
    public void AcceptedPastTheSavedRows_WithKeptKv_RestoresNothing()
    {
        // All accepted rows are at or past the last saved row.
        var (_, count) = Gemma4Model.SwaSlotsToRestore(
            backupFirstRow: 6, backupRows: 8, acceptedRows: 7, verifyRows: 7, verifyKvKept: true);
        Assert.Equal(0, count);
    }
}
