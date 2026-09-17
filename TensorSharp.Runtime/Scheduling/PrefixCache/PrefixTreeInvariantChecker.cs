// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>One violated invariant: its id (I1-I30) and a description.</summary>
public readonly record struct InvariantViolation(string Id, string Message)
{
    public override string ToString() => $"{Id}: {Message}";
}

/// <summary>What the caller knows about the moment of the check.</summary>
internal readonly record struct InvariantCheckContext(
    bool TransactionOpen = false,      // I18: DonationPending is legal only inside an admission transaction
    bool AfterEvictionTrigger = false, // I21: caps must hold
    bool Quiescent = false);           // I22 (tree part): no request runs or waits

/// <summary>
/// The tree-level invariants of DESIGN §4.10 (I1-I16, I18-I21, I24, I25, I28, I29 and the tree
/// parts of I10, I14 and I22). Engine-level ones (I17, I23, I26, I27, I30 and the model sides of
/// I10, I14, I22) are added in M4. Compiled in every configuration; cost O(nodes + pages + receipts
/// + key elements). Keeps the last (version, state hash) so I19 can detect an unversioned mutation.
/// </summary>
internal sealed class PrefixTreeInvariantChecker
{
    private long _lastVersion = -1;
    private ulong _lastHash;

    /// <summary>Runs every tree-level check once, without I19 history.</summary>
    internal static IReadOnlyList<InvariantViolation> CheckOnce(PrefixTree tree, InvariantCheckContext context = default)
        => new PrefixTreeInvariantChecker().Check(tree, context);

    internal IReadOnlyList<InvariantViolation> Check(PrefixTree tree, InvariantCheckContext context = default)
    {
        var v = new List<InvariantViolation>();
        void Fail(string id, string message) => v.Add(new InvariantViolation(id, message));

        RadixNode root = tree.Root;
        if (root.Parent is not null || root.Depth != 0 || !root.InTree) Fail("I1", "root must have no parent, depth 0 and be in the tree");
        if (root.EndState is not null || root.PageCount > 0) Fail("I1", "root carries no payload");

        var nodes = new List<RadixNode>();
        var seen = new HashSet<RadixNode>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<RadixNode>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            RadixNode n = stack.Pop();
            var keys = new HashSet<(int, long)>();
            int enumerated = 0;
            foreach (RadixNode c in n.Children)
            {
                enumerated++;
                if (!seen.Add(c)) { Fail("I1", $"{c} reachable twice"); continue; }
                if (c.Edge.Rope is not null && c.Edge.Length > 0 && !keys.Add((c.ScopeIx, c.Edge[0])))
                    Fail("I2", $"two children of {n} share (scope {c.ScopeIx}, first {c.Edge[0]})");
                if (!ReferenceEquals(c.Parent, n)) Fail("I1", $"{c} parent pointer does not match");
                nodes.Add(c);
                stack.Push(c);
            }
            if (enumerated != n.Children.Count) Fail("I2", $"{n} child count {n.Children.Count} != enumerated {enumerated}");
        }
        if (nodes.Count != tree.NodeCount) Fail("I1", $"NodeCount {tree.NodeCount} != reachable {nodes.Count}");

        // Receipts (I6, I24)
        Dictionary<RadixNode, int>? expectedPath = null, expectedState = null;
        if (tree.Ledger is not null)
        {
            expectedPath = new Dictionary<RadixNode, int>(ReferenceEqualityComparer.Instance);
            expectedState = new Dictionary<RadixNode, int>(ReferenceEqualityComparer.Instance);
            foreach (LockReceipt r in tree.Ledger.Open)
            {
                if (r.TreeSerial != tree.TreeSerial) Fail("I24", $"open {r} belongs to tree serial {r.TreeSerial}, tree is {tree.TreeSerial}");
                if (r.PathAnchor is not null)
                {
                    if (!r.PathAnchor.InTree || r.PathAnchor.Generation != r.PathGeneration) { Fail("I6", $"open {r} anchors a recycled node"); }
                    else
                        for (RadixNode n = r.PathAnchor; !n.IsRoot; n = n.Parent!)
                            expectedPath[n] = expectedPath.GetValueOrDefault(n) + 1;
                }
                if (r.StateAnchor is not null)
                {
                    if (!r.StateAnchor.InTree || r.StateAnchor.Generation != r.StateGeneration) Fail("I6", $"open {r} state-locks a recycled node");
                    else expectedState[r.StateAnchor] = expectedState.GetValueOrDefault(r.StateAnchor) + 1;
                }
            }
        }

        ResourceVector cached = default, prot = default;
        var keyOwners = new Dictionary<string, RadixNode>(StringComparer.Ordinal);
        var blockOwners = new Dictionary<KvBlock, RadixNode>(ReferenceEqualityComparer.Instance);
        int primaryResidents = 0, publicTop = 0, scopedEndStates = 0, publicEndStates = 0, nativeSlots = 0;
        int blockSize = tree.Options.BlockSize;
        IPrefixTreePageHost host = tree.PageHost;
        ulong stateHash = 0;

        foreach (RadixNode n in nodes)
        {
            RadixNode parent = n.Parent!;
            bool scopeLive = tree.Scopes.IsLive(n.ScopeIx);
            // I1
            if (!n.InTree) Fail("I1", $"{n} is reachable but not marked in the tree");
            if (n.Edge.Rope is null || n.Edge.Length < 1) { Fail("I1", $"{n} has an empty edge"); continue; }
            if (n.Edge.Rope.Disposed || n.Edge.Start < 0 || n.Edge.Start + n.Edge.Length > n.Edge.Rope.Length)
            { Fail("I1", $"{n} edge lies outside its rope"); continue; }
            if (!ReferenceEquals(parent.Children.Find(n.ScopeIx, n.Edge[0]), n)) Fail("I1", $"{n} is not found under its (scope, first element) key");
            if (n.Depth != parent.Depth + n.Edge.Length) Fail("I1", $"{n} depth != parent depth + edge length");
            // I3, I4
            if (parent.ScopeIx != 0 && n.ScopeIx != parent.ScopeIx) Fail("I3", $"{n} scope {n.ScopeIx} under scoped parent {parent.ScopeIx}");
            if (n.IsPublicBoundary && n.ScopeIx != 0) Fail("I4", $"{n} is a public boundary in scope {n.ScopeIx}");
            // I5
            if (n.Children.Count == 0 && !n.AnyLock && !n.HasPayload) Fail("I5", $"{n} is an unlocked payload-less leaf");
            // I6
            if (expectedPath is not null)
            {
                if (n.LockRef != expectedPath.GetValueOrDefault(n)) Fail("I6", $"{n} LockRef {n.LockRef} != open receipts {expectedPath.GetValueOrDefault(n)}");
                if (n.StateLockRef != expectedState!.GetValueOrDefault(n)) Fail("I6", $"{n} StateLockRef {n.StateLockRef} != open receipts {expectedState!.GetValueOrDefault(n)}");
            }
            if (n.LockRef < 0 || n.StateLockRef < 0 || n.PinRef < 0) Fail("I6", $"{n} has a negative lock count");
            // I7
            if (!parent.IsRoot && n.LockRef > parent.LockRef) Fail("I7", $"{n} LockRef {n.LockRef} > parent {parent.LockRef}");
            // I8
            ResourceVector own = default, pageBytes = default;
            foreach (PageRef page in n.PageSpan)
            {
                var pb = new ResourceVector { PoolPages = 1 };
                if (page.HasA1) pb.HostKv = tree.Options.PageHostBytes;
                pageBytes += pb;
            }
            own = pageBytes;
            if (n.EndState is not null) own += n.EndState.Bytes;
            if (own != n.Bytes) Fail("I8", $"{n} Bytes {n.Bytes} != payload bytes {own}");
            cached += n.Bytes;
            if (n.LockRef > 0) prot += pageBytes;
            if (n.EndState is not null && (n.StateLockRef > 0 || n.PinRef > 0 || n.IsDonationPending)) prot += n.EndState.Bytes;
            // I9
            byte want = scopeLive ? tree.DesiredList(n) : n.LruList;
            if (n.LruList != want) Fail("I9", $"{n} is in list {n.LruList}, predicates say {want}");
            // I10 (tree part)
            if (n.EndState is { } es)
            {
                if (es.Key is null || !es.Key.StartsWith("pc:", StringComparison.Ordinal)) Fail("I10", $"{n} key '{es.Key}' is not tree-minted");
                else if (!keyOwners.TryAdd(es.Key, n)) Fail("I10", $"key {es.Key} appears on two nodes");
                if (!tree.TryGetNodeByKey(es.Key ?? string.Empty, out RadixNode indexed) || !ReferenceEquals(indexed, n))
                    Fail("I10", $"key index does not map {es.Key} to {n}");
                if (es.Kind == EndStateKind.PrimaryResident)
                {
                    primaryResidents++;
                    if (n.StateLockRef > 1) Fail("I14", $"PrimaryResident {n} StateLockRef {n.StateLockRef} > 1");
                    if (!es.Footprint.Bytes.IsZero) Fail("I14", $"PrimaryResident {n} charges bytes");
                }
                else if (n.ScopeIx == 0) publicEndStates++;
                else scopedEndStates++;
                if (es.Kind == EndStateKind.NativeSlot) nativeSlots++;
                if (es.Persisted && (!n.IsPublicBoundary || PathHasMedia(n))) Fail("I20", $"{n} persisted end state is not a media-free public boundary");
                if (PrefixTree.IsInsideSpan(n, n.Depth)) Fail("I13", $"{n} end state lies strictly inside a media span");
            }
            // I11, I12
            int lastIndex = -1;
            foreach (PageRef page in n.PageSpan)
            {
                int end = (page.PageIndex + 1) * blockSize;
                if (end - 1 < n.EdgeStartDepth || end - 1 >= n.Depth || end > n.Depth)
                    Fail("I11", $"page {page.PageIndex} is not owned by the node containing token {end - 1} ({n})");
                if (page.PageIndex <= lastIndex) Fail("I11", $"{n} pages are not unique and sorted");
                lastIndex = page.PageIndex;
                if (page.Block is null) { Fail("I11", $"{n} page {page.PageIndex} has no block"); continue; }
                if (!blockOwners.TryAdd(page.Block, n)) Fail("I11", $"block {page.Block.Id} is owned by two tree pages");
                if (!tree.TryGetBlockOwner(page.Block, out RadixNode recorded) || !ReferenceEquals(recorded, n))
                    Fail("I11", $"block owner index disagrees for block {page.Block.Id}");
                if (host.RefCount(page.Block) < 1) Fail("I11", $"block {page.Block.Id} RefCount {host.RefCount(page.Block)} < tree refs");
                if (page.Store != PageStore.A1HostSlab && page.Store != PageStore.A2ModelPaged && page.Store != PageStore.Both)
                    Fail("I12", $"page {page.PageIndex} of {n} has no store of record");
                if (page.HasA1 && (!host.HoldsSnapshotBytes(page.Block) || host.UsedTokens(page.Block) < blockSize))
                    Fail("I12", $"A1 page {page.PageIndex} of {n} has no full snapshot slab");
                if (page.HasA2 && !host.HoldsModelPagedKv(page.Block))
                    Fail("I12", $"A2 page {page.PageIndex} of {n} is not in model-paged storage");
            }
            // I13: media records sit on the node containing their Start; every media element is covered.
            foreach (MediaSpanRecord rec in n.SpanRecords)
                if (rec.Start < n.EdgeStartDepth || rec.Start >= n.Depth || rec.End <= rec.Start)
                    Fail("I13", $"{n} stores span [{rec.Start},{rec.End}) that does not start in its edge");
            for (int i = 0; i < n.Edge.Length; i++)
            {
                long e = n.Edge[i];
                if (!KeyElem.IsMedia(e)) continue;
                int pos = n.EdgeStartDepth + i;
                int covering = 0;
                bool valueOk = false;
                for (RadixNode? cur = n; cur is not null && !cur.IsRoot; cur = cur.Parent)
                    foreach (MediaSpanRecord rec in cur.SpanRecords)
                        if (rec.Start <= pos && pos < rec.End)
                        {
                            covering++;
                            valueOk = e == KeyElem.Media(rec.Id);
                        }
                if (covering != 1 || !valueOk)
                {
                    Fail("I13", $"media element at {pos} on {n} is covered by {covering} span records");
                    break;
                }
            }
            // I16
            if (scopeLive && tree.TierOf(n) == EvictionTier.PublicTop) publicTop++;
            // I18
            if (n.IsDonationPending && !context.TransactionOpen) Fail("I18", $"{n} is DonationPending outside an admission transaction");
            // I24
            if (n.Depth > tree.Options.ContextLength) Fail("I24", $"{n} depth exceeds the context length");
            // I28
            if (!scopeLive || tree.Scopes[n.ScopeIx].Index != n.ScopeIx)
                Fail("I28", $"{n} refers to scope index {n.ScopeIx} with no live record");
            if (n.ScopeIx == 0 && (n.Flags & NodeFlags.Retired) != 0) Fail("I28", $"public node {n} flagged Retired");

            stateHash += NodeHash(n);
        }

        // I1 (rope side): every rope's slice count and live tokens equal the edges that reference it.
        var ropes = new Dictionary<KeyRope, (int Refs, long Tokens)>(ReferenceEqualityComparer.Instance);
        foreach (RadixNode n in nodes)
        {
            if (n.Edge.Rope is null) continue;
            (int refs, long tokens) = ropes.GetValueOrDefault(n.Edge.Rope);
            ropes[n.Edge.Rope] = (refs + 1, tokens + n.Edge.Length);
        }
        foreach (KeyValuePair<KeyRope, (int Refs, long Tokens)> kv in ropes)
            if (kv.Key.SliceRefs != kv.Value.Refs || kv.Key.LiveSliceTokens != kv.Value.Tokens)
                Fail("I1", $"rope slice accounting {kv.Key.SliceRefs}/{kv.Key.LiveSliceTokens} != edges {kv.Value.Refs}/{kv.Value.Tokens}");

        // I8 totals
        if (cached != tree.Cached) Fail("I8", $"Σ node bytes {cached} != Cached {tree.Cached}");
        if (prot != tree.Protected) Fail("I8", $"Σ protected {prot} != Protected {tree.Protected}");
        if (tree.Protected.AnyNegative || (tree.Cached - tree.Protected).AnyNegative) Fail("I8", "Protected must lie in [0, Cached]");
        if (tree.PendingReclaim.AnyNegative) Fail("I8", "PendingReclaim is negative");

        // I9 list structure
        for (byte id = 1; id <= 2 * EvictionLists.TierCount; id++)
        {
            int count = 0;
            RadixNode? prev = null;
            for (RadixNode? n = tree.Lists.First(id); n is not null; n = n.LruNext)
            {
                count++;
                if (n.LruList != id) Fail("I9", $"{n} is linked in list {id} but records list {n.LruList}");
                if (!ReferenceEquals(n.LruPrev, prev)) Fail("I9", $"list {id} back pointer broken at {n}");
                if (prev is not null && prev.LastAccess > n.LastAccess) Fail("I9", $"list {id} is not ordered by LastAccess at {n}");
                if (!n.InTree || !seen.Contains(n)) Fail("I9", $"list {id} holds {n}, which is not in the tree");
                prev = n;
                if (count > nodes.Count + 1) { Fail("I9", $"list {id} has a cycle"); break; }
            }
            if (!ReferenceEquals(tree.Lists.Last(id), prev)) Fail("I9", $"list {id} tail pointer is wrong");
            if (count != tree.Lists.Count(id)) Fail("I9", $"list {id} count {tree.Lists.Count(id)} != linked {count}");
        }

        // I10 index size
        if (keyOwners.Count != tree.PayloadKeys.Count) Fail("I10", $"key index holds {tree.PayloadKeys.Count} keys, tree holds {keyOwners.Count}");
        // I14 (tree part)
        if (primaryResidents > 1) Fail("I14", $"{primaryResidents} PrimaryResident payloads exist");
        // I15
        var listed = new HashSet<RadixNode>(ReferenceEqualityComparer.Instance);
        foreach (ScopeRecord rec in tree.Scopes.LiveRecords())
        {
            if (rec.Index == 0) continue;
            int count = 0;
            ResourceVector bytes = default;
            RadixNode? prev = null;
            for (RadixNode? n = rec.FirstNode; n is not null; n = n.ScopeNext)
            {
                count++;
                if (n.ScopeIx != rec.Index) Fail("I15", $"{n} is in the list of scope {rec.Index}");
                if (!seen.Contains(n)) Fail("I15", $"scope {rec.Index} lists {n}, which is not in the tree");
                if (!ReferenceEquals(n.ScopePrev, prev)) Fail("I15", $"scope {rec.Index} back pointer broken at {n}");
                bytes += n.Bytes;
                listed.Add(n);
                if (rec.Retired && !SubtreeLocked(n)) Fail("I15", $"retired scope {rec.Index} still holds unlocked {n}");
                prev = n;
                if (count > nodes.Count + 1) { Fail("I15", $"scope {rec.Index} list has a cycle"); break; }
            }
            if (count != rec.NodeCount) Fail("I15", $"scope {rec.Index} NodeCount {rec.NodeCount} != listed {count}");
            if (bytes != rec.Bytes) Fail("I15", $"scope {rec.Index} Bytes {rec.Bytes} != Σ node bytes {bytes}");
            if (rec.NewestLeaf is not null && (!rec.NewestLeaf.InTree || rec.NewestLeaf.ScopeIx != rec.Index))
                Fail("I15", $"scope {rec.Index} NewestLeaf is not one of its nodes");
        }
        foreach (RadixNode n in nodes)
        {
            if (n.ScopeIx == 0 || !tree.Scopes.IsLive(n.ScopeIx)) continue;
            if (!listed.Contains(n)) Fail("I15", $"{n} is missing from its scope list");
        }
        // I16
        if (publicTop > tree.Options.PublicMax) Fail("I16", $"{publicTop} nodes in tier PublicTop > PublicMax {tree.Options.PublicMax}");
        // I19
        if (_lastVersion >= 0)
        {
            if (tree.Version < _lastVersion) Fail("I19", $"Version decreased from {_lastVersion} to {tree.Version}");
            else if (tree.Version == _lastVersion && stateHash != _lastHash) Fail("I19", "the tree changed without a Version bump");
        }
        _lastVersion = tree.Version;
        _lastHash = stateHash;
        // I21
        if (context.AfterEvictionTrigger)
        {
            for (int i = 0; i < ResourceVector.ClassCount; i++)
            {
                var c = (ResourceClass)i;
                long cap = tree.EffectiveCap(c);
                if (tree.Cached[c] > cap) Fail("I21", $"Cached[{c}] {tree.Cached[c]} > EffectiveCap {cap}");
            }
            if (tree.Options.ScopedEndStateLeavesMax > 0 && scopedEndStates > tree.Options.ScopedEndStateLeavesMax)
                Fail("I21", $"{scopedEndStates} scoped end states > sub-cap {tree.Options.ScopedEndStateLeavesMax}");
            if (publicEndStates > tree.Options.PublicMax)
                Fail("I21", $"{publicEndStates} public end states > PublicMax {tree.Options.PublicMax}");
            if (tree.Caps.MaxRetainedNativeSlots > 0 && nativeSlots > tree.Caps.MaxRetainedNativeSlots)
                Fail("I21", $"{nativeSlots} native slots > {tree.Caps.MaxRetainedNativeSlots}");
        }
        // I22 (tree part)
        if (context.Quiescent)
        {
            foreach (RadixNode n in nodes)
                if (n.AnyLock) { Fail("I22", $"{n} is locked at quiescence"); break; }
            if (!tree.Protected.IsZero) Fail("I22", $"Protected {tree.Protected} != 0 at quiescence");
            if (tree.Ledger is { Count: > 0 }) Fail("I22", $"{tree.Ledger.Count} receipts open at quiescence");
        }
        // I25
        foreach (string key in tree.Reclaim.Keys)
            if (tree.TryGetNodeByKey(key, out _) || keyOwners.ContainsKey(key)) Fail("I25", $"queued key {key} is still referenced by a node");
        // I29
        var unscopedIds = new HashSet<ScopeId>();
        var allIds = new Dictionary<ScopeId, int>();
        foreach (ScopeRecord rec in tree.Scopes.LiveRecords())
        {
            if (rec.Index == 0) continue;
            if (rec.Id.IsPublic) Fail("I29", $"scope {rec.Index} has the public id");
            if (!rec.Retired && !allIds.TryAdd(rec.Id, rec.Index)) Fail("I29", $"scope id {rec.Id} is interned twice");
            if (rec.Kind == ScopeKind.Unscoped && !unscopedIds.Add(rec.Id)) Fail("I29", $"unscoped scope id {rec.Id} is shared");
        }
        return v;
    }

    private static ulong NodeHash(RadixNode n)
    {
        var h = new HashCode();
        h.Add(n.Id);
        h.Add(n.Depth);
        h.Add(n.ScopeIx);
        h.Add((byte)n.Flags);
        h.Add(n.EndState?.Key);
        h.Add(n.PageCount);
        for (int i = 0; i < n.PageCount; i++) h.Add((byte)n.Pages![i].Store);
        h.Add(n.LockRef > 0);
        h.Add(n.StateLockRef > 0);
        h.Add(n.PinRef > 0);
        h.Add(n.Children.Count);
        h.Add(n.Parent?.Id);
        ulong x = (ulong)(uint)h.ToHashCode();
        // spread 32 bits over 64 so the order-independent sum rarely cancels
        x *= 0x9E3779B97F4A7C15UL;
        return x ^ (x >> 29);
    }

    private static bool PathHasMedia(RadixNode n)
    {
        for (RadixNode? cur = n; cur is not null && !cur.IsRoot; cur = cur.Parent)
        {
            if (cur.MediaSpanCount > 0) return true;
            for (int i = 0; i < cur.Edge.Length; i++)
                if (KeyElem.IsMedia(cur.Edge[i])) return true;
        }
        return false;
    }

    private static bool SubtreeLocked(RadixNode n)
    {
        if (n.AnyLock || n.IsDonationPending) return true;
        foreach (RadixNode c in n.Children)
            if (SubtreeLocked(c)) return true;
        return false;
    }
}
