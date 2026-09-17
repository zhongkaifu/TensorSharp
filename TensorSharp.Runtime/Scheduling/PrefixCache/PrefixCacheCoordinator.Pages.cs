// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

internal sealed partial class PrefixCacheCoordinator
{
    private sealed class PageHost(BlockPool pool) : IPrefixTreePageHost
    {
        public void RetainPage(KvBlock block) => pool.Touch(block);
        public void FreePage(KvBlock block) => pool.Free(block);
        public bool HoldsModelPagedKv(KvBlock block) => block.HoldsModelPagedKv;
        public bool HoldsSnapshotBytes(KvBlock block) => block.HoldsSnapshotBytes;
        public int UsedTokens(KvBlock block) => block.Used;
        public int RefCount(KvBlock block) => block.RefCount;
    }

    internal void CapturePages(SequenceState seq)
    {
        if (_tree.Caps.Pages == PageSupport.None || !_scheduler.PrefixCacheConfigured)
            return;
        int blockSize = _pool.BlockSize;
        int length = Math.Min(seq.NumComputedTokens, seq.NumTotalTokens);
        if (seq.CacheBreakpoints != null)
            length = Math.Min(length, seq.CacheBreakpointLimit);
        if (_tree.Caps.PageWindowTokens > 0)
            length = Math.Min(length, _tree.Caps.PageWindowTokens);
        length = PromptMediaSpans.ClampReusablePrefix(length, seq.MediaSpans, seq.MediaSpans,
            _tree.Caps.ReuseAcrossMediaSpan);
        int count = Math.Min(length / blockSize, seq.BlockTable.NumBlocks);
        if (count == 0) return;

        var pages = new List<PageRef>(count);
        for (int i = 0; i < count; i++)
        {
            KvBlock block = seq.BlockTable.Blocks[i];
            bool a1 = block.HoldsSnapshotBytes && block.Used == blockSize
                && _tree.Caps.Pages is PageSupport.A1HostSlab or PageSupport.Both;
            bool a2 = block.HoldsModelPagedKv && !_tree.Caps.PagesNeedStateAtEnd
                && _tree.Caps.Pages is PageSupport.A2ModelPaged or PageSupport.Both;
            if (!a1 && !a2) continue;
            if (_tree.TryGetBlockOwner(block, out _)) continue;
            pages.Add(new PageRef(block, i, a1 && a2 ? PageStore.Both
                : a1 ? PageStore.A1HostSlab : PageStore.A2ModelPaged, block.IsRestorablePrefixEnd));
        }
        if (pages.Count == 0) return;

        int end = count * blockSize;
        // A radix edge may span a partial final page, but never invents a resumable
        // endpoint inside a media span. The tree's media records validate lookup.
        KeyRope key = GetKey(seq);
        int boundary = Math.Min(seq.SharedPrefixTokens, length);
        RadixNode node = _tree.Insert(key, end, GetScope(seq), boundary,
            end == seq.PromptTokens.Count ? NodeFlags.PromptEnd : NodeFlags.None, GetSpans(seq));
        try
        {
            while (pages.Count > 0 && (pages[^1].PageIndex + 1) * blockSize > node.Depth)
                pages.RemoveAt(pages.Count - 1);
            _tree.AttachPages(node, CollectionsMarshal.AsSpan(pages));
        }
        finally
        {
            _tree.CollectIfEmpty(node);
        }
        _tree.EnforceCaps(EvictionTier.PublicTop);
    }

    internal bool TryAdoptPages(SequenceState seq, MatchPlan plan, in LockReceipt receipt)
    {
        if (seq.BlockTable.NumBlocks != 0 || receipt.PathAnchor == null) return false;
        int count = plan.Length / _pool.BlockSize;
        if (count == 0) return false;
        var blocks = new KvBlock[count];
        for (RadixNode node = receipt.PathAnchor; !node.IsRoot; node = node.Parent!)
        {
            foreach (PageRef page in node.PageSpan)
            {
                if (page.PageIndex >= count) continue;
                if (!_tree.StoreBacked(page)) return false;
                blocks[page.PageIndex] = page.Block;
            }
        }
        foreach (KvBlock block in blocks)
            if (block == null) return false;

        // Reserve all managed metadata before transferring a holder or acquiring
        // page references, so an allocation failure cannot strand ownership.
        seq.BlockTable.EnsureBlockCapacity(count);

        if (plan.Mode == MaterializeMode.CopyA2PagesToHolder)
        {
            var ids = new int[count];
            for (int i = 0; i < count; i++) ids[i] = blocks[i].Id;
            if (!_cacheModel.TryCopyPagedToHolder(ids, plan.Length, seq.RequestId))
                return false;
        }
        else if (plan.Mode is not (MaterializeMode.InjectA1Pages or MaterializeMode.BindPagesInPlace))
            return false;

        foreach (KvBlock block in blocks)
        {
            _pool.Touch(block);
            seq.BlockTable.AppendBlock(block);
        }
        seq.SetComputedTokensForPrefixAdoption(plan.Length);
        seq.PrefixCacheReusedTokens = plan.Length;
        seq.KvStateInPagedStorage = plan.Mode == MaterializeMode.BindPagesInPlace;
        return true;
    }

    internal void EnsureFreePages(int needed)
    {
        // Tree references keep cached blocks out of the pool's free queue. Evict
        // before preempting work. A released tree reference may still belong to a
        // running sequence, so keep checking physical capacity after each pass.
        while (_pool.NumFreeBlocks < needed)
        {
            long before = _tree.Cached.PoolPages;
            _tree.Evict(ResourceClass.PoolPages, needed - _pool.NumFreeBlocks,
                ReleaseReason.Evicted, EvictionTier.PublicTop);
            if (_tree.Cached.PoolPages >= before) break;
        }
    }
}
