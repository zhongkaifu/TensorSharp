// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Muse-Glimmer's side of the radix prefix cache contract (DESIGN §6.3.2, class S).
// Not an IBatchedPagedModel: A1 pages within the ring rows, the primary cache,
// rewinds within the ring slack (rows - W - 1, the M0c guard), no end states,
// no reuse across media until validated. Readiness=Legacy.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class MuseGlimmerModel : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => RingSlack > 0
                ? PageFamilyCapabilities(FamilyClass.S, TruncationKind.WithinRingSlack, RingSlack)
                : PageFamilyCapabilities(FamilyClass.S, TruncationKind.None);

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        /// <summary>How far the primary may rewind on a wrapped ring and stay exact (M0c).</summary>
        private int RingSlack => _kvSwaRows - _slidingWindow - 1;

        // ---- diagnostics ----

        public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();

        public int PrivateHolderCount => 0;

        public int PrimaryCacheLength => _cacheSeqLen;
    }
}
