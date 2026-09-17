// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Muse-Glimmer's radix prefix cache: host pages within the ring rows and the
// resident primary cache. Rewinds respect ring slack; media reuse stays disabled.
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
