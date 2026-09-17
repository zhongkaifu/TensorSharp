// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Mistral3's side of the radix prefix cache contract (DESIGN §6.2.3, class P).
// No holders: pages and the primary cache. Reuse across an image span stays off
// until its M5 PR carries an image-turn exactness test. Readiness=Legacy.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Mistral3Model : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any, reuseAcrossMediaSpan: false);

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---- diagnostics ----

        public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();

        public int PrivateHolderCount => 0;

        public int PrimaryCacheLength => _cacheSeqLen;
    }
}
