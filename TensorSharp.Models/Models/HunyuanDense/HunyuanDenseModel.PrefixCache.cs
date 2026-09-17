// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// HunyuanDense's side of the radix prefix cache contract (DESIGN §6.2.4, class P).
// Pages on its batched paged route (and host slabs where it snapshots) plus the
// primary cache. Readiness=Legacy until its own readiness PR.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class HunyuanDenseModel : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any);

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---- diagnostics ----

        public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();

        public int PrivateHolderCount => 0;

        public int PrimaryCacheLength => _cacheSeqLen;
    }
}
