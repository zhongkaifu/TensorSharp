// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// The radix prefix cache contract, once for every family on the runtime: pages (host slabs
    /// where snapshots restore across sequences, model-paged storage where the batched forward
    /// runs) and the resident primary cache; no end states. A stack with recurrent layers is
    /// class R: its pages do not carry the running state, it cannot rewind, and it offers no
    /// snapshots, so a shared prefix is re-prefilled rather than restored wrong. Reuse across an
    /// image span stays off pending image-turn exactness coverage.
    /// </summary>
    public abstract partial class StandardDecoderModel : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => _anyRecurrent
                ? PageFamilyCapabilities(FamilyClass.R, TruncationKind.None, pagesNeedStateAtEnd: true)
                : PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any);

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---- diagnostics ----

        public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();

        public int PrivateHolderCount => 0;

        public int PrimaryCacheLength => _cacheSeqLen;
    }
}
