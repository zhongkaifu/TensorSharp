// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GPT-OSS's side of the radix prefix cache contract (DESIGN §6.2.2, class P).
// M2 parity, as Qwen3: pages and the primary cache, no end states until M7a. The
// sliding window is a mask over a linear cache, so any rewind is exact.
// Readiness=Legacy, so no engine calls a state member.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class GptOssModel : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any, reuseAcrossMediaSpan: true);

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---- diagnostics ----

        public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.SeqLen ?? 0);
    }
}
