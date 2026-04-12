// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    public sealed class DefaultKvCachePolicy : IKVCachePolicy
    {
        public static DefaultKvCachePolicy Shared { get; } = new();

        public int ComputeReusablePrefix(IModelArchitecture model, List<int> cachedTokens, List<int> inputTokens, bool hasMultimodal)
        {
            if (model == null || hasMultimodal || !model.SupportsKVCacheTruncation)
                return 0;
            if (cachedTokens == null || cachedTokens.Count == 0 || inputTokens == null || inputTokens.Count == 0)
                return 0;

            int raw = FindCommonPrefix(cachedTokens, inputTokens);
            if (raw <= 0)
                return 0;

            int slidingWindow = model.Config?.SlidingWindow ?? 0;
            if (slidingWindow > 0)
                raw = Math.Max(0, raw - slidingWindow);

            if (raw < 4)
                return 0;

            double savingsRatio = (double)raw / inputTokens.Count;
            if (savingsRatio < 0.10)
                return 0;

            return raw;
        }

        private static int FindCommonPrefix(List<int> cachedTokens, List<int> inputTokens)
        {
            int maxLen = Math.Min(cachedTokens.Count, inputTokens.Count);
            int prefix = 0;
            for (int i = 0; i < maxLen; i++)
            {
                if (cachedTokens[i] != inputTokens[i])
                    break;
                prefix++;
            }

            if (prefix == 0 || prefix >= inputTokens.Count)
                return 0;

            return prefix;
        }
    }
}

