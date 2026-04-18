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
            // Multimodal prompt safety is handled by the injector, which clamps reuse so
            // we never split an injected media span and only re-queues embeddings for the
            // suffix that is actually replayed.
            if (model == null || !model.SupportsKVCacheTruncation)
                return 0;
            if (cachedTokens == null || cachedTokens.Count == 0 || inputTokens == null || inputTokens.Count == 0)
                return 0;

            int raw = FindCommonPrefix(cachedTokens, inputTokens);
            if (raw <= 0)
                return 0;

            int replayWindow = GetRequiredReplayWindow(model);
            if (replayWindow > 0)
                raw = Math.Max(0, raw - replayWindow);

            return raw;
        }

        private static int GetRequiredReplayWindow(IModelArchitecture model)
        {
            var config = model.Config;
            if (config == null || !config.UsesCircularKvCache)
                return 0;

            return Math.Max(0, config.SlidingWindow);
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

            return prefix;
        }
    }
}

