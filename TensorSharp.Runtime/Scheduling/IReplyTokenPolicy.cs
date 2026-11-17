// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A model whose vocabulary holds tokens its replies never sample as text: media placeholders, and the tokens that switch
    /// a reply into speech (LFM3-Omni: HF's generate suppresses &lt;|audio|&gt;, &lt;|audio_out|&gt;, &lt;|image|&gt;,
    /// &lt;|video|&gt; in every text slot, and a text-only reply bans &lt;|audio_start|&gt; and &lt;|mixed_start|&gt;). The
    /// engine gives every request's sampler these tokens at a logit of -inf, so every host and API gets the same replies.
    /// </summary>
    public interface IReplyTokenPolicy
    {
        /// <summary>The tokens a reply never draws as text: a plain text reply (<paramref name="speechReply"/> false), or a
        /// reply whose slots an <see cref="ISequenceStepDriver"/> chooses (true: it may switch into speech).</summary>
        IReadOnlyList<int> SuppressedReplyTokens(bool speechReply);
    }
}
