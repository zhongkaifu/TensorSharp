// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// A <c>role=user</c> message the HOST wrote into a tool loop's conversation — a
    /// correction, a sub-agent handover — as opposed to one the user typed.
    ///
    /// <para>
    /// It has to be a user turn, because strict chat templates allow nothing else there,
    /// but it must never be taken for the user's request. History compaction anchors on
    /// the latest GENUINE user message and may drop everything before it; a host note
    /// taken for that anchor had the model answering "now write your final answer" with
    /// the actual question compacted away. The type is an in-process marker only:
    /// renderers see an ordinary <see cref="ChatMessage"/>, and nothing on the wire or on
    /// disk changes.
    /// </para>
    /// </summary>
    public class HostAuthoredUserMessage : ChatMessage
    {
        /// <summary>A user-role host note with <paramref name="content"/>.</summary>
        public static HostAuthoredUserMessage Create(string content) => new() { Role = "user", Content = content };
    }
}
