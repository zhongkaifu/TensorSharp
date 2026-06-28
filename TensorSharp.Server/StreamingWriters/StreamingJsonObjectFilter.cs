// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;

namespace TensorSharp.Server.StreamingWriters
{
    /// <summary>
    /// Incrementally extracts the first balanced JSON object from a model's
    /// streamed output, dropping anything before the opening <c>{</c> (markdown
    /// code fences, leading prose, stray special tags) and anything after the
    /// matching <c>}</c> (closing fences, trailing tags). This lets
    /// <c>response_format: { "type": "json_object" }</c> stream token-by-token —
    /// so time-to-first-token reflects prefill latency rather than the full
    /// decode — while still delivering clean JSON, the same shape the buffered
    /// non-streaming path produces via <c>StructuredOutputValidator</c>.
    ///
    /// Brace counting is string- and escape-aware so braces inside string values
    /// never close the object early. Mirrors
    /// <c>StructuredOutputValidator.TryReadBalancedObject</c>, but stateful so it
    /// can run across streamed fragments.
    /// </summary>
    internal sealed class StreamingJsonObjectFilter
    {
        private bool _started;   // seen the opening '{'
        private bool _done;      // emitted the matching closing '}'
        private int _depth;
        private bool _inString;
        private bool _escaping;

        /// <summary>True once the complete JSON object has been emitted.</summary>
        public bool Done => _done;

        /// <summary>True if any object content has been emitted yet.</summary>
        public bool Started => _started;

        /// <summary>
        /// Feed a raw output fragment; returns only the portion that belongs to
        /// the JSON object and should be streamed to the client (may be empty).
        /// </summary>
        public string Feed(string text)
        {
            if (_done || string.IsNullOrEmpty(text))
                return string.Empty;

            var sb = new StringBuilder(text.Length);
            foreach (char ch in text)
            {
                if (!_started)
                {
                    // Skip everything ahead of the object (```json fences, prose,
                    // whitespace, leaked tags) until the object actually begins.
                    if (ch == '{')
                    {
                        _started = true;
                        _depth = 1;
                        sb.Append(ch);
                    }
                    continue;
                }

                sb.Append(ch);

                if (_escaping)
                {
                    _escaping = false;
                    continue;
                }
                if (_inString)
                {
                    if (ch == '\\') _escaping = true;
                    else if (ch == '"') _inString = false;
                    continue;
                }
                if (ch == '"')
                {
                    _inString = true;
                }
                else if (ch == '{')
                {
                    _depth++;
                }
                else if (ch == '}')
                {
                    _depth--;
                    if (_depth == 0)
                    {
                        _done = true;
                        break; // stop at the matching close; drop any trailing junk
                    }
                }
            }

            return sb.ToString();
        }
    }
}
