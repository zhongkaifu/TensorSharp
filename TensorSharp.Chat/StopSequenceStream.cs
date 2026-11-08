// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Server;

/// <summary>
/// Streams decoded text against a request's string stop sequences the way the OpenAI, Ollama and
/// Anthropic APIs return it: a matched stop sequence is not sent, and neither is anything after it.
/// Text that could still be the start of one is held back until the next piece settles it, so a
/// stop sequence split across tokens never leaks its first part either. Every piece used to be sent
/// as it was decoded, and the matching piece too: <c>stop: ["6"]</c> on "count from 1 to 10" sent
/// "1, 2, 3, 4, 5, 6".
/// </summary>
internal sealed class StopSequenceStream
{
    private readonly string[] _stops;
    private readonly StringBuilder _pending = new();
    private readonly StringBuilder _sent = new();

    private StopSequenceStream(string[] stops) => _stops = stops;

    /// <summary>A stream for <paramref name="stops"/>, or null when none can match (none set, or only empty ones).</summary>
    public static StopSequenceStream? For(IEnumerable<string>? stops)
    {
        string[] usable = stops?.Where(s => !string.IsNullOrEmpty(s)).Distinct(StringComparer.Ordinal).ToArray()
            ?? Array.Empty<string>();
        return usable.Length == 0 ? null : new StopSequenceStream(usable);
    }

    /// <summary>True once a stop sequence matched: the turn ends, and nothing more is sent.</summary>
    public bool Matched { get; private set; }

    /// <summary>Everything sent so far: the reply as the client has it.</summary>
    public string SentText => _sent.ToString();

    /// <summary>Takes the next decoded piece and returns the text that can be sent now.</summary>
    public string Append(string piece)
    {
        if (Matched || string.IsNullOrEmpty(piece))
            return string.Empty;
        _pending.Append(piece);
        // Only the unsent tail can hold a match: what was sent could not begin a stop sequence.
        string pending = _pending.ToString();
        int at = -1;
        foreach (string stop in _stops)
        {
            int i = pending.IndexOf(stop, StringComparison.Ordinal);
            if (i >= 0 && (at < 0 || i < at))
                at = i;
        }
        if (at >= 0)
        {
            Matched = true;
            _pending.Clear();
            return Send(pending.Substring(0, at));
        }

        // Hold back the longest tail that is the start of some stop sequence.
        int hold = 0;
        foreach (string stop in _stops)
        {
            for (int k = Math.Min(stop.Length - 1, pending.Length); k > hold; k--)
            {
                if (pending.AsSpan(pending.Length - k).SequenceEqual(stop.AsSpan(0, k)))
                {
                    hold = k;
                    break;
                }
            }
        }
        _pending.Clear();
        _pending.Append(pending, pending.Length - hold, hold);
        return Send(pending.Substring(0, pending.Length - hold));
    }

    /// <summary>Sends what was held back: the turn ended another way (end of sequence, a token limit,
    /// another of this layer's stops), so it cannot become a stop sequence any more.</summary>
    public string Flush()
    {
        if (Matched || _pending.Length == 0)
            return string.Empty;
        string rest = _pending.ToString();
        _pending.Clear();
        return Send(rest);
    }

    private string Send(string text)
    {
        _sent.Append(text);
        return text;
    }
}
