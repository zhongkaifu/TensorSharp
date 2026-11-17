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
using System.IO;

namespace TensorSharp.Models.Media
{
    /// <summary>
    /// Media a request carried (an image, a recording, a video) that could not be read: bytes no decoder on this host
    /// understands, a container the decoders refuse, a decode that ran out of time. The client's input rather than a server
    /// fault, so a server answers it with a 400 and <see cref="Exception.Message"/>, which names the attachment by its file
    /// name only (never a directory of the host). An <see cref="IOException"/> (the file could not be read as media), which
    /// the callers that tolerate or refuse undecodable media already catch.
    /// </summary>
    public sealed class MediaInputException : IOException
    {
        public MediaInputException(string message, Exception inner = null) : base(message, inner)
        {
        }

        /// <summary>Whether <paramref name="ex"/> is how a decoder refuses its input (malformed or unsupported data, a
        /// missing decoder, a timeout), as opposed to a failure of the server itself.</summary>
        public static bool IsDecodeFailure(Exception ex)
            => ex is InvalidDataException or NotSupportedException or TimeoutException or FileNotFoundException
                or EndOfStreamException;

        /// <summary><paramref name="ex"/> as a <see cref="MediaInputException"/> (itself when it is one): its message with
        /// every directory of <paramref name="paths"/> left out, so the reply names the attachment and not where the host
        /// keeps it.</summary>
        public static MediaInputException From(Exception ex, IEnumerable<string> paths)
        {
            ArgumentNullException.ThrowIfNull(ex);
            if (ex is MediaInputException media)
                return media;
            return new MediaInputException("A media attachment of the request could not be read: " + Scrub(ex.Message, paths), ex);
        }

        /// <summary><paramref name="text"/> with each of <paramref name="paths"/> replaced by its file name.</summary>
        public static string Scrub(string text, IEnumerable<string> paths)
        {
            if (string.IsNullOrEmpty(text) || paths == null)
                return text ?? string.Empty;
            foreach (string path in paths)
            {
                if (string.IsNullOrEmpty(path))
                    continue;
                text = text.Replace(path, Path.GetFileName(path), StringComparison.Ordinal);
                string full;
                try { full = Path.GetFullPath(path); }
                catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException) { continue; }
                text = text.Replace(full, Path.GetFileName(full), StringComparison.Ordinal);
                string dir = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(dir))
                    text = text.Replace(dir + Path.DirectorySeparatorChar, string.Empty, StringComparison.Ordinal);
            }
            return text;
        }
    }
}
