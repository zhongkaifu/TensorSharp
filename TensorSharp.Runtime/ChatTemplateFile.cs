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
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// A Jinja chat template read from a file, used instead of the one in the model's GGUF
    /// (<c>tokenizer.chat_template</c>): the <c>--chat-template-file</c> option of
    /// <c>TensorSharp.Cli</c> and <c>TensorSharp.Server</c>, spelled as llama.cpp spells it.
    ///
    /// <para>
    /// Model repositories ship a corrected or extended <c>chat_template.jinja</c> beside
    /// their GGUFs, and a GGUF converted before the fix keeps the old one (issue #285
    /// asked for exactly this). The file replaces the template text only: the output
    /// parser, the BOS decision and the tool-call grammar stay the model family's, so a
    /// template that changes the tool-call format will not be parsed back.
    /// </para>
    /// <para>
    /// The file is parsed by this engine when it is read, so a template it cannot parse
    /// stops the host at startup instead of every request falling back to the built-in
    /// renderer. A family that normally bypasses its GGUF template for a purpose-built
    /// renderer renders the operator's file (<see cref="ChatTemplate.IsOperatorTemplate"/>);
    /// a file the engine renders wrongly still takes the reported fallback to that renderer.
    /// </para>
    /// </summary>
    public sealed class ChatTemplateFile
    {
        public const string Flag = "--chat-template-file";

        /// <summary>The largest template accepted. Shipped templates are 2-30 KB.</summary>
        public const int MaxBytes = 1024 * 1024;

        private ChatTemplateFile(string path, string template)
        {
            Path = path;
            Template = template;
        }

        /// <summary>The file, as an absolute path.</summary>
        public string Path { get; }

        /// <summary>The template text.</summary>
        public string Template { get; }

        /// <summary>
        /// Read and parse the template at <paramref name="path"/>, and register it as an
        /// operator template.
        /// </summary>
        /// <param name="source">Where the path came from, for the error.</param>
        /// <exception cref="ArgumentException">
        /// The file does not exist, cannot be read, is empty, is larger than
        /// <see cref="MaxBytes"/>, or does not parse.
        /// </exception>
        public static ChatTemplateFile Load(string? path, string source = Flag)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException($"{source} requires a path to a Jinja chat template file.");

            string full;
            try
            {
                full = System.IO.Path.GetFullPath(path.Trim());
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                throw new ArgumentException($"Invalid value for {source}: '{path}' is not a valid path.", ex);
            }

            var info = new FileInfo(full);
            if (!info.Exists)
                throw new ArgumentException($"Invalid value for {source}: '{path}' is not an existing file.");
            if (info.Length > MaxBytes)
                throw new ArgumentException(
                    $"Invalid value for {source}: '{path}' is {info.Length:N0} bytes; a chat template is at most {MaxBytes:N0}.");

            string template;
            try
            {
                template = File.ReadAllText(full, Encoding.UTF8);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new ArgumentException($"Invalid value for {source}: '{path}' could not be read ({ex.Message}).", ex);
            }

            // A leading BOM is the editor's, not the template's.
            template = template.TrimStart('﻿');
            if (string.IsNullOrWhiteSpace(template))
                throw new ArgumentException($"Invalid value for {source}: '{path}' is empty.");

            try
            {
                _ = new Jinja2Template(template);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                throw new ArgumentException(
                    $"Invalid value for {source}: '{path}' is not a chat template TensorSharp can parse: {ex.Message}", ex);
            }

            ChatTemplate.RegisterOperatorTemplate(template);
            return new ChatTemplateFile(full, template);
        }

        /// <summary>
        /// Take <c>--chat-template-file PATH</c> or <c>--chat-template-file=PATH</c> out of
        /// <paramref name="args"/> (matched case-insensitively, the last one winning) and copy
        /// the other arguments to <paramref name="remaining"/>. For the CLI, whose own switch
        /// ignores what it does not know, so a spelling the server accepts is never dropped.
        /// </summary>
        /// <returns>The path as given, or null when the flag is absent.</returns>
        /// <exception cref="ArgumentException">The flag has no value.</exception>
        public static string? TakePath(IReadOnlyList<string> args, List<string> remaining)
        {
            ArgumentNullException.ThrowIfNull(args);
            ArgumentNullException.ThrowIfNull(remaining);
            string? result = null;
            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i];
                if (string.Equals(arg, Flag, StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 >= args.Count)
                        throw new ArgumentException($"{Flag} requires a path to a Jinja chat template file.");
                    result = args[++i];
                }
                else if (arg.StartsWith(Flag + "=", StringComparison.OrdinalIgnoreCase))
                {
                    result = arg.Substring(Flag.Length + 1);
                }
                else
                {
                    remaining.Add(arg);
                }
            }
            return result;
        }
    }
}
