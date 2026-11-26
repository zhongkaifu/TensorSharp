// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's tokenizer files, without the weights. The text-encoder GGUF carries no
// tokenizer, so it is assembled from loose files, and the one that is easy to leave out
// - tokenizer_config.json - is the only place the vision markers are defined. These
// tests build a tiny byte-level vocabulary in a temporary folder so they run anywhere.
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.MiniMaxH3;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3TokenizerTests : IDisposable
    {
        private readonly string _dir =
            Path.Combine(Path.GetTempPath(), "h3-tokenizer-" + Guid.NewGuid().ToString("N"));

        public MiniMaxH3TokenizerTests()
        {
            Directory.CreateDirectory(_dir);
            // Printable ASCII maps to itself in the byte-level alphabet, so one entry per
            // character spells every marker out of single-character pieces; no merges.
            var vocab = Enumerable.Range(33, 94).ToDictionary(c => ((char)c).ToString(), c => c - 33);
            vocab["Ġ"] = vocab.Count;   // the byte-level spelling of a space
            File.WriteAllText(Path.Combine(_dir, "vocab.json"), JsonSerializer.Serialize(vocab));
            File.WriteAllText(Path.Combine(_dir, "merges.txt"), "#version: 0.2\n");
        }

        public void Dispose()
        {
            try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
        }

        [Fact]
        public void APictureIsRefusedWhenTokenizerConfigIsMissing()
        {
            var tokenizer = MiniMaxH3TextEncoder.LoadTokenizer(_dir);

            // Without the config the placeholder is thirteen ordinary pieces, which is the
            // silent failure: a prompt longer than the placeholders it was built from.
            Assert.Equal(13, tokenizer.Encode("<|image_pad|>", addSpecial: false).Count);
            var ex = Assert.Throws<InvalidOperationException>(
                () => MiniMaxH3Requests.RequireVisionTokens(tokenizer));
            Assert.Contains("tokenizer_config.json", ex.Message);
        }

        [Fact]
        public void ReferencesThatAreAllSoundsNeedNoTokenizerConfig()
        {
            var tokenizer = MiniMaxH3TextEncoder.LoadTokenizer(_dir);
            MiniMaxH3Requests.MiniMaxH3Reference Of(MiniMaxH3ReferenceKind kind) => new() { Kind = kind };

            // A soundtrack reaches the prompt only as its "<Audio N>: " label.
            MiniMaxH3Requests.RequireVisionTokens(tokenizer,
                new[] { Of(MiniMaxH3ReferenceKind.Audio), Of(MiniMaxH3ReferenceKind.Audio) });
            // Anything the vision tower is shown needs the markers, a clip with its sound too.
            foreach (var shown in new[] { MiniMaxH3ReferenceKind.Image, MiniMaxH3ReferenceKind.Video, MiniMaxH3ReferenceKind.VideoAudio })
            {
                var ex = Assert.Throws<InvalidOperationException>(() => MiniMaxH3Requests.RequireVisionTokens(
                    tokenizer, new[] { Of(MiniMaxH3ReferenceKind.Audio), Of(shown) }));
                Assert.Contains("tokenizer_config.json", ex.Message);
            }
        }

        [Fact]
        public void TheVisionMarkersAreSingleTokensWithTheConfig()
        {
            // The real file's added_tokens_decoder entries for the three markers.
            File.WriteAllText(Path.Combine(_dir, "tokenizer_config.json"), """
                { "added_tokens_decoder": {
                    "151652": { "content": "<|vision_start|>", "special": true },
                    "151653": { "content": "<|vision_end|>", "special": true },
                    "151655": { "content": "<|image_pad|>", "special": true } } }
                """);
            var tokenizer = MiniMaxH3TextEncoder.LoadTokenizer(_dir);

            Assert.Equal(new[] { 151655 }, tokenizer.Encode("<|image_pad|>", addSpecial: false));
            MiniMaxH3Requests.RequireVisionTokens(tokenizer);
        }
    }
}
