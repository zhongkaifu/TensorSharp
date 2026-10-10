// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// A multiple-choice question put to Qwen-Image-2.1's own text encoder. Qwen3-VL-8B-Instruct
// carries an instruction-tuned language-model head next to the trunk the image model
// conditions on, so the probability of each option's letter as the first token of the
// assistant's reply answers a question without a decoder, a KV cache or a second model.
//
// Everything here except the logits themselves is weight-free: the ChatML rendering, the
// letter-to-token mapping and the position debiasing take the tokenizer and the scorer as
// delegates, so they are tested without the 5 GB encoder.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// The image model cannot answer this question at all: its text encoder has no output
    /// head, the backend is not GGML, or the question is over the scoring cap. The caller
    /// decides without the model. Every other failure (cancellation, a native fault,
    /// non-finite logits) is a different exception and propagates.
    /// </summary>
    internal sealed class ImageIntentUnavailableException(string message) : Exception(message);

    internal static class QwenImageIntentScorer
    {
        /// <summary>The option labels, in order. Each must be one token so that its logit is one option's.</summary>
        internal const string Letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        /// <summary>
        /// Qwen3-VL-8B-Instruct's <c>tokenizer.chat_template</c> for one system and one user message
        /// with <c>add_generation_prompt</c>: each role is followed by a newline, each turn by
        /// <c>&lt;|im_end|&gt;\n</c>, and the Instruct template opens the assistant turn with no think
        /// block. Both arguments must already be free of control markers (<see cref="StripControlMarkers"/>).
        /// </summary>
        internal static string RenderChatMl(string system, string user) =>
            "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";

        /// <summary>
        /// <paramref name="text"/> without the <c>&lt;|</c> and <c>|&gt;</c> that delimit Qwen's special
        /// tokens. The tokenizer parses <c>&lt;|im_end|&gt;</c> typed into a message, a picture's prompt
        /// or a file name as the token itself, which would end the user turn early and let that text
        /// write turns of its own. Removing one pair can join the halves of another
        /// (<c>&lt;&lt;||&gt;&gt;</c>), so this repeats until none is left.
        /// </summary>
        internal static string StripControlMarkers(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            while (text.Contains("<|", StringComparison.Ordinal) || text.Contains("|>", StringComparison.Ordinal))
                text = text.Replace("<|", "", StringComparison.Ordinal).Replace("|>", "", StringComparison.Ordinal);
            return text;
        }

        /// <summary>
        /// The user turn: <paramref name="user"/>, then the options' texts lettered A, B, ... in
        /// <paramref name="order"/> (position i shows <c>options[order[i]]</c>), one per line, then the
        /// reply format. An option's line breaks become spaces so it stays one option.
        /// </summary>
        internal static string RenderQuestion(string user, IReadOnlyList<string> options, IReadOnlyList<int> order)
        {
            var text = new StringBuilder(StripControlMarkers(user).TrimEnd());
            if (text.Length > 0) text.Append("\n\n");
            for (int position = 0; position < order.Count; position++)
                text.Append(Letters[position]).Append(". ")
                    .Append(StripControlMarkers(options[order[position]]).ReplaceLineEndings(" ").Trim()).Append('\n');
            return text.Append("Answer with the letter of one option.").ToString();
        }

        /// <summary>
        /// Argument checks, made before any availability check so a caller's mistake is
        /// reported the same way on every backend.
        /// <para>Options that are bare letters are refused: the caller has lettered its own
        /// list and is passing the labels. Scored, that list would be lettered a second time
        /// (<c>A. B</c>), every other pass would credit each letter to a different option, and
        /// the passes would average every answer towards a tie -- a question that silently
        /// never decides anything.</para>
        /// </summary>
        internal static void Validate(string system, string user, IReadOnlyList<string> options)
        {
            ArgumentNullException.ThrowIfNull(system);
            ArgumentNullException.ThrowIfNull(user);
            ArgumentNullException.ThrowIfNull(options);
            if (options.Count < 2 || options.Count > Letters.Length)
                throw new ArgumentException($"A question needs between 2 and {Letters.Length} options; got {options.Count}.", nameof(options));
            for (int i = 0; i < options.Count; i++)
                if (string.IsNullOrWhiteSpace(options[i]))
                    throw new ArgumentException($"Option {i} is empty.", nameof(options));
            if (options.All(IsBareLetter))
                throw new ArgumentException(
                    "The options are bare letters. Pass each option's text: the scorer letters the options itself, in every order it asks.",
                    nameof(options));
        }

        private static bool IsBareLetter(string option)
        {
            string text = option.Trim().TrimEnd('.', ')', ':');
            return text.Length == 1 && Letters.Contains(char.ToUpperInvariant(text[0]));
        }

        /// <summary>
        /// Ask the question once per rotation of the options -- as given, then each shifted one
        /// place further, so that every option is shown at every position once -- and average each
        /// option's probability over the passes. A model reading lettered options favours some
        /// positions (most often A) whatever they say; shown at each position once, every option
        /// gets the same share of that preference, whatever its shape, and an answer that reads
        /// nothing but the positions comes out an exact tie. Two options are the given order and
        /// its reverse. Reversing alone left a middle option where it was: with three options, a
        /// model that split between A and B gave the middle one half and each of the others a
        /// quarter. Probabilities are a softmax over the answer letters' logits only: the share
        /// of the reply's first token that goes to each option.
        /// </summary>
        /// <param name="tokenize">The encoder's tokenizer (no BOS: Qwen adds none).</param>
        /// <param name="nextTokenLogits">Logits of the given candidate token ids after the given prompt.</param>
        /// <param name="maxTokens">The longest prompt the scorer takes.</param>
        internal static ImageIntentChoice Choose(string system, string user, IReadOnlyList<string> options,
            Func<string, IReadOnlyList<int>> tokenize, Func<int[], int[], float[]> nextTokenLogits,
            int maxTokens, CancellationToken cancellationToken)
        {
            Validate(system, user, options);
            ArgumentNullException.ThrowIfNull(tokenize);
            ArgumentNullException.ThrowIfNull(nextTokenLogits);
            int count = options.Count;

            var letterIds = new int[count];
            for (int i = 0; i < count; i++)
            {
                IReadOnlyList<int> ids = tokenize(Letters[i].ToString());
                if (ids.Count != 1)
                    throw new ImageIntentUnavailableException(
                        $"the option letter '{Letters[i]}' is {ids.Count} tokens in this tokenizer, so no single logit is that option's");
                letterIds[i] = ids[0];
            }
            if (letterIds.Distinct().Count() != count)
                throw new ImageIntentUnavailableException("two option letters are the same token in this tokenizer");

            string cleanSystem = StripControlMarkers(system);
            // Position i of rotation r shows option (i + r) mod count.
            int[][] orders = Enumerable.Range(0, count)
                .Select(rotation => Enumerable.Range(0, count).Select(position => (position + rotation) % count).ToArray())
                .ToArray();
            // Tokenize every order before scoring any, so an over-long question is refused
            // before any device work rather than after the first pass.
            int[][] prompts = orders
                .Select(order => tokenize(RenderChatMl(cleanSystem, RenderQuestion(user, options, order))).ToArray())
                .ToArray();
            int longest = prompts.Max(p => p.Length);
            if (longest > maxTokens)
                throw new ImageIntentUnavailableException($"the question is {longest} tokens, over the {maxTokens}-token scoring cap");

            var averaged = new double[count];
            for (int pass = 0; pass < orders.Length; pass++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float[] logits = nextTokenLogits(prompts[pass], letterIds);
                if (logits == null || logits.Length != count)
                    throw new InvalidOperationException($"The scorer returned {logits?.Length ?? 0} logits for {count} options.");
                double[] shares = Softmax(logits);
                for (int position = 0; position < count; position++)
                    averaged[orders[pass][position]] += shares[position] / orders.Length;
            }

            int best = 0;
            for (int i = 1; i < count; i++)
                if (averaged[i] > averaged[best]) best = i;
            double runnerUp = averaged.Where((_, i) => i != best).Max();
            return new ImageIntentChoice(best, (float)averaged[best], (float)(averaged[best] - runnerUp))
            {
                Probabilities = averaged.Select(p => (float)p).ToArray(),
                PromptTokens = longest,
            };
        }

        private static double[] Softmax(float[] logits)
        {
            foreach (float logit in logits)
                if (!float.IsFinite(logit))
                    throw new InvalidOperationException("The text encoder produced a non-finite logit.");
            double max = logits.Max();
            double[] shares = logits.Select(l => Math.Exp(l - max)).ToArray();
            double sum = shares.Sum();
            for (int i = 0; i < shares.Length; i++) shares[i] /= sum;
            return shares;
        }
    }
}
