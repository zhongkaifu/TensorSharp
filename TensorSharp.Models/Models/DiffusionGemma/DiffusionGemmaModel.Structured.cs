// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TensorSharp.Models
{
    public sealed partial class DiffusionGemmaModel
    {
        /// <summary>The structured reads' own cache (a sequence of the runtime): repeated reads of one prompt reuse its rows.</summary>
        private DiffusionSeqState _structured;

        /// <summary>
        /// Read a seeded Jev canvas once, without sampling, committing tokens, or self-conditioning. Each result row is the
        /// temperature-one distribution over its requested label token IDs, in the caller's order, after the model's final
        /// logit softcap. Positions are zero-based within the canvas, which may be shorter than <see cref="CanvasLength"/>. Only
        /// the requested positions' rows go through the head (the canvas's head, so a row's logits are the full read's) and
        /// come back; the other rows' vocabulary logits are never made. The caller's arrays are never modified and the
        /// returned arrays are owned.
        /// </summary>
        /// <remarks>Like generation, this needs exclusive model access. It keeps one prompt's cache for repeated reads;
        /// <see cref="ClearStructuredCache"/> releases it.</remarks>
        public float[][] ReadStructured(int[] promptTokens, int[] seedCanvas, int[] positions, int[][] tokenIds,
            CancellationToken cancellationToken = default)
            => ReadStructuredCore(promptTokens, seedCanvas, positions, tokenIds, false, cancellationToken);

        /// <summary>Full-vocabulary reference for <see cref="ReadStructured"/>: the same forward, the whole canvas's logits read
        /// back before the labels are picked and normalized (more memory, by design).</summary>
        public float[][] ReadStructuredReference(int[] promptTokens, int[] seedCanvas, int[] positions, int[][] tokenIds,
            CancellationToken cancellationToken = default)
            => ReadStructuredCore(promptTokens, seedCanvas, positions, tokenIds, true, cancellationToken);

        private float[][] ReadStructuredCore(int[] promptTokens, int[] seedCanvas, int[] positions, int[][] tokenIds, bool fullVocabulary,
            CancellationToken cancellationToken)
        {
            ValidateStructuredRequest(promptTokens, seedCanvas, positions, tokenIds, VocabSize, CanvasLength, MaxContextLength);
            cancellationToken.ThrowIfCancellationRequested();
            _structured ??= CreateSeqState();
            BindSequenceCache(_structured.Id);
            _structured.Prefilled = true;
            try
            {
                PrefillActive(promptTokens, ActiveVision, _structured.Cache, cancellationToken);
            }
            finally
            {
                RestorePrimaryCache();
            }
            cancellationToken.ThrowIfCancellationRequested();

            int vocab = VocabSize;
            var rows = new float[positions.Length][];
            if (fullVocabulary)
            {
                float[] logits = CanvasLogits(_structured.Id, seedCanvas, null);
                for (int r = 0; r < rows.Length; r++)
                {
                    rows[r] = new float[tokenIds[r].Length];
                    for (int j = 0; j < rows[r].Length; j++) rows[r][j] = logits[checked((long)positions[r] * vocab + tokenIds[r][j])];
                }
            }
            else
            {
                int[] keep = positions.Distinct().Order().ToArray();
                float[] logits = CanvasRows(_structured.Id, seedCanvas, keep);
                for (int r = 0; r < rows.Length; r++)
                {
                    long at = (long)Array.BinarySearch(keep, positions[r]) * vocab;
                    rows[r] = new float[tokenIds[r].Length];
                    for (int j = 0; j < rows[r].Length; j++) rows[r][j] = logits[at + tokenIds[r][j]];
                }
            }
            cancellationToken.ThrowIfCancellationRequested();
            foreach (float[] row in rows) NormalizeStructuredLabels(row);
            return rows;
        }

        /// <summary>The softcapped logits [keep.Length, vocab] of a canvas step's rows at <paramref name="keep"/> (ascending positions).</summary>
        private float[] CanvasRows(string sequence, int[] canvas, int[] keep)
        {
            CachedGraph graph = null;
            bool owned = false;
            Tensor embedded = null;
            bool bound = Bind(sequence);
            try
            {
                embedded = Embedding(canvas);
                graph = RunBlockRows(null, new[] { canvas.Length }, step => EmitCanvas(step, embedded, null, CanvasHead.Rows, null, keep),
                    out StepOutputs outputs, out owned, "diffusion-canvas-rows");
                return (float[])graph.Outputs[outputs.Logits].Clone();
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded?.Dispose();
                if (bound) RestorePrimaryCache();
            }
        }

        internal static void NormalizeStructuredLabels(float[] logits)
        {
            float max = float.NegativeInfinity;
            foreach (float value in logits)
            {
                if (!float.IsFinite(value))
                    throw new InvalidOperationException("The model produced non-finite structured label logits.");
                max = Math.Max(max, value);
            }
            double denominator = 0;
            foreach (float value in logits) denominator += Math.Exp((double)value - max);
            for (int i = 0; i < logits.Length; i++)
                logits[i] = (float)(Math.Exp((double)logits[i] - max) / denominator);
        }

        internal static void ValidateStructuredRequest(int[] promptTokens, int[] seedCanvas, int[] positions,
            int[][] tokenIds, int vocabSize, int canvasLength, int contextLength)
        {
            ArgumentNullException.ThrowIfNull(promptTokens);
            ArgumentNullException.ThrowIfNull(seedCanvas);
            ArgumentNullException.ThrowIfNull(positions);
            ArgumentNullException.ThrowIfNull(tokenIds);
            if (promptTokens.Length == 0) throw new ArgumentException("A nonempty prompt is required.", nameof(promptTokens));
            if (seedCanvas.Length == 0 || seedCanvas.Length > canvasLength)
                throw new ArgumentOutOfRangeException(nameof(seedCanvas), $"Canvas length must be between 1 and {canvasLength}.");
            if (contextLength > 0 && (long)promptTokens.Length + seedCanvas.Length > contextLength)
                throw new ArgumentException($"Prompt plus canvas exceeds the {contextLength}-token context window.", nameof(promptTokens));
            if (positions.Length == 0 || positions.Length != tokenIds.Length)
                throw new ArgumentException("Positions and label rows must have the same nonzero length.", nameof(positions));
            foreach (int id in promptTokens)
                if ((uint)id >= (uint)vocabSize) throw new ArgumentOutOfRangeException(nameof(promptTokens), "Prompt token outside vocabulary.");
            foreach (int id in seedCanvas)
                if ((uint)id >= (uint)vocabSize) throw new ArgumentOutOfRangeException(nameof(seedCanvas), "Canvas token outside vocabulary.");
            for (int r = 0; r < positions.Length; r++)
            {
                if ((uint)positions[r] >= (uint)seedCanvas.Length)
                    throw new ArgumentOutOfRangeException(nameof(positions), "Position outside canvas.");
                if (tokenIds[r] == null || tokenIds[r].Length == 0)
                    throw new ArgumentException("Each label row must contain at least one token.", nameof(tokenIds));
                var seen = new HashSet<int>();
                foreach (int id in tokenIds[r])
                {
                    if ((uint)id >= (uint)vocabSize)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), "Label token outside vocabulary.");
                    if (!seen.Add(id)) throw new ArgumentException("Label tokens must be distinct within each row.", nameof(tokenIds));
                }
            }
        }

        /// <summary>Release the prompt cache the structured reads keep.</summary>
        public void ClearStructuredCache()
        {
            DisposeSeqState(_structured);
            _structured = null;
        }
    }
}
