// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime;
using System.IO;

namespace TensorSharp.Models
{
    /// <summary>
    /// Engram lookup metadata embedded in the checkpoint GGUF and the n-gram
    /// hashing it drives. This is the managed counterpart of dsv41_engram.h and
    /// dsv41_engram_gguf.h; multipliers, bucket geometry and token IDs must
    /// match exactly.
    /// </summary>
    internal sealed class Dsv41EngramData
    {
        /// <summary>One Engram table: which layer owns it, how many rows it has,
        /// and the hash parameters that address them.</summary>
        internal sealed class LayerLayout
        {
            public int Id;
            public long Rows;
            public ulong[] Multipliers;   // [MaxNgramSize]
            public uint[] Primes;         // [HashColumns] - bucket sizes
            public long[] Offsets;        // [HashColumns] - running sum of Primes
        }

        /// <summary>
        /// The one tensor that holds every Engram table in llama.cpp's layout
        /// (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs). TensorSharp's layout keeps
        /// one <see cref="TableName"/> per Engram layer instead.
        /// </summary>
        public const string CombinedTableName = "per_layer_token_embd.weight";

        /// <summary>One Engram layer's rows [FirstRow, FirstRow + Rows) of the combined table.</summary>
        internal readonly record struct CombinedSlice(int LayerId, long FirstRow, long Rows);

        /// <summary>A layer's own table in TensorSharp's layout, and the name the
        /// executors read it by in either layout.</summary>
        public static string TableName(int layerId) => $"blk.{layerId}.engram_embd.weight";

        public uint VocabSize;
        public uint CompressedVocabSize;
        public uint PadId;               // already compressed; do not apply TokenMap again
        public uint MaxNgramSize;
        public uint HeadCount;
        public uint HeadDim;
        public ulong TokenizerHash;
        public int CandidateSourceLayerId = -1;
        public uint CandidateTopkBlocks;
        public uint CandidateBlockSize;
        public int[] KvSourceLayerIds;
        public int[] IndexSourceLayerIds;
        public int[] TokenMap;            // [VocabSize] -> compressed id
        public LayerLayout[] Layers;

        /// <summary>
        /// llama.cpp's layout only, null in TensorSharp's: where each of
        /// <see cref="Layers"/> sits in <see cref="CombinedTableName"/>, in the same
        /// order. <see cref="LayerLayout.Offsets"/> are already rebased onto these
        /// slices, so the executor must expose each one as that layer's
        /// <see cref="TableName"/> (a view of the same bytes, see
        /// <see cref="TableViews"/>) before it reads a row.
        /// </summary>
        public CombinedSlice[] Slices;

        /// <summary>Hash columns per token per table: one per (lookback, head).</summary>
        public uint HashColumns => (MaxNgramSize - 1) * HeadCount;

        /// <summary>The per-layer tables of llama.cpp's layout as row ranges of
        /// <see cref="CombinedTableName"/>, under the names TensorSharp's layout
        /// gives them; empty in TensorSharp's layout.</summary>
        public (string Name, long FirstRow, long Rows)[] TableViews() => Slices == null
            ? Array.Empty<(string, long, long)>()
            : Slices.Select(s => (TableName(s.LayerId), s.FirstRow, s.Rows)).ToArray();

        /// <summary>
        /// True when the file describes Engram with llama.cpp's <c>deepseek41.ple.*</c>
        /// keys: the <c>ple</c> layer list is present and TensorSharp's
        /// <c>deepseek41.engram.layer_ids</c>, which wins when a file carries both, is
        /// not. The same test as the native loader's has_combined_engram_metadata, so
        /// every backend reads a given file in the same layout.
        /// </summary>
        public static bool HasCombinedMetadata(GgufFile file) =>
            !file.Metadata.ContainsKey("deepseek41.engram.layer_ids") &&
            file.Metadata.ContainsKey("deepseek41.ple.layers");

        /// <summary>
        /// FNV-1a over each token's byte length then its bytes, folded across the
        /// whole vocabulary. Vision artifacts use this fingerprint to identify their
        /// parent checkpoint.
        /// </summary>
        public static ulong FingerprintToken(ulong hash, string token)
        {
            ulong size = (ulong)System.Text.Encoding.UTF8.GetByteCount(token);
            for (int i = 0; i < 8; i++)
                hash = (hash ^ (byte)(size >> (8 * i))) * 1099511628211UL;
            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(token))
                hash = (hash ^ b) * 1099511628211UL;
            return hash;
        }

        private static InvalidDataException Invalid(string key) =>
            new InvalidDataException("Missing or invalid DeepSeek V4.1 GGUF metadata: " + key);

        private static long Integer(GgufFile file, string key, long? fallback = null)
        {
            if (!file.Metadata.TryGetValue(key, out object raw))
                return fallback ?? throw Invalid(key);
            try
            {
                return raw switch
                {
                    uint u32 => u32, int i32 => i32,
                    ulong u64 => checked((long)u64), long i64 => i64,
                    _ => throw Invalid(key),
                };
            }
            catch (OverflowException) { throw Invalid(key); }
        }

        private static uint U32(GgufFile file, string key, long? fallback = null)
        {
            long value = Integer(file, key, fallback);
            if (value < 0 || value > uint.MaxValue) throw Invalid(key);
            return (uint)value;
        }

        private static ulong[] ArrayValues(GgufFile file, string key, int maxCount)
        {
            if (!file.Metadata.TryGetValue(key, out object raw) || raw is not Array array ||
                array.Length == 0 || array.Length > maxCount)
                throw Invalid(key);
            var result = new ulong[array.Length];
            try
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] = array.GetValue(i) switch
                    {
                        uint u32 => u32, int i32 => checked((ulong)i32),
                        ulong u64 => u64, long i64 => checked((ulong)i64),
                        _ => throw Invalid(key),
                    };
            }
            catch (OverflowException) { throw Invalid(key); }
            return result;
        }

        /// <summary>
        /// combined_engram_slices (dsv41_engram_gguf.h). llama.cpp's layout addresses
        /// one combined table with offsets that run on from one Engram layer to the
        /// next. Each layer's buckets must still be one contiguous row range, which
        /// is what lets that range stand in for the layer's own table.
        /// </summary>
        public static CombinedSlice[] CombinedSlices(GgufFile file, ulong tableRows)
        {
            uint ngram = U32(file, "deepseek41.ple.ngram_size");
            uint heads = U32(file, "deepseek41.ple.heads_per_ngram");
            if (ngram < 2 || ngram > 16 || heads == 0 || heads > 128)
                throw Invalid("deepseek41.ple dimensions");
            int columns = (int)((ngram - 1) * heads);
            ulong[] ids = ArrayValues(file, "deepseek41.ple.layers", 128);
            ulong[] primes = ArrayValues(file, "deepseek41.ple.head_vocab_sizes", ids.Length * columns);
            ulong[] offsets = ArrayValues(file, "deepseek41.ple.head_offsets", ids.Length * columns);
            if (primes.Length != ids.Length * columns || offsets.Length != ids.Length * columns)
                throw Invalid("deepseek41.ple flattened array lengths");
            var slices = new CombinedSlice[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] > int.MaxValue) throw Invalid("deepseek41.ple.layers");
                ulong first = offsets[i * columns], rows = 0;
                for (int j = 0; j < columns; j++)
                {
                    ulong prime = primes[i * columns + j];
                    if (prime < 2 || offsets[i * columns + j] != first + rows)
                        throw Invalid("deepseek41.ple bucket layout is not contiguous per layer");
                    rows += prime;
                }
                if (rows > int.MaxValue || first > tableRows || rows > tableRows - first)
                    throw Invalid("deepseek41.ple bucket rows exceed " + CombinedTableName);
                slices[i] = new CombinedSlice((int)ids[i], (long)first, (long)rows);
            }
            return slices;
        }

        /// <summary>
        /// Reads either Engram layout, deciding as the native loader does
        /// (<see cref="HasCombinedMetadata"/>). TensorSharp's: <c>deepseek41.engram.*</c>
        /// keys, one <see cref="TableName"/> per layer and a pad id that is already
        /// compressed. llama.cpp's: <c>deepseek41.ple.*</c> keys plus
        /// <c>deepseek41.engram.token_map</c>/<c>pad_token_id</c> and one
        /// <see cref="CombinedTableName"/>; the per-layer offsets are rebased onto
        /// each layer's slice and the raw pad token is mapped through the token map,
        /// exactly as engram.py's NgramHashState does. <paramref name="findTensor"/>
        /// must still show the combined table (not views of it): a file with both a
        /// combined table and per-layer ones is refused, as is a table nothing
        /// describes. No Engram table payload is read.
        /// </summary>
        public static Dsv41EngramData Load(GgufFile file,
            Func<string, GgufTensorInfo> findTensor = null)
        {
            findTensor ??= name => file.Tensors.TryGetValue(name, out var tensor) ? tensor : null;
            string[] tokens = file.GetStringArray("tokenizer.ggml.tokens");
            if (tokens == null || tokens.Length == 0 || tokens.Length > 1048576)
                throw Invalid("tokenizer.ggml.tokens");
            var embedding = findTensor("token_embd.weight");
            if (embedding == null || embedding.Shape.Length < 2 || embedding.Shape[1] != (ulong)tokens.Length)
                throw Invalid("token embedding vocabulary does not match tokenizer");
            bool combined = HasCombinedMetadata(file);
            GgufTensorInfo table = findTensor(CombinedTableName);
            if (!combined && table != null)
                throw new InvalidDataException(file.Metadata.ContainsKey("deepseek41.engram.layer_ids")
                    ? "DeepSeek V4.1 GGUF has both TensorSharp's per-layer Engram metadata " +
                      $"(deepseek41.engram.layer_ids) and llama.cpp's combined Engram table {CombinedTableName}"
                    : $"DeepSeek V4.1 GGUF has llama.cpp's combined Engram table {CombinedTableName} " +
                      "but no deepseek41.ple.layers metadata that addresses it");
            string Key(string tensorSharp, string llamaCpp) =>
                combined ? "deepseek41.ple." + llamaCpp : "deepseek41.engram." + tensorSharp;
            var data = new Dsv41EngramData
            {
                VocabSize = (uint)tokens.Length,
                TokenizerHash = 14695981039346656037UL,
            };
            foreach (string token in tokens)
                data.TokenizerHash = FingerprintToken(data.TokenizerHash, token);
            if (combined)
            {
                data.MaxNgramSize = U32(file, "deepseek41.ple.ngram_size");
                data.HeadCount = U32(file, "deepseek41.ple.heads_per_ngram");
                data.HeadDim = U32(file, "deepseek41.embedding_length_per_layer_input");
            }
            else
            {
                data.PadId = U32(file, "deepseek41.engram.pad_id");
                data.MaxNgramSize = U32(file, "deepseek41.engram.max_ngram_size");
                data.HeadCount = U32(file, "deepseek41.engram.head_count");
                data.HeadDim = U32(file, "deepseek41.engram.key_length");
            }
            if (data.MaxNgramSize < 2 || data.MaxNgramSize > 16 || data.HeadCount == 0 ||
                data.HeadCount > 128 || data.HeadDim == 0 || data.HeadDim > 65536)
                throw Invalid("deepseek41.engram dimensions");
            ulong[] ids = ArrayValues(file, Key("layer_ids", "layers"), 128);
            int columns = (int)data.HashColumns;
            ulong[] multipliers = ArrayValues(file, Key("multipliers", "layer_multipliers"), ids.Length * (int)data.MaxNgramSize);
            ulong[] primes = ArrayValues(file, Key("primes", "head_vocab_sizes"), ids.Length * columns);
            ulong[] offsets = ArrayValues(file, Key("offsets", "head_offsets"), ids.Length * columns);
            if (multipliers.Length != ids.Length * data.MaxNgramSize ||
                primes.Length != ids.Length * columns || offsets.Length != ids.Length * columns)
                throw Invalid("deepseek41.engram flattened array lengths");
            if (combined)
            {
                if (table == null)
                    throw new InvalidDataException(
                        $"DeepSeek V4.1 Engram metadata names {CombinedTableName} but no shard holds it");
                if (table.Shape.Length < 2 || table.Shape.Skip(2).Any(d => d != 1))
                    throw new InvalidDataException($"{CombinedTableName} must be two-dimensional");
                data.Slices = CombinedSlices(file, table.Shape[1]);
                // Rebase each layer's offsets onto its own slice of the combined table.
                for (int i = 0; i < ids.Length; i++)
                {
                    ulong first = offsets[i * columns];
                    for (int j = 0; j < columns; j++)
                        offsets[i * columns + j] -= first;
                }
            }
            ulong[] map = ArrayValues(file, "deepseek41.engram.token_map", tokens.Length);
            data.TokenMap = new int[map.Length];
            for (int i = 0; i < map.Length; i++)
            {
                if (map[i] >= data.VocabSize) throw Invalid("deepseek41.engram.token_map");
                data.TokenMap[i] = (int)map[i];
                data.CompressedVocabSize = Math.Max(data.CompressedVocabSize, (uint)map[i] + 1);
            }
            if (combined)
            {
                // engram.py: `self.pad_id = token_map[args.engram_pad_id]`. llama.cpp's
                // layout stores the raw token, so compress it here.
                uint padToken = U32(file, "deepseek41.engram.pad_token_id");
                if (padToken >= data.TokenMap.Length) throw Invalid("deepseek41.engram.pad_token_id");
                data.PadId = (uint)data.TokenMap[padToken];
                if (file.Metadata.ContainsKey("deepseek41.engram.compressed_vocab_size") &&
                    U32(file, "deepseek41.engram.compressed_vocab_size") != data.CompressedVocabSize)
                    throw Invalid("deepseek41.engram.compressed_vocab_size does not match the token map");
            }
            uint nLayer = U32(file, "deepseek41.block_count");
            if (nLayer == 0 || nLayer > 128) throw Invalid("deepseek41.block_count");
            data.Layers = new LayerLayout[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] >= nLayer) throw Invalid(Key("layer_ids", "layers"));
                // In llama.cpp's layout a layer's table is its slice of the combined one.
                var tensor = combined
                    ? new GgufTensorInfo { Shape = new[] { table.Shape[0], (ulong)data.Slices[i].Rows } }
                    : findTensor(TableName((int)ids[i]));
                if (tensor == null || tensor.Shape.Length < 2 || tensor.Shape[0] != data.HeadDim ||
                    tensor.Shape[1] == 0 || tensor.Shape[1] > int.MaxValue ||
                    tensor.Shape.Skip(2).Any(d => d != 1))
                    throw Invalid($"Engram embedding tensor dimensions at layer {ids[i]}");
                var layer = new LayerLayout
                {
                    Id = (int)ids[i], Rows = (long)tensor.Shape[1],
                    Multipliers = multipliers.AsSpan(i * (int)data.MaxNgramSize, (int)data.MaxNgramSize).ToArray(),
                    Primes = new uint[columns], Offsets = new long[columns],
                };
                for (int j = 0; j < columns; j++)
                {
                    if (primes[i * columns + j] > uint.MaxValue || offsets[i * columns + j] > long.MaxValue)
                        throw Invalid("deepseek41.engram bucket dimensions");
                    layer.Primes[j] = (uint)primes[i * columns + j];
                    layer.Offsets[j] = (long)offsets[i * columns + j];
                }
                data.Layers[i] = layer;
            }
            ulong[] ratios = ArrayValues(file, "deepseek41.attention.compress_ratios", 256);
            if (ratios.Length < nLayer) throw Invalid("deepseek41.attention.compress_ratios");
            var kv = new List<int>();
            var index = new List<int>();
            int candidate = -1;
            for (int i = 0; i < nLayer; i++)
            {
                if (findTensor(TableName(i)) != null)
                {
                    if (combined)
                        throw new InvalidDataException(
                            $"DeepSeek V4.1 GGUF has both {TableName(i)} and llama.cpp's combined Engram table {CombinedTableName}");
                    if (!ids.Contains((ulong)i))
                        throw Invalid("Engram embedding tensor is absent from deepseek41.engram.layer_ids");
                }
                if (findTensor($"blk.{i}.attn_compressor_kv.weight") != null) kv.Add(i);
                if (findTensor($"blk.{i}.indexer.attn_q_b.weight") != null)
                {
                    index.Add(i);
                    if (candidate < 0 && ratios[i] == 1) candidate = i;
                }
            }
            data.KvSourceLayerIds = kv.ToArray();
            data.IndexSourceLayerIds = index.ToArray();
            // Published Flash GGUF omits pruning geometry. The first ratio-1
            // indexer owns the mask; these are its original checkpoint defaults.
            long source = Integer(file, "deepseek41.attention.indexer.candidate_source_layer", candidate);
            if (source < -1 || source >= nLayer) throw Invalid("deepseek41.attention.indexer.candidate_source_layer");
            data.CandidateSourceLayerId = (int)source;
            if (source >= 0)
            {
                // The 1017 GGUFs spell it candidate_top_k_blocks; TensorSharp's own files and
                // fixtures candidate_top_k. A file with both must not let them disagree. The
                // native loader (dsv41_engram_gguf.h) reads it the same way, so ggml prunes alike.
                const string blocksKey = "deepseek41.attention.indexer.candidate_top_k_blocks";
                const string topkKey = "deepseek41.attention.indexer.candidate_top_k";
                data.CandidateTopkBlocks = U32(file, file.Metadata.ContainsKey(blocksKey) ? blocksKey : topkKey, 2048);
                if (file.Metadata.ContainsKey(blocksKey) && file.Metadata.ContainsKey(topkKey) &&
                    U32(file, topkKey) != data.CandidateTopkBlocks)
                    throw Invalid($"{blocksKey} disagrees with {topkKey}");
                data.CandidateBlockSize = U32(file, "deepseek41.attention.indexer.candidate_block_size", 8);
                if (!index.Contains((int)source)) throw Invalid("candidate source must own an indexer");
            }
            data.Validate((uint)tokens.Length);
            return data;
        }

        private void Validate(uint expectedVocab)
        {
            if (VocabSize != expectedVocab || VocabSize == 0 || VocabSize > 1048576 ||
                CompressedVocabSize == 0 || CompressedVocabSize > VocabSize || PadId >= CompressedVocabSize ||
                Layers.Length == 0 || Layers.Length > 128 || MaxNgramSize < 2 || MaxNgramSize > 16 ||
                HeadCount == 0 || HeadCount > 128 || HeadDim == 0 || HeadDim > 65536 ||
                KvSourceLayerIds.Length == 0 || KvSourceLayerIds.Length > 128 ||
                IndexSourceLayerIds.Length == 0 || IndexSourceLayerIds.Length > 128 ||
                CandidateSourceLayerId < -1 ||
                (CandidateSourceLayerId >= 0 && (CandidateTopkBlocks == 0 || CandidateBlockSize == 0 ||
                    CandidateTopkBlocks > int.MaxValue || CandidateBlockSize > int.MaxValue)))
                throw new InvalidDataException("DeepSeek V4.1 GGUF Engram metadata has invalid dimensions");
            if (TokenMap.Length != VocabSize)
                throw new InvalidDataException("DeepSeek V4.1 GGUF Engram token map does not match the tokenizer vocabulary");
            var seen = new bool[CompressedVocabSize];
            foreach (int value in TokenMap)
            {
                if (value < 0 || value >= CompressedVocabSize)
                    throw new InvalidDataException("DeepSeek V4.1 compressed token id is out of bounds");
                seen[value] = true;
            }
            if (seen.Any(present => !present))
                throw new InvalidDataException("DeepSeek V4.1 compressed token map has missing ids");
            for (int i = 0; i < Layers.Length; i++)
            {
                LayerLayout layer = Layers[i];
                if (layer.Id < 0 || (i > 0 && layer.Id <= Layers[i - 1].Id) ||
                    layer.Rows <= 0 || layer.Rows > int.MaxValue || layer.Multipliers.Length != MaxNgramSize ||
                    layer.Primes.Length != HashColumns || layer.Offsets.Length != HashColumns)
                    throw new InvalidDataException("Invalid DeepSeek V4.1 Engram table dimensions");
                foreach (ulong multiplier in layer.Multipliers)
                    if ((multiplier & 1) == 0 || multiplier > (ulong)long.MaxValue / CompressedVocabSize)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram hash multiplier");
                long total = 0;
                for (int j = 0; j < layer.Primes.Length; j++)
                {
                    if (layer.Primes[j] < 2 || layer.Offsets[j] != total)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram bucket layout");
                    total += layer.Primes[j];
                }
                if (total != layer.Rows)
                    throw new InvalidDataException("DeepSeek V4.1 Engram bucket sizes do not match table rows");
            }
        }

        /// <summary>
        /// Row ids for one ubatch, laid out [layer][token][hash column].
        ///
        /// <para><paramref name="history"/> is the sequence's compressed-token
        /// history and is extended in place, because a lookback reaches tokens
        /// from earlier calls. A negative input token marks an image position: it
        /// stores -1, which blocks that position and every later lookback that
        /// reaches it, substituting the pad token instead.</para>
        /// </summary>
        public int[] HashTokens(ReadOnlySpan<int> tokens, int startPos, ref int[] history, ref int historyLength)
        {
            if (startPos > historyLength)
                throw new InvalidOperationException("DeepSeek V4.1 Engram history is not contiguous");
            foreach (int token in tokens)
            {
                if (token >= 0 && (uint)token >= VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokens), "DeepSeek V4.1 token is out of bounds");
            }

            int needed = startPos + tokens.Length;
            if (history == null || history.Length < needed)
                Array.Resize(ref history, Math.Max(needed, (history?.Length ?? 0) * 2));
            for (int i = 0; i < tokens.Length; i++)
                history[startPos + i] = tokens[i] < 0 ? -1 : TokenMap[tokens[i]];
            historyLength = needed;
            return HashWindow(history, 0, startPos, tokens.Length);
        }

        /// <summary>
        /// The row ids <see cref="HashTokens"/> would return for <paramref name="tokens"/> at
        /// <paramref name="startPos"/>, without touching the history: the Engram reader uses it to read a
        /// forward's rows ahead of the layers. It reads only the lookback before
        /// <paramref name="startPos"/> from the history and the rest from <paramref name="tokens"/>, exactly
        /// as <see cref="HashTokens"/> would see them once it has stored them.
        /// </summary>
        public int[] PreviewHashes(ReadOnlySpan<int> tokens, int startPos, int[] history, int historyLength)
        {
            if (startPos > historyLength || (startPos > 0 && (history == null || history.Length < startPos)))
                throw new InvalidOperationException("DeepSeek V4.1 Engram history is not contiguous");
            foreach (int token in tokens)
            {
                if (token >= 0 && (uint)token >= VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokens), "DeepSeek V4.1 token is out of bounds");
            }
            int lookback = (int)Math.Min((uint)startPos, MaxNgramSize > 0 ? MaxNgramSize - 1 : 0);
            int first = startPos - lookback;
            var window = new int[lookback + tokens.Length];
            for (int i = 0; i < lookback; i++)
                window[i] = history[first + i];
            for (int i = 0; i < tokens.Length; i++)
                window[lookback + i] = tokens[i] < 0 ? -1 : TokenMap[tokens[i]];
            return HashWindow(window, first, startPos, tokens.Length);
        }

        /// <summary>
        /// The hashing loop of <see cref="HashTokens"/>: <paramref name="history"/>[k] holds the compressed token
        /// at absolute position <paramref name="historyBase"/> + k, for every position from the first lookback
        /// the ubatch reaches up to its end.
        /// </summary>
        private int[] HashWindow(int[] history, int historyBase, int startPos, int count)
        {
            uint columns = HashColumns;
            var hashes = new int[Layers.Length * count * columns];
            for (int li = 0; li < Layers.Length; li++)
            {
                LayerLayout layer = Layers[li];
                for (int i = 0; i < count; i++)
                {
                    int pos = startPos + i;
                    ulong rolling = 0;
                    bool blocked = false;
                    for (uint shift = 0; shift < MaxNgramSize; shift++)
                    {
                        // Once a lookback runs off the start of the sequence or
                        // hits an image position, every longer one is blocked too.
                        blocked = blocked || pos < shift || history[pos - shift - historyBase] < 0;
                        int token = blocked ? (int)PadId : history[pos - shift - historyBase];
                        rolling ^= (ulong)token * layer.Multipliers[shift];
                        if (shift == 0)
                            continue;
                        for (uint head = 0; head < HeadCount; head++)
                        {
                            uint column = (shift - 1) * HeadCount + head;
                            // Every head at this lookback sees the same rolling
                            // value; the prime and offset are what separate them.
                            hashes[(li * count + i) * columns + column] =
                                (int)(rolling % layer.Primes[column] + (ulong)layer.Offsets[column]);
                        }
                    }
                }
            }
            return hashes;
        }
    }
}
