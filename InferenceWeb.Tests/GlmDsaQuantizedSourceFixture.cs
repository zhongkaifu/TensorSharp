// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Bounded synthetic GLM expert tensors made from independent original
/// quantization blocks. Dense/recurrent weights remain synthetic; this exercises
/// actual quantized layouts rather than claiming real-model semantics.</summary>
internal static class GlmDsaQuantizedSourceFixture
{
    internal sealed record Crop(string TargetTensor, GgmlTensorType Type, byte[] Bytes);

    internal static IReadOnlyList<Crop> Read(string sourcePath, int hidden, int ffn, int experts,
        int requestedGroups, string manifestPath)
    {
        using var file = GgufFile.OpenForMemoryPlanning(sourcePath);
        string architecture = file.GetString("general.architecture") ?? "";
        if (architecture is not ("glm5next" or "glm5-next"))
            throw new NotSupportedException("Expert source must be GLM5Next.");
        int layers = checked((int)file.GetUint32(architecture + ".block_count"));
        int next = checked((int)file.GetUint32(architecture + ".nextn_predict_layers"));
        if (layers <= 0 || next < 0 || next >= layers || requestedGroups <= 0)
            throw new InvalidDataException("Invalid source trunk/group geometry.");
        int trunk = layers - next;
        var groups = new HashSet<string>(StringComparer.Ordinal);
        var output = new List<Crop>();
        var evidence = new List<object>();
        string[] parts = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
        for (int layer = 0; layer < trunk && groups.Count < requestedGroups; ++layer)
        {
            var tensors = parts.Select(part => file.Tensors.GetValueOrDefault($"blk.{layer}.{part}")).ToArray();
            if (tensors.Any(t => t == null)) continue;
            if (tensors.Any(t => t.Type is not (GgmlTensorType.IQ1_S or GgmlTensorType.IQ2_XXS
                or GgmlTensorType.IQ3_XXS or GgmlTensorType.IQ4_XS))) continue;
            string key = string.Join(",", tensors.Select(t => (int)t.Type));
            if (!groups.Add(key)) continue;
            int targetLayer = groups.Count;
            for (int part = 0; part < parts.Length; ++part)
            {
                var tensor = tensors[part];
                int k = part == 2 ? ffn : hidden, m = part == 2 ? hidden : ffn;
                long block = GgufFile.GetBlockSize(tensor.Type), blockBytes = GgufFile.GetTypeSize(tensor.Type);
                if (k <= 0 || m <= 0 || experts <= 0 || tensor.Shape.Length != 3 || k % block != 0
                    || tensor.Shape[0] % (ulong)block != 0 || tensor.Shape[0] < (ulong)k
                    || tensor.Shape[1] < (ulong)m || tensor.Shape[2] < (ulong)experts)
                    throw new InvalidDataException("Crop is not aligned to complete independent quantization blocks: " + tensor.Name);
                long sourceRow = checked((long)tensor.Shape[0] / block * blockBytes);
                long sourceExpert = checked(sourceRow * (long)tensor.Shape[1]);
                int targetRow = checked((int)(k / block * blockBytes));
                var region = file.GetTensorFileRegion(tensor.Name);
                if (region.ByteLength != checked(sourceExpert * (long)tensor.Shape[2]))
                    throw new InvalidDataException("Source tensor byte geometry changed: " + tensor.Name);
                byte[] bytes = new byte[checked(targetRow * m * experts)];
                var before = new FileInfo(region.Path);
                long sourceLength = before.Length;
                DateTime sourceModified = before.LastWriteTimeUtc;
                using (var handle = File.OpenHandle(region.Path, FileMode.Open, FileAccess.Read, FileShare.Read))
                    for (int expert = 0; expert < experts; ++expert)
                    for (int row = 0; row < m; ++row)
                    {
                        long offset = checked(region.Offset + expert * sourceExpert + row * sourceRow);
                        var destination = bytes.AsSpan(checked((expert * m + row) * targetRow), targetRow);
                        int read = 0;
                        while (read < destination.Length)
                        {
                            int count = RandomAccess.Read(handle, destination[read..], checked(offset + read));
                            if (count == 0) throw new EndOfStreamException(tensor.Name);
                            read += count;
                        }
                    }
                var after = new FileInfo(region.Path);
                if (after.Length != sourceLength || after.LastWriteTimeUtc != sourceModified)
                    throw new IOException("Source file changed while reading bounded expert blocks: " + region.Path);
                string target = $"blk.{targetLayer}.{parts[part]}";
                output.Add(new(target, tensor.Type, bytes));
                evidence.Add(new
                {
                    target_tensor = target, source_tensor = tensor.Name, type = tensor.Type.ToString(),
                    source_file = region.Path, source_file_length = sourceLength, source_modified_utc = sourceModified,
                    source_tensor_shape = tensor.Shape, source_tensor_offset = region.Offset,
                    source_tensor_bytes = region.ByteLength, source_row_bytes = sourceRow, source_expert_bytes = sourceExpert,
                    source_expert_first_offsets = Enumerable.Range(0, experts).Select(e => checked(region.Offset + e * sourceExpert)).ToArray(),
                    copied_rows_per_expert = m, copied_bytes_per_row = targetRow, copied_shape = new[] { k, m, experts },
                    copied_bytes = bytes.Length, copied_sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant()
                });
            }
        }
        if (groups.Count != requestedGroups) throw new NotSupportedException($"Required {requestedGroups} distinct actual IQ layouts; found {groups.Count}.");
        if (manifestPath != null)
        {
            if (File.Exists(manifestPath)) throw new IOException("Refusing to overwrite expert source evidence.");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(new
            {
                source = Path.GetFullPath(sourcePath), architecture, groups = groups.ToArray(), tensors = evidence,
                limitation = "Complete original quantization blocks are copied verbatim from the first expert rows/columns. " +
                    "All dense, attention, recurrent and routing weights are synthetic. This qualifies quantized layouts and source reuse, not published-model semantics or performance."
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        return output;
    }
}
