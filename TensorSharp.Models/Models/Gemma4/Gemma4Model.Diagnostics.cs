// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Text.Json;

namespace TensorSharp.Models;

public partial class Gemma4Model
{
    private string _gemmaDiagnosticDirectory, _gemmaDiagnosticTag;
    private int _gemmaDiagnosticLayers;
    private int _gemmaDiagnosticCurrentLayer;

    private void BeginStreamingDiagnostic(int[] tokens, int startPos)
    {
        _gemmaDiagnosticTag = null;
        string directory = Environment.GetEnvironmentVariable("TS_GEMMA4_TENSOR_DUMP");
        if (string.IsNullOrEmpty(directory) || startPos != 0 || tokens.Length <= 1) return;
        ulong hash = 14695981039346656037UL;
        foreach (int token in tokens)
            for (int shift = 0; shift < 32; shift += 8)
                hash = unchecked((hash ^ (byte)((uint)token >> shift)) * 1099511628211UL);
        _gemmaDiagnosticDirectory = directory;
        _gemmaDiagnosticTag = $"stream.p{startPos}.n{tokens.Length}.t{hash:x16}";
        _gemmaDiagnosticLayers = int.TryParse(Environment.GetEnvironmentVariable("TS_GEMMA4_TENSOR_DUMP_LAYERS"), out int count)
            ? Math.Clamp(count, 0, Config.NumLayers) : Math.Min(6, Config.NumLayers);
    }

    private unsafe void DumpStreamingTensor(Tensor tensor, string stage, int detailLayer = -1)
    {
        if (_gemmaDiagnosticTag == null || tensor == null || detailLayer >= _gemmaDiagnosticLayers) return;
        // Materialization failures remain real compute failures. Only optional
        // diagnostic I/O is swallowed; it must never retry an advanced model.
        using var contiguous = Ops.NewContiguous(tensor);
        var bytes = new ReadOnlySpan<byte>(GetFloatPtr(contiguous), checked((int)contiguous.ElementCount() * sizeof(float)));
        try
        {
            Directory.CreateDirectory(_gemmaDiagnosticDirectory);
            string path = Path.Combine(_gemmaDiagnosticDirectory, _gemmaDiagnosticTag + "." + stage);
            using (var file = File.Create(path + ".f32")) file.Write(bytes);
            File.WriteAllText(path + ".json", JsonSerializer.Serialize(new {
                Stage = stage, Shape = tensor.Sizes.ToArray(), Dtype = "F32", Layout = "Contiguous TensorSharp row-major logical order",
                Scope = "Optional streamed intermediate; matching native files pin existing fusion boundaries. Normalized Q/K inside the native fused RMSNorm/Mul/RoPE block are not exposed separately." }));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            Console.Error.WriteLine($"[Gemma4 diagnostic] Cannot write {stage}: {error.Message}");
            _gemmaDiagnosticTag = null;
        }
    }
}
