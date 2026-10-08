// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

internal sealed class LogitCapture : IDisposable
{
    private readonly string _dataPath;
    private readonly string _indexPath;
    private readonly FileStream _stream;
    private readonly List<object> _rows = [];

    internal LogitCapture(string directory)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Capture requires little-endian F32.");
        _dataPath = Path.Combine(directory, "logits.f32");
        _indexPath = Path.Combine(directory, "logits.json");
        if (File.Exists(_indexPath)) throw new IOException("Use a fresh output directory for full-logit capture.");
        _stream = new FileStream(_dataPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
    }

    internal void Append(float[] logits, int run, int step, IReadOnlyList<int> history, int argmax)
    {
        if (logits.Length == 0 || logits.Any(value => !float.IsFinite(value)))
            throw new ArithmeticException("Capture requires finite, nonempty full-vocabulary logits.");
        long offset = _stream.Position;
        var bytes = MemoryMarshal.AsBytes(logits.AsSpan());
        _stream.Write(bytes);
        _stream.Flush();
        _rows.Add(new { Run = run, Step = step, Stage = step == 0 ? "prefill" : "decode",
            InputTokens = history.ToArray(), Elements = logits.Length, ByteOffset = offset,
            Sha256 = Convert.ToHexString(SHA256.HashData(bytes)), Argmax = argmax });
        File.WriteAllText(_indexPath, JsonSerializer.Serialize(new
        {
            Format = "f32le", DataPath = _dataPath, Rows = _rows,
            Qualification = "Incremental diagnostic data; only a successful final report proves a completed run."
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    public void Dispose() => _stream.Dispose();

    internal object Evidence()
    {
        using var data = File.OpenRead(_dataPath);
        return new { Format = "f32le", DataPath = _dataPath, IndexPath = _indexPath,
            Bytes = data.Length, Sha256 = Convert.ToHexString(SHA256.HashData(data)), Rows = _rows };
    }
}
