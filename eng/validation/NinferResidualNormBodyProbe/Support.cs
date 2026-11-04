// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Validation;

internal static class Support
{
    internal const int Vocab = 248320, Valid = 248077, Guard = 64, GuardBits = 0x4e57a193, IntGuard = 0x6327194e;
    internal const long Budget = 512L << 20;
    internal static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    internal sealed record Pin(string Path, long Bytes, string Sha256);
    internal static void Need(bool b, string message) { if (!b) throw new InvalidDataException(message); }
    internal static string Hash(ReadOnlySpan<byte> b) => Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();
    internal static Pin FilePin(string path)
    { using var f = File.OpenRead(path); return new(System.IO.Path.GetFullPath(path), f.Length, Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant()); }
    internal static string Encoded(object value) => JsonSerializer.Serialize(value, Json);
    internal static void Save(string path, object value, bool once = false)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(Encoded(value) + "\n");
        using (var f = new FileStream(path, once ? FileMode.CreateNew : FileMode.Create, FileAccess.Write, FileShare.None)) { f.Write(bytes); f.Flush(true); }
        if (once) { using var f = new FileStream(path + ".sha256", FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(System.Text.Encoding.ASCII.GetBytes(Hash(bytes) + "  " + System.IO.Path.GetFileName(path) + "\n")); f.Flush(true); }
    }
    internal static void CheckPins(IEnumerable<Pin> pins)
    { foreach (var p in pins) Need(FilePin(p.Path) == p, "Pinned input/native/assembly changed: " + p.Path); }
    internal static bool Link(string p) => new FileInfo(p).LinkTarget != null || new DirectoryInfo(p).LinkTarget != null;
    internal static void Fresh(string path)
    {
        Need(System.IO.Path.IsPathFullyQualified(path) && path == System.IO.Path.GetFullPath(path), "Canonical absolute fresh output required.");
        string[] pieces = path.Split(System.IO.Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(pieces.Contains("artifacts") || Enumerable.Range(0, pieces.Length - 1).Any(i => pieces[i] == "docs" && pieces[i + 1] == "validation"), "Ignored evidence path required.");
        Need(!System.IO.Path.Exists(path) && !Link(path), "Output already exists or is aliased.");
        for (string? p = System.IO.Path.GetDirectoryName(path); p != null; p = System.IO.Path.GetDirectoryName(p)) Need(!Link(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    internal static Dictionary<string, string> Options(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i]; Need(key is "--execute" or "--self-test" or "--compare" or "--mode" or "--artifact" or "--expect-directory-sha" or "--inputs" or "--expect-inputs-sha" or "--native-pins" or "--expect-native-pins-sha" or "--out" or "--old" or "--off" or "--on", "Unknown option: " + key);
            Need(!values.ContainsKey(key), "Duplicate option.");
            if (key is "--execute" or "--self-test" or "--compare") values.Add(key, "1");
            else { Need(++i < args.Length, "Missing option value."); values.Add(key, args[i]); }
        }
        Need(new[] { "--execute", "--self-test", "--compare" }.Count(values.ContainsKey) <= 1, "Exclusive modes required."); return values;
    }
    internal static string Required(Dictionary<string, string> options, string key) => options.TryGetValue(key, out string? s) ? s : throw new ArgumentException("Missing " + key);
    internal static T Member<T>(object value, string name)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        Type t = value.GetType(); object? result = t.GetProperty(name, flags)?.GetValue(value) ?? t.GetField(name, flags)?.GetValue(value);
        return result is T v ? v : throw new InvalidDataException("Runtime diagnostic member unavailable/wrong type: " + t.FullName + "." + name);
    }
    internal static object? Optional(object value, string name) => value.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public)?.GetValue(value);
    internal static float[] Poison(int rows, int pass)
    {
        var a = Enumerable.Repeat(BitConverter.Int32BitsToSingle(pass == 0 ? unchecked((int)0x7fc1f17e) : unchecked((int)0x7fc2f17e)), checked(rows * Vocab + Guard)).ToArray();
        for (int i = rows * Vocab; i < a.Length; i++) a[i] = BitConverter.Int32BitsToSingle(GuardBits); return a;
    }
    internal static void Head(float[] a, int rows, int[]? drawn = null)
    {
        Need(a.Length == rows * Vocab + Guard && a.AsSpan(rows * Vocab).ToArray().All(x => BitConverter.SingleToInt32Bits(x) == GuardBits), "Full-head guard changed.");
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int c = 0; c < Vocab; c++) { Need(float.IsFinite(a[r * Vocab + c]), "Nonfinite physical head."); if (a[r * Vocab + c] > a[r * Vocab + best]) best = c; }
            Need(best < Valid && (drawn == null || drawn[r] == best), "Physical-head argmax/padding differs.");
        }
        if (drawn != null) Need(drawn.AsSpan(rows).ToArray().All(x => x == IntGuard), "Argmax output guard changed.");
    }
    internal sealed class Store
    {
        private readonly string _output;
        private readonly HashSet<string> _files = new(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _repeat = new(StringComparer.Ordinal);
        internal readonly List<object> Vectors = new();
        internal long Bytes; internal int HeadRows, TapCaptures, RepeatChecks;
        internal Store(string output) { _output = output; Directory.CreateDirectory(System.IO.Path.Combine(output, "vectors")); }
        internal void Raw(string id, int pass, int row, string format, ReadOnlySpan<byte> data)
        {
            string sha = Hash(data), file = "vectors/" + sha + "." + format, key = id + "/" + row + "/" + format;
            string path = System.IO.Path.Combine(_output, file);
            if (_files.Add(file)) { Need(Bytes + data.Length <= Budget, "Raw evidence budget exhausted; incomplete, never PASS."); using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(data); Bytes += data.Length; }
            else Need(File.ReadAllBytes(path).AsSpan().SequenceEqual(data), "CAS collision or retained bytes changed.");
            Vectors.Add(new { id, pass, row, format, bytes = data.Length, file, sha256 = sha });
            if (pass == 0) Need(_repeat.TryAdd(key, sha), "Duplicate reset reference: " + key);
            else { Need(_repeat.TryGetValue(key, out var old) && old == sha, "Reset repeat changed raw output: " + key); RepeatChecks++; }
        }
        internal void Record(string id, int pass, float[] head, int rows, int[]? drawn = null)
        {
            Head(head, rows, drawn);
            for (int r = 0; r < rows; r++) Raw(id, pass, r, "f32", MemoryMarshal.AsBytes(head.AsSpan(r * Vocab, Vocab)));
            HeadRows += rows;
        }
        internal void VerifyFiles() { foreach (string file in _files) Need(FilePin(System.IO.Path.Combine(_output, file)).Sha256 == System.IO.Path.GetFileNameWithoutExtension(file), "Raw CAS changed before completion."); }
    }
}
