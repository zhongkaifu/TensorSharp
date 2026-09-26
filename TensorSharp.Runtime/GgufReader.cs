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
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TensorSharp.Runtime
{
    public enum GgufValueType : uint
    {
        Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
        Uint32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
        String = 8, Array = 9, Uint64 = 10, Int64 = 11, Float64 = 12
    }

    public enum GgmlTensorType : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
        IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19,
        IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
        I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28,
        IQ1_M = 29, BF16 = 30,
        TQ1_0 = 34, TQ2_0 = 35,
        MXFP4 = 39,
        NVFP4 = 40,
        Q1_0 = 41,
        Q2_0 = 42,
        // PrismML Bonsai2 on-disk formats; transcoded by TensorSharp, never
        // passed to upstream ggml as ggml_type values.
        PQ2_0 = 142, PTQ1_0 = 143,
    }

    public class GgufTensorInfo
    {
        public string Name { get; set; } = string.Empty;
        public ulong[] Shape { get; set; } = Array.Empty<ulong>();
        public GgmlTensorType Type { get; set; }
        public ulong Offset { get; set; }

        public long NumElements
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= (long)d;
                return n;
            }
        }
    }

    public partial class GgufFile : IDisposable
    {
        public uint Version { get; private set; }
        public Dictionary<string, object> Metadata { get; } = new();
        public Dictionary<string, GgufTensorInfo> Tensors { get; } = new();
        public long DataOffset { get; private set; }

        /// <summary>Unaligned end of the KV + tensor table; a shard with no
        /// tensor data of its own may legitimately end here, before the
        /// alignment padding that <see cref="DataOffset"/> assumes.</summary>
        private long _tableEnd;

        private FileStream _stream;
        private string _path;
        private MemoryMappedFile? _mappedFile;
        private MemoryMappedViewAccessor? _mappedView;
        private unsafe byte* _mappedBase;
        private bool _mappedPointerAcquired;
        private unsafe byte* _lockedBase;
        private ulong _lockedLength;

        /// <summary>
        /// Sibling shards of a split GGUF (<c>NAME-00001-of-000NN.gguf</c>), in split
        /// order and excluding this one. Empty for a single-file model.
        /// </summary>
        private readonly List<GgufFile> _shards = new();

        /// <summary>
        /// Owning shard of every tensor that lives in a sibling file. Tensors absent
        /// from this map are stored in this file, so the single-file read path is
        /// unchanged (the map stays empty).
        /// </summary>
        private readonly Dictionary<string, GgufFile> _tensorOwner = new(StringComparer.Ordinal);

        public GgufFile(string path) : this(path, isShard: false) { }

        /// <summary>
        /// Opens one GGUF and reads THIS file only, leaving its sibling shards
        /// alone.
        ///
        /// <para>llama.cpp's gguf-split stamps split.count into every shard while
        /// keeping the whole metadata block -- including a 130k-token vocabulary --
        /// in shard 1, so opening any shard normally pulls in all of them. A caller
        /// that is enumerating the shards itself would then open N files N times:
        /// N^2 file handles and N redundant vocabulary parses, which on a 7-shard
        /// checkpoint measured ~240 ms and ~215 MB of allocation for nothing, and
        /// left every tensor attributed to the last shard opened.</para>
        /// </summary>
        public static GgufFile OpenWithoutSiblingShards(string path) => new GgufFile(path, isShard: true);

        private GgufFile(string path, bool isShard)
        {
            _path = path;
            _stream = File.OpenRead(path);
            try
            {
                Parse();
                if (!isShard)
                    OpenSiblingShards();
            }
            catch
            {
                // in case of exceptions, the constructor doesn't complete and Dispose won't be called, so we need to clean up here
                // We need to call Dispose() and not just close the stream, since the shards may have been opened and need to be disposed as well.
                Dispose();
                // rethrow the original exception to preserve the stack trace
                throw;
            }
        }

        /// <summary>Every file this model is stored in, starting with this one.</summary>
        public IReadOnlyList<string> FilePaths
        {
            get
            {
                var paths = new List<string>(_shards.Count + 1) { _path };
                foreach (var s in _shards)
                    paths.Add(s._path);
                return paths;
            }
        }

        /// <summary>True when the model is stored across more than one GGUF file.</summary>
        public bool IsSplit => _shards.Count > 0;

        /// <summary>
        /// Open the remaining files of a split GGUF and merge their tensor tables into
        /// this one, so callers see a single flat <see cref="Tensors"/> table and read
        /// through the same API however the checkpoint was sharded.
        ///
        /// <para>Split checkpoints (llama.cpp's <c>gguf-split</c> layout, which every
        /// very large release ships in — GLM-5.2 is six files) put all metadata in the
        /// first shard and zero tensors in it; the weights live in the siblings.
        /// Without this, opening the first shard yields a model with no weights.</para>
        ///
        /// <para>A missing sibling is a hard error: a partially loaded model would
        /// otherwise fail much later as a missing-tensor exception inside a model
        /// constructor, which reads as an unsupported architecture rather than as an
        /// incomplete download.</para>
        /// </summary>
        private void OpenSiblingShards()
        {
            int splitCount = (int)GetUint32("split.count", 0);
            if (splitCount <= 1)
                return;

            // gguf-split names shards "<prefix>-%05d-of-%05d.gguf". Derive the prefix
            // from this file's own name rather than from metadata, so a renamed set
            // still resolves as long as the shards were renamed together.
            string dir = Path.GetDirectoryName(Path.GetFullPath(_path)) ?? ".";
            string name = Path.GetFileName(_path);
            var m = System.Text.RegularExpressions.Regex.Match(
                name, @"^(?<prefix>.*)-(?<no>\d{5})-of-(?<count>\d{5})\.gguf$",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            if (!m.Success)
                return;

            string prefix = m.Groups["prefix"].Value;
            int selfNo = int.Parse(m.Groups["no"].Value);

            for (int i = 1; i <= splitCount; i++)
            {
                if (i == selfNo)
                    continue;
                string shardPath = Path.Combine(dir, $"{prefix}-{i:D5}-of-{splitCount:D5}.gguf");
                if (!File.Exists(shardPath))
                    throw new FileNotFoundException(
                        $"{_path} is shard {selfNo} of {splitCount}, but {Path.GetFileName(shardPath)} is missing. " +
                        "Every shard of a split GGUF must sit in the same directory.", shardPath);

                var shard = new GgufFile(shardPath, isShard: true);
                _shards.Add(shard);
                foreach (var kv in shard.Tensors)
                {
                    Tensors[kv.Key] = kv.Value;
                    _tensorOwner[kv.Key] = shard;
                }
            }
        }

        /// <summary>The file a tensor's bytes live in: a sibling shard, or this file.</summary>
        private GgufFile OwnerOf(GgufTensorInfo tensorInfo) =>
            tensorInfo != null && _tensorOwner.TryGetValue(tensorInfo.Name, out var owner) ? owner : this;

        /// <summary>
        /// Pins the GGUF mmap region in physical RAM via mlock(2). This
        /// prevents the kernel from evicting model-weight pages between
        /// inference passes (which would otherwise force the next forward
        /// to page-fault every weight back from SSD/swap). Best-effort:
        /// silently no-ops on failure (e.g. when the process memlock
        /// rlimit is too low, or the kernel rejects the wire request).
        /// Idempotent — safe to call multiple times.
        /// </summary>
        public unsafe bool TryLockMappedRegion()
        {
            if (_lockedBase != null)
                return true;
            foreach (var shard in _shards)
                shard.TryLockMappedRegion();
            EnsureMappedView();
            if (_mappedBase == null)
                return false;
            try
            {
                long capacity = _mappedView!.Capacity;
                if (capacity <= 0)
                    return false;
                ulong len = (ulong)capacity;

                // First try a single mlock for the whole region. macOS XNU
                // sometimes returns EAGAIN when asked to wire many GB at
                // once even though the global limit allows it — split into
                // chunks and try again. 256 MB chunks are large enough to
                // amortise syscall overhead and small enough to avoid the
                // single-call rejection.
                int rc = mlock(_mappedBase, (nuint)len);
                if (rc == 0)
                {
                    _lockedBase = _mappedBase;
                    _lockedLength = len;
                    return true;
                }
                LastLockError = Marshal.GetLastWin32Error();

                const ulong chunk = 256UL * 1024 * 1024;
                ulong locked = 0;
                while (locked < len)
                {
                    ulong remaining = len - locked;
                    ulong step = remaining < chunk ? remaining : chunk;
                    int rcChunk = mlock(_mappedBase + locked, (nuint)step);
                    if (rcChunk != 0)
                    {
                        LastLockError = Marshal.GetLastWin32Error();
                        if (locked > 0)
                            _ = munlock(_mappedBase, (nuint)locked);
                        return false;
                    }
                    locked += step;
                }

                _lockedBase = _mappedBase;
                _lockedLength = len;
                LastLockError = 0;
                return true;
            }
            catch (DllNotFoundException) { return false; }
            catch (EntryPointNotFoundException) { return false; }
        }

        public int LastLockError { get; private set; }

        private readonly List<(IntPtr Address, nuint Length)> _lockedRanges = new();

        /// <summary>
        /// Pins only these byte ranges of the mapped file (page-aligned and merged), not
        /// the whole region. Once most tensors have been copied into device buffers and
        /// their pages released, locking the whole file would fault those pages back in
        /// and wire every copied byte a second time; the tensors still read straight
        /// from the mapping are the only ones that need to stay resident.
        /// Best-effort like <see cref="TryLockMappedRegion"/>: returns false if any range
        /// could not be locked (the others stay locked), and unlocks them all on dispose.
        /// </summary>
        public unsafe bool TryLockRanges(IEnumerable<(IntPtr Address, long Length)> ranges)
        {
            long page = Environment.SystemPageSize;
            var aligned = new List<(long Start, long End)>();
            foreach (var (address, length) in ranges)
            {
                if (address == IntPtr.Zero || length <= 0)
                    continue;
                long start = (long)address & ~(page - 1);
                long end = ((long)address + length + page - 1) & ~(page - 1);
                aligned.Add((start, end));
            }
            aligned.Sort((a, b) => a.Start.CompareTo(b.Start));

            var merged = new List<(long Start, long End)>();
            foreach (var range in aligned)
            {
                if (merged.Count > 0 && range.Start <= merged[^1].End)
                    merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, range.End));
                else
                    merged.Add(range);
            }

            bool all = merged.Count > 0;
            try
            {
                foreach (var (start, end) in merged)
                {
                    nuint length = (nuint)(end - start);
                    if (mlock((void*)start, length) == 0)
                        _lockedRanges.Add(((IntPtr)start, length));
                    else
                    {
                        LastLockError = Marshal.GetLastWin32Error();
                        all = false;
                    }
                }
            }
            catch (DllNotFoundException) { return false; }
            catch (EntryPointNotFoundException) { return false; }
            if (all)
                LastLockError = 0;
            return all;
        }

        /// <summary>Bytes currently pinned by <see cref="TryLockRanges"/>.</summary>
        public long LockedRangeBytes
        {
            get
            {
                long total = 0;
                foreach (var (_, length) in _lockedRanges)
                    total += (long)length;
                return total;
            }
        }

        private bool _prefaulted;

        /// <summary>
        /// Warm the OS page cache for the whole GGUF with parallel positional
        /// reads before the loaders touch it. The load path reads the file
        /// through at most a couple of streams (the serial F32/dequant reads,
        /// plus one mmap-faulting upload thread per TP rank), which caps a
        /// cold load at single-stream throughput. Network-backed model storage
        /// is far faster in parallel (measured on a MooseFS volume: ~440 MB/s
        /// on one stream, ~1.8 GB/s at 8-16 streams), so a 16-stream warm-up
        /// pass first makes every subsequent read a RAM hit. On an
        /// already-cached file the pass is a quick no-op (reads hit the page
        /// cache), so it is safe to call unconditionally; it skips itself when
        /// the file cannot reasonably fit in memory (> half of available RAM —
        /// huge models stream through their own chunked loaders instead).
        /// Controlled by TS_GGUF_PREFAULT=0 (disable) and
        /// TS_GGUF_PREFAULT_THREADS (default: min(16, cores)).
        /// </summary>
        public void PrefaultFileCache()
        {
            if (_prefaulted)
                return;
            _prefaulted = true;

            if (Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT") == "0")
                return;

            // iOS/iPadOS: reading the whole model through 16 threads only inflates
            // the app's dirty-page footprint against the jetsam limit, and local
            // flash has none of the network-filesystem readahead this optimizes
            // for. Let the mmap'd pages fault in on demand instead.
            if (OperatingSystem.IsIOS() || OperatingSystem.IsTvOS())
                return;

            // Split GGUF: warm every shard, not just the (tensor-less) first one.
            foreach (var shard in _shards)
                shard.PrefaultFileCache();

            long length;
            try { length = _stream.Length; }
            catch { return; }
            if (length <= 0)
                return;

            var memInfo = GC.GetGCMemoryInfo();
            if (memInfo.TotalAvailableMemoryBytes > 0 && length > memInfo.TotalAvailableMemoryBytes / 2)
                return;

            int threads = Math.Min(16, Environment.ProcessorCount);
            {
                string? env = Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT_THREADS");
                if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int t) && t > 0)
                    threads = t;
            }

            // One contiguous region per thread: every stream reads strictly
            // sequentially, which is what network filesystems' readahead
            // optimizes for (dynamic chunk claiming interleaves the workers
            // across the file and was measured ~3x slower on MooseFS).
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                long regionBytes = (length + threads - 1) / threads;
                Parallel.For(0, threads,
                    new ParallelOptions { MaxDegreeOfParallelism = threads },
                    () => new byte[8 << 20],
                    (region, _, buffer) =>
                    {
                        // A private handle per stream: FUSE-backed filesystems
                        // (MooseFS et al.) keep per-open readahead state, and
                        // sixteen streams sharing one handle serialize on it.
                        using var handle = File.OpenHandle(_path, FileMode.Open, FileAccess.Read, FileShare.Read);
                        long offset = region * regionBytes;
                        long end = Math.Min(offset + regionBytes, length);
                        while (offset < end)
                        {
                            int want = (int)Math.Min(buffer.Length, end - offset);
                            int read = RandomAccess.Read(handle, buffer.AsSpan(0, want), offset);
                            if (read <= 0)
                                break;
                            offset += read;
                        }
                        return buffer;
                    },
                    _ => { });
            }
            catch
            {
                return; // Best-effort: a failed warm-up just means cold-speed reads.
            }
            sw.Stop();

            // Only worth mentioning when it actually pulled data off storage.
            if (sw.Elapsed.TotalSeconds >= 1.0)
            {
                double gb = length / (1024.0 * 1024.0 * 1024.0);
                Console.WriteLine($"  Prefaulted GGUF page cache: {gb:F1} GiB with {threads} read streams in {sw.Elapsed.TotalSeconds:F1}s ({gb / sw.Elapsed.TotalSeconds:F2} GiB/s)");
            }
        }

        [LibraryImport("libc", EntryPoint = "mlock", SetLastError = true)]
        private static unsafe partial int mlock(void* addr, nuint len);

        [LibraryImport("libc", EntryPoint = "munlock", SetLastError = true)]
        private static unsafe partial int munlock(void* addr, nuint len);

        private void Parse()
        {
            using var reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);

            uint magic = reader.ReadUInt32();
            if (magic != 0x46554747) // "GGUF" in little-endian
                throw new InvalidDataException($"Not a GGUF file (magic: 0x{magic:X8})");

            Version = reader.ReadUInt32();
            if (Version < 2)
                throw new NotSupportedException($"GGUF version {Version} not supported");

            ulong tensorCount = reader.ReadUInt64();
            ulong kvCount = reader.ReadUInt64();

            for (ulong i = 0; i < kvCount; i++)
            {
                string key = ReadString(reader);
                var valType = (GgufValueType)reader.ReadUInt32();
                object value = ReadValue(reader, valType);
                Metadata[key] = value;
            }

            for (ulong i = 0; i < tensorCount; i++)
            {
                var info = new GgufTensorInfo();
                info.Name = ReadString(reader);
                uint dims = reader.ReadUInt32();
                info.Shape = new ulong[dims];
                for (uint d = 0; d < dims; d++)
                    info.Shape[d] = reader.ReadUInt64();
                info.Type = (GgmlTensorType)reader.ReadUInt32();
                info.Offset = reader.ReadUInt64();
                Tensors[info.Name] = info;
            }

            long pos = _stream.Position;
            int alignment = 32;
            if (Metadata.TryGetValue("general.alignment", out var a))
                alignment = Convert.ToInt32(a);
            _tableEnd = pos;
            DataOffset = pos + (alignment - pos % alignment) % alignment;
        }

        /// <summary>
        /// Byte length this file must have for every tensor in its table to be
        /// present: where the last tensor's data ends. Tensors whose footprint
        /// this reader cannot size (an unknown quantization type) are skipped
        /// rather than throwing, so this stays usable as a sanity check on a
        /// file the caller may only want metadata from.
        /// </summary>
        public long GetRequiredLength(out string? lastTensorName)
        {
            // Split GGUFs often front-load a metadata-only first shard: every
            // tensor in its table lives in a sibling file, so the file ends
            // right after the table and the alignment padding DataOffset
            // assumes never exists. Only demand bytes past the table when a
            // tensor actually claims them.
            long required = _tableEnd;
            lastTensorName = null;
            foreach (var t in Tensors.Values)
            {
                // Merged-in shard tensors are sized against their own file.
                if (_tensorOwner.ContainsKey(t.Name))
                    continue;

                long bytes;
                try { bytes = GetTensorByteCount(t); }
                catch (NotSupportedException) { continue; }
                catch (IndexOutOfRangeException) { continue; }

                long end = DataOffset + (long)t.Offset + bytes;
                if (end > required)
                {
                    required = end;
                    lastTensorName = t.Name;
                }
            }
            return required;
        }

        /// <summary>
        /// Throws when the file is shorter than its tensor table says it should
        /// be. A truncated GGUF (an interrupted download or copy) otherwise
        /// fails as a short read deep inside weight loading, after the loader
        /// has already committed its buffers - which reads as a loader bug
        /// rather than a bad file. Call this before allocating anything.
        /// </summary>
        public void ThrowIfTruncated()
        {
            foreach (var shard in _shards)
                shard.ThrowIfTruncated();

            long required = GetRequiredLength(out string? lastTensorName);
            long actual = _stream.Length;
            if (actual < required)
            {
                double missingGiB = (required - actual) / (1024.0 * 1024.0 * 1024.0);
                throw new IOException(
                    $"{_path} is incomplete: the file is {actual} bytes but its {Tensors.Count} tensors need " +
                    $"{required} ({missingGiB:F2} GiB missing; {lastTensorName ?? "?"} is the last one). " +
                    "Re-download this file.");
            }

            // A file LONGER than its tensor table is not truncated, so it loads and
            // runs — but the surplus has to have come from somewhere, and in practice
            // it means two writers appended to the same partial download. That
            // corrupts the interior silently: the model reports sensible shapes, runs
            // at full speed, and emits garbage. Alignment padding is tens of bytes,
            // so anything past a mebibyte is worth saying out loud.
            const long SlackBytes = 1L << 20;
            if (actual - required > SlackBytes)
            {
                double extraGiB = (actual - required) / (1024.0 * 1024.0 * 1024.0);
                Console.Error.WriteLine(
                    $"warning: {_path} is {extraGiB:F2} GiB LARGER than its {Tensors.Count} tensors " +
                    $"need ({actual} bytes on disk, {required} accounted for). A GGUF should end where " +
                    "its tensor data ends; surplus usually means an interrupted download was resumed " +
                    "by a second writer, which corrupts the interior without shortening the file. " +
                    "Verify the size against the publisher and re-download if it differs.");
            }
        }

        public string? GetString(string key, string? defaultValue = null)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return v as string ?? defaultValue;
        }

        public uint GetUint32(string key, uint defaultValue = 0)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is int[] ia && ia.Length > 0) return (uint)ia[0];
            if (v is uint[] ua && ua.Length > 0) return ua[0];
            return Convert.ToUInt32(v);
        }

        public float GetFloat32(string key, float defaultValue = 0f)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is float[] fa && fa.Length > 0) return fa[0];
            return Convert.ToSingle(v);
        }

        public bool GetBool(string key, bool defaultValue = false)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return Convert.ToBoolean(v);
        }

        public string[]? GetStringArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is string[] sa) return sa;
            return null;
        }

        public float[]? GetFloatArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is float[] fa) return fa;
            return null;
        }

        public int[]? GetInt32Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is int[] ia) return ia;
            if (v is uint[] ua)
            {
                var result = new int[ua.Length];
                for (int i = 0; i < ua.Length; i++) result[i] = (int)ua[i];
                return result;
            }
            return null;
        }

        public bool[]? GetBoolArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is bool[] ba) return ba;
            return null;
        }

        /// <summary>
        /// A UINT64 metadata array. Used by the qwen4exp PLE n-gram hash, whose
        /// multipliers and per-head vocabulary sizes are 64-bit by construction -
        /// the hash multiplies token ids by ~2^44 constants and takes the result
        /// modulo a ~20 M row count, so nothing narrower carries it.
        /// </summary>
        public ulong[]? GetUint64Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is ulong[] ua) return ua;
            if (v is uint[] u32)
            {
                var result = new ulong[u32.Length];
                for (int i = 0; i < u32.Length; i++) result[i] = u32[i];
                return result;
            }
            if (v is long[] i64)
            {
                var result = new ulong[i64.Length];
                for (int i = 0; i < i64.Length; i++) result[i] = (ulong)i64[i];
                return result;
            }
            return null;
        }

        public uint[]? GetUint32Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is uint[] ua) return ua;
            if (v is int[] ia)
            {
                var result = new uint[ia.Length];
                for (int i = 0; i < ia.Length; i++) result[i] = (uint)ia[i];
                return result;
            }
            return null;
        }

        public byte[] ReadTensorData(GgufTensorInfo tensorInfo)
        {
            var owner = OwnerOf(tensorInfo);
            if (!ReferenceEquals(owner, this))
                return owner.ReadTensorData(tensorInfo);

            long byteCount = GetTensorByteCount(tensorInfo);
            byte[] data = new byte[byteCount];
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            _stream.ReadExactly(data, 0, data.Length);
            return data;
        }

        /// <summary>
        /// Read F32 tensor data directly into a float array in chunks (for tensors > 2GB raw bytes).
        /// </summary>
        public unsafe void ReadTensorDataToFloat32(GgufTensorInfo tensorInfo, float[] dest, long numElements)
        {
            var owner = OwnerOf(tensorInfo);
            if (!ReferenceEquals(owner, this))
            {
                owner.ReadTensorDataToFloat32(tensorInfo, dest, numElements);
                return;
            }

            long totalBytes = numElements * 4;
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            const int chunkBytes = 16 * 1024 * 1024;
            byte[] buffer = new byte[chunkBytes];
            long bytesRead = 0;

            fixed (float* destBase = dest)
            {
                while (bytesRead < totalBytes)
                {
                    int toRead = (int)Math.Min(totalBytes - bytesRead, chunkBytes);
                    _stream.ReadExactly(buffer, 0, toRead);
                    fixed (byte* srcPtr = buffer)
                    {
                        Buffer.MemoryCopy(srcPtr, (byte*)destBase + bytesRead,
                            totalBytes - bytesRead, toRead);
                    }
                    bytesRead += toRead;
                }
            }
        }

        /// <summary>
        /// Read F32 tensor data directly into native memory pointed to by dest (for tensors > 2G elements).
        /// </summary>
        public unsafe void ReadTensorDataToFloat32Native(GgufTensorInfo tensorInfo, IntPtr dest, long numElements)
        {
            var owner = OwnerOf(tensorInfo);
            if (!ReferenceEquals(owner, this))
            {
                owner.ReadTensorDataToFloat32Native(tensorInfo, dest, numElements);
                return;
            }

            // Straight into the caller's native memory. This used to allocate a
            // flat 16 MB managed buffer per call and copy through it -- for a model
            // carrying ~300 small F32 tensors (norms, biases) that is ~4.5 GiB of
            // large-object allocation and one redundant copy of every byte, on the
            // load critical path.
            ReadExactlyInto(DataOffset + (long)tensorInfo.Offset, (byte*)dest, numElements * 4);
        }

        /// <summary>Reads <paramref name="byteCount"/> bytes at
        /// <paramref name="offset"/> straight into native memory.</summary>
        private unsafe void ReadExactlyInto(long offset, byte* dest, long byteCount)
        {
            _stream.Seek(offset, SeekOrigin.Begin);
            long done = 0;
            while (done < byteCount)
            {
                // int-sized spans, so a >2 GiB tensor still reads in one pass per
                // chunk rather than needing a staging buffer.
                int want = (int)Math.Min(byteCount - done, int.MaxValue);
                _stream.ReadExactly(new Span<byte>(dest + done, want));
                done += want;
            }
        }

        /// <summary>
        /// Read tensor data directly into pre-allocated native memory (for tensors > 2GB).
        /// </summary>
        public unsafe void ReadTensorDataToNative(GgufTensorInfo tensorInfo, IntPtr dest, long byteCount)
        {
            var owner = OwnerOf(tensorInfo);
            if (!ReferenceEquals(owner, this))
            {
                owner.ReadTensorDataToNative(tensorInfo, dest, byteCount);
                return;
            }

            ReadExactlyInto(DataOffset + (long)tensorInfo.Offset, (byte*)dest.ToPointer(), byteCount);
        }

        private unsafe void ReadTensorDataToNativeLegacy(GgufTensorInfo tensorInfo, IntPtr dest, long byteCount)
        {
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            byte[] buffer = new byte[Math.Min(byteCount, 8 * 1024 * 1024)];
            long remaining = byteCount;
            byte* destPtr = (byte*)dest.ToPointer();
            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                _stream.ReadExactly(buffer, 0, toRead);
                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, (IntPtr)destPtr, toRead);
                destPtr += toRead;
                remaining -= toRead;
            }
        }

        public unsafe bool TryGetTensorDataPointer(GgufTensorInfo tensorInfo, out IntPtr dataPtr)
        {
            var owner = OwnerOf(tensorInfo);
            if (!ReferenceEquals(owner, this))
                return owner.TryGetTensorDataPointer(tensorInfo, out dataPtr);

            dataPtr = IntPtr.Zero;
            if (tensorInfo == null)
                return false;

            EnsureMappedView();
            if (_mappedBase == null)
                return false;

            dataPtr = (IntPtr)(_mappedBase + DataOffset + (long)tensorInfo.Offset);
            return true;
        }

        public long GetTensorByteCount(GgufTensorInfo tensorInfo)
        {
            long ne0 = (long)tensorInfo.Shape[0];
            long rows = 1;
            for (int i = 1; i < tensorInfo.Shape.Length; i++)
                rows *= (long)tensorInfo.Shape[i];

            long rowBytes = GetRowBytes(tensorInfo.Type, ne0);
            return rowBytes * rows;
        }

        private static long GetRowBytes(GgmlTensorType type, long ne0)
        {
            long blockSize = GetBlockSize(type);
            long typeSize = GetTypeSize(type);
            return (ne0 / blockSize) * typeSize;
        }

        public static long GetBlockSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                case GgmlTensorType.F16:
                case GgmlTensorType.BF16:
                case GgmlTensorType.I8:
                case GgmlTensorType.I16:
                case GgmlTensorType.I32:
                case GgmlTensorType.I64:
                case GgmlTensorType.F64:
                    return 1;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q5_1:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                case GgmlTensorType.IQ4_NL:
                case GgmlTensorType.MXFP4:
                    return 32;
                case GgmlTensorType.NVFP4:
                case GgmlTensorType.Q2_0:
                    return 64;
                case GgmlTensorType.Q1_0:
                case GgmlTensorType.PQ2_0:
                case GgmlTensorType.PTQ1_0:
                    return 128;
                default:
                    return 256;
            }
        }

        public static long GetTypeSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32: return 4;
                case GgmlTensorType.F16: return 2;
                case GgmlTensorType.BF16: return 2;
                case GgmlTensorType.Q4_0: return 2 + 32 / 2;
                case GgmlTensorType.Q4_1: return 2 + 2 + 32 / 2;
                case GgmlTensorType.Q5_0: return 2 + 4 + 32 / 2;
                case GgmlTensorType.Q5_1: return 2 + 2 + 4 + 32 / 2;
                case GgmlTensorType.Q8_0: return 2 + 32;
                case GgmlTensorType.Q8_1: return 2 + 2 + 32;
                case GgmlTensorType.Q2_K: return 256 / 16 + 256 / 4 + 2 + 2;
                case GgmlTensorType.Q3_K: return 256 / 8 + 256 / 4 + 12 + 2;
                case GgmlTensorType.Q4_K: return 2 + 2 + 12 + 256 / 2;
                case GgmlTensorType.Q5_K: return 2 + 2 + 12 + 256 / 8 + 256 / 2;
                case GgmlTensorType.Q6_K: return 256 / 2 + 256 / 4 + 256 / 16 + 2;
                case GgmlTensorType.Q8_K: return 4 + 256 + 2 * 256 / 16;
                case GgmlTensorType.IQ2_XXS: return 2 + 256 / 8 * 2;           // 66
                case GgmlTensorType.IQ2_XS: return 2 + 256 / 8 * 2 + 256 / 32; // 74
                case GgmlTensorType.IQ3_XXS: return 2 + 3 * (256 / 8);         // 98
                case GgmlTensorType.IQ1_S: return 2 + 256 / 8 + 256 / 16;      // 50
                case GgmlTensorType.IQ4_NL: return 2 + 32 / 2;                 // 18
                case GgmlTensorType.IQ3_S: return 2 + 13 * (256 / 32) + 256 / 64; // 110
                case GgmlTensorType.IQ2_S: return 2 + 256 / 4 + 256 / 16;      // 82
                case GgmlTensorType.IQ4_XS: return 2 + 2 + 256 / 64 + 256 / 2; // 136
                case GgmlTensorType.IQ1_M: return 256 / 8 + 256 / 16 + 256 / 32; // 56
                case GgmlTensorType.TQ1_0: return 2 + 256 / 64 + (256 - 4 * 256 / 64) / 5; // 54
                case GgmlTensorType.TQ2_0: return 2 + 256 / 4;                 // 66
                case GgmlTensorType.MXFP4: return 1 + 32 / 2;                  // 17
                case GgmlTensorType.NVFP4: return 4 + 64 / 2;               // 36
                case GgmlTensorType.Q1_0: return 2 + 128 / 8;               // 18
                case GgmlTensorType.Q2_0: return 2 + 64 / 4;                // 18
                case GgmlTensorType.PQ2_0: return 2 + 128 / 4;              // 34
                case GgmlTensorType.PTQ1_0: return 2 + 26;                  // 28
                case GgmlTensorType.I8: return 1;
                case GgmlTensorType.I16: return 2;
                case GgmlTensorType.I32: return 4;
                case GgmlTensorType.I64: return 8;
                case GgmlTensorType.F64: return 8;
                default:
                    throw new NotSupportedException($"Unknown GGML tensor type: {type}");
            }
        }

        /// <summary>
        /// One string from the header. Reads into a reusable buffer rather than a
        /// fresh byte[] per string: a header carries one string per vocabulary
        /// entry, per merge and per tensor, so on a 150k-token vocabulary the
        /// throwaway arrays alone were tens of MB of garbage per parse, and a load
        /// parses the header more than once.
        /// </summary>
        private byte[] _stringScratch = new byte[256];

        private string ReadString(BinaryReader reader)
        {
            ulong len = reader.ReadUInt64();
            if (len > int.MaxValue)
                throw new InvalidDataException($"GGUF string length {len} is out of range");
            int n = (int)len;
            if (n == 0)
                return string.Empty;
            if (_stringScratch.Length < n)
                _stringScratch = new byte[Math.Max(n, _stringScratch.Length * 2)];
            reader.BaseStream.ReadExactly(_stringScratch, 0, n);
            return Encoding.UTF8.GetString(_stringScratch, 0, n);
        }

        private object ReadValue(BinaryReader reader, GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.Uint8: return reader.ReadByte();
                case GgufValueType.Int8: return reader.ReadSByte();
                case GgufValueType.Uint16: return reader.ReadUInt16();
                case GgufValueType.Int16: return reader.ReadInt16();
                case GgufValueType.Uint32: return reader.ReadUInt32();
                case GgufValueType.Int32: return reader.ReadInt32();
                case GgufValueType.Float32: return reader.ReadSingle();
                case GgufValueType.Bool: return reader.ReadByte() != 0;
                case GgufValueType.String: return ReadString(reader);
                case GgufValueType.Uint64: return reader.ReadUInt64();
                case GgufValueType.Int64: return reader.ReadInt64();
                case GgufValueType.Float64: return reader.ReadDouble();
                case GgufValueType.Array: return ReadArray(reader);
                default:
                    throw new NotSupportedException($"Unknown GGUF value type: {type}");
            }
        }

        private object ReadArray(BinaryReader reader)
        {
            var elemType = (GgufValueType)reader.ReadUInt32();
            ulong count = reader.ReadUInt64();

            switch (elemType)
            {
                case GgufValueType.Uint32:
                {
                    var arr = new uint[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt32();
                    return arr;
                }
                case GgufValueType.Int32:
                {
                    var arr = new int[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt32();
                    return arr;
                }
                case GgufValueType.Float32:
                {
                    var arr = new float[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSingle();
                    return arr;
                }
                case GgufValueType.String:
                {
                    var arr = new string[count];
                    for (ulong i = 0; i < count; i++) arr[i] = ReadString(reader);
                    return arr;
                }
                case GgufValueType.Uint8:
                {
                    var arr = new byte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte();
                    return arr;
                }
                case GgufValueType.Int8:
                {
                    var arr = new sbyte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSByte();
                    return arr;
                }
                case GgufValueType.Uint16:
                {
                    var arr = new ushort[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt16();
                    return arr;
                }
                case GgufValueType.Int16:
                {
                    var arr = new short[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt16();
                    return arr;
                }
                case GgufValueType.Uint64:
                {
                    var arr = new ulong[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt64();
                    return arr;
                }
                case GgufValueType.Int64:
                {
                    var arr = new long[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt64();
                    return arr;
                }
                case GgufValueType.Float64:
                {
                    var arr = new double[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadDouble();
                    return arr;
                }
                case GgufValueType.Bool:
                {
                    var arr = new bool[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte() != 0;
                    return arr;
                }
                default:
                    throw new NotSupportedException($"Unknown array element type: {elemType}");
            }
        }

        public unsafe void Dispose()
        {
            foreach (var shard in _shards)
                shard.Dispose();
            _shards.Clear();
            _tensorOwner.Clear();

            if (_lockedBase != null)
            {
                try { _ = munlock(_lockedBase, (nuint)_lockedLength); } catch { }
                _lockedBase = null;
                _lockedLength = 0;
            }
            foreach (var (address, length) in _lockedRanges)
            {
                try { _ = munlock((void*)address, length); } catch { }
            }
            _lockedRanges.Clear();
            if (_mappedPointerAcquired && _mappedView != null)
            {
                _mappedView.SafeMemoryMappedViewHandle.ReleasePointer();
                _mappedPointerAcquired = false;
                _mappedBase = null;
            }

            _mappedView?.Dispose();
            _mappedView = null;
            _mappedFile?.Dispose();
            _mappedFile = null;
            _stream?.Dispose();
            _stream = null!;
        }

        private unsafe void EnsureMappedView()
        {
            if (_mappedBase != null)
                return;

            if (_mappedFile == null)
            {
                // Map from an explicitly read-only, share-read stream: the string-path
                // CreateFromFile overload would open it writable and collide with the
                // other read handles on the same GGUF (loader streams, second instances).
                var fs = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read);
                _mappedFile = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read,
                    HandleInheritability.None, leaveOpen: false);
            }
            _mappedView ??= _mappedFile.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            byte* viewPtr = null;
            _mappedView.SafeMemoryMappedViewHandle.AcquirePointer(ref viewPtr);
            viewPtr += _mappedView.PointerOffset;
            _mappedBase = viewPtr;
            _mappedPointerAcquired = true;
        }
    }
}
