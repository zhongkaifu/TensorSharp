// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// The shards of a (possibly split) GGUF checkpoint as a whole-model direct-CUDA engine loads
    /// them: one tensor table across every shard, weight descriptors that name a shard and an
    /// offset instead of holding bytes (the engine's loader streams them straight into VRAM), and
    /// the small tensors the host itself reads (norms, gates, routers) prefetched in parallel.
    /// </summary>
    internal sealed unsafe class GgufShardSet : IDisposable
    {
        private readonly string _tag;
        private readonly List<GgufFile> _files = new List<GgufFile>();
        private readonly List<string> _paths = new List<string>();
        private readonly Dictionary<GgufFile, int> _indexOf = new Dictionary<GgufFile, int>();
        private readonly Dictionary<string, (GgufFile File, GgufTensorInfo Info)> _tensors
            = new Dictionary<string, (GgufFile, GgufTensorInfo)>(StringComparer.Ordinal);
        private readonly List<IntPtr> _owned = new List<IntPtr>();
        private ShardSource[] _sources;
        private Dictionary<string, IntPtr> _prefetched;

        /// <summary>The checkpoint's first shard, which carries the metadata.</summary>
        public GgufFile First => _files[0];

        /// <summary>A separate GGUF loaded beside the checkpoint (a drafter), or null.</summary>
        public GgufFile Extra { get; }

        /// <param name="tag">Log and error prefix, e.g. "dsv4-cuda".</param>
        /// <param name="extraPath">A separate GGUF whose tensors join the same table (their
        /// names must not collide with the checkpoint's), or null.</param>
        public GgufShardSet(string firstPath, string tag, string extraPath = null)
        {
            _tag = tag;
            // Standalone: this set enumerates the shards itself from split.count, so letting
            // shard 1 also pull in its siblings would build a second, duplicate set of files and
            // attribute every tensor to whichever shard happened to be opened last.
            var first = GgufFile.OpenWithoutSiblingShards(firstPath);
            _files.Add(first);
            _paths.Add(firstPath);

            int splitCount = (int)first.GetUint32("split.count", 1);
            if (splitCount > 1)
            {
                const string marker = "-00001-of-";
                int pos = firstPath.IndexOf(marker, StringComparison.Ordinal);
                if (pos >= 0)
                {
                    for (int i = 2; i <= splitCount; i++)
                    {
                        string path = firstPath.Substring(0, pos) + $"-{i:D5}-of-" + firstPath.Substring(pos + marker.Length);
                        _files.Add(GgufFile.OpenWithoutSiblingShards(path));
                        _paths.Add(path);
                    }
                }
            }

            if (!string.IsNullOrEmpty(extraPath))
            {
                Extra = new GgufFile(extraPath);
                _files.Add(Extra);
                _paths.Add(extraPath);
            }

            // Before anything is sized: a shard cut short by an interrupted download would
            // otherwise fail as a short read well into the upload, with the weight buffers
            // already committed.
            foreach (var file in _files)
                file.ThrowIfTruncated();

            for (int s = 0; s < _files.Count; s++)
            {
                _indexOf[_files[s]] = s;
                foreach (var kv in _files[s].Tensors)
                    _tensors[kv.Key] = (_files[s], kv.Value);
            }
        }

        public bool Has(string name) => _tensors.ContainsKey(name);

        /// <summary>The tensor's header, or null when no shard has it.</summary>
        public GgufTensorInfo InfoOf(string name) => _tensors.TryGetValue(name, out var e) ? e.Info : null;

        /// <summary>Every tensor of the table (row views by their own names) and its byte count, from the
        /// headers alone: what a placement computed before any data is read prices.</summary>
        public IEnumerable<(string Name, GgufTensorInfo Info, long Bytes)> TensorBytes()
        {
            foreach (var kv in _tensors)
                yield return (kv.Key, kv.Value.Info, kv.Value.File.GetTensorByteCount(kv.Value.Info));
        }

        /// <summary>
        /// Replaces the 2-D tensor <paramref name="source"/> in the table with row views of it (the
        /// static overload below). Call it before <see cref="BeginStreaming"/>.
        /// </summary>
        public void ReplaceWithRowViews(string source, IReadOnlyList<(string Name, long FirstRow, long Rows)> views)
            => ReplaceWithRowViews(_tensors, source, views, _tag);

        /// <summary>
        /// Replaces the 2-D tensor <paramref name="source"/> in a per-shard tensor table with row
        /// views: each view is a header for rows [FirstRow, FirstRow + Rows) of the same bytes in
        /// the same shard, so every reader of the table (weight descriptors, mappings, staged
        /// copies) sees that row range as a tensor of its own and no byte is copied. The source
        /// entry is removed, so nothing can also stream, stage or buffer the whole tensor beside
        /// its views. This is how llama.cpp's combined DeepSeek V4.1 Engram table becomes the
        /// per-layer tables the executors read, as the native loader does it
        /// (ggml_ops_deepseek4.cpp).
        /// </summary>
        /// <remarks>
        /// The table's files must each be ONE shard (<see cref="GgufFile.OpenWithoutSiblingShards"/>):
        /// a view's offset is relative to the shard that holds the source. Each view also keeps
        /// the source's <see cref="GgufTensorInfo.Name"/>, because a GgufFile that aggregates its
        /// siblings resolves a tensor's owning shard by that name and would look a view named
        /// after its layer up in the wrong file.
        /// </remarks>
        internal static void ReplaceWithRowViews(Dictionary<string, (GgufFile File, GgufTensorInfo Info)> tensors,
            string source, IReadOnlyList<(string Name, long FirstRow, long Rows)> views, string tag)
        {
            if (!tensors.TryGetValue(source, out var entry))
                throw new InvalidOperationException($"[{tag}] missing tensor: {source}");
            GgufTensorInfo info = entry.Info;
            if (info.Shape.Length < 2 || Array.Exists(info.Shape, d => d == 0) ||
                Array.FindIndex(info.Shape, 2, d => d != 1) >= 0)
                throw new InvalidDataException($"[{tag}] {source} must be a two-dimensional tensor to be viewed by rows");
            ulong tableRows = info.Shape[1];
            long rowBytes = entry.File.GetTensorByteCount(new GgufTensorInfo { Shape = new[] { info.Shape[0] }, Type = info.Type });
            var built = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
            foreach (var (name, firstRow, rows) in views)
            {
                if (firstRow < 0 || rows <= 0 || (ulong)firstRow > tableRows || (ulong)rows > tableRows - (ulong)firstRow)
                    throw new InvalidDataException(
                        $"[{tag}] {name}: rows [{firstRow}, {firstRow + rows}) lie outside {source}'s {tableRows} rows");
                if (name == source || tensors.ContainsKey(name) || built.ContainsKey(name))
                    throw new InvalidDataException($"[{tag}] {name} already exists; cannot view {source} under that name");
                built[name] = new GgufTensorInfo
                {
                    Name = info.Name,
                    Shape = new[] { info.Shape[0], (ulong)rows },
                    Type = info.Type,
                    Offset = info.Offset + (ulong)firstRow * (ulong)rowBytes,
                };
            }
            foreach (var kv in built)
                tensors[kv.Key] = (entry.File, kv.Value);
            tensors.Remove(source);
        }

        /// <summary>
        /// Drop each range the loader reads from the page cache as soon as it has been copied out (Linux, set before
        /// <see cref="BeginStreaming"/>). A weight on its way to VRAM is never read from the file again, and on a host
        /// whose memory cgroup also has to keep a host-mapped table warm, the weights' pages would otherwise push the
        /// table out: on the 1017's 100 GB VMs a node held 36.8 GB of weight cache after every load. Shards on tmpfs
        /// or ramfs are left alone (their pages are the file).
        /// </summary>
        public bool DropPagesAfterRead { get; set; }

        /// <summary>Bytes the loader dropped from the page cache, and the drops the kernel refused (logs, tests).</summary>
        public long DroppedBytes
        {
            get
            {
                if (_loadEnded)
                    return _droppedAtEnd;
                long total = 0;
                foreach (var s in _sources ?? Array.Empty<ShardSource>())
                    total += s.DroppedBytes;
                return total;
            }
        }

        private bool _loadEnded;
        private long _droppedAtEnd;

        /// <summary>
        /// Switch to streaming: large weights are read exactly once, on their way to VRAM, so the
        /// engine streams them from the shards through pinned chunks instead of staging the whole
        /// (hundreds of GB) model in host RAM. The host's own small tensors are read now, in
        /// parallel: individually tiny, but there are hundreds of them and a serial read of each
        /// costs a full round trip on a network filesystem.
        /// </summary>
        /// <param name="prefetch">Which small tensors to read ahead (null: all of them). A stage of a
        /// distributed layer split reads only its own layers'; a tensor left out is still read on demand.</param>
        public void BeginStreaming(Func<string, bool> prefetch = null)
        {
            _sources = new ShardSource[_paths.Count];
            for (int s = 0; s < _paths.Count; s++)
            {
                bool drop = DropPagesAfterRead && OperatingSystem.IsLinux() && !PageCacheHints.IsMemoryBacked(_paths[s]);
                _sources[s] = new ShardSource(_paths[s], _tag, drop);
            }

            var names = new List<string>();
            foreach (var kv in _tensors)
            {
                var type = kv.Value.Info.Type;
                if ((type == GgmlTensorType.F32 || type == GgmlTensorType.I32) && (prefetch == null || prefetch(kv.Key)))
                    names.Add(kv.Key);
            }
            PrefetchedNames = names.ToArray();
            if (names.Count == 0)
                return;
            var slots = new IntPtr[names.Count];
            Parallel.For(0, names.Count, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {
                var entry = _tensors[names[i]];
                long bytes = entry.File.GetTensorByteCount(entry.Info);
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                _sources[_indexOf[entry.File]].Read(entry.File.DataOffset + (long)entry.Info.Offset, buf, bytes);
                slots[i] = buf;
            });
            _prefetched = new Dictionary<string, IntPtr>(names.Count, StringComparer.Ordinal);
            for (int i = 0; i < names.Count; i++)
            {
                _prefetched[names[i]] = slots[i];
                _owned.Add(slots[i]);
            }
        }

        /// <summary>The tensors <see cref="BeginStreaming"/> read ahead (kept after <see cref="EndLoad"/>, names
        /// only); empty before streaming.</summary>
        public IReadOnlyList<string> PrefetchedNames { get; private set; } = Array.Empty<string>();

        /// <summary>The tensor's bytes in host memory (prefetched, read, or mapped), or
        /// (zero, null) when it is absent and not required.</summary>
        public (IntPtr Ptr, GgufTensorInfo Info) Raw(string name, bool required = true)
        {
            if (!_tensors.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[{_tag}] missing tensor: {name}");
                return (IntPtr.Zero, null);
            }

            GgufTensorInfo info = entry.Info;
            if (_prefetched != null && _prefetched.TryGetValue(name, out IntPtr cached))
                return (cached, info);

            long bytes = entry.File.GetTensorByteCount(info);
            if (_sources != null)
            {
                IntPtr staged = Marshal.AllocHGlobal((nint)bytes);
                _owned.Add(staged);
                _sources[_indexOf[entry.File]].Read(entry.File.DataOffset + (long)info.Offset, staged, bytes);
                return (staged, info);
            }

            if (entry.File.TryGetTensorDataPointer(info, out IntPtr mapped))
                return (mapped, info);

            IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
            _owned.Add(buf);
            entry.File.ReadTensorDataToNative(info, buf, bytes);
            return (buf, info);
        }

        /// <summary>
        /// Describes a bulk weight to the engine. While streaming this touches no tensor data: it
        /// hands over the shard and the file offset, and the engine's loader pool moves the bytes.
        /// </summary>
        public CudaWeightDesc Weight(string name, bool required = true)
        {
            if (!_tensors.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[{_tag}] missing tensor: {name}");
                return default;
            }

            GgufTensorInfo info = entry.Info;
            var desc = new CudaWeightDesc
            {
                GgmlType = (int)info.Type,
                Ne0 = (int)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (int)info.Shape[1] : 1,
                Ne2 = info.Shape.Length > 2 ? (int)info.Shape[2] : 1,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Name = name,
            };
            if (_sources != null)
            {
                desc.Source = _sources[_indexOf[entry.File]];
                desc.SourceOffset = entry.File.DataOffset + (long)info.Offset;
            }
            else
            {
                var (ptr, _) = Raw(name, required);
                if (ptr == IntPtr.Zero)
                    return default;
                desc.HostPtr = ptr;
            }
            return desc;
        }

        /// <summary>A tensor of any storable type dequantized to F32 on the host; null when absent
        /// and not required.</summary>
        public float[] Floats(string name, bool required = true)
        {
            var (ptr, info) = Raw(name, required);
            if (ptr == IntPtr.Zero)
                return null;
            long n = info.NumElements;
            var arr = new float[n];
            ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, ptr, arr, 0, n);
            return arr;
        }

        public int[] Ints(string name)
        {
            var (ptr, info) = Raw(name);
            if (info.Type != GgmlTensorType.I32)
                throw new InvalidOperationException($"[{_tag}] {name}: expected I32, got {info.Type}");
            long n = info.NumElements;
            var arr = new int[n];
            Marshal.Copy(ptr, arr, 0, (int)n);
            return arr;
        }

        /// <summary>
        /// A tensor mapped read-only for the engine's lifetime (tables read row by row, far too
        /// large to stage): its shard's path and byte range, and the mapping. Throws when the
        /// platform cannot map rather than silently allocating RAM the box does not have.
        /// </summary>
        public (string Path, long Offset, long Bytes, IntPtr Mapped) MapTensor(string name)
        {
            if (!_tensors.TryGetValue(name, out var entry))
                throw new InvalidOperationException($"[{_tag}] missing tensor: {name}");
            GgufTensorInfo info = entry.Info;
            long bytes = entry.File.GetTensorByteCount(info);
            long offset = entry.File.DataOffset + (long)info.Offset;
            IntPtr mapped = IntPtr.Zero;
            if (_sources != null)
                _sources[_indexOf[entry.File]].TryMapRange(offset, bytes, out mapped);
            if (mapped == IntPtr.Zero && !entry.File.TryGetTensorDataPointer(info, out mapped))
                throw new InvalidOperationException(
                    $"[{_tag}] cannot memory-map {name} ({bytes / (1024.0 * 1024 * 1024):F1} GiB).");
            return (_paths[_indexOf[entry.File]], offset, bytes, mapped);
        }

        /// <summary>
        /// Post-load cleanup: everything lives in VRAM now, so the host copies go, and so do the
        /// per-thread read handles, which only serve the load. A source whose mapping the engine
        /// borrowed stays alive until <see cref="Dispose"/>.
        /// </summary>
        public void EndLoad()
        {
            _prefetched = null;
            foreach (var p in _owned)
                Marshal.FreeHGlobal(p);
            _owned.Clear();
            if (_sources == null)
                return;
            bool anyMapped = false;
            long dropped = 0;
            int failures = 0, dropping = 0;
            foreach (var s in _sources)
            {
                s.DisposeReaders();
                anyMapped |= s.HasMapping;
                dropped += s.DroppedBytes;
                failures += s.DropFailures;
                dropping += s.Drops ? 1 : 0;
            }
            if (dropping > 0)
                Console.Error.WriteLine($"[{_tag}] dropped {dropped / (double)(1L << 30):F1} GiB of weight pages from the page cache " +
                    $"after reading them ({dropping} of {_sources.Length} shards" +
                    (failures > 0 ? $"; the kernel refused {failures} drops" : string.Empty) + "; TS_DSV4_LOAD_DROP_CACHE=0 keeps them)");
            else if (DropPagesAfterRead && OperatingSystem.IsLinux())
                Console.Error.WriteLine($"[{_tag}] weight pages kept in the page cache: every shard is on tmpfs or ramfs");
            _droppedAtEnd = dropped;
            _loadEnded = true;
            if (!anyMapped)
                _sources = null;
        }

        public void Dispose()
        {
            if (_sources != null)
            {
                foreach (var s in _sources)
                    s?.Dispose();
                _sources = null;
            }
            foreach (var p in _owned)
                Marshal.FreeHGlobal(p);
            _owned.Clear();
            foreach (var f in _files)
                f.Dispose();
            _files.Clear();
        }

        /// <summary>
        /// Positional-read view over one GGUF shard, handed to the CUDA engine so big weights go
        /// file -> pinned chunk -> VRAM without a host-RAM copy of the whole model.
        /// </summary>
        /// <remarks>
        /// One descriptor per reader thread, deliberately. pread(2) on a shared handle is correct
        /// but on FUSE filesystems it is also serialized: on MooseFS, 16 threads sharing one
        /// descriptor read at 0.69 GB/s while the same 16 threads on their own descriptors read at
        /// 2.4 GB/s, independent of the access pattern.
        /// </remarks>
        private sealed class ShardSource : ICudaMappedWeightSource, IDisposable
        {
            private readonly string _path;
            private readonly string _tag;
            private readonly ThreadLocal<Microsoft.Win32.SafeHandles.SafeFileHandle> _handles;

            // Whole-file read-only mapping, created lazily by TryMapRange. The engine borrows raw
            // pointers into it for as long as it lives, so the mapping survives DisposeReaders
            // (the post-load cleanup) and only Dispose -- called after the engine is gone --
            // releases it.
            private readonly object _mapLock = new object();
            private System.IO.MemoryMappedFiles.MemoryMappedFile _map;
            private System.IO.MemoryMappedFiles.MemoryMappedViewAccessor _view;
            private byte* _mapBase;
            private long _mapLength;

            private readonly bool _drop;
            private long _dropped;
            private int _dropFailures;

            public ShardSource(string path, string tag, bool dropPagesAfterRead = false)
            {
                _path = path;
                _tag = tag;
                _drop = dropPagesAfterRead;
                _handles = new ThreadLocal<Microsoft.Win32.SafeHandles.SafeFileHandle>(
                    () => File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read),
                    trackAllValues: true);
            }

            public bool Drops => _drop;

            public long DroppedBytes => Interlocked.Read(ref _dropped);

            public int DropFailures => Volatile.Read(ref _dropFailures);

            public void Read(long offset, IntPtr dst, long bytes)
            {
                var handle = _handles.Value;
                byte* p = (byte*)dst;
                long done = 0;
                while (done < bytes)
                {
                    int want = (int)Math.Min(1 << 30, bytes - done);
                    int got = RandomAccess.Read(handle, new Span<byte>(p + done, want), offset + done);
                    if (got <= 0)
                        throw new IOException($"[{_tag}] short read at offset {offset + done} in {_path}");
                    done += got;
                }
                // The bytes are in the caller's buffer now (a pinned chunk on its way to VRAM, or a small tensor the
                // host keeps): this process never reads these pages from the file again.
                if (_drop)
                {
                    if (PageCacheHints.DropFileRange(handle, offset, bytes, out long whole) == 0)
                        Interlocked.Add(ref _dropped, whole);
                    else
                        Interlocked.Increment(ref _dropFailures);
                }
            }

            public bool HasMapping => _mapBase != null;

            public bool TryMapRange(long offset, long bytes, out IntPtr ptr)
            {
                ptr = IntPtr.Zero;
                if (offset < 0 || bytes <= 0)
                    return false;
                lock (_mapLock)
                {
                    if (_mapBase == null)
                    {
                        try
                        {
                            long length = new FileInfo(_path).Length;
                            var map = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
                                _path, FileMode.Open, mapName: null, 0,
                                System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
                            var view = map.CreateViewAccessor(0, 0,
                                System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
                            byte* b = null;
                            view.SafeMemoryMappedViewHandle.AcquirePointer(ref b);
                            _map = map;
                            _view = view;
                            _mapBase = b;
                            _mapLength = length;
                        }
                        catch
                        {
                            return false; // caller copies instead
                        }
                    }
                    if (offset + bytes > _mapLength)
                        return false;
                    ptr = (IntPtr)(_mapBase + offset);
                    return true;
                }
            }

            /// <summary>Drops the per-thread read handles (only needed while the loader streams
            /// weights) but keeps any mapping the engine borrowed pointers into.</summary>
            public void DisposeReaders()
            {
                foreach (var h in _handles.Values)
                    h?.Dispose();
                _handles.Dispose();
            }

            public void Dispose()
            {
                try { DisposeReaders(); } catch (ObjectDisposedException) { }
                lock (_mapLock)
                {
                    if (_mapBase != null)
                    {
                        _view.SafeMemoryMappedViewHandle.ReleasePointer();
                        _view.Dispose();
                        _map.Dispose();
                        _view = null;
                        _map = null;
                        _mapBase = null;
                        _mapLength = 0;
                    }
                }
            }
        }
    }
}
