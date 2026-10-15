// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using Microsoft.Win32.SafeHandles;
using System.Security.Cryptography;

namespace TensorSharp.Memory;

/// <summary>One handle shared by all regions in a shard. No mmap, prefault, full-file
/// read or tensor-sized temporary. Keep the file immutable and this source alive until
/// every registered region is unregistered.</summary>
public sealed class FileDataSource : IDisposable
{
    private readonly SafeFileHandle _handle;
    public FileDataSource(string path) : this(path, FileShare.Read) { }
    internal FileDataSource(string path, FileShare sharing)
    {
        _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, sharing,
            FileOptions.Asynchronous | FileOptions.RandomAccess);
        try { ByteLength = RandomAccess.GetLength(_handle); }
        catch { _handle.Dispose(); throw; }
    }
    public long ByteLength { get; }
    internal async ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken)
    {
        MemoryRange.Check(ByteLength, offset, destination.Length);
        while (!destination.IsEmpty)
        {
            int read = await RandomAccess.ReadAsync(_handle, destination, offset, cancellationToken).ConfigureAwait(false);
            if (read == 0) throw new EndOfStreamException("Backing file was truncated.");
            destination = destination[read..];
            offset += read;
        }
    }
    public void Dispose() => _handle.Dispose();
}

public sealed class FileRegionSource : IResourceSource
{
    private readonly FileDataSource _file;
    private readonly long _offset;
    public FileRegionSource(FileDataSource file, long offset, long byteLength)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (offset < 0 || byteLength <= 0 || offset > file.ByteLength || byteLength > file.ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
        _file = file;
        _offset = offset;
        ByteLength = byteLength;
    }
    public long ByteLength { get; }
    public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    {
        MemoryRange.Check(ByteLength, offset, destination.Length);
        return _file.ReadAsync(checked(_offset + offset), destination, cancellationToken);
    }
}

/// <summary>Lossless, process-private spill storage for mutable state. Quota includes
/// a new snapshot while the old copy remains valid. A partial/cancelled write is never
/// published. Buffered filesystem I/O may populate the OS page cache; this is not a
/// hard process-RSS or cgroup-memory limiter.</summary>
public sealed class SsdSpillStore : IDisposable
{
    private readonly MemoryBudget _budget;
    private readonly string _pool;
    private readonly BoundedTransfers _transfers;
    private readonly string _directory;
    private readonly object _gate = new();
    private int _files;
    private bool _disposed;

    public SsdSpillStore(MemoryBudget budget, string pool, string rootDirectory, BoundedTransfers transfers)
    {
        _budget = budget;
        _pool = pool;
        _transfers = transfers;
        _directory = Path.Combine(Path.GetFullPath(rootDirectory), "ts-memory-" + Guid.NewGuid().ToString("N"));
        if (OperatingSystem.IsWindows()) Directory.CreateDirectory(_directory);
        else Directory.CreateDirectory(_directory, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
    }

    internal async ValueTask<Snapshot> WriteAsync(IResourceSource source, CancellationToken cancellationToken,
        BudgetReservation? allocationOwner = null)
    {
        lock (_gate) { ObjectDisposedException.ThrowIf(_disposed, this); _files++; }
        string path = Path.Combine(_directory, Guid.NewGuid().ToString("N") + ".pending");
        BudgetReservation? reservation = null;
        try
        {
            var charges = new[] { new MemoryCharge(_pool, MemoryRange.Align(source.ByteLength, 4096)) };
            reservation = allocationOwner == null ? _budget.Reserve(charges)
                : _budget.TryReserveFollowing(allocationOwner, charges)
                    ?? throw new MemoryPressureException("The spill does not fit its owner's reserved SSD budget.");
            using (var handle = File.OpenHandle(path, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                FileOptions.Asynchronous | FileOptions.SequentialScan))
            {
                using var slot = await _transfers.RentAsync(cancellationToken).ConfigureAwait(false);
                using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                for (long offset = 0; offset < source.ByteLength;)
                {
                    int count = (int)Math.Min(slot.Memory.Length, source.ByteLength - offset);
                    var memory = slot.Memory[..count];
                    await source.ReadAsync(offset, memory, cancellationToken).ConfigureAwait(false);
                    hash.AppendData(memory.Span);
                    await RandomAccess.WriteAsync(handle, memory, offset, cancellationToken).ConfigureAwait(false);
                    offset += count;
                }
                cancellationToken.ThrowIfCancellationRequested();
                // This process-private swap file is not a restart checkpoint.
                // Awaited writes make all bytes visible to the reader; forcing
                // durable media synchronization for every evicted page stalls
                // decode without providing a recoverable session after a crash.
                // SafeFileHandle must close before opening the committed reader on Windows.
                handle.Dispose();
                string committedPath = Path.ChangeExtension(path, ".bin");
                File.Move(path, committedPath);
                path = committedPath;
                var file = new FileDataSource(path, FileShare.Read | FileShare.Delete);
                reservation.Commit();
                return new Snapshot(this, file, path, source.ByteLength, Convert.ToHexString(hash.GetHashAndReset()), reservation);
            }
        }
        catch
        {
            // Release accounting only after disk bytes have actually been removed.
            // If deletion itself fails, retain the charge (fail closed).
            File.Delete(path);
            reservation?.Dispose();
            lock (_gate) _files--;
            throw;
        }
    }

    internal sealed class Snapshot(SsdSpillStore owner, FileDataSource file, string path, long byteLength,
        string sha256, BudgetReservation reservation) : IResourceSource, IChecksummedSource, IDisposable
    {
        private bool _disposed;
        public long ByteLength => byteLength;
        public string Sha256 => sha256;
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return file.ReadAsync(offset, destination, cancellationToken);
        }
        public void Dispose()
        {
            if (_disposed) return;
            // Delete while the read handle is still valid. A deletion failure must
            // leave a usable authoritative snapshot, not a closed source that the
            // residency directory could mistake for recoverable state.
            File.Delete(path);
            file.Dispose();
            reservation.Dispose();
            _disposed = true;
            lock (owner._gate) owner._files--;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            if (_files != 0) throw new InvalidOperationException("Unregister every spilled resource before disposing the store.");
            Directory.Delete(_directory);
            _disposed = true;
        }
    }
}
