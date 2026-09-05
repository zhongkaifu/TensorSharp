using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Enforces the operator-configured storage limits on the upload directory:
    /// a per-file cap on client-originated writes (<c>--upload-max-mb</c>), a
    /// total directory budget (<c>--upload-quota-mb</c>), and a file age limit
    /// (<c>--upload-ttl-hours</c>). Usage is an in-memory tally seeded by one
    /// directory scan at construction; admission reserves bytes up front so
    /// concurrent writes cannot overshoot the quota, and the TTL sweep returns
    /// what it deletes. Files removed behind the server's back are only
    /// reconciled at the next restart.
    /// </summary>
    public sealed class UploadStoragePolicy
    {
        /// <summary>Default per-file cap; equals the Kestrel request-body limit, so it changes nothing until lowered.</summary>
        public const long DefaultMaxFileBytes = 500L * 1024 * 1024;

        private long _usedBytes;

        public UploadStoragePolicy(
            string uploadDirectory,
            long maxFileBytes = DefaultMaxFileBytes,
            long quotaBytes = 0,
            TimeSpan? ttl = null)
        {
            if (string.IsNullOrEmpty(uploadDirectory)) throw new ArgumentNullException(nameof(uploadDirectory));
            DirectoryPath = uploadDirectory;
            MaxFileBytes = maxFileBytes;
            QuotaBytes = quotaBytes;
            Ttl = ttl;
            _usedBytes = ScanDirectory(uploadDirectory);
        }

        /// <summary>Absolute path of the upload directory this policy governs.</summary>
        public string DirectoryPath { get; }

        /// <summary>Largest single client-originated file accepted, in bytes.</summary>
        public long MaxFileBytes { get; }

        /// <summary>Total budget for the directory in bytes; 0 disables the quota.</summary>
        public long QuotaBytes { get; }

        /// <summary>Age after which the cleanup sweep deletes a file; null disables TTL cleanup.</summary>
        public TimeSpan? Ttl { get; }

        public bool QuotaEnabled => QuotaBytes > 0;

        /// <summary>Bytes currently attributed to the directory (startup scan + reservations - deletions).</summary>
        public long UsedBytes => Interlocked.Read(ref _usedBytes);

        /// <summary>
        /// Admit a client-originated write of known size (a multipart upload or
        /// a decoded base64 attachment) and reserve its bytes against the
        /// quota. On failure <paramref name="statusCode"/> is 413 (file over
        /// the per-file cap) or 507 (directory quota exhausted). A successful
        /// reservation must be paired with <see cref="Release"/> if the write
        /// subsequently fails.
        /// </summary>
        public bool TryReserveClientWrite(long bytes, out string error, out int statusCode)
        {
            if (bytes > MaxFileBytes)
            {
                error = string.Format(CultureInfo.InvariantCulture,
                    "File is too large ({0:0.#} MB); this server accepts at most {1:0.#} MB per file.",
                    bytes / (1024.0 * 1024.0), MaxFileBytes / (1024.0 * 1024.0));
                statusCode = 413;
                return false;
            }

            if (!TryReserve(bytes))
            {
                error = QuotaExhaustedMessage;
                statusCode = 507;
                return false;
            }

            error = null;
            statusCode = 0;
            return true;
        }

        /// <summary>
        /// <see cref="TryReserveClientWrite"/> as a throw, for the base64
        /// materialisation paths inside the chat parsers where a status code
        /// cannot be returned; adapters translate the exception into their
        /// protocol's error shape.
        /// </summary>
        public void ReserveClientWriteOrThrow(long bytes)
        {
            if (!TryReserveClientWrite(bytes, out string error, out int statusCode))
                throw new UploadLimitExceededException(error, statusCode);
        }

        /// <summary>Return a reservation whose write failed (or whose file was removed).</summary>
        public void Release(long bytes) => Interlocked.Add(ref _usedBytes, -bytes);

        /// <summary>
        /// Gate for requests that generate output files of unknown size (image
        /// edits, videos). Checked at request start so a full directory fails
        /// in milliseconds instead of after minutes of GPU work; the produced
        /// bytes are counted afterwards via <see cref="RecordFile"/>, so a
        /// request admitted with little headroom can finish slightly over
        /// budget rather than being killed mid-generation.
        /// </summary>
        public bool HasQuotaHeadroom(out string error)
        {
            if (QuotaEnabled && Interlocked.Read(ref _usedBytes) >= QuotaBytes)
            {
                error = QuotaExhaustedMessage;
                return false;
            }
            error = null;
            return true;
        }

        /// <summary>Count a file written outside the reservation path (generated output, extracted frame). Missing paths are ignored.</summary>
        public void RecordFile(string path)
        {
            var info = new FileInfo(path);
            if (info.Exists)
                Interlocked.Add(ref _usedBytes, info.Length);
        }

        public void RecordFiles(IEnumerable<string> paths)
        {
            foreach (string path in paths)
                RecordFile(path);
        }

        /// <summary>
        /// Delete files whose last write is older than <see cref="Ttl"/>.
        /// Files that cannot be deleted (in use, permissions) are left for the
        /// next sweep. No-op when no TTL is configured. Returns the number of
        /// files deleted; <paramref name="freedBytes"/> reports their total size.
        /// </summary>
        public int CleanupExpired(out long freedBytes)
        {
            freedBytes = 0;
            if (Ttl is not TimeSpan ttl)
                return 0;

            DateTime cutoffUtc = DateTime.UtcNow - ttl;
            int deleted = 0;
            foreach (string path in Directory.EnumerateFiles(DirectoryPath))
            {
                try
                {
                    var info = new FileInfo(path);
                    if (!info.Exists || info.LastWriteTimeUtc >= cutoffUtc)
                        continue;
                    long length = info.Length;
                    info.Delete();
                    Interlocked.Add(ref _usedBytes, -length);
                    freedBytes += length;
                    deleted++;
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Locked or in use; retried on the next sweep.
                }
            }
            return deleted;
        }

        private bool TryReserve(long bytes)
        {
            if (!QuotaEnabled)
            {
                Interlocked.Add(ref _usedBytes, bytes);
                return true;
            }

            while (true)
            {
                long used = Interlocked.Read(ref _usedBytes);
                if (used + bytes > QuotaBytes)
                    return false;
                if (Interlocked.CompareExchange(ref _usedBytes, used + bytes, used) == used)
                    return true;
            }
        }

        // Deliberately does not disclose the configured budget: the quota is
        // operator configuration a client cannot act on, and its size is a
        // config detail unauthenticated clients have no need to learn. The
        // number is in the startup "Upload storage limits" log line. The
        // per-file cap IS stated in its 413 message, because that one is
        // client-actionable (shrink the file and retry).
        private const string QuotaExhaustedMessage =
            "Upload storage quota exceeded; the server cannot accept more uploads until space is freed.";

        private static long ScanDirectory(string directory)
        {
            long total = 0;
            if (!Directory.Exists(directory))
                return 0;
            // Every server write site targets the directory root, so the scan
            // is deliberately non-recursive.
            foreach (string path in Directory.EnumerateFiles(directory))
            {
                try { total += new FileInfo(path).Length; }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            }
            return total;
        }
    }

    /// <summary>
    /// Thrown by <see cref="UploadStoragePolicy.ReserveClientWriteOrThrow"/>
    /// when a decoded attachment violates an upload limit. Carries the HTTP
    /// status (413 or 507) the adapter should answer with.
    /// </summary>
    internal sealed class UploadLimitExceededException : Exception
    {
        public UploadLimitExceededException(string message, int statusCode) : base(message)
        {
            StatusCode = statusCode;
        }

        public int StatusCode { get; }
    }
}
