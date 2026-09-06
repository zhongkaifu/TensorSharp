using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Periodic sweep behind <c>--upload-ttl-hours</c>: deletes upload-directory
    /// files older than the TTL via <see cref="UploadStoragePolicy.CleanupExpired"/>.
    /// Registered only when a TTL is configured. The sweep interval is TTL/4
    /// clamped to [1, 15] minutes, so short TTLs stay timely without a hot loop;
    /// one sweep also runs at startup to clear files that expired while the
    /// server was down.
    /// </summary>
    public sealed class UploadCleanupService : BackgroundService
    {
        private readonly UploadStoragePolicy _uploads;
        private readonly ILogger _logger;

        public UploadCleanupService(UploadStoragePolicy uploads, ILoggerFactory loggerFactory)
        {
            _uploads = uploads ?? throw new ArgumentNullException(nameof(uploads));
            _logger = loggerFactory.CreateLogger("TensorSharp.Server.UploadCleanup");
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (_uploads.Ttl is not TimeSpan ttl)
                return;

            var period = TimeSpan.FromMinutes(Math.Clamp(ttl.TotalMinutes / 4, 1, 15));
            using var timer = new PeriodicTimer(period);
            try
            {
                do
                {
                    int deleted = _uploads.CleanupExpired(out long freedBytes);
                    if (deleted > 0)
                    {
                        _logger.LogInformation(LogEventIds.UploadCleanup,
                            "Upload TTL sweep: deleted {Deleted} file(s) older than {TtlHours:0.##} h, freed {FreedMB} MB ({UsedMB} MB in use)",
                            deleted, ttl.TotalHours, freedBytes / (1024 * 1024), _uploads.UsedBytes / (1024 * 1024));
                    }
                }
                while (await timer.WaitForNextTickAsync(stoppingToken));
            }
            catch (OperationCanceledException)
            {
                // Host shutdown.
            }
        }
    }
}
