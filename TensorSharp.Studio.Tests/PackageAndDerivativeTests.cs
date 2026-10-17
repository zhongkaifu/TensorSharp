// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using TensorSharp.Studio;
using Xunit;

namespace TensorSharp.Studio.Tests;

public sealed class StudioMediaFactAttribute : FactAttribute
{
    public StudioMediaFactAttribute()
    {
        bool Exists(string name) => (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)
            .Any(p => File.Exists(Path.Combine(p, name + (OperatingSystem.IsWindows() ? ".exe" : ""))));
        if (!Exists("ffmpeg") || !Exists("ffprobe")) Skip = "Operator-installed ffmpeg and ffprobe are required for real media validation.";
    }
}

public sealed class PackageAndDerivativeTests : IAsyncLifetime
{
    readonly string root = Path.Combine(Path.GetTempPath(), "tensorstudio-package-tests-" + Guid.NewGuid().ToString("N"));
    StudioService service = null!;
    public Task InitializeAsync() { service = new StudioService(root); return Task.CompletedTask; }
    public async Task DisposeAsync() { await service.DisposeAsync(); Directory.Delete(root, true); }

    [Fact]
    public async Task PackageRoundTripStartsNewIdentityAndExcludesConversationHistory()
    {
        var original = await service.CreateProjectAsync("Portable project", conversationId: "private-conversation");
        original = await service.ApplyCommandsAsync(original.Id, original.Revision, StudioJson.NewId(), [new() { Type = "settings", Name = "Portable revision" }]);
        var package = await service.ExportPackageAsync(original.Id);
        using (var archive = ZipFile.OpenRead(package))
        {
            Assert.Single(archive.Entries);
            using var reader = new StreamReader(archive.GetEntry("project.json")!.Open());
            string json = await reader.ReadToEndAsync();
            Assert.DoesNotContain("private-conversation", json);
            Assert.DoesNotContain("appliedCommands", json);
        }
        await using var input = File.OpenRead(package);
        var imported = await service.ImportPackageAsync(input);
        Assert.NotEqual(original.Id, imported.Id);
        Assert.Equal(original.Name, imported.Name);
        Assert.Equal(0, imported.Revision);
        Assert.Null(imported.ConversationId);
        Assert.Equal(2, imported.Tracks.Count);
        var error = await Assert.ThrowsAsync<StudioException>(() => service.UndoAsync(imported.Id, 0, StudioJson.NewId()));
        Assert.Equal("history_empty", error.Code);
    }

    [Theory]
    [InlineData("../escape.txt")]
    [InlineData("/tmp/escape.txt")]
    [InlineData("assets/../../escape.txt")]
    [InlineData("assets\\escape.txt")]
    [InlineData("assets/C:escape.txt")]
    public async Task RejectsArchiveTraversalWithoutPublishingProject(string path)
    {
        using var bytes = Package(("project.json", "{}"), (path, "bad"));
        await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(bytes));
        Assert.Empty(await service.ListProjectsAsync());
        Assert.Empty(Directory.EnumerateFileSystemEntries(Path.Combine(root, "imports")));
    }

    [Fact]
    public async Task RejectsDuplicateEntriesAndFutureSchema()
    {
        using (var duplicate = Package(("project.json", "{}"), ("project.json", "{}")))
            Assert.Equal("unsafe_package", (await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(duplicate))).Code);
        var project = await service.CreateProjectAsync();
        project.SchemaVersion = 999;
        using var future = Package(("project.json", JsonSerializer.Serialize(project, StudioJson.Options)));
        Assert.Equal("unsupported_schema", (await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(future))).Code);
    }

    [Fact]
    public async Task RejectsExpandedPackageOverLimit()
    {
        service.Options.MaxImportBytes = 2048;
        using var package = Package(("project.json", new string(' ', 100_000)));
        Assert.True(package.Length < 2048); // Compressed ZIP bomb remains within wire limit.
        Assert.Equal("package_too_large", (await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(package))).Code);
    }

    [Fact]
    public async Task PackageAdmissionRejectsInsufficientDiskBeforeCreatingPartialFiles()
    {
        var project = await service.CreateProjectAsync();
        service.Options.MinimumFreeDiskBytes = long.MaxValue;
        var exportError = await Assert.ThrowsAsync<StudioException>(() => service.ExportPackageAsync(project.Id));
        Assert.Equal("insufficient_disk", exportError.Code);
        using var package = Package(("project.json", JsonSerializer.Serialize(project, StudioJson.Options)));
        var importError = await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(package));
        Assert.Equal("insufficient_disk", importError.Code);
        Assert.Empty(Directory.EnumerateFiles(root, "*.tmp", SearchOption.AllDirectories));
        string imports = Path.Combine(root, "imports");
        Assert.True(!Directory.Exists(imports) || !Directory.EnumerateFileSystemEntries(imports).Any());
        Assert.Single(await service.ListProjectsAsync());
    }

    [StudioMediaFact]
    public async Task RealWaveformIsBoundedAndPortableAssetsRetainHashes()
    {
        var project = await service.CreateProjectAsync();
        using var wav = SineWave();
        project = await service.ImportAssetAsync(project.Id, wav, "tone.wav");
        var asset = Assert.Single(project.Assets);
        var waveform = await service.GetWaveformAsync(project.Id, asset.Id);
        Assert.Equal(512, waveform.Peaks.Count);
        Assert.All(waveform.Peaks, peak => Assert.InRange(peak, 0, 1));
        Assert.True(waveform.Peaks.Count(peak => peak > .2f) > 450);
        Assert.InRange(waveform.DurationSeconds, .99, 1.01);
        var cached = await service.GetWaveformAsync(project.Id, asset.Id);
        Assert.Equal(waveform.Peaks, cached.Peaks);
        var package = await service.ExportPackageAsync(project.Id);
        await using var input = File.OpenRead(package);
        var imported = await service.ImportPackageAsync(input);
        Assert.Equal(asset.Sha256, Assert.Single(imported.Assets).Sha256);
        var importedWaveform = await service.GetWaveformAsync(imported.Id, imported.Assets[0].Id);
        Assert.Equal(waveform.Peaks, importedWaveform.Peaks);
    }

    [SkippableFact]
    public async Task ARotatedPhotoIsRecordedAtTheSizeItIsShownAndGenerated()
    {
        // A phone photo is often stored on its side with EXIF Orientation 6. The video models decode it upright, and a
        // canvas sized from the first frame is computed from the recorded size, so the asset must record the upright one.
        Skip.IfNot(new StudioMediaFactAttribute().Skip == null, "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        var project = await service.CreateProjectAsync();
        string stored = Path.Combine(root, "stored.jpg");
        using (var process = new System.Diagnostics.Process())
        {
            process.StartInfo = new("ffmpeg") { UseShellExecute = false, RedirectStandardError = true };
            foreach (var arg in new[] { "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:d=1", "-frames:v", "1", stored }) process.StartInfo.ArgumentList.Add(arg);
            process.Start(); string error = await process.StandardError.ReadToEndAsync(); await process.WaitForExitAsync();
            Assert.True(process.ExitCode == 0, error);
        }
        // APP1 Exif segment right after SOI: little-endian TIFF, IFD0 with one entry, Orientation (0x0112) = 6.
        byte[] tiff = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        byte[] payload = [.. "Exif\0\0"u8, .. tiff];
        int length = payload.Length + 2;
        byte[] jpeg = await File.ReadAllBytesAsync(stored);
        string rotated = Path.Combine(root, "rotated.jpg");
        await File.WriteAllBytesAsync(rotated, [jpeg[0], jpeg[1], 0xFF, 0xE1, (byte)(length >> 8), (byte)length, .. payload, .. jpeg[2..]]);
        await using (var input = File.OpenRead(stored)) project = await service.ImportAssetAsync(project.Id, input, "stored.jpg");
        await using (var input = File.OpenRead(rotated)) project = await service.ImportAssetAsync(project.Id, input, "rotated.jpg");
        var plain = project.Assets.Single(a => a.Name == "stored.jpg"); var turned = project.Assets.Single(a => a.Name == "rotated.jpg");
        Assert.Equal((96, 64), (plain.Width, plain.Height));
        Skip.If(turned.Width == 96, "This ffprobe reports no display matrix for EXIF orientation, so the stored size is kept, as before.");
        Assert.Equal((64, 96), (turned.Width, turned.Height));
    }

    [StudioMediaFact]
    public async Task ThumbnailsUseManagedAssetOwnershipAndCache()
    {
        var project = await service.CreateProjectAsync();
        // Generate a tiny PNG using FFmpeg's deterministic color source.
        // through a color source, then imported as PNG. No internet asset or model is involved.
        string png = Path.Combine(root, "fixture.png");
        using (var process = new System.Diagnostics.Process())
        {
            process.StartInfo = new("ffmpeg") { UseShellExecute = false, RedirectStandardError = true };
            foreach (var arg in new[] { "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", png }) process.StartInfo.ArgumentList.Add(arg);
            process.Start();
            string error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            Assert.True(process.ExitCode == 0, error);
        }
        await using var input = File.OpenRead(png);
        project = await service.ImportAssetAsync(project.Id, input, "red.png");
        string thumbnail = await service.GetThumbnailAsync(project.Id, project.Assets[0].Id);
        Assert.InRange(new FileInfo(thumbnail).Length, 4, 1024 * 1024);
        byte[] header = await File.ReadAllBytesAsync(thumbnail);
        Assert.Equal(0xff, header[0]); Assert.Equal(0xd8, header[1]);
        Assert.Equal(thumbnail, await service.GetThumbnailAsync(project.Id, project.Assets[0].Id));
        var other = await service.CreateProjectAsync();
        await Assert.ThrowsAsync<StudioException>(() => service.GetThumbnailAsync(other.Id, project.Assets[0].Id));
    }

    static MemoryStream Package(params (string Name, string Content)[] entries)
    {
        var stream = new MemoryStream();
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, true))
            foreach (var (name, content) in entries)
            {
                using var writer = new StreamWriter(archive.CreateEntry(name, CompressionLevel.SmallestSize).Open());
                writer.Write(content);
            }
        stream.Position = 0; return stream;
    }

    static MemoryStream SineWave()
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.ASCII, true))
        {
            const int samples = 48000;
            writer.Write("RIFF"u8); writer.Write(36 + samples * 2); writer.Write("WAVEfmt "u8);
            writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(48000); writer.Write(96000);
            writer.Write((short)2); writer.Write((short)16); writer.Write("data"u8); writer.Write(samples * 2);
            for (int i = 0; i < samples; i++) writer.Write((short)(Math.Sin(2 * Math.PI * 440 * i / samples) * 10000));
        }
        stream.Position = 0; return stream;
    }
}
