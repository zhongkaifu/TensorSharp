using System.Text.Json;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioRecoveryTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-recovery-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    static void Write(string path, string text = "partial") { Directory.CreateDirectory(Path.GetDirectoryName(path)!); File.WriteAllText(path, text); }

    [Fact] public async Task RestartCleansOnlyManagedScratchAndPreservesCommittedFiles()
    {
        string projectId;
        await using (var service = new StudioService(root)) projectId = (await service.CreateProjectAsync()).Id;
        var projectDirectory = Path.Combine(root, "projects", projectId);
        var projectTemp = Path.Combine(projectDirectory, "project.json." + StudioJson.NewId() + ".tmp"); Write(projectTemp);
        var keepAsset = Path.Combine(projectDirectory, "assets", StudioJson.NewId() + ".mp4"); Write(keepAsset, "source-asset");
        var keepProjectNote = Path.Combine(projectDirectory, "project.json.notes.tmp"); Write(keepProjectNote);
        var package = Path.Combine(root, "packages", projectId + "-0.tsstudio"); Write(package, "completed-package");
        var packageTemp = package + "." + StudioJson.NewId() + ".tmp"; Write(packageTemp);
        var keepPackageNote = Path.Combine(root, "packages", "notes.tmp"); Write(keepPackageNote);
        var stageId = StudioJson.NewId(); var stageFile = Path.Combine(root, "imports", stageId, "assets", "partial.mp4"); Write(stageFile);
        var archive = Path.Combine(root, "imports", stageId + ".zip"); Write(archive);
        var keepImport = Path.Combine(root, "imports", "manual-notes", "keep.txt"); Write(keepImport);
        var jobId = StudioJson.NewId(); var jobDirectory = Path.Combine(root, "jobs", jobId);
        Write(Path.Combine(jobDirectory, "job.json"), JsonSerializer.Serialize(new { job = new StudioJob { Id = jobId, ProjectId = projectId, State = "completed", Stage = "completed", OutputFileName = "output.mp4" }, fingerprint = "saved" }, StudioJson.Options));
        var jobOutput = Path.Combine(jobDirectory, "output.mp4"); Write(jobOutput, "completed-render");
        var jobTemp = Path.Combine(jobDirectory, "job.json." + StudioJson.NewId() + ".tmp"); Write(jobTemp);
        var partialRender = Path.Combine(jobDirectory, "work", "partial.mp4"); Write(partialRender);
        var orphanWork = Path.Combine(root, "jobs", StudioJson.NewId(), "work", "partial.mp4"); Write(orphanWork);
        var unrelatedJob = Path.Combine(root, "jobs", "manual-backup", "keep.txt"); Write(unrelatedJob);
        await using var reopened = new StudioService(root);
        foreach (var path in new[] { projectTemp, packageTemp, stageFile, archive, jobTemp, partialRender, orphanWork }) Assert.False(File.Exists(path), path);
        foreach (var path in new[] { keepAsset, keepProjectNote, package, keepPackageNote, keepImport, jobOutput, unrelatedJob }) Assert.True(File.Exists(path), path);
        Assert.Equal("completed", Assert.Single(await reopened.ListJobsAsync(projectId)).State);
        Assert.Equal("Untitled video", (await reopened.GetProjectAsync(projectId)).Name);
    }

    [SkippableFact] public async Task ScratchDirectoryLinksAreNeverFollowedIntoOtherFiles()
    {
        Skip.If(OperatingSystem.IsWindows(), "Creating directory symlinks requires developer-mode or elevated Windows privileges.");
        await using (var service = new StudioService(root)) await service.CreateProjectAsync();
        var outside = Path.Combine(root, "outside-scratch"); var keep = Path.Combine(outside, "keep.txt"); Write(keep);
        Directory.CreateDirectory(Path.Combine(root, "imports"));
        var link = Path.Combine(root, "imports", StudioJson.NewId()); Directory.CreateSymbolicLink(link, outside);
        await using var reopened = new StudioService(root);
        Assert.True(File.Exists(keep)); Assert.False(Directory.Exists(link));
    }

    [Fact] public async Task StartupScratchCleanupHasABoundedEntryBudget()
    {
        await using (var service = new StudioService(root)) await service.CreateProjectAsync();
        string stage = Path.Combine(root, "imports", StudioJson.NewId());
        for (int i = 0; i < 300; i++) Write(Path.Combine(stage, i + ".partial"));
        await using (var reopened = new StudioService(root))
        {
            int remaining = Directory.EnumerateFiles(stage).Count();
            Assert.InRange(remaining, 44, 299); // At most 256 entries removed per startup.
        }
        await using var next = new StudioService(root);
        Assert.False(Directory.Exists(stage)); // Later recovery safely finishes the remaining bounded work.
    }
}
