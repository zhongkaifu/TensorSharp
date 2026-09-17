using System;
using System.IO;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class TestGateModelPathTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-model-gate-" + Guid.NewGuid().ToString("N"));

    public TestGateModelPathTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public void ExplicitFileOverride_IsReturnedByBothLoadersWithoutTheDirectoryNameFilter()
    {
        string model = Write("explicit-model.gguf", 3);

        Assert.Equal(model, TestGates.FindGguf(model, "different-quant"));
        Assert.Equal(model, TestGates.FindSmallestGguf(model, "different-quant"));
    }

    [Fact]
    public void DirectorySearch_ExcludesCompanionsAndAcceptsAlternativeNames()
    {
        Write("target-mmproj.gguf", 1);
        Write("target-assistant.gguf", 1);
        Write("unrelated.gguf", 1);
        string model = Write("TARGET-model.gguf", 3);

        Assert.Equal(model, TestGates.FindGguf(_directory, "absent|target"));
        Assert.Null(TestGates.FindGguf(_directory, "missing"));
    }

    [Fact]
    public void SmallestDirectoryMatch_IsSelectedByFileSize()
    {
        Write("target-large.gguf", 5);
        string small = Write("target-small.gguf", 2);

        Assert.Equal(small, TestGates.FindSmallestGguf(_directory, "target"));
    }

    private string Write(string name, int length)
    {
        string path = Path.Combine(_directory, name);
        File.WriteAllBytes(path, new byte[length]);
        return path;
    }

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
