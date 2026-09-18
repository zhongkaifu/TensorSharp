// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

public sealed class CodeRepairHintTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "ts-repair-hint-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _manager;

    public CodeRepairHintTests()
    {
        _manager = new SessionWorkspaceManager(_root);
    }

    [Fact]
    public void PythonFailurePointsToTheSmallestEditableRegion()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("python");
        string source = Path.Combine(workspace.WorkDirectory, "calculator.py");
        File.WriteAllText(source, "def divide(a, b):\n    return a / b\n\nprint(divide(4, 0))\n");

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"calculator.py\", line 4, in <module>\n"
            + "    print(divide(4, 0))\n"
            + "ZeroDivisionError: division by zero",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'calculator.py' around line 4", hint, StringComparison.Ordinal);
        Assert.Contains(NumberedListing.Prefix(4) + "print(divide(4, 0))", hint, StringComparison.Ordinal);
        Assert.DoesNotContain(ShellTools.EditToolName, hint, StringComparison.Ordinal);
        Assert.Contains("`" + ShellTools.PatchToolName + "`", hint, StringComparison.Ordinal);
        Assert.Contains("Do not use `" + ShellTools.WriteToolName + "`", hint, StringComparison.Ordinal);
        Assert.Contains("run the same check again", hint, StringComparison.Ordinal);

        FileLedger.ReadState shown = workspace.Reads.Check(source, File.ReadAllText(source));
        Assert.Equal(ReadFreshness.Fresh, shown.Freshness);
        Assert.True(shown.Complete, "the four-line file was shown in full by the repair excerpt");
    }

    [Fact]
    public void DeepestResolvablePythonFrameWins()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("frames");
        string source = Path.Combine(workspace.WorkDirectory, "app.py");
        File.WriteAllText(source, "start()\nraise ValueError('bad')\n");

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"app.py\", line 1, in <module>\n"
            + "  File \"../env/library.py\", line 99, in start\n"
            + "ValueError: bad",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'app.py' around line 1", hint, StringComparison.Ordinal);
        Assert.DoesNotContain("library.py' around line 99", hint, StringComparison.Ordinal);
    }

    [Fact]
    public void NodeFileUriFrameResolvesBackToTheWorkspaceSource()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("node");
        string source = Path.Combine(workspace.WorkDirectory, "app.mjs");
        File.WriteAllText(source, "const answer = missing + 1;\nconsole.log(answer);\n");
        string fileUri = new Uri(source).AbsoluteUri;

        string? hint = CodeRepairHint.Create(
            "ReferenceError: missing is not defined\n    at " + fileUri + ":1:16",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'app.mjs' around line 1", hint, StringComparison.Ordinal);
        Assert.Contains("const answer = missing + 1", hint, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void NodeFramePreservesParenthesesInsideTheSourcePath(bool functionWrapper)
    {
        SessionWorkspace workspace = _manager.GetOrCreate(
            functionWrapper ? "node-parentheses-wrapper" : "node-parentheses-bare");
        string directory = Path.Combine(workspace.WorkDirectory, "scripts (dev)");
        Directory.CreateDirectory(directory);
        string source = Path.Combine(directory, "app.mjs");
        File.WriteAllText(source, "const answer = missing + 1;\n");
        string frame = functionWrapper
            ? "    at main (" + source + ":1:16)"
            : "    at " + source + ":1:16";

        string? hint = CodeRepairHint.Create(
            "ReferenceError: missing is not defined\n" + frame,
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'scripts (dev)/app.mjs' around line 1", hint, StringComparison.Ordinal);
    }

    [Fact]
    public void FrameShapedTextInsideAJavaScriptErrorMessageCannotForgeARepairTarget()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("node-forged-frame");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "victim.mjs"), "doNotEdit();\n");

        string? hint = CodeRepairHint.Create(
            "ReferenceError: rejected input said at victim.mjs:1:1",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.Null(hint);
    }

    [Fact]
    public void PythonFrameTextAfterTheTerminalExceptionCannotOverrideTheRealTraceback()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("python-forged-frame");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "actual.py"), "raise ValueError('bad')\n");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "victim.py"), "do_not_edit()\n");

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"actual.py\", line 1, in <module>\n"
            + "ValueError: rejected text follows\n"
            + "  File \"victim.py\", line 1, in <module>\n",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'actual.py' around line 1", hint, StringComparison.Ordinal);
        Assert.DoesNotContain("victim.py", hint, StringComparison.Ordinal);
    }

    [Fact]
    public void CompilerDiagnosticIsActionableEvenWithoutATracebackExceptionName()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("compiler");
        string source = Path.Combine(workspace.WorkDirectory, "Program.cs");
        File.WriteAllText(source, "class Program\n{\n    static void Main() => Missing();\n}\n");

        string? hint = CodeRepairHint.Create(
            "Program.cs(3,27): error CS0103: The name 'Missing' does not exist in the current context",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("'Program.cs' around line 3", hint, StringComparison.Ordinal);
        Assert.Contains("static void Main() => Missing();", hint, StringComparison.Ordinal);
    }

    [Fact]
    public void ClippedSourceLineIsNotCreditedAsRead()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("clipped");
        string source = Path.Combine(workspace.WorkDirectory, "wide.py");
        string wide = "value = '" + new string('x', NumberedListing.MaxExcerptLineChars + 50) + "'";
        File.WriteAllText(source, wide + "\nraise ValueError('bad')\n");

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"wide.py\", line 2, in <module>\n"
            + "ValueError: bad",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("line continues", hint, StringComparison.Ordinal);
        FileLedger.ReadState shown = workspace.Reads.Check(
            source, File.ReadAllText(source).Replace("\r\n", "\n", StringComparison.Ordinal));
        Assert.Equal(ReadFreshness.Unread, shown.Freshness);
    }

    [Fact]
    public void SymlinkNamedByTracebackIsNeverFollowed()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("symlink");
        string outside = Path.Combine(_root, "outside.py");
        string link = Path.Combine(workspace.WorkDirectory, "linked.py");
        File.WriteAllText(outside, "SECRET_OUTSIDE_SOURCE\n");
        try
        {
            File.CreateSymbolicLink(link, outside);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                      or PlatformNotSupportedException)
        {
            return;
        }

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"linked.py\", line 1, in <module>\n"
            + "NameError: bad",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.Null(hint);
    }

    [Fact]
    public void FifoNamedByTracebackCannotBlockRepairCoaching()
    {
        if (OperatingSystem.IsWindows())
            return;

        SessionWorkspace workspace = _manager.GetOrCreate("fifo");
        string source = Path.Combine(workspace.WorkDirectory, "fifo.py");
        Assert.Equal(0, MkFifo(source, Convert.ToUInt32("600", 8)));

        int flags = 2 | (OperatingSystem.IsMacOS() ? 0x00000004 | 0x01000000
                                                   : 0x00000800 | 0x00080000);
        int fd = OpenUnix(source, flags);
        Assert.True(fd >= 0, $"open failed with errno {Marshal.GetLastWin32Error()}");

        using var keeper = new SafeFileHandle((IntPtr)fd, ownsHandle: true);
        string? hint = null;
        Exception? failure = null;
        var reader = new Thread(() =>
        {
            try
            {
                hint = CodeRepairHint.Create(
                    "Traceback (most recent call last):\n"
                    + "  File \"fifo.py\", line 1, in <module>\n"
                    + "NameError: bad",
                    workspace,
                    workspace.WorkDirectory,
                    networkConfined: true);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        }) { IsBackground = true };

        reader.Start();
        bool completedWithoutRescue = reader.Join(TimeSpan.FromSeconds(2));
        if (!completedWithoutRescue)
        {
            keeper.Dispose();
            Assert.True(reader.Join(TimeSpan.FromSeconds(2)),
                "the blocking repair-hint reader could not be rescued");
        }

        Assert.Null(failure);
        Assert.True(completedWithoutRescue, "repair coaching blocked on a FIFO source path");
        Assert.NotNull(hint);
        Assert.Contains("points to 'fifo.py' at line 1", hint, StringComparison.Ordinal);
    }

    [Fact]
    public void OversizedSourceProducesPathOnlyAdviceWithoutLedgerCredit()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("oversized");
        string source = Path.Combine(workspace.WorkDirectory, "large.py");
        File.WriteAllText(source, new string('x', 1024 * 1024 + 1));

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"large.py\", line 1, in <module>\n"
            + "NameError: bad",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.NotNull(hint);
        Assert.Contains("points to 'large.py' at line 1", hint, StringComparison.Ordinal);
        Assert.DoesNotContain(NumberedListing.Prefix(1), hint, StringComparison.Ordinal);
        Assert.Equal(
            ReadFreshness.Unread,
            workspace.Reads.Check(source, File.ReadAllText(source)).Freshness);
    }

    [Fact]
    public void EnvironmentFailureDoesNotRecommendEditing()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("environment");

        string? hint = CodeRepairHint.Create(
            "ModuleNotFoundError: No module named 'missing_package'",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.Null(hint);
    }

    [Fact]
    public void InlineFailureDoesNotRecommendAFileToolThatCannotFixIt()
    {
        SessionWorkspace workspace = _manager.GetOrCreate("inline");

        string? hint = CodeRepairHint.Create(
            "Traceback (most recent call last):\n"
            + "  File \"command\", line 1, in <module>\n"
            + "NameError: name 'answer' is not defined",
            workspace,
            workspace.WorkDirectory,
            networkConfined: true);

        Assert.Null(hint);
    }

    public void Dispose()
    {
        foreach (string id in new[]
                 { "python", "frames", "node", "compiler", "clipped", "symlink", "fifo",
                   "oversized", "environment", "inline" })
            _manager.Release(id);
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        GC.SuppressFinalize(this);
    }

    [DllImport("libc", EntryPoint = "mkfifo", SetLastError = true)]
    private static extern int MkFifo(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, uint mode);

    [DllImport("libc", EntryPoint = "open", SetLastError = true)]
    private static extern int OpenUnix(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int flags);
}
