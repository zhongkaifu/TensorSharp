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
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Noticing a whole file re-typed to change two lines of it — and, just as importantly,
/// never saying that about a command that did something else.
///
/// <para>
/// The note this class produces is an accusation, so every case where it is WRONG costs
/// more than the case where it is right saves: a model told its correct action was
/// wasteful will stop taking it. The append case below is not hypothetical — it was the
/// shipped behaviour until an adversarial review found it.
/// </para>
/// </summary>
public class RewriteWatchTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;

    public RewriteWatchTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-rewrite-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
    }

    public void Dispose()
    {
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static string Program(int lines, string? replacementAt = null, int at = 5) =>
        string.Join("\n", Enumerable.Range(0, lines).Select(
            i => i == at && replacementAt != null ? replacementAt : $"line_{i} = {i}")) + "\n";

    private void Write(string name, string content) =>
        File.WriteAllText(Path.Combine(_workspace.WorkDirectory, name), content);

    private string? Run(string command, string before, string after, string name = "deck.py")
    {
        Write(name, before);
        RewriteWatch? watch = RewriteWatch.Before(command, _workspace, _workspace.WorkDirectory);
        Write(name, after);
        return watch?.Describe(_workspace, "apply_patch");
    }

    // ---- the case it exists for ----------------------------------------------

    [Fact]
    public void AWholeFileRetypedToChangeOneLine_IsNamedWithTheLineItself()
    {
        string? note = Run(
            "cat > deck.py <<'EOF'\n...\nEOF",
            Program(60),
            Program(60, replacementAt: "line_5 = 'LAYOUT_16x9'"));

        Assert.NotNull(note);
        Assert.Contains("deck.py already existed", note!, StringComparison.Ordinal);
        Assert.Contains("replaced all 60 lines", note!, StringComparison.Ordinal);
        Assert.Contains("1 line is different", note!, StringComparison.Ordinal);
        Assert.Contains("apply_patch", note!, StringComparison.Ordinal);

        // The bytes, not just the count. "Send those 3 lines to apply_patch" without
        // saying WHICH asks the model to rebuild them from memory — the exact failure the
        // patch tool exists to avoid, so the advice would cause the thing it warns about.
        Assert.Contains("- line_5 = 5", note!, StringComparison.Ordinal);
        Assert.Contains("+ line_5 = 'LAYOUT_16x9'", note!, StringComparison.Ordinal);
        Assert.Contains("patch argument", note!, StringComparison.Ordinal);
        Assert.Contains("*** Begin Patch / *** End Patch", note!, StringComparison.Ordinal);
        Assert.Contains("*** Update File: <path>", note!, StringComparison.Ordinal);
        Assert.Contains("@@ hunks", note!, StringComparison.Ordinal);
        Assert.Contains("single-file or multi-file change", note!, StringComparison.Ordinal);
        Assert.Contains("they are not a patch to apply now", note!, StringComparison.Ordinal);
        Assert.DoesNotContain("old_string", note!, StringComparison.Ordinal);
        Assert.DoesNotContain("new_string", note!, StringComparison.Ordinal);
        Assert.DoesNotContain("edit_file", note!, StringComparison.Ordinal);
    }

    // ---- and every case where it must stay quiet ------------------------------

    /// <summary>
    /// The false statement an adversarial review found in the shipped version. An APPEND
    /// adds lines; it replaces nothing. To a line-count comparison a 190-line file with
    /// five lines appended looks exactly like a 195-line file with five lines changed —
    /// and the model was told its correct action had been wasteful.
    /// </summary>
    [Fact]
    public void AnAppendIsNeverCalledARewrite()
    {
        string? note = Run(
            "cat >> deck.py <<'EOF'\nextra\nEOF",
            Program(60),
            Program(60) + "extra_1 = 1\nextra_2 = 2\n");

        Assert.Null(note);
    }

    /// <summary>
    /// "Before" and "after" must be the same FILE. The watch remembered a relative name
    /// and resolved it twice — once against the directory the command started in, once
    /// against the directory it ended in — so `cd sub &amp;&amp; cat > deck.py …` read
    /// `work/deck.py` going in, `work/sub/deck.py` coming out, and then reported that one
    /// line of "it" had changed while describing two different files.
    /// </summary>
    [Fact]
    public void TheFileComparedIsTheOneThatWasRemembered()
    {
        Directory.CreateDirectory(Path.Combine(_workspace.WorkDirectory, "sub"));
        // Same name, different directories, deliberately different contents.
        Write("deck.py", Program(60));
        File.WriteAllText(
            Path.Combine(_workspace.WorkDirectory, "sub", "deck.py"),
            Program(60, replacementAt: "line_5 = 'moved'"));

        // Remembered from the work directory...
        RewriteWatch? watch = RewriteWatch.Before(
            "cat > deck.py <<'EOF'\n...\nEOF", _workspace, _workspace.WorkDirectory);
        Assert.NotNull(watch);

        // ...and the work-directory file is genuinely untouched, so there is nothing to say.
        Assert.Null(watch!.Describe(_workspace, "apply_patch"));
    }

    [Fact]
    public void ABrandNewFileIsNotARewrite()
    {
        RewriteWatch? watch = RewriteWatch.Before(
            "cat > fresh.py <<'EOF'\n...\nEOF", _workspace, _workspace.WorkDirectory);
        Write("fresh.py", Program(60));

        Assert.Null(watch?.Describe(_workspace, "apply_patch"));
    }

    [Fact]
    public void AShortFileIsNotWorthMentioning()
    {
        Assert.Null(Run(
            "cat > deck.py <<'EOF'\n...\nEOF",
            Program(10),
            Program(10, replacementAt: "line_5 = 99", at: 5)));
    }

    /// <summary>
    /// A model that genuinely rewrote the program must not be told it should have patched
    /// it. The note is for a small edit typed the long way, and nothing else.
    /// </summary>
    [Fact]
    public void AGenuineRewriteIsNotNagged()
    {
        Assert.Null(Run(
            "cat > deck.py <<'EOF'\n...\nEOF",
            Program(60),
            string.Join("\n", Enumerable.Range(0, 60).Select(i => $"completely_different_{i}()")) + "\n"));
    }

    [Fact]
    public void AnUnchangedFileSaysNothing()
    {
        Assert.Null(Run("cat > deck.py <<'EOF'\n...\nEOF", Program(60), Program(60)));
    }

    [Fact]
    public void ACommandThatRedirectsNowhereIsNotWatchedAtAll()
    {
        Write("deck.py", Program(60));
        Assert.Null(RewriteWatch.Before("python3 deck.py", _workspace, _workspace.WorkDirectory));
        Assert.Null(RewriteWatch.Before("ls -la", _workspace, _workspace.WorkDirectory));
        Assert.Null(RewriteWatch.Before("python3 deck.py 2>&1", _workspace, _workspace.WorkDirectory));
    }

    /// <summary>
    /// <see cref="File.ReadAllLines(string)"/> decodes UTF-16-without-a-BOM as Latin-1 and
    /// yields lines laced with NULs, so both the line count and the lines themselves would
    /// be nonsense — and the note states both to the model as facts about its own file.
    /// </summary>
    [Fact]
    public void ABinaryOrUtf16FileIsNotReasonedAbout()
    {
        string path = Path.Combine(_workspace.WorkDirectory, "data.bin");
        File.WriteAllBytes(path, new UnicodeEncoding(bigEndian: false, byteOrderMark: false)
            .GetBytes(Program(60)));

        RewriteWatch? watch = RewriteWatch.Before(
            "cat > data.bin <<'EOF'\n...\nEOF", _workspace, _workspace.WorkDirectory);

        Assert.Null(watch?.Describe(_workspace, "apply_patch"));
    }

    /// <summary>A minified bundle is one line of half a megabyte; it must not be echoed whole.</summary>
    [Fact]
    public void AVeryLongLineIsClippedRatherThanEchoed()
    {
        string huge = new string('x', 50_000);
        string? note = Run(
            "cat > deck.py <<'EOF'\n...\nEOF",
            Program(60),
            Program(60, replacementAt: huge));

        Assert.NotNull(note);
        Assert.DoesNotContain(huge, note!);
        Assert.True(note!.Length < 4000, $"note was {note.Length} characters");
    }
}
