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
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>read_file</c>, <c>edit_file</c> and <c>write_file</c> end to end, against a real
/// workspace on a real disk.
///
/// <para>
/// These exist because the point of the change is not that a string got replaced — that
/// is <c>FileEditTests</c> — but that the model is put in a position where replacing a
/// string is the cheapest thing it can do. So what is asserted here is mostly about
/// MESSAGES: that a refusal hands over the file's real bytes rather than leaving a
/// rewrite as the cheapest next move, that a refusal which quotes the file also counts as
/// having shown it, that a success says the rest of the file is untouched so nothing has
/// to be read back, and that a whole-file rewrite is named with numbers.
/// </para>
/// <para>
/// The read gate is asserted in the direction that matters. It must never be possible for
/// the model to be told "read it first" and handed nothing to read — that is a round spent
/// on a demand the host could have satisfied itself, and a model that loses a round to a
/// gate is a model that goes back to heredocs.
/// </para>
/// <para>
/// No shell is needed for any of this: the file tools never launch anything. That is the
/// property that lets them work identically on every host, and it is why these tests have
/// nothing to gate on.
/// </para>
/// </summary>
public class FileToolsTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;
    private readonly SessionWorkspace _workspace;
    private readonly ShellRunner _runner;

    public FileToolsTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-filetools-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _workspaces.GetOrCreate("s");
        _runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = _base,
        });
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { _workspaces.Release("s"); } catch (Exception ex) when (ex is IOException) { }
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        GC.SuppressFinalize(this);
    }

    private string Write(string name, string content)
    {
        string path = Path.Combine(_workspace.WorkDirectory, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        return path;
    }

    private string Read(string name) => File.ReadAllText(Path.Combine(_workspace.WorkDirectory, name));

    private CodeExecResult DoRead(string path, int offset = 0, int limit = 0) =>
        _runner.ReadFile(new ShellTools.ReadRequest(path, offset, limit), _workspace);

    private CodeExecResult DoEdit(string path, string oldString, string newString, bool all = false) =>
        _runner.EditFile(new ShellTools.EditRequest(path, oldString, newString, all), _workspace);

    private CodeExecResult DoWrite(string path, string content, bool overwrite = false) =>
        _runner.WriteFile(
            new ShellTools.WriteRequest(path, content) { Overwrite = overwrite },
            _workspace);

    private static string Numbered(int line) => NumberedListing.Prefix(line);

    [Theory]
    [InlineData("create")]
    [InlineData("edit")]
    [InlineData("overwrite")]
    public void AFileMutation_PublishesOnlyItsOutput_WithDownloadableBytes(string operation)
    {
        var store = new CodeArtifactStore(Path.Combine(_base, "artifacts"));
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ArtifactUriPrefix = "/api/code/artifacts",
        }, null, store);
        Write("untouched.txt", "not an output\n");
        if (operation != "create")
            Write("reports/result.json", "{\"sum\":702}\n");

        CodeExecResult result = operation == "edit"
            ? runner.EditFile(new ShellTools.EditRequest("reports/result.json", "702", "703", false), _workspace)
            : runner.WriteFile(new ShellTools.WriteRequest("reports/result.json", "{\"sum\":703}\n")
                { Overwrite = operation == "overwrite" }, _workspace);

        Assert.True(result.Ok, result.Content);
        CodeArtifact artifact = Assert.Single(result.Artifacts);
        Assert.Equal("reports/result.json", artifact.Path);
        Assert.Equal(result.RunId, artifact.RunId);
        Assert.Equal($"/api/code/artifacts/{artifact.RunId}/reports/result.json", artifact.Pointer);
        Assert.Contains($"[reports/result.json]({artifact.Pointer})", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("untouched.txt", result.Content, StringComparison.Ordinal);
        Assert.True(store.TryResolve(artifact.RunId, artifact.Path, out string? captured, out _));
        Assert.Equal("{\"sum\":703}\n", File.ReadAllText(captured!));

        // The download is a retained copy, independent of later workspace edits.
        Write("reports/result.json", "later\n");
        Assert.Equal("{\"sum\":703}\n", File.ReadAllText(captured!));
    }

    [Fact]
    public void ReadsRefusalsAndIdenticalWrites_DoNotRepublishExistingFiles()
    {
        var store = new CodeArtifactStore(Path.Combine(_base, "artifacts"));
        using var runner = new ShellRunner(new CodeExecOptions { Enabled = true }, null, store);
        Write("result.json", "{}\n");

        Assert.Empty(runner.ReadFile(new ShellTools.ReadRequest("result.json", 0, 0), _workspace).Artifacts);
        Assert.Empty(runner.WriteFile(new ShellTools.WriteRequest("result.json", "{}\n") { Overwrite = false }, _workspace).Artifacts);
        CodeExecResult refused = runner.WriteFile(new ShellTools.WriteRequest("result.json", "different") { Overwrite = false }, _workspace);
        Assert.False(refused.Ok);
        Assert.Empty(refused.Artifacts);
        Assert.Empty(runner.EditFile(new ShellTools.EditRequest("result.json", "missing", "new", false), _workspace).Artifacts);
    }

    [Fact]
    public void AFileMutation_ReportsArtifactLimitsWithoutLosingItsSuccessfulWrite()
    {
        var store = new CodeArtifactStore(Path.Combine(_base, "artifacts"),
            new CodeArtifactLimits { MaxFileBytes = 2 });
        using var runner = new ShellRunner(new CodeExecOptions { Enabled = true }, null, store);

        CodeExecResult result = runner.WriteFile(new ShellTools.WriteRequest("result.txt", "too large"), _workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Equal("too large", Read("result.txt"));
        Assert.Empty(result.Artifacts);
        Assert.Contains("per-file limit", result.Content, StringComparison.Ordinal);
    }

    // ---- read_file ---------------------------------------------------------

    [Fact]
    public void AReadShowsTheLinesNumbered_AndSaysHowLongTheFileIs()
    {
        Write("main.py", "import os\nprint(1)\nprint(2)\n");

        CodeExecResult result = DoRead("main.py");

        Assert.True(result.Ok, result.Content);
        Assert.Contains("of 3:", result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(1) + "import os", result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(3) + "print(2)", result.Content, StringComparison.Ordinal);

        // The trailing empty element from splitting "…\n" is a POSITION, not a line. If
        // it were counted the file would be reported one line longer than it is, and an
        // offset built from that number would read nothing.
        Assert.DoesNotContain(Numbered(4), result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSeparatorIsAPipe_NotATab()
    {
        // A tab is exactly what `nl -ba` and `grep -n` emit, and copying that back into an
        // edit produces text with a number glued to the front. The pipe makes the number
        // visibly not part of the line.
        Write("a.txt", string.Join("\n", Enumerable.Range(1, 40).Select(i => "line " + i)) + "\n");

        CodeExecResult result = DoRead("a.txt");

        Assert.Contains(" | line 1", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("\tline 1", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ALongFile_IsWindowed_AndSaysHowToSeeTheRest()
    {
        // Never silently short: a model told nothing about the remainder reads the window
        // as the whole file and edits against lines it was never shown.
        Write("big.txt", string.Join("\n", Enumerable.Range(1, 900).Select(i => "line " + i)) + "\n");

        CodeExecResult result = DoRead("big.txt");

        Assert.True(result.Ok);
        Assert.Contains("more line(s) below", result.Content, StringComparison.Ordinal);
        Assert.Contains("offset 401", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOffsetAndLimit_ShowExactlyThatRange()
    {
        Write("big.txt", string.Join("\n", Enumerable.Range(1, 100).Select(i => "line " + i)) + "\n");

        CodeExecResult result = DoRead("big.txt", offset: 10, limit: 3);

        Assert.Contains("line 10 to 12 of 100", result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(10) + "line 10", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(Numbered(13), result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void RereadingAnUnchangedFile_SaysSoInsteadOfRepeatingIt()
    {
        // Claude Code's anti-re-read. The saving is real but the message is the point: it
        // tells the model the earlier result is still authoritative, which is what stops
        // the read-after-every-edit loop.
        Write("main.py", "print(1)\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult again = DoRead("main.py");

        Assert.True(again.Ok);
        Assert.Contains("unchanged since you last read it", again.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(Numbered(1), again.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatChangedSinceTheLastRead_IsRenderedAgain()
    {
        // Freshness is decided by comparing CONTENT, so a change made by anything at all —
        // here, straight through the filesystem behind the host's back — is seen.
        Write("main.py", "print(1)\n");
        Assert.True(DoRead("main.py").Ok);
        Write("main.py", "print(2)\n");

        CodeExecResult again = DoRead("main.py");

        Assert.Contains(Numbered(1) + "print(2)", again.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingFile_IsRefusedWithTheNearestNameThatExists()
    {
        // Half the failures in the earlier editing era were about WHERE, not what. A list
        // of everything is a haystack; the nearest name is an answer.
        Write("create_slides.py", "x = 1\n");

        CodeExecResult result = DoRead("create_slide.py");

        Assert.False(result.Ok);
        Assert.Contains("create_slides.py", result.Content, StringComparison.Ordinal);
        Assert.StartsWith("No change was made:", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ADirectory_IsRefusedByPointingAtTheShell()
    {
        Directory.CreateDirectory(Path.Combine(_workspace.WorkDirectory, "src"));

        CodeExecResult result = DoRead("src");

        Assert.False(result.Ok);
        Assert.Contains("is a directory", result.Content, StringComparison.Ordinal);
        Assert.Contains(ShellTools.ShellToolName, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ABinaryFile_IsNotRenderedAsLines()
    {
        // Reading UTF-16-without-a-BOM as Latin-1 yields lines laced with NULs, so both
        // the count and the lines would be nonsense stated to the model as facts about
        // its own file.
        File.WriteAllBytes(Path.Combine(_workspace.WorkDirectory, "blob.bin"),
            new byte[] { 0x41, 0x00, 0x42, 0x00, 0x43, 0x00 });

        CodeExecResult result = DoRead("blob.bin");

        Assert.False(result.Ok);
        Assert.Contains("not a text file", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void APathOutsideTheWorkspace_IsRefused()
    {
        // These run in the HOST process, which is not sandboxed. A read tool that renders
        // whatever file the model names is a strictly larger exposure than a patch that
        // writes one.
        CodeExecResult result = DoRead("../../../etc/passwd");

        Assert.False(result.Ok);
    }

    // ---- edit_file ---------------------------------------------------------

    [Fact]
    public void AnEdit_ChangesOnlyWhatItNames()
    {
        Write("main.py", "import os\nvalue = 1\nprint(value)\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "value = 1", "value = 2");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("import os\nvalue = 2\nprint(value)\n", Read("main.py"));
        Assert.Contains("line 2", result.Content, StringComparison.Ordinal);

        // The whole economic argument, said to the model: there is nothing to verify.
        Assert.Contains("no need to read it back", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditCanDeleteText()
    {
        // An empty new_string. Without it the only way to remove a line would be to
        // rewrite the file around it — the behaviour this tool exists to replace.
        Write("main.py", "import os\nimport sys\nprint(1)\n");
        Assert.True(DoRead("main.py").Ok);

        Assert.True(DoEdit("main.py", "import sys\n", string.Empty).Ok);
        Assert.Equal("import os\nprint(1)\n", Read("main.py"));
    }

    [Fact]
    public void AnAmbiguousEdit_IsRefused_AndTheMatchesAreLocated()
    {
        // Deliberately the OPPOSITE policy to this host's patcher, which applies at the
        // first match and says so. The reasoning does not transfer: a hunk's context is a
        // fixed few lines the model cannot easily widen, whereas old_string is unbounded
        // and "include more of the surrounding lines" can be acted on immediately.
        Write("main.py", "x = 1\ny = 2\nx = 1\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 3");

        Assert.False(result.Ok);
        Assert.Equal("x = 1\ny = 2\nx = 1\n", Read("main.py"));
        Assert.Contains("appears 2 times", result.Content, StringComparison.Ordinal);
        Assert.Contains("replace_all", result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(1), result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(3), result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ReplaceAll_ChangesEveryOccurrence()
    {
        Write("main.py", "x = 1\ny = 2\nx = 1\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 3", all: true);

        Assert.True(result.Ok, result.Content);
        Assert.Equal("x = 3\ny = 2\nx = 3\n", Read("main.py"));
        Assert.Contains("2 places", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditThatMatchesNothing_ShowsTheFilesRealBytes()
    {
        // THE message. A refusal that says only "not found" leaves re-typing the file as
        // the cheapest next move, which is the behaviour the whole tool exists to remove.
        Write("main.py", "def solve():\n        return 1\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "    return 99", "    return 2");

        Assert.False(result.Ok);
        Assert.Equal("def solve():\n        return 1\n", Read("main.py"));

        // What the model sent, echoed with its whitespace intact...
        Assert.Contains("return 99", result.Content, StringComparison.Ordinal);
        // ...and what is actually there, numbered.
        Assert.Contains(Numbered(2) + "        return 1", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditWhoseIndentationIsWrong_IsToldThatIsTheProblem()
    {
        // Indentation is the near-miss that actually happens and the one that is invisible
        // in a summary, so it is diagnosed by name rather than left to be inferred.
        Write("main.py", "def solve():\n\t\treturn 1\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "  return 1", "  return 2");

        Assert.False(result.Ok);
        Assert.Contains("not with the spacing you wrote", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditAimedAtTheWrongFileEntirely_IsToldToCheckTheFile()
    {
        // Nothing resembles the text. Inventing a "closest match" would send the model to
        // rebuild an edit against a region that has nothing to do with it.
        Write("main.py", "alpha\nbeta\ngamma\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "def totally_unrelated():", "x");

        Assert.False(result.Ok);
        Assert.Contains("Check you are editing the right file", result.Content, StringComparison.Ordinal);
        Assert.Contains(Numbered(1) + "alpha", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFailedEditIsFramedAsAFileOperation_NotAsACommand()
    {
        // With five code tools instead of two, telling the model "The command was not run"
        // about an edit costs a round working out which tool that sentence describes.
        Write("main.py", "alpha\n");
        CodeExecResult result = DoEdit("main.py", "nope", "x");

        Assert.StartsWith("No change was made:", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("The command was not run", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void EditingAFileThatDoesNotExist_PointsAtWriteFile()
    {
        CodeExecResult result = DoEdit("nope.py", "a", "b");

        Assert.False(result.Ok);
        Assert.Contains(ShellTools.WriteToolName, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditThatWouldChangeNothing_IsRefused()
    {
        Write("main.py", "x = 1\n");
        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 1");

        Assert.False(result.Ok);
        Assert.Contains("identical", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ACrlfFile_KeepsItsLineEndings()
    {
        // Converting a CRLF file to LF is a whole-file change nobody asked for, and it
        // shows up in someone's diff the next morning as every line modified.
        Write("main.py", "alpha\r\nbeta\r\ngamma\r\n");
        Assert.True(DoRead("main.py").Ok);

        Assert.True(DoEdit("main.py", "beta", "BETA").Ok);
        Assert.Equal("alpha\r\nBETA\r\ngamma\r\n", Read("main.py"));
    }

    [Fact]
    public void ATolerantMatch_IsReported_AndWritesTheFilesOwnPunctuation()
    {
        Write("main.py", "say(‘HELLO’)\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "say('HELLO')", "say('GOODBYE')");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("say(‘GOODBYE’)\n", Read("main.py"));
        Assert.Contains("typographic", result.Content, StringComparison.Ordinal);
    }

    // ---- the read gate -----------------------------------------------------

    [Fact]
    public void AnEditWithoutAPriorRead_IsAppliedWhenItIsUnambiguous_AndSaysSo()
    {
        // Applied rather than refused, because the anchor was unique: the model
        // demonstrably had the bytes, and refusing would cost a round to prove something
        // already proved. Saying so is what keeps the invariant meaningful.
        Write("main.py", "value = 1\n");

        CodeExecResult result = DoEdit("main.py", "value = 1", "value = 2");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("value = 2\n", Read("main.py"));
        Assert.Contains("had not read", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ReplaceAllOnAFileNeverRead_IsRefused()
    {
        // The one case where "it matched, so apply it" is not good enough: a whole-file
        // substitution may match in places the model has never looked at.
        Write("main.py", "x = 1\ny = 2\nx = 1\n");

        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 3", all: true);

        Assert.False(result.Ok);
        Assert.Equal("x = 1\ny = 2\nx = 1\n", Read("main.py"));
        Assert.Contains(ShellTools.ReadToolName, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ReplaceAllOnAFileThatWasRead_IsAllowed()
    {
        Write("main.py", "x = 1\ny = 2\nx = 1\n");
        Assert.True(DoRead("main.py").Ok);

        Assert.True(DoEdit("main.py", "x = 1", "x = 3", all: true).Ok);
    }

    [Fact]
    public void AFailedEditCountsAsHavingShownTheFile_SoTheNextOneIsNotGated()
    {
        // There is no state in which the model is told "you have not read this" and handed
        // nothing to read. A refusal that quotes the file HAS shown it those lines, so the
        // next edit against them is not treated as blind.
        Write("main.py", "x = 1\ny = 2\nx = 1\n");

        // A miss, whose message prints the file's opening lines.
        Assert.False(DoEdit("main.py", "nowhere", "x").Ok);

        CodeExecResult result = DoEdit("main.py", "y = 2", "y = 20");

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("had not read", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void OneNarrowEdit_DoesNotUnlockAWholeFileReplaceAll()
    {
        // The gate is "have you seen the WHOLE file", not "have you touched it". An edit
        // proves the model had the bytes it named and says nothing about the rest — so
        // recording the post-edit file as fully seen would let one narrow edit silently
        // authorise a rename across regions it has never looked at.
        Write("main.py", "x = 1\ny = 2\nx = 1\n");

        Assert.True(DoEdit("main.py", "y = 2", "y = 20").Ok);

        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 3", all: true);

        Assert.False(result.Ok);
        Assert.Contains("whole file", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void APartialRead_DoesNotAuthoriseAWholeFileReplaceAll()
    {
        // "I read lines 1-2 of 3" is exactly as blind about line 3 as having read nothing.
        Write("main.py", "x = 1\ny = 2\nx = 1\n");
        Assert.True(DoRead("main.py", offset: 1, limit: 2).Ok);

        CodeExecResult result = DoEdit("main.py", "x = 1", "x = 3", all: true);

        Assert.False(result.Ok);
        Assert.Contains(ShellTools.ReadToolName, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditAfterTheFileChangedUnderneath_AppliesAndWarns()
    {
        // Recover, do not reject: the edit matched, so it is applied — but the file holds
        // other changes the model's context does not have, and only the host knows that.
        Write("main.py", "value = 1\nother = 0\n");
        Assert.True(DoRead("main.py").Ok);
        Write("main.py", "value = 1\nother = 999\n");

        CodeExecResult result = DoEdit("main.py", "value = 1", "value = 2");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("value = 2\nother = 999\n", Read("main.py"));
        Assert.Contains("had changed since you last read it", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditRightAfterAnEdit_IsNotGated()
    {
        // A successful edit leaves the ledger holding exactly what was written, so a
        // sequence of edits costs one read, not one read each.
        Write("main.py", "a = 1\nb = 2\n");
        Assert.True(DoRead("main.py").Ok);

        Assert.True(DoEdit("main.py", "a = 1", "a = 10").Ok);

        CodeExecResult second = DoEdit("main.py", "b = 2", "b = 20");
        Assert.True(second.Ok, second.Content);
        Assert.DoesNotContain("had changed since", second.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("had not read", second.Content, StringComparison.Ordinal);
    }

    // ---- write_file --------------------------------------------------------

    [Fact]
    public void AWriteCreatesAFile_AndSaysNothingNeedsCheckingAfterwards()
    {
        CodeExecResult result = DoWrite("new.py", "print(1)\n");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("print(1)\n", Read("new.py"));
        Assert.Contains("Created new.py", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AWriteCreatesParentDirectories()
    {
        Assert.True(DoWrite("a/b/c.py", "x\n").Ok);
        Assert.Equal("x\n", Read(Path.Combine("a", "b", "c.py")));
    }

    [Fact]
    public void AWriteDoesNotReplaceAnExistingFileWithoutExplicitConfirmation()
    {
        Write("main.py", "value = 1\nkeep = true\n");

        CodeExecResult result = DoWrite("main.py", "value = 2\nkeep = true\n");

        Assert.False(result.Ok);
        Assert.Equal("value = 1\nkeep = true\n", Read("main.py"));
        Assert.Contains(ShellTools.PatchToolName, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(ShellTools.EditToolName, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("overwrite=true", result.Content, StringComparison.Ordinal);
        Assert.Contains("write_file only to create", result.Content, StringComparison.Ordinal);
        Assert.Contains("Nothing was written", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheWriteToolAdvertisesOnlyNewFileCreation()
    {
        ToolFunction declaration = ShellTools.DeclareWrite();

        Assert.DoesNotContain("overwrite", declaration.Parameters.Keys);
        Assert.DoesNotContain("overwrite", declaration.Required);
        Assert.Contains(ShellTools.PatchToolName, declaration.Description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLegacyOverwriteArgumentIsParsedAsExplicitConfirmation()
    {
        var call = new ToolCall
        {
            Name = ShellTools.WriteToolName,
            Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["path"] = "main.py",
                ["content"] = "print(2)\n",
                ["overwrite"] = true,
            },
        };

        Assert.True(ShellTools.TryReadWrite(call, out ShellTools.WriteRequest request, out string? error), error);
        Assert.True(request.Overwrite);
    }

    [Fact]
    public void AStructuredJsonContentArgumentIsWrittenAsJson_NotAClrTypeName()
    {
        var call = new ToolCall
        {
            Name = ShellTools.WriteToolName,
            Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["path"] = "pptx_spec.json",
                ["content"] = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["title"] = "Apple M6 vs M5",
                    ["slides"] = new object[]
                    {
                        new Dictionary<string, object>
                        {
                            ["layout"] = "title",
                            ["title"] = "Apple M6 vs M5",
                        },
                    },
                },
            },
        };

        Assert.True(ShellTools.TryReadWrite(
            call, out ShellTools.WriteRequest request, out string? error), error);
        Assert.DoesNotContain("System.Collections", request.Content, StringComparison.Ordinal);
        Assert.True(_runner.WriteFile(request, _workspace).Ok);

        using JsonDocument document = JsonDocument.Parse(Read("pptx_spec.json"));
        Assert.Equal("Apple M6 vs M5", document.RootElement.GetProperty("title").GetString());
        Assert.Equal(
            "title",
            document.RootElement.GetProperty("slides")[0].GetProperty("layout").GetString());
    }

    [Fact]
    public void ExistingTwoArgumentHostCallersRetainLegacyReplacementSemantics()
    {
        var request = new ShellTools.WriteRequest("main.py", "print(2)\n");

        Assert.True(request.Overwrite);
    }

    [Fact]
    public void CreateOnlyWriteCannotClobberAFileThatWinsTheExistenceRace()
    {
        string path = Write("race.txt", "background job won\n");

        bool written = ShellRunner.TryWriteFileBytes(
            path,
            "model content\n",
            "\n",
            overwrite: false,
            out string? error);

        Assert.False(written);
        Assert.Contains("overwrite was not authorized", error, StringComparison.Ordinal);
        Assert.Equal("background job won\n", File.ReadAllText(path));
        Assert.Empty(Directory.GetFiles(
            _workspace.WorkDirectory,
            ".tensorsharp-write-*.tmp",
            SearchOption.AllDirectories));
    }

    [Fact]
    public void CreateOnlyWriteConstructionFailureIsNotReportedAsATargetCollision()
    {
        string blocker = Write("blocked", "a file cannot be a parent directory\n");
        string path = Path.Combine(blocker, "report.txt");

        bool written = ShellRunner.TryWriteFileBytes(
            path,
            "model content\n",
            "\n",
            overwrite: false,
            out string? error);

        Assert.False(written);
        Assert.DoesNotContain("path appeared", error, StringComparison.Ordinal);
        Assert.DoesNotContain("overwrite was not authorized", error, StringComparison.Ordinal);
        Assert.Equal("a file cannot be a parent directory\n", File.ReadAllText(blocker));
        Assert.Empty(Directory.GetFiles(
            _workspace.WorkDirectory,
            ".tensorsharp-write-*.tmp",
            SearchOption.AllDirectories));
    }

    [Fact]
    public void AnIdempotentWriteSucceedsAndAuthorizesAFollowingEdit()
    {
        Write("main.txt", "value = 1\n");

        CodeExecResult same = DoWrite("main.txt", "value = 1\n");
        CodeExecResult edit = DoEdit("main.txt", "value = 1", "value = 2");

        Assert.True(same.Ok, same.Content);
        Assert.Contains("nothing was written", same.Content, StringComparison.Ordinal);
        Assert.True(edit.Ok, edit.Content);
        Assert.DoesNotContain("had not read", edit.Content, StringComparison.Ordinal);
        Assert.Equal("value = 2\n", Read("main.txt"));
    }

    [Fact]
    public void AWriteThatRetypesAFileToChangeOneLine_IsNamedWithTheNumbers()
    {
        // The legacy overwrite API can still notice a rewrite by construction rather
        // than by scanning a command line for a redirect.
        string before = string.Join("\n", Enumerable.Range(1, 60).Select(i => $"line{i} = {i}")) + "\n";
        Write("deck.py", before);
        Assert.True(DoRead("deck.py").Ok);

        string after = before.Replace("line30 = 30", "line30 = 999", StringComparison.Ordinal);
        CodeExecResult result = DoWrite("deck.py", after, overwrite: true);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("replaced all 60 lines", result.Content, StringComparison.Ordinal);
        Assert.Contains("only 1 line is different", result.Content, StringComparison.Ordinal);
        Assert.Contains("59 lines came back exactly as they already were", result.Content, StringComparison.Ordinal);

        // Show the differing lines as evidence and explain the actual patch argument.
        // A multiset comparison cannot safely produce an ordered contextual patch.
        Assert.Contains("- line30 = 30", result.Content, StringComparison.Ordinal);
        Assert.Contains("+ line30 = 999", result.Content, StringComparison.Ordinal);
        Assert.Contains(ShellTools.PatchToolName, result.Content, StringComparison.Ordinal);
        Assert.Contains("patch argument", result.Content, StringComparison.Ordinal);
        Assert.Contains("*** Update File:", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(ShellTools.EditToolName, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("old_string=", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AGenuineRewrite_IsNotNagged()
    {
        // The note is an accusation, so every case where it is WRONG costs more than the
        // case where it is right saves: a model told its correct action was wasteful will
        // stop taking it.
        Write("old.py", string.Join("\n", Enumerable.Range(1, 60).Select(i => $"old{i}()")) + "\n");
        Assert.True(DoRead("old.py").Ok);

        CodeExecResult result = DoWrite(
            "old.py", string.Join("\n", Enumerable.Range(1, 60).Select(i => $"brand_new{i}()")) + "\n",
            overwrite: true);

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("came back exactly as they already were", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AShortFileRewrite_IsNotNagged()
    {
        // Re-typing ten lines is not the waste this exists to name, and saying so would
        // be nagging.
        Write("tiny.py", "a\nb\nc\n");
        Assert.True(DoRead("tiny.py").Ok);

        CodeExecResult result = DoWrite("tiny.py", "a\nB\nc\n", overwrite: true);

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("came back exactly", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFortyLineFileRetypedToChangeEleven_IsStillNamed()
    {
        // The case a RATIO let through: 11 of 40 is 27.5% and was exempt under the old
        // quarter-of-the-file rule, while 29 lines that were already correct had just been
        // re-emitted and re-rolled. What costs tokens is the identical lines, so that is
        // what is counted.
        string before = string.Join("\n", Enumerable.Range(1, 40).Select(i => $"v{i} = {i}")) + "\n";
        Write("mid.py", before);
        Assert.True(DoRead("mid.py").Ok);

        var sb = new StringBuilder();
        for (int i = 1; i <= 40; i++)
            sb.Append(i <= 11 ? $"v{i} = CHANGED{i}" : $"v{i} = {i}").Append('\n');

        CodeExecResult result = DoWrite("mid.py", sb.ToString(), overwrite: true);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("29 lines came back exactly as they already were", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void OverwritingAFileThatWasNeverRead_SaysWhatWasLost()
    {
        Write("notes.txt", "something the user wrote\n");

        CodeExecResult result = DoWrite("notes.txt", "replaced\n", overwrite: true);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("had not read", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AWriteLeavesTheFileReadable_SoAnEditMayFollowWithoutAReRead()
    {
        Assert.True(DoWrite("new.py", "value = 1\n").Ok);

        CodeExecResult result = DoEdit("new.py", "value = 1", "value = 2");

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("had not read", result.Content, StringComparison.Ordinal);
    }

    // ---- what happens with no workspace at all -----------------------------

    [Fact]
    public void WithoutAWorkspace_EveryFileToolRefusesRatherThanThrowing()
    {
        // The stateless endpoints do not declare these, but a caller can still reach the
        // runner, and a null-reference escaping into the tool dispatcher's catch-all is a
        // turn that renders nothing.
        Assert.False(_runner.ReadFile(new ShellTools.ReadRequest("a", 0, 0), null).Ok);
        Assert.False(_runner.EditFile(new ShellTools.EditRequest("a", "b", "c", false), null).Ok);
        Assert.False(_runner.WriteFile(new ShellTools.WriteRequest("a", "b"), null).Ok);
    }

    // ---- regressions from the adversarial review ---------------------------

    [Fact]
    public void AnEditThatCopiedLineNumbersIntoBothArguments_WritesNeitherIntoTheFile()
    {
        // The rung exists to absorb a model pasting a read_file result back. It stripped
        // the prefixes from what it searched for and wrote the replacement verbatim, so
        // "   2 |     return 1" landed in the source under a result saying the edit
        // succeeded and the rest of the file was untouched.
        Write("main.py", "def f():\n    return 1\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit(
            "main.py",
            Numbered(1) + "def f():\n" + Numbered(2) + "    return 1",
            Numbered(1) + "def f():\n" + Numbered(2) + "    return 2");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("def f():\n    return 2\n", Read("main.py"));
        Assert.DoesNotContain("|", Read("main.py"), StringComparison.Ordinal);
    }

    [Fact]
    public void AnEditOfACrlfFile_DoesNotClaimTheFileChangedUnderneath()
    {
        // read_file recorded the raw bytes while every other site hashed the LF form, so
        // a Windows file was hashed one way when read and another when edited — and every
        // first edit reported a change nobody had made.
        Write("main.py", "alpha\r\nbeta\r\n");
        Assert.True(DoRead("main.py").Ok);

        CodeExecResult result = DoEdit("main.py", "beta", "BETA");

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("had changed since you last read it", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("had not read", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAbsurdLimit_DoesNotOverflowIntoGarbage()
    {
        // The argument reader clamps an oversized JSON number to int.MaxValue, so
        // "read the rest of it" written as limit=99999999999 arrived here and wrapped:
        // a header reading "line 2 to -2147483648", a footer offering a negative offset,
        // and a ledger window that poisoned every later read of that path.
        Write("x.py", string.Join("\n", Enumerable.Range(1, 10).Select(i => "line " + i)) + "\n");

        CodeExecResult result = DoRead("x.py", offset: 2, limit: int.MaxValue);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("line 2 to 10 of 10", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("-", result.Content.Split('\n')[0], StringComparison.Ordinal);
        Assert.DoesNotContain("more line(s) below", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void PagingThroughAWholeFile_AuthorisesAReplaceAll()
    {
        // The gate demanded a single complete read, which a paged read can never produce.
        // A model following read_file's own footer ("read them with offset 401") was
        // refused with "read the whole file first" — advice it had already followed and
        // could not follow differently, because a plain re-read returns the same first
        // page. That loop ends in write_file and a rewritten file.
        string body = string.Join("\n", Enumerable.Range(1, 900)
            .Select(i => i % 100 == 0 ? "TOKEN = 1" : "line" + i + " = " + i)) + "\n";
        Write("app.py", body);

        Assert.True(DoRead("app.py", offset: 1, limit: 400).Ok);
        Assert.True(DoRead("app.py", offset: 401, limit: 400).Ok);
        Assert.True(DoRead("app.py", offset: 801, limit: 400).Ok);

        CodeExecResult result = DoEdit("app.py", "TOKEN = 1", "TOKEN = 2", all: true);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("9 places", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheReplaceAllRefusal_NamesACallThatWouldActuallySatisfyIt()
    {
        // A refusal that cannot be acted on is worse than no gate: it costs a round and
        // teaches the model that the editor is the thing standing in its way.
        Write("app.py", string.Join("\n", Enumerable.Range(1, 50)
            .Select(i => i % 10 == 0 ? "T = 1" : "l" + i)) + "\n");
        Assert.True(DoRead("app.py", offset: 1, limit: 5).Ok);

        CodeExecResult result = DoEdit("app.py", "T = 1", "T = 2", all: true);

        Assert.False(result.Ok);
        Assert.Contains("offset 1 and limit 50", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileTooBigForTheHost_IsRefusedRatherThanRead()
    {
        // These run in the HOST process, unsandboxed, with the path chosen by the model,
        // and .NET decodes UTF-8 to UTF-16 — so an unbounded read is an
        // OutOfMemoryException that takes the server with it.
        string big = new string('x', 9 * 1024 * 1024);
        Write("huge.txt", big);

        CodeExecResult result = DoRead("huge.txt");

        Assert.False(result.Ok);
        Assert.Contains("too large", result.Content, StringComparison.Ordinal);
        Assert.Contains(ShellTools.ShellToolName, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnErrorAboutAFile_DoesNotLeakTheHostsAbsolutePath()
    {
        // Every other result on this surface is scrubbed; the file tools' raw exception
        // messages were not, and a filesystem exception names the host's real path.
        Write("locked", "x");
        CodeExecResult result = DoRead("nope-does-not-exist.txt");

        Assert.False(result.Ok);
        Assert.DoesNotContain(_workspace.WorkDirectory, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(_base, result.Content, StringComparison.Ordinal);
    }
}
