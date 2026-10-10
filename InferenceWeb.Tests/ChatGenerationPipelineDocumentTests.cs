namespace InferenceWeb.Tests;

public class ChatGenerationPipelineDocumentTests
{
    [Fact]
    public void ImageDetectionIncludesMediaFromAnEarlierTurn()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "look", ImagePaths = new() { "photo.png" } },
            new() { Role = "assistant", Content = "I see it" },
            new() { Role = "user", Content = "what color is it?" },
        };

        Assert.True(ChatGenerationPipeline.HasImageAttachments(history));
        Assert.False(ChatGenerationPipeline.HasImageAttachments(
            new List<ChatMessage> { new() { Role = "user", Content = "text only" } }));
    }

    private static Dictionary<string, string> Staged(
        params (string Path, string WorkspaceName)[] files) =>
        files.ToDictionary(file => file.Path, file => file.WorkspaceName, StringComparer.Ordinal);

    [Fact]
    public void RejectAttachedDocumentOverflow_AllowsCompleteDocumentThatFits()
    {
        ChatGenerationPipeline.RejectAttachedDocumentOverflow(
            promptTokens: 68_847,
            maxTokens: 4_096,
            modelContextLimit: 131_072,
            preserveAllInput: true);
    }

    [Fact]
    public void RejectAttachedDocumentOverflow_RejectsInsteadOfSilentlyTruncating()
    {
        var ex = Assert.Throws<PromptContextOverflowException>(() =>
            ChatGenerationPipeline.RejectAttachedDocumentOverflow(
                promptTokens: 130_000,
                maxTokens: 4_096,
                modelContextLimit: 131_072,
                preserveAllInput: true));

        Assert.Contains("No document content was truncated", ex.Message);
        Assert.Contains("130000 prompt tokens", ex.Message);
        Assert.Contains("131072 context tokens", ex.Message);
    }

    [Fact]
    public void PromptThatAloneExceedsContext_ExplainsThatReplyLimitCannotIncreaseContext()
    {
        var ex = Assert.Throws<PromptContextOverflowException>(() =>
            ChatGenerationPipeline.RejectAttachedDocumentOverflow(
                promptTokens: 59_086,
                maxTokens: 1,
                modelContextLimit: 32_768,
                preserveAllInput: true));

        Assert.Contains("59086 tokens", ex.Message);
        Assert.Contains("effective model/engine context limit of 32768 tokens", ex.Message);
        Assert.Contains("reply-length setting controls only generated output", ex.Message);
        Assert.Contains("large CSV or table", ex.Message);
        Assert.DoesNotContain("increase the scheduler", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void HasTextFileAttachments_DetectsUploadedTextOrPdfPath()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "analyze this" },
            new()
            {
                Role = "user",
                Content = "[File: book.pdf]...",
                TextFilePaths = new List<string> { "uploads/book.pdf" },
            },
        };

        Assert.True(ChatGenerationPipeline.HasTextFileAttachments(history));
        Assert.True(ChatGenerationPipeline.HasTextFileAttachments(
            new List<ChatMessage>
            {
                new()
                {
                    Role = "user",
                    Content = "[File: book.pdf]\ncomplete text\n[End of file]\nSummarize it",
                },
            }));
        Assert.False(ChatGenerationPipeline.HasTextFileAttachments(
            new List<ChatMessage> { new() { Role = "user", Content = "plain chat" } }));
    }

    [Fact]
    public void ToolBackedCsv_ReplacesTheInlineTableAndPreservesTheCompleteFileReference()
    {
        string rows = string.Join('\n', Enumerable.Range(0, 12_000)
            .Select(i => $"customer-{i},region-{i % 7},{i * 13}"));
        string originalContent = "[File: responses.csv]\nname,region,score\n" + rows
            + "\n[End of file]\n\nPlease analyze this form.";
        var original = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = originalContent,
                TextFilePaths = new List<string> { "/safe/uploads/9f3.csv" },
                TextFileNames = new List<string> { "responses.csv" },
                AttachmentPaths = new List<string> { "/safe/uploads/9f3.csv" },
                AttachmentNames = new List<string> { "responses.csv" },
            },
        };

        List<ChatMessage> prepared = ChatHistoryPreparer.UseFileBackedCsvAttachments(
            original, Staged(("/safe/uploads/9f3.csv", "responses.csv")));

        ChatMessage message = Assert.Single(prepared);
        Assert.NotSame(original, prepared);
        Assert.NotSame(original[0], message);
        Assert.Equal(originalContent, original[0].Content); // request/transcript data was not mutated
        Assert.StartsWith("[Attached CSV available to tools: 'responses.csv']", message.Content);
        Assert.Contains("Please analyze this form.", message.Content);
        Assert.DoesNotContain("customer-11999", message.Content);
        Assert.True(message.Content.Length < 600, "the table was still copied into the model prompt");
        Assert.Null(message.TextFilePaths);
        Assert.Null(message.TextFileNames);
        Assert.Equal(original[0].AttachmentPaths, message.AttachmentPaths);
        Assert.False(ChatGenerationPipeline.HasTextFileAttachments(prepared));

        // Every tool-loop round normalizes the history again. The marker makes that
        // operation stable instead of adding another reference on every round.
        Assert.Same(prepared,
            ChatHistoryPreparer.UseFileBackedCsvAttachments(
                prepared, Staged(("/safe/uploads/9f3.csv", "responses.csv"))));
    }

    [Fact]
    public void MetadataOnlyCsv_GainsACompactReferenceAfterItsFileIsStaged()
    {
        var original = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "Please analyze this form.",
                TextFilePaths = new List<string> { "/safe/uploads/stored.csv" },
                TextFileNames = new List<string> { "responses.csv" },
                AttachmentPaths = new List<string> { "/safe/uploads/stored.csv" },
                AttachmentNames = new List<string> { "responses.csv" },
                HasFileBackedTextAttachments = true,
            },
        };

        List<ChatMessage> prepared = ChatHistoryPreparer.UseFileBackedCsvAttachments(
            original, Staged(("/safe/uploads/stored.csv", "responses.csv")));

        ChatMessage message = Assert.Single(prepared);
        Assert.StartsWith("[Attached CSV available to tools: 'responses.csv']", message.Content);
        Assert.EndsWith("Please analyze this form.", message.Content, StringComparison.Ordinal);
        Assert.Null(message.TextFilePaths);
        Assert.False(message.HasFileBackedTextAttachments);
        Assert.False(ChatGenerationPipeline.HasTextFileAttachments(prepared));
        Assert.Equal("Please analyze this form.", original[0].Content);
        Assert.True(original[0].HasFileBackedTextAttachments);
    }

    [Fact]
    public void CsvThatWasNotStaged_RemainsInlineRatherThanSilentlyLosingRows()
    {
        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "[File: data.csv]\na,b\n1,2\n[End of file]\n\nAnalyze it",
                TextFilePaths = new List<string> { "/safe/uploads/data.csv" },
                TextFileNames = new List<string> { "data.csv" },
            },
        };

        Assert.Same(history,
            ChatHistoryPreparer.UseFileBackedCsvAttachments(
                history, Staged(("/safe/uploads/some-other.csv", "data.csv"))));
        Assert.Contains("a,b\n1,2", history[0].Content);
    }

    [Fact]
    public void NonTabularTextAttachment_RemainsInlineEvenWhenShellIsAvailable()
    {
        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "[File: notes.md]\nkeep every paragraph\n[End of file]\n\nSummarize it",
                TextFilePaths = new List<string> { "/safe/uploads/notes.md" },
                TextFileNames = new List<string> { "notes.md" },
            },
        };

        Assert.Same(history,
            ChatHistoryPreparer.UseFileBackedCsvAttachments(
                history, Staged(("/safe/uploads/notes.md", "notes.md"))));
        Assert.Contains("keep every paragraph", history[0].Content);
    }

    [Fact]
    public void ToolBackedCsv_DoesNotConsumeAnEndMarkerTypedByTheUser()
    {
        const string question = "Does the phrase [End of file] appear in the data?";
        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "[File: data.csv]\na,b\n1,2\n[End of file]\n\n" + question,
                TextFilePaths = new List<string> { "/safe/uploads/data.csv" },
                TextFileNames = new List<string> { "data.csv" },
            },
        };

        List<ChatMessage> prepared = ChatHistoryPreparer.UseFileBackedCsvAttachments(
            history, Staged(("/safe/uploads/data.csv", "data.csv")));

        Assert.Contains(question, Assert.Single(prepared).Content);
        Assert.DoesNotContain("a,b\n1,2", prepared[0].Content);
    }

    [Fact]
    public void ToolBackedCsv_LeavesACompanionMarkdownDocumentInline()
    {
        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "[File: data.csv]\na,b\n1,2\n[End of file]\n\n"
                    + "[File: notes.md]\nKeep this paragraph.\n[End of file]\n\nCompare them.",
                TextFilePaths = new List<string>
                {
                    "/safe/uploads/data.csv",
                    "/safe/uploads/notes.md",
                },
                TextFileNames = new List<string> { "data.csv", "notes.md" },
            },
        };

        List<ChatMessage> prepared = ChatHistoryPreparer.UseFileBackedCsvAttachments(
            history, Staged(("/safe/uploads/data.csv", "data.csv")));

        ChatMessage message = Assert.Single(prepared);
        Assert.DoesNotContain("a,b\n1,2", message.Content);
        Assert.Contains("[File: notes.md]\nKeep this paragraph.\n[End of file]", message.Content);
        Assert.EndsWith("Compare them.", message.Content, StringComparison.Ordinal);
        Assert.Equal(new[] { "/safe/uploads/notes.md" }, message.TextFilePaths);
        Assert.Equal(new[] { "notes.md" }, message.TextFileNames);
        Assert.True(ChatGenerationPipeline.HasTextFileAttachments(prepared));
    }

    [Fact]
    public void MetadataOnlyCsv_IsRestoredInlineWhenNoReadableWorkspaceIsAvailable()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-csv-fallback-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "stored.csv");
        File.WriteAllText(path, "name,total\nalice,17\nbob,23");
        try
        {
            var history = new List<ChatMessage>
            {
                new()
                {
                    Role = "user",
                    Content = "Please analyze this form.",
                    TextFilePaths = new List<string> { path },
                    TextFileNames = new List<string> { "responses.csv" },
                    HasFileBackedTextAttachments = true,
                },
            };

            List<ChatMessage> restored =
                ChatHistoryPreparer.RestoreUnstagedFileBackedCsvAttachments(history);

            ChatMessage message = Assert.Single(restored);
            Assert.Contains("[File: responses.csv]\nname,total\nalice,17\nbob,23\n[End of file]", message.Content);
            Assert.EndsWith("Please analyze this form.", message.Content, StringComparison.Ordinal);
            Assert.False(message.HasFileBackedTextAttachments);
            Assert.True(ChatGenerationPipeline.HasTextFileAttachments(restored));
            Assert.Equal("Please analyze this form.", history[0].Content);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void MetadataOnlyCsvFallback_PreservesDistinctFilesWithTheSameDisplayName()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-csv-fallback-collision-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string firstPath = Path.Combine(directory, "stored-one.csv");
        string secondPath = Path.Combine(directory, "stored-two.csv");
        File.WriteAllText(firstPath, "id,value\n1,FIRST_DISTINCT_UPLOAD");
        File.WriteAllText(secondPath, "id,value\n2,SECOND_DISTINCT_UPLOAD");
        try
        {
            var history = new List<ChatMessage>
            {
                new()
                {
                    Role = "user",
                    Content = "Compare both forms.",
                    TextFilePaths = new List<string> { firstPath, secondPath },
                    TextFileNames = new List<string> { "responses.csv", "responses.csv" },
                    HasFileBackedTextAttachments = true,
                },
            };

            ChatMessage restored = Assert.Single(
                ChatHistoryPreparer.RestoreUnstagedFileBackedCsvAttachments(history));

            Assert.Contains("FIRST_DISTINCT_UPLOAD", restored.Content, StringComparison.Ordinal);
            Assert.Contains("SECOND_DISTINCT_UPLOAD", restored.Content, StringComparison.Ordinal);
            Assert.Equal(2, restored.Content.Split("[File: responses.csv]", StringSplitOptions.None).Length - 1);
            Assert.EndsWith("Compare both forms.", restored.Content, StringComparison.Ordinal);
            Assert.False(restored.HasFileBackedTextAttachments);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ReuploadedCsvWithTheSameDisplayName_IsNeverCompactedAgainstStaleBytes()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-csv-collision-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string firstPath = Path.Combine(directory, "upload-one.csv");
        string secondPath = Path.Combine(directory, "upload-two.csv");
        File.WriteAllText(firstPath, "id,value\n1,FIRST_VERSION");
        File.WriteAllText(secondPath, "id,value\n1,SECOND_VERSION");
        try
        {
            ChatMessage Attached(string content, string path) => new()
            {
                Role = "user",
                Content = content,
                TextFilePaths = new List<string> { path },
                TextFileNames = new List<string> { "responses.csv" },
                HasFileBackedTextAttachments = true,
            };
            var history = new List<ChatMessage>
            {
                Attached("Analyze version one.", firstPath),
                new() { Role = "assistant", Content = "Done." },
                Attached("Now analyze the replacement.", secondPath),
            };

            // CollectCodeInputFiles keeps the first meaning of a workspace name (a later
            // upload with the same display name is staged under its own, numbered name).
            // The path-keyed map therefore authorizes only that exact source for
            // compaction, never every later upload with the same display name.
            List<ChatMessage> compacted = ChatHistoryPreparer.UseFileBackedCsvAttachments(
                history, Staged((firstPath, "responses.csv")));

            Assert.StartsWith("[Attached CSV available to tools: 'responses.csv']", compacted[0].Content);
            Assert.True(compacted[2].HasFileBackedTextAttachments);
            Assert.DoesNotContain("[Attached CSV available to tools:", compacted[2].Content);

            List<ChatMessage> safe =
                ChatHistoryPreparer.RestoreUnstagedFileBackedCsvAttachments(compacted);
            Assert.DoesNotContain("FIRST_VERSION", safe[0].Content);
            Assert.Contains("SECOND_VERSION", safe[2].Content);
            Assert.DoesNotContain("FIRST_VERSION", safe[2].Content);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
