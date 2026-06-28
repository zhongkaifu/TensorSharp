using System.Text;
using System.Text.Json;
using TensorSharp.Server.StreamingWriters;

namespace InferenceWeb.Tests;

public class StructuredOutputTests
{
    // Stream a model output in many small fragments through the json_object
    // streaming filter and return the concatenation actually sent to the client.
    private static string FeedInChunks(string modelOutput, int chunkSize = 3)
    {
        var filter = new StreamingJsonObjectFilter();
        var sb = new StringBuilder();
        for (int i = 0; i < modelOutput.Length; i += chunkSize)
            sb.Append(filter.Feed(modelOutput.Substring(i, System.Math.Min(chunkSize, modelOutput.Length - i))));
        return sb.ToString();
    }

    [Fact]
    public void StreamingJsonFilterStripsCodeFencesAndTrailingTags()
    {
        // Exactly the messy shape observed live (markdown fence + a leaked
        // Gemma channel tag after the object).
        string raw = "```json\n{\n  \"name\": \"Mars\",\n  \"diameter_km\": 6779,\n  \"has_moons\": true\n}\n```<channel|>";
        string streamed = FeedInChunks(raw);

        using var doc = JsonDocument.Parse(streamed); // must be valid JSON
        Assert.Equal("Mars", doc.RootElement.GetProperty("name").GetString());
        Assert.DoesNotContain("```", streamed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("channel", streamed, System.StringComparison.Ordinal);
    }

    [Fact]
    public void StreamingJsonFilterKeepsBracesInsideStrings()
    {
        string raw = "prefix {\"text\": \"a } b { c\", \"n\": 1} trailing";
        string streamed = FeedInChunks(raw, chunkSize: 1);

        using var doc = JsonDocument.Parse(streamed);
        Assert.Equal("a } b { c", doc.RootElement.GetProperty("text").GetString());
        Assert.Equal("""{"text": "a } b { c", "n": 1}""", streamed);
    }

    [Fact]
    public void StreamingJsonFilterStopsAtFirstBalancedObject()
    {
        var filter = new StreamingJsonObjectFilter();
        string emitted = filter.Feed("{\"a\":1}{\"b\":2}");
        Assert.Equal("""{"a":1}""", emitted);
        Assert.True(filter.Done);
        Assert.Equal("", filter.Feed("more text")); // nothing after close
    }


    [Fact]
    public void Qwen35NoThinkingTemplateKeepsPriorAnswerAsNextTurnPrefix()
    {
        const string jinjaTemplate = "{{ 'from-jinja' }}";

        var turn1 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the tallest mountain in the world?" }
        };
        string renderedTurn1 = ChatTemplate.RenderFromGgufTemplate(
            jinjaTemplate, turn1, addGenerationPrompt: true, architecture: "qwen35", enableThinking: false);

        const string answer = "Mount Everest";
        var turn2 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the tallest mountain in the world?" },
            new() { Role = "assistant", Content = answer },
            new() { Role = "user", Content = "How tall is it in meters?" }
        };
        string renderedTurn2 = ChatTemplate.RenderFromGgufTemplate(
            jinjaTemplate, turn2, addGenerationPrompt: true, architecture: "qwen35", enableThinking: false);

        Assert.DoesNotContain("from-jinja", renderedTurn1, StringComparison.Ordinal);
        Assert.StartsWith(renderedTurn1 + answer, renderedTurn2, StringComparison.Ordinal);
    }

    [Fact]
    public void ParserAcceptsDocumentedChatCompletionsJsonSchemaShape()
    {
        using var body = JsonDocument.Parse("""
        {
          "response_format": {
            "type": "json_schema",
            "json_schema": {
              "name": "research_paper_extraction",
              "strict": true,
              "schema": {
                "type": "object",
                "properties": {
                  "title": { "type": "string" }
                },
                "required": ["title"],
                "additionalProperties": false
              }
            }
          }
        }
        """);

        bool ok = OpenAIResponseFormatParser.TryParse(body.RootElement, out var format, out var error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.NotNull(format);
        Assert.Equal(StructuredOutputKind.JsonSchema, format!.Kind);
        Assert.Equal("research_paper_extraction", format.Name);
        Assert.True(format.Strict);
    }

    [Fact]
    public void JsonSchemaValidationRejectsRootAnyOfAndMissingRequired()
    {
        var format = StructuredOutputFormat.JsonSchema("bad_schema", """
        {
          "anyOf": [
            {
              "type": "object",
              "properties": {
                "answer": { "type": "string" }
              },
              "required": [],
              "additionalProperties": false
            }
          ]
        }
        """);

        var validation = StructuredOutputValidator.ValidateSchema(format);

        Assert.False(validation.IsValid);
        Assert.Contains(validation.Errors, e => e.Contains("root schema", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(validation.Errors, e => e.Contains("required", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void JsonSchemaValidationRequiresStrictAndAdditionalPropertiesFalse()
    {
        var format = StructuredOutputFormat.JsonSchema("weather", """
        {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
        """, strict: false);

        var validation = StructuredOutputValidator.ValidateSchema(format);

        Assert.False(validation.IsValid);
        Assert.Contains(validation.Errors, e => e.Contains("strict", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(validation.Errors, e => e.Contains("additionalProperties", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void JsonObjectNormalizationExtractsCodeFencedJson()
    {
        var normalized = StructuredOutputValidator.NormalizeOutput("""
        Here you go:

        ```json
        {
          "answer": 5
        }
        ```
        """, StructuredOutputFormat.JsonObject());

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"answer":5}""", normalized.NormalizedContent);
    }

    [Fact]
    public void JsonSchemaNormalizationDropsExtrasFillsNullableFieldsAndPreservesSchemaOrder()
    {
        var format = StructuredOutputFormat.JsonSchema("result", """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" },
            "optional_note": { "type": ["string", "null"] },
            "done": { "type": "boolean" }
          },
          "required": ["answer", "optional_note", "done"],
          "additionalProperties": false
        }
        """);

        var normalized = StructuredOutputValidator.NormalizeOutput("""
        {
          "done": true,
          "extra": "remove me",
          "answer": "ok"
        }
        """, format);

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"answer":"ok","optional_note":null,"done":true}""", normalized.NormalizedContent);
    }

    [Fact]
    public void JsonSchemaNormalizationSupportsDefsAndAnyOf()
    {
        var format = StructuredOutputFormat.JsonSchema("container", """
        {
          "type": "object",
          "properties": {
            "item": {
              "anyOf": [
                { "$ref": "#/$defs/person" },
                {
                  "type": "object",
                  "properties": {
                    "city": { "type": "string" }
                  },
                  "required": ["city"],
                  "additionalProperties": false
                }
              ]
            }
          },
          "$defs": {
            "person": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
              },
              "required": ["name", "age"],
              "additionalProperties": false
            }
          },
          "required": ["item"],
          "additionalProperties": false
        }
        """);

        var normalized = StructuredOutputValidator.NormalizeOutput("""
        {
          "item": {
            "age": 30,
            "name": "Ada",
            "ignored": true
          }
        }
        """, format);

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"item":{"name":"Ada","age":30}}""", normalized.NormalizedContent);
    }
}


