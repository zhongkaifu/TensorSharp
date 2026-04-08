using System.Text.Json;
using InferenceEngine;
using InferenceWeb;

namespace InferenceWeb.Tests;

public class StructuredOutputTests
{
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
