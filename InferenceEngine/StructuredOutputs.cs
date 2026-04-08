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
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace InferenceEngine
{
    public enum StructuredOutputKind
    {
        JsonObject,
        JsonSchema
    }

    public sealed class StructuredOutputFormat
    {
        private StructuredOutputFormat(StructuredOutputKind kind, string name = null,
            string schemaJson = null, bool strict = false, string description = null)
        {
            Kind = kind;
            Name = name;
            SchemaJson = schemaJson;
            Strict = strict;
            Description = description;
        }

        public StructuredOutputKind Kind { get; }
        public string Name { get; }
        public string SchemaJson { get; }
        public bool Strict { get; }
        public string Description { get; }

        public static StructuredOutputFormat JsonObject()
            => new(StructuredOutputKind.JsonObject);

        public static StructuredOutputFormat JsonSchema(string name, string schemaJson,
            bool strict = true, string description = null)
            => new(StructuredOutputKind.JsonSchema, name, schemaJson, strict, description);
    }

    public static class StructuredOutputPrompt
    {
        public static List<ChatMessage> Apply(List<ChatMessage> messages, StructuredOutputFormat format)
        {
            if (format == null)
                return messages;

            var result = new List<ChatMessage>(messages?.Count + 1 ?? 1);
            string instruction = BuildInstruction(format);

            if (messages != null && messages.Count > 0 &&
                (messages[0].Role == "system" || messages[0].Role == "developer"))
            {
                var first = CloneMessage(messages[0]);
                first.Content = string.IsNullOrWhiteSpace(first.Content)
                    ? instruction
                    : first.Content.TrimEnd() + "\n\n" + instruction;
                result.Add(first);

                for (int i = 1; i < messages.Count; i++)
                    result.Add(CloneMessage(messages[i]));
                return result;
            }

            result.Add(new ChatMessage { Role = "system", Content = instruction });
            if (messages != null)
            {
                foreach (var msg in messages)
                    result.Add(CloneMessage(msg));
            }
            return result;
        }

        public static string BuildInstruction(StructuredOutputFormat format)
        {
            if (format == null)
                return "";

            if (format.Kind == StructuredOutputKind.JsonObject)
            {
                return "You must answer with valid JSON. Return exactly one JSON object and nothing else. " +
                       "Do not include Markdown code fences, explanations, or extra text before or after the JSON.";
            }

            var sb = new StringBuilder();
            sb.Append("You must answer with valid JSON that matches the provided JSON Schema exactly. ");
            sb.Append("Return exactly one JSON object and nothing else. ");
            sb.Append("Do not include Markdown code fences, explanations, or extra text before or after the JSON.");
            sb.Append("\n\nStructured output schema name: ");
            sb.Append(format.Name);

            if (!string.IsNullOrWhiteSpace(format.Description))
            {
                sb.Append("\nSchema description: ");
                sb.Append(format.Description.Trim());
            }

            sb.Append("\nRules:");
            sb.Append("\n- Include every required property.");
            sb.Append("\n- Do not add properties that are not defined in the schema.");
            sb.Append("\n- Use `null` only when the schema allows it.");
            sb.Append("\n- Keep property order aligned with the schema.");
            sb.Append("\n\nJSON Schema:");
            sb.Append("\n");
            sb.Append(format.SchemaJson);
            return sb.ToString();
        }

        private static ChatMessage CloneMessage(ChatMessage msg)
        {
            if (msg == null)
                return null;

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = msg.ImagePaths != null ? new List<string>(msg.ImagePaths) : null,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls != null ? new List<ToolCall>(msg.ToolCalls) : null,
                Thinking = msg.Thinking
            };
        }
    }

    public sealed class StructuredOutputSchemaValidationResult
    {
        public bool IsValid { get; init; }
        public List<string> Errors { get; init; } = new();

        public string ErrorMessage
            => Errors.Count == 0 ? null : string.Join("; ", Errors);
    }

    public sealed class StructuredOutputNormalizationResult
    {
        public bool IsValid { get; init; }
        public string NormalizedContent { get; init; }
        public List<string> Errors { get; init; } = new();

        public string ErrorMessage
            => Errors.Count == 0 ? null : string.Join("; ", Errors);
    }

    public static class StructuredOutputValidator
    {
        private static readonly HashSet<string> SupportedPrimitiveTypes = new(StringComparer.Ordinal)
        {
            "string", "number", "integer", "boolean", "object", "array", "null"
        };

        private static readonly HashSet<string> UnsupportedKeywords = new(StringComparer.Ordinal)
        {
            "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
            "minLength", "maxLength", "pattern", "format",
            "minimum", "maximum", "multipleOf",
            "patternProperties",
            "minItems", "maxItems"
        };

        public static StructuredOutputSchemaValidationResult ValidateSchema(StructuredOutputFormat format)
        {
            var result = new StructuredOutputSchemaValidationResult();
            if (format == null || format.Kind == StructuredOutputKind.JsonObject)
                return new StructuredOutputSchemaValidationResult { IsValid = true };

            bool missingCriticalField = false;
            if (string.IsNullOrWhiteSpace(format.Name))
            {
                result.Errors.Add("response_format.json_schema.name is required.");
                missingCriticalField = true;
            }
            if (string.IsNullOrWhiteSpace(format.SchemaJson))
            {
                result.Errors.Add("response_format.json_schema.schema is required.");
                missingCriticalField = true;
            }
            if (!format.Strict)
                result.Errors.Add("response_format.json_schema.strict must be true.");

            if (missingCriticalField)
                return result;

            JsonDocument schemaDoc;
            try
            {
                schemaDoc = JsonDocument.Parse(format.SchemaJson);
            }
            catch (Exception ex)
            {
                result.Errors.Add("response_format.json_schema.schema is not valid JSON: " + ex.Message);
                return result;
            }

            using (schemaDoc)
            {
                var root = schemaDoc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                {
                    result.Errors.Add("Structured outputs require the schema root to be a JSON object.");
                    return result;
                }

                if (root.TryGetProperty("anyOf", out _))
                    result.Errors.Add("Structured outputs do not allow `anyOf` at the root schema.");

                if (!SchemaAllowsObject(root, root))
                    result.Errors.Add("Structured outputs require the schema root to describe a JSON object.");

                var ctx = new SchemaValidationContext(root, result.Errors);
                ValidateSchemaNode(root, "$", 1, true, ctx);

                if (ctx.TotalPropertyCount > 5000)
                    result.Errors.Add("Structured outputs allow at most 5000 object properties across the schema.");
                if (ctx.TotalEnumValues > 1000)
                    result.Errors.Add("Structured outputs allow at most 1000 enum values across the schema.");
                if (ctx.TotalStringBytes > 120000)
                    result.Errors.Add("Structured outputs allow at most 120000 total characters across names, enum values, and const values.");
            }

            return new StructuredOutputSchemaValidationResult
            {
                IsValid = result.Errors.Count == 0,
                Errors = result.Errors
            };
        }

        public static StructuredOutputNormalizationResult NormalizeOutput(string rawOutput, StructuredOutputFormat format)
        {
            if (format == null)
            {
                return new StructuredOutputNormalizationResult
                {
                    IsValid = true,
                    NormalizedContent = rawOutput ?? ""
                };
            }

            if (string.IsNullOrWhiteSpace(rawOutput))
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string> { "The model returned an empty response." }
                };
            }

            if (format.Kind == StructuredOutputKind.JsonObject)
                return NormalizeJsonObject(rawOutput);

            var schemaValidation = ValidateSchema(format);
            if (!schemaValidation.IsValid)
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string>(schemaValidation.Errors)
                };
            }

            if (!TryExtractJsonObject(rawOutput, out string candidateJson, out string extractError))
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string> { extractError }
                };
            }

            try
            {
                using var valueDoc = JsonDocument.Parse(candidateJson);
                using var schemaDoc = JsonDocument.Parse(format.SchemaJson);

                var errors = new List<string>();
                if (!TryNormalizeValue(valueDoc.RootElement, schemaDoc.RootElement, schemaDoc.RootElement,
                    "$", 1, errors, out JsonNode normalizedNode))
                {
                    return new StructuredOutputNormalizationResult { Errors = errors };
                }

                if (normalizedNode is not JsonObject)
                {
                    return new StructuredOutputNormalizationResult
                    {
                        Errors = new List<string> { "Structured outputs require a JSON object response." }
                    };
                }

                return new StructuredOutputNormalizationResult
                {
                    IsValid = true,
                    NormalizedContent = normalizedNode.ToJsonString(new JsonSerializerOptions { WriteIndented = false })
                };
            }
            catch (Exception ex)
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string> { "The model response could not be parsed as JSON: " + ex.Message }
                };
            }
        }

        private static StructuredOutputNormalizationResult NormalizeJsonObject(string rawOutput)
        {
            if (!TryExtractJsonObject(rawOutput, out string candidateJson, out string extractError))
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string> { extractError }
                };
            }

            try
            {
                using var doc = JsonDocument.Parse(candidateJson);
                if (doc.RootElement.ValueKind != JsonValueKind.Object)
                {
                    return new StructuredOutputNormalizationResult
                    {
                        Errors = new List<string> { "response_format.type=json_object requires the model to return a JSON object." }
                    };
                }

                return new StructuredOutputNormalizationResult
                {
                    IsValid = true,
                    NormalizedContent = JsonSerializer.Serialize(doc.RootElement)
                };
            }
            catch (Exception ex)
            {
                return new StructuredOutputNormalizationResult
                {
                    Errors = new List<string> { "The model response could not be parsed as JSON: " + ex.Message }
                };
            }
        }

        private static void ValidateSchemaNode(JsonElement schema, string path, int depth,
            bool isRoot, SchemaValidationContext ctx)
        {
            if (ctx.Errors.Count >= 16)
                return;

            if (depth > 10)
            {
                ctx.Errors.Add($"Schema nesting exceeds the supported depth of 10 at {path}.");
                return;
            }

            RegisterUnsupportedKeywords(schema, path, ctx);
            RegisterEnumAndConstStats(schema, path, ctx);

            if (schema.TryGetProperty("$ref", out var refEl))
            {
                string refPath = refEl.GetString();
                if (!TryResolveRef(ctx.RootSchema, refPath, out JsonElement target, out string refError))
                {
                    ctx.Errors.Add($"{path}: {refError}");
                    return;
                }

                if (ctx.ActiveRefs.Contains(refPath))
                    return;

                ctx.ActiveRefs.Add(refPath);
                ValidateSchemaNode(target, $"{path}->$ref({refPath})", depth + 1, false, ctx);
                ctx.ActiveRefs.Remove(refPath);
                return;
            }

            if (schema.TryGetProperty("anyOf", out var anyOfEl))
            {
                if (isRoot)
                    ctx.Errors.Add("Structured outputs do not allow `anyOf` at the root schema.");

                if (anyOfEl.ValueKind != JsonValueKind.Array || anyOfEl.GetArrayLength() == 0)
                {
                    ctx.Errors.Add($"{path}: `anyOf` must be a non-empty array.");
                    return;
                }

                int idx = 0;
                foreach (var variant in anyOfEl.EnumerateArray())
                {
                    ValidateSchemaNode(variant, $"{path}.anyOf[{idx}]", depth + 1, false, ctx);
                    idx++;
                }
            }

            var types = ReadTypeList(schema, path, ctx);
            bool objectLike = SchemaIsObjectLike(schema, ctx.RootSchema);
            bool arrayLike = schema.TryGetProperty("items", out _) || types.Contains("array");

            if (objectLike)
            {
                if (!schema.TryGetProperty("additionalProperties", out var apEl) ||
                    apEl.ValueKind != JsonValueKind.False)
                {
                    ctx.Errors.Add($"{path}: `additionalProperties: false` is required for every object schema.");
                }

                var propertyNames = new HashSet<string>(StringComparer.Ordinal);
                if (schema.TryGetProperty("properties", out var propsEl))
                {
                    if (propsEl.ValueKind != JsonValueKind.Object)
                    {
                        ctx.Errors.Add($"{path}: `properties` must be an object.");
                    }
                    else
                    {
                        foreach (var prop in propsEl.EnumerateObject())
                        {
                            propertyNames.Add(prop.Name);
                            ctx.TotalPropertyCount++;
                            ctx.TotalStringBytes += prop.Name.Length;
                            ValidateSchemaNode(prop.Value, $"{path}.properties.{prop.Name}", depth + 1, false, ctx);
                        }
                    }
                }

                if (!schema.TryGetProperty("required", out var requiredEl) || requiredEl.ValueKind != JsonValueKind.Array)
                {
                    ctx.Errors.Add($"{path}: `required` must be an array listing every property.");
                }
                else
                {
                    var requiredNames = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var item in requiredEl.EnumerateArray())
                    {
                        if (item.ValueKind == JsonValueKind.String && item.GetString() is string reqName)
                            requiredNames.Add(reqName);
                    }

                    foreach (var propName in propertyNames)
                    {
                        if (!requiredNames.Contains(propName))
                            ctx.Errors.Add($"{path}: property `{propName}` must be listed in `required`.");
                    }

                    foreach (var reqName in requiredNames)
                    {
                        if (!propertyNames.Contains(reqName))
                            ctx.Errors.Add($"{path}: `required` contains unknown property `{reqName}`.");
                    }
                }
            }

            if (arrayLike)
            {
                if (!schema.TryGetProperty("items", out var itemsEl))
                    ctx.Errors.Add($"{path}: array schemas must define `items`.");
                else
                    ValidateSchemaNode(itemsEl, $"{path}.items", depth + 1, false, ctx);
            }

            if (schema.TryGetProperty("$defs", out var defsEl))
            {
                if (defsEl.ValueKind != JsonValueKind.Object)
                {
                    ctx.Errors.Add($"{path}: `$defs` must be an object.");
                }
                else
                {
                    foreach (var def in defsEl.EnumerateObject())
                    {
                        ctx.TotalStringBytes += def.Name.Length;
                        ValidateSchemaNode(def.Value, $"{path}.$defs.{def.Name}", depth + 1, false, ctx);
                    }
                }
            }
        }

        private static void RegisterUnsupportedKeywords(JsonElement schema, string path, SchemaValidationContext ctx)
        {
            foreach (var prop in schema.EnumerateObject())
            {
                if (UnsupportedKeywords.Contains(prop.Name))
                    ctx.Errors.Add($"{path}: `{prop.Name}` is not supported by structured outputs.");
            }
        }

        private static void RegisterEnumAndConstStats(JsonElement schema, string path, SchemaValidationContext ctx)
        {
            if (schema.TryGetProperty("enum", out var enumEl))
            {
                if (enumEl.ValueKind != JsonValueKind.Array)
                {
                    ctx.Errors.Add($"{path}: `enum` must be an array.");
                    return;
                }

                foreach (var item in enumEl.EnumerateArray())
                {
                    ctx.TotalEnumValues++;
                    if (item.ValueKind == JsonValueKind.String)
                        ctx.TotalStringBytes += item.GetString()?.Length ?? 0;
                }
            }

            if (schema.TryGetProperty("const", out var constEl) && constEl.ValueKind == JsonValueKind.String)
                ctx.TotalStringBytes += constEl.GetString()?.Length ?? 0;
        }

        private static HashSet<string> ReadTypeList(JsonElement schema, string path, SchemaValidationContext ctx)
        {
            var types = new HashSet<string>(StringComparer.Ordinal);
            if (!schema.TryGetProperty("type", out var typeEl))
                return types;

            if (typeEl.ValueKind == JsonValueKind.String)
            {
                string typeName = typeEl.GetString();
                if (SupportedPrimitiveTypes.Contains(typeName))
                    types.Add(typeName);
                else
                    ctx.Errors.Add($"{path}: unsupported type `{typeName}`.");
                return types;
            }

            if (typeEl.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in typeEl.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.String)
                    {
                        ctx.Errors.Add($"{path}: `type` arrays may only contain strings.");
                        continue;
                    }

                    string typeName = item.GetString();
                    if (SupportedPrimitiveTypes.Contains(typeName))
                        types.Add(typeName);
                    else
                        ctx.Errors.Add($"{path}: unsupported type `{typeName}`.");
                }
                return types;
            }

            ctx.Errors.Add($"{path}: `type` must be a string or array of strings.");
            return types;
        }

        private static bool TryNormalizeValue(JsonElement value, JsonElement schema, JsonElement rootSchema,
            string path, int depth, List<string> errors, out JsonNode normalized)
        {
            normalized = null;
            if (depth > 128)
            {
                errors.Add($"{path}: response exceeded the supported normalization depth.");
                return false;
            }

            if (schema.TryGetProperty("$ref", out var refEl))
            {
                string refPath = refEl.GetString();
                if (!TryResolveRef(rootSchema, refPath, out JsonElement target, out string refError))
                {
                    errors.Add($"{path}: {refError}");
                    return false;
                }
                return TryNormalizeValue(value, target, rootSchema, path, depth + 1, errors, out normalized);
            }

            if (schema.TryGetProperty("anyOf", out var anyOfEl))
            {
                foreach (var variant in anyOfEl.EnumerateArray())
                {
                    var variantErrors = new List<string>();
                    if (TryNormalizeValue(value, variant, rootSchema, path, depth + 1, variantErrors, out normalized))
                        return true;
                }

                errors.Add($"{path}: value did not match any schema in `anyOf`.");
                return false;
            }

            var typeCtx = new SchemaValidationContext(rootSchema, new List<string>());
            var types = ReadTypeList(schema, path, typeCtx);

            if (types.Contains("null") && value.ValueKind == JsonValueKind.Null)
            {
                normalized = null;
                return true;
            }

            if (SchemaMatchesObject(value, schema, rootSchema, path, depth, errors, out normalized))
                return CheckEnumAndConst(schema, normalized, errors, path, allowNullFallback: true);

            if (SchemaMatchesArray(value, schema, rootSchema, path, depth, errors, out normalized))
                return CheckEnumAndConst(schema, normalized, errors, path, allowNullFallback: true);

            if (TryNormalizePrimitive(value, types, path, errors, out normalized))
                return CheckEnumAndConst(schema, normalized, errors, path, allowNullFallback: true);

            if (typeCtx.Errors.Count > 0)
                errors.AddRange(typeCtx.Errors);
            return false;
        }

        private static bool SchemaMatchesObject(JsonElement value, JsonElement schema, JsonElement rootSchema,
            string path, int depth, List<string> errors, out JsonNode normalized)
        {
            normalized = null;
            bool objectLike = SchemaIsObjectLike(schema, rootSchema);
            if (!objectLike)
                return false;

            if (value.ValueKind != JsonValueKind.Object)
            {
                errors.Add($"{path}: expected an object.");
                return false;
            }

            var result = new JsonObject();
            if (!schema.TryGetProperty("properties", out var propsEl) || propsEl.ValueKind != JsonValueKind.Object)
            {
                normalized = result;
                return true;
            }

            foreach (var prop in propsEl.EnumerateObject())
            {
                if (value.TryGetProperty(prop.Name, out var childValue))
                {
                    if (!TryNormalizeValue(childValue, prop.Value, rootSchema, $"{path}.{prop.Name}", depth + 1, errors, out JsonNode childNode))
                        return false;
                    result[prop.Name] = childNode;
                }
                else if (AllowsNull(prop.Value, rootSchema))
                {
                    result[prop.Name] = null;
                }
                else
                {
                    errors.Add($"{path}: missing required property `{prop.Name}`.");
                    return false;
                }
            }

            normalized = result;
            return true;
        }

        private static bool SchemaMatchesArray(JsonElement value, JsonElement schema, JsonElement rootSchema,
            string path, int depth, List<string> errors, out JsonNode normalized)
        {
            normalized = null;
            bool arrayLike = schema.TryGetProperty("items", out var itemsEl)
                || (schema.TryGetProperty("type", out var typeEl) &&
                    ((typeEl.ValueKind == JsonValueKind.String && typeEl.GetString() == "array") ||
                     (typeEl.ValueKind == JsonValueKind.Array && ArrayContainsType(typeEl, "array"))));
            if (!arrayLike)
                return false;

            if (value.ValueKind != JsonValueKind.Array)
            {
                errors.Add($"{path}: expected an array.");
                return false;
            }

            if (!schema.TryGetProperty("items", out itemsEl))
            {
                normalized = JsonNode.Parse(value.GetRawText());
                return true;
            }

            var result = new JsonArray();
            int index = 0;
            foreach (var item in value.EnumerateArray())
            {
                if (!TryNormalizeValue(item, itemsEl, rootSchema, $"{path}[{index}]", depth + 1, errors, out JsonNode childNode))
                    return false;
                result.Add(childNode);
                index++;
            }

            normalized = result;
            return true;
        }

        private static bool TryNormalizePrimitive(JsonElement value, HashSet<string> types,
            string path, List<string> errors, out JsonNode normalized)
        {
            normalized = null;

            if (types.Count == 0)
            {
                if (value.ValueKind == JsonValueKind.Null)
                    return true;
                normalized = JsonNode.Parse(value.GetRawText());
                return normalized != null;
            }

            if (types.Contains("string") && value.ValueKind == JsonValueKind.String)
            {
                normalized = JsonValue.Create(value.GetString());
                return true;
            }

            if (types.Contains("boolean") &&
                (value.ValueKind == JsonValueKind.True || value.ValueKind == JsonValueKind.False))
            {
                normalized = JsonValue.Create(value.GetBoolean());
                return true;
            }

            if (types.Contains("integer") && value.ValueKind == JsonValueKind.Number)
            {
                if (value.TryGetInt64(out long asInt))
                {
                    normalized = JsonValue.Create(asInt);
                    return true;
                }

                if (value.TryGetDouble(out double asDouble) && Math.Abs(asDouble - Math.Round(asDouble)) < 1e-9)
                {
                    normalized = JsonValue.Create((long)Math.Round(asDouble));
                    return true;
                }

                errors.Add($"{path}: expected an integer.");
                return false;
            }

            if (types.Contains("number") && value.ValueKind == JsonValueKind.Number)
            {
                normalized = JsonNode.Parse(value.GetRawText());
                return normalized != null;
            }

            if (types.Contains("null") && value.ValueKind == JsonValueKind.Null)
            {
                normalized = null;
                return true;
            }

            errors.Add($"{path}: value does not match the schema type.");
            return false;
        }

        private static bool CheckEnumAndConst(JsonElement schema, JsonNode normalized,
            List<string> errors, string path, bool allowNullFallback)
        {
            if (normalized == null && allowNullFallback)
                return true;

            if (schema.TryGetProperty("const", out var constEl))
            {
                JsonNode constNode = JsonNode.Parse(constEl.GetRawText());
                if (!JsonNode.DeepEquals(normalized, constNode))
                {
                    errors.Add($"{path}: value does not match the schema const.");
                    return false;
                }
            }

            if (schema.TryGetProperty("enum", out var enumEl))
            {
                bool matched = false;
                foreach (var enumValue in enumEl.EnumerateArray())
                {
                    JsonNode enumNode = JsonNode.Parse(enumValue.GetRawText());
                    if (JsonNode.DeepEquals(normalized, enumNode))
                    {
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    errors.Add($"{path}: value is not in the schema enum.");
                    return false;
                }
            }

            return true;
        }

        private static bool SchemaAllowsObject(JsonElement schema, JsonElement rootSchema)
        {
            if (schema.TryGetProperty("$ref", out var refEl))
            {
                return TryResolveRef(rootSchema, refEl.GetString(), out JsonElement target, out _)
                    && SchemaAllowsObject(target, rootSchema);
            }

            if (schema.TryGetProperty("type", out var typeEl))
            {
                if (typeEl.ValueKind == JsonValueKind.String)
                    return string.Equals(typeEl.GetString(), "object", StringComparison.Ordinal);

                if (typeEl.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in typeEl.EnumerateArray())
                    {
                        if (item.ValueKind == JsonValueKind.String &&
                            string.Equals(item.GetString(), "object", StringComparison.Ordinal))
                            return true;
                    }
                }
            }

            return schema.TryGetProperty("properties", out _)
                || schema.TryGetProperty("additionalProperties", out _);
        }

        private static bool SchemaIsObjectLike(JsonElement schema, JsonElement rootSchema)
        {
            if (schema.TryGetProperty("$ref", out var refEl) &&
                TryResolveRef(rootSchema, refEl.GetString(), out JsonElement target, out _))
            {
                return SchemaIsObjectLike(target, rootSchema);
            }

            return schema.TryGetProperty("properties", out _)
                || schema.TryGetProperty("additionalProperties", out _)
                || (schema.TryGetProperty("type", out var typeEl) &&
                    ((typeEl.ValueKind == JsonValueKind.String && typeEl.GetString() == "object") ||
                     (typeEl.ValueKind == JsonValueKind.Array && ArrayContainsType(typeEl, "object"))));
        }

        private static bool AllowsNull(JsonElement schema, JsonElement rootSchema)
        {
            if (schema.TryGetProperty("$ref", out var refEl))
            {
                return TryResolveRef(rootSchema, refEl.GetString(), out JsonElement target, out _)
                    && AllowsNull(target, rootSchema);
            }

            if (schema.TryGetProperty("type", out var typeEl))
            {
                if (typeEl.ValueKind == JsonValueKind.String)
                    return typeEl.GetString() == "null";
                if (typeEl.ValueKind == JsonValueKind.Array)
                    return ArrayContainsType(typeEl, "null");
            }

            if (schema.TryGetProperty("anyOf", out var anyOfEl))
            {
                foreach (var variant in anyOfEl.EnumerateArray())
                {
                    if (AllowsNull(variant, rootSchema))
                        return true;
                }
            }

            return false;
        }

        private static bool ArrayContainsType(JsonElement typeArray, string expected)
        {
            foreach (var item in typeArray.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.String &&
                    string.Equals(item.GetString(), expected, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        private static bool TryResolveRef(JsonElement rootSchema, string refPath,
            out JsonElement target, out string error)
        {
            target = default;
            error = null;

            if (string.IsNullOrWhiteSpace(refPath))
            {
                error = "Encountered an empty `$ref`.";
                return false;
            }

            if (refPath == "#")
            {
                target = rootSchema;
                return true;
            }

            if (!refPath.StartsWith("#/", StringComparison.Ordinal))
            {
                error = $"Only local refs are supported, but received `{refPath}`.";
                return false;
            }

            JsonElement current = rootSchema;
            string[] segments = refPath.Substring(2).Split('/');
            foreach (var rawSegment in segments)
            {
                string segment = rawSegment.Replace("~1", "/").Replace("~0", "~");
                if (current.ValueKind != JsonValueKind.Object || !current.TryGetProperty(segment, out current))
                {
                    error = $"Could not resolve `$ref` path `{refPath}`.";
                    return false;
                }
            }

            target = current;
            return true;
        }

        private static bool TryExtractJsonObject(string rawOutput, out string json, out string error)
        {
            json = null;
            error = null;

            string trimmed = rawOutput.Trim();
            if (TryParseCandidate(trimmed, out json))
                return true;

            var fenceMatches = Regex.Matches(trimmed, "```(?:json)?\\s*(.*?)\\s*```",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);
            foreach (Match match in fenceMatches)
            {
                if (match.Groups.Count > 1 && TryParseCandidate(match.Groups[1].Value.Trim(), out json))
                    return true;
            }

            for (int i = 0; i < trimmed.Length; i++)
            {
                if (trimmed[i] != '{')
                    continue;

                if (TryReadBalancedObject(trimmed, i, out string candidate) && TryParseCandidate(candidate, out json))
                    return true;
            }

            error = "The model response did not contain a valid JSON object.";
            return false;
        }

        private static bool TryParseCandidate(string candidate, out string json)
        {
            json = null;
            if (string.IsNullOrWhiteSpace(candidate))
                return false;

            try
            {
                using var doc = JsonDocument.Parse(candidate);
                if (doc.RootElement.ValueKind != JsonValueKind.Object)
                    return false;
                json = JsonSerializer.Serialize(doc.RootElement);
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static bool TryReadBalancedObject(string text, int startIndex, out string json)
        {
            json = null;
            bool inString = false;
            bool escaping = false;
            int depth = 0;

            for (int i = startIndex; i < text.Length; i++)
            {
                char ch = text[i];
                if (escaping)
                {
                    escaping = false;
                    continue;
                }

                if (ch == '\\' && inString)
                {
                    escaping = true;
                    continue;
                }

                if (ch == '"')
                {
                    inString = !inString;
                    continue;
                }

                if (inString)
                    continue;

                if (ch == '{')
                    depth++;
                else if (ch == '}')
                    depth--;

                if (depth == 0)
                {
                    json = text.Substring(startIndex, i - startIndex + 1);
                    return true;
                }
            }

            return false;
        }

        private sealed class SchemaValidationContext
        {
            public SchemaValidationContext(JsonElement rootSchema, List<string> errors)
            {
                RootSchema = rootSchema;
                Errors = errors;
            }

            public JsonElement RootSchema { get; }
            public List<string> Errors { get; }
            public HashSet<string> ActiveRefs { get; } = new(StringComparer.Ordinal);
            public int TotalPropertyCount { get; set; }
            public int TotalEnumValues { get; set; }
            public int TotalStringBytes { get; set; }
        }
    }
}
