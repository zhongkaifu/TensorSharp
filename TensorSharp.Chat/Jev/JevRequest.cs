// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace TensorSharp.Server.Jev;

/// <summary>A validated Jev decision request. Labels are evaluated at temperature one.</summary>
public sealed record JevQuestion(string Id, string Type, string Instructions,
    string[] Names, string?[] Descriptions, string[] Labels, JsonElement[]? Levels = null);

/// <summary>One decoded inline image from a request, before it is written to media storage.
/// The format is sniffed from the bytes, never taken from the declared media type.</summary>
public sealed record JevImage(byte[] Bytes, string Format);

public sealed class JevValidationException(string message) : ArgumentException(message);
public sealed class JevQueueFullException() : InvalidOperationException("The Jev inference queue is full; retry later.");
public sealed class JevModelUnavailableException(string message) : InvalidOperationException(message);
public sealed class JevModelNotFoundException(string message) : ArgumentException(message);

public sealed record JevRequest(string? Model, string State, string? Instructions,
    JevQuestion[] Questions, int Samples, int AutoMax, double AutoThreshold, int Seed,
    int? ChunkRows, bool SharedPrompt, JevImage[] Images)
{
    /// <summary>File, document, audio and video inputs, decoded or referring to upload storage.</summary>
    public JevAttachment[] Attachments { get; init; } = [];

    public const int MaxQuestions = 64;
    public const int MaxSamples = 32;
    /// <summary>Images per request. Each one costs its encoder's own soft-token count —
    /// up to 280 rows for this tower, fewer for a small picture — in every chunk prompt,
    /// so the ceiling is deliberately low; the context check rejects whatever does not
    /// fit regardless.</summary>
    public const int MaxImages = 8;

    /// <summary>Parse the public /v1/systemone JSON contract without retaining document storage.</summary>
    public static JevRequest Parse(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object) throw Error("request must be an object");
        Unique(root, "request");
        var supportedKeys = new HashSet<string>(["model", "state", "instructions", "questions", "samples", "auto_max", "auto_threshold",
            "seed", "chunk_rows", "chunk_prompt", "steps", "think", "images", "files", "documents", "videos", "audios", "ask", "sequential"], StringComparer.Ordinal);
        foreach (var property in root.EnumerateObject())
        {
            if (property.Name is "audio" or "input_audio" or "video")
                throw Error($"'{property.Name}' is not supported; use the 'audios' or 'videos' attachment array.");
            if (!supportedKeys.Contains(property.Name)) throw Error($"unsupported request field '{property.Name}'");
        }
        if (!root.TryGetProperty("state", out var state) || state.ValueKind == JsonValueKind.Null)
            throw Error("state: required");
        if (state.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
            throw Error("state must be a string, object, or array");
        var images = new List<JevImage>();
        if (root.TryGetProperty("images", out var attachments) && attachments.ValueKind != JsonValueKind.Null)
        {
            if (attachments.ValueKind != JsonValueKind.Array)
                throw Error("images must be an array of inline images");
            foreach (var attachment in attachments.EnumerateArray())
            {
                if (images.Count == MaxImages) throw Error($"images: at most {MaxImages} images per request");
                images.Add(JevImageInput.Decode(attachment, images.Count));
            }
        }
        var fileAttachments = JevAttachmentInput.Parse(root, images.Sum(image => (long)image.Bytes.Length));
        string? model = OptionalString(root, "model");
        if (model != null && string.IsNullOrWhiteSpace(model)) throw Error("model must not be empty");
        if (!root.TryGetProperty("questions", out var questions) || questions.ValueKind != JsonValueKind.Object)
            throw Error("questions: needs a non-empty map of id -> question");
        Unique(questions, "questions");
        var parsed = new List<JevQuestion>();
        foreach (var property in questions.EnumerateObject())
        {
            string id = property.Name;
            if (string.IsNullOrWhiteSpace(id) || id != id.Trim() || id.Length > 128 || id.Any(char.IsControl) || id.Contains(':'))
                throw Error("question ids must be 1 to 128 characters, without ':' or control characters or surrounding whitespace");
            var q = property.Value;
            if (q.ValueKind != JsonValueKind.Object) throw Error($"question '{id}': must be an object");
            Unique(q, $"question '{id}'");
            foreach (var field in q.EnumerateObject())
                if (field.Name is not ("type" or "instructions" or "criteria" or "depends_on" or "ask_if" or "alone"))
                    throw Error($"question '{id}': unsupported field '{field.Name}'");
            foreach (string unsupported in new[] { "depends_on", "ask_if", "alone" })
                if (q.TryGetProperty(unsupported, out _)) throw Error($"question '{id}': '{unsupported}' is not supported");
            string type = OptionalString(q, "type") ?? throw Error($"question '{id}': type is required");
            string instructions = Text(q, "instructions") ?? "";
            q.TryGetProperty("criteria", out var criteria);
            var names = new List<string>();
            var descriptions = new List<string?>();
            JsonElement[]? levels = null;
            if (type == "noul")
            {
                if (criteria.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.Object))
                    throw Error($"question '{id}': noul criteria must be an object with true and false descriptions");
                if (criteria.ValueKind == JsonValueKind.Object)
                {
                    Unique(criteria, $"question '{id}' criteria");
                    if (criteria.EnumerateObject().Any(p => p.Name is not ("true" or "false")))
                        throw Error($"question '{id}': noul criteria keys must be true or false");
                }
                names.AddRange(["yes", "no"]);
                descriptions.Add(criteria.ValueKind == JsonValueKind.Object ? Text(criteria, "true") : null);
                descriptions.Add(criteria.ValueKind == JsonValueKind.Object ? Text(criteria, "false") : null);
            }
            else if (type == "choice")
            {
                if (criteria.ValueKind != JsonValueKind.Object) throw Error($"question '{id}': choice criteria must map names to descriptions");
                Unique(criteria, $"question '{id}' criteria");
                foreach (var option in criteria.EnumerateObject())
                {
                    if (string.IsNullOrWhiteSpace(option.Name)) throw Error($"question '{id}': choice names must not be empty");
                    names.Add(option.Name);
                    descriptions.Add(JsonText(option.Value));
                }
            }
            else if (type == "score")
            {
                if (criteria.ValueKind != JsonValueKind.Array) throw Error($"question '{id}': score criteria must be an ordered array of levels");
                levels = criteria.EnumerateArray().Select(level => level.Clone()).ToArray();
                foreach (var level in levels)
                {
                    names.Add(JsonText(level) ?? "null");
                    descriptions.Add(null);
                }
            }
            else throw Error($"question '{id}': unknown type '{type}'");
            if (names.Count is < 2 or > 26) throw Error($"question '{id}': needs 2 to 26 alternatives");
            string[] labels = type == "noul" ? ["yes", "no"]
                : Enumerable.Range(0, names.Count).Select(i => type == "score" && names.Count <= 9
                    ? (i + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) : ((char)('A' + i)).ToString()).ToArray();
            parsed.Add(new(id, type, instructions, names.ToArray(), descriptions.ToArray(), labels, levels));
            if (parsed.Count > MaxQuestions) throw Error($"questions: at most {MaxQuestions} questions");
        }
        if (parsed.Count == 0) throw Error("questions: needs a non-empty map of id -> question");
        int samples = 0;
        if (root.TryGetProperty("samples", out var s) && !(s.ValueKind == JsonValueKind.String && s.GetString() == "auto"))
            samples = Integer(s, "samples", 1, MaxSamples);
        int autoMax = Integer(root, "auto_max", 4, 1, MaxSamples);
        double threshold = 0.1;
        if (root.TryGetProperty("auto_threshold", out var t) &&
            (t.ValueKind != JsonValueKind.Number || !t.TryGetDouble(out threshold) || !double.IsFinite(threshold) || threshold < 0))
            throw Error("auto_threshold must be a finite nonnegative number");
        int seed = Integer(root, "seed", 42, int.MinValue, int.MaxValue);
        int? rows = root.TryGetProperty("chunk_rows", out var r) ? Integer(r, "chunk_rows", 8, 4096) : null;
        string chunkPrompt = OptionalString(root, "chunk_prompt") ?? "own";
        if (chunkPrompt is not ("own" or "shared")) throw Error("chunk_prompt must be own or shared");
        if (Integer(root, "steps", 1, 1, 8) != 1) throw Error("steps: only one-step structured reads (steps=1) are supported");
        if (Integer(root, "think", 0, 0, 4096) != 0) throw Error("think: thought generation is not supported; use think=0");
        foreach (string unsupported in new[] { "ask", "sequential" })
            if (root.TryGetProperty(unsupported, out var value) &&
                !(unsupported == "sequential" && value.ValueKind == JsonValueKind.False))
                throw Error($"'{unsupported}' is not supported by one-step Jev inference");
        return new(model, JsonText(state)!, Text(root, "instructions"), parsed.ToArray(), samples,
            autoMax, threshold, seed, rows, chunkPrompt == "shared", images.ToArray()) { Attachments = fileAttachments };
    }

    private static string? OptionalString(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var v)) return null;
        if (v.ValueKind != JsonValueKind.String) throw Error($"{key} must be a string");
        return v.GetString();
    }
    private static string? Text(JsonElement root, string key) => root.TryGetProperty(key, out var v) ? JsonText(v) : null;
    private static string? JsonText(JsonElement v) => v.ValueKind switch
    { JsonValueKind.String => v.GetString(), JsonValueKind.Null => null, _ => v.GetRawText() };
    private static int Integer(JsonElement root, string name, int fallback, int min, int max)
        => root.TryGetProperty(name, out var v) ? Integer(v, name, min, max) : fallback;
    private static int Integer(JsonElement v, string name, int min, int max)
    {
        if (v.ValueKind != JsonValueKind.Number || !v.TryGetInt32(out int n) || n < min || n > max)
            throw Error($"{name} must be an integer from {min} to {max}");
        return n;
    }
    private static void Unique(JsonElement root, string name)
    {
        var keys = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in root.EnumerateObject())
            if (!keys.Add(p.Name)) throw Error($"{name}: duplicate key '{p.Name}'");
    }
    private static JevValidationException Error(string message) => new(message);
}
